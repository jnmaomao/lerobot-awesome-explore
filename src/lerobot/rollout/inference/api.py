# Copyright 2025 The HuggingFace Inc. team. All rights reserved. Licensed under the Apache License, Version 2.0.

"""Remote API inference engine: pull action chunks from a GPU inference server.

本引擎把模型推理的重计算部分卸载到远端服务器(见 ``server/policy_server.py``)。
本地只负责:采集观测 -> JPEG 编码 -> 一次 HTTP/msgpack RPC -> 拿回整 chunk ->
本地 ``deque`` 缓存逐帧 pop -> ``make_robot_action`` 重排 -> 交给控制环。

pi0 是 chunk-based policy:一次前向生成 ``n_action_steps``(=50)步动作。因此
RPC 频率极低(约每 1.67s 一次 @30fps),网络延迟被 chunk 缓冲 + ``ActionInterpolator``
自然掩盖。队列由客户端持有,服务器无状态 —— 这与 ``SyncInferenceEngine`` 走
``select_action`` 单步 pop 的区别只在请求粒度,对控制环(``send_next_action``)
返回的 1D 动作 tensor 格式完全一致。

依赖(客户端,机器人主机):``requests``(lerobot 已依赖)、``msgpack``(新增,纯 Python)。
"""

from __future__ import annotations

import logging
import time
from collections import deque

import cv2
import msgpack
import numpy as np
import requests
import torch

from lerobot.policies.utils import make_robot_action
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from .base import InferenceEngine

logger = logging.getLogger(__name__)

# 观测 dict 中图像键的前缀。build_dataset_frame 产出形如
# "observation.images.front" / "observation.images.wrist" 的键。
_IMG_PREFIX = OBS_IMAGES + "."  # "observation.images."
_STATE_KEY = OBS_STATE  # "observation.state"


class ApiInferenceEngine(InferenceEngine):
    """Remote inference backend: fetch action chunks over HTTP.

    对外接口与 ``SyncInferenceEngine`` 一致(``get_action`` 返回 1D 动作 tensor,
    按 ``ordered_action_keys`` 排序),因此 ``base`` 等策略无需改动即可切换。
    """

    def __init__(
        self,
        server_url: str,
        dataset_features: dict,
        ordered_action_keys: list[str],
        task: str,
        n_action_steps: int,
        device: str | None = None,
        timeout: float = 5.0,
        jpeg_quality: int = 90,
    ) -> None:
        self._server_url = server_url.rstrip("/")
        self._dataset_features = dataset_features
        self._ordered_action_keys = ordered_action_keys
        self._task = task
        self._timeout = timeout
        self._jpeg_quality = jpeg_quality
        # 客户端持有 chunk 队列:队列空时才发一次 RPC 拉取整 chunk。
        self._chunk_queue: deque = deque(maxlen=max(1, n_action_steps))
        # 复用 sync.py 的 device 语义(本引擎无本地模型,device 仅用于返回 tensor)。
        self._device = torch.device(device or "cpu")
        self._session = requests.Session()
        self._rpc_count = 0
        logger.info(
            "ApiInferenceEngine initialized (server=%s, n_action_steps=%d, action_keys=%d)",
            self._server_url, n_action_steps, len(ordered_action_keys),
        )

    # -- InferenceEngine lifecycle ------------------------------------------

    def start(self) -> None:
        logger.info("ApiInferenceEngine started (remote mode — no local GPU)")
        # 启动时探活,尽早暴露服务器不可达问题。
        try:
            r = self._session.get(f"{self._server_url}/health", timeout=self._timeout)
            r.raise_for_status()
            info = r.json()
            logger.info("Server health: %s", info)
        except Exception as e:
            logger.warning("Server health check failed: %s", e)

    def stop(self) -> None:
        logger.info("ApiInferenceEngine stopped (rpc_count=%d)", self._rpc_count)
        self._session.close()

    def reset(self) -> None:
        """清空 chunk 队列 —— 对应 ``policy.reset()`` 清 ``_action_queue``。"""
        logger.info("Resetting ApiInferenceEngine state (chunk queue cleared)")
        self._chunk_queue.clear()

    # -- Action production --------------------------------------------------

    def get_action(self, obs_frame: dict | None) -> torch.Tensor | None:
        """返回下一帧动作(1D tensor,按 ``ordered_action_keys`` 排序)。

        队列空时发一次 RPC 拉取整 chunk 并填入队列;否则直接 pop 一帧。
        与 ``SyncInferenceEngine.get_action`` 的返回格式逐位一致。
        """
        if obs_frame is None:
            return None

        if len(self._chunk_queue) == 0:
            chunk = self._fetch_chunk(obs_frame)  # (n_action_steps, action_dim) np.float32
            # 逐行填入队列,每行是一帧动作 (action_dim,)。
            for i in range(chunk.shape[0]):
                self._chunk_queue.append(chunk[i])

        action_np = self._chunk_queue.popleft()  # (action_dim,) float32
        action_tensor = torch.from_numpy(action_np.copy())

        # 复用 sync.py:119-122 的重排逻辑,保证与 sync 引擎输出格式一致。
        action_dict = make_robot_action(action_tensor, self._dataset_features)
        return torch.tensor([action_dict[k] for k in self._ordered_action_keys])

    # -- internals ----------------------------------------------------------

    def _encode_obs(self, obs_frame: dict) -> dict:
        """obs_frame(numpy dict) -> msgpack payload{images, state, task}。

        图像 uint8 (H,W,C) -> JPEG bytes 省带宽;state -> list。
        """
        images: dict[str, bytes] = {}
        state = None
        for key, val in obs_frame.items():
            if key.startswith(_IMG_PREFIX):
                cam_name = key[len(_IMG_PREFIX):]
                arr = np.asarray(val)
                if arr.dtype != np.uint8:
                    # prepare_observation_for_inference 接受 float [0,1];这里统一转 uint8 编码。
                    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                # cv2.imencode 按通道位置无损往返,直接编码 RGB 数组即可(见服务器 _decode_jpeg 注释)。
                ok, buf = cv2.imencode(
                    ".jpg", arr,
                    [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
                )
                if not ok:
                    raise ValueError(f"JPEG encode failed for camera '{cam_name}'")
                images[cam_name] = buf.tobytes()
            elif key == _STATE_KEY:
                state = np.asarray(val, dtype=np.float32).tolist()

        if not images:
            raise ValueError(f"no image keys (prefix '{_IMG_PREFIX}') found in obs_frame")
        if state is None:
            raise ValueError(f"state key '{_STATE_KEY}' not found in obs_frame")

        return {"images": images, "state": state, "task": self._task}

    def _fetch_chunk(self, obs_frame: dict) -> np.ndarray:
        """POST /predict -> (n_action_steps, action_dim) float32 数组。"""
        payload = self._encode_obs(obs_frame)
        body = msgpack.packb(payload, use_bin_type=True)
        url = f"{self._server_url}/predict"
        t0 = time.perf_counter()
        r = self._session.post(
            url,
            data=body,
            headers={"Content-Type": "application/x-msgpack"},
            timeout=self._timeout,
        )
        r.raise_for_status()
        result = msgpack.unpackb(r.content, raw=False, strict_map_key=False)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._rpc_count += 1
        chunk = np.frombuffer(result["chunk"], dtype=np.float32).reshape(result["shape"])
        logger.debug(
            "RPC #%d: %d cameras, %.1fKB up, chunk %s, RTT %.0fms",
            self._rpc_count, len(payload["images"]), len(body) / 1024.0, list(chunk.shape), dt_ms,
        )
        return chunk
