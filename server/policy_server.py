#!/usr/bin/env python
"""LeRobot policy inference API service — FastAPI + msgpack.

把模型推理的重计算部分(Pi0 flow 的 10 步 flow-matching 前向 / Pi0-FAST 的自回归
token 解码,均过 PaliGemma 3B)从本地机器搬到更强大的推理服务器(如 4090)。本地
LeRobot 客户端通过 HTTP 调用本服务,只传输原始观测(JPEG 图像 + 关节状态 + task
字符串),服务器返回完整动作 chunk。

服务器按 checkpoint ``config.json`` 的 ``type`` 字段自动 dispatch 加载哪个 policy
类(当前支持 ``pi0`` / ``pi0_fast``),加载路径分别照搬已验证的部署脚本:
  - pi0(flow):     ``scripts/test_load_pi0_int8_5060.py``
  - pi0_fast:      ``scripts/test_load_5060.py``
两者 pre/post processor 均走通用 ``PolicyProcessorPipeline.from_pretrained``,
推理均用 ``predict_action_chunk`` 一次返回整 chunk,服务器无状态,chunk 队列由
客户端持有。

Usage:
    # 4090 大显存:走原始 bf16(默认 quantization=none)
    python server/policy_server.py \\
        --ckpt outputs/train/nexarm_pi0/checkpoints/040000/pretrained_model \\
        --host 0.0.0.0 --port 8000

    # Pi0-FAST(自回归解码;不支持 int8 量化,走 bf16)
    python server/policy_server.py \\
        --ckpt outputs/train/nexarm_pi0fast/checkpoints/040000/pretrained_model

    # 小显存机器 + Pi0 flow:开 int8 weight-only 量化(同 test_load_pi0_int8_5060.py 路径)
    python server/policy_server.py --ckpt <ckpt> --quantization int8wo

客户端(本地):
    lerobot-rollout --config_path=examples/nexarm_pi0/inference_api_client.yaml \\
        --policy.path=<ckpt>   # 仅读 config.json 元数据,不加载权重

协议:
    POST /predict
        Content-Type: application/x-msgpack
        Body(msgpack): {
            "images": {"front": <jpeg bytes>, "wrist": <jpeg bytes>},  # 相机名→JPEG
            "state":  [6 floats],                                       # 原始舵机值
            "task":   "Put blue cube in cup"
        }
        Response(msgpack): {
            "chunk":  <bytes of float32 (n_action_steps, action_dim)>,
            "shape":  [n_action_steps, action_dim]
        }
    GET /health  -> {"ready": bool, "chunk_size": int, "action_dim": int, ...}
"""

from __future__ import annotations

import argparse
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import msgpack
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

logger = logging.getLogger("policy_server")

# 全局推理状态,在 lifespan 启动时填充。
_POLICY = None
_PREPROCESSOR = None
_POSTPROCESSOR = None
_DEVICE = None
_CFG = None
# GPU 前向串行化锁:模型前向占满显存,并发请求会打爆显存。
_INFER_LOCK = threading.Lock()
READY = threading.Event()


# chunk-based policy 注册表:config.json 的 type -> (Policy 类路径, Config 类路径)。
# 新增 policy 只需在此追加一行,前提是该 policy 实现了 predict_action_chunk 且
# pre/post processor 走通用 PolicyProcessorPipeline.from_pretrained。
_POLICY_REGISTRY: dict[str, tuple[str, str, str, str]] = {
    "pi0": (
        "lerobot.policies.pi0.modeling_pi0", "PI0Policy",
        "lerobot.policies.pi0.configuration_pi0", "PI0Config",
    ),
    "pi0_fast": (
        "lerobot.policies.pi0_fast.modeling_pi0_fast", "PI0FastPolicy",
        "lerobot.policies.pi0_fast.configuration_pi0_fast", "PI0FastConfig",
    ),
}


def _detect_policy_type(ckpt_path: Path) -> str:
    """从 checkpoint ``config.json`` 的 ``type`` 字段判定 policy 类型。"""
    import json

    cfg_path = ckpt_path / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in checkpoint: {ckpt_path}")
    with open(cfg_path) as f:
        data = json.load(f)
    ptype = data.get("type") or data.get("policy_type")
    if not ptype:
        raise ValueError(f"cannot determine policy type from {cfg_path} (no 'type' field)")
    return ptype


def _import_attr(dotted: str, name: str):
    """``importlib`` 懒导入:``dotted`` 模块路径 + ``name`` 属性名。"""
    import importlib

    mod = importlib.import_module(dotted)
    return getattr(mod, name)


def _load_policy(ckpt: str, quantization: str, num_inference_steps: int | None) -> None:
    """加载 policy + pre/post processor(通用路径)。

    按 ``config.json`` 的 ``type`` dispatch 到具体 Policy/Config 类:
      - pi0(flow):照搬 ``test_load_pi0_int8_5060.py:32-84``,支持 quantization/num_inference_steps
      - pi0_fast :照搬 ``test_load_5060.py``,自回归解码,不支持 quantization,
                   强制 use_kv_cache=True;quantization/steps 参数被忽略并告警
    pre/post processor 走通用 ``PolicyProcessorPipeline.from_pretrained``,两者同形。
    """
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor import (
        batch_to_transition,
        transition_to_batch,
        policy_action_to_transition,
        transition_to_policy_action,
    )

    ckpt_path = Path(ckpt)
    assert ckpt_path.exists(), f"checkpoint not found: {ckpt_path}"

    ptype = _detect_policy_type(ckpt_path)
    if ptype not in _POLICY_REGISTRY:
        raise ValueError(
            f"unsupported policy type '{ptype}'. Registered: {list(_POLICY_REGISTRY)}. "
            f"Add it to _POLICY_REGISTRY in server/policy_server.py."
        )
    policy_mod, policy_cls_name, cfg_mod, cfg_cls_name = _POLICY_REGISTRY[ptype]
    PolicyCls = _import_attr(policy_mod, policy_cls_name)
    ConfigCls = _import_attr(cfg_mod, cfg_cls_name)
    logger.info("detected policy type='%s' -> %s / %s", ptype, policy_cls_name, cfg_cls_name)

    cfg = ConfigCls.from_pretrained(pretrained_name_or_path=str(ckpt_path))
    cfg.dtype = "bfloat16"
    cfg.device = "cuda"
    cfg.gradient_checkpointing = False

    # 类型相关开关
    if ptype == "pi0":
        cfg.quantization = quantization  # "none"(大显存)或 "int8wo"(小显存,torchao int8 weight-only)
        if num_inference_steps is not None:
            cfg.num_inference_steps = num_inference_steps
    elif ptype == "pi0_fast":
        # pi0_fast 无 quantization / num_inference_steps 字段(自回归解码,非 flow 去噪)。
        # 必须开 KV-cache,否则 256 步自回归逐 token 重算前向,极慢。
        cfg.use_kv_cache = True
        if quantization != "none":
            logger.warning(
                "pi0_fast does not support --quantization (ignored '%s'); running bf16. "
                "Use pi0 (flow) checkpoint for int8wo path.", quantization,
            )
        if num_inference_steps is not None:
            logger.warning("pi0_fast has no flow denoising steps; --num_inference_steps ignored.")

    logger.info(
        "config loaded: type=%s dtype=%s device=%s chunk_size=%s n_action_steps=%s",
        ptype, cfg.dtype, cfg.device, cfg.chunk_size, cfg.n_action_steps,
    )
    if hasattr(cfg, "quantization"):
        logger.info("  quantization=%s", cfg.quantization)
    if hasattr(cfg, "num_inference_steps"):
        logger.info("  num_inference_steps=%s", cfg.num_inference_steps)
    if hasattr(cfg, "use_kv_cache"):
        logger.info("  use_kv_cache=%s max_decoding_steps=%s", cfg.use_kv_cache, cfg.max_decoding_steps)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable - this server needs a GPU. Run the client on the robot host.")
    device = torch.device("cuda")
    policy = PolicyCls.from_pretrained(pretrained_name_or_path=str(ckpt_path), config=cfg)
    policy.eval()

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(ckpt_path),
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(ckpt_path),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
        overrides={},
    )
    logger.info("policy + pre/post processor loaded")

    globals()["_POLICY"] = policy
    globals()["_PREPROCESSOR"] = preprocessor
    globals()["_POSTPROCESSOR"] = postprocessor
    globals()["_DEVICE"] = device
    globals()["_CFG"] = cfg
    globals()["_POLICY_TYPE"] = ptype


def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """JPEG bytes -> (H, W, 3) uint8 RGB。

    cv2.imencode / cv2.imdecode 按通道位置无损往返(字节顺序保持),因此客户端
    把相机返回的 RGB 数组直接编码、服务器直接解码即可还原原始 RGB,无需通道翻转。
    模型在相机默认 color_mode=RGB 上训练,此处保持一致。
    """
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode JPEG image")
    return img


def _build_batch(payload: dict) -> dict:
    """msgpack payload -> 观测 batch dict(numpy uint8 图像 + float32 状态)。

    与 sync.py:111 的 ``prepare_observation_for_inference`` 输入约定一致:
    images 形状 (H, W, C) uint8(或 float [0,1]),state 为 1D 数组。
    """
    images = payload.get("images", {})
    if not images:
        raise ValueError("payload missing 'images'")
    batch: dict = {}
    for cam_name, jpeg_bytes in images.items():
        batch[f"observation.images.{cam_name}"] = _decode_jpeg(jpeg_bytes)

    state = payload.get("state")
    if state is None:
        raise ValueError("payload missing 'state'")
    batch["observation.state"] = np.asarray(state, dtype=np.float32)

    # task / robot_type 由 prepare_observation_for_inference 注入(同 sync.py:111-113)。
    return batch


def _predict(payload: dict) -> dict:
    """完整推理:prepare -> preprocess -> predict_action_chunk -> postprocess。

    返回 {chunk: float32 bytes, shape: [n, dim]}。
    """
    policy = _POLICY
    preprocessor = _PREPROCESSOR
    postprocessor = _POSTPROCESSOR
    device = _DEVICE
    task = payload.get("task") or ""

    batch = _build_batch(payload)
    # 复用 src/lerobot/policies/utils.py:97 —— numpy->tensor、加 batch 维、[0,1] 归一化。
    from lerobot.policies.utils import prepare_observation_for_inference

    observation = prepare_observation_for_inference(batch, device, task=task, robot_type="nexarm_follower")
    observation = preprocessor(observation)
    with _INFER_LOCK, torch.inference_mode():
        # predict_action_chunk 一次返回整 chunk (1, n_action_steps, action_dim),
        # 不走 select_action(单步 pop)—— 队列由客户端持有,服务器无状态。
        chunk = policy.predict_action_chunk(observation)
    chunk = postprocessor(chunk)
    arr = chunk.squeeze(0).cpu().numpy().astype(np.float32)
    return {"chunk": arr.tobytes(), "shape": list(arr.shape)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = app.state.args
    logger.info("Loading policy from '%s' (quantization=%s)...", args.ckpt, args.quantization)
    _load_policy(args.ckpt, args.quantization, args.num_inference_steps)
    if _DEVICE is not None and _DEVICE.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        logger.info("VRAM after load: free=%.2fG peak_allocated=%.2fG", free / 1024**3, peak)
    READY.set()
    logger.info("Policy server READY")
    yield
    READY.clear()


app = FastAPI(title="LeRobot Policy Inference Service", lifespan=lifespan)


@app.get("/health")
def health():
    ready = READY.is_set()
    info = {"ready": ready}
    if ready and _CFG is not None:
        info.update(
            policy_type=globals().get("_POLICY_TYPE"),
            chunk_size=_CFG.chunk_size,
            n_action_steps=_CFG.n_action_steps,
            action_dim=_CFG.output_features["action"].shape[0]
            if hasattr(_CFG, "output_features") and "action" in getattr(_CFG, "output_features", {})
            else None,
            # pi0_fast 无 num_inference_steps / quantization 字段,用 getattr 守卫。
            num_inference_steps=getattr(_CFG, "num_inference_steps", None),
            quantization=getattr(_CFG, "quantization", "none"),
        )
    return info


@app.post("/predict")
async def predict(request: Request) -> Response:
    if not READY.is_set():
        raise HTTPException(status_code=503, detail="server not ready")
    body = await request.body()
    try:
        payload = msgpack.unpackb(body, raw=False, strict_map_key=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"msgpack decode failed: {e}")
    try:
        result = _predict(payload)
    except Exception as e:
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail=str(e))
    return Response(content=msgpack.packb(result, use_bin_type=True), media_type="application/x-msgpack")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LeRobot policy inference API service (pi0 / pi0_fast)")
    p.add_argument("--ckpt", required=True, help="path to pretrained_model dir (config.json 的 type 决定 policy)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--quantization",
        choices=["none", "int8wo"],
        default="none",
        help="none=原始 bf16(大显存,默认); int8wo=torchao int8 weight-only(小显存)。"
        "仅 pi0(flow)支持;pi0_fast 会忽略并走 bf16。",
    )
    p.add_argument(
        "--num_inference_steps", type=int, default=None,
        help="覆盖 flow 去噪步数(默认用 config)。仅 pi0(flow)生效;pi0_fast 忽略。",
    )
    p.add_argument("--log_level", default="info")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, workers=1)


if __name__ == "__main__":
    main()
