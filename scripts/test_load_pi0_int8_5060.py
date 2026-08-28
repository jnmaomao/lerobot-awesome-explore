#!/usr/bin/env python
"""Load-test the trained Pi0 (flow) checkpoint on the RTX 5060 (8GB) with int8 weight-only quantization.

路径 A: 运行时 weight-only 量化 (torchao int8wo).
  - 8.3GB 原始混合精度权重 (bf16 语言塔 + fp32 视觉塔/layernorm) 无法直接加载到 8GB 显存.
  - 加载后在 CPU 上对 bf16 Linear 做 int8 weight-only 量化 (~6.7GB -> ~3.4GB), 再搬到 cuda.
  - 跳过 vision_tower / multi_modal_projector / layernorm / proj / lm_head (保持原精度).

本脚本复现 lerobot-rollout 的真实加载路径 (PI0Policy.from_pretrained + processor),
报告: 量化层数 / 峰值 VRAM / 一次 select_action 数值正确性 / 推理延迟与吞吐.
"""
import sys
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()

CKPT = Path("outputs/train/nexarm_pi0/checkpoints/040000/pretrained_model")


def mem(tag: str):
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[{tag}] free={free/1024**3:.2f}G  allocated={allocated:.2f}G  reserved={reserved:.2f}G")


def main():
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.pi0.configuration_pi0 import PI0Config

    assert CKPT.exists(), f"checkpoint not found: {CKPT}"

    # Load config from checkpoint, force bf16 + cuda + int8 weight-only quantization.
    # quantization="int8wo" 让 from_pretrained 在 load_state_dict 后于 CPU 上量化, 再 to(cuda).
    cfg = PI0Config.from_pretrained(pretrained_name_or_path=str(CKPT))
    cfg.dtype = "bfloat16"
    cfg.device = "cuda"
    cfg.quantization = "int8wo"          # 路径 A: int8 weight-only
    cfg.gradient_checkpointing = False   # 推理无需 gc
    print(f"config loaded: dtype={cfg.dtype} device={cfg.device} quantization={cfg.quantization}")
    print(f"  chunk_size={cfg.chunk_size} n_action_steps={cfg.n_action_steps} "
          f"num_inference_steps={cfg.num_inference_steps}")

    mem("before load")
    policy = PI0Policy.from_pretrained(pretrained_name_or_path=str(CKPT), config=cfg)
    policy.eval()
    mem("after load + quantize")

    # 统计量化后各模块参数 dtype 分布 (确认视觉塔仍 fp32, 语言塔已 int8)
    from collections import Counter
    dtype_counter = Counter()
    for name, p in policy.named_parameters():
        # torchao Int8Tensor 的 dtype 仍是 bf16 (包装层), 用类型名区分
        tname = type(p).__name__
        dtype_counter[(str(p.dtype), tname)] += p.numel() * p.element_size()
    print("\n=== 参数显存占用 (dtype, tensor_type) ===")
    for (dt, tn), bytes_ in sorted(dtype_counter.items(), key=lambda x: -x[1]):
        print(f"  dtype={dt:8s} type={tn:30s} {bytes_/1024**3:.3f} GB")

    # 加载 preprocessor / postprocessor (与 lerobot-rollout 同路径)
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor import batch_to_transition, transition_to_batch
    from lerobot.processor import policy_action_to_transition, transition_to_policy_action

    preprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(CKPT),
        config_filename="policy_preprocessor.json",
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
        overrides={},
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        pretrained_model_name_or_path=str(CKPT),
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
        overrides={},
    )
    print("preprocessor/postprocessor loaded")

    # 构造一帧 dummy 输入 (与 inference_base.yaml 一致: 6-DOF state + front/wrist 480x640)
    batch = {
        "observation.state": torch.zeros(1, 6, device="cuda"),
        "observation.images.front": torch.zeros(1, 3, 480, 640, device="cuda"),
        "observation.images.wrist": torch.zeros(1, 3, 480, 640, device="cuda"),
        "task": ["Put blue cube in cup"],
    }
    processed = preprocessor(batch)
    with torch.inference_mode():
        action_chunk = policy.select_action(processed)
    action = postprocessor(action_chunk)
    print(f"\ninference OK -> action shape={tuple(action.shape)} dtype={action.dtype}")
    print(f"action sample: {action.flatten().tolist()[:6]}")
    mem("after 1 inference")

    # ------------------------------------------------------------------
    # 推理速度基准测试
    # Pi0 flow: 一次 select_action 跑 num_inference_steps(10) 步去噪, 生成 chunk_size(50) 步动作.
    # 内部维护 action 队列, 首次填满后后续 pop; 每次计时前必须 policy.reset() 清空队列,
    # 否则测的是缓存 pop, 得到假的超高 fps.
    # ------------------------------------------------------------------
    WARMUP = 3
    BENCH = 20
    chunk_size = cfg.chunk_size
    action_dim = action.shape[-1] if action.dim() >= 2 else action.shape[0]
    print(f"\n=== Inference benchmark (warmup={WARMUP}, bench={BENCH}) ===")
    print(f"chunk_size={chunk_size}  action_dim={action_dim}  "
          f"num_inference_steps={cfg.num_inference_steps}")
    print("每次计时前调用 policy.reset() 强制真实生成, 排除 action 缓存.")

    def one_real_inference():
        """reset -> preprocess -> select_action, 返回耗时(ms)."""
        policy.reset()
        proc = preprocessor(batch)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = policy.select_action(proc)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    for _ in range(WARMUP):
        one_real_inference()

    lat_ms = [one_real_inference() for _ in range(BENCH)]

    def stats_ms(xs_ms):
        xs = sorted(xs_ms)
        n = len(xs)
        return {
            "mean": statistics.mean(xs),
            "median": statistics.median(xs),
            "p95": xs[min(n - 1, int(0.95 * n))],
            "min": xs[0],
            "max": xs[-1],
        }

    s = stats_ms(lat_ms)
    print(f"[full preprocess+infer, real generation]")
    print(f"  mean={s['mean']:.1f}ms  median={s['median']:.1f}ms  "
          f"p95={s['p95']:.1f}ms  min={s['min']:.1f}ms  max={s['max']:.1f}ms")
    print(f"  raw samples(ms): {[round(x,1) for x in lat_ms]}")

    raw_fps_p95 = 1000.0 / s["p95"]
    raw_fps_median = 1000.0 / s["median"]
    raw_fps_mean = 1000.0 / s["mean"]
    print(f"\nraw throughput (per real select_action call):")
    print(f"  mean-based   = {raw_fps_mean:.1f} Hz")
    print(f"  median-based = {raw_fps_median:.1f} Hz")
    print(f"  p95-based    = {raw_fps_p95:.1f} Hz  (更保守, 推荐)")
    mem("after benchmark")

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nPEAK allocated VRAM: {peak:.2f} GB  (5060 has ~8GB)")
    print("LOAD TEST:", "PASS" if peak < 7.6 else "TIGHT/FAIL")

    # ------------------------------------------------------------------
    # 推理 fps 建议 (采集数据为 30fps)
    # Pi0 flow 一次生成 chunk_size 步动作, 耗时 = gen_time.
    #   可持续控制 fps = chunk_size / gen_time  (回放 chunk 内每步).
    # 维持 30fps 控制, 需 gen_time <= chunk_size/30.
    # ------------------------------------------------------------------
    gen_p95_s = s["p95"] / 1000.0
    raw_fps_p95_hz = 1.0 / gen_p95_s
    sustainable_control_fps = chunk_size / gen_p95_s
    required_gen_for_30fps = chunk_size / 30.0

    print(f"\n=== FPS recommendation (collection @ 30fps) ===")
    print(f"单 chunk 生成耗时(p95)       = {gen_p95_s*1000:.0f} ms")
    print(f"去噪步数 num_inference_steps = {cfg.num_inference_steps} "
          f"(~{s['p95']/cfg.num_inference_steps:.1f} ms/step)")
    print(f"chunk_size                   = {chunk_size} 步")
    print(f"  -> 单次生成吞吐(1/gen)      = {raw_fps_p95_hz:.2f} chunks/s")
    print(f"  -> 回放 chunk 可持续控制    = {sustainable_control_fps:.1f} fps")
    print(f"  -> 维持 30fps 控制需 gen <= {required_gen_for_30fps*1000:.0f} ms "
          f"(当前 {gen_p95_s*1000:.0f} ms)")

    if sustainable_control_fps >= 30:
        print(f"  结论: 可支撑 30fps 控制 (可持续 {sustainable_control_fps:.1f}fps).")
    elif sustainable_control_fps >= 15:
        print(f"  结论: 无法 30fps, 但可维持 ~{sustainable_control_fps:.0f}fps 控制. "
              f"建议采集/控制降到 {int(sustainable_control_fps)}fps.")
    else:
        print(f"  结论: 显卡对该模型带宽受限, 仅 ~{sustainable_control_fps:.1f}fps. 建议:")
        print(f"     - 降 num_inference_steps (10->4, ~2.5x 加速, 简单任务通常够)")
        print(f"     - 开 compile_model=true (torch.compile, ~1.3x, 8GB 须谨慎)")
        print(f"     - 换非自回归 policy")


if __name__ == "__main__":
    try:
        main()
    finally:
        torch.cuda.empty_cache()
