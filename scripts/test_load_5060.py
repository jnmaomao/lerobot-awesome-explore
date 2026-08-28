#!/usr/bin/env python
"""Load-test the trained Pi0-FAST checkpoint on the RTX 5060 (8GB).

Mirrors the real inference load path (PI0FastPolicy.from_pretrained) and reports
peak VRAM after model load + one dummy select_action pass.
"""
import sys
import statistics
import time
from pathlib import Path

import torch
from huggingface_hub.utils import disable_progress_bars

disable_progress_bars()

CKPT = Path("outputs/train/nexarm_pi0fast/checkpoints/040000/pretrained_model")


def mem(tag: str):
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[{tag}] free={free/1024**3:.2f}G  allocated={allocated:.2f}G  reserved={reserved:.2f}G")


def main():
    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy
    from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig

    assert CKPT.exists(), f"checkpoint not found: {CKPT}"

    # Load config from checkpoint, force bf16 + cuda (inference safety net)
    cfg = PI0FastConfig.from_pretrained(pretrained_name_or_path=str(CKPT))
    cfg.dtype = "bfloat16"
    cfg.device = "cuda"
    cfg.use_kv_cache = True
    cfg.gradient_checkpointing = False  # 推理无需 gc
    print(f"config loaded: dtype={cfg.dtype} device={cfg.device} kv_cache={cfg.use_kv_cache}")

    mem("before load")
    policy = PI0FastPolicy.from_pretrained(pretrained_name_or_path=str(CKPT), config=cfg)
    policy.eval()
    mem("after load")

    # 加载 preprocessor / postprocessor（与 lerobot-rollout 同路径）
    from lerobot.processor.pipeline import PolicyProcessorPipeline
    from lerobot.processor import batch_to_transition, transition_to_batch
    from lerobot.processor import policy_action_to_transition, transition_to_policy_action

    # checkpoint 的 policy_preprocessor.json 已改为 lerobot/fast-action-tokenizer，
    # 无需再传 action_tokenizer_processor override（模拟 rollout 真实路径）。
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

    # 构造一帧 dummy 输入（与 inference_base.yaml 一致：6-DOF state + front/wrist 480x640）
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
    print(f"inference OK -> action shape={tuple(action.shape)} dtype={action.dtype}")
    print(f"action sample: {action.flatten().tolist()[:6]}")
    mem("after 1 inference")

    # ------------------------------------------------------------------
    # 推理速度基准测试
    # 关键: Pi0-FAST 这类 chunk-based policy 内部维护 action 队列,
    #   首次 select_action 跑完整生成填满队列, 后续调用只是 pop 下一个动作,
    #   直到队列空才重新推理. 因此每次计时前必须 policy.reset() 清空队列,
    #   否则 48/50 次测的都是 0.5ms 的缓存 pop, 得到假的超高 fps.
    # ------------------------------------------------------------------
    WARMUP = 3
    BENCH = 20
    # 真实动作 chunk 长度 = n_action_steps (一次生成含多少步动作),
    # 不是 action.shape[-1] (那是 action_dim=6).
    chunk_size = cfg.n_action_steps
    action_dim = action.shape[-1] if action.dim() >= 2 else action.shape[0]
    print(f"\n=== Inference benchmark (warmup={WARMUP}, bench={BENCH}) ===")
    print(f"chunk_size(n_action_steps)={chunk_size}  action_dim={action_dim}  "
          f"max_decoding_steps={cfg.max_decoding_steps}")
    print("每次计时前调用 policy.reset() 强制真实生成, 排除 action 缓存.")

    def one_real_inference():
        """reset -> preprocess -> select_action, 返回耗时(ms)."""
        policy.reset()  # 清空 action 队列, 强制本次跑完整前向
        proc = preprocessor(batch)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = policy.select_action(proc)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    # warmup（稳定 cudnn autotuner / 编译, 不计入统计）
    for _ in range(WARMUP):
        one_real_inference()

    # 计时: 每次都是真实完整生成
    lat_ms = [one_real_inference() for _ in range(BENCH)]
    lat_sorted = sorted(lat_ms)
    n = len(lat_sorted)

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

    # 基于 p95 的稳健吞吐：保证 95% 的帧能在周期内完成
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
    # 推理 fps 建议（采集数据为 30fps）
    # Pi0-FAST 自回归解码是 显存带宽受限: 每生成一个 token 都要读一遍 ~6GB 权重.
    # 一次 select_action 生成一个 chunk_size 步的动作序列, 耗时 = gen_time.
    #   - 单次生成吞吐 raw_fps = 1/gen_time (这是 "每秒能生成几个 chunk")
    #   - 可持续控制 fps = chunk_size / gen_time  (回放 chunk 内每步)
    # 要维持 30fps 控制, 需 gen_time <= chunk_size/30.
    # ------------------------------------------------------------------
    gen_p95_s = s["p95"] / 1000.0
    # 单次"每帧都调模型"的吞吐(理论上限, chunk_size=1 时)
    raw_fps_p95_hz = 1.0 / gen_p95_s
    # 回放 chunk 时的可持续控制 fps
    sustainable_control_fps = chunk_size / gen_p95_s
    # 维持 30fps 控制所需的最大单 chunk 生成时间
    required_gen_for_30fps = chunk_size / 30.0

    print(f"\n=== FPS recommendation (collection @ 30fps) ===")
    print(f"单 chunk 生成耗时(p95)       = {gen_p95_s*1000:.0f} ms")
    print(f"解码步数 max_decoding_steps  = {cfg.max_decoding_steps} "
          f"(~{s['p95']/cfg.max_decoding_steps:.1f} ms/token, 带宽受限)")
    print(f"chunk_size                   = {chunk_size} 步")
    print(f"  -> 单次生成吞吐(1/gen)      = {raw_fps_p95_hz:.2f} chunks/s")
    print(f"  -> 回放 chunk 可持续控制    = {sustainable_control_fps:.1f} fps")
    print(f"  -> 维持 30fps 控制需 gen <= {required_gen_for_30fps*1000:.0f} ms "
          f"(当前 {gen_p95_s*1000:.0f} ms, 差 {gen_p95_s/required_gen_for_30fps:.1f}x)")

    if sustainable_control_fps >= 30:
        print(f"  结论: 可支撑 30fps 控制 (可持续 {sustainable_control_fps:.1f}fps).")
    elif sustainable_control_fps >= 15:
        print(f"  结论: 无法 30fps, 但可维持 ~{sustainable_control_fps:.0f}fps 控制. "
              f"建议采集/控制降到 {int(sustainable_control_fps)}fps.")
    else:
        print(f"  结论: 显卡对该模型带宽受限, 仅 ~{sustainable_control_fps:.1f}fps. "
              f"建议采集降到 {max(1,int(sustainable_control_fps))}fps, 或:")
        print(f"     - 降 max_decoding_steps + 加 EOS 早停 (减少 token 数)")
        print(f"     - 开 compile_model=true (torch.compile, ~1.3x)")
        print(f"     - int8/int4 量化 (减半/减 4x 权重读取, 最有效但需支持)")
        print(f"     - 换非自回归 policy (diffusion/pi0-flow, 并行解码)")


if __name__ == "__main__":
    try:
        main()
    finally:
        torch.cuda.empty_cache()
