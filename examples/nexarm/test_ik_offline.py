"""Offline verification for NexarmIK (no hardware required).

Run:
    conda activate nexarm
    python examples/nexarm/test_ik_offline.py

What it does:
  1. Reachability round-trip: sample 100 random joint configurations, compute
     FK poses, then drive the IK (with per-tick step limit) toward each pose
     and measure final position / tool-axis error.
  2. Speed: report per-solve wall time (excluding the first JIT-ish call).
  3. Continuity: random-walk targets, verify no command step exceeds the
     configured MAX_STEP_RAD.

Acceptance (from the implementation plan): pos error < 5 mm, tool-axis error
< 3 deg, solve time < 33 ms per frame.
"""
# OpenMP/BLAS thread pinning (Linux only) — MUST run before numpy/casadi imports
# (same rationale as teleoperate_pico.py; ~27x IK speedup on many-core CPUs).
# Windows untouched: pip casadi there ships a serial IPOPT/MUMPS build.
import os
if os.name != "nt":
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ.setdefault(_var, "1")
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")

import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "src" / "lerobot"
sys.path.append(str(SRC))  # append (not insert): avoids shadowing stdlib modules

from teleop_pico.config import JOINT_LIMIT_RAD, MAX_STEP_RAD  # noqa: E402
from teleop_pico.nexarm_ik import NexarmIK  # noqa: E402
from teleop_pico.config import TOOL_AXIS_LOCAL  # noqa: E402

N_SAMPLES = 100
SETTLE_TICKS = 60  # ticks allowed per sample for the step-limited IK to converge


def make_target(q: np.ndarray, ik: NexarmIK) -> np.ndarray:
    p, R = ik.fk(q)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def axis_error_deg(q_result: np.ndarray, T_target: np.ndarray, ik: NexarmIK) -> float:
    a_res = ik.tool_axis_world(q_result)
    a_tgt = T_target[:3, :3] @ np.asarray(TOOL_AXIS_LOCAL)
    cos = np.clip(np.dot(a_res, a_tgt) / (np.linalg.norm(a_res) * np.linalg.norm(a_tgt)), -1, 1)
    return float(np.degrees(np.arccos(cos)))


def reachability_study(ik: NexarmIK, rng, mode: str, n: int, settle: int):
    """mode='near': start within ±0.5 rad of the goal (teleop-representative).
    mode='far': start uniform random (stress test for local minima)."""
    pos_errs, dir_errs, ok_flags = [], [], []
    for _ in range(n):
        q_goal = rng.uniform(-JOINT_LIMIT_RAD, JOINT_LIMIT_RAD, ik.nq)
        T_target = make_target(q_goal, ik)

        if mode == "near":
            q_cur = np.clip(q_goal + rng.uniform(-0.5, 0.5, ik.nq),
                            -JOINT_LIMIT_RAD, JOINT_LIMIT_RAD)
        else:
            q_cur = rng.uniform(-JOINT_LIMIT_RAD, JOINT_LIMIT_RAD, ik.nq)
        q_res = q_cur.copy()
        ik.last_q = q_cur.copy()  # reset warm start state per sample
        for _ in range(settle):
            q_res = ik.solve_ik(T_target, q_res)

        p_res, _ = ik.fk(q_res)
        pos_errs.append(np.linalg.norm(p_res - T_target[:3, 3]) * 1e3)
        dir_errs.append(axis_error_deg(q_res, T_target, ik))
        ok_flags.append(ik.last_solve_ok)

    pos_errs = np.array(pos_errs)
    dir_errs = np.array(dir_errs)
    print(f"\n--- reachability [{mode}, n={n}, settle={settle}] "
          f"(solver-ok {sum(ok_flags)}/{n}) ---")
    print(f"position error  mm : mean={pos_errs.mean():.2f}  p95={np.percentile(pos_errs, 95):.2f}  max={pos_errs.max():.2f}")
    print(f"tool-axis error deg: mean={dir_errs.mean():.2f}  p95={np.percentile(dir_errs, 95):.2f}  max={dir_errs.max():.2f}")
    return pos_errs, dir_errs


def main() -> int:
    rng = np.random.default_rng(0)
    ik = NexarmIK()
    print(f"model nq={ik.nq}, frame={ik.model.frames[ik.frame_id].name}")
    print(f"joint limits: lower={ik.lower_limit}, upper={ik.upper_limit}")

    # --- 1. Reachability round-trip ---------------------------------------
    # Primary metric: teleop-representative (target near current configuration).
    pos_errs, dir_errs = reachability_study(ik, rng, "near", N_SAMPLES, 60)
    # Informational: worst-case far jumps.
    reachability_study(ik, rng, "far", N_SAMPLES, 200)

    # --- 2. Speed ----------------------------------------------------------
    q_cur = rng.uniform(-1, 1, ik.nq)
    T = make_target(q_cur + rng.uniform(-0.5, 0.5, ik.nq), ik)
    ik.solve_ik(T, q_cur)  # warm-up solve, excluded from timing
    times = []
    for _ in range(50):
        T = make_target(q_cur + rng.uniform(-0.05, 0.05, ik.nq), ik)
        t0 = time.perf_counter()
        ik.solve_ik(T, q_cur)
        times.append((time.perf_counter() - t0) * 1e3)
    times = np.array(times)
    print(f"\n--- solve time ms (n=50, small local moves) ---")
    print(f"mean={times.mean():.2f}  p95={np.percentile(times, 95):.2f}  max={times.max():.2f}")

    # --- 3. Continuity under random-walk targets ---------------------------
    # The commanded sequence is the stream of solve_ik() outputs; the hard
    # post-clamp guarantees |q_out - q_current| <= MAX_STEP, and in this test
    # q_current(t) == q_out(t-1), so consecutive outputs must differ by at
    # most MAX_STEP.
    q_cur = np.zeros(ik.nq)
    ik.last_q = q_cur.copy()
    max_step_seen = 0.0
    prev_out = q_cur.copy()
    for _ in range(200):
        p, R = ik.fk(q_cur)
        wander = rng.uniform(-0.02, 0.02, 3)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p + wander
        q_cur = ik.solve_ik(T, q_cur)
        max_step_seen = max(max_step_seen, float(np.max(np.abs(q_cur - prev_out))))
        prev_out = q_cur.copy()
    print(f"\n--- continuity ---")
    print(f"max command step across 200 ticks: {max_step_seen:.4f} rad (limit {MAX_STEP_RAD})")

    # --- verdict -----------------------------------------------------------
    ok = (
        pos_errs.mean() < 5.0
        and np.percentile(dir_errs, 95) < 3.0
        and np.percentile(times, 95) < 33.0
        and max_step_seen <= MAX_STEP_RAD + 1e-6
    )
    print("\nVERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
