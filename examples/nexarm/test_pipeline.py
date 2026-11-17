"""End-to-end pipeline smoke test without XR hardware.

Run:
    conda activate nexarm
    python examples/nexarm/test_pipeline.py

Simulates 300 ticks of the ACTIVE teleop loop:
  alignment (anchored at activation) -> NexarmIK -> rad_to_tick -> gripper,
with a random-walk XR wrist motion. Asserts:
  - no NaN anywhere,
  - all servo ticks within 0..4095,
  - consecutive commanded jumps bounded by MAX_STEP_RAD,
  - IK solver never hard-fails.
"""
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "src" / "lerobot"
sys.path.append(str(SRC))  # append: avoids shadowing stdlib modules

from teleop_pico.alignment import RelativeAlignment  # noqa: E402
from teleop_pico.config import MAX_STEP_RAD  # noqa: E402
from teleop_pico.nexarm_ik import NexarmIK  # noqa: E402
from teleop_pico.unit_convert import rad_to_tick, trigger_to_gripper  # noqa: E402

N_TICKS = 300


def main() -> int:
    rng = np.random.default_rng(2)
    ik = NexarmIK()

    # Activation state: arm at some q0, XR wrist anchored at an offset pose.
    q0 = rng.uniform(-0.8, 0.8, ik.nq)
    p0, R0 = ik.fk(q0)
    robot_pose = np.eye(4)
    robot_pose[:3, :3] = R0
    robot_pose[:3, 3] = p0

    xr_anchor = np.eye(4)
    xr_anchor[:3, :3] = np.array([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ])  # typical "controller pointing forward-down" orientation
    xr_anchor[:3, 3] = p0 + np.array([0.05, 0.02, 0.0])

    alignment = RelativeAlignment(robot_pose, xr_anchor, blend_seconds=0.0)
    ik.last_q = q0.copy()
    ik._filter._data_queue = []
    ik._filter.add_data(q0.copy())

    failures = []
    prev_cmd = q0.copy()
    xr_pose = xr_anchor.copy()
    for t in range(N_TICKS):
        # random-walk controller motion (2 cm / 0.03 rad per tick)
        xr_pose[:3, 3] += rng.uniform(-0.02, 0.02, 3)
        xr_pose[:3, :3] = (
            __import__("scipy.spatial.transform", fromlist=["Rotation"])
            .Rotation.from_rotvec(rng.uniform(-0.03, 0.03, 3)).as_matrix()
            @ xr_pose[:3, :3]
        )
        # re-orthonormalize to satisfy valid_pose
        u, _, vt = np.linalg.svd(xr_pose[:3, :3])
        xr_pose[:3, :3] = u @ vt

        try:
            target = alignment.poses(xr_pose)  # no `now`: same clock as the real loop
        except ValueError as exc:
            failures.append(f"tick {t}: alignment error {exc}")
            break

        q_cmd = ik.solve_ik(target, prev_cmd)
        if not np.isfinite(q_cmd).all():
            failures.append(f"tick {t}: NaN in q_cmd")
            break
        if not ik.last_solve_ok:
            failures.append(f"tick {t}: IK solver hard-failed")
            break
        if np.max(np.abs(q_cmd - prev_cmd)) > MAX_STEP_RAD + 1e-9:
            failures.append(f"tick {t}: command jump {np.max(np.abs(q_cmd - prev_cmd)):.3f} rad")
            break

        ticks = rad_to_tick(q_cmd)
        if not all(0 <= x <= 4095 for x in ticks):
            failures.append(f"tick {t}: servo ticks out of range {ticks}")
            break

        trigger = float(np.clip(rng.uniform(-0.1, 1.1), 0, 1))
        grip = trigger_to_gripper(trigger)
        if not (0 <= grip <= 4095):
            failures.append(f"tick {t}: gripper tick {grip} out of range")
            break

        prev_cmd = q_cmd.copy()

    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print(f"PIPELINE SMOKE TEST PASSED ({N_TICKS} ticks, "
          f"mean solve {np.mean([ik.last_solve_time_ms]):.1f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
