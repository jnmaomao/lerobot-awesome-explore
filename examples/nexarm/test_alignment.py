"""Unit tests for teleop_pico.alignment (no hardware required).

Run:
    conda activate nexarm
    python examples/nexarm/test_alignment.py

Checks:
  1. valid_pose rejects malformed input.
  2. Zero-jump activation: with blend_seconds=0 (fixed-relative), the first
     mapped target equals the robot pose exactly while the controller is
     stationary.
  3. Follows motion: after activation, moving the XR wrist by delta moves the
     EE target by the same delta (positions in the shared robot basis).
  4. Displacement guard: XR motion beyond 0.6 m raises ValueError.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SRC = Path(__file__).resolve().parents[2] / "src" / "lerobot"
sys.path.append(str(SRC))  # append: avoids shadowing stdlib modules

from teleop_pico.alignment import RelativeAlignment, valid_pose  # noqa: E402


def rot(axis, angle):
    return Rotation.from_rotvec(np.asarray(axis) * angle).as_matrix()


def make_pose(axis, angle, trans):
    T = np.eye(4)
    T[:3, :3] = rot(axis, angle)
    T[:3, 3] = trans
    return T


def main() -> int:
    rng = np.random.default_rng(1)
    failures = []

    # --- 1. valid_pose ------------------------------------------------------
    try:
        valid_pose(np.eye(3))
        failures.append("valid_pose accepted a 3x3 matrix")
    except ValueError:
        pass
    try:
        T = np.eye(4)
        T[:3, :3] = rot([0, 0, 1], 0.5) * 1.01  # non-orthonormal
        valid_pose(T)
        failures.append("valid_pose accepted a non-orthonormal rotation")
    except ValueError:
        pass

    # --- 2. zero-jump activation --------------------------------------------
    robot_pose = make_pose([1, 0, 0], 0.3, [0.25, 0.0, 0.30])
    xr_anchor = make_pose([0, 1, 0], 1.2, [0.30, 0.10, 0.35])  # intentionally offset
    align = RelativeAlignment(robot_pose, xr_anchor, blend_seconds=0.0, now=100.0)
    out = align.poses(xr_anchor, now=100.1)
    if not np.allclose(out, robot_pose, atol=1e-9):
        failures.append(f"activation jump! mapped={out[:3,3]} vs robot={robot_pose[:3,3]}")

    # --- 3. follows motion ----------------------------------------------------
    delta_p = np.array([0.04, -0.03, 0.05])
    R_delta = rot([0, 0, 1], 0.2)
    xr_now = np.eye(4)
    xr_now[:3, :3] = R_delta @ xr_anchor[:3, :3]
    xr_now[:3, 3] = xr_anchor[:3, 3] + delta_p
    out = align.poses(xr_now, now=100.2)
    expect_p = robot_pose[:3, 3] + delta_p
    if not np.allclose(out[:3, 3], expect_p, atol=1e-6):
        failures.append(f"position not following: {out[:3,3]} vs {expect_p}")
    expect_R = R_delta @ robot_pose[:3, :3]
    if not np.allclose(out[:3, :3], expect_R, atol=1e-6):
        failures.append("rotation not following delta rotation")

    # --- 4. displacement guard -------------------------------------------------
    xr_far = xr_now.copy()
    xr_far[:3, 3] += np.array([0.0, 0.0, 0.7])  # > 0.6 m from robot anchor
    try:
        align.poses(xr_far, now=100.3)
        failures.append("displacement guard did not trigger")
    except ValueError:
        pass

    # --- report ---------------------------------------------------------------
    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("ALL ALIGNMENT TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
