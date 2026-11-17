"""Single-arm relative PICO alignment, ported from the R1-A7 project.

Source: data/unitree-lerobot/unitree_lerobot/r1a7_dagger.py
(RelativeAlignment + valid_pose). Adaptations for a single 5-DOF arm:
  - two-arm pose lists collapsed to single poses;
  - gripper pickup-latch removed (Nexarm gripper is driven directly from the
    trigger via unit_convert.trigger_to_gripper);
  - Ownership / ZMQ contracts not ported (R1 policy-vs-human switching).

Semantics (adapted for Nexarm bring-up):
  - "No movement at activation": when the grip button is pressed, the
    alignment is constructed from the current robot EE pose and the current
    XR wrist pose; the mapped target then tracks the controller DELTA from
    that activation pose (robot-anchored relative mapping), scaled by
    pos_gain / rot_gain. A gain below 1.0 attenuates hand motion so tiny
    physical tremors stay inside the safety cage; 1.0 is the upstream R1
    1:1 behavior.
  - Deltas live in the shared robot basis: TeleVuerWrapper already expresses
    XR wrist poses in the robot base frame.
  - Optional quintic "fade" blends the initial offset away over time.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation

# Movement-speed guards used to size the fade duration (from the R1 source):
# 0.10 m/s max linear, 0.6 rad/s max angular.
_MAX_LIN_SPEED = 0.10
_MAX_ANG_SPEED = 0.6
_MAX_DISPLACEMENT = 0.6  # XR displacement beyond this requires realignment


def valid_pose(value) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("invalid wrist pose")
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("invalid homogeneous wrist pose")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-3
    ):
        raise ValueError("invalid wrist rotation")
    return value.copy()


class RelativeAlignment:
    """Relative mapping between one XR wrist and the robot EE."""

    def __init__(self, robot_pose, xr_pose, *, blend_seconds=0.0, pos_gain=1.0,
                 rot_gain=1.0, now=None):
        if not np.isfinite(blend_seconds) or blend_seconds < 0:
            raise ValueError("blend_seconds must be finite and non-negative")
        for name, gain in (("pos_gain", pos_gain), ("rot_gain", rot_gain)):
            if not np.isfinite(gain) or gain <= 0 or gain > 4:
                raise ValueError(f"{name} must be in (0, 4]")
        self.pos_gain = float(pos_gain)
        self.rot_gain = float(rot_gain)
        self.robot = valid_pose(robot_pose)
        self.xr = valid_pose(xr_pose)

        rotvec = Rotation.from_matrix(self.robot[:3, :3].T @ self.xr[:3, :3]).as_rotvec()
        distance = float(np.linalg.norm(self.robot[:3, 3] - self.xr[:3, 3]))
        angle = float(np.linalg.norm(rotvec))
        # Quintic smoothstep has a maximum derivative of 1.875.
        self.blend_duration = (
            float(max(float(blend_seconds),
                      1.875 * distance / _MAX_LIN_SPEED,
                      1.875 * angle / _MAX_ANG_SPEED))
            if blend_seconds else 0.0
        )
        self.blend_elapsed = 0.0
        # Fixed-relative mode (blend_seconds=0): full relative tracking from
        # the first tick. Fade mode ramps the weight 0 -> 1 over the blend.
        self.blend_weight = 1.0 if self.blend_duration == 0 else 0.0
        self.last_blend_time = time.monotonic() if now is None else float(now)
        if not np.isfinite(self.last_blend_time):
            raise ValueError("invalid alignment timestamp")

    def blend_snapshot(self) -> dict:
        return {
            "mode": "fade-in-relative" if self.blend_duration else "fixed-relative",
            "weight": float(self.blend_weight),
            "duration_seconds": float(self.blend_duration),
        }

    def poses(self, xr_pose, *, now=None) -> np.ndarray:
        """Map the current XR wrist pose onto the robot EE target pose."""
        current = valid_pose(xr_pose)
        stamp = time.monotonic() if now is None else float(now)
        if not np.isfinite(stamp) or stamp < self.last_blend_time:
            raise ValueError("invalid blend clock")
        if self.blend_duration:
            self.blend_elapsed += min(stamp - self.last_blend_time, 1 / 30)
            fraction = min(self.blend_elapsed / self.blend_duration, 1.0)
            self.blend_weight = (
                1.0 if fraction >= 1 else fraction**3 * (10 + fraction * (-15 + 6 * fraction))
            )
        self.last_blend_time = stamp

        w = self.blend_weight

        # Robot-anchored relative mapping: target = anchor EE pose composed
        # (world-frame, pre-multiplied) with the blend-weighted, gain-scaled
        # controller delta from the activation pose. At w=0 the target equals
        # the robot pose exactly (zero-jump activation).
        delta_p = self.pos_gain * (current[:3, 3] - self.xr[:3, 3])
        delta_R = current[:3, :3] @ self.xr[:3, :3].T
        delta_r = self.rot_gain * Rotation.from_matrix(delta_R).as_rotvec()
        pose = self.robot.copy()
        pose[:3, 3] = self.robot[:3, 3] + w * delta_p
        pose[:3, :3] = Rotation.from_rotvec(w * delta_r).as_matrix() @ self.robot[:3, :3]
        if np.linalg.norm(pose[:3, 3] - self.robot[:3, 3]) > _MAX_DISPLACEMENT:
            raise ValueError(
                f"XR displacement exceeded {_MAX_DISPLACEMENT} m; realignment required"
            )
        return pose
