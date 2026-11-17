"""Unit conversion between IK radians, servo ticks, PICO trigger and the gripper.

Nexarm convention (see lerobot-awesome-explore/src/lerobot/motors/nexarm/
nexarm.py): 6 servos, positions 0..4095 ticks, center 2048. The URDF joint
limits are ±2.09 rad; the linear tick<->rad mapping below must be verified
and its direction fixed on real hardware ([CALIBRATE] in config.py).
"""
from __future__ import annotations

import numpy as np

from .config import (
    GRIPPER_CLOSED_TICK,
    GRIPPER_INVERT,
    GRIPPER_OPEN_TICK,
    GRIPPER_TRIGGER_DEADZONE,
    JOINT_LIMIT_RAD,
    JOINT_SIGN,
    POSITION_CENTER_TICK,
    TICK_PER_RAD,
)


# --------------------------------------------------------------------------- arm
SERVO_TICK_MIN = 0
SERVO_TICK_MAX = 4095


def rad_to_tick(q_rad: np.ndarray) -> list[int]:
    """Convert (5,) arm joint angles in rad to servo ticks (clamped to range)."""
    q_rad = np.asarray(q_rad, dtype=float) * np.asarray(JOINT_SIGN)
    ticks = POSITION_CENTER_TICK + q_rad * TICK_PER_RAD
    ticks = np.clip(ticks, SERVO_TICK_MIN, SERVO_TICK_MAX)
    return [int(round(t)) for t in ticks]


def tick_to_rad(ticks) -> np.ndarray:
    """Convert servo ticks (list of 5, or the 5 arm entries of an observation)
    back to joint angles in rad."""
    ticks = np.asarray(ticks, dtype=float)
    return (ticks - POSITION_CENTER_TICK) / TICK_PER_RAD * np.asarray(JOINT_SIGN)


# ----------------------------------------------------------------------- gripper
def trigger_to_gripper(trigger: float) -> int:
    """Map a PICO trigger value to the gripper servo tick.

    TeleData trigger scale (see televuer/tv_wrapper.py): 10.0 = fully
    released, 0.0 = fully squeezed (pinch-like, reversed 0..10).
    Semantics: squeezing the trigger closes the jaws (matching the R1
    project where grip = grab). Values near the released end are treated as
    fully open to kill released-state jitter.
    """
    t = min(10.0, max(0.0, float(trigger)))
    t_norm = 1.0 - t / 10.0  # 0 = released (open), 1 = squeezed (closed)
    if t_norm <= GRIPPER_TRIGGER_DEADZONE:
        t_norm = 0.0
    tick = GRIPPER_OPEN_TICK + (GRIPPER_CLOSED_TICK - GRIPPER_OPEN_TICK) * t_norm
    if GRIPPER_INVERT:
        tick = 2 * GRIPPER_OPEN_TICK - tick
    return int(round(tick))
