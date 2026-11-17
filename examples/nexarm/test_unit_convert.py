"""Unit tests for teleop_pico.unit_convert (no hardware required).

Run:
    conda activate nexarm
    python examples/nexarm/test_unit_convert.py
"""
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "src" / "lerobot"
sys.path.append(str(SRC))  # append: avoids shadowing stdlib modules

from teleop_pico.config import (  # noqa: E402
    GRIPPER_CLOSED_TICK,
    GRIPPER_OPEN_TICK,
    JOINT_LIMIT_RAD,
    POSITION_CENTER_TICK,
    TICK_PER_RAD,
)
from teleop_pico.unit_convert import rad_to_tick, tick_to_rad, trigger_to_gripper  # noqa: E402


def main() -> int:
    failures = []

    # rad <-> tick round trip at boundaries and center.
    # One tick ≈ 2.09/2048 ≈ 1e-3 rad; allow ~1 tick (quantization + the
    # range clamp at the +limit boundary costs one extra tick).
    HALF_TICK_RAD = 1.0 / TICK_PER_RAD
    for q in (-JOINT_LIMIT_RAD, 0.0, JOINT_LIMIT_RAD, 0.37, -1.23):
        ticks = rad_to_tick([q])
        back = tick_to_rad(ticks)
        if not np.isclose(back[0], q, atol=1.1 * HALF_TICK_RAD):
            failures.append(f"round trip failed for q={q}: {back[0]}")
    if rad_to_tick([0.0])[0] != POSITION_CENTER_TICK:
        failures.append("zero rad should map to center tick 2048")

    # limits map inside 0..4095
    lo, hi = rad_to_tick([-JOINT_LIMIT_RAD, JOINT_LIMIT_RAD])
    if not (0 <= lo <= 4095 and 0 <= hi <= 4095):
        failures.append(f"limit ticks out of servo range: {lo}, {hi}")

    # gripper mapping endpoints and dead zone.
    # TeleData trigger scale: 10.0 = fully released (open), 0.0 = fully
    # squeezed (closed) — reversed 0..10 per televuer/tv_wrapper.py.
    if trigger_to_gripper(10.0) != GRIPPER_OPEN_TICK:
        failures.append("trigger 10 (released) should map to open tick")
    if trigger_to_gripper(0.0) != GRIPPER_CLOSED_TICK:
        failures.append("trigger 0 (squeezed) should map to closed tick")
    if trigger_to_gripper(9.96) != GRIPPER_OPEN_TICK:
        failures.append("released jitter within dead zone should read fully open")
    mid = trigger_to_gripper(5.0)
    if not (min(GRIPPER_OPEN_TICK, GRIPPER_CLOSED_TICK) <= mid <= max(GRIPPER_OPEN_TICK, GRIPPER_CLOSED_TICK)):
        failures.append(f"mid trigger out of range: {mid}")

    # all gripper outputs stay inside servo range (over the whole 0..10 scale
    # and beyond, since raw values can transiently overshoot)
    for t in np.linspace(-2, 12, 141):
        g = trigger_to_gripper(t)
        if not (0 <= g <= 4095):
            failures.append(f"tick {g} out of servo range at trigger={t}")
            break

    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("ALL UNIT_CONVERT TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
