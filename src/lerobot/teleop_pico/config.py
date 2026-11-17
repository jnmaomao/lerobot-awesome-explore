"""Calibration constants and tuning parameters for the PICO4 -> Nexarm teleop pipeline.

Every value marked [CALIBRATE] must be verified on the real hardware before
serious use (small step-by-step moves, keep e-stop ready).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# nexarm.urdf lives at the repo root of lerobot-awesome-explore.
URDF_PATH = Path(__file__).resolve().parents[3] / "nexarm.urdf"

# ---------------------------------------------------------------------------
# Nexarm kinematic limits (from nexarm.urdf)
# ---------------------------------------------------------------------------
# All 5 arm joints are ±2.09 rad in the URDF.
JOINT_LIMIT_RAD = 2.09

# servo tick <-> rad mapping (linear). 2048 ticks = half range = JOINT_LIMIT_RAD.
POSITION_CENTER_TICK = 2048
TICK_PER_RAD = 2048.0 / JOINT_LIMIT_RAD  # [CALIBRATE] verify direction/ratio on hardware

# Per-joint direction sign: +1 assumes URDF positive rotation == servo tick
# increase.  Flip to -1 for any joint whose real motion is inverted.
# Verify on first power-up with the arm supported: grip, move 1 cm, check
# each axis follows the hand.  Order: shoulder_pan, shoulder_lift,
# elbow_flex, wrist_flex, wrist_roll.
# Verified on hardware 2026-11-13 via examples/nexarm/calibrate_joint_sign.py
# + real-teleop cross-check: pan/lift/elbow/wrist_roll reversed, wrist_flex as
# modeled. (pan was first mis-read as -1 due to the left/right viewing-frame
# ambiguity in the calibration prompt; teleop showed horizontal follow
# inverted with pan=-1, which proves pan is +1. wrist_roll's "same direction"
# answer was unreliable — the prompt gave no spin-direction reference — and
# teleop showed the tool twist inverted with roll=+1, which proves roll is -1.)
JOINT_SIGN = (1, -1, -1, 1, -1)

# ---------------------------------------------------------------------------
# IK tuning
# ---------------------------------------------------------------------------
# End-effector frame in the URDF used as IK target.
EE_FRAME_NAME = "gripper_base_link"

# Local axis of the EE frame that represents the tool pointing direction
# (+X = gripper approach direction per URDF layout).
TOOL_AXIS_LOCAL = (1.0, 0.0, 0.0)

# Cost weights (see nexarm_ik.py).
W_POSITION = 100.0   # position tracking, dominant
W_DIRECTION = 20.0   # tool-axis pointing
W_REGULARIZATION = 0.02  # pull toward zero pose (tie-breaker only)
W_SMOOTH = 0.1       # pull toward previous solution (anti-jump; R1 uses 0.1)

# Per-tick joint step bound (30 Hz tick => ~10 rad/s max joint speed).
MAX_STEP_RAD = 0.35  # [CALIBRATE] tighten for real hardware

# Output smoothing filter (window 4, same style as the R1 project).
FILTER_WEIGHTS = (0.4, 0.3, 0.2, 0.1)

# ---------------------------------------------------------------------------
# Teleop gain — controller delta is scaled before becoming an IK target.
# Hand tremor of 1-2 deg / 2 cm maps 1:1 under gain=1.0, which already
# saturates a level-0 safety cage (±2.9 deg/axis). Gain 0.3 attenuates:
# 2 cm hand motion -> 6 mm EE target, 1.5 deg hand rotation -> 0.45 deg.
# Raise towards 1.0 together with SAFETY_LEVEL.
# ---------------------------------------------------------------------------
TELEOP_POS_GAIN = 0.3   # [CALIBRATE: raise with cage level]
TELEOP_ROT_GAIN = 0.3   # [CALIBRATE: raise with cage level]

# ---------------------------------------------------------------------------
# Safety cage — hardware bring-up confinement.
# On activation the arm's current pose is recorded as the anchor; all IK
# output is clamped into |q - anchor| <= max_dev_rad per axis, step-limited
# to max_step_rad per tick (30Hz), and the gripper is confined to
# POSITION_CENTER_TICK +/- grip_half_span. Raise SAFETY_LEVEL step by step
# only after the previous level has been verified stable on hardware.
# ---------------------------------------------------------------------------
SAFETY_CAGE_ENABLE = True
SAFETY_LEVEL = 0  # 0 = tightest [CALIBRATE: raise to unlock progressively]
SAFETY_LEVELS = {
    0: dict(max_dev_rad=0.05, max_step_rad=0.015, grip_half_span=60),    # ~2.9°/axis, ~0.45 rad/s, jaw ±60 tick
    1: dict(max_dev_rad=0.15, max_step_rad=0.030, grip_half_span=150),   # ~8.6°/axis
    2: dict(max_dev_rad=0.40, max_step_rad=0.060, grip_half_span=400),   # ~23°/axis
    3: None,  # cage off — full workspace
}

# ---------------------------------------------------------------------------
# Gripper (servo #6) mapping
# TeleData trigger scale: 10.0 = fully released (open), 0.0 = fully squeezed
# (closed) — see televuer/tv_wrapper.py.
# ---------------------------------------------------------------------------
# Verified on hardware 2026-11-13: with the trigger released the arm held
# 2448 and the jaws were CLOSED, so the pair below is swapped relative to the
# original assumption (2448 open / 1848 closed).
GRIPPER_OPEN_TICK = 1848    # trigger fully released: jaws open   [CALIBRATE]
GRIPPER_CLOSED_TICK = 2448  # trigger fully squeezed: jaws closed [CALIBRATE]
GRIPPER_INVERT = False      # flip if open/close direction is reversed [CALIBRATE]
GRIPPER_TRIGGER_DEADZONE = 0.05  # fraction of squeeze; squeeze below this reads as open
