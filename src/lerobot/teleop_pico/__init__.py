"""PICO4 teleoperation + IK for the Nexarm 5-DOF arm + gripper.

Ported/adapted from the r1-a7-autonomous-checkout XR teleoperation project:
  - televuer/            : TeleVuer wrapper (XR input over vuer/WebRTC), copied as-is
  - alignment.py         : single-arm RelativeAlignment (pose delta mapping)
  - nexarm_ik.py         : CasADi + Pinocchio NLP IK for 5 joints
  - unit_convert.py      : rad <-> servo tick, trigger -> gripper mapping
  - weighted_moving_filter.py : output smoothing (copied as-is, viz removed)
  - config.py            : calibration constants
"""
