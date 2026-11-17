"""CasADi + Pinocchio numerical IK for the Nexarm 5-DOF arm.

Structure follows R1_A7_ArmIK (robot/xr-teleoperate/teleop/robot_control/
robot_arm_ik.py) but is simplified for a 5-joint arm:

  - Decision variable: q (5,), one MX per arm joint.
  - Costs (all soft, weighted):
      position  : || p_ee(q) - p_target ||^2                    (3-dim, dominant)
      direction : || a_ee(q) - a_target ||^2                    (tool axis, 2-dim)
      regularization : || q ||^2                                (tie-breaker)
      smooth    : || q - q_last ||^2                            (anti-jump)
  - Constraints:
      joint limits   : lowerPositionLimit <= q <= upperPositionLimit (URDF)
      step limit     : |q - q_last| <= MAX_STEP_RAD                 (per tick)

Because the arm has 5 DOF and the task is 3 (position) + 2 (tool axis) = 5
dimensions, the problem is generically square: a good target is fully
reachable, an unreachable one degrades gracefully through the weights.

The twist about the tool axis (wrist_roll) is invisible to that task (it
moves neither the EE position nor the tool axis), so it bypasses IK: the
controller's roll delta is extracted from the target rotation and added
directly onto wrist_roll (see _roll_delta / solve_ik).
"""
from __future__ import annotations

import logging

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

from .config import (
    EE_FRAME_NAME,
    FILTER_WEIGHTS,
    JOINT_LIMIT_RAD,
    MAX_STEP_RAD,
    TOOL_AXIS_LOCAL,
    W_DIRECTION,
    W_POSITION,
    W_REGULARIZATION,
    W_SMOOTH,
    URDF_PATH,
)
from .weighted_moving_filter import WeightedMovingFilter

logger = logging.getLogger(__name__)

# Weights for the final output smoothing filter (window 4).
_OUTPUT_FILTER_WEIGHTS = np.array(FILTER_WEIGHTS)


class NexarmIK:
    """Single-solve IK for Nexarm: pose target -> 5 joint angles (rad)."""

    def __init__(self, urdf_path: str | Path | None = None, verbose: bool = False):
        self.urdf_path = str(urdf_path or URDF_PATH)

        # --- Pinocchio model (model-only: no mesh files needed) -------------
        model = pin.buildModelFromUrdf(self.urdf_path)
        if model.nq == 0:
            raise ValueError(f"No joints parsed from {self.urdf_path}")

        # Lock every non-arm joint (gripper + jaws). Fixed joints (camera,
        # gripper fixed links) never become pinocchio joints.
        joints_to_lock = []
        for name in ("gripper_base_joint", "right_jaw_joint", "left_jaw_joint"):
            jid = model.getJointId(name)
            if jid != 0:  # 0 = universe; missing joints are silently skipped
                joints_to_lock.append(jid)
        self.model = pin.buildReducedModel(
            model, joints_to_lock, np.zeros(model.nq)
        )
        self.nq = self.model.nq
        if self.nq != 5:
            raise ValueError(
                f"Expected 5 arm joints after locking, got nq={self.nq}. "
                "Check the URDF joint names."
            )
        self.frame_id = self.model.getFrameId(EE_FRAME_NAME)
        if self.frame_id >= len(self.model.frames) or self.model.frames[
            self.frame_id
        ].name != EE_FRAME_NAME:
            raise ValueError(f"Frame '{EE_FRAME_NAME}' not found in {self.urdf_path}")

        # --- CasADi symbolic model ------------------------------------------
        self.cmodel = cpin.Model(self.model)
        self.cdata = self.cmodel.createData()
        self.cq = casadi.SX.sym("q", self.nq, 1)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        ee_rot = self.cdata.oMf[self.frame_id].rotation
        self.c_tool_axis = ee_rot @ casadi.DM(np.asarray(TOOL_AXIS_LOCAL).reshape(3, 1))

        # Symbolic error functions (kept as casadi.Function like the R1 code,
        # useful for debugging/plotting the cost landscape).
        self.cTf = casadi.SX.sym("tf", 4, 4)
        self.position_error = casadi.Function(
            "position_error",
            [self.cq, self.cTf],
            [self.cdata.oMf[self.frame_id].translation - self.cTf[:3, 3]],
        )
        self.direction_error = casadi.Function(
            "direction_error",
            [self.cq, self.cTf],
            [
                self.c_tool_axis
                - self.cTf[:3, :3] @ casadi.DM(np.asarray(TOOL_AXIS_LOCAL).reshape(3, 1))
            ],
        )

        # --- Optimization problem -------------------------------------------
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.nq)
        self.param_q_last = self.opti.parameter(self.nq)  # warm start / smoothness
        self.param_tf = self.opti.parameter(4, 4)

        position_cost = casadi.sumsqr(
            self.position_error(self.var_q, self.param_tf)
        )
        direction_cost = casadi.sumsqr(
            self.direction_error(self.var_q, self.param_tf)
        )
        regularization_cost = casadi.sumsqr(self.var_q)
        smooth_cost = casadi.sumsqr(self.var_q - self.param_q_last)

        lower = self.model.lowerPositionLimit.copy()
        upper = self.model.upperPositionLimit.copy()
        # URDF velocity/effort fields are 0; position limits are ±2.09 — trust
        # them but clamp against config in case of URDF oddities.
        np.clip(lower, -JOINT_LIMIT_RAD, JOINT_LIMIT_RAD, out=lower)
        np.clip(upper, -JOINT_LIMIT_RAD, JOINT_LIMIT_RAD, out=upper)
        self.lower_limit = lower
        self.upper_limit = upper

        self.opti.subject_to(self.opti.bounded(lower, self.var_q, upper))
        self.opti.subject_to(
            self.opti.bounded(-MAX_STEP_RAD, self.var_q - self.param_q_last, MAX_STEP_RAD)
        )
        self.opti.minimize(
            W_POSITION * position_cost
            + W_DIRECTION * direction_cost
            + W_REGULARIZATION * regularization_cost
            + W_SMOOTH * smooth_cost
        )

        opts = {
            # CasADi-level options
            "expand": True,
            "detect_simple_bounds": True,
            "calc_lam_p": False,  # avoids IPOPT "NaN detected" issues
            "print_time": False,
            # IPOPT solver options
            "ipopt.sb": "yes",
            "ipopt.print_level": 0 if not verbose else 5,
            "ipopt.max_iter": 100,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 5e-4,
            "ipopt.acceptable_iter": 5,
            "ipopt.warm_start_init_point": "yes",
        }
        self.opti.solver("ipopt", opts)

        # --- Runtime state ---------------------------------------------------
        self.last_q = np.zeros(self.nq)  # previous solution (warm start)
        self._filter = WeightedMovingFilter(_OUTPUT_FILTER_WEIGHTS, self.nq)
        self._filter.add_data(self.last_q.copy())
        self.last_solve_ok = True
        self.last_solve_time_ms = 0.0

    # ------------------------------------------------------------------ API
    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Numerical forward kinematics.

        Returns:
            (p_ee, R_ee): EE position (3,) and rotation (3,3) of the frame
            ``EE_FRAME_NAME`` expressed in the URDF base frame.
        """
        data = self.model.createData()
        q = np.asarray(q, dtype=float).reshape(self.nq)
        pin.framesForwardKinematics(self.model, data, q)
        frame = data.oMf[self.frame_id]
        return frame.translation.copy(), frame.rotation.copy()

    def tool_axis_world(self, q: np.ndarray) -> np.ndarray:
        """Current tool pointing axis (unit vector) in base frame."""
        _, rot = self.fk(q)
        return rot @ np.asarray(TOOL_AXIS_LOCAL)

    def _roll_delta(self, target_pose: np.ndarray, q_current: np.ndarray) -> float:
        """Twist component (rad) of the target rotation about the tool axis,
        relative to the current EE rotation.

        wrist_roll spins the tool about its own axis: it moves neither the EE
        position nor the tool pointing axis, so the 5-dim IK task
        (3 position + 2 direction) cannot observe it — its IK value is driven
        only by the regularizer and drifts. The roll command therefore bypasses
        IK and is added onto the solved wrist_roll directly (see solve_ik)."""
        a = self.tool_axis_world(q_current)
        helper = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(a, helper)
        u /= np.linalg.norm(u)
        v = np.cross(a, u)

        R_cur = self.fk(q_current)[1]
        R_tgt = target_pose[:3, :3]
        uc, vc = R_cur @ u, R_cur @ v
        ut, vt = R_tgt @ u, R_tgt @ v
        cos_t = float(ut @ uc + vt @ vc)
        sin_t = float(ut @ vc - vt @ uc)
        return float(np.arctan2(sin_t, cos_t))

    def solve_ik(self, target_pose: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        """Solve IK for one control tick.

        Args:
            target_pose: 4x4 homogeneous EE target (base frame). Only the
                translation and the tool-axis column of the rotation are used.
            q_current: current joint angles (5,) rad — smoothness anchor and
                step-limit center.

        Returns:
            Filtered joint target (5,) rad. On solver failure the previous
            solution is returned (hold behavior, no jump).
        """
        import time

        q_current = np.asarray(q_current, dtype=float).reshape(self.nq)
        target_pose = np.asarray(target_pose, dtype=float)

        t0 = time.perf_counter()
        self.opti.set_initial(self.var_q, self.last_q)
        self.opti.set_value(self.param_q_last, q_current)
        self.opti.set_value(self.param_tf, target_pose)

        try:
            sol = self.opti.solve()
            q_target = np.array(sol.value(self.var_q)).flatten()
            # Roll bypass (successful solves only — a failed solve holds
            # everything, including roll): apply the controller's twist about
            # the tool axis directly onto wrist_roll, clamped to the URDF
            # range. wrist_roll cannot be seen by the position/direction costs,
            # so overriding it does not disturb the IK solution.
            q_target[4] = np.clip(
                q_current[4] + self._roll_delta(target_pose, q_current),
                self.lower_limit[4], self.upper_limit[4],
            )
            self.last_solve_ok = True
        except RuntimeError:
            # Keep the last good solution: holding position is the safe
            # fallback for teleoperation (same philosophy as R1_A7_ArmIK).
            q_target = self.last_q.copy()
            self.last_solve_ok = False
            logger.warning("IK solve failed — holding previous solution")

        self.last_q = q_target.copy()
        self._filter.add_data(q_target.tolist())
        q_out = np.array(self._filter.filtered_data).flatten()

        # Hard post-clamp: the filtered output is a lagging combination of past
        # raw targets, so it can drift beyond the per-tick step bound relative
        # to the hardware state. Clamp it so the commanded jump is guaranteed
        # bounded — the ratcheting toward the goal then happens naturally over
        # the following ticks.
        delta = np.clip(q_out - q_current, -MAX_STEP_RAD, MAX_STEP_RAD)
        q_out = q_current + delta

        self.last_solve_time_ms = (time.perf_counter() - t0) * 1e3
        return q_out
