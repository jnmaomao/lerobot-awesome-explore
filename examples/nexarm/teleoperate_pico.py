#!/usr/bin/env python

# PICO4 teleoperation for NexArm: VR controller pose -> IK -> follower arm.
#
# Usage (hardware):
#   conda activate nexarm
#   python examples/nexarm/teleoperate_pico.py --follower-port COM19
#
# Dry run (no hardware; PICO still streams, commands are printed):
#   python examples/nexarm/teleoperate_pico.py --dry-run
#
# Activation semantics (same as the R1 project's grip-grab convention):
#   - Squeeze the RIGHT controller grip button -> arm follows the controller.
#     The target is anchored at the press edge, so activation never jumps.
#   - Release the grip (or stale XR data, or B button) -> HOLD, the arm keeps
#     its last commanded position.
#   - Trigger (index finger) -> gripper: squeeze = close, release = open.
#
# Requires the vuer page to be open on the PICO browser (WebXR), same setup
# as the r1-a7-autonomous-checkout project.

# OpenMP/BLAS thread pinning (Linux only) — MUST run before numpy/casadi imports.
# conda-forge casadi/ipopt (Linux) links MUMPS/spral built with OpenMP; on
# many-core CPUs every IK solve spawns one thread per core and the sync
# overhead (~20ms) dwarfs the 5-DOF solve itself (measured 21ms vs 0.8ms on a
# 24-core machine). Windows pip casadi ships a serial IPOPT/MUMPS build and is
# unaffected, so Win11 behavior is left untouched. setdefault so an explicit
# shell env still wins.
import os
if os.name != "nt":
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ.setdefault(_var, "1")
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "src" / "lerobot"
sys.path.append(str(SRC))  # append: avoids shadowing stdlib modules

from teleop_pico.config import (  # noqa: E402
    SAFETY_CAGE_ENABLE,
    SAFETY_LEVEL,
    SAFETY_LEVELS,
    TELEOP_POS_GAIN,
    TELEOP_ROT_GAIN,
)
from teleop_pico.alignment import RelativeAlignment  # noqa: E402
from teleop_pico.nexarm_ik import NexarmIK  # noqa: E402
from teleop_pico.unit_convert import rad_to_tick, tick_to_rad, trigger_to_gripper  # noqa: E402

try:
    from lerobot.utils.robot_utils import precise_sleep
except ImportError:
    def precise_sleep(duration):  # dry-run fallback (lerobot not installed)
        time.sleep(max(0.0, duration))

XR_STALE_S = 0.25  # XR data older than this -> HOLD (same as R1 project)


def parse_args():
    parser = argparse.ArgumentParser(description="PICO4 teleop for NexArm")
    parser.add_argument("--follower-port", default=None, help="Serial port for follower ESP32 (e.g. COM19)")
    parser.add_argument("--dry-run", action="store_true", help="No hardware: print commands instead of sending")
    parser.add_argument("--no-camera", action="store_true",
                        help="Skip camera connect (pass-through display does not need it)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--blend-seconds", type=float, default=0.0,
                        help="Fade duration to absorb the initial offset (0 = fixed-relative, no fade)")
    parser.add_argument("--stale-s", type=float, default=XR_STALE_S)
    parser.add_argument("--safety-level", type=int, default=None, choices=[0, 1, 2, 3],
                        help="Override SAFETY_LEVEL from config.py (0 tightest, 3 = cage off)")
    parser.add_argument("--pos-gain", type=float, default=None,
                        help="Override TELEOP_POS_GAIN (raise together with cage level)")
    parser.add_argument("--rot-gain", type=float, default=None,
                        help="Override TELEOP_ROT_GAIN (raise together with cage level)")
    return parser.parse_args()


class VirtualArm:
    """Dry-run stand-in for NexArmFollower: perfect servo model (jumps allowed)."""

    def __init__(self, n_joints: int = 5):
        self.q_rad = np.zeros(n_joints)
        self.gripper_tick = 2048.0

    def apply(self, arm_ticks: list[int], gripper_tick: int):
        self.q_rad = tick_to_rad(arm_ticks)
        self.gripper_tick = float(gripper_tick)

    def q_current(self) -> np.ndarray:
        return self.q_rad.copy()


def main():
    args = parse_args()
    if not args.dry_run and not args.follower_port:
        sys.exit("error: --follower-port is required unless --dry-run")

    # Runtime overrides for progressive hardware bring-up (config.py values
    # stay the defaults, so existing Win11 behavior is untouched).
    global SAFETY_LEVEL, TELEOP_POS_GAIN, TELEOP_ROT_GAIN
    if args.safety_level is not None:
        SAFETY_LEVEL = args.safety_level
    if args.pos_gain is not None:
        TELEOP_POS_GAIN = args.pos_gain
    if args.rot_gain is not None:
        TELEOP_ROT_GAIN = args.rot_gain

    ik = NexarmIK(verbose=False)
    print(f"[IK] nq={ik.nq}, EE frame='{ik.model.frames[ik.frame_id].name}', "
          f"limits ±{ik.model.upperPositionLimit[0]:.2f} rad")

    # --- hardware / virtual follower -------------------------------------
    follower = None
    if not args.dry_run:
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.nexarm_follower import NexArmFollower, NexArmFollowerConfig

        camera_config = (
            {"front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=args.fps)}
            if not args.no_camera else {}
        )
        follower = NexArmFollower(
            NexArmFollowerConfig(port=args.follower_port, cameras=camera_config,
                                 disable_torque_on_disconnect=True)
        )
        follower.connect()

    virtual = VirtualArm(ik.nq)

    def current_q_rad(obs: dict | None) -> np.ndarray:
        """Current arm joint angles in rad, from hardware obs or the virtual arm."""
        if follower is not None and obs is not None:
            arm_keys = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                        "wrist_flex.pos", "wrist_roll.pos"]
            return tick_to_rad([obs[k] for k in arm_keys])
        return virtual.q_current()

    # --- XR input ----------------------------------------------------------
    from teleop_pico.televuer import TeleVuerWrapper  # heavy import: only needed now

    xr = TeleVuerWrapper(use_hand_tracking=False, binocular=False,
                         img_shape=(360, 640), display_mode="pass-through",
                         zmq=False, webrtc=False, arm_reference_mode="head_yaw")

    state = "HOLD"           # HOLD | ACTIVE
    alignment: RelativeAlignment | None = None
    prev_squeeze = False
    last_print = 0.0
    tick = 0
    q_anchor: np.ndarray | None = None   # safety-cage anchor (pose at activation)
    grip_anchor: int | None = None       # safety-cage gripper anchor (tick at activation)
    q_prev_cmd: np.ndarray | None = None # previous commanded q (step limiting)
    # XR event-rate diagnostics: track inter-arrival gaps of controller events
    prev_motion_ts = 0.0
    event_dts = []                       # rolling window of recent event gaps (s)

    cage = cage_limits()
    if cage is not None:
        max_dev, max_step, grip_half = cage
        print(f"[SAFETY CAGE] level={SAFETY_LEVEL} dev=±{max_dev:.3f}rad/axis "
              f"step<={max_step:.3f}rad/tick grip=anchor±{grip_half}tick")
    print(f"[TELEOP GAIN] pos={TELEOP_POS_GAIN} rot={TELEOP_ROT_GAIN} "
          f"(hand motion scaled before IK)")

    lan_ip = _detect_lan_ip()
    print("Waiting for PICO motion data...")
    print(f"PICO browser:  https://{lan_ip}:8012/?ws=wss://{lan_ip}:8012")
    print("  (accept the self-signed certificate warning, then start the XR session)")
    print("Control: squeeze RIGHT grip = follow, release = HOLD, B = HOLD, trigger = gripper.")

    try:
        while True:
            start = time.perf_counter()
            tick += 1

            tele = xr.get_tele_data()
            now = time.monotonic()
            age = now - tele.motion_data_timestamp
            # event-rate diagnostics: record gap between consecutive XR events
            ts = tele.motion_data_timestamp
            if ts != prev_motion_ts:
                if prev_motion_ts > 0:
                    event_dts.append(max(0.0, ts - prev_motion_ts))
                    if len(event_dts) > 60:
                        event_dts.pop(0)
                prev_motion_ts = ts
            fresh = bool(tele.motion_data_ready) and 0 <= age <= args.stale_s

            # Right grip activation with hysteresis (same as R1 update_start_button).
            value = float(getattr(tele, "right_ctrl_squeezeValue", 0.0))
            pressed = bool(tele.right_ctrl_squeeze) or value >= 0.8
            if not pressed and value > 0.2:
                pressed = prev_squeeze  # hysteresis against partially held grip
            press_edge = pressed and not prev_squeeze
            prev_squeeze = pressed

            # Observed state (hardware read or virtual model)
            obs = follower.get_observation() if follower is not None else None
            q_cur = current_q_rad(obs)

            emergency = bool(getattr(tele, "right_ctrl_bButton", False))

            if state == "HOLD":
                if emergency:
                    reason = "B button"
                elif not fresh:
                    reason = f"stale XR (age={age:.2f}s)"
                elif press_edge:
                    # Activate: anchor at current pose — zero jump by construction.
                    robot_pose = np.eye(4)
                    robot_pose[:3, :3] = _ee_rotation(ik, q_cur)
                    robot_pose[:3, 3] = _ee_position(ik, q_cur)
                    try:
                        alignment = RelativeAlignment(
                            robot_pose, tele.right_wrist_pose,
                            blend_seconds=args.blend_seconds,
                            pos_gain=TELEOP_POS_GAIN, rot_gain=TELEOP_ROT_GAIN)
                        ik.last_q = q_cur.copy()
                        ik._filter._data_queue = []
                        ik._filter.add_data(q_cur.copy())
                        q_anchor = q_cur.copy()
                        q_prev_cmd = q_cur.copy()
                        grip_anchor = trigger_to_gripper(tele.right_ctrl_triggerValue)
                        state = "ACTIVE"
                        print(f"[{tick}] ACTIVE — anchored at q={np.round(q_cur, 3)}")
                    except ValueError as exc:
                        print(f"[{tick}] activation rejected: {exc}")
                    reason = None
                if state == "HOLD" and tick - last_print >= 60:
                    last_print = tick
                    if event_dts:
                        mean_dt = sum(event_dts) / len(event_dts)
                        rate = 1.0 / mean_dt if mean_dt > 1e-3 else 0.0
                        print(f"[{tick}] HOLD ({reason or 'waiting for grip'}) "
                              f"[events: n={len(event_dts)} mean={mean_dt * 1e3:.0f}ms "
                              f"last={event_dts[-1] * 1e3:.0f}ms rate={rate:.1f}Hz]")
                    else:
                        print(f"[{tick}] HOLD ({reason or 'waiting for grip'}) "
                              f"[events: none]")

            if state == "ACTIVE":
                if not pressed:
                    state = "HOLD"
                    print(f"[{tick}] HOLD — grip released")
                elif emergency:
                    state = "HOLD"
                    print(f"[{tick}] HOLD — B button")
                elif not fresh:
                    state = "HOLD"
                    print(f"[{tick}] HOLD — stale XR (age={age:.2f}s)")
                else:
                    try:
                        target = alignment.poses(tele.right_wrist_pose)
                    except ValueError as exc:
                        state = "HOLD"
                        print(f"[{tick}] HOLD — {exc}")
                        target = None
                    if target is not None:
                        q_cmd = ik.solve_ik(target, q_cur)
                        grip_tick = trigger_to_gripper(tele.right_ctrl_triggerValue)
                        cage_reason = ""
                        if cage is not None:
                            q_cmd, grip_tick, cage_reason = apply_safety_cage(
                                q_cmd, grip_tick, q_anchor, grip_anchor, q_prev_cmd)
                        q_prev_cmd = q_cmd.copy()
                        if cage_reason and tick - last_print >= 30:
                            last_print = tick
                            print(f"[{tick}] CAGE — output clipped ({cage_reason})")
                        arm_ticks = rad_to_tick(q_cmd)
                        if follower is not None:
                            action = {
                                "shoulder_pan.pos": arm_ticks[0],
                                "shoulder_lift.pos": arm_ticks[1],
                                "elbow_flex.pos": arm_ticks[2],
                                "wrist_flex.pos": arm_ticks[3],
                                "wrist_roll.pos": arm_ticks[4],
                                "gripper.pos": grip_tick,
                            }
                            follower.send_action(action)
                        else:
                            virtual.apply(arm_ticks, grip_tick)
                        if tick - last_print >= 30:
                            last_print = tick
                            ok = "ok" if ik.last_solve_ok else "IK-FALLBACK"
                            print(f"[{tick}] {ok} ik={ik.last_solve_time_ms:.1f}ms "
                                  f"q={np.round(q_cmd, 3)} grip={grip_tick}")

            precise_sleep(1.0 / args.fps - (time.perf_counter() - start))
    except KeyboardInterrupt:
        print("\nStopping PICO teleoperation.")
    finally:
        if follower is not None:
            # Hold a moment at the current position before cutting torque so
            # the arm does not drop mid-air (same as NexArmFollower.disconnect).
            try:
                positions = follower.bus.read_positions()
                follower.bus.write_positions(positions)
                time.sleep(0.4)
            except Exception:
                pass
            follower.disconnect()
        xr.close()


def _ee_position(ik: NexarmIK, q: np.ndarray) -> np.ndarray:
    p, _ = ik.fk(q)
    return p


def _ee_rotation(ik: NexarmIK, q: np.ndarray) -> np.ndarray:
    _, R = ik.fk(q)
    return R


def _detect_lan_ip() -> str:
    """Best-effort LAN IPv4 (no packets actually sent; UDP connect only picks the route)."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def cage_limits():
    """Return (max_dev_rad, max_step_rad, grip_half_span) for the configured
    safety level, or None when the cage is disabled/off."""
    if not SAFETY_CAGE_ENABLE:
        return None
    limits = SAFETY_LEVELS.get(SAFETY_LEVEL)
    if limits is None:
        return None
    return limits["max_dev_rad"], limits["max_step_rad"], limits["grip_half_span"]


def apply_safety_cage(
    q_cmd: np.ndarray,
    grip_tick: int,
    q_anchor: np.ndarray,
    grip_anchor: int,
    q_prev_cmd: np.ndarray | None,
) -> tuple[np.ndarray, int, str]:
    """Clamp IK output into the safety cage: anchor box + per-tick step limit
    + gripper band around the activation-time grip value. Returns
    (clamped_q, clamped_grip, clip_reason); clip_reason names the limits that
    engaged ("box"/"step"/"grip", "" when free)."""
    max_dev, max_step, grip_half = cage_limits()
    q_box = np.clip(q_cmd, q_anchor - max_dev, q_anchor + max_dev)
    box_hit = not np.array_equal(q_box, q_cmd)
    if q_prev_cmd is not None:
        q_step = np.clip(q_box, q_prev_cmd - max_step, q_prev_cmd + max_step)
    else:
        q_step = q_box
    step_hit = not np.array_equal(q_step, q_box)
    grip_out = int(np.clip(grip_tick, grip_anchor - grip_half, grip_anchor + grip_half))
    grip_hit = grip_out != grip_tick
    reasons = [name for name, hit in (("box", box_hit), ("step", step_hit)) if hit]
    if grip_hit:
        reasons.append("grip")
    return q_step, grip_out, "+".join(reasons)


if __name__ == "__main__":
    main()
