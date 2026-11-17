#!/usr/bin/env python
"""Per-joint direction calibration for NexArm (real hardware).

Drives each joint by a small step from center and asks the operator whether
the observed motion matches the URDF-model expectation. At the end it prints
the JOINT_SIGN tuple to paste into src/lerobot/teleop_pico/config.py.

Safety: small steps (150 tick ~ 4.3 deg), slow ramp (30 Hz, servo speed
capped at 1000), torque off on Ctrl+C / exit.

Usage:
    conda activate nexarm_pico
    python examples/nexarm/calibrate_joint_sign.py --port /dev/ttyUSB0
"""

import argparse
import logging
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "lerobot"
sys.path.append(str(SRC))  # append: avoids shadowing stdlib modules

from lerobot.motors.nexarm.nexarm import JOINT_NAMES  # noqa: E402
from lerobot.robots.nexarm_follower import NexArmFollower, NexArmFollowerConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Expected EE motion per joint for a positive tick step (from the URDF model,
# measured near q=0 with the arm extended forward — see per-joint FK deltas).
# Operator only answers "same direction / reversed" per axis.
EXPECT = [
    # NOTE: "arm's left" = the arm's own left = the operator's RIGHT when facing
    # the arm. Stated both ways to avoid the viewing-frame mix-up that once
    # flipped pan's sign.
    ("shoulder_pan", "EE 水平平移——朝机械臂自己的左侧（你面对机械臂时，即朝你的右手边）"),
    ("shoulder_lift", "EE 向下 + 略前伸"),
    ("elbow_flex", "EE 向下 + 略收回"),
    ("wrist_flex", "EE 向下（小幅）"),
    ("wrist_roll", "EE 位置不动，夹爪绕工具轴自转——从夹爪前端朝臂身看为逆时针（右手系绕工具指向轴）"),
    ("gripper", "夹爪张开"),
]

STEP_TICK = 400   # ~23 deg per axis — large enough to see clearly
RAMP_STEPS = 40   # ~1.3 s per ramp (10 tick/frame)
RAMP_FPS = 30
OBSERVE_S = 3.0   # pause at the far end for observation


def parse_args():
    parser = argparse.ArgumentParser(description="NexArm joint-direction calibration")
    parser.add_argument("--port", required=True, help="Serial port (e.g. /dev/ttyUSB0)")
    return parser.parse_args()


def ramp_to(follower, goals: dict[str, int], current: dict[str, int]):
    """Ramp all joints toward goals by direct bus writes.

    Deliberately bypasses send_action(): its first-frame damper (designed for
    policy-inference gaps) rewrites goals back toward the cached position once
    the observation gap exceeds 0.2 s, which fights the slow calibration ramp
    and makes the arm twitch in place. Small bounded steps (<=10 tick/frame,
    stay within ~2048±400) are safe to write directly."""
    for s in range(1, RAMP_STEPS + 1):
        t = s / RAMP_STEPS
        frame = [int(round(current[k] + (goals[k] - current[k]) * t)) for k in JOINT_NAMES]
        follower.bus.write_positions(frame)
        time.sleep(1 / RAMP_FPS)


def main():
    args = parse_args()
    follower = NexArmFollower(
        NexArmFollowerConfig(port=args.port, cameras={}, disable_torque_on_disconnect=True)
    )
    follower.connect()
    print("\n=== NexArm 逐轴方向标定 ===")
    print("提示：连接时臂已回中 (2048)。每个关节将缓慢步进 +400 tick (~23°) 后停住")
    print("      供观察，随后自动返回中位。Ctrl+C 随时安全退出（卸扭矩）。\n")

    answers = []
    try:
        obs = follower.get_observation()
        rest = {name: int(obs[f"{name}.pos"]) for name, _ in EXPECT}

        for name, expect_desc in EXPECT:
            input(f"[{name}] 期望：{expect_desc}。按 Enter 开始点动...")
            base = {k: rest[k] for k in rest}
            obs = follower.get_observation()
            base = {k: int(obs[f"{k}.pos"]) for k, _ in EXPECT}  # re-read: stay at real pose
            far = dict(base)
            far[name] = base[name] + STEP_TICK
            ramp_to(follower, far, base)
            time.sleep(OBSERVE_S)
            ramp_to(follower, base, far)
            time.sleep(0.5)
            rest = dict(base)

            while True:
                ans = input(f"[{name}] 实际运动与期望同向? [y=同向 / n=反向]: ").strip().lower()
                if ans in ("y", "n"):
                    break
                print("  请输入 y 或 n")
            answers.append(ans == "y")

        signs = tuple(1 if ok else -1 for ok in answers)
        print("\n=== 标定结果 ===")
        for (name, _), ok in zip(EXPECT, answers):
            print(f"  {name:14s} {'同向 (+1)' if ok else '反向 (-1)'}")
        print(f"\n将 config.py 中 JOINT_SIGN 改为:\nJOINT_SIGN = {signs}")
        print("(wrist_roll 若难以观察，可先按模型默认 +1，后续再验证)")
    except KeyboardInterrupt:
        print("\n中止标定（已卸扭矩）。")
    finally:
        follower.disconnect()


if __name__ == "__main__":
    main()
