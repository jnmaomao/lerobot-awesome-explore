# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NexArm follower (slave) robot — LeRobot Robot subclass.

Connects to the slave ESP32 via USB serial. Uses CMD 68 to enter bridge
mode, then CMD 96/97/98 through the AT32 co-processor to read/write
6-DOF servo positions.
"""

from __future__ import annotations

import contextlib
import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import MotorCalibration
from lerobot.motors.nexarm import NexArmMotorsBus
from lerobot.motors.nexarm.nexarm import JOINT_NAMES, POSITION_CENTER, POSITION_MAX, POSITION_MIN
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.robot_utils import precise_sleep

from ..robot import Robot
from .config_nexarm_follower import NexArmFollowerConfig

logger = logging.getLogger(__name__)


class NexArmFollower(Robot):
    """
    NexArm 6-DOF desktop robot arm (follower/slave).

    The follower connects to a slave ESP32 via USB serial at 1 Mbps. The ESP32
    forwards commands through an AT32F421 co-processor to HX-30HM serial bus
    servos. CMD 68 enters LeRobot bridge mode for direct servo access via
    CMD 96 (read positions) and CMD 97 (write positions).
    """

    config_class = NexArmFollowerConfig
    name = "nexarm_follower"

    def __init__(self, config: NexArmFollowerConfig):
        super().__init__(config)
        self.config = config
        self.bus = NexArmMotorsBus(port=config.port, baudrate=config.baudrate)
        self.cameras = make_cameras_from_configs(config.cameras)

        # Safety state for anomaly detection and recovery
        self._last_valid_obs: dict[str, Any] | None = None
        self._consecutive_anomaly_count: int = 0
        self._anomaly_logged: bool = False
        # Cache current positions from last observation (used for relative action limit)
        self._current_positions_cache: dict[str, float] | None = None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in JOINT_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def safe_rest_position(self) -> dict[str, float]:
        """Safe rest position for NexArm — all joints at center (2048).

        This is the position the robot should move to when returning to a
        "safe" pose on shutdown, instead of the raw zero (0) which causes
        rapid contraction. Each joint defaults to POSITION_CENTER (2048)
        but can be overridden per-joint via ``config.safe_rest_positions``.
        """
        defaults = {f"{name}.pos": float(POSITION_CENTER) for name in JOINT_NAMES}
        if self.config.safe_rest_positions:
            defaults.update(self.config.safe_rest_positions)
        return defaults

    @property
    def position_range(self) -> tuple[int, int]:
        """Min/max servo position range for this robot (0, 4095)."""
        return (POSITION_MIN, POSITION_MAX)

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return len(self.calibration) > 0 or self.calibration_fpath.is_file()

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        self.bus.enter_lerobot_mode()

        # Warm-up: discard initial frames that may contain stale/zero data.
        # After enter_lerobot_mode() the AT32 co-processor needs a few cycles
        # to start reporting real servo positions.  Without this, the first
        # read_positions() can return values near 0 (boundary anomaly).
        for _ in range(5):
            try:
                self.bus.read_positions()
            except TimeoutError:
                pass
            time.sleep(0.02)

        if calibrate and not self.is_calibrated:
            self.calibrate()
        elif self.calibration_fpath.is_file() and not self.calibration:
            self._load_calibration()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

        # Interpolate from current servo positions to safe rest / goal to avoid sudden jerky motion.
        if self.config.safe_rest_positions:
            goal_positions = [
                int(self.config.safe_rest_positions.get(f"{name}.pos", POSITION_CENTER))
                for name in JOINT_NAMES
            ]
        else:
            goal_positions = [POSITION_CENTER] * len(JOINT_NAMES)

        try:
            current_positions = self.bus.read_positions()
        except (TimeoutError, OSError):
            current_positions = [POSITION_CENTER] * len(JOINT_NAMES)

        _CONNECT_MOVE_STEPS = 50  # ~1s at 50Hz
        _CONNECT_MOVE_FPS = 50
        for step in range(1, _CONNECT_MOVE_STEPS + 1):
            t = step / _CONNECT_MOVE_STEPS
            interp = [int(round(cur * (1 - t) + goal * t)) for cur, goal in zip(current_positions, goal_positions)]
            self.bus.write_positions(interp)
            precise_sleep(1 / _CONNECT_MOVE_FPS)

        logger.info(f"{self} connected.")

    def calibrate(self) -> None:
        for i, name in enumerate(JOINT_NAMES):
            self.calibration[name] = MotorCalibration(
                id=i + 1,
                drive_mode=0,
                homing_offset=2048,
                range_min=POSITION_MIN,
                range_max=POSITION_MAX,
            )
        self._save_calibration()
        logger.info(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.write_motion_params(acc=self.config.motion_acc, speed=self.config.motion_speed)
        self.bus.set_torque(True)

    # Velocity watchdog: warn when a joint jumps more than this per tick.
    _MAX_POS_DELTA = 300

    # Positions suspiciously close to 0 or 4095 are treated as corrupt reads.
    # The firmware occasionally returns these on a dropped/corrupted packet.
    _CORRUPT_MARGIN = 5

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        raw = self.bus.read_positions()
        dt_ms = (time.perf_counter() - start) * 1e3

        prev = getattr(self, "_prev_positions", None)

        # Sanitize: replace corrupt reads (≤margin or ≥4095-margin) with prev value.
        positions = list(raw)
        for i, name in enumerate(JOINT_NAMES):
            p = raw[i]
            is_corrupt = p <= self._CORRUPT_MARGIN or p >= POSITION_MAX - self._CORRUPT_MARGIN
            if is_corrupt and prev is not None:
                logger.warning(
                    "servo %s corrupt read %d — using previous value %d",
                    name,
                    p,
                    prev[i],
                )
                positions[i] = prev[i]
            elif prev is not None:
                delta = abs(p - prev[i])
                if delta > self._MAX_POS_DELTA:
                    logger.warning(
                        "servo %s jumped %d raw units (prev=%d now=%d) - possible slip or lost packet",
                        name,
                        delta,
                        prev[i],
                        p,
                    )

        logger.debug(
            "servo_pos [%s]  read=%.1fms",
            " ".join(f"{name}={positions[i]}" for i, name in enumerate(JOINT_NAMES)),
            dt_ms,
        )

        self._prev_positions = positions
        obs_dict: RobotObservation = {
            f"{name}.pos": float(positions[i]) for i, name in enumerate(JOINT_NAMES)
        }

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        # State anomaly detection and recovery
        if self.config.state_safety_margin > 0:
            obs_dict = self._validate_and_recover_state(obs_dict)

        # Cache current positions for use in send_action() relative limit.
        # This MUST happen after anomaly recovery so that send_action() uses the
        # recovered (valid) positions, not the raw anomalous ones.
        # _validate_and_recover_state() also updates the cache when it replaces
        # values, but we set it here unconditionally so normal frames are cached too.
        self._current_positions_cache = {k: v for k, v in obs_dict.items() if k.endswith(".pos")}

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command arm to move to a target joint configuration.

        Args:
            action (RobotAction): The goal positions for the motors.

        Returns:
            RobotAction: The action sent to the motors.
        """
        goal = {}
        for name in JOINT_NAMES:
            val = float(action[f"{name}.pos"])
            goal[f"{name}.pos"] = max(POSITION_MIN, min(POSITION_MAX, int(round(val))))

        # Apply relative movement safety limit using cached positions from last get_observation()
        if self._current_positions_cache is not None and self.config.max_relative_target < POSITION_MAX:
            goal_present_pos = {}
            for name in JOINT_NAMES:
                key = f"{name}.pos"
                if key in self._current_positions_cache:
                    goal_present_pos[key] = (goal[key], self._current_positions_cache[key])
            if goal_present_pos:
                goal = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        goal_list = [goal[f"{name}.pos"] for name in JOINT_NAMES]
        self.bus.write_positions(goal_list)
        return {f"{name}.pos": float(goal_list[i]) for i, name in enumerate(JOINT_NAMES)}

    @check_if_not_connected
    def disconnect(self) -> None:
        for cam in self.cameras.values():
            with contextlib.suppress(Exception):
                cam.disconnect()

        if self.config.disable_torque_on_disconnect:
            with contextlib.suppress(Exception):
                # Hold current position briefly before killing torque.
                # Without this the servos lose power mid-air and drop.
                try:
                    positions = self.bus.read_positions()
                    self.bus.write_positions(positions)
                    time.sleep(0.4)
                except Exception:
                    pass
                self.bus.set_torque(False)

        with contextlib.suppress(Exception):
            self.bus.exit_lerobot_mode()

        self.bus.disconnect()
        logger.info(f"{self} disconnected.")

    def _validate_and_recover_state(self, obs: dict) -> dict:
        """Validate state values and recover from anomalies.

        Detects extreme boundary values (likely from drop-packet errors) and
        replaces them with the last valid state. Tracks consecutive anomalies
        and triggers a severe warning when threshold is exceeded.

        When recovery replaces joint values, the position cache
        (``_current_positions_cache``) is also updated so that
        ``send_action()`` clamps relative to the *recovered* positions,
        not the raw anomalous ones.
        """
        margin = self.config.state_safety_margin
        pos_min = POSITION_MIN + margin
        pos_max = POSITION_MAX - margin

        anomalous_joints = []
        for name in JOINT_NAMES:
            key = f"{name}.pos"
            val = obs.get(key)
            if val is not None and (val < pos_min or val > pos_max):
                anomalous_joints.append((name, val))

        if not anomalous_joints:
            self._last_valid_obs = obs.copy()
            if self._consecutive_anomaly_count > 0:
                logger.info(
                    "State values recovered to normal after %d anomalous frame(s)",
                    self._consecutive_anomaly_count,
                )
            self._consecutive_anomaly_count = 0
            self._anomaly_logged = False
            return obs

        # Anomaly detected
        self._consecutive_anomaly_count += 1
        anomaly_details = ", ".join(f"{n}={v:.0f}" for n, v in anomalous_joints)

        if self._consecutive_anomaly_count >= self.config.max_consecutive_anomalies:
            logger.error(
                "CRITICAL: %d consecutive anomalous frames detected! Affected joints: [%s]. "
                "This indicates a persistent communication issue or hardware fault. "
                "Consider checking serial connection and servo status.",
                self._consecutive_anomaly_count,
                anomaly_details,
            )
        elif not self._anomaly_logged:
            logger.warning(
                "State anomaly detected (frame %d): [%s]. "
                "Using last valid state for recovery. Safe range: [%d, %d]",
                self._consecutive_anomaly_count,
                anomaly_details,
                pos_min,
                pos_max,
            )
            self._anomaly_logged = True

        if self._last_valid_obs is not None:
            recovered_obs = self._last_valid_obs.copy()
            for key in obs:
                if key not in recovered_obs:
                    recovered_obs[key] = obs[key]
        else:
            # No valid cache yet — re-read from hardware instead of using
            # center positions.  Center (2048) is NOT the robot's actual
            # position; using it poisons _current_positions_cache and causes
            # ensure_safe_goal_position to clamp subsequent moves to wildly
            # wrong targets (e.g. shoulder_lift clamped to 500 instead of 2370).
            logger.warning(
                "No valid state cache available — re-reading from hardware"
            )
            for _retry in range(3):
                try:
                    positions = self.bus.read_positions()
                    retry_obs = {f"{name}.pos": float(positions[i]) for i, name in enumerate(JOINT_NAMES)}
                    # Check if the retry produced valid values
                    retry_anomalous = any(
                        v < pos_min or v > pos_max
                        for v in retry_obs.values()
                    )
                    if not retry_anomalous:
                        recovered_obs = obs.copy()
                        for name in JOINT_NAMES:
                            recovered_obs[f"{name}.pos"] = retry_obs[f"{name}.pos"]
                        break
                except Exception:
                    pass
            else:
                # All retries failed or still anomalous — fall back to center
                # but log prominently so the user knows the cache is unreliable
                logger.warning(
                    "Hardware re-read still anomalous after 3 retries — "
                    "falling back to center positions.  move_to_start_position "
                    "may produce incorrect clamping!"
                )
                recovered_obs = obs.copy()
                for name in JOINT_NAMES:
                    recovered_obs[f"{name}.pos"] = float(POSITION_CENTER)

        # Keep _current_positions_cache in sync with recovered values
        # so that send_action() clamps relative to the *recovered* positions,
        # not the raw anomalous ones.  Without this, a first-frame anomaly
        # (e.g. raw ≈ 0) would cause ensure_safe_goal_position to compute
        # wildly wrong clamps (e.g. goal 2050 clamped to 500).
        if self._current_positions_cache is None:
            self._current_positions_cache = {}
        for name in JOINT_NAMES:
            key = f"{name}.pos"
            if key in recovered_obs:
                self._current_positions_cache[key] = recovered_obs[key]

        return recovered_obs
