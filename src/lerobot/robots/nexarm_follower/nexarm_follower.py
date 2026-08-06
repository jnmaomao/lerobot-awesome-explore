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

        # First-frame damper state
        self._last_obs_time: float | None = None
        self._damp_frame_count: int = 0

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
        #
        # We validate the warm-up reads: if all joints are 0 (or near-boundary)
        # after the initial burst, we wait longer and retry.  The AT32 may need
        # up to ~1s to fully initialize its servo bus after entering bridge mode.
        _WARMUP_MAX_ATTEMPTS = 5
        _WARMUP_READS_PER_ATTEMPT = 5
        _WARMUP_READ_GAP_S = 0.02
        _WARMUP_RETRY_GAP_S = 0.3

        warmup_ok = False
        warmup_positions = None
        for attempt in range(_WARMUP_MAX_ATTEMPTS):
            for _ in range(_WARMUP_READS_PER_ATTEMPT):
                try:
                    self.bus.read_positions()
                except TimeoutError:
                    pass
                time.sleep(_WARMUP_READ_GAP_S)

            # After warm-up burst, check if we get valid reads
            try:
                positions = self.bus.read_positions()
                # Reject corrupt values (0 or near-boundary) — these are never real
                valid_count = sum(
                    1 for p in positions
                    if not self._is_corrupt(p)
                )
                if valid_count == len(JOINT_NAMES):
                    logger.info(
                        "Warm-up read OK (attempt %d): all joints valid — %s",
                        attempt + 1,
                        " ".join(f"{n}={positions[i]}" for i, n in enumerate(JOINT_NAMES)),
                    )
                    warmup_ok = True
                    warmup_positions = positions
                    break
                else:
                    logger.warning(
                        "Warm-up read (attempt %d): %d/%d corrupt joints — %s. "
                        "Retrying in %.1fs…",
                        attempt + 1,
                        len(JOINT_NAMES) - valid_count,
                        len(JOINT_NAMES),
                        " ".join(f"{n}={positions[i]}" for i, n in enumerate(JOINT_NAMES)),
                        _WARMUP_RETRY_GAP_S,
                    )
            except TimeoutError:
                logger.warning(
                    "Warm-up read (attempt %d): timeout. Retrying in %.1fs…",
                    attempt + 1,
                    _WARMUP_RETRY_GAP_S,
                )

            time.sleep(_WARMUP_RETRY_GAP_S)

        if not warmup_ok:
            logger.error(
                "CRITICAL: Warm-up never produced valid servo reads after %d attempts! "
                "The AT32 co-processor may not be communicating with the servos. "
                "Check: 1) servo power 2) servo cable connections 3) AT32 firmware",
                _WARMUP_MAX_ATTEMPTS,
            )

        # Initialize _prev_positions with the first valid real readings.
        # This ensures get_observation()'s corrupt-read detection always has
        # a previous value to fall back on — _prev_positions is never None.
        if warmup_positions is not None:
            self._prev_positions = warmup_positions
        else:
            # Warm-up failed — make one more attempt with retries
            try:
                self._prev_positions = self._read_positions_clean()
            except (TimeoutError, OSError):
                logger.error("Cannot read servo positions — initializing prev to center")
                self._prev_positions = [POSITION_CENTER] * len(JOINT_NAMES)

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

        # Use the already-validated positions from warm-up/initialization
        # as the interpolation start point (no need to re-read).
        current_positions = self._prev_positions
        has_corrupt = any(self._is_corrupt(p) for p in current_positions)

        if has_corrupt:
            logger.warning(
                "Current positions contain corrupt values — skipping interpolation "
                "to avoid violent motion: %s",
                " ".join(f"{n}={current_positions[i]}" for i, n in enumerate(JOINT_NAMES)),
            )
        else:
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

    # Positions at or near 0 or 4095 are physically impossible for real servos
    # and always indicate a communication error (dropped/corrupted packet,
    # AT32 not yet initialized, etc.).  We never accept these values — instead
    # we re-read from hardware until we get real positions.
    _CORRUPT_MARGIN = 5
    _CORRUPT_RETRY_COUNT = 1  # re-read once on corrupt; prev is the fast fallback

    def _is_corrupt(self, p: int) -> bool:
        """True if a position value is a communication artifact, not a real reading."""
        return p <= self._CORRUPT_MARGIN or p >= POSITION_MAX - self._CORRUPT_MARGIN

    def _read_positions_clean(self) -> list[int]:
        """Read servo positions, rejecting corrupt values.

        Values of 0 or 4095 are physically impossible and always indicate a
        communication error.  Strategy:
          1. Read from hardware.
          2. If any joint is corrupt, re-read ONCE (cheap, ~10ms).
          3. For joints still corrupt after retry, replace with previous
             frame's value (guaranteed valid after connect() init).

        This avoids retry storms: at most 1 extra read (~10ms), and the
        fallback to prev is instant.  Total overhead in the corrupt case
        is ~10ms, well within the 33ms budget at 30 FPS.
        """
        prev = getattr(self, "_prev_positions", None)
        margin = self.config.state_safety_margin

        try:
            raw = self.bus.read_positions()
        except (TimeoutError, OSError):
            # Timeout is rare and expensive (150ms); don't retry, use prev
            if prev is not None:
                logger.warning("read_positions() timed out — using previous values")
                return list(prev)
            raise

        corrupt_indices = [i for i, p in enumerate(raw) if self._is_corrupt(p)]
        if not corrupt_indices:
            return raw

        # One retry — if it fixes all corrupt joints, great; otherwise
        # fall back to prev for the remaining ones (no further retries)
        if self._CORRUPT_RETRY_COUNT > 0:
            logger.debug(
                "Corrupt read: %s — retrying once",
                " ".join(f"{JOINT_NAMES[i]}={raw[i]}" for i in corrupt_indices),
            )
            time.sleep(0.005)
            try:
                retry = self.bus.read_positions()
            except (TimeoutError, OSError):
                retry = None

            if retry is not None:
                still_corrupt = []
                positions = list(retry)
                for i in corrupt_indices:
                    if self._is_corrupt(retry[i]):
                        still_corrupt.append(i)
                    # else: retry gave a valid value for this joint — keep it
                corrupt_indices = still_corrupt
                raw = retry  # use retry as base for prev fallback

                if not corrupt_indices:
                    return positions

        # Replace remaining corrupt joints with previous values
        positions = list(raw)
        if prev is not None:
            for i in corrupt_indices:
                logger.warning(
                    "servo %s corrupt read %d — using previous value %d",
                    JOINT_NAMES[i], raw[i], prev[i],
                )
                positions[i] = prev[i]
        else:
            for i in corrupt_indices:
                logger.error(
                    "servo %s corrupt read %d with no previous value — clamping to safe boundary",
                    JOINT_NAMES[i], raw[i],
                )
                positions[i] = POSITION_MIN + margin if raw[i] <= self._CORRUPT_MARGIN else POSITION_MAX - margin

        return positions

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        start = time.perf_counter()
        positions = self._read_positions_clean()
        dt_ms = (time.perf_counter() - start) * 1e3

        prev = getattr(self, "_prev_positions", None)
        if prev is not None:
            for i, name in enumerate(JOINT_NAMES):
                delta = abs(positions[i] - prev[i])
                if delta > 300:
                    logger.warning(
                        "servo %s jumped %d raw units (prev=%d now=%d) - possible slip or lost packet",
                        name, delta, prev[i], positions[i],
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

        # Record observation timestamp for first-frame damper in send_action()
        self._last_obs_time = time.perf_counter()

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command arm to move to a target joint configuration.

        Two-layer safety:
          Layer 1 — Absolute goal clamping: goal is never sent to the danger
            zone [0, margin] or [4095-margin, 4095].  This directly prevents
            the motor from being commanded to position 0 (violent contraction)
            regardless of cache state.
          Layer 2 — Relative movement limit (ensure_safe_goal_position): caps
            the per-tick displacement to max_relative_target.  This prevents
            erroneous high-speed commands from damaging the arm.  Only applied
            when _current_positions_cache is trustworthy (i.e. we have had at
            least one valid observation).

        Args:
            action (RobotAction): The goal positions for the motors.

        Returns:
            RobotAction: The action sent to the motors.
        """
        margin = self.config.state_safety_margin
        safe_min = POSITION_MIN + margin   # 100
        safe_max = POSITION_MAX - margin   # 3995

        # --- Layer 1: Absolute goal clamping ---
        # Clamp goal to safe range [safe_min, safe_max].  This is the primary
        # defense against position-0 commands that cause violent contraction.
        # Unlike the relative limit, absolute clamping does NOT depend on
        # _current_positions_cache, so it works correctly even when the cache
        # is poisoned by anomalous reads.
        goal = {}
        for name in JOINT_NAMES:
            val = float(action[f"{name}.pos"])
            clamped = max(safe_min, min(safe_max, int(round(val))))
            goal[f"{name}.pos"] = clamped
            if clamped != int(round(val)):
                logger.warning(
                    "Goal %s=%.0f clamped to safe range [%d, %d] → %d",
                    name, val, safe_min, safe_max, clamped,
                )

        # --- First-frame damper ---
        # Prevents the "spasm" that occurs when the first action after a long
        # inference delay (e.g. ~556ms for SmolVLA's initial forward pass)
        # causes the robot to suddenly jump to the policy's target position.
        #
        # Detection (dual-condition to avoid false positives in teleop/record):
        #   1. gap_since_obs > first_frame_gap_threshold_s  (time since last observation)
        #   2. max_delta > first_frame_min_delta            (positional change magnitude)
        #
        # Behaviour: linearly ramp the execution factor from 0.1 to 1.0 over
        # first_frame_warmup_frames ticks.  After warmup, full-speed execution
        # resumes.
        #
        # Guard: only apply when _last_valid_obs exists (at least one valid
        # observation established).  Without this, the damper interpolates
        # FROM a poisoned cache (e.g. center=2048 instead of real=1800),
        # pulling the goal AWAY from the correct target.
        now = time.perf_counter()
        gap_since_obs = (now - self._last_obs_time) if self._last_obs_time else 0.0

        # Compute max positional delta between goal and current position
        max_delta = 0
        cache_valid = self._current_positions_cache is not None and self._last_valid_obs is not None
        if cache_valid:
            for name in JOINT_NAMES:
                key = f"{name}.pos"
                if key in self._current_positions_cache and key in goal:
                    delta = abs(goal[key] - self._current_positions_cache[key])
                    max_delta = max(max_delta, delta)

        # Activate damper when: long gap since observation + significant positional change + cache valid
        cfg = self.config
        if (cache_valid
                and gap_since_obs > cfg.first_frame_gap_threshold_s
                and max_delta > cfg.first_frame_min_delta):
            self._damp_frame_count = 0
            logger.info(
                "First-frame damper activated (gap=%.0fms, max_delta=%d) — "
                "damping next %d frames",
                gap_since_obs * 1000,
                max_delta,
                cfg.first_frame_warmup_frames,
            )

        # Apply damping: linearly ramp execution factor from 0.1 to 1.0
        if (self._damp_frame_count < cfg.first_frame_warmup_frames
                and cache_valid):
            warmup = cfg.first_frame_warmup_frames
            min_factor = 0.1
            factor = min_factor + (1.0 - min_factor) * (self._damp_frame_count / warmup)
            for name in JOINT_NAMES:
                key = f"{name}.pos"
                if key in self._current_positions_cache and key in goal:
                    current = self._current_positions_cache[key]
                    delta = goal[key] - current
                    goal[key] = current + delta * factor
            self._damp_frame_count += 1
            if self._damp_frame_count >= warmup:
                logger.info("First-frame damper released — full speed resumed")
        # --- End first-frame damper ---

        # --- Layer 2: Relative movement safety limit ---
        # Cap per-tick displacement to prevent erroneous high-speed commands.
        # Only apply when the cache is trustworthy (we have had at least one
        # valid observation).  When the cache is poisoned (anomalous values),
        # skip the relative limit — clamping relative to a wrong present_pos
        # is worse than no clamping at all.  Layer 1 (absolute clamping) still
        # protects against extreme goal values.
        cache_valid = (
            self._current_positions_cache is not None
            and self._last_valid_obs is not None  # at least one valid observation established
            and self.config.max_relative_target < POSITION_MAX
        )
        if cache_valid:
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

        Checks whether any joint position falls within the extended danger zone
        [0, margin] or [4095-margin, 4095].  Corrupt reads (0 or 4095) should
        already be handled by _read_positions_clean(); this catches values in
        the wider margin (e.g. [6, 99] or [3996, 4090]) that pass the corrupt
        check but are still physically dangerous.

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

        # Anomaly detected — replace with last valid values
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

        recovered_obs = obs.copy()
        if self._last_valid_obs is not None:
            # Replace anomalous joints with last valid values
            for name, _ in anomalous_joints:
                key = f"{name}.pos"
                if key in self._last_valid_obs:
                    recovered_obs[key] = self._last_valid_obs[key]
        else:
            # No valid cache — clamp to safe boundary as last resort
            for name, val in anomalous_joints:
                key = f"{name}.pos"
                recovered_obs[key] = pos_min if val < pos_min else pos_max

        # Keep _current_positions_cache in sync with recovered values
        if self._current_positions_cache is None:
            self._current_positions_cache = {}
        for name in JOINT_NAMES:
            key = f"{name}.pos"
            if key in recovered_obs:
                self._current_positions_cache[key] = recovered_obs[key]

        return recovered_obs
