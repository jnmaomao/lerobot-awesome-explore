# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE_2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NexArm leader (master) teleoperator — LeRobot Teleoperator subclass.

Connects to the master ESP32 via USB serial. Reads servo positions via
CMD 96, maps them from leader space to follower space. The master runs
in torque-off mode so the operator can freely drag the arm.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from lerobot.motors import MotorCalibration
from lerobot.motors.nexarm import NexArmMotorsBus
from lerobot.motors.nexarm.nexarm import (
    JOINT_NAMES,
    POSITION_CENTER,
    POSITION_MAX,
    POSITION_MIN,
    map_leader_to_follower,
)
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_nexarm_leader import NexArmLeaderConfig

logger = logging.getLogger(__name__)

# Same margin as follower: positions near 0 or 4095 are physically impossible
# and always indicate a communication error.
_CORRUPT_MARGIN = 5
_CORRUPT_RETRY_COUNT = 1


class NexArmLeader(Teleoperator):
    config_class = NexArmLeaderConfig
    name = "nexarm_leader"

    def __init__(self, config: NexArmLeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus = NexArmMotorsBus(port=config.port, baudrate=config.baudrate)
        self._prev_leader_pos: list[int] | None = None

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{name}.pos": float for name in JOINT_NAMES}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @property
    def is_calibrated(self) -> bool:
        return len(self.calibration) > 0 or self.calibration_fpath.is_file()

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()

        # Read until we get valid positions — initialize _prev_leader_pos
        # so get_action() always has a fallback for corrupt reads.
        for attempt in range(10):
            try:
                raw = self.bus.read_positions()
                corrupt = [i for i, p in enumerate(raw) if p <= _CORRUPT_MARGIN or p >= POSITION_MAX - _CORRUPT_MARGIN]
                if not corrupt:
                    self._prev_leader_pos = raw
                    break
                logger.warning(
                    "Leader warm-up read (attempt %d): %d corrupt joints — %s",
                    attempt + 1,
                    len(corrupt),
                    " ".join(f"{JOINT_NAMES[i]}={raw[i]}" for i in corrupt),
                )
            except (TimeoutError, OSError):
                logger.warning("Leader warm-up read (attempt %d): timeout", attempt + 1)
            time.sleep(0.1)

        if self._prev_leader_pos is None:
            logger.error("Leader warm-up never produced valid reads — using center positions")
            self._prev_leader_pos = [POSITION_CENTER] * len(JOINT_NAMES)

        if calibrate and not self.is_calibrated:
            self.calibrate()
        elif self.calibration_fpath.is_file() and not self.calibration:
            self._load_calibration()

        self.configure()
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
        self.bus.set_torque(False)

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        leader_pos = self._read_positions_clean()
        follower_pos = map_leader_to_follower(leader_pos)
        return {f"{name}.pos": float(follower_pos[i]) for i, name in enumerate(JOINT_NAMES)}

    def _read_positions_clean(self) -> list[int]:
        """Read leader positions, rejecting corrupt values.

        Same strategy as NexArmFollower._read_positions_clean(): one retry
        on corrupt data, then fall back to previous values.  At most 1 extra
        read (~10ms) in the corrupt case.
        """
        try:
            raw = self.bus.read_positions()
        except (TimeoutError, OSError):
            logger.warning("Leader read_positions() timed out — using previous values")
            return list(self._prev_leader_pos)

        corrupt_indices = [i for i, p in enumerate(raw) if p <= _CORRUPT_MARGIN or p >= POSITION_MAX - _CORRUPT_MARGIN]
        if not corrupt_indices:
            self._prev_leader_pos = list(raw)
            return raw

        # One retry
        if _CORRUPT_RETRY_COUNT > 0:
            logger.debug(
                "Leader corrupt read: %s — retrying once",
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
                    if retry[i] <= _CORRUPT_MARGIN or retry[i] >= POSITION_MAX - _CORRUPT_MARGIN:
                        still_corrupt.append(i)
                corrupt_indices = still_corrupt
                raw = retry

                if not corrupt_indices:
                    self._prev_leader_pos = list(positions)
                    return positions

        # Replace remaining corrupt joints with previous values
        positions = list(raw)
        for i in corrupt_indices:
            logger.warning(
                "Leader servo %s corrupt read %d — using previous value %d",
                JOINT_NAMES[i], raw[i], self._prev_leader_pos[i],
            )
            positions[i] = self._prev_leader_pos[i]

        self._prev_leader_pos = list(positions)
        return positions

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")
