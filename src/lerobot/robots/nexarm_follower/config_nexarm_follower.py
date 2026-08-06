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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("nexarm_follower")
@dataclass
class NexArmFollowerConfig(RobotConfig):
    # Serial port for the slave ESP32 (e.g. "COM19", "/dev/ttyUSB1")
    port: str

    # Serial baud rate — NexArm uses 1 Mbps
    baudrate: int = 1_000_000

    disable_torque_on_disconnect: bool = True

    # Motion parameters sent via CMD 56 after connect.
    # acc: 0–254. 0 = max (no ramp), higher = softer ramp.
    # motion_speed: 0–3400 raw units/s. 0 = no limit.
    motion_acc: int = 100
    motion_speed: int = 2000
    safe_rest_positions: dict[str, float] = field(default_factory=dict)
    state_safety_margin: int = 100
    max_consecutive_anomalies: int = 5
    max_relative_target: float | dict[str, float] = 500.0

    # First-frame damper: smooths the transition from idle to active inference.
    # When the gap between last get_observation() and send_action() exceeds
    # first_frame_gap_threshold_s AND the positional delta exceeds
    # first_frame_min_delta, the damper activates for first_frame_warmup_frames
    # ticks, linearly ramping the execution factor from 0.1 to 1.0.
    # This prevents the "spasm" caused by the first action after a long
    # inference delay (e.g. ~556ms for SmolVLA's initial forward pass).
    first_frame_gap_threshold_s: float = 0.2
    first_frame_min_delta: int = 50
    first_frame_warmup_frames: int = 15

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
