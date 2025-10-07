#!/usr/bin/env python

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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("heuristic")
@dataclass
class HeuristicConfig(PreTrainedConfig):
    """Configuration class for Heuristic Policy.

    This is a rule-based policy that doesn't require training.
    You can customize the heuristic parameters here.

    Args:
        n_obs_steps: Number of observation steps (default: 1, no history needed for heuristics)
        chunk_size: Action chunk size (default: 1, heuristics typically output single actions)
        n_action_steps: Number of action steps to execute (default: 1)
        gripper_threshold: Threshold for gripper open/close decision
        speed_factor: Multiplier for action speed
        use_vision: Whether to use vision-based heuristics (requires OpenCV)
        target_color: Target color to detect in HSV format [H_min, S_min, V_min, H_max, S_max, V_max]
    """

    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 1
    n_action_steps: int = 1

    # Normalization (usually not needed for heuristics, but kept for compatibility)
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Heuristic-specific parameters
    gripper_threshold: float = 0.5
    speed_factor: float = 1.0
    use_vision: bool = False

    # Vision parameters (HSV color range for red object detection)
    target_color_lower: list[int] = field(default_factory=lambda: [0, 100, 100])
    target_color_upper: list[int] = field(default_factory=lambda: [10, 255, 255])

    # Control gains
    position_gain: float = 0.1

    # Waypoint-based control
    waypoints_path: str | None = None
    waypoint_tolerance: float = 0.01  # Distance threshold to consider waypoint reached
    waypoint_speed: float = 0.1  # Speed multiplier when moving to waypoint

    # Training (not used for heuristics, but required by base class)
    optimizer_lr: float = 0.0  # No training
    optimizer_weight_decay: float = 0.0

    def __post_init__(self):
        super().__post_init__()

        # Validation
        if self.gripper_threshold < 0 or self.gripper_threshold > 1:
            raise ValueError(f"gripper_threshold must be in [0, 1], got {self.gripper_threshold}")

    def get_optimizer_preset(self) -> None:
        # Heuristic policies don't need optimization
        return None

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        # Heuristic policy needs at least state observations
        if not self.state_feature:
            raise ValueError("Heuristic policy requires 'observation.state' feature")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
