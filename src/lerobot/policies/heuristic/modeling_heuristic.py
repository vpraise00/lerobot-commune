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

"""
Heuristic Policy - A rule-based policy for robot control.

This policy implements simple heuristics and can be extended with:
- Vision-based control (object detection, tracking)
- State-based rules (threshold-based decisions)
- Custom control logic

Example usage:
```python
from lerobot.policies.heuristic import HeuristicPolicy, HeuristicConfig

config = HeuristicConfig(
    gripper_threshold=0.5,
    speed_factor=1.0,
    use_vision=True
)
policy = HeuristicPolicy(config)

# In control loop
action = policy.select_action(observation)
```
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.heuristic.configuration_heuristic import HeuristicConfig
from lerobot.policies.pretrained import PreTrainedPolicy

logger = logging.getLogger(__name__)


class HeuristicPolicy(PreTrainedPolicy):
    """
    Heuristic Policy - Rule-based control without learning.

    This policy can be used for:
    1. Teleoperation data collection (with simple automation)
    2. Baseline comparisons
    3. Safety fallbacks
    4. Testing robot setups
    """

    config_class = HeuristicConfig
    name = "heuristic"

    def __init__(self, config: HeuristicConfig):
        """
        Args:
            config: HeuristicConfig instance with policy parameters
        """
        super().__init__(config)
        self.config = config

        # Initialize vision processor if needed
        if self.config.use_vision:
            try:
                import cv2

                self.cv2 = cv2
                logger.info("Vision-based heuristics enabled (OpenCV loaded)")
            except ImportError:
                logger.warning(
                    "OpenCV not found. Vision-based heuristics disabled. "
                    "Install with: pip install opencv-python"
                )
                self.config.use_vision = False
                self.cv2 = None
        else:
            self.cv2 = None

        self.reset()
        logger.info(f"Initialized {self.name} policy with config: {self.config}")

    def reset(self):
        """Reset policy state (called when environment resets)"""
        # Add any stateful variables here if needed
        self._step_count = 0
        logger.debug("Policy reset")

    def get_optim_params(self) -> dict:
        """Heuristic policies don't have trainable parameters"""
        return {}

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """
        Forward pass for training (not used for heuristics).

        Heuristic policies are rule-based and don't require training.
        This method returns zero loss for compatibility.
        """
        loss = torch.tensor(0.0, device=batch[list(batch.keys())[0]].device)
        info = {"heuristic_loss": 0.0}
        return loss, info

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Select action based on heuristic rules.

        Args:
            batch: Dictionary containing observations:
                - 'observation.state': Robot joint positions (required)
                - 'observation.images.*': Camera images (optional, if use_vision=True)

        Returns:
            Tensor: Action to execute, shape (batch_size, action_dim)
        """
        self.eval()
        self._step_count += 1

        # Get state from batch
        state = batch.get("observation.state")
        if state is None:
            raise ValueError("observation.state is required for heuristic policy")

        batch_size = state.shape[0]
        action_dim = state.shape[1]  # Assume action_dim == state_dim

        # Initialize action as current state (no-op baseline)
        action = state.clone()

        # Apply heuristic rules
        if self.config.use_vision and self.cv2 is not None:
            # Vision-based control
            action = self._vision_based_control(batch, action)
        else:
            # Simple state-based control
            action = self._state_based_control(state, action)

        # Apply speed factor
        action = action * self.config.speed_factor

        return action

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Predict action chunk (for compatibility with action chunking interface).

        For heuristics, we just return a single action with chunk dimension.
        """
        action = self.select_action(batch)
        # Add chunk dimension: (batch, action_dim) -> (batch, chunk_size, action_dim)
        return action.unsqueeze(1).expand(-1, self.config.chunk_size, -1)

    def _state_based_control(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Simple state-based heuristic control.

        Example rule: Control gripper based on threshold.

        Args:
            state: Current robot state (batch_size, state_dim)
            action: Action to modify (batch_size, action_dim)

        Returns:
            Modified action tensor
        """
        # Assuming last dimension is gripper position
        # State format for SO101: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
        gripper_pos = state[:, -1]  # Last dimension

        # Example heuristic: Toggle gripper based on threshold
        # If gripper > threshold, close it (set to 0)
        # If gripper <= threshold, open it (set to 1)
        gripper_action = torch.where(
            gripper_pos > self.config.gripper_threshold,
            torch.tensor(0.0, device=state.device),  # Close
            torch.tensor(1.0, device=state.device),  # Open
        )

        action[:, -1] = gripper_action

        # You can add more heuristic rules here
        # Example: Keep other joints at current position (no-op)
        # action[:, :-1] = state[:, :-1]

        return action

    def _vision_based_control(self, batch: dict[str, Tensor], action: Tensor) -> Tensor:
        """
        Vision-based heuristic control using object detection.

        Example: Detect red object and move arm towards it.

        Args:
            batch: Observation batch containing images
            action: Action to modify

        Returns:
            Modified action tensor
        """
        # Find image keys
        image_keys = [k for k in batch.keys() if k.startswith("observation.image")]

        if not image_keys:
            logger.warning("No images found for vision-based control, falling back to state-based")
            return self._state_based_control(batch["observation.state"], action)

        # Use first camera
        image = batch[image_keys[0]][0]  # (C, H, W)

        # Detect object
        cx, cy = self._detect_object(image)

        if cx is not None:
            # Object detected - adjust arm to center it
            state = batch["observation.state"][0]
            img_height, img_width = image.shape[1], image.shape[2]

            # Calculate error from image center
            center_x = img_width / 2
            center_y = img_height / 2

            error_x = (cx - center_x) / center_x
            error_y = (cy - center_y) / center_y

            # Apply proportional control to shoulder joints
            # Assuming: shoulder_pan (index 0), shoulder_lift (index 1)
            action[0, 0] = state[0] - error_x * self.config.position_gain  # Pan
            action[0, 1] = state[1] + error_y * self.config.position_gain  # Lift

            logger.debug(f"Object detected at ({cx:.1f}, {cy:.1f}), error: ({error_x:.2f}, {error_y:.2f})")

        return action

    def _detect_object(self, image: Tensor) -> tuple[float | None, float | None]:
        """
        Detect object in image using color-based detection.

        Args:
            image: Image tensor (C, H, W) in [0, 1] range

        Returns:
            Tuple of (center_x, center_y) or (None, None) if not found
        """
        if self.cv2 is None:
            return None, None

        # Convert tensor to numpy: (C, H, W) -> (H, W, C)
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        img_np = (img_np * 255).astype(np.uint8)

        # Convert RGB to HSV
        hsv = self.cv2.cvtColor(img_np, self.cv2.COLOR_RGB2HSV)

        # Create mask for target color
        lower = np.array(self.config.target_color_lower)
        upper = np.array(self.config.target_color_upper)
        mask = self.cv2.inRange(hsv, lower, upper)

        # Find contours
        contours, _ = self.cv2.findContours(mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        # Find largest contour
        largest_contour = max(contours, key=self.cv2.contourArea)

        # Calculate center
        moments = self.cv2.moments(largest_contour)
        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            return cx, cy

        return None, None


# Waypoint-based policy
class WaypointPolicy(HeuristicPolicy):
    """
    Waypoint-based heuristic policy.

    This policy loads waypoints from a JSON file and sequentially moves
    the robot to each waypoint using simple proportional control.

    Example usage:
    ```python
    config = HeuristicConfig(
        waypoints_path="waypoints/pick_and_place.json",
        waypoint_tolerance=0.01,
        waypoint_speed=0.1
    )
    policy = WaypointPolicy(config)
    ```
    """

    def __init__(self, config: HeuristicConfig):
        """
        Args:
            config: HeuristicConfig with waypoints_path set
        """
        super().__init__(config)

        if not self.config.waypoints_path:
            raise ValueError("waypoints_path must be specified for WaypointPolicy")

        # Load waypoints
        self.waypoints = self._load_waypoints(self.config.waypoints_path)
        self.current_waypoint_idx = 0
        self.waypoints_completed = False

        logger.info(f"Loaded {len(self.waypoints)} waypoints from {self.config.waypoints_path}")

    def _load_waypoints(self, waypoints_path: str) -> list[dict[str, float]]:
        """Load waypoints from JSON file."""
        path = Path(waypoints_path)
        if not path.exists():
            raise FileNotFoundError(f"Waypoints file not found: {waypoints_path}")

        with open(path, "r") as f:
            data = json.load(f)

        waypoints = []
        for wp in data["waypoints"]:
            # Extract state (joint positions)
            waypoints.append(wp["state"])

        return waypoints

    def reset(self):
        """Reset waypoint tracking."""
        super().reset()
        self.current_waypoint_idx = 0
        self.waypoints_completed = False
        logger.info("Waypoint tracking reset")

    def _state_based_control(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Move to waypoints sequentially.

        Args:
            state: Current robot state (batch_size, state_dim)
            action: Action to modify (batch_size, action_dim)

        Returns:
            Modified action tensor
        """
        if self.waypoints_completed:
            # Stay at last waypoint
            logger.debug("All waypoints completed, staying at final position")
            return action

        # Get current target waypoint
        target_waypoint = self.waypoints[self.current_waypoint_idx]

        # Convert to tensor
        target_state = torch.tensor(
            list(target_waypoint.values()),
            dtype=state.dtype,
            device=state.device
        ).unsqueeze(0)  # Add batch dimension

        # Calculate distance to target
        distance = torch.norm(state - target_state, dim=1)

        # Check if waypoint reached
        if distance < self.config.waypoint_tolerance:
            logger.info(f"✓ Waypoint {self.current_waypoint_idx + 1}/{len(self.waypoints)} reached!")

            # Move to next waypoint
            self.current_waypoint_idx += 1

            if self.current_waypoint_idx >= len(self.waypoints):
                logger.info("All waypoints completed!")
                self.waypoints_completed = True
                return action
            else:
                logger.info(f"Moving to waypoint {self.current_waypoint_idx + 1}/{len(self.waypoints)}")
                # Update target for next waypoint
                target_waypoint = self.waypoints[self.current_waypoint_idx]
                target_state = torch.tensor(
                    list(target_waypoint.values()),
                    dtype=state.dtype,
                    device=state.device
                ).unsqueeze(0)

        # Proportional control: move towards target
        error = target_state - state
        action = state + error * self.config.waypoint_speed

        logger.debug(
            f"Waypoint {self.current_waypoint_idx + 1}/{len(self.waypoints)}, "
            f"distance: {distance.item():.4f}"
        )

        return action


# Example: Custom heuristic policy
class CustomHeuristicPolicy(HeuristicPolicy):
    """
    Example of extending the base HeuristicPolicy with custom logic.

    Override _state_based_control() or _vision_based_control() to implement
    your own heuristics.
    """

    def _state_based_control(self, state: Tensor, action: Tensor) -> Tensor:
        """
        Your custom heuristic logic here!

        Example ideas:
        1. Impedance control
        2. Trajectory following
        3. Collision avoidance
        4. Task-specific behaviors
        """
        # Example: Simple sinusoidal motion on first joint
        # action[:, 0] = torch.sin(torch.tensor(self._step_count * 0.1))

        # Keep base class gripper logic
        return super()._state_based_control(state, action)
