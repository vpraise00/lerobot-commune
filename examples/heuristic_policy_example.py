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
Example usage of HeuristicPolicy with SO101 robot.

This demonstrates:
1. Local control (direct robot connection)
2. Remote control (via PolicyServer)
3. Data collection with heuristic policy
"""

import torch

from lerobot.policies.heuristic import HeuristicConfig, HeuristicPolicy
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig


def example_1_local_control():
    """Example 1: Direct local control with heuristic policy"""
    print("=" * 60)
    print("Example 1: Local Control with Heuristic Policy")
    print("=" * 60)

    # Configure robot
    robot_config = SO101FollowerConfig(
        port="/dev/ttyACM0",  # Change to your port
        id="my_so101",
        cameras={
            "front": {
                "type": "opencv",
                "index_or_path": 0,
                "width": 640,
                "height": 480,
                "fps": 30,
            }
        },
    )

    # Configure heuristic policy
    policy_config = HeuristicConfig(
        gripper_threshold=0.5,
        speed_factor=1.0,
        use_vision=False,  # Start with simple state-based control
        state_features=["observation.state"],
        action_features=["action"],
    )

    # Initialize
    robot = SO101Follower(robot_config)
    robot.connect()

    policy = HeuristicPolicy(policy_config)

    # Control loop
    try:
        for i in range(100):
            # Get observation
            obs = robot.get_observation()

            # Convert to batch format
            batch = {
                "observation.state": torch.tensor(
                    [
                        [
                            obs["shoulder_pan.pos"],
                            obs["shoulder_lift.pos"],
                            obs["elbow_flex.pos"],
                            obs["wrist_flex.pos"],
                            obs["wrist_roll.pos"],
                            obs["gripper.pos"],
                        ]
                    ],
                    dtype=torch.float32,
                )
            }

            # Get action from policy
            action = policy.select_action(batch)

            # Send to robot
            action_dict = {
                "shoulder_pan.pos": float(action[0, 0]),
                "shoulder_lift.pos": float(action[0, 1]),
                "elbow_flex.pos": float(action[0, 2]),
                "wrist_flex.pos": float(action[0, 3]),
                "wrist_roll.pos": float(action[0, 4]),
                "gripper.pos": float(action[0, 5]),
            }

            robot.send_action(action_dict)

            if i % 10 == 0:
                print(f"Step {i}: Gripper={action_dict['gripper.pos']:.2f}")

    finally:
        robot.disconnect()

    print("Example 1 completed!\n")


def example_2_vision_based_control():
    """Example 2: Vision-based heuristic control"""
    print("=" * 60)
    print("Example 2: Vision-Based Heuristic Control")
    print("=" * 60)

    # Configure with vision enabled
    policy_config = HeuristicConfig(
        gripper_threshold=0.5,
        speed_factor=1.0,
        use_vision=True,  # Enable vision
        position_gain=0.1,
        target_color_lower=[0, 100, 100],  # Red object (HSV)
        target_color_upper=[10, 255, 255],
    )

    policy = HeuristicPolicy(policy_config)

    # Dummy observation with image
    dummy_obs = {
        "observation.state": torch.randn(1, 6),
        "observation.image.front": torch.rand(1, 3, 480, 640),  # Random image
    }

    action = policy.select_action(dummy_obs)
    print(f"Action shape: {action.shape}")
    print("Example 2 completed!\n")


def example_3_remote_control_setup():
    """Example 3: Setup for remote control via PolicyServer"""
    print("=" * 60)
    print("Example 3: Remote Control Setup (Instructions)")
    print("=" * 60)

    print(
        """
To use heuristic policy with PolicyServer:

1. Start PolicyServer on GPU machine:

   python src/lerobot/async_inference/policy_server.py \\
       --host=0.0.0.0 \\
       --port=8080 \\
       --fps=30

2. Start RobotClient on robot machine:

   python src/lerobot/async_inference/robot_client.py \\
       --robot.type=so101_follower \\
       --robot.port=/dev/ttyACM0 \\
       --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \\
       --robot.id=my_so101 \\
       --server_address=192.168.1.100:8080 \\
       --policy_type=heuristic \\
       --pretrained_name_or_path=local/path/or/hf/repo \\
       --policy_device=cpu

Note: For heuristic policy, pretrained_name_or_path can be a dummy path
since the policy doesn't load weights.
"""
    )


def example_4_custom_heuristic():
    """Example 4: Create custom heuristic by subclassing"""
    print("=" * 60)
    print("Example 4: Custom Heuristic Implementation")
    print("=" * 60)

    from lerobot.policies.heuristic.modeling_heuristic import HeuristicPolicy

    class MyCustomHeuristic(HeuristicPolicy):
        """Custom heuristic with task-specific logic"""

        def _state_based_control(self, state, action):
            """
            Custom control logic.

            Example: Implement a simple pick-and-place behavior
            """
            # Phase 1: Move to object
            if self._step_count < 50:
                # Open gripper
                action[:, -1] = 1.0
                # Move arm down
                action[:, 1] = state[:, 1] - 0.01

            # Phase 2: Grasp
            elif self._step_count < 100:
                # Close gripper
                action[:, -1] = 0.0

            # Phase 3: Lift
            elif self._step_count < 150:
                # Move arm up
                action[:, 1] = state[:, 1] + 0.01

            # Phase 4: Place
            else:
                # Open gripper
                action[:, -1] = 1.0

            return action

    # Use custom policy
    config = HeuristicConfig()
    custom_policy = MyCustomHeuristic(config)

    dummy_obs = {"observation.state": torch.randn(1, 6)}
    action = custom_policy.select_action(dummy_obs)
    print(f"Custom policy action: {action}")
    print("Example 4 completed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Heuristic Policy Examples")
    print("=" * 60 + "\n")

    # Run examples (comment out the ones that need real hardware)
    # example_1_local_control()  # Requires real robot
    example_2_vision_based_control()
    example_3_remote_control_setup()
    example_4_custom_heuristic()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
