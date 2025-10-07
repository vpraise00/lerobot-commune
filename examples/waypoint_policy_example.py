#!/usr/bin/env python

"""
Example: Using Waypoint-based Heuristic Policy

This example demonstrates how to:
1. Save waypoints via teleoperation
2. Replay waypoints using WaypointPolicy

Step 1: Save waypoints
------------------------
```bash
lerobot-save-waypoints \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_so101 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader \
    --output_path=waypoints/pick_and_place.json \
    --max_waypoints=10
```

Step 2: Replay waypoints
------------------------
```bash
lerobot-eval \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_so101 \
    --policy.type=heuristic \
    --policy.waypoints_path=waypoints/pick_and_place.json \
    --policy.waypoint_tolerance=0.01 \
    --policy.waypoint_speed=0.1 \
    --num_episodes=5
```

Step 3: Programmatic usage
---------------------------
"""

from lerobot.policies.heuristic import HeuristicConfig, WaypointPolicy
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
import torch


def example_waypoint_policy():
    """Example of using WaypointPolicy programmatically."""

    # Robot configuration
    robot_config = SO101FollowerConfig(
        port="/dev/ttyACM0",
        id="my_so101"
    )

    # Policy configuration with waypoints
    policy_config = HeuristicConfig(
        waypoints_path="waypoints/pick_and_place.json",
        waypoint_tolerance=0.01,  # 1cm tolerance
        waypoint_speed=0.1,  # 10% speed per step
    )

    # Initialize robot and policy
    robot = SO101Follower(robot_config)
    robot.connect()

    policy = WaypointPolicy(policy_config)

    # Control loop
    try:
        for step in range(1000):
            # Get observation
            obs = robot.get_observation()

            # Convert to batch format
            batch = {
                "observation.state": torch.tensor([[
                    obs["shoulder_pan.pos"],
                    obs["shoulder_lift.pos"],
                    obs["elbow_flex.pos"],
                    obs["wrist_flex.pos"],
                    obs["wrist_roll.pos"],
                    obs["gripper.pos"],
                ]], dtype=torch.float32)
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

            # Check if all waypoints completed
            if policy.waypoints_completed:
                print("All waypoints completed!")
                break

    finally:
        robot.disconnect()


if __name__ == "__main__":
    example_waypoint_policy()
