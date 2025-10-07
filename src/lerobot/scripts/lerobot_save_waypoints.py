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
Script to save waypoints via teleoperation for waypoint-based heuristic policy.

This script allows you to control a robot via teleoperation and save waypoints
by pressing a key (e.g., spacebar). The saved waypoints can then be replayed
using the WaypointPolicy.

Example:

```shell
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

Controls:
- Teleoperate the robot to desired position
- Press SPACE to save current position as waypoint
- Press 'q' or ESC to finish and save waypoints file
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass
class SaveWaypointsConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    output_path: str = "waypoints/waypoints.json"
    max_waypoints: int = 10
    fps: int = 30


def save_waypoints_from_teleop(config: SaveWaypointsConfig):
    """
    Teleoperate robot and save waypoints when spacebar is pressed.

    Args:
        config: SaveWaypointsConfig instance
    """
    init_logging()

    # Create output directory
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize robot and teleoperator
    logger.info("Initializing robot and teleoperator...")
    robot = make_robot_from_config(config.robot)
    teleop = make_teleoperator_from_config(config.teleop)

    # Connect
    robot.connect()
    teleop.connect()

    # Create processor pipeline
    processors = make_default_processors(config.robot)
    pipeline = RobotProcessorPipeline(processors)

    # Waypoints storage
    waypoints = []

    # Keyboard input (simplified - in real usage, use pynput or similar)
    logger.info("\n" + "=" * 60)
    logger.info("Waypoint Recording Started")
    logger.info("=" * 60)
    logger.info("Controls:")
    logger.info("  - Move robot to desired position")
    logger.info("  - Press ENTER in terminal to save waypoint")
    logger.info("  - Type 'q' and press ENTER to finish")
    logger.info(f"  - Max waypoints: {config.max_waypoints}")
    logger.info("=" * 60 + "\n")

    dt = 1.0 / config.fps

    try:
        # Start non-blocking input mode
        import sys
        import select

        is_windows = sys.platform.startswith('win')

        if is_windows:
            # Windows doesn't support select on stdin
            logger.warning("Windows detected. Please use manual ENTER key press to save waypoints.")
            logger.info("Type 's' + ENTER to save, 'q' + ENTER to quit")

        while len(waypoints) < config.max_waypoints:
            start_time = time.perf_counter()

            # Get teleoperator action
            teleop_obs = teleop.get_observation()
            action_dict = pipeline.to_action(RobotAction(teleop_obs))

            # Send action to robot
            robot.send_action(action_dict)

            # Get current robot state
            robot_obs = robot.get_observation()

            # Check for keyboard input (non-blocking)
            user_input = None
            if is_windows:
                # Windows: blocking input (requires ENTER)
                if sys.stdin.isatty():
                    try:
                        user_input = input().strip().lower()
                    except:
                        pass
            else:
                # Linux/Mac: non-blocking check
                if select.select([sys.stdin], [], [], 0)[0]:
                    user_input = sys.stdin.readline().strip().lower()

            # Handle user input
            if user_input:
                if user_input == 'q':
                    logger.info("Quitting...")
                    break
                elif user_input == 's' or user_input == '':
                    # Save waypoint
                    waypoint = {
                        "index": len(waypoints),
                        "state": robot_obs,
                        "timestamp": time.time()
                    }
                    waypoints.append(waypoint)
                    logger.info(f"✓ Waypoint {len(waypoints)}/{config.max_waypoints} saved!")
                    logger.info(f"  State: {robot_obs}")

            # Wait for next control step
            dt_remaining = dt - (time.perf_counter() - start_time)
            busy_wait(dt_remaining)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user (Ctrl+C)")

    finally:
        # Disconnect
        robot.disconnect()
        teleop.disconnect()

        # Save waypoints to file
        if waypoints:
            waypoints_data = {
                "robot_type": config.robot.type,
                "num_waypoints": len(waypoints),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "waypoints": waypoints
            }

            with open(output_path, "w") as f:
                json.dump(waypoints_data, f, indent=2)

            logger.info(f"\n✓ Saved {len(waypoints)} waypoints to {output_path}")
            logger.info(f"  Use with: lerobot-eval --policy.type=waypoint --policy.waypoints_path={output_path}")
        else:
            logger.info("\nNo waypoints saved.")


def main():
    parser_config = parser.init(SaveWaypointsConfig)
    save_waypoints_from_teleop(parser_config)


if __name__ == "__main__":
    main()
