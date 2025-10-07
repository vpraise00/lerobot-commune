# Heuristic Policy

A rule-based policy for robot control without machine learning.

## Overview

The Heuristic Policy allows you to implement custom control logic using:
- **State-based rules**: Simple threshold-based decisions
- **Vision-based control**: Object detection and tracking using OpenCV
- **Custom behaviors**: Easy to extend with your own heuristics

## Use Cases

1. **Teleoperation automation**: Add simple automation during data collection
2. **Baseline comparisons**: Compare learned policies against hand-crafted rules
3. **Safety fallbacks**: Fallback behavior when learned policies fail
4. **Testing**: Quick robot setup testing without training
5. **Prototyping**: Rapid iteration on control ideas

## Installation

```bash
# Base installation (state-based heuristics)
pip install -e .

# For vision-based heuristics
pip install opencv-python
```

## Quick Start

### 1. Local Control (Direct Robot Connection)

```python
from lerobot.policies.heuristic import HeuristicPolicy, HeuristicConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
import torch

# Configure policy
config = HeuristicConfig(
    gripper_threshold=0.5,  # Toggle gripper at this position
    speed_factor=1.0,       # Action speed multiplier
    use_vision=False        # State-based control
)

policy = HeuristicPolicy(config)

# Configure robot
robot_config = SO101FollowerConfig(port="/dev/ttyACM0", id="my_so101")
robot = SO101Follower(robot_config)
robot.connect()

# Control loop
while True:
    obs = robot.get_observation()

    # Format observation
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

    # Get action
    action = policy.select_action(batch)

    # Send to robot
    robot.send_action({
        "shoulder_pan.pos": float(action[0, 0]),
        # ... other joints ...
    })
```

### 2. Remote Control (via PolicyServer)

**Server (GPU machine):**
```bash
python src/lerobot/async_inference/policy_server.py \
    --host=0.0.0.0 \
    --port=8080 \
    --fps=30
```

**Client (Robot machine):**
```bash
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
    --server_address=192.168.1.100:8080 \
    --policy_type=heuristic \
    --pretrained_name_or_path=dummy \
    --policy_device=cpu
```

### 3. Vision-Based Control

```python
config = HeuristicConfig(
    use_vision=True,
    position_gain=0.1,
    target_color_lower=[0, 100, 100],   # Red object (HSV)
    target_color_upper=[10, 255, 255]
)

policy = HeuristicPolicy(config)

# Observation must include images
batch = {
    "observation.state": torch.randn(1, 6),
    "observation.image.front": torch.rand(1, 3, 480, 640)
}

action = policy.select_action(batch)
```

## Custom Heuristics

Extend the base class to implement your own logic:

```python
from lerobot.policies.heuristic.modeling_heuristic import HeuristicPolicy
from torch import Tensor

class PickAndPlaceHeuristic(HeuristicPolicy):
    """Custom pick-and-place behavior"""

    def _state_based_control(self, state: Tensor, action: Tensor) -> Tensor:
        """State machine for pick and place"""

        # Phase 1: Approach (steps 0-50)
        if self._step_count < 50:
            action[:, -1] = 1.0  # Open gripper
            action[:, 1] = state[:, 1] - 0.01  # Move down

        # Phase 2: Grasp (steps 50-100)
        elif self._step_count < 100:
            action[:, -1] = 0.0  # Close gripper

        # Phase 3: Lift (steps 100-150)
        elif self._step_count < 150:
            action[:, 1] = state[:, 1] + 0.01  # Move up

        # Phase 4: Place (steps 150+)
        else:
            action[:, -1] = 1.0  # Open gripper

        return action
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gripper_threshold` | float | 0.5 | Gripper position threshold |
| `speed_factor` | float | 1.0 | Action speed multiplier |
| `use_vision` | bool | False | Enable vision-based control |
| `position_gain` | float | 0.1 | Proportional control gain |
| `target_color_lower` | list[int] | [0,100,100] | HSV lower bound |
| `target_color_upper` | list[int] | [10,255,255] | HSV upper bound |

## Architecture

```
HeuristicPolicy
├── select_action()              # Main entry point
│   ├── _state_based_control()   # Override for state rules
│   └── _vision_based_control()  # Override for vision rules
│       └── _detect_object()     # Color-based detection
└── predict_action_chunk()       # Action chunking interface
```

## Examples

See `examples/heuristic_policy_example.py` for complete examples:
- Local control
- Vision-based control
- Remote control setup
- Custom heuristic implementation

## Debugging

Enable logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Notes

- No training required - pure rule-based control
- Compatible with all LeRobot infrastructure (PolicyServer, data collection, etc.)
- Can be mixed with learned policies (e.g., heuristic for reset, learned for task)
- Zero GPU usage

## Contributing

To add new heuristic features:
1. Extend `HeuristicPolicy` class
2. Override `_state_based_control()` or `_vision_based_control()`
3. Add configuration parameters to `HeuristicConfig`

## Citation

If you use heuristic policies as baselines in your research:

```bibtex
@software{lerobot_heuristic_policy,
  author = {LeRobot Team},
  title = {Heuristic Policy for Robot Control},
  year = {2024},
  url = {https://github.com/huggingface/lerobot}
}
```
