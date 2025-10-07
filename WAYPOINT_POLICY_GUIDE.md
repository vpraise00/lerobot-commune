# Waypoint-based Heuristic Policy - 가이드

## 📋 개요

Waypoint Policy는 teleoperation을 통해 저장한 waypoint들을 순차적으로 재생하는 heuristic policy입니다.

**주요 기능:**
- ✅ Teleoperation으로 waypoint 저장 (5~10개)
- ✅ 저장된 waypoint를 순차적으로 재생
- ✅ Proportional control로 부드러운 이동
- ✅ Waypoint 도달 감지 (tolerance 기반)
- ✅ JSON 포맷으로 waypoint 저장/로드

---

## 🚀 빠른 시작

### 1단계: Waypoint 저장

Teleoperation을 통해 원하는 위치로 로봇을 이동하고 waypoint를 저장합니다:

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

**조작 방법:**
1. Leader 로봇으로 follower를 원하는 위치로 이동
2. 터미널에서 `s` + ENTER를 눌러 waypoint 저장
3. `q` + ENTER를 눌러 저장 완료

**저장되는 JSON 포맷:**
```json
{
  "robot_type": "so101_follower",
  "num_waypoints": 5,
  "created_at": "2025-10-03 12:34:56",
  "waypoints": [
    {
      "index": 0,
      "state": {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": 0.5,
        "elbow_flex.pos": -0.3,
        "wrist_flex.pos": 0.1,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 1.0
      },
      "timestamp": 1728045296.123
    },
    ...
  ]
}
```

---

### 2단계: Waypoint 재생

저장한 waypoint를 순차적으로 재생합니다:

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

**파라미터 설명:**
- `--policy.waypoints_path`: waypoint JSON 파일 경로
- `--policy.waypoint_tolerance`: waypoint 도달 판정 거리 (기본: 0.01)
- `--policy.waypoint_speed`: 이동 속도 계수 (0.1 = 에러의 10%씩 이동)
- `--num_episodes`: 재생 반복 횟수

---

## 📝 프로그래밍 방식 사용

### 기본 사용법

```python
from lerobot.policies.heuristic import HeuristicConfig, WaypointPolicy
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
import torch

# 설정
robot_config = SO101FollowerConfig(port="/dev/ttyACM0", id="my_so101")
policy_config = HeuristicConfig(
    waypoints_path="waypoints/pick_and_place.json",
    waypoint_tolerance=0.01,
    waypoint_speed=0.1
)

# 초기화
robot = SO101Follower(robot_config)
robot.connect()
policy = WaypointPolicy(policy_config)

# 제어 루프
for step in range(1000):
    # 관찰
    obs = robot.get_observation()
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

    # 액션 계산
    action = policy.select_action(batch)

    # 로봇에 전송
    action_dict = {
        "shoulder_pan.pos": float(action[0, 0]),
        "shoulder_lift.pos": float(action[0, 1]),
        "elbow_flex.pos": float(action[0, 2]),
        "wrist_flex.pos": float(action[0, 3]),
        "wrist_roll.pos": float(action[0, 4]),
        "gripper.pos": float(action[0, 5]),
    }
    robot.send_action(action_dict)

    # 완료 체크
    if policy.waypoints_completed:
        print("모든 waypoint 완료!")
        break

robot.disconnect()
```

---

## 🔧 고급 사용법

### 1. Waypoint를 직접 생성

JSON 파일을 직접 만들어서 사용할 수도 있습니다:

```python
import json

waypoints_data = {
    "robot_type": "so101_follower",
    "num_waypoints": 3,
    "created_at": "2025-10-03 12:00:00",
    "waypoints": [
        {
            "index": 0,
            "state": {
                "shoulder_pan.pos": 0.0,
                "shoulder_lift.pos": 0.0,
                "elbow_flex.pos": 0.0,
                "wrist_flex.pos": 0.0,
                "wrist_roll.pos": 0.0,
                "gripper.pos": 1.0
            },
            "timestamp": 0
        },
        # ... more waypoints
    ]
}

with open("waypoints/custom.json", "w") as f:
    json.dump(waypoints_data, f, indent=2)
```

### 2. 실시간으로 waypoint 추가

```python
class DynamicWaypointPolicy(WaypointPolicy):
    """동적으로 waypoint를 추가할 수 있는 policy"""

    def add_waypoint(self, state_dict):
        """새로운 waypoint 추가"""
        self.waypoints.append(state_dict)
        print(f"Waypoint {len(self.waypoints)} 추가됨")

    def clear_waypoints(self):
        """모든 waypoint 제거"""
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoints_completed = False
```

### 3. Waypoint 보간 (Interpolation)

부드러운 궤적을 위해 waypoint 사이를 보간:

```python
import numpy as np

class InterpolatedWaypointPolicy(WaypointPolicy):
    """Waypoint 사이를 선형 보간하는 policy"""

    def __init__(self, config):
        super().__init__(config)
        self.interpolation_steps = 50  # 각 waypoint 사이 스텝 수
        self.interpolated_waypoints = self._interpolate_waypoints()

    def _interpolate_waypoints(self):
        """Waypoint 사이를 선형 보간"""
        interpolated = []

        for i in range(len(self.waypoints) - 1):
            current = np.array(list(self.waypoints[i].values()))
            next_wp = np.array(list(self.waypoints[i + 1].values()))

            # 선형 보간
            for t in np.linspace(0, 1, self.interpolation_steps):
                interpolated_state = current * (1 - t) + next_wp * t
                interpolated.append(dict(zip(self.waypoints[i].keys(), interpolated_state)))

        return interpolated
```

---

## 📊 제어 알고리즘

WaypointPolicy는 간단한 **Proportional Control**을 사용합니다:

```python
# 현재 상태와 목표 waypoint 사이의 오차 계산
error = target_waypoint - current_state

# 오차의 일정 비율만큼 이동
action = current_state + error * waypoint_speed

# waypoint_speed = 0.1 이면 매 스텝마다 오차의 10%씩 이동
```

### 파라미터 튜닝

| 파라미터 | 낮은 값 | 높은 값 |
|---------|--------|---------|
| `waypoint_speed` | 느리고 부드러움 | 빠르지만 오버슈팅 가능 |
| `waypoint_tolerance` | 정확하지만 느림 | 빠르지만 부정확 |

**권장 값:**
- `waypoint_speed`: 0.05 ~ 0.2
- `waypoint_tolerance`: 0.005 ~ 0.02

---

## 🎯 활용 사례

### 1. Pick and Place

```bash
# 1. Waypoint 저장
lerobot-save-waypoints \
    --output_path=waypoints/pick_and_place.json \
    --max_waypoints=6
# Waypoints: [초기위치] -> [물체 위] -> [물체 잡기] -> [들기] -> [목표 위] -> [놓기]

# 2. 재생
lerobot-eval --policy.waypoints_path=waypoints/pick_and_place.json
```

### 2. 반복 작업

```bash
# 동일한 작업을 100번 반복
lerobot-eval \
    --policy.waypoints_path=waypoints/assembly.json \
    --num_episodes=100
```

### 3. 데이터 수집

Waypoint policy로 일관된 데이터를 수집:

```bash
lerobot-record \
    --policy.type=heuristic \
    --policy.waypoints_path=waypoints/task.json \
    --dataset.num_episodes=50
```

---

## 🐛 문제 해결

### 1. Waypoint가 저장되지 않음

**원인:** Windows에서 non-blocking 입력 미지원

**해결:**
- Windows: `s` + ENTER로 저장, `q` + ENTER로 종료
- Linux/Mac: 자동으로 감지됨

### 2. 로봇이 waypoint에 도달하지 못함

**원인:** `waypoint_tolerance`가 너무 작음

**해결:**
```bash
--policy.waypoint_tolerance=0.02  # 기본값 0.01에서 증가
```

### 3. 움직임이 너무 느림

**원인:** `waypoint_speed`가 너무 작음

**해결:**
```bash
--policy.waypoint_speed=0.2  # 기본값 0.1에서 증가
```

### 4. 로봇이 진동함 (oscillation)

**원인:** `waypoint_speed`가 너무 큼

**해결:**
```bash
--policy.waypoint_speed=0.05  # 속도 감소
```

---

## 📚 API 레퍼런스

### WaypointPolicy

```python
class WaypointPolicy(HeuristicPolicy):
    def __init__(self, config: HeuristicConfig)
    def reset(self) -> None
    def select_action(self, batch: dict[str, Tensor]) -> Tensor

    # 속성
    waypoints: list[dict[str, float]]  # 로드된 waypoint 리스트
    current_waypoint_idx: int          # 현재 목표 waypoint 인덱스
    waypoints_completed: bool          # 모든 waypoint 완료 여부
```

### HeuristicConfig (Waypoint 관련)

```python
@dataclass
class HeuristicConfig:
    waypoints_path: str | None = None           # waypoint JSON 파일 경로
    waypoint_tolerance: float = 0.01            # waypoint 도달 판정 거리
    waypoint_speed: float = 0.1                 # 이동 속도 계수 (0~1)
```

---

## 🎓 예제 파일

- **스크립트**: `src/lerobot/scripts/lerobot_save_waypoints.py`
- **Policy**: `src/lerobot/policies/heuristic/modeling_heuristic.py`
- **예제**: `examples/waypoint_policy_example.py`
- **설정**: `src/lerobot/policies/heuristic/configuration_heuristic.py`

---

## 🚀 다음 단계

1. ✅ Waypoint 저장 테스트
2. ✅ 간단한 pick-and-place 작업
3. ✅ 파라미터 튜닝
4. ✅ 데이터 수집 및 학습된 policy와 비교

**Happy coding!** 🤖
