# Heuristic Policy - 완벽 가이드

## 📋 목차
1. [설치 확인](#설치-확인)
2. [기본 사용법](#기본-사용법)
3. [원격 제어 (PolicyServer)](#원격-제어)
4. [커스텀 휴리스틱 만들기](#커스텀-휴리스틱)
5. [데이터 수집](#데이터-수집)
6. [문제 해결](#문제-해결)

---

## 설치 확인

```bash
# 프로젝트 디렉토리에서
cd /path/to/lerobot

# 설치 확인
python -c "from lerobot.policies.heuristic import HeuristicPolicy; print('✅ 설치 완료')"

# Vision 기능 사용시 OpenCV 설치
pip install opencv-python
```

---

## 기본 사용법

### 1. 로컬에서 직접 제어

```python
from lerobot.policies.heuristic import HeuristicPolicy, HeuristicConfig
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
import torch

# ========== 설정 ==========
robot_config = SO101FollowerConfig(
    port="/dev/ttyACM0",  # lerobot-find-port로 찾은 포트
    id="my_so101"
)

policy_config = HeuristicConfig(
    gripper_threshold=0.5,   # 그리퍼 열기/닫기 임계값
    speed_factor=1.0,        # 동작 속도 배율
    use_vision=False         # 비전 사용 여부
)

# ========== 초기화 ==========
robot = SO101Follower(robot_config)
robot.connect()

policy = HeuristicPolicy(policy_config)

# ========== 제어 루프 ==========
try:
    for step in range(100):
        # 1. 관찰 받기
        obs = robot.get_observation()

        # 2. Batch 형식으로 변환
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

        # 3. 정책에서 액션 받기
        action = policy.select_action(batch)

        # 4. 로봇에 액션 전송
        action_dict = {
            "shoulder_pan.pos": float(action[0, 0]),
            "shoulder_lift.pos": float(action[0, 1]),
            "elbow_flex.pos": float(action[0, 2]),
            "wrist_flex.pos": float(action[0, 3]),
            "wrist_roll.pos": float(action[0, 4]),
            "gripper.pos": float(action[0, 5]),
        }
        robot.send_action(action_dict)

        if step % 10 == 0:
            print(f"Step {step}: Gripper={action_dict['gripper.pos']:.2f}")

finally:
    robot.disconnect()
```

---

## 원격 제어

### 아키텍처
```
┌──────────────────┐                    ┌──────────────────┐
│  로봇 머신        │ gRPC observations  │  서버 머신        │
│  - SO101 연결    │ -----------------> │  - HeuristicPolicy│
│  - RobotClient   │ <----------------- │  - PolicyServer   │
│                  │    gRPC actions    │                  │
└──────────────────┘                    └──────────────────┘
```

### 1단계: 서버 시작 (서버 머신)

```bash
python src/lerobot/async_inference/policy_server.py \
    --host=0.0.0.0 \
    --port=8080 \
    --fps=30
```

### 2단계: 클라이언트 시작 (로봇 머신)

```bash
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras='{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
    --robot.id=my_so101 \
    --server_address=192.168.1.100:8080 \
    --policy_type=heuristic \
    --pretrained_name_or_path=dummy \
    --policy_device=cpu
```

**주요 파라미터:**
- `--server_address`: 서버 IP:포트
- `--policy_type=heuristic`: 휴리스틱 정책 사용
- `--pretrained_name_or_path=dummy`: 휴리스틱은 가중치가 없으므로 dummy 값

---

## 커스텀 휴리스틱

### 예제 1: 간단한 Pick-and-Place

```python
from lerobot.policies.heuristic.modeling_heuristic import HeuristicPolicy
from lerobot.policies.heuristic import HeuristicConfig
from torch import Tensor

class PickAndPlacePolicy(HeuristicPolicy):
    """4단계 Pick-and-Place 휴리스틱"""

    def _state_based_control(self, state: Tensor, action: Tensor) -> Tensor:
        """
        상태 기반 제어 로직

        Args:
            state: 현재 상태 (batch, 6) - [pan, lift, elbow, wrist, roll, gripper]
            action: 수정할 액션 (batch, 6)
        """
        step = self._step_count

        # Phase 1: 접근 (0-50 steps)
        if step < 50:
            action[:, -1] = 1.0              # 그리퍼 열기
            action[:, 1] = state[:, 1] - 0.01  # 아래로 이동

        # Phase 2: 잡기 (50-100 steps)
        elif step < 100:
            action[:, -1] = 0.0              # 그리퍼 닫기

        # Phase 3: 들어올리기 (100-150 steps)
        elif step < 150:
            action[:, 1] = state[:, 1] + 0.02  # 위로 이동

        # Phase 4: 놓기 (150+ steps)
        else:
            action[:, -1] = 1.0              # 그리퍼 열기
            action[:, 0] = 0.5               # Pan을 특정 위치로

        return action


# 사용
config = HeuristicConfig()
policy = PickAndPlacePolicy(config)
```

### 예제 2: Vision 기반 제어

```python
class VisionTrackingPolicy(HeuristicPolicy):
    """빨간 물체를 따라가는 정책"""

    def __init__(self, config):
        super().__init__(config)
        # 비전 활성화 필수
        config.use_vision = True

    def _vision_based_control(self, batch: dict, action: Tensor) -> Tensor:
        """이미지에서 물체를 감지하고 따라감"""

        # 첫 번째 카메라 이미지 가져오기
        image_keys = [k for k in batch.keys() if k.startswith("observation.image")]
        if not image_keys:
            return action

        image = batch[image_keys[0]][0]  # (C, H, W)
        state = batch["observation.state"][0]

        # 물체 감지
        cx, cy = self._detect_object(image)

        if cx is not None:
            # 이미지 중심과의 오차 계산
            img_h, img_w = image.shape[1], image.shape[2]
            error_x = (cx - img_w/2) / (img_w/2)
            error_y = (cy - img_h/2) / (img_h/2)

            # 비례 제어
            action[0, 0] = state[0] - error_x * 0.1  # Pan
            action[0, 1] = state[1] + error_y * 0.1  # Lift

            # 물체가 중앙에 가까우면 그리퍼 닫기
            if abs(error_x) < 0.1 and abs(error_y) < 0.1:
                action[0, -1] = 0.0  # Close gripper

        return action


# 사용
config = HeuristicConfig(
    use_vision=True,
    target_color_lower=[0, 100, 100],   # 빨간색 (HSV)
    target_color_upper=[10, 255, 255]
)
policy = VisionTrackingPolicy(config)
```

### 예제 3: 임피던스 제어

```python
class ImpedanceControlPolicy(HeuristicPolicy):
    """부드러운 임피던스 제어"""

    def __init__(self, config):
        super().__init__(config)
        self.target_position = None
        self.stiffness = 0.5
        self.damping = 0.1

    def _state_based_control(self, state: Tensor, action: Tensor) -> Tensor:
        """임피던스 제어 로직"""

        if self.target_position is None:
            # 목표 위치 설정 (초기 상태 기준)
            self.target_position = state.clone()
            self.target_position[:, 1] += 0.2  # 위쪽으로 20cm

        # 임피던스 제어: F = K*(x_target - x) - D*v
        # 간단화: velocity를 0으로 가정
        error = self.target_position - state
        action = state + error * self.stiffness

        return action
```

---

## 데이터 수집

### 휴리스틱 정책으로 데이터 수집하기

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --robot.id=my_so101 \
    --policy.path=dummy \
    --policy.type=heuristic \
    --dataset.repo_id=${HF_USER}/heuristic_dataset \
    --dataset.num_episodes=10 \
    --dataset.single_task="Pick and place cube"
```

**참고:** `--policy.path=dummy`는 휴리스틱 정책에는 가중치가 없어서 dummy 값 사용

---

## 문제 해결

### 1. ModuleNotFoundError: No module named 'lerobot.policies.heuristic'

```bash
# 확인
ls src/lerobot/policies/heuristic/

# 있어야 할 파일들:
# - __init__.py
# - configuration_heuristic.py
# - modeling_heuristic.py
# - README.md

# 없으면 다시 생성
```

### 2. "heuristic policy not found" 에러

**확인 사항:**
1. `src/lerobot/policies/__init__.py`에 HeuristicConfig 추가됨
2. `src/lerobot/policies/factory.py`에 heuristic 케이스 추가됨
3. `src/lerobot/async_inference/constants.py`에 "heuristic" 추가됨

### 3. Vision 기능이 안 됨

```bash
# OpenCV 설치 확인
python -c "import cv2; print(cv2.__version__)"

# 설치
pip install opencv-python
```

### 4. 로봇 연결 안 됨

```bash
# 포트 확인
lerobot-find-port

# 권한 확인 (Linux)
sudo chmod 666 /dev/ttyACM0

# 캘리브레이션 확인
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_so101
```

---

## 고급 사용

### PolicyServer에 데이터 저장 기능 추가

서버에서 observation + action을 저장하려면 `policy_server.py` 수정:

```python
# policy_server.py 수정

class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    def __init__(self, config: PolicyServerConfig):
        # ... 기존 코드 ...

        # 데이터셋 저장 추가
        if config.record_dataset:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            self.dataset = LeRobotDataset.create(
                repo_id=config.dataset_repo_id,
                fps=config.fps,
                robot_type="so101_follower",
                features=...,
                use_videos=True,
            )
        else:
            self.dataset = None

    def _predict_action_chunk(self, obs):
        action_chunk = self.policy.predict_action_chunk(obs)

        # 데이터 저장
        if self.dataset is not None:
            frame = {**obs, **action_chunk}
            self.dataset.add_frame(frame)

        return action_chunk
```

---

## 참고 자료

- **정책 파일**: `src/lerobot/policies/heuristic/`
- **예제**: `examples/heuristic_policy_example.py`
- **README**: `src/lerobot/policies/heuristic/README.md`
- **설정**: `HeuristicConfig` in `configuration_heuristic.py`

---

## 다음 단계

1. ✅ 로컬에서 테스트
2. ✅ 커스텀 휴리스틱 구현
3. ✅ PolicyServer로 원격 제어
4. ✅ 데이터 수집
5. ✅ 학습된 정책과 비교

**Good luck!** 🚀
