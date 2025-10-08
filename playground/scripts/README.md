# LeRobot Scripts

LeRobot 기본 명령어들을 쉽게 사용하기 위한 wrapper 스크립트 모음입니다.

## Scripts

### 1. check_ports.sh
USB 포트를 찾는 스크립트입니다.

```bash
./playground/scripts/check_ports.sh
```

### 2. calibrate.sh
Robot 또는 Teleoperator를 calibration하는 스크립트입니다.

```bash
# Robot calibration
./playground/scripts/calibrate.sh --robot.type=so100_follower --robot.port=/dev/ttyUSB0

# Teleoperator calibration
./playground/scripts/calibrate.sh --teleop.type=so100_leader --teleop.port=/dev/ttyUSB0
```

## LeRobot 기본 명령어

스크립트 없이 직접 사용할 수도 있습니다:

- `lerobot-find-port` - 포트 찾기
- `lerobot-calibrate` - Calibration
- `lerobot-setup-motors` - 모터 ID 및 baudrate 설정
- `lerobot-find-cameras` - 카메라 찾기
- `lerobot-record` - 데이터 기록
- `lerobot-replay` - 데이터 재생
- `lerobot-teleoperate` - Teleoperation
