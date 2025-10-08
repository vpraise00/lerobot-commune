#!/bin/bash

# LeRobot Calibration 스크립트
#
# 간단 사용법:
#   ./playground/scripts/calibrate.sh follower
#   ./playground/scripts/calibrate.sh leader
#
# 직접 지정:
#   ./playground/scripts/calibrate.sh --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower
#
# 현재 설정:
#   /dev/ttyACM0 = follower (robot)
#   /dev/ttyACM1 = leader (teleoperator)

echo "======================================================================"
echo "LeRobot Calibration"
echo "======================================================================"

if [ $# -eq 0 ]; then
    echo ""
    echo "사용법:"
    echo "  ./playground/scripts/calibrate.sh follower"
    echo "  ./playground/scripts/calibrate.sh leader"
    echo ""
    echo "또는 직접 지정:"
    echo "  ./playground/scripts/calibrate.sh --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower"
    echo ""
    exit 1
fi

# 간단 명령어 처리
if [ "$1" = "follower" ]; then
    echo ""
    echo "Follower (Robot) calibration 시작..."
    echo "실행: lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower"
    echo ""
    lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=follower
elif [ "$1" = "leader" ]; then
    echo ""
    echo "Leader (Teleoperator) calibration 시작..."
    echo "실행: lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader"
    echo ""
    lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=leader
else
    # 직접 인자 전달
    echo ""
    echo "실행: lerobot-calibrate $@"
    echo ""
    lerobot-calibrate "$@"
fi
