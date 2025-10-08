#!/bin/bash

# LeRobot Teleoperation 스크립트
#
# 사용법:
#   ./playground/scripts/teleoperate.sh
#
# 현재 설정:
#   /dev/ttyACM0 = follower (robot)
#   /dev/ttyACM1 = leader (teleoperator)
#   카메라 0 = front camera
#
# 카메라 없이 실행:
#   ./playground/scripts/teleoperate.sh --robot.cameras=null
#
# Visualization 끄기:
#   ./playground/scripts/teleoperate.sh --display_data=false

sudo chmod 777 /dev/ttyACM0
sudo chmod 777 /dev/ttyACM1

echo "======================================================================"
echo "LeRobot Teleoperation"
echo "======================================================================"
echo ""
echo "Follower (Robot): /dev/ttyACM0"
echo "Leader (Teleoperator): /dev/ttyACM1"
echo "Camera #0: /dev/video0 (top view)"
echo "Camera #2: /dev/video4 (wrist view)"
echo "Visualization: Enabled (Rerun)"
echo ""
echo "실행: lerobot-teleoperate \\"
echo "  --robot.type=so101_follower \\"
echo "  --robot.port=/dev/ttyACM0 \\"
echo "  --robot.id=follower \\"
echo "  --robot.cameras='{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}' \\"
echo "  --teleop.type=so101_leader \\"
echo "  --teleop.port=/dev/ttyACM1 \\"
echo "  --teleop.id=leader \\"
echo "  --display_data=true"
echo ""

v4l2-ctl -d /dev/video0 --set-ctrl=focus_automatic_continuous=0
v4l2-ctl -d /dev/video0 --set-ctrl=focus_absolute=20
v4l2-ctl -d /dev/video4 --set-ctrl=focus_automatic_continuous=0
v4l2-ctl -d /dev/video4 --set-ctrl=focus_absolute=240

# verifing
echo "/dev/video0"
v4l2-ctl -d /dev/video0 --get-ctrl=focus_automatic_continuous,focus_absolute
echo "/dev/video4"
v4l2-ctl -d /dev/video4 --get-ctrl=focus_automatic_continuous,focus_absolute


lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower \
  --robot.cameras='{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=leader \
  --display_data=true \
  "$@"
