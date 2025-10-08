export PYTHONPATH="$PYTHONPATH:/home/csi-lerobot-client/lerobot_src_oldver/lerobot/src"

sudo chmod 777 /dev/ttyACM0

# v4l2 auto_focus on
# v4l2-ctl -d /dev/video0 --set-ctrl=focus_automatic_continuous=1
# v4l2-ctl -d /dev/video2 --set-ctrl=focus_automatic_continuous=1

# # v4l2 auto_focus off
v4l2-ctl -d /dev/video0 --set-ctrl=focus_automatic_continuous=0
v4l2-ctl -d /dev/video0 --set-ctrl=focus_absolute=20
v4l2-ctl -d /dev/video4 --set-ctrl=focus_automatic_continuous=0
v4l2-ctl -d /dev/video4 --set-ctrl=focus_absolute=240

# verifing
echo "/dev/video0"
v4l2-ctl -d /dev/video0 --get-ctrl=focus_automatic_continuous,focus_absolute
echo "/dev/video4"
v4l2-ctl -d /dev/video4 --get-ctrl=focus_automatic_continuous,focus_absolute


python -m lerobot.scripts.server.robot_client \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
  --robot.id=follower \
  --task="Grab the block of the red color and put it in the cup" \
  --server_address=115.145.173.248:8080 \
  --policy_type=smolvla \
  --pretrained_name_or_path=vpraise00/lerobot_singletask_10k_iter \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --debug_visualize_queue_size=True 

# Grab the blue block and put it in the cup
# Grab the block of the blue color and put it in the cup