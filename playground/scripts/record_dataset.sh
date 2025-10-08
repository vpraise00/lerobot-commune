sudo chmod 777 /dev/ttyACM0
sudo chmod 777 /dev/ttyACM1

v4l2-ctl -d /dev/video0 --set-ctrl=focus_automatic_continuous=0
v4l2-ctl -d /dev/video0 --set-ctrl=focus_absolute=20
v4l2-ctl -d /dev/video2 --set-ctrl=focus_automatic_continuous=0
v4l2-ctl -d /dev/video2 --set-ctrl=focus_absolute=400

# verifing
echo "/dev/video0"
v4l2-ctl -d /dev/video0 --get-ctrl=focus_automatic_continuous,focus_absolute
echo "/dev/video2"
v4l2-ctl -d /dev/video2 --get-ctrl=focus_automatic_continuous,focus_absolute


# blue block
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader \
    --display_data=true \
    --dataset.repo_id=vpraise00/multitask_pnp \
    --dataset.single_task="Grab the blue block and put it in the cup" \
    --dataset.push_to_hub=False \
    --dataset.episode_time_s=200 \
    --dataset.num_episodes=48 \
    --dataset.reset_time_s=15


# # red block
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader \
    --display_data=true \
    --dataset.repo_id=vpraise00/multitask_pnp \
    --dataset.single_task="Grab the red block and put it in the cup" \
    --dataset.push_to_hub=False \
    --dataset.episode_time_s=200 \
    --dataset.num_episodes=48 \
    --dataset.reset_time_s=15 \
    --resume=true

# # green block
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader \
    --display_data=true \
    --dataset.repo_id=vpraise00/multitask_pnp \
    --dataset.single_task="Grab the green block and put it in the cup" \
    --dataset.push_to_hub=False \
    --dataset.episode_time_s=200 \
    --dataset.num_episodes=48 \
    --dataset.reset_time_s=15 \
    --resume=true


# Grab the blue block and put it in the pink cup
# Grab the green block and put it in the pink cup
# Grab the red block and put it in the pink cup