#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

source /home/ros/miniforge3/bin/activate
conda activate uarm

python src/simulation/teleop_sim.py \
    --robot piper \
    --record \
    --record-dir ./lerobot_data/eazy_sim_data \
    --repo-id local/teleop_sim \
    --task "put red box to blue plate" \
    --record-cameras d435_top_camera,wrist_camera \
    --record-fps 30
