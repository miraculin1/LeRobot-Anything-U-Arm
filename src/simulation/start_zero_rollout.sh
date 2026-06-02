#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

source /home/ros/miniforge3/bin/activate
conda activate uarm

python src/simulation/rollout_sim.py \
    --robot piper \
    --policy-mode zero \
    --task "put red box to blue plate" \
    --record-cameras d435_top_camera,wrist_camera \
    --record-fps 30
