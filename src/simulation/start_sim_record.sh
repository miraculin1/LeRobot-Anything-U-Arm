#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

source /home/ros/miniforge3/bin/activate
conda activate uarm

echo "[INFO] Raw recording includes initial object poses and per-frame end-effector poses."

python src/simulation/teleop_sim.py \
    --robot piper \
    --render-mode sensors \
    --no-env-render \
    --control-dwell 0.0 \
    --no-render-preflight \
    --randomize-all-task-objects \
    --randomize-object-yaw \
    --random-workspace-inner-diameter 0.60 \
    --random-workspace-outer-diameter 1.20 \
    --record \
    --record-dir ./raw_data/hard_sim_data \
    --repo-id local/teleop_sim \
    --task "put red box to blue plate" \
    --record-cameras d435_top_camera,wrist_camera \
    --record-fps 30 \
    --image-writer-processes 0 \
    --image-writer-threads 8 \
    --raw-writer-queue-size 256
