#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

source /home/ros/miniforge3/bin/activate
conda activate uarm

python src/simulation/rollout_sim.py \
    --robot piper \
    --policy-mode openpi \
    --host localhost \
    --port 8000 \
    --prompt "put red box to yellow plate" \
    --open-loop-horizon 10 \
    --action-mode absolute \
    --render-mode sensors \
    --no-env-render \
    --no-render-preflight \
    --randomize-all-task-objects \
    --randomize-object-yaw \
    --random-workspace-inner-diameter 0.60 \
    --random-workspace-outer-diameter 1.20 \
    --no-show-gripper-plot \
    --no-record
