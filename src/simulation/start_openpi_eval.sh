#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

eval_args=("$@")
set --
source /home/ros/miniforge3/bin/activate
conda activate uarm
set -- "${eval_args[@]}"

python src/simulation/evaluate_openpi_rollout.py \
    --robot piper \
    --policy-mode openpi \
    --host localhost \
    --port 8000 \
    --prompt "put red box to blue plate" \
    --open-loop-horizon 10 \
    --action-mode absolute \
    --render-mode sensors \
    --display-cameras \
    --no-env-render \
    --no-render-preflight \
    --randomize-all-task-objects \
    --randomize-object-yaw \
    --random-workspace-inner-diameter 0.60 \
    --random-workspace-outer-diameter 1.20 \
    "$@"
