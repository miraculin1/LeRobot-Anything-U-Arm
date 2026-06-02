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
    --prompt "put red box to blue plate" \
    --open-loop-horizon 10 \
    --action-mode delta \
    --no-record
