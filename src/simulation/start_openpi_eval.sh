#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

eval_args=("$@")
default_box_args=(--target-box red)
default_plate_args=(--target-plate green)
for arg in "${eval_args[@]}"; do
    case "$arg" in
        --prompt|--prompt=*)
            default_box_args=()
            default_plate_args=()
            ;;
        --target-box|--target-box=*)
            default_box_args=()
            ;;
        --target-plate|--target-plate=*)
            default_plate_args=()
            ;;
    esac
done
set --
source /home/ros/miniforge3/bin/activate
conda activate uarm
set -- "${eval_args[@]}"

python src/simulation/evaluate_openpi_rollout.py \
    --robot piper \
    --policy-mode openpi \
    --host localhost \
    --port 8000 \
    "${default_box_args[@]}" \
    "${default_plate_args[@]}" \
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
