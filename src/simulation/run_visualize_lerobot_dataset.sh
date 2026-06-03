#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/../.."

source /home/ros/miniforge3/bin/activate
conda activate uarm

MODE="${1:-view}"
if [[ $# -gt 0 ]]; then
    shift
fi

DATASET_ROOT="${DATASET_ROOT:-./lerobot_data/eazy_sim_data/local/teleop_sim}"
EPISODE="${EPISODE:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"

case "$MODE" in
    view)
        python src/simulation/visualize_lerobot_dataset.py view \
            "$DATASET_ROOT" \
            --episode "$EPISODE" \
            "$@"
        ;;
    serve)
        python src/simulation/visualize_lerobot_dataset.py serve \
            "$DATASET_ROOT" \
            --episode "$EPISODE" \
            --host "$HOST" \
            --port "$PORT" \
            --chunk-size "$CHUNK_SIZE" \
            --loop \
            "$@"
        ;;
    both)
        python src/simulation/visualize_lerobot_dataset.py both \
            "$DATASET_ROOT" \
            --episode "$EPISODE" \
            --host "$HOST" \
            --port "$PORT" \
            --chunk-size "$CHUNK_SIZE" \
            --loop \
            "$@"
        ;;
    *)
        echo "Usage:"
        echo "  $0 view  [extra visualize args]"
        echo "  $0 serve [extra server args]"
        echo "  $0 both  [extra visualize/server args]"
        echo
        echo "Environment overrides:"
        echo "  DATASET_ROOT=./lerobot_data/eazy_sim_data/local/teleop_sim"
        echo "  EPISODE=0"
        echo "  HOST=0.0.0.0"
        echo "  PORT=8000"
        echo "  CHUNK_SIZE=10"
        exit 2
        ;;
esac
