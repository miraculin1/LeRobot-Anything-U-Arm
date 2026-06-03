#!/usr/bin/env python3
"""Convert raw simulation recordings to the current LeRobot delta-action format."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


def load_jsonl(path: Path) -> List[dict]:
    frames = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def find_raw_episodes(raw_root: Path) -> List[Path]:
    episodes_root = raw_root / "episodes"
    if not episodes_root.is_dir():
        raise FileNotFoundError(f"Raw episodes directory not found: {episodes_root}")
    episodes = []
    for episode_dir in sorted(episodes_root.glob("episode_*")):
        if (episode_dir / "meta.json").is_file() and (episode_dir / "frames.jsonl").is_file():
            episodes.append(episode_dir)
    if not episodes:
        raise FileNotFoundError(f"No complete raw episodes found under: {episodes_root}")
    return episodes


def read_rgb_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def infer_camera_shapes(
    episode_dir: Path, meta: dict, frames: List[dict]
) -> Dict[str, Tuple[int, int, int]]:
    shapes = {}
    for camera_name in meta.get("record_cameras", []):
        meta_shape = meta.get("camera_shapes", {}).get(camera_name)
        if meta_shape:
            shapes[camera_name] = tuple(int(v) for v in meta_shape)
            continue
        for frame in frames:
            rel_path = frame.get("images", {}).get(camera_name)
            if rel_path:
                shapes[camera_name] = tuple(read_rgb_image(episode_dir / rel_path).shape)
                break
        if camera_name not in shapes:
            raise ValueError(f"Camera '{camera_name}' has no frames in {episode_dir}")
    return shapes


def create_lerobot_dataset(output_root: Path, output_repo_id: str, fps: int, camera_shapes: dict):
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "Conversion requires lerobot in the active Python environment. "
            "Run with: source /home/ros/miniforge3/bin/activate && conda activate uarm"
        ) from exc

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": JOINT_NAMES,
        },
    }
    for camera_name, shape in camera_shapes.items():
        features[f"observation.images.{camera_name}"] = {
            "dtype": "image",
            "shape": tuple(shape),
            "names": ["height", "width", "channels"],
        }

    return LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=fps,
        features=features,
        root=output_root,
        robot_type="piper",
        use_videos=False,
    )


def convert_episode(dataset, episode_dir: Path, task_override: Optional[str]):
    meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
    frames = load_jsonl(episode_dir / "frames.jsonl")
    if not frames:
        print(f"[WARN] Skipping empty episode: {episode_dir}")
        return

    task = task_override or meta.get("task") or "put red box to blue plate"
    fps = float(meta.get("fps") or 30)
    previous_state = None
    for output_index, frame_meta in enumerate(frames):
        state = np.asarray(frame_meta["observation_state"], dtype=np.float32)
        if state.shape != (7,):
            raise ValueError(f"Expected 7D observation_state in {episode_dir}, got {state.shape}")
        if previous_state is None:
            action = np.zeros(7, dtype=np.float32)
        else:
            action = (state - previous_state).astype(np.float32)

        frame = {
            "observation.state": state,
            "action": action,
        }
        for camera_name, rel_path in frame_meta.get("images", {}).items():
            frame[f"observation.images.{camera_name}"] = read_rgb_image(episode_dir / rel_path)

        timestamp = float(frame_meta.get("timestamp", output_index / fps))
        dataset.add_frame(frame, task=task, timestamp=timestamp)
        previous_state = state.copy()

    dataset.save_episode()
    print(f"[CONVERT] Saved {episode_dir.name}: {len(frames)} frames")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert uarm_sim_raw_v1 raw recordings to LeRobot v2-style local datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", required=True, type=Path, help="Raw dataset root")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output LeRobot dataset root path, usually <parent>/<repo-id>",
    )
    parser.add_argument(
        "--output-repo-id",
        default="local/teleop_sim",
        help="LeRobot repo id stored in dataset metadata",
    )
    parser.add_argument("--fps", type=int, default=None, help="Override FPS; default reads raw metadata")
    parser.add_argument("--task", type=str, default=None, help="Override task string")
    parser.add_argument("--force", action="store_true", help="Delete existing output directory first")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_root = args.raw_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()

    episodes = find_raw_episodes(raw_root)
    first_meta = json.loads((episodes[0] / "meta.json").read_text(encoding="utf-8"))
    first_frames = load_jsonl(episodes[0] / "frames.jsonl")
    fps = int(args.fps or first_meta.get("fps") or 30)
    camera_shapes = infer_camera_shapes(episodes[0], first_meta, first_frames)

    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"Output directory already exists. Use --force to replace: {output_root}")
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)

    dataset = create_lerobot_dataset(output_root, args.output_repo_id, fps, camera_shapes)
    for episode_dir in episodes:
        convert_episode(dataset, episode_dir, args.task)

    finalize = getattr(dataset, "finalize", None)
    if callable(finalize):
        finalize()
    stop_image_writer = getattr(dataset, "stop_image_writer", None)
    if callable(stop_image_writer):
        stop_image_writer()
    print(f"[CONVERT] Done: {output_root}")


if __name__ == "__main__":
    main()
