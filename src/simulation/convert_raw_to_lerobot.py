#!/usr/bin/env python3
"""Convert raw simulation recordings to LeRobot format for OpenPI training."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
INFO_PATH = Path("meta/info.json")
TASKS_PATH = Path("meta/tasks.jsonl")
EPISODES_PATH = Path("meta/episodes.jsonl")
EPISODES_STATS_PATH = Path("meta/episodes_stats.jsonl")
DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_IMAGE_PATH = "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png"


def load_jsonl(path: Path) -> List[dict]:
    frames = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def find_raw_episodes(raw_root: Path) -> List[Path]:
    episodes_root = raw_root / "episodes"
    if not episodes_root.is_dir():
        raise FileNotFoundError(f"Raw episodes directory not found: {episodes_root}")
    episodes = []
    for episode_dir in sorted(episodes_root.glob("episode_*")):
        if (episode_dir / "meta.json").is_file() and (
            episode_dir / "frames.jsonl"
        ).is_file():
            episodes.append(episode_dir)
    if not episodes:
        raise FileNotFoundError(
            f"No complete raw episodes found under: {episodes_root}"
        )
    return episodes


def resolve_output_root(output_parent: Path, output_repo_id: str) -> Path:
    repo_path = Path(output_repo_id)
    if repo_path.is_absolute() or any(part == ".." for part in repo_path.parts):
        raise ValueError(
            "--output-repo-id must be a relative repo id such as 'local/teleop_sim'"
        )
    return (output_parent.expanduser().resolve() / repo_path).resolve()


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
                shapes[camera_name] = tuple(
                    read_rgb_image(episode_dir / rel_path).shape
                )
                break
        if camera_name not in shapes:
            raise ValueError(f"Camera '{camera_name}' has no frames in {episode_dir}")
    return shapes


def build_lerobot_features(camera_shapes: dict) -> dict:
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

    return features


def import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "Conversion requires lerobot in the active Python environment. "
            "Run with: source /home/ros/miniforge3/bin/activate && conda activate uarm"
        ) from exc
    return LeRobotDataset


def create_lerobot_dataset(
    output_root: Path,
    output_repo_id: str,
    fps: int,
    camera_shapes: dict,
    image_writer_processes: int,
    image_writer_threads: int,
):
    LeRobotDataset = import_lerobot_dataset()
    return LeRobotDataset.create(
        repo_id=output_repo_id,
        fps=fps,
        features=build_lerobot_features(camera_shapes),
        root=output_root,
        robot_type="piper",
        use_videos=False,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
    )


def normalize_feature_for_compare(feature: Any) -> Any:
    if isinstance(feature, dict):
        return {key: normalize_feature_for_compare(value) for key, value in feature.items()}
    if isinstance(feature, (list, tuple)):
        return tuple(feature)
    return feature


def validate_resume_compatibility(
    output_root: Path, info: dict, fps: int, camera_shapes: dict
) -> None:
    existing_fps = int(info.get("fps") or 0)
    if existing_fps != fps:
        raise ValueError(
            f"Existing dataset fps={existing_fps} does not match requested/raw fps={fps}. "
            "Use --force to rebuild."
        )

    expected_features = normalize_feature_for_compare(
        build_lerobot_features(camera_shapes)
    )
    existing_features = normalize_feature_for_compare(info.get("features", {}))
    for key, expected in expected_features.items():
        if existing_features.get(key) != expected:
            raise ValueError(
                f"Existing dataset feature mismatch for {key!r} in {output_root}. "
                "Use --force to rebuild."
            )

    expected_keys = set(expected_features)
    existing_data_keys = {
        key
        for key, feature in existing_features.items()
        if isinstance(feature, dict)
        and key not in {"index", "episode_index", "frame_index", "timestamp", "task_index"}
    }
    if existing_data_keys != expected_keys:
        raise ValueError(
            f"Existing dataset feature keys do not match this conversion. "
            f"existing={sorted(existing_data_keys)} expected={sorted(expected_keys)}. "
            "Use --force to rebuild."
        )


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    if chunks_size <= 0:
        return 0
    return episode_index // chunks_size


def episode_data_path(output_root: Path, info: dict, episode_index: int) -> Path:
    data_path = info.get("data_path", DEFAULT_DATA_PATH)
    chunk = episode_chunk(episode_index, int(info.get("chunks_size", 1000)))
    return output_root / data_path.format(
        episode_chunk=chunk, episode_index=episode_index
    )


def episode_image_dir(output_root: Path, image_key: str, episode_index: int) -> Path:
    image_path = DEFAULT_IMAGE_PATH.format(
        image_key=image_key, episode_index=episode_index, frame_index=0
    )
    return (output_root / image_path).parent


def image_feature_keys(info: dict) -> List[str]:
    return [
        key
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict) and feature.get("dtype") == "image"
    ]


def check_existing_episode(
    output_root: Path,
    info: dict,
    episode: dict,
    stats_by_index: dict,
    raw_episode_dir: Path,
) -> Tuple[bool, str]:
    episode_index = int(episode.get("episode_index", -1))
    expected_length = int(episode.get("length", -1))
    raw_frames_path = raw_episode_dir / "frames.jsonl"
    if not raw_frames_path.is_file():
        return False, f"missing raw frames file: {raw_frames_path}"
    raw_length = len(load_jsonl(raw_frames_path))
    if expected_length <= 0:
        return False, f"invalid episode length in metadata: {expected_length}"
    if expected_length != raw_length:
        return (
            False,
            f"metadata length {expected_length} does not match raw length {raw_length}",
        )
    if episode_index not in stats_by_index:
        return False, f"missing episode stats for episode {episode_index}"

    parquet_path = episode_data_path(output_root, info, episode_index)
    if not parquet_path.is_file():
        return False, f"missing parquet: {parquet_path}"

    try:
        import pandas as pd

        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        return False, f"cannot read parquet {parquet_path}: {exc}"

    if len(df) != expected_length:
        return (
            False,
            f"parquet row count {len(df)} does not match metadata length {expected_length}",
        )
    required_columns = {"episode_index", "frame_index", "timestamp", "task_index"}
    required_columns.update(
        key
        for key, feature in info.get("features", {}).items()
        if isinstance(feature, dict)
        and key not in {"index", "episode_index", "frame_index", "timestamp", "task_index"}
    )
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        return False, f"parquet missing columns: {missing_columns}"
    if sorted(set(int(value) for value in df["episode_index"].tolist())) != [episode_index]:
        return False, "parquet contains unexpected episode_index values"
    if df["frame_index"].tolist() != list(range(expected_length)):
        return False, "parquet frame_index is not contiguous from zero"

    for image_key in image_feature_keys(info):
        image_dir = episode_image_dir(output_root, image_key, episode_index)
        if not image_dir.is_dir():
            return False, f"missing image directory: {image_dir}"
        for frame_index in range(expected_length):
            image_path = output_root / DEFAULT_IMAGE_PATH.format(
                image_key=image_key,
                episode_index=episode_index,
                frame_index=frame_index,
            )
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                return False, f"missing or unreadable image: {image_path}"

    return True, ""


def remove_output_episode(output_root: Path, info: dict, episode_index: int) -> None:
    parquet_path = episode_data_path(output_root, info, episode_index)
    if parquet_path.exists():
        parquet_path.unlink()
    for image_key in image_feature_keys(info):
        image_dir = episode_image_dir(output_root, image_key, episode_index)
        if image_dir.exists():
            shutil.rmtree(image_dir)


def rewrite_resume_metadata(output_root: Path, info: dict, keep_episodes: List[dict]) -> None:
    keep_indices = {int(episode["episode_index"]) for episode in keep_episodes}
    stats_rows = load_jsonl(output_root / EPISODES_STATS_PATH)
    keep_stats_rows = [
        row for row in stats_rows if int(row.get("episode_index", -1)) in keep_indices
    ]

    total_frames = sum(int(episode["length"]) for episode in keep_episodes)
    total_episodes = len(keep_episodes)
    chunks_size = int(info.get("chunks_size", 1000))
    total_chunks = 0
    if keep_episodes:
        total_chunks = (
            max(episode_chunk(int(ep["episode_index"]), chunks_size) for ep in keep_episodes)
            + 1
        )

    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_chunks"] = total_chunks
    info["splits"] = {"train": f"0:{total_episodes}"}
    if "total_videos" in info:
        info["total_videos"] = total_episodes * len(
            [
                key
                for key, feature in info.get("features", {}).items()
                if isinstance(feature, dict) and feature.get("dtype") == "video"
            ]
        )

    (output_root / INFO_PATH).write_text(
        json.dumps(info, indent=4, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_jsonl(output_root / EPISODES_PATH, keep_episodes)
    write_jsonl(output_root / EPISODES_STATS_PATH, keep_stats_rows)


def prepare_resume_dataset(
    output_root: Path,
    output_repo_id: str,
    raw_episodes: List[Path],
    fps: int,
    camera_shapes: dict,
    image_writer_processes: int,
    image_writer_threads: int,
):
    info_path = output_root / INFO_PATH
    if not info_path.is_file():
        raise FileNotFoundError(
            f"--resume requires an existing LeRobot dataset with metadata: {info_path}"
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    validate_resume_compatibility(output_root, info, fps, camera_shapes)

    episodes_path = output_root / EPISODES_PATH
    stats_path = output_root / EPISODES_STATS_PATH
    tasks_path = output_root / TASKS_PATH
    missing_metadata_paths = [
        path for path in [episodes_path, stats_path, tasks_path] if not path.is_file()
    ]
    if missing_metadata_paths:
        if int(info.get("total_episodes") or 0) == 0:
            print("[RESUME] Existing output has no saved episodes; recreating dataset.")
            shutil.rmtree(output_root)
            output_root.parent.mkdir(parents=True, exist_ok=True)
            dataset = create_lerobot_dataset(
                output_root,
                output_repo_id,
                fps,
                camera_shapes,
                image_writer_processes,
                image_writer_threads,
            )
            return dataset, 0
        raise FileNotFoundError(
            "Missing resume metadata file(s): "
            + ", ".join(str(path) for path in missing_metadata_paths)
        )

    episode_rows = load_jsonl(episodes_path)
    stats_rows = load_jsonl(stats_path)
    stats_by_index = {int(row["episode_index"]): row for row in stats_rows}
    episode_indices = [int(row.get("episode_index", -1)) for row in episode_rows]
    expected_indices = list(range(len(episode_rows)))
    if episode_indices != expected_indices:
        raise ValueError(
            f"Existing episode metadata must be contiguous from zero; got {episode_indices}"
        )
    if len(episode_rows) > len(raw_episodes):
        raise ValueError(
            f"Existing output has {len(episode_rows)} episodes, but raw input only has "
            f"{len(raw_episodes)} episodes."
        )

    bad_by_index = {}
    for row in episode_rows:
        episode_index = int(row["episode_index"])
        ok, reason = check_existing_episode(
            output_root,
            info,
            row,
            stats_by_index,
            raw_episodes[episode_index],
        )
        if not ok:
            bad_by_index[episode_index] = reason

    if bad_by_index:
        first_bad_index = min(bad_by_index)
        tail_indices = set(range(first_bad_index, len(episode_rows)))
        if set(bad_by_index) != tail_indices:
            details = "; ".join(
                f"episode {idx}: {reason}" for idx, reason in sorted(bad_by_index.items())
            )
            raise RuntimeError(
                "Resume found corruption before complete later episodes. "
                "Only trailing incomplete episodes are cleaned automatically. "
                f"Details: {details}"
            )

        tail_rows = episode_rows[first_bad_index:]
        for row in tail_rows:
            episode_index = int(row["episode_index"])
            remove_output_episode(output_root, info, episode_index)
        keep_rows = episode_rows[:first_bad_index]
        rewrite_resume_metadata(output_root, info, keep_rows)
        print(
            "[RESUME] Removed incomplete trailing output episodes "
            f"{first_bad_index}:{len(episode_rows)} ({bad_by_index[first_bad_index]})"
        )
        episode_rows = keep_rows
    else:
        print(f"[RESUME] Existing output is complete through episode {len(episode_rows) - 1}")

    if not episode_rows:
        print("[RESUME] No complete output episodes remain; recreating dataset.")
        shutil.rmtree(output_root)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        dataset = create_lerobot_dataset(
            output_root,
            output_repo_id,
            fps,
            camera_shapes,
            image_writer_processes,
            image_writer_threads,
        )
        return dataset, 0

    LeRobotDataset = import_lerobot_dataset()
    dataset = LeRobotDataset(repo_id=output_repo_id, root=output_root)
    if image_writer_processes or image_writer_threads:
        dataset.start_image_writer(image_writer_processes, image_writer_threads)
    dataset.episode_buffer = dataset.create_episode_buffer()
    return dataset, len(episode_rows)


def frame_state(frame_meta: dict, episode_dir: Path) -> np.ndarray:
    state = np.asarray(frame_meta["observation_state"], dtype=np.float32)
    if state.shape != (7,):
        raise ValueError(
            f"Expected 7D observation_state in {episode_dir}, got {state.shape}"
        )
    return state


def normalize_piper_action(action: np.ndarray, episode_dir: Path) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape == (7,):
        return action
    if action.shape == (8,):
        if not np.isclose(action[6], action[7], rtol=1e-5, atol=1e-6):
            raise ValueError(
                "Expected 8D Piper teleop_target to have matching left/right "
                f"gripper values in {episode_dir}, got {action[6:8].tolist()}"
            )
        return np.concatenate([action[:6], action[6:7]]).astype(np.float32)
    raise ValueError(
        f"Expected 7D policy action or 8D Piper env action in {episode_dir}, "
        f"got {action.shape}"
    )


def frame_teleop_target(frame_meta: dict, episode_dir: Path) -> np.ndarray:
    if "teleop_target" not in frame_meta:
        raise ValueError(
            f"Raw frame in {episode_dir} does not contain teleop_target. "
            "Use --action-source current_state_delta for older raw recordings."
        )
    action = np.asarray(frame_meta["teleop_target"], dtype=np.float32)
    return normalize_piper_action(action, episode_dir)


def action_from_source(
    frames: List[dict],
    frame_index: int,
    state: np.ndarray,
    previous_state: Optional[np.ndarray],
    episode_dir: Path,
    action_source: str,
) -> np.ndarray:
    if action_source == "teleop_target":
        return frame_teleop_target(frames[frame_index], episode_dir)
    if action_source == "next_state":
        next_index = min(frame_index + 1, len(frames) - 1)
        return frame_state(frames[next_index], episode_dir)
    if action_source == "current_state_delta":
        if previous_state is None:
            return np.zeros(7, dtype=np.float32)
        return (state - previous_state).astype(np.float32)
    raise ValueError(f"Unsupported action source: {action_source}")


def convert_episode(
    dataset, episode_dir: Path, task_override: Optional[str], action_source: str
):
    meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
    frames = load_jsonl(episode_dir / "frames.jsonl")
    if not frames:
        print(f"[WARN] Skipping empty episode: {episode_dir}")
        return

    task = task_override or meta.get("task") or "put red box to blue plate"
    fps = float(meta.get("fps") or 30)
    previous_state = None
    for output_index, frame_meta in enumerate(frames):
        state = frame_state(frame_meta, episode_dir)
        action = action_from_source(
            frames,
            output_index,
            state,
            previous_state,
            episode_dir,
            action_source,
        )

        frame = {
            "observation.state": state,
            "action": action,
        }
        for camera_name, rel_path in frame_meta.get("images", {}).items():
            frame[f"observation.images.{camera_name}"] = read_rgb_image(
                episode_dir / rel_path
            )

        timestamp = float(frame_meta.get("timestamp", output_index / fps))
        dataset.add_frame(frame, task=task, timestamp=timestamp)
        previous_state = state.copy()

    dataset.save_episode()
    print(
        f"[CONVERT] Saved {episode_dir.name}: {len(frames)} frames action_source={action_source}"
    )


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
        help="Output parent directory. Dataset is written to <output-dir>/<output-repo-id>",
    )
    parser.add_argument(
        "--output-repo-id",
        default="local/teleop_sim",
        help="LeRobot repo id stored in dataset metadata",
    )
    parser.add_argument(
        "--fps", type=int, default=None, help="Override FPS; default reads raw metadata"
    )
    parser.add_argument("--task", type=str, default=None, help="Override task string")
    parser.add_argument(
        "--action-source",
        choices=["teleop_target", "next_state", "current_state_delta"],
        default="teleop_target",
        help=(
            "Source for LeRobot action. teleop_target writes absolute mapped robot targets "
            "for OpenPI relative-action training; current_state_delta keeps the previous "
            "sequential delta behavior."
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help="Delete existing output directory first"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate an existing output dataset and continue from the first missing episode",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=0,
        help="Number of async image writer processes used by LeRobotDataset",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=8,
        help="Number of async image writer threads used by LeRobotDataset. Use 0 to disable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_root = args.raw_dir.expanduser().resolve()
    output_root = resolve_output_root(args.output_dir, args.output_repo_id)

    if args.force and args.resume:
        raise ValueError("--force and --resume are mutually exclusive")
    if args.image_writer_processes < 0 or args.image_writer_threads < 0:
        raise ValueError(
            "--image-writer-processes and --image-writer-threads must be non-negative"
        )

    episodes = find_raw_episodes(raw_root)
    first_meta = json.loads((episodes[0] / "meta.json").read_text(encoding="utf-8"))
    first_frames = load_jsonl(episodes[0] / "frames.jsonl")
    fps = int(args.fps or first_meta.get("fps") or 30)
    camera_shapes = infer_camera_shapes(episodes[0], first_meta, first_frames)

    start_index = 0
    if output_root.exists():
        if args.resume:
            dataset, start_index = prepare_resume_dataset(
                output_root,
                args.output_repo_id,
                episodes,
                fps,
                camera_shapes,
                args.image_writer_processes,
                args.image_writer_threads,
            )
        elif not args.force:
            raise FileExistsError(
                f"Output directory already exists. Use --resume to continue or "
                f"--force to replace: {output_root}"
            )
        else:
            shutil.rmtree(output_root)
            output_root.parent.mkdir(parents=True, exist_ok=True)
            dataset = create_lerobot_dataset(
                output_root,
                args.output_repo_id,
                fps,
                camera_shapes,
                args.image_writer_processes,
                args.image_writer_threads,
            )
    else:
        if args.resume:
            raise FileNotFoundError(
                f"--resume requires an existing output directory: {output_root}"
            )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        dataset = create_lerobot_dataset(
            output_root,
            args.output_repo_id,
            fps,
            camera_shapes,
            args.image_writer_processes,
            args.image_writer_threads,
        )

    if start_index >= len(episodes):
        print(f"[RESUME] All {len(episodes)} raw episodes are already converted.")

    for episode_dir in episodes[start_index:]:
        convert_episode(dataset, episode_dir, args.task, args.action_source)

    finalize = getattr(dataset, "finalize", None)
    if callable(finalize):
        finalize()
    stop_image_writer = getattr(dataset, "stop_image_writer", None)
    if callable(stop_image_writer):
        stop_image_writer()
    print(f"[CONVERT] Done: {output_root}")


if __name__ == "__main__":
    main()
