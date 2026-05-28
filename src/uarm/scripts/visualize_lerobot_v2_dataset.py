#!/usr/bin/env python3
"""Visualize a local LeRobot v2 dataset episode and print actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def load_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Cannot find LeRobot metadata: {info_path}")
    with info_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def episode_chunk(episode_index: int, chunks_size: int) -> int:
    if chunks_size <= 0:
        return 0
    return episode_index // chunks_size


def episode_path(dataset_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    data_path_template = info.get(
        "data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    )
    chunk = episode_chunk(episode_index, int(info.get("chunks_size", 1000)))
    return dataset_root / data_path_template.format(
        episode_chunk=chunk,
        episode_index=episode_index,
    )


def image_columns(info: dict[str, Any], requested: list[str] | None) -> list[str]:
    features = info.get("features", {})
    columns = [
        name
        for name, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") in {"image", "video"}
    ]
    if requested:
        missing = sorted(set(requested) - set(columns))
        if missing:
            raise ValueError(f"Requested image columns are not in metadata: {missing}")
        return requested
    if not columns:
        raise ValueError("No image/video columns found in meta/info.json")
    return columns


def as_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, list):
        return np.asarray(value)
    if hasattr(value, "as_py"):
        return as_array(value.as_py())
    raise TypeError(f"Unsupported array value type: {type(value)!r}")


def decode_image(value: Any, dataset_root: Path, column: str, episode_index: int) -> np.ndarray:
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            encoded = np.frombuffer(value["bytes"], dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to decode image bytes for column {column}")
            return image

        if value.get("path"):
            image_path = Path(value["path"])
            candidates = []
            if image_path.is_absolute():
                candidates.append(image_path)
            else:
                candidates.extend(
                    [
                        dataset_root / image_path,
                        dataset_root / "images" / column / f"episode_{episode_index:06d}" / image_path.name,
                    ]
                )
            for candidate in candidates:
                image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
                if image is not None:
                    return image
            raise FileNotFoundError(f"Could not load image for {column}; tried {candidates}")

    image = as_array(value)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    raise ValueError(f"Unsupported image shape for {column}: {image.shape}")


def format_vector(value: Any) -> str:
    array = np.asarray(as_array(value), dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{x:.4f}" for x in array) + "]"


def make_mosaic(images: list[tuple[str, np.ndarray]], max_width: int) -> np.ndarray:
    if not images:
        raise ValueError("No images to display")

    rendered = []
    target_height = min(360, max(image.shape[0] for _, image in images))
    for name, image in images:
        scale = target_height / image.shape[0]
        width = max(1, int(image.shape[1] * scale))
        resized = cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)
        cv2.putText(
            resized,
            name,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        rendered.append(resized)

    mosaic = cv2.hconcat(rendered) if len(rendered) > 1 else rendered[0]
    if mosaic.shape[1] > max_width:
        scale = max_width / mosaic.shape[1]
        mosaic = cv2.resize(
            mosaic,
            (max_width, max(1, int(mosaic.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return mosaic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize LeRobot v2 dataset images and print action vectors."
    )
    parser.add_argument("dataset_root", type=Path, help="LeRobot v2 dataset root directory")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to play")
    parser.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        help="Image column to show. Can be passed multiple times. Default: all image columns.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS. Default: dataset fps")
    parser.add_argument("--start", type=int, default=0, help="Start row/frame inside the episode")
    parser.add_argument("--step", type=int, default=1, help="Frame stride")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many shown frames")
    parser.add_argument("--max-width", type=int, default=1600, help="Maximum display mosaic width")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Only print actions; useful on machines without a GUI.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    info = load_info(dataset_root)
    columns = image_columns(info, args.cameras)
    parquet_path = episode_path(dataset_root, info, args.episode)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Cannot find episode parquet: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    fps = float(args.fps if args.fps is not None else info.get("fps", 30))
    delay_ms = max(1, int(round(1000.0 / fps)))

    print(f"Dataset: {dataset_root}")
    print(f"Episode: {args.episode} ({len(df)} frames)")
    print(f"Image columns: {', '.join(columns)}")
    print("Keys: ESC/q quit, SPACE pause/resume, n next frame while paused")

    shown = 0
    paused = False
    indices = range(max(0, args.start), len(df), max(1, args.step))
    for row_index in indices:
        row = df.iloc[row_index]
        action_text = format_vector(row["action"]) if "action" in df.columns else "<missing>"
        frame_index = int(row["frame_index"]) if "frame_index" in df.columns else row_index
        timestamp = float(row["timestamp"]) if "timestamp" in df.columns else float("nan")
        print(
            f"episode={args.episode:06d} frame={frame_index:06d} "
            f"row={row_index:06d} timestamp={timestamp:.4f} action={action_text}",
            flush=True,
        )

        if not args.no_display:
            frames = [
                (column, decode_image(row[column], dataset_root, column, args.episode))
                for column in columns
                if column in df.columns
            ]
            mosaic = make_mosaic(frames, args.max_width)
            cv2.imshow("LeRobot v2 Dataset Viewer", mosaic)

            while True:
                key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
                if key in (27, ord("q")):
                    cv2.destroyAllWindows()
                    return
                if key == ord(" "):
                    paused = not paused
                    continue
                if paused and key == ord("n"):
                    break
                if not paused:
                    break

        shown += 1
        if args.max_frames is not None and shown >= args.max_frames:
            break

    if not args.no_display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
