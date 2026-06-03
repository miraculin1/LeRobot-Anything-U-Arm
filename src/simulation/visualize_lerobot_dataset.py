#!/usr/bin/env python3
"""View a LeRobot dataset episode or replay its actions as an OpenPI policy server."""

from __future__ import annotations

import argparse
import asyncio
import struct
import http
import json
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import websockets
import websockets.asyncio.server as websocket_server


ACTION_COLUMNS = ("action",)
STATE_COLUMNS = ("observation.state", "observation_state")


@dataclass
class LoadedEpisode:
    dataset_root: Path
    info: dict[str, Any]
    episode_index: int
    df: pd.DataFrame
    image_columns: list[str]
    fps: float


class ReplayActionSource:
    def __init__(
        self,
        actions: np.ndarray,
        start: int = 0,
        chunk_size: int = 10,
        loop: bool = False,
    ):
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError(f"Expected actions with shape (N, 7), got {actions.shape}")
        if actions.shape[0] == 0:
            raise ValueError("Cannot serve an empty action sequence")
        self.actions = actions.astype(np.float32)
        self.index = min(max(0, int(start)), max(0, len(actions) - 1))
        self.chunk_size = max(1, int(chunk_size))
        self.loop = loop
        self.lock = threading.Lock()

    def next_chunk(self) -> np.ndarray:
        with self.lock:
            if self.index >= len(self.actions):
                if not self.loop:
                    return self.actions[-1:].copy()
                self.index = 0

            end = min(len(self.actions), self.index + self.chunk_size)
            chunk = self.actions[self.index:end]
            self.index = end

            if len(chunk) < self.chunk_size and self.loop:
                remaining = self.chunk_size - len(chunk)
                wrap_end = min(len(self.actions), remaining)
                chunk = np.concatenate([chunk, self.actions[:wrap_end]], axis=0)
                self.index = wrap_end

            return chunk.astype(np.float32, copy=True)


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


def vector_from_row(row: pd.Series, columns: tuple[str, ...]) -> np.ndarray | None:
    for column in columns:
        if column in row.index:
            return np.asarray(as_array(row[column]), dtype=np.float32).reshape(-1)
    return None


def action_matrix(df: pd.DataFrame) -> np.ndarray:
    if "action" not in df.columns:
        raise ValueError("Dataset episode does not contain an 'action' column")
    values = [np.asarray(as_array(value), dtype=np.float32).reshape(-1) for value in df["action"]]
    actions = np.stack(values, axis=0)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected action column to contain 7D vectors, got {actions.shape}")
    return actions


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
                        dataset_root
                        / "images"
                        / column
                        / f"episode_{episode_index:06d}"
                        / image_path.name,
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


def format_vector(value: np.ndarray | None) -> str:
    if value is None:
        return "<missing>"
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{x:.4f}" for x in array) + "]"


def put_text_block(image: np.ndarray, lines: list[str]) -> np.ndarray:
    rendered = image.copy()
    y = 26
    for line in lines:
        cv2.putText(
            rendered,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            rendered,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        y += 22
    return rendered


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


def load_episode(
    dataset_root: Path,
    episode_index: int,
    requested_cameras: list[str] | None,
    fps_override: float | None,
) -> LoadedEpisode:
    dataset_root = dataset_root.expanduser().resolve()
    info = load_info(dataset_root)
    columns = image_columns(info, requested_cameras)
    parquet_path = episode_path(dataset_root, info, episode_index)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Cannot find episode parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    fps = float(fps_override if fps_override is not None else info.get("fps", 30))
    return LoadedEpisode(dataset_root, info, episode_index, df, columns, fps)


def visualize_episode(
    episode: LoadedEpisode,
    start: int,
    step: int,
    max_frames: int | None,
    max_width: int,
    no_display: bool,
    stop_event: threading.Event | None = None,
) -> None:
    delay_ms = max(1, int(round(1000.0 / episode.fps)))
    print(f"Dataset: {episode.dataset_root}")
    print(f"Episode: {episode.episode_index} ({len(episode.df)} frames)")
    print(f"Image columns: {', '.join(episode.image_columns)}")
    if not no_display:
        print("Keys: ESC/q quit, SPACE pause/resume, n next frame while paused")

    shown = 0
    paused = False
    indices = range(max(0, start), len(episode.df), max(1, step))
    for row_index in indices:
        if stop_event is not None and stop_event.is_set():
            break

        row = episode.df.iloc[row_index]
        action = vector_from_row(row, ACTION_COLUMNS)
        state = vector_from_row(row, STATE_COLUMNS)
        frame_index = int(row["frame_index"]) if "frame_index" in episode.df.columns else row_index
        timestamp = float(row["timestamp"]) if "timestamp" in episode.df.columns else float("nan")
        action_text = format_vector(action)
        state_text = format_vector(state)
        print(
            f"episode={episode.episode_index:06d} frame={frame_index:06d} "
            f"row={row_index:06d} timestamp={timestamp:.4f} "
            f"state={state_text} action={action_text}",
            flush=True,
        )

        if not no_display:
            frames = [
                (
                    column,
                    decode_image(row[column], episode.dataset_root, column, episode.episode_index),
                )
                for column in episode.image_columns
                if column in episode.df.columns
            ]
            mosaic = make_mosaic(frames, max_width)
            mosaic = put_text_block(
                mosaic,
                [
                    f"ep {episode.episode_index:06d} frame {frame_index:06d} t={timestamp:.3f}",
                    f"state {state_text}",
                    f"action {action_text}",
                ],
            )
            cv2.imshow("LeRobot Dataset Viewer", mosaic)

            while True:
                key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
                if key in (27, ord("q")):
                    if stop_event is not None:
                        stop_event.set()
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
        if max_frames is not None and shown >= max_frames:
            break

    if not no_display:
        cv2.destroyAllWindows()


def import_msgpack_numpy():
    try:
        from openpi_client import msgpack_numpy
    except ImportError:
        try:
            import msgpack_numpy
        except ImportError as exc:
            try:
                import msgpack
            except ImportError:
                msgpack = None

            class LocalMsgpackNumpy:
                @classmethod
                def packb(cls, obj):
                    if isinstance(obj, np.ndarray):
                        obj = cls.pack_array(obj)
                    if isinstance(obj, np.generic):
                        obj = cls.pack_array(obj)
                    if obj is None:
                        return b"\xc0"
                    if obj is True:
                        return b"\xc3"
                    if obj is False:
                        return b"\xc2"
                    if isinstance(obj, int):
                        if 0 <= obj <= 0x7F:
                            return bytes([obj])
                        if -32 <= obj < 0:
                            return struct.pack("b", obj)
                        if 0 <= obj <= 0xFFFFFFFF:
                            return b"\xce" + struct.pack(">I", obj)
                        return b"\xd3" + struct.pack(">q", obj)
                    if isinstance(obj, float):
                        return b"\xcb" + struct.pack(">d", obj)
                    if isinstance(obj, str):
                        data = obj.encode("utf-8")
                        return cls.pack_str_header(len(data)) + data
                    if isinstance(obj, bytes):
                        return cls.pack_bin_header(len(obj)) + obj
                    if isinstance(obj, (list, tuple)):
                        return cls.pack_array_header(len(obj)) + b"".join(cls.packb(v) for v in obj)
                    if isinstance(obj, dict):
                        return cls.pack_map_header(len(obj)) + b"".join(
                            cls.packb(k) + cls.packb(v) for k, v in obj.items()
                        )
                    raise TypeError(f"Cannot msgpack encode {type(obj)!r}")

                @staticmethod
                def pack_str_header(length):
                    if length <= 31:
                        return bytes([0xA0 | length])
                    if length <= 0xFF:
                        return b"\xd9" + struct.pack(">B", length)
                    if length <= 0xFFFF:
                        return b"\xda" + struct.pack(">H", length)
                    return b"\xdb" + struct.pack(">I", length)

                @staticmethod
                def pack_bin_header(length):
                    if length <= 0xFF:
                        return b"\xc4" + struct.pack(">B", length)
                    if length <= 0xFFFF:
                        return b"\xc5" + struct.pack(">H", length)
                    return b"\xc6" + struct.pack(">I", length)

                @staticmethod
                def pack_array_header(length):
                    if length <= 15:
                        return bytes([0x90 | length])
                    if length <= 0xFFFF:
                        return b"\xdc" + struct.pack(">H", length)
                    return b"\xdd" + struct.pack(">I", length)

                @staticmethod
                def pack_map_header(length):
                    if length <= 15:
                        return bytes([0x80 | length])
                    if length <= 0xFFFF:
                        return b"\xde" + struct.pack(">H", length)
                    return b"\xdf" + struct.pack(">I", length)

                @staticmethod
                def pack_array(obj):
                    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
                        raise ValueError(f"Unsupported dtype: {obj.dtype}")
                    if isinstance(obj, np.ndarray):
                        return {
                            b"__ndarray__": True,
                            b"data": obj.tobytes(),
                            b"dtype": obj.dtype.str,
                            b"shape": obj.shape,
                        }
                    if isinstance(obj, np.generic):
                        return {
                            b"__npgeneric__": True,
                            b"data": obj.item(),
                            b"dtype": obj.dtype.str,
                        }
                    return obj

                @staticmethod
                def unpack_array(obj):
                    if b"__ndarray__" in obj:
                        return np.ndarray(
                            buffer=obj[b"data"],
                            dtype=np.dtype(obj[b"dtype"]),
                            shape=obj[b"shape"],
                        )
                    if b"__npgeneric__" in obj:
                        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
                    return obj

                @classmethod
                def Packer(cls):
                    if msgpack is not None:
                        return msgpack.Packer(default=cls.pack_array)

                    class Packer:
                        def pack(self, obj):
                            return cls.packb(obj)

                    return Packer()

                @classmethod
                def unpackb(cls, data):
                    return msgpack.unpackb(data, object_hook=cls.unpack_array)

            return LocalMsgpackNumpy
    return msgpack_numpy


def health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def serve_policy(args: argparse.Namespace, episode: LoadedEpisode, stop_event: threading.Event):
    msgpack_numpy = import_msgpack_numpy()
    packer = msgpack_numpy.Packer()
    action_source = ReplayActionSource(
        action_matrix(episode.df),
        start=args.start,
        chunk_size=args.chunk_size,
        loop=args.loop,
    )
    metadata = {
        "source": "lerobot_dataset_replay",
        "dataset_root": str(episode.dataset_root),
        "episode": episode.episode_index,
        "action_shape": [7],
        "chunk_size": args.chunk_size,
        "loop": args.loop,
    }

    async def handler(websocket):
        await websocket.send(packer.pack(metadata))
        while not stop_event.is_set():
            try:
                await websocket.recv()
                infer_start = time.monotonic()
                actions = action_source.next_chunk()
                response = {
                    "actions": actions,
                    "server_timing": {"infer_ms": (time.monotonic() - infer_start) * 1000.0},
                }
                await websocket.send(packer.pack(response))
                print(
                    f"[SERVE] sent actions shape={actions.shape} next_index={action_source.index}",
                    flush=True,
                )
            except websockets.ConnectionClosed:
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=1011, reason="Internal server error")
                raise

    print(f"[SERVE] Dataset: {episode.dataset_root}")
    print(f"[SERVE] Episode: {episode.episode_index} ({len(episode.df)} frames)")
    print(f"[SERVE] Listening on ws://{args.host}:{args.port}")
    async with websocket_server.serve(
        handler,
        args.host,
        args.port,
        compression=None,
        max_size=None,
        process_request=health_check,
    ):
        while not stop_event.is_set():
            await asyncio.sleep(0.1)


def run_server(args: argparse.Namespace, episode: LoadedEpisode, stop_event: threading.Event):
    try:
        asyncio.run(serve_policy(args, episode, stop_event))
    except KeyboardInterrupt:
        stop_event.set()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset_root", type=Path, help="Converted LeRobot dataset root")
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        help="Image column to show. Can be passed multiple times. Default: all image columns.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS override")
    parser.add_argument("--start", type=int, default=0, help="Start row/frame")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize LeRobot datasets or replay actions as an OpenPI policy server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    view = subparsers.add_parser("view", help="Visualize images, state, and action")
    add_common_args(view)
    view.add_argument("--step", type=int, default=1, help="Frame stride")
    view.add_argument("--max-frames", type=int, default=None, help="Stop after N shown frames")
    view.add_argument("--max-width", type=int, default=1600, help="Maximum display mosaic width")
    view.add_argument("--no-display", action="store_true", help="Only print values")

    serve = subparsers.add_parser("serve", help="Serve dataset actions as an OpenPI policy server")
    add_common_args(serve)
    serve.add_argument("--host", type=str, default="0.0.0.0", help="Websocket bind host")
    serve.add_argument("--port", type=int, default=8000, help="Websocket bind port")
    serve.add_argument("--chunk-size", type=int, default=10, help="Number of actions per response")
    serve.add_argument("--loop", action="store_true", help="Loop when reaching episode end")

    both = subparsers.add_parser("both", help="Serve actions and visualize the same episode")
    add_common_args(both)
    both.add_argument("--host", type=str, default="0.0.0.0", help="Websocket bind host")
    both.add_argument("--port", type=int, default=8000, help="Websocket bind port")
    both.add_argument("--chunk-size", type=int, default=10, help="Number of actions per response")
    both.add_argument("--loop", action="store_true", help="Loop when reaching episode end")
    both.add_argument("--step", type=int, default=1, help="Frame stride")
    both.add_argument("--max-frames", type=int, default=None, help="Stop after N shown frames")
    both.add_argument("--max-width", type=int, default=1600, help="Maximum display mosaic width")
    both.add_argument("--no-display", action="store_true", help="Only print values")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode = load_episode(args.dataset_root, args.episode, args.cameras, args.fps)
    stop_event = threading.Event()

    if args.mode == "view":
        visualize_episode(
            episode,
            start=args.start,
            step=args.step,
            max_frames=args.max_frames,
            max_width=args.max_width,
            no_display=args.no_display,
        )
        return

    if args.mode == "serve":
        run_server(args, episode, stop_event)
        return

    if args.mode == "both":
        server_thread = threading.Thread(
            target=run_server,
            args=(args, episode, stop_event),
            daemon=True,
        )
        server_thread.start()
        try:
            visualize_episode(
                episode,
                start=args.start,
                step=args.step,
                max_frames=args.max_frames,
                max_width=args.max_width,
                no_display=args.no_display,
                stop_event=stop_event,
            )
        finally:
            stop_event.set()
            server_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
