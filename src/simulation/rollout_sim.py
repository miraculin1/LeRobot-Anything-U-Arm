import argparse
import os
import select
import signal
import sys
import time
from contextlib import contextmanager
from threading import Event, Lock, Thread

import gymnasium as gym
import mani_skill.envs  # Must import to register all env/agent
import numpy as np
from transforms3d.euler import euler2quat

try:
    from teleop_sim import (
        ServoTeleoperatorSim,
        build_camera_shader_config,
        normalize_shader_pack,
        PREBUILT_SHADER_CONFIGS,
        SHADER_PACK_ALIASES,
    )
except ModuleNotFoundError:
    from .teleop_sim import (
        ServoTeleoperatorSim,
        build_camera_shader_config,
        normalize_shader_pack,
        PREBUILT_SHADER_CONFIGS,
        SHADER_PACK_ALIASES,
    )


DEFAULT_INITIAL_STATE = np.array(
    [
        0.010968562901414746,
        0.0006513808895519403,
        -0.002636399535867838,
        -0.018102952698976972,
        0.022893221634536675,
        0.06279406425217975,
        0.03781127277761698,
    ],
    dtype=np.float32,
)


class TimingStats:
    def __init__(self, enabled: bool, interval: int, warmup: int, target_period: float):
        self.enabled = enabled
        self.interval = max(1, int(interval))
        self.warmup = max(0, int(warmup))
        self.target_period = target_period
        self.step = 0
        self.data = {}

    @contextmanager
    def time(self, key: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(key, time.perf_counter() - start)

    def record(self, key: str, seconds: float):
        if not self.enabled or self.step < self.warmup:
            return
        values = self.data.setdefault(key, [])
        values.append(float(seconds) * 1000.0)

    def end_step(self, label: str):
        if not self.enabled:
            return
        self.step += 1
        if self.step <= self.warmup:
            return
        measured_steps = self.step - self.warmup
        if measured_steps % self.interval != 0:
            return
        self.print_summary(label, measured_steps)
        self.data.clear()

    def print_summary(self, label: str, measured_steps: int):
        loop_values = self.data.get("loop.total", [])
        avg_loop_ms = float(np.mean(loop_values)) if loop_values else 0.0
        fps = 1000.0 / avg_loop_ms if avg_loop_ms > 0 else 0.0
        target_ms = self.target_period * 1000.0
        status = "OVERRUN" if avg_loop_ms > target_ms else "OK"
        print(
            f"[TIMING] {label} steps={measured_steps} "
            f"avg_loop={avg_loop_ms:.1f}ms fps={fps:.1f} "
            f"target={target_ms:.1f}ms {status}"
        )
        for key in sorted(self.data):
            values = self.data[key]
            if not values:
                continue
            print(
                f"[TIMING]   {key}: "
                f"avg={np.mean(values):.1f}ms max={np.max(values):.1f}ms "
                f"last={values[-1]:.1f}ms n={len(values)}"
            )


class ZeroActionRolloutSim(ServoTeleoperatorSim):
    """Piper simulation rollout without serial or teleoperation threads."""

    def __init__(
        self,
        scene: str,
        robot_uids: str,
        object_pos=None,
        object_size: float = 0.04,
        spawn_object: bool = True,
        render_mode: str = "sensors",
        wrist_camera_width: int = 320,
        wrist_camera_height: int = 320,
        show_wrist_camera: bool = False,
        wrist_camera_display_rate: float = 10.0,
        wrist_camera_display_scale: float = 2.0,
        shader_pack: str = "fast_rt",
        rt_samples_per_pixel: int = 2,
        rt_path_depth: int = 1,
        rt_denoiser: str = "oidn",
        render_preflight: bool = True,
        record: bool = False,
        record_dir: str = "./lerobot_data/eazy_sim_data",
        repo_id: str = "local/teleop_sim",
        task: str = "put red box to blue plate",
        record_cameras=None,
        record_fps: int = 30,
        rate: float = 30.0,
        display_cameras: bool = True,
        initial_state=None,
        debug_timing: bool = False,
        debug_interval: int = 30,
        debug_warmup: int = 5,
        env_render: bool = True,
    ):
        self.SERIAL_PORT = None
        self.BAUDRATE = None
        self.ser = None

        self.scene = scene
        self.robot_uids = robot_uids
        self.gripper_range = 0.43
        self.object_pos = object_pos if object_pos is not None else [0.457, -1.612, 0.956]
        self.object_size = object_size
        self.spawn_object = spawn_object
        self.grasp_object = None
        self.render_mode = render_mode
        self.wrist_camera_width = wrist_camera_width
        self.wrist_camera_height = wrist_camera_height
        self.show_wrist_camera = show_wrist_camera
        self.wrist_camera_display_period = 1.0 / max(wrist_camera_display_rate, 1e-6)
        self.wrist_camera_display_scale = max(wrist_camera_display_scale, 1e-6)
        self.shader_pack = shader_pack
        self.normalized_shader_pack = normalize_shader_pack(shader_pack)
        self.rt_samples_per_pixel = rt_samples_per_pixel
        self.rt_path_depth = rt_path_depth
        self.rt_denoiser = rt_denoiser
        self.render_preflight = render_preflight
        self.env_render = env_render
        self.default_render_sensor_names = ("d435_top_camera", "wrist_camera")
        self.show_default_sensor_cameras = display_cameras
        self._camera_window_initialized = {}
        self._last_wrist_camera_display_time = 0.0
        self.cv2 = None
        if self.show_wrist_camera:
            if self.robot_uids != "piper":
                raise ValueError("--show-wrist-camera is only supported for --robot piper")
        if self.show_wrist_camera or self.show_default_sensor_cameras:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError(
                    "Camera display requires OpenCV. Install opencv-python "
                    "in this environment, then run again."
                ) from exc
            self.cv2 = cv2

        self.zero_angles = [0.0] * 7
        self.sim_init_angles = [0.0] * 7
        self.stop_event = Event()
        self.rate = float(rate)
        self.timing = TimingStats(
            enabled=debug_timing,
            interval=debug_interval,
            warmup=debug_warmup,
            target_period=max(1.0 / self.rate, 1e-6),
        )
        self.record_enabled = record
        self.record_dir = os.path.expanduser(record_dir)
        self.repo_id = repo_id
        self.task = "put red box to blue plate"
        if task != self.task:
            print(f"[WARN] Ignoring --task '{task}'. Fixed task is: {self.task}")
        self.record_cameras = tuple(record_cameras or ("d435_top_camera", "wrist_camera"))
        self.record_fps = int(record_fps)
        self.record_period = 1.0 / max(float(self.record_fps), 1.0)
        self.dataset = None
        self.record_state = "disabled" if not self.record_enabled else "initializing"
        self.record_lock = Lock()
        self.pending_stop_episode = False
        self.pending_episode_decision = None
        self.pending_quit = False
        self.exit_after_episode_decision = False
        self.last_record_time = 0.0
        self.last_recorded_state = None
        self.current_episode_frames = 0
        self.countdown_end_time = None
        self.status_message = "Recording disabled"
        self._ui_buttons = {}
        self.plates = []
        self.plate_specs = [
            ("blue_plate", [0.05, 0.18, 1.0, 1.0]),
            ("green_plate", [0.1, 0.7, 0.2, 1.0]),
            ("yellow_plate", [1.0, 0.85, 0.05, 1.0]),
        ]
        self.plate_radius = 0.045
        self.plate_half_height = 0.004
        self.plate_quat = euler2quat(0, np.pi / 2, 0)
        self.random_workspace = dict(x=(0.395, 0.673), y=(-1.959, -0.991))
        self.min_object_spacing = 0.10
        self.fixed_red_box_xy = np.array([0.457, -1.612], dtype=np.float64)
        self.fixed_blue_plate_xy = np.array([0.577, -1.612], dtype=np.float64)
        self.initial_state = self._validate_initial_state(initial_state)
        self.initial_env_action = self.piper_state_to_env_action(self.initial_state)

        if robot_uids == "x_fetch":
            self.control_mode = "pd_joint_pos_dual_arm"
        elif robot_uids == "unitree_h1":
            self.control_mode = "pd_joint_pos"
        else:
            self.control_mode = "pd_joint_pos"

        camera_shader_config = build_camera_shader_config(
            shader_pack=self.shader_pack,
            rt_samples_per_pixel=self.rt_samples_per_pixel,
            rt_path_depth=self.rt_path_depth,
            rt_denoiser=self.rt_denoiser,
        )
        sensor_configs = dict(shader_config=camera_shader_config)
        if robot_uids == "piper":
            sensor_configs["wrist_camera"] = dict(
                width=self.wrist_camera_width,
                height=self.wrist_camera_height,
                shader_config=camera_shader_config,
            )

        try:
            self.env = gym.make(
                scene,
                robot_uids=robot_uids,
                render_mode=self.render_mode,
                control_mode=self.control_mode,
                sensor_configs=sensor_configs,
                human_render_camera_configs=dict(shader_config=camera_shader_config),
                viewer_camera_configs=dict(shader_config=camera_shader_config),
                sim_config=dict(
                    default_materials_config=dict(
                        static_friction=10.0,
                        dynamic_friction=10.0,
                        restitution=0.0,
                    )
                ),
            )
        except Exception:
            self._print_render_diagnostics()
            raise

        self.env.reset(seed=0)
        print("Action space:", self.env.action_space)
        if self.spawn_object:
            self._spawn_grasp_object()
            self._spawn_plates()
            self._randomize_task_objects()
        self._set_initial_robot_state()

        self._setup_camera_pose()
        if self.render_preflight:
            self._run_render_preflight()
        if self.record_enabled:
            self._setup_lerobot_dataset()
            self._begin_recording_episode()

    def zero_policy_action(self) -> np.ndarray:
        return np.zeros(7, dtype=np.float32)

    def zero_env_action(self) -> np.ndarray:
        if self.robot_uids != "piper":
            action_shape = self.env.action_space.shape[0]
            return np.zeros(action_shape, dtype=np.float32)
        return self.initial_env_action.copy()

    def _validate_initial_state(self, initial_state) -> np.ndarray:
        state = DEFAULT_INITIAL_STATE if initial_state is None else initial_state
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (7,):
            raise ValueError(f"Expected --initial-state to contain 7 floats, got {state.shape}")
        return state

    def _set_initial_robot_state(self):
        if self.robot_uids != "piper":
            return
        agent = getattr(self.env.unwrapped, "agent", None)
        if agent is None:
            raise RuntimeError("Cannot set initial robot state: env has no agent")
        agent.reset(init_qpos=self.initial_env_action)
        controller = getattr(agent, "controller", None)
        reset_controller = getattr(controller, "reset", None)
        if callable(reset_controller):
            reset_controller()
        print(f"[INFO] Set Piper initial state: {self.initial_state.tolist()}")

    def get_policy_observation(self, prompt: str, image_size: int):
        try:
            from openpi_client import image_tools
        except ImportError as exc:
            raise RuntimeError(
                "OpenPI policy mode requires openpi-client. Install it in the uarm "
                "environment, for example: pip install -e /workspace/openpi/packages/openpi-client"
            ) from exc

        with self.timing.time("policy.get_state"):
            state = self._get_piper_record_state()
        if state is None:
            raise RuntimeError("Failed to read Piper state from simulation")

        with self.timing.time("policy.get_sensor_images"):
            sensor_images = self.env.unwrapped.get_sensor_images()
        with self.timing.time("policy.extract_base_image"):
            base_image = self._extract_policy_image(sensor_images, "d435_top_camera")
        with self.timing.time("policy.extract_wrist_image"):
            wrist_image = self._extract_policy_image(sensor_images, "wrist_camera")
        with self.timing.time("policy.resize_images"):
            base_image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(base_image, image_size, image_size)
            )
            wrist_image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_image, image_size, image_size)
            )

        return {
            "observation/state": state.astype(np.float32),
            "observation/image": base_image,
            "observation/wrist_image": wrist_image,
            "prompt": prompt,
        }

    def _extract_policy_image(self, sensor_images, sensor_name: str) -> np.ndarray:
        images = sensor_images.get(sensor_name)
        if not images:
            raise RuntimeError(f"Missing sensor images for '{sensor_name}'")
        frame = images.get("rgb")
        if frame is None:
            frame = next(iter(images.values()))
        frame = self._sensor_frame_to_rgb(frame)
        if frame is None:
            raise RuntimeError(f"Failed to convert '{sensor_name}' image to RGB")
        return frame

    def policy_action_to_env_action(self, policy_action, action_mode: str) -> np.ndarray:
        policy_action = np.asarray(policy_action, dtype=np.float32).reshape(-1)
        if policy_action.shape != (7,):
            raise ValueError(
                f"Expected a 7D Piper policy action, got shape {policy_action.shape}"
            )
        if action_mode == "delta":
            state = self._get_piper_record_state()
            if state is None:
                raise RuntimeError("Failed to read Piper state for delta action conversion")
            target_state = state.astype(np.float32) + policy_action
        elif action_mode == "absolute":
            target_state = policy_action
        else:
            raise ValueError(f"Unsupported action mode: {action_mode}")
        return self.piper_state_to_env_action(target_state)

    def piper_state_to_env_action(self, state_7) -> np.ndarray:
        state_7 = np.asarray(state_7, dtype=np.float32).reshape(-1)
        if state_7.shape != (7,):
            raise ValueError(f"Expected a 7D Piper state/action, got shape {state_7.shape}")
        env_action = np.zeros(8, dtype=np.float32)
        env_action[:6] = state_7[:6]
        env_action[6:] = state_7[6]
        return env_action

    def validate_action_chunk(self, response) -> np.ndarray:
        if not isinstance(response, dict) or "actions" not in response:
            raise RuntimeError("Policy server response must be a dict containing 'actions'")
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise RuntimeError(
                f"Expected policy server actions with shape (N, 7), got {actions.shape}"
            )
        if actions.shape[0] == 0:
            raise RuntimeError("Policy server returned an empty action chunk")
        return actions

    def idle_step(self):
        """Advance physics and refresh render/camera windows before policy inference starts."""
        with self.timing.time("idle.env_step"):
            self.env.step(self.zero_env_action())
        if self.env_render:
            with self.timing.time("idle.env_render"):
                self.env.render()
        with self.timing.time("idle.display_cameras"):
            self._display_default_sensor_cameras()
            self._display_wrist_camera()

    def rollout_step(self, env_action):
        self._process_recording_events()
        with self.timing.time("sim.env_step"):
            self.env.step(env_action)
        if self.env_render:
            with self.timing.time("sim.env_render"):
                self.env.render()
        with self.timing.time("sim.display_cameras"):
            self._display_default_sensor_cameras()
            self._display_wrist_camera()
        with self.timing.time("sim.record_frame"):
            self._record_current_frame()
        self._process_recording_events()

    def close_resources(self):
        if self.record_enabled and self.dataset is not None:
            finalize = getattr(self.dataset, "finalize", None)
            if callable(finalize):
                finalize()
        if getattr(self, "env", None) is not None:
            self.env.close()
        if self.cv2 is not None:
            self.cv2.destroyAllWindows()

    def _record_current_frame(self):
        if not self.record_enabled:
            return
        with self.record_lock:
            if self.record_state != "recording":
                return
        now = time.monotonic()
        if self.last_record_time and now - self.last_record_time < self.record_period:
            return
        state = self._get_piper_record_state()
        camera_frames = self._collect_record_camera_frames()
        if state is None or camera_frames is None:
            return
        frame = {
            "observation.state": state.astype(np.float32),
            "action": self.zero_policy_action(),
        }
        frame.update(camera_frames)
        timestamp = self.current_episode_frames / float(self.record_fps)
        self.dataset.add_frame(frame, task=self.task, timestamp=timestamp)
        self.last_recorded_state = state.copy()
        self.last_record_time = now
        self.current_episode_frames += 1
        if self.current_episode_frames == 1 or self.current_episode_frames % self.record_fps == 0:
            episode_index = self._current_dataset_episode_index()
            print(f"[RECORD] Episode {episode_index}: {self.current_episode_frames} frames")

    def run(
        self,
        max_steps=None,
        save_on_exit: bool = True,
        policy_mode: str = "zero",
        policy_client=None,
        prompt: str = "put red box to blue plate",
        open_loop_horizon: int = 10,
        image_size: int = 224,
        action_mode: str = "delta",
    ):
        print(f"Starting {policy_mode} rollout...")
        print("Policy action shape: 7")
        print(f"Control frequency: {self.rate} Hz")
        if policy_mode == "openpi":
            if policy_client is None:
                raise ValueError("policy_client is required when --policy-mode openpi")
            if open_loop_horizon <= 0:
                raise ValueError("--open-loop-horizon must be positive")
            if image_size <= 0:
                raise ValueError("--image-size must be positive")

        command_thread = None
        if self.record_enabled:
            command_thread = Thread(target=self.command_loop, daemon=True)
            command_thread.start()

        step_count = 0
        actions_from_chunk_completed = 0
        pred_action_chunk = None
        period = max(1.0 / self.rate, 1e-6)
        next_time = time.monotonic()
        try:
            while not self.stop_event.is_set():
                if policy_mode == "zero":
                    env_action = self.zero_env_action()
                elif policy_mode == "openpi":
                    if (
                        pred_action_chunk is None
                        or actions_from_chunk_completed >= min(
                            open_loop_horizon, pred_action_chunk.shape[0]
                        )
                    ):
                        actions_from_chunk_completed = 0
                        with self.timing.time("policy.get_observation_total"):
                            obs = self.get_policy_observation(prompt, image_size)
                        with self.timing.time("policy.infer"):
                            with prevent_keyboard_interrupt():
                                response = policy_client.infer(obs)
                        with self.timing.time("policy.validate_chunk"):
                            pred_action_chunk = self.validate_action_chunk(response)
                    policy_action = pred_action_chunk[actions_from_chunk_completed]
                    actions_from_chunk_completed += 1
                    with self.timing.time("action.convert"):
                        env_action = self.policy_action_to_env_action(policy_action, action_mode)
                else:
                    raise ValueError(f"Unsupported policy mode: {policy_mode}")

                loop_start = time.perf_counter()
                self.rollout_step(env_action)
                step_count += 1
                if max_steps is not None and step_count >= max_steps:
                    break
                next_time += period
                sleep_dt = next_time - time.monotonic()
                if sleep_dt > 0:
                    sleep_start = time.perf_counter()
                    time.sleep(sleep_dt)
                    self.timing.record("loop.sleep", time.perf_counter() - sleep_start)
                else:
                    next_time = time.monotonic()
                self.timing.record("loop.total", time.perf_counter() - loop_start)
                self.timing.end_step(policy_mode)
        except KeyboardInterrupt:
            print("Received interrupt signal, preparing to stop...")
        finally:
            self.stop_event.set()
            if command_thread is not None:
                command_thread.join(timeout=0.2)
            if (
                save_on_exit
                and self.record_enabled
                and self.dataset is not None
                and self.current_episode_frames > 0
            ):
                self._save_current_episode()
            self.close_resources()
            print(f"{policy_mode} rollout stopped after {step_count} steps")


@contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        del signum, frame
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Piper simulation rollout with zero or OpenPI policy actions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot", "-r", default="piper", choices=["piper"])
    parser.add_argument("--scene", "-s", default="ReplicaCAD_SceneManipulation-v1")
    parser.add_argument("--rate", type=float, default=30.0, help="Control frequency (Hz)")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum rollout steps; omit to run until interrupted",
    )
    parser.add_argument(
        "--policy-mode",
        choices=["zero", "openpi"],
        default="openpi",
        help="Action source for the rollout",
    )
    parser.add_argument("--host", type=str, default="localhost", help="OpenPI policy server host")
    parser.add_argument("--port", type=int, default=8000, help="OpenPI policy server port")
    parser.add_argument("--api-key", type=str, default=None, help="OpenPI policy server API key")
    parser.add_argument(
        "--prompt",
        type=str,
        default="put red box to blue plate",
        help="Language instruction sent to OpenPI policy server",
    )
    parser.add_argument(
        "--open-loop-horizon",
        type=int,
        default=10,
        help="Number of actions to execute from each policy chunk before querying again",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image size sent to OpenPI policy server",
    )
    parser.add_argument(
        "--action-mode",
        choices=["delta", "absolute"],
        default="delta",
        help="Interpret OpenPI 7D actions as delta from current state or absolute target state",
    )
    parser.add_argument(
        "--wait-for-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for terminal confirmation after simulation starts before OpenPI inference",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="sensors",
        choices=["human", "rgb_array", "sensors", "all"],
    )
    parser.add_argument(
        "--env-render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call env.render() each step; disable to show only OpenCV camera windows",
    )
    parser.add_argument(
        "--shader-pack",
        type=str,
        default="fast_rt",
        choices=sorted(PREBUILT_SHADER_CONFIGS.keys()) + sorted(SHADER_PACK_ALIASES.keys()),
    )
    parser.add_argument("--rt-samples-per-pixel", type=int, default=2)
    parser.add_argument("--rt-path-depth", type=int, default=1)
    parser.add_argument("--rt-denoiser", type=str, default="oidn", choices=["none", "oidn", "optix"])
    parser.add_argument("--no-render-preflight", action="store_true")
    parser.add_argument("--wrist-camera-width", type=int, default=320)
    parser.add_argument("--wrist-camera-height", type=int, default=320)
    parser.add_argument("--show-wrist-camera", action="store_true")
    parser.add_argument("--no-display-cameras", action="store_true")
    parser.add_argument("--wrist-camera-display-rate", type=float, default=10.0)
    parser.add_argument("--wrist-camera-display-scale", type=float, default=2.0)
    parser.add_argument(
        "--object-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[0.457, -1.612, 0.956],
    )
    parser.add_argument("--object-size", type=float, default=0.04)
    parser.add_argument("--no-object", action="store_true")
    parser.add_argument(
        "--initial-state",
        type=float,
        nargs=7,
        default=DEFAULT_INITIAL_STATE.tolist(),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "GRIPPER"),
        help="Initial 7D Piper state: 6 arm joints plus one gripper value",
    )
    parser.add_argument("--debug-timing", action="store_true", help="Print rollout timing diagnostics")
    parser.add_argument(
        "--debug-interval",
        type=int,
        default=30,
        help="Print timing summary every N measured steps",
    )
    parser.add_argument(
        "--debug-warmup",
        type=int,
        default=5,
        help="Skip the first N steps when collecting timing diagnostics",
    )
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--record-dir", type=str, default="./lerobot_data/eazy_sim_data")
    parser.add_argument("--repo-id", type=str, default="local/teleop_sim")
    parser.add_argument("--task", type=str, default="put red box to blue plate")
    parser.add_argument("--record-cameras", type=str, default="d435_top_camera,wrist_camera")
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument("--save-on-exit", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def wait_for_human_start(sim: ZeroActionRolloutSim, enabled: bool, policy_mode: str):
    if not enabled or policy_mode != "openpi":
        return True
    if not sys.stdin.isatty():
        print("[INFO] Non-interactive stdin detected; skipping human start confirmation.")
        return True

    print("[READY] Simulation environment is running.")
    print("[READY] Press Enter to connect OpenPI and start inference. Press Ctrl+C to quit.")
    period = max(1.0 / sim.rate, 1e-6)
    next_time = time.monotonic()
    try:
        while not sim.stop_event.is_set():
            loop_start = time.perf_counter()
            sim.idle_step()
            if sim.pending_quit:
                sim.stop_event.set()
                return False
            if sys.stdin in select.select([sys.stdin], [], [], 0.0)[0]:
                sys.stdin.readline()
                return True
            next_time += period
            sleep_dt = next_time - time.monotonic()
            if sleep_dt > 0:
                sleep_start = time.perf_counter()
                time.sleep(sleep_dt)
                sim.timing.record("idle.sleep", time.perf_counter() - sleep_start)
            else:
                next_time = time.monotonic()
            sim.timing.record("loop.total", time.perf_counter() - loop_start)
            sim.timing.end_step("idle")
    except KeyboardInterrupt:
        print("Received interrupt signal before inference start.")
        return False
    return False


def main():
    args = parse_args()
    record_cameras = tuple(
        camera.strip() for camera in args.record_cameras.split(",") if camera.strip()
    )
    if not args.env_render and args.render_mode == "human":
        print(
            "[WARN] --no-env-render skips per-step env.render(), but --render-mode human "
            "may still create a SAPIEN viewer. Use --render-mode sensors to hide it."
        )

    print("=" * 60)
    print("    Piper Simulation Rollout")
    print("=" * 60)
    print(f"Robot arm type: {args.robot}")
    print(f"Simulation scene: {args.scene}")
    print(f"Control frequency: {args.rate} Hz")
    print(f"Policy mode: {args.policy_mode}")
    if args.policy_mode == "openpi":
        print(f"OpenPI server: {args.host}:{args.port}")
        print(f"Open-loop horizon: {args.open_loop_horizon}")
        print(f"Policy image size: {args.image_size}")
        print(f"Action mode: {args.action_mode}")
    print(f"Prompt: {args.prompt}")
    print(f"Render mode: {args.render_mode}")
    print(f"Env render: {'enabled' if args.env_render else 'disabled'}")
    print(f"Shader pack: {args.shader_pack} ({normalize_shader_pack(args.shader_pack)})")
    print(f"RT samples: {args.rt_samples_per_pixel}")
    print(f"RT path depth: {args.rt_path_depth}")
    print(f"RT denoiser: {args.rt_denoiser}")
    print(f"Render preflight: {'enabled' if not args.no_render_preflight else 'disabled'}")
    print(f"Wrist camera: {args.wrist_camera_width}x{args.wrist_camera_height}")
    print(f"Wrist display: {'enabled' if args.show_wrist_camera else 'disabled'}")
    print(f"Default camera display: {'disabled' if args.no_display_cameras else 'enabled'}")
    if args.no_object:
        print("Grasp object: disabled")
    else:
        print(f"Grasp object pos: {args.object_pos}")
        print(f"Grasp object size: {args.object_size} m")
    print(f"Initial state: {np.asarray(args.initial_state, dtype=np.float32).tolist()}")
    print(f"Debug timing: {'enabled' if args.debug_timing else 'disabled'}")
    print(f"Recording: {'enabled' if args.record else 'disabled'}")
    if args.record:
        print(f"Record dir: {os.path.expanduser(args.record_dir)}")
        print(f"Repo id: {args.repo_id}")
        print("Task: put red box to blue plate")
        print(f"Record cameras: {record_cameras}")
        print(f"Record FPS: {args.record_fps}")
        print(f"Save on exit: {'enabled' if args.save_on_exit else 'disabled'}")
    print("-" * 60)

    sim = ZeroActionRolloutSim(
        scene=args.scene,
        robot_uids=args.robot,
        object_pos=args.object_pos,
        object_size=args.object_size,
        spawn_object=not args.no_object,
        render_mode=args.render_mode,
        wrist_camera_width=args.wrist_camera_width,
        wrist_camera_height=args.wrist_camera_height,
        show_wrist_camera=args.show_wrist_camera,
        wrist_camera_display_rate=args.wrist_camera_display_rate,
        wrist_camera_display_scale=args.wrist_camera_display_scale,
        shader_pack=args.shader_pack,
        rt_samples_per_pixel=args.rt_samples_per_pixel,
        rt_path_depth=args.rt_path_depth,
        rt_denoiser=args.rt_denoiser,
        render_preflight=not args.no_render_preflight,
        record=args.record,
        record_dir=args.record_dir,
        repo_id=args.repo_id,
        task=args.task,
        record_cameras=record_cameras,
        record_fps=args.record_fps,
        rate=args.rate,
        display_cameras=not args.no_display_cameras,
        initial_state=args.initial_state,
        debug_timing=args.debug_timing,
        debug_interval=args.debug_interval,
        debug_warmup=args.debug_warmup,
        env_render=args.env_render,
    )

    if not wait_for_human_start(sim, args.wait_for_start, args.policy_mode):
        sim.close_resources()
        return

    policy_client = None
    if args.policy_mode == "openpi":
        try:
            from openpi_client import websocket_client_policy
        except ImportError as exc:
            raise RuntimeError(
                "OpenPI policy mode requires openpi-client. Install it in the uarm "
                "environment, for example: pip install -e /workspace/openpi/packages/openpi-client"
            ) from exc
        policy_client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
        )
        print(f"OpenPI server metadata: {policy_client.get_server_metadata()}")

    sim.run(
        max_steps=args.max_steps,
        save_on_exit=args.save_on_exit,
        policy_mode=args.policy_mode,
        policy_client=policy_client,
        prompt=args.prompt,
        open_loop_horizon=args.open_loop_horizon,
        image_size=args.image_size,
        action_mode=args.action_mode,
    )


if __name__ == "__main__":
    main()
