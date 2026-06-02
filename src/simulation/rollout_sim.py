import argparse
import os
import time
from threading import Event, Lock, Thread

import gymnasium as gym
import mani_skill.envs  # Must import to register all env/agent
import numpy as np
from transforms3d.euler import euler2quat

from teleop_sim import (
    ServoTeleoperatorSim,
    build_camera_shader_config,
    normalize_shader_pack,
    PREBUILT_SHADER_CONFIGS,
    SHADER_PACK_ALIASES,
)


class ZeroActionRolloutSim(ServoTeleoperatorSim):
    """Piper simulation rollout with a fixed zero action source."""

    def __init__(
        self,
        scene: str,
        robot_uids: str,
        object_pos=None,
        object_size: float = 0.04,
        spawn_object: bool = True,
        render_mode: str = "human",
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
        return np.zeros(8, dtype=np.float32)

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

    def run(self, max_steps=None, save_on_exit: bool = True):
        print("Starting zero-action rollout...")
        print("Policy action shape: 7")
        print(f"Control frequency: {self.rate} Hz")
        command_thread = None
        if self.record_enabled:
            command_thread = Thread(target=self.command_loop, daemon=True)
            command_thread.start()

        step_count = 0
        period = max(1.0 / self.rate, 1e-6)
        next_time = time.monotonic()
        try:
            while not self.stop_event.is_set():
                self.teleop_sim_handler(self.zero_env_action(), dwell=0.0)
                step_count += 1
                if max_steps is not None and step_count >= max_steps:
                    break
                next_time += period
                sleep_dt = next_time - time.monotonic()
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
                else:
                    next_time = time.monotonic()
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
            if self.record_enabled and self.dataset is not None:
                finalize = getattr(self.dataset, "finalize", None)
                if callable(finalize):
                    finalize()
            self.env.close()
            if self.cv2 is not None:
                self.cv2.destroyAllWindows()
            print(f"Zero-action rollout stopped after {step_count} steps")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Piper zero-action simulation rollout",
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
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array", "sensors", "all"],
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
    parser.add_argument("--record", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--record-dir", type=str, default="./lerobot_data/eazy_sim_data")
    parser.add_argument("--repo-id", type=str, default="local/teleop_sim")
    parser.add_argument("--task", type=str, default="put red box to blue plate")
    parser.add_argument("--record-cameras", type=str, default="d435_top_camera,wrist_camera")
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument("--save-on-exit", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    record_cameras = tuple(
        camera.strip() for camera in args.record_cameras.split(",") if camera.strip()
    )

    print("=" * 60)
    print("    Piper Zero-Action Simulation Rollout")
    print("=" * 60)
    print(f"Robot arm type: {args.robot}")
    print(f"Simulation scene: {args.scene}")
    print(f"Control frequency: {args.rate} Hz")
    print(f"Render mode: {args.render_mode}")
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
    )
    sim.run(max_steps=args.max_steps, save_on_exit=args.save_on_exit)


if __name__ == "__main__":
    main()
