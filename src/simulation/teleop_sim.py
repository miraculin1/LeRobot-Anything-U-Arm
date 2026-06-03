import copy
import json
import os
import shutil
import serial
import sys
import time 
import numpy as np
import re
import gymnasium as gym
import mani_skill.envs  # Must import to register all env/agent
from threading import Event, Thread, Lock
from queue import Queue, Empty, Full
import torch
import sapien
import argparse
from transforms3d.euler import euler2quat
from mani_skill.render import PREBUILT_SHADER_CONFIGS
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors


SHADER_PACK_ALIASES = {
    "fast_rt": "rt-fast",
}


class TimingStats:
    def __init__(self, enabled: bool, interval: int, warmup: int, target_period: float):
        self.enabled = enabled
        self.interval = max(1, int(interval))
        self.warmup = max(0, int(warmup))
        self.target_period = target_period
        self.step = 0
        self.data = {}

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


class NullTimer:
    def __init__(self, stats: TimingStats, key: str):
        self.stats = stats
        self.key = key
        self.start = None

    def __enter__(self):
        if self.stats.enabled:
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.start is not None:
            self.stats.record(self.key, time.perf_counter() - self.start)
        return False


def normalize_shader_pack(shader_pack: str) -> str:
    """Map user-facing shader aliases to ManiSkill shader config keys."""
    return SHADER_PACK_ALIASES.get(shader_pack, shader_pack)


def build_camera_shader_config(
    shader_pack: str,
    rt_samples_per_pixel: int,
    rt_path_depth: int,
    rt_denoiser: str,
):
    normalized_shader_pack = normalize_shader_pack(shader_pack)
    if normalized_shader_pack not in PREBUILT_SHADER_CONFIGS:
        valid_shader_packs = sorted(PREBUILT_SHADER_CONFIGS.keys()) + sorted(
            SHADER_PACK_ALIASES.keys()
        )
        raise ValueError(
            f"Unknown shader pack '{shader_pack}'. Valid options: {valid_shader_packs}"
        )

    shader_config = copy.deepcopy(PREBUILT_SHADER_CONFIGS[normalized_shader_pack])
    if shader_config.shader_pack[:2] == "rt":
        shader_config.shader_pack_config.update(
            ray_tracing_samples_per_pixel=rt_samples_per_pixel,
            ray_tracing_path_depth=rt_path_depth,
            ray_tracing_denoiser=rt_denoiser,
        )
    return shader_config


class ServoTeleoperatorSim: 
    """Robot arm teleoperation simulation system
    
    Supports reading servo angles through serial port and mapping to different types of robot arm simulation environments.
    Supported robot arm types: arx-x5, so100, xarm6_robotiq, panda, x_fetch, unitree_h1
    """
    
    def __init__(self, scene: str, robot_uids: str, serial_port: str = '/dev/ttyUSB0',
                 object_pos=None, object_size: float = 0.04, spawn_object: bool = True,
                 render_mode: str = "human", wrist_camera_width: int = 320,
                 wrist_camera_height: int = 320, show_wrist_camera: bool = False,
                 wrist_camera_display_rate: float = 10.0,
                 wrist_camera_display_scale: float = 2.0,
                 shader_pack: str = "fast_rt",
                 rt_samples_per_pixel: int = 2,
                 rt_path_depth: int = 1,
                 rt_denoiser: str = "oidn",
                 render_preflight: bool = True,
                 record: bool = False,
                 record_dir: str = "~/lerobot_sim_data",
                 repo_id: str = "local/teleop_sim",
                 task: str = "put red box to blue plate",
                 record_cameras=None,
                 record_fps: int = 30,
                 image_writer_processes: int = 0,
                 image_writer_threads: int = 4,
                 raw_writer_queue_size: int = 256,
                 env_render: bool = True,
                 control_dwell: float = 0.0,
                 debug_timing: bool = False,
                 debug_interval: int = 30,
                 debug_warmup: int = 5):
        """Initialize teleoperation system
        
        Args:
            scene: Simulation scene name
            robot_uids: Robot arm type identifier
            serial_port: Serial port device path
            object_pos: Position of the grasp object center [x, y, z]
            object_size: Edge length of the grasp object
            spawn_object: Whether to spawn a grasp object
            render_mode: ManiSkill render mode
            wrist_camera_width: Width of the piper wrist camera image
            wrist_camera_height: Height of the piper wrist camera image
            show_wrist_camera: Whether to show the piper wrist camera in an OpenCV window
            wrist_camera_display_rate: Wrist camera display refresh rate in Hz
            wrist_camera_display_scale: Display scaling factor for the OpenCV window
            shader_pack: ManiSkill shader pack, or fast_rt alias for rt-fast
            rt_samples_per_pixel: Ray tracing samples per pixel
            rt_path_depth: Ray tracing path depth
            rt_denoiser: Ray tracing denoiser backend
            render_preflight: Whether to render one frame during initialization
        """
        # Serial port configuration
        self.SERIAL_PORT = serial_port
        self.BAUDRATE = 115200
        self.ser = serial.Serial(self.SERIAL_PORT, self.BAUDRATE, timeout=0.01)

        # System configuration
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
        self.control_dwell = max(0.0, float(control_dwell))
        self.default_render_sensor_names = ("d435_top_camera", "wrist_camera")
        self.show_default_sensor_cameras = True
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
        self.zero_angles = [0.0] * 7  # Initial calibration angles for servos
        self.sim_init_angles = [0.0] * 7  # Simulation initial angles
        self.stop_event = Event()
        self.rate = 50.0  # Control frequency
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
        self.image_writer_processes = max(0, int(image_writer_processes))
        self.image_writer_threads = max(1, int(image_writer_threads))
        self.raw_writer_queue_size = max(1, int(raw_writer_queue_size))
        self.raw_writer_queue = None
        self.raw_writer_threads = []
        self.raw_writer_errors = []
        self.raw_writer_backpressure_warned = False
        self.raw_dataset_root = None
        self.current_episode_index = 0
        self.current_episode_dir = None
        self.current_episode_frames_data = []
        self.current_episode_camera_shapes = {}
        self.current_episode_started_at = None
        self.latest_teleop_target = None
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
        
        # Initialize servos and calibrate zero position
        self._init_servos()

        # Thread-safe data exchange queue
        self.arm_pos_queue: "Queue[list]" = Queue(maxsize=1)

        # Select control mode based on robot type
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

        # Create simulation environment
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
                        static_friction=10.0,  # Static friction
                        dynamic_friction=10.0, # Dynamic friction
                        restitution=0.0       # Restitution coefficient
                    )
                ),
            )
        except Exception:
            self._print_render_diagnostics()
            raise
        obs, _ = self.env.reset(seed=0)
        print("Action space:", self.env.action_space)
        if self.spawn_object:
            self._spawn_grasp_object()
            self._spawn_plates()
            self._randomize_task_objects()
        
        # Set initial standing pose for H1
        if robot_uids == "unitree_h1":
            self._setup_h1_standing_pose()

        # Create producer thread (read servo angles)
        self.produce_thread = Thread(
            target=self.angle_stream_loop, 
            args=(self.default_sender,), 
            daemon=True
        )

        # Create consumer thread (control simulation)
        self.consume_thread = Thread(
            target=self.pose_consumer_loop, 
            args=(self.teleop_sim_handler,), 
            daemon=True
        )

        self._setup_camera_pose()
        if self.render_preflight:
            self._run_render_preflight()
        if self.record_enabled:
            self._setup_raw_recorder()
            self._begin_recording_episode()


    def _print_render_diagnostics(self):
        print("[ERROR] Render initialization failed. Diagnostics:")
        print(f"DISPLAY={os.environ.get('DISPLAY')}")
        print(f"XDG_RUNTIME_DIR={os.environ.get('XDG_RUNTIME_DIR')}")
        for device_path in ("/dev/dri", "/dev/nvidia0", "/dev/nvidiactl"):
            print(f"{device_path}: {'present' if os.path.exists(device_path) else 'missing'}")
        try:
            print("SAPIEN render device summary:")
            print(sapien.render.get_device_summary())
        except Exception as exc:
            print(f"SAPIEN render device summary unavailable: {exc}")

    def _run_render_preflight(self):
        try:
            frame = self.env.render()
        except Exception:
            self._print_render_diagnostics()
            raise
        frame_shape = getattr(frame, "shape", None)
        print(f"[INFO] Render preflight completed. Frame shape: {frame_shape}")

    def _setup_camera_pose(self):
        agent = getattr(self.env.unwrapped, "agent", None)
        pose = sapien.Pose()
        if agent is not None:
            pose = agent.robot.get_pose()  # Returns sapien.Pose
            print(f"Robot initial position: {pose}")
        camera_pose = sapien_utils.look_at(
            [0.0, -1.5, 1.7], pose.p
        )
        camera_viewer = getattr(self.env.unwrapped, "viewer", None)
        if camera_viewer is not None:
            print(camera_pose)
            camera_pose_arr = camera_pose.raw_pose.squeeze().cpu().numpy()
            camera_position = camera_pose_arr[:3]
            camera_quaternion = camera_pose_arr[3:]
            camera_viewer.set_camera_pose(sapien.Pose(camera_position, camera_quaternion))

    def _spawn_grasp_object(self):
        """Spawn a simple dynamic cube for teleoperation grasp tests."""
        half_size = self.object_size / 2.0
        self.grasp_object = actors.build_cube(
            self.env.unwrapped.scene,
            half_size=half_size,
            color=[1, 0, 0, 1],
            name="teleop_cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=self.object_pos),
        )
        print(
            f"Spawned grasp object 'teleop_cube' at {self.object_pos} "
            f"with size {self.object_size} m"
        )

    def _spawn_plates(self):
        """Create reusable colored plate actors. Episode resets only move them."""
        self.plates = []
        table_z = self.object_pos[2] - self.object_size / 2.0
        plate_z = table_z + self.plate_half_height
        for idx, (name, color) in enumerate(self.plate_specs):
            plate = actors.build_cylinder(
                self.env.unwrapped.scene,
                radius=self.plate_radius,
                half_length=self.plate_half_height,
                color=color,
                name=name,
                body_type="dynamic",
                initial_pose=sapien.Pose(
                    p=[self.object_pos[0] + 0.12 * idx, self.object_pos[1], plate_z],
                    q=self.plate_quat,
                ),
            )
            self.plates.append(plate)
        print(f"[INFO] Spawned {len(self.plates)} colored plates: blue, green, yellow")

    def _sample_non_overlapping_xy(self, count: int, existing_points=None):
        rng = np.random.default_rng()
        xs = self.random_workspace["x"]
        ys = self.random_workspace["y"]
        points = [np.asarray(point, dtype=np.float64) for point in (existing_points or [])]
        sampled_points = []
        for _ in range(count):
            for _attempt in range(200):
                point = np.array(
                    [
                        rng.uniform(xs[0], xs[1]),
                        rng.uniform(ys[0], ys[1]),
                    ],
                    dtype=np.float64,
                )
                if all(np.linalg.norm(point - prev) >= self.min_object_spacing for prev in points):
                    points.append(point)
                    sampled_points.append(point)
                    break
            else:
                points.append(point)
                sampled_points.append(point)
        return sampled_points

    def _randomize_task_objects(self):
        if not self.spawn_object or self.grasp_object is None:
            return
        cube_pose = sapien.Pose(
            p=[self.fixed_red_box_xy[0], self.fixed_red_box_xy[1], self.object_pos[2]]
        )
        self.grasp_object.set_pose(cube_pose)
        table_z = self.object_pos[2] - self.object_size / 2.0
        plate_z = table_z + self.plate_half_height
        fixed_points = [self.fixed_red_box_xy, self.fixed_blue_plate_xy]
        random_points = self._sample_non_overlapping_xy(
            max(0, len(self.plates) - 1),
            existing_points=fixed_points,
        )
        plate_points = [self.fixed_blue_plate_xy] + random_points
        for plate, point in zip(self.plates, plate_points):
            plate.set_pose(sapien.Pose(p=[point[0], point[1], plate_z], q=self.plate_quat))
            self._zero_actor_velocity(plate)
        print(
            "[INFO] Randomized task objects: "
            f"red box xy={self.fixed_red_box_xy.round(3).tolist()}, "
            f"blue plate xy={self.fixed_blue_plate_xy.round(3).tolist()}, "
            f"random plate xys={[point.round(3).tolist() for point in random_points]}"
        )

    def _zero_actor_velocity(self, actor):
        for method_name in ("set_linear_velocity", "set_angular_velocity"):
            method = getattr(actor, method_name, None)
            if callable(method):
                method(np.zeros(3, dtype=np.float32))

    def _setup_h1_standing_pose(self):
        """Set initial standing pose for H1 robot"""
        try:
            agent = getattr(self.env.unwrapped, "agent", None)
            if agent is not None:
                # Use H1 predefined standing pose
                standing_keyframe = agent.keyframes["standing"]
                
                # Check qpos dimensions
                if hasattr(standing_keyframe.qpos, '__len__') and len(standing_keyframe.qpos) >= 19:
                    agent.reset(standing_keyframe.qpos)
                    agent.robot.set_root_pose(standing_keyframe.pose)
                    print("H1 set to standing pose")
                else:
                    print("Warning: standing_keyframe.qpos dimensions incorrect, using default standing pose")
                    # Use default standing pose
                    default_standing = np.array([
                        0, 0, 0, 0, 0, 0, 0, -0.4, -0.4, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0, -0.4, -0.4, 0.0, 0.0
                    ])
                    agent.reset(default_standing)
                    agent.robot.set_root_pose(standing_keyframe.pose)
                    print("H1 set to default standing pose")
        except Exception as e:
            print(f"Failed to set H1 standing pose: {e}")
    
    def _init_servos(self):
        """Initialize servos and calibrate zero position angles"""
        self.send_command('#000PVER!')
        for i in range(7):
            self.send_command("#000PCSK!")
            self.send_command(f'#{i:03d}PULK!')
            response = self.send_command(f'#{i:03d}PRAD!')
            angle = self.pwm_to_angle(response.strip(), i)
            self.zero_angles[i] = angle if angle is not None else 0.0
        print("[INFO] Servo initial angle calibration completed")

    def send_command(self, cmd: str) -> str:
        """Send serial command and read response
        
        Args:
            cmd: Command string to send
            
        Returns:
            Response string, returns empty string if no response
        """
        self.ser.write(cmd.encode('ascii'))
        time.sleep(0.008)
        response = self.ser.read_all()
        return response.decode('ascii', errors='ignore') if response else ""
    
    def pwm_to_angle(self, response_str: str, servo_num: int,pwm_min: int = 500, 
                     pwm_max: int = 2500, angle_range: float = 270):
        """Convert PWM response to angle
        
        Args:
            response_str: Servo response string
            pwm_min: PWM minimum value
            pwm_max: PWM maximum value
            angle_range: Angle range (degrees)
            
        Returns:
            Angle value, returns None if parsing fails
        """
        pattern = f"#{servo_num:03d}P(\\d{{4}})"
        match = re.search(pattern, response_str)
        if not match:
            return None
        pwm_val = int(match.group(1))
        pwm_span = pwm_max - pwm_min
        angle = (pwm_val - pwm_min) / pwm_span * angle_range
        return angle
    
    def publish_arm_pos(self, arm_pos: list):
        """Publish latest arm position to queue, overwriting old values"""
        try:
            # Clear old data from queue
            while True:
                self.arm_pos_queue.get_nowait()
        except Empty:
            pass
        try:
            # Add new data
            self.arm_pos_queue.put_nowait(list(arm_pos))
        except Exception:
            pass
    
    def get_latest_arm_pos(self, timeout: float = 0.0):
        """Get latest arm position snapshot
        
        Args:
            timeout: Timeout time, 0 means return immediately
            
        Returns:
            Latest arm position list, returns None if queue is empty
        """
        try:
            return self.arm_pos_queue.get(timeout=timeout) if timeout and timeout > 0 else self.arm_pos_queue.get_nowait()
        except Empty:
            return None
    
    def angle_to_gripper(self, angle_rad: float, pos_min: float, pos_max: float, 
                        angle_range: float = 1.5 * np.pi) -> float:
        """Map servo angle to gripper position
        
        Args:
            angle_rad: Servo angle (radians)
            pos_min: Gripper minimum position
            pos_max: Gripper maximum position
            angle_range: Servo angle range
            
        Returns:
            Gripper position value
        """
        ratio = max(0, 1 - (angle_rad / angle_range))
        position = pos_min + (pos_max - pos_min) * ratio
        return float(np.clip(position, pos_min, pos_max))

    def convert_pose_to_action(self, pose: list) -> np.ndarray: 
        """Convert servo position to simulation action based on different robot arm types
        
        Args:
            pose: 7-dimensional servo angle list (radians)
            
        Returns:
            Corresponding robot arm action vector
        """
        action = np.array([])

        if self.robot_uids == "arx-x5":  # 6-axis robot arm + dual-finger gripper
            action = np.array(pose)
            # Handle gripper: map last dimension to gripper position
            action[-1] = self.angle_to_gripper(action[-1], 0, 0.044)
            action = np.concatenate([action, [action[-1]]])

            action[2] = -action[2]
            action[4], action[5] = -action[5], -action[4]  # Swap joints 4 and 5
        
        elif self.robot_uids == "piper":  # 6-axis robot arm + dual-finger gripper
            action = np.array(pose)
            press_ratio = np.clip(-action[-1] / np.radians(48), 0.0, 1.0)
            action[-1] = float(0.04 * (1.0 - press_ratio))

            action = np.concatenate([action, [action[-1]]])
            action[4] = -action[4]

        elif self.robot_uids == "so100":  # 5-axis robot arm
            pose_copy = pose.copy()
            pose_copy.pop(5)  # Remove 6th dimension (so100 only has 5 axes)
            action = np.array(pose_copy)
            action[-1] = self.angle_to_gripper(action[-1], -1.1, 1.1)
            
            action[0] = -action[0]
            action[3] = -action[3]
            action[4] = -action[4]

        elif self.robot_uids == "xarm6_robotiq":  # 6-axis robot arm + Robotiq gripper
            action = np.array(pose)
            action[3], action[4] = action[4], -action[3]  # Swap joints 3 and 4
            # action[1] = -action[1]
            action[-1] = 0.81 - self.angle_to_gripper(action[-1], 0, 0.81)

        elif self.robot_uids == "panda":  # 7-axis robot arm
            pose_copy = pose.copy()
            pose_copy.insert(2, 0.0)  # Insert 0 at 3rd position (Panda's 3rd joint)
            action = np.array(pose_copy)
            # action[1] = -action[1]
            action[3] = -action[3]
            action[4], action[5] = action[5], action[4]  # Swap joints 4 and 5
            action[-1] = self.angle_to_gripper(action[-1], -1.0, 1.0)

        elif self.robot_uids == "x_fetch":  # Dual-arm robot + mobile base
            pose_copy = pose.copy()
            pose_copy.pop(5)  # Remove 6th dimension
            action = np.array(pose_copy)
            action[-1] = self.angle_to_gripper(action[-1], -1.1, 1.1)
            # Adjust joint directions
            action[0] = -action[0]
            action[1] = -action[1]
            action[3] = -action[3]
            action[4] = -action[4]
            # Build dual-arm action
            left_arm_action = action.copy()
            right_arm_action = left_arm_action.copy()
            right_arm_action[0] = -right_arm_action[0]
            right_arm_action[-2] = -right_arm_action[-2]
            # Combine: left arm joints + right arm joints + left/right grippers + base motion (0)
            zero_action = np.zeros(6)
            action = np.concatenate([
                left_arm_action[0:-1], 
                right_arm_action[0:-1], 
                [left_arm_action[-1], right_arm_action[-1]], 
                np.zeros(4) 
            ])

        elif self.robot_uids == "widowx250s":  # 6-axis robot arm + dual-finger gripper
            action = np.array(pose)
            action[-1] = self.angle_to_gripper(action[-1], 0, 0.04)
            action = np.concatenate([action, [action[-1]]])
            action[3], action[4] = action[4], -action[3] 
        
        elif self.robot_uids == "unitree_h1":  # Humanoid robot
            raw = np.array(pose, dtype=np.float32)
            action = np.zeros(19, dtype=np.float32)

            # Only modify arm joint increments (relative to current state)
            # Left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
            action[5] = raw[0]  # left_shoulder_pitch increment
            action[9] = raw[1]   # left_shoulder_roll increment
            action[13] = raw[2]  # left_shoulder_yaw increment
            action[17] = raw[3]  # left_elbow increment

            # Right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
            action[6] = raw[4]   # right_shoulder_pitch increment
            action[10] = raw[5]  # right_shoulder_roll increment
            action[14] = raw[6]  # right_shoulder_yaw increment
            action[18] = raw[3]  # right_elbow increment (reuse 4th servo)

        else: 
            raise ValueError(f"Unsupported robot arm type: {self.robot_uids}")

        return action

    def default_sender(self, arm_pos: list): 
        """Default angle sending callback (for debugging)"""
        print(f"Servo angles (degrees): {np.degrees(arm_pos)}")

    def _setup_raw_recorder(self):
        if self.robot_uids != "piper":
            raise ValueError("--record is currently implemented for --robot piper only")
        if self.record_fps <= 0:
            raise ValueError("--record-fps must be positive")
        self.raw_dataset_root = os.path.join(self.record_dir, self.repo_id)
        if os.path.exists(os.path.join(self.raw_dataset_root, "meta", "info.json")):
            raise RuntimeError(
                f"Record root looks like an existing LeRobot dataset: {self.raw_dataset_root}. "
                "Use a raw-data directory for --record-dir, then convert with convert_raw_to_lerobot.py."
            )
        os.makedirs(os.path.join(self.raw_dataset_root, "episodes"), exist_ok=True)

        camera_shapes = self._expected_record_camera_shapes()
        for camera_name in self.record_cameras:
            if camera_name not in camera_shapes:
                raise ValueError(
                    f"Unsupported --record-cameras entry '{camera_name}'. "
                    "Supported cameras: d435_top_camera,wrist_camera"
                )
        print(f"[INFO] Raw recording root: {self.raw_dataset_root}")
        if self.image_writer_processes:
            print(
                "[WARN] --image-writer-processes is ignored for raw recording; "
                "--image-writer-threads controls raw PNG writer threads."
            )
        self._start_raw_writer()
        print(
            "[INFO] Raw async image writer enabled: "
            f"threads={self.image_writer_threads}, queue_size={self.raw_writer_queue_size}"
        )

    def _start_raw_writer(self):
        if self.raw_writer_queue is not None:
            return
        self.raw_writer_queue = Queue(maxsize=self.raw_writer_queue_size)
        self.raw_writer_threads = []
        self.raw_writer_errors = []
        for index in range(self.image_writer_threads):
            thread = Thread(
                target=self._raw_writer_loop,
                name=f"raw-image-writer-{index}",
                daemon=True,
            )
            thread.start()
            self.raw_writer_threads.append(thread)

    def _raw_writer_loop(self):
        while True:
            job = self.raw_writer_queue.get()
            try:
                if job is None:
                    return
                for _camera_name, frame, abs_path in job:
                    bgr_frame = self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR)
                    if not self.cv2.imwrite(abs_path, bgr_frame):
                        raise RuntimeError(f"Failed to write raw camera frame: {abs_path}")
            except Exception as exc:
                with self.record_lock:
                    self.raw_writer_errors.append(str(exc))
            finally:
                self.raw_writer_queue.task_done()

    def _enqueue_raw_image_write(self, job):
        if self.raw_writer_queue is None:
            raise RuntimeError("Raw writer queue is not initialized")
        try:
            self.raw_writer_queue.put(job, timeout=0.001)
        except Full:
            if not self.raw_writer_backpressure_warned:
                print(
                    "[WARN] Raw image writer queue is full; blocking control loop until "
                    "disk writer catches up."
                )
                self.raw_writer_backpressure_warned = True
            self.raw_writer_queue.put(job)

    def _wait_raw_writer(self, raise_errors=True):
        if self.raw_writer_queue is not None:
            self.raw_writer_queue.join()
        with self.record_lock:
            errors = list(self.raw_writer_errors)
            self.raw_writer_errors = []
        if errors and raise_errors:
            raise RuntimeError("Raw image writer failed: " + "; ".join(errors[:3]))
        if errors:
            print("[WARN] Raw image writer errors ignored during cleanup: " + "; ".join(errors[:3]))

    def _stop_raw_writer(self):
        if self.raw_writer_queue is None:
            return
        self._wait_raw_writer(raise_errors=False)
        for _thread in self.raw_writer_threads:
            self.raw_writer_queue.put(None)
        self.raw_writer_queue.join()
        for thread in self.raw_writer_threads:
            thread.join(timeout=2.0)
        self.raw_writer_queue = None
        self.raw_writer_threads = []

    def _begin_recording_episode(self):
        episode_index = self._next_raw_episode_index()
        episode_dir = os.path.join(
            self.raw_dataset_root, "episodes", f"episode_{episode_index:06d}"
        )
        if os.path.isdir(episode_dir):
            shutil.rmtree(episode_dir)
        for camera_name in self.record_cameras:
            os.makedirs(os.path.join(episode_dir, "images", camera_name), exist_ok=True)
        with self.record_lock:
            self.record_state = "recording"
            self.pending_stop_episode = False
            self.pending_episode_decision = None
            self.last_record_time = 0.0
            self.last_recorded_state = None
            self.current_episode_frames = 0
            self.current_episode_frames_data = []
            self.current_episode_camera_shapes = {}
            self.current_episode_index = episode_index
            self.current_episode_dir = episode_dir
            self.current_episode_started_at = time.time()
            self.countdown_end_time = None
            self.status_message = f"RECORDING episode {episode_index}"
        print(
            f"[RECORD] Recording episode {episode_index}. "
            "Use terminal 'n' or GUI Stop Episode to finish."
        )

    def _current_dataset_episode_index(self):
        return int(self.current_episode_index)

    def _next_raw_episode_index(self):
        episodes_dir = os.path.join(self.raw_dataset_root, "episodes")
        max_index = -1
        if os.path.isdir(episodes_dir):
            for name in os.listdir(episodes_dir):
                if not name.startswith("episode_"):
                    continue
                try:
                    max_index = max(max_index, int(name.split("_")[-1]))
                except ValueError:
                    continue
        return max_index + 1

    def _expected_record_camera_shapes(self):
        return {
            "d435_top_camera": (480, 640, 3),
            "wrist_camera": (self.wrist_camera_height, self.wrist_camera_width, 3),
        }

    def _request_stop_episode(self):
        with self.record_lock:
            if self.record_state == "recording":
                self.pending_stop_episode = True
            elif self.record_state == "confirm_save":
                print("[RECORD] Episode already stopped. Choose save or discard.")

    def _request_episode_decision(self, decision: str):
        with self.record_lock:
            if self.record_state != "confirm_save":
                print("[RECORD] No episode is waiting for save/discard confirmation.")
                return
            self.pending_episode_decision = decision

    def _request_quit(self):
        with self.record_lock:
            if self.record_enabled and self.record_state == "recording" and self.current_episode_frames > 0:
                self.pending_stop_episode = True
                self.exit_after_episode_decision = True
            elif self.record_enabled and self.record_state == "confirm_save":
                self.exit_after_episode_decision = True
                print("[RECORD] Confirm save/discard before quitting.")
            else:
                self.pending_quit = True

    def _enter_save_confirmation(self):
        with self.record_lock:
            self.record_state = "confirm_save"
            self.pending_stop_episode = False
            frames = self.current_episode_frames
            episode_index = self._current_dataset_episode_index()
            self.status_message = (
                f"CONFIRM episode {episode_index}: {frames} frames. "
                "Save [y] or discard [d]."
            )
        print(
            f"[RECORD] Episode {episode_index} stopped with {frames} frames. "
            "Save episode? [y]/save or discard with [d]/discard. [q] quits after decision."
        )

    def _save_current_episode(self):
        if self.current_episode_dir is None or self.current_episode_frames == 0:
            print("[RECORD] No frames recorded; skipping save.")
            self._clear_current_episode_buffer()
            return
        episode_index = self._current_dataset_episode_index()
        print(f"[RECORD] Saving episode {episode_index} ({self.current_episode_frames} frames)...")
        self._wait_raw_writer()
        meta = {
            "format": "uarm_sim_raw_v1",
            "episode_index": episode_index,
            "robot_type": self.robot_uids,
            "task": self.task,
            "fps": self.record_fps,
            "record_cameras": list(self.record_cameras),
            "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
            "state_key": "observation_state",
            "teleop_target_key": "teleop_target",
            "teleop_target_semantics": "mapped_robot_absolute_target",
            "num_frames": self.current_episode_frames,
            "camera_shapes": self.current_episode_camera_shapes,
            "started_at_unix": self.current_episode_started_at,
            "saved_at_unix": time.time(),
        }
        with open(os.path.join(self.current_episode_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        frames_path = os.path.join(self.current_episode_dir, "frames.jsonl")
        with open(frames_path, "w", encoding="utf-8") as f:
            for frame in self.current_episode_frames_data:
                f.write(json.dumps(frame, sort_keys=True) + "\n")
        print(f"[RECORD] Saved raw episode {episode_index}: {self.current_episode_dir}")
        self._reset_current_episode_buffer()

    def _clear_current_episode_buffer(self):
        self._wait_raw_writer(raise_errors=False)
        if self.current_episode_dir and os.path.isdir(self.current_episode_dir):
            shutil.rmtree(self.current_episode_dir)
        self._reset_current_episode_buffer()

    def _reset_current_episode_buffer(self):
        self.current_episode_dir = None
        self.current_episode_frames_data = []
        self.current_episode_camera_shapes = {}
        self.current_episode_started_at = None

    def _discard_current_episode(self):
        episode_index = self._current_dataset_episode_index()
        print(f"[RECORD] Discarding episode {episode_index} ({self.current_episode_frames} frames).")
        self._clear_current_episode_buffer()

    def _start_next_episode_countdown(self):
        self._randomize_task_objects()
        with self.record_lock:
            self.record_state = "countdown"
            self.countdown_end_time = time.monotonic() + 3.0
            self.last_recorded_state = None
            self.current_episode_frames = 0
            self.status_message = "STARTING IN 3"

    def _process_recording_events(self):
        if not self.record_enabled:
            return
        with self.record_lock:
            pending_stop = self.pending_stop_episode
            pending_decision = self.pending_episode_decision
            pending_quit = self.pending_quit
            state = self.record_state
        if pending_quit:
            self.stop_event.set()
            return
        if pending_stop and state == "recording":
            self._enter_save_confirmation()
            return
        if pending_decision and state == "confirm_save":
            with self.record_lock:
                self.pending_episode_decision = None
                self.status_message = "SAVING" if pending_decision == "save" else "DISCARDING"
            if pending_decision == "save":
                self._save_current_episode()
            else:
                self._discard_current_episode()
            if self.exit_after_episode_decision:
                with self.record_lock:
                    self.pending_quit = True
                return
            self._start_next_episode_countdown()
            return
        if state == "countdown":
            now = time.monotonic()
            remaining = max(0.0, (self.countdown_end_time or now) - now)
            with self.record_lock:
                self.status_message = f"STARTING IN {int(np.ceil(remaining))}"
            if remaining <= 0:
                self._begin_recording_episode()

    def _sensor_frame_to_bgr(self, frame):
        if frame is None:
            return None
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)
        if frame.ndim == 4:
            frame = frame[0]
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR)

    def _sensor_frame_to_rgb(self, frame):
        if frame is None:
            return None
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)
        if frame.ndim == 4:
            frame = frame[0]
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def _collect_camera_frames(self, sensor_names):
        sensor_images = self.env.unwrapped.get_sensor_images()
        frames = {}
        for sensor_name in sensor_names:
            images = sensor_images.get(sensor_name)
            if not images:
                continue
            frame = images.get("rgb")
            if frame is None:
                frame = next(iter(images.values()))
            frame = self._sensor_frame_to_rgb(frame)
            if frame is not None:
                frames[sensor_name] = frame
        return frames

    def _camera_frames_to_bgr_list(self, camera_frames, sensor_names):
        frames = []
        for sensor_name in sensor_names:
            frame = camera_frames.get(sensor_name)
            if frame is None:
                continue
            frames.append(self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR))
        return frames

    def _collect_record_camera_frames(self, camera_frames=None):
        if camera_frames is None:
            camera_frames = self._collect_camera_frames(self.record_cameras)
        frames = {}
        for sensor_name in self.record_cameras:
            frame = camera_frames.get(sensor_name)
            if frame is None:
                return None
            frames[f"observation.images.{sensor_name}"] = frame
        return frames

    def _get_piper_record_state(self):
        agent = getattr(self.env.unwrapped, "agent", None)
        if agent is None:
            return None
        qpos = agent.robot.get_qpos()
        if hasattr(qpos, "detach"):
            qpos = qpos.detach().cpu().numpy()
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if qpos.size < 6:
            return None
        state = np.zeros(7, dtype=np.float32)
        state[:6] = qpos[:6]
        if qpos.size >= 8:
            state[6] = float(np.mean(qpos[6:8]))
        elif qpos.size >= 7:
            state[6] = float(qpos[6])
        return state

    def _record_current_frame(self, camera_frames=None):
        if not self.record_enabled:
            return
        with self.record_lock:
            if self.record_state != "recording":
                return
        now = time.monotonic()
        if self.last_record_time and now - self.last_record_time < self.record_period:
            return
        with NullTimer(self.timing, "record.state"):
            state = self._get_piper_record_state()
        if camera_frames is None:
            with NullTimer(self.timing, "record.cameras"):
                record_camera_frames = self._collect_record_camera_frames()
        else:
            record_camera_frames = self._collect_record_camera_frames(camera_frames)
        if state is None or record_camera_frames is None:
            return
        with self.record_lock:
            teleop_target = None if self.latest_teleop_target is None else self.latest_teleop_target.copy()
            episode_dir = self.current_episode_dir
        if teleop_target is None or episode_dir is None:
            return

        frame_index = self.current_episode_frames
        timestamp = self.current_episode_frames / float(self.record_fps)
        image_paths = {}
        write_job = []
        with NullTimer(self.timing, "record.enqueue_frame"):
            for camera_key, frame in record_camera_frames.items():
                camera_name = camera_key.split(".")[-1]
                rel_path = os.path.join("images", camera_name, f"frame_{frame_index:06d}.png")
                abs_path = os.path.join(episode_dir, rel_path)
                image_paths[camera_name] = rel_path
                self.current_episode_camera_shapes[camera_name] = list(frame.shape)
                write_job.append((camera_name, np.ascontiguousarray(frame).copy(), abs_path))
            self._enqueue_raw_image_write(write_job)
            self.current_episode_frames_data.append(
                {
                    "frame_index": frame_index,
                    "timestamp": timestamp,
                    "observation_state": state.astype(np.float32).tolist(),
                    "teleop_target": teleop_target.astype(np.float32).tolist(),
                    "images": image_paths,
                }
            )
        self.last_recorded_state = state.copy()
        self.last_record_time = now
        self.current_episode_frames += 1
        if self.current_episode_frames == 1 or self.current_episode_frames % self.record_fps == 0:
            episode_index = self._current_dataset_episode_index()
            print(f"[RECORD] Episode {episode_index}: {self.current_episode_frames} frames")

    def _display_camera_frames(self, window_name: str, frames):
        if self.cv2 is None or not frames:
            return
        if len(frames) == 1:
            frame = frames[0]
        else:
            target_height = min(frame.shape[0] for frame in frames)
            resized_frames = []
            for frame in frames:
                scale = target_height / frame.shape[0]
                target_width = max(1, int(frame.shape[1] * scale))
                resized_frames.append(
                    self.cv2.resize(
                        frame,
                        (target_width, target_height),
                        interpolation=self.cv2.INTER_LINEAR,
                    )
                )
            frame = np.concatenate(resized_frames, axis=1)
        if self.wrist_camera_display_scale != 1.0:
            display_size = (
                int(frame.shape[1] * self.wrist_camera_display_scale),
                int(frame.shape[0] * self.wrist_camera_display_scale),
            )
            frame = self.cv2.resize(frame, display_size, interpolation=self.cv2.INTER_LINEAR)
        if window_name == "default_sensor_cameras":
            frame = self._draw_recording_overlay(frame, window_name)
        if not self._camera_window_initialized.get(window_name, False):
            self.cv2.namedWindow(window_name, self.cv2.WINDOW_NORMAL)
            self.cv2.resizeWindow(window_name, frame.shape[1], frame.shape[0])
            if window_name == "default_sensor_cameras":
                self.cv2.setMouseCallback(window_name, self._on_camera_window_mouse)
            self._camera_window_initialized[window_name] = True
        self.cv2.imshow(window_name, frame)
        key = self.cv2.waitKey(1) & 0xFF
        self._handle_gui_key(key)

    def _draw_recording_overlay(self, frame, window_name: str):
        if self.cv2 is None:
            return frame
        frame = frame.copy()
        with self.record_lock:
            state = self.record_state
            status = self.status_message
            frames = self.current_episode_frames
        self.cv2.rectangle(frame, (0, 0), (frame.shape[1], 72), (20, 20, 20), -1)
        color = (0, 220, 0) if state == "recording" else (0, 220, 255)
        self.cv2.putText(
            frame,
            status,
            (12, 28),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            self.cv2.LINE_AA,
        )
        self.cv2.putText(
            frame,
            f"frames: {frames} | terminal: n stop, y save, d discard, q quit",
            (12, 56),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            self.cv2.LINE_AA,
        )
        buttons = []
        if state == "recording":
            buttons.append(("Stop Episode", "stop", (frame.shape[1] - 190, 16, 176, 40)))
        elif state == "confirm_save":
            buttons.append(("Save", "save", (frame.shape[1] - 190, 16, 82, 40)))
            buttons.append(("Discard", "discard", (frame.shape[1] - 100, 16, 88, 40)))
        self._ui_buttons[window_name] = []
        for label, action, (x, y, w, h) in buttons:
            self.cv2.rectangle(frame, (x, y), (x + w, y + h), (245, 245, 245), -1)
            self.cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), 1)
            self.cv2.putText(
                frame,
                label,
                (x + 10, y + 26),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (25, 25, 25),
                1,
                self.cv2.LINE_AA,
            )
            self._ui_buttons[window_name].append((x, y, w, h, action))
        return frame

    def _on_camera_window_mouse(self, event, x, y, flags, param):
        if self.cv2 is None or event != self.cv2.EVENT_LBUTTONDOWN:
            return
        for bx, by, bw, bh, action in self._ui_buttons.get("default_sensor_cameras", []):
            if bx <= x <= bx + bw and by <= y <= by + bh:
                if action == "stop":
                    self._request_stop_episode()
                elif action == "save":
                    self._request_episode_decision("save")
                elif action == "discard":
                    self._request_episode_decision("discard")
                return

    def _handle_gui_key(self, key: int):
        if key in (255, -1):
            return
        if key in (ord("n"), ord("s")):
            self._request_stop_episode()
        elif key in (ord("y"), ord("\r")):
            self._request_episode_decision("save")
        elif key in (ord("d"), ord("x")):
            self._request_episode_decision("discard")
        elif key == ord("q"):
            self._request_quit()

    def _display_default_sensor_cameras(self, camera_frames=None):
        """Display the default top-down D435 camera and wrist camera when available."""
        if not self.show_default_sensor_cameras:
            return
        now = time.monotonic()
        if now - self._last_wrist_camera_display_time < self.wrist_camera_display_period:
            return
        self._last_wrist_camera_display_time = now
        if camera_frames is None:
            camera_frames = self._collect_camera_frames(self.default_render_sensor_names)
        frames = self._camera_frames_to_bgr_list(camera_frames, self.default_render_sensor_names)
        self._display_camera_frames("default_sensor_cameras", frames)

    def _display_wrist_camera(self, camera_frames=None):
        """Display the piper wrist camera in a separate OpenCV window."""
        if not self.show_wrist_camera:
            return
        if camera_frames is None:
            camera_frames = self._collect_camera_frames(("wrist_camera",))
        frames = self._camera_frames_to_bgr_list(camera_frames, ("wrist_camera",))
        self._display_camera_frames("wrist_camera", frames)

    def _should_record_this_step(self):
        if not self.record_enabled:
            return False
        with self.record_lock:
            if self.record_state != "recording":
                return False
        now = time.monotonic()
        return not (self.last_record_time and now - self.last_record_time < self.record_period)

    def _should_display_default_cameras_this_step(self):
        if not self.show_default_sensor_cameras:
            return False
        now = time.monotonic()
        return now - self._last_wrist_camera_display_time >= self.wrist_camera_display_period

    def teleop_sim_handler(self, action: np.ndarray):
        """Simulation control handler function
        
        Args:
            action: Robot arm action vector
        """
        if self.env is None or action is None:
            return
            
        # All robot types execute actions
        loop_start = time.perf_counter()
        self._process_recording_events()
        with NullTimer(self.timing, "sim.env_step"):
            self.env.step(action)
        if self.env_render:
            with NullTimer(self.timing, "sim.env_render"):
                self.env.render()
        needs_record_cameras = self._should_record_this_step()
        needs_default_display = self._should_display_default_cameras_this_step()
        needs_wrist_display = self.show_wrist_camera
        step_camera_frames = None
        if needs_record_cameras or needs_default_display or needs_wrist_display:
            sensor_names = set()
            if needs_record_cameras:
                sensor_names.update(self.record_cameras)
            if needs_default_display:
                sensor_names.update(self.default_render_sensor_names)
            if needs_wrist_display:
                sensor_names.add("wrist_camera")
            with NullTimer(self.timing, "step.cameras"):
                step_camera_frames = self._collect_camera_frames(tuple(sensor_names))
        with NullTimer(self.timing, "sim.display_cameras"):
            self._display_default_sensor_cameras(step_camera_frames)
            self._display_wrist_camera(step_camera_frames)
        with NullTimer(self.timing, "sim.record_frame"):
            self._record_current_frame(step_camera_frames)
        self._process_recording_events()
        if self.control_dwell > 0:
            sleep_start = time.perf_counter()
            time.sleep(self.control_dwell)
            self.timing.record("loop.sleep", time.perf_counter() - sleep_start)
        self.timing.record("loop.total", time.perf_counter() - loop_start)
        self.timing.end_step("teleop")
    
    def angle_stream_loop(self, on_send):
        """Angle data producer thread: periodically read servo angles
        
        Args:
            on_send: Callback function that receives angle data list
        """
        num_joints = 7
        arm_pos = [0.0] * num_joints

        period = max(1.0 / self.rate, 1e-6)
        next_time = time.monotonic()

        while not self.stop_event.is_set():
            # Read all joint angles
            for i in range(num_joints):
                response = self.send_command(f'#{i:03d}PRAD!')
                angle = self.pwm_to_angle(response.strip(), i)
                if angle is not None:
                    # Calculate angle relative to zero position
                    new_angle = angle - self.zero_angles[i]
                    arm_pos[i] = np.radians(new_angle)
                else: 
                    raise ValueError(f"Servo {i} response error: {response.strip()}")
            
            # Publish latest data and call callback
            self.publish_arm_pos(arm_pos)
            try:
                on_send(list(arm_pos))
            except Exception as e:
                print(f"Angle sending callback error: {e}")
                
            # Maintain fixed frequency
            next_time += period
            sleep_dt = next_time - time.monotonic()
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            else:
                # If falling behind significantly, resync time
                next_time = time.monotonic()

    def pose_consumer_loop(self, on_pose):
        """Simulation control consumer thread: periodically get angle data and control simulation
        
        Args:
            on_pose: Callback function that receives action vector
        """
        period = max(1.0 / self.rate, 1e-6)
        next_time = time.monotonic()

        # Safely get action space dimensions
        try:
            action_shape = self.env.action_space.shape[0] if self.env.action_space is not None else 0
        except (AttributeError, TypeError, IndexError):
            action_shape = 0
        print("Action Space Shape:", action_shape)
        
        while not self.stop_event.is_set():
            pose = self.get_latest_arm_pos(timeout=0.0)
            if pose is not None:
                try:
                    action = self.convert_pose_to_action(pose)
                    with self.record_lock:
                        self.latest_teleop_target = np.asarray(action, dtype=np.float32).copy()
                    on_pose(action)
                except Exception as e:
                    print(f"Simulation control callback error: {e}")
                    
            # Maintain fixed frequency
            next_time += period
            sleep_dt = next_time - time.monotonic()
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            else:
                next_time = time.monotonic()

    def command_loop(self):
        """Read terminal commands for episode control without blocking simulation."""
        if not self.record_enabled:
            return
        print("[RECORD] Terminal commands: n/next stop episode, y/save save, d/discard discard, q/quit quit.")
        while not self.stop_event.is_set():
            try:
                command = sys.stdin.readline()
            except Exception as exc:
                print(f"[RECORD] Terminal command input stopped: {exc}")
                return
            if command == "":
                time.sleep(0.1)
                continue
            command = command.strip().lower()
            if command in ("n", "next", "stop", "end"):
                self._request_stop_episode()
            elif command in ("y", "yes", "save"):
                self._request_episode_decision("save")
            elif command in ("d", "discard", "drop", "no"):
                self._request_episode_decision("discard")
            elif command in ("q", "quit", "exit"):
                self._request_quit()
            elif command:
                print("[RECORD] Unknown command. Use n, y, d, or q.")
    
    def run(self):
        """Start teleoperation system"""
        print("Starting angle reading thread...")
        self.produce_thread.start()
        print("Starting simulation control thread...")
        self.consume_thread.start()
        self.command_thread = None
        if self.record_enabled:
            self.command_thread = Thread(target=self.command_loop, daemon=True)
            self.command_thread.start()
        
        try: 
            print("System running, press Ctrl+C to stop...")
            while not self.stop_event.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Received interrupt signal, preparing to stop...")
            if self.record_enabled:
                self._request_quit()
        finally:
            self.stop_event.set()
            self.produce_thread.join(timeout=2.0)
            self.consume_thread.join(timeout=2.0)
            if self.command_thread is not None:
                self.command_thread.join(timeout=0.2)
            print("All threads stopped")
            if self.record_enabled:
                try:
                    self._stop_raw_writer()
                except Exception as exc:
                    print(f"[WARN] Raw writer shutdown failed: {exc}")
            self.env.close()
            self.ser.close()
            if self.cv2 is not None:
                self.cv2.destroyAllWindows()
            print("Resource cleanup completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Robot arm teleoperation simulation program',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--robot', '-r', 
        type=str, 
        default='so100',
        choices=['arx-x5', 'so100', 'xarm6_robotiq', 'panda', 'x_fetch', 'piper', 'widowx250s'],
        help='Select robot arm type to control'
    )
    parser.add_argument(
        '--scene', '-s', 
        type=str, 
        default='ReplicaCAD_SceneManipulation-v1',
        help='Simulation scene name'
    )
    parser.add_argument(
        '--rate', 
        type=float, 
        default=50.0,
        help='Control frequency (Hz)'
    )
    parser.add_argument(
        '--serial-port', 
        type=str, 
        default='/dev/ttyUSB0',
        help='Serial port device path'
    )
    parser.add_argument(
        '--render-mode',
        type=str,
        default='sensors',
        choices=['human', 'rgb_array', 'sensors', 'all'],
        help='ManiSkill render mode'
    )
    parser.add_argument(
        '--env-render',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Call env.render() each step; disable to show only OpenCV camera windows'
    )
    parser.add_argument(
        '--control-dwell',
        type=float,
        default=0.0,
        help='Extra sleep inside each simulation control step; 0 uses only the outer rate limiter'
    )
    parser.add_argument(
        '--shader-pack',
        type=str,
        default='fast_rt',
        choices=sorted(PREBUILT_SHADER_CONFIGS.keys()) + sorted(SHADER_PACK_ALIASES.keys()),
        help='Shader pack to use for all cameras; fast_rt is an alias for rt-fast'
    )
    parser.add_argument(
        '--rt-samples-per-pixel',
        type=int,
        default=2,
        help='Ray tracing samples per pixel for rt shader packs'
    )
    parser.add_argument(
        '--rt-path-depth',
        type=int,
        default=1,
        help='Ray tracing path depth for rt shader packs'
    )
    parser.add_argument(
        '--rt-denoiser',
        type=str,
        default='oidn',
        choices=['none', 'oidn', 'optix'],
        help='Ray tracing denoiser backend'
    )
    parser.add_argument(
        '--no-render-preflight',
        action='store_true',
        help='Skip the one-frame render preflight during initialization'
    )
    parser.add_argument(
        '--wrist-camera-width',
        type=int,
        default=320,
        help='Piper wrist camera image width'
    )
    parser.add_argument(
        '--wrist-camera-height',
        type=int,
        default=320,
        help='Piper wrist camera image height'
    )
    parser.add_argument(
        '--show-wrist-camera',
        action='store_true',
        help='Show the piper wrist camera in a separate OpenCV window'
    )
    parser.add_argument(
        '--wrist-camera-display-rate',
        type=float,
        default=10.0,
        help='Piper wrist camera display refresh rate in Hz'
    )
    parser.add_argument(
        '--wrist-camera-display-scale',
        type=float,
        default=2.0,
        help='Piper wrist camera OpenCV display scale'
    )
    parser.add_argument(
        '--object-pos',
        type=float,
        nargs=3,
        metavar=('X', 'Y', 'Z'),
        default=[0.457, -1.612, 0.956],
        help='Grasp object center position in world coordinates'
    )
    parser.add_argument(
        '--object-size',
        type=float,
        default=0.04,
        help='Grasp object edge length in meters'
    )
    parser.add_argument(
        '--no-object',
        action='store_true',
        help='Disable spawning the grasp object'
    )
    parser.add_argument(
        '--record',
        action='store_true',
        help='Record raw simulation episodes for later LeRobot conversion'
    )
    parser.add_argument(
        '--record-dir',
        type=str,
        default='~/lerobot_sim_data',
        help='Parent directory for local LeRobot datasets'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        default='local/teleop_sim',
        help='Raw dataset id; used as a subdirectory under --record-dir'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='put red box to blue plate',
        help='Task string saved with each frame; this script fixes it to put red box to blue plate'
    )
    parser.add_argument(
        '--record-cameras',
        type=str,
        default='d435_top_camera,wrist_camera',
        help='Comma-separated cameras to record'
    )
    parser.add_argument(
        '--record-fps',
        type=int,
        default=30,
        help='Dataset frame rate'
    )
    parser.add_argument(
        '--image-writer-processes',
        type=int,
        default=0,
        help='Deprecated for raw recording; ignored by --record'
    )
    parser.add_argument(
        '--image-writer-threads',
        type=int,
        default=4,
        help='Raw PNG writer thread count'
    )
    parser.add_argument(
        '--raw-writer-queue-size',
        type=int,
        default=256,
        help='Maximum queued raw image write jobs before recording applies backpressure'
    )
    parser.add_argument(
        '--debug-timing',
        action='store_true',
        help='Print teleoperation/recording timing diagnostics'
    )
    parser.add_argument(
        '--debug-interval',
        type=int,
        default=30,
        help='Print timing summary every N measured steps'
    )
    parser.add_argument(
        '--debug-warmup',
        type=int,
        default=5,
        help='Skip the first N steps when collecting timing diagnostics'
    )
    
    args = parser.parse_args()
    if not args.env_render and args.render_mode == "human":
        print(
            "[WARN] --no-env-render skips per-step env.render(), but --render-mode human "
            "may still create a SAPIEN viewer. Use --render-mode sensors to hide it."
        )
    
    # Display startup information
    print("=" * 60)
    print("    Robot Arm Teleoperation Simulation System")
    print("=" * 60)
    print(f"Robot arm type: {args.robot}")
    print(f"Simulation scene:   {args.scene}")
    print(f"Control frequency:   {args.rate} Hz")
    print(f"Serial device:   {args.serial_port}")
    print(f"Render mode:     {args.render_mode}")
    print(f"Env render:      {'enabled' if args.env_render else 'disabled'}")
    print(f"Control dwell:   {args.control_dwell} s")
    print(f"Shader pack:     {args.shader_pack} ({normalize_shader_pack(args.shader_pack)})")
    print(f"RT samples:      {args.rt_samples_per_pixel}")
    print(f"RT path depth:   {args.rt_path_depth}")
    print(f"RT denoiser:     {args.rt_denoiser}")
    print(f"Render preflight: {'enabled' if not args.no_render_preflight else 'disabled'}")
    print(f"Debug timing:    {'enabled' if args.debug_timing else 'disabled'}")
    if args.robot == "piper":
        print(f"Wrist camera:    {args.wrist_camera_width}x{args.wrist_camera_height}")
        print(f"Wrist display:   {'enabled' if args.show_wrist_camera else 'disabled'}")
        if args.show_wrist_camera:
            print(f"Wrist scale:     {args.wrist_camera_display_scale}x")
    if args.no_object:
        print("Grasp object:     disabled")
    else:
        print(f"Grasp object pos: {args.object_pos}")
        print(f"Grasp object size: {args.object_size} m")
    record_cameras = tuple(
        camera.strip() for camera in args.record_cameras.split(",") if camera.strip()
    )
    print(f"Recording:        {'enabled' if args.record else 'disabled'}")
    if args.record:
        print(f"Record dir:       {os.path.expanduser(args.record_dir)}")
        print(f"Repo id:          {args.repo_id}")
        print(f"Task:             put red box to blue plate")
        print(f"Record cameras:   {record_cameras}")
        print(f"Record FPS:       {args.record_fps}")
        print(f"Raw writer:       {max(1, args.image_writer_threads)} threads, queue={args.raw_writer_queue_size}")
        print("Record format:    raw uarm_sim_raw_v1")
    print("-" * 60)
    
    # Create and run simulation instance
    try:
        sim = ServoTeleoperatorSim(
            scene=args.scene,
            robot_uids=args.robot,
            serial_port=args.serial_port,
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
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
            raw_writer_queue_size=args.raw_writer_queue_size,
            env_render=args.env_render,
            control_dwell=args.control_dwell,
            debug_timing=args.debug_timing,
            debug_interval=args.debug_interval,
            debug_warmup=args.debug_warmup,
        )
        sim.rate = args.rate
        sim.timing.target_period = max(1.0 / sim.rate, 1e-6)
        sim.run()
    except Exception as e:
        print(f"Program runtime error: {e}")
        import traceback
        traceback.print_exc()
