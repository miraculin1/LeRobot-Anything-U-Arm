import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from rollout_sim import DEFAULT_INITIAL_STATE, ZeroActionRolloutSim, normalize_shader_pack


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate OpenPI rollout success rate in the Piper simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot", default="piper", choices=["piper"])
    parser.add_argument("--scene", default="ReplicaCAD_SceneManipulation-v1")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--policy-mode", choices=["zero", "openpi"], default="openpi")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="put red box to blue plate")
    parser.add_argument("--open-loop-horizon", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--action-mode",
        choices=["absolute", "relative", "delta"],
        default="absolute",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="sensors",
        choices=["human", "rgb_array", "sensors", "all"],
    )
    parser.add_argument("--env-render", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show live sensor camera windows and the gripper/contact plot during evaluation",
    )
    parser.add_argument(
        "--display-cameras",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show live OpenCV windows for the default top and wrist sensor cameras",
    )
    parser.add_argument(
        "--show-gripper-plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show the live gripper opening, policy action, and contact plot",
    )
    parser.add_argument("--gripper-plot-history", type=int, default=300)
    parser.add_argument("--gripper-plot-update-rate", type=float, default=5.0)
    parser.add_argument("--shader-pack", type=str, default="fast_rt")
    parser.add_argument("--rt-samples-per-pixel", type=int, default=2)
    parser.add_argument("--rt-path-depth", type=int, default=1)
    parser.add_argument("--rt-denoiser", type=str, default="oidn", choices=["none", "oidn", "optix"])
    parser.add_argument("--render-preflight", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wrist-camera-width", type=int, default=320)
    parser.add_argument("--wrist-camera-height", type=int, default=320)
    parser.add_argument(
        "--object-pos",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[0.457, -1.612, 0.956],
    )
    parser.add_argument("--object-size", type=float, default=0.04)
    parser.add_argument("--randomize-all-task-objects", action="store_true", default=True)
    parser.add_argument("--no-randomize-all-task-objects", dest="randomize_all_task_objects", action="store_false")
    parser.add_argument("--randomize-object-yaw", action="store_true", default=True)
    parser.add_argument("--no-randomize-object-yaw", dest="randomize_object_yaw", action="store_false")
    parser.add_argument("--random-workspace-inner-diameter", type=float, default=0.60)
    parser.add_argument("--random-workspace-outer-diameter", type=float, default=1.20)
    parser.add_argument(
        "--initial-state",
        type=float,
        nargs=7,
        default=DEFAULT_INITIAL_STATE.tolist(),
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "GRIPPER"),
    )
    parser.add_argument("--success-contact-force-threshold", type=float, default=1e-4)
    parser.add_argument("--home-joint-l2-threshold", type=float, default=0.35)
    parser.add_argument("--home-joint-max-threshold", type=float, default=0.20)
    parser.add_argument(
        "--screenshot-cameras",
        type=str,
        default="d435_top_camera,wrist_camera",
        help="Comma-separated sensor camera names to save at the end of each episode",
    )
    parser.add_argument("--output-dir", type=str, default="eval_results/openpi_rollout_eval")
    parser.add_argument("--debug-timing", action="store_true")
    parser.add_argument("--debug-interval", type=int, default=30)
    parser.add_argument("--debug-warmup", type=int, default=5)
    parser.add_argument("--debug-action-interval", type=int, default=0)
    return parser.parse_args()


def force_norm_between(sim: ZeroActionRolloutSim, actor_a, actor_b) -> float:
    try:
        forces = sim.env.unwrapped.scene.get_pairwise_contact_forces(actor_a, actor_b)
    except (AttributeError, IndexError, RuntimeError):
        return 0.0
    if hasattr(forces, "detach"):
        forces = forces.detach().cpu().numpy()
    forces = np.asarray(forces, dtype=np.float32).reshape(-1, 3)
    return float(np.linalg.norm(forces, axis=1).max(initial=0.0))


def is_home(state, initial_state, l2_threshold: float, max_threshold: float):
    if state is None:
        return False, None, None
    diff = np.asarray(state[:6], dtype=np.float32) - np.asarray(initial_state[:6], dtype=np.float32)
    l2_error = float(np.linalg.norm(diff))
    max_error = float(np.max(np.abs(diff), initial=0.0))
    return l2_error <= l2_threshold and max_error <= max_threshold, l2_error, max_error


def save_episode_screenshots(sim: ZeroActionRolloutSim, camera_names, screenshot_dir: Path, episode_index: int):
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    frames = sim._collect_camera_frames(camera_names)
    paths = {}
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Saving screenshots requires OpenCV in the active environment") from exc
    for camera_name in camera_names:
        frame = frames.get(camera_name)
        if frame is None:
            paths[camera_name] = None
            print(f"[WARN] Episode {episode_index}: missing screenshot frame for {camera_name}")
            continue
        path = screenshot_dir / f"episode_{episode_index:06d}_{camera_name}.png"
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), bgr):
            raise RuntimeError(f"Failed to write screenshot: {path}")
        paths[camera_name] = str(path)
    return paths


def make_policy_client(args):
    if args.policy_mode != "openpi":
        return None
    try:
        from openpi_client import websocket_client_policy
    except ImportError as exc:
        raise RuntimeError(
            "OpenPI policy mode requires openpi-client in the uarm environment."
        ) from exc
    policy_client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    print(f"OpenPI server metadata: {policy_client.get_server_metadata()}")
    return policy_client


def run_episode(sim: ZeroActionRolloutSim, args, policy_client, episode_index: int):
    sim.stop_event.clear()
    sim.task = args.prompt
    sim._set_initial_robot_state()
    sim._randomize_task_objects()

    step_count = 0
    actions_from_chunk_completed = 0
    pred_action_chunk = None
    representation_base_state = None
    previous_delta_target = None
    success = False
    max_contact_force = 0.0
    first_success_step = None
    period = max(1.0 / args.rate, 1e-6)
    next_time = time.monotonic()

    blue_plate = sim.plates[0] if sim.plates else None
    if sim.grasp_object is None or blue_plate is None:
        raise RuntimeError("Evaluation requires the red box and blue plate actors")

    while step_count < args.max_steps:
        if args.policy_mode == "zero":
            policy_action = sim.zero_policy_action()
            env_action = sim.zero_env_action()
        elif args.policy_mode == "openpi":
            if pred_action_chunk is None or actions_from_chunk_completed >= min(
                args.open_loop_horizon, pred_action_chunk.shape[0]
            ):
                actions_from_chunk_completed = 0
                previous_delta_target = None
                obs = sim.get_policy_observation(args.prompt, args.image_size)
                representation_base_state = np.asarray(
                    obs["observation/state"], dtype=np.float32
                ).reshape(-1)
                response = policy_client.infer(obs)
                pred_action_chunk = sim.validate_action_chunk(response)
            policy_action = pred_action_chunk[actions_from_chunk_completed]
            actions_from_chunk_completed += 1
            env_action, previous_delta_target = sim.policy_action_to_env_action(
                policy_action,
                args.action_mode,
                representation_base_state=representation_base_state,
                previous_delta_target=previous_delta_target,
            )
        else:
            raise ValueError(f"Unsupported policy mode: {args.policy_mode}")

        sim.rollout_step(env_action)
        step_count += 1
        contact_force = force_norm_between(sim, sim.grasp_object, blue_plate)
        max_contact_force = max(max_contact_force, contact_force)
        if not success and contact_force > args.success_contact_force_threshold:
            success = True
            first_success_step = step_count

        state = sim._get_piper_record_state()
        sim._log_openpi_action_debug(
            step_count,
            args.action_mode,
            policy_action,
            env_action,
            state,
            representation_base_state=representation_base_state,
        )
        sim._update_gripper_plot(step_count, state, policy_action)

        next_time += period
        sleep_dt = next_time - time.monotonic()
        if sleep_dt > 0:
            time.sleep(sleep_dt)
        else:
            next_time = time.monotonic()

    final_state = sim._get_piper_record_state()
    home, home_l2_error, home_max_error = is_home(
        final_state,
        sim.initial_state,
        args.home_joint_l2_threshold,
        args.home_joint_max_threshold,
    )
    screenshot_paths = save_episode_screenshots(
        sim,
        tuple(name.strip() for name in args.screenshot_cameras.split(",") if name.strip()),
        Path(args.output_dir) / "screenshots",
        episode_index,
    )

    return {
        "episode": episode_index,
        "steps": step_count,
        "success": success,
        "first_success_step": first_success_step,
        "max_contact_force": max_contact_force,
        "returned_home": home,
        "home_joint_l2_error": home_l2_error,
        "home_joint_max_error": home_max_error,
        "final_state": None if final_state is None else final_state.astype(float).tolist(),
        "screenshots": screenshot_paths,
    }


def write_results(output_dir: Path, args, episode_results):
    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = sum(1 for result in episode_results if result["success"])
    home_count = sum(1 for result in episode_results if result["returned_home"])
    total = len(episode_results)
    payload = {
        "config": vars(args),
        "summary": {
            "episodes": total,
            "success_count": success_count,
            "success_rate": success_count / total if total else 0.0,
            "returned_home_count": home_count,
            "returned_home_rate": home_count / total if total else 0.0,
        },
        "episodes": episode_results,
    }
    results_path = output_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return results_path, payload


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.open_loop_horizon <= 0:
        raise ValueError("--open-loop-horizon must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_names = tuple(name.strip() for name in args.screenshot_cameras.split(",") if name.strip())
    if not camera_names:
        raise ValueError("--screenshot-cameras must contain at least one camera name")

    print("=" * 60)
    print("    Piper OpenPI Rollout Evaluation")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Policy mode: {args.policy_mode}")
    if args.policy_mode == "openpi":
        print(f"OpenPI server: {args.host}:{args.port}")
        print(f"Open-loop horizon: {args.open_loop_horizon}")
        print(f"Action mode: {args.action_mode}")
    print(f"Prompt: {args.prompt}")
    print(f"Render mode: {args.render_mode}")
    print(f"Env render: {'enabled' if args.env_render else 'disabled'}")
    print(f"Live camera display: {'enabled' if args.display_cameras or args.visualize else 'disabled'}")
    print(f"Gripper plot: {'enabled' if args.show_gripper_plot or args.visualize else 'disabled'}")
    print(f"Shader pack: {args.shader_pack} ({normalize_shader_pack(args.shader_pack)})")
    print(f"Screenshot cameras: {camera_names}")
    print(f"Output dir: {output_dir}")
    print("-" * 60)

    policy_client = make_policy_client(args)
    sim = ZeroActionRolloutSim(
        scene=args.scene,
        robot_uids=args.robot,
        object_pos=args.object_pos,
        object_size=args.object_size,
        spawn_object=True,
        render_mode=args.render_mode,
        wrist_camera_width=args.wrist_camera_width,
        wrist_camera_height=args.wrist_camera_height,
        show_wrist_camera=False,
        shader_pack=args.shader_pack,
        rt_samples_per_pixel=args.rt_samples_per_pixel,
        rt_path_depth=args.rt_path_depth,
        rt_denoiser=args.rt_denoiser,
        render_preflight=args.render_preflight,
        record=False,
        rate=args.rate,
        display_cameras=args.display_cameras or args.visualize,
        initial_state=args.initial_state,
        debug_timing=args.debug_timing,
        debug_interval=args.debug_interval,
        debug_warmup=args.debug_warmup,
        env_render=args.env_render,
        show_gripper_plot=args.show_gripper_plot or args.visualize,
        gripper_plot_history=args.gripper_plot_history,
        gripper_plot_update_rate=args.gripper_plot_update_rate,
        debug_action_interval=args.debug_action_interval,
        randomize_all_task_objects=args.randomize_all_task_objects,
        randomize_object_yaw=args.randomize_object_yaw,
        show_random_workspace=False,
        random_workspace_inner_diameter=args.random_workspace_inner_diameter,
        random_workspace_outer_diameter=args.random_workspace_outer_diameter,
    )

    episode_results = []
    try:
        for episode_index in range(1, args.episodes + 1):
            print(f"[EVAL] Starting episode {episode_index}/{args.episodes}")
            result = run_episode(sim, args, policy_client, episode_index)
            episode_results.append(result)
            print(
                "[EVAL] Episode {episode}: success={success} "
                "first_success_step={first_success_step} max_contact_force={max_contact_force:.6f} "
                "returned_home={returned_home} home_l2={home_joint_l2_error} "
                "home_max={home_joint_max_error}".format(**result)
            )
            write_results(output_dir, args, episode_results)
    finally:
        sim.close_resources()

    results_path, payload = write_results(output_dir, args, episode_results)
    summary = payload["summary"]
    print("-" * 60)
    print(
        f"[EVAL] Success: {summary['success_count']}/{summary['episodes']} "
        f"({summary['success_rate']:.3f})"
    )
    print(
        f"[EVAL] Returned home: {summary['returned_home_count']}/{summary['episodes']} "
        f"({summary['returned_home_rate']:.3f})"
    )
    print(f"[EVAL] Results saved to: {results_path}")


if __name__ == "__main__":
    main()
