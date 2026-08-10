# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Render short orbit videos of live OmniReset reset configurations."""

from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True, help="Registered OmniReset task ID.")
parser.add_argument("--output_dir", default="./Datasets/vis_resets", help="Video output directory.")
parser.add_argument("--num_videos", type=int, default=10, help="Number of independently reset videos.")
parser.add_argument("--start_index", type=int, default=0, help="Starting index for resumable batches.")
parser.add_argument(
    "--source_num_envs",
    type=int,
    default=1,
    help="Parallel environments to simulate before selecting the largest-motion dynamics videos.",
)
parser.add_argument("--frames", type=int, default=36, help="Frames per orbit video.")
parser.add_argument("--fps", type=int, default=18, help="Encoded video frame rate.")
parser.add_argument(
    "--settle_seconds",
    type=float,
    default=0.0,
    help="Render a fixed-camera zero-action settling trajectory instead of a static orbit.",
)
parser.add_argument(
    "--flat_output",
    action="store_true",
    help="Write videos directly into --output_dir instead of a task-name subdirectory.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

import isaaclab_tasks  # noqa: F401
import isaaclab.utils.math as math_utils
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    del agent_cfg
    env_cfg.scene.num_envs = args_cli.source_num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # Keep rendering enabled during the parallel rollout.  The selected
    # trajectories are replayed in the same simulator for video capture, so a
    # very large render interval leaves the viewport annotator with black data.
    env_cfg.sim.render_interval = 1
    env_cfg.seed = None

    # Dynamics videos must retain the same sampled state for the full window.
    # Disable automatic termination/reset; this does not alter the controller
    # or physics that produce the motion being diagnosed.
    if args_cli.settle_seconds > 0.0:
        env_cfg.terminations.time_out = None
        env_cfg.terminations.abnormal_robot = None
        env_cfg.terminations.success = None

    task_name = args_cli.task.removeprefix("OmniReset-UR5eRobotiq2f140-").removesuffix("-v0")
    task_output_dir = args_cli.output_dir if args_cli.flat_output else os.path.join(args_cli.output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    unwrapped = env.unwrapped

    if args_cli.settle_seconds > 0.0 and args_cli.source_num_envs > 1:
        env.reset()
        robot = unwrapped.scene["robot"]
        block = unwrapped.scene["insertive_object"]
        box = unwrapped.scene["receptive_object"]
        ee_body_index = robot.data.body_names.index("robotiq_base_link")
        actions = torch.zeros(unwrapped.action_space.shape, device=unwrapped.device, dtype=torch.float32)
        if "EEGrasped" in task_name:
            actions[:, -1] = -1.0
        else:
            actions[:, -1] = (
                torch.randint(0, 2, (unwrapped.num_envs,), device=unwrapped.device, dtype=torch.float32) * 2 - 1
            )

        def snapshot() -> dict[str, torch.Tensor]:
            return {
                "robot_root_pose": robot.data.root_pose_w.detach().cpu().clone(),
                "robot_joint_pos": robot.data.joint_pos.detach().cpu().clone(),
                "ee_pos": robot.data.body_link_pos_w[:, ee_body_index].detach().cpu().clone(),
                "block_root_pose": block.data.root_pose_w.detach().cpu().clone(),
                "box_root_pose": box.data.root_pose_w.detach().cpu().clone(),
            }

        step_dt = float(unwrapped.step_dt)
        num_steps = int(round(args_cli.settle_seconds / step_dt))
        snapshots = [snapshot()]
        for _ in range(num_steps):
            env.step(actions)
            snapshots.append(snapshot())

        final_displacement = torch.linalg.vector_norm(
            snapshots[-1]["ee_pos"] - snapshots[0]["ee_pos"], dim=-1
        )
        selected_env_ids = torch.topk(final_displacement, k=args_cli.num_videos).indices.tolist()
        dynamics_rows = []
        for video_offset, source_env_id in enumerate(selected_env_ids):
            video_index = args_cli.start_index + video_offset
            # The viewport does not reliably expose distant cloned environments.
            # Replay the sampled source trajectory in visible env 0, translating
            # world poses by the difference between the two environment origins.
            replay_ids = torch.tensor([0], dtype=torch.long, device=unwrapped.device)
            origin_shift = (
                unwrapped.scene.env_origins[0] - unwrapped.scene.env_origins[source_env_id]
            ).detach().cpu()
            box_position = (
                snapshots[0]["box_root_pose"][source_env_id, :3] + origin_shift
            ).numpy()
            target = box_position + np.array([0.0, 0.0, 0.12], dtype=np.float32)
            eye = target + np.array([0.58, 0.46, 0.34], dtype=np.float32)
            unwrapped.sim.set_camera_view(eye=eye.tolist(), target=target.tolist())
            frames = []
            trajectory = []
            initial_ee = snapshots[0]["ee_pos"][source_env_id]
            initial_block = snapshots[0]["block_root_pose"][source_env_id, :3]
            for frame_index, state in enumerate(snapshots):
                robot_root_pose = state["robot_root_pose"][source_env_id].clone()
                robot_root_pose[:3] += origin_shift
                robot.write_root_pose_to_sim(
                    robot_root_pose.to(unwrapped.device).unsqueeze(0), env_ids=replay_ids
                )
                joint_position = state["robot_joint_pos"][source_env_id].to(unwrapped.device).unsqueeze(0)
                robot.write_joint_state_to_sim(
                    joint_position,
                    torch.zeros_like(joint_position),
                    env_ids=replay_ids,
                )
                block_root_pose = state["block_root_pose"][source_env_id].clone()
                block_root_pose[:3] += origin_shift
                block.write_root_pose_to_sim(
                    block_root_pose.to(unwrapped.device).unsqueeze(0), env_ids=replay_ids
                )
                box_root_pose = state["box_root_pose"][source_env_id].clone()
                box_root_pose[:3] += origin_shift
                box.write_root_pose_to_sim(
                    box_root_pose.to(unwrapped.device).unsqueeze(0), env_ids=replay_ids
                )
                unwrapped.sim.forward()
                unwrapped.scene.update(0.0)
                if frame_index == 0:
                    unwrapped.render()
                    unwrapped.render()
                frame = unwrapped.render()
                if frame is None:
                    raise RuntimeError("The environment returned no RGB frame; pass --enable_cameras.")
                ee_displacement = float(torch.linalg.vector_norm(state["ee_pos"][source_env_id] - initial_ee))
                block_displacement = float(
                    torch.linalg.vector_norm(state["block_root_pose"][source_env_id, :3] - initial_block)
                )
                image = Image.fromarray(np.asarray(frame)[..., :3])
                draw = ImageDraw.Draw(image)
                lines = [
                    f"source env={source_env_id}",
                    f"t={frame_index * step_dt:0.1f}s",
                    f"EE displacement={ee_displacement * 100.0:0.1f} cm",
                    f"block displacement={block_displacement * 100.0:0.1f} cm",
                    f"gripper action={int(actions[source_env_id, -1].item()):+d}",
                ]
                draw.rectangle((8, 8, 305, 98), fill=(0, 0, 0))
                draw.multiline_text((14, 12), "\n".join(lines), fill=(255, 255, 255), spacing=2)
                frames.append(np.asarray(image))
                trajectory.append(
                    {
                        "time_s": frame_index * step_dt,
                        "ee_displacement_m": ee_displacement,
                        "block_displacement_m": block_displacement,
                    }
                )

            output_path = os.path.join(task_output_dir, f"reset_{video_index:02d}.mp4")
            imageio.mimsave(output_path, frames, fps=args_cli.fps, codec="libx264", quality=8)
            dynamics_rows.append(
                {
                    "video": video_index,
                    "source_env_id": source_env_id,
                    "step_dt_s": step_dt,
                    "gripper_action": int(actions[source_env_id, -1].item()),
                    "trajectory": trajectory,
                }
            )
            print(f"Rendered {output_path} from source environment {source_env_id}")

        with open(os.path.join(task_output_dir, "dynamics.json"), "w", encoding="utf-8") as file:
            json.dump(dynamics_rows, file, indent=2)
        env.close()
        return

    grasp_event = None
    if task_name in {"ObjectAnywhereEEGrasped", "ObjectRestingEEGrasped"}:
        grasp_event = unwrapped.event_manager.get_term_cfg("reset_end_effector_pose_from_grasp_dataset").func
    validate_in_box = bool(grasp_event is not None and grasp_event.workspace_filter_active)
    validation_rows = []
    dynamics_rows = []
    for video_index in range(args_cli.start_index, args_cli.start_index + args_cli.num_videos):
        env.reset()
        if validate_in_box:
            block = unwrapped.scene["insertive_object"]
            box = unwrapped.scene["receptive_object"]
            local_position, _ = math_utils.subtract_frame_transforms(
                box.data.root_pos_w[:1],
                box.data.root_quat_w[:1],
                block.data.root_pos_w[:1],
                block.data.root_quat_w[:1],
            )
            local_z = torch.tensor([[0.0, 0.0, 1.0]], device=unwrapped.device)
            world_z = math_utils.quat_apply(block.data.root_quat_w[:1], local_z)
            tilt = torch.acos(torch.clamp(world_z[:, 2], -1.0, 1.0))
            collision_free = bool(
                grasp_event.workspace_collision_analyzer(
                    unwrapped, torch.zeros(1, dtype=torch.long, device=unwrapped.device)
                )[0]
            )
            row = {
                "video": video_index,
                "block_x_in_box": float(local_position[0, 0]),
                "block_y_in_box": float(local_position[0, 1]),
                "block_z_in_box": float(local_position[0, 2]),
                "block_world_z_tilt": float(tilt[0]),
                "robot_box_collision_free": collision_free,
            }
            if not (
                abs(row["block_x_in_box"]) <= 0.09 + 1.0e-6
                and abs(row["block_y_in_box"]) <= 0.09 + 1.0e-6
                and row["block_z_in_box"] < 0.08 + 1.0e-6
                and row["block_world_z_tilt"] <= np.pi / 6.0 + 1.0e-6
                and collision_free
            ):
                raise RuntimeError(f"Reset violates requested in-box constraints: {row}")
            validation_rows.append(row)

        box_position = unwrapped.scene["receptive_object"].data.root_pos_w[0].detach().cpu().numpy()
        target = box_position + np.array([0.0, 0.0, 0.12], dtype=np.float32)
        if args_cli.settle_seconds > 0.0:
            eye = target + np.array([0.58, 0.46, 0.34], dtype=np.float32)
            unwrapped.sim.set_camera_view(eye=eye.tolist(), target=target.tolist())
            unwrapped.render()
            unwrapped.render()

            robot = unwrapped.scene["robot"]
            block = unwrapped.scene["insertive_object"]
            ee_body_index = robot.data.body_names.index("robotiq_base_link")
            initial_ee = robot.data.body_link_pos_w[0, ee_body_index].clone()
            initial_block = block.data.root_pos_w[0].clone()
            actions = torch.zeros(unwrapped.action_space.shape, device=unwrapped.device, dtype=torch.float32)
            if "EEGrasped" in task_name:
                actions[:, -1] = -1.0
            else:
                actions[:, -1] = float(torch.randint(0, 2, (1,), device=unwrapped.device).item() * 2 - 1)

            step_dt = float(unwrapped.step_dt)
            num_steps = int(round(args_cli.settle_seconds / step_dt))
            frames = []
            trajectory = []

            def capture_frame(elapsed: float) -> None:
                frame = env.render()
                if frame is None:
                    raise RuntimeError("The environment returned no RGB frame; pass --enable_cameras.")
                ee_displacement = float(
                    torch.linalg.vector_norm(robot.data.body_link_pos_w[0, ee_body_index] - initial_ee)
                )
                block_displacement = float(torch.linalg.vector_norm(block.data.root_pos_w[0] - initial_block))
                image = Image.fromarray(np.asarray(frame)[..., :3])
                draw = ImageDraw.Draw(image)
                lines = [
                    f"t={elapsed:0.1f}s",
                    f"EE displacement={ee_displacement * 100.0:0.1f} cm",
                    f"block displacement={block_displacement * 100.0:0.1f} cm",
                    f"gripper action={int(actions[0, -1].item()):+d}",
                ]
                draw.rectangle((8, 8, 305, 82), fill=(0, 0, 0))
                draw.multiline_text((14, 12), "\n".join(lines), fill=(255, 255, 255), spacing=2)
                frames.append(np.asarray(image))
                trajectory.append(
                    {
                        "time_s": elapsed,
                        "ee_displacement_m": ee_displacement,
                        "block_displacement_m": block_displacement,
                    }
                )

            capture_frame(0.0)
            for step_index in range(num_steps):
                env.step(actions)
                capture_frame((step_index + 1) * step_dt)

            output_path = os.path.join(task_output_dir, f"reset_{video_index:02d}.mp4")
            imageio.mimsave(output_path, frames, fps=args_cli.fps, codec="libx264", quality=8)
            dynamics_rows.append(
                {
                    "video": video_index,
                    "step_dt_s": step_dt,
                    "gripper_action": int(actions[0, -1].item()),
                    "trajectory": trajectory,
                }
            )
            print(f"Rendered {output_path}")
            continue

        # The viewport annotator is created lazily; warm it up so the encoded
        # video never begins with an empty black frame.
        unwrapped.render()
        unwrapped.render()
        frames = []
        for frame_index in range(args_cli.frames):
            angle = 2.0 * np.pi * frame_index / args_cli.frames
            eye = target + np.array(
                [0.62 * np.cos(angle), 0.62 * np.sin(angle), 0.34], dtype=np.float32
            )
            unwrapped.sim.set_camera_view(eye=eye.tolist(), target=target.tolist())
            frame = env.render()
            if frame is None:
                raise RuntimeError("The environment returned no RGB frame; pass --enable_cameras.")
            frames.append(np.asarray(frame)[..., :3])

        output_path = os.path.join(task_output_dir, f"reset_{video_index:02d}.mp4")
        imageio.mimsave(output_path, frames, fps=args_cli.fps, codec="libx264", quality=8)
        print(f"Rendered {output_path}")

    if validation_rows:
        with open(os.path.join(task_output_dir, "validation.json"), "w", encoding="utf-8") as file:
            json.dump(validation_rows, file, indent=2)
    if dynamics_rows:
        with open(os.path.join(task_output_dir, "dynamics.json"), "w", encoding="utf-8") as file:
            json.dump(dynamics_rows, file, indent=2)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
