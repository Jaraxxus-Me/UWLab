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
parser.add_argument("--frames", type=int, default=36, help="Frames per orbit video.")
parser.add_argument("--fps", type=int, default=18, help="Encoded video frame rate.")
parser.add_argument(
    "--max_filter_attempts",
    type=int,
    default=256,
    help="Maximum reset attempts used to obtain each accepted grasped pose.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab.utils.math as math_utils
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    del agent_cfg
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.render_interval = 1
    env_cfg.seed = None

    task_name = args_cli.task.removeprefix("OmniReset-UR5eRobotiq2f140-").removesuffix("-v0")
    task_output_dir = os.path.join(args_cli.output_dir, task_name)
    os.makedirs(task_output_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    unwrapped = env.unwrapped
    insertive = unwrapped.scene["insertive_object"]
    receptive = unwrapped.scene["receptive_object"]
    robot = unwrapped.scene["robot"]
    resync_pair = (
        getattr(insertive.cfg, "reset_profile", None) == "resync_cube_region"
        and getattr(receptive.cfg, "reset_profile", None) == "resync_cube_region"
    )
    ee_body_idx = robot.data.body_names.index("robotiq_base_link")
    finger_joint_idx = robot.joint_names.index("finger_joint")
    grasped_tasks = {
        "ObjectAnywhereEEGrasped",
        "ObjectRestingEEGrasped",
        "ObjectPartiallyAssembledEEGrasped",
    }
    grasp_event = None
    if task_name in grasped_tasks:
        grasp_event = unwrapped.event_manager.get_term_cfg("reset_end_effector_pose_from_grasp_dataset").func
    validate_in_box = bool(grasp_event is not None and grasp_event.workspace_filter_active)
    success_term = unwrapped.termination_manager.get_term_cfg("success").func
    approach_local = torch.tensor(
        success_term.gripper_approach_direction, device=unwrapped.device, dtype=torch.float32
    ).unsqueeze(0)
    table_collision_analyzer = next(
        (
            analyzer
            for analyzer in success_term.collision_analyzers
            if any(obstacle_cfg.name == "table" for obstacle_cfg in analyzer.cfg.obstacle_cfgs)
        ),
        None,
    )
    if task_name in grasped_tasks and table_collision_analyzer is None:
        raise RuntimeError(f"{args_cli.task} does not configure robot-table collision rejection.")

    def reset_until_pose_filter_accepts() -> tuple[int, float, bool, bool]:
        """Apply the reset-pose subset of the termination filter before rendering."""
        for attempt in range(1, args_cli.max_filter_attempts + 1):
            env.reset()
            ee_quat = robot.data.body_link_quat_w[:1, ee_body_idx]
            approach_world = math_utils.quat_apply(ee_quat, approach_local)
            approach_z = float(approach_world[0, 2])
            table_collision_free = bool(
                table_collision_analyzer(
                    unwrapped, torch.zeros(1, dtype=torch.long, device=unwrapped.device)
                )[0]
            )
            robot_above_tabletop = bool(
                success_term.robot_above_minimum_height(
                    torch.zeros(1, dtype=torch.long, device=unwrapped.device)
                )[0]
            )
            if approach_z < -0.5 and table_collision_free and robot_above_tabletop:
                return attempt, approach_z, table_collision_free, robot_above_tabletop
        raise RuntimeError(
            f"Could not sample a downward, robot-table-collision-free pose for {args_cli.task} "
            f"after {args_cli.max_filter_attempts} attempts."
        )

    validation_rows = []
    for video_index in range(args_cli.start_index, args_cli.start_index + args_cli.num_videos):
        filter_attempts = 1
        filtered_approach_z = float("nan")
        robot_table_collision_free = True
        robot_above_tabletop = True
        if task_name in grasped_tasks:
            (
                filter_attempts,
                filtered_approach_z,
                robot_table_collision_free,
                robot_above_tabletop,
            ) = reset_until_pose_filter_accepts()
        else:
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

        if resync_pair:
            cube_pos = insertive.data.root_pos_w[0]
            region_pos = receptive.data.root_pos_w[0]
            ee_pos = robot.data.body_link_pos_w[0, ee_body_idx]
            ee_quat = robot.data.body_link_quat_w[0, ee_body_idx]
            cube_region_xy = torch.linalg.vector_norm((cube_pos - region_pos)[:2])
            ee_cube_delta = ee_pos - cube_pos
            world_approach = math_utils.quat_apply(ee_quat.unsqueeze(0), approach_local)[0]
            approach_angle = torch.acos(torch.clamp(-world_approach[2], -1.0, 1.0))
            finger_error = torch.abs(
                robot.data.joint_pos[0, finger_joint_idx] - robot.data.default_joint_pos[0, finger_joint_idx]
            )
            row = {
                "video": video_index,
                "region_x": float(region_pos[0]),
                "region_y": float(region_pos[1]),
                "cube_region_xy_distance": float(cube_region_xy),
                "ee_cube_xy_distance": float(torch.linalg.vector_norm(ee_cube_delta[:2])),
                "ee_cube_height": float(ee_cube_delta[2]),
                "ee_approach_angle_degrees": float(torch.rad2deg(approach_angle)),
                "finger_open_error": float(finger_error),
                "reset_filter_attempts": filter_attempts,
                "gripper_approach_world_z": filtered_approach_z,
                "robot_table_collision_free": robot_table_collision_free,
                "robot_above_tabletop": robot_above_tabletop,
            }
            if not (-0.65 - 1.0e-5 <= row["region_x"] <= -0.35 + 1.0e-5):
                raise RuntimeError(f"ReSYNC region X reset is out of range: {row}")
            if not (-0.2 - 1.0e-5 <= row["region_y"] <= 0.2 + 1.0e-5):
                raise RuntimeError(f"ReSYNC region Y reset is out of range: {row}")

            if task_name == "ObjectAnywhereEEAnywhere":
                if not (
                    0.08 - 1.0e-5 <= row["cube_region_xy_distance"] <= 0.12 + 1.0e-5
                    and row["ee_cube_xy_distance"] < 0.05
                    and 0.25 - 1.0e-5 <= row["ee_cube_height"] <= 0.35 + 1.0e-5
                    and row["ee_approach_angle_degrees"] <= 15.0 + 1.0e-4
                    and row["finger_open_error"] < 1.0e-4
                ):
                    raise RuntimeError(f"ReSYNC top-down pre-grasp reset violates its constraints: {row}")
            elif task_name in grasped_tasks:
                if not (
                    row["ee_cube_height"] <= 0.3 + 1.0e-5
                    and row["gripper_approach_world_z"] < -0.5
                    and row["robot_table_collision_free"]
                    and row["robot_above_tabletop"]
                ):
                    raise RuntimeError(f"ReSYNC grasped EE reset violates its pose filter: {row}")
            validation_rows.append(row)

        box_position = receptive.data.root_pos_w[0].detach().cpu().numpy()
        target = box_position + np.array([0.0, 0.0, 0.12], dtype=np.float32)
        # The viewport annotator is created lazily; warm it up so the encoded
        # video never begins with an empty black frame.
        unwrapped.render(recompute=True)
        unwrapped.render(recompute=True)
        frames = []
        for frame_index in range(args_cli.frames):
            angle = 2.0 * np.pi * frame_index / args_cli.frames
            eye = target + np.array(
                [0.82 * np.cos(angle), 0.82 * np.sin(angle), 0.48], dtype=np.float32
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

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
