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
    grasp_event = None
    if task_name in {"ObjectAnywhereEEGrasped", "ObjectRestingEEGrasped"}:
        grasp_event = unwrapped.event_manager.get_term_cfg("reset_end_effector_pose_from_grasp_dataset").func
    validate_in_box = bool(grasp_event is not None and grasp_event.workspace_filter_active)
    validation_rows = []
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
        # The viewport annotator is created lazily; warm it up so the encoded
        # video never begins with an empty black frame.
        unwrapped.render(recompute=True)
        unwrapped.render(recompute=True)
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

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
