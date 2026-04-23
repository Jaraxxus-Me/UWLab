# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize receptive_object (box) placement at a pinned (x, y, yaw).

Pins the task's ``reset_receptive_object_pose`` event to exactly the CLI-specified
pose, steps ``--settle_steps`` zero-action policy steps so physics can settle, then
keeps the Isaac Sim viewport open for inspection.

Use this to tune the uniform pose range currently set in
``ResetStatesBaseEventCfg.reset_receptive_object_pose`` for a new receptive asset.

Usage (activate the project venv first):
    python demos/visualize_box_pos.py \\
        --x 0.425 --y 0.1 --yaw 0.0 \\
        env.scene.insertive_object=block env.scene.receptive_object=box
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize receptive_object placement at a pinned (x, y, yaw).")
parser.add_argument(
    "--task",
    type=str,
    default="OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0",
    help="Task id with a reset_receptive_object_pose event (any reset_states variant works).",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--x", type=float, default=0.425, help="Pinned box x (m, support frame).")
parser.add_argument("--y", type=float, default=0.10, help="Pinned box y (m, support frame).")
parser.add_argument("--yaw", type=float, default=0.0, help="Pinned box yaw (rad).")
parser.add_argument("--settle_steps", type=int, default=20, help="Zero-action policy steps after reset.")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=hydra_args)
def main(env_cfg, agent_cfg) -> None:
    # Pin the receptive-object reset pose to the CLI point.
    rr = env_cfg.events.reset_receptive_object_pose.params["pose_range"]
    rr["x"] = (args_cli.x, args_cli.x)
    rr["y"] = (args_cli.y, args_cli.y)
    rr["yaw"] = (args_cli.yaw, args_cli.yaw)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 0
    # Long episode so physics doesn't auto-reset mid-inspection.
    env_cfg.episode_length_s = 1e6

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    assert isinstance(env, ManagerBasedRLEnv)

    env.reset()

    zero_action = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
    for _ in range(args_cli.settle_steps):
        env.step(zero_action)

    receptive = env.scene["receptive_object"]
    pos = receptive.data.root_pos_w[0].tolist()
    quat = receptive.data.root_quat_w[0].tolist()
    print("---")
    print(f"Pinned receptive x={args_cli.x:.3f} y={args_cli.y:.3f} yaw={args_cli.yaw:.3f}")
    print(f"World pose: pos={[f'{p:.4f}' for p in pos]} quat={[f'{q:.4f}' for q in quat]}")
    print("Viewport is live. Close the window or Ctrl+C to quit.")

    while simulation_app.is_running():
        env.step(zero_action)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
