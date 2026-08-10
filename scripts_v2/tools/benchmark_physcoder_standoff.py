# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Measure one-shot PhysCoder reset acceptance over fixed standoff ranges."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True, help="Registered ObjectAnywhereEEAnywhere task ID.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--lower_start", type=float, default=0.15)
parser.add_argument("--lower_stop", type=float, default=0.35)
parser.add_argument("--step", type=float, default=0.01)
parser.add_argument("--range_width", type=float, default=0.05)
parser.add_argument("--trials", type=int, default=1, help="Independent 256-environment batches per range.")
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    del agent_cfg
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    term = env.event_manager.get_term_cfg("reset_end_effector_pose").func
    if not getattr(term, "_physcoder_active", False):
        raise RuntimeError("The selected assets did not activate the PhysCoder reset branch.")

    env_ids = torch.arange(env.num_envs, device=env.device)
    count = round((args_cli.lower_stop - args_cli.lower_start) / args_cli.step) + 1
    lower_bounds = [args_cli.lower_start + index * args_cli.step for index in range(count)]
    print("lower_m upper_m accepted total acceptance_pct worst_batch_pct")
    for lower in lower_bounds:
        accepted_per_trial = []
        for _ in range(args_cli.trials):
            term._physcoder_reset_robot(env, env_ids)
            term._physcoder_reset_box_block(env, env_ids)
            standoff = lower + args_cli.range_width * torch.rand(env.num_envs, device=env.device)
            target_pos, target_quat = term._physcoder_sample_sphere(env_ids, standoff)
            term._physcoder_place(env, env_ids, target_pos, target_quat)
            valid = term._physcoder_valid(env, env_ids, target_pos, require_box_bounds=True)
            accepted_per_trial.append(int(valid.sum().item()))
        accepted = sum(accepted_per_trial)
        total = env.num_envs * args_cli.trials
        print(
            f"{lower:.3f} {lower + args_cli.range_width:.3f} "
            f"{accepted} {total} {100.0 * accepted / total:.2f} "
            f"{100.0 * min(accepted_per_trial) / env.num_envs:.2f}",
            flush=True,
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
