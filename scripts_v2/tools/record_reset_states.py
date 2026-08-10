# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to record reset states using IsaacLab framework."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import torch
from tqdm import tqdm
from typing import cast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record reset states for object pairs.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task", type=str, default="OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0", help="Name of the task."
)
parser.add_argument(
    "--dataset_dir",
    type=str,
    default="./Datasets/OmniResetRealWorkspace/",
    help="Root Datasets/OmniResetRealWorkspace/ directory.",
)
parser.add_argument(
    "--reset_type",
    type=str,
    default=None,
    help="Reset type name (e.g. ObjectAnywhereEEAnywhere). Auto-inferred from --task if omitted.",
)
parser.add_argument(
    "--num_reset_states", type=int, default=100, help="Number of reset states to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_reset_conditions",
    type=int,
    default=None,
    help="Stop after evaluating this many reset candidates, regardless of how many succeed.",
)
parser.add_argument(
    "--report_initial_collisions",
    action="store_true",
    help="Report collision-free rates immediately after reset, before stepping physics.",
)
parser.add_argument(
    "--initial_collision_batches",
    type=int,
    default=1,
    help="Number of freshly sampled batches to aggregate for --report_initial_collisions.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import time

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.recorder_manager import DatasetExportMode

from uwlab.utils.datasets.torch_dataset_file_handler import TorchDatasetFileHandler

import uwlab_tasks  # noqa: F401
import uwlab_tasks.manager_based.manipulation.omnireset.mdp as task_mdp
from uwlab_tasks.utils.hydra import hydra_task_compose

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    """Main function to record reset states."""
    # create directory if it does not exist
    if not os.path.exists(args_cli.dataset_dir):
        os.makedirs(args_cli.dataset_dir, exist_ok=True)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # make sure environment is non-deterministic for diverse pose discovery
    env_cfg.seed = None

    # Derive pair directory and reset type for output path
    insertive_usd_path = env_cfg.scene.insertive_object.spawn.usd_path
    receptive_usd_path = env_cfg.scene.receptive_object.spawn.usd_path
    pair = task_mdp.utils.compute_pair_dir(insertive_usd_path, receptive_usd_path)

    # Auto-infer reset_type from task name if not provided
    reset_type = args_cli.reset_type
    if reset_type is None:
        for candidate in [
            "ObjectAnywhereEEAnywhere",
            "ObjectRestingEEGrasped",
            "ObjectAnywhereEEGrasped",
            "ObjectPartiallyAssembledEEGrasped",
        ]:
            if candidate in args_cli.task:
                reset_type = candidate
                break
        if reset_type is None:
            raise ValueError(f"Could not infer reset_type from task '{args_cli.task}'. Pass --reset_type explicitly.")

    print(f"Recording reset states for: {pair} / {reset_type}")
    print(f"Insertive: {insertive_usd_path}")
    print(f"Receptive: {receptive_usd_path}")

    # Setup recording configuration
    output_dir = os.path.join(args_cli.dataset_dir, "Resets", pair)
    os.makedirs(output_dir, exist_ok=True)
    output_file_name = f"resets_{reset_type}.pt"

    env_cfg.recorders = task_mdp.StableStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = TorchDatasetFileHandler

    # create environment
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    env.reset()

    if args_cli.report_initial_collisions:
        success_term = env.termination_manager.get_term_cfg("success").func
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        collision_labels = ["robot-object", "robot-receptive", "object-receptive"]
        collision_free_counts = torch.zeros(len(collision_labels), dtype=torch.long)
        for batch_index in range(args_cli.initial_collision_batches):
            if batch_index > 0:
                env.reset()
            for analyzer_index, collision_analyzer in enumerate(success_term.collision_analyzers):
                collision_free_counts[analyzer_index] += collision_analyzer(env, all_env_ids).sum().cpu()
        total_initial_states = env.num_envs * args_cli.initial_collision_batches
        print(f"Initial collision-free rates ({total_initial_states} states):")
        for label, collision_free_count in zip(collision_labels, collision_free_counts, strict=True):
            print(
                f"  {label}: {int(collision_free_count)}/{total_initial_states} "
                f"({collision_free_count.item() / total_initial_states:.2%})"
            )

    # Run reset state sampling
    num_reset_conditions_evaluated = 0
    current_successful_reset_conditions = 0
    actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
    ee_grasped = "EEGrasped" in args_cli.task
    if ee_grasped:
        actions[:, -1] = -1.0
    else:
        actions[:, -1] = (
            torch.randint(0, 2, (env.num_envs,), device=env.device, dtype=torch.float32) * 2 - 1
        )  # Randomly choose between -1 and 1

    # Create progress bar. A fixed evaluation budget is useful for measuring
    # comparable success rates without waiting for a low-rate reset type to
    # accumulate an arbitrary number of successful states.
    fixed_evaluation_budget = args_cli.num_reset_conditions is not None
    pbar = tqdm(
        total=args_cli.num_reset_conditions if fixed_evaluation_budget else args_cli.num_reset_states,
        desc="Evaluated reset conditions" if fixed_evaluation_budget else "Successful reset states",
        unit="reset conditions" if fixed_evaluation_budget else "reset states",
    )

    start_time = time.time()

    while (
        num_reset_conditions_evaluated < args_cli.num_reset_conditions
        if fixed_evaluation_budget
        else current_successful_reset_conditions < args_cli.num_reset_states
    ):
        # Step environment (this will evaluate grasps in parallel across environments)
        _, _, terminated, truncated, _ = env.step(actions)
        dones = terminated | truncated
        done_idx = torch.where(dones)[0]

        # Reset actions for environments that are done
        if done_idx.numel() > 0 and not ee_grasped:
            actions[done_idx, -1] = (
                torch.randint(0, 2, (done_idx.numel(),), device=env.device, dtype=torch.float32) * 2 - 1
            )

        # Update progress based on successful reset conditions
        new_successful_count = env.recorder_manager.exported_successful_episode_count
        if new_successful_count > current_successful_reset_conditions:
            increment = new_successful_count - current_successful_reset_conditions
            current_successful_reset_conditions = new_successful_count
            if not fixed_evaluation_budget:
                pbar.update(increment)

        # Count total reset conditions evaluated (sum across all environments)
        evaluated_increment = dones.sum().item()
        num_reset_conditions_evaluated += evaluated_increment
        if fixed_evaluation_budget:
            pbar.update(evaluated_increment)

        if env.sim.is_stopped():
            break

    pbar.close()

    # Get final statistics
    final_successful_reset_conditions = env.recorder_manager.exported_successful_episode_count
    print("Reset state recording complete!")
    print(f"Total reset conditions evaluated: {num_reset_conditions_evaluated}")
    print(f"Successful reset conditions: {final_successful_reset_conditions}")
    if num_reset_conditions_evaluated > 0:
        print(f"Success rate: {final_successful_reset_conditions / num_reset_conditions_evaluated:.2%}")
        print(f"Time taken: {(time.time() - start_time) / 60:.2f} minutes")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
