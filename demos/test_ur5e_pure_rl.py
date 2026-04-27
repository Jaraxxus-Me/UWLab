# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone demo: load a pre-trained RL policy for the UR5e OmniReset task and
run it for a specified number of episodes, collecting per-episode statistics.

This mirrors the rsl_rl play.py pipeline but as a self-contained script with
explicit episode counting and per-episode logging.

Usage (activate the project venv first):
    # Rectangle + wall, 10 episodes, max 200 steps each
    python demos/test_ur5e_pure_rl.py \\
        --checkpoint rectangle_state_rl_expert_seed0.pt \\
        --num_envs 1 --num_episodes 10 --max_steps 200 \\
        env.scene.insertive_object=rectangle \\
        env.scene.receptive_object=wall

    # Peg + hole
    python demos/test_ur5e_pure_rl.py \\
        --checkpoint peg_state_rl_expert_seed42.pt \\
        --num_envs 1 --num_episodes 10 --max_steps 200 \\
        env.scene.insertive_object=peg \\
        env.scene.receptive_object=peghole

    # Headless
    python demos/test_ur5e_pure_rl.py \
        --checkpoint logs/rsl_rl/ur5e_robotiq_2f140_omnireset_agent/2026-04-26_02-37-26/model_6900.pt \
        --num_envs 4 --num_episodes 20 --max_steps 200 \
        env.scene.insertive_object=cube \
        env.scene.receptive_object=cube
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR5e pre-trained policy evaluation.")
parser.add_argument("--task", type=str,
                    default="OmniReset-Ur5eRobotiq2f140-RelCartesianOSC-State-Play-v0",
                    help="Registered gym task ID.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, default=10, help="Total number of episodes to run.")
parser.add_argument("--max_steps", type=int, default=200,
                    help="Max policy steps per episode (policy runs at ~10 Hz with decimation=12).")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible.")
AppLauncher.add_app_launcher_args(parser)

# Split CLI into known args and Hydra overrides (e.g. env.scene.insertive_object=rectangle)
args_cli, hydra_args = parser.parse_known_args()
# Clear sys.argv so Hydra doesn't see argparse flags
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import functools
import time

import torch

# Ensure prints are not buffered
print = functools.partial(print, flush=True)  # type: ignore[assignment]

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import importlib.metadata as metadata
import inspect

from isaaclab.managers import ManagerTermBase

import isaaclab_tasks  # noqa: F401  — registers Isaac Lab gym envs
import uwlab_tasks  # noqa: F401  — registers UW Lab gym envs
from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
from uwlab_tasks.utils.hydra import hydra_task_config


def force_init_event_term_classes(env) -> None:
    """Instantiate any class-based ``ManagerTermBase`` event terms up front.

    The ``EventManager`` defers class-term construction to a timeline PLAY callback
    that can silently fail — when it does, ``term_cfg.func`` is still the class and
    the next apply call hits ``__init__`` with the param kwargs (e.g. ``dataset_dir``)
    instead of ``__call__``, raising ``TypeError``.
    """
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)


def sanitize_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg) -> RslRlBaseRunnerCfg:
    """Strip algorithm-config keys the installed rsl-rl version doesn't accept."""
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    alg_cfg = agent_cfg.algorithm
    class_name = getattr(alg_cfg, "class_name", None)
    if class_name is not None:
        from rsl_rl import algorithms
        alg_class = getattr(algorithms, class_name, None)
        if alg_class is not None:
            accepted = set(inspect.signature(alg_class.__init__).parameters.keys())
            for key in list(vars(alg_cfg)):
                if key != "class_name" and key not in accepted:
                    delattr(alg_cfg, key)
    return agent_cfg


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Load policy and run episodes."""

    # ------------------------------------------------------------------
    # 1. Configure environment
    # ------------------------------------------------------------------
    agent_cfg = sanitize_rsl_rl_cfg(agent_cfg)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ------------------------------------------------------------------
    # 2. Create environment + RSL-RL wrapper
    # ------------------------------------------------------------------
    env = gym.make(args_cli.task, cfg=env_cfg)
    force_init_event_term_classes(env.unwrapped)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    num_envs = env.num_envs
    device = env.device

    # ------------------------------------------------------------------
    # 3. Load checkpoint
    # ------------------------------------------------------------------
    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=device)

    # Extract the neural network module for reset() calls
    try:
        policy_nn = runner.alg.policy  # rsl-rl >= 2.3
    except AttributeError:
        policy_nn = runner.alg.actor_critic  # rsl-rl < 2.3

    # ------------------------------------------------------------------
    # 4. Run episodes
    # ------------------------------------------------------------------
    dt = env.unwrapped.step_dt

    # Per-env tracking
    episode_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_return = torch.zeros(num_envs, device=device)

    # Aggregate stats
    completed_returns = []
    completed_lengths = []

    total_target = args_cli.num_episodes
    max_steps = args_cli.max_steps

    obs = env.get_observations()
    print(f"[INFO] Running {total_target} episodes, max {max_steps} steps/episode, {num_envs} env(s).")
    print("-" * 80)

    while simulation_app.is_running():
        # Check if we've collected enough episodes
        if len(completed_returns) >= total_target:
            break

        start_time = time.time()

        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
            policy_nn.reset(dones)

        episode_step += 1
        episode_return += rewards

        # Check for done envs (terminated or truncated) or max steps reached
        timed_out = episode_step >= max_steps
        finished = (dones.bool() | timed_out)

        if finished.any():
            for idx in finished.nonzero(as_tuple=False).squeeze(-1).tolist():
                if len(completed_returns) >= total_target:
                    break

                ep_num = len(completed_returns) + 1
                ep_ret = episode_return[idx].item()
                ep_len = episode_step[idx].item()
                completed_returns.append(ep_ret)
                completed_lengths.append(ep_len)

                done_reason = "done" if dones[idx].bool().item() else "max_steps"
                print(
                    f"  Episode {ep_num:3d}/{total_target} | "
                    f"env {idx} | steps: {ep_len:4d} | return: {ep_ret:8.3f} | {done_reason}"
                )

                # Reset tracking for this env
                episode_step[idx] = 0
                episode_return[idx] = 0.0
                episode_count[idx] += 1

            # Force-reset envs that hit max_steps but env didn't auto-reset
            force_reset_mask = timed_out & ~dones.bool()
            if force_reset_mask.any():
                # The env auto-resets on done; for max_steps we just zero the counters
                # and let the env continue (it will reset on next done).
                # If you need a hard reset, you'd call env.reset() — but with vectorized
                # envs that resets ALL envs. Instead we just track episodes ourselves.
                episode_step[force_reset_mask] = 0
                episode_return[force_reset_mask] = 0.0

        # Real-time pacing
        if args_cli.real_time:
            elapsed = time.time() - start_time
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print("-" * 80)
    if completed_returns:
        returns_t = torch.tensor(completed_returns)
        lengths_t = torch.tensor(completed_lengths, dtype=torch.float)
        print(f"[RESULTS] {len(completed_returns)} episodes completed:")
        print(f"  Return  — mean: {returns_t.mean():.3f}, std: {returns_t.std():.3f}, "
              f"min: {returns_t.min():.3f}, max: {returns_t.max():.3f}")
        print(f"  Length  — mean: {lengths_t.mean():.1f}, std: {lengths_t.std():.1f}, "
              f"min: {lengths_t.min():.0f}, max: {lengths_t.max():.0f}")
    else:
        print("[RESULTS] No episodes completed.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
