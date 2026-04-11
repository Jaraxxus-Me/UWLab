# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hybrid policy demo: scripted "bring-to-goal" phase followed by RL policy takeover.

Each episode:
  1. **Scripted phase** — Computes position/orientation error between the current EE
     pose and a desired goal pose, then outputs normalized OSC delta actions to drive
     the arm toward the goal. The gripper stays open during this phase.
  2. **RL phase** — Once the EE is within a position/orientation threshold of the goal,
     the pre-trained RL policy takes over for the insertion task.

The desired goal pose is visualized as a coordinate frame in the simulator.

The env action space is [arm(6), gripper(1)]:
  - arm: (dx, dy, dz, drx, dry, drz) relative Cartesian, scaled by (0.02, 0.02, 0.02, 0.02, 0.02, 0.2)
  - gripper: 0=open, 1=close

Usage (activate the project venv first):
    python demos/test_ur5e_hybrid_policy.py \\
        --checkpoint rectangle_state_rl_expert_seed0.pt \\
        --num_envs 1 --num_episodes 5 --max_steps 400 \\
        env.scene.insertive_object=rectangle \\
        env.scene.receptive_object=wall
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR5e hybrid (scripted + RL) policy demo.")
parser.add_argument("--task", type=str,
                    default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0",
                    help="Registered gym task ID.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt checkpoint file.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, default=10, help="Total number of episodes to run.")
parser.add_argument("--max_steps", type=int, default=400,
                    help="Max policy steps per episode.")
parser.add_argument("--pos_threshold", type=float, default=0.02,
                    help="Position error (m) to switch from scripted to RL.")
parser.add_argument("--rot_threshold", type=float, default=0.1,
                    help="Orientation error (rad) to switch from scripted to RL.")
parser.add_argument("--goal_pos", type=float, nargs=3, default=[0.4, 0.0, 0.4],
                    help="EE goal position in robot base frame (x y z).")
parser.add_argument("--goal_rpy", type=float, nargs=3, default=[3.14159, 0.0, 0.0],
                    help="EE goal orientation as roll-pitch-yaw in radians.")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import functools
import importlib.metadata as metadata
import inspect
import time

import torch

print = functools.partial(print, flush=True)  # type: ignore[assignment]

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import (
    axis_angle_from_quat,
    compute_pose_error,
    euler_xyz_from_quat,
    quat_from_euler_xyz,
    subtract_frame_transforms,
)
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

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


def get_ee_pose_in_base(env):
    """Get EE (wrist_3_link) pose in robot base frame. Returns (pos_b, quat_b) each (N,3)/(N,4)."""
    robot = env.unwrapped.scene["robot"]
    # Find wrist_3_link body index
    ee_idx = robot.find_bodies("wrist_3_link")[0][0]
    # World-frame poses
    ee_pos_w = robot.data.body_pos_w[:, ee_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_idx]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    # Transform to base frame
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    return ee_pos_b, ee_quat_b


class ScriptedBringToGoal:
    """Outputs OSC delta actions to drive EE toward a goal pose.

    The action is: error / scale, clipped to [-1, 1]. This is a simple
    proportional controller in the OSC action space.
    """

    def __init__(self, goal_pos, goal_quat, action_scale, num_envs, device, gain=1.0):
        """
        Args:
            goal_pos: (3,) tensor — desired EE position in robot base frame.
            goal_quat: (4,) tensor — desired EE orientation (w,x,y,z) in robot base frame.
            action_scale: (6,) tensor — the OSC action scaling factors.
            num_envs: Number of parallel environments.
            device: Torch device.
            gain: Proportional gain multiplier (1.0 = full correction per step).
        """
        self.goal_pos = goal_pos.to(device).unsqueeze(0).expand(num_envs, -1)
        self.goal_quat = goal_quat.to(device).unsqueeze(0).expand(num_envs, -1)
        self.action_scale = action_scale.to(device)
        self.gain = gain

    def compute(self, ee_pos_b, ee_quat_b):
        """Compute normalized OSC actions toward the goal.

        Returns:
            arm_actions: (N, 6) tensor clipped to [-1, 1]
            pos_error_norm: (N,) scalar position error magnitude
            rot_error_norm: (N,) scalar rotation error magnitude
        """
        pos_error, rot_error_aa = compute_pose_error(
            ee_pos_b, ee_quat_b, self.goal_pos, self.goal_quat, rot_error_type="axis_angle"
        )

        # Normalize by action scale to get raw actions
        raw_actions = torch.zeros_like(pos_error.new_empty(pos_error.shape[0], 6))
        raw_actions[:, :3] = pos_error / self.action_scale[:3]
        raw_actions[:, 3:] = rot_error_aa / self.action_scale[3:]

        # Apply gain and clip
        arm_actions = (self.gain * raw_actions).clamp(-1.0, 1.0)

        pos_err_norm = pos_error.norm(dim=-1)
        rot_err_norm = rot_error_aa.norm(dim=-1)
        return arm_actions, pos_err_norm, rot_err_norm


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):

    # --- Config ---
    agent_cfg = sanitize_rsl_rl_cfg(agent_cfg)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # --- Create env ---
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    num_envs = env.num_envs
    device = env.device

    # --- Load RL policy ---
    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # --- Scripted policy setup ---
    # OSC action scale from the env config (matching rl_state_cfg actions)
    action_scale = torch.tensor([0.02, 0.02, 0.02, 0.02, 0.02, 0.2])

    # Goal pose in robot base frame
    goal_pos = torch.tensor(args_cli.goal_pos, dtype=torch.float32)
    r, p, y = args_cli.goal_rpy
    goal_quat = quat_from_euler_xyz(
        torch.tensor([r]), torch.tensor([p]), torch.tensor([y])
    ).squeeze(0)

    scripted_policy = ScriptedBringToGoal(
        goal_pos=goal_pos,
        goal_quat=goal_quat,
        action_scale=action_scale,
        num_envs=num_envs,
        device=device,
        gain=0.5,  # conservative: half-correction per step avoids overshoot
    )

    print(f"[INFO] Goal pose (base frame): pos={goal_pos.tolist()}, quat={goal_quat.tolist()}")

    # --- Goal frame visualization ---
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))

    # Goal in world frame for visualization (offset by env origins)
    robot = env.unwrapped.scene["robot"]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w

    # --- Total action dim (arm + gripper) ---
    num_actions = env.num_actions

    # --- Per-env state ---
    # False = scripted phase, True = RL phase
    use_rl = torch.zeros(num_envs, dtype=torch.bool, device=device)
    episode_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_return = torch.zeros(num_envs, device=device)

    completed_returns = []
    completed_lengths = []
    total_target = args_cli.num_episodes

    obs = env.get_observations()
    print(f"[INFO] Running {total_target} episodes, max {args_cli.max_steps} steps/episode.")
    print(f"[INFO] Switch thresholds: pos={args_cli.pos_threshold}m, rot={args_cli.rot_threshold}rad")
    print("-" * 90)

    while simulation_app.is_running():
        if len(completed_returns) >= total_target:
            break

        with torch.inference_mode():
            # --- Get current EE pose ---
            ee_pos_b, ee_quat_b = get_ee_pose_in_base(env)

            # --- Compute scripted actions ---
            scripted_arm, pos_err, rot_err = scripted_policy.compute(ee_pos_b, ee_quat_b)

            # --- Check switching condition ---
            reached_goal = (pos_err < args_cli.pos_threshold) & (rot_err < args_cli.rot_threshold)
            # Once switched to RL, stay on RL for the rest of the episode
            use_rl = use_rl | reached_goal

            # --- Compute RL actions ---
            rl_actions = policy(obs)

            # --- Assemble hybrid actions ---
            actions = torch.zeros(num_envs, num_actions, device=device)
            for i in range(num_envs):
                if use_rl[i]:
                    actions[i] = rl_actions[i]
                else:
                    actions[i, :6] = scripted_arm[i]
                    actions[i, 6] = 0.0  # gripper open during approach

            # --- Step ---
            obs, rewards, dones, extras = env.step(actions)
            policy_nn.reset(dones)

        episode_step += 1
        episode_return += rewards

        # --- Update visualization ---
        # Recompute root pose (may change across steps)
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        # Goal in world frame: goal is in base frame, add root offset
        from isaaclab.utils.math import combine_frame_transforms
        goal_pos_w, goal_quat_w = combine_frame_transforms(
            root_pos_w, root_quat_w,
            scripted_policy.goal_pos, scripted_policy.goal_quat,
        )
        goal_marker.visualize(goal_pos_w, goal_quat_w)

        ee_pos_w = robot.data.body_pos_w[:, robot.find_bodies("wrist_3_link")[0][0]]
        ee_quat_w = robot.data.body_quat_w[:, robot.find_bodies("wrist_3_link")[0][0]]
        ee_marker.visualize(ee_pos_w, ee_quat_w)

        # --- Episode bookkeeping ---
        timed_out = episode_step >= args_cli.max_steps
        finished = dones.bool() | timed_out

        if finished.any():
            for idx in finished.nonzero(as_tuple=False).squeeze(-1).tolist():
                if len(completed_returns) >= total_target:
                    break
                ep_num = len(completed_returns) + 1
                ep_ret = episode_return[idx].item()
                ep_len = episode_step[idx].item()
                phase = "RL" if use_rl[idx].item() else "scripted"
                reason = "done" if dones[idx].bool().item() else "max_steps"
                completed_returns.append(ep_ret)
                completed_lengths.append(ep_len)
                print(
                    f"  Ep {ep_num:3d}/{total_target} | env {idx} | "
                    f"steps: {ep_len:4d} | return: {ep_ret:8.3f} | "
                    f"ended in: {phase} | {reason}"
                )
                # Reset per-env state for next episode
                episode_step[idx] = 0
                episode_return[idx] = 0.0
                use_rl[idx] = False  # next episode starts scripted again

            force_reset = timed_out & ~dones.bool()
            if force_reset.any():
                episode_step[force_reset] = 0
                episode_return[force_reset] = 0.0
                use_rl[force_reset] = False

        # --- Periodic logging ---
        step_total = episode_step[0].item()
        if step_total % 50 == 0 and step_total > 0:
            mode = "RL" if use_rl[0].item() else "scripted"
            print(
                f"    [env 0] step {step_total:4d} | mode: {mode:8s} | "
                f"pos_err: {pos_err[0]:.4f}m | rot_err: {rot_err[0]:.3f}rad"
            )

    # --- Summary ---
    print("-" * 90)
    if completed_returns:
        returns_t = torch.tensor(completed_returns)
        lengths_t = torch.tensor(completed_lengths, dtype=torch.float)
        print(f"[RESULTS] {len(completed_returns)} episodes:")
        print(f"  Return — mean: {returns_t.mean():.3f}, std: {returns_t.std():.3f}, "
              f"min: {returns_t.min():.3f}, max: {returns_t.max():.3f}")
        print(f"  Length — mean: {lengths_t.mean():.1f}, min: {lengths_t.min():.0f}, "
              f"max: {lengths_t.max():.0f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
