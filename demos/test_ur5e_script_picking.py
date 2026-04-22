# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-scripted pick-and-lift demo (no RL).

Each episode runs a two-phase pick-and-lift sequence:

  1. **APPROACH + GRASP**: a ``BringToGoal`` proportional controller drives the EE to the
     receptive object pose with a top-down orientation (gripper parallel to the block
     faces). Once the EE is within pos/rot thresholds, the gripper closes and holds for
     a few steps to let the fingers settle on the object.

  2. **LIFT**: a second ``BringToGoal`` target at the same XY but raised in Z lifts the
     object. The gripper stays closed.

Works with any Ur5eRobotiq task (2F-85 or 2F-140) since the action space is identical:
(6) Cartesian OSC deltas + (1) binary gripper.

Usage (activate the project venv first):
    # 2F-85
    python demos/test_ur5e_script_picking.py \\
        --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \\
        --num_envs 1 --num_episodes 3 --max_steps 400 \\
        env.scene.insertive_object=cube env.scene.receptive_object=cube

    # 2F-140
    python demos/test_ur5e_script_picking.py \\
        --task OmniReset-Ur5eRobotiq2f140-RelCartesianOSC-State-Play-v0 \\
        --num_envs 1 --num_episodes 3 --max_steps 400 \\
        env.scene.insertive_object=cube env.scene.receptive_object=cube
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR5e pure-scripted pick-and-lift demo.")
parser.add_argument("--task", type=str,
                    default="OmniReset-Ur5eRobotiq2f140-RelCartesianOSC-State-Play-v0",
                    help="Registered gym task ID.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--num_episodes", type=int, default=3, help="Total number of episodes to run.")
parser.add_argument("--max_steps", type=int, default=400, help="Max policy steps per episode.")
parser.add_argument("--pos_threshold", type=float, default=0.05,
                    help="Position error (m) threshold to consider a goal reached.")
parser.add_argument("--rot_threshold", type=float, default=0.1,
                    help="Orientation error (rad) threshold to consider a goal reached.")
parser.add_argument("--max_approach_steps", type=int, default=150,
                    help="Force transition from approach to pre-grasp after this many steps, "
                         "even if thresholds not met. Handles cases where the arm saturates "
                         "just above the threshold.")
parser.add_argument("--max_pre_grasp_steps", type=int, default=80,
                    help="Force transition from pre-grasp to close after this many steps.")
parser.add_argument("--approach_wrist_offset", type=float, default=0.25,
                    help="Wrist_3 Z hover offset above cube for initial approach (m). "
                         "Higher than grasp_wrist_offset to clear the object.")
parser.add_argument("--grasp_wrist_offset", type=float, default=0.15,
                    help="Wrist_3 Z offset above cube center at pre-grasp/grasp (m). "
                         "Roughly the gripper TCP length so fingers straddle the object.")
parser.add_argument("--lift_height", type=float, default=0.15,
                    help="Distance (m) to lift the object above the grasp pose.")
parser.add_argument("--close_hold_steps", type=int, default=30,
                    help="Number of steps to hold position while the gripper closes.")
parser.add_argument("--control_gain", type=float, default=2.0,
                    help="Proportional gain for the BringToGoal controller.")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import functools
import importlib.metadata as metadata
import inspect

import torch

print = functools.partial(print, flush=True)  # type: ignore[assignment]

import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    quat_from_euler_xyz,
    subtract_frame_transforms,
)
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_config

TOP_DOWN_RPY = torch.tensor([3.14159, 0.0, 0.0])

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def sanitize_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg) -> RslRlBaseRunnerCfg:
    """Strip algorithm-config keys the installed rsl-rl version doesn't accept.

    Not strictly needed for the scripted demo (we don't build a runner), but kept
    so the ``rsl_rl_cfg_entry_point`` hydra decorator still resolves cleanly.
    """
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
    """Return the EE (wrist_3_link) pose expressed in the robot root frame: (pos_b, quat_b)."""
    robot = env.unwrapped.scene["robot"]
    ee_idx = robot.find_bodies("wrist_3_link")[0][0]
    ee_pos_w = robot.data.body_pos_w[:, ee_idx]
    ee_quat_w = robot.data.body_quat_w[:, ee_idx]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    return subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)


def get_object_pos_in_base(env, asset_name: str):
    """Return the object's position expressed in the robot root frame: (pos_b,)."""
    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene[asset_name]
    obj_pos_w = obj.data.root_pos_w
    obj_quat_w = obj.data.root_quat_w
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, obj_pos_w, obj_quat_w)
    return pos_b


class BringToGoal:
    """Proportional controller in OSC action space that drives the EE toward a goal pose.

    Outputs a 6-dim action (scaled delta position + axis-angle), clipped to [-1, 1].
    The goal is given in the robot base (root) frame and can be updated per step via
    :meth:`set_goal`.
    """

    def __init__(self, action_scale: torch.Tensor, num_envs: int, device, gain: float = 0.5):
        self.action_scale = action_scale.to(device)
        self.device = device
        self.num_envs = num_envs
        self.gain = gain
        self.goal_pos = torch.zeros(num_envs, 3, device=device)
        self.goal_quat = torch.zeros(num_envs, 4, device=device)
        self.goal_quat[:, 0] = 1.0  # identity

    def set_goal(self, goal_pos: torch.Tensor, goal_quat: torch.Tensor):
        """Update the goal; broadcasts (3,)/(4,) tensors to all envs."""
        if goal_pos.dim() == 1:
            goal_pos = goal_pos.unsqueeze(0).expand(self.num_envs, -1)
        if goal_quat.dim() == 1:
            goal_quat = goal_quat.unsqueeze(0).expand(self.num_envs, -1)
        self.goal_pos = goal_pos.to(self.device)
        self.goal_quat = goal_quat.to(self.device)

    def compute(self, ee_pos_b: torch.Tensor, ee_quat_b: torch.Tensor):
        """Return (arm_actions (N,6), pos_err (N,), rot_err (N,))."""
        pos_err, rot_err_aa = compute_pose_error(
            ee_pos_b, ee_quat_b, self.goal_pos, self.goal_quat, rot_error_type="axis_angle"
        )
        raw = pos_err.new_zeros(pos_err.shape[0], 6)
        raw[:, :3] = pos_err / self.action_scale[:3]
        raw[:, 3:] = rot_err_aa / self.action_scale[3:]
        arm_actions = (self.gain * raw).clamp(-1.0, 1.0)
        return arm_actions, pos_err.norm(dim=-1), rot_err_aa.norm(dim=-1)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

# Phase encoding
PHASE_APPROACH = 0   # hover above object, gripper open
PHASE_PRE_GRASP = 1  # descend to grasp height, gripper still open
PHASE_CLOSE = 2      # hold pre-grasp pose, close gripper
PHASE_LIFT = 3       # rise to hover height, gripper closed
PHASE_NAMES = {
    PHASE_APPROACH: "approach",
    PHASE_PRE_GRASP: "pre_grasp",
    PHASE_CLOSE: "close",
    PHASE_LIFT: "lift",
}


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):

    agent_cfg = sanitize_rsl_rl_cfg(agent_cfg)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    num_envs = env.num_envs
    device = env.device
    num_actions = env.num_actions

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Action space: {num_actions}-dim (6 Cartesian OSC + 1 binary gripper)")

    # OSC scale from the env's action cfg. Hardcoded to match ``UR5E_ROBOTIQ_*_RELATIVE_OSC``
    # in actions.py (the Play/Eval cfg uses the soft Kp variant with this scale).
    action_scale = torch.tensor([0.02, 0.02, 0.02, 0.02, 0.02, 0.2])

    # Top-down grasp orientation in the robot base frame: wrist X-axis points down,
    # so the gripper fingers extend downward and close horizontally, parallel to
    # the block's vertical faces. Matches the default-pose convention UR5e uses
    # when the wrist is flipped upside-down (roll=pi).
    top_down_quat = quat_from_euler_xyz(
        TOP_DOWN_RPY[0:1], TOP_DOWN_RPY[1:2], TOP_DOWN_RPY[2:3]
    ).squeeze(0)

    controller = BringToGoal(
        action_scale=action_scale, num_envs=num_envs, device=device, gain=args_cli.control_gain
    )

    # Visualization: current EE frame + active goal frame.
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))

    robot = env.unwrapped.scene["robot"]

    # Per-env state
    phase = torch.zeros(num_envs, dtype=torch.long, device=device)  # phase id, see PHASE_*
    close_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    approach_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    pre_grasp_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_return = torch.zeros(num_envs, device=device)
    # Grasp position captured at the moment we transition into CLOSE; LIFT uses same XY.
    grasp_pos = torch.zeros(num_envs, 3, device=device)
    # Approach (hover) position captured at start of episode; LIFT returns to this height.
    approach_pos = torch.zeros(num_envs, 3, device=device)

    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    completed_final_phase: list[str] = []
    total_target = args_cli.num_episodes

    obs = env.get_observations()  # noqa: F841 — env needs the first call, obs unused by scripted policy
    print(f"[INFO] Running {total_target} episodes, max {args_cli.max_steps} steps/episode.")
    print(f"[INFO] Thresholds: pos<{args_cli.pos_threshold}m, rot<{args_cli.rot_threshold}rad.")
    print(
        f"[INFO] Wrist offsets — approach (hover): {args_cli.approach_wrist_offset}m, "
        f"pre-grasp/grasp: {args_cli.grasp_wrist_offset}m, lift height: {args_cli.lift_height}m."
    )
    print("-" * 90)

    while simulation_app.is_running():
        if len(completed_returns) >= total_target:
            break

        with torch.inference_mode():
            ee_pos_b, ee_quat_b = get_ee_pose_in_base(env)
            obj_pos_b = get_object_pos_in_base(env, "insertive_object")

            # --- Compute goal per env based on current phase ---
            goal_pos = torch.zeros(num_envs, 3, device=device)
            for i in range(num_envs):
                ph = phase[i].item()
                if ph == PHASE_APPROACH:
                    # Hover above the cube with clearance.
                    goal_pos[i] = obj_pos_b[i].clone()
                    goal_pos[i, 2] += args_cli.approach_wrist_offset
                elif ph == PHASE_PRE_GRASP:
                    # Descend to grasp height: fingers straddle the cube, gripper still open.
                    goal_pos[i] = obj_pos_b[i].clone()
                    goal_pos[i, 2] += args_cli.grasp_wrist_offset
                elif ph == PHASE_CLOSE:
                    # Hold the latched grasp pose while the fingers close.
                    goal_pos[i] = grasp_pos[i]
                else:  # PHASE_LIFT
                    # Return to the hover height over the (now hopefully grasped) object.
                    goal_pos[i] = grasp_pos[i].clone()
                    goal_pos[i, 2] += args_cli.lift_height

            goal_quat = top_down_quat.to(device).unsqueeze(0).expand(num_envs, -1)
            controller.set_goal(goal_pos, goal_quat)

            arm_actions, pos_err, rot_err = controller.compute(ee_pos_b, ee_quat_b)

            # --- Compose gripper action ---
            # BinaryJointAction convention: actions >= 0 → OPEN, actions < 0 → CLOSE.
            gripper_actions = torch.zeros(num_envs, 1, device=device)
            for i in range(num_envs):
                ph = phase[i].item()
                if ph == PHASE_APPROACH or ph == PHASE_PRE_GRASP:
                    gripper_actions[i] = 1.0  # open during approach + descent
                else:
                    gripper_actions[i] = -1.0  # closed through CLOSE and LIFT

            # Full action tensor: [6 arm, 1 gripper]
            actions = torch.cat([arm_actions, gripper_actions], dim=-1)

            obs, rewards, dones, extras = env.step(actions)

            # --- Advance phase for envs that hit their conditions ---
            reached_goal = (pos_err < args_cli.pos_threshold) & (rot_err < args_cli.rot_threshold)

            for i in range(num_envs):
                ph = phase[i].item()
                if ph == PHASE_APPROACH:
                    approach_steps[i] += 1
                    if reached_goal[i].item() or approach_steps[i].item() >= args_cli.max_approach_steps:
                        # Latch the hover position; lift target will be same XY at a higher Z.
                        approach_pos[i] = goal_pos[i].clone()
                        phase[i] = PHASE_PRE_GRASP
                        pre_grasp_steps[i] = 0
                elif ph == PHASE_PRE_GRASP:
                    pre_grasp_steps[i] += 1
                    if reached_goal[i].item() or pre_grasp_steps[i].item() >= args_cli.max_pre_grasp_steps:
                        # Latch the grasp pose; CLOSE holds here, LIFT rises from here.
                        grasp_pos[i] = goal_pos[i].clone()
                        phase[i] = PHASE_CLOSE
                        close_steps[i] = 0
                elif ph == PHASE_CLOSE:
                    close_steps[i] += 1
                    if close_steps[i].item() >= args_cli.close_hold_steps:
                        phase[i] = PHASE_LIFT

        episode_step += 1
        episode_return += rewards

        # --- Visualization markers (in world frame) ---
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        goal_pos_w, goal_quat_w = combine_frame_transforms(
            root_pos_w, root_quat_w, controller.goal_pos, controller.goal_quat
        )
        goal_marker.visualize(goal_pos_w, goal_quat_w)
        ee_idx = robot.find_bodies("wrist_3_link")[0][0]
        ee_marker.visualize(robot.data.body_pos_w[:, ee_idx], robot.data.body_quat_w[:, ee_idx])

        # --- Periodic logging ---
        step0 = episode_step[0].item()
        if step0 % 25 == 0 and step0 > 0:
            print(
                f"    [env 0] step {step0:4d} | phase: {PHASE_NAMES[phase[0].item()]:8s} | "
                f"pos_err: {pos_err[0]:.4f}m | rot_err: {rot_err[0]:.3f}rad"
            )

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
                final_phase = PHASE_NAMES[phase[idx].item()]
                reason = "done" if dones[idx].bool().item() else "max_steps"
                completed_returns.append(ep_ret)
                completed_lengths.append(ep_len)
                completed_final_phase.append(final_phase)
                print(
                    f"  Ep {ep_num:3d}/{total_target} | env {idx} | "
                    f"steps: {ep_len:4d} | return: {ep_ret:8.3f} | "
                    f"final phase: {final_phase:8s} | {reason}"
                )
                # Reset per-env state for next episode.
                episode_step[idx] = 0
                episode_return[idx] = 0.0
                phase[idx] = PHASE_APPROACH
                close_steps[idx] = 0
                approach_steps[idx] = 0
                pre_grasp_steps[idx] = 0

            force_reset = timed_out & ~dones.bool()
            if force_reset.any():
                episode_step[force_reset] = 0
                episode_return[force_reset] = 0.0
                phase[force_reset] = PHASE_APPROACH
                close_steps[force_reset] = 0
                approach_steps[force_reset] = 0
                pre_grasp_steps[force_reset] = 0

    # --- Summary ---
    print("-" * 90)
    if completed_returns:
        returns_t = torch.tensor(completed_returns)
        lengths_t = torch.tensor(completed_lengths, dtype=torch.float)
        lifted = sum(p == "lift" for p in completed_final_phase)
        closed = sum(p == "close" for p in completed_final_phase)
        pre_grasping = sum(p == "pre_grasp" for p in completed_final_phase)
        approaching = sum(p == "approach" for p in completed_final_phase)
        print(f"[RESULTS] {len(completed_returns)} episodes:")
        print(f"  Return — mean: {returns_t.mean():.3f}, min: {returns_t.min():.3f}, max: {returns_t.max():.3f}")
        print(f"  Length — mean: {lengths_t.mean():.1f}, min: {lengths_t.min():.0f}, max: {lengths_t.max():.0f}")
        print(
            f"  Final phase tally — approach: {approaching}, pre_grasp: {pre_grasping}, "
            f"close: {closed}, lift: {lifted}"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
