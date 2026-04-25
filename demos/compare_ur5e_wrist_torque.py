# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare UR5e OSC wrist torques across two robot USD/task configs.

The script creates each task with one environment, disables reset/randomization
events, writes the same six UR5e arm qpos/qvel, and drives wrist_3_link toward
the same base-frame goal pose.  For each policy step it prints the OSC torque
before clamp, after clamp, and the actuator's applied torque after the env step.

Example:
    python demos/compare_ur5e_wrist_torque.py --headless --steps 40
"""

from __future__ import annotations

import argparse
import functools
from dataclasses import dataclass

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Compare UR5e wrist OSC torques for two task/USD configs.")
parser.add_argument(
    "--task_a",
    type=str,
    default="OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0",
    help="First registered gym task ID. Defaults to the working 2F-85 full-gripper task.",
)
parser.add_argument(
    "--task_b",
    type=str,
    default="OmniReset-Ur5eRobotiq2f140-RelCartesianOSC-State-Play-v0",
    help="Second registered gym task ID. Defaults to the 2F-140 arm-only debug task.",
)
parser.add_argument("--steps", type=int, default=40, help="Number of policy steps to run per task.")
parser.add_argument("--print_every", type=int, default=1, help="Print every N policy steps.")
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help=(
        "Episode length to use for each diagnostic case. By default this is derived from --steps "
        "so long runs do not hit the task timeout and auto-reset mid-test."
    ),
)
parser.add_argument(
    "--start_qpos",
    type=float,
    nargs=6,
    default=[0.0, -1.5708, 1.5708, -1.5708, -1.5708, -1.5708],
    metavar=("SH_PAN", "SH_LIFT", "ELBOW", "WR1", "WR2", "WR3"),
    help="Shared start qpos for the six UR5e arm joints.",
)
parser.add_argument(
    "--goal_mode",
    choices=("delta_axis_angle", "absolute_rpy"),
    default="delta_axis_angle",
    help=(
        "How to construct the shared goal from task A's start pose. "
        "'delta_axis_angle' applies --goal_axis_angle_delta to the start wrist orientation; "
        "'absolute_rpy' uses --goal_rpy directly."
    ),
)
parser.add_argument(
    "--goal_axis_angle_delta",
    type=float,
    nargs=3,
    default=[0.35, -0.35, 0.25],
    metavar=("RX", "RY", "RZ"),
    help="Orientation delta from task A's start wrist orientation, in axis-angle radians.",
)
parser.add_argument(
    "--goal_rpy",
    type=float,
    nargs=3,
    default=[3.14159, 0.0, 0.0],
    metavar=("ROLL", "PITCH", "YAW"),
    help="Shared absolute base-frame wrist goal orientation, in radians.",
)
parser.add_argument(
    "--goal_pos_delta",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.0],
    metavar=("DX", "DY", "DZ"),
    help="Goal position delta from task A's start wrist position, in robot base frame.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402

from isaaclab.utils.math import (  # noqa: E402
    axis_angle_from_quat,
    compute_pose_error,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from uwlab_assets.robots.ur5e_robotiq_gripper.kinematics import compute_jacobian_analytical  # noqa: E402
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.actions.task_space_actions import (  # noqa: E402
    RelCartesianOSCAction,
)

print = functools.partial(print, flush=True)  # type: ignore[assignment]


ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
WRIST_JOINT_NAMES = ["wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

EVENTS_TO_DISABLE = [
    "robot_material",
    "insertive_object_material",
    "receptive_object_material",
    "table_material",
    "randomize_robot_mass",
    "randomize_insertive_object_mass",
    "randomize_receptive_object_mass",
    "randomize_table_mass",
    "randomize_gripper_actuator_parameters",
    "randomize_arm_sysid",
    "randomize_osc_gains",
    "reset_from_reset_states",
]


@dataclass
class TorqueDebug:
    pos_err_norm: float
    rot_err_norm: float
    tau_raw_wrist: list[float]
    tau_clamped_wrist: list[float]
    saturated_wrist: list[bool]


def _disable_random_events(env_cfg) -> None:
    if not hasattr(env_cfg, "events") or env_cfg.events is None:
        return
    for name in EVENTS_TO_DISABLE:
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)


def _make_env(task_name: str, device: str):
    omni.usd.get_context().new_stage()
    env_cfg = parse_env_cfg(task_name, device=device, num_envs=1)
    _disable_random_events(env_cfg)
    policy_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    min_episode_length_s = policy_dt * (args_cli.steps + 2)
    env_cfg.episode_length_s = (
        float(args_cli.episode_length_s)
        if args_cli.episode_length_s is not None
        else max(float(env_cfg.episode_length_s), min_episode_length_s)
    )
    env = gym.make(task_name, cfg=env_cfg)
    env.reset()
    return env


def _write_start_state(env, start_qpos: torch.Tensor) -> list[int]:
    robot = env.unwrapped.scene["robot"]
    arm_joint_ids, arm_joint_names = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
    if arm_joint_names != ARM_JOINT_NAMES:
        raise RuntimeError(f"Unexpected arm joint order: {arm_joint_names}")

    qpos = robot.data.default_joint_pos.clone()
    qvel = torch.zeros_like(robot.data.default_joint_vel)
    qpos[:, arm_joint_ids] = start_qpos.unsqueeze(0)

    robot.write_joint_state_to_sim(qpos, qvel)
    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.step()
    env.unwrapped.scene.update(env.unwrapped.sim.get_physics_dt())
    return arm_joint_ids


def _get_ee_pose_in_base(env) -> tuple[torch.Tensor, torch.Tensor]:
    robot = env.unwrapped.scene["robot"]
    ee_idx = robot.find_bodies("wrist_3_link")[0][0]
    return subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        robot.data.body_pos_w[:, ee_idx],
        robot.data.body_quat_w[:, ee_idx],
    )


def _get_arm_action_to_goal(
    arm_term: RelCartesianOSCAction,
    ee_pos_b: torch.Tensor,
    ee_quat_b: torch.Tensor,
    goal_pos_b: torch.Tensor,
    goal_quat_b: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    pos_err, rot_err_aa = compute_pose_error(ee_pos_b, ee_quat_b, goal_pos_b, goal_quat_b, rot_error_type="axis_angle")
    raw_action = torch.zeros(ee_pos_b.shape[0], 6, device=ee_pos_b.device)
    raw_action[:, :3] = pos_err / arm_term._scale[:3]
    raw_action[:, 3:] = rot_err_aa / arm_term._scale[3:]
    return raw_action.clamp(-1.0, 1.0), float(pos_err.norm(dim=-1)[0]), float(rot_err_aa.norm(dim=-1)[0])


def _compute_torque_debug(
    arm_term: RelCartesianOSCAction,
    arm_action: torch.Tensor,
    goal_pos_err: float,
    goal_rot_err: float,
) -> TorqueDebug:
    arm_term.process_actions(arm_action)

    ee_pos_b, ee_quat_b = arm_term._get_ee_pose_root_frame()
    joint_pos = arm_term._asset.data.joint_pos[:, arm_term._joint_ids]
    joint_vel = arm_term._asset.data.joint_vel[:, arm_term._joint_ids]

    jacobian = compute_jacobian_analytical(joint_pos, device=str(arm_term.device))
    ee_vel = torch.bmm(jacobian, joint_vel.unsqueeze(-1)).squeeze(-1)

    pos_error = arm_term._ee_pos_des - ee_pos_b
    quat_error = quat_mul(arm_term._ee_quat_des, quat_inv(ee_quat_b))
    axis_angle_error = axis_angle_from_quat(quat_error)
    pose_error = torch.cat([pos_error, axis_angle_error], dim=-1)

    tau_raw, tau_clamped = arm_term._compute_joint_torques(jacobian, pose_error, ee_vel)

    wrist_rel_ids = [arm_term._joint_names.index(name) for name in WRIST_JOINT_NAMES]
    raw = tau_raw[0, wrist_rel_ids].detach().cpu()
    clamped = tau_clamped[0, wrist_rel_ids].detach().cpu()
    saturated = (raw - clamped).abs() > 1e-5

    return TorqueDebug(
        pos_err_norm=goal_pos_err,
        rot_err_norm=goal_rot_err,
        tau_raw_wrist=[float(x) for x in raw],
        tau_clamped_wrist=[float(x) for x in clamped],
        saturated_wrist=[bool(x) for x in saturated],
    )


def _fmt_float_list(values: list[float]) -> str:
    return "[" + ", ".join(f"{value:+10.5f}" for value in values) + "]"


def _fmt_bool_list(values: list[bool]) -> str:
    return "[" + ", ".join("Y" if value else "." for value in values) + "]"


def _quat_from_axis_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    axis_angle = axis_angle.reshape(1, 3)
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    safe_angle = torch.where(angle > 1e-6, angle, torch.ones_like(angle))
    axis = axis_angle / safe_angle
    axis = torch.where(angle > 1e-6, axis, torch.zeros_like(axis))
    half_angle = 0.5 * angle
    return torch.cat([torch.cos(half_angle), axis * torch.sin(half_angle)], dim=-1)


def _run_case(task_name: str, device: str, start_qpos: torch.Tensor, goal_pose=None):
    env = _make_env(task_name, device)
    robot = env.unwrapped.scene["robot"]
    arm_joint_ids = _write_start_state(env, start_qpos)
    arm_term = env.unwrapped.action_manager._terms.get("arm")
    if not isinstance(arm_term, RelCartesianOSCAction):
        raise RuntimeError(f"Task {task_name} has no RelCartesianOSCAction named 'arm'.")

    start_pos_b, start_quat_b = _get_ee_pose_in_base(env)
    if goal_pose is None:
        goal_pos_b = start_pos_b + torch.tensor(args_cli.goal_pos_delta, device=env.unwrapped.device).unsqueeze(0)
        if args_cli.goal_mode == "absolute_rpy":
            rpy = torch.tensor(args_cli.goal_rpy, device=env.unwrapped.device)
            goal_quat_b = quat_from_euler_xyz(rpy[0:1], rpy[1:2], rpy[2:3])
        else:
            delta = torch.tensor(args_cli.goal_axis_angle_delta, device=env.unwrapped.device)
            goal_quat_b = quat_mul(_quat_from_axis_angle(delta), start_quat_b)
    else:
        goal_pos_b, goal_quat_b = goal_pose
        goal_pos_b = goal_pos_b.to(env.unwrapped.device)
        goal_quat_b = goal_quat_b.to(env.unwrapped.device)

    print("")
    print(f"[CASE] {task_name}")
    print(f"  USD: {env.unwrapped.cfg.scene.robot.spawn.usd_path}")
    print(f"  episode_length_s: {env.unwrapped.cfg.episode_length_s}")
    action_dim = env.unwrapped.action_manager.total_action_dim
    print(f"  action_dim: {action_dim}")
    print(f"  arm joints: {arm_term._joint_names}")
    print(f"  kp: {[float(x) for x in arm_term._kp[0].detach().cpu()]}")
    print(f"  torque_limit: {[float(x) for x in arm_term._torque_max.detach().cpu()]}")
    print(f"  use_task_space_inertia: {arm_term._use_task_space_inertia}")
    print(f"  start_ee_pos_b: {[round(float(x), 5) for x in start_pos_b[0].detach().cpu()]}")
    print(f"  goal_pos_b: {[round(float(x), 5) for x in goal_pos_b[0].detach().cpu()]}")
    print(f"  goal_quat_b(wxyz): {[round(float(x), 5) for x in goal_quat_b[0].detach().cpu()]}")
    print(
        "  step | pos_err | rot_err | raw wrist tau [w1,w2,w3] | "
        "clamped wrist tau [w1,w2,w3] | sat | applied after step"
    )

    has_gripper_action = action_dim == 7
    for step in range(args_cli.steps):
        ee_pos_b, ee_quat_b = _get_ee_pose_in_base(env)
        arm_action, pos_err, rot_err = _get_arm_action_to_goal(arm_term, ee_pos_b, ee_quat_b, goal_pos_b, goal_quat_b)
        torque_debug = _compute_torque_debug(arm_term, arm_action, pos_err, rot_err)

        if has_gripper_action:
            actions = torch.cat([arm_action, torch.ones(1, 1, device=arm_action.device)], dim=-1)
        else:
            actions = arm_action

        env.step(actions)
        applied = robot.data.applied_torque[0, arm_joint_ids].detach().cpu()
        applied_wrist = [float(applied[ARM_JOINT_NAMES.index(name)]) for name in WRIST_JOINT_NAMES]

        if step % args_cli.print_every == 0:
            print(
                f"  {step:4d} | {torque_debug.pos_err_norm:7.4f} | {torque_debug.rot_err_norm:7.4f} | "
                f"{_fmt_float_list(torque_debug.tau_raw_wrist)} | "
                f"{_fmt_float_list(torque_debug.tau_clamped_wrist)} | "
                f"{_fmt_bool_list(torque_debug.saturated_wrist)} | {_fmt_float_list(applied_wrist)}"
            )

    final_pos_b, final_quat_b = _get_ee_pose_in_base(env)
    _, final_rot_err_aa = compute_pose_error(final_pos_b, final_quat_b, goal_pos_b, goal_quat_b, rot_error_type="axis_angle")
    print(f"  final_rot_err: {float(final_rot_err_aa.norm(dim=-1)[0]):.6f} rad")

    goal_pose = (goal_pos_b.detach().cpu(), goal_quat_b.detach().cpu())
    env.close()
    return goal_pose


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    start_qpos = torch.tensor(args_cli.start_qpos, device=device, dtype=torch.float32)

    print("[INFO] Identical start qpos:", [float(x) for x in start_qpos.detach().cpu()])
    print("[INFO] Randomization/reset-state events disabled for this diagnostic.")
    goal_pose = _run_case(args_cli.task_a, device, start_qpos, goal_pose=None)
    _run_case(args_cli.task_b, device, start_qpos, goal_pose=goal_pose)


if __name__ == "__main__":
    main()
    simulation_app.close()
