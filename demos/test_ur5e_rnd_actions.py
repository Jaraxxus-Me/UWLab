# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone demo: UR5e + Robotiq 2F-85 with rectangle/wall scene, sending random
joint effort commands to the arm and binary gripper commands.

The arm actuator uses stiffness=0 / damping=0 (torque-controlled), matching the
OmniReset task setup. We drive it with a joint-space PD controller that tracks
randomly sampled target positions — converting position offsets into effort targets.

Usage (activate the project venv first):
    # GUI mode
    python demos/test_ur5e_rnd_actions.py

    # Headless
    python demos/test_ur5e_rnd_actions.py --headless

    # More envs / steps
    python demos/test_ur5e_rnd_actions.py --num_envs 4 --num_steps 2000
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR5e random-action demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--num_steps", type=int, default=10000, help="Number of simulation steps to run.")
parser.add_argument("--action_scale", type=float, default=0.1, help="Scale of random joint offsets (rad).")
parser.add_argument("--control_freq", type=int, default=12, help="Apply new random action every N sim steps (decimation).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import functools

import torch

# Ensure prints are not buffered (critical for headless runs where stdout is captured)
print = functools.partial(print, flush=True)  # type: ignore[assignment]

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
from uwlab_assets.robots.ur5e_robotiq_gripper import IMPLICIT_UR5E_ROBOTIQ_2F85

# ------------------------------------------------------------------
# Scene — rectangle + wall (from OmniReset variants)
# ------------------------------------------------------------------

RECTANGLE_USD = f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Rectangle/rectangle.usd"
WALL_USD = f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Wall/wall.usd"
TABLE_USD = f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention/pat_vention.usd"
SUPPORT_USD = f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention2/Ur5MetalSupport/ur5plate.usd"


@configclass
class Ur5eRectangleWallSceneCfg(InteractiveSceneCfg):

    robot = IMPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

    insertive_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=RECTANGLE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.1, 0.02)),
    )

    receptive_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=WALL_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.1, 0.0)),
    )

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, -0.881), rot=(0.707, 0.0, 0.0, -0.707)),
    )

    support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Support",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SUPPORT_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.013)),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.868)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ------------------------------------------------------------------
# Joint-space PD gains for effort control (arm actuator has stiffness=0)
# ------------------------------------------------------------------

ARM_KP = torch.tensor([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
ARM_KD = torch.tensor([20.0, 20.0, 20.0, 5.0, 5.0, 5.0])
NUM_ARM_JOINTS = 6
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.785398


# ------------------------------------------------------------------
# Simulation loop
# ------------------------------------------------------------------

def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    insertive = scene["insertive_object"]
    receptive = scene["receptive_object"]
    sim_dt = sim.get_physics_dt()
    num_envs = scene.num_envs
    device = sim.device

    arm_kp = ARM_KP.to(device)
    arm_kd = ARM_KD.to(device)
    num_joints = robot.data.joint_pos.shape[1]

    # Target joint positions for the arm — start from default
    desired_arm_pos = robot.data.default_joint_pos[:, :NUM_ARM_JOINTS].clone()

    for step in range(args_cli.num_steps):
        if not simulation_app.is_running():
            break

        # --- Reset every 500 steps ---
        if step % 500 == 0:
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            )

            ins_state = insertive.data.default_root_state.clone()
            ins_state[:, :3] += scene.env_origins
            insertive.write_root_pose_to_sim(ins_state[:, :7])
            insertive.write_root_velocity_to_sim(ins_state[:, 7:])

            rec_state = receptive.data.default_root_state.clone()
            rec_state[:, :3] += scene.env_origins
            receptive.write_root_pose_to_sim(rec_state[:, :7])

            scene.reset()
            desired_arm_pos = robot.data.default_joint_pos[:, :NUM_ARM_JOINTS].clone()
            print(f"[INFO] Step {step}: scene reset.")

        # --- Sample new random target every control_freq steps ---
        if step % args_cli.control_freq == 0:
            arm_delta = args_cli.action_scale * torch.randn(num_envs, NUM_ARM_JOINTS, device=device)
            desired_arm_pos = robot.data.default_joint_pos[:, :NUM_ARM_JOINTS] + arm_delta

        # --- Compute PD torques for arm ---
        pos_error = desired_arm_pos - robot.data.joint_pos[:, :NUM_ARM_JOINTS]
        vel_error = -robot.data.joint_vel[:, :NUM_ARM_JOINTS]
        arm_torques = arm_kp * pos_error + arm_kd * vel_error

        efforts = torch.zeros(num_envs, num_joints, device=device)
        efforts[:, :NUM_ARM_JOINTS] = arm_torques
        robot.set_joint_effort_target(efforts)

        # --- Gripper: binary open/close via position target (has stiffness=17) ---
        if step % args_cli.control_freq == 0:
            gripper_cmd = torch.where(
                torch.rand(num_envs, 1, device=device) > 0.5,
                torch.full((num_envs, 1), GRIPPER_CLOSED, device=device),
                torch.full((num_envs, 1), GRIPPER_OPEN, device=device),
            )
            gripper_target = robot.data.default_joint_pos.clone()
            gripper_target[:, NUM_ARM_JOINTS] = gripper_cmd.squeeze(-1)
            robot.set_joint_position_target(gripper_target)

        # --- Write and step ---
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        # --- Print state periodically ---
        if step % 100 == 0:
            cur_pos = robot.data.joint_pos[0, :NUM_ARM_JOINTS]
            cur_vel = robot.data.joint_vel[0, :NUM_ARM_JOINTS]
            gripper_pos = robot.data.joint_pos[0, NUM_ARM_JOINTS]
            ins_pos = insertive.data.root_pos_w[0]
            print(
                f"  Step {step:5d} | "
                f"arm pos: [{', '.join(f'{v:.3f}' for v in cur_pos.tolist())}] | "
                f"arm vel: [{', '.join(f'{v:.2f}' for v in cur_vel.tolist())}] | "
                f"gripper: {gripper_pos:.3f} | "
                f"rect: [{ins_pos[0]:.3f}, {ins_pos[1]:.3f}, {ins_pos[2]:.3f}]"
            )


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.5, 1.5, 1.0), target=(0.4, 0.0, 0.0))

    scene_cfg = Ur5eRectangleWallSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.5)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] Setup complete. Starting random-action loop...")
    print(f"[INFO] {args_cli.num_steps} steps, {args_cli.num_envs} env(s), "
          f"action_scale={args_cli.action_scale}, control_freq={args_cli.control_freq}")
    run_simulator(sim, scene)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
