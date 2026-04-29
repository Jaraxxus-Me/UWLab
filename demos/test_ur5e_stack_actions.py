# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn the UR5e + Robotiq 2F-140 block/box scene in the real workspace.

This is a static smoke test: it loads the remote block/box USDs, places the
workspace at -X in the robot base frame, resets the robot to its configured
default joint positions, and steps physics without sending arm actions.

Usage:
    python demos/test_ur5e_stack_actions.py --headless
    python demos/test_ur5e_rnd_actions.py --headless
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Static UR5e 2F-140 block/box workspace smoke test.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--num_steps", type=int, default=50, help="Number of simulation steps to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import functools

import torch

print = functools.partial(print, flush=True)  # type: ignore[assignment]

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets import UWLAB_ASSETS_EXT_DIR, UWLAB_CLOUD_ASSETS_DIR, custom_cloud_path
from uwlab_assets.robots.ur5e_robotiq_gripper import IMPLICIT_UR5E_ROBOTIQ_2F140

CORNERED_BLOCK_ASSET_DIR = custom_cloud_path(
    "Props/Custom/CorneredBlock",
    f"{UWLAB_ASSETS_EXT_DIR}/uwlab_assets/cornered_block",
)

BLOCK_USD = f"{CORNERED_BLOCK_ASSET_DIR}/block/block.usd"
BOX_USD = f"{CORNERED_BLOCK_ASSET_DIR}/box/box.usd"
TABLE_USD = f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention/pat_vention.usd"
SUPPORT_USD = f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention2/Ur5MetalSupport/ur5plate.usd"


@configclass
class Ur5eBlockBoxStaticSceneCfg(InteractiveSceneCfg):
    """UR5e 2F-140, cornered block/box, and real-workspace table/support."""

    robot = IMPLICIT_UR5E_ROBOTIQ_2F140.replace(prim_path="{ENV_REGEX_NS}/Robot")

    insertive_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BLOCK_USD,
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.49, -0.08, 0.027), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    receptive_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=BOX_USD,
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.0, 0.0215), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.4, 0.0, -0.881), rot=(0.707, 0.0, 0.0, 0.707)),
    )

    ur5_metal_support = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/UR5MetalSupport",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SUPPORT_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.013), rot=(1.0, 0.0, 0.0, 0.0)),
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


def reset_to_defaults(scene: InteractiveScene) -> None:
    """Write configured default root and joint states into the simulator."""
    robot = scene["robot"]

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())

    for name in ("insertive_object", "receptive_object", "table", "ur5_metal_support"):
        asset = scene[name]
        asset_state = asset.data.default_root_state.clone()
        asset_state[:, :3] += scene.env_origins
        asset.write_root_pose_to_sim(asset_state[:, :7])
        asset.write_root_velocity_to_sim(asset_state[:, 7:])

    scene.reset()


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    insertive = scene["insertive_object"]
    receptive = scene["receptive_object"]
    sim_dt = sim.get_physics_dt()
    device = sim.device

    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    zero_efforts = torch.zeros(scene.num_envs, robot.data.joint_pos.shape[1], device=device)

    reset_to_defaults(scene)
    print("[INFO] Scene reset to configured default joint positions.")

    for step in range(args_cli.num_steps):
        if not simulation_app.is_running():
            break

        robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
        robot.set_joint_effort_target(zero_efforts)
        robot.set_joint_position_target(default_joint_pos)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)

        if step == 0 or (step + 1) % 10 == 0 or step + 1 == args_cli.num_steps:
            arm_pos = robot.data.joint_pos[0, :6]
            block_pos = insertive.data.root_pos_w[0]
            box_pos = receptive.data.root_pos_w[0]
            print(
                f"  Step {step + 1:3d}/{args_cli.num_steps} | "
                f"arm: [{', '.join(f'{v:.3f}' for v in arm_pos.tolist())}] | "
                f"block: [{block_pos[0]:.3f}, {block_pos[1]:.3f}, {block_pos[2]:.3f}] | "
                f"box: [{box_pos[0]:.3f}, {box_pos[1]:.3f}, {box_pos[2]:.3f}]"
            )


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(-1.3, 1.2, 0.9), target=(-0.5, 0.0, 0.05))

    scene_cfg = Ur5eBlockBoxStaticSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.5)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] Setup complete. Running static block/box workspace smoke test.")
    print(f"[INFO] Running {args_cli.num_steps} step(s) with {args_cli.num_envs} env(s).")
    run_simulator(sim, scene)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
