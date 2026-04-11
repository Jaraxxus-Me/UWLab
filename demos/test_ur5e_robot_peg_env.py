# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone demo: spawn a UR5e + Robotiq 2F-85 gripper with rectangle and wall,
step physics with sinusoidal joint effort commands, and optionally render RGB frames
to disk in headless mode.

The arm actuator uses stiffness=0 / damping=0 (torque-controlled), so we drive it
with a simple joint-space PD controller computing effort targets — the same approach
the OmniReset OSC action term uses under the hood.

Usage (activate the project venv first):
    # GUI mode (interactive viewer)
    python demos/test_ur5e_robot_peg_env.py

    # Headless with camera rendering + image saving
    python demos/test_ur5e_robot_peg_env.py --headless --enable_cameras --save

    # More environments
    python demos/test_ur5e_robot_peg_env.py --num_envs 4
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR5e rectangle-wall scene demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--num_steps", type=int, default=1000, help="Number of simulation steps to run.")
parser.add_argument("--save", action="store_true", default=False, help="Save rendered camera images to disk.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import functools
import os

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
# Scene configuration — rectangle + wall (from OmniReset variants)
# ------------------------------------------------------------------

RECTANGLE_USD = f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Rectangle/rectangle.usd"
WALL_USD = f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Wall/wall.usd"
TABLE_USD = f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention/pat_vention.usd"
SUPPORT_USD = f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention2/Ur5MetalSupport/ur5plate.usd"


@configclass
class Ur5eRectangleWallSceneCfg(InteractiveSceneCfg):
    """Scene: UR5e robot, rectangle (insertive), wall (receptive), table, support."""

    robot = IMPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Rectangle (rigid, small mass, affected by gravity)
    insertive_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=RECTANGLE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, kinematic_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.1, 0.02)),
    )

    # Wall (kinematic — fixed in place, acts as the receptacle)
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

# Stiffness (Nm/rad) and damping (Nm·s/rad) for a simple joint PD controller.
# The OmniReset task uses task-space OSC; here we use joint-space PD for simplicity.
ARM_KP = torch.tensor([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
ARM_KD = torch.tensor([20.0, 20.0, 20.0, 5.0, 5.0, 5.0])
NUM_ARM_JOINTS = 6


# ------------------------------------------------------------------
# Camera (optional, for headless rendering)
# ------------------------------------------------------------------

def create_camera():
    """Create a camera sensor for off-screen rendering."""
    import isaacsim.core.utils.prims as prim_utils
    from isaaclab.sensors.camera import Camera, CameraCfg

    prim_utils.create_prim("/World/CameraMount", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/CameraMount/Camera",
        update_period=0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    return Camera(cfg=camera_cfg)


# ------------------------------------------------------------------
# Simulation loop
# ------------------------------------------------------------------

def run_simulator(sim: SimulationContext, scene: InteractiveScene, camera=None, rep_writer=None):
    """Step physics with sinusoidal joint effort targets."""
    robot = scene["robot"]
    insertive = scene["insertive_object"]
    receptive = scene["receptive_object"]
    sim_dt = sim.get_physics_dt()
    device = sim.device

    arm_kp = ARM_KP.to(device)
    arm_kd = ARM_KD.to(device)

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
            print(f"[INFO] Step {step}: scene reset.")

        # --- Compute sinusoidal joint targets and PD torques for arm ---
        t = step * sim_dt
        desired_pos = robot.data.default_joint_pos[:, :NUM_ARM_JOINTS].clone()
        arm_offset = 0.3 * torch.sin(
            2.0 * torch.pi * 0.2 * t
            + torch.linspace(0, torch.pi, NUM_ARM_JOINTS, device=device)
        )
        desired_pos += arm_offset.unsqueeze(0)

        pos_error = desired_pos - robot.data.joint_pos[:, :NUM_ARM_JOINTS]
        vel_error = -robot.data.joint_vel[:, :NUM_ARM_JOINTS]
        arm_torques = arm_kp * pos_error + arm_kd * vel_error

        # Build full effort tensor (arm torques + zeros for gripper joints)
        num_joints = robot.data.joint_pos.shape[1]
        efforts = torch.zeros(scene.num_envs, num_joints, device=device)
        efforts[:, :NUM_ARM_JOINTS] = arm_torques
        robot.set_joint_effort_target(efforts)

        # Gripper: position target (actuator has stiffness=17, so this works)
        gripper_target = robot.data.default_joint_pos.clone()
        gripper_target[:, NUM_ARM_JOINTS] = 0.0  # keep open
        robot.set_joint_position_target(gripper_target)

        # --- Write and step ---
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        # --- Camera ---
        if camera is not None:
            camera.update(dt=sim_dt)
            if rep_writer is not None and args_cli.save and step % 10 == 0:
                from isaaclab.utils import convert_dict_to_backend

                cam_data = convert_dict_to_backend(
                    {k: v[0] for k, v in camera.data.output.items()}, backend="numpy"
                )
                cam_info = camera.data.info[0]
                rep_output = {"annotators": {}}
                for key, data, info in zip(cam_data.keys(), cam_data.values(), cam_info.values()):
                    if info is not None:
                        rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                    else:
                        rep_output["annotators"][key] = {"render_product": {"data": data}}
                rep_output["trigger_outputs"] = {"on_time": camera.frame[0]}
                rep_writer.write(rep_output)

        # --- Print state periodically ---
        if step % 100 == 0:
            cur_arm_pos = robot.data.joint_pos[0, :NUM_ARM_JOINTS]
            ins_pos = insertive.data.root_pos_w[0]
            rec_pos = receptive.data.root_pos_w[0]
            print(
                f"  Step {step:5d} | "
                f"arm: [{', '.join(f'{v:.3f}' for v in cur_arm_pos.tolist())}] | "
                f"rect: [{ins_pos[0]:.3f}, {ins_pos[1]:.3f}, {ins_pos[2]:.3f}] | "
                f"wall: [{rec_pos[0]:.3f}, {rec_pos[1]:.3f}, {rec_pos[2]:.3f}]"
            )


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.5, 1.5, 1.0), target=(0.4, 0.0, 0.0))

    scene_cfg = Ur5eRectangleWallSceneCfg(num_envs=args_cli.num_envs, env_spacing=1.5)
    scene = InteractiveScene(scene_cfg)

    camera = None
    rep_writer = None
    if args_cli.save:
        import omni.replicator.core as rep

        camera = create_camera()
        camera.set_world_poses_from_view(
            torch.tensor([[1.5, 1.5, 1.0]]),
            torch.tensor([[0.4, 0.0, 0.0]]),
        )
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "ur5e_rectangle_wall")
        rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)
        print(f"[INFO] Saving camera images to: {output_dir}")

    sim.reset()
    print("[INFO] Setup complete. Starting simulation loop...")
    print(f"[INFO] Running {args_cli.num_steps} steps with {args_cli.num_envs} env(s).")
    run_simulator(sim, scene, camera, rep_writer)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
