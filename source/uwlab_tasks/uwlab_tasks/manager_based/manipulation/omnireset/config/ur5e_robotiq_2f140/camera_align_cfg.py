# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scene and env config for sim2real camera alignment.

Minimal env with robot + cameras (from data_collection_rgb_cfg) but NO
randomization.  The interactive alignment script (scripts_v2/tools/sim2real/align_cameras.py)
uses keyboard controls to move/rotate the sim camera and overlay the sim
render on a real reference image, then prints the final (pos, rot, focal_length)
to paste back into data_collection_rgb_cfg.py.

Mirrors the sysid pattern:
  sysid_cfg.py  +  scripts_v2/tools/sim2real/sysid_ur5e_osc.py
  camera_align_cfg.py  +  scripts_v2/tools/sim2real/align_cameras.py
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
from uwlab_assets.robots.ur5e_robotiq_gripper import EXPLICIT_UR5E_ROBOTIQ_2F140

from ... import mdp as task_mdp
from .actions import Ur5eRobotiq2f140SysidOSCAction
from .rl_state_cfg import RlStateSceneCfg

# Same sim dt as sysid / finetune (500 Hz)
CAMERA_ALIGN_SIM_DT = 1.0 / 500.0


@configclass
class CameraAlignSceneCfg(RlStateSceneCfg):
    """Scene for camera alignment.

    Inherits from RlStateSceneCfg (robot, table, ur5_metal_support, ground,
    sky_light, insertive/receptive objects) and adds curtains + cameras.
    Same structure as DataCollectionRGBObjectSceneCfg but with NO randomization.
    """

    # Use explicit (sysid-tuned) actuator model
    robot = EXPLICIT_UR5E_ROBOTIQ_2F140.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Use a light table material during camera alignment so the black 2F-140
    # gripper is easier to separate from the workspace in RGB overlays.
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.4, 0.0, -0.881), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention/pat_vention.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.68, 0.72), roughness=0.85),
        ),
    )

    # --- Background curtains (match real workspace) ---
    curtain_left = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainLeft",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.4, 0.68, 0.519), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.0, 1.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    curtain_back = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainBack",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.0, 0.519), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.3, 1.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    curtain_right = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainRight",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.4, -0.68, 0.519), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.0, 1.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    # --- Cameras (initial poses from data_collection_rgb_cfg) ---
    front_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rgb_front_camera",
        update_period=0,
        height=480,
        width=640,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.9101819, 0.0358507, 0.4750194),
            rot=(0.67165214, 0.23014157, -0.25002852, -0.65833426),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.40),
    )

    side_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rgb_side_camera",
        update_period=0,
        height=480,
        width=640,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.5430160, -0.2917557, 0.4118886),
            rot=(0.93103116, 0.35759344, -0.06975901, 0.02101393),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.40),
    )

    wrist_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_link/robotiq_base_link/rgb_wrist_camera",
        update_period=0,
        height=480,
        width=640,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.06192727, -0.08342348, -0.02054465),
            rot=(0.02589054,  0.99876904,  0.00439874, -0.04208001),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=20.02, clipping_range=(0.03, 1e6)),
    )

# Calibration:
#   Path: /home/bowenli2/CoRL2026/diffusion_policy/scripts/sim2real/perception/calibrations/front.json
#   Camera serial: 207122078046
#   Image size: 640 x 480

# Policy sees current joint angles:
#   arm_joint_pos [rad]:  0.20109713, -1.55775622, -1.83930719, -1.33317194,  1.56346416, -1.36284811
#   arm_joint_pos [deg]:  11.52201687, -89.25285672, -105.38453916, -76.38512525,  89.57989806, -78.08544465

# Policy sees current EE pose in Isaac/sim base:
#   Position (x, y, z): -0.48972641,  0.03756944,  0.38740799
#   Quaternion [w, x, y, z]:  0.00766793,  0.00428674,  0.99994664,  0.00543577
#   Axis-angle [rx, ry, rz]:  0.01340184,  3.12618172,  0.01699411

# Isaac Sim camera pose in Isaac/sim base:
#   Position (x, y, z): -0.88067880,  0.03436852,  0.42247376
#   Quaternion [w, x, y, z]: -0.66482655, -0.23723004,  0.22472031,  0.67173532
#   Axis-angle [rx, ry, rz]: -1.45962521,  1.38265552,  4.13304234

# Isaac Sim camera pose in policy EE frame:
#   Position (x, y, z):  0.39032824, -0.00613616, -0.04110765
#   Quaternion [w, x, y, z]:  0.22224493, -0.66944707,  0.67068330, -0.22941604
#   Axis-angle [rx, ry, rz]: -1.84931158,  1.85272659, -0.63374947

# Calibration:
#   Path: /home/bowenli2/CoRL2026/diffusion_policy/scripts/sim2real/perception/calibrations/side.json
#   Camera serial: 207222070875
#   Image size: 640 x 480

# Policy sees current joint angles:
#   arm_joint_pos [rad]:  0.20106630, -1.55774898, -1.83931911, -1.33321788,  1.56346393, -1.36286718
#   arm_joint_pos [deg]:  11.52025041, -89.25244221, -105.38522218, -76.38775743,  89.57988440, -78.08653748

# Policy sees current EE pose in Isaac/sim base:
#   Position (x, y, z): -0.48971675,  0.03758629,  0.38740121
#   Quaternion [w, x, y, z]:  0.00769282,  0.00429252,  0.99994639,  0.00544086
#   Axis-angle [rx, ry, rz]:  0.01341970,  3.12613178,  0.01700976

# Isaac Sim camera pose in Isaac/sim base:
#   Position (x, y, z): -0.58335175, -0.31205614,  0.41541872
#   Quaternion [w, x, y, z]:  0.93228655,  0.35914301, -0.02178747, -0.03719405
#   Axis-angle [rx, ry, rz]:  0.73495082, -0.04458590, -0.07611396

# Isaac Sim camera pose in policy EE frame:
#   Position (x, y, z):  0.09015993, -0.35009811, -0.03323889
#   Quaternion [w, x, y, z]:  0.01327512, -0.03583448,  0.93451788, -0.35385871
#   Axis-angle [rx, ry, rz]: -0.11163573,  2.91131865, -1.10238175

# Calibration:
#   Path: /home/bowenli2/CoRL2026/diffusion_policy/scripts/sim2real/perception/calibrations/wrist.json
#   Camera serial: 752112070737
#   Image size: 640 x 480

# Policy sees current joint angles:
#   arm_joint_pos [rad]:  0.20112464, -1.55775261, -1.83931553, -1.33318608,  1.56344867, -1.36288292
#   arm_joint_pos [deg]:  11.52359293, -89.25265026, -105.38501727, -76.38593548,  89.57901013, -78.08743906

# Policy sees current EE pose in Isaac/sim base:
#   Position (x, y, z): -0.48972390,  0.03755827,  0.38740433
#   Quaternion [w, x, y, z]:  0.00767563,  0.00425547,  0.99994666,  0.00544526
#   Axis-angle [rx, ry, rz]:  0.01330403,  3.12616659,  0.01702369

# Isaac Sim camera pose in Isaac/sim base:
#   Position (x, y, z): -0.45246176, -0.04559970,  0.39201947
#   Quaternion [w, x, y, z]:  0.02041356,  0.00118244, -0.03714957,  0.99910049
#   Axis-angle [rx, ry, rz]:  0.00366723, -0.11521602,  3.09861922

# Isaac Sim camera pose in policy EE frame:
#   Position (x, y, z): -0.03804170, -0.08278546, -0.00494076
#   Quaternion [w, x, y, z]:  0.03154552,  0.99932729,  0.01645241, -0.00889803
#   Axis-angle [rx, ry, rz]:  3.07795205,  0.05067383, -0.02740615
# ---------------------------------------------------------------------------
# Minimal MDP (camera alignment only needs RGB obs + joint_pos action)
# ---------------------------------------------------------------------------
@configclass
class CameraAlignObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=task_mdp.joint_pos)

        front_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb",
                "process_image": False,
                "output_size": (240, 320),
            },
        )
        side_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("side_camera"),
                "data_type": "rgb",
                "process_image": False,
                "output_size": (240, 320),
            },
        )
        wrist_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                "process_image": False,
                "output_size": (240, 320),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class CameraAlignRewardsCfg:
    pass


@configclass
class CameraAlignTerminationsCfg:
    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)


@configclass
class CameraAlignEnvCfg(ManagerBasedRLEnvCfg):
    """Env for interactive sim2real camera alignment.

    Uses the same robot/action as sysid so the robot can be positioned
    at arbitrary joint angles.  Only 1 env needed (interactive tool).
    """

    scene: CameraAlignSceneCfg = CameraAlignSceneCfg(num_envs=1, env_spacing=2.0)
    actions: Ur5eRobotiq2f140SysidOSCAction = Ur5eRobotiq2f140SysidOSCAction()
    observations: CameraAlignObservationsCfg = CameraAlignObservationsCfg()
    rewards: CameraAlignRewardsCfg = CameraAlignRewardsCfg()
    terminations: CameraAlignTerminationsCfg = CameraAlignTerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 1
        self.episode_length_s = 99999.0
        self.sim.dt = CAMERA_ALIGN_SIM_DT

        # Place robot at average real-world position (reset_states_cfg y avg = -0.039).
        self.scene.robot.init_state.pos = (0.0, -0.039, 0.0)
        self.scene.ur5_metal_support.init_state.pos = (0.0, -0.039, -0.013)

        # Render settings for visual fidelity
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True
        self.sim.render_interval = 1

        # rerender on reset
        self.num_rerenders_on_reset = 1
