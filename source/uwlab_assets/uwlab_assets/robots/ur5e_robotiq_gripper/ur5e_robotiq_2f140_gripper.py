# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the UR5e + Robotiq 2F-140 robot.

Mirrors the 2F-85 configuration so tasks can swap between the two grippers
by changing only the articulation cfg import. The USD is a self-contained
local asset (flattened arm + Omniverse 2F-140) that lives alongside this
module under ``usd/ur5e_robotiq2f140.usd``.

The following configurations are available:

* :obj:`UR5E_ROBOTIQ_2F140_ARTICULATION`: Base articulation (USD, init state).
* :obj:`EXPLICIT_UR5E_ROBOTIQ_2F140`: Full robot with DelayedPDActuator arm (PD delay, for sim2real finetuning).
* :obj:`IMPLICIT_UR5E_ROBOTIQ_2F140`: Full robot with ImplicitActuator arm (no motor delay, for RL training).
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "usd", "ur5e_robotiq2f140.usd")

# After patching the 2F-140 USD to match the 2F-85 closed-chain modeling pattern:
# - ``*_outer_finger_joint`` converted from revolute to fixed
# - ``*_inner_finger_joint`` excluded from articulation (breaks the kinematic loop)
# - ``*_inner_knuckle_joint`` and ``*_inner_finger_pad_joint`` carry mimic constraints
# This gives 6 gripper DOFs in the articulation, matching 2F-85's count.
ROBOTIQ_2F140_DEFAULT_JOINT_POS = {
    "finger_joint": 0.0,
    "right_outer_knuckle_joint": 0.0,
    "left_inner_knuckle_joint": 0.0,
    "right_inner_knuckle_joint": 0.0,
    "left_inner_finger_pad_joint": 0.0,
    "right_inner_finger_pad_joint": 0.0,
}

UR5E_DEFAULT_JOINT_POS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": -1.5708,
    **ROBOTIQ_2F140_DEFAULT_JOINT_POS,
}

UR5E_VELOCITY_LIMITS = {
    "shoulder_pan_joint": 1.5708,
    "shoulder_lift_joint": 1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": 3.1415,
    "wrist_2_joint": 3.1415,
    "wrist_3_joint": 3.1415,
}

UR5E_EFFORT_LIMITS = {
    "shoulder_pan_joint": 150.0,
    "shoulder_lift_joint": 150.0,
    "elbow_joint": 150.0,
    "wrist_1_joint": 28.0,
    "wrist_2_joint": 28.0,
    "wrist_3_joint": 28.0,
}

UR5E_ROBOTIQ_2F140_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=36, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), joint_pos=UR5E_DEFAULT_JOINT_POS),
    soft_joint_pos_limit_factor=1,
)

_GRIPPER_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=["finger_joint"],
    stiffness=17,
    damping=10,
    effort_limit_sim=60,
)

EXPLICIT_UR5E_ROBOTIQ_2F140 = UR5E_ROBOTIQ_2F140_ARTICULATION.copy()  # type: ignore
EXPLICIT_UR5E_ROBOTIQ_2F140.actuators = {
    "arm": DelayedPDActuatorCfg(
        joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
        stiffness=0.0,
        damping=0.0,
        effort_limit=UR5E_EFFORT_LIMITS,
        effort_limit_sim=UR5E_EFFORT_LIMITS,
        velocity_limit=UR5E_VELOCITY_LIMITS,
        velocity_limit_sim=UR5E_VELOCITY_LIMITS,
        min_delay=0,
        max_delay=1,
    ),
    "gripper": _GRIPPER_ACTUATOR,
}

IMPLICIT_UR5E_ROBOTIQ_2F140 = UR5E_ROBOTIQ_2F140_ARTICULATION.copy()  # type: ignore
IMPLICIT_UR5E_ROBOTIQ_2F140.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
        stiffness=0.0,
        damping=0.0,
        effort_limit_sim=UR5E_EFFORT_LIMITS,
        velocity_limit_sim=UR5E_VELOCITY_LIMITS,
    ),
    "gripper": _GRIPPER_ACTUATOR,
}
