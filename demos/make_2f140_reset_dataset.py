# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a 2F-140-specific OmniReset dataset by zeroing the gripper joints.

The upstream OmniReset reset-state datasets were recorded on a 2F-85 gripper.
Applied to a 2F-140 articulation, indices 10-11 of the recorded joint-position
tensor land on ``*_inner_finger_pad_joint`` instead of the 2F-85
``*_inner_finger_knuckle_joint``, and the remaining gripper joints (6-9)
assume 2F-85 finger kinematics that do not match the 2F-140 geometry.

This script zeroes joint positions and velocities for indices 6-11 (all six
gripper DOFs), putting the gripper in its documented open pose
(``finger_open_joint_angle: 0.0`` in the 2F-140 metadata.yaml and the
``ROBOTIQ_2F140_DEFAULT_JOINT_POS`` dict). It only rewrites gripper state —
arm qpos, object root poses, and root velocities are preserved.

Only safe for reset types that do not assume the object is held by the
gripper (e.g. ``ObjectAnywhereEEAnywhere``). Grasped reset types need a
different treatment.
"""
from __future__ import annotations

import argparse
import os
import pathlib

import torch

ARM_DOF = 6
GRIPPER_DOF = 6


def process_file(src: pathlib.Path, dst: pathlib.Path) -> None:
    data = torch.load(src, map_location="cpu", weights_only=False)
    robot_state = data["initial_state"]["articulation"]["robot"]

    jp_list = robot_state["joint_position"]
    jv_list = robot_state["joint_velocity"]
    n = len(jp_list)

    jp_stack = torch.stack([t.cpu() for t in jp_list])
    jv_stack = torch.stack([t.cpu() for t in jv_list])

    grip_before = jp_stack[:, ARM_DOF:ARM_DOF + GRIPPER_DOF]
    print(
        f"  {src.name}: n={n}, gripper qpos before → "
        f"min={grip_before.min(dim=0).values.tolist()}, "
        f"max={grip_before.max(dim=0).values.tolist()}, "
        f"mean={grip_before.mean(dim=0).tolist()}"
    )

    jp_stack[:, ARM_DOF:ARM_DOF + GRIPPER_DOF] = 0.0
    jv_stack[:, ARM_DOF:ARM_DOF + GRIPPER_DOF] = 0.0

    robot_state["joint_position"] = [jp_stack[i] for i in range(n)]
    robot_state["joint_velocity"] = [jv_stack[i] for i in range(n)]

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, dst)
    print(f"    → wrote {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        default=os.path.expanduser("~/.cache/uwlab/assets/Datasets/OmniReset"),
        help="Root of the original 2F-85 reset-state datasets (contains Resets/).",
    )
    parser.add_argument(
        "--dst",
        default=os.path.expanduser("~/.cache/uwlab/assets/Datasets/OmniReset2f140"),
        help="Output root for the 2F-140 reset-state datasets.",
    )
    args = parser.parse_args()

    src_resets = pathlib.Path(args.src) / "Resets"
    dst_resets = pathlib.Path(args.dst) / "Resets"
    if not src_resets.is_dir():
        raise SystemExit(f"src Resets/ not found: {src_resets}")

    for pair_dir in sorted(src_resets.iterdir()):
        if not pair_dir.is_dir():
            continue
        print(f"Pair: {pair_dir.name}")
        for pt in sorted(pair_dir.glob("resets_*.pt")):
            process_file(pt, dst_resets / pair_dir.name / pt.name)


if __name__ == "__main__":
    main()
