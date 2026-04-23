# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Patch the 2F-140 USD to add PhysxMimicJoint constraints on the knuckle joints.

The upstream 2F-140 USD we ship has no mimic constraints, so when ``finger_joint``
is commanded closed, the 3 other knuckle revolute joints are independent DOFs with
no actuator driving them; the solver has to resolve the closed-chain linkage via
loop-closure constraints on the pad_joints, which is slow and often leaves the
gripper visually frozen.

The 2F-85 USD (``ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd``) is known
good and uses the following mimic pattern on ``finger_joint``:

    right_outer_knuckle_joint          gearing -1.0
    left_inner_knuckle_joint           gearing -1.0
    right_inner_knuckle_joint          gearing -1.0
    left_inner_finger_knuckle_joint    gearing +1.0
    right_inner_finger_knuckle_joint   gearing +1.0

The 2F-140 has the same 4-bar topology but uses ``*_inner_finger_pad_joint``
instead of ``*_inner_finger_knuckle_joint`` for the second-loop closers, so the
gearing-+1 pair maps to pad_joint here. The 2 free ``*_inner_finger_joint``s
remain unconstrained — they close the loop physically, matching the 2F-85's free
``*_inner_finger_joint``s.

Attributes written match the 2F-85 convention verbatim:
    physxMimicJoint:rotZ:gearing
    physxMimicJoint:rotZ:offset
    physxMimicJoint:rotZ:referenceJoint   (relationship)

Usage::

    python demos/patch_2f140_usd_mimic.py \\
        --src source/uwlab_assets/uwlab_assets/robots/ur5e_robotiq_gripper/usd/ur5e_robotiq2f140.usd \\
        --dst source/uwlab_assets/uwlab_assets/robots/ur5e_robotiq_gripper/usd/ur5e_robotiq2f140_mimic.usd
"""
from __future__ import annotations

import argparse
import sys

from pxr import Sdf, Usd, UsdPhysics

try:
    from pxr import PhysxSchema
    _HAS_PHYSX_SCHEMA = True
except ImportError:
    _HAS_PHYSX_SCHEMA = False

MIMIC_PLAN: dict[str, float] = {
    # 3 knuckle joints coupled to finger_joint with gearing = -1 (parallel-jaw mirror)
    "right_outer_knuckle_joint": -1.0,
    "left_inner_knuckle_joint": -1.0,
    "right_inner_knuckle_joint": -1.0,
    # 2 pad_joint loop closers coupled with gearing = +1 (same direction as finger_joint)
    "left_inner_finger_pad_joint": +1.0,
    "right_inner_finger_pad_joint": +1.0,
}

# Any joint that the mimic would drive to a target outside its native range needs its
# limits widened. The 2F-140 USD has right_outer_knuckle_joint at (0°, 75°), but with
# gearing = -1 and finger_joint ∈ [0°, 45°], the mimic target lives in [-45°, 0°] —
# PhysX clamps at the 0° lower limit and the right side never closes.
LIMIT_OVERRIDES: dict[str, tuple[float, float]] = {
    "right_outer_knuckle_joint": (-75.0, 75.0),
}

REFERENCE_JOINT_NAME = "finger_joint"


def find_joint(stage: Usd.Stage, name: str) -> Usd.Prim | None:
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint) and prim.GetName() == name:
            return prim
    return None


def add_mimic(joint_prim: Usd.Prim, reference_path: Sdf.Path, gearing: float) -> None:
    # Raw attributes match the 2F-85 USD authoring exactly (no schema API applied there).
    gearing_attr = joint_prim.CreateAttribute("physxMimicJoint:rotZ:gearing", Sdf.ValueTypeNames.Float)
    gearing_attr.Set(float(gearing))

    offset_attr = joint_prim.CreateAttribute("physxMimicJoint:rotZ:offset", Sdf.ValueTypeNames.Float)
    offset_attr.Set(0.0)

    rel = joint_prim.CreateRelationship("physxMimicJoint:rotZ:referenceJoint")
    rel.SetTargets([reference_path])

    # Also apply as an explicit schema when PhysxSchema is available — 2F-85's
    # authoring works without this, but being explicit avoids schema-resolution
    # surprises across PhysX / Isaac Sim versions.
    if _HAS_PHYSX_SCHEMA:
        PhysxSchema.PhysxMimicJointAPI.Apply(joint_prim, "rotZ")


def widen_limits(joint_prim: Usd.Prim, lower_deg: float, upper_deg: float) -> tuple[float, float]:
    rj = UsdPhysics.RevoluteJoint(joint_prim)
    old_lo = rj.GetLowerLimitAttr().Get()
    old_hi = rj.GetUpperLimitAttr().Get()
    rj.GetLowerLimitAttr().Set(float(lower_deg))
    rj.GetUpperLimitAttr().Set(float(upper_deg))
    return old_lo, old_hi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", required=True, help="Path to the source 2F-140 USD.")
    parser.add_argument("--dst", required=True, help="Where to write the patched USD.")
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.src)
    if stage is None:
        sys.exit(f"Failed to open {args.src}")

    ref_prim = find_joint(stage, REFERENCE_JOINT_NAME)
    if ref_prim is None:
        sys.exit(f"Reference joint {REFERENCE_JOINT_NAME!r} not found in {args.src}")
    ref_path = ref_prim.GetPath()
    print(f"reference joint: {ref_path}")

    patched = 0
    for joint_name, gearing in MIMIC_PLAN.items():
        prim = find_joint(stage, joint_name)
        if prim is None:
            print(f"  [skip] {joint_name}: not found")
            continue
        add_mimic(prim, ref_path, gearing)
        patched += 1
        print(f"  [mimic] {joint_name:35s} gearing = {gearing:+.1f}  ref = {ref_path.name}")

    for joint_name, (lo, hi) in LIMIT_OVERRIDES.items():
        prim = find_joint(stage, joint_name)
        if prim is None or not prim.IsA(UsdPhysics.RevoluteJoint):
            print(f"  [skip] {joint_name}: revolute joint not found for limit override")
            continue
        old_lo, old_hi = widen_limits(prim, lo, hi)
        print(f"  [limit] {joint_name:35s} {old_lo}° → {old_hi}°  =>  {lo}° → {hi}°")

    if patched == 0:
        sys.exit("No joints patched — aborting write.")

    stage.Export(args.dst)
    print(f"\nWrote {patched} mimic constraints to {args.dst}")


if __name__ == "__main__":
    main()
