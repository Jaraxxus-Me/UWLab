# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Filter reset-state datasets using simulation collision checks.

This script replays recorded ``resets_*.pt`` files in an Isaac Lab environment
and keeps states that satisfy all of the following:

1. robot/block/box are collision-free under the same collision analyzers used by
   reset-state collection,
2. the configured EE body local +Z axis is within ``--top_down_angle_deg`` of
   world -Z,
3. the block/box pair is not already in the assembly success region.

The input structure is preserved. By default output is written to a sibling
``block__box_filtered`` directory; use ``--in_place`` to overwrite input files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--input_dir",
    type=str,
    default="./Datasets/PhysCoder_2f140/Resets/block__box",
    help="Directory containing resets_*.pt files.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to write filtered files. Defaults to '<input_dir>_filtered'.",
)
parser.add_argument("--in_place", action="store_true", help="Overwrite input files with filtered data.")
parser.add_argument(
    "--task",
    type=str,
    default="OmniReset-UR5eRobotiq2f140-ObjectAnywhereEEAnywhere-v0",
    help="Reset-state task used to construct the scene and collision analyzers.",
)
parser.add_argument("--batch_size", type=int, default=1024, help="Number of states replayed per simulation batch.")
parser.add_argument("--ee_body_name", type=str, default="wrist_3_link", help="Body whose +Z axis is checked.")
parser.add_argument("--top_down_angle_deg", type=float, default=30.0, help="Max angle between body +Z and world -Z.")
parser.add_argument("--success_position_threshold", type=float, default=0.04)
parser.add_argument("--success_orientation_threshold", type=float, default=0.025)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import isaaclab.utils.math as math_utils

import isaaclab_tasks  # noqa: F401
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _install_namespace_package(name: str, package_path: Path) -> None:
    package = sys.modules.get(name)
    if package is None:
        package = types.ModuleType(name)
        package.__path__ = [str(package_path)]
        package.__package__ = name
        sys.modules[name] = package


def _bootstrap_uwlab_sources() -> None:
    uwlab_root = Path(__file__).resolve().parents[2]
    _install_namespace_package("uwlab", uwlab_root / "source" / "uwlab" / "uwlab")
    _install_namespace_package("uwlab.envs", uwlab_root / "source" / "uwlab" / "uwlab" / "envs")
    _install_namespace_package("uwlab_tasks", uwlab_root / "source" / "uwlab_tasks" / "uwlab_tasks")
    _install_namespace_package("uwlab_rl", uwlab_root / "source" / "uwlab_rl" / "uwlab_rl")


_bootstrap_uwlab_sources()
import uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f140  # noqa: F401,E402
import uwlab_tasks.manager_based.manipulation.omnireset.mdp as task_mdp
from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f140 import reset_states_cfg
from uwlab_tasks.utils.hydra import hydra_task_compose


INSERTIVE_ASSEMBLED_OFFSET_POS = (0.0, 0.0, -0.021639)
INSERTIVE_ASSEMBLED_OFFSET_QUAT = (1.0, 0.0, 0.0, 0.0)
RECEPTIVE_ASSEMBLED_OFFSET_POS = (0.0, 0.0, 0.01)
RECEPTIVE_ASSEMBLED_OFFSET_QUAT = (1.0, 0.0, 0.0, 0.0)


def _num_records(payload: dict[str, Any]) -> int:
    return len(payload["initial_state"]["rigid_object"]["insertive_object"]["root_pose"])


def _slice_nested(data: Any, indices: Sequence[int]) -> Any:
    if isinstance(data, dict):
        return {k: _slice_nested(v, indices) for k, v in data.items()}
    if isinstance(data, list):
        return [data[i] for i in indices]
    return data


def _apply_keep_mask(data: Any, keep_indices: list[int]) -> Any:
    return _slice_nested(data, keep_indices)


def _stack_field(payload: dict[str, Any], path: Sequence[str], indices: Sequence[int], device: torch.device) -> torch.Tensor:
    data: Any = payload
    for key in path:
        data = data[key]
    return torch.stack([data[i] for i in indices], dim=0).to(device=device, dtype=torch.float32)


def _write_batch_state(env: ManagerBasedRLEnv, payload: dict[str, Any], indices: Sequence[int]) -> None:
    env_ids = torch.arange(len(indices), device=env.device)
    scene = env.scene

    for name, articulation in scene._articulations.items():
        state = payload["initial_state"].get("articulation", {}).get(name)
        if state is None:
            continue
        root_pose = _stack_field(payload, ("initial_state", "articulation", name, "root_pose"), indices, env.device)
        root_pose[:, :3] += scene.env_origins[env_ids]
        root_velocity = _stack_field(
            payload, ("initial_state", "articulation", name, "root_velocity"), indices, env.device
        )
        joint_position = _stack_field(
            payload, ("initial_state", "articulation", name, "joint_position"), indices, env.device
        )
        joint_velocity = _stack_field(
            payload, ("initial_state", "articulation", name, "joint_velocity"), indices, env.device
        )
        articulation.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        articulation.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        articulation.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)
        articulation.set_joint_position_target(joint_position, env_ids=env_ids)
        articulation.set_joint_velocity_target(joint_velocity, env_ids=env_ids)

    for name, rigid_object in scene._rigid_objects.items():
        state = payload["initial_state"].get("rigid_object", {}).get(name)
        if state is None:
            continue
        root_pose = _stack_field(payload, ("initial_state", "rigid_object", name, "root_pose"), indices, env.device)
        root_pose[:, :3] += scene.env_origins[env_ids]
        root_velocity = _stack_field(
            payload, ("initial_state", "rigid_object", name, "root_velocity"), indices, env.device
        )
        rigid_object.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        rigid_object.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)

    scene.write_data_to_sim()
    env.sim.forward()
    scene.update(dt=0.0)


def _offset_pose(
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
    offset_pos: tuple[float, float, float],
    offset_quat: tuple[float, float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    n = pos_w.shape[0]
    pos = torch.tensor(offset_pos, device=pos_w.device, dtype=pos_w.dtype).expand(n, 3)
    quat = torch.tensor(offset_quat, device=quat_w.device, dtype=quat_w.dtype).expand(n, 4)
    return math_utils.combine_frame_transforms(pos_w, quat_w, pos, quat)


def _initial_success_mask(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    insertive: RigidObject = env.scene["insertive_object"]
    receptive: RigidObject = env.scene["receptive_object"]
    ins_pos, ins_quat = _offset_pose(
        insertive.data.root_pos_w[env_ids],
        insertive.data.root_quat_w[env_ids],
        INSERTIVE_ASSEMBLED_OFFSET_POS,
        INSERTIVE_ASSEMBLED_OFFSET_QUAT,
    )
    rec_pos, rec_quat = _offset_pose(
        receptive.data.root_pos_w[env_ids],
        receptive.data.root_quat_w[env_ids],
        RECEPTIVE_ASSEMBLED_OFFSET_POS,
        RECEPTIVE_ASSEMBLED_OFFSET_QUAT,
    )
    rel_pos, rel_quat = math_utils.subtract_frame_transforms(rec_pos, rec_quat, ins_pos, ins_quat)
    e_x, e_y, _ = math_utils.euler_xyz_from_quat(rel_quat)
    euler_xy_dist = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
    xyz_dist = torch.norm(rel_pos, dim=1)
    return (xyz_dist < args_cli.success_position_threshold) & (
        euler_xy_dist < args_cli.success_orientation_threshold
    )


def _top_down_mask(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    body_idx = robot.data.body_names.index(args_cli.ee_body_name)
    quat_w = robot.data.body_link_quat_w[env_ids, body_idx]
    local_z = torch.tensor((0.0, 0.0, 1.0), device=env.device, dtype=torch.float32).expand(len(env_ids), 3)
    z_axis_w = math_utils.quat_apply(quat_w, local_z)
    world_neg_z = torch.tensor((0.0, 0.0, -1.0), device=env.device, dtype=torch.float32)
    cos_angle = torch.sum(z_axis_w * world_neg_z, dim=1)
    return cos_angle >= torch.cos(torch.tensor(args_cli.top_down_angle_deg * torch.pi / 180.0, device=env.device))


def _make_collision_analyzers(env: ManagerBasedRLEnv):
    cfgs = [
        task_mdp.CollisionAnalyzerCfg(
            num_points=1024,
            max_dist=0.5,
            min_dist=-0.0005,
            asset_cfg=SceneEntityCfg("robot"),
            obstacle_cfgs=[SceneEntityCfg("insertive_object")],
        ),
        task_mdp.CollisionAnalyzerCfg(
            num_points=1024,
            max_dist=0.5,
            min_dist=0.0,
            asset_cfg=SceneEntityCfg("robot"),
            obstacle_cfgs=[SceneEntityCfg("receptive_object")],
        ),
        task_mdp.CollisionAnalyzerCfg(
            num_points=1024,
            max_dist=0.5,
            min_dist=-0.001,
            asset_cfg=SceneEntityCfg("insertive_object"),
            obstacle_cfgs=[SceneEntityCfg("receptive_object")],
        ),
    ]
    return [cfg.class_type(cfg, env) for cfg in cfgs]


def _collision_free_mask(env: ManagerBasedRLEnv, analyzers, env_ids: torch.Tensor) -> torch.Tensor:
    return torch.all(torch.stack([analyzer(env, env_ids) for analyzer in analyzers]), dim=0)


def filter_file(env: ManagerBasedRLEnv, analyzers, input_path: str, output_path: str) -> dict[str, int]:
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    if "initial_state" not in payload:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(payload, output_path)
        return {"copied_as_is": 1}

    total = _num_records(payload)
    keep_indices: list[int] = []
    dropped_collision = 0
    dropped_not_top_down = 0
    dropped_initial_success = 0

    for start in range(0, total, env.num_envs):
        end = min(start + env.num_envs, total)
        batch_indices = list(range(start, end))
        env_ids = torch.arange(len(batch_indices), device=env.device)
        _write_batch_state(env, payload, batch_indices)

        collision_free = _collision_free_mask(env, analyzers, env_ids)
        top_down = _top_down_mask(env, env_ids)
        initial_success = _initial_success_mask(env, env_ids)
        fail_collision = ~collision_free
        fail_not_top_down = collision_free & ~top_down
        fail_initial_success = collision_free & top_down & initial_success
        keep = collision_free & top_down & (~initial_success)

        keep_indices.extend([batch_indices[i] for i in torch.nonzero(keep, as_tuple=False).squeeze(-1).tolist()])
        dropped_collision += int(fail_collision.sum().item())
        dropped_not_top_down += int(fail_not_top_down.sum().item())
        dropped_initial_success += int(fail_initial_success.sum().item())

    filtered = _apply_keep_mask(payload, keep_indices)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(filtered, output_path)
    return {
        "total": total,
        "kept": len(keep_indices),
        "dropped_collision": dropped_collision,
        "dropped_not_top_down": dropped_not_top_down,
        "dropped_initial_success": dropped_initial_success,
        "dropped_any": total - len(keep_indices),
    }


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=hydra_args)
def main(env_cfg, _agent_cfg) -> None:
    input_dir = os.path.abspath(args_cli.input_dir)
    if args_cli.in_place:
        output_dir = input_dir
    else:
        output_dir = os.path.abspath(args_cli.output_dir or f"{input_dir}_filtered")
        if output_dir == input_dir:
            raise ValueError("output_dir must differ from input_dir unless --in_place is set")

    env_cfg.scene.num_envs = args_cli.batch_size
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.scene.insertive_object = reset_states_cfg.variants["scene.insertive_object"]["block"]
    env_cfg.scene.receptive_object = reset_states_cfg.variants["scene.receptive_object"]["box"]

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()
    analyzers = _make_collision_analyzers(env)

    files = sorted(f for f in os.listdir(input_dir) if f.startswith("resets_") and f.endswith(".pt"))
    if not files:
        raise FileNotFoundError(f"No resets_*.pt files found in {input_dir}")

    print(f"Filtering {len(files)} reset files: {input_dir} -> {output_dir}", flush=True)
    print(
        f"Top-down filter: angle({args_cli.ee_body_name} +Z, world -Z) <= {args_cli.top_down_angle_deg} deg",
        flush=True,
    )

    totals = {"total": 0, "kept": 0, "dropped_collision": 0, "dropped_not_top_down": 0, "dropped_initial_success": 0}
    report = {"input_dir": input_dir, "output_dir": output_dir, "files": {}, "total": totals}
    for fname in files:
        stats = filter_file(env, analyzers, os.path.join(input_dir, fname), os.path.join(output_dir, fname))
        report["files"][fname] = stats
        if stats.get("copied_as_is"):
            print(f"{fname}: copied as-is", flush=True)
            continue
        for key in totals:
            totals[key] += stats[key]
        print(
            f"{fname}: total={stats['total']} kept={stats['kept']} "
            f"filtered_any={stats['dropped_any']} collision={stats['dropped_collision']} "
            f"not_top_down={stats['dropped_not_top_down']} initial_success={stats['dropped_initial_success']}",
            flush=True,
        )

    print(
        "TOTAL: "
        f"total={totals['total']} kept={totals['kept']} "
        f"filtered_any={totals['total'] - totals['kept']} collision={totals['dropped_collision']} "
        f"not_top_down={totals['dropped_not_top_down']} initial_success={totals['dropped_initial_success']}",
        flush=True,
    )
    report["total"] = totals | {"dropped_any": totals["total"] - totals["kept"]}
    report_path = os.path.join(output_dir, "filter_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"Wrote report: {report_path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
