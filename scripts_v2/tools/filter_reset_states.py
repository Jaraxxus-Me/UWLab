# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Filter recorded reset states.

Keeps reset states where (a) the episode is NOT already in the success region
and (b) the insertive object's (x, y) lies inside the receptive object's
footprint (default ``|dx|, |dy| <= 0.11`` in the receptive object's frame).

Input files are the ``resets_*.pt`` torch files produced by
``record_reset_states.py``. Output files have the same nested structure with
every per-episode list filtered by the same mask.
"""

from __future__ import annotations

import argparse
import os
import torch
import yaml
from typing import Any

import isaaclab.utils.math as math_utils

# Defaults matching source/uwlab_assets/uwlab_assets/cornered_block/{block,box}/metadata.yaml
DEFAULT_INSERTIVE_OFFSET_POS = (0.0, 0.0, -0.02)
DEFAULT_INSERTIVE_OFFSET_QUAT = (1.0, 0.0, 0.0, 0.0)
DEFAULT_RECEPTIVE_OFFSET_POS = (0.0, 0.0, 0.01)
DEFAULT_RECEPTIVE_OFFSET_QUAT = (1.0, 0.0, 0.0, 0.0)
DEFAULT_SUCCESS_POSITION_THRESHOLD = 0.04
DEFAULT_SUCCESS_ORIENTATION_THRESHOLD = 0.025


def _stack(poses: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(poses).to(dtype=torch.float32)


def _offset_from_metadata(metadata_path: str | None, default_pos, default_quat):
    if metadata_path is None:
        return tuple(default_pos), tuple(default_quat)
    with open(metadata_path) as f:
        meta = yaml.safe_load(f)
    pos = tuple(meta["assembled_offset"]["pos"])
    quat = tuple(meta["assembled_offset"]["quat"])
    return pos, quat


def _thresholds_from_metadata(metadata_path: str | None, default_pos_thr, default_ori_thr):
    if metadata_path is None:
        return default_pos_thr, default_ori_thr
    with open(metadata_path) as f:
        meta = yaml.safe_load(f)
    thresholds = meta.get("success_thresholds") or {}
    return (
        float(thresholds.get("position", default_pos_thr)),
        float(thresholds.get("orientation", default_ori_thr)),
    )


def compute_masks(
    insertive_pose: torch.Tensor,
    receptive_pose: torch.Tensor,
    insertive_offset_pos: tuple[float, float, float],
    insertive_offset_quat: tuple[float, float, float, float],
    receptive_offset_pos: tuple[float, float, float],
    receptive_offset_quat: tuple[float, float, float, float],
    success_position_threshold: float,
    success_orientation_threshold: float,
    dx_max: float,
    dy_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (success_mask, inside_box_mask, xyz_distance) for diagnostics.

    ``insertive_pose`` and ``receptive_pose`` are (N, 7) tensors ordered as
    ``[x, y, z, qw, qx, qy, qz]``.
    """
    n = insertive_pose.shape[0]
    device = insertive_pose.device

    ins_pos_w = insertive_pose[:, :3]
    ins_quat_w = insertive_pose[:, 3:7]
    rec_pos_w = receptive_pose[:, :3]
    rec_quat_w = receptive_pose[:, 3:7]

    ins_off_pos = torch.tensor(insertive_offset_pos, device=device, dtype=ins_pos_w.dtype).expand(n, 3)
    ins_off_quat = torch.tensor(insertive_offset_quat, device=device, dtype=ins_quat_w.dtype).expand(n, 4)
    rec_off_pos = torch.tensor(receptive_offset_pos, device=device, dtype=rec_pos_w.dtype).expand(n, 3)
    rec_off_quat = torch.tensor(receptive_offset_quat, device=device, dtype=rec_quat_w.dtype).expand(n, 4)

    ins_align_pos_w, ins_align_quat_w = math_utils.combine_frame_transforms(
        ins_pos_w, ins_quat_w, ins_off_pos, ins_off_quat
    )
    rec_align_pos_w, rec_align_quat_w = math_utils.combine_frame_transforms(
        rec_pos_w, rec_quat_w, rec_off_pos, rec_off_quat
    )

    ins_in_rec_pos, ins_in_rec_quat = math_utils.subtract_frame_transforms(
        rec_align_pos_w, rec_align_quat_w, ins_align_pos_w, ins_align_quat_w
    )

    e_x, e_y, _ = math_utils.euler_xyz_from_quat(ins_in_rec_quat)
    euler_xy_distance = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
    xyz_distance = torch.norm(ins_in_rec_pos, dim=1)

    success_mask = (xyz_distance < success_position_threshold) & (euler_xy_distance < success_orientation_threshold)

    # "Block x, y inside box" uses the block's position in the receptive object's body frame
    # (no alignment offset applied, no yaw), so the tolerance is a footprint in box-local xy.
    block_in_box_pos, _ = math_utils.subtract_frame_transforms(rec_pos_w, rec_quat_w, ins_pos_w, ins_quat_w)
    inside_box_mask = (block_in_box_pos[:, 0].abs() <= dx_max) & (block_in_box_pos[:, 1].abs() <= dy_max)

    return success_mask, inside_box_mask, xyz_distance


def _apply_mask(data: Any, keep_idx: list[int]) -> Any:
    if isinstance(data, dict):
        return {k: _apply_mask(v, keep_idx) for k, v in data.items()}
    if isinstance(data, list):
        return [data[i] for i in keep_idx]
    return data


def filter_file(
    input_path: str,
    output_path: str,
    insertive_offset_pos,
    insertive_offset_quat,
    receptive_offset_pos,
    receptive_offset_quat,
    success_position_threshold: float,
    success_orientation_threshold: float,
    dx_max: float,
    dy_max: float,
) -> dict[str, int]:
    payload = torch.load(input_path, map_location="cpu", weights_only=False)

    if "initial_state" not in payload:
        # e.g. partial_assemblies.pt has a different structure; copy as-is.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(payload, output_path)
        return {"total": 0, "copied_as_is": 1}

    ins_list = payload["initial_state"]["rigid_object"]["insertive_object"]["root_pose"]
    rec_list = payload["initial_state"]["rigid_object"]["receptive_object"]["root_pose"]
    n = len(ins_list)
    if n == 0:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(payload, output_path)
        return {"total": 0, "kept": 0, "dropped_success": 0, "dropped_outside": 0}

    ins_pose = _stack(ins_list)
    rec_pose = _stack(rec_list)

    success_mask, inside_mask, _ = compute_masks(
        ins_pose,
        rec_pose,
        insertive_offset_pos,
        insertive_offset_quat,
        receptive_offset_pos,
        receptive_offset_quat,
        success_position_threshold,
        success_orientation_threshold,
        dx_max,
        dy_max,
    )

    keep_mask = (~success_mask) & inside_mask
    keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1).tolist()

    filtered = _apply_mask(payload, keep_idx)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(filtered, output_path)

    return {
        "total": n,
        "kept": int(keep_mask.sum().item()),
        "dropped_success": int(success_mask.sum().item()),
        "dropped_outside": int((~inside_mask).sum().item()),
        "dropped_both": int((success_mask & ~inside_mask).sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./Datasets/OmniReset/Resets/block__box_original",
        help="Directory of resets_*.pt files to filter.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./Datasets/OmniReset/Resets/block__box",
        help="Directory to write filtered files into.",
    )
    parser.add_argument("--dx_max", type=float, default=0.11, help="Max |dx| of block in box frame (meters).")
    parser.add_argument("--dy_max", type=float, default=0.11, help="Max |dy| of block in box frame (meters).")
    parser.add_argument(
        "--success_position_threshold",
        type=float,
        default=DEFAULT_SUCCESS_POSITION_THRESHOLD,
        help="xyz distance below which an episode is treated as already successful.",
    )
    parser.add_argument(
        "--success_orientation_threshold",
        type=float,
        default=DEFAULT_SUCCESS_ORIENTATION_THRESHOLD,
        help="euler xy distance below which an episode is treated as already successful.",
    )
    parser.add_argument(
        "--insertive_metadata",
        type=str,
        default=None,
        help="Optional path to metadata.yaml for the insertive object (overrides default block offsets).",
    )
    parser.add_argument(
        "--receptive_metadata",
        type=str,
        default=None,
        help=(
            "Optional path to metadata.yaml for the receptive object (overrides default box offsets and"
            " success thresholds)."
        ),
    )
    args = parser.parse_args()

    insertive_offset_pos, insertive_offset_quat = _offset_from_metadata(
        args.insertive_metadata, DEFAULT_INSERTIVE_OFFSET_POS, DEFAULT_INSERTIVE_OFFSET_QUAT
    )
    receptive_offset_pos, receptive_offset_quat = _offset_from_metadata(
        args.receptive_metadata, DEFAULT_RECEPTIVE_OFFSET_POS, DEFAULT_RECEPTIVE_OFFSET_QUAT
    )
    success_pos_thr, success_ori_thr = _thresholds_from_metadata(
        args.receptive_metadata, args.success_position_threshold, args.success_orientation_threshold
    )

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    if input_dir == output_dir:
        raise ValueError(f"input_dir and output_dir must differ, both are {input_dir}")
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")

    print(f"Filtering reset states from {input_dir} -> {output_dir}")
    print(
        f"  insertive offset pos={insertive_offset_pos} quat={insertive_offset_quat}\n"
        f"  receptive offset pos={receptive_offset_pos} quat={receptive_offset_quat}\n"
        f"  success thresholds: position<{success_pos_thr} orientation<{success_ori_thr}\n"
        f"  inside-box bounds:  |dx|<={args.dx_max} |dy|<={args.dy_max}"
    )

    files = sorted(f for f in os.listdir(input_dir) if f.endswith(".pt"))
    if not files:
        print(f"No .pt files found in {input_dir}")
        return

    grand_total = 0
    grand_kept = 0
    for fname in files:
        stats = filter_file(
            os.path.join(input_dir, fname),
            os.path.join(output_dir, fname),
            insertive_offset_pos,
            insertive_offset_quat,
            receptive_offset_pos,
            receptive_offset_quat,
            success_pos_thr,
            success_ori_thr,
            args.dx_max,
            args.dy_max,
        )
        if stats.get("copied_as_is"):
            print(f"  {fname}: copied as-is (no initial_state field)")
            continue
        total = stats["total"]
        kept = stats["kept"]
        grand_total += total
        grand_kept += kept
        pct = 100.0 * kept / max(total, 1)
        print(
            f"  {fname}: kept {kept}/{total} ({pct:.1f}%)"
            f"  dropped_success={stats['dropped_success']}"
            f"  dropped_outside={stats['dropped_outside']}"
            f"  dropped_both={stats['dropped_both']}"
        )

    if grand_total:
        print(f"Total kept {grand_kept}/{grand_total} ({100.0 * grand_kept / grand_total:.1f}%)")


if __name__ == "__main__":
    main()
