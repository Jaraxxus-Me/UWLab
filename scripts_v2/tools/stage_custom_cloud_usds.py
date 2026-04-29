#!/usr/bin/env python3
"""Stage project USD assets into the CUSTOM_CLOUD_ASSETS_DIR layout.

Example:
    python scripts_v2/tools/stage_custom_cloud_usds.py --out /tmp/custom_assets

Upload the output directory contents to your asset host, then run with:
    CUSTOM_CLOUD_ASSETS_DIR=https://host.example/path/to/custom_assets
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ASSETS: tuple[tuple[str, str], ...] = (
    (
        "source/uwlab_assets/uwlab_assets/cornered_block/block/block.usd",
        "Props/Custom/CorneredBlock/block/block.usd",
    ),
    (
        "source/uwlab_assets/uwlab_assets/cornered_block/block/block_blue.usd",
        "Props/Custom/CorneredBlock/block/block_blue.usd",
    ),
    (
        "source/uwlab_assets/uwlab_assets/cornered_block/block/metadata.yaml",
        "Props/Custom/CorneredBlock/block/metadata.yaml",
    ),
    (
        "source/uwlab_assets/uwlab_assets/cornered_block/box/box.usd",
        "Props/Custom/CorneredBlock/box/box.usd",
    ),
    (
        "source/uwlab_assets/uwlab_assets/cornered_block/box/metadata.yaml",
        "Props/Custom/CorneredBlock/box/metadata.yaml",
    ),
    (
        "source/uwlab_assets/uwlab_assets/robots/ur5e_robotiq_gripper/usd/ur5e_robotiq2f140.usd",
        "Robots/UniversalRobots/Ur5eRobotiq2f140/ur5e_robotiq2f140.usd",
    ),
    (
        "source/uwlab_assets/uwlab_assets/robots/ur5e_robotiq_gripper/usd/old/ur5e_robotiq2f140.bak.usd",
        "Robots/UniversalRobots/Ur5eRobotiq2f140/old/ur5e_robotiq2f140.bak.usd",
    ),
    (
        "source/uwlab_assets/uwlab_assets/robots/ur5e_robotiq_gripper/usd/metadata.yaml",
        "Robots/UniversalRobots/Ur5eRobotiq2f140/metadata.yaml",
    ),
)

OPTIONAL_ASSETS: tuple[tuple[str, str], ...] = (
    (
        "source/uwlab_assets/uwlab_assets/robots/ur5e_robotiq_gripper/usd/robotiq_2f140_gripper.usd",
        "Robots/UniversalRobots/Robotiq2f140/robotiq_2f140_gripper.usd",
    ),
)


def is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as f:
        return b"version https://git-lfs.github.com/spec/v1" in f.read(256)


def stage_asset(repo_root: Path, out_dir: Path, src_rel: str, dst_rel: str) -> None:
    src = repo_root / src_rel
    if not src.is_file():
        raise FileNotFoundError(f"Missing source asset: {src}")
    if is_lfs_pointer(src):
        raise RuntimeError(f"Refusing to stage unresolved Git LFS pointer: {src}")

    dst = out_dir / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"{src_rel} -> {dst_rel}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Directory to populate with cloud-layout assets.")
    parser.add_argument(
        "--include-standalone-gripper",
        action="store_true",
        help="Also stage the standalone 2F-140 gripper USD if it has been hydrated.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out).expanduser().resolve()
    assets = ASSETS + (OPTIONAL_ASSETS if args.include_standalone_gripper else ())

    for src_rel, dst_rel in assets:
        stage_asset(repo_root, out_dir, src_rel, dst_rel)

    print(f"\nStaged {len(assets)} assets under {out_dir}")


if __name__ == "__main__":
    main()
