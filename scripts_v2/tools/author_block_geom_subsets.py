"""Author visual material GeomSubsets for the cornered block asset."""

from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom


def _face_indices_by_center_z(mesh: UsdGeom.Mesh, threshold: float) -> tuple[list[int], list[int]]:
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()

    top: list[int] = []
    body: list[int] = []
    offset = 0
    for face_idx, count in enumerate(counts):
        face_points = [points[indices[offset + i]] for i in range(count)]
        center_z = sum(p[2] for p in face_points) / count
        if center_z > threshold:
            top.append(face_idx)
        else:
            body.append(face_idx)
        offset += count
    return body, top


def _set_subset(mesh: UsdGeom.Mesh, name: str, indices: list[int], region_name: str) -> None:
    subset = UsdGeom.Subset(mesh.GetPrim().GetChild(name))
    if not subset:
        raise RuntimeError(f"Expected existing subset '{name}' under {mesh.GetPath()}")
    subset.CreateElementTypeAttr("face")
    subset.CreateFamilyNameAttr("materialBind")
    subset.CreateIndicesAttr(indices)
    subset.GetPrim().SetCustomDataByKey("uwlab:region", region_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Input USD path.")
    parser.add_argument("--dst", required=True, help="Output USD path.")
    parser.add_argument("--mesh-path", default="/cube/visuals/cube/Mesh")
    parser.add_argument("--threshold-z", type=float, default=0.035)
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())

    stage = Usd.Stage.Open(str(dst))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(args.mesh_path))
    if not mesh:
        raise RuntimeError(f"Mesh not found: {args.mesh_path}")

    body_indices, top_indices = _face_indices_by_center_z(mesh, args.threshold_z)
    if not body_indices or not top_indices:
        raise RuntimeError(
            f"Invalid split at z>{args.threshold_z}: body={len(body_indices)}, top={len(top_indices)}"
        )

    with Sdf.ChangeBlock():
        _set_subset(mesh, "PreviewSurface", body_indices, "body_region")
        _set_subset(mesh, "PreviewSurfaceFace", top_indices, "top_region")

    stage.GetRootLayer().Save()
    print(f"Wrote {dst}")
    print(f"PreviewSurface/body_region faces: {len(body_indices)}")
    print(f"PreviewSurfaceFace/top_region faces: {len(top_indices)}")


if __name__ == "__main__":
    main()
