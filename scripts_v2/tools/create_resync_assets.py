#!/usr/bin/env python3
"""Create the ReSync cube and goal-region USD assets."""

from __future__ import annotations

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = REPO_ROOT / "source/uwlab_assets/uwlab_assets/resync"


def _create_box_mesh(stage: Usd.Stage, path: str, size: tuple[float, float, float]) -> UsdGeom.Mesh:
    hx, hy, hz = (dimension / 2.0 for dimension in size)
    points = [
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    ]
    # Faces are ordered bottom, top, +Y, -Y, +X, -X.
    face_vertex_indices = [
        0, 3, 2, 1,
        4, 5, 6, 7,
        3, 7, 6, 2,
        0, 1, 5, 4,
        1, 2, 6, 5,
        0, 4, 7, 3,
    ]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr([(-hx, -hy, -hz), (hx, hy, hz)])
    return mesh


def _create_preview_material(
    stage: Usd.Stage, path: str, color: tuple[float, float, float]
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def _create_physics_material(stage: Usd.Stage, path: str) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(0.5)
    physics_material.CreateDynamicFrictionAttr(0.5)
    physics_material.CreateRestitutionAttr(0.0)
    return material


def _create_stage(path: Path, root_name: str, *, kinematic: bool) -> tuple[Usd.Stage, UsdGeom.Xform]:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{root_name}")
    stage.SetDefaultPrim(root.GetPrim())
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid_body.CreateRigidBodyEnabledAttr(True)
    rigid_body.CreateKinematicEnabledAttr(kinematic)
    return stage, root


def create_cube_resync() -> Path:
    path = ASSET_ROOT / "cube_resync/cube_resync.usd"
    stage, root = _create_stage(path, "cube_resync", kinematic=False)

    UsdGeom.Xform.Define(stage, f"{root.GetPath()}/visuals")
    UsdGeom.Scope.Define(stage, f"{root.GetPath()}/visuals/Looks")
    visual = _create_box_mesh(stage, f"{root.GetPath()}/visuals/cube", (0.06, 0.06, 0.06))
    purple = _create_preview_material(stage, f"{root.GetPath()}/visuals/Looks/Purple", (0.45, 0.08, 0.75))
    red = _create_preview_material(stage, f"{root.GetPath()}/visuals/Looks/RedTop", (1.0, 0.0, 0.0))

    purple_faces = UsdGeom.Subset.Define(stage, f"{visual.GetPath()}/PurpleFaces")
    purple_faces.CreateElementTypeAttr(UsdGeom.Tokens.face)
    purple_faces.CreateFamilyNameAttr("materialBind")
    purple_faces.CreateIndicesAttr([0, 2, 3, 4, 5])
    UsdShade.MaterialBindingAPI.Apply(purple_faces.GetPrim()).Bind(purple)

    red_top = UsdGeom.Subset.Define(stage, f"{visual.GetPath()}/RedTop")
    red_top.CreateElementTypeAttr(UsdGeom.Tokens.face)
    red_top.CreateFamilyNameAttr("materialBind")
    red_top.CreateIndicesAttr([1])
    UsdShade.MaterialBindingAPI.Apply(red_top.GetPrim()).Bind(red)
    UsdGeom.Subset.SetFamilyType(visual, "materialBind", UsdGeom.Tokens.partition)

    UsdGeom.Xform.Define(stage, f"{root.GetPath()}/collisions")
    collision = _create_box_mesh(stage, f"{root.GetPath()}/collisions/cube", (0.06, 0.06, 0.06))
    collision.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(collision.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim()).CreateApproximationAttr("convexHull")
    physics_material = _create_physics_material(stage, f"{root.GetPath()}/PhysicsMaterial")
    UsdShade.MaterialBindingAPI.Apply(collision.GetPrim()).Bind(
        physics_material, materialPurpose="physics"
    )

    stage.GetRootLayer().Save()
    return path


def create_region_resync() -> Path:
    path = ASSET_ROOT / "region_resync/region_resync.usd"
    stage, root = _create_stage(path, "region_resync", kinematic=True)

    UsdGeom.Xform.Define(stage, f"{root.GetPath()}/visuals")
    UsdGeom.Scope.Define(stage, f"{root.GetPath()}/visuals/Looks")
    visual = _create_box_mesh(stage, f"{root.GetPath()}/visuals/region", (0.062, 0.062, 0.001))
    red = _create_preview_material(stage, f"{root.GetPath()}/visuals/Looks/Red", (1.0, 0.0, 0.0))
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(red)

    # Deliberately omit PhysicsCollisionAPI and collision geometry: this asset is
    # a kinematic visual goal marker, while the table provides physical support.
    stage.GetRootLayer().Save()
    return path


def main() -> None:
    for asset_path in (create_cube_resync(), create_region_resync()):
        print(asset_path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
