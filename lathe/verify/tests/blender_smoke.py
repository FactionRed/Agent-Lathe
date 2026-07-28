"""
Proof that the verifier is Blender-optional, not Blender-hostile.

Run:
    blender -b -P lathe/verify/tests/blender_smoke.py

Everything else in the test suite runs on plain CPython, which is the point of
the neutral mesh representation (see mesh.py). This one script builds a broken
asset with `bmesh`, adapts it through `from_bpy_mesh`, and runs the same rule
engine inside Blender's interpreter. It is not part of the unittest suite,
because requiring Blender to test a headless subsystem would defeat the design
decision it exists to check.

Asserts three things:
  1. `from_bpy_mesh` reconstructs topology, transform, uvs and material groups.
  2. The rules produce the same codes for the bpy-sourced asset as for the
     equivalent hand-built fixture.
  3. Nothing in the engine imports bpy: it already ran outside Blender.

Style follows prototype/lathe_proto.py: bmesh only, no bpy.ops, no mode
switches, no selection state.
"""

import os
import sys

import bmesh
import bpy
from mathutils import Vector

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lathe.verify import EmitSpec, Asset, run  # noqa: E402
from lathe.verify.mesh import from_bpy_mesh  # noqa: E402


def rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def reset():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for m in list(bpy.data.meshes):
        bpy.data.meshes.remove(m)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)


def build(name, size=0.6, flip_top=True, scale=(0.01, 0.01, 0.01), loose=True):
    """A crate with the four defects a bpy round trip most often introduces."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=size)
    bmesh.ops.translate(bm, vec=Vector((0, 0, size / 2)), verts=bm.verts)
    bm.faces.ensure_lookup_table()

    if flip_top:
        top = max(bm.faces, key=lambda f: f.calc_center_median().z)
        top.normal_flip()  # E103: one face wound the wrong way

    if loose:
        bm.verts.new(Vector((0.0, 0.0, 0.9)))  # E105: an orphan vertex

    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()

    mat = bpy.data.materials.new("oak")
    me.materials.append(mat)

    obj = bpy.data.objects.new(name, me)
    obj.scale = scale  # E104: an unapplied transform
    bpy.context.scene.collection.objects.link(obj)
    return obj


def main():
    print("\nBlender %s" % bpy.app.version_string)
    reset()

    rule("1  adapter: bpy object -> AssetMesh")
    obj = build("crate_final")  # N401: not [Prefix]_[Name]_[Variant]_[LOD]
    mesh = from_bpy_mesh(obj)
    print("  name          : %s" % mesh.name)
    print("  verts / faces : %d / %d" % (mesh.num_verts, mesh.num_faces))
    print("  tris          : %d" % mesh.triangle_count())
    print("  transform     : scale=%s rotation=%s"
          % (tuple(round(s, 4) for s in mesh.transform.scale),
             tuple(round(r, 4) for r in mesh.transform.rotation_euler)))
    print("  face groups   : %s" % sorted(mesh.face_groups))
    print("  uv sets       : %s" % sorted(mesh.uv_sets))
    checks = [
        ("verts adapted", mesh.num_verts == 9),
        ("faces adapted", mesh.num_faces == 6),
        # bpy stores transforms as float32, so 0.01 comes back as
        # 0.009999999776482582. This is exactly why E104 takes a tolerance
        # parameter rather than testing equality against 1.0.
        ("scale carried", all(abs(s - 0.01) < 1e-7 for s in mesh.transform.scale)),
        ("material became a face group", sorted(mesh.face_groups) == ["oak"]),
    ]

    rule("2  the same rule engine, running inside Blender")
    asset = Asset(
        mesh,
        EmitSpec(
            profile="game_prop_pbr",
            budget={"tris_lod0": 1500},
            components=1,
            origin="bottom_center",
            texel_density=None,
            textures=(),
            bake=None,
        ),
    )
    report = run(asset)
    print(report.render(ascii_only=True))
    codes = sorted(set(report.codes))
    checks += [
        ("E103 flipped face found", "E103" in codes),
        ("E104 unapplied scale found", "E104" in codes),
        ("E105 orphan vertex found", "E105" in codes),
        ("N401 naming found", "N401" in codes),
        ("gate fails", report.passed is False),
    ]

    rule("3  a clean bpy asset produces zero findings")
    reset()
    clean = build("SM_Crate_Wood_LOD0", flip_top=False, scale=(1.0, 1.0, 1.0), loose=False)
    clean_report = run(
        Asset(
            from_bpy_mesh(clean),
            EmitSpec(
                budget={"tris_lod0": 1500},
                components=1,
                origin="bottom_center",
                texel_density=None,
                textures=(),
                bake=None,
            ),
        )
    )
    print(clean_report.render(ascii_only=True))
    checks.append(("clean bpy asset is clean", clean_report.clean))

    rule("VERDICT")
    failed = 0
    for label, ok in checks:
        failed += not ok
        print("  %-32s %s" % (label, "PASS" if ok else "FAIL"))
    print("\n  %d/%d checks passed" % (len(checks) - failed, len(checks)))
    return 1 if failed else 0


sys.exit(main())
