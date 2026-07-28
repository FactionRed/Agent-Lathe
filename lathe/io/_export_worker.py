"""
Runs inside Blender: `blender -b -P _export_worker.py -- job.json`

Only for formats not worth hand-writing: glTF, FBX, USD, PLY. OBJ and STL are
written natively in export.py.

Transforms are applied before the exporter runs, and the FBX path passes
apply_unit_scale plus FBX_SCALE_ALL. DESIGN.md section 7 E104 exists because
this is where scale bugs are born.
"""

import json
import sys

import bpy
import numpy as np


def main():
    job = json.load(open(sys.argv[sys.argv.index("--") + 1]))
    bpy.ops.wm.read_factory_settings(use_empty=True)

    data = np.load(job["mesh_npz"])
    verts, tris, part_id = data["verts"], data["tris"], data["part_id"]
    tag_of = {int(k): v for k, v in job["tags"].items()}

    me = bpy.data.meshes.new(job["name"])
    me.from_pydata([tuple(map(float, v)) for v in verts], [],
                   [tuple(map(int, t)) for t in tris])
    me.update()
    obj = bpy.data.objects.new(job["name"], me)
    bpy.context.scene.collection.objects.link(obj)

    # One material per tag, named so the target engine can bind to it.
    slots, order = {}, []
    for pid in sorted(set(int(p) for p in part_id)):
        tag = tag_of.get(pid, "part_%d" % pid)
        mat = bpy.data.materials.new("M_%s" % tag)
        mat.use_nodes = True
        slots[pid] = len(order)
        order.append(mat)
    for mat in order:
        me.materials.append(mat)
    for poly, pid in zip(me.polygons, part_id):
        poly.material_index = slots[int(pid)]

    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    # Belt and braces: the geometry already carries its transform, and this
    # guarantees scale is (1,1,1) and rotation is 0 at the moment of writing.
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    out, fmt = job["out"], job["fmt"]
    if fmt in ("glb", "gltf"):
        bpy.ops.export_scene.gltf(
            filepath=out,
            export_format="GLB" if fmt == "glb" else "GLTF_SEPARATE",
            use_selection=True, export_apply=True,
            export_yup=True, export_materials="EXPORT")
    elif fmt == "fbx":
        bpy.ops.export_scene.fbx(
            filepath=out, use_selection=True,
            apply_unit_scale=True, apply_scale_options="FBX_SCALE_ALL",
            axis_forward="-Y", axis_up="Z",
            mesh_smooth_type="FACE", use_mesh_modifiers=True,
            add_leaf_bones=False, bake_anim=False)
    elif fmt in ("usd", "usdc"):
        bpy.ops.wm.usd_export(filepath=out, selected_objects_only=True)
    elif fmt == "ply":
        bpy.ops.wm.ply_export(filepath=out, export_selected_objects=True)
    else:
        raise SystemExit("unsupported format %r in the Blender path" % fmt)

    print("LATHE_EXPORTED %s -> %s" % (fmt, out))


main()
