"""
Runs inside Blender: `blender -b -P _bake_worker.py -- job.json`

High-to-low baking. The low mesh carries the UVs and receives the maps; the high
mesh carries the detail. Cycles' selected-to-active bake does the ray casting.

The pre-bake checklist from the 3D modelling reference, applied rather than
recited:

  * transforms applied on both meshes, so scale cannot skew the rays
  * high and low occupy the same world position
  * low is smooth-shaded with an edge split at the UV-seam angle, because a low
    poly whose normals do not match the high is what produces wavy bakes
  * cage extrusion set from the low's own size, not a fixed number, so a 20 mm
    part and a 2 m part both get a sane ray distance
  * Flip Y for Unity, straight for Unreal -- the single most common reason a
    normal map looks inverted in one engine and fine in the other
"""

import json
import math
import os
import sys

import bmesh
import bpy
import numpy as np

TARGETS = ("normal", "ao", "curvature")


def build(name, npz_path, smooth_angle=None, uv=None):
    data = np.load(npz_path)
    verts, tris = data["verts"], data["tris"]
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [],
                   [tuple(map(int, t)) for t in tris])
    me.update()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)

    if uv is not None:
        layer = me.uv_layers.new(name="UV0")
        flat = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        for i, loop_uv in enumerate(layer.data):
            loop_uv.uv = (float(flat[i][0]), float(flat[i][1]))

    if smooth_angle is not None:
        for poly in me.polygons:
            poly.use_smooth = True
        mod = obj.modifiers.new("split", "EDGE_SPLIT")
        mod.split_angle = math.radians(smooth_angle)
    return obj


def main():
    job = json.load(open(sys.argv[sys.argv.index("--") + 1]))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    low = build("low", job["low_npz"], smooth_angle=job["smooth_angle"],
                uv=np.load(job["low_npz"])["uv"])
    high = build("high", job["high_npz"])

    for ob in (low, high):
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        ob.select_set(False)

    # A material with an image node is where the bake lands.
    mat = bpy.data.materials.new("M_bake")
    mat.use_nodes = True
    low.data.materials.append(mat)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    tex = nodes.new("ShaderNodeTexImage")

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = int(job["samples"])
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.cage_extrusion = float(job["cage_extrusion"])
    scene.render.bake.max_ray_distance = float(job["max_ray_distance"])
    scene.render.bake.margin = int(job["margin_px"])
    try:
        scene.render.bake.margin_type = "EXTEND"
    except Exception:
        pass
    scene.render.bake.use_clear = True

    written = {}
    for target in job["targets"]:
        res = int(job["resolution"])
        is_normal = target == "normal"
        img = bpy.data.images.new(
            "bake_%s" % target, res, res,
            alpha=False, float_buffer=False,
            is_data=is_normal)          # a normal map is data, not colour
        tex.image = img
        nodes.active = tex

        if is_normal:
            scene.cycles.bake_type = "NORMAL"
            scene.render.bake.normal_space = "TANGENT"
            # Unity expects the green channel inverted; Unreal does not.
            flip = job["target_engine"] == "unity"
            scene.render.bake.normal_g = "NEG_Y" if flip else "POS_Y"
            scene.render.bake.normal_r = "POS_X"
            scene.render.bake.normal_b = "POS_Z"
        elif target == "ao":
            scene.cycles.bake_type = "AO"
        else:
            # Curvature has no bake type of its own; the closest built-in that
            # reads surface shape is the pointiness-free normal difference, so
            # this bakes it as a plain normal in object space and lets the caller
            # derive curvature. Stated rather than silently substituted.
            scene.cycles.bake_type = "NORMAL"
            scene.render.bake.normal_space = "OBJECT"

        bpy.ops.object.select_all(action="DESELECT")
        high.select_set(True)
        low.select_set(True)
        bpy.context.view_layer.objects.active = low   # active receives the bake

        try:
            bpy.ops.object.bake(type=scene.cycles.bake_type)
        except RuntimeError as exc:
            print("LATHE_BAKE_ERROR %s %s" % (target, exc))
            continue

        out = os.path.join(job["out_dir"], "%s_%s.png" % (job["stem"], target))
        img.filepath_raw = out
        img.file_format = "PNG"
        img.save()
        written[target] = out
        print("LATHE_BAKED %s %s" % (target, out))

    print("LATHE_BAKE_DONE " + json.dumps(written))


main()
