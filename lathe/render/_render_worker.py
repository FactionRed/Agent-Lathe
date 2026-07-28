"""
Runs inside Blender: `blender -b -P _render_worker.py -- job.json`

DESIGN.md section 6 Tier 3. Renders a solid from several fixed views so the agent
sees shape rather than one flattering angle, plus the two modes the doc argues
matter more than beauty: a silhouette matte, because agents read form from
outline far more reliably than from shading, and wireframe-over-shaded, which
makes topology density legible.

Materials are keyed by tag, so the provenance from section 5.2 is what drives
shading. A solid that knows which triangles are `seat` and which are `leg` can
be shaded wood-on-wood without anyone painting anything.
"""

import json
import math
import sys

import bmesh
import bpy
import numpy as np

TAG_LAYER = "lathe_part"

# (base_color RGBA, roughness, metallic)
MATERIALS = {
    "wood_light": ((0.62, 0.42, 0.22, 1), 0.55, 0.0),
    "wood_mid":   ((0.44, 0.27, 0.13, 1), 0.50, 0.0),
    "wood_dark":  ((0.26, 0.15, 0.07, 1), 0.45, 0.0),
    "ceramic":    ((0.90, 0.89, 0.86, 1), 0.18, 0.0),
    "ceramic_blue": ((0.42, 0.55, 0.68, 1), 0.20, 0.0),
    "apple":      ((0.68, 0.09, 0.09, 1), 0.28, 0.0),
    "orange":     ((0.85, 0.42, 0.06, 1), 0.42, 0.0),
    "banana":     ((0.88, 0.76, 0.20, 1), 0.38, 0.0),
    "pear":       ((0.72, 0.75, 0.28, 1), 0.33, 0.0),
    "leaf":       ((0.22, 0.42, 0.16, 1), 0.45, 0.0),
    "cloth_red":  ((0.52, 0.16, 0.14, 1), 0.85, 0.0),
    "cloth_blue": ((0.20, 0.26, 0.42, 1), 0.85, 0.0),
    "cloth_tan":  ((0.60, 0.50, 0.36, 1), 0.85, 0.0),
    "skin":       ((0.80, 0.60, 0.47, 1), 0.42, 0.0),
    "hair_brown": ((0.20, 0.12, 0.07, 1), 0.60, 0.0),
    "leather":    ((0.32, 0.22, 0.15, 1), 0.55, 0.0),
    "metal":      ((0.62, 0.63, 0.66, 1), 0.32, 1.0),
    "default":    ((0.72, 0.72, 0.72, 1), 0.45, 0.0),
}


def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def make_material(name, key):
    spec = MATERIALS.get(key, MATERIALS["default"])
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = spec[0]
    bsdf.inputs["Roughness"].default_value = spec[1]
    bsdf.inputs["Metallic"].default_value = spec[2]
    return mat


def build_object(job):
    data = np.load(job["mesh_npz"])
    verts, tris, part_id = data["verts"], data["tris"], data["part_id"]
    tag_of = {int(k): v for k, v in job["tags"].items()}
    mat_for_tag = job["materials"]

    me = bpy.data.meshes.new(job["name"])
    me.from_pydata([tuple(map(float, v)) for v in verts], [],
                   [tuple(map(int, t)) for t in tris])
    me.update()
    obj = bpy.data.objects.new(job["name"], me)
    bpy.context.scene.collection.objects.link(obj)

    # One material slot per distinct tag, assigned per polygon from part_id.
    slots, order = {}, []
    for pid in sorted(set(int(p) for p in part_id)):
        tag = tag_of.get(pid, "part:%d" % pid)
        key = mat_for_tag.get(tag, job.get("default_material", "default"))
        slots[pid] = len(order)
        order.append(make_material("M_%s" % tag, key))
    for mat in order:
        obj.data.materials.append(mat)
    for poly, pid in zip(me.polygons, part_id):
        poly.material_index = slots[int(pid)]

    # Coplanar triangles merged, then smooth shading with an angle split. Without
    # this a revolved bowl reads as faceted even at 48 segments.
    if job.get("merge_coplanar", True):
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.normal_update()
        flat = [e for e in bm.edges
                if len(e.link_faces) == 2
                and e.calc_face_angle(0.0) < math.radians(0.5)
                and e.link_faces[0].material_index == e.link_faces[1].material_index]
        if flat:
            bmesh.ops.dissolve_edges(bm, edges=flat, use_verts=False,
                                     use_face_split=False)
        bm.to_mesh(me)
        bm.free()
        me.update()
    if job.get("smooth_angle_deg", 0) > 0:
        for poly in me.polygons:
            poly.use_smooth = True
        me.shade_smooth()
        mod = obj.modifiers.new("smooth_split", "EDGE_SPLIT")
        mod.split_angle = math.radians(job["smooth_angle_deg"])
    return obj


def add_ground(job, obj):
    """A neutral floor plus a metric grid, so scale is readable in the frame."""
    lo = [min(v[i] for v in [obj.bound_box[j] for j in range(8)]) for i in range(3)]
    z = job["bbox"][0][2]
    bpy.ops.mesh.primitive_plane_add(size=job["ground_size"], location=(0, 0, z))
    ground = bpy.context.active_object
    mat = bpy.data.materials.new("M_ground")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.9
    checker = nt.nodes.new("ShaderNodeTexChecker")
    checker.inputs["Color1"].default_value = (0.30, 0.30, 0.32, 1)
    checker.inputs["Color2"].default_value = (0.35, 0.35, 0.37, 1)
    # One check per 100 mm: the grid *is* the ruler.
    checker.inputs["Scale"].default_value = job["ground_size"] / 0.1
    nt.links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
    ground.data.materials.append(mat)
    return ground


def setup_world(job, mode):
    world = bpy.data.worlds.new("W")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    if mode == "matte":
        bg.inputs["Color"].default_value = (1, 1, 1, 1)
        bg.inputs["Strength"].default_value = 1.0
    else:
        bg.inputs["Color"].default_value = (0.05, 0.055, 0.065, 1)
        bg.inputs["Strength"].default_value = 1.0


def add_lights(radius):
    """Three-quarter key, cool fill, rim. Sized to the subject."""
    d = max(radius, 0.05)
    # Power scales with d^2 so exposure holds as the subject changes size: the
    # lights sit at a fixed multiple of d, and irradiance falls off as 1/dist^2.
    specs = [
        ("key",  (2.2, -2.6, 2.8), 140.0, 2.2),
        ("fill", (-2.8, -1.4, 1.4), 42.0, 3.0),
        ("rim",  (-0.8, 2.9, 2.2), 65.0, 2.0),
    ]
    for name, pos, power, size in specs:
        light = bpy.data.lights.new(name, "AREA")
        light.energy = power * (d / 1.0) ** 2
        light.size = size * d
        ob = bpy.data.objects.new(name, light)
        ob.location = tuple(p * d for p in pos)
        bpy.context.scene.collection.objects.link(ob)
        direction = -np.array(ob.location, dtype=float)
        ob.rotation_euler = _look(direction)


def _look(direction):
    d = np.array(direction, dtype=float)
    n = np.linalg.norm(d)
    if n == 0:
        return (0, 0, 0)
    d /= n
    return (math.acos(-d[2]) if abs(d[2]) <= 1 else 0.0,
            0.0,
            math.atan2(-d[0], d[1]))


VIEWS = {
    "front": (0.0, 0.0),
    "back":  (180.0, 0.0),
    "side":  (90.0, 0.0),
    "top":   (0.0, 89.0),
    "iso":   (38.0, 26.0),
    "iso2":  (-52.0, 20.0),
    "low":   (24.0, 6.0),
}


def place_camera(view, center, radius, ortho):
    az, el = VIEWS[view]
    a, e = math.radians(az), math.radians(el)
    dist = radius * 5.0
    loc = (center[0] + dist * math.sin(a) * math.cos(e),
           center[1] - dist * math.cos(a) * math.cos(e),
           center[2] + dist * math.sin(e))
    cam_data = bpy.data.cameras.new("cam")
    if ortho:
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = radius * 2.75
    else:
        cam_data.lens = 52
    cam = bpy.data.objects.new("cam", cam_data)
    cam.location = loc
    bpy.context.scene.collection.objects.link(cam)
    cam.rotation_euler = _look(np.array(center) - np.array(loc))
    bpy.context.scene.camera = cam
    return cam


def apply_mode(mode, obj, ground, radius=1.0):
    scene = bpy.context.scene
    if mode == "matte":
        # Flat black on white. The doc's claim is that outline reads more
        # reliably than shading, and shading is exactly what this removes.
        if ground:
            bpy.data.objects.remove(ground)
        mat = bpy.data.materials.new("M_matte")
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        emit = nt.nodes.new("ShaderNodeEmission")
        emit.inputs["Color"].default_value = (0.04, 0.04, 0.05, 1)
        emit.inputs["Strength"].default_value = 1.0
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        for poly in obj.data.polygons:
            poly.material_index = 0
    elif mode == "wire":
        mod = obj.modifiers.new("wire", "WIREFRAME")
        # Scale with the subject. A fixed 1.5 mm was invisible on a chair and
        # would have swallowed a 20 mm part whole.
        mod.thickness = max(radius * 0.006, 1e-4)
        mod.use_replace = False
        mod.material_offset = len(obj.data.materials)
        mat = bpy.data.materials.new("M_wire")
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.02, 0.75, 0.85, 1)
        bsdf.inputs["Roughness"].default_value = 0.4
        obj.data.materials.append(mat)


def main():
    job = json.load(open(sys.argv[sys.argv.index("--") + 1]))
    lo, hi = job["bbox"]
    center = [(a + b) / 2 for a, b in zip(lo, hi)]
    radius = max(max(b - a for a, b in zip(lo, hi)), 1e-3) * 0.62

    for mode in job["modes"]:
        for view in job["views"]:
            clear()
            scene = bpy.context.scene
            obj = build_object(job)
            ground = None
            if mode == "shaded" and job.get("ground", True):
                ground = add_ground(job, obj)
            setup_world(job, mode)
            if mode != "matte":
                add_lights(radius)
            apply_mode(mode, obj, ground, radius)
            place_camera(view, center, radius, job.get("ortho", False)
                         or view in ("front", "side", "top", "back"))

            scene.render.engine = "CYCLES"
            scene.cycles.samples = job["samples"] if mode != "matte" else 8
            scene.cycles.use_denoising = True
            try:
                scene.cycles.device = "CPU"
            except Exception:
                pass
            scene.render.resolution_x = job["size"]
            scene.render.resolution_y = job["size"]
            scene.render.film_transparent = False
            scene.view_settings.view_transform = (
                "Standard" if mode == "matte" else "AgX")
            scene.render.filepath = job["out_pattern"] % (mode, view)
            bpy.ops.render.render(write_still=True)
            print("LATHE_RENDERED %s %s" % (mode, view))


main()
