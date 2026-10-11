"""
Runs inside Blender: `blender -b -P _bevel_worker.py -- in.npz out.npz params.json`

Manifold ships no bevel, so DESIGN.md section 11 routes it through bmesh in a
subprocess. This file is the only place bpy is imported.

The tag discipline is the point. bmesh interpolates ancestor attributes onto new
geometry, which is what silently widened `faces(#cut_face)` from 3 matches to 78
in the spike. New faces here get the bevel id and nothing else, so a selector
keeps meaning what the program said.
"""

import json
import sys

import bmesh
import bpy
import numpy as np

TAG_LAYER = "lathe_part"


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    in_path, out_path, params_path = argv[0], argv[1], argv[2]
    p = json.load(open(params_path))

    data = np.load(in_path)
    verts = data["verts"]
    tris = data["tris"]
    part_id = data["part_id"]

    bm = bmesh.new()
    bverts = [bm.verts.new((float(x), float(y), float(z))) for x, y, z in verts]
    bm.verts.ensure_lookup_table()
    lay = bm.faces.layers.int.new(TAG_LAYER)
    for t, pid in zip(tris, part_id):
        f = bm.faces.new((bverts[int(t[0])], bverts[int(t[1])], bverts[int(t[2])]))
        f[lay] = int(pid)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    # Required. Faces built with faces.new() carry a zero normal until this runs,
    # and calc_face_angle reads f.normal -- so every edge would measure as
    # coplanar and the merge below would flatten the whole solid.
    bm.normal_update()

    report = {"in_tris": len(tris), "in_verts": len(verts)}

    # 1. Merge coplanar triangles back into logical faces. Manifold hands us a
    #    triangulation, and bevelling a triangulated cube would round the
    #    diagonals too. Only merge across edges whose faces share a part, so a
    #    cut face coplanar with a body face stays two distinct faces.
    flat = []
    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue
        a, b = e.link_faces
        if a[lay] != b[lay]:
            continue
        if e.calc_face_angle(0.0) < np.radians(p.get("coplanar_deg", 1.0)):
            flat.append(e)
    if flat:
        # dissolve_edges, not dissolve_limit. dissolve_limit re-derives its own
        # edge set from the angle limit and silently declined to merge anything
        # here; this says exactly which edges to remove.
        bmesh.ops.dissolve_edges(bm, edges=flat, use_verts=False,
                                 use_face_split=False)
        bm.faces.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.normal_update()
    report["coplanar_edges_merged"] = len(flat)
    report["faces_after_dissolve"] = len(bm.faces)

    # 2. Pick edges by predicate, exactly as the program asked.
    only = p.get("only_parts")
    min_d = np.radians(p.get("min_dihedral", 30.0))
    max_d = np.radians(p.get("max_dihedral", 180.0))
    min_len = p.get("min_length", 0.0)
    picked = []
    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue
        ang = e.calc_face_angle(0.0)
        if not (min_d < ang < max_d):
            continue
        if e.calc_length() <= min_len:
            continue
        if only is not None:
            if not all(f[lay] in only for f in e.link_faces):
                continue
        picked.append(e)
    report["edges_selected"] = len(picked)

    if not picked:
        report["status"] = "no_edges_selected"
        _write(out_path, bm, lay, report)
        return

    # Where the picked edges are *before* the bevel moves everything, with the
    # plane of each adjacent face and the direction into it. The wrapper measures
    # the width actually delivered against these, because bmesh does not say
    # when clamp_overlap shrank it. Row layout: a, b, then (normal, inward) for
    # each of the two faces.
    picked_xyz = []
    for e in picked:
        row = [list(e.verts[0].co), list(e.verts[1].co)]
        for f in e.link_faces:
            # The edge as this face winds it: inward is to its left.
            d_f = None
            for loop in f.loops:
                if loop.edge == e:
                    d_f = loop.link_loop_next.vert.co - loop.vert.co
                    break
            inward = f.normal.cross(d_f).normalized()
            row += [list(f.normal), list(inward)]
        picked_xyz.append(row)

    # 3. Bevel.
    res = bmesh.ops.bevel(
        bm, geom=picked, offset=float(p["width"]), offset_type="OFFSET",
        segments=int(p.get("segments", 2)), profile=float(p.get("profile", 0.5)),
        affect="EDGES", clamp_overlap=bool(p.get("clamp", True)),
        miter_outer="ARC" if p.get("miter_arc", True) else "SHARP",
        harden_normals=False,
    )
    new_faces = [f for f in res.get("faces", []) if f.is_valid]
    report["faces_created"] = len(new_faces)

    # 4. THE tag rule. Non-inheriting: the new faces carry the bevel id only.
    #    Leaving bmesh's interpolated masks in place is precisely the section 13
    #    drift, measured at 97 wrongly attributed triangles.
    bevel_id = int(p["bevel_id"])
    for f in new_faces:
        f[lay] = bevel_id

    # 5. Back to triangles for Manifold. Triangles of a bevel face are still
    #    bevel faces, so inheritance is correct here and only here.
    bm.faces.ensure_lookup_table()
    bmesh.ops.triangulate(bm, faces=list(bm.faces), quad_method="BEAUTY",
                          ngon_method="BEAUTY")
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    report["status"] = "ok"
    _write(out_path, bm, lay, report, picked_xyz)


def _write(out_path, bm, lay, report, picked_xyz=()):
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    for i, v in enumerate(bm.verts):
        v.index = i
    verts = np.array([[v.co.x, v.co.y, v.co.z] for v in bm.verts], dtype=np.float64)
    tris, parts = [], []
    for f in bm.faces:
        if len(f.verts) != 3:
            continue
        tris.append([v.index for v in f.verts])
        parts.append(f[lay])
    report["out_tris"] = len(tris)
    report["out_verts"] = len(verts)
    np.savez(
        out_path,
        verts=verts,
        tris=np.array(tris, dtype=np.int64).reshape(-1, 3),
        part_id=np.array(parts, dtype=np.int64),
        picked_edges=np.array(picked_xyz, dtype=np.float64).reshape(-1, 6, 3),
        report=np.array([json.dumps(report)]),
    )
    bm.free()
    print("LATHE_BEVEL_REPORT " + json.dumps(report))


main()
