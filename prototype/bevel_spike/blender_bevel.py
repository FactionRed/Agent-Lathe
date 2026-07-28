"""
The in-Blender half of approach B. Runs under Blender's own Python:

    blender -b -P blender_bevel.py -- <config.json>

Reads a mesh plus a per-triangle provenance bitmask from .npz, bevels edges
chosen by predicate, and writes the result back as .npz. No bpy.ops, no mode
switches, no selection state -- everything is a function of explicit arguments,
per DESIGN.md section 2.

Tag handling is the point of the exercise, so it is a switch:

  raw        set the face attribute before the bevel and then touch nothing.
             Whatever the new faces come back carrying is bmesh's own
             inheritance behaviour, which is the thing DESIGN.md section 5.2
             accuses of causing silent selector drift.
  inherit    raw, plus OR in the #bevel bit -- a naive implementation.
  exclusive  new faces get the #bevel bit and nothing else -- the DESIGN.md
             section 5.2 default.
"""

import json
import math
import sys
import time

import bmesh
import bpy
import numpy as np

TAG_ATTR = "lathe_tag"
FID_ATTR = "lathe_fid"
TAG_BODY, TAG_CUT, TAG_BEVEL = 1, 2, 4


def build_bmesh(verts, tris, tri_tag):
    bm = bmesh.new()
    bverts = [bm.verts.new((float(p[0]), float(p[1]), float(p[2])))
              for p in verts]
    bm.verts.ensure_lookup_table()
    lay = bm.faces.layers.int.new(TAG_ATTR)
    for i, t in enumerate(tris):
        try:
            f = bm.faces.new((bverts[t[0]], bverts[t[1]], bverts[t[2]]))
        except ValueError:
            continue                      # duplicate face, skip
        f[lay] = int(tri_tag[i])
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    # Faces created with bm.faces.new() have a ZERO normal until this call, and
    # BMEdge.calc_face_angle then returns 0 for every edge -- so the whole
    # `edges(dihedral > 60deg)` predicate silently selects nothing and
    # dissolve_limit silently dissolves nothing. Costly to notice, because
    # dissolve_limit updates normals as a side effect and the next read looks
    # correct.
    bm.normal_update()
    return bm, lay


def dissolve_coplanar(bm, lay, angle_deg=0.05):
    """Merge coplanar triangles into n-gons, so the bevel sees logical faces
    the way Manifold's face_id does rather than the triangulation."""
    before = len(bm.faces)
    bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(angle_deg),
                             use_dissolve_boundaries=False,
                             verts=[], edges=list(bm.edges), delimit=set())
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.normal_update()
    untagged = sum(1 for f in bm.faces if f[lay] == 0)
    return {"faces_before": before, "faces_after": len(bm.faces),
            "untagged_after_dissolve": untagged}


def select_edges(bm, min_dihedral_deg=60.0, want=("convex", "concave")):
    out = []
    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue
        ang = math.degrees(e.calc_face_angle(0.0))
        if ang <= min_dihedral_deg:
            continue
        signed = e.calc_face_angle_signed(0.0)
        kind = "convex" if signed > 0 else "concave"
        if kind not in want:
            continue
        out.append(e)
    return out


def do_bevel(bm, lay, edges, offset, segments, mode):
    res = bmesh.ops.bevel(bm, geom=edges, offset=offset,
                          offset_type="OFFSET", profile_type="SUPERELLIPSE",
                          segments=segments, profile=0.5, affect="EDGES",
                          clamp_overlap=True, loop_slide=True,
                          harden_normals=False)
    new_faces = res.get("faces", [])
    inherited = [int(f[lay]) for f in new_faces]
    if mode == "inherit":
        for f in new_faces:
            f[lay] = int(f[lay]) | TAG_BEVEL
    elif mode == "exclusive":
        for f in new_faces:
            f[lay] = TAG_BEVEL
    elif mode != "raw":
        raise ValueError("unknown tag mode %r" % mode)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.normal_update()
    hist = {}
    for v in inherited:
        hist[str(v)] = hist.get(str(v), 0) + 1
    return {"new_faces": len(new_faces),
            "inherited_mask_histogram": hist}


def export(bm, lay):
    # Stamp a logical-face id BEFORE triangulating, so the caller can rebuild
    # Manifold's face_id channel. Without it, coplanar analysis is the only way
    # to group triangles back into faces, and that fails on curved surfaces.
    fid_lay = bm.faces.layers.int.get(FID_ATTR) or bm.faces.layers.int.new(
        FID_ATTR)
    for i, f in enumerate(bm.faces):
        f[fid_lay] = i + 1
    bmesh.ops.triangulate(bm, faces=list(bm.faces), quad_method="BEAUTY",
                          ngon_method="BEAUTY")
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    V = np.array([[v.co.x, v.co.y, v.co.z] for v in bm.verts],
                 dtype=np.float64)
    T, G, F = [], [], []
    for f in bm.faces:
        loops = [l.vert.index for l in f.loops]
        if len(loops) != 3:
            continue
        T.append(loops)
        G.append(int(f[lay]) & 0xFF)
        F.append(int(f[fid_lay]))
    return (V, np.array(T, dtype=np.int64), np.array(G, dtype=np.uint8),
            np.array(F, dtype=np.int64))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    cfg = json.load(open(argv[0]))
    t_start = time.perf_counter()

    z = np.load(cfg["in_npz"])
    verts = np.array(z["verts"], dtype=np.float64)
    tris = np.array(z["tris"], dtype=np.int64)
    tri_tag = np.array(z["tri_tag"], dtype=np.uint8)

    out = {"blender": bpy.app.version_string,
           "in_tris": int(tris.shape[0])}

    t0 = time.perf_counter()
    bm, lay = build_bmesh(verts, tris, tri_tag)
    out["build_s"] = time.perf_counter() - t0

    if cfg.get("dissolve", True):
        out["dissolve"] = dissolve_coplanar(bm, lay)

    passes = cfg.get("passes", [{"want": ["convex", "concave"],
                                 "offset": 0.02}])
    out["passes"] = []
    t_bevel = 0.0
    for p in passes:
        edges = select_edges(bm, cfg.get("min_dihedral", 60.0),
                             want=tuple(p.get("want",
                                              ["convex", "concave"])))
        info = {"want": p.get("want"), "offset": p["offset"],
                "edges_selected": len(edges)}
        if edges:
            t0 = time.perf_counter()
            info.update(do_bevel(bm, lay, edges, p["offset"],
                                 cfg.get("segments", 3),
                                 cfg.get("tag_mode", "exclusive")))
            t_bevel += time.perf_counter() - t0
        out["passes"].append(info)
    out["bevel_s"] = t_bevel

    t0 = time.perf_counter()
    V, T, G, F = export(bm, lay)
    out["export_s"] = time.perf_counter() - t0
    bm.free()

    np.savez(cfg["out_npz"], verts=V, tris=T, tri_tag=G, face_id=F)
    out["out_tris"] = int(T.shape[0])
    out["script_s"] = time.perf_counter() - t_start
    json.dump(out, open(cfg["out_json"], "w"), indent=1)


main()
