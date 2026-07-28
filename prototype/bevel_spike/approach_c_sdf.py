"""
Approach C -- implicit / SDF rounding.

    python approach_c_sdf.py

Three routes, all measured:

  C1  manifold3d's Manifold.level_set on an analytic SDF of the fixture, with
      rounded-box primitives and a smooth subtraction. Guaranteed-manifold
      output by construction, but the SDF is a Python callable evaluated per
      grid point.
  C2  the same SDF sampled on a numpy grid and meshed with skimage's marching
      cubes. Vectorised, so it isolates how much of C1's runtime is the Python
      callback rather than the algorithm.
  C3  Minkowski morphology inside Manifold itself: erode then dilate with a
      sphere rounds convex edges, dilate then erode rounds concave ones. This
      needs no SDF at all and stays in the kernel DESIGN.md already chose.

The provenance question for this whole family is structural, and the code
below measures it rather than arguing it: every one of these routes returns a
brand new mesh, so the only provenance available is re-derived from geometry.
"""

import json
import math
import os
import sys
import time

import numpy as np
import manifold3d as m

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

RADIUS = 0.02


# ------------------------------------------------------------------- the SDF
# manifold3d's level_set wants POSITIVE INSIDE, which is the opposite of the
# usual convention. Everything below is written negative-inside and negated
# once, at the boundary.

def sd_round_box(p, half, r):
    """Rounded box, negative inside. Outer dimensions are preserved: the
    surface sits r outside a box shrunk by r, so the flat faces stay at +-half
    and only the edges pick up the radius."""
    q = np.abs(p) - (np.asarray(half) - r)
    return (np.linalg.norm(np.maximum(q, 0.0), axis=-1)
            + np.minimum(np.max(q, axis=-1), 0.0) - r)


def smooth_max(a, b, k):
    """Polynomial smooth max. k = 0 falls back to the sharp max, which is what
    a plain boolean subtraction is."""
    if k <= 0:
        return np.maximum(a, b)
    h = np.clip(0.5 - 0.5 * (a - b) / k, 0.0, 1.0)
    return a * (1 - h) + b * h + k * h * (1 - h)


def fixture_sdf(fx, prim_radius=0.0, blend=0.0):
    """Negative-inside SDF of the slotted cube.

    prim_radius rounds each primitive's own edges -- that is what reaches the
    12 cube edges, which no smooth boolean can touch because they are not a
    boolean crease. blend rounds the crease the subtraction creates, i.e. the
    concave slot floor edges.
    """
    body_half = np.array([fx.half, fx.half, fx.half])
    cut_half = np.array([fx.slot_w / 2, fx.half * 1.4, fx.slot_depth])
    centers = [np.array([cx, 0.0, fx.half]) for cx in fx.centers]

    def f(p):
        p = np.atleast_2d(np.asarray(p, dtype=np.float64))
        d = sd_round_box(p, body_half, prim_radius)
        for c in centers:
            dc = sd_round_box(p - c, cut_half, prim_radius)
            d = smooth_max(d, -dc, blend)
        return d

    return f


# ------------------------------------------------------------------------ C1

def level_set_mesh(fx, edge_length, prim_radius, blend, pad=0.05):
    f = fixture_sdf(fx, prim_radius, blend)
    calls = [0]

    def scalar(x, y, z):                      # positive inside for Manifold
        calls[0] += 1
        return float(-f(np.array([x, y, z]))[0])

    h = fx.half + pad
    bounds = [-h, -h, -h, h, h, h]
    t0 = time.perf_counter()
    man = m.Manifold.level_set(scalar, bounds, edge_length)
    el = time.perf_counter() - t0
    mesh = man.to_mesh64()
    verts = np.array(mesh.vert_properties, dtype=np.float64, copy=True)[:, :3]
    tris = np.array(mesh.tri_verts, dtype=np.int64, copy=True)
    ids = sorted(set(np.asarray(mesh.run_original_id).tolist()))
    return verts, tris, {"seconds": el, "sdf_calls": calls[0],
                         "distinct_run_original_ids": len(ids)}


# ------------------------------------------------------------------------ C2

def marching_cubes_mesh(fx, edge_length, prim_radius, blend, pad=0.05):
    from skimage import measure
    f = fixture_sdf(fx, prim_radius, blend)
    h = fx.half + pad
    n = int(math.ceil(2 * h / edge_length)) + 1
    ax = np.linspace(-h, h, n)
    t0 = time.perf_counter()
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    vol = f(pts).reshape(n, n, n)
    t_sample = time.perf_counter() - t0
    t0 = time.perf_counter()
    verts, faces, _n, _v = measure.marching_cubes(vol, level=0.0)
    t_mc = time.perf_counter() - t0
    verts = verts * (2 * h / (n - 1)) - h
    return (np.asarray(verts, dtype=np.float64),
            np.asarray(faces, dtype=np.int64),
            {"grid": n, "sample_s": t_sample, "marching_cubes_s": t_mc,
             "sdf_calls": int(n ** 3)})


# ------------------------------------------------------------------------ C3

def minkowski_round(man, radius, mode, sphere_segments=32):
    """Morphological rounding, entirely inside Manifold.

      open   erode then dilate  -> rounds CONVEX edges
      close  dilate then erode  -> rounds CONCAVE edges
      both   close then open
    """
    m.set_circular_segments(sphere_segments)
    s = m.Manifold.sphere(radius, sphere_segments)
    t0 = time.perf_counter()
    if mode == "open":
        out = man.minkowski_difference(s).minkowski_sum(s)
    elif mode == "close":
        out = man.minkowski_sum(s).minkowski_difference(s)
    elif mode == "both":
        out = (man.minkowski_sum(s).minkowski_difference(s)
               .minkowski_difference(s).minkowski_sum(s))
    else:
        raise ValueError(mode)
    el = time.perf_counter() - t0
    mesh = out.to_mesh64()
    verts = np.array(mesh.vert_properties, dtype=np.float64, copy=True)[:, :3]
    tris = np.array(mesh.tri_verts, dtype=np.int64, copy=True)
    ids = sorted(set(np.asarray(mesh.run_original_id).tolist()))
    return verts, tris, {"seconds": el, "sphere_tris": s.num_tri(),
                         "distinct_run_original_ids": len(ids),
                         "status": C._status_name(out),
                         "empty": out.is_empty()}


# -------------------------------------------------- geometric re-attribution

def reattribute(fx, verts, tris, dist_tol, ang_tol_deg=8.0):
    """The only provenance available once the mesh has been rebuilt: guess.

    Match each triangle to a nominal reference plane by normal and offset. This
    can only work because the fixture is made of axis-aligned planes whose
    equations the program knows. It has no answer for a curved or freeform
    surface, and it cannot distinguish two coplanar surfaces of different
    ancestry.
    """
    n, _area = C.tri_normals(verts, tris)
    cen = C.tri_centroids(verts, tris)
    cos_tol = math.cos(math.radians(ang_tol_deg))
    tag = np.zeros(tris.shape[0], dtype=np.uint8)
    best = np.full(tris.shape[0], np.inf)
    for kind, axis, off in fx.planes():
        para = np.abs(n[:, axis]) >= cos_tol
        dist = np.abs(cen[:, axis] - off)
        hit = para & (dist <= dist_tol) & (dist < best)
        best = np.where(hit, dist, best)
        tag[hit] = C.TAG_CUT if kind == "cut" else C.TAG_BODY
    tag[tag == 0] = C.TAG_BEVEL
    return tag


def score(fx, verts, tris, before, label, referee_tol):
    tag = reattribute(fx, verts, tris, dist_tol=referee_tol)
    prov = C.provenance_report(fx, verts, tris, tag, before)
    ref_loose = C.referee(fx, verts, tris, dist_tol=referee_tol,
                          ang_tol_deg=8.0)
    chk = C.check_mesh(verts, tris, label)
    fid = C.plane_fidelity(fx, verts, tris, ang_tol_deg=8.0)
    fid.pop("per_plane")
    return {
        "check": chk,
        "provenance": prov,
        "referee_tolerance": referee_tol,
        "loose_referee_counts": ref_loose["counts"],
        "plane_fidelity": fid,
        "sharp_edges": C.sharp_edge_report(verts, tris, 60.0),
    }


def run():
    res = {"approach": "C_sdf", "radius": RADIUS,
           "manifold3d": "3.5.2"}
    fx = C.Fixture()
    man, bid, cid = fx.manifold()
    v0, t0, g0 = C.manifold_arrays(man, bid, cid)
    before = C.before_snapshot(fx, v0, t0, g0)
    res["before"] = before
    res["input_sharp_edges"] = C.sharp_edge_report(v0, t0, 60.0)

    # ----------------------------------------------- C1 Manifold.level_set
    c1 = {}
    for edge_length, prim_r, blend, key in (
            (0.03, 0.0, 0.0, "L=0.03 no rounding (baseline)"),
            (0.03, RADIUS, RADIUS, "L=0.03 rounded"),
            (0.02, RADIUS, RADIUS, "L=0.02 rounded"),
            (0.01, RADIUS, RADIUS, "L=0.01 rounded")):
        try:
            v, t, info = level_set_mesh(fx, edge_length, prim_r, blend)
            s = score(fx, v, t, before, "level_set %s" % key,
                      referee_tol=3 * edge_length)
            c1[key] = {
                "level_set": info,
                "tris": s["check"]["tris"],
                "raw_watertight": s["check"]["raw_watertight"],
                "coincident_vertex_positions": s["check"][
                    "coincident_vertex_positions"],
                "watertight_after_weld": s["check"]["watertight"],
                "manifold_status": s["check"]["manifold_status"],
                "components": s["check"]["components"],
                "genus": s["check"]["genus"],
                "volume": s["check"]["volume"],
                "reusable_as_boolean_operand": s["check"][
                    "reusable_as_boolean_operand"],
                "worst_flat_face_offset_error": s["plane_fidelity"][
                    "worst_max_offset_error"],
                "mean_flat_face_offset_error": s["plane_fidelity"][
                    "worst_mean_offset_error"],
                "planes_found": s["plane_fidelity"]["planes_found"],
                "sharp_edges_remaining": s["sharp_edges"],
                "sel_cut_faces_before": before["sel_cut_any_faces"],
                "sel_cut_faces_after_reattribution": s["provenance"][
                    "sel_cut_any_faces"],
                "expected_cut_faces": fx.n_cut_faces(),
                "reattribution_tolerance": s["referee_tolerance"],
            }
        except Exception as exc:                             # noqa: BLE001
            c1[key] = {"error": "%s: %s" % (type(exc).__name__, exc)}
    res["C1_level_set"] = c1

    # ------------------------------------------------ C2 marching cubes
    c2 = {}
    for edge_length in (0.02, 0.01, 0.005):
        try:
            v, t, info = marching_cubes_mesh(fx, edge_length, RADIUS, RADIUS)
            s = score(fx, v, t, before, "marching_cubes L=%g" % edge_length,
                      referee_tol=3 * edge_length)
            c2["L=%g" % edge_length] = {
                "sampling": info,
                "tris": s["check"]["tris"],
                "raw_watertight": s["check"]["raw_watertight"],
                "coincident_vertex_positions": s["check"][
                    "coincident_vertex_positions"],
                "watertight_after_weld": s["check"]["watertight"],
                "non_manifold_edges": s["check"]["non_manifold_edges"],
                "manifold_status": s["check"]["manifold_status"],
                "components": s["check"]["components"],
                "volume": s["check"]["volume"],
                "reusable_as_boolean_operand": s["check"][
                    "reusable_as_boolean_operand"],
                "worst_flat_face_offset_error": s["plane_fidelity"][
                    "worst_max_offset_error"],
                "sharp_edges_remaining": s["sharp_edges"],
                "sel_cut_faces_after_reattribution": s["provenance"][
                    "sel_cut_any_faces"],
                "expected_cut_faces": fx.n_cut_faces(),
            }
        except Exception as exc:                             # noqa: BLE001
            c2["L=%g" % edge_length] = {"error": "%s: %s"
                                        % (type(exc).__name__, exc)}
    res["C2_marching_cubes"] = c2

    # ------------------------------------------------------ C3 Minkowski
    c3 = {}
    for mode in ("open", "close", "both"):
        for segs in (16, 32):
            key = "%s,segments=%d" % (mode, segs)
            try:
                v, t, info = minkowski_round(man, RADIUS, mode, segs)
                s = score(fx, v, t, before, "minkowski %s" % key,
                          referee_tol=1e-4)
                chk = s["check"]
                c3[key] = {
                    "minkowski": info,
                    "tris": chk["tris"],
                    "raw_watertight": chk["raw_watertight"],
                    "raw_non_manifold_edges": chk["raw_non_manifold_edges"],
                    "coincident_vertex_positions": chk[
                        "coincident_vertex_positions"],
                    "watertight_after_weld": chk["watertight"],
                    "non_manifold_edges_after_weld": chk[
                        "non_manifold_edges"],
                    "degenerate_after_weld": chk["dropped_degenerate"],
                    "manifold_status": s["check"]["manifold_status"],
                    "components_after_weld": s["check"]["components"],
                    "volume": s["check"]["volume"],
                    "worst_flat_face_offset_error": s["plane_fidelity"][
                        "worst_max_offset_error"],
                    "planes_found": s["plane_fidelity"]["planes_found"],
                    "sharp_edges_remaining": s["sharp_edges"],
                    "sel_cut_faces_after_reattribution": s["provenance"][
                        "sel_cut_any_faces"],
                    "expected_cut_faces": fx.n_cut_faces(),
                    "false_positive_real_tris": s["provenance"][
                        "false_positive_real_tris"],
                }
            except Exception as exc:                         # noqa: BLE001
                c3[key] = {"error": "%s: %s" % (type(exc).__name__, exc)}
    res["C3_minkowski"] = c3

    # Minkowski cost is documented as scaling with the product of face counts
    # for non-convex inputs. Measure that against slot count.
    scale = {}
    for slots in (1, 3, 5):
        fxd = C.Fixture(slots=slots, slot_w=0.05 if slots > 1 else 0.15)
        md, _b, _c = fxd.manifold()
        try:
            t0 = time.perf_counter()
            v, t, info = minkowski_round(md, RADIUS, "open", 16)
            ck = C.check_mesh(v, t, "mink")
            scale["slots=%d" % slots] = {
                "in_tris": md.num_tri(),
                "seconds": time.perf_counter() - t0,
                "out_tris": int(t.shape[0]),
                "status": info["status"],
                "raw_watertight": ck["raw_watertight"],
                "coincident_vertex_positions": ck[
                    "coincident_vertex_positions"],
            }
        except Exception as exc:                             # noqa: BLE001
            scale["slots=%d" % slots] = {"error": "%s: %s"
                                         % (type(exc).__name__, exc)}
    res["C3_minkowski_scaling"] = scale

    # ------------------------- does the smooth boolean bulge the flat faces?
    # A smooth subtraction rounds by ADDING material, so at the convex slot
    # opening it should raise a lip on the top face rather than round the edge.
    try:
        v, t, _i = level_set_mesh(fx, 0.02, RADIUS, RADIUS)
        n, area = C.tri_normals(v, t)
        cen = C.tri_centroids(v, t)
        near_top = (n[:, 2] > 0.9) & (np.abs(cen[:, 2] - fx.half) < 0.03)
        d = cen[near_top, 2] - fx.half
        res["C6_top_face_signed_deviation"] = {
            "n_tris": int(np.sum(near_top)),
            "max_outward_mm": float(np.max(d) * 1000),
            "max_inward_mm": float(np.min(d) * 1000),
            "requested_blend_mm": RADIUS * 1000,
            "note": "Positive is outward. The measured answer is that nothing "
                    "bulges out and the top face is pulled IN by far more than "
                    "the requested blend, so the smooth-min parameter is not a "
                    "width. Compare OCCT, where the measured fillet area "
                    "matches the analytic quarter cylinder to 0.05%.",
        }
    except Exception as exc:                                 # noqa: BLE001
        res["C6_top_face_signed_deviation"] = {"error": str(exc)}

    # -------------------------------------------- can we SDF a mesh at all?
    api = [x for x in dir(m.Manifold) if not x.startswith("_")]
    res["C4_mesh_to_sdf"] = {
        "manifold_api": api,
        "has_signed_distance_query": any(
            k in api for k in ("signed_distance", "distance", "sdf",
                               "closest_point", "nearest")),
        "note": ("Manifold.level_set takes a callable SDF, not a mesh, and the "
                 "Manifold API exposes no signed-distance or closest-point "
                 "query. So an implicit round cannot be applied to an "
                 "arbitrary boolean result -- it needs the whole model "
                 "expressed as an SDF."),
    }

    # per-edge control question
    res["C5_control"] = {
        "per_edge_width": False,
        "granularity": "one radius per primitive (rounded box) plus one blend "
                       "radius per boolean; no per-edge and no predicate",
        "edge_predicate_selection": False,
        "segment_count": "not a parameter -- tessellation density is the grid "
                         "edge length, which is global",
    }
    return res


def report(res):
    C.rule("APPROACH C -- implicit / SDF rounding")
    C.kv("input sharp edges (dihedral>60)", res["input_sharp_edges"])
    C.kv("faces(#cut_face) before", res["before"]["sel_cut_any_faces"])
    print()
    for k, v in res["C1_level_set"].items():
        C.kv("level_set %s" % k, "")
        for kk, vv in v.items():
            C.kv("    " + kk, vv)
    print()
    for k, v in res["C2_marching_cubes"].items():
        C.kv("marching_cubes %s" % k, "")
        for kk, vv in v.items():
            C.kv("    " + kk, vv)
    print()
    for k, v in res["C3_minkowski"].items():
        C.kv("minkowski %s" % k, "")
        for kk, vv in v.items():
            C.kv("    " + kk, vv)
    print()
    for k, v in res["C3_minkowski_scaling"].items():
        C.kv("minkowski scaling %s" % k, v)
    C.kv("top face signed deviation (L=0.02 rounded)",
         res["C6_top_face_signed_deviation"])
    print()
    C.kv("signed-distance query in Manifold API",
         res["C4_mesh_to_sdf"]["has_signed_distance_query"])
    C.kv("control granularity", res["C5_control"])


if __name__ == "__main__":
    r = run()
    report(r)
    print("@@RESULT@@")
    print(json.dumps(r, default=str))
