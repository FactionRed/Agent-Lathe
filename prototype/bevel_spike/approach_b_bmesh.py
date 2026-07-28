"""
Approach B -- wrap Blender's bmesh bevel, headless.

    python approach_b_bmesh.py

Blender is a subprocess, not a library: `blender -b -P blender_bevel.py`. The
mesh crosses the boundary as .npz together with a per-triangle provenance
bitmask, so this measures the whole realistic pipeline
Manifold -> npz -> bmesh -> bevel -> npz -> Manifold, including the process
start cost, which is the part that actually shows up in a build.

Measurements:
  B1  tag_mode=raw        what bmesh does to ancestor attributes on its own
  B2  tag_mode=inherit    a naive wrapper
  B3  tag_mode=exclusive  the DESIGN.md section 5.2 non-inheriting default
  B4  per-edge width, which bmesh.ops.bevel does not offer, via two passes
  B5  edge predicate, including convex/concave
  B6  cost and behaviour on denser input
"""

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLENDER = os.environ.get(
    "LATHE_BLENDER",
    r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")
SCRIPT = os.path.join(HERE, "blender_bevel.py")

RADIUS, SEGMENTS = 0.02, 3


def blender_available():
    return os.path.exists(BLENDER)


def measure_startup():
    """Blender process start plus quit, with no work done. This is the floor
    on every bevel call if Blender is a subprocess."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "noop.py")
        open(p, "w").write("pass\n")
        t0 = time.perf_counter()
        subprocess.run([BLENDER, "-b", "-P", p], capture_output=True,
                       check=False)
        return time.perf_counter() - t0


def call_blender(verts, tris, tri_tag, **cfg):
    d = tempfile.mkdtemp(prefix="lathe_bevel_")
    in_npz = os.path.join(d, "in.npz")
    out_npz = os.path.join(d, "out.npz")
    out_json = os.path.join(d, "out.json")
    cfg_path = os.path.join(d, "cfg.json")
    C.dump_mesh(in_npz, verts, tris, tri_tag)
    cfg.update({"in_npz": in_npz, "out_npz": out_npz, "out_json": out_json})
    json.dump(cfg, open(cfg_path, "w"))
    t0 = time.perf_counter()
    proc = subprocess.run([BLENDER, "-b", "-P", SCRIPT, "--", cfg_path],
                          capture_output=True, text=True, check=False)
    wall = time.perf_counter() - t0
    if not os.path.exists(out_json):
        raise RuntimeError("blender produced no result\nSTDOUT:\n%s\nSTDERR:\n%s"
                           % (proc.stdout[-3000:], proc.stderr[-3000:]))
    info = json.load(open(out_json))
    info["wall_s"] = wall
    info["process_overhead_s"] = wall - info["script_s"]
    z = np.load(out_npz)
    v = np.array(z["verts"], dtype=np.float64)
    t = np.array(z["tris"], dtype=np.int64)
    g = np.array(z["tri_tag"], dtype=np.uint8)
    face_id = (np.array(z["face_id"], dtype=np.int64)
               if "face_id" in z.files else None)
    return v, t, g, face_id, info


def one_run(fx, verts, tris, tri_tag, before, label, downstream=False, **cfg):
    v, t, g, face_id, info = call_blender(verts, tris, tri_tag, **cfg)
    chk = C.check_mesh(v, t, label)
    prov = C.provenance_report(fx, v, t, g, before)
    fid = C.plane_fidelity(fx, v, t)
    fid.pop("per_plane")
    out = {
        "blender_info": info,
        "check": chk,
        "provenance": prov,
        "plane_fidelity": fid,
        "sharp_edges": C.sharp_edge_report(v, t, 60.0),
        "area_by_truth": C.referee(fx, v, t)["area_by_truth"],
    }
    if downstream:
        out["downstream"] = C.downstream_boolean_test(v, t, g, face_id)
    return out


def run():
    res = {"approach": "B_bmesh", "blender_path": BLENDER,
           "blender_available": blender_available(),
           "radius": RADIUS, "segments": SEGMENTS}
    if not res["blender_available"]:
        res["error"] = "blender not found at %s" % BLENDER
        return res

    res["startup_overhead_s"] = measure_startup()

    fx = C.Fixture()
    man, bid, cid = fx.manifold()
    v0, t0, g0 = C.manifold_arrays(man, bid, cid)
    before = C.before_snapshot(fx, v0, t0, g0)
    res["before"] = before

    # -------------------------------------------------- B1/B2/B3 tag modes
    modes = {}
    for mode in ("raw", "inherit", "exclusive"):
        modes[mode] = one_run(
            fx, v0, t0, g0, before, "bmesh %s" % mode,
            downstream=(mode == "exclusive"),
            dissolve=True, segments=SEGMENTS, tag_mode=mode,
            passes=[{"want": ["convex", "concave"], "offset": RADIUS}])
    res["B_tag_modes"] = modes

    # ------------------------------------------------------ B5 predicates
    pred = {}
    for want in (["convex", "concave"], ["convex"], ["concave"]):
        key = "+".join(want)
        try:
            pred[key] = one_run(
                fx, v0, t0, g0, before, "bmesh %s" % key,
                dissolve=True, segments=SEGMENTS, tag_mode="exclusive",
                passes=[{"want": want, "offset": RADIUS}])
        except Exception as exc:                             # noqa: BLE001
            pred[key] = {"error": str(exc)}
    res["B5_predicate"] = pred

    # ------------------------------------------- B4 per-edge width, 2 passes
    try:
        res["B4_two_pass_widths"] = one_run(
            fx, v0, t0, g0, before, "bmesh two-pass",
            dissolve=True, segments=SEGMENTS, tag_mode="exclusive",
            passes=[{"want": ["convex"], "offset": 0.03},
                    {"want": ["concave"], "offset": 0.008}])
        res["B4_note"] = (
            "bmesh.ops.bevel takes one scalar offset, so per-edge width needs "
            "one call per width. The second call re-resolves its predicate "
            "against topology the first call already changed.")
    except Exception as exc:                                 # noqa: BLE001
        res["B4_two_pass_widths"] = {"error": str(exc)}

    # ------------------------------------------------- no-dissolve variant
    try:
        res["B_no_dissolve"] = one_run(
            fx, v0, t0, g0, before, "bmesh no dissolve",
            dissolve=False, segments=SEGMENTS, tag_mode="exclusive",
            passes=[{"want": ["convex", "concave"], "offset": RADIUS}])
    except Exception as exc:                                 # noqa: BLE001
        res["B_no_dissolve"] = {"error": str(exc)}

    # ----------------------------------------------------------- B6 density
    dense = {}
    cases = [("slots=1", C.Fixture(), None),
             ("slots=5,slot_w=0.05", C.Fixture(slots=5, slot_w=0.05), None),
             ("slots=5,slot_w=0.10", C.Fixture(slots=5, slot_w=0.10), None),
             ("slots=1 refined 0.05", C.Fixture(), 0.05),
             ("slots=1 refined 0.02", C.Fixture(), 0.02),
             ("slots=1 refined 0.01", C.Fixture(), 0.01)]
    for key, fxd, refine in cases:
        try:
            md, b_, c_ = fxd.manifold()
            if refine:
                md = md.refine_to_length(refine)
            vv, tt, gg = C.manifold_arrays(md, b_, c_)
            bsnap = C.before_snapshot(fxd, vv, tt, gg)
            r = one_run(fxd, vv, tt, gg, bsnap, key,
                        dissolve=True, segments=SEGMENTS,
                        tag_mode="exclusive",
                        passes=[{"want": ["convex", "concave"],
                                 "offset": RADIUS}])
            dense[key] = {
                "in_tris": int(tt.shape[0]),
                "out_tris": r["check"]["tris"],
                "wall_s": r["blender_info"]["wall_s"],
                "in_blender_script_s": r["blender_info"]["script_s"],
                "bmesh_build_s": r["blender_info"]["build_s"],
                "bevel_s": r["blender_info"]["bevel_s"],
                "edges_selected": r["blender_info"]["passes"][0][
                    "edges_selected"],
                "dissolve": r["blender_info"].get("dissolve"),
                "watertight": r["check"]["watertight"],
                "manifold_status": r["check"]["manifold_status"],
                "reusable_as_boolean_operand": r["check"][
                    "reusable_as_boolean_operand"],
                "sel_cut_faces_before": bsnap["sel_cut_any_faces"],
                "sel_cut_faces_after": r["provenance"]["sel_cut_any_faces"],
                "expected_cut_faces": fxd.n_cut_faces(),
                "provenance_exact": r["provenance"]["provenance_exact"],
                "provenance_exact_ignoring_slivers": r["provenance"][
                    "provenance_exact_ignoring_slivers"],
                "false_positive_tris": r["provenance"]["false_positive_tris"],
                "false_positive_real_tris": r["provenance"][
                    "false_positive_real_tris"],
                "sliver_tris": r["provenance"]["sliver_tris"],
                "false_negative_tris": r["provenance"]["false_negative_tris"],
                "fp_dist_to_cut_plane": r["provenance"].get(
                    "false_positive_dist_to_cut_plane"),
            }
        except Exception as exc:                             # noqa: BLE001
            dense[key] = {"error": "%s: %s" % (type(exc).__name__, exc)}
    res["B6_density"] = dense
    return res


def _summary(entry):
    if "error" in entry:
        return entry["error"]
    p, c = entry["provenance"], entry["check"]
    return ("cut_faces %d->%d (expect %d) drift=%s  fp=%d (real %d) fn=%d "
            "untagged=%d  tris=%d  watertight=%s  status=%s"
            % (p["before"]["sel_cut_any_faces"], p["sel_cut_any_faces"],
               p["expected_cut_faces"], "YES" if p["drift_any"] else "no",
               p["false_positive_tris"], p["false_positive_real_tris"],
               p["false_negative_tris"],
               p["untagged_tris"], c["tris"], c["watertight"],
               c["manifold_status"]))


def report(res):
    C.rule("APPROACH B -- bmesh.ops.bevel via headless Blender")
    if not res.get("blender_available"):
        C.kv("blender", res.get("error"))
        return
    C.kv("blender", res["blender_path"])
    C.kv("process startup+quit floor",
         "%.2f s" % res["startup_overhead_s"])
    C.kv("faces(#cut_face) before bevel",
         res["before"]["sel_cut_any_faces"])
    print()
    for mode, entry in res["B_tag_modes"].items():
        C.kv("tag_mode=%s" % mode, _summary(entry))
        if "blender_info" in entry:
            hist = entry["blender_info"]["passes"][0].get(
                "inherited_mask_histogram")
            C.kv("   mask bmesh put on new faces by itself", hist)
            C.kv("   wall / in-blender bevel",
                 "%.2f s / %.4f s" % (entry["blender_info"]["wall_s"],
                                      entry["blender_info"]["bevel_s"]))
            C.kv("   sharp edges left", entry["sharp_edges"])
        if "downstream" in entry:
            C.kv("   back into Manifold + one more boolean", "")
            for k, v in entry["downstream"].items():
                C.kv("       " + k, v)
    print()
    for k, entry in res["B5_predicate"].items():
        C.kv("predicate %s" % k, _summary(entry))
        if "blender_info" in entry:
            C.kv("   edges selected",
                 entry["blender_info"]["passes"][0]["edges_selected"])
    print()
    C.kv("two-pass per-edge width", _summary(res["B4_two_pass_widths"]))
    if "blender_info" in res["B4_two_pass_widths"]:
        for p in res["B4_two_pass_widths"]["blender_info"]["passes"]:
            C.kv("   pass %s offset=%s" % (p["want"], p["offset"]),
                 "edges=%d new_faces=%s" % (p["edges_selected"],
                                            p.get("new_faces")))
    C.kv("no-dissolve variant", _summary(res["B_no_dissolve"]))
    print()
    for k, v in res["B6_density"].items():
        C.kv("density %s" % k, v)


if __name__ == "__main__":
    r = run()
    report(r)
    print("@@RESULT@@")
    print(json.dumps(r, default=str))
