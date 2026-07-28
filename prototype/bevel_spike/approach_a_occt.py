"""
Approach A -- OpenCASCADE fillets (BRepFilletAPI_MakeFillet).

Needs an interpreter with BOTH `OCP` and `manifold3d`. pythonocc-core has no
PyPI wheels (conda only), so this uses `cadquery-ocp`, which does:

    py -3.12 -m venv occtvenv
    occtvenv\\Scripts\\python -m pip install cadquery-ocp manifold3d

Measured install cost is in README.md. Run:

    occtvenv\\Scripts\\python approach_a_occt.py

What this measures, in order:
  A1  provenance through a fillet, scored against the geometric referee
  A2  per-edge radius
  A3  edge selection by predicate, including convex/concave discrimination
  A4  cost and behaviour on a denser input (5-slot crate) and finer meshing
  A5  the architectural question: can OCCT fillet a solid that came out of
      Manifold as a triangle mesh, or must the model be built in OCCT?
"""

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from OCP.BRep import BRep_Tool                                    # noqa: E402
from OCP.BRepAdaptor import BRepAdaptor_Curve                     # noqa: E402
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut                       # noqa: E402
from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakeFace,          # noqa: E402
                                BRepBuilderAPI_MakePolygon,
                                BRepBuilderAPI_MakeSolid,
                                BRepBuilderAPI_Sewing)
from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet            # noqa: E402
from OCP.BRepGProp import BRepGProp, BRepGProp_Face               # noqa: E402
from OCP.GProp import GProp_GProps                                # noqa: E402
from OCP.BRepMesh import BRepMesh_IncrementalMesh                 # noqa: E402
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox                   # noqa: E402
from OCP.BRepTools import BRepTools                               # noqa: E402
from OCP.TopAbs import (TopAbs_EDGE, TopAbs_FACE,                 # noqa: E402
                        TopAbs_REVERSED, TopAbs_SHELL)
from OCP.TopExp import TopExp, TopExp_Explorer                    # noqa: E402
from OCP.TopLoc import TopLoc_Location                            # noqa: E402
from OCP.GCPnts import GCPnts_AbscissaPoint                       # noqa: E402
from OCP.TopoDS import TopoDS                                     # noqa: E402
from OCP.TopTools import (TopTools_IndexedDataMapOfShapeListOfShape,  # noqa: E402
                          TopTools_IndexedMapOfShape)
from OCP.gp import gp_Pnt, gp_Vec                                 # noqa: E402


# ------------------------------------------------------------------ helpers

def sub_shapes(shape, kind):
    ex = TopExp_Explorer(shape, kind)
    out = []
    while ex.More():
        out.append(ex.Current())
        ex.Next()
    return out


def faces_of(shape):
    return [TopoDS.Face_s(s) for s in sub_shapes(shape, TopAbs_FACE)]


def face_normal(face):
    """Outward normal at the parametric midpoint. BRepGProp_Face already
    accounts for face orientation, so do NOT flip it again -- doing so points
    every normal of a box the wrong way, which silently inverts every
    convexity test downstream."""
    u0, u1, v0, v1 = BRepTools.UVBounds_s(face)
    p, n = gp_Pnt(), gp_Vec()
    BRepGProp_Face(face).Normal((u0 + u1) / 2.0, (v0 + v1) / 2.0, p, n)
    n.Normalize()
    return np.array([n.X(), n.Y(), n.Z()]), np.array([p.X(), p.Y(), p.Z()])


def edge_midpoint(edge):
    """BRep_Tool.Curve_s uses OCCT output parameters, which OCP does not
    unpack, so go through the adaptor instead."""
    ad = BRepAdaptor_Curve(edge)
    p = ad.Value((ad.FirstParameter() + ad.LastParameter()) / 2.0)
    return np.array([p.X(), p.Y(), p.Z()])


def edge_length(edge):
    ad = BRepAdaptor_Curve(edge)
    return float(GCPnts_AbscissaPoint.Length_s(ad))


# --------------------------------------------------------------- build (A1)

def build_solid(fixture):
    """Body and cutters as native OCCT boxes, one BRepAlgoAPI_Cut per cutter.

    This is the point of architectural friction: the solid has to be built in
    OCCT. A Manifold result cannot be handed over without going through A5.
    """
    boxes = fixture.boxes()
    (_k, c, s) = boxes[0]
    shape = BRepPrimAPI_MakeBox(
        gp_Pnt(c[0] - s[0] / 2, c[1] - s[1] / 2, c[2] - s[2] / 2),
        s[0], s[1], s[2]).Shape()
    kinds = {}  # face index -> kind, refreshed after each cut
    fmap = TopTools_IndexedMapOfShape()
    for f in faces_of(shape):
        fmap.Add(f)
    kinds = {i: "body" for i in range(1, fmap.Extent() + 1)}

    for (_k, c, s) in boxes[1:]:
        cutter = BRepPrimAPI_MakeBox(
            gp_Pnt(c[0] - s[0] / 2, c[1] - s[1] / 2, c[2] - s[2] / 2),
            s[0], s[1], s[2]).Shape()
        op = BRepAlgoAPI_Cut(shape, cutter)
        op.Build()
        if not op.IsDone():
            raise RuntimeError("BRepAlgoAPI_Cut failed")
        new_shape = op.Shape()
        nmap = TopTools_IndexedMapOfShape()
        for f in faces_of(new_shape):
            nmap.Add(f)
        new_kinds = {}
        # carry existing labels forward through the boolean's history
        for i in range(1, fmap.Extent() + 1):
            f = TopoDS.Face_s(fmap.FindKey(i))
            if op.IsDeleted(f):
                continue
            got = list(op.Modified(f)) or [f]
            for r in got:
                j = nmap.FindIndex(r)
                if j:
                    new_kinds[j] = kinds[i]
        # cutter faces become cut surface
        for f in faces_of(cutter):
            if op.IsDeleted(f):
                continue
            got = list(op.Modified(f)) or [f]
            for r in got:
                j = nmap.FindIndex(r)
                if j:
                    new_kinds[j] = "cut"
        shape, fmap, kinds = new_shape, nmap, new_kinds
    unlabelled = [i for i in range(1, fmap.Extent() + 1) if i not in kinds]
    return shape, fmap, kinds, unlabelled


# ------------------------------------------------------- edge selection (A3)

def face_centroid(face):
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    c = props.CentreOfMass()
    return np.array([c.X(), c.Y(), c.Z()])


def select_edges(shape, min_dihedral_deg=60.0, want=("convex", "concave")):
    """Edges by predicate, exactly the `edges(dihedral > 60deg)` of the IR.

    Convexity: for a convex edge each face's centroid lies on the *inside* of
    the other face's plane, so

        s = n1 . (c2 - pe) + n2 . (c1 - pe)

    is negative. A first attempt probed a point offset along n1+n2 and asked
    the solid classifier -- that is degenerate, because that direction leaves
    the solid for convex AND concave edges alike, and it silently reported all
    24 edges of the slotted cube as convex.
    """
    em = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, em)
    out = []
    for i in range(1, em.Extent() + 1):
        edge = TopoDS.Edge_s(em.FindKey(i))
        adj = list(em.FindFromIndex(i))
        if len(adj) != 2:
            continue
        f1, f2 = TopoDS.Face_s(adj[0]), TopoDS.Face_s(adj[1])
        n1, _ = face_normal(f1)
        n2, _ = face_normal(f2)
        d = float(np.clip(np.dot(n1, n2), -1.0, 1.0))
        ang = math.degrees(math.acos(d))
        if ang <= min_dihedral_deg:
            continue
        pe = edge_midpoint(edge)
        s = float(np.dot(n1, face_centroid(f2) - pe)
                  + np.dot(n2, face_centroid(f1) - pe))
        kind = "convex" if s < 0 else "concave"
        if kind not in want:
            continue
        out.append({"edge": edge, "angle": ang, "kind": kind,
                    "length": edge_length(edge), "mid": pe.tolist()})
    return out, em.Extent()


# -------------------------------------------------------------- fillet (A1)

class FilletFailure(RuntimeError):
    def __init__(self, diag):
        super().__init__("BRepFilletAPI_MakeFillet did not converge: %s" % diag)
        self.diag = diag


def fillet_diagnostics(fil):
    """Everything OCCT is willing to say about a failure. Worth recording,
    because DESIGN.md section 11 warns that OCCT failures are opaque -- this
    measures how opaque."""
    d = {}
    for name in ("IsDone", "HasResult", "NbContours", "NbFaultyContours",
                 "NbFaultyVertices"):
        try:
            d[name] = getattr(fil, name)()
        except Exception as exc:                             # noqa: BLE001
            d[name] = "raised %s" % type(exc).__name__
    faulty = []
    try:
        for i in range(1, int(d.get("NbFaultyContours") or 0) + 1):
            faulty.append(fil.FaultyContour(i))
    except Exception:                                        # noqa: BLE001
        pass
    d["faulty_contours"] = faulty
    stripe = []
    try:
        for i in range(1, int(d.get("NbContours") or 0) + 1):
            stripe.append(str(fil.StripeStatus(i)).split(".")[-1])
    except Exception:                                        # noqa: BLE001
        pass
    d["stripe_status"] = sorted(set(stripe))
    return d


def fillet(shape, fmap, kinds, edge_specs):
    """edge_specs: [(edge, radius)]. Returns (out_shape, omap, labels)."""
    fil = BRepFilletAPI_MakeFillet(shape)
    for edge, radius in edge_specs:
        fil.Add(radius, edge)
    try:
        fil.Build()
    except Exception as exc:                                 # noqa: BLE001
        raise FilletFailure({"exception": "%s: %s"
                             % (type(exc).__name__, exc)}) from exc
    if not fil.IsDone():
        raise FilletFailure(fillet_diagnostics(fil))
    out = fil.Shape()
    omap = TopTools_IndexedMapOfShape()
    for f in faces_of(out):
        omap.Add(f)

    labels = {}
    for i in range(1, fmap.Extent() + 1):
        f = TopoDS.Face_s(fmap.FindKey(i))
        if fil.IsDeleted(f):
            continue
        got = list(fil.Modified(f)) or [f]
        for r in got:
            j = omap.FindIndex(r)
            if j:
                labels[j] = kinds[i]
    # positively tag the fillet surfaces themselves, rather than inferring
    n_generated = 0
    for edge, _r in edge_specs:
        for g in fil.Generated(edge):
            j = omap.FindIndex(g)
            if j and j not in labels:
                labels[j] = "bevel"
                n_generated += 1
    residual = [j for j in range(1, omap.Extent() + 1) if j not in labels]
    for j in residual:
        labels[j] = "bevel"          # corner / vertex blend patches
    return out, omap, labels, n_generated, len(residual)


# ---------------------------------------------------------- triangulate (A1)

KIND_BIT = {"body": C.TAG_BODY, "cut": C.TAG_CUT, "bevel": C.TAG_BEVEL}


def triangulate(shape, fmap, labels, deflection=1e-4, angular=0.35):
    # BRepMesh caches the triangulation on the shape, so a second call with a
    # COARSER deflection is a no-op and a deflection sweep silently returns the
    # same triangle count every time. Clean first.
    BRepTools.Clean_s(shape)
    BRepMesh_IncrementalMesh(shape, deflection, False, angular, True)
    V, T, G, F = [], [], [], []
    for i in range(1, fmap.Extent() + 1):
        f = TopoDS.Face_s(fmap.FindKey(i))
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is None:
            continue
        trsf = loc.Transformation()
        n0 = len(V)
        for k in range(1, tri.NbNodes() + 1):
            p = tri.Node(k).Transformed(trsf)
            V.append((p.X(), p.Y(), p.Z()))
        rev = f.Orientation() == TopAbs_REVERSED
        bit = KIND_BIT[labels.get(i, "bevel")]
        for k in range(1, tri.NbTriangles() + 1):
            a, b, c = tri.Triangle(k).Get()
            if rev:
                b, c = c, b
            T.append((n0 + a - 1, n0 + b - 1, n0 + c - 1))
            G.append(bit)
            F.append(i)                     # the B-Rep face this came from
    return (np.array(V, dtype=np.float64),
            np.array(T, dtype=np.int64),
            np.array(G, dtype=np.uint8),
            np.array(F, dtype=np.int64))


# ------------------------------------------------------------------- A5 sew

def sew_mesh_to_solid(verts, tris):
    """Every triangle becomes a planar B-Rep face, sewn into a shell.

    This is the only bridge from a Manifold mesh into OCCT without a feature
    history. It is included to be measured, not because it is expected to work.
    """
    sew = BRepBuilderAPI_Sewing(1e-6)
    for t in tris:
        poly = BRepBuilderAPI_MakePolygon()
        for k in (0, 1, 2):
            p = verts[t[k]]
            poly.Add(gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
        poly.Close()
        face = BRepBuilderAPI_MakeFace(poly.Wire())
        if face.IsDone():
            sew.Add(face.Face())
    sew.Perform()
    sewn = sew.SewedShape()
    shells = sub_shapes(sewn, TopAbs_SHELL)
    if not shells:
        return None
    mk = BRepBuilderAPI_MakeSolid(TopoDS.Shell_s(shells[0]))
    return mk.Solid()


# ------------------------------------------------------------------- the run

def run():
    res = {"approach": "A_occt", "occt_available": True}
    try:
        import OCP
        res["ocp_file"] = OCP.__file__
    except Exception:                                        # noqa: BLE001
        pass

    RADIUS, SEGMENTS_EQUIV = 0.02, "exact (analytic arc)"
    res["radius"] = RADIUS
    res["segments"] = SEGMENTS_EQUIV

    # ---------------------------------------------------------------- A1
    fx = C.Fixture()
    t = {}
    t0 = time.perf_counter()
    shape, fmap, kinds, unlab = build_solid(fx)
    t["build_s"] = time.perf_counter() - t0
    res["A1_boolean_faces"] = fmap.Extent()
    res["A1_boolean_unlabelled_faces"] = len(unlab)
    res["A1_boolean_kind_counts"] = {
        k: sum(1 for v in kinds.values() if v == k) for k in ("body", "cut")}

    # pre-bevel baseline, meshed the same way as the post-bevel result
    v0, t0v, g0, _f0 = triangulate(shape, fmap, kinds)
    before = C.before_snapshot(fx, v0, t0v, g0)
    res["A1_before"] = before
    res["A1_before_check"] = C.check_mesh(v0, t0v, "occt pre-fillet")

    t0 = time.perf_counter()
    sel, n_edges = select_edges(shape, 60.0)
    t["select_s"] = time.perf_counter() - t0
    res["A1_edges_total"] = n_edges
    res["A1_edges_selected"] = len(sel)
    res["A1_edges_convex"] = sum(1 for e in sel if e["kind"] == "convex")
    res["A1_edges_concave"] = sum(1 for e in sel if e["kind"] == "concave")

    t0 = time.perf_counter()
    out, omap, labels, n_gen, n_resid = fillet(
        shape, fmap, kinds, [(e["edge"], RADIUS) for e in sel])
    t["fillet_s"] = time.perf_counter() - t0
    res["A1_out_faces"] = omap.Extent()
    res["A1_fillet_faces_from_generated"] = n_gen
    res["A1_residual_faces_tagged_bevel"] = n_resid

    t0 = time.perf_counter()
    v1, t1, g1, f1 = triangulate(out, omap, labels)
    t["mesh_s"] = time.perf_counter() - t0
    res["A1_timing"] = t
    res["A1_total_s"] = sum(t.values())

    res["A1_check"] = C.check_mesh(v1, t1, "occt filleted")
    res["A1_provenance"] = C.provenance_report(fx, v1, t1, g1, before)
    fid = C.plane_fidelity(fx, v1, t1)
    fid.pop("per_plane")
    res["A1_plane_fidelity"] = fid

    ref = C.referee(fx, v1, t1)
    res["A1_area_by_truth"] = ref["area_by_truth"]

    # A6 -- does the labelled result survive re-entry into Manifold and one
    # more boolean? This is the integration question, not a nicety: OCCT is
    # only usable if its history can be handed to the kernel DESIGN.md picked.
    res["A6_downstream"] = C.downstream_boolean_test(v1, t1, g1, face_id=f1)

    # ---------------------------------------------------------------- A2
    # per-edge radius: 0.03 on convex, 0.008 on concave
    specs = [(e["edge"], 0.03 if e["kind"] == "convex" else 0.008)
             for e in sel]
    try:
        out2, omap2, labels2, _g, _r = fillet(shape, fmap, kinds, specs)
        v2, t2, g2, _f2 = triangulate(out2, omap2, labels2)
        chk2 = C.check_mesh(v2, t2, "occt mixed radii")
        res["A2_per_edge_radius"] = {
            "supported": True,
            "api": "BRepFilletAPI_MakeFillet.Add(radius, edge) per edge; "
                   "Add(r1, r2, edge) gives a variable radius along one edge",
            "volume_uniform": res["A1_check"]["manifold_volume"],
            "volume_mixed": chk2["manifold_volume"],
            "distinct": abs((chk2["manifold_volume"] or 0)
                            - (res["A1_check"]["manifold_volume"] or 0)) > 1e-9,
            "watertight": chk2["watertight"],
        }
    except Exception as exc:                                 # noqa: BLE001
        res["A2_per_edge_radius"] = {"supported": False, "error": str(exc)}

    # single-edge fillet -- does the radius land where the API says?
    one = [e for e in sel if e["kind"] == "convex" and e["length"] > 0.5][:1]
    if one:
        out3, omap3, labels3, _g, _r = fillet(
            shape, fmap, kinds, [(one[0]["edge"], 0.03)])
        v3, t3, g3, _f3 = triangulate(out3, omap3, labels3)
        r3 = C.referee(fx, v3, t3)
        res["A2_single_edge"] = {
            "edge_length": one[0]["length"],
            "new_surface_area_measured": r3["area_by_truth"]["new-bevel"],
            "analytic_quarter_cylinder": math.pi / 2 * 0.03 * one[0]["length"],
            "watertight": C.check_mesh(v3, t3, "occt one edge")["watertight"],
        }

    # variable radius along a single edge -- claimed by the API, so measure it
    if one:
        try:
            filv = BRepFilletAPI_MakeFillet(shape)
            filv.Add(0.005, 0.03, one[0]["edge"])
            filv.Build()
            if not filv.IsDone():
                raise FilletFailure(fillet_diagnostics(filv))
            outv = filv.Shape()
            omv = TopTools_IndexedMapOfShape()
            for f in faces_of(outv):
                omv.Add(f)
            lbv = {}
            for i in range(1, fmap.Extent() + 1):
                f = TopoDS.Face_s(fmap.FindKey(i))
                if filv.IsDeleted(f):
                    continue
                for r in (list(filv.Modified(f)) or [f]):
                    j = omv.FindIndex(r)
                    if j:
                        lbv[j] = kinds[i]
            for j in range(1, omv.Extent() + 1):
                lbv.setdefault(j, "bevel")
            vv, tv, gv, _fv = triangulate(outv, omv, lbv)
            rv = C.referee(fx, vv, tv)
            L = one[0]["length"]
            res["A2_variable_radius"] = {
                "supported": True,
                "r1": 0.005, "r2": 0.03,
                "new_surface_area_measured": rv["area_by_truth"]["new-bevel"],
                "analytic_if_constant_r1": math.pi / 2 * 0.005 * L,
                "analytic_if_constant_r2": math.pi / 2 * 0.03 * L,
                "watertight": C.check_mesh(vv, tv, "occt var r")["watertight"],
            }
        except Exception as exc:                             # noqa: BLE001
            res["A2_variable_radius"] = {"supported": False,
                                         "error": str(exc)}

    # ---------------------------------------------------------------- A3
    conv, _ = select_edges(shape, 60.0, want=("convex",))
    outc, omapc, labelsc, _g, _r = fillet(
        shape, fmap, kinds, [(e["edge"], RADIUS) for e in conv])
    vc, tc, gc, _fc = triangulate(outc, omapc, labelsc)
    rc = C.referee(fx, vc, tc)
    res["A3_predicate"] = {
        "convex_only_edges": len(conv),
        "all_sharp_edges": len(sel),
        "convex_only_watertight": C.check_mesh(vc, tc, "occt convex only")[
            "watertight"],
        "cut_surface_area_all_edges": res["A1_area_by_truth"]["cut-surface"],
        "cut_surface_area_convex_only": rc["area_by_truth"]["cut-surface"],
        "note": "concave floor edges left sharp keeps more of the cut surface "
                "flat, so the cut-surface area is larger",
    }

    # ---------------------------------------------------------------- A4
    dense = {}
    for slots, slot_w in ((1, 0.15), (3, 0.05), (5, 0.05), (5, 0.06),
                          (5, 0.08), (5, 0.10)):
        fxd = C.Fixture(slots=slots, slot_w=slot_w)
        key = "slots=%d,slot_w=%g" % (slots, slot_w)
        try:
            t0 = time.perf_counter()
            sh, fm, kn, _u = build_solid(fxd)
            se, ne = select_edges(sh, 60.0)
            o, om, lb, _g, _r = fillet(sh, fm, kn,
                                       [(e["edge"], RADIUS) for e in se])
            vv, tt, gg, _fv = triangulate(o, om, lb)
            el = time.perf_counter() - t0
            chk = C.check_mesh(vv, tt, key)
            bv, bt, bg, _fb = triangulate(sh, fm, kn)
            bsnap = C.before_snapshot(fxd, bv, bt, bg)
            prov = C.provenance_report(fxd, vv, tt, gg, bsnap)
            dense[key] = {
                "seconds": el, "brep_edges": ne, "edges_filleted": len(se),
                "concave_edges": sum(1 for e in se if e["kind"] == "concave"),
                "brep_faces_out": om.Extent(), "tris": chk["tris"],
                "watertight": chk["watertight"],
                "manifold_status": chk["manifold_status"],
                "sel_cut_faces_before": bsnap["sel_cut_any_faces"],
                "sel_cut_faces_after": prov["sel_cut_any_faces"],
                "expected_cut_faces": fxd.n_cut_faces(),
                "provenance_exact": prov["provenance_exact"],
            }
        except FilletFailure as exc:
            wall = (2 * fxd.half * fxd.span_frac / (slots - 1) - slot_w
                    if slots > 1 else None)
            dense[key] = {"error": "fillet did not converge",
                          "occt_diagnostics": exc.diag,
                          "wall_thickness_between_slots": wall,
                          "radius": RADIUS,
                          "geometrically_feasible": (wall is None
                                                     or wall > 2 * RADIUS)}
        except Exception as exc:                             # noqa: BLE001
            dense[key] = {"error": "%s: %s" % (type(exc).__name__, exc)}
    res["A4_density_by_slots"] = dense

    mesh_sweep = {}
    for defl, ang in ((1e-2, 0.7), (1e-3, 0.5), (1e-4, 0.35), (1e-5, 0.2)):
        t0 = time.perf_counter()
        vv, tt, gg, _fs = triangulate(out, omap, labels, deflection=defl,
                                     angular=ang)
        el = time.perf_counter() - t0
        chk = C.check_mesh(vv, tt, "defl=%g" % defl)
        mesh_sweep["deflection=%g,angular=%g" % (defl, ang)] = {
            "seconds": el, "tris": chk["tris"],
            "watertight": chk["watertight"],
            "manifold_status": chk["manifold_status"],
            "degenerate_tris_emitted": chk["dropped_degenerate"],
            "provenance_exact": C.provenance_report(
                fx, vv, tt, gg, before)["provenance_exact"],
        }
    res["A4_mesh_deflection_sweep"] = mesh_sweep

    # ---------------------------------------------------------------- A5
    a5 = {}
    try:
        man, bid, cid = fx.manifold()
        mv, mt, mg = C.manifold_arrays(man, bid, cid)
        t0 = time.perf_counter()
        solid = sew_mesh_to_solid(mv, mt)
        a5["sew_s"] = time.perf_counter() - t0
        if solid is None:
            a5["result"] = "sewing produced no shell"
        else:
            fm5 = TopTools_IndexedMapOfShape()
            for f in faces_of(solid):
                fm5.Add(f)
            a5["brep_faces"] = fm5.Extent()
            a5["input_tris"] = int(mt.shape[0])
            se5, ne5 = select_edges(solid, 60.0)
            a5["brep_edges"] = ne5
            a5["edges_selected_dihedral_gt_60"] = len(se5)
            a5["note"] = ("every triangle is its own B-Rep face, so each "
                          "logical cube edge arrives as several collinear "
                          "edge segments")
            t0 = time.perf_counter()
            try:
                kn5 = {i: "body" for i in range(1, fm5.Extent() + 1)}
                o5, om5, lb5, _g, _r = fillet(
                    solid, fm5, kn5, [(e["edge"], 0.02) for e in se5])
                v5, t5, g5, _f5 = triangulate(o5, om5, lb5)
                chk5 = C.check_mesh(v5, t5, "occt sewn fillet")
                a5["fillet_s"] = time.perf_counter() - t0
                a5["result"] = "fillet returned IsDone"
                a5["out_faces"] = om5.Extent()
                a5["watertight"] = chk5["watertight"]
                a5["manifold_status"] = chk5["manifold_status"]
                a5["volume_before"] = float(man.volume())
                a5["volume_after"] = chk5["manifold_volume"]
            except Exception as exc:                          # noqa: BLE001
                a5["fillet_s"] = time.perf_counter() - t0
                a5["result"] = "FAILED: %s: %s" % (type(exc).__name__, exc)
    except Exception as exc:                                  # noqa: BLE001
        a5["result"] = "FAILED early: %s: %s" % (type(exc).__name__, exc)
    res["A5_mesh_to_brep"] = a5
    return res


def report(res):
    C.rule("APPROACH A -- OpenCASCADE fillet (cadquery-ocp / OCP %s)"
           % os.environ.get("OCCT_VERSION", "7.9.3"))
    C.kv("B-Rep faces after boolean", res["A1_boolean_faces"])
    C.kv("faces with no provenance label", res["A1_boolean_unlabelled_faces"])
    C.kv("kind counts", res["A1_boolean_kind_counts"])
    C.kv("B-Rep edges / selected dihedral>60",
         "%d / %d  (convex %d, concave %d)"
         % (res["A1_edges_total"], res["A1_edges_selected"],
            res["A1_edges_convex"], res["A1_edges_concave"]))
    C.kv("B-Rep faces after fillet", res["A1_out_faces"])
    C.kv("  of which Generated(edge) fillet faces",
         res["A1_fillet_faces_from_generated"])
    C.kv("  residual corner patches", res["A1_residual_faces_tagged_bevel"])
    chk = res["A1_check"]
    C.kv("triangles", chk["tris"])
    C.kv("watertight / manifold status",
         "%s / %s" % (chk["watertight"], chk["manifold_status"]))
    C.kv("reusable as a Manifold boolean operand",
         chk["reusable_as_boolean_operand"])
    C.kv("volume", "%.6f" % (chk["manifold_volume"] or float("nan")))
    p = res["A1_provenance"]
    print()
    C.kv("faces(#cut_face) before bevel", p["before"]["sel_cut_any_faces"])
    C.kv("faces(#cut_face) after bevel", p["sel_cut_any_faces"])
    C.kv("expected", p["expected_cut_faces"])
    C.kv("DRIFT", "YES" if p["drift_any"] else "no")
    C.kv("false positives (bevel tris claiming #cut_face)",
         p["false_positive_tris"])
    C.kv("false negatives (cut tris that lost the tag)",
         p["false_negative_tris"])
    C.kv("untagged triangles", p["untagged_tris"])
    C.kv("provenance exact vs geometric referee",
         C.ok(p["provenance_exact"]))
    C.kv("original planes still found / expected",
         "%d / %d" % (res["A1_plane_fidelity"]["planes_found"],
                      res["A1_plane_fidelity"]["planes_expected"]))
    C.kv("worst flat-face offset error",
         res["A1_plane_fidelity"]["worst_max_offset_error"])
    C.kv("total runtime (build+select+fillet+mesh)",
         "%.3f s" % res["A1_total_s"])
    print()
    C.kv("A6 back into Manifold, then one more boolean", "")
    for k, v in res["A6_downstream"].items():
        C.kv("    " + k, v)
    print()
    C.kv("per-edge radius", res["A2_per_edge_radius"])
    if "A2_single_edge" in res:
        C.kv("single-edge fillet", res["A2_single_edge"])
    if "A2_variable_radius" in res:
        C.kv("variable radius along one edge", res["A2_variable_radius"])
    C.kv("predicate (convex only)", res["A3_predicate"])
    print()
    for k, v in res["A4_density_by_slots"].items():
        C.kv("density %s" % k, v)
    for k, v in res["A4_mesh_deflection_sweep"].items():
        C.kv("mesh %s" % k, v)
    print()
    C.kv("A5 mesh->B-Rep sew then fillet", "")
    for k, v in res["A5_mesh_to_brep"].items():
        C.kv("    " + k, v)


if __name__ == "__main__":
    r = run()
    report(r)
    print("@@RESULT@@")
    print(json.dumps(r, default=str))
