"""
Shared fixture, referee and metrics for the bevel/fillet spike.

DESIGN.md section 13 names bevel as the top risk, because the earlier prototype
proved bevel is where provenance breaks: `bmesh.ops.bevel` interpolated ancestor
attributes onto new faces and silently widened `faces(#cut_face)` from 2 matches
to 10. Manifold ships no bevel, so something has to fill that hole.

This module is deliberately kernel-agnostic. Every candidate is required to hand
back the same three things:

    verts (N,3) float64,  tris (M,3) int,  tri_tag (M,) uint8 bitmask

and nothing else. The bitmask is the *candidate's own claim* about provenance:

    bit 0  TAG_BODY    this triangle is part of the original body surface
    bit 1  TAG_CUT     this triangle is part of the original boolean cut surface
    bit 2  TAG_BEVEL   this triangle is new geometry made by the bevel

The referee below recomputes the truth from geometry alone and scores the claim
against it. That separation is the whole point: an approach cannot pass by
asserting that its provenance is fine.

Fixture: a cube with a slot milled across the top. One connected component, per
the m5() lesson in prototype/manifold_provenance.py -- a cutter spanning the
full height severs the body instead of slotting it.

        +Z
         |     ___     ___        slot walls  x = +/- 0.075   (cut surface)
         |    |   |   |   |       slot floor  z = +0.15       (cut surface)
         |____|   |___|   |____   cube        |x|,|y|,|z| = 0.3  (body surface)
"""

import math
import time

import numpy as np
import manifold3d as m

# ----------------------------------------------------------------- tag vocab

TAG_BODY = 1
TAG_CUT = 2
TAG_BEVEL = 4

TAG_NAMES = {TAG_BODY: "body", TAG_CUT: "cut_face", TAG_BEVEL: "bevel"}


def tag_str(mask):
    parts = [n for b, n in sorted(TAG_NAMES.items()) if mask & b]
    return "+".join(parts) if parts else "<untagged>"


# ------------------------------------------------------------------ fixture

class Fixture:
    """Slotted cube. `slots` slots, cut down from the top, through in Y."""

    def __init__(self, half=0.3, slot_w=0.15, slot_depth=0.15, slots=1,
                 span_frac=0.7):
        self.half = half
        self.slot_w = slot_w
        self.slot_depth = slot_depth
        self.slots = slots
        self.span_frac = span_frac
        self.floor_z = half - slot_depth
        self.centers = self._centers()

    def _centers(self):
        if self.slots == 1:
            return [0.0]
        span = 2 * self.half * self.span_frac
        return [-span / 2 + (i / (self.slots - 1.0)) * span
                for i in range(self.slots)]

    # -- reference planes, used by the referee ----------------------------

    def planes(self):
        """[(kind, axis, offset)] -- kind is 'body' or 'cut'."""
        h = self.half
        out = [("body", 0, -h), ("body", 0, h),
               ("body", 1, -h), ("body", 1, h),
               ("body", 2, -h), ("body", 2, h)]
        for cx in self.centers:
            out.append(("cut", 0, cx - self.slot_w / 2))
            out.append(("cut", 0, cx + self.slot_w / 2))
        out.append(("cut", 2, self.floor_z))
        return out

    def n_cut_faces(self):
        """Logical cut faces expected: two walls + one floor per slot."""
        return 3 * self.slots

    # -- construction, kernel specific -----------------------------------

    def manifold(self):
        """Build in Manifold, with ancestor ids in run_original_id.

        Body gets id BODY, every cutter gets id CUT, so the output triangle
        runs already carry the section 5.2 provenance the candidates must
        preserve.
        """
        base = m.Manifold.reserve_ids(2)
        body_id, cut_id = base, base + 1
        s = 2 * self.half
        body = _with_id(m.Manifold.cube((s, s, s), True), body_id)
        out = body
        for cx in self.centers:
            cutter = _with_id(
                m.Manifold.cube(
                    (self.slot_w, s * 1.4, self.slot_depth * 2), True
                ).translate((cx, 0.0, self.half)),
                cut_id)
            out = out - cutter
        return out, body_id, cut_id

    def boxes(self):
        """[(kind, (cx,cy,cz), (sx,sy,sz))] -- for kernels with own primitives."""
        s = 2 * self.half
        out = [("body", (0.0, 0.0, 0.0), (s, s, s))]
        for cx in self.centers:
            out.append(("cut", (cx, 0.0, self.half),
                        (self.slot_w, s * 1.4, self.slot_depth * 2)))
        return out

    def __repr__(self):
        return ("Fixture(half=%g slot_w=%g depth=%g slots=%d)"
                % (self.half, self.slot_w, self.slot_depth, self.slots))


def _with_id(man, tag_id):
    """Stamp every triangle of `man` with run_original_id == tag_id.

    to_mesh() hands back read-only views and np.ascontiguousarray is a no-op on
    them, so the copy has to be explicit. This is the hour-costing footgun
    recorded in prototype/README.md.
    """
    mesh = man.to_mesh()
    tv = np.array(mesh.tri_verts, dtype=np.uint32, order="C", copy=True)
    vp = np.array(mesh.vert_properties, dtype=np.float32, order="C", copy=True)
    return m.Manifold(m.Mesh(
        vert_properties=vp,
        tri_verts=tv,
        run_index=np.array([0, tv.shape[0] * 3], dtype=np.uint32),
        run_original_id=np.array([tag_id], dtype=np.uint32)))


def manifold_arrays(man, body_id, cut_id):
    """Manifold -> (verts, tris, tri_tag) using native run_original_id."""
    mesh = man.to_mesh64()
    verts = np.array(mesh.vert_properties, dtype=np.float64, copy=True)[:, :3]
    tris = np.array(mesh.tri_verts, dtype=np.int64, copy=True)
    run_idx = np.asarray(mesh.run_index, dtype=np.int64)
    run_ids = np.asarray(mesh.run_original_id, dtype=np.int64)
    tag = np.zeros(tris.shape[0], dtype=np.uint8)
    for i, rid in enumerate(run_ids):
        a, b = run_idx[i] // 3, run_idx[i + 1] // 3
        if int(rid) == int(body_id):
            tag[a:b] = TAG_BODY
        elif int(rid) == int(cut_id):
            tag[a:b] = TAG_CUT
    return verts, tris, tag


def to_manifold(verts, tris):
    """Arrays -> Manifold. Returns (manifold, status_name)."""
    v = np.array(verts, dtype=np.float64, order="C", copy=True)
    t = np.array(tris, dtype=np.uint64, order="C", copy=True)
    man = m.Manifold(m.Mesh64(vert_properties=v, tri_verts=t))
    return man, _status_name(man)


def _status_name(man):
    st = man.status
    if callable(st):
        st = st()
    return str(st).split(".")[-1]


def to_manifold_with_tags(verts, tris, tri_tag, face_id=None):
    """Push a candidate's provenance back into Manifold's NATIVE channels.

    This is the integration question the recommendation turns on: whichever
    kernel does the bevel, the result has to re-enter Manifold and keep its
    provenance through the next boolean. Triangles are sorted into one run per
    tag value, each run gets a reserved original id, and face_id carries the
    logical-face grouping that coplanar analysis cannot recover on curved
    surfaces.
    """
    tris = np.asarray(tris, dtype=np.int64)
    tri_tag = np.asarray(tri_tag, dtype=np.uint8)
    # OCCT emits a separate vertex array per B-Rep face, so the raw mesh is a
    # pile of disconnected patches and Manifold rejects it as NotManifold.
    # Weld first, carrying the tags. (Manifold's merge_from_vert/merge_to_vert
    # is the lossless alternative if the producer can supply the pairs.)
    verts, welded, keep = weld(verts, tris)
    tris = welded[keep]
    tri_tag = tri_tag[keep]
    if face_id is not None:
        face_id = np.asarray(face_id)[keep]
    values = sorted(set(tri_tag.tolist()))
    order = np.concatenate([np.where(tri_tag == v)[0] for v in values])
    sorted_tris = tris[order]

    base = m.Manifold.reserve_ids(len(values))
    id_map = {int(v): int(base + i) for i, v in enumerate(values)}
    counts = [int(np.sum(tri_tag == v)) for v in values]
    run_index = np.concatenate([[0], np.cumsum(counts) * 3]).astype(np.uint64)
    run_ids = np.array([id_map[int(v)] for v in values], dtype=np.uint32)

    kwargs = dict(
        vert_properties=np.array(verts, dtype=np.float64, order="C",
                                 copy=True),
        tri_verts=np.array(sorted_tris, dtype=np.uint64, order="C", copy=True),
        run_index=run_index,
        run_original_id=run_ids)
    if face_id is not None:
        kwargs["face_id"] = np.array(
            np.asarray(face_id)[order], dtype=np.uint64, order="C", copy=True)
    man = m.Manifold(m.Mesh64(**kwargs))
    return man, id_map, _status_name(man)


def id_census(man, id_map, extra_names=None):
    """Triangles per tag name, read back out of run_original_id."""
    inv = {v: tag_str(k) for k, v in id_map.items()}
    inv.update(extra_names or {})
    mesh = man.to_mesh64()
    idx = np.asarray(mesh.run_index, dtype=np.int64)
    ids = np.asarray(mesh.run_original_id, dtype=np.int64)
    out = {}
    for i, rid in enumerate(ids):
        name = inv.get(int(rid), "unknown:%d" % rid)
        out[name] = out.get(name, 0) + int((idx[i + 1] - idx[i]) // 3)
    return out


def downstream_boolean_test(verts, tris, tri_tag, face_id=None):
    """Re-import a bevelled result into Manifold with native provenance, then
    run another boolean on it and check the tags still separate."""
    out = {}
    try:
        man, id_map, status = to_manifold_with_tags(verts, tris, tri_tag,
                                                    face_id)
        out["import_status"] = status
        if status != "NoError" or man.is_empty():
            out["result"] = "import failed"
            return out
        out["census_after_import"] = id_census(man, id_map)
        probe_id = m.Manifold.reserve_ids(1)
        probe = _with_id(
            m.Manifold.cube((0.06, 0.06, 5.0), True).translate(
                (0.2, 0.2, 0.0)), probe_id)
        res = man - probe
        out["boolean_status"] = _status_name(res)
        out["census_after_boolean"] = id_census(
            res, id_map, extra_names={probe_id: "probe"})
        # every triangle must still be attributable, and the pre-bevel tags
        # must still be present and distinct
        out["unknown_after_boolean"] = sum(
            v for k, v in out["census_after_boolean"].items()
            if k.startswith("unknown"))
        out["tags_still_separable"] = (
            set(out["census_after_import"]) <= set(out["census_after_boolean"])
            and out["unknown_after_boolean"] == 0)
        mesh = res.to_mesh64()
        fid = np.asarray(mesh.face_id, dtype=np.int64)
        out["distinct_face_ids_after_boolean"] = len(set(fid.tolist()))
        out["result"] = "ok"
    except Exception as exc:                                 # noqa: BLE001
        out["result"] = "FAILED: %s: %s" % (type(exc).__name__, exc)
    return out


# ------------------------------------------------------------------ topology

def weld(verts, tris, tol=1e-7):
    """Merge coincident vertices so topology tests see one surface."""
    key = np.round(np.asarray(verts, dtype=np.float64) / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True,
                              return_inverse=True)
    inv = inv.reshape(-1)
    wv = np.asarray(verts, dtype=np.float64)[first]
    wt = inv[np.asarray(tris)]
    keep = (wt[:, 0] != wt[:, 1]) & (wt[:, 1] != wt[:, 2]) & (wt[:, 2] != wt[:, 0])
    return wv, wt, keep


def tri_normals(verts, tris):
    v = np.asarray(verts, dtype=np.float64)
    a, b, c = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1)
    area = 0.5 * ln
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / np.where(ln[:, None] == 0, 1.0, ln[:, None])
    return n, area


def tri_centroids(verts, tris):
    v = np.asarray(verts, dtype=np.float64)
    return (v[tris[:, 0]] + v[tris[:, 1]] + v[tris[:, 2]]) / 3.0


def signed_volume(verts, tris):
    v = np.asarray(verts, dtype=np.float64)
    a, b, c = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


def check_mesh(verts, tris, label="", weld_tol=1e-7):
    """Independent validity audit. Does not trust any kernel's own claim.

    weld_tol matters: on a dense mesh a loose weld can merge vertices the
    kernel keeps apart and manufacture a fake non-manifold edge, so any
    disagreement with Manifold's own verdict should be re-checked tighter
    before it is reported as a defect.
    """
    tris = np.asarray(tris, dtype=np.int64)
    wv, wt, keep = weld(verts, tris, tol=weld_tol)
    dropped = int(np.sum(~keep))
    wt = wt[keep]

    he = np.concatenate([wt[:, [0, 1]], wt[:, [1, 2]], wt[:, [2, 0]]])
    und = np.sort(he, axis=1)
    _, cnt = np.unique(und, axis=0, return_counts=True)
    non_manifold_edges = int(np.sum(cnt != 2))
    boundary_edges = int(np.sum(cnt == 1))

    # oriented: every directed half-edge appears exactly once
    _, dcnt = np.unique(he, axis=0, return_counts=True)
    dup_directed = int(np.sum(dcnt > 1))

    _, tcnt = np.unique(np.sort(wt, axis=1), axis=0, return_counts=True)
    dup_tris = int(np.sum(tcnt > 1))

    _, area = tri_normals(wv, wt)
    zero_area = int(np.sum(area < 1e-14))

    # components by union-find over welded verts
    parent = np.arange(wv.shape[0])

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in und:
        ra, rb = find(e[0]), find(e[1])
        if ra != rb:
            parent[ra] = rb
    used = np.unique(wt)
    comps = len({find(int(x)) for x in used})

    V = int(used.shape[0])
    E = int(np.unique(und, axis=0).shape[0])
    F = int(wt.shape[0])
    chi = V - E + F
    genus = (2 * comps - chi) // 2 if non_manifold_edges == 0 else None

    vol = signed_volume(wv, wt)

    # Raw audit, on the producer's own indices with no welding. For a producer
    # that already shares vertices (Manifold, bmesh) this is the true topology.
    # For one that emits a separate vertex array per face (OCCT) every edge
    # looks like a boundary, so the welded numbers are the meaningful ones.
    # Reporting only the welded audit is how a clean Minkowski result gets
    # mislabelled non-manifold: welding coincident positions fuses two surface
    # sheets that legitimately only touch.
    rhe = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    rund = np.sort(rhe, axis=1)
    _, rcnt = np.unique(rund, axis=0, return_counts=True)
    raw_nm = int(np.sum(rcnt != 2))
    raw_boundary = int(np.sum(rcnt == 1))
    _, rd = np.unique(rhe, axis=0, return_counts=True)
    raw_dup_directed = int(np.sum(rd > 1))
    n_in = int(np.asarray(verts).shape[0])
    coincident = n_in - int(np.unique(
        np.round(np.asarray(verts, dtype=np.float64) / weld_tol
                 ).astype(np.int64), axis=0).shape[0])

    man, status = None, "not-attempted"
    boolean_ok = None
    try:
        man, status = to_manifold(wv, wt)
        if status == "NoError" and not man.is_empty():
            probe = m.Manifold.cube((0.05, 0.05, 5.0), True).translate(
                (0.13, 0.0, 0.0))
            r = man - probe
            boolean_ok = (_status_name(r) == "NoError" and not r.is_empty())
    except Exception as exc:                                # noqa: BLE001
        status = "exception: %s" % exc

    return {
        "label": label,
        "weld_tol": weld_tol,
        "tris": F,
        "verts": V,
        "dropped_degenerate": dropped,
        "non_manifold_edges": non_manifold_edges,
        "boundary_edges": boundary_edges,
        "duplicate_directed_halfedges": dup_directed,
        "duplicate_tris": dup_tris,
        "zero_area_tris": zero_area,
        "components": comps,
        "genus": genus,
        "volume": vol,
        "raw_non_manifold_edges": raw_nm,
        "raw_boundary_edges": raw_boundary,
        "raw_duplicate_directed_halfedges": raw_dup_directed,
        "raw_watertight": (raw_nm == 0 and raw_boundary == 0
                           and raw_dup_directed == 0),
        "coincident_vertex_positions": coincident,
        "manifold_status": status,
        "manifold_volume": (float(man.volume()) if man is not None
                            and status == "NoError" and not man.is_empty()
                            else None),
        "reusable_as_boolean_operand": boolean_ok,
        "watertight": (non_manifold_edges == 0 and boundary_edges == 0
                       and dup_directed == 0),
    }


def sharp_edge_report(verts, tris, thresh_deg=60.0):
    """Did the bevel actually remove the sharp edges it was asked to remove?

    Counts edges whose dihedral still exceeds the threshold, and their total
    length. A correct bevel of every sharp edge leaves none. An approach that
    only rounds the creases created by one boolean leaves the rest, and this is
    where that shows up.
    """
    tris = np.asarray(tris, dtype=np.int64)
    wv, wt, keep = weld(verts, tris)
    wt = wt[keep]
    n, _ = tri_normals(wv, wt)
    edge_map = {}
    for i in range(wt.shape[0]):
        t = wt[i]
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_map.setdefault((int(min(a, b)), int(max(a, b))), []).append(i)
    cos_t = math.cos(math.radians(thresh_deg))
    count, length = 0, 0.0
    for (a, b), fs in edge_map.items():
        if len(fs) != 2:
            continue
        if float(np.dot(n[fs[0]], n[fs[1]])) < cos_t:
            count += 1
            length += float(np.linalg.norm(wv[a] - wv[b]))
    return {"sharp_edges_over_%gdeg" % thresh_deg: count,
            "total_sharp_length": length}


def coplanar_groups(verts, tris, ang_tol_deg=1.0, dist_tol=1e-6):
    """Connected coplanar triangle groups -- the logical-face analogue of
    Manifold's face_id, computed the same way for every candidate so the
    `faces(#cut_face)` counts are comparable."""
    tris = np.asarray(tris, dtype=np.int64)
    wv, wt, keep = weld(verts, tris)
    idx = np.where(keep)[0]
    wt = wt[keep]
    n, _ = tri_normals(wv, wt)
    cen = tri_centroids(wv, wt)
    d = np.einsum("ij,ij->i", n, cen)

    edge_map = {}
    for i in range(wt.shape[0]):
        t = wt[i]
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            k = (int(min(a, b)), int(max(a, b)))
            edge_map.setdefault(k, []).append(i)

    parent = list(range(wt.shape[0]))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cos_tol = math.cos(math.radians(ang_tol_deg))
    for pair in edge_map.values():
        for a in range(len(pair)):
            for b in range(a + 1, len(pair)):
                i, j = pair[a], pair[b]
                if float(np.dot(n[i], n[j])) < cos_tol:
                    continue
                if abs(d[i] - d[j]) > dist_tol:
                    continue
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    gid = np.full(np.asarray(tris).shape[0], -1, dtype=np.int64)
    roots = {}
    for i in range(wt.shape[0]):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        gid[idx[i]] = roots[r]
    return gid, len(roots)


# -------------------------------------------------------------- the referee

BODY, CUT, NEW = 0, 1, 2
TRUTH_NAMES = {BODY: "body-surface", CUT: "cut-surface", NEW: "new-bevel"}

# Below this area a triangle's normal is numerical noise, so the referee cannot
# place it on a plane. 1e-12 m^2 is a 1 micron feature on a 0.6 m part.
SLIVER_AREA = 1e-12


def referee(fixture, verts, tris, dist_tol=1e-6, ang_tol_deg=0.5):
    """Ground truth from geometry alone.

    A triangle lies on a reference plane when its normal is parallel to the
    plane normal and its centroid sits on the plane. Everything else is new
    surface made by the bevel. Body and cut planes share axes, so the offset is
    what separates them -- x=+-0.3 is body, x=+-0.075 is the slot wall.
    """
    tris = np.asarray(tris, dtype=np.int64)
    n, area = tri_normals(verts, tris)
    cen = tri_centroids(verts, tris)
    truth = np.full(tris.shape[0], NEW, dtype=np.uint8)
    dev = np.full(tris.shape[0], np.inf)
    cos_tol = math.cos(math.radians(ang_tol_deg))

    for kind, axis, off in fixture.planes():
        para = np.abs(n[:, axis]) >= cos_tol
        dist = np.abs(cen[:, axis] - off)
        hit = para & (dist <= dist_tol)
        truth[hit] = CUT if kind == "cut" else BODY
        dev = np.minimum(dev, np.where(para, dist, np.inf))

    return {
        "truth": truth,
        "area": area,
        "plane_deviation": dev,
        "counts": {TRUTH_NAMES[k]: int(np.sum(truth == k))
                   for k in (BODY, CUT, NEW)},
        "area_by_truth": {TRUTH_NAMES[k]: float(np.sum(area[truth == k]))
                          for k in (BODY, CUT, NEW)},
    }


def plane_fidelity(fixture, verts, tris, ang_tol_deg=1.0, search=0.05):
    """Are the original flat faces still where the program put them?

    For every reference plane, take the triangles whose normal is aligned with
    it and which sit within `search` of it, then report the area-weighted mean
    offset error and the worst deviation. An exact kernel answers 0 for both.
    An implicit round moves the flat regions, and that error is exactly what
    makes geometric re-attribution unreliable, so it has to be measured rather
    than assumed.
    """
    tris = np.asarray(tris, dtype=np.int64)
    n, area = tri_normals(verts, tris)
    cen = tri_centroids(verts, tris)
    cos_tol = math.cos(math.radians(ang_tol_deg))
    per_plane = []
    for kind, axis, off in fixture.planes():
        m_ = (np.abs(n[:, axis]) >= cos_tol) & (
            np.abs(cen[:, axis] - off) <= search)
        if not np.any(m_):
            per_plane.append({"kind": kind, "axis": axis, "offset": off,
                              "area": 0.0, "mean_error": None,
                              "max_error": None})
            continue
        d = cen[m_, axis] - off
        a = area[m_]
        per_plane.append({
            "kind": kind, "axis": axis, "offset": off,
            "area": float(np.sum(a)),
            "mean_error": float(np.abs(np.sum(d * a) / np.sum(a))),
            "max_error": float(np.max(np.abs(d))),
        })
    have = [p for p in per_plane if p["area"] > 0]
    return {
        "planes_found": len(have),
        "planes_expected": len(per_plane),
        "worst_mean_offset_error": (max(p["mean_error"] for p in have)
                                   if have else None),
        "worst_max_offset_error": (max(p["max_error"] for p in have)
                                   if have else None),
        "per_plane": per_plane,
    }


# ---------------------------------------------------------------- provenance

def provenance_report(fixture, verts, tris, tri_tag, before_counts):
    """Score a candidate's provenance claim against the referee.

    Reports the exact quantity DESIGN.md section 13 cares about: how many
    logical faces answer `faces(#cut_face)` before and after the bevel, and
    whether any of the post-bevel matches are actually new bevel surface.
    """
    tris = np.asarray(tris, dtype=np.int64)
    tri_tag = np.asarray(tri_tag, dtype=np.uint8)
    ref = referee(fixture, verts, tris)
    truth = ref["truth"]
    gid, n_groups = coplanar_groups(verts, tris)

    def groups_of(mask):
        return len(set(gid[mask].tolist())) if np.any(mask) else 0

    claim_cut_any = (tri_tag & TAG_CUT) != 0
    claim_cut_primary = tri_tag == TAG_CUT
    claim_bevel = (tri_tag & TAG_BEVEL) != 0
    truth_cut = truth == CUT
    truth_new = truth == NEW

    false_pos = claim_cut_any & truth_new     # bevel surface answering #cut_face
    false_neg = truth_cut & ~claim_cut_any    # real cut surface that lost its tag
    untagged = tri_tag == 0

    out = {
        "n_logical_faces": n_groups,
        "truth_counts": ref["counts"],
        "claim_tri": {tag_str(int(v)): int(np.sum(tri_tag == v))
                      for v in sorted(set(tri_tag.tolist()))},
        "sel_cut_primary_tris": int(np.sum(claim_cut_primary)),
        "sel_cut_any_tris": int(np.sum(claim_cut_any)),
        "sel_cut_primary_faces": groups_of(claim_cut_primary),
        "sel_cut_any_faces": groups_of(claim_cut_any),
        "sel_bevel_faces": groups_of(claim_bevel),
        "truth_cut_faces": groups_of(truth_cut),
        "truth_new_faces": groups_of(truth_new),
        "false_positive_tris": int(np.sum(false_pos)),
        "false_negative_tris": int(np.sum(false_neg)),
        "untagged_tris": int(np.sum(untagged)),
        "expected_cut_faces": fixture.n_cut_faces(),
        "before": dict(before_counts),
    }
    # Where do the wrong matches actually sit, and are they real surface?
    #
    # A false positive a whole bevel-width from the cut plane is genuine drift:
    # new surface answering to an ancestor tag. A false positive sitting ON the
    # plane with ~zero area is a sliver whose normal is numerical noise, so the
    # referee could not place it -- that is a mesh-quality defect (E105), not a
    # provenance defect, and conflating the two would invent a failure that is
    # not there.
    ref_area = ref["area"]
    sliver = ref_area < SLIVER_AREA
    out["sliver_tris"] = int(np.sum(sliver))
    out["false_positive_sliver_tris"] = int(np.sum(false_pos & sliver))
    out["false_positive_real_tris"] = int(np.sum(false_pos & ~sliver))
    if np.any(false_pos):
        cen = tri_centroids(verts, tris)[false_pos]
        cut_planes = [(ax, off) for kind, ax, off in fixture.planes()
                      if kind == "cut"]
        d = np.min(np.stack([np.abs(cen[:, ax] - off)
                             for ax, off in cut_planes]), axis=0)
        out["false_positive_dist_to_cut_plane"] = {
            "median": float(np.median(d)), "max": float(np.max(d))}
        out["false_positive_area_total"] = float(np.sum(ref_area[false_pos]))
    out["drift_any"] = (out["sel_cut_any_faces"] !=
                        before_counts["sel_cut_any_faces"])
    out["drift_primary"] = (out["sel_cut_primary_faces"] !=
                            before_counts["sel_cut_primary_faces"])
    out["truth_matches_expectation"] = (
        out["truth_cut_faces"] == fixture.n_cut_faces())
    out["provenance_exact"] = (out["false_positive_tris"] == 0
                               and out["false_negative_tris"] == 0
                               and out["untagged_tris"] == 0)
    out["provenance_exact_ignoring_slivers"] = (
        out["false_positive_real_tris"] == 0
        and out["false_negative_tris"] == 0
        and out["untagged_tris"] == 0)
    return out


def before_snapshot(fixture, verts, tris, tri_tag):
    """The pre-bevel baseline the drift test compares against."""
    gid, n_groups = coplanar_groups(verts, tris)
    tri_tag = np.asarray(tri_tag, dtype=np.uint8)

    def groups_of(mask):
        return len(set(gid[mask].tolist())) if np.any(mask) else 0

    return {
        "tris": int(np.asarray(tris).shape[0]),
        "n_logical_faces": n_groups,
        "sel_cut_primary_faces": groups_of(tri_tag == TAG_CUT),
        "sel_cut_any_faces": groups_of((tri_tag & TAG_CUT) != 0),
        "sel_cut_primary_tris": int(np.sum(tri_tag == TAG_CUT)),
    }


# ------------------------------------------------------------------ plumbing

class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0


def rule(title):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def kv(k, v):
    print("  %-42s %s" % (k, v))


def ok(b):
    return "PASS" if b else "FAIL"


def dump_mesh(path, verts, tris, tri_tag):
    np.savez(path,
             verts=np.asarray(verts, dtype=np.float64),
             tris=np.asarray(tris, dtype=np.int64),
             tri_tag=np.asarray(tri_tag, dtype=np.uint8))


def load_mesh(path):
    z = np.load(path)
    return (np.array(z["verts"], dtype=np.float64),
            np.array(z["tris"], dtype=np.int64),
            np.array(z["tri_tag"], dtype=np.uint8))
