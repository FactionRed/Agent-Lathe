"""
Tier 1 -- numeric probes.  DESIGN.md 6.

    measure(final.bbox)              -> 0.600 x 0.600 x 0.600 m
    measure(distance(leg_01,leg_02)) -> 0.320 m
    count(final.tris)                -> 1,284
    volume(final)                    -> 0.164 m3
    center_of_mass(final)            -> (0, 0, 0.287)
    raycast(from=(0,-2,0.3), dir=+Y) -> hit final #cut_face at 1.71 m
    is_watertight(final)             -> true
    symmetry(final, plane = YZ)      -> 0.998

DESIGN.md 13 says agents will reach for renders when a probe would be faster and
more reliable, and that the mitigation is to make Tier 1 genuinely excellent. So:

  * every probe is a pure function of an immutable Mesh -- no session, no
    selection, no active object (DESIGN.md 2, principle 1);
  * every length carries a unit (DESIGN.md 4.1), including on the way *in*: a
    bare number where a length is expected is a TypeError, not a guess;
  * a probe that cannot answer exactly says which method it used and how far off
    it might be, instead of returning a bare number the agent will over-trust;
  * a probe that finds a problem says so on its own line, in the words the
    verifier uses (E101, E103, E105 ...), so the next step is obvious.

Beyond the doc's list this module adds surface_area, counts of
edges/faces/components/genus, degenerates, orientation and origin_placement.
Each is justified in README.md: they are the cheap checks that catch the failure
modes DESIGN.md 7 names as errors, and an agent that cannot see them has to
guess from a render.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np

from . import fmt, kernel
from . import mesh as _mesh
from .mesh import Mesh
from .units import (
    AXIS_NAMES,
    Angle,
    Area,
    Length,
    Point,
    Volume,
    as_direction,
    as_length,
    as_point,
    plane_name,
    plane_normal_axis,
)

# ------------------------------------------------------------------ plumbing


@dataclass(frozen=True)
class Probe:
    """One probe answer: an aligned headline plus optional detail lines.

    `fields` is the machine-readable form (units named in the keys) for callers
    that want values rather than text.
    """

    probe: str
    value: str
    fields: dict = field(default_factory=dict)
    detail: tuple[tuple[str, str], ...] = ()
    problems: tuple[str, ...] = ()

    def lines(self, width: int = fmt.LABEL_WIDTH, detail: bool = True) -> list[str]:
        out = [fmt.probe_line(self.probe, self.value, width=width)]
        if detail:
            out += [fmt.field_line(k, v) for k, v in self.detail]
        out += [fmt.field_line("problem", p) for p in self.problems]
        return out

    def text(self, width: int = fmt.LABEL_WIDTH, detail: bool = True) -> str:
        return "\n".join(self.lines(width=width, detail=detail))

    def __str__(self) -> str:
        return self.text()

    @property
    def ok(self) -> bool:
        return not self.problems


class _Verdict(Probe):
    """A Probe whose truth value is meaningful: `if is_watertight(m): ...`."""

    def __bool__(self) -> bool:
        return bool(self.fields.get("verdict", False))


class _Scored(Probe):
    """A Probe carrying a 0..1 score: `float(symmetry(m, "YZ"))`."""

    def __float__(self) -> float:
        return float(self.fields.get("score", float("nan")))


def _pt(arr) -> Point:
    return Point(float(arr[0]), float(arr[1]), float(arr[2]))


def _plural(n: int, noun: str, plural: str | None = None) -> str:
    return "%s %s" % (fmt.count(n), noun if n == 1 else (plural or noun + "s"))


def _diagonal(mesh: Mesh) -> float:
    lo, hi = _mesh.bounds(mesh)
    return float(np.linalg.norm(hi - lo))


def _scoped(mesh: Mesh, part: str | None) -> tuple[Mesh, str]:
    """Apply an optional part scope, and give back the expression prefix."""
    if part is None:
        return mesh, ""
    return mesh.part(part), "%s." % part


# ------------------------------------------------------------------- probes


def bbox(mesh: Mesh, part: str | None = None, unit: str = "m") -> Probe:
    """Axis-aligned bounds. `measure(final.bbox)` in DESIGN.md 6."""
    m, prefix = _scoped(mesh, part)
    lo, hi = _mesh.bounds(m)
    size = hi - lo
    centre = (lo + hi) / 2.0
    diag = float(np.linalg.norm(size))
    return Probe(
        probe="measure(%sbbox)" % prefix,
        value=_pt(size).format_size(unit),
        fields={
            "min_m": tuple(lo.tolist()),
            "max_m": tuple(hi.tolist()),
            "size_m": tuple(size.tolist()),
            "center_m": tuple(centre.tolist()),
            "diagonal_m": diag,
        },
        detail=(
            ("min / max", "%s / %s" % (_pt(lo).format(unit), _pt(hi).format(unit))),
            ("center", _pt(centre).format(unit)),
            ("diagonal", Length(diag).format(unit)),
        ),
    )


def distance(a: Mesh, b: Mesh, unit: str = "m", exact_gap: bool = True) -> Probe:
    """Separation of two meshes. `measure(distance(leg_01, leg_02))`.

    Headline is centre-to-centre of the two bounding boxes, which is what a
    program that placed parts by relationship (DESIGN.md 5.4) asked for. The
    surface gap -- the number that answers "do these collide?" -- is on the
    detail line, with its method named, because the kernel-free path can
    overestimate it (see README).
    """
    lo_a, hi_a = _mesh.bounds(a)
    lo_b, hi_b = _mesh.bounds(b)
    ca, cb = (lo_a + hi_a) / 2.0, (lo_b + hi_b) / 2.0
    delta = cb - ca
    centre_dist = float(np.linalg.norm(delta))

    # a bbox-to-bbox lower bound is free and bounds the search for min_gap
    sep = np.maximum(np.maximum(lo_b - hi_a, lo_a - hi_b), 0.0)
    bbox_gap = float(np.linalg.norm(sep))

    gap, method = None, "not computed"
    if exact_gap:
        man_a, why_a = _mesh.to_manifold(a)
        man_b, why_b = _mesh.to_manifold(b)
        if kernel.HAVE_MANIFOLD and man_a is not None and man_b is not None:
            search = max(centre_dist * 2.0, _diagonal(a) + _diagonal(b), 1e-9)
            gap = kernel.min_gap_exact(man_a, man_b, search)
            method = "kernel_min_gap (exact)"
            if gap >= search:
                method = "kernel_min_gap (no contact within %s)" % Length(search).format(unit)
        else:
            gap = kernel.vertex_to_surface_gap(a, b)
            method = "vertex_to_surface (lower bound; %s)" % (why_a or why_b or "no kernel")

    detail = [
        ("centers", "%s -> %s" % (_pt(ca).format(unit), _pt(cb).format(unit))),
        (
            "axis delta",
            "  ".join(
                "d%s %s" % (AXIS_NAMES[i].lower(), Length(abs(delta[i])).format(unit))
                for i in range(3)
            ),
        ),
        ("bbox gap", Length(bbox_gap).format(unit)),
    ]
    fields = {
        "center_distance_m": centre_dist,
        "center_a_m": tuple(ca.tolist()),
        "center_b_m": tuple(cb.tolist()),
        "delta_m": tuple(delta.tolist()),
        "bbox_gap_m": bbox_gap,
        "surface_gap_m": gap,
        "gap_method": method,
    }
    if gap is not None:
        detail.insert(2, ("surface gap", "%s   method=%s" % (Length(gap).format(unit), method)))
    problems = []
    if gap is not None and gap <= 0.0 and bbox_gap <= 0.0:
        problems.append("%s and %s interpenetrate or touch (gap 0)" % (a.name, b.name))
    return Probe(
        probe="measure(distance(%s, %s))" % (a.name, b.name),
        value=Length(centre_dist).format(unit),
        fields=fields,
        detail=tuple(detail),
        problems=tuple(problems),
    )


def counts(mesh: Mesh, part: str | None = None) -> Probe:
    """Cardinality of everything an agent asserts on. `count(final.tris)`.

    Includes components and genus because DESIGN.md 7 makes an unexpected
    component count an error (E108) and the prototype found the doc's own crate
    example severed into six pieces with nothing to say so.
    """
    m, prefix = _scoped(mesh, part)
    topo = m.topology
    n_faces, degrees = _logical_faces(m)
    per_comp_genus = _genus_per_component(m)
    # genus is only defined for a closed orientable surface; say so rather than
    # printing a confident 0 for a mesh with a hole in it
    total_genus = sum(per_comp_genus) if per_comp_genus else None

    fields = {
        "tris": m.n_tris,
        "verts": int(np.count_nonzero(topo.vert_used)),
        "verts_stored": m.n_verts,
        "edges": topo.n_edges,
        "faces_logical": n_faces,
        "face_degrees": degrees,
        "components": topo.n_components,
        "genus": total_genus,
        "genus_per_component": tuple(per_comp_genus),
        "boundary_edges": int(len(topo.boundary_edges)),
        "nonmanifold_edges": int(len(topo.nonmanifold_edges)),
    }
    detail = [
        ("verts / edges", "%s / %s" % (fmt.count(fields["verts"]), fmt.count(topo.n_edges))),
        ("faces (logical)", "%s   %s" % (fmt.count(n_faces), _degree_text(degrees))),
        ("components", fmt.count(topo.n_components)),
        ("genus", _genus_text(total_genus, per_comp_genus)),
    ]
    problems = []
    if fields["boundary_edges"]:
        problems.append(
            "E101 %s -- the surface is open"
            % _plural(fields["boundary_edges"], "boundary edge")
        )
    if fields["nonmanifold_edges"]:
        problems.append(
            "E101 %s shared by more than two faces"
            % _plural(fields["nonmanifold_edges"], "edge")
        )
    return _Counts(
        probe="count(%stris)" % prefix,
        value=fmt.count(m.n_tris),
        fields=fields,
        detail=tuple(detail),
        problems=tuple(problems),
    )


class _Counts(Probe):
    """Counts render as several `count(x)` lines: they are separate questions."""

    def lines(self, width: int = fmt.LABEL_WIDTH, detail: bool = True) -> list[str]:
        f = self.fields
        genus = _genus_text(f["genus"], f["genus_per_component"])
        out = [
            fmt.probe_line("count(tris)", fmt.count(f["tris"]), width=width),
            fmt.probe_line(
                "count(verts) / count(edges)",
                "%s / %s" % (fmt.count(f["verts"]), fmt.count(f["edges"])),
                width=width,
            ),
            fmt.probe_line(
                "count(faces)",
                "%s logical   %s" % (fmt.count(f["faces_logical"]),
                                     _degree_text(f["face_degrees"])),
                width=width,
            ),
            fmt.probe_line("count(components)", fmt.count(f["components"]), width=width),
            fmt.probe_line("genus", genus, width=width),
        ]
        out += [fmt.field_line("problem", p) for p in self.problems]
        return out


def _genus_text(total, per_component) -> str:
    if total is None:
        return "n/a (not a closed orientable surface)"
    if len(per_component) > 1:
        return "%d  (per component %s)" % (
            total,
            ", ".join(str(g) for g in per_component),
        )
    return "%d" % total


def _degree_name(d: int) -> str:
    return {3: "tri", 4: "quad"}.get(d, "%d-gon" % d)


def _degree_text(degrees: dict) -> str:
    if not degrees:
        return "(none)"
    return "  ".join(
        "%s:%s" % (_degree_name(d), fmt.count(n)) for d, n in sorted(degrees.items())
    )


def _logical_faces(mesh: Mesh) -> tuple[int, dict]:
    """Group triangles by face_id and recover each group's polygon degree.

    A coplanar group's outline is the set of its edges used exactly once inside
    the group; interior diagonals are used twice. Counting the once-used edges
    gives the n-gon degree, which is what E102 (n-gons in emitted geometry) is
    about. Vectorised: one sort over (group, edge) pairs, no Python loop over
    faces.
    """
    if mesh.face_id is None or mesh.n_tris == 0:
        return mesh.n_tris, ({3: mesh.n_tris} if mesh.n_tris else {})
    topo = mesh.topology
    # face_id is scoped to the original solid it came from: two different
    # ancestors reuse the same small ids. The logical face key is therefore
    # (part_id, face_id), never face_id alone -- see README, "kernel footguns".
    if mesh.part_id is not None:
        span = int(mesh.face_id.max()) + 1 if mesh.n_tris else 1
        key = mesh.part_id * span + mesh.face_id
    else:
        key = mesh.face_id
    groups, group_index = np.unique(key, return_inverse=True)
    group_index = group_index.reshape(-1)
    n_edges = max(topo.n_edges, 1)
    pair_key = np.repeat(group_index, 3) * n_edges + topo.tri_edges.reshape(-1)
    uniq, cnt = np.unique(pair_key, return_counts=True)
    outline_groups = (uniq[cnt == 1] // n_edges).astype(np.int64)
    degree_per_group = np.bincount(outline_groups, minlength=len(groups))
    hist = np.bincount(degree_per_group)
    degrees = {int(d): int(n) for d, n in enumerate(hist) if n}
    return int(len(groups)), degrees


def _genus_per_component(mesh: Mesh) -> list[int]:
    """Euler characteristic per component: V - E + F = 2 - 2g (closed only)."""
    topo = mesh.topology
    if mesh.n_tris == 0 or not topo.is_closed or not topo.is_edge_manifold:
        return []
    labels = topo.vert_component
    tri_label = labels[mesh.tris[:, 0]]
    out = []
    for comp in np.unique(tri_label):
        tri_rows = np.flatnonzero(tri_label == comp)
        f = len(tri_rows)
        e = len(np.unique(topo.tri_edges[tri_rows].reshape(-1)))
        v = len(np.unique(mesh.tris[tri_rows].reshape(-1)))
        chi = v - e + f
        out.append(int(round((2 - chi) / 2)))
    return out


def volume(mesh: Mesh, part: str | None = None, unit: str = "m^3") -> Probe:
    """Enclosed volume by the divergence theorem. `volume(final)`.

    Exact for a closed, consistently wound mesh: a unit cube returns exactly
    1.0 m3. On an open mesh the sum still evaluates but means nothing, so the
    probe says so instead of returning a confident wrong number.
    """
    m, prefix = _scoped(mesh, part)
    signed = _mesh.signed_volume(m)
    topo = m.topology
    closed = topo.is_closed and topo.is_edge_manifold and topo.is_oriented
    inverted = closed and signed < 0.0

    detail = [("signed", Volume(signed).format(unit))]
    problems = []
    if not closed:
        reasons = []
        if not topo.is_closed:
            reasons.append(_plural(len(topo.boundary_edges), "boundary edge"))
        if not topo.is_edge_manifold:
            reasons.append(_plural(len(topo.nonmanifold_edges), "non-manifold edge"))
        if not topo.is_oriented:
            reasons.append(
                _plural(topo.duplicate_halfedges, "edge") + " wound twice the same way"
            )
        problems.append(
            "volume is undefined on an open surface (%s); the number above is "
            "the divergence-theorem sum, not an enclosed volume" % ", ".join(reasons)
        )
    if inverted:
        problems.append("E103 negative signed volume -- face normals point inward")
    return Probe(
        probe="volume(%s)" % (prefix.rstrip(".") or m.name),
        value=Volume(abs(signed)).format(unit) + ("" if closed else "  UNRELIABLE"),
        fields={
            "volume_m3": abs(signed),
            "signed_volume_m3": signed,
            "closed": bool(closed),
            "inverted": bool(inverted),
        },
        detail=tuple(detail),
        problems=tuple(problems),
    )


def surface_area(mesh: Mesh, part: str | None = None, unit: str = "m^2") -> Probe:
    """Total triangle area. Needed for texel density (W302) and as the cheapest
    sanity check that a boolean did not leave interior shells behind."""
    m, prefix = _scoped(mesh, part)
    areas = _mesh.tri_areas(m)
    total = float(areas.sum())
    return Probe(
        probe="surface_area(%s)" % (prefix.rstrip(".") or m.name),
        value=Area(total).format(unit),
        fields={
            "area_m2": total,
            "tri_area_min_m2": float(areas.min()) if len(areas) else 0.0,
            "tri_area_max_m2": float(areas.max()) if len(areas) else 0.0,
        },
    )


def center_of_mass(mesh: Mesh, part: str | None = None, unit: str = "m") -> Probe:
    """Centroid of the enclosed solid at uniform density. `center_of_mass(final)`.

    Exact tetrahedron decomposition for a closed mesh. For an open mesh there is
    no solid, so the probe falls back to the area-weighted shell centroid and
    labels it -- those are different points and conflating them hides real bugs.
    """
    m, prefix = _scoped(mesh, part)
    topo = m.topology
    closed = topo.is_closed and topo.is_edge_manifold and topo.is_oriented
    detail = []
    problems = []
    if closed:
        centre, vol = _mesh.volume_centroid(m)
        method = "volume_centroid (exact)"
    else:
        centre, _area = _mesh.area_centroid(m)
        method = "area_centroid (mesh is not closed; this is the shell centre)"
        problems.append("no enclosed solid, so there is no volumetric centre of mass")
    if centre is None:
        centre = np.zeros(3)
        method = "undefined (zero measure)"
    lo, hi = _mesh.bounds(m)
    size = np.maximum(hi - lo, 1e-15)
    rel = (centre - lo) / size
    detail.append(("method", method))
    detail.append(
        ("relative to bbox", "(%.3f, %.3f, %.3f) of size" % (rel[0], rel[1], rel[2]))
    )
    return Probe(
        probe="center_of_mass(%s)" % (prefix.rstrip(".") or m.name),
        value=_pt(centre).format(unit),
        fields={
            "center_m": tuple(centre.tolist()),
            "relative_to_bbox": tuple(rel.tolist()),
            "method": method,
        },
        detail=tuple(detail),
        problems=tuple(problems),
    )


# ------------------------------------------------------------------ raycast


def _ray_hits(mesh: Mesh, origin: np.ndarray, direction: np.ndarray,
              max_distance: float, eps: float):
    """Moller-Trumbore over every triangle. Returns (t, tri, front) sorted.

    No BVH: O(T) per ray, vectorised. Hits closer together than `eps` are merged
    so that a ray crossing a shared edge is one hit, not two.
    """
    a, b, c = _mesh.tri_corners(mesh)
    e1, e2 = b - a, c - a
    h = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, h)
    parallel = np.abs(det) < 1e-15
    inv = np.where(parallel, 1.0, 1.0 / np.where(parallel, 1.0, det))
    s = origin - a
    u = np.einsum("ij,ij->i", s, h) * inv
    q = np.cross(s, e1)
    v = np.einsum("j,ij->i", direction, q) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    bary_eps = 1e-9
    hit = (
        ~parallel
        & (u >= -bary_eps)
        & (v >= -bary_eps)
        & (u + v <= 1.0 + bary_eps)
        & (t >= 0.0)
        & (t <= max_distance)
    )
    idx = np.flatnonzero(hit)
    if len(idx) == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    order = np.argsort(t[idx], kind="stable")
    idx = idx[order]
    ts = t[idx]
    keep = np.concatenate([[True], np.diff(ts) > eps])
    idx, ts = idx[keep], ts[keep]
    # det = e1 . (dir x e2) = -dir . (e1 x e2) = -dir . normal, so a positive
    # det means the ray runs against the outward normal: a front-face hit.
    front = det[idx] > 0.0
    return ts, idx, front


def raycast(mesh: Mesh, origin, direction, unit: str | None = None,
            max_distance: Length | None = None, report_unit: str = "m") -> Probe:
    """Shoot a ray and report what was hit, where, and how far.

        raycast(from = (0,-2,0.3), dir = +Y) -> hit final #cut_face at 1.71 m

    `origin` is a Point, or a 3-tuple *with* `unit=` -- a bare tuple with no
    unit is the scale bug DESIGN.md 4.1 exists to kill. `direction` is
    dimensionless: '+Y', '-Z' or any 3-vector.

    On a miss the probe also reports the ray's nearest approach to the mesh,
    because "you missed by 41 mm" is actionable and "miss" is not.
    """
    o = as_point(origin, unit, what="origin")
    d = np.array(as_direction(direction), dtype=np.float64)
    op = np.array(o.si, dtype=np.float64)
    diag = _diagonal(mesh)
    limit = 4.0 * diag if max_distance is None else as_length(max_distance).si
    eps = max(diag, 1.0) * 1e-9

    ts, tris, front = _ray_hits(mesh, op, d, limit, eps)
    dir_txt = _direction_text(d, direction)
    probe_expr = "raycast(from = %s, dir = %s)" % (o.format(report_unit), dir_txt)

    if len(ts) == 0:
        approach_d, approach_t = _nearest_approach(mesh, op, d, limit)
        return Probe(
            probe=probe_expr,
            value="miss",
            fields={
                "hit": False,
                "searched_m": limit,
                "nearest_approach_m": approach_d,
                "nearest_approach_at_t_m": approach_t,
            },
            detail=(
                ("searched", "%s along %s" % (Length(limit).format(report_unit), dir_txt)),
                (
                    "nearest approach",
                    "%s at %s along the ray"
                    % (Length(approach_d).format(report_unit),
                       Length(approach_t).format(report_unit)),
                ),
            ),
            problems=("no geometry along that ray -- check the aim, not the mesh",),
        )

    t0, tri0 = float(ts[0]), int(tris[0])
    pos = op + d * t0
    normal = _mesh.tri_normals(mesh)[tri0]
    # name what was hit the way DESIGN.md 6 does -- "hit final #cut_face" -- using
    # the provenance tag when the mesh carries one
    part = None
    if mesh.part_id is not None:
        part = mesh.parts.get(int(mesh.part_id[tri0]))
    face = int(mesh.face_id[tri0]) if mesh.face_id is not None else tri0
    material = (
        mesh.material_name(mesh.material_id[tri0]) if mesh.material_id is not None else None
    )
    incidence = Angle(math.acos(min(1.0, abs(float(np.dot(normal, d))))))
    detail = [
        ("position", _pt(pos).format(report_unit)),
        (
            "normal",
            "(%.3f, %.3f, %.3f)   incidence %s   %s side"
            % (normal[0], normal[1], normal[2], incidence.format(1),
               "front" if front[0] else "back"),
        ),
        ("part / face", "%s / face group %d" % (part or "(unnamed)", face)),
    ]
    if material:
        detail.append(("material", material))
    if len(ts) > 1:
        detail.append(
            (
                "hits along ray",
                "%d   first %s  last %s  span %s"
                % (len(ts), Length(t0).format(report_unit),
                   Length(float(ts[-1])).format(report_unit),
                   Length(float(ts[-1]) - t0).format(report_unit)),
            )
        )
    else:
        detail.append(("hits along ray", "1"))
    detail.append(("searched", Length(limit).format(report_unit)))

    problems = []
    if not front[0]:
        problems.append(
            "E103 first hit is a back face -- the ray starts inside the solid, or "
            "that normal is inverted, or the front surface is missing"
        )
    if len(ts) % 2 == 1:
        problems.append(
            "odd crossing count (%d) -- the surface is not closed along this ray "
            "(a hole), or the ray started inside it" % len(ts)
        )
    return _Verdict(
        probe=probe_expr,
        value="hit %s%s at %s"
        % (mesh.name, " #%s" % part if part else "", Length(t0).format(report_unit)),
        fields={
            "hit": True,
            "verdict": True,
            "distance_m": t0,
            "position_m": tuple(pos.tolist()),
            "normal": tuple(normal.tolist()),
            "tri": tri0,
            "face_id": face,
            "part": part,
            "material": material,
            "front_face": bool(front[0]),
            "hits": int(len(ts)),
            "all_distances_m": tuple(float(x) for x in ts),
            "searched_m": limit,
        },
        detail=tuple(detail),
        problems=tuple(problems),
    )


def _direction_text(d: np.ndarray, original) -> str:
    if isinstance(original, str):
        return original.strip().upper().replace(" ", "")
    for i in range(3):
        for sign, label in ((1.0, "+"), (-1.0, "-")):
            probe = np.zeros(3)
            probe[i] = sign
            if np.allclose(d, probe):
                return label + AXIS_NAMES[i]
    return "(%.3f, %.3f, %.3f)" % tuple(d)


def _nearest_approach(mesh: Mesh, origin: np.ndarray, direction: np.ndarray,
                      limit: float):
    """Closest distance between the ray segment and any vertex of the mesh."""
    used = np.unique(mesh.tris) if mesh.n_tris else np.arange(mesh.n_verts)
    if len(used) == 0:
        return float("inf"), 0.0
    v = mesh.verts[used]
    t = np.clip((v - origin) @ direction, 0.0, limit)
    d = np.linalg.norm(v - (origin + t[:, None] * direction), axis=1)
    i = int(np.argmin(d))
    return float(d[i]), float(t[i])


# --------------------------------------------------------------- integrity


def is_watertight(mesh: Mesh, part: str | None = None) -> _Verdict:
    """`is_watertight(final) -> true`, and when false, why.

    Three independent conditions, reported separately because the repairs differ:
      closed          every edge has exactly two triangles (no holes)
      edge-manifold   no edge has three or more triangles (E101)
      oriented        no edge traversed the same way twice (E103, winding clash)
    """
    m, prefix = _scoped(mesh, part)
    topo = m.topology
    n_boundary = int(len(topo.boundary_edges))
    n_nonman = int(len(topo.nonmanifold_edges))
    verdict = topo.is_closed and topo.is_edge_manifold and topo.is_oriented
    loops = topo.boundary_loops()
    signed = _mesh.signed_volume(m)

    reasons = []
    if n_boundary:
        reasons.append("%s in %s"
                       % (_plural(n_boundary, "boundary edge"), _plural(loops, "hole")))
    if n_nonman:
        reasons.append(_plural(n_nonman, "non-manifold edge"))
    if not topo.is_oriented:
        reasons.append(_plural(topo.duplicate_halfedges, "duplicate half-edge"))

    detail = [
        ("closed", fmt.boolean(topo.is_closed)),
        ("edge manifold", fmt.boolean(topo.is_edge_manifold)),
        ("oriented", fmt.boolean(topo.is_oriented)),
        ("components", fmt.count(topo.n_components)),
        ("orientation", "outward" if signed >= 0 else "inward (inverted)"),
    ]
    if n_boundary:
        detail.append(("holes", "%d  (%s)" % (loops, _plural(n_boundary, "boundary edge"))))
    problems = []
    if not verdict:
        problems.append("E101 not watertight: " + "; ".join(reasons))
    if verdict and signed < 0:
        problems.append("E103 closed but inside out (negative signed volume)")
    return _Verdict(
        probe="is_watertight(%s)" % (prefix.rstrip(".") or m.name),
        value=fmt.boolean(verdict) + ("" if verdict else "  (%s)" % "; ".join(reasons)),
        fields={
            "verdict": bool(verdict),
            "closed": bool(topo.is_closed),
            "edge_manifold": bool(topo.is_edge_manifold),
            "oriented": bool(topo.is_oriented),
            "boundary_edges": n_boundary,
            "holes": loops,
            "nonmanifold_edges": n_nonman,
            "duplicate_halfedges": int(topo.duplicate_halfedges),
            "components": int(topo.n_components),
            "outward": bool(signed >= 0),
        },
        detail=tuple(detail),
        problems=tuple(problems),
    )


def degenerates(mesh: Mesh, part: str | None = None,
                weld_tol: Length | None = None) -> _Verdict:
    """Zero-area faces, welded-but-not-merged verts, orphans, duplicate faces.

    This is E105 in DESIGN.md 7 (loose verts/edges, zero-area faces). Cheap, and
    catches the residue booleans and hand-built arrays leave behind.
    """
    m, prefix = _scoped(mesh, part)
    diag = max(_diagonal(m), 1e-12)
    tol = (Length(diag * 1e-6) if weld_tol is None else as_length(weld_tol)).si
    areas = _mesh.tri_areas(m)
    area_tol = (diag * 1e-6) ** 2
    zero_area = int(np.count_nonzero(areas <= area_tol))
    repeated_idx = int(
        np.count_nonzero(
            (m.tris[:, 0] == m.tris[:, 1])
            | (m.tris[:, 1] == m.tris[:, 2])
            | (m.tris[:, 0] == m.tris[:, 2])
        )
    )
    topo = m.topology
    orphan = int(np.count_nonzero(~topo.vert_used))
    coincident = _mesh._count_duplicate_rows(
        np.round(m.verts[topo.vert_used] / max(tol, 1e-15)).astype(np.int64)
    )
    duplicate_faces = _mesh._count_duplicate_rows(np.sort(m.tris, axis=1))

    found = {
        "zero_area_tris": zero_area,
        "degenerate_index_tris": repeated_idx,
        "orphan_verts": orphan,
        "coincident_verts": coincident,
        "duplicate_tris": duplicate_faces,
    }
    bad = {k: v for k, v in found.items() if v}
    value = "none" if not bad else "  ".join("%s:%s" % (k, fmt.count(v)) for k, v in bad.items())
    problems = []
    if bad:
        problems.append(
            "E105 degenerate geometry: "
            + ", ".join("%s %s" % (fmt.count(v), k.replace("_", " ")) for k, v in bad.items())
        )
    return _Verdict(
        probe="degenerates(%s)" % (prefix.rstrip(".") or m.name),
        value=value,
        fields={"verdict": not bad, "weld_tol_m": tol, **found},
        detail=(("weld tolerance", Length(tol).format("mm")),),
        problems=tuple(problems),
    )


_ORIGIN_NAMES = {
    (0.5, 0.5, 0.0): "bottom_center",
    (0.5, 0.5, 0.5): "center",
    (0.5, 0.5, 1.0): "top_center",
    (0.0, 0.0, 0.0): "min_corner",
}


def origin_placement(mesh: Mesh, tol: Length | None = None,
                     unit: str = "mm") -> _Verdict:
    """Where the world origin sits inside the bounds, and what to call it.

    DESIGN.md 4 declares origin in the program ("@ origin(bottom_center)") and
    W201 warns when the emitted origin is not at the declared reference point.
    An agent cannot check that from a render; it is two subtractions here.
    """
    lo, hi = _mesh.bounds(mesh)
    size = hi - lo
    safe = np.where(np.abs(size) > 1e-15, size, 1.0)
    rel = (0.0 - lo) / safe
    tol_m = (Length(max(float(np.linalg.norm(size)), 1e-12) * 1e-4)
             if tol is None else as_length(tol)).si
    label = "off"
    for key, name in _ORIGIN_NAMES.items():
        target = lo + np.array(key) * size
        if np.all(np.abs(target) <= tol_m):
            label = name
            break
    inside = bool(np.all(lo - tol_m <= 0.0) and np.all(hi + tol_m >= 0.0))
    offset_from_center = -(lo + hi) / 2.0
    return _Verdict(
        probe="origin_placement(%s)" % mesh.name,
        value="%s   at (%.3f, %.3f, %.3f) of bounds" % (label, rel[0], rel[1], rel[2]),
        fields={
            "verdict": label != "off",
            "label": label,
            "relative": tuple(rel.tolist()),
            "inside_bbox": inside,
            "offset_from_bbox_center_m": tuple(offset_from_center.tolist()),
            "tol_m": tol_m,
        },
        detail=(
            ("offset from center", _pt(offset_from_center).format(unit, dp=1)),
            ("origin inside bbox", fmt.boolean(inside)),
        ),
        problems=()
        if label != "off"
        else (
            "W201 origin is not at a named reference point -- "
            "relative position (%.3f, %.3f, %.3f) of the bounds" % tuple(rel.tolist()),
        ),
    )


# ---------------------------------------------------------------- symmetry


def symmetry(mesh: Mesh, plane="YZ", about="bbox_center", method: str = "auto",
             unit: str = "m") -> _Scored:
    """`symmetry(final, plane = YZ) -> 0.998`. A 0..1 score, method declared.

    Two methods, both reported by name because they are not the same number:

    volume_iou (default when the mesh is a closed solid, exact)
        Mirror the solid across the plane and take the volumetric
        intersection-over-union: vol(A and mirror(A)) / vol(A or mirror(A)).
        This is the fraction of the object that maps onto itself. Exactly 1.0
        for an exactly symmetric solid, and tessellation cannot fake it.

    surface_sample (fallback for open or non-manifold meshes, approximate)
        Mirror deterministic area-weighted surface samples and measure the
        distance from each back to the nearest *triangle*. Reports rms and worst
        deviation as lengths; the score is 1 - rms/bbox_diagonal, clamped to
        0..1. Monotone with asymmetry but NOT comparable to an IoU -- read the
        deviation lengths, not the score, when this method is in play.

    `about` picks the mirror plane position: "bbox_center" (default), "origin",
    "centroid", or a Length offset along the plane normal. The chosen position
    is always printed, because "symmetric about its own middle" and "symmetric
    about the world YZ plane" are different questions and confusing them is a
    real bug.
    """
    axis = plane_normal_axis(plane)
    lo, hi = _mesh.bounds(mesh)
    if isinstance(about, str):
        key = about.lower()
        if key in ("bbox_center", "bbox_centre", "center", "centre"):
            offset = float((lo[axis] + hi[axis]) / 2.0)
        elif key == "origin":
            offset = 0.0
        elif key in ("centroid", "com", "center_of_mass"):
            centre, _ = _mesh.volume_centroid(mesh)
            if centre is None:
                centre, _ = _mesh.area_centroid(mesh)
            offset = float(centre[axis])
        else:
            raise ValueError(
                "about must be 'bbox_center', 'origin', 'centroid' or a Length, got %r"
                % (about,)
            )
        about_label = key
    else:
        offset = as_length(about, what="about").si
        about_label = "explicit"

    topo = mesh.topology
    solid_ok = topo.is_closed and topo.is_edge_manifold and topo.is_oriented
    man, why = (None, "mesh is not a closed solid")
    if solid_ok:
        man, why = _mesh.to_manifold(mesh)
    want = method.lower()
    use_iou = want in ("auto", "volume_iou") and man is not None and kernel.HAVE_MANIFOLD
    if want == "volume_iou" and not use_iou:
        raise ValueError("volume_iou needs a closed solid and manifold3d (%s)" % why)
    if want == "surface_sample":
        why = "method=surface_sample requested"

    plane_txt = "%s = %s (%s)" % (
        AXIS_NAMES[axis].lower(),
        Length(offset).format(unit),
        about_label,
    )
    probe_expr = "symmetry(%s, plane = %s)" % (mesh.name, plane_name(axis))

    if use_iou:
        mirrored = kernel.mirror_about(man, axis, offset)
        ov = kernel.volume_overlap(man, mirrored)
        score = max(0.0, min(1.0, ov["iou"]))
        asym = max(0.0, ov["union"] - ov["intersection"])  # never print "-0.00%"
        detail = [
            ("method", "volume_iou (exact, manifold kernel)"),
            ("mirror plane", plane_txt),
            (
                "asymmetric vol",
                "%s of %s union   %s"
                % (
                    Volume(asym).format("m^3", dp=6),
                    Volume(ov["union"]).format("m^3", dp=6),
                    fmt.pct(asym / ov["union"] if ov["union"] else 0.0, 2),
                ),
            ),
        ]
        fields = {
            "score": score,
            "method": "volume_iou",
            "plane": plane_name(axis),
            "plane_offset_m": offset,
            "intersection_m3": ov["intersection"],
            "union_m3": ov["union"],
            "asymmetric_m3": asym,
        }
    else:
        dev = kernel.surface_mirror_deviation(mesh, axis, offset)
        diag = max(_diagonal(mesh), 1e-12)
        score = max(0.0, min(1.0, 1.0 - dev["rms"] / diag))
        detail = [
            ("method", "surface_sample (approximate; %s)" % why),
            ("mirror plane", plane_txt),
            (
                "deviation",
                "rms %s   max %s at %s"
                % (
                    Length(dev["rms"]).format("mm", dp=2),
                    Length(dev["max"]).format("mm", dp=2),
                    _pt(dev["max_at"]).format(unit),
                ),
            ),
            (
                "score formula",
                "1 - rms / bbox_diagonal (%s); not an IoU"
                % Length(diag).format(unit),
            ),
            (
                "samples",
                fmt.count(dev["samples"])
                + (
                    ""
                    if dev["stride"] == 1
                    else " of %s (every %dth -- see README on the sampling budget)"
                    % (fmt.count(dev["samples_available"]), dev["stride"])
                ),
            ),
        ]
        fields = {
            "score": score,
            "method": "surface_sample",
            "plane": plane_name(axis),
            "plane_offset_m": offset,
            "rms_deviation_m": dev["rms"],
            "max_deviation_m": dev["max"],
            "max_deviation_at_m": tuple(float(x) for x in dev["max_at"]),
            "samples": dev["samples"],
            "sample_stride": dev["stride"],
        }
    return _Scored(
        probe=probe_expr,
        value=fmt.score(score),
        fields=fields,
        detail=tuple(detail),
    )


def symmetry_planes(mesh: Mesh, about="bbox_center", method: str = "auto") -> Probe:
    """All three principal planes at once -- the question agents actually ask."""
    results = {p: symmetry(mesh, p, about=about, method=method) for p in ("YZ", "XZ", "XY")}
    value = "  ".join("%s %s" % (p, r.value) for p, r in results.items())
    used = sorted({r.fields["method"] for r in results.values()})
    return _Scored(
        probe="symmetry(%s, all planes)" % mesh.name,
        value=value,
        fields={
            "score": min(float(r) for r in results.values()),
            "scores": {p: float(r) for p, r in results.items()},
            "method": "+".join(used),
        },
        detail=(("method", "+".join(used)),),
    )


# ------------------------------------------------------------------ report


def report(mesh: Mesh, unit: str = "m", detail: bool = False,
           symmetry_method: str = "auto") -> str:
    """Every Tier 1 probe in one block. One call, no state, no images.

    This is the thing an agent should reach for after every `apply`. `detail`
    adds the per-probe sub-lines; the default is the compact card.
    """
    probes = [
        bbox(mesh, unit=unit),
        counts(mesh),
        volume(mesh),
        surface_area(mesh),
        center_of_mass(mesh, unit=unit),
        is_watertight(mesh),
        degenerates(mesh),
        origin_placement(mesh),
    ]
    if symmetry_method != "off":  # the only probe that costs more than microseconds
        probes.append(symmetry_planes(mesh, method=symmetry_method))
    lines = []
    for p in probes:
        lines.extend(p.lines(detail=detail))
    problems = [pr for p in probes for pr in p.problems]
    head = fmt.header(
        "probe %s" % mesh.name,
        "tier 1 · %s tris · %s"
        % (fmt.count(mesh.n_tris), "clean" if not problems else "%d problem(s)" % len(problems)),
    )
    if problems and not detail:
        lines.append("")
        lines.extend("  ! " + p for p in problems)
    return "\n".join([head] + lines)


__all__ = [
    "Probe",
    "bbox",
    "distance",
    "counts",
    "volume",
    "surface_area",
    "center_of_mass",
    "raycast",
    "is_watertight",
    "degenerates",
    "origin_placement",
    "symmetry",
    "symmetry_planes",
    "report",
]
