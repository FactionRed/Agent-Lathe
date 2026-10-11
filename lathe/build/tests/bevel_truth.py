"""
A referee for bevel on curved input. DESIGN.md section 14.7.

The bevel spike's referee (prototype/bevel_spike/common.py) recomputes provenance
from a hand-listed set of reference planes, which is why it could only grade
axis-aligned boxes: nobody lists the 48 facet planes of a cylinder by hand. This
one gets truth for any *faceted solid of revolution* from one fact.

    A solid revolved in n segments is a stack of regular n-gons. In
    (apothem, z) space it is a plain 2D polygon, and every quantity worth
    checking is a closed form over that polygon.

The apothem is the revolve radius times cos(pi/n), and a bevel's width is measured
along the facets, i.e. in exactly this slice, so no approximation enters. Bevel a
corner of the 2D polygon with a circular arc, sum frusta, and the result is the
volume and the area of every face of the beveled 3D solid -- independent of
anything bmesh does. If the two disagree, bmesh is wrong, or the spec for what
`bevel(width=...)` means is.

Pure numpy. No Blender, no Manifold: these formulas are testable on their own,
and test_bevel_curved.py checks them before it trusts them to grade anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

BEVEL = "bevel"


@dataclass(frozen=True)
class Profile:
    """A closed polygon in (apothem, z). Edge i runs vertex i -> i+1 (wrapping)
    and carries tags[i]. None marks an edge on the axis, which is not a surface."""

    pts: Tuple[Tuple[float, float], ...]
    tags: Tuple[Optional[str], ...]

    def __post_init__(self):
        if len(self.pts) != len(self.tags):
            raise ValueError("one tag per edge: %d points, %d tags"
                             % (len(self.pts), len(self.tags)))

    def edges(self):
        n = len(self.pts)
        for i in range(n):
            yield i, self.pts[i], self.pts[(i + 1) % n], self.tags[i]


def profile_from_revolve(points: Sequence[Tuple[float, float]], n: int,
                         tag: str = "body") -> Profile:
    """The polygon `revolve(points, segments=n)` is, in apothem space.

    `points` is the same (radius, height) list `revolve` takes. The closing edge
    runs back along the axis when the first and last points are not both on it.
    """
    c = math.cos(math.pi / n)
    pts = [(float(r) * c, float(z)) for r, z in points]
    tags: List[Optional[str]] = []
    for i, (p, q) in enumerate(zip(pts, pts[1:] + pts[:1])):
        on_axis = abs(p[0]) < 1e-12 and abs(q[0]) < 1e-12
        tags.append(None if on_axis else tag)
    return Profile(tuple(pts), tuple(tags))


def stacked_cylinders(parts: Sequence[Tuple[float, float, str]], n: int) -> Profile:
    """Cylinders sitting on each other, bottom to top: (radius, height, tag).

    The silhouette of a union meant to be one body. Each part keeps its own side
    wall; the exposed tread of a wider part below carries the lower part's tag,
    which is what a boolean leaves behind.
    """
    c = math.cos(math.pi / n)
    pts: List[Tuple[float, float]] = [(0.0, 0.0)]
    tags: List[Optional[str]] = []
    z = 0.0
    prev = None
    for r, h, tag in parts:
        a = r * c
        if prev is None:
            pts.append((a, 0.0))
            tags.append(tag)                      # the base
        else:
            if a >= prev[0]:
                raise ValueError("each part must be narrower than the one below")
            pts.append((a, z))
            tags.append(prev[1])                  # the tread: the lower part's top
        z += h
        pts.append((a, z))
        tags.append(tag)                          # this part's wall
        prev = (a, tag)
    pts.append((0.0, z))
    tags.append(prev[1])                          # the top
    tags.append(None)                             # back down the axis
    return Profile(tuple(pts), tuple(tags))


# -- the bevel ----------------------------------------------------------------

def _turning_deg(u: np.ndarray, v: np.ndarray) -> float:
    """The dihedral across a corner: the angle between its two faces' normals,
    which for a profile corner is the turning angle between its two edges."""
    cos_t = float(np.clip(np.dot(-u, v), -1.0, 1.0))
    return math.degrees(math.acos(cos_t))


MODELS = ("skew", "circle")


def bevel_profile(profile: Profile, width: float, segments: int,
                  min_dihedral: float = 30.0, max_dihedral: float = 180.0,
                  only_tags: Optional[Sequence[str]] = None,
                  new_tag: str = BEVEL, model: str = "skew") -> Profile:
    """Round every corner `bevel()` would round.

    OFFSET semantics, as the worker asks for: the arc starts and ends `width`
    from the corner *along each adjacent face*. Convex corners lose material and
    concave corners gain it; the construction is the same.

    Two models of the arc between those endpoints, because they are different
    and the difference is a finding:

      "skew"    what bmesh does with profile=0.5. A circle quadrant drawn in the
                frame whose axes are the two faces, so at a right angle it is a
                circle and at any other angle it is an *ellipse*, tangent to both
                faces but not a circle of any radius.
      "circle"  the fillet a designer means: the circle tangent to both faces at
                those endpoints, radius width * tan(theta / 2).

    They coincide at 90 degrees and nowhere else.

    A corner is skipped when it is on the axis, when either neighbour is an axis
    edge (that is a pole, not an edge), or when two bevels would overlap on one
    edge -- the caller picks widths where that cannot happen, because what
    bmesh's clamp does there has no closed form.
    """
    pts = [np.array(p, dtype=np.float64) for p in profile.pts]
    n = len(pts)
    wanted = None if only_tags is None else set(only_tags)

    picked: Dict[int, bool] = {}
    for i in range(n):
        e_in, e_out = profile.tags[i - 1], profile.tags[i]
        if e_in is None or e_out is None:
            continue
        if pts[i][0] < 1e-12:
            continue
        if wanted is not None and not (e_in in wanted and e_out in wanted):
            continue
        u = pts[i - 1] - pts[i]
        v = pts[(i + 1) % n] - pts[i]
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        if lu < 1e-12 or lv < 1e-12:
            continue
        dih = 180.0 - _turning_deg(u / lu, v / lv)
        if abs(dih) < 1e-9:
            continue                      # collinear: not a corner at all
        if min_dihedral <= dih <= max_dihedral:
            picked[i] = True

    # Overlap on one edge is the caller's mistake in this referee.
    for i in picked:
        j = (i + 1) % n
        if j in picked:
            length = float(np.linalg.norm(pts[j] - pts[i]))
            t_i = _tangent_length(pts, i, width)
            t_j = _tangent_length(pts, j, width)
            if t_i + t_j > length + 1e-12:
                raise ValueError(
                    "bevel width %g overlaps on the edge between corners %d and "
                    "%d (%g long); what clamping does there has no closed form"
                    % (width, i, j, length))

    out_pts: List[Tuple[float, float]] = []
    out_tags: List[Optional[str]] = []
    for i in range(n):
        if i not in picked:
            out_pts.append((float(pts[i][0]), float(pts[i][1])))
            out_tags.append(profile.tags[i])
            continue
        arc = (_arc_skew if model == "skew" else _arc_circle)(pts, i, width, segments)
        for k, a in enumerate(arc):
            out_pts.append((float(a[0]), float(a[1])))
            # Edges inside the arc are bevel surface; the last arc point starts
            # the edge that leaves the corner along the original outgoing face.
            out_tags.append(new_tag if k < len(arc) - 1 else profile.tags[i])
    return Profile(tuple(out_pts), tuple(out_tags))


def _tangent_length(pts, i, width) -> float:
    return width      # OFFSET: the arc starts `width` along each adjacent face


def _arc_skew(pts, i, width, segments) -> List[np.ndarray]:
    n = len(pts)
    p = pts[i]
    u = pts[i - 1] - p
    v = pts[(i + 1) % n] - p
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    return [p + width * ((1.0 - math.sin(math.pi / 2 * k / segments)) * u
                         + (1.0 - math.cos(math.pi / 2 * k / segments)) * v)
            for k in range(segments + 1)]


def _arc_circle(pts, i, width, segments) -> List[np.ndarray]:
    n = len(pts)
    p = pts[i]
    u = pts[i - 1] - p
    v = pts[(i + 1) % n] - p
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    theta = math.acos(float(np.clip(np.dot(u, v), -1.0, 1.0)))   # interior angle
    t1, t2 = p + width * u, p + width * v
    bis = u + v
    bis = bis / np.linalg.norm(bis)
    centre = p + bis * (width / math.cos(theta / 2.0))
    r1, r2 = t1 - centre, t2 - centre
    a1 = math.atan2(r1[1], r1[0])
    sweep = math.pi - theta
    # Walk the short way round, from t1 toward t2.
    cross = r1[0] * r2[1] - r1[1] * r2[0]
    sign = 1.0 if cross > 0 else -1.0
    rho = float(np.linalg.norm(r1))
    return [centre + rho * np.array([math.cos(a1 + sign * sweep * k / segments),
                                     math.sin(a1 + sign * sweep * k / segments)])
            for k in range(segments + 1)]


# -- quantities ---------------------------------------------------------------

def k_of(n: int) -> float:
    """Area of a regular n-gon is k * apothem^2."""
    return n * math.tan(math.pi / n)


def volume(profile: Profile, n: int) -> float:
    """Frusta, summed. Exact for a stack of regular n-gons."""
    s = 0.0
    for _i, p, q, _t in profile.edges():
        s += (p[0] * p[0] + p[0] * q[0] + q[0] * q[0]) * (q[1] - p[1])
    return abs(k_of(n) * s / 3.0)


def area_by_tag(profile: Profile, n: int) -> Dict[str, float]:
    """Surface area per tag. A facet strip between apothems x1 and x2 and slant
    length l is a trapezoid ring of area k * (x1 + x2) * l."""
    out: Dict[str, float] = {}
    k = k_of(n)
    for _i, p, q, tag in profile.edges():
        if tag is None:
            continue
        ell = math.hypot(q[0] - p[0], q[1] - p[1])
        out[tag] = out.get(tag, 0.0) + k * (p[0] + q[0]) * ell
    return out


# -- grading a mesh -----------------------------------------------------------

def classify(profile: Profile, n: int, verts: np.ndarray, tris: np.ndarray,
             tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Truth tag per triangle, from geometry alone.

    A triangle lies in one facet, so its normal fixes the azimuth of the
    (apothem, z) slice it lives in; its centroid projected onto that horizontal
    normal is its apothem coordinate. Then it is a point in the profile's plane,
    and the nearest profile edge is its source.

    Returns (tag index per triangle or -1, distance to that edge, area).
    """
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    nrm = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(nrm, axis=1)
    area = ln / 2.0
    unit = nrm / np.maximum(ln[:, None], 1e-300)
    cen = (p0 + p1 + p2) / 3.0
    horiz = np.hypot(unit[:, 0], unit[:, 1])
    hx = np.where(horiz > 1e-9, unit[:, 0] / np.maximum(horiz, 1e-300), 1.0)
    hy = np.where(horiz > 1e-9, unit[:, 1] / np.maximum(horiz, 1e-300), 0.0)
    # Horizontal facets have no azimuth of their own: any slice through the
    # centroid works, and the radial one is the natural choice. Their apothem
    # coordinate is then the radius of the centroid, which sits *inside* the
    # polygon, so they are matched on z alone below.
    rad = np.hypot(cen[:, 0], cen[:, 1])
    # abs(): an inner wall's outward normal points *toward* the axis, so the
    # projection comes out negative. The apothem is a distance.
    x = np.where(horiz > 1e-9, np.abs(cen[:, 0] * hx + cen[:, 1] * hy), rad)
    z = cen[:, 2]

    best = np.full(len(tris), -1, dtype=np.int64)
    bestd = np.full(len(tris), np.inf)
    for i, p, q, tag in profile.edges():
        if tag is None:
            continue
        pa, qa = np.array(p), np.array(q)
        d = qa - pa
        ll = float(np.dot(d, d))
        if ll < 1e-300:
            continue
        for_h = np.abs(d[1]) < 1e-12        # a horizontal edge: match z, and
        t = np.clip(((x - pa[0]) * d[0] + (z - pa[1]) * d[1]) / ll, 0.0, 1.0)
        dist = np.hypot(x - (pa[0] + t * d[0]), z - (pa[1] + t * d[1]))
        if for_h:
            # Only triangles that are themselves horizontal can be on it, and
            # their centroid must fall inside the annulus [x_lo, x_hi].
            lo, hi = sorted((pa[0], qa[0]))
            hi = hi / math.cos(math.pi / n)     # a corner of the outer n-gon
            inside = (horiz < 1e-9) & (rad >= lo - tol) & (rad <= hi + tol)
            dist = np.where(inside, np.abs(z - pa[1]), np.inf)
        else:
            dist = np.where(horiz > 1e-9, dist, np.inf)
        better = dist < bestd
        best = np.where(better, i, best)
        bestd = np.where(better, dist, bestd)
    best = np.where(bestd <= tol, best, -1)
    return best, bestd, area


@dataclass
class Grade:
    volume_truth: float
    volume_measured: float
    area_truth: Dict[str, float]
    area_measured: Dict[str, float]      # by the *claimed* tag
    unmatched: int                       # triangles on no expected surface
    unmatched_area: float
    mis_tagged: int                      # claim disagrees with geometry
    mis_tagged_area: float
    worst_distance: float

    def rel_volume_error(self) -> float:
        return (self.volume_measured - self.volume_truth) / self.volume_truth

    def rel_area_error(self, tag: str) -> float:
        t = self.area_truth.get(tag, 0.0)
        return (self.area_measured.get(tag, 0.0) - t) / t if t else float("nan")


def grade(profile: Profile, n: int, verts: np.ndarray, tris: np.ndarray,
          claimed: Sequence[str], measured_volume: float,
          tol: float = 1e-6) -> Grade:
    """Score a mesh and its provenance claim against `profile`.

    `claimed` is the tag the program under test gave each triangle. Truth for a
    triangle is the tag of the profile edge it lies on.
    """
    best, dist, area = classify(profile, n, verts, tris, tol)
    edge_tag = [t for _i, _p, _q, t in profile.edges()]
    mis, mis_a, un, un_a = 0, 0.0, 0, 0.0
    by_claim: Dict[str, float] = {}
    for ti in range(len(tris)):
        by_claim[claimed[ti]] = by_claim.get(claimed[ti], 0.0) + float(area[ti])
        if best[ti] < 0:
            un += 1
            un_a += float(area[ti])
            continue
        if edge_tag[best[ti]] != claimed[ti]:
            mis += 1
            mis_a += float(area[ti])
    finite = dist[np.isfinite(dist)]
    return Grade(
        volume_truth=volume(profile, n), volume_measured=float(measured_volume),
        area_truth=area_by_tag(profile, n), area_measured=by_claim,
        unmatched=un, unmatched_area=un_a, mis_tagged=mis, mis_tagged_area=mis_a,
        worst_distance=float(finite.max()) if finite.size else 0.0)
