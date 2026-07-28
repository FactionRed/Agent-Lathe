"""
Overlap measures, exact through the kernel and approximate without it.

Two probes need to compare a shape with another shape rather than measure one
shape: ``symmetry`` (how much of the solid maps onto itself under a mirror) and
the LOD ``silhouette deviation`` in Tier 2. Both reduce to an overlap ratio, and
both have an exact form and a kernel-free form:

  =================  ==============================  =========================
  measure            exact (manifold3d)              fallback (numpy only)
  =================  ==============================  =========================
  volume overlap     boolean intersect/union         surface-sample deviation
  outline overlap    project() + 2D boolean          rasterised projection
  =================  ==============================  =========================

The exact form is used whenever the input is a closed solid, because Manifold is
the committed kernel (DESIGN.md 11). The fallback exists for two reasons: an
open or non-manifold mesh has no kernel representation at all, and having two
independent implementations lets the test suite cross-check the numbers rather
than trust one of them. Every probe that has a choice reports which method it
used -- a number whose provenance is invisible is exactly the kind of thing an
agent over-trusts.
"""

from __future__ import annotations

import math

import numpy as np

from . import mesh as _mesh
from .mesh import Mesh

try:  # manifold3d is optional at import time, required for the exact paths
    import manifold3d as _m3

    HAVE_MANIFOLD = True
except ImportError:  # pragma: no cover - environment dependent
    _m3 = None
    HAVE_MANIFOLD = False


# --------------------------------------------------------------- exact paths


def mirror_about(man, axis: int, offset: float):
    """Mirror a solid across the plane {x_axis == offset}. Winding is preserved
    by Manifold, so the result is still a positive-volume solid."""
    normal = [0.0, 0.0, 0.0]
    normal[axis] = 1.0
    shift = [0.0, 0.0, 0.0]
    shift[axis] = -offset
    back = [0.0, 0.0, 0.0]
    back[axis] = offset
    return man.translate(shift).mirror(normal).translate(back)


def volume_overlap(man_a, man_b) -> dict:
    """Exact volume intersection/union of two solids (m^3) and their IoU."""
    inter = (man_a ^ man_b).volume()  # ^ is intersection in manifold3d
    union = (man_a + man_b).volume()
    return {
        "volume_a": man_a.volume(),
        "volume_b": man_b.volume(),
        "intersection": inter,
        "union": union,
        "iou": (inter / union) if union > 0.0 else 1.0,
    }


def min_gap_exact(man_a, man_b, search_length: float) -> float:
    """Exact minimum surface-to-surface gap, clamped at search_length."""
    return man_a.min_gap(man_b, search_length)


_PROJECT_ROTATION = {0: (0.0, 90.0, 0.0), 1: (90.0, 0.0, 0.0), 2: (0.0, 0.0, 0.0)}


def outline(man, axis: int):
    """The silhouette of a solid seen down `axis`, as a filled CrossSection.

    project() can return self-intersecting contours, so the result is refilled
    with the positive rule exactly as the manifold3d docs advise.
    """
    rot = _PROJECT_ROTATION[axis]
    raw = (man.rotate(rot) if any(rot) else man).project()
    return _m3.CrossSection(raw.to_polygons(), _m3.FillRule.Positive)


def outline_overlap_exact(man_a, man_b, axis: int) -> dict:
    """Exact 2D silhouette intersection/union area (m^2) and IoU."""
    a, b = outline(man_a, axis), outline(man_b, axis)
    inter = (a ^ b).area()
    union = (a + b).area()
    return {
        "area_a": a.area(),
        "area_b": b.area(),
        "intersection": inter,
        "union": union,
        "iou": (inter / union) if union > 0.0 else 1.0,
    }


# ------------------------------------------------------- kernel-free paths


def sample_surface(mesh: Mesh, per_tri: int = 4):
    """Deterministic points on the surface, with area weights.

    Fixed barycentric coordinates, so the sampling is a pure function of the
    mesh -- no RNG, no seed to remember (DESIGN.md 2, principle 1). `per_tri`
    of 1 is the centroid; 4 adds the three mid-edge-biased points, which is
    enough to notice a mirror mismatch that a centroid alone would miss.
    """
    bary = [(1 / 3, 1 / 3, 1 / 3)]
    if per_tri >= 4:
        bary += [(2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)]
    bary = bary[:per_tri] if per_tri < 4 else bary
    a, b, c = _mesh.tri_corners(mesh)
    areas = _mesh.tri_areas(mesh)
    pts, wts = [], []
    for u, v, w in bary:
        pts.append(a * u + b * v + c * w)
        wts.append(areas / len(bary))
    return np.concatenate(pts, axis=0), np.concatenate(wts, axis=0)


def surface_mirror_deviation(mesh: Mesh, axis: int, offset: float,
                             per_tri: int = 4, max_pairs: int = 1_000_000) -> dict:
    """Distance from mirrored surface samples back to the surface.

    Kernel-free companion to the volumetric symmetry score. Measures a length
    (how far the shape is from its own mirror), not a ratio, which is often the
    more actionable number: "worst deviation 12 mm at (x,y,z)" tells an agent
    where to look. Distances are point-to-*triangle*, not point-to-vertex, so
    the result does not move when the tessellation changes.
    """
    pts, wts = sample_surface(mesh, per_tri=per_tri)
    # distance-to-surface is O(points * tris); above the budget, keep every
    # k-th sample. Deterministic, so the probe stays a pure function, and the
    # stride is reported so the agent knows the number is a sample.
    n_total = len(pts)
    budget = max(1, max_pairs // max(mesh.n_tris, 1))
    stride = 1 if n_total <= budget else int(math.ceil(n_total / budget))
    if stride > 1:
        pts, wts = pts[::stride], wts[::stride]
    mirrored = pts.copy()
    mirrored[:, axis] = 2.0 * offset - mirrored[:, axis]
    d2 = _mesh.point_triangle_distance2(mirrored, mesh)
    d = np.sqrt(np.maximum(d2, 0.0))
    total_w = float(wts.sum())
    rms = float(np.sqrt((wts * d2).sum() / total_w)) if total_w > 0 else 0.0
    worst = int(np.argmax(d)) if len(d) else -1
    return {
        "samples": int(len(d)),
        "samples_available": int(n_total),
        "stride": stride,
        "rms": rms,
        "max": float(d[worst]) if worst >= 0 else 0.0,
        "max_at": mirrored[worst].copy() if worst >= 0 else np.zeros(3),
        "mean": float((wts * d).sum() / total_w) if total_w > 0 else 0.0,
    }


def rasterize_projection(mesh: Mesh, axis: int, res: int,
                         lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """(res,res) bool coverage of the silhouette down `axis` in a shared frame.

    Per triangle: map to grid space, test the cell centres inside its own
    bounding box. Cost is O(covered cells), not O(res^2 * tris).
    """
    keep = [i for i in range(3) if i != axis]
    a, b, c = _mesh.tri_corners(mesh)
    span = np.maximum(hi[keep] - lo[keep], 1e-12)
    scale = (res - 1) / span

    def to_grid(p):
        return (p[:, keep] - lo[keep]) * scale

    ga, gb, gc = to_grid(a), to_grid(b), to_grid(c)
    grid = np.zeros((res, res), dtype=bool)
    lo_i = np.floor(np.minimum(np.minimum(ga, gb), gc)).astype(int)
    hi_i = np.ceil(np.maximum(np.maximum(ga, gb), gc)).astype(int)
    lo_i = np.clip(lo_i, 0, res - 1)
    hi_i = np.clip(hi_i, 0, res - 1)
    for t in range(len(ga)):
        x0, y0 = lo_i[t]
        x1, y1 = hi_i[t]
        xs = np.arange(x0, x1 + 1)
        ys = np.arange(y0, y1 + 1)
        if not len(xs) or not len(ys):
            continue
        px, py = np.meshgrid(xs, ys, indexing="ij")
        p = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.float64)
        v0 = gb[t] - ga[t]
        v1 = gc[t] - ga[t]
        det = v0[0] * v1[1] - v0[1] * v1[0]
        if abs(det) < 1e-12:
            continue
        w = p - ga[t]
        u = (w[:, 0] * v1[1] - w[:, 1] * v1[0]) / det
        v = (v0[0] * w[:, 1] - v0[1] * w[:, 0]) / det
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
        if inside.any():
            grid[px.ravel()[inside], py.ravel()[inside]] = True
    return grid


def outline_overlap_raster(mesh_a: Mesh, mesh_b: Mesh, axis: int,
                           res: int = 512) -> dict:
    """Rasterised silhouette IoU. Accurate to roughly one cell of perimeter."""
    lo_a, hi_a = _mesh.bounds(mesh_a)
    lo_b, hi_b = _mesh.bounds(mesh_b)
    lo = np.minimum(lo_a, lo_b)
    hi = np.maximum(hi_a, hi_b)
    ga = rasterize_projection(mesh_a, axis, res, lo, hi)
    gb = rasterize_projection(mesh_b, axis, res, lo, hi)
    inter = int(np.count_nonzero(ga & gb))
    union = int(np.count_nonzero(ga | gb))
    keep = [i for i in range(3) if i != axis]
    cell = float(np.prod(np.maximum(hi[keep] - lo[keep], 0.0)) / max(res - 1, 1) ** 2)
    return {
        "area_a": np.count_nonzero(ga) * cell,
        "area_b": np.count_nonzero(gb) * cell,
        "intersection": inter * cell,
        "union": union * cell,
        "iou": (inter / union) if union else 1.0,
        "resolution": res,
    }


def vertex_to_surface_gap(mesh_a: Mesh, mesh_b: Mesh) -> float:
    """Smallest distance found between A's verts and B's surface, and back.

    Kernel-free lower bound on the true surface gap. It is exact whenever the
    closest pair involves a vertex (vertex-face, vertex-vertex, vertex-edge),
    which covers axis-aligned and most real placements, and it *overestimates*
    when the true closest pair is edge-to-edge on two skew edges. probe_distance
    prefers the kernel's exact min_gap and labels the method it used.
    """
    va = mesh_a.verts[np.unique(mesh_a.tris)] if mesh_a.n_tris else mesh_a.verts
    vb = mesh_b.verts[np.unique(mesh_b.tris)] if mesh_b.n_tris else mesh_b.verts
    d2 = min(
        float(_mesh.point_triangle_distance2(va, mesh_b).min()),
        float(_mesh.point_triangle_distance2(vb, mesh_a).min()),
    )
    return float(np.sqrt(max(d2, 0.0)))


__all__ = [
    "HAVE_MANIFOLD",
    "mirror_about",
    "volume_overlap",
    "min_gap_exact",
    "outline",
    "outline_overlap_exact",
    "sample_surface",
    "surface_mirror_deviation",
    "rasterize_projection",
    "outline_overlap_raster",
    "vertex_to_surface_gap",
]
