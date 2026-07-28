"""
Geometry primitives for the verifier.

Pure functions over coordinate arrays: no asset, no rule, no profile. They live
apart from mesh.py because DESIGN.md section 6 (perception) needs the same
areas, normals and clipping for its Tier 1 probes, and because a function with
no dependencies is a function you can test by hand.

Written in scalars rather than vectorised numpy in the small cases. These are
called once per face per rule, where numpy's per-call overhead dominates: the
scalar Newell area measured 3x faster than `np.cross` per edge on a 2,000-face
asset.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

EdgeKey = Tuple[int, int]


def edge_key(a: int, b: int) -> EdgeKey:
    return (a, b) if a < b else (b, a)


def seg_key(p, q):
    a = (round(float(p[0]), 9), round(float(p[1]), 9))
    b = (round(float(q[0]), 9), round(float(q[1]), 9))
    return (a, b) if a <= b else (b, a)


def newell_normal(poly: np.ndarray) -> np.ndarray:
    """Newell's method: correct for non-planar and concave polygons alike."""
    n = np.zeros(3)
    m = poly.shape[0]
    for i in range(m):
        a, b = poly[i], poly[(i + 1) % m]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    ln = float(np.linalg.norm(n))
    return n / ln if ln > 0 else n


def polygon_area_3d(poly) -> float:
    """Newell area. Written out in scalars rather than `np.cross` per edge:
    these run once per face per rule, and numpy's per-call overhead dominates
    at triangle scale. Measured 3x faster on a 2,000-face asset."""
    pts = [(float(p[0]), float(p[1]), float(p[2])) for p in poly]
    m = len(pts)
    if m < 3:
        return 0.0
    tx = ty = tz = 0.0
    for i in range(m):
        ax, ay, az = pts[i]
        bx, by, bz = pts[(i + 1) % m]
        tx += ay * bz - az * by
        ty += az * bx - ax * bz
        tz += ax * by - ay * bx
    return math.sqrt(tx * tx + ty * ty + tz * tz) * 0.5


def polygon_area_2d(poly) -> float:
    """Signed shoelace area. The sign carries UV winding, which E106 needs."""
    pts = [(float(p[0]), float(p[1])) for p in poly]
    m = len(pts)
    if m < 3:
        return 0.0
    s = 0.0
    for i in range(m):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % m]
        s += x1 * y2 - x2 * y1
    return s * 0.5


def clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of one convex polygon by another. Both inputs
    are triangles in E106, so the result stays convex and the area is exact."""
    poly = [np.asarray(p, dtype=np.float64) for p in subject]
    cl = [np.asarray(p, dtype=np.float64) for p in clip]
    if polygon_area_2d(np.array(cl)) < 0:
        cl = cl[::-1]
    if polygon_area_2d(np.array(poly)) < 0:
        poly = poly[::-1]
    for i in range(len(cl)):
        a, b = cl[i], cl[(i + 1) % len(cl)]
        edge = b - a
        out = []
        m = len(poly)
        if m == 0:
            return np.zeros((0, 2))
        for j in range(m):
            p, q = poly[j], poly[(j + 1) % m]
            sp = edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])
            sq = edge[0] * (q[1] - a[1]) - edge[1] * (q[0] - a[0])
            if sp >= 0:
                out.append(p)
            if (sp > 0 > sq) or (sp < 0 < sq):
                t = sp / (sp - sq)
                out.append(p + (q - p) * t)
        poly = out
    return np.array(poly) if poly else np.zeros((0, 2))


def overlap_area(tri_a: np.ndarray, tri_b: np.ndarray) -> float:
    poly = clip_convex(tri_a, tri_b)
    if poly.shape[0] < 3:
        return 0.0
    return abs(polygon_area_2d(poly))


def seg_seg_distance(p1, p2, q1, q2) -> float:
    """Minimum distance between two 2D segments. Used by W202."""
    p1, p2, q1, q2 = (np.asarray(v, dtype=np.float64) for v in (p1, p2, q1, q2))
    if segments_intersect(p1, p2, q1, q2):
        return 0.0
    return min(
        point_seg_distance(p1, q1, q2),
        point_seg_distance(p2, q1, q2),
        point_seg_distance(q1, p1, p2),
        point_seg_distance(q2, p1, p2),
    )


def point_seg_distance(p, a, b) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab)) / denom))
    return float(np.linalg.norm(p - (a + ab * t)))


def segments_intersect(p1, p2, q1, q2) -> bool:
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1, d2 = cross(q1, q2, p1), cross(q1, q2, p2)
    d3, d4 = cross(p1, p2, q1), cross(p1, p2, q2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))
