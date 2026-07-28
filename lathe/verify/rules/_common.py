"""Formatting helpers shared by the rule modules.

Evidence is numeric, so every number in a report goes through here. One place
to decide how many digits a measurement deserves.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from ..findings import _num as num  # noqa: F401  (re-export: the number formatter)
from ..mesh import AssetMesh


def loc(p: Sequence[float]) -> str:
    """A coordinate locator. Coordinates, not indices: an index is meaningless
    to the next evaluation of the program (DESIGN.md section 5.1), a position is
    something the agent can look at and select on."""
    return "(%.3f, %.3f, %.3f)" % tuple(_unsign(v, 3) for v in p[:3])


def loc2(p: Sequence[float]) -> str:
    return "(%.4f, %.4f)" % tuple(_unsign(v, 4) for v in p[:2])


def _unsign(v: float, digits: int) -> float:
    """Kill negative zero. `(-0.000, 0.000, 0.000)` reads as a bug in the
    verifier, and a report the agent distrusts is a report it ignores."""
    return round(float(v), digits) + 0.0


def plural(n: int, singular: str, suffix: str = "s") -> str:
    """`1 edge`, `3 edges`. Evidence lines are read closely; broken agreement
    makes a numeric claim look sloppy."""
    return "%d %s%s" % (n, singular, "" if n == 1 else suffix)


def mm(v: float) -> str:
    return "%s mm" % num(float(v) * 1000.0)


def pct(v: float) -> str:
    return "%s%%" % num(v)


def deg(v: float) -> str:
    return "%s°" % num(np.degrees(float(v)))


def face_label(mesh: AssetMesh, fi: int) -> str:
    """`side_L` when the mesh gave the face a group name, else a coordinate."""
    g = mesh.group_of_face(fi)
    return g if g else "face at %s" % loc(mesh.face_centroid(fi, world=True))


def face_locator(mesh: AssetMesh, fi: int) -> str:
    """`side_L at (0.300, 0.000, 0.300)`, or just the position when the face
    belongs to no named group."""
    g = mesh.group_of_face(fi)
    p = loc(mesh.face_centroid(fi, world=True))
    return "%s at %s" % (g, p) if g else "at %s" % p


def group_summary(mesh: AssetMesh, faces: Iterable[int], limit: int = 4) -> str:
    """Named face groups covering a set of faces, for the entity field."""
    seen = []
    for fi in faces:
        g = mesh.group_of_face(fi)
        key = g if g else "<ungrouped>"
        if key not in seen:
            seen.append(key)
    if not seen:
        return "<none>"
    shown = seen[:limit]
    tail = "" if len(seen) <= limit else " +%d" % (len(seen) - limit)
    return ", ".join(shown) + tail


def entity_for(mesh: AssetMesh, faces: Optional[Sequence[int]] = None) -> str:
    """`SM_Crate_Wood_LOD0` or `SM_Crate_Wood_LOD0.side_L` when the offending
    faces all live in one named group."""
    if not faces:
        return mesh.name
    groups = {mesh.group_of_face(fi) for fi in faces}
    if len(groups) == 1:
        only = next(iter(groups))
        if only:
            return "%s.%s" % (mesh.name, only)
    return mesh.name


def histogram(pairs) -> str:
    return ", ".join("%s: %s" % (k, v) for k, v in pairs)
