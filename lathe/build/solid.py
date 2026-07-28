"""
The authoring kernel -- DESIGN.md sections 2, 4 and 5 made executable.

Every operation here is a pure function of its explicit arguments. No mode, no
selection state, no active object. A `Solid` is an immutable value: operations
return new ones.

Provenance (section 5.2) rides on Manifold's native channels. Each primitive
becomes an *original* with its own id, and a `Solid` carries the map from those
ids to tag names. Booleans merge the maps, so the result knows which of its
triangles came from which source solid. The logical face key is the pair
`(part_id, face_id)` -- `face_id` alone collides across solids, which section 5.2
records the hard way.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

import manifold3d as m3

from ..perceive import mesh as pmesh


def mm(x: float) -> float:
    """Millimetres to metres. The build layer works in metres throughout."""
    return x / 1000.0


class LatheError(Exception):
    pass


class CardinalityError(LatheError):
    """Section 5.3. A selector matched a different set than the program assumed."""


# --------------------------------------------------------------- selections

@dataclass(frozen=True)
class Selection:
    """Resolved query result. `domain` is "faces" or "edges".

    For faces, `indices` are triangle indices and `groups` are the distinct
    (part_id, face_id) logical faces they cover. For edges, `indices` index the
    mesh topology's unique undirected edge list.
    """

    solid: "Solid"
    domain: str
    indices: np.ndarray
    expr: str
    groups: Tuple[Tuple[int, int], ...] = ()

    def count(self) -> int:
        """Logical faces for a face selection, edges for an edge selection.

        Triangles are a tessellation detail. A program that asserts on face
        counts means faces, so counting triangles here would make section 5.3
        assertions fire on a remesh that changed nothing meaningful.
        """
        return len(self.groups) if self.domain == "faces" else len(self.indices)

    @property
    def n_tris(self) -> int:
        return len(self.indices) if self.domain == "faces" else 0

    def __repr__(self) -> str:
        return "<Selection %s %s n=%d>" % (self.domain, self.expr, self.count())


def assert_count(sel: Selection, expected: int, label: Optional[str] = None) -> Selection:
    """Section 5.3. Cheap to write, and it converts silent corruption into a
    loud, localized error naming the assumption that broke."""
    got = sel.count()
    if got != expected:
        raise CardinalityError(
            "assert count(%s) == %d\n"
            "  actual   : %d\n"
            "  selector : %s\n"
            "  on       : %s (%d tris, %d parts)\n"
            "  meaning  : the selector matched a different set than the program\n"
            "             assumed. An upstream parameter probably changed."
            % (label or sel.expr, expected, got, sel.expr, sel.solid.name,
               sel.solid.n_tris, len(sel.solid.tags))
        )
    return sel


# -------------------------------------------------------------------- solid

@dataclass(frozen=True)
class Solid:
    man: object                      # manifold3d.Manifold
    tags: Mapping[int, str] = field(default_factory=dict)
    name: str = "solid"
    _mesh: Optional[pmesh.Mesh] = field(default=None, compare=False, repr=False)

    # -- lifting -----------------------------------------------------------

    def mesh(self) -> pmesh.Mesh:
        """The neutral mesh, memoised. Carries part_id and face_id across."""
        if self._mesh is None:
            msh = pmesh.from_manifold(self.man, name=self.name, parts=dict(self.tags))
            object.__setattr__(self, "_mesh", msh)
        return self._mesh

    @property
    def n_tris(self) -> int:
        return self.man.num_tri()

    def volume(self) -> float:
        return self.man.volume()

    def bbox(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        lo, hi = self.man.bounding_box()[:3], self.man.bounding_box()[3:]
        return tuple(lo), tuple(hi)

    def components(self) -> int:
        return len(self.man.decompose())

    def status_ok(self) -> bool:
        return self.man.status() == m3.Error.NoError and not self.man.is_empty()

    def rename(self, name: str) -> "Solid":
        return Solid(self.man, dict(self.tags), name)

    def retag(self, **renames: str) -> "Solid":
        """Rename tags: `s.retag(body="seat")`."""
        flipped = {v: k for k, v in renames.items()}
        tags = {i: flipped.get(t, t) for i, t in self.tags.items()}
        return Solid(self.man, tags, self.name)

    # -- transforms --------------------------------------------------------

    def translate(self, v) -> "Solid":
        return Solid(self.man.translate(tuple(float(x) for x in v)), dict(self.tags), self.name)

    def rotate(self, degrees) -> "Solid":
        return Solid(self.man.rotate(tuple(float(x) for x in degrees)), dict(self.tags), self.name)

    def scale(self, v) -> "Solid":
        if np.isscalar(v):
            v = (v, v, v)
        return Solid(self.man.scale(tuple(float(x) for x in v)), dict(self.tags), self.name)

    def mirror(self, normal) -> "Solid":
        return Solid(self.man.mirror(tuple(float(x) for x in normal)), dict(self.tags), self.name)

    def at(self, x=0.0, y=0.0, z=0.0) -> "Solid":
        return self.translate((x, y, z))

    # -- booleans ----------------------------------------------------------

    def _combine(self, other: "Solid", op: str) -> "Solid":
        if op == "-":
            man = self.man - other.man
        elif op == "+":
            man = self.man + other.man
        else:
            man = self.man ^ other.man
        merged = dict(self.tags)
        merged.update(other.tags)
        out = Solid(man, merged, self.name)
        if not out.status_ok():
            raise LatheError(
                "boolean %r on %s and %s produced %s (empty=%s).\n"
                "  Check for coincident faces and whether the cutter fully "
                "penetrates." % (op, self.name, other.name, man.status(), man.is_empty())
            )
        return out

    def __sub__(self, other) -> "Solid":
        return self._combine(_as_solid(other), "-")

    def __add__(self, other) -> "Solid":
        return self._combine(_as_solid(other), "+")

    def __and__(self, other) -> "Solid":
        return self._combine(_as_solid(other), "^")

    # -- selectors (section 5.1) -------------------------------------------

    def faces(self, tag=None, normal=None, normal_tol=0.15, area_gt=None,
              area_lt=None, z_gt=None, z_lt=None, all=False) -> Selection:
        msh = self.mesh()
        terms = []
        keep = np.ones(msh.n_tris, dtype=bool)
        if all:
            terms.append("all")
        if tag is not None:
            terms.append("#%s" % tag)
            ids = [i for i, t in self.tags.items() if t == tag]
            if not ids:
                raise LatheError(
                    "no such tag %r on %s. Tags present: %s"
                    % (tag, self.name, sorted(set(self.tags.values())))
                )
            keep &= np.isin(msh.part_id, ids)
        nrm = pmesh.tri_normals(msh)
        if normal is not None:
            want = np.asarray(normal, dtype=np.float64)
            want = want / np.linalg.norm(want)
            terms.append("normal ~ (%.3g, %.3g, %.3g)" % tuple(want))
            keep &= np.linalg.norm(nrm - want, axis=1) <= normal_tol
        if area_gt is not None or area_lt is not None:
            areas = pmesh.tri_areas(msh)
            if area_gt is not None:
                terms.append("area > %g" % area_gt)
                keep &= areas > area_gt
            if area_lt is not None:
                terms.append("area < %g" % area_lt)
                keep &= areas < area_lt
        if z_gt is not None or z_lt is not None:
            cz = pmesh.tri_centroids(msh)[:, 2]
            if z_gt is not None:
                terms.append("z > %g" % z_gt)
                keep &= cz > z_gt
            if z_lt is not None:
                terms.append("z < %g" % z_lt)
                keep &= cz < z_lt
        idx = np.flatnonzero(keep)
        groups = _logical_faces(msh, idx)
        return Selection(self, "faces", idx, "faces(%s)" % ", ".join(terms), groups)

    def edges(self, dihedral_gt=None, dihedral_lt=None, length_gt=None,
              tag=None, all=False) -> Selection:
        msh = self.mesh()
        topo = msh.topology
        terms = []
        e_tris = _edge_tris(topo)
        # Only manifold interior edges have a dihedral angle at all.
        keep = topo.edge_tri_count == 2
        if all:
            terms.append("all")
        if dihedral_gt is not None or dihedral_lt is not None:
            ang = _dihedral(msh, topo, e_tris)
            if dihedral_gt is not None:
                terms.append("dihedral > %gdeg" % dihedral_gt)
                keep &= ang > dihedral_gt
            if dihedral_lt is not None:
                terms.append("dihedral < %gdeg" % dihedral_lt)
                keep &= ang < dihedral_lt
        if length_gt is not None:
            terms.append("length > %g" % length_gt)
            v = msh.verts
            elen = np.linalg.norm(v[topo.edges[:, 0]] - v[topo.edges[:, 1]], axis=1)
            keep &= elen > length_gt
        if tag is not None:
            terms.append("#%s on both sides" % tag)
            ids = [i for i, t in self.tags.items() if t == tag]
            tagged = np.isin(msh.part_id, ids)
            both = tagged[e_tris[:, 0]] & tagged[e_tris[:, 1]]
            keep &= both
        idx = np.flatnonzero(keep)
        return Selection(self, "edges", idx, "edges(%s)" % ", ".join(terms))


def _as_solid(x) -> Solid:
    if isinstance(x, Solid):
        return x
    raise TypeError("expected a Solid, got %r" % type(x).__name__)


def _logical_faces(msh: pmesh.Mesh, tri_idx) -> Tuple[Tuple[int, int], ...]:
    """Distinct (part_id, face_id) pairs -- the composite key from section 5.2."""
    if len(tri_idx) == 0:
        return ()
    if msh.face_id is None or msh.part_id is None:
        return tuple((0, int(i)) for i in tri_idx)
    pairs = set(zip(msh.part_id[tri_idx].tolist(), msh.face_id[tri_idx].tolist()))
    return tuple(sorted((int(a), int(b)) for a, b in pairs))


def _edge_tris(topo) -> np.ndarray:
    """(E,2) the two triangles on each edge, -1 where there is only one."""
    E = topo.n_edges
    out = np.full((E, 2), -1, dtype=np.int64)
    slot = np.zeros(E, dtype=np.int64)
    tri_of = np.repeat(np.arange(len(topo.tri_edges)), 3)
    for e, t in zip(topo.tri_edges.reshape(-1), tri_of):
        if slot[e] < 2:
            out[e, slot[e]] = t
            slot[e] += 1
    return out


def _dihedral(msh: pmesh.Mesh, topo, e_tris) -> np.ndarray:
    """Angle between face normals in degrees. 0 is flat, 90 is a square corner."""
    nrm = pmesh.tri_normals(msh)
    a = e_tris[:, 0].copy()
    b = e_tris[:, 1].copy()
    ok = (a >= 0) & (b >= 0)
    a[~ok] = 0
    b[~ok] = 0
    dot = np.clip(np.einsum("ij,ij->i", nrm[a], nrm[b]), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    ang[~ok] = 0.0
    return ang


# --------------------------------------------------------------- primitives

def _original(man, tag: str, name: str) -> Solid:
    """Mark a fresh primitive as an original so its triangles stay attributable.

    Safe here and only here: `as_original()` drops ancestor relations, so calling
    it on a boolean result would destroy provenance (DESIGN.md section 14).
    """
    man = man.as_original()
    return Solid(man, {int(man.original_id()): tag}, name)


def box(sx, sy, sz, center=True, tag="body", name="box") -> Solid:
    return _original(m3.Manifold.cube((sx, sy, sz), center), tag, name)


def cylinder(height, radius, radius_top=None, segments=32, center=True,
             tag="body", name="cylinder") -> Solid:
    rt = radius if radius_top is None else radius_top
    return _original(
        m3.Manifold.cylinder(height, radius, rt, segments, center), tag, name)


def sphere(radius, segments=32, tag="body", name="sphere") -> Solid:
    return _original(m3.Manifold.sphere(radius, segments), tag, name)


def revolve(profile: Sequence[Tuple[float, float]], segments=48, degrees=360.0,
            tag="body", name="revolve") -> Solid:
    """Revolve a (radius, height) polygon about the Z axis.

    Manifold revolves the cross-section about its own Y axis and calls that Z,
    so profile x is radius and profile y becomes world z.
    """
    pts = [(float(x), float(y)) for x, y in profile]
    cs = m3.CrossSection([pts])
    return _original(m3.Manifold.revolve(cs, segments, degrees), tag, name)


def extrude(profile: Sequence[Tuple[float, float]], height, tag="body",
            name="extrude") -> Solid:
    pts = [(float(x), float(y)) for x, y in profile]
    cs = m3.CrossSection([pts])
    return _original(m3.Manifold.extrude(cs, float(height)), tag, name)


# ------------------------------------------------------------------ helpers

def union_all(solids: Iterable[Solid], name=None) -> Solid:
    """Fold a union. Keeps every source solid's tag, unlike Manifold.compose,
    which would leave the result a single unattributable blob."""
    items = list(solids)
    if not items:
        raise LatheError("union_all got nothing to union")
    out = items[0]
    for s in items[1:]:
        out = out + s
    return out.rename(name or out.name)


def array(factory, count, axis="x", span=0.0, center=True, tag_each=False):
    """`count` copies of `factory(i)` spread along an axis.

    factory takes the index so each instance can differ. With tag_each, every
    instance keeps its own tag, so a selector can address one of them.
    """
    ax = {"x": 0, "y": 1, "z": 2}[axis]
    out = []
    for i in range(count):
        t = 0.5 if count == 1 else i / (count - 1.0)
        off = [0.0, 0.0, 0.0]
        off[ax] = (-span / 2 + t * span) if center else t * span
        s = factory(i)
        if tag_each:
            s = s.retag(**{"%s_%02d" % (list(s.tags.values())[0], i):
                           list(s.tags.values())[0]})
        out.append(s.translate(off))
    return out


def bowl_profile(radius, height, thickness, foot_radius=None, foot_height=None):
    """A closed (radius, height) polygon for `revolve`: a bowl with a flat foot.

    Returned counter-clockwise, which is what CrossSection wants for a solid
    region; a clockwise ring would come back as a hole.
    """
    fr = radius * 0.45 if foot_radius is None else foot_radius
    fh = thickness if foot_height is None else foot_height
    inner_r = radius - thickness
    return [
        (0.0, 0.0),
        (fr, 0.0),
        (fr, fh * 0.6),
        (radius * 0.72, fh * 0.9),
        (radius, height),
        (inner_r, height),
        (inner_r * 0.62, fh + thickness * 0.7),
        (0.0, fh + thickness),
    ]
