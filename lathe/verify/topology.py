"""
Derived connectivity: components, edge maps, manifoldness, degeneracy.

Split from mesh.py because the representation and the analysis of it are
different jobs, and because every topology rule (E101, E103, E105, E108) reads
exactly this object. It is computed once per asset and cached on the rule
context (see engine.RuleContext.topology).

CONNECTIVITY IS A PROPERTY OF POSITIONS, NOT OF VERTEX INDICES. Every question
here -- how many shells, which edges are open, where two fans pinch -- is asked
of the surface, and the surface does not know how an exporter numbered its
corners. So faces are rewritten onto one canonical vertex per position before
anything is counted (`canonical_vertices`), at a tolerance the asset declares
(`EmitSpec.weld_tolerance`). Read by index instead, a flat-shaded glTF export
is a pile of open pieces with pinched vertices, and a real pinch between two
shells that never shared an index is invisible. Both happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .geom import EdgeKey, edge_key
from .mesh import AssetMesh, Face

# Metres. Bit-identical duplicates, which is what a vertex split produces,
# always land in the same cell; two points that round to the same cell weld
# as well. Well under any feature size a game asset carries.
WELD_TOLERANCE = 1e-6


def canonical_vertices(verts: np.ndarray, tolerance: Optional[float]) -> np.ndarray:
    """Map every vertex index to the first vertex at the same position.

    Positions are binned on a grid of `tolerance`. None or 0 is the identity
    map: index connectivity, exactly as the mesh arrived.
    """
    n = int(verts.shape[0])
    canon = np.arange(n, dtype=np.int64)
    if not tolerance or n == 0:
        return canon
    cells = np.round(np.asarray(verts, dtype=np.float64) / float(tolerance)).astype(np.int64)
    first: Dict[Tuple[int, int, int], int] = {}
    for i, key in enumerate(map(tuple, cells.tolist())):
        canon[i] = first.setdefault(key, i)
    return canon


@dataclass(frozen=True)
class Component:
    """One position-connected shell. E108 counts these."""

    index: int
    faces: Tuple[int, ...]
    verts: Tuple[int, ...]
    tri_count: int
    closed: bool  # every edge has exactly two faces
    consistent: bool  # no edge traversed the same way by both its faces
    signed_volume: float  # positive == outward-facing under CCW convention
    bbox_min: Tuple[float, float, float]
    bbox_max: Tuple[float, float, float]

    @property
    def name(self) -> str:
        return "component_%02d" % self.index

    @property
    def extents(self) -> Tuple[float, float, float]:
        return tuple(b - a for a, b in zip(self.bbox_min, self.bbox_max))


class Topology:
    """Derived connectivity. Computed once per asset, shared by every rule.

    Cheap and eager. At production density this wants to be incremental, which
    is a real limitation and is recorded in the README rather than hidden.
    """

    def __init__(self, mesh: AssetMesh, weld: Optional[float] = WELD_TOLERANCE) -> None:
        self.mesh = mesh
        self.weld = weld
        # Faces on canonical vertices: one index per position. Every index
        # this class hands out (bowtie verts, edge keys, component verts) is a
        # canonical one, and always a valid index into `mesh.verts`.
        self.canonical: np.ndarray = canonical_vertices(mesh.verts, weld)
        self.duplicate_verts: int = int(
            np.count_nonzero(self.canonical != np.arange(mesh.num_verts, dtype=np.int64))
        )
        canon = self.canonical
        self.faces: Tuple[Face, ...] = tuple(
            tuple(int(canon[i]) for i in f) for f in mesh.faces
        )

        self.edge_faces: Dict[EdgeKey, List[int]] = {}
        self.directed: Dict[EdgeKey, List[int]] = {}
        self.vert_faces: List[List[int]] = [[] for _ in range(mesh.num_verts)]
        # A face using one vertex twice in a row. After the weld this also
        # catches a face whose two corners sit on one point, which is a
        # zero-length edge and belongs to E105 as much as an index repeat does.
        self.self_edge_faces: List[int] = []

        for fi, f in enumerate(self.faces):
            n = len(f)
            for j in range(n):
                a, b = f[j], f[(j + 1) % n]
                if a == b:
                    self.self_edge_faces.append(fi)
                    continue
                self.directed.setdefault((a, b), []).append(fi)
                self.edge_faces.setdefault(edge_key(a, b), []).append(fi)
            for v in set(f):
                self.vert_faces[v].append(fi)

        self.boundary_edges: Tuple[EdgeKey, ...] = tuple(
            sorted(e for e, fs in self.edge_faces.items() if len(fs) == 1)
        )
        self.nonmanifold_edges: Tuple[EdgeKey, ...] = tuple(
            sorted(e for e, fs in self.edge_faces.items() if len(fs) > 2)
        )
        self.inconsistent_edges: Tuple[EdgeKey, ...] = tuple(
            sorted(e for e in self.edge_faces if self._is_inconsistent(e))
        )
        self.bowtie_verts: Tuple[int, ...] = tuple(self._find_bowties())
        self.components: Tuple[Component, ...] = tuple(self._components())

        # Loose verts are judged on the ORIGINAL indices. A duplicate that a
        # face references is used, not loose, even though no canonical face
        # names it any more.
        used = {i for f in mesh.faces for i in f}
        wired = {i for e in mesh.wire_edges for i in e}
        self.loose_verts: Tuple[int, ...] = tuple(
            i for i in range(mesh.num_verts) if i not in used and i not in wired
        )
        self.loose_wire_edges: Tuple[EdgeKey, ...] = tuple(
            e for e in mesh.wire_edges
            if edge_key(int(canon[e[0]]), int(canon[e[1]])) not in self.edge_faces
        )

    # -- predicates --------------------------------------------------------

    def _is_inconsistent(self, edge: EdgeKey) -> bool:
        """Only meaningful where exactly two faces meet: two faces sharing an
        edge must traverse it in opposite directions."""
        faces = self.edge_faces[edge]
        if len(faces) != 2:
            return False
        a, b = edge
        fwd = len(self.directed.get((a, b), ()))
        rev = len(self.directed.get((b, a), ()))
        return not (fwd == 1 and rev == 1)

    def fan_count(self, v: int) -> int:
        """How many separate face fans meet at a vertex. One is normal; more
        than one is a bowtie -- two shells pinched at a point."""
        faces = self.vert_faces[v]
        if len(faces) < 2:
            return len(faces)
        # Union-find over the fan: two faces are joined when they share an edge
        # that contains v.
        parent = {f: f for f in faces}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for _edge, efaces in self.edges_at(v):
            for other in efaces[1:]:
                ra, rb = find(efaces[0]), find(other)
                if ra != rb:
                    parent[ra] = rb
        return len({find(f) for f in faces})

    def _find_bowties(self) -> List[int]:
        """A vertex is non-manifold when its incident faces form more than one
        fan. Independent of edge manifoldness: a bowtie can sit in a mesh whose
        every edge has exactly two faces."""
        return [
            v
            for v in range(self.mesh.num_verts)
            if len(self.vert_faces[v]) >= 2 and self.fan_count(v) > 1
        ]

    def edges_at(self, v: int) -> Iterable[Tuple[EdgeKey, List[int]]]:
        seen = set()
        for fi in self.vert_faces[v]:
            f = self.faces[fi]
            n = len(f)
            for j, a in enumerate(f):
                b = f[(j + 1) % n]
                if a == b:
                    continue
                if v in (a, b):
                    k = edge_key(a, b)
                    if k not in seen:
                        seen.add(k)
                        yield k, self.edge_faces[k]

    def _components(self) -> List[Component]:
        mesh = self.mesh
        parent = list(range(mesh.num_verts))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        canon = self.canonical
        for f in self.faces:
            for j in range(len(f)):
                union(f[j], f[(j + 1) % len(f)])
        for a, b in mesh.wire_edges:
            union(int(canon[a]), int(canon[b]))

        groups: Dict[int, List[int]] = {}
        for fi, f in enumerate(self.faces):
            groups.setdefault(find(f[0]), []).append(fi)

        wv = mesh.world_verts()
        out: List[Component] = []
        # deterministic order: biggest first, then by first face index
        keys = sorted(groups, key=lambda r: (-len(groups[r]), min(groups[r])))
        for idx, root in enumerate(keys):
            faces = tuple(groups[root])
            vset = sorted({i for fi in faces for i in self.faces[fi]})
            edges = {
                edge_key(self.faces[fi][j], self.faces[fi][(j + 1) % len(self.faces[fi])])
                for fi in faces
                for j in range(len(self.faces[fi]))
                if self.faces[fi][j] != self.faces[fi][(j + 1) % len(self.faces[fi])]
            }
            closed = all(len(self.edge_faces[e]) == 2 for e in edges)
            consistent = not any(self._is_inconsistent(e) for e in edges)
            vol = 0.0
            for fi in faces:
                poly = wv[list(self.faces[fi])]
                for t in range(1, len(poly) - 1):
                    a, b, c = poly[0], poly[t], poly[t + 1]
                    vol += float(np.dot(a, np.cross(b, c))) / 6.0
            pts = wv[vset]
            out.append(
                Component(
                    index=idx,
                    faces=faces,
                    verts=tuple(vset),
                    tri_count=sum(max(len(mesh.faces[fi]) - 2, 0) for fi in faces),
                    closed=closed,
                    consistent=consistent,
                    signed_volume=vol,
                    bbox_min=tuple(pts.min(axis=0)),
                    bbox_max=tuple(pts.max(axis=0)),
                )
            )
        return out

    # -- readouts ----------------------------------------------------------

    def edge_midpoint(self, edge: EdgeKey, world: bool = True) -> np.ndarray:
        v = self.mesh.world_verts() if world else self.mesh.verts
        return (v[edge[0]] + v[edge[1]]) * 0.5

    def degenerate_faces(self, min_area: float) -> Tuple[Tuple[int, float, str], ...]:
        """(face, area, reason). The reason matters: a face can be degenerate by
        area or by repeating a vertex, and reporting the second as the first
        makes the evidence line untrue."""
        areas = self.mesh.face_areas(world=True)
        out: Dict[int, Tuple[float, List[str]]] = {}
        for i, a in enumerate(areas):
            reasons: List[str] = []
            if a <= min_area:
                reasons.append("zero area")
            if len(self.mesh.faces[i]) < 3:
                reasons.append("fewer than 3 verts")
            if reasons:
                out[i] = (float(a), reasons)
        for fi in self.self_edge_faces:
            area, reasons = out.get(fi, (float(areas[fi]), []))
            if "repeated vertex" not in reasons:
                reasons = reasons + ["repeated vertex"]
            out[fi] = (area, reasons)
        return tuple(
            (fi, area, ", ".join(reasons)) for fi, (area, reasons) in sorted(out.items())
        )


# --------------------------------------------------------------- UV layout
