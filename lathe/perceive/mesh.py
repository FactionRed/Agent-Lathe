"""
The neutral mesh the perception layer sees.

DESIGN.md 11 says the core must be headless and Blender-optional, and 11 picks
Manifold as the boolean kernel. So probes operate on a small neutral value type
-- vertices, triangles, and optional provenance/material/UV channels -- and
Manifold is an *adapter*, not a dependency:

  * ``from_manifold(m)`` lifts a Manifold solid, carrying ``run_original_id``
    across as part ids and ``face_id`` across as logical face groups (DESIGN.md
    5.2 provenance);
  * a mesh built from raw arrays needs no kernel at all, which matters because
    Manifold *cannot represent* a non-watertight mesh (``Manifold(open_mesh)``
    returns empty with ``Error.NotManifold``). The probes must still be able to
    look at broken geometry and say what is wrong with it.

Purity (DESIGN.md 2, principle 1): the arrays are set read-only at construction,
so a probe cannot mutate its input even by accident. Derived topology is
memoised on the value; because the value is immutable, a warm memo and a cold
memo produce byte-identical probe output (asserted in the test suite). Lengths
are stored in metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

EMPTY_MAP: Mapping[int, str] = MappingProxyType({})


def _freeze(arr) -> np.ndarray:
    arr = np.array(arr, copy=True)
    arr.setflags(write=False)
    return arr


def _freeze_as(arr, dtype) -> np.ndarray:
    arr = np.array(arr, dtype=dtype, copy=True, order="C")
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False, repr=False)
class Mesh:
    """A triangle mesh plus the channels perception needs. Immutable.

    verts        (V,3) float64, metres
    tris         (T,3) int64 vertex indices, CCW seen from outside
    face_id      (T,)  int64 logical coplanar face group (Manifold face_id)
    part_id      (T,)  int64 part/original id (Manifold run_original_id)
    parts        {part_id: name}
    material_id  (T,)  int64
    materials    {material_id: name}
    uv           (T,3,2) float64 per-corner UV, so seams are representable
    source       the manifold3d.Manifold this came from, when it came from one.
                 Probes that can be *exact* with a kernel say so in `method`.
    """

    verts: np.ndarray
    tris: np.ndarray
    name: str = "mesh"
    face_id: np.ndarray | None = None
    part_id: np.ndarray | None = None
    parts: Mapping[int, str] = EMPTY_MAP
    material_id: np.ndarray | None = None
    materials: Mapping[int, str] = EMPTY_MAP
    uv: np.ndarray | None = None
    uv_set: str = "UV0"
    source: object | None = None
    _topo: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        set_ = object.__setattr__
        set_(self, "verts", _freeze_as(self.verts, np.float64).reshape(-1, 3))
        set_(self, "tris", _freeze_as(self.tris, np.int64).reshape(-1, 3))
        if self.tris.size and (
            self.tris.min() < 0 or self.tris.max() >= len(self.verts)
        ):
            raise ValueError(
                "%s: triangle references vertex %d but there are %d verts"
                % (self.name, int(self.tris.max()), len(self.verts))
            )
        for chan in ("face_id", "part_id", "material_id"):
            val = getattr(self, chan)
            if val is None:
                continue
            arr = _freeze_as(val, np.int64).reshape(-1)
            if len(arr) != len(self.tris):
                raise ValueError(
                    "%s: %s has %d entries, expected one per triangle (%d)"
                    % (self.name, chan, len(arr), len(self.tris))
                )
            set_(self, chan, arr)
        if self.uv is not None:
            uv = _freeze_as(self.uv, np.float64)
            if uv.shape != (len(self.tris), 3, 2):
                raise ValueError(
                    "%s: uv must be (T,3,2) per-corner, got %s" % (self.name, (uv.shape,))
                )
            set_(self, "uv", uv)
        set_(self, "parts", MappingProxyType(dict(self.parts)))
        set_(self, "materials", MappingProxyType(dict(self.materials)))

    # -- basic counts ------------------------------------------------------
    @property
    def n_tris(self) -> int:
        return len(self.tris)

    @property
    def n_verts(self) -> int:
        return len(self.verts)

    def part_name(self, pid: int) -> str:
        return self.parts.get(int(pid), "part:%d" % int(pid))

    def material_name(self, mid: int) -> str:
        return self.materials.get(int(mid), "material:%d" % int(mid))

    def part_names(self) -> list[str]:
        if self.part_id is None:
            return []
        return [self.part_name(p) for p in sorted(set(self.part_id.tolist()))]

    # -- derived, memoised -------------------------------------------------
    @property
    def topology(self) -> "Topology":
        if self._topo is None:
            object.__setattr__(self, "_topo", build_topology(self))
        return self._topo  # type: ignore[return-value]

    # -- restriction -------------------------------------------------------
    def select_tris(self, mask, name: str | None = None) -> "Mesh":
        """A Mesh made of the selected triangles, verts compacted.

        Drops `source`: a subset of a solid is generally not a solid, and a
        probe must not silently claim kernel exactness for it.
        """
        mask = np.asarray(mask)
        if mask.dtype == bool:
            idx = np.flatnonzero(mask)
        else:
            idx = mask.astype(np.int64)
        tris = self.tris[idx]
        used = np.unique(tris)
        remap = np.full(len(self.verts), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        return Mesh(
            verts=self.verts[used],
            tris=remap[tris],
            name=name or self.name,
            face_id=None if self.face_id is None else self.face_id[idx],
            part_id=None if self.part_id is None else self.part_id[idx],
            parts=self.parts,
            material_id=None if self.material_id is None else self.material_id[idx],
            materials=self.materials,
            uv=None if self.uv is None else self.uv[idx],
            uv_set=self.uv_set,
            source=None,
        )

    def part(self, name_or_id) -> "Mesh":
        """The sub-mesh of one named part. Raises if the name is unknown."""
        if self.part_id is None:
            raise KeyError("%s carries no part ids" % self.name)
        if isinstance(name_or_id, str):
            ids = [k for k, v in self.parts.items() if v == name_or_id]
            if not ids:
                raise KeyError(
                    "no part named %r in %s; known parts: %s"
                    % (name_or_id, self.name, ", ".join(sorted(self.parts.values())))
                )
            pid = ids[0]
            label = name_or_id
        else:
            pid = int(name_or_id)
            label = self.part_name(pid)
        mask = self.part_id == pid
        if not mask.any():
            raise KeyError("part %r has no triangles in %s" % (label, self.name))
        return self.select_tris(mask, name="%s.%s" % (self.name, label))

    def __repr__(self):
        return "<Mesh %s verts=%d tris=%d>" % (self.name, self.n_verts, self.n_tris)


# ------------------------------------------------------------------ topology


@dataclass(frozen=True, eq=False, repr=False)
class Topology:
    """Edge/vertex bookkeeping shared by several probes.

    edges          (E,2) int64 unique undirected edges, low index first
    edge_tri_count (E,)  how many triangles use each edge (2 == manifold)
    tri_edges      (T,3) edge index per triangle side
    valence        (V,)  unique-edge degree per vertex
    vert_component (V,)  component label, -1 for unreferenced verts
    n_components   components counted over *used* vertices
    duplicate_halfedges  directed (a,b) sides seen more than once: winding clash
    """

    edges: np.ndarray
    edge_tri_count: np.ndarray
    tri_edges: np.ndarray
    valence: np.ndarray
    vert_component: np.ndarray
    n_components: int
    duplicate_halfedges: int
    vert_used: np.ndarray

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def boundary_edges(self) -> np.ndarray:
        return np.flatnonzero(self.edge_tri_count == 1)

    @property
    def nonmanifold_edges(self) -> np.ndarray:
        return np.flatnonzero(self.edge_tri_count > 2)

    @property
    def is_closed(self) -> bool:
        return len(self.boundary_edges) == 0

    @property
    def is_edge_manifold(self) -> bool:
        return len(self.nonmanifold_edges) == 0

    @property
    def is_oriented(self) -> bool:
        return self.duplicate_halfedges == 0

    def boundary_loops(self) -> int:
        """Connected components of the boundary-edge graph.

        Two loops that share a single vertex are counted as one; that is a
        pinch point and rare enough to be worth the simpler implementation.
        """
        be = self.edges[self.boundary_edges]
        if len(be) == 0:
            return 0
        return _count_components(be, int(be.max()) + 1)[1]


def _pair_key(pairs: np.ndarray, stride: int) -> np.ndarray:
    """Encode an (N,2) integer pair array as one int64 per row.

    np.unique(..., axis=0) lexsorts a structured view and costs about eight times
    a plain 1-D unique on the same data (measured: 39 ms against 4.8 ms for a
    32k-triangle sphere). Every pair here is bounded by the vertex or edge count,
    so a single int64 holds both components with room to spare.
    """
    a = pairs[:, 0].astype(np.int64, copy=False)
    b = pairs[:, 1].astype(np.int64, copy=False)
    return a * np.int64(stride) + b


def _count_duplicate_rows(rows: np.ndarray) -> int:
    """How many rows are repeats of an earlier identical row."""
    if len(rows) == 0:
        return 0
    order = np.lexsort(tuple(rows[:, i] for i in range(rows.shape[1] - 1, -1, -1)))
    srt = rows[order]
    return int(np.count_nonzero(np.all(srt[1:] == srt[:-1], axis=1)))


def connected_labels(edges: np.ndarray, n_nodes: int) -> np.ndarray:
    """Component label per node: the smallest node id in its component.

    Min-label propagation with pointer jumping, fully vectorised. Each round
    takes the minimum over each node's neighbours (one `minimum.reduceat` over a
    pre-sorted adjacency) and then compresses the label forest; labels only ever
    decrease and are bounded, so it converges, and at the fixed point every edge
    joins equal labels. The Python union-find this replaces cost 56 ms on a
    32k-triangle sphere.
    """
    labels = np.arange(n_nodes, dtype=np.int64)
    if len(edges) == 0:
        return labels
    src = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    dst = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64)
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    uniq, starts = np.unique(src, return_index=True)
    while True:
        previous = labels
        candidate = labels.copy()
        candidate[uniq] = np.minimum(
            labels[uniq], np.minimum.reduceat(labels[dst], starts)
        )
        while True:  # pointer jumping; candidate[i] <= i always holds
            jumped = candidate[candidate]
            if np.array_equal(jumped, candidate):
                break
            candidate = jumped
        if np.array_equal(candidate, previous):
            return candidate
        labels = candidate


def _count_components(edges: np.ndarray, n_nodes: int):
    """(labels, number of components among the nodes an edge touches)."""
    labels = connected_labels(edges, n_nodes)
    if len(edges) == 0:
        return labels, 0
    touched = np.zeros(n_nodes, dtype=bool)
    touched[edges.reshape(-1)] = True
    return labels, int(len(np.unique(labels[touched])))


def build_topology(mesh: Mesh) -> Topology:
    """Pure function: mesh arrays in, edge/vertex bookkeeping out."""
    tris = mesh.tris
    nv = mesh.n_verts
    if len(tris) == 0:
        z = np.zeros((0, 2), dtype=np.int64)
        return Topology(
            edges=z,
            edge_tri_count=np.zeros(0, dtype=np.int64),
            tri_edges=np.zeros((0, 3), dtype=np.int64),
            valence=np.zeros(nv, dtype=np.int64),
            vert_component=np.full(nv, -1, dtype=np.int64),
            n_components=0,
            duplicate_halfedges=0,
            vert_used=np.zeros(nv, dtype=bool),
        )

    halfedges = np.stack(
        [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=1
    ).reshape(-1, 2)
    undirected = np.sort(halfedges, axis=1)
    stride = max(nv, 1)
    keys, inverse, counts = np.unique(
        _pair_key(undirected, stride), return_inverse=True, return_counts=True
    )
    edges = np.stack([keys // stride, keys % stride], axis=1)
    tri_edges = inverse.reshape(-1, 3)

    # a directed side seen twice means two triangles traverse it the same way,
    # i.e. their winding disagrees
    _, he_counts = np.unique(_pair_key(halfedges, stride), return_counts=True)
    duplicate_halfedges = int(np.sum(he_counts - 1))

    valence = np.zeros(nv, dtype=np.int64)
    np.add.at(valence, edges.reshape(-1), 1)

    vert_used = np.zeros(nv, dtype=bool)
    vert_used[tris.reshape(-1)] = True
    labels, ncomp = _count_components(edges, nv)
    labels = np.where(vert_used, labels, -1)

    return Topology(
        edges=_freeze(edges),
        edge_tri_count=_freeze_as(counts, np.int64),
        tri_edges=_freeze_as(tri_edges, np.int64),
        valence=_freeze(valence),
        vert_component=_freeze(labels),
        n_components=int(ncomp),
        duplicate_halfedges=duplicate_halfedges,
        vert_used=_freeze(vert_used),
    )


# ------------------------------------------------------------ geometry maths
# Closed-form, exact for a closed mesh. No sampling, no tolerance knobs.


def tri_corners(mesh: Mesh):
    v = mesh.verts
    t = mesh.tris
    return v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]


def tri_cross(mesh: Mesh) -> np.ndarray:
    a, b, c = tri_corners(mesh)
    return np.cross(b - a, c - a)


def tri_areas(mesh: Mesh) -> np.ndarray:
    """(T,) triangle areas in m^2. |(b-a) x (c-a)| / 2 -- exact."""
    return 0.5 * np.linalg.norm(tri_cross(mesh), axis=1)


def tri_normals(mesh: Mesh) -> np.ndarray:
    n = tri_cross(mesh)
    ln = np.linalg.norm(n, axis=1)
    safe = np.where(ln > 0, ln, 1.0)[:, None]
    return n / safe


def tri_centroids(mesh: Mesh) -> np.ndarray:
    a, b, c = tri_corners(mesh)
    return (a + b + c) / 3.0


def signed_volume(mesh: Mesh) -> float:
    """Divergence theorem: sum of signed tetrahedra (origin, a, b, c) / 6.

    Exact for a closed, consistently wound mesh -- a unit cube gives exactly
    1.0. Positive means outward normals; negative means the mesh is inside out.
    Meaningless (but still computable) for an open mesh, which is why
    probe_volume() reports whether the mesh was closed.
    """
    a, b, c = tri_corners(mesh)
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def volume_centroid(mesh: Mesh):
    """(centroid (3,), signed volume) for a uniform-density solid.

    Each triangle spans a tetrahedron with the origin; the tet centroid is
    (a+b+c)/4 and its signed volume weights the sum. Exact for closed meshes.
    """
    a, b, c = tri_corners(mesh)
    vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))
    total = float(vol6.sum() / 6.0)
    if total == 0.0:
        return None, 0.0
    centre = ((a + b + c) / 4.0 * vol6[:, None]).sum(axis=0) / vol6.sum()
    return centre, total


def area_centroid(mesh: Mesh):
    """(centroid, total area) weighted by triangle area -- the shell centre.

    Used when the mesh is open, where a volume centroid does not exist.
    """
    areas = tri_areas(mesh)
    total = float(areas.sum())
    if total == 0.0:
        return None, 0.0
    return (tri_centroids(mesh) * areas[:, None]).sum(axis=0) / total, total


def bounds(mesh: Mesh):
    """(min (3,), max (3,)) over *referenced* vertices.

    Unreferenced vertices are excluded on purpose: a stray vertex 100 m away
    would otherwise silently multiply the reported size of the asset.
    """
    if mesh.n_tris:
        used = np.zeros(mesh.n_verts, dtype=bool)  # scatter, not sort
        used[mesh.tris.reshape(-1)] = True
        v = mesh.verts[used]
    else:
        v = mesh.verts
    if len(v) == 0:
        z = np.zeros(3)
        return z, z
    return v.min(axis=0), v.max(axis=0)


def _segment_distance2(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(P,T) squared distance from points to segments, clamped at both ends."""
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    denom = np.where(denom > 0.0, denom, 1.0)
    ap = p[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("pij,ij->pi", ap, ab) / denom, 0.0, 1.0)
    diff = ap - t[:, :, None] * ab[None, :, :]
    return np.einsum("pij,pij->pi", diff, diff)


def point_triangle_distance2(points: np.ndarray, mesh: Mesh,
                             chunk: int | None = None) -> np.ndarray:
    """(P,) squared distance from each point to the nearest triangle. Exact.

    For each point-triangle pair: if the point projects inside the triangle the
    answer is the plane distance, otherwise it is the nearest of the three edge
    segments (which covers the vertex regions too). Vectorised over triangles,
    chunked over points. Cost is O(P*T) with no acceleration structure -- see
    README for the size at which that starts to hurt.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    a, b, c = tri_corners(mesh)
    if len(a) == 0 or len(points) == 0:
        return np.full(len(points), np.inf)
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    nlen2 = np.einsum("ij,ij->i", normal, normal)
    degenerate = nlen2 <= 0.0
    nlen2_safe = np.where(degenerate, 1.0, nlen2)
    d00 = np.einsum("ij,ij->i", ab, ab)
    d01 = np.einsum("ij,ij->i", ab, ac)
    d11 = np.einsum("ij,ij->i", ac, ac)
    denom = d00 * d11 - d01 * d01
    denom_safe = np.where(np.abs(denom) > 0.0, denom, 1.0)

    if chunk is None:  # keep the (P,T,3) temporaries near 50 MB
        chunk = max(1, 2_000_000 // max(len(a), 1))

    out = np.empty(len(points))
    for start in range(0, len(points), chunk):
        p = points[start:start + chunk]
        ap = p[:, None, :] - a[None, :, :]
        d20 = np.einsum("pij,ij->pi", ap, ab)
        d21 = np.einsum("pij,ij->pi", ap, ac)
        u = (d11 * d20 - d01 * d21) / denom_safe
        v = (d00 * d21 - d01 * d20) / denom_safe
        inside = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & ~degenerate[None, :]
        # distance to the triangle plane
        signed = np.einsum("pij,ij->pi", ap, normal)
        d2_plane = signed * signed / nlen2_safe
        # distance to the three edges
        d2_edge = np.minimum(
            np.minimum(_segment_distance2(p, a, b), _segment_distance2(p, b, c)),
            _segment_distance2(p, c, a),
        )
        d2 = np.where(inside, d2_plane, d2_edge)
        out[start:start + chunk] = d2.min(axis=1)
    return out


# ------------------------------------------------------------------ adapters


def from_manifold(man, name: str = "solid", parts: Mapping[int, str] | None = None,
                  materials: Mapping[int, str] | None = None,
                  material_id=None, uv=None, keep_source: bool = True) -> Mesh:
    """Lift a manifold3d.Manifold into a neutral Mesh.

    Carries DESIGN.md 5.2 provenance across: ``run_original_id`` becomes
    ``part_id`` (exact attribution, one id per source solid) and ``face_id``
    becomes the logical coplanar face group, which is what an n-gon histogram
    and ``seat.top_face()`` both need.

    Two kernel footguns handled here. ``to_mesh()`` hands back read-only views
    and ``np.ascontiguousarray`` is a no-op on them, so every array is copied
    explicitly (prototype/README.md records the hour that cost). And ``to_mesh()``
    returns *float32* positions even though Manifold works in double internally,
    which turns a 600 mm cube into 600.00002 mm and quietly caps every probe at
    ~1e-7 relative accuracy -- so this reads ``to_mesh64()`` when it exists.
    """
    md = man.to_mesh64() if hasattr(man, "to_mesh64") else man.to_mesh()
    vp = np.array(md.vert_properties, dtype=np.float64, copy=True)
    verts = vp[:, :3]
    tris = np.array(md.tri_verts, dtype=np.int64, copy=True)

    face_id = None
    if getattr(md, "face_id", None) is not None:
        fid = np.array(md.face_id, dtype=np.int64, copy=True)
        if len(fid) == len(tris):
            face_id = fid

    part_id = None
    run_index = np.array(md.run_index, dtype=np.int64, copy=True)
    run_ids = np.array(md.run_original_id, dtype=np.int64, copy=True)
    if len(run_ids) and len(run_index) == len(run_ids) + 1:
        part_id = np.zeros(len(tris), dtype=np.int64)
        for i, rid in enumerate(run_ids.tolist()):
            lo, hi = run_index[i] // 3, run_index[i + 1] // 3
            part_id[lo:hi] = rid

    return Mesh(
        verts=verts,
        tris=tris,
        name=name,
        face_id=face_id,
        part_id=part_id,
        parts=parts or {},
        material_id=material_id,
        materials=materials or {},
        uv=uv,
        source=man if keep_source else None,
    )


def to_manifold(mesh: Mesh):
    """Rebuild a Manifold from a neutral Mesh, or return None with a reason.

    Returns (manifold, None) on success, (None, reason) when the mesh cannot be
    a solid -- an open or non-manifold mesh yields Error.NotManifold, which is
    exactly why the neutral type exists.
    """
    if mesh.source is not None:
        return mesh.source, None
    try:
        import manifold3d as m3
    except ImportError:  # pragma: no cover - environment dependent
        return None, "manifold3d is not installed"
    # Mesh64 keeps float64 positions, so the round trip is lossless; the float32
    # Mesh would quietly cost ~1e-7 relative precision on every probe below.
    verts = np.array(mesh.verts, dtype=np.float64, order="C", copy=True)
    tris = np.array(mesh.tris, dtype=np.uint64, order="C", copy=True)
    try:
        man = m3.Manifold(m3.Mesh64(vert_properties=verts, tri_verts=tris))
    except Exception as exc:  # pragma: no cover - kernel refusal path
        return None, "manifold3d rejected the mesh: %s" % exc
    status = man.status()
    if man.is_empty() or status != m3.Error.NoError:
        return None, "not a solid (%s)" % str(status).split(".")[-1]
    return man, None


__all__ = [
    "Mesh",
    "Topology",
    "build_topology",
    "from_manifold",
    "to_manifold",
    "tri_areas",
    "tri_normals",
    "tri_centroids",
    "tri_corners",
    "signed_volume",
    "volume_centroid",
    "area_centroid",
    "bounds",
    "point_triangle_distance2",
]
