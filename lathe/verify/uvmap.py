"""
UV topology: islands, island area, texel density, island outlines.

Split from mesh.py for the same reason as topology.py, and because UVs are the
one part of the asset where "the same vertex" and "the same point in the map"
are different questions -- E106, W202 and W302 all turn on that distinction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geom import EdgeKey, edge_key, polygon_area_2d, seg_key
from .mesh import AssetMesh


@dataclass(frozen=True)
class Island:
    """A UV island, named where the mesh gave us a name to use."""

    index: int
    name: str
    faces: Tuple[int, ...]
    uv_area: float  # in unit-square space
    world_area: float  # m^2
    bbox_min: Tuple[float, float]
    bbox_max: Tuple[float, float]

    def texel_density(self, resolution: int) -> float:
        """px/cm. sqrt of both areas, so it is a length ratio, not an area one."""
        if self.world_area <= 0.0:
            return 0.0
        return (math.sqrt(self.uv_area) * resolution) / (math.sqrt(self.world_area) * 100.0)


class UVLayout:
    """UV topology for one uv set: islands, per-island area, triangles.

    UVs live on face corners (loops), not vertices, because seams are exactly
    the case where one vertex carries two UVs.
    """

    def __init__(self, mesh: AssetMesh, set_name: str, weld_eps: float = 1e-7) -> None:
        if set_name not in mesh.uv_sets:
            raise KeyError(set_name)
        self.mesh = mesh
        self.set_name = set_name
        self.uv = mesh.uv_sets[set_name]
        self.eps = weld_eps
        self._starts = mesh.loop_starts
        self.islands: Tuple[Island, ...] = tuple(self._build_islands())
        self.island_of_face: Dict[int, int] = {
            f: isl.index for isl in self.islands for f in isl.faces
        }

    # -- helpers -----------------------------------------------------------

    def face_uv(self, fi: int) -> np.ndarray:
        s = self._starts[fi]
        return self.uv[s : s + len(self.mesh.faces[fi])]

    def uv_of_vert_in_face(self, fi: int, v: int) -> Optional[np.ndarray]:
        f = self.mesh.faces[fi]
        if v not in f:
            return None
        return self.uv[self._starts[fi] + f.index(v)]

    def face_uv_tris(self, fi: int) -> List[np.ndarray]:
        poly = self.face_uv(fi)
        return [np.array([poly[0], poly[t], poly[t + 1]]) for t in range(1, len(poly) - 1)]

    def uv_area_of_face(self, fi: int) -> float:
        return abs(polygon_area_2d(self.face_uv(fi)))

    def flipped_faces(self) -> Tuple[int, ...]:
        """Negative signed UV area == mirrored winding in UV space."""
        return tuple(
            fi
            for fi in range(self.mesh.num_faces)
            if polygon_area_2d(self.face_uv(fi)) < -1e-12
        )

    # -- islands -----------------------------------------------------------

    def _build_islands(self) -> List[Island]:
        mesh = self.mesh
        topo_edges: Dict[EdgeKey, List[int]] = {}
        for fi, f in enumerate(mesh.faces):
            for j in range(len(f)):
                a, b = f[j], f[(j + 1) % len(f)]
                if a != b:
                    topo_edges.setdefault(edge_key(a, b), []).append(fi)

        parent = list(range(mesh.num_faces))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for (a, b), faces in topo_edges.items():
            for i in range(len(faces) - 1):
                for j in range(i + 1, len(faces)):
                    fa, fb = faces[i], faces[j]
                    if self._uv_continuous(fa, fb, a, b):
                        ra, rb = find(fa), find(fb)
                        if ra != rb:
                            parent[ra] = rb

        groups: Dict[int, List[int]] = {}
        for fi in range(mesh.num_faces):
            groups.setdefault(find(fi), []).append(fi)

        keys = sorted(groups, key=lambda r: min(groups[r]))
        out: List[Island] = []
        used_names: Dict[str, int] = {}
        for idx, root in enumerate(keys):
            faces = tuple(sorted(groups[root]))
            name = self._name_for(faces, used_names)
            uvs = np.vstack([self.face_uv(fi) for fi in faces])
            out.append(
                Island(
                    index=idx,
                    name=name,
                    faces=faces,
                    uv_area=float(sum(self.uv_area_of_face(fi) for fi in faces)),
                    world_area=float(sum(mesh.face_area(fi) for fi in faces)),
                    bbox_min=tuple(uvs.min(axis=0)),
                    bbox_max=tuple(uvs.max(axis=0)),
                )
            )
        return out

    def _uv_continuous(self, fa: int, fb: int, a: int, b: int) -> bool:
        ua, ub = self.uv_of_vert_in_face(fa, a), self.uv_of_vert_in_face(fb, a)
        va, vb = self.uv_of_vert_in_face(fa, b), self.uv_of_vert_in_face(fb, b)
        if ua is None or ub is None or va is None or vb is None:
            return False
        return bool(
            np.allclose(ua, ub, atol=self.eps) and np.allclose(va, vb, atol=self.eps)
        )

    def _name_for(self, faces: Sequence[int], used: Dict[str, int]) -> str:
        """Most common face group wins; ties and repeats get a suffix, so island
        names are stable and unique for the report."""
        votes: Dict[str, int] = {}
        for fi in faces:
            g = self.mesh.group_of_face(fi)
            if g:
                votes[g] = votes.get(g, 0) + 1
        base = max(sorted(votes), key=lambda k: (votes[k], k)) if votes else "island_%02d" % (
            len(used) if used else 0
        )
        if base not in used:
            used[base] = 1
            return base
        used[base] += 1
        return "%s#%d" % (base, used[base])

    def island_boundary_segments(self, isl: Island) -> List[np.ndarray]:
        """UV-space boundary of an island: edges with no UV-continuous partner.
        Used by W202 padding, which needs distance between island outlines."""
        counts: Dict[Tuple[Tuple[float, float], Tuple[float, float]], int] = {}
        for fi in isl.faces:
            poly = self.face_uv(fi)
            n = len(poly)
            for j in range(n):
                p, q = poly[j], poly[(j + 1) % n]
                key = seg_key(p, q)
                counts[key] = counts.get(key, 0) + 1
        segs = []
        for (p, q), c in counts.items():
            if c == 1:
                segs.append(np.array([p, q], dtype=np.float64))
        return segs


# --------------------------------------------------------------- math bits
