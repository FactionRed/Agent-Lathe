"""
UV unwrapping and packing -- the data the section 7 UV rules were waiting for.

E106, W202, W203 and W302 have existed since the verifier landed. Every one of
them reported *not checked*, because nothing in the pipeline produced a UV set.
This closes that: xatlas generates charts and packs them, and a rescale pass
drives measured texel density onto a declared target.

Two decisions worth stating.

**UVs are per corner, not per vertex.** A seam is exactly the place where one
vertex needs two different UVs, so a per-vertex channel cannot represent one
without splitting the vertex -- and splitting it would mean handing Manifold a
mesh it would immediately weld back together. Storing `(T, 3, 2)` keeps the solid
untouched and the seam representable.

**Unwrap last.** A boolean or a bevel creates surface the unwrap never saw, so
those operations drop the UV channel rather than carry a stale one. That is the
order a real pipeline uses anyway: model, then unwrap, then bake.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .solid import LatheError, Solid

# DESIGN.md section 7 W202: minimum padding in pixels by texture resolution.
PADDING_TABLE = {256: 2, 512: 2, 1024: 4, 2048: 8, 4096: 16, 8192: 32}


def _xatlas():
    try:
        import xatlas
    except ImportError:  # pragma: no cover - environment dependent
        raise LatheError(
            "UV unwrapping needs xatlas.\n"
            "  pip install xatlas\n"
            "  Everything else in Lathe works without it; only unwrap and the "
            "four UV rules depend on it.")
    return xatlas


def unwrap(solid: Solid, max_angle: float = 66.0, resolution: int = 1024,
           padding_px: Optional[int] = None, straighten: bool = True,
           uv_set: str = "UV0", texel_density: Optional[float] = None,
           chart_cost: float = 0.35) -> Solid:
    """Generate a UV set for `solid`. Returns a new Solid carrying it.

    max_angle is the chart-splitting threshold in degrees: a crease sharper than
    this becomes a seam. Padding defaults to the section 7 table for the given
    resolution, so a freshly unwrapped asset passes W202 rather than needing a
    second pass to fix what this pass created.

    `texel_density` in px/cm is handed to the packer as texels-per-unit, so the
    density target and the padding are satisfied by the same pass. Rescaling
    afterwards was the first approach and it was worse: repacking by bounding box
    closed the gaps the packer had just placed, and W202 said so.

    Two limitations the section 7 rules will report, both real:

    **W202 padding.** xatlas pads chart *bounding rectangles*; the verifier
    measures *outline to outline*. A concave chart can nest into a neighbour's
    notch, so outlines come closer than the rectangle gap. Raising padding does
    not close it -- measured at 4, 6, 8 and 12 px, all still reported. Fixing it
    properly needs a polygon packer, not a bigger margin.

    **W302 density variance.** Padding and block alignment impose a minimum chart
    size, so a sliver gets more texels than its area asks for. On a mesh whose
    bevels are modelled rather than baked, those slivers are numerous and the
    spread is wide. The production answer is to bake the bevel into a normal map
    instead of carrying it in geometry, not to loosen the tolerance.
    """
    xat = _xatlas()
    msh = solid.mesh()
    if msh.n_tris == 0:
        raise LatheError("cannot unwrap %s: it has no triangles" % solid.name)

    pad = PADDING_TABLE.get(resolution) if padding_px is None else padding_px
    if pad is None:
        pad = max(2, int(round(resolution / 256.0)))

    atlas = xat.Atlas()
    atlas.add_mesh(
        np.ascontiguousarray(msh.verts, dtype=np.float32),
        np.ascontiguousarray(msh.tris, dtype=np.uint32))

    copt = xat.ChartOptions()
    # 0.35, not the library default. At looser costs a chart on the chair wrapped
    # far enough round a bevel to overlap itself, and E106 caught it: a chart that
    # folds is an under-split chart. Splitting harder costs 101 charts instead of
    # 84 and removes the fold entirely.
    copt.max_cost = float(chart_cost)
    copt.normal_deviation_weight = 2.0
    copt.straightness_weight = 6.0 if straighten else 1.0
    copt.max_chart_area = 0.0
    copt.max_boundary_length = 0.0

    popt = xat.PackOptions()
    popt.resolution = int(resolution)
    popt.padding = int(pad)
    popt.bilinear = True
    popt.blockAlign = True
    popt.bruteForce = False
    # px/cm -> texels per metre, because the kernel works in metres.
    popt.texels_per_unit = (0.0 if texel_density is None
                            else float(texel_density) * 100.0)

    atlas.generate(chart_options=copt, pack_options=popt)
    vmapping, indices, uvs = atlas.get_mesh(0)

    uv_corner = _corner_uvs(msh.tris, indices, uvs, resolution)
    uv_corner = _inset_border(uv_corner, pad, resolution)
    return solid.with_uv(uv_corner, uv_set=uv_set)


def _inset_border(uv: np.ndarray, pad_px: int, resolution: int) -> np.ndarray:
    """Keep the whole layout `pad_px` clear of the atlas edge.

    xatlas pads between charts but will place one flush against the boundary, and
    W202 counts the border as a padding case -- correctly, since bleed at the edge
    of a map is the same defect as bleed between islands. The inset costs a
    fraction of a percent of density, which is far less than the warning it clears.
    """
    if uv.size == 0:
        return uv
    margin = pad_px / float(resolution)
    lo = uv.reshape(-1, 2).min(axis=0)
    hi = uv.reshape(-1, 2).max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    room = 1.0 - 2.0 * margin
    if room <= 0:
        return uv
    scale = min(1.0, float(np.min(room / span)))
    return margin + (uv - lo) * scale


def _corner_uvs(orig_tris, indices, uvs, resolution) -> np.ndarray:
    """Lift xatlas' re-indexed output back onto the original corners.

    xatlas splits vertices at seams and returns its own index buffer, but it keeps
    face order, so face i of the output corresponds to face i of the input. That
    correspondence is what lets the UV land on `(T, 3, 2)` without the solid ever
    learning about the split.
    """
    indices = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    uvs = np.asarray(uvs, dtype=np.float64)
    if len(indices) != len(orig_tris):
        raise LatheError(
            "xatlas returned %d faces for %d input faces, so face order cannot "
            "be assumed. Refusing to guess a UV mapping."
            % (len(indices), len(orig_tris)))
    # xatlas reports uv in texels for the packed atlas; normalise to 0..1.
    if uvs.size and float(np.nanmax(uvs)) > 1.5:
        uvs = uvs / float(resolution)
    return uvs[indices.reshape(-1)].reshape(-1, 3, 2)


# --------------------------------------------------------------- measurement

def island_labels(uv: np.ndarray, tris) -> np.ndarray:
    """Group triangles into islands by shared UV position.

    Connectivity is in *UV* space, not on the mesh: two triangles adjacent on the
    surface but split across a seam belong to different islands, which is the
    whole point of a seam.
    """
    n = len(uv)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    # Quantise so corners that coincide in uv space hash together.
    key = np.round(uv.reshape(-1, 2), 6)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    corner_of = inverse.reshape(-1, 3)

    parent = np.arange(n, dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    seen: Dict[int, int] = {}
    for t in range(n):
        for c in corner_of[t]:
            c = int(c)
            if c in seen:
                ra, rb = find(seen[c]), find(t)
                if ra != rb:
                    parent[rb] = ra
            else:
                seen[c] = t
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def _tri_uv_area(uv: np.ndarray) -> np.ndarray:
    a, b, c = uv[:, 0], uv[:, 1], uv[:, 2]
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                        - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def texel_density(solid: Solid, resolution: int = 1024) -> Dict[str, object]:
    """Measured density per island, in pixels per centimetre.

    density = sqrt(uv_area) * resolution / sqrt(world_area), converted to cm.
    Per island, so a single internally stretched island can still average onto
    target -- W302 documents that limitation rather than hiding it.
    """
    from ..perceive import mesh as pmesh

    uv = solid.uv
    if uv is None:
        raise LatheError("%s has no UV set; call unwrap() first" % solid.name)
    msh = solid.mesh()
    world = pmesh.tri_areas(msh)
    uva = _tri_uv_area(uv)
    labels = island_labels(uv, msh.tris)

    out = []
    for lab in range(int(labels.max()) + 1 if len(labels) else 0):
        sel = labels == lab
        w, u = float(world[sel].sum()), float(uva[sel].sum())
        if w <= 0 or u <= 0:
            continue
        # px per metre -> px per cm
        d = (math.sqrt(u) * resolution) / math.sqrt(w) / 100.0
        out.append({"island": int(lab), "tris": int(sel.sum()),
                    "world_m2": w, "uv_area": u, "px_per_cm": d})
    if not out:
        return {"islands": [], "min": 0.0, "max": 0.0, "mean": 0.0}
    ds = [o["px_per_cm"] for o in out]
    return {"islands": out, "min": min(ds), "max": max(ds),
            "mean": sum(ds) / len(ds)}


def rescale_to_density(solid: Solid, target_px_per_cm: float,
                       resolution: int = 1024, padding_px: Optional[int] = None,
                       uv_set: str = "UV0") -> Solid:
    """Scale every island so its texel density matches the target, then repack.

    Uniform density is what W302 asks for, and an unwrap alone does not give it:
    xatlas packs to fill the atlas, which stretches whichever islands happen to
    need it. Each island is scaled by target/measured, then all islands are
    shelf-packed with the section 7 padding.
    """
    uv = solid.uv
    if uv is None:
        raise LatheError("%s has no UV set; call unwrap() first" % solid.name)
    from ..perceive import mesh as pmesh

    msh = solid.mesh()
    world = pmesh.tri_areas(msh)
    labels = island_labels(uv, msh.tris)
    n_isl = int(labels.max()) + 1 if len(labels) else 0
    out = uv.copy()

    boxes = []
    for lab in range(n_isl):
        sel = labels == lab
        w = float(world[sel].sum())
        u = float(_tri_uv_area(uv[sel]).sum())
        if w <= 0 or u <= 0:
            boxes.append(None)
            continue
        have = (math.sqrt(u) * resolution) / math.sqrt(w) / 100.0
        factor = (target_px_per_cm / have) if have > 0 else 1.0
        pts = out[sel]
        origin = pts.reshape(-1, 2).min(axis=0)
        out[sel] = origin + (pts - origin) * factor
        ext = out[sel].reshape(-1, 2)
        boxes.append((lab, ext.min(axis=0), ext.max(axis=0)))

    pad_px = PADDING_TABLE.get(resolution, 4) if padding_px is None else padding_px
    _shelf_pack(out, labels, boxes, pad_px / float(resolution))
    return solid.with_uv(out, uv_set=uv_set)


def _shelf_pack(uv: np.ndarray, labels: np.ndarray, boxes, gap: float) -> None:
    """Tallest-first shelf packing, in place. Not optimal, but it respects the
    padding, which is the part W202 measures."""
    live = [b for b in boxes if b is not None]
    live.sort(key=lambda b: -(b[2][1] - b[1][1]))
    x = y = gap
    shelf_h = 0.0
    for lab, lo, hi in live:
        w, h = (hi[0] - lo[0]), (hi[1] - lo[1])
        if x + w + gap > 1.0 and shelf_h > 0:
            x = gap
            y += shelf_h + gap
            shelf_h = 0.0
        sel = labels == lab
        uv[sel] = uv[sel] - lo + np.array([x, y])
        x += w + gap
        shelf_h = max(shelf_h, h)


def overlap_fraction(solid: Solid) -> float:
    """Rough share of UV area that two islands both claim.

    A cheap screen for E106 before the verifier runs its exact clip: rasterises
    island bounding boxes into a coarse grid and counts cells claimed twice.
    """
    uv = solid.uv
    if uv is None:
        return 0.0
    labels = island_labels(uv, solid.mesh().tris)
    n = int(labels.max()) + 1 if len(labels) else 0
    grid = np.zeros((256, 256), dtype=np.int16)
    for lab in range(n):
        pts = uv[labels == lab].reshape(-1, 2)
        lo = np.clip((pts.min(axis=0) * 256).astype(int), 0, 255)
        hi = np.clip((pts.max(axis=0) * 256).astype(int) + 1, 1, 256)
        grid[lo[1]:hi[1], lo[0]:hi[0]] += 1
    claimed = (grid > 0).sum()
    return float((grid > 1).sum()) / float(claimed) if claimed else 0.0
