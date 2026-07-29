"""
Tier 2 -- structural summaries.  DESIGN.md 6.

    "Topology histograms, per-island UV stats, material assignment table, LOD
     chain summary, poly distribution by named part. Compact, and agents read
     them far more reliably than pixels."

Five summaries, all text, all pure functions of an immutable Mesh (or of a list
of them, for a LOD chain). Each returns a `Summary`: `str(s)` is the block an
agent reads, `s.fields` is the same content as data for a caller that wants
numbers. Where a summary knows the verifier rule it feeds -- E102 n-gons, E106
overlapping UVs, W202 padding, W204/W205 LOD steps, W302 texel density, N402
dense flat regions -- it prints the code, because DESIGN.md 7 is explicit that a
named code plus numeric evidence is what actually gets repaired.

Milliseconds, not microseconds: these walk every edge and rasterise every UV
island.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import fmt, kernel
from . import mesh as _mesh
from .mesh import Mesh, _count_components
from .tier1 import _degree_text, _genus_per_component, _logical_faces, _pt
from .units import AXIS_NAMES, Area, Length, Volume


@dataclass(frozen=True)
class Summary:
    """A Tier 2 block: text for the agent, `fields` for the caller."""

    title: str
    body: tuple[str, ...] = ()
    fields: dict = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def text(self, ascii_only: bool | None = None) -> str:
        """The block as one string, folded to ASCII if stdout needs it.

        `title` and `body` stay UTF-8 whatever the console is; see
        `tier1.Probe.text`.
        """
        lines = [self.title, *self.body]
        if self.warnings:
            lines.append("")
            lines.extend("  ! " + w for w in self.warnings)
        return fmt.for_stdout("\n".join(lines), ascii_only)

    def __str__(self) -> str:
        return self.text()


# ------------------------------------------------------------------ helpers


def _pick_length_unit(median: float) -> str:
    """One unit per line, chosen from the median: mixing units inside a
    min/p50/max line is how a reader mis-reads it."""
    return "m" if median >= 1.0 else "mm"


def _pick_area_unit(median: float) -> str:
    return "m^2" if median >= 1e-2 else "mm^2"


def _dp_for(unit: str) -> int:
    """Decimals that do not pretend to a precision the number does not have."""
    return 1 if unit.startswith("mm") else 3


def _quantiles(values: np.ndarray, qs=(0, 50, 100)) -> list[float]:
    if len(values) == 0:
        return [0.0 for _ in qs]
    return [float(np.percentile(values, q)) for q in qs]


def _edge_face_pairs(mesh: Mesh):
    """(edge_index, tri_a, tri_b) for every edge with exactly two triangles."""
    topo = mesh.topology
    flat = topo.tri_edges.reshape(-1)
    tri_of = np.repeat(np.arange(mesh.n_tris), 3)
    order = np.argsort(flat, kind="stable")
    tri_sorted = tri_of[order]
    starts = np.concatenate([[0], np.cumsum(topo.edge_tri_count)[:-1]])
    pick = topo.edge_tri_count == 2
    idx = starts[pick]
    return np.flatnonzero(pick), tri_sorted[idx], tri_sorted[idx + 1]


def _edge_lengths(mesh: Mesh) -> np.ndarray:
    e = mesh.topology.edges
    if len(e) == 0:
        return np.zeros(0)
    return np.linalg.norm(mesh.verts[e[:, 1]] - mesh.verts[e[:, 0]], axis=1)


def _aspect_ratios(mesh: Mesh) -> np.ndarray:
    """Longest edge over shortest altitude, normalised so equilateral == 1.0.

    1.0 is ideal, a right isoceles triangle (a split quad) is sqrt(3) ~ 1.73,
    and 8+ is a sliver that will shade and bake badly. Degenerate triangles come
    back as inf.
    """
    a, b, c = _mesh.tri_corners(mesh)
    e0 = np.linalg.norm(b - a, axis=1)
    e1 = np.linalg.norm(c - b, axis=1)
    e2 = np.linalg.norm(a - c, axis=1)
    longest = np.maximum(np.maximum(e0, e1), e2)
    area = _mesh.tri_areas(mesh)
    with np.errstate(divide="ignore", invalid="ignore"):
        altitude = np.where(longest > 0, 2.0 * area / longest, 0.0)
        ratio = np.where(altitude > 0, longest / altitude, np.inf)
    return ratio / (2.0 / math.sqrt(3.0))


def _bucket_counts(values: np.ndarray, edges) -> list[tuple[str, int]]:
    out = []
    lo = -1
    for hi, label in edges:
        out.append((label, int(np.count_nonzero((values > lo) & (values <= hi)))))
        lo = hi
    return out


# --------------------------------------------------------------- 1. topology


def topology_summary(mesh: Mesh) -> Summary:
    """Histograms an agent would otherwise try to read off a wireframe render."""
    topo = mesh.topology
    n_faces, degrees = _logical_faces(mesh)
    areas = _mesh.tri_areas(mesh)
    lengths = _edge_lengths(mesh)
    ratios = _aspect_ratios(mesh)
    finite = ratios[np.isfinite(ratios)]
    valence = topo.valence[topo.vert_used]
    genus_list = _genus_per_component(mesh)
    genus = sum(genus_list) if genus_list else None

    a_unit = _pick_area_unit(float(np.median(areas)) if len(areas) else 0.0)
    l_unit = _pick_length_unit(float(np.median(lengths)) if len(lengths) else 0.0)
    a_lo, a_mid, a_hi = _quantiles(areas)
    l_lo, l_mid, l_hi = _quantiles(lengths)
    r_mid, r_95, r_max = _quantiles(finite, (50, 95, 100))
    slivers = int(np.count_nonzero(ratios > 8.0))
    valence_hist = _bucket_counts(
        valence,
        [(2, "<=2"), (3, "3"), (4, "4"), (5, "5"), (6, "6"), (7, "7"), (8, "8"),
         (10**9, "9+")],
    )

    lines = [
        fmt.probe_line(
            "tris / verts / edges",
            "%s / %s / %s"
            % (
                fmt.count(mesh.n_tris),
                fmt.count(int(np.count_nonzero(topo.vert_used))),
                fmt.count(topo.n_edges),
            ),
        ),
        fmt.probe_line(
            "faces (logical)",
            "%s   %s" % (fmt.count(n_faces), _degree_text(degrees)),
            note=None if set(degrees) <= {3, 4} else "E102 n-gons present",
        ),
        fmt.probe_line(
            "components / genus / holes",
            "%s / %s / %s"
            % (
                fmt.count(topo.n_components),
                "n/a" if genus is None else fmt.count(genus),
                fmt.count(topo.boundary_loops()),
            ),
        ),
        fmt.probe_line(
            "tri area",
            "min %s  p50 %s  max %s"
            % (
                Area(a_lo).format(a_unit, dp=_dp_for(a_unit)),
                Area(a_mid).format(a_unit, dp=_dp_for(a_unit)),
                Area(a_hi).format(a_unit, dp=_dp_for(a_unit)),
            ),
            note="max/min %s" % (fmt.ratio(a_hi / a_lo, 1) if a_lo > 0 else "inf"),
        ),
        fmt.probe_line(
            "edge length",
            "min %s  p50 %s  max %s"
            % (
                Length(l_lo).format(l_unit, dp=_dp_for(l_unit)),
                Length(l_mid).format(l_unit, dp=_dp_for(l_unit)),
                Length(l_hi).format(l_unit, dp=_dp_for(l_unit)),
            ),
        ),
        fmt.probe_line(
            "aspect ratio",
            "p50 %s  p95 %s  max %s"
            % (fmt.ratio(r_mid, 2), fmt.ratio(r_95, 2), fmt.ratio(r_max, 2)),
            note="slivers>8: %s" % fmt.count(slivers),
        ),
        fmt.histogram_line(
            "vertex valence", valence_hist,
            total=int(np.count_nonzero(topo.vert_used)),
        ),
    ]

    fields = {
        "tris": mesh.n_tris,
        "verts": int(np.count_nonzero(topo.vert_used)),
        "edges": topo.n_edges,
        "faces_logical": n_faces,
        "face_degrees": degrees,
        "components": topo.n_components,
        "genus": genus,
        "holes": topo.boundary_loops(),
        "tri_area_m2": (a_lo, a_mid, a_hi),
        "edge_length_m": (l_lo, l_mid, l_hi),
        "aspect_ratio": (r_mid, r_95, r_max),
        "slivers_over_8": slivers,
        "valence_hist": dict(valence_hist),
        "dihedral_hist": {},
    }
    warnings = []
    if set(degrees) - {3, 4}:
        warnings.append(
            "E102 %s logical face(s) have more than 4 sides (largest %d)"
            % (
                fmt.count(sum(n for d, n in degrees.items() if d > 4)),
                max(degrees) if degrees else 0,
            )
        )

    e_idx, ta, tb = _edge_face_pairs(mesh)
    if len(e_idx):
        normals = _mesh.tri_normals(mesh)
        dots = np.clip(np.einsum("ij,ij->i", normals[ta], normals[tb]), -1.0, 1.0)
        deg = np.degrees(np.arccos(dots))
        buckets = [
            ("flat<1°", int(np.count_nonzero(deg < 1.0))),
            ("1-30°", int(np.count_nonzero((deg >= 1.0) & (deg < 30.0)))),
            ("30-60°", int(np.count_nonzero((deg >= 30.0) & (deg < 60.0)))),
            ("hard>60°", int(np.count_nonzero(deg >= 60.0))),
        ]
        lines.append(
            fmt.histogram_line("dihedral (normal deviation)", buckets, total=len(deg))
        )
        fields["dihedral_hist"] = dict(buckets)
        flat_frac = buckets[0][1] / len(deg)
        fields["flat_edge_fraction"] = flat_frac
        if flat_frac > 0.6 and mesh.n_tris > 200:
            warnings.append(
                "N402 %s of interior edges are flat -- dense topology in a "
                "low-curvature region" % fmt.pct(flat_frac)
            )
    return Summary(
        title=fmt.header("topology %s" % mesh.name, "tier 2"),
        body=tuple(lines),
        fields=fields,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------- 2. UV


def uv_islands(mesh: Mesh, tol: float = 1e-9) -> np.ndarray:
    """Island label per triangle.

    Two triangles join when they share a mesh edge *and* both of their UV
    corners agree across it. Splitting on UV discontinuity rather than on UV
    position is what keeps two islands that happen to sit exactly on top of each
    other from being counted as one -- which is precisely the E106 case.
    """
    if mesh.uv is None:
        raise ValueError("%s carries no UV set" % mesh.name)
    e_idx, ta, tb = _edge_face_pairs(mesh)
    if len(e_idx) == 0:
        return np.arange(mesh.n_tris)
    edges = mesh.topology.edges[e_idx]
    v0, v1 = edges[:, 0], edges[:, 1]
    tris = mesh.tris

    def corner(tri_ids, vert_ids):
        return np.argmax(tris[tri_ids] == vert_ids[:, None], axis=1)

    a0, a1 = corner(ta, v0), corner(ta, v1)
    b0, b1 = corner(tb, v0), corner(tb, v1)
    uv = mesh.uv
    same = np.all(np.abs(uv[ta, a0] - uv[tb, b0]) <= tol, axis=1) & np.all(
        np.abs(uv[ta, a1] - uv[tb, b1]) <= tol, axis=1
    )
    pairs = np.stack([ta[same], tb[same]], axis=1)
    labels, _ = _count_components(pairs, mesh.n_tris)
    _, inverse = np.unique(labels, return_inverse=True)
    return inverse.reshape(-1)


def _uv_signed_areas(mesh: Mesh) -> np.ndarray:
    uv = mesh.uv
    d1 = uv[:, 1] - uv[:, 0]
    d2 = uv[:, 2] - uv[:, 0]
    return 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])


def _rasterize_uv(mesh: Mesh, rows: np.ndarray, res: int) -> np.ndarray:
    """Bool coverage of the given triangles' UVs on a res x res grid of [0,1]^2."""
    grid = np.zeros((res, res), dtype=bool)
    uv = mesh.uv[rows] * (res - 1)
    lo = np.clip(np.floor(uv.min(axis=1)).astype(int), 0, res - 1)
    hi = np.clip(np.ceil(uv.max(axis=1)).astype(int), 0, res - 1)
    for i in range(len(rows)):
        xs = np.arange(lo[i, 0], hi[i, 0] + 1)
        ys = np.arange(lo[i, 1], hi[i, 1] + 1)
        if not len(xs) or not len(ys):
            continue
        px, py = np.meshgrid(xs, ys, indexing="ij")
        p = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.float64)
        a, b, c = uv[i]
        v0, v1 = b - a, c - a
        det = v0[0] * v1[1] - v0[1] * v1[0]
        if abs(det) < 1e-12:
            continue
        w = p - a
        u = (w[:, 0] * v1[1] - w[:, 1] * v1[0]) / det
        v = (v0[0] * w[:, 1] - v0[1] * w[:, 0]) / det
        inside = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
        if inside.any():
            grid[px.ravel()[inside], py.ravel()[inside]] = True
    return grid


def texel_density_px_cm(uv_area: float, surface_area_m2: float,
                        texture_px: int) -> float:
    """px per centimetre of surface, the definition W302 is written against.

        px_per_m  = texture_px * sqrt(uv_area / surface_area)
        px_per_cm = px_per_m / 100
    """
    if surface_area_m2 <= 0.0 or uv_area <= 0.0:
        return 0.0
    return texture_px * math.sqrt(uv_area / surface_area_m2) / 100.0


def uv_summary(mesh: Mesh, texture_px: int = 1024,
               target_density_px_cm: float | None = None,
               raster: int = 512, max_islands_listed: int = 12) -> Summary:
    """Per-island UV stats: area, texel density, stretch, overlap, padding.

    Overlap (E106) is measured on a `raster` x `raster` grid, so it detects
    anything larger than one cell and cannot see anything smaller. Padding
    (W202) is measured between island UV bounding boxes, a conservative lower
    bound on the true gap: it can warn early, never late.
    """
    title = fmt.header("uv %s" % mesh.name, "tier 2")
    if mesh.uv is None:
        return Summary(
            title=title,
            body=(fmt.probe_line("uv set", "absent -- nothing to report"),),
            fields={"islands": 0, "has_uv": False},
        )
    labels = uv_islands(mesh)
    n_islands = int(labels.max()) + 1 if len(labels) else 0
    signed = _uv_signed_areas(mesh)
    uv_area = np.abs(signed)
    tri_area = _mesh.tri_areas(mesh)
    flipped = int(np.count_nonzero(signed < 0))

    per_island = []
    for isl in range(n_islands):
        sel = np.flatnonzero(labels == isl)
        uva = float(uv_area[sel].sum())
        m2 = float(tri_area[sel].sum())
        per_tri = np.array(
            [texel_density_px_cm(float(uv_area[i]), float(tri_area[i]), texture_px)
             for i in sel]
        )
        stretch = float(per_tri.max() / per_tri.min()) if per_tri.min() > 0 else float("inf")
        box_lo = mesh.uv[sel].reshape(-1, 2).min(axis=0)
        box_hi = mesh.uv[sel].reshape(-1, 2).max(axis=0)
        name = (
            mesh.part_name(mesh.part_id[sel[0]])
            if mesh.part_id is not None
            else "island_%d" % isl
        )
        per_island.append(
            {
                "island": isl,
                "name": name,
                "tris": int(len(sel)),
                "uv_area": uva,
                "area_m2": m2,
                "density_px_cm": texel_density_px_cm(uva, m2, texture_px),
                "stretch": stretch,
                "uv_bbox": (tuple(box_lo.tolist()), tuple(box_hi.tolist())),
                "rows": sel,
            }
        )
    per_island.sort(key=lambda d: -d["uv_area"])

    coverage = float(uv_area.sum())
    densities = np.array([d["density_px_cm"] for d in per_island])
    out_of_range = int(np.count_nonzero((mesh.uv < -1e-9) | (mesh.uv > 1.0 + 1e-9)))

    # overlap, E106
    masks = [_rasterize_uv(mesh, d["rows"], raster) for d in per_island]
    stack = np.zeros((raster, raster), dtype=np.int16)
    for msk in masks:
        stack += msk
    overlap_cells = int(np.count_nonzero(stack > 1))
    covered_cells = int(np.count_nonzero(stack > 0))
    overlap_fraction = overlap_cells / max(covered_cells, 1)
    pairs = []
    if overlap_cells and len(masks) <= 24:
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                both = int(np.count_nonzero(masks[i] & masks[j]))
                if both:
                    ref = min(int(np.count_nonzero(masks[i])),
                              int(np.count_nonzero(masks[j])))
                    pairs.append((both / max(ref, 1), per_island[i]["name"],
                                  per_island[j]["name"]))
        pairs.sort(reverse=True)

    # padding, W202 -- gap between island UV bounding boxes, in texture pixels
    padding_px = None
    if len(per_island) > 1:
        best = float("inf")
        for i in range(len(per_island)):
            lo_i, hi_i = (np.array(x) for x in per_island[i]["uv_bbox"])
            for j in range(i + 1, len(per_island)):
                lo_j, hi_j = (np.array(x) for x in per_island[j]["uv_bbox"])
                gap = np.maximum(np.maximum(lo_j - hi_i, lo_i - hi_j), 0.0)
                best = min(best, float(np.linalg.norm(gap)))
        padding_px = best * texture_px

    lines = [
        fmt.probe_line("uv set / texture", "%s / %d px" % (mesh.uv_set, texture_px)),
        fmt.probe_line(
            "islands", fmt.count(n_islands),
            note="coverage %s of the sheet" % fmt.pct(coverage),
        ),
    ]
    if len(densities):
        lines.append(
            fmt.probe_line(
                "texel density px/cm",
                "min %.3f  p50 %.3f  max %.3f"
                % (densities.min(), float(np.median(densities)), densities.max()),
                note="spread %s"
                % fmt.pct(
                    (densities.max() - densities.min()) / max(float(np.median(densities)), 1e-12)
                ),
            )
        )
    warnings = []
    worst_dev = None
    if target_density_px_cm and len(densities):
        dev = (densities - target_density_px_cm) / target_density_px_cm
        worst_dev = float(np.max(np.abs(dev)))
        lines.append(
            fmt.probe_line(
                "vs target %.3f px/cm" % target_density_px_cm,
                "worst deviation %s" % fmt.pct(worst_dev),
                note="W302" if worst_dev > 0.10 else "within ±10%",
            )
        )
        if worst_dev > 0.10:
            warnings.append(
                "W302 texel density %.3f-%.3f px/cm against a %.3f ±10%% target"
                % (densities.min(), densities.max(), target_density_px_cm)
            )
    lines.append(
        fmt.probe_line(
            "overlap",
            "none detected"
            if not overlap_cells
            else "%s of covered area on %s cells"
            % (fmt.pct(overlap_fraction), fmt.count(overlap_cells)),
            note="E106" if overlap_cells else "raster %d²" % raster,
        )
    )
    for frac, a, b in pairs[:4]:
        lines.append(
            fmt.field_line("overlaps", "%s ∩ %s  %s of the smaller island"
                           % (a, b, fmt.pct(frac)))
        )
    if overlap_cells:
        worst = pairs[0] if pairs else (overlap_fraction, "?", "?")
        warnings.append(
            "E106 UV islands overlap: %s ∩ %s at %s of the smaller island"
            % (worst[1], worst[2], fmt.pct(worst[0]))
        )
    if flipped:
        lines.append(
            fmt.probe_line("flipped tris", fmt.count(flipped), note="E106 mirrored UVs")
        )
        warnings.append("E106 %s triangle(s) have negative UV area (mirrored)"
                        % fmt.count(flipped))
    want_padding = max(4.0, texture_px / 256.0)
    if padding_px is not None:
        lines.append(
            fmt.probe_line(
                "island padding",
                "%.1f px (bbox lower bound)" % padding_px,
                note="W202 below %.0f px" % want_padding
                if padding_px < want_padding
                else "OK",
            )
        )
        if padding_px < want_padding:
            warnings.append(
                "W202 island padding %.1f px below the %.0f px table value for %d px"
                % (padding_px, want_padding, texture_px)
            )
    if out_of_range:
        lines.append(
            fmt.probe_line("uv outside 0..1", fmt.count(out_of_range),
                           note="check UDIM intent")
        )
    rows = []
    for d in per_island[:max_islands_listed]:
        lo, hi = d["uv_bbox"]
        rows.append(
            [
                d["name"],
                fmt.count(d["tris"]),
                "%.4f" % d["uv_area"],
                Area(d["area_m2"]).format("m^2"),
                "%.3f" % d["density_px_cm"],
                "%.2f" % d["stretch"] if math.isfinite(d["stretch"]) else "inf",
                "(%.3f,%.3f)-(%.3f,%.3f)" % (lo[0], lo[1], hi[0], hi[1]),
            ]
        )
    lines.append("")
    lines.extend(
        fmt.table(
            ["island", "tris", "uv area", "3d area", "px/cm", "stretch", "uv bbox"],
            rows,
            align="lrrrrrl",
        )
    )
    if len(per_island) > max_islands_listed:
        lines.append("    ... %d more islands, largest first"
                     % (len(per_island) - max_islands_listed))

    for d in per_island:
        d.pop("rows", None)
    return Summary(
        title=title,
        body=tuple(lines),
        fields={
            "has_uv": True,
            "texture_px": texture_px,
            "islands": n_islands,
            "coverage": coverage,
            "density_px_cm": (
                float(densities.min()), float(np.median(densities)),
                float(densities.max()),
            )
            if len(densities)
            else (0.0, 0.0, 0.0),
            "worst_density_deviation": worst_dev,
            "overlap_cells": overlap_cells,
            "overlap_fraction": overlap_fraction,
            "overlap_pairs": tuple(pairs),
            "flipped_tris": flipped,
            "padding_px": padding_px,
            "uv_outside_unit_square": out_of_range,
            "per_island": tuple(per_island),
            "raster": raster,
        },
        warnings=tuple(warnings),
    )


# -------------------------------------------------------------- 3. materials


def material_table(mesh: Mesh) -> Summary:
    """Which material covers what, by triangle count and by area.

    Area share is the number that matters for texture budget; triangle share is
    the one that matters for draw calls. Both are cheap, so print both.
    """
    title = fmt.header("materials %s" % mesh.name, "tier 2")
    if mesh.material_id is None:
        return Summary(
            title=title,
            body=(fmt.probe_line("materials", "no assignment channel on this mesh"),),
            fields={"materials": ()},
        )
    areas = _mesh.tri_areas(mesh)
    total_area = float(areas.sum())
    ids = np.unique(mesh.material_id)
    rows, records = [], []
    for mid in ids:
        sel = mesh.material_id == mid
        names = []
        if mesh.part_id is not None:
            names = sorted({mesh.part_name(p) for p in np.unique(mesh.part_id[sel])})
        a = float(areas[sel].sum())
        n = int(np.count_nonzero(sel))
        records.append(
            {
                "name": mesh.material_name(mid),
                "tris": n,
                "tri_fraction": n / max(mesh.n_tris, 1),
                "area_m2": a,
                "area_fraction": a / total_area if total_area else 0.0,
                "parts": tuple(names),
            }
        )
        rows.append(
            [
                mesh.material_name(mid),
                fmt.count(n),
                fmt.pct(n / max(mesh.n_tris, 1)),
                Area(a).format("m^2"),
                fmt.pct(a / total_area if total_area else 0.0),
                ", ".join(names[:3]) + (" +%d" % (len(names) - 3) if len(names) > 3 else ""),
            ]
        )
    lines = [
        fmt.probe_line(
            "materials", "%s assigned" % fmt.count(len(ids)),
            note="slots: " + ", ".join(mesh.material_name(i) for i in ids),
        ),
        "",
    ]
    lines.extend(
        fmt.table(["material", "tris", "%tris", "area", "%area", "parts"], rows,
                  align="lrrrrl")
    )
    return Summary(title=title, body=tuple(lines),
                   fields={"materials": tuple(records), "area_m2": total_area})


# ------------------------------------------------------------------ 4. parts


def part_summary(mesh: Mesh) -> Summary:
    """Polygon distribution by named part -- DESIGN.md 6's last Tier 2 item.

    Volume is printed only for parts that are closed on their own. A part of a
    unioned solid usually is not (its seam with the neighbour is open), and
    printing a plausible-looking number there would be exactly the sort of quiet
    lie this subsystem exists to prevent.
    """
    title = fmt.header("parts %s" % mesh.name, "tier 2")
    if mesh.part_id is None:
        return Summary(
            title=title,
            body=(fmt.probe_line("parts", "no provenance channel on this mesh"),),
            fields={"parts": ()},
        )
    areas = _mesh.tri_areas(mesh)
    ids = np.unique(mesh.part_id)
    rows, records = [], []
    for pid in ids:
        sel = mesh.part_id == pid
        sub = mesh.select_tris(sel)
        topo = sub.topology
        closed = topo.is_closed and topo.is_edge_manifold and topo.is_oriented
        lo, hi = _mesh.bounds(sub)
        n = int(np.count_nonzero(sel))
        vol = abs(_mesh.signed_volume(sub)) if closed else None
        records.append(
            {
                "name": mesh.part_name(pid),
                "tris": n,
                "tri_fraction": n / max(mesh.n_tris, 1),
                "verts": sub.n_verts,
                "size_m": tuple((hi - lo).tolist()),
                "area_m2": float(areas[sel].sum()),
                "volume_m3": vol,
                "closed": bool(closed),
            }
        )
        rows.append(
            [
                mesh.part_name(pid),
                fmt.count(n),
                fmt.pct(n / max(mesh.n_tris, 1)),
                fmt.count(sub.n_verts),
                _pt(hi - lo).format_size("m"),
                Area(float(areas[sel].sum())).format("m^2"),
                Volume(vol).format("m^3", dp=5) if vol is not None else "-",
                "yes" if closed else "no",
            ]
        )
    biggest = max(records, key=lambda r: r["tris"])
    lines = [
        fmt.probe_line("parts", fmt.count(len(ids))),
        fmt.probe_line(
            "largest",
            "%s (%s of tris)" % (biggest["name"], fmt.pct(biggest["tri_fraction"])),
        ),
        "",
    ]
    lines.extend(
        fmt.table(
            ["part", "tris", "%tris", "verts", "bbox", "area", "volume", "closed"],
            rows,
            align="lrrrlrrr",
        )
    )
    return Summary(title=title, body=tuple(lines), fields={"parts": tuple(records)})


# -------------------------------------------------------------------- 5. LOD


def _silhouette_deviation(a: Mesh, b: Mesh, axes, raster: int):
    """Worst 1 - IoU of the projected outlines over the given axes."""
    man_a, _ = _mesh.to_manifold(a)
    man_b, _ = _mesh.to_manifold(b)
    worst = 0.0
    if kernel.HAVE_MANIFOLD and man_a is not None and man_b is not None:
        for ax in axes:
            worst = max(worst, 1.0 - kernel.outline_overlap_exact(man_a, man_b, ax)["iou"])
        return worst, "outline_iou (exact)"
    for ax in axes:
        worst = max(worst, 1.0 - kernel.outline_overlap_raster(a, b, ax, res=raster)["iou"])
    return worst, "raster %d² (approximate)" % raster


def lod_summary(chain, target_ratio: float = 0.5, ratio_tol: float = 0.15,
                silhouette_max: float = 0.02, axes=(0, 1, 2),
                raster: int = 512) -> Summary:
    """LOD chain: reduction ratios, volume/area drift, silhouette deviation.

    Silhouette deviation is the render-free form of W205: project both solids
    down each axis and compare outlines, deviation = 1 - IoU. Exact through the
    kernel when both levels are closed solids (a 2D boolean on the projected
    outline), rasterised otherwise. The method is printed, since the two differ
    by a fraction of a percent.
    """
    if not chain:
        return Summary(title="lod chain (empty)", fields={"levels": ()})
    names = [n for n, _ in chain]
    meshes = [m for _, m in chain]
    base = meshes[0]
    base_tris = base.n_tris
    base_vol = abs(_mesh.signed_volume(base))
    base_area = float(_mesh.tri_areas(base).sum())
    base_lo, base_hi = _mesh.bounds(base)

    rows, records, warnings = [], [], []
    method_used = set()
    for i, (name, m) in enumerate(zip(names, meshes)):
        tris = m.n_tris
        vol = abs(_mesh.signed_volume(m))
        area = float(_mesh.tri_areas(m).sum())
        lo, hi = _mesh.bounds(m)
        bbox_delta = float(np.max(np.abs(np.concatenate([lo - base_lo, hi - base_hi]))))
        if i == 0:
            records.append(
                {"name": name, "tris": tris, "ratio_prev": None, "ratio_base": 1.0,
                 "volume_delta": 0.0, "area_delta": 0.0, "bbox_delta_m": 0.0,
                 "silhouette_deviation": 0.0}
            )
            rows.append([name, fmt.count(tris), "-", "1.000", "-", "-", "-", "-"])
            continue
        prev = meshes[i - 1].n_tris
        r_prev = tris / prev if prev else float("nan")
        r_base = tris / base_tris if base_tris else float("nan")
        dev, method = _silhouette_deviation(base, m, axes, raster)
        method_used.add(method)
        d_vol = (vol - base_vol) / base_vol if base_vol else 0.0
        d_area = (area - base_area) / base_area if base_area else 0.0
        records.append(
            {"name": name, "tris": tris, "ratio_prev": r_prev, "ratio_base": r_base,
             "volume_delta": d_vol, "area_delta": d_area, "bbox_delta_m": bbox_delta,
             "silhouette_deviation": dev}
        )
        rows.append(
            [
                name,
                fmt.count(tris),
                fmt.ratio(r_prev),
                fmt.ratio(r_base),
                fmt.pct(d_vol, 2),
                fmt.pct(d_area, 2),
                Length(bbox_delta).format("mm", dp=2),
                fmt.pct(dev, 2),
            ]
        )
        if abs(r_prev - target_ratio) > ratio_tol:
            warnings.append(
                "W204 %s reduces %s from the previous level (target %s ±%s)"
                % (name, fmt.ratio(r_prev), fmt.ratio(target_ratio), fmt.ratio(ratio_tol))
            )
        if dev > silhouette_max:
            warnings.append(
                "W205 %s silhouette deviates %s from LOD0 (limit %s)"
                % (name, fmt.pct(dev, 2), fmt.pct(silhouette_max, 2))
            )
    lines = [
        fmt.probe_line("levels", fmt.count(len(chain))),
        fmt.probe_line(
            "silhouette method",
            " + ".join(sorted(method_used)) if method_used else "n/a",
            note="deviation = 1 - outline IoU, worst of %s"
            % "/".join(AXIS_NAMES[a] for a in axes),
        ),
        "",
    ]
    lines.extend(
        fmt.table(
            ["level", "tris", "vs prev", "vs LOD0", "Δvolume", "Δarea", "Δbbox",
             "silhouette"],
            rows,
            align="lrrrrrrr",
        )
    )
    return Summary(
        title=fmt.header("lod chain %s" % names[0].rsplit("_LOD", 1)[0], "tier 2"),
        body=tuple(lines),
        fields={"levels": tuple(records), "method": tuple(sorted(method_used))},
        warnings=tuple(warnings),
    )


# ------------------------------------------------------------------ report


def report(mesh: Mesh, texture_px: int = 1024,
           target_density_px_cm: float | None = None,
           ascii_only: bool | None = None) -> str:
    """Every Tier 2 summary that applies to this mesh, in one block."""
    blocks = [topology_summary(mesh)]
    if mesh.material_id is not None:
        blocks.append(material_table(mesh))
    if mesh.part_id is not None:
        blocks.append(part_summary(mesh))
    if mesh.uv is not None:
        blocks.append(
            uv_summary(mesh, texture_px=texture_px,
                       target_density_px_cm=target_density_px_cm)
        )
    return "\n\n".join(b.text(ascii_only=ascii_only) for b in blocks)


__all__ = [
    "Summary",
    "topology_summary",
    "uv_summary",
    "uv_islands",
    "texel_density_px_cm",
    "material_table",
    "part_summary",
    "lod_summary",
    "report",
]
