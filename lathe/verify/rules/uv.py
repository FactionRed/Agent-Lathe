"""
UV rules: E106, W202, W302.

These are the rules whose report shape section 7 shows verbatim, so they are
the ones held closest to the sample output.

Division of labour, so that one defect produces one finding: E106 owns islands
that intersect, W202 owns islands that are merely too close. A pair with zero
gap is an overlap, so W202 defers to E106 rather than reporting the same pair
twice. An agent that receives two codes for one defect will fix it twice.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from ..engine import rule
from ..findings import ERROR, WARN, Raw, Repair
from ..geom import overlap_area, polygon_area_2d, seg_seg_distance
from ._common import loc2, num

# --------------------------------------------------------------------- E106

_GRID = 64  # uv-space hash resolution for the broad phase


@rule(
    "E106",
    ERROR,
    "Overlapping UVs on unique bake",
    needs=("emit.bake", "uv"),
    defaults={"min_overlap_fraction": 0.005, "max_pairs": 6},
)
def e106(ctx):
    """Two islands sharing texture space is fine for a tiling map and fatal for
    a unique bake: the second island overwrites the first, and the normal map
    lights one of them wrong. Applicability is driven by the emit declaration,
    not guessed from the geometry."""
    bake = ctx.emit.bake
    if bake is None or not bake.unique_uv:
        ctx.not_applicable("emit declares no unique-uv bake, so overlap is legal")
    if not bake.targets:
        ctx.not_applicable("emit declares a unique uv set but no bake targets")
    uvl = ctx.uv(bake.uv_set)

    pairs, self_overlaps = _overlap_report(uvl, float(ctx.p("min_overlap_fraction")))
    total = len(pairs) + len(self_overlaps)
    if not total:
        return

    flipped = uvl.flipped_faces()
    bake_line = "bake target = %s, uv_set = %s" % (", ".join(bake.targets), bake.uv_set)

    for (a, b), area in pairs[: ctx.p("max_pairs")]:
        ia, ib = uvl.islands[a], uvl.islands[b]
        frac = area / max(min(ia.uv_area, ib.uv_area), 1e-12)
        # Mirrored islands take a different repair from merely badly packed
        # ones, so the distinction has to come out of the measurement.
        mirrored = _is_mirrored(uvl, ia, ib) or any(
            f in ia.faces or f in ib.faces for f in flipped
        )
        yield ctx.finding(
            entity="%s.%s" % (ctx.name, ia.name),
            where=[
                (
                    "islands",
                    "%s overlaps %s (%s%% area)" % (ia.name, ib.name, num(frac * 100.0)),
                ),
                (
                    "uv",
                    "%s bbox %s→%s   %s bbox %s→%s"
                    % (
                        ia.name,
                        loc2(ia.bbox_min),
                        loc2(ia.bbox_max),
                        ib.name,
                        loc2(ib.bbox_min),
                        loc2(ib.bbox_max),
                    ),
                ),
            ],
            evidence="%s; overlap = %s uv² of %s uv² (%s%%)%s%s"
            % (
                bake_line,
                num(area),
                num(min(ia.uv_area, ib.uv_area)),
                num(frac * 100.0),
                "; islands are mirrored" if mirrored else "",
                "" if total == 1 else "; %d overlapping island pairs in this uv set" % total,
            ),
            data={
                "islands": [ia.name, ib.name],
                "overlap_uv_area": area,
                "island_uv_area": [ia.uv_area, ib.uv_area],
                "overlap_fraction": frac,
                "mirrored": mirrored,
                "bake_targets": list(bake.targets),
                "uv_set": bake.uv_set,
                "flipped_face_count": len(flipped),
            },
            repair=_overlap_repair(ctx, mirrored, [ia.name, ib.name]),
        )

    for isl_index, area in self_overlaps[: ctx.p("max_pairs")]:
        isl = uvl.islands[isl_index]
        frac = area / max(isl.uv_area, 1e-12)
        yield ctx.finding(
            summary="Overlapping UVs on unique bake: %s folds onto itself" % isl.name,
            entity="%s.%s" % (ctx.name, isl.name),
            where=[
                ("islands", "%s self-overlaps (%s%% area)" % (isl.name, num(frac * 100.0))),
                ("faces", "%d faces, %d with mirrored uv winding"
                 % (len(isl.faces), len([f for f in flipped if f in isl.faces]))),
            ],
            evidence="%s; self-overlap = %s uv² of %s uv² (%s%%)"
            % (bake_line, num(area), num(isl.uv_area), num(frac * 100.0)),
            data={
                "island": isl.name,
                "self_overlap_uv_area": area,
                "island_uv_area": isl.uv_area,
                "overlap_fraction": frac,
                "flipped_faces_in_island": [f for f in flipped if f in isl.faces],
            },
            repair=_overlap_repair(ctx, True, [isl.name]),
        )


def _overlap_repair(ctx, mirrored: bool, islands: List[str]) -> Repair:
    res = ctx.emit.texture_resolution
    if mirrored:
        return Repair(
            "uv.unwrap",
            (Raw(ctx.name),),
            {"split_mirrored": True},
            alternatives=(
                "emit with bake_uv_set = UV1",
                "uv.pack(%s, islands = [%s], margin_px = %d, resolution = %d)"
                % (ctx.name, ", ".join(islands), _margin_px(res), res),
            ),
        )
    return Repair(
        "uv.pack",
        (Raw(ctx.name),),
        {"islands": [Raw(i) for i in islands], "margin_px": _margin_px(res),
         "resolution": res},
        alternatives=("emit with bake_uv_set = UV1",),
    )


def _is_mirrored(uvl, ia, ib) -> bool:
    """Signed uv area of opposite sign means one island is a mirror of the
    other -- the classic cause of an exact stacked overlap."""
    sa = sum(polygon_area_2d(uvl.face_uv(f)) for f in ia.faces)
    sb = sum(polygon_area_2d(uvl.face_uv(f)) for f in ib.faces)
    return (sa > 0) != (sb > 0)


def _overlap_report(uvl, min_fraction: float):
    """Broad phase over a uv grid, then exact convex clip. Returns
    (island pairs, self-overlapping islands), largest first."""
    tris: List[Tuple[int, int, np.ndarray]] = []
    boxes: List[Tuple[float, float, float, float]] = []
    for isl in uvl.islands:
        for fi in isl.faces:
            for t in uvl.face_uv_tris(fi):
                tris.append((isl.index, fi, t))
                boxes.append(
                    (float(t[:, 0].min()), float(t[:, 1].min()),
                     float(t[:, 0].max()), float(t[:, 1].max()))
                )

    buckets: Dict[Tuple[int, int], List[int]] = {}
    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        for cx in range(_cell(x0), _cell(x1) + 1):
            for cy in range(_cell(y0), _cell(y1) + 1):
                buckets.setdefault((cx, cy), []).append(idx)

    pair_area: Dict[Tuple[int, int], float] = {}
    self_area: Dict[int, float] = {}
    tested = set()
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if (a, b) in tested:
                    continue
                tested.add((a, b))
                ia, fa, ta = tris[a]
                ib, fb, tb = tris[b]
                if fa == fb:
                    continue  # fan triangles of one face never overlap
                # Sharing a grid cell is not sharing area. The bbox test with
                # `<=` also rejects edge-adjacent triangles, whose boxes touch
                # but do not overlap -- that is the common case in a dense mesh.
                ax0, ay0, ax1, ay1 = boxes[a]
                bx0, by0, bx1, by1 = boxes[b]
                if ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0:
                    continue
                area = overlap_area(ta, tb)
                if area <= 1e-12:
                    continue
                if ia == ib:
                    self_area[ia] = self_area.get(ia, 0.0) + area
                else:
                    key = (min(ia, ib), max(ia, ib))
                    pair_area[key] = pair_area.get(key, 0.0) + area

    pairs = []
    for (a, b), area in pair_area.items():
        smallest = max(min(uvl.islands[a].uv_area, uvl.islands[b].uv_area), 1e-12)
        if area / smallest >= min_fraction:
            pairs.append(((a, b), area))
    selves = [
        (i, area)
        for i, area in self_area.items()
        if area / max(uvl.islands[i].uv_area, 1e-12) >= min_fraction
    ]
    pairs.sort(key=lambda kv: (-kv[1], kv[0]))
    selves.sort(key=lambda kv: (-kv[1], kv[0]))
    return pairs, selves


def _cell(v: float) -> int:
    return max(0, min(_GRID - 1, int(v * _GRID)))


# --------------------------------------------------------------------- W202

# Minimum island gap in pixels, by target resolution. Two texels of padding per
# mip level that matters; the table is a profile parameter, not a law.
PADDING_TABLE = {256: 2, 512: 2, 1024: 4, 2048: 8, 4096: 16, 8192: 32}


@rule(
    "W202",
    WARN,
    "UV padding below the table for target resolution",
    needs=("uv", "emit.texture_resolution"),
    defaults={"padding_table": PADDING_TABLE, "max_report": 6},
    approximation="Measures distance between island outlines in uv space and "
    "converts to pixels with the declared resolution. It does not rasterise, so "
    "it cannot see bleed that only appears at a specific mip level, and it "
    "assumes square, non-tiling textures.",
)
def w202(ctx):
    """Islands closer than the padding table bleed into each other when the
    texture is mipped. Also checks the atlas border, which is the same defect
    against the edge of the map."""
    uvl = ctx.uv()
    res = int(ctx.emit.texture_resolution)
    table = dict(ctx.p("padding_table"))
    need_px = _lookup_padding(table, res)
    need_uv = need_px / float(res)

    offenders = []
    overlapping = 0
    islands = uvl.islands
    for i in range(len(islands)):
        for j in range(i + 1, len(islands)):
            a, b = islands[i], islands[j]
            if _bbox_gap(a, b) >= need_uv:
                continue  # bbox gap is a lower bound on the real gap
            gap = _island_gap(uvl, a, b)
            if gap <= 0.0:
                overlapping += 1  # an overlap, not a padding problem: see E106
                continue
            if gap < need_uv:
                offenders.append((a.name, b.name, gap))

    border = []
    for isl in islands:
        d = min(
            isl.bbox_min[0], isl.bbox_min[1], 1.0 - isl.bbox_max[0], 1.0 - isl.bbox_max[1]
        )
        if d < need_uv:
            border.append((isl.name, d))

    if not offenders and not border:
        return

    offenders.sort(key=lambda t: t[2])
    border.sort(key=lambda t: t[1])
    rows = [
        "%s ↔ %s: %s px (need %s px)" % (a, b, num(g * res), num(need_px))
        for a, b, g in offenders[: ctx.p("max_report")]
    ]
    rows += [
        "%s ↔ atlas border: %s px (need %s px)" % (n, num(d * res), num(need_px))
        for n, d in border[: ctx.p("max_report")]
    ]
    worst = min([g for _, _, g in offenders] + [d for _, d in border])
    yield ctx.finding(
        summary="UV padding %s px below the %d table minimum of %s px"
        % (num(worst * res), res, num(need_px)),
        entity=ctx.name,
        where=[("islands", rows)],
        evidence="worst gap = %s px at %d² (table requires %s px); %d island pairs and "
        "%d border cases under the minimum%s"
        % (
            num(worst * res),
            res,
            num(need_px),
            len(offenders),
            len(border),
            "; %d further pairs overlap outright (E106)" % overlapping if overlapping else "",
        ),
        data={
            "resolution": res,
            "required_px": need_px,
            "worst_gap_px": worst * res,
            "pairs": [{"a": a, "b": b, "gap_px": g * res} for a, b, g in offenders],
            "border": [{"island": n, "gap_px": d * res} for n, d in border],
            "overlapping_pairs": overlapping,
        },
        repair=Repair(
            "uv.pack",
            (Raw(ctx.name),),
            {"margin_px": int(math.ceil(need_px)), "resolution": res},
            alternatives=(
                "emit with texture_resolution = %d, which lowers the required margin"
                % _next_res_down(table, res),
            ),
        ),
    )


def _lookup_padding(table: Dict[int, int], res: int) -> float:
    keys = sorted(table)
    best = keys[0]
    for k in keys:
        if k <= res:
            best = k
    return float(table[best])


def _next_res_down(table: Dict[int, int], res: int) -> int:
    keys = [k for k in sorted(table) if k < res]
    return keys[-1] if keys else res


def _bbox_gap(a, b) -> float:
    dx = max(a.bbox_min[0] - b.bbox_max[0], b.bbox_min[0] - a.bbox_max[0], 0.0)
    dy = max(a.bbox_min[1] - b.bbox_max[1], b.bbox_min[1] - a.bbox_max[1], 0.0)
    return math.hypot(dx, dy)


def _island_gap(uvl, a, b) -> float:
    sa = uvl.island_boundary_segments(a)
    sb = uvl.island_boundary_segments(b)
    best = float("inf")
    for s1 in sa:
        for s2 in sb:
            d = seg_seg_distance(s1[0], s1[1], s2[0], s2[1])
            if d < best:
                best = d
                if best == 0.0:
                    return 0.0
    return best


# --------------------------------------------------------------------- W302


@rule(
    "W302",
    WARN,
    "Texel density variance outside target ±10%",
    needs=("uv", "emit.texel_density"),
    defaults={"tolerance_pct": 10.0, "max_report": 8},
    approximation="Density is computed per island as sqrt(uv area) * resolution "
    "/ sqrt(world area) in cm, so it is the island average. A single island that "
    "is stretched internally can average onto the target and pass. A per-face "
    "measure would catch that and would also flag every legitimately foreshortened "
    "face, so the average is the deliberate choice.",
)
def w302(ctx):
    """`texel_density : 5.12 px/cm` from the section 4 emit block, checked."""
    target = ctx.emit.texel_density
    if target is None:
        ctx.not_applicable("emit declares no texel_density target")
    uvl = ctx.uv()
    res = int(ctx.emit.texture_resolution)
    tol = float(ctx.p("tolerance_pct")) / 100.0
    lo, hi = target * (1.0 - tol), target * (1.0 + tol)

    measured = [(isl, isl.texel_density(res)) for isl in uvl.islands]
    measured = [(isl, d) for isl, d in measured if isl.world_area > 0 and isl.uv_area > 0]
    if not measured:
        ctx.not_applicable("no island has both uv area and world area")
    offenders = [(isl, d) for isl, d in measured if d < lo or d > hi]
    if not offenders:
        return

    densities = [d for _, d in measured]
    offenders.sort(key=lambda t: abs(t[1] - target), reverse=True)
    names = [isl.name for isl, _ in offenders]
    rows = [
        "%-14s %s px/cm  (%s%% off target)"
        % (isl.name, num(d), num(100.0 * (d - target) / target))
        for isl, d in offenders[: ctx.p("max_report")]
    ]
    yield ctx.finding(
        summary="Texel density variance %s–%s px/cm (target %s ±%s%%)"
        % (num(min(densities)), num(max(densities)), num(target), num(tol * 100.0)),
        entity=ctx.name,
        where=[("islands", ", ".join(names[: ctx.p("max_report")])), ("measured", rows)],
        evidence="target %s px/cm at %d², allowed %s–%s; %d of %d islands outside, "
        "observed %s–%s px/cm"
        % (
            num(target),
            res,
            num(lo),
            num(hi),
            len(offenders),
            len(measured),
            num(min(densities)),
            num(max(densities)),
        ),
        data={
            "target_px_cm": target,
            "tolerance_pct": float(ctx.p("tolerance_pct")),
            "resolution": res,
            "observed_min": min(densities),
            "observed_max": max(densities),
            "islands": [
                {"island": isl.name, "px_cm": d, "off_target_pct": 100.0 * (d - target) / target}
                for isl, d in offenders
            ],
        },
        repair=Repair(
            "uv.rescale",
            (),
            {"islands": [Raw(n) for n in names], "to": target},
            alternatives=(
                "emit with texel_density = %s px/cm if the current layout is intended"
                % num(sum(densities) / len(densities)),
            ),
        ),
    )


def _margin_px(res: int) -> int:
    return int(_lookup_padding(PADDING_TABLE, res))
