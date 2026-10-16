"""
Bevel, per the DESIGN.md section 11 decision.

The spike measured three candidates. OCCT gives the best bevel but fillets a
feature history rather than a mesh, and the implicit routes collapse provenance
to a single id. So bevel runs bmesh in a headless Blender subprocess, and this
wrapper -- not the caller -- enforces the non-inheriting tag default from
section 5.2.

Provenance round-trips back into Manifold's *native* channels: triangles are
reordered into contiguous runs by part, and the rebuilt Mesh carries explicit
`run_index` / `run_original_id`. So a bevelled solid survives a further boolean
with every triangle still attributable.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

import manifold3d as m3

from .solid import LatheError, Selection, Solid

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bevel_worker.py")

_CANDIDATES = [
    r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
    "/Applications/Blender.app/Contents/MacOS/Blender",
    "/usr/bin/blender",
]


# A delivered width this close to the request counts as delivered. Exact results
# measure to ~1e-7, but a face bmesh merged across up to coplanar_deg (1 degree)
# is not planar, so its outline is tilted against the edge and reads up to ~0.3%
# off. A clamp that matters is far above this: the smallest seen is 1%.
WIDTH_TOLERANCE = 5e-3


@dataclass(frozen=True)
class BevelReport:
    """What bevel() actually did, as opposed to what it was asked.

    `width` is a ceiling when clamp is on: bmesh shrinks the bevel where it fears
    overlap, by an amount that depends on incidental geometry, and says nothing.
    Measured on curved input an 8 mm request delivered 3.4 to 7.6 mm and the
    Solid came back valid, correctly tagged and 41% off in bevel area.
    DESIGN.md section 14.7.

    An edge the measurement could not read is neither delivered nor short, and
    counting it as either is how the first version reported 99.94 mm delivered
    for a 5 mm request. It is `unmeasured`, and a report with any is not `ok`.
    """

    requested: float          # metres
    edges: int                # edges the predicate selected
    short: int                # measured edges that got less than `requested`
    min_delivered: float      # metres; worst measured edge (nan if none)
    max_delivered: float      # metres; best measured edge (nan if none)
    clamp: bool
    unmeasured: int = 0       # edges whose delivered width could not be read

    @property
    def clamped(self) -> bool:
        return self.short > 0

    @property
    def ok(self) -> bool:
        """Every edge measured, and every one delivered in full."""
        return self.short == 0 and self.unmeasured == 0

    def describe(self) -> str:
        mm = lambda v: "%.3g mm" % (v * 1000.0)
        measured = self.edges - self.unmeasured
        if self.ok:
            return ("bevel: %s requested, delivered on all %d edges"
                    % (mm(self.requested), self.edges))
        parts = []
        if self.short:
            lo = 100.0 * self.min_delivered / self.requested if self.requested else 0.0
            parts.append("%d of %d edges came out short, the worst at %s (%.0f%%)"
                         % (self.short, self.edges, mm(self.min_delivered), lo))
        elif measured:
            parts.append("delivered on the %d edges it could measure" % measured)
        if self.unmeasured:
            parts.append("%d of %d edges could not be measured"
                         % (self.unmeasured, self.edges))
        return "bevel: %s requested; %s" % (mm(self.requested), "; ".join(parts))

    def advice(self) -> str:
        out = []
        if self.clamped:
            if self.clamp:
                out.append(
                    "bmesh clamps the width where it fears overlap, by an amount "
                    "that depends on incidental geometry. clamp=False delivers the "
                    "width exactly and raises if the result is not a valid "
                    "manifold; a smaller width is the other way out.")
            else:
                out.append(
                    "clamp was off, so a short edge here is not bmesh's clamp: the "
                    "geometry could not take the width. Try a smaller one.")
        if self.unmeasured:
            out.append(
                "An unmeasured edge sits beside a face too small or too noisy to "
                "find the bevel boundary on. That is not a pass: check the width "
                "by eye, or bevel with a width the smallest face can take.")
        return " ".join(out)


_SINK: "contextvars.ContextVar[Optional[List[BevelReport]]]" = contextvars.ContextVar(
    "lathe_bevel_reports", default=None)


@contextlib.contextmanager
def collecting_reports():
    """Gather the BevelReport of every bevel() called inside the block.

    The result Solid of a program is usually *not* the bevel's own, since a
    boolean or a transform comes after, so a report that only rode on the
    returned Solid would be gone by the time anyone could read it. A context
    variable, not a global: lathe_apply runs programs on worker threads, and
    asyncio.to_thread copies the context.
    """
    sink: List[BevelReport] = []
    token = _SINK.set(sink)
    try:
        yield sink
    finally:
        _SINK.reset(token)


def _recentre_origin(verts: np.ndarray) -> np.ndarray:
    """The point to subtract before the mesh goes to Blender's float32.

    Zero for anything near the origin, so ordinary models are untouched: a face
    on z = 0 stays exactly there, which origin checks and exact-comparison tests
    rely on. Only a model far from the origin, where float32 has already lost the
    bevel, is moved, and then by a multiple of a power of two sized to the model,
    so that subtracting it and adding it back is exact in float64.
    """
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    centre = 0.5 * (lo + hi)
    extent = max(float(np.max(hi - lo)), 1e-9)
    if float(np.max(np.abs(centre))) <= 4.0 * extent:
        return np.zeros(3)
    quantum = 2.0 ** np.floor(np.log2(extent / 1024.0))
    return np.round(centre / quantum) * quantum


def _noise_floor(maxabs: float) -> float:
    """How far a vertex can sit from where it should, from float32 alone.

    Blender stores coordinates as float32 (half an ulp is 2**-24 relative) and the
    bevel arithmetic compounds that, so 64 half-ulps of the largest coordinate
    is allowed. The caller recentres the mesh first, so `maxabs` is the model's
    size, not its distance from the origin.
    """
    return 64.0 * 2.0 ** -24 * max(float(maxabs), 1e-9)


def delivered_widths(edges: np.ndarray, side_verts: np.ndarray,
                     side_off: np.ndarray, width: float) -> np.ndarray:
    """Width delivered along each picked edge.

    `edges` is the worker's (m, 6, 3): for each picked edge its two endpoints,
    then (face normal, direction into the face) for each adjacent face, all
    measured before the bevel. `side_verts` and `side_off` are, for each edge's
    two sides in order, the vertices of the polygon that face became, in loop
    order.

    A bevel shrinks each adjacent face to a smaller polygon. Along the old edge
    its outline is a segment *parallel* to the edge, `delivered` away: that
    offset is the width on that side, and an edge's width is the smaller of its
    two sides.

    Parallel segments, because the outline is not only that. At a corner the
    face's outline follows the vertex patch's arc, which comes within half a
    width of the edge's end: the nearest outline point read the chair's exact
    0.99 mm bevel as 0.41 mm. Overlapping the edge's extent, because where the
    face turns away from the edge the boundary line runs past its ends. A face
    whose edge is shorter than the bevel has no such segment, only the corner
    where its neighbours' lines meet; that vertex is the fallback.

    A side whose polygon is gone, or has nothing in range, is NaN: not zero and
    not the request. An unmeasured edge is its own state (see BevelReport),
    because reading 0.05 mm as "99.94 mm delivered" is the failure this function
    exists to prevent.

    Linear in the mesh: per edge the cost is the size of two polygons. The first
    version tested every output vertex against every edge, 52 s at 32k vertices.

    Returns metres, one per edge, NaN where the width could not be read.
    """
    m = len(edges)
    out = np.full(m, np.nan, dtype=np.float64)
    if m == 0:
        return out
    sv = np.asarray(side_verts, dtype=np.float64).reshape(-1, 3)
    off = np.asarray(side_off, dtype=np.int64)
    maxabs = float(np.abs(edges[:, :2]).max())
    if len(sv):
        maxabs = max(maxabs, float(np.abs(sv).max()))
    noise = _noise_floor(maxabs)
    cap = float(width) * (1.0 + WIDTH_TOLERANCE) + noise
    floor = 4.0 * noise
    for e in range(m):
        a, b = edges[e, 0], edges[e, 1]
        d = b - a
        length = float(np.linalg.norm(d))
        if length <= 1e-12:
            continue
        dh = d / length
        sides = []
        for k in (0, 1):
            vs = sv[off[2 * e + k]:off[2 * e + k + 1]]
            if len(vs) < 2:
                break
            u = edges[e, 3 + 2 * k]
            rel = vs - a
            t, s = rel @ dh, rel @ u
            t2, s2 = np.roll(t, -1), np.roll(s, -1)
            seg = np.roll(vs, -1, axis=0) - vs
            seg_len = np.linalg.norm(seg, axis=1)
            # Parallel to the edge. Tight on a planar face. A face bmesh merged
            # across up to coplanar_deg is not planar, and its boundary line is
            # tilted against the edge by up to about a degree: it gets the slack,
            # and only it, so a planar face cannot lend a stray segment.
            n_k = edges[e, 2 + 2 * k]
            size = float(np.linalg.norm(vs.max(axis=0) - vs.min(axis=0)))
            planar = float(np.abs(rel @ n_k).max()) <= 1e-4 * size + 8.0 * noise
            slack = 1e-4 if planar else 0.03
            cross = np.linalg.norm(np.cross(seg, dh), axis=1)
            parallel = (seg_len > floor) & (cross <= slack * seg_len + 8.0 * noise)
            lo_t, hi_t = np.minimum(t, t2), np.maximum(t, t2)
            reach = np.minimum(hi_t, length) - np.maximum(lo_t, 0.0)
            overlaps = parallel & (reach > -noise)
            if overlaps.any():
                # The offset where the segment crosses the middle of the edge.
                tm = np.clip(0.5 * length, lo_t, hi_t)
                span = t2 - t
                moving = np.abs(span) > 1e-18
                frac = np.where(moving, (tm - t) / np.where(moving, span, 1.0), 0.0)
                soff = s + frac * (s2 - s)
                ok = overlaps & (soff > floor) & (soff <= cap)
                if ok.any():
                    sides.append(float(soff[ok].min()))
                    continue
            near = (t >= -noise) & (t <= length + noise) & (s > floor) & (s <= cap)
            if near.any():
                sides.append(float(s[near].min()))
                continue
            break
        if len(sides) == 2:
            out[e] = min(sides)
    return out


def blender_path() -> Optional[str]:
    env = os.environ.get("LATHE_BLENDER")
    if env and os.path.exists(env):
        return env
    found = shutil.which("blender")
    if found:
        return found
    for c in _CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def bevel_available() -> bool:
    return blender_path() is not None


def bevel(solid: Solid, width: float, segments: int = 2, min_dihedral: float = 30.0,
          max_dihedral: float = 180.0, min_length: float = 0.0,
          only_tags: Optional[Sequence[str]] = None, new_tag: str = "bevel",
          profile: float = 0.5, clamp: bool = True, coplanar_deg: float = 1.0,
          verbose: bool = False, edges: Optional[Selection] = None) -> Solid:
    """Round the edges matching the predicate. Returns a new Solid.

    only_tags restricts the bevel to edges whose *both* faces carry one of those
    tags, which is how you bevel a seat without touching the legs.

    `edges` is an edge Selection taken from this same solid, and then those edges
    are exactly the ones beveled: the dihedral, length and tag predicates are the
    selection's own and are not applied again. Without it the predicates here pick.
    (The .lathe language used to recover only a selection's dihedral threshold, so
    `bevel(carved, rim)` beveled every edge over it, whatever else `rim` said.)

    `width` is the distance from the edge along each adjacent face. With
    clamp=True, the default, it is a *ceiling*, not a promise: bmesh shrinks the
    bevel where it fears overlap, by an amount that depends on incidental
    geometry. Measured on curved input an 8 mm request delivered 3.4 to 7.6 mm.
    The returned Solid's `bevel_report` says what was delivered (and on how many
    edges that could not be read), and lathe_apply prints it when it is not ok. clamp=False delivers the width exactly and
    raises if the result is not a valid manifold, but cannot promise the surface
    does not fold over itself. DESIGN.md section 14.7.
    """
    # Arguments before the environment, and before the work. A width of zero, a
    # negative one or NaN used to run, change nothing, and be reported as
    # delivered; a non-finite width reached bmesh itself.
    if not (isinstance(width, (int, float)) and np.isfinite(width) and width > 0):
        raise LatheError(
            "bevel width must be a positive, finite length in metres, got %r. "
            "`mm(2)` is 2 mm." % (width,))
    if int(segments) < 1:
        raise LatheError("bevel segments must be at least 1, got %r" % (segments,))

    exe = blender_path()
    if exe is None:
        raise LatheError(
            "bevel needs Blender. Set LATHE_BLENDER to blender.exe, or put it on PATH.")

    msh = solid.mesh()
    if msh.part_id is None:
        raise LatheError("bevel needs part ids; %s has none" % solid.name)

    only_pairs = None
    if edges is not None:
        if getattr(edges, "domain", None) != "edges":
            raise LatheError("bevel(edges=) needs an edge selection, got %r"
                             % (edges,))
        src = edges.solid.mesh()
        if (src.verts.shape != msh.verts.shape or src.tris.shape != msh.tris.shape
                or not np.array_equal(src.verts, msh.verts)
                or not np.array_equal(src.tris, msh.tris)):
            raise LatheError(
                "the edge selection %s belongs to a different solid than the one "
                "being beveled (%s). Select the edges on the solid you bevel."
                % (edges.expr, solid.name))
        if len(edges.indices) == 0:
            raise LatheError(
                "bevel selected no edges on %s: the selection %s is empty. Loosen "
                "its predicates." % (solid.name, edges.expr))
        only_pairs = np.asarray(src.topology.edges, dtype=np.int64)[edges.indices]

    only_parts = None
    if only_tags is not None:
        wanted = set(only_tags)
        only_parts = sorted(i for i, t in solid.tags.items() if t in wanted)
        if not only_parts:
            raise LatheError(
                "only_tags=%r matched no tag on %s. Present: %s"
                % (list(only_tags), solid.name, sorted(set(solid.tags.values()))))

    bevel_id = (max(solid.tags) if solid.tags else 0) + 1000
    while bevel_id in solid.tags:
        bevel_id += 1

    tmp = tempfile.mkdtemp(prefix="lathe_bevel_")
    try:
        in_p = os.path.join(tmp, "in.npz")
        out_p = os.path.join(tmp, "out.npz")
        par_p = os.path.join(tmp, "params.json")
        # Blender holds float32. A model 1 km from the origin has 0.1 mm of
        # coordinate noise before anything happens, and 100 km loses the bevel
        # altogether (volume -20%, reported delivered). So the worker sees the
        # model centred, and the result is put back in float64.
        origin = _recentre_origin(msh.verts)
        arrays = dict(verts=msh.verts - origin, tris=msh.tris,
                      part_id=msh.part_id)
        if only_pairs is not None:
            arrays["only_pairs"] = only_pairs
        np.savez(in_p, **arrays)
        json.dump({
            "width": float(width), "segments": int(segments),
            "min_dihedral": float(min_dihedral), "max_dihedral": float(max_dihedral),
            "min_length": float(min_length), "only_parts": only_parts,
            "bevel_id": int(bevel_id), "profile": float(profile),
            "clamp": bool(clamp), "coplanar_deg": float(coplanar_deg),
        }, open(par_p, "w"))

        proc = subprocess.run(
            [exe, "-b", "--factory-startup", "-noaudio", "-P", _WORKER,
             "--", in_p, out_p, par_p],
            capture_output=True, text=True, timeout=600,
            # stdin=DEVNULL is not optional. capture_output only redirects
            # stdout and stderr, so a child would inherit *our* stdin -- and
            # under the MCP server that is the JSON-RPC pipe. Blender then holds
            # the transport until its timeout, and may eat protocol bytes.
            stdin=subprocess.DEVNULL)
        if not os.path.exists(out_p):
            raise LatheError(
                "bevel worker produced no output.\n--- stdout ---\n%s\n--- stderr ---\n%s"
                % (proc.stdout[-3000:], proc.stderr[-3000:]))

        res = np.load(out_p)
        report = json.loads(str(res["report"][0]))
        if verbose:
            print("  bevel: %s" % report)
        if report.get("status") == "no_edges_selected":
            raise LatheError(
                "bevel selected no edges on %s.\n"
                "  predicate : dihedral %g..%g deg, length > %g, tags %s\n"
                "  mesh      : %d tris, %d logical faces after coplanar merge\n"
                "  Loosen min_dihedral, or check the tag names."
                % (solid.name, min_dihedral, max_dihedral, min_length,
                   list(only_tags) if only_tags else "any",
                   report["in_tris"], report.get("faces_after_dissolve", -1)))

        centred = np.asarray(res["verts"], dtype=np.float64)
        man = _tagged_manifold(centred + origin, res["tris"], res["part_id"])
        tags = dict(solid.tags)
        tags[int(bevel_id)] = new_tag
        out = Solid(man, tags, solid.name)
        rep = None
        if "picked_edges" in res.files and len(res["picked_edges"]):
            got = delivered_widths(res["picked_edges"], res["side_verts"],
                                   res["side_off"], float(width))
            seen = np.isfinite(got)
            short = seen & (got < float(width) * (1.0 - WIDTH_TOLERANCE))
            rep = BevelReport(
                requested=float(width), edges=int(len(got)),
                short=int(short.sum()),
                min_delivered=float(got[seen].min()) if seen.any() else float("nan"),
                max_delivered=float(got[seen].max()) if seen.any() else float("nan"),
                clamp=bool(clamp), unmeasured=int((~seen).sum()))
            object.__setattr__(out, "bevel_report", rep)
            sink = _SINK.get()
            if sink is not None:
                sink.append(rep)
        # A bevel removes a sliver along each edge. One that shrank the bounding
        # box by a quarter, or left under a quarter of the volume, has not
        # rounded anything: it has collapsed the surface, and nothing else would
        # say so. A dome revolved from a finely sampled arc came back a flat disc
        # (16 mm tall to 4 mm), "delivered on all 96 edges", verifier PASS. The
        # volume alone cannot tell it from a legitimate maximum rounding (a cube
        # with an oversize width is 41% and fine), so the bounding box leads.
        v0, v1 = float(solid.volume()), float(out.volume())
        e0 = np.array(solid.bbox()[1]) - np.array(solid.bbox()[0])
        e1 = np.array(out.bbox()[1]) - np.array(out.bbox()[0])
        if v0 > 0.0 and (v1 < 0.25 * v0 or np.any(e1 < 0.75 * e0 - 1e-12)):
            raise LatheError(
                "bevel on %s collapsed the solid: %.0f%% of its volume is left and "
                "its bounding box went from %s to %s mm. A width of %g mm is too "
                "large for the smallest feature, or the surface is tessellated "
                "finer than coplanar_deg (%g) and was merged into one face. "
                "Narrow the width, or lower coplanar_deg."
                % (solid.name, 100.0 * v1 / v0,
                   " x ".join("%.1f" % (1000 * x) for x in e0),
                   " x ".join("%.1f" % (1000 * x) for x in e1),
                   width * 1000.0, coplanar_deg))
        if not out.status_ok():
            raise LatheError(
                "bevel on %s produced %s. Width %g is probably too large for the "
                "smallest feature; try clamp=True or a smaller width."
                % (solid.name, man.status(), width))
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _tagged_manifold(verts, tris, part_id):
    """Rebuild a Manifold whose native run_original_id carries our tags.

    Manifold groups triangles into *runs*, one id per run, so the triangles must
    be reordered to make each part contiguous. Without this the rebuilt solid
    would come back as one anonymous blob and every tag-filtered selector would
    stop working after a bevel.
    """
    verts = np.array(verts, dtype=np.float64, order="C", copy=True)
    tris = np.array(tris, dtype=np.int64, order="C", copy=True)
    part_id = np.asarray(part_id, dtype=np.int64).reshape(-1)

    order = np.argsort(part_id, kind="stable")
    tris = tris[order]
    parts = part_id[order]
    _, starts = np.unique(parts, return_index=True)
    starts = np.sort(starts)
    run_index = np.append(starts * 3, len(tris) * 3)
    run_ids = parts[starts]

    # Mesh64, not Mesh: the float32 Mesh would cap every downstream probe at
    # ~1e-7 relative accuracy, which is the section 11 to_mesh64 note in reverse.
    mesh = m3.Mesh64(
        vert_properties=np.array(verts, dtype=np.float64, order="C", copy=True),
        tri_verts=np.array(tris, dtype=np.uint64, order="C", copy=True),
        run_index=np.array(run_index, dtype=np.uint64, order="C", copy=True),
        run_original_id=np.array(run_ids, dtype=np.uint64, order="C", copy=True),
    )
    return m3.Manifold(mesh)
