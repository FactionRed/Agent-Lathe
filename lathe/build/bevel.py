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

import json
import os
import shutil
import subprocess
import tempfile
from typing import Optional, Sequence

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
          verbose: bool = False) -> Solid:
    """Round the edges matching the predicate. Returns a new Solid.

    only_tags restricts the bevel to edges whose *both* faces carry one of those
    tags, which is how you bevel a seat without touching the legs.

    `width` is the distance from the edge along each adjacent face. With
    clamp=True, the default, it is a *ceiling*, not a promise: bmesh shrinks the
    bevel where it fears overlap, by an amount that depends on incidental
    geometry, and the returned Solid does not say it did. Measured on curved
    input an 8 mm request delivered 3.4 to 7.6 mm. clamp=False delivers the
    width exactly and raises if the result is not a valid manifold, but cannot
    promise the surface does not fold over itself. DESIGN.md section 14.7.
    """
    exe = blender_path()
    if exe is None:
        raise LatheError(
            "bevel needs Blender. Set LATHE_BLENDER to blender.exe, or put it on PATH.")

    msh = solid.mesh()
    if msh.part_id is None:
        raise LatheError("bevel needs part ids; %s has none" % solid.name)

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
        np.savez(in_p, verts=msh.verts, tris=msh.tris, part_id=msh.part_id)
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

        man = _tagged_manifold(res["verts"], res["tris"], res["part_id"])
        tags = dict(solid.tags)
        tags[int(bevel_id)] = new_tag
        out = Solid(man, tags, solid.name)
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
