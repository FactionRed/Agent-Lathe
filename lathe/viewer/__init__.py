"""
Interactive preview -- a self-contained WebGL viewer for a Solid or a Mesh.

Static contact sheets answer "is the proportion right". They do not answer "what
does the back of that joint look like", and a still frame cannot be turned over.
This writes one HTML file with the mesh embedded and a small renderer inline.

Hand-rolled WebGL, deliberately. The page has to work under a strict CSP that
blocks every external host, so a CDN import is impossible, and inlining three.js
would mean shipping ~600 KB of minified library in every preview. A Lambert-plus-
Blinn shader with an orbit camera is about 300 lines and has no supply chain.

What the viewer shows:
  * full orbit -- drag to tumble, wheel to zoom, right or shift drag to pan,
    double-click to recentre, plus six preset views
  * adjustable lighting -- key intensity, azimuth and height, fill, rim, ambient
    and gloss, with five rig presets
  * vertex and polygon statistics, including logical faces, components, genus and
    a per-tag triangle breakdown driven by section 5.2 provenance

It takes a neutral Mesh as well as a Solid, so an imported .glb can be looked at
whether or not it is watertight. That is the case that matters most: geometry you
authored, you already understand, and an open shell is the *normal* state of an
asset from somewhere else.
"""

from __future__ import annotations

import base64
import colorsys
import json
import os
from typing import Dict, Optional

import numpy as np

from ..build.solid import Solid
from ..io.export import _PALETTE
from ..perceive import mesh as pmesh

_SHELL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shell.html")

# Shown when a tag has no material assigned. Spread around the hue circle so
# adjacent parts stay distinguishable rather than all landing on grey.
_FALLBACK_HUES = (0.58, 0.09, 0.33, 0.78, 0.15, 0.47, 0.68, 0.02, 0.25, 0.88)


def _b64(arr: np.ndarray, dtype) -> str:
    return base64.b64encode(
        np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode("ascii")


def _material_colours(tags, materials: Dict[str, str]) -> list:
    """How the asset is meant to look: oak, ceramic, leather."""
    out = []
    for i, tag in enumerate(tags):
        key = materials.get(tag)
        if key and key in _PALETTE:
            out.append([round(c, 4) for c in _PALETTE[key]])
            continue
        h = _FALLBACK_HUES[i % len(_FALLBACK_HUES)]
        r, g, b = colorsys.hsv_to_rgb(h, 0.42, 0.78)
        out.append([round(r, 4), round(g, 4), round(b, 4)])
    return out


def _identity_colours(tags) -> list:
    """One saturated hue per tag, for *reading provenance* rather than looks.

    Two wooden parts share a material colour and are then impossible to tell
    apart, which defeats the point of section 5.2 tags being visible at all.
    """
    n = max(len(tags), 1)
    out = []
    for i in range(len(tags)):
        # Golden-ratio hue stepping keeps neighbours far apart at any count.
        h = (i * 0.6180339887) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.62, 0.92)
        out.append([round(r, 4), round(g, 4), round(b, 4)])
    return out


def _wire_edges(msh: pmesh.Mesh, min_dihedral: float = 12.0) -> np.ndarray:
    """Line pairs for the wireframe overlay.

    Only edges where the surface actually creases. Drawing every triangulation
    edge would bury the shape under diagonals that are not features -- the same
    reason the bevel worker merges coplanar faces before selecting.
    """
    topo = msh.topology
    verts, tris = msh.verts, msh.tris
    if topo.n_edges == 0:
        return np.zeros((0, 3), dtype=np.float32)

    nrm = pmesh.tri_normals(msh)
    # Map each edge to the triangles that use it.
    tri_of = np.repeat(np.arange(len(tris)), 3)
    flat = topo.tri_edges.reshape(-1)
    order = np.argsort(flat, kind="stable")
    e_sorted, t_sorted = flat[order], tri_of[order]
    starts = np.searchsorted(e_sorted, np.arange(topo.n_edges))
    counts = topo.edge_tri_count

    keep = []
    cos_lim = np.cos(np.radians(min_dihedral))
    for e in range(topo.n_edges):
        if counts[e] == 2:
            a, b = t_sorted[starts[e]], t_sorted[starts[e] + 1]
            if float(np.dot(nrm[a], nrm[b])) < cos_lim:
                keep.append(e)
        elif counts[e] == 1:
            keep.append(e)          # a boundary edge is always a feature
    if not keep:
        return np.zeros((0, 3), dtype=np.float32)
    pairs = topo.edges[np.asarray(keep, dtype=np.int64)]
    return verts[pairs.reshape(-1)].astype(np.float32)


def _subject(subject) -> tuple:
    """(mesh, {part id: tag}, name). Accepts a Solid or a neutral Mesh.

    A Mesh is accepted because an imported .glb is generally *not* a solid -- an
    open shell is the normal case for scanned and sculpted assets -- and being
    unable to look at the one kind of geometry you did not author would make the
    viewer useless exactly when it is most needed.
    """
    if isinstance(subject, Solid):
        return subject.mesh(), dict(subject.tags), subject.name
    if isinstance(subject, pmesh.Mesh):
        return subject, dict(subject.parts or {}), subject.name
    raise TypeError("expected a Solid or a perceive Mesh, got %s"
                    % type(subject).__name__)


def _stats(msh: pmesh.Mesh) -> dict:
    """Read every number off the mesh, not off the kernel.

    The kernel is not available for an imported mesh, and a viewer that needed it
    could not show the geometry most in need of looking at. Tier 1 already
    computes all of this closed-form, including saying `n/a` for the volume of an
    open surface rather than printing a confident wrong number.
    """
    from ..perceive import tier1
    counts = tier1.counts(msh).fields
    vol = tier1.volume(msh).fields
    lo, hi = pmesh.bounds(msh)
    return {
        "verts": int(counts["verts"]),
        "tris": int(counts["tris"]),
        "edges": int(counts["edges"]),
        "faces": int(counts["faces_logical"]),
        "components": int(counts["components"]),
        "genus": counts["genus"] if counts["genus"] is not None else "n/a",
        "watertight": bool(vol["closed"]),
        "size": "%.0f x %.0f x %.0f mm" % tuple((h - l) * 1000 for l, h in zip(lo, hi)),
        "volume": ("%.4f m3" % vol["volume_m3"] if vol["closed"]
                   else "n/a (open surface)"),
        "area": "%.4f m2" % tier1.surface_area(msh).fields["area_m2"],
        "boundary_edges": int(counts["boundary_edges"]),
        "nonmanifold_edges": int(counts["nonmanifold_edges"]),
    }


def to_html(subject, materials: Optional[Dict[str, str]] = None,
            title: Optional[str] = None, wire_angle: float = 12.0) -> str:
    """Render a Solid or a neutral Mesh into a standalone HTML document."""
    msh, part_tags, subject_name = _subject(subject)
    materials = dict(materials or {})

    tags = sorted(set(part_tags.values())) or ["mesh"]
    tag_index = {t: i for i, t in enumerate(tags)}
    per_tri = np.zeros(msh.n_tris, dtype=np.float32)
    if msh.part_id is not None and part_tags:
        lut = {pid: tag_index.get(tag, 0) for pid, tag in part_tags.items()}
        per_tri = np.array([lut.get(int(p), 0) for p in msh.part_id],
                           dtype=np.float32)

    # Un-index so every triangle carries its own face normal: flat shading is
    # honest about faceting, and a smoothed normal would hide tessellation.
    corners = msh.verts[msh.tris.reshape(-1)]
    normals = np.repeat(pmesh.tri_normals(msh), 3, axis=0)
    tag_attr = np.repeat(per_tri, 3)

    lo, hi = pmesh.bounds(msh)
    counts = np.bincount(per_tri.astype(np.int64), minlength=len(tags))

    data = {
        "pos": _b64(corners, np.float32),
        "nrm": _b64(normals, np.float32),
        "tag": _b64(tag_attr, np.float32),
        "edge": _b64(_wire_edges(msh, wire_angle), np.float32),
        "tags": tags,
        "cols": _material_colours(tags, materials),
        "tagCols": _identity_colours(tags),
        "tagTris": [int(c) for c in counts],
        "bbox": {
            "center": [round((a + b) / 2, 6) for a, b in zip(lo, hi)],
            "size": [round(b - a, 6) for a, b in zip(lo, hi)],
        },
        "stats": _stats(msh),
    }

    shell = open(_SHELL, "r", encoding="utf-8").read()
    name = title or subject_name
    return (shell
            .replace("__TITLE__", _escape(name))
            .replace("__NAME__", _escape(name))
            .replace("__DATA__", _json_for_html(data)))


def _json_for_html(data: dict) -> str:
    """JSON safe to drop inside a <script> element.

    `json.dumps` does not escape `<`, so a tag named `</script><script>` would
    close the data block and run as markup. Tag names used to come only from the
    program the agent wrote; since import they can come from a stranger's .glb,
    so the payload is escaped here as well as sanitised at the reader. `\\u003c`
    is an ordinary JSON escape, so the parsed value is unchanged.
    """
    return (json.dumps(data, separators=(",", ":"))
            .replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("&", "\\u0026"))


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def write(subject, path: str, materials: Optional[Dict[str, str]] = None,
          title: Optional[str] = None) -> dict:
    """Write the preview to `path`. Returns a small summary."""
    html = to_html(subject, materials, title)
    msh = _subject(subject)[0]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    # newline="" so the file is byte-identical to the string. Without it, Windows
    # rewrites every \n as \r\n and the file no longer matches what to_html
    # returned -- which is also what an Artifact or a browser would receive.
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(html)
    return {
        "path": os.path.abspath(path),
        "bytes": os.path.getsize(path),
        "tris": int(msh.n_tris),
        "self_contained": True,
    }
