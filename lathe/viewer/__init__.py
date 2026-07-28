"""
Interactive preview -- a self-contained WebGL viewer for a Solid.

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


def _stats(solid: Solid, msh: pmesh.Mesh) -> dict:
    from ..perceive import tier1
    topo = msh.topology
    lo, hi = solid.bbox()
    faces = (len(set(zip(msh.part_id.tolist(), msh.face_id.tolist())))
             if msh.part_id is not None and msh.face_id is not None
             else int(msh.n_tris))
    wt = tier1.is_watertight(msh)
    return {
        "verts": int(msh.n_verts),
        "tris": int(msh.n_tris),
        "edges": int(topo.n_edges),
        "faces": faces,
        "components": int(solid.components()),
        "genus": int(solid.man.genus()),
        "watertight": bool(getattr(wt, "value", None) is True or "true" in str(wt).lower()),
        "size": "%.0f x %.0f x %.0f mm" % tuple((h - l) * 1000 for l, h in zip(lo, hi)),
        "volume": "%.4f m3" % solid.volume(),
        "area": "%.4f m2" % solid.man.surface_area(),
    }


def to_html(solid: Solid, materials: Optional[Dict[str, str]] = None,
            title: Optional[str] = None, wire_angle: float = 12.0) -> str:
    """Render `solid` into a standalone HTML document."""
    msh = solid.mesh()
    materials = dict(materials or {})

    tags = sorted(set(solid.tags.values())) or ["mesh"]
    tag_index = {t: i for i, t in enumerate(tags)}
    per_tri = np.zeros(msh.n_tris, dtype=np.float32)
    if msh.part_id is not None:
        lut = {pid: tag_index.get(tag, 0) for pid, tag in solid.tags.items()}
        per_tri = np.array([lut.get(int(p), 0) for p in msh.part_id],
                           dtype=np.float32)

    # Un-index so every triangle carries its own face normal: flat shading is
    # honest about faceting, and a smoothed normal would hide tessellation.
    corners = msh.verts[msh.tris.reshape(-1)]
    normals = np.repeat(pmesh.tri_normals(msh), 3, axis=0)
    tag_attr = np.repeat(per_tri, 3)

    lo, hi = solid.bbox()
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
        "stats": _stats(solid, msh),
    }

    shell = open(_SHELL, "r", encoding="utf-8").read()
    name = title or solid.name
    return (shell
            .replace("__TITLE__", _escape(name))
            .replace("__NAME__", _escape(name))
            .replace("__DATA__", json.dumps(data, separators=(",", ":"))))


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def write(solid: Solid, path: str, materials: Optional[Dict[str, str]] = None,
          title: Optional[str] = None) -> dict:
    """Write the preview to `path`. Returns a small summary."""
    html = to_html(solid, materials, title)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    # newline="" so the file is byte-identical to the string. Without it, Windows
    # rewrites every \n as \r\n and the file no longer matches what to_html
    # returned -- which is also what an Artifact or a browser would receive.
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(html)
    return {
        "path": os.path.abspath(path),
        "bytes": os.path.getsize(path),
        "tris": int(solid.n_tris),
        "self_contained": True,
    }
