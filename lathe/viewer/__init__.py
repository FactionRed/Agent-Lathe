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
  * given a `pose.Rig`, a slider per joint bounded by that joint's declared
    limit, live articulation, and the resulting pose as copyable program text

**Why a read-only viewer grew a control.** Section 11 put a review surface first
and "direct manipulation that writes back to the IR" later, and section 10 says
what that has to look like:

    When a human adjusts something in the viewer, the app **writes back a
    parameter or constraint edit**, not an opaque mesh mutation ... Human
    correction becomes machine-legible input. That feedback loop is the actual
    product.

So the page never hands back geometry. It hands back a handful of named, typed
angles -- text a human can read, the history DAG can diff, and the agent can act
on. A dragged vertex would be none of those things.

Poses specifically, and not yet geometry parameters, because a pose is the one
edit this page can actually *show*: articulating a rig is rigid transforms of
parts already in the payload, so it costs some linear algebra and no rebuild.
Changing `seat_h` needs Manifold and a re-evaluation, which a file with no
network cannot do. Offering a slider that silently failed to preview would be
worse than not offering it.

**The page checks itself.** Forward kinematics is implemented twice -- once in
`lathe/pose/skeleton.py` and once in JavaScript here -- and two implementations
drift. A flipped axis convention would be invisible in a still frame and wrong
in every pose, which is the failure this whole system exists to prevent. So the
payload carries Python's own rest positions and the page recomputes them on
load; if they disagree it refuses to open and says which joint and by how much.

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


def _per_tri_tags(msh: pmesh.Mesh, part_tags: Dict[int, str], tag_index) -> np.ndarray:
    """Tag index per triangle, via section 5.2 provenance."""
    if msh.part_id is None or not part_tags:
        return np.zeros(msh.n_tris, dtype=np.float32)
    lut = {pid: tag_index.get(tag, 0) for pid, tag in part_tags.items()}
    return np.array([lut.get(int(p), 0) for p in msh.part_id], dtype=np.float32)


def _as_rig(subject):
    """The subject as a `pose.Rig`, or None.

    Imported lazily and by duck type so the viewer keeps working if `lathe.pose`
    is ever absent -- the viewer's job is to show geometry, and it should not
    stop being able to do that because an unrelated subsystem failed to import.
    """
    try:
        from ..pose import Rig
    except Exception:  # pragma: no cover - defensive, pose has no hard deps
        return None
    return subject if isinstance(subject, Rig) else None


def _rig_geometry(rig) -> tuple:
    """Per-joint geometry for a `pose.Rig`, kept separable.

    The parts are concatenated rather than booleaned. `Rig.posed()` unions, which
    is right for an asset and wrong for a viewer: a union welds the parts into
    one vertex set, and a hinge cannot then move one of them without the others.
    Every corner instead carries the index of the joint that drives it, which is
    one-bone-per-vertex rigid skinning and exactly what a hinge needs.

    Returns (corners, normals, tag_attr, joint_attr, edges, edge_joints, tags).
    """
    names = list(rig.bound)
    tags = sorted({t for n in names for t in rig.bindings[n].tags.values()}) or ["mesh"]
    tag_index = {t: i for i, t in enumerate(tags)}

    corners, normals, tag_attr, joint_attr = [], [], [], []
    edges, edge_joints = [], []
    for j, name in enumerate(names):
        msh = rig.bindings[name].mesh()
        part_tags = dict(rig.bindings[name].tags)
        per_tri = np.zeros(msh.n_tris, dtype=np.float32)
        if msh.part_id is not None and part_tags:
            lut = {pid: tag_index.get(tag, 0) for pid, tag in part_tags.items()}
            per_tri = np.array([lut.get(int(p), 0) for p in msh.part_id],
                               dtype=np.float32)
        corners.append(msh.verts[msh.tris.reshape(-1)])
        normals.append(np.repeat(pmesh.tri_normals(msh), 3, axis=0))
        tag_attr.append(np.repeat(per_tri, 3))
        joint_attr.append(np.full(msh.n_tris * 3, j, dtype=np.float32))
        e = _wire_edges(msh)
        edges.append(e)
        edge_joints.append(np.full(len(e), j, dtype=np.float32))

    cat = lambda xs, w: (np.concatenate(xs) if xs else
                         np.zeros((0, w) if w else 0, dtype=np.float32))
    return (cat(corners, 3), cat(normals, 3), cat(tag_attr, 0),
            cat(joint_attr, 0), cat(edges, 3), cat(edge_joints, 0), tags)


def _rig_payload(rig, pose) -> dict:
    """The skeleton as data, plus the numbers that let the page check itself.

    `restWorld` is forward kinematics evaluated *in Python*, at rest. The page
    re-derives the same thing in JavaScript to drive the sliders, and the two
    implementations could drift -- a flipped axis convention or a different
    rotation order would be invisible in a still frame and wrong in every pose.
    So the page compares its own result against these numbers on load and says
    so if they disagree. A viewer that quietly disagrees with the kernel is
    worse than one that fails to open.
    """
    from ..pose import degrees, forward

    rest = forward(rig.skeleton, None)
    start = dict(rig.skeleton.rest())
    if pose:
        start.update({k: float(v) for k, v in pose.items()})

    joints = []
    for name in rig.skeleton.names:
        j = rig.skeleton.joint(name)
        joints.append({
            "name": name,
            "parent": j.parent,
            "offset": [round(float(v), 9) for v in j.offset],
            "axis": [round(float(v), 9) for v in j.unit_axis()],
            "limit": [round(degrees(j.limit[0]), 6), round(degrees(j.limit[1]), 6)],
            "locked": bool(j.locked),
            "bound": name in rig.bindings,
            # Two different baselines, deliberately not sharing a name.
            # `start` is the angle the page opens at -- the caller's pose, or
            # the skeleton's zero -- and it is what Reset returns to and what
            # the write-back diffs against, because the user wants to send the
            # change they made, not the change since the joint tree's zero.
            # `restWorld` is the structural rest, used only to check this page's
            # kinematics against Python's.
            "start": round(degrees(start[name]), 6),
            "restWorld": [round(float(v), 9) for v in rest[name][:3, 3]],
        })
    return {"name": rig.name, "joints": joints, "bound": list(rig.bound)}


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
            title: Optional[str] = None, wire_angle: float = 12.0,
            pose: Optional[Dict[str, float]] = None) -> str:
    """Render a Solid, a neutral Mesh, or a `pose.Rig` into standalone HTML.

    Given a Rig the page gains joint sliders and poses the model live, then
    hands back the pose as text to paste into the program. That is the section
    10 loop: the human adjusts something and the app "writes back a parameter or
    constraint edit, not an opaque mesh mutation".
    """
    rig = _as_rig(subject)
    materials = dict(materials or {})
    joint_attr = edge_joint = None

    if rig is not None:
        subject_name = rig.name
        # Statistics describe the *assembled* asset -- the boolean union, which
        # is what gets budgeted, verified and exported. The drawn geometry is
        # the parts kept separate so a hinge can move one of them, and a union
        # welds overlapping parts into a different triangle count. Counting the
        # HUD off the parts would make the panel disagree with the verifier, so
        # both numbers come from the union and only the vertices come from the
        # parts.
        assembled = rig.at_rest()
        msh = assembled.mesh()
        (corners, normals, tag_attr, joint_attr,
         edge_xyz, edge_joint, tags) = _rig_geometry(rig)
        tag_index = {t: i for i, t in enumerate(tags)}
        per_tri = _per_tri_tags(msh, dict(assembled.tags), tag_index)
    else:
        msh, part_tags, subject_name = _subject(subject)
        tags = sorted(set(part_tags.values())) or ["mesh"]
        tag_index = {t: i for i, t in enumerate(tags)}
        per_tri = _per_tri_tags(msh, part_tags, tag_index)
        # Un-index so every triangle carries its own face normal: flat shading is
        # honest about faceting, and a smoothed normal would hide tessellation.
        corners = msh.verts[msh.tris.reshape(-1)]
        normals = np.repeat(pmesh.tri_normals(msh), 3, axis=0)
        tag_attr = np.repeat(per_tri, 3)
        edge_xyz = _wire_edges(msh, wire_angle)

    lo, hi = pmesh.bounds(msh)
    counts = np.bincount(per_tri.astype(np.int64), minlength=len(tags))

    data = {
        "pos": _b64(corners, np.float32),
        "nrm": _b64(normals, np.float32),
        "tag": _b64(tag_attr, np.float32),
        "edge": _b64(edge_xyz, np.float32),
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
    if rig is not None:
        data["rig"] = _rig_payload(rig, pose)
        data["jnt"] = _b64(joint_attr, np.float32)
        data["edgeJnt"] = _b64(edge_joint, np.float32)

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
          title: Optional[str] = None,
          pose: Optional[Dict[str, float]] = None) -> dict:
    """Write the preview to `path`. Returns a small summary."""
    html = to_html(subject, materials, title, pose=pose)
    rig = _as_rig(subject)
    msh = rig.at_rest().mesh() if rig is not None else _subject(subject)[0]
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
