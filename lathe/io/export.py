"""
Export, with the hygiene DESIGN.md section 7 asks for.

OBJ and STL are written natively -- no dependencies, and the writer controls
exactly what lands on disk. glTF and FBX route through headless Blender, because
writing a correct binary glTF is not the interesting part of this project.

Every path applies the transform before writing. E104 exists because unapplied
scale is the single most common export bug, and an exporter that can emit it is
an exporter that will.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import Dict, Optional

import numpy as np

from ..build.bevel import blender_path
from ..build.solid import LatheError, Solid

NATIVE = ("obj", "stl")
VIA_BLENDER = ("glb", "gltf", "fbx", "usd", "usdc", "ply")
FORMATS = NATIVE + VIA_BLENDER

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_export_worker.py")


def export(solid: Solid, path: str, materials: Optional[Dict[str, str]] = None,
           origin: str = "as_built") -> dict:
    """Write `solid` to `path`. Format comes from the extension.

    origin:
      as_built      leave coordinates alone
      bottom_center recentre XY on the bounding box, drop Z to zero
      center        recentre on the bounding box in all three axes
    """
    fmt = os.path.splitext(path)[1].lstrip(".").lower()
    if fmt not in FORMATS:
        raise LatheError(
            "unknown export format %r. Native: %s. Via Blender: %s."
            % (fmt, ", ".join(NATIVE), ", ".join(VIA_BLENDER)))

    solid = _place(solid, origin)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    if fmt == "obj":
        info = _write_obj(solid, path, materials or {})
    elif fmt == "stl":
        info = _write_stl(solid, path)
    else:
        info = _via_blender(solid, path, fmt, materials or {})

    lo, hi = solid.bbox()
    info.update({
        "format": fmt,
        "path": os.path.abspath(path),
        "bytes": os.path.getsize(path) if os.path.exists(path) else 0,
        "tris": solid.n_tris,
        "origin": origin,
        "bbox_m": [[round(v, 6) for v in lo], [round(v, 6) for v in hi]],
        "transform_applied": True,
    })
    return info


def _place(solid: Solid, origin: str) -> Solid:
    if origin == "as_built":
        return solid
    lo, hi = solid.bbox()
    if origin == "bottom_center":
        return solid.at(x=-(lo[0] + hi[0]) / 2, y=-(lo[1] + hi[1]) / 2, z=-lo[2])
    if origin == "center":
        return solid.at(x=-(lo[0] + hi[0]) / 2, y=-(lo[1] + hi[1]) / 2,
                        z=-(lo[2] + hi[2]) / 2)
    raise LatheError(
        "unknown origin %r. Use as_built, bottom_center or center." % origin)


# ------------------------------------------------------------------ native

def _write_obj(solid: Solid, path: str, materials: Dict[str, str]) -> dict:
    """OBJ with one group per tag, so the parts stay addressable downstream."""
    msh = solid.mesh()
    verts, tris = msh.verts, msh.tris
    part = (msh.part_id if msh.part_id is not None
            else np.zeros(len(tris), dtype=np.int64))
    tag_of = dict(solid.tags)

    mtl_name = os.path.splitext(os.path.basename(path))[0] + ".mtl"
    groups = {}
    for i, pid in enumerate(part.tolist()):
        groups.setdefault(tag_of.get(pid, "part_%d" % pid), []).append(i)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# %s -- written by Lathe\n" % solid.name)
        f.write("# %d verts, %d tris, %d groups\n" % (len(verts), len(tris), len(groups)))
        f.write("mtllib %s\n" % mtl_name)
        for v in verts:
            f.write("v %.9g %.9g %.9g\n" % (v[0], v[1], v[2]))
        # Normals are per-face here: the mesh is flat-shaded triangles, and
        # inventing vertex normals would misrepresent the geometry.
        nrm = _face_normals(verts, tris)
        for n in nrm:
            f.write("vn %.6g %.6g %.6g\n" % (n[0], n[1], n[2]))
        for tag in sorted(groups):
            f.write("g %s\n" % tag)
            f.write("usemtl %s\n" % tag)
            for i in groups[tag]:
                a, b, c = (int(x) + 1 for x in tris[i])
                f.write("f %d//%d %d//%d %d//%d\n" % (a, i + 1, b, i + 1, c, i + 1))

    with open(os.path.join(os.path.dirname(os.path.abspath(path)), mtl_name),
              "w", encoding="utf-8") as f:
        f.write("# %s\n" % solid.name)
        for tag in sorted(groups):
            r, g, b = _PALETTE.get(materials.get(tag, ""), (0.72, 0.72, 0.72))
            f.write("\nnewmtl %s\n" % tag)
            f.write("Kd %.4f %.4f %.4f\n" % (r, g, b))
            f.write("Ka 0 0 0\nKs 0.1 0.1 0.1\nNs 32\nd 1\nillum 2\n")
    return {"groups": sorted(groups), "sidecar": mtl_name}


def _write_stl(solid: Solid, path: str) -> dict:
    """Binary STL. No tags, no materials -- the format has nowhere to put them."""
    msh = solid.mesh()
    verts, tris = msh.verts, msh.tris
    nrm = _face_normals(verts, tris)
    with open(path, "wb") as f:
        f.write(b"Lathe " + solid.name.encode("ascii", "replace")[:74].ljust(74, b" "))
        f.write(np.uint32(len(tris)).tobytes())
        for i, t in enumerate(tris):
            f.write(np.asarray(nrm[i], dtype="<f4").tobytes())
            for vi in t:
                f.write(np.asarray(verts[vi], dtype="<f4").tobytes())
            f.write(b"\x00\x00")
    return {"note": "STL carries no tags or materials"}


def _face_normals(verts, tris) -> np.ndarray:
    a = verts[tris[:, 0]]
    n = np.cross(verts[tris[:, 1]] - a, verts[tris[:, 2]] - a)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)


_PALETTE = {
    "wood_light": (0.62, 0.42, 0.22), "wood_mid": (0.44, 0.27, 0.13),
    "wood_dark": (0.26, 0.15, 0.07), "ceramic": (0.90, 0.89, 0.86),
    "ceramic_blue": (0.42, 0.55, 0.68), "apple": (0.68, 0.09, 0.09),
    "orange": (0.85, 0.42, 0.06), "banana": (0.88, 0.76, 0.20),
    "pear": (0.72, 0.75, 0.28), "leaf": (0.22, 0.42, 0.16),
    "cloth_red": (0.52, 0.16, 0.14), "cloth_blue": (0.20, 0.26, 0.42),
    "cloth_tan": (0.60, 0.50, 0.36), "skin": (0.80, 0.60, 0.47),
    "hair_brown": (0.20, 0.12, 0.07), "leather": (0.32, 0.22, 0.15),
    "metal": (0.62, 0.63, 0.66),
}


# ----------------------------------------------------------- via Blender

def _via_blender(solid: Solid, path: str, fmt: str, materials: Dict[str, str]) -> dict:
    exe = blender_path()
    if exe is None:
        raise LatheError(
            "%s export needs Blender. Set LATHE_BLENDER, or export .obj/.stl "
            "which are written natively." % fmt)
    msh = solid.mesh()
    tmp = tempfile.mkdtemp(prefix="lathe_export_")
    try:
        npz = os.path.join(tmp, "mesh.npz")
        np.savez(npz, verts=msh.verts, tris=msh.tris,
                 part_id=(msh.part_id if msh.part_id is not None
                          else np.zeros(msh.n_tris, dtype=np.int64)))
        job = {
            "mesh_npz": npz, "name": solid.name, "fmt": fmt,
            "out": os.path.abspath(path),
            "tags": {str(k): v for k, v in solid.tags.items()},
            "materials": materials,
        }
        job_p = os.path.join(tmp, "job.json")
        json.dump(job, open(job_p, "w"))
        proc = subprocess.run(
            [exe, "-b", "--factory-startup", "-noaudio", "-P", _WORKER, "--", job_p],
            capture_output=True, text=True, timeout=900)
        if not os.path.exists(path):
            raise LatheError(
                "%s export produced no file.\n--- stdout ---\n%s\n--- stderr ---\n%s"
                % (fmt, proc.stdout[-2000:], proc.stderr[-3000:]))
        return {"via": "blender"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
