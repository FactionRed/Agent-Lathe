"""
High-to-low baking -- the piece that lets a bevel live in texture instead of
geometry.

This closes the W302 story the UV work opened. Those assets warn about texel
density variance because their bevels are *modelled*: padding imposes a minimum
chart size, so the many resulting slivers get more texels than their area asks
for. Baking is the answer a technical artist would give — carry the bevel in a
normal map and let the low poly be flat-faced and few-charted.

Blender does the ray casting, through the same subprocess pattern as the bevel.
Writing a cage-based ray caster that handles skirts and self-occlusion correctly
is a project in itself, and not this project.

The pre-bake checklist is enforced in code rather than documented and hoped for:
transforms applied on both meshes, matching world position, the low smooth-shaded
with an edge split at the seam angle, cage extrusion derived from the low's own
size, and Flip Y switched by target engine.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import Dict, Optional, Sequence

import numpy as np

from ..build.bevel import blender_path
from ..build.solid import LatheError, Solid
from ..build.uvmap import PADDING_TABLE

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bake_worker.py")

TARGETS = ("normal", "ao")


def bake(low: Solid, high: Solid, out_dir: str, stem: Optional[str] = None,
         targets: Sequence[str] = TARGETS, resolution: int = 1024,
         samples: int = 32, target_engine: str = "unreal",
         cage_extrusion: Optional[float] = None,
         max_ray_distance: Optional[float] = None,
         margin_px: Optional[int] = None,
         smooth_angle: float = 32.0, verbose: bool = False) -> Dict[str, object]:
    """Bake `high` onto `low`'s UV set. Returns {target: path} plus the settings.

    `low` must carry UVs -- the maps land in its layout. `high` needs none.
    """
    if not low.has_uv:
        raise LatheError(
            "the low poly has no UV set, so a bake has nowhere to land.\n"
            "  Call unwrap() on the low mesh first.")
    for name, s in (("low", low), ("high", high)):
        if not s.status_ok():
            raise LatheError("the %s mesh is not a valid solid" % name)

    lo, hi = low.bbox()
    diag = max((b - a) for a, b in zip(lo, hi)) or 1.0
    # Ray distance from the subject's own size. A fixed number is wrong at both
    # ends: too small on a 2 m asset gives black patches where rays miss, too
    # large on a 20 mm one lets rays hit the far side.
    cage = diag * 0.02 if cage_extrusion is None else cage_extrusion
    ray = diag * 0.05 if max_ray_distance is None else max_ray_distance
    margin = (PADDING_TABLE.get(resolution, 4) if margin_px is None else margin_px)

    # Arguments first, environment second. A misspelt target is the caller's
    # mistake whether or not Blender is installed, and answering "baking needs
    # Blender" to it sends them to install something that would not have helped.
    unknown = [t for t in targets if t not in ("normal", "ao", "curvature")]
    if unknown:
        raise LatheError(
            "unknown bake target(s) %s. Available: normal, ao, curvature."
            % ", ".join(unknown))

    exe = blender_path()
    if exe is None:
        raise LatheError("baking needs Blender. Set LATHE_BLENDER or put it on PATH.")

    os.makedirs(out_dir, exist_ok=True)
    stem = stem or low.name
    tmp = tempfile.mkdtemp(prefix="lathe_bake_")
    try:
        lmesh, hmesh = low.mesh(), high.mesh()
        low_p = os.path.join(tmp, "low.npz")
        high_p = os.path.join(tmp, "high.npz")
        np.savez(low_p, verts=lmesh.verts, tris=lmesh.tris, uv=low.uv)
        np.savez(high_p, verts=hmesh.verts, tris=hmesh.tris)

        job = {
            "low_npz": low_p, "high_npz": high_p,
            "out_dir": os.path.abspath(out_dir), "stem": stem,
            "targets": list(targets), "resolution": int(resolution),
            "samples": int(samples), "target_engine": target_engine,
            "cage_extrusion": float(cage), "max_ray_distance": float(ray),
            "margin_px": int(margin), "smooth_angle": float(smooth_angle),
        }
        job_p = os.path.join(tmp, "job.json")
        json.dump(job, open(job_p, "w"))

        proc = subprocess.run(
            [exe, "-b", "--factory-startup", "-noaudio", "-P", _WORKER, "--", job_p],
            capture_output=True, text=True, timeout=3600,
            stdin=subprocess.DEVNULL)

        written: Dict[str, str] = {}
        errors = []
        for line in proc.stdout.splitlines():
            if line.startswith("LATHE_BAKED "):
                _, target, path = line.split(" ", 2)
                written[target] = path
            elif line.startswith("LATHE_BAKE_ERROR "):
                errors.append(line[len("LATHE_BAKE_ERROR "):])
        if verbose:
            print("  bake wrote %s" % ", ".join(written) or "nothing")
        if not written:
            raise LatheError(
                "the bake produced no maps.\n"
                "  errors: %s\n--- stderr ---\n%s"
                % ("; ".join(errors) or "none reported", proc.stderr[-2500:]))

        return {
            "maps": written,
            "errors": errors,
            "resolution": int(resolution),
            "samples": int(samples),
            "target_engine": target_engine,
            "flip_green": target_engine == "unity",
            "cage_extrusion_m": round(cage, 6),
            "max_ray_distance_m": round(ray, 6),
            "margin_px": int(margin),
            "low_tris": int(low.n_tris),
            "high_tris": int(high.n_tris),
            "ratio": round(high.n_tris / max(low.n_tris, 1), 2),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
