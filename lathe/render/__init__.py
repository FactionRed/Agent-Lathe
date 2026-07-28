"""
Perception Tier 3 -- DESIGN.md section 6.

Renders always come back as a *contact sheet*, never a single view, because one
flattering angle is how a proportion error survives review. Two non-beauty modes
carry most of the diagnostic weight: `matte` (flat silhouette, no shading to
confound the outline) and `wire` (topology density made visible).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..build.bevel import blender_path
from ..build.solid import LatheError, Solid

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_render_worker.py")

DEFAULT_VIEWS = ("front", "side", "iso", "iso2")


def render(solid: Solid, out_path: str, views: Sequence[str] = DEFAULT_VIEWS,
           modes: Sequence[str] = ("shaded",), materials: Optional[Dict[str, str]] = None,
           default_material: str = "default", size: int = 720, samples: int = 64,
           ground: bool = True, smooth_angle_deg: float = 32.0,
           label: Optional[str] = None, verbose: bool = False,
           sheet: bool = True) -> Dict[str, Any]:
    """Render `solid` and composite each mode into its own contact sheet.

    materials maps tag name -> material key (see _render_worker.MATERIALS), so
    section 5.2 provenance is what drives shading.

    Returns {mode: sheet_path}. With sheet=False, returns
    {mode: [(view, frame_path), ...]} of raw frames instead -- which is what a
    caller building its own grid wants, since compositing sheets into a sheet
    doubles every label.
    """
    exe = blender_path()
    if exe is None:
        raise LatheError("render needs Blender. Set LATHE_BLENDER or put it on PATH.")

    msh = solid.mesh()
    lo, hi = solid.bbox()
    tmp = tempfile.mkdtemp(prefix="lathe_render_")
    try:
        npz = os.path.join(tmp, "mesh.npz")
        np.savez(npz, verts=msh.verts, tris=msh.tris,
                 part_id=msh.part_id if msh.part_id is not None
                 else np.zeros(msh.n_tris, dtype=np.int64))
        job = {
            "mesh_npz": npz,
            "name": solid.name,
            "tags": {str(k): v for k, v in solid.tags.items()},
            "materials": dict(materials or {}),
            "default_material": default_material,
            "bbox": [list(lo), list(hi)],
            "views": list(views),
            "modes": list(modes),
            "size": int(size),
            "samples": int(samples),
            "ground": bool(ground),
            "ground_size": max(max(h - l for l, h in zip(lo, hi)) * 4.0, 1.0),
            "smooth_angle_deg": float(smooth_angle_deg),
            "out_pattern": os.path.join(tmp, "%s_%s.png"),
        }
        job_p = os.path.join(tmp, "job.json")
        json.dump(job, open(job_p, "w"))

        proc = subprocess.run(
            [exe, "-b", "--factory-startup", "-noaudio", "-P", _WORKER, "--", job_p],
            capture_output=True, text=True, timeout=3600,
            # See bevel.py: without this the child inherits the MCP JSON-RPC
            # stdin and blocks the transport.
            stdin=subprocess.DEVNULL)
        made = [l for l in proc.stdout.splitlines() if l.startswith("LATHE_RENDERED")]
        if verbose:
            print("  rendered %d frames" % len(made))
        if not made:
            raise LatheError(
                "render produced nothing.\n--- stdout ---\n%s\n--- stderr ---\n%s"
                % (proc.stdout[-3000:], proc.stderr[-4000:]))

        out: Dict[str, Any] = {}
        base, ext = os.path.splitext(out_path)
        for mode in modes:
            tiles = [(v, job["out_pattern"] % (mode, v)) for v in views]
            tiles = [(v, p) for v, p in tiles if os.path.exists(p)]
            if not tiles:
                continue
            if not sheet:
                kept = []
                for v, p in tiles:
                    dest = "%s_%s_%s%s" % (base, mode, v, ext)
                    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".",
                                exist_ok=True)
                    shutil.copyfile(p, dest)
                    kept.append((v, dest))
                out[mode] = kept
                continue
            dest = out_path if len(modes) == 1 else "%s_%s%s" % (base, mode, ext)
            _contact_sheet(tiles, dest,
                           title="%s  [%s]" % (label or solid.name, mode))
            out[mode] = dest
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _contact_sheet(tiles, dest, title, cols=2, pad=10, header=34):
    from PIL import Image, ImageDraw

    imgs = [(v, Image.open(p).convert("RGB")) for v, p in tiles]
    w, h = imgs[0][1].size
    cols = min(cols, len(imgs))
    rows = (len(imgs) + cols - 1) // cols
    dark = imgs[0][1].getpixel((2, 2))[0] < 128
    bg = (18, 19, 22) if dark else (250, 250, 250)
    fg = (232, 232, 236) if dark else (24, 24, 28)

    sheet = Image.new("RGB",
                      (cols * w + pad * (cols + 1),
                       header + rows * (h + 20) + pad * (rows + 1)), bg)
    draw = ImageDraw.Draw(sheet)
    draw.text((pad + 2, 10), title, fill=fg)
    for i, (view, im) in enumerate(imgs):
        r, c = divmod(i, cols)
        x = pad + c * (w + pad)
        y = header + pad + r * (h + 20 + pad)
        sheet.paste(im, (x, y))
        draw.text((x + 4, y + h + 4), view, fill=fg)
    os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
    sheet.save(dest)
    return dest
