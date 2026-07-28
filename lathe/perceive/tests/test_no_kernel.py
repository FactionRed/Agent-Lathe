"""
DESIGN.md 11: the core must be headless and Blender-optional.

This runs the probes in a subprocess with `manifold3d` and `bpy` blocked at import
time, on a cube built from raw arrays. It asserts that the package still imports,
that the closed-form probes give the same exact answers, that `symmetry` falls
back and names the reason, and that nothing anywhere pulled in `bpy`.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

SCRIPT = textwrap.dedent(
    """
    import sys
    import numpy as np
    from importlib.abc import MetaPathFinder

    class Blocker(MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("manifold3d", "bpy"):
                raise ImportError("blocked: %s" % name)
            return None

    sys.meta_path.insert(0, Blocker())
    sys.path.insert(0, "__ROOT__")

    from lathe.perceive import kernel, tier1, tier2
    from lathe.perceive.mesh import Mesh

    verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                      [0,0,1],[1,0,1],[1,1,1],[0,1,1]], float)
    tris = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                     [1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    cube = Mesh(verts=verts, tris=tris, name="raw_cube")
    open_cube = cube.select_tris([i for i in range(12) if i not in (2, 3)],
                                 name="raw_open")

    print("HAVE_MANIFOLD", kernel.HAVE_MANIFOLD)
    print("BPY_LOADED", "bpy" in sys.modules)
    print("VOLUME", repr(tier1.volume(cube).fields["volume_m3"]))
    print("AREA", repr(tier1.surface_area(cube).fields["area_m2"]))
    print("COM", tier1.center_of_mass(cube).fields["center_m"])
    print("GENUS", tier1.counts(cube).fields["genus"])
    print("WATERTIGHT", bool(tier1.is_watertight(cube)))
    print("OPEN_WATERTIGHT", bool(tier1.is_watertight(open_cube)))
    print("OPEN_BOUNDARY", tier1.is_watertight(open_cube).fields["boundary_edges"])
    sym = tier1.symmetry(cube, "YZ")
    print("SYM_METHOD", sym.fields["method"])
    print("SYM_SCORE", float(sym))
    print("SYM_WHY", "manifold3d is not installed" in sym.text())
    print("RAY", tier1.raycast(cube, (0.5, 0.5, -1.0), "+Z", unit="m").fields["distance_m"])
    print("TOPOLOGY_OK", "tris / verts / edges" in tier2.topology_summary(cube).text())
    print("REPORT_OK", "measure(bbox)" in tier1.report(cube))
    """
).replace("__ROOT__", str(ROOT).replace("\\", "\\\\"))


def test_probes_work_without_manifold3d_or_bpy():
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={"PYTHONIOENCODING": "utf-8", "PATH": ""},
    )
    assert proc.returncode == 0, proc.stderr
    out = dict(
        line.split(" ", 1) for line in proc.stdout.strip().splitlines() if " " in line
    )
    assert out["HAVE_MANIFOLD"] == "False"
    assert out["BPY_LOADED"] == "False"
    # the closed-form probes are unaffected by the missing kernel
    assert out["VOLUME"] == "1.0"
    assert out["AREA"] == "6.0"
    assert out["COM"] == "(0.5, 0.5, 0.5)"
    assert out["GENUS"] == "0"
    assert out["WATERTIGHT"] == "True"
    assert out["RAY"] == "1.0"
    # the broken mesh is still diagnosed
    assert out["OPEN_WATERTIGHT"] == "False"
    assert out["OPEN_BOUNDARY"] == "4"
    # symmetry degrades to the sampled method and says why
    assert out["SYM_METHOD"] == "surface_sample"
    assert out["SYM_SCORE"] == "1.0"
    assert out["SYM_WHY"] == "True"
    assert out["TOPOLOGY_OK"] == "True"
    assert out["REPORT_OK"] == "True"
