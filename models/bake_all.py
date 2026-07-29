"""
The game-art path: build a high/low pair, unwrap the low, bake the high onto it.

    python models/bake_all.py            # chair and bowl
    python models/bake_all.py chair

Kept separate from build_all.py on purpose. build_all emits the *bevelled* mesh,
which is the right thing when the bevel is the deliverable. This emits the low
poly plus maps, which is the right thing when it ships in an engine — and the two
answer different questions, so they stay two scripts.

Why it matters: the bevelled meshes legitimately warn on W302, because padding
imposes a minimum chart size and modelled bevels produce many slivers. Moving the
bevel into a normal map removes the slivers rather than tolerating them.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lathe.build as LB  # noqa: E402
from lathe.bake import bake  # noqa: E402
from lathe.perceive import tier1  # noqa: E402
from lathe.verify import engine as vengine  # noqa: E402
from lathe.verify import mesh as vmesh  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "bake")

# Only assets with a bevel have anything to bake down. The villager has none, and
# models/villager.py says so rather than pretending.
ASSETS = {
    "chair": dict(module="models.chair", stem="T_Chair_Oak",
                  resolution=1024, density=5.12, budget=3000),
    "bowl": dict(module="models.bowl", stem="T_BowlFruit_Ceramic",
                 resolution=1024, density=5.12, budget=9000),
}


def run(key, samples=16):
    cfg = ASSETS[key]
    mod = __import__(cfg["module"], fromlist=["build_pair"])
    print("\n" + "=" * 74)
    print("%s  high-to-low" % key)
    print("=" * 74)

    t0 = time.perf_counter()
    low, high = mod.build_pair()
    print("  built pair in %.1f s   low %d tris   high %d tris   ratio %.2f:1"
          % (time.perf_counter() - t0, low.n_tris, high.n_tris,
             high.n_tris / max(low.n_tris, 1)))

    res, dens = cfg["resolution"], cfg["density"]
    low = LB.unwrap(low, resolution=res, texel_density=dens)
    d = LB.texel_density(low, res)
    dh = LB.texel_density(LB.unwrap(high, resolution=res, texel_density=dens), res)
    print("  low  : %3d islands   %.2f-%.2f px/cm" % (len(d["islands"]), d["min"], d["max"]))
    print("  high : %3d islands   %.2f-%.2f px/cm   <- what modelling the bevel costs"
          % (len(dh["islands"]), dh["min"], dh["max"]))

    t0 = time.perf_counter()
    info = bake(low, high, OUT, stem=cfg["stem"], targets=("normal", "ao"),
                resolution=res, samples=samples, target_engine="unreal")
    print("  baked in %.0f s   cage %.1f mm   ray %.1f mm   %s"
          % (time.perf_counter() - t0, info["cage_extrusion_m"] * 1000,
             info["max_ray_distance_m"] * 1000,
             ", ".join(os.path.basename(p) for p in info["maps"].values())))

    print()
    print("  " + str(tier1.counts(low.mesh())).strip().replace("\n", "\n  "))

    spec = vmesh.EmitSpec(
        profile="game_prop_pbr", target="unreal", components=1,
        origin="bottom_center", budget={"tris_lod0": cfg["budget"]},
        texel_density=dens, texture_resolution=res,
        textures=tuple(
            vmesh.TextureDecl(name="%s_%s" % (cfg["stem"], ch), width=res,
                              height=res, channel=ch)
            for ch in ("N", "AO")),
        bake=vmesh.BakeSpec(targets=tuple(info["maps"]), uv_set=low.uv_set,
                            unique_uv=True))
    report = vengine.check(vmesh.Asset(
        mesh=vmesh.from_manifold(low.man, name=low.name,
                                 uv_sets={low.uv_set: low.uv}), spec=spec))
    print()
    print("  " + str(report).strip().replace("\n", "\n  "))
    return low, high, info, report


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    keys = args or list(ASSETS)
    os.makedirs(OUT, exist_ok=True)
    out = {}
    for k in keys:
        if k not in ASSETS:
            print("unknown asset %r; known: %s" % (k, ", ".join(ASSETS)))
            return 2
        out[k] = run(k)

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    worst = 0
    for k, (low, high, info, report) in out.items():
        n_err, n_warn = len(report.errors), len(report.warnings)
        worst = max(worst, 1 if n_err else 0)
        print("  %-8s low %5d tris (was %5d, %.1f%% of it)  %d maps  %d error %d warn  %s"
              % (k, low.n_tris, high.n_tris, 100.0 * low.n_tris / high.n_tris,
                 len(info["maps"]), n_err, n_warn, "PASS" if not n_err else "FAIL"))
    return worst


if __name__ == "__main__":
    sys.exit(main())
