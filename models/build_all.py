"""
Build, verify, probe and render every test asset.

    python models/build_all.py            # everything
    python models/build_all.py chair      # one asset
    python models/build_all.py --no-render

This is the whole loop DESIGN.md describes, end to end: a program produces
geometry, the verifier gates it against a declared spec, the probes measure it,
and the renderer shows it. Nothing here touches a viewport.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lathe.perceive import tier1, tier2  # noqa: E402
from lathe.verify import engine as vengine  # noqa: E402
from lathe.verify import mesh as vmesh  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")

# Texel density targets follow furniture and character practice: large props at
# 5.12 px/cm, a figure a little higher because it carries a face.
ASSETS = {
    "chair": dict(
        module="models.chair", views=("front", "side", "iso", "iso2"),
        spec=dict(components=1, origin="bottom_center", budget={"tris_lod0": 3000}),
        uv=dict(resolution=1024, texel_density=5.12, channel="D"),
    ),
    "bowl": dict(
        module="models.bowl", views=("front", "side", "iso", "top"),
        # The fruit rest against the bowl and each other, so the union really is
        # one body. Declared explicitly rather than read back from the geometry:
        # a declaration copied from the measurement checks nothing.
        spec=dict(components=1, origin="bottom_center", budget={"tris_lod0": 9000}),
        uv=dict(resolution=1024, texel_density=5.12, channel="D"),
    ),
    "villager": dict(
        module="models.villager", views=("front", "side", "iso", "iso2"),
        spec=dict(components=1, origin="bottom_center", budget={"tris_lod0": 4000}),
        uv=dict(resolution=2048, texel_density=10.24, channel="D"),
    ),
}


def build_one(key, render_it=True, size=440, samples=48, modes=("shaded",)):
    cfg = ASSETS[key]
    mod = __import__(cfg["module"], fromlist=["build"])
    print("\n" + "=" * 74)
    print("%s  (%s)" % (key, cfg["module"]))
    print("=" * 74)

    t0 = time.perf_counter()
    solid = mod.build()
    build_s = time.perf_counter() - t0
    print("  built in %.1f s   %d tris   %d component(s)   tags: %s"
          % (build_s, solid.n_tris, solid.components(),
             ", ".join(sorted(set(solid.tags.values())))))

    # -- unwrap, after the booleans and the bevel ---------------------------
    # A boolean drops the UV channel because it creates surface the unwrap never
    # saw, so unwrapping has to come last. That is the real pipeline order too.
    uvcfg = cfg.get("uv")
    if uvcfg:
        import lathe.build as LB
        t0 = time.perf_counter()
        solid = LB.unwrap(solid, resolution=uvcfg["resolution"],
                          texel_density=uvcfg["texel_density"])
        d = LB.texel_density(solid, uvcfg["resolution"])
        print("  unwrapped in %.1f s   %d islands   %.2f-%.2f px/cm at %d"
              % (time.perf_counter() - t0, len(d["islands"]), d["min"], d["max"],
                 uvcfg["resolution"]))

    # -- probes (section 6) ------------------------------------------------
    msh = solid.mesh()
    print()
    for probe in (tier1.bbox, tier1.volume, tier1.counts, tier1.is_watertight,
                  tier1.center_of_mass):
        print("  " + str(probe(msh)).strip().replace("\n", "\n  "))
    print("  " + str(tier1.symmetry(msh, "YZ")).strip().replace("\n", "\n  "))

    # -- verify (section 7) ------------------------------------------------
    spec_kw = dict(cfg["spec"])
    if spec_kw.get("components") is None:
        spec_kw["components"] = solid.components()
    uv_sets = None
    if solid.has_uv and uvcfg:
        res = uvcfg["resolution"]
        base = solid.name.replace("SM_", "T_", 1).replace("_LOD0", "")
        spec_kw.update(
            texel_density=uvcfg["texel_density"],
            texture_resolution=res,
            textures=(vmesh.TextureDecl(name="%s_%s" % (base, uvcfg["channel"]),
                                        width=res, height=res,
                                        channel=uvcfg["channel"]),),
            bake=vmesh.BakeSpec(targets=("normal", "ao"), uv_set=solid.uv_set,
                                unique_uv=True),
        )
        uv_sets = {solid.uv_set: solid.uv}
    spec = vmesh.EmitSpec(profile="game_prop_pbr", target="unreal", **spec_kw)
    asset = vmesh.Asset(
        mesh=vmesh.from_manifold(solid.man, name=solid.name, uv_sets=uv_sets),
        spec=spec)
    report = vengine.check(asset)
    print()
    print("  " + str(report).strip().replace("\n", "\n  "))

    # -- render (section 6, tier 3) ---------------------------------------
    sheets = {}
    if render_it:
        from lathe.render import render
        t0 = time.perf_counter()
        sheets = render(
            solid, os.path.join(OUT, "%s.png" % key),
            views=cfg["views"], modes=modes,
            materials=mod.MATERIALS, size=size, samples=samples,
            label=solid.name)
        print("\n  rendered in %.0f s: %s"
              % (time.perf_counter() - t0,
                 ", ".join(os.path.basename(p) for p in sheets.values())))
    return solid, report, sheets


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    render_it = "--no-render" not in sys.argv
    keys = args or list(ASSETS)
    os.makedirs(OUT, exist_ok=True)
    results = {}
    for k in keys:
        if k not in ASSETS:
            print("unknown asset %r; known: %s" % (k, ", ".join(ASSETS)))
            return 2
        results[k] = build_one(k, render_it=render_it)

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    worst = 0
    for k, (solid, report, _) in results.items():
        # Use the report's own accessors. Comparing f.severity to the string
        # "error" is always False -- severity is a Severity, not a str -- so the
        # summary printed PASS over a failing report until this was fixed.
        n_err = len(report.errors)
        n_warn = len(report.warnings)
        worst = max(worst, 1 if n_err else 0)
        print("  %-10s %6d tris  %d comp  %d error %d warn  %s"
              % (k, solid.n_tris, solid.components(), n_err, n_warn,
                 "PASS" if not n_err else "FAIL"))
    return worst


if __name__ == "__main__":
    sys.exit(main())
