"""
Drive every MCP tool the way an agent would, in order.

    python lathe/mcp/tests/smoke.py            # no Blender: skips render tools
    python lathe/mcp/tests/smoke.py --render   # everything

This calls the tool functions directly rather than over stdio. The point is to
prove the surface is sufficient: could an agent author, inspect, validate,
branch, compare and export an asset using only these nine tools?
"""

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from lathe.mcp import server as S  # noqa: E402

STEP1 = '''
seat  = box(0.42, 0.40, mm(35), tag="seat", name="SM_Stool_Oak_LOD0").at(z=0.4325)
legs  = [box(0.045, 0.045, 0.415, tag="leg").at(x=sx * 0.16, y=sy * 0.15, z=0.2075)
         for sx in (-1, 1) for sy in (-1, 1)]
stool = union_all([seat] + legs, name="SM_Stool_Oak_LOD0")
assert_count(stool.faces(tag="seat"), 6, "seat faces")
spec = {"components": 1, "origin": "bottom_center", "budget": {"tris_lod0": 4000}}
result = stool
'''

STEP2 = STEP1.replace(
    "result = stool",
    'rail = [box(0.30, 0.022, 0.045, tag="rail").at(y=sy * 0.15, z=0.13)\n'
    '        for sy in (-1, 1)]\n'
    'stool = union_all([stool] + rail, name="SM_Stool_Oak_LOD0")\n'
    'result = stool')

VARIANT = STEP2.replace("0.42, 0.40", "0.36, 0.36")


def show(title, body):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)
    text = body if isinstance(body, str) else "<%s>" % type(body).__name__
    print(text if len(text) < 2600 else text[:2600] + "\n  ... truncated")


async def main():
    do_render = "--render" in sys.argv

    show("lathe_apply  (root commit)",
         await S.lathe_apply(S.ApplyInput(program=STEP1, message="seat and legs")))

    show("lathe_apply  (same program again -- must be a no-op)",
         await S.lathe_apply(S.ApplyInput(program=STEP1, message="again")))

    show("lathe_apply  (add rails -- note the inline report and the diff)",
         await S.lathe_apply(S.ApplyInput(program=STEP2, message="add side rails")))

    for expr in ("bbox", "counts", "watertight", "tags", "faces(tag=rail)",
                 "edges(dihedral_gt=60, length_gt=0.02)", "symmetry(YZ)",
                 "raycast(from=0,-2,0.44, dir=0,1,0)", "parts"):
        show("lathe_query  %s" % expr,
             await S.lathe_query(S.QueryInput(expr=expr)))

    show("lathe_query  (bad expression returns the grammar)",
         await S.lathe_query(S.QueryInput(expr="how_tall_is_it")))

    show("lathe_check  game_prop_pbr",
         await S.lathe_check(S.CheckInput()))
    show("lathe_check  character_skinned (wrong profile on purpose)",
         await S.lathe_check(S.CheckInput(profile="character_skinned")))

    show("lathe_branch  narrow", await S.lathe_branch(S.BranchInput(name="narrow")))
    show("lathe_apply   (variant on the branch)",
         await S.lathe_apply(S.ApplyInput(program=VARIANT, message="narrower seat")))
    show("lathe_log", await S.lathe_log(S.LogInput()))
    show("lathe_checkout main", await S.lathe_checkout(S.CheckoutInput(ref="main")))

    show("lathe_apply  (cardinality failure is loud and specific)",
         await S.lathe_apply(S.ApplyInput(
             program=STEP2.replace('faces(tag="seat"), 6', 'faces(tag="seat"), 99'),
             message="wrong assertion")))
    show("lathe_apply  (program error names the line)",
         await S.lathe_apply(S.ApplyInput(program="result = frobnicate(3)")))

    with tempfile.TemporaryDirectory(prefix="lathe_smoke_") as tmp:
        for ext in ("obj", "stl"):
            show("lathe_export  .%s" % ext, await S.lathe_export(S.ExportInput(
                path=os.path.join(tmp, "SM_Stool_Oak_LOD0." + ext),
                materials={"seat": "wood_mid", "leg": "wood_dark", "rail": "wood_dark"})))

        if do_render:
            show("lathe_observe  shaded", await S.lathe_observe(S.ObserveInput(
                views=["front", "iso"], size=300, samples=16,
                materials={"seat": "wood_mid", "leg": "wood_dark",
                           "rail": "wood_dark"})))
            show("lathe_compare  main vs narrow", await S.lathe_compare(
                S.CompareInput(refs=["main", "narrow"], size=280, samples=16)))
            show("lathe_export  .glb", await S.lathe_export(S.ExportInput(
                path=os.path.join(tmp, "SM_Stool_Oak_LOD0.glb"))))
        else:
            print("\n(skipped lathe_observe, lathe_compare and .glb -- pass --render)")

    print("\n" + "=" * 74)
    print("all nine tools exercised")
    print("=" * 74)


asyncio.run(main())
