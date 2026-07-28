"""
Talk to the server the way a client actually does: spawn it, speak MCP over
stdio, and drive a real session.

Everything else tests the Python functions. This tests the *server* -- that it
starts, advertises ten tools with usable schemas, and returns content over the
wire. A surface that works when imported and fails when spawned is not a surface.

    python lathe/mcp/tests/test_stdio.py            # no Blender needed
    python lathe/mcp/tests/test_stdio.py --render    # includes lathe_observe
"""

import asyncio
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

EXPECTED = {
    "lathe_setup", "lathe_apply", "lathe_query", "lathe_observe", "lathe_check",
    "lathe_export", "lathe_log", "lathe_branch", "lathe_checkout", "lathe_compare",
    "lathe_view",
}

PROGRAM = '''
top  = cylinder(mm(34), 0.17, segments=48, center=False, tag="top",
                name="SM_Stdio_Table_LOD0").at(z=0.50 - mm(34))
post = cylinder(0.50, 0.030, 0.038, segments=32, center=False, tag="post")
foot = revolve([(0.0, 0.0), (0.12, 0.0), (0.12, mm(18)), (0.05, mm(26)),
                (0.0, mm(26))], segments=48, tag="foot")
piece = union_all([top, post, foot], name="SM_Stdio_Table_LOD0")
if piece.components() != 1:
    raise LatheError("the post is not reaching both the top and the foot")
spec = {"components": 1, "origin": "bottom_center", "budget": {"tris_lod0": 9000}}
result = piece
'''

PASS, FAIL = [], []


def check(label, ok, detail=""):
    (PASS if ok else FAIL).append(label)
    print("  %-4s %s%s" % ("ok" if ok else "FAIL", label,
                           ("   " + detail) if detail and not ok else ""))


def text_of(result):
    out = []
    for block in result.content:
        if getattr(block, "type", None) == "text":
            out.append(block.text)
        else:
            out.append("<%s>" % getattr(block, "type", type(block).__name__))
    return "\n".join(out)


def kinds(result):
    return [getattr(b, "type", type(b).__name__) for b in result.content]


async def main():
    do_render = "--render" in sys.argv
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "lathe.mcp.server"], cwd=REPO,
        env=dict(os.environ, PYTHONPATH=REPO))

    print("spawning: %s -m lathe.mcp.server" % os.path.basename(sys.executable))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            init = await s.initialize()
            print("connected to %r %s\n"
                  % (init.serverInfo.name, init.serverInfo.version))

            # -- handshake and schemas ------------------------------------
            listed = await s.list_tools()
            names = {t.name for t in listed.tools}
            check("eleven tools advertised", names == EXPECTED,
                  "missing %s, unexpected %s" % (EXPECTED - names, names - EXPECTED))
            check("every tool has a description",
                  all((t.description or "").strip() for t in listed.tools))
            check("every tool has an input schema",
                  all(t.inputSchema for t in listed.tools))
            annotated = [t for t in listed.tools if t.annotations]
            check("every tool is annotated", len(annotated) == len(listed.tools))
            ro = {t.name for t in listed.tools
                  if t.annotations and t.annotations.readOnlyHint}
            check("read-only hints are right",
                  ro == {"lathe_query", "lathe_observe", "lathe_check",
                         "lathe_log", "lathe_compare"}, "got %s" % sorted(ro))

            # -- setup ----------------------------------------------------
            r = await s.call_tool("lathe_setup", {"params": {
                "seed": 20260728, "style": "lantern", "check_upstream": False}})
            body = text_of(r)
            check("lathe_setup reports READY", "READY" in body)
            check("lathe_setup builds a welcome piece", "welcome piece" in body)
            check("lathe_setup hands back program text",
                  "result = piece" in body or "result = pot" in body)
            check("lathe_setup honours a forced style", "'lantern'" in body)

            # -- apply ----------------------------------------------------
            r = await s.call_tool("lathe_apply", {"params": {
                "program": PROGRAM, "message": "stdio table", "branch": "stdio"}})
            body = text_of(r)
            check("lathe_apply builds", "SM_Stdio_Table_LOD0" in body)
            check("lathe_apply returns the report inline", "check  SM_Stdio" in body)
            check("lathe_apply passes the profile", "PASS" in body)

            r = await s.call_tool("lathe_apply", {"params": {"program": PROGRAM}})
            check("re-applying is a no-op", "no change" in text_of(r))

            # -- query ----------------------------------------------------
            for expr, want in (("bbox", "0.340"), ("watertight", "true"),
                               ("tags", "post"), ("faces(tag=top)", "matched"),
                               ("symmetry(YZ)", "volume_iou"),
                               ("raycast(from=0,-2,0.3, dir=0,1,0)", "raycast")):
                r = await s.call_tool("lathe_query", {"params": {"expr": expr}})
                check("query %s" % expr, want in text_of(r),
                      text_of(r)[:120].replace("\n", " "))

            r = await s.call_tool("lathe_query", {"params": {"expr": "nonsense"}})
            check("bad query returns the grammar",
                  "Valid query expressions" in text_of(r))

            # -- branch, log, compare-by-checkout -------------------------
            await s.call_tool("lathe_branch", {"params": {"name": "wide"}})
            await s.call_tool("lathe_apply", {"params": {
                "program": PROGRAM.replace("0.17,", "0.24,"), "message": "wider"}})
            r = await s.call_tool("lathe_log", {"params": {}})
            body = text_of(r)
            check("log lists branches", "wide" in body and "stdio" in body)
            r = await s.call_tool("lathe_checkout", {"params": {"ref": "stdio"}})
            check("checkout switches head", "HEAD -> stdio" in text_of(r))

            # -- failure paths --------------------------------------------
            r = await s.call_tool("lathe_apply", {"params": {
                "program": "result = 1"}})
            check("wrong result type is explained",
                  "expected a Solid" in text_of(r))
            r = await s.call_tool("lathe_apply", {"params": {
                "program": "result = nope(1)"}})
            check("unknown name names the line and lists the API",
                  "Available:" in text_of(r))

            # -- export ---------------------------------------------------
            out = os.path.join(REPO, "out", "SM_Stdio_Table_LOD0.obj")
            r = await s.call_tool("lathe_export", {"params": {
                "path": out, "materials": {"top": "wood_mid", "post": "metal",
                                           "foot": "wood_dark"}}})
            body = text_of(r)
            ok = os.path.exists(out)
            check("export writes a file", ok)
            if ok:
                info = json.loads(body)
                check("export applied the transform", info["transform_applied"])
                check("export grouped by tag",
                      set(info["groups"]) >= {"top", "post", "foot"},
                      str(info.get("groups")))
                check("export bbox sits on z=0",
                      abs(info["bbox_m"][0][2]) < 1e-9)
                os.remove(out)
                mtl = out.replace(".obj", ".mtl")
                if os.path.exists(mtl):
                    os.remove(mtl)

            # -- interactive preview (no Blender needed) -------------------
            import re as _re
            view_path = os.path.join(REPO, "out", "stdio_preview.html")
            r = await s.call_tool("lathe_view", {"params": {
                "path": view_path,
                "materials": {"top": "wood_mid", "post": "metal"}}})
            body = text_of(r)
            check("view writes a preview", os.path.exists(view_path), body[:160])
            if os.path.exists(view_path):
                info = json.loads(body)
                check("view reports the real byte size",
                      info["bytes"] == os.path.getsize(view_path),
                      "%s vs %s" % (info["bytes"], os.path.getsize(view_path)))
                check("view page is self contained", info["self_contained"])
                page = open(view_path, encoding="utf-8").read()
                check("view page embeds the mesh", '"pos":"' in page)
                check("view page has no external references",
                      not any(u.startswith(("http:", "https:", "//"))
                              for u in _re.findall(
                                  r'(?:src|href)\s*=\s*"([^"]+)"', page)))
                os.remove(view_path)
            r = await s.call_tool("lathe_view", {"params": {"path": "nope.txt"}})
            check("view rejects a non-html path", "must end in .html" in text_of(r))

            # -- render ---------------------------------------------------
            if do_render:
                r = await s.call_tool("lathe_observe", {"params": {
                    "views": ["front", "iso"], "size": 260, "samples": 12,
                    "materials": {"top": "wood_mid", "post": "metal",
                                  "foot": "wood_dark"}}})
                check("observe returns an image", "image" in kinds(r),
                      str(kinds(r)))
                r = await s.call_tool("lathe_compare", {"params": {
                    "refs": ["stdio", "wide"], "size": 240, "samples": 12}})
                check("compare returns an image", "image" in kinds(r),
                      str(kinds(r)))
            else:
                print("  --   skipped observe and compare (pass --render)")

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("  FAILED: %s" % f)
    return 1 if FAIL else 0


sys.exit(asyncio.run(main()))
