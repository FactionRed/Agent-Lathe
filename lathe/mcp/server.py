#!/usr/bin/env python3
"""
MCP server for Lathe -- DESIGN.md section 9.

Nine tools, deliberately. A large tool surface drowns the model; a small surface
plus a rich language does not. The expressive weight sits in the program text
that `lathe_apply` takes and the expression language `lathe_query` speaks.

`lathe_apply` returns the validation report inline. Section 9 makes that point
explicitly: the agent should see the consequences of a change in the same turn it
makes the change, rather than needing to remember to look.

Run:  python -m lathe.mcp.server
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from enum import Enum
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP, Image
from pydantic import BaseModel, ConfigDict, Field

from ..build.solid import CardinalityError, LatheError
from ..io import FORMATS, export as do_export
from ..verify import engine as vengine
from ..verify import mesh as vmesh
from . import probes
from .session import ProgramError, Workspace

mcp = FastMCP("lathe_mcp")

WS = Workspace()
OUT_DIR = os.environ.get(
    "LATHE_OUT", os.path.join(os.path.expanduser("~"), ".lathe", "out"))


# --------------------------------------------------------------- formatting

def _fail(exc: Exception) -> str:
    """One error path. Section 7's shape: what broke, and what to do next."""
    if isinstance(exc, CardinalityError):
        return "CARDINALITY ASSERTION FAILED\n%s" % exc
    if isinstance(exc, ProgramError):
        return "PROGRAM ERROR\n%s" % exc
    if isinstance(exc, LatheError):
        return "ERROR\n  %s" % exc
    return "UNEXPECTED %s\n  %s" % (type(exc).__name__, exc)


def _spec_from(commit_spec: Dict[str, Any], solid) -> vmesh.EmitSpec:
    kw = dict(commit_spec or {})
    kw.setdefault("profile", "game_prop_pbr")
    if kw.get("components") is None:
        kw.pop("components", None)
    return vmesh.EmitSpec(**kw)


def _check(ref: Optional[str], profile: Optional[str] = None) -> str:
    solid = WS.solid(ref)
    c = WS.commit(ref)
    spec = _spec_from(c.spec, solid)
    asset = vmesh.Asset(
        mesh=vmesh.from_manifold(solid.man, name=solid.name), spec=spec)
    return str(vengine.check(asset, profile)).rstrip()


async def _off_loop(fn, *args, **kw):
    """Run blocking work in a thread.

    Booleans, bevels and renders take seconds. Doing that on the event loop makes
    the server deaf for the duration -- it cannot answer a ping or honour a
    cancellation while a Blender subprocess runs.
    """
    return await asyncio.to_thread(lambda: fn(*args, **kw))


def _headline(solid, commit) -> str:
    return ("  %s  %s\n"
            "  tris %d   components %d   volume %.6f m3   tags: %s"
            % (commit.id, solid.name, solid.n_tris, solid.components(),
               solid.volume(), ", ".join(commit.tags) or "none"))


# ------------------------------------------------------------------- apply

class ApplyInput(BaseModel):
    """Input for lathe_apply."""
    model_config = ConfigDict(str_strip_whitespace=False, extra="forbid")

    program: str = Field(
        ...,
        description=(
            "Python source that builds geometry and assigns the finished Solid to "
            "`result`. The lathe.build API is already in scope -- do not import it. "
            "Optionally assign `spec` (a dict) to declare what the verifier should "
            "check, e.g. spec = {'components': 1, 'origin': 'bottom_center', "
            "'budget': {'tris_lod0': 3000}}."),
        min_length=1, max_length=200000)
    message: str = Field(
        default="", description="Short commit message, e.g. 'add top rail'",
        max_length=200)
    branch: Optional[str] = Field(
        default=None,
        description=("Branch to commit on. Creates it from the current tip if new. "
                     "Omit to stay on the current branch."))


@mcp.tool(
    name="lathe_apply",
    annotations={"title": "Apply a Lathe Program",
                 "readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False})
async def lathe_apply(params: ApplyInput) -> str:
    """Build geometry by running a program, then commit and validate it.

    The unit of work in Lathe. Every operation is a pure function of its explicit
    arguments -- there is no mode, no selection state and no active object, so the
    same program always produces the same geometry.

    Write the program against the API below, addressing geometry by *semantic
    selector* rather than by index. `carved.edges(dihedral_gt=60)` re-resolves on
    every evaluation, so an upstream edit does not invalidate it.

    Guard your assumptions with `assert_count`. It converts a selector silently
    matching the wrong set -- the most dangerous failure mode here -- into a loud
    error naming the assumption that broke.

    API in scope (lengths in metres; `mm(35)` converts):
        box(sx, sy, sz, center=True, tag=, name=)
        cylinder(height, radius, radius_top=None, segments=32, center=True, tag=)
        sphere(radius, segments=32, tag=)
        revolve(profile, segments=48, degrees=360, tag=)   profile = [(radius, height), ...]
        extrude(profile, height, tag=)
        bowl_profile(radius, height, thickness, foot_radius=None)
        bevel(solid, width, segments=2, min_dihedral=30, only_tags=None, new_tag='bevel')
        union_all([...], name=)   array(factory, count, axis='x', span=, center=True)
        assert_count(selection, expected, label=None)
        mm(x)

    Solid methods:
        a + b   a - b   a & b        union, difference, intersection
        .at(x=, y=, z=)  .translate(v)  .rotate((rx,ry,rz))  .scale(v)  .mirror(n)
        .faces(tag=, normal=, area_gt=, z_gt=, z_lt=, all=False)
        .edges(dihedral_gt=, dihedral_lt=, length_gt=, tag=)
        .rename(name)  .retag(new=old)  .components()  .volume()  .bbox()

    Args:
        params (ApplyInput): program, optional message, optional branch.

    Returns:
        str: Commit headline, the program diff against the parent, and the full
        validation report inline. Findings carry a code, the offending named
        entity, numeric evidence and an executable repair.

        On failure, a CARDINALITY ASSERTION FAILED or PROGRAM ERROR block naming
        the line and what to do about it.

    Examples:
        - "make a 600 mm crate with five slots" -> a program ending
          `result = carved`, with `assert_count(rim, 96)` guarding the bevel.
        - Iterating: re-send the whole program with the edit. Each commit stores
          the complete program, so history replays.
        - Don't use for measuring -- that is lathe_query. Don't use for looking --
          that is lathe_observe.

    Security note:
        The program runs via `exec` in a namespace preloaded with the Lathe API.
        That is a restricted namespace, NOT a sandbox: builtins are present, so a
        program can reach the filesystem. This is intended for a local stdio
        server whose caller could already run Python. Do not expose it remotely.
    """
    try:
        commit, solid, diff, is_new = await _off_loop(
            WS.apply, params.program, params.message, params.branch)
    except Exception as exc:
        return _fail(exc)

    parts = []
    if not is_new:
        parts.append("  no change: this program on this parent is already %s"
                     % commit.id)
    parts.append(_headline(solid, commit))
    parts.append("  branch %s   parent %s   geometry %s"
                 % (commit.branch, commit.parent or "(root)", commit.geom_hash))
    if diff:
        parts.append("\nprogram diff\n" + diff)
    try:
        parts.append("\n" + _check(commit.id))
    except Exception as exc:
        parts.append("\n  validation could not run: %s" % exc)
    return "\n".join(parts)


# ------------------------------------------------------------------- query

class QueryInput(BaseModel):
    """Input for lathe_query."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    expr: str = Field(
        ..., description=("Query expression, e.g. 'bbox', 'volume', 'counts', "
                          "'watertight', 'tags', 'symmetry(YZ)', "
                          "'faces(tag=seat)', 'edges(dihedral_gt=60)'. Send an "
                          "invalid value to get the full grammar back."),
        min_length=1, max_length=400)
    ref: Optional[str] = Field(
        default=None,
        description="Commit id, branch, HEAD or HEAD~n. Defaults to HEAD.")


@mcp.tool(
    name="lathe_query",
    annotations={"title": "Measure the Current Solid",
                 "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False})
async def lathe_query(params: QueryInput) -> str:
    """Measure geometry numerically. Prefer this over rendering.

    Closed-form probes cost 30-90 microseconds and answer in text. A render costs
    seconds, shows one interpretation of the shape, and is easy to over-trust.
    Reach for this first and `lathe_observe` second.

    `symmetry` is the exception on cost: it needs kernel booleans, so it runs in
    milliseconds rather than microseconds. It also reports which method it used,
    because its exact path and its sampling fallback produce scores that are not
    comparable.

    Args:
        params (QueryInput): expr, and an optional ref.

    Returns:
        str: Labelled text with units. Selector queries report the logical face or
        edge count plus the (part, face) keys matched.

        On an unknown expression, the complete grammar.

    Examples:
        - "how tall is it?" -> expr='bbox'
        - "is it a single closed solid?" -> expr='watertight' or 'counts'
        - "did the bevel touch the cut faces?" -> expr='faces(tag=cut_face)'
        - "is it symmetric?" -> expr='symmetry(YZ)'
        - Don't use to change geometry -- that is lathe_apply.
    """
    try:
        return probes.evaluate(WS.solid(params.ref), params.expr)
    except Exception as exc:
        return _fail(exc)


# ----------------------------------------------------------------- observe

class RenderMode(str, Enum):
    SHADED = "shaded"
    MATTE = "matte"
    WIRE = "wire"


class ObserveInput(BaseModel):
    """Input for lathe_observe."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    views: List[str] = Field(
        default_factory=lambda: ["front", "side", "iso", "iso2"],
        description=("Views for the contact sheet. Any of front, back, side, top, "
                     "iso, iso2, low."),
        min_length=1, max_length=7)
    mode: RenderMode = Field(
        default=RenderMode.SHADED,
        description=("shaded for appearance; matte for a flat silhouette, which "
                     "reads shape more reliably than shading; wire for topology "
                     "density."))
    materials: Optional[Dict[str, str]] = Field(
        default=None,
        description=("Tag -> material key, e.g. {'seat': 'wood_mid'}. Keys: "
                     "wood_light, wood_mid, wood_dark, ceramic, ceramic_blue, "
                     "apple, orange, banana, pear, leaf, cloth_red, cloth_blue, "
                     "cloth_tan, skin, hair_brown, leather, metal."))
    size: int = Field(default=420, description="Pixels per tile", ge=160, le=1200)
    samples: int = Field(default=40, description="Cycles samples per tile", ge=4, le=512)
    ref: Optional[str] = Field(default=None, description="Commit or branch. Default HEAD.")


@mcp.tool(
    name="lathe_observe",
    annotations={"title": "Render a Contact Sheet",
                 "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False})
async def lathe_observe(params: ObserveInput) -> Any:
    """Look at the geometry as a multi-view contact sheet.

    Never a single view. One flattering angle is how a proportion error survives
    review, so several fixed views come back in one image with a metric ground
    grid for scale.

    Use `matte` when judging form: a flat silhouette removes the shading that
    confounds an outline. Use `wire` to judge topology density. On hard surface
    with merged coplanar faces the wireframe looks sparse, and that emptiness is
    itself the reading -- flat faces really are single quads.

    Needs Blender. Set LATHE_BLENDER if it is not on PATH.

    Args:
        params (ObserveInput): views, mode, optional materials, size, samples, ref.

    Returns:
        Image: The contact sheet as a PNG. On failure, an error string naming the
        cause.

    Examples:
        - "show me the chair" -> defaults are fine
        - "are the proportions right?" -> mode='matte', views=['front','side']
        - "how dense is the mesh?" -> mode='wire'
        - Prefer lathe_query for anything measurable. This costs seconds.
    """
    try:
        from ..render import render
        solid = WS.solid(params.ref)
        commit = WS.commit(params.ref)
        os.makedirs(OUT_DIR, exist_ok=True)
        dest = os.path.join(OUT_DIR, "%s_%s.png" % (commit.id, params.mode.value))
        sheets = await _off_loop(
            render, solid, dest, views=tuple(params.views),
            modes=(params.mode.value,), materials=params.materials or {},
            size=params.size, samples=params.samples,
            label="%s  %s" % (solid.name, commit.id))
        path = sheets.get(params.mode.value)
        if not path or not os.path.exists(path):
            return "ERROR\n  render produced no sheet for mode %r" % params.mode.value
        with open(path, "rb") as fh:
            return Image(data=fh.read(), format="png")
    except Exception as exc:
        return _fail(exc)


# ------------------------------------------------------------------- check

class CheckInput(BaseModel):
    """Input for lathe_check."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    profile: Optional[str] = Field(
        default=None,
        description=("Rule profile: game_prop_pbr, character_skinned or mobile_vr. "
                     "Defaults to whatever the program's `spec` declared."))
    ref: Optional[str] = Field(default=None, description="Commit or branch. Default HEAD.")


@mcp.tool(
    name="lathe_check",
    annotations={"title": "Validate Against a Profile",
                 "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False})
async def lathe_check(params: CheckInput) -> str:
    """Run the rule engine and report every violation with a repair.

    `lathe_apply` already returns this inline, so reach for it directly when
    re-checking against a different profile, or after changing a declaration.

    Rules compare geometry against what the program *declared*. A rule with
    nothing to compare against reports itself as not checked rather than passing
    silently -- so an empty report with four skips is not a clean bill of health.

    Args:
        params (CheckInput): optional profile, optional ref.

    Returns:
        str: A pass/fail headline, then one block per finding carrying a stable
        code (E101-E108, W201-W302, N401), the offending named entity, numeric
        evidence and an executable repair. Then the not-checked list.

    Examples:
        - "is this game-ready?" -> defaults
        - "would this pass for a character?" -> profile='character_skinned'
        - Don't use to measure -- that is lathe_query.
    """
    try:
        return _check(params.ref, params.profile)
    except Exception as exc:
        return _fail(exc)


# ------------------------------------------------------------------ export

class ExportInput(BaseModel):
    """Input for lathe_export."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    path: str = Field(
        ..., description=("Output path. Format comes from the extension: "
                          "obj and stl are written natively; glb, gltf, fbx, usd "
                          "and ply route through Blender."),
        min_length=3, max_length=1024)
    origin: str = Field(
        default="bottom_center",
        description="as_built, bottom_center or center. Applied before writing.")
    materials: Optional[Dict[str, str]] = Field(
        default=None, description="Tag -> material key, used for the OBJ sidecar.")
    ref: Optional[str] = Field(default=None, description="Commit or branch. Default HEAD.")


@mcp.tool(
    name="lathe_export",
    annotations={"title": "Export to a File",
                 "readOnlyHint": False, "destructiveHint": True,
                 "idempotentHint": True, "openWorldHint": False})
async def lathe_export(params: ExportInput) -> str:
    """Write the solid to disk, with the transform applied.

    Overwrites the destination if it exists. Scale is (1,1,1) and rotation is zero
    at the moment of writing, because unapplied scale is the most common export
    bug there is and an exporter that can emit it eventually will.

    Run lathe_check first. This does not gate on the report.

    Args:
        params (ExportInput): path, origin, optional materials, optional ref.

    Returns:
        str: JSON with the format, absolute path, byte count, triangle count,
        applied origin, bounding box in metres, and the OBJ group list where
        applicable.

    Examples:
        - "save as glb for Godot" -> path='.../SM_Chair_Oak_LOD0.glb'
        - "give me an obj I can open anywhere" -> path='....obj'
        - FBX and glTF need Blender. OBJ and STL do not.
    """
    try:
        fmt = os.path.splitext(params.path)[1].lstrip(".").lower()
        if fmt not in FORMATS:
            return ("ERROR\n  unknown format %r. Supported: %s."
                    % (fmt, ", ".join(FORMATS)))
        info = await _off_loop(
            do_export, WS.solid(params.ref), params.path,
            materials=params.materials, origin=params.origin)
        return json.dumps(info, indent=2)
    except Exception as exc:
        return _fail(exc)


# ----------------------------------------------------------------- history

class LogInput(BaseModel):
    """Input for lathe_log."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    limit: int = Field(default=20, description="Commits to show", ge=1, le=200)
    branch: Optional[str] = Field(default=None, description="Branch. Default current.")


@mcp.tool(
    name="lathe_log",
    annotations={"title": "Session History",
                 "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False})
async def lathe_log(params: LogInput) -> str:
    """List commits, branches, and which branch is current.

    Operations are pure, so a commit stores the whole program and checking one out
    means replaying it. Exploration is therefore cheap: branch, try a variant,
    compare the tips.

    Args:
        params (LogInput): limit, optional branch.

    Returns:
        str: Current branch, every branch and its tip, then one line per commit
        with id, asset name, triangle count, component count and message.
    """
    try:
        lines = ["  HEAD -> %s" % WS.head, "", "  branches"]
        for name in sorted(WS.branches):
            tip = WS.branches[name] or "(empty)"
            lines.append("      %-16s %s%s"
                         % (name, tip, "   <- current" if name == WS.head else ""))
        log = WS.log(params.limit, params.branch)
        lines += ["", "  commits (newest first)"]
        if not log:
            lines.append("      none yet -- call lathe_apply")
        for c in log:
            lines.append("      " + c.short())
        return "\n".join(lines)
    except Exception as exc:
        return _fail(exc)


class BranchInput(BaseModel):
    """Input for lathe_branch."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(..., description="New branch name, e.g. 'wider-seat'",
                      min_length=1, max_length=80)
    from_ref: Optional[str] = Field(
        default=None, description="Commit or branch to branch from. Default HEAD.")


@mcp.tool(
    name="lathe_branch",
    annotations={"title": "Create a Branch",
                 "readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": False, "openWorldHint": False})
async def lathe_branch(params: BranchInput) -> str:
    """Create a branch and switch to it, for trying a variant.

    Use this before exploring alternatives. Twelve proportions across twelve
    branches, then one lathe_compare grid, is far cheaper than twelve
    build-look-undo cycles.

    Args:
        params (BranchInput): name, optional from_ref.

    Returns:
        str: The new branch, where it starts, and a reminder that it is current.
    """
    try:
        name = WS.new_branch(params.name, params.from_ref)
        return ("  created branch %s at %s\n  HEAD -> %s"
                % (name, WS.branches[name] or "(empty)", WS.head))
    except Exception as exc:
        return _fail(exc)


class CheckoutInput(BaseModel):
    """Input for lathe_checkout."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    ref: str = Field(..., description="Branch name, commit id, HEAD or HEAD~n",
                     min_length=1, max_length=80)


@mcp.tool(
    name="lathe_checkout",
    annotations={"title": "Switch Branch or Commit",
                 "readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False})
async def lathe_checkout(params: CheckoutInput) -> str:
    """Switch the working head to a branch or commit.

    Checking out a bare commit names a branch at it, because detached state is not
    worth modelling here.

    Args:
        params (CheckoutInput): ref.

    Returns:
        str: The branch now current and a headline for its solid.
    """
    try:
        name = WS.checkout(params.ref)
        solid, commit = WS.solid(None), WS.commit(None)
        return "  HEAD -> %s\n%s" % (name, _headline(solid, commit))
    except Exception as exc:
        return _fail(exc)


class CompareInput(BaseModel):
    """Input for lathe_compare."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    refs: List[str] = Field(
        ..., description="Commits or branches to put side by side",
        min_length=2, max_length=12)
    view: str = Field(default="iso", description="One view, used for every tile")
    mode: RenderMode = Field(default=RenderMode.SHADED, description="Render mode")
    materials: Optional[Dict[str, str]] = Field(default=None, description="Tag -> material")
    size: int = Field(default=340, description="Pixels per tile", ge=160, le=900)
    samples: int = Field(default=32, description="Cycles samples", ge=4, le=256)


@mcp.tool(
    name="lathe_compare",
    annotations={"title": "Compare Variants Side by Side",
                 "readOnlyHint": True, "destructiveHint": False,
                 "idempotentHint": True, "openWorldHint": False})
async def lathe_compare(params: CompareInput) -> Any:
    """Render several commits into one grid, for picking between variants.

    This is what makes breadth cheap, which is the thing agents are actually good
    at. Generate a dozen variants on branches, compare them in one image, keep the
    two that work.

    Args:
        params (CompareInput): refs, view, mode, optional materials, size, samples.

    Returns:
        Image: One grid, one tile per ref, labelled with commit id and triangle
        count. Also lists which refs share a geometry hash, since two programs can
        differ in text and not in geometry.
    """
    try:
        from ..render import _contact_sheet, render
        os.makedirs(OUT_DIR, exist_ok=True)
        tiles, notes = [], []
        with tempfile.TemporaryDirectory(prefix="lathe_cmp_") as tmp:
            for ref in params.refs:
                solid, commit = WS.solid(ref), WS.commit(ref)
                one = os.path.join(tmp, "%s.png" % commit.id)
                # sheet=False: raw frames, so the grid below is the only grid.
                got = await _off_loop(
                    render, solid, one, views=(params.view,),
                    modes=(params.mode.value,), materials=params.materials or {},
                    size=params.size, samples=params.samples, sheet=False)
                frames = got.get(params.mode.value) or []
                for _view, src in frames:
                    if os.path.exists(src):
                        tiles.append(("%s  %d tris" % (commit.id, commit.n_tris), src))
                notes.append((ref, commit.id, commit.geom_hash))
            if not tiles:
                return "ERROR\n  nothing rendered for refs %s" % ", ".join(params.refs)
            dest = os.path.join(OUT_DIR, "compare_%s.png" % "_".join(
                n[1] for n in notes)[:60])
            cols = 2 if len(tiles) <= 4 else 3
            _contact_sheet(tiles, dest, title="compare  %s  [%s]" % (
                " vs ".join(n[1] for n in notes), params.mode.value), cols=cols)
        with open(dest, "rb") as fh:
            return Image(data=fh.read(), format="png")
    except Exception as exc:
        return _fail(exc)


# -------------------------------------------------------------------- setup

class SetupInput(BaseModel):
    """Input for lathe_setup."""
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    confirm: bool = Field(
        default=False,
        description=("False reports what is missing and asks. True installs the "
                     "missing Python packages with pip. Only set True after the "
                     "user has agreed in conversation."))
    seed: Optional[int] = Field(
        default=None,
        description=("Seed for the welcome piece. Omit for one derived from the "
                     "machine and account, which is stable for this user. Pass "
                     "any integer for a different piece."))
    style: Optional[str] = Field(
        default=None,
        description="Force a style: stool, planter, lantern or pedestal. Omit to let the seed pick.")
    build: bool = Field(
        default=True,
        description="Build and commit the welcome piece. False audits only.")
    check_upstream: bool = Field(
        default=True, description="Read the project's public GitHub metadata.")


@mcp.tool(
    name="lathe_setup",
    annotations={"title": "First-Run Setup and Welcome Piece",
                 "readOnlyHint": False, "destructiveHint": False,
                 "idempotentHint": False, "openWorldHint": True})
async def lathe_setup(params: SetupInput) -> str:
    """Audit the install, check the project page, and build a welcome piece.

    Call this first in a new session. It answers three questions: is this
    environment able to build and render, is the local copy current, and what does
    a working Lathe program actually look like.

    The welcome piece is generated as **program text**, seeded from the machine and
    account name. Two users get different pieces; the same user gets the same piece
    every time until they pass a new seed. Style and palette both vary. The program
    is committed to the session, so `lathe_observe` will render it immediately and
    `lathe_export` will write it out.

    It hands back the program rather than only the geometry on purpose -- a first
    artefact should be something the user can read and edit, which is the whole
    premise here.

    Two-step by design. With `confirm=False` it reports and asks. Relay the missing
    dependencies to the user, get an answer, and only then call again with
    `confirm=True`. A tool call cannot pause to ask mid-flight, and installing
    packages is not something to do unasked.

    Blender is never installed for you. It is a large third-party installer, so the
    report points at blender.org instead. Without it, building, measuring,
    validating and OBJ/STL export all still work; only `lathe_observe`,
    `lathe_compare` and glTF/FBX export need it.

    Args:
        params (SetupInput): confirm, optional seed, optional style, build,
            check_upstream.

    Returns:
        str: An environment table, the upstream project line, and -- when the
        environment can build -- the welcome piece's commit, its style and palette,
        the material map for `lathe_observe`, and the full program text.

    Examples:
        - "set up lathe" -> defaults; relay anything missing before confirming.
        - "give me a different one" -> pass any new seed.
        - "show me a lantern" -> style='lantern'.
        - Don't call with confirm=True until the user has actually agreed.

    Network note:
        With check_upstream, this reads api.github.com for the repo's public
        metadata. That response is treated strictly as data -- a version string and
        a description. Nothing fetched is executed or followed.
    """
    from . import welcome

    a = welcome.audit()
    out: List[str] = ["Lathe %s   Python %s   %s"
                      % (a["version"], a["python"], a["platform"]), ""]

    out.append("environment")
    for import_name, why, pip_name in welcome.DEPS:
        ok = import_name in a["present"]
        out.append("    %-12s %-4s %s" % (import_name, "ok" if ok else "MISSING", why))
    out.append("    %-12s %-4s %s"
               % ("blender", "ok" if a["blender_ok"] else "--",
                  a["blender"] or "not found: lathe_observe, lathe_compare and "
                                  "glTF/FBX export are unavailable"))
    out.append("")

    if params.check_upstream:
        up = welcome.fetch_upstream()
        if up.get("reachable"):
            out.append("project   %s%s"
                       % (up["url"], "   (private)" if up.get("private") else ""))
            if up.get("description"):
                out.append("          %s" % up["description"])
            out.append("          branch %s   last push %s   %s stars   %s open issues"
                       % (up.get("default_branch"), up.get("pushed_at"),
                          up.get("stars"), up.get("open_issues")))
            out.append("          read via %s. This is a local checkout, so "
                       "`git pull` updates it." % up.get("via", "the public API"))
        else:
            out.append("project   %s (not reachable: %s)"
                       % (up["url"], up.get("reason", "")))
        out.append("")

    # -- the gate ---------------------------------------------------------
    if a["missing"]:
        out.append("NOT READY -- %d package(s) missing" % len(a["missing"]))
        out.append("")
        for d in a["missing"]:
            out.append("    %-12s %s" % (d["module"], d["why"]))
        out.append("")
        out.append("    %s" % a["pip_command"])
        out.append("")
        if not params.confirm:
            out.append("Ask the user whether to install these, then call")
            out.append("lathe_setup again with confirm=true. Do not confirm on")
            out.append("their behalf.")
            return "\n".join(out)

        out.append("installing (confirmed)...")
        import subprocess
        args = [sys.executable, "-m", "pip", "install"] + [
            d["pip"] for d in a["missing"]]
        try:
            proc = await asyncio.to_thread(
                subprocess.run, args, capture_output=True, text=True, timeout=900)
            tail = (proc.stdout or "").strip().splitlines()[-4:]
            out += ["    " + line for line in tail]
            again = welcome.audit()
            if again["missing"]:
                out.append("")
                out.append("    still missing: %s"
                           % ", ".join(d["module"] for d in again["missing"]))
                out.append("    Install them yourself and call lathe_setup again.")
                return "\n".join(out)
            out.append("    all dependencies present now")
            a = again
        except Exception as exc:
            out.append("    pip failed: %s: %s" % (type(exc).__name__, exc))
            out.append("    Run the command above yourself, then call this again.")
            return "\n".join(out)
        out.append("")

    if not params.build:
        out.append("READY. Pass build=true for a welcome piece, or write your own "
                   "program and call lathe_apply.")
        return "\n".join(out)

    # -- the welcome piece -------------------------------------------------
    try:
        w = welcome.generate(params.seed, params.style)
    except ValueError as exc:
        return "\n".join(out + ["", "ERROR", "  %s" % exc])

    try:
        commit, solid, _diff, _new = await _off_loop(
            WS.apply, w["program"],
            "welcome piece (%s, seed %s)" % (w["style"], w["seed_label"]),
            branch="welcome")
    except Exception as exc:
        return "\n".join(out + ["", "the welcome program failed to build:", "",
                                _fail(exc), "",
                                "This is a bug in the generator, not in your "
                                "environment. The program text was:", "",
                                w["program"]])

    out += [
        "READY",
        "",
        "your welcome piece   %s, %s palette" % (w["style"], w["palette"]),
        "                     seed %s -- pass a different seed for another"
        % w["seed_label"],
        "                     %s" % "; ".join(w["notes"]),
        "",
        _headline(solid, commit),
        "  on branch %s" % commit.branch,
        "",
    ]
    try:
        out.append(_check(commit.id))
    except Exception as exc:
        out.append("  validation could not run: %s" % exc)

    out += [
        "",
        "materials for lathe_observe:",
        "    %s" % json.dumps(w["materials"]),
        "",
        "next: lathe_observe to see it, lathe_export to write it out, or edit the",
        "program below and lathe_apply it. The seed only chose the numbers -- the",
        "program is ordinary Lathe text.",
        "",
        "-" * 70,
        w["program"].rstrip(),
        "-" * 70,
    ]
    if not a["blender_ok"]:
        out.append("")
        out.append("Blender is not installed, so lathe_observe cannot render this.")
        out.append("Get it from https://www.blender.org/download/ and set "
                   "LATHE_BLENDER, or keep going without renders.")
    return "\n".join(out)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
