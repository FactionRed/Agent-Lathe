"""
First-run setup: environment audit, upstream version check, and a welcome model
that differs per user.

The welcome model is generated as *program text*, not as geometry. A new user's
first artefact should be something they can read, edit and re-apply -- which is
the whole premise of the system. Handing back an opaque mesh would teach nothing.

Everything is derived from one seed, so a given user gets the same piece every
time until they ask for another. The seed comes from the machine and account name
by default, which is stable per user and different across users.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import platform
import random
import shutil
import sys
from typing import Dict, List, Optional, Tuple

REPO = "FactionRed/Agent-3D"
REPO_URL = "https://github.com/%s" % REPO
API_URL = "https://api.github.com/repos/%s" % REPO

# name -> (import name, why it is needed, pip name)
DEPS: Tuple[Tuple[str, str, str], ...] = (
    ("manifold3d", "the boolean kernel -- nothing works without it", "manifold3d"),
    ("numpy", "mesh arrays", "numpy"),
    ("PIL", "contact sheets in lathe_observe and lathe_compare", "pillow"),
    ("mcp", "this server", "mcp[cli]"),
)


# ------------------------------------------------------------------ audit

def audit() -> Dict[str, object]:
    """What is installed, what is missing, and what to run about it."""
    from .. import __version__

    missing: List[Dict[str, str]] = []
    present: List[str] = []
    for import_name, why, pip_name in DEPS:
        if importlib.util.find_spec(import_name) is None:
            missing.append({"module": import_name, "why": why, "pip": pip_name})
        else:
            present.append(import_name)

    blender = None
    try:
        from ..build.bevel import blender_path
        blender = blender_path()
    except Exception:
        pass

    ready = not missing
    return {
        "version": __version__,
        "python": "%d.%d.%d" % sys.version_info[:3],
        "platform": "%s %s" % (platform.system(), platform.release()),
        "present": present,
        "missing": missing,
        "blender": blender,
        "blender_ok": blender is not None,
        "ready_to_build": ready,
        "ready_to_render": ready and blender is not None,
        "pip_command": (
            "pip install " + " ".join(
                ('"%s"' % d["pip"]) if "[" in d["pip"] else d["pip"]
                for d in missing)) if missing else None,
    }


def fetch_upstream(timeout: float = 6.0) -> Dict[str, object]:
    """Read the repo's public metadata.

    Treated strictly as data. Nothing fetched here is executed, and no field is
    interpreted as an instruction -- it is a version string and a description.
    """
    import json
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(
            API_URL, headers={"Accept": "application/vnd.github+json",
                              "User-Agent": "lathe-setup"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {
            "reachable": True,
            "url": REPO_URL,
            "description": (data.get("description") or "")[:300],
            "default_branch": data.get("default_branch"),
            "pushed_at": data.get("pushed_at"),
            "stars": data.get("stargazers_count"),
            "open_issues": data.get("open_issues_count"),
        }
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            reason = ("404 -- the repository is private or renamed, so the "
                      "unauthenticated API cannot see it. Nothing local is "
                      "affected.")
        elif exc.code in (403, 429):
            reason = ("%d -- GitHub rate-limited this request. Try later; nothing "
                      "local is affected." % exc.code)
        else:
            reason = "HTTP %d" % exc.code
        return {"reachable": False, "url": REPO_URL, "reason": reason}
    except Exception as exc:
        return {
            "reachable": False,
            "url": REPO_URL,
            "reason": "%s: %s -- offline, or the host is unreachable. Nothing "
                      "local is affected." % (type(exc).__name__, exc),
        }


# ------------------------------------------------------------------ seed

def default_seed() -> int:
    """Stable for one user on one machine, different across users."""
    raw = "|".join([
        os.environ.get("USERNAME") or os.environ.get("USER") or "anon",
        platform.node() or "host",
    ])
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12], 16)


def seed_label(seed: int) -> str:
    return "%08x" % (seed & 0xFFFFFFFF)


# ---------------------------------------------------------------- palettes

PALETTES: Tuple[Tuple[str, Dict[str, str]], ...] = (
    ("oak and iron", {"primary": "wood_mid", "secondary": "metal", "accent": "wood_dark"}),
    ("pale ash", {"primary": "wood_light", "secondary": "wood_mid", "accent": "leather"}),
    ("walnut", {"primary": "wood_dark", "secondary": "wood_mid", "accent": "metal"}),
    ("glazed blue", {"primary": "ceramic_blue", "secondary": "ceramic", "accent": "metal"}),
    ("bone china", {"primary": "ceramic", "secondary": "wood_light", "accent": "leather"}),
    ("verdigris", {"primary": "metal", "secondary": "leaf", "accent": "wood_dark"}),
    ("orchard", {"primary": "wood_light", "secondary": "leaf", "accent": "apple"}),
    ("tack room", {"primary": "leather", "secondary": "wood_dark", "accent": "metal"}),
)

STYLES = ("stool", "planter", "lantern", "pedestal")


# --------------------------------------------------------------- generator

def generate(seed: Optional[int] = None,
             style: Optional[str] = None) -> Dict[str, object]:
    """Build a welcome program. Returns the text plus what was chosen."""
    if seed is None:
        seed = default_seed()
    rng = random.Random(seed)
    style = style or rng.choice(STYLES)
    if style not in STYLES:
        raise ValueError(
            "unknown style %r. Choose one of %s, or omit it to let the seed pick."
            % (style, ", ".join(STYLES)))
    pal_name, pal = PALETTES[rng.randrange(len(PALETTES))]

    builder = {
        "stool": _stool, "planter": _planter,
        "lantern": _lantern, "pedestal": _pedestal,
    }[style]
    body, materials, notes = builder(rng, pal)

    header = "\n".join([
        "# Lathe welcome piece -- style %r, palette %r" % (style, pal_name),
        "# seed %s. Same seed, same piece. Pass a different seed for another." % seed_label(seed),
        "# Everything below is ordinary Lathe program text: edit it and re-apply.",
        "",
    ])
    program = header + body
    return {
        "seed": seed, "seed_label": seed_label(seed), "style": style,
        "palette": pal_name, "materials": materials, "program": program,
        "notes": notes,
    }


def _r(rng, lo, hi, nd=4):
    return round(rng.uniform(lo, hi), nd)


def _stool(rng, pal):
    seat_r = _r(rng, 0.145, 0.185)
    height = _r(rng, 0.44, 0.60)
    legs = rng.choice((3, 4))
    round_seat = rng.random() < 0.55
    taper = _r(rng, 0.55, 0.85, 2)
    seat_t = _r(rng, 0.030, 0.046)
    leg_r = _r(rng, 0.019, 0.026)
    ring = rng.random() < 0.6
    ring_z = _r(rng, 0.14, 0.22)

    seat = ('cylinder(%s, %s, segments=48, center=False, tag="top", '
            'name="SM_Welcome_Stool_LOD0").at(z=%s)'
            % (seat_t, seat_r, round(height - seat_t, 4))) if round_seat else (
        'box(%s, %s, %s, tag="top", name="SM_Welcome_Stool_LOD0").at(z=%s)'
        % (round(seat_r * 1.8, 4), round(seat_r * 1.8, 4), seat_t,
           round(height - seat_t / 2, 4)))

    lines = [
        "seat_r = %s" % seat_r,
        "height = %s" % height,
        "leg_r  = %s" % leg_r,
        "",
        "top = %s" % seat,
        "",
        "# %d legs on a circle, each tapering toward the floor." % legs,
        "import math",
        "legs = []",
        "for i in range(%d):" % legs,
        "    a = 2 * math.pi * i / %d" % legs,
        "    x = math.cos(a) * seat_r * 0.66",
        "    y = math.sin(a) * seat_r * 0.66",
        "    legs.append(cylinder(height, leg_r * %s, leg_r, segments=20,"
        % taper,
        "                         center=False, tag=\"leg\").at(x=x, y=y))",
        "",
        "piece = union_all([top] + legs, name=\"SM_Welcome_Stool_LOD0\")",
    ]
    if ring:
        lines += [
            "",
            "# A stretcher ring braces the legs. Revolved, so it is one surface.",
            "ring = revolve([(seat_r * 0.52, 0.0), (seat_r * 0.60, 0.0),",
            "                (seat_r * 0.60, %s), (seat_r * 0.52, %s)],"
            % (round(0.022, 4), round(0.022, 4)),
            "               segments=40, tag=\"ring\").at(z=%s)" % ring_z,
            "piece = piece + ring",
        ]
    lines += [""]
    if round_seat:
        # No face-count assertion on a cylinder. Its side is curved, so it
        # fragments into one facet per segment and the count tracks tessellation
        # rather than shape -- DESIGN.md 5.3. Assert what actually holds.
        lines += [
            "# A round seat is a cylinder: its side is one facet per segment, so a",
            "# logical face count would track tessellation, not shape. Assert the",
            "# invariant instead.",
            "if piece.components() != 1:",
            '    raise LatheError("a leg is not reaching the seat")',
        ]
    else:
        lines += [
            "assert_count(piece.faces(tag=\"top\"), 6, \"seat faces\")",
        ]
    lines += [
        "",
        "piece = bevel(piece, width=mm(%s), segments=2, min_dihedral=55)"
        % _r(rng, 1.2, 2.6, 1),
        "",
        'spec = {"components": 1, "origin": "bottom_center",',
        '        "budget": {"tris_lod0": 12000}}',
        "result = piece",
    ]
    mats = {"top": pal["primary"], "leg": pal["secondary"],
            "ring": pal["accent"], "bevel": pal["primary"]}
    return "\n".join(lines) + "\n", mats, [
        "%s seat, %d legs, %s" % ("round" if round_seat else "square", legs,
                                  "stretcher ring" if ring else "no stretcher")]


def _planter(rng, pal):
    top_r = _r(rng, 0.10, 0.155)
    height = _r(rng, 0.11, 0.22)
    wall = _r(rng, 0.006, 0.011)
    lip = rng.random() < 0.65
    foot = rng.random() < 0.5
    waist = _r(rng, 0.62, 0.86, 2)

    lines = [
        "top_r  = %s" % top_r,
        "height = %s" % height,
        "wall   = %s" % wall,
        "",
        "# A revolved profile: (radius, height) pairs, spun about Z.",
        "profile = [",
        "    (0.0, 0.0),",
        "    (top_r * %s, 0.0)," % waist,
        "    (top_r, height * 0.62),",
        "    (top_r%s, height)," % (" * 1.06" if lip else ""),
        "    (top_r - wall%s, height)," % (" * 1.06" if lip else ""),
        "    (top_r * %s - wall, height * 0.60)," % waist,
        "    (0.0, wall * 2),",
        "]",
        'pot = revolve(profile, segments=%d, tag="pot",' % rng.choice((48, 56, 64)),
        '              name="SM_Welcome_Planter_LOD0")',
    ]
    if foot:
        lines += [
            "",
            "# A short foot ring lifts it off the surface. Its top must sit ABOVE",
            "# z=0 so it overlaps the pot's base -- placed flush, or below it, the",
            "# foot is a separate component (DESIGN.md 13).",
            "lap    = 0.003",
            "foot_h = %s" % _r(rng, 0.010, 0.018),
            'foot = cylinder(foot_h, top_r * %s, top_r * %s, segments=44,'
            % (round(waist * 0.92, 3), round(waist * 0.82, 3)),
            '                center=False, tag="foot").at(z=-(foot_h - lap))',
            "pot = pot + foot",
        ]
    lines += [
        "",
        "# Curved surfaces fragment into one facet per segment, so assert the",
        "# invariants that hold rather than a logical face count.",
        "if pot.components() != 1:",
        '    raise LatheError("the foot detached from the pot")',
        "",
        'spec = {"components": 1, "origin": "bottom_center",',
        '        "budget": {"tris_lod0": 12000}}',
        "result = pot",
    ]
    mats = {"pot": pal["primary"], "foot": pal["accent"], "bevel": pal["primary"]}
    return "\n".join(lines) + "\n", mats, [
        "%s profile, %s, %s" % ("flared" if lip else "straight",
                                "footed" if foot else "flat-bottomed",
                                "waisted" if waist < 0.75 else "full")]


def _lantern(rng, pal):
    w = _r(rng, 0.085, 0.125)
    body_h = _r(rng, 0.13, 0.21)
    sides = rng.choice((4, 6, 8))
    roof_h = _r(rng, 0.045, 0.080)
    base_h = _r(rng, 0.016, 0.030)
    glazed = rng.random() < 0.7

    lines = [
        "w      = %s" % w,
        "body_h = %s" % body_h,
        "",
        "# Stacked parts must overlap. Meeting exactly on a shared plane is",
        "# measure-zero contact, and whether a boolean joins it or leaves two",
        "# components depends on the exact coordinates -- DESIGN.md 13.",
        "lap = 0.004",
        "",
        "# A %d-sided body: a low-segment cylinder is a prism." % sides,
        'body = cylinder(body_h + lap * 2, w, w, segments=%d, center=False,' % sides,
        '                tag="body", name="SM_Welcome_Lantern_LOD0").at(z=%s - lap)'
        % base_h,
        "",
        'base = cylinder(%s + lap, w * %s, w * %s, segments=%d, center=False,'
        % (base_h, _r(rng, 1.10, 1.28, 2), _r(rng, 1.02, 1.14, 2), sides),
        '                tag="base")',
        "",
        "# The roof is a cone: a cylinder with a smaller top radius.",
        'roof = cylinder(%s + lap, w * %s, w * %s, segments=%d, center=False,'
        % (roof_h, _r(rng, 1.12, 1.30, 2), _r(rng, 0.10, 0.30, 2), sides),
        '                tag="roof").at(z=%s - lap)' % round(base_h + body_h, 4),
        "",
        "piece = union_all([base, body, roof], name=\"SM_Welcome_Lantern_LOD0\")",
    ]
    if glazed:
        # Window width is a fraction of the face *chord*, not of the radius. A
        # fixed fraction of w cuts straight through the corner posts once the
        # sides get narrow, which detaches the roof from the base.
        chord_frac = _r(rng, 0.42, 0.60, 2)
        lines += [
            "",
            "# Cut window openings, leaving corner posts. The cutter's own faces",
            "# become the reveal, and they stay addressable as #window.",
            "import math",
            "# Size the opening against the face chord, so the posts survive at any",
            "# side count. A fixed fraction of the radius does not.",
            "chord = 2 * w * math.sin(math.pi / %d)" % sides,
            "for i in range(%d):" % sides,
            "    a = 360.0 * i / %d" % sides,
            "    cut = box(chord * %s, w * 3, body_h * %s, tag=\"window\")"
            % (chord_frac, _r(rng, 0.46, 0.62, 2)),
            "    cut = cut.rotate((0, 0, a)).at(z=%s)"
            % round(base_h + body_h * 0.55, 4),
            "    piece = piece - cut",
            "",
            "if piece.components() != 1:",
            '    raise LatheError("the windows severed the corner posts")',
        ]
    lines += [
        "",
        "piece = bevel(piece, width=mm(%s), segments=2, min_dihedral=50,"
        % _r(rng, 0.8, 1.8, 1),
        "              min_length=mm(3))",
        "",
        'spec = {"components": 1, "origin": "bottom_center",',
        '        "budget": {"tris_lod0": 14000}}',
        "result = piece",
    ]
    mats = {"body": pal["primary"], "roof": pal["secondary"],
            "base": pal["secondary"], "window": pal["accent"],
            "bevel": pal["primary"]}
    return "\n".join(lines) + "\n", mats, [
        "%d sides, %s" % (sides, "glazed openings" if glazed else "solid body")]


def _pedestal(rng, pal):
    tiers = rng.choice((2, 3))
    height = _r(rng, 0.42, 0.72)
    # Derived from height, not drawn independently. Independent draws produced a
    # 6:1 slenderness that read as a bollard rather than a pedestal.
    col_r = round(height * _r(rng, 0.10, 0.145, 3), 4)
    base_w = round(height * _r(rng, 0.34, 0.46, 3), 4)
    square = rng.random() < 0.5
    fluted = rng.random() < 0.45

    cap_t = _r(rng, 0.026, 0.042)
    lines = [
        "base_w = %s" % base_w,
        "col_r  = %s" % col_r,
        "height = %s" % height,
        "cap_t  = %s" % cap_t,
        "",
        "# Parts must OVERLAP, not abut. Two solids meeting exactly on a shared",
        "# plane have measure-zero contact, and a boolean treats that as",
        "# disconnected -- DESIGN.md 13. This lap is what keeps it one piece.",
        "lap = 0.006",
        "",
        "# %d stacked tiers, each a little smaller than the one below." % tiers,
        "tiers, z = [], 0.0",
        "for i in range(%d):" % tiers,
        "    t = %s * (0.80 ** i)" % _r(rng, 0.028, 0.046),
        "    s = base_w * (0.82 ** i)",
        "    tiers.append(%s)" % (
            'box(s, s, t + lap, tag="base").at(z=z + t / 2)' if square else
            'cylinder(t + lap, s / 2, s / 2, segments=44, center=False, '
            'tag="base").at(z=z)'),
        "    z += t",
        "",
        "# The column runs from inside the top tier to inside the cap.",
        "col_bottom = z - lap",
        "col_h = (height - cap_t + lap) - col_bottom",
        'column = cylinder(col_h, col_r * %s, col_r, segments=%d, center=False,'
        % (_r(rng, 0.86, 1.0, 2), 8 if fluted else 40),
        '                  tag="column").at(z=col_bottom)',
        "",
        'cap = %s' % (
            'box(base_w * 0.66, base_w * 0.66, cap_t, tag="cap")'
            '.at(z=height - cap_t / 2)'
            if square else
            'cylinder(cap_t, base_w * 0.34, base_w * 0.30, segments=44, '
            'center=False, tag="cap").at(z=height - cap_t)'),
        "",
        'piece = union_all(tiers + [column, cap], name="SM_Welcome_Pedestal_LOD0")',
        "",
        "if piece.components() != 1:",
        '    raise LatheError("a tier is not touching the column")',
        "",
        "piece = bevel(piece, width=mm(%s), segments=2, min_dihedral=52,"
        % _r(rng, 1.0, 2.2, 1),
        "              min_length=mm(4))",
        "",
        'spec = {"components": 1, "origin": "bottom_center",',
        '        "budget": {"tris_lod0": 14000}}',
        "result = piece",
    ]
    mats = {"base": pal["secondary"], "column": pal["primary"],
            "cap": pal["secondary"], "bevel": pal["primary"]}
    return "\n".join(lines) + "\n", mats, [
        "%d tiers, %s plan, %s column" % (
            tiers, "square" if square else "round",
            "fluted" if fluted else "smooth")]
