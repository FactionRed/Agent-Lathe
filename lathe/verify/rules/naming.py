"""
Naming rule: N401.

Naming is the cheapest rule in the profile and the one studios care about
first, which is exactly the adoption wedge DESIGN.md section 13 describes:
"the wedge is the verifier and the export hygiene -- those provide value even
to teams that never let an agent model anything."

The convention is data. `[Prefix]_[Name]_[Variant]_[LOD]`, e.g.
`SM_Chair_Wood_LOD0`, is the default; a studio with a different convention
edits the profile parameter, not this module.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from ..engine import rule
from ..findings import NOTE, Raw, Repair

# (label, pattern, hint) -- positional segments, joined by "_"
OBJECT_SEGMENTS: Tuple[Tuple[str, str, str], ...] = (
    ("prefix", r"SM|SK|SKM|SMI|BP|WS", "asset class: SM static mesh, SK skeletal"),
    ("name", r"[A-Z][A-Za-z0-9]*", "PascalCase, no separators"),
    ("variant", r"[A-Z][A-Za-z0-9]*", "PascalCase, no separators"),
    ("lod", r"LOD\d+", "LOD0 for the base mesh"),
)

TEXTURE_SEGMENTS: Tuple[Tuple[str, str, str], ...] = (
    ("prefix", r"T", "textures are prefixed T"),
    ("name", r"[A-Z][A-Za-z0-9]*", "PascalCase, matching the mesh name"),
    ("variant", r"[A-Z][A-Za-z0-9]*", "PascalCase, matching the mesh variant"),
    ("channel", r"D|N|ORM|R|M|AO|E|MASK", "packed channel code"),
)


@rule(
    "N401",
    NOTE,
    "Naming does not match [Prefix]_[Name]_[Variant]_[LOD]",
    needs=(),
    defaults={
        "object_segments": OBJECT_SEGMENTS,
        "texture_segments": TEXTURE_SEGMENTS,
        "default_prefix": "SM",
        "check_textures": True,
    },
)
def n401(ctx):
    """Reports which segment failed and what the corrected name is, because
    "naming convention violation" is not a repair and a corrected string is."""
    segs = tuple(ctx.p("object_segments"))
    yield from _check(ctx, ctx.name, segs, "object", ctx.p("default_prefix"), ctx.emit.lod)
    if ctx.p("check_textures"):
        tsegs = tuple(ctx.p("texture_segments"))
        for tex in ctx.emit.textures:
            yield from _check(ctx, tex.name, tsegs, "texture", "T", ctx.emit.lod)


# "lod" is an initialism, and `[Lod]` in a report about naming conventions is
# not a good look for a rule about naming conventions.
_SEG_DISPLAY = {"lod": "LOD", "uv": "UV"}


def _display(label: str) -> str:
    return _SEG_DISPLAY.get(label, label.capitalize())


def _check(ctx, name: str, segs, kind: str, default_prefix: str, lod: int):
    parts = _split(name)
    pattern = "_".join("[%s]" % _display(s[0]) for s in segs)
    problems: List[str] = []

    if len(parts) != len(segs):
        problems.append(
            "%d segment%s, expected %d (%s)"
            % (len(parts), "" if len(parts) == 1 else "s", len(segs), pattern)
        )
    for i, (label, rx, hint) in enumerate(segs):
        actual = parts[i] if i < len(parts) else None
        if actual is None:
            problems.append("%s missing — %s" % (label, hint))
        elif not re.fullmatch(rx, actual):
            problems.append("%s %r does not match %s — %s" % (label, actual, rx, hint))
    if not problems:
        return

    suggestion = _suggest(parts, segs, default_prefix, lod)
    entity = ctx.name if kind == "object" else "%s/%s" % (ctx.name, name)
    yield ctx.finding(
        summary="Naming: %r does not match %s" % (name, pattern),
        entity=entity,
        where=[
            ("expected", "%s   e.g. %s"
             % (pattern, "SM_Chair_Wood_LOD0" if kind == "object" else "T_Chair_Wood_D")),
            ("actual", "%r → segments [%s]" % (name, ", ".join(parts) or "—")),
            ("problems", problems),
            ("suggest", suggestion),
        ],
        evidence="%d naming problem%s across %d segments; %s name %r, suggested %r"
        % (len(problems), "" if len(problems) == 1 else "s", len(segs), kind, name,
           suggestion),
        data={
            "kind": kind,
            "actual": name,
            "segments": parts,
            "expected_pattern": pattern,
            "problems": problems,
            "suggested": suggestion,
        },
        repair=Repair(
            "emit.rename" if kind == "object" else "texture.rename",
            (Raw(name),),
            {"to": Raw(suggestion)},
            alternatives=(
                "override N401.object_segments in the profile if the studio "
                "convention differs" if kind == "object" else
                "override N401.texture_segments in the profile",
            ),
        ),
    )


def _split(name: str) -> List[str]:
    return [p for p in re.split(r"[_\s\-.]+", name) if p]


def _pascal(word: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]", "", word)
    if not cleaned:
        return "Part"
    if cleaned[0].isdigit():
        # `Cube.001` -> `V001`. A segment must start with a letter, and dropping
        # the number would lose the only thing distinguishing two assets.
        cleaned = "V" + cleaned
    return cleaned[0].upper() + cleaned[1:]


def _suggest(parts: Sequence[str], segs, default_prefix: str, lod: int) -> str:
    """Build the corrected name from whatever the current one supplies. This is
    the executable half of the finding, so it must always produce something
    valid, even from `Cube.001`."""
    rest = list(parts)
    prefix_rx = segs[0][1]
    if rest and re.fullmatch(prefix_rx, rest[0], flags=re.IGNORECASE):
        prefix = rest.pop(0).upper()  # `sm_chair_wood` meant `SM`, not a new name
    else:
        prefix = default_prefix

    last_label, last_rx = segs[-1][0], segs[-1][1]
    if rest and re.fullmatch(last_rx, rest[-1], flags=re.IGNORECASE):
        tail = rest.pop(-1).upper()
    elif last_label == "lod":
        tail = "LOD%d" % lod
    else:
        tail = "D"

    middle_count = len(segs) - 2
    middles = [_pascal(p) for p in rest]
    while len(middles) < middle_count:
        middles.append("Base")
    if len(middles) > middle_count:
        middles = middles[: middle_count - 1] + ["".join(middles[middle_count - 1 :])]
    return "_".join([prefix] + middles + [tail])
