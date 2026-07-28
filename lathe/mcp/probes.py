"""
The `query` expression language -- DESIGN.md section 9's "small tool surface plus
a rich language".

One tool, many expressions, rather than twenty tools. Unknown expressions return
the full grammar, because an agent that guessed wrong needs the list more than it
needs an apology.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple

from ..build.solid import LatheError, Solid
from ..perceive import tier1, tier2

# name -> (callable(solid) -> str, one-line help)
_SCALARS: Dict[str, Tuple[Any, str]] = {}


def _scalar(name, help_text):
    def deco(fn):
        _SCALARS[name] = (fn, help_text)
        return fn
    return deco


@_scalar("bbox", "bounding box, centre and diagonal in metres")
def _bbox(s: Solid):
    return tier1.bbox(s.mesh())


@_scalar("volume", "enclosed volume, exact closed form")
def _volume(s: Solid):
    return tier1.volume(s.mesh())


@_scalar("surface_area", "total surface area")
def _area(s: Solid):
    return tier1.surface_area(s.mesh())


@_scalar("counts", "verts, edges, logical faces, components, genus")
def _counts(s: Solid):
    return tier1.counts(s.mesh())


@_scalar("watertight", "closed, edge-manifold, oriented, outward")
def _wt(s: Solid):
    return tier1.is_watertight(s.mesh())


@_scalar("degenerates", "loose verts and edges, zero-area faces (E105)")
def _deg(s: Solid):
    return tier1.degenerates(s.mesh())


@_scalar("center_of_mass", "volume centroid, and where it sits in the bbox")
def _com(s: Solid):
    return tier1.center_of_mass(s.mesh())


@_scalar("topology", "triangle/quad/n-gon histogram and valence spread")
def _topo(s: Solid):
    return tier2.topology_summary(s.mesh())


@_scalar("parts", "triangle and volume distribution per tag")
def _parts(s: Solid):
    return tier2.part_summary(s.mesh())


@_scalar("materials", "material assignment table")
def _mats(s: Solid):
    return tier2.material_table(s.mesh())


@_scalar("uv", "per-island UV stats, or a note that there is no UV set")
def _uv(s: Solid):
    return tier2.uv_summary(s.mesh())


def _tags(s: Solid) -> str:
    by = {}
    for pid, tag in s.tags.items():
        by.setdefault(tag, []).append(pid)
    lines = ["  tags on %s: %d distinct" % (s.name, len(by))]
    for tag in sorted(by):
        lines.append("      %-18s part id(s) %s"
                     % (tag, ", ".join(str(i) for i in sorted(by[tag]))))
    return "\n".join(lines)


_SEL = re.compile(r"^(faces|edges)\s*\((.*)\)$", re.S)
_SYM = re.compile(r"^symmetry\s*\(\s*([XYZ]{2})\s*\)$", re.I)
_RAY = re.compile(r"^raycast\s*\((.*)\)$", re.S)


def grammar() -> str:
    lines = ["Valid query expressions:", ""]
    for name in sorted(_SCALARS):
        lines.append("  %-16s %s" % (name, _SCALARS[name][1]))
    lines += [
        "  %-16s %s" % ("tags", "every tag and the part ids behind it"),
        "  %-16s %s" % ("symmetry(YZ)", "mirror symmetry 0..1 about a plane"),
        "",
        "  faces(tag=seat)            logical face count for a selector",
        "  faces(normal=0,0,1)        faces pointing a direction",
        "  faces(z_gt=0.4)            faces above a height, metres",
        "  edges(dihedral_gt=60)      sharp edges",
        "  edges(dihedral_gt=60, length_gt=0.02)",
        "  edges(tag=seat)            edges with that tag on both sides",
        "  raycast(from=0,-2,0.3, dir=0,1,0)",
        "  raycast(from=0,-2000,300, dir=0,1,0, unit=mm)",
        "",
        "Lengths are metres unless you pass unit=mm/cm/in/ft. Angles are degrees.",
    ]
    return "\n".join(lines)


def _kwargs(text: str) -> Dict[str, Any]:
    """Parse `a=1, b=2,3,4, tag=seat` into kwargs. Tuples are bare comma lists."""
    out: Dict[str, Any] = {}
    for part in re.split(r",(?=\s*[A-Za-z_]+\s*=)", text):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise LatheError(
                "cannot read %r. Selector arguments are name=value pairs, e.g. "
                "faces(tag=seat) or edges(dihedral_gt=60)." % part)
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        nums = [x.strip() for x in v.split(",")]
        try:
            vals = [float(x) for x in nums]
            out[k] = vals[0] if len(vals) == 1 else tuple(vals)
        except ValueError:
            out[k] = v.strip("'\"")
    return out


def evaluate(solid: Solid, expr: str) -> str:
    expr = (expr or "").strip()
    if not expr:
        raise LatheError("empty query.\n\n" + grammar())

    if expr in _SCALARS:
        return str(_SCALARS[expr][0](solid)).rstrip()
    if expr == "tags":
        return _tags(solid)

    m = _SYM.match(expr)
    if m:
        return str(tier1.symmetry(solid.mesh(), m.group(1).upper())).rstrip()

    m = _RAY.match(expr)
    if m:
        kw = _kwargs(m.group(1))
        if "from" not in kw or "dir" not in kw:
            raise LatheError(
                "raycast needs from= and dir=, e.g. "
                "raycast(from=0,-2,0.3, dir=0,1,0)")
        # The probe layer refuses a bare tuple on purpose -- section 4.1 makes an
        # unmarked length a type error. State the unit rather than route around it.
        return str(tier1.raycast(solid.mesh(), kw["from"], kw["dir"],
                                 unit=str(kw.get("unit", "m")))).rstrip()

    m = _SEL.match(expr)
    if m:
        domain, argtext = m.group(1), m.group(2)
        kw = _kwargs(argtext)
        sel = (solid.faces(**kw) if domain == "faces" else solid.edges(**kw))
        lines = ["  %s" % sel.expr,
                 "      matched            %d %s"
                 % (sel.count(), "logical face(s)" if domain == "faces" else "edge(s)")]
        if domain == "faces":
            lines.append("      triangles          %d" % sel.n_tris)
            if sel.groups:
                shown = ", ".join("(%d,%d)" % g for g in sel.groups[:8])
                more = "" if len(sel.groups) <= 8 else "  +%d more" % (len(sel.groups) - 8)
                lines.append("      (part, face) keys  %s%s" % (shown, more))
        return "\n".join(lines)

    raise LatheError("unknown query %r.\n\n%s" % (expr, grammar()))
