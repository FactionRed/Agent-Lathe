"""
Transform rules: E104, W201.

DESIGN.md section 4: "The program declares origin and scale, and the emitter
must honor them. This designs out the most common export bug instead of relying
on memory." E104 and W201 are the check that the emitter actually did.

E104 is the rule that catches "the object is 100x too large" from section 1.
It is also the cheapest rule in the profile: six floats.
"""

from __future__ import annotations

import numpy as np

from ..engine import rule
from ..findings import ERROR, WARN, Raw, Repair
from ..mesh import reference_point
from ._common import loc, mm, num

# --------------------------------------------------------------------- E104


@rule(
    "E104",
    ERROR,
    "Object scale ≠ (1,1,1) or rotation ≠ 0 at emit",
    needs=("mesh.transform",),
    defaults={"scale_tolerance": 1e-6, "rotation_tolerance_deg": 1e-4},
)
def e104(ctx):
    """An unapplied transform means the mesh data and the object disagree about
    units and orientation. Every importer resolves that disagreement its own
    way, and normals, physics bodies and instanced copies each pick a different
    answer."""
    mesh = ctx.mesh
    tr = mesh.transform
    stol = float(ctx.p("scale_tolerance"))
    rtol = np.radians(float(ctx.p("rotation_tolerance_deg")))

    bad_scale = [i for i, s in enumerate(tr.scale) if abs(s - 1.0) > stol]
    bad_rot = [i for i, r in enumerate(tr.rotation_euler) if abs(r) > rtol]
    if not (bad_scale or bad_rot):
        return

    axis = "XYZ"
    problems = []
    if bad_scale:
        problems.append(
            "scale %s on %s"
            % (
                "(%s)" % ", ".join(num(s) for s in tr.scale),
                "".join(axis[i] for i in bad_scale),
            )
        )
    if bad_rot:
        problems.append(
            "rotation %s on %s"
            % (
                "(%s)°" % ", ".join(num(np.degrees(r)) for r in tr.rotation_euler),
                "".join(axis[i] for i in bad_rot),
            )
        )

    lo_l, hi_l = mesh.bbox(world=False)
    lo_w, hi_w = mesh.bbox(world=True)
    local = hi_l - lo_l
    world = hi_w - lo_w
    yield ctx.finding(
        summary="Unapplied object transform at emit: " + ", ".join(problems),
        entity=mesh.name,
        where=[
            ("scale", "(%s)  expected (1, 1, 1)" % ", ".join(num(s) for s in tr.scale)),
            (
                "rotation",
                "(%s)°  expected (0, 0, 0)°"
                % ", ".join(num(np.degrees(r)) for r in tr.rotation_euler),
            ),
            (
                "bbox",
                "mesh %s m  →  world %s m"
                % (" × ".join(num(v) for v in local), " × ".join(num(v) for v in world)),
            ),
        ],
        evidence="scale = (%s), rotation = (%s)°; the mesh measures %s m but exports "
        "as %s m"
        % (
            ", ".join(num(s) for s in tr.scale),
            ", ".join(num(np.degrees(r)) for r in tr.rotation_euler),
            " × ".join(num(v) for v in local),
            " × ".join(num(v) for v in world),
        ),
        data={
            "scale": [float(s) for s in tr.scale],
            "rotation_deg": [float(np.degrees(r)) for r in tr.rotation_euler],
            "bbox_local_m": [float(v) for v in local],
            "bbox_world_m": [float(v) for v in world],
        },
        repair=Repair(
            "transform.apply",
            (Raw(mesh.name),),
            {"scale": bool(bad_scale), "rotation": bool(bad_rot)},
            alternatives=(
                "bake the transform into the program instead: express the size as a "
                "param (DESIGN.md §4.1 makes `600mm` a type, so scale bugs die there)",
            ),
        ),
    )


# --------------------------------------------------------------------- W201


@rule(
    "W201",
    WARN,
    "Origin not at declared reference point",
    needs=("emit.origin",),
    defaults={"tolerance_mm": 1.0},
)
def w201(ctx):
    """`box(...) @ origin(bottom_center)` is a promise about where (0,0,0) sits
    relative to the geometry. Placement code downstream depends on it."""
    mesh = ctx.mesh
    declared = ctx.emit.origin
    if not declared:
        ctx.not_applicable("emit declares no origin reference point")
    want = reference_point(mesh, declared)
    if want is None:
        ctx.not_applicable(
            "unknown origin reference %r — known: bottom_center, top_center, center, "
            "min_corner, max_corner, world_origin" % declared
        )

    tol = float(ctx.p("tolerance_mm")) / 1000.0
    delta = -np.asarray(want, dtype=np.float64)  # origin (0,0,0) minus reference
    dist = float(np.linalg.norm(delta))
    if dist <= tol:
        return

    lo, hi = mesh.bbox(world=False)
    yield ctx.finding(
        summary="Origin is %s from the declared %s" % (mm(dist), declared),
        entity=mesh.name,
        where=[
            ("declared", "%s at %s" % (declared, loc(want))),
            ("actual", "object origin at %s" % loc((0.0, 0.0, 0.0))),
            ("offset", "(%s, %s, %s)" % tuple(mm(d) for d in delta)),
            ("bbox", "%s → %s (mesh space)" % (loc(lo), loc(hi))),
        ],
        evidence="offset = (%s, %s, %s), distance %s, tolerance %s"
        % (mm(delta[0]), mm(delta[1]), mm(delta[2]), mm(dist), mm(tol)),
        data={
            "declared": declared,
            "reference_point": [float(v) for v in want],
            "offset_mm": [float(d) * 1000.0 for d in delta],
            "distance_mm": dist * 1000.0,
            "tolerance_mm": float(ctx.p("tolerance_mm")),
        },
        repair=Repair(
            "transform.set_origin",
            (Raw(mesh.name),),
            {"at": Raw(declared)},
            alternatives=(
                "emit with origin = %s if the current placement is the intended one"
                % _nearest_reference(mesh),
            ),
        ),
    )


def _nearest_reference(mesh) -> str:
    """Which declared reference the geometry actually matches. Half of these
    findings are a wrong declaration, not wrong geometry."""
    best, best_d = "world_origin", float("inf")
    for name in ("bottom_center", "top_center", "center", "min_corner", "max_corner",
                 "world_origin"):
        p = reference_point(mesh, name)
        d = float(np.linalg.norm(p))
        if d < best_d:
            best, best_d = name, d
    return best
