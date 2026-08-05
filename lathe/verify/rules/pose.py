"""
Articulation rules: E501, E502, W503, W504, W505.

DESIGN.md section 7's table stops at the static asset. These grade the *rig* an
asset was posed by, against what the program declared about it -- the same
comparison every other rule makes, applied to joints instead of triangles.

**Nothing here knows what a body is.** No rule looks for a joint called
"shoulder", counts limbs, assumes two of anything, assumes one root, assumes a
chain is short, or assumes symmetry is bilateral. Those are all statements about
vertebrates, and this kernel is at least as likely to be pointed at a lamp arm,
a spider, a starfish, a tentacle or a mechanism. Every plan-shaped question is
answered by a declaration in `ArticulationSpec` and by nothing else:

  * how many roots the rig has            -> `roots`, compared not assumed
  * whether it is symmetric, and how      -> `SymmetryDecl`, including `none`
  * where a named tip is supposed to end  -> `ReachDecl`

A rule that inferred the plan from the joint names would be the exact failure
this subsystem exists to prevent: a confident answer about the wrong thing.

Forward kinematics is imported from `lathe.pose` rather than reimplemented.
There is already a second implementation in the viewer, cross-checked against
this one on load; a third hand-rolled copy here would be a liability rather than
independent evidence. The import is lazy, so `lathe.verify` still loads for an
asset that declares no articulation at all.
"""

from __future__ import annotations

import math

import numpy as np

from ..engine import rule
from ..findings import ERROR, WARN, Raw, Repair
from ._common import num

_DEG = 180.0 / math.pi


def _articulation(ctx):
    """The declaration, or *not checked*. Never inferred from the geometry."""
    art = ctx.emit.articulation
    if art is None or not art.joints:
        ctx.not_applicable(
            "emit declares no articulation — add `articulation` to the emit "
            "spec, e.g. from `verify.mesh.articulation_from_rig(rig, pose)`"
        )
    return art


def _world(art):
    """Joint name -> world 4x4 at the declared pose, via `lathe.pose`."""
    from ...pose import Joint, Skeleton, forward

    sk = Skeleton(tuple(
        Joint(name=j.name, parent=j.parent, offset=tuple(j.offset),
              axis=tuple(j.axis), limit=tuple(j.limit))
        for j in art.joints
    ))
    return sk, forward(sk, art.pose())


# --------------------------------------------------------------------- E501


@rule(
    "E501",
    ERROR,
    "Joint angle outside its declared limit",
    needs=("emit.articulation",),
    defaults={"max_report": 8, "tolerance_deg": 1e-6},
)
def e501(ctx):
    """The rig said how far the joint goes. The emitted pose went further.

    Exact, and independent of body plan: a limit is a property of one hinge, not
    of the creature it belongs to. `pose.solve` clamps, so this catches the
    poses nothing clamped -- one typed by a human, lifted off a photograph, or
    retargeted from a rig with a different range.
    """
    art = _articulation(ctx)
    tol = float(ctx.p("tolerance_deg")) / _DEG

    bad = []
    for j in art.joints:
        lo, hi = float(j.limit[0]), float(j.limit[1])
        a = float(j.angle)
        if a < lo - tol:
            bad.append((j, a, lo - a, "below"))
        elif a > hi + tol:
            bad.append((j, a, a - hi, "above"))
    if not bad:
        return

    bad.sort(key=lambda t: -t[2])
    rows = ["%-14s %9.3f°  %s its %s bound by %.3f°"
            % (j.name, a * _DEG, way, "lower" if way == "below" else "upper",
               over * _DEG)
            for j, a, over, way in bad[: ctx.p("max_report")]]
    worst = bad[0]
    yield ctx.finding(
        summary="%d joint%s outside the declared limit, worst %s by %s°"
        % (len(bad), "" if len(bad) == 1 else "s", worst[0].name,
           num(worst[2] * _DEG)),
        entity="%s/%s" % (ctx.name, worst[0].name),
        where=[("joints", rows),
               ("declared", "%s allows %.3f° … %.3f°"
                % (worst[0].name, worst[0].limit[0] * _DEG, worst[0].limit[1] * _DEG))],
        evidence="%d of %d declared joints out of range; worst %s at %s° against "
        "[%s°, %s°], over by %s°"
        % (len(bad), len(art.joints), worst[0].name, num(worst[1] * _DEG),
           num(worst[0].limit[0] * _DEG), num(worst[0].limit[1] * _DEG),
           num(worst[2] * _DEG)),
        data={
            "count": len(bad),
            "joints": [{"joint": j.name, "angle_deg": a * _DEG,
                        "limit_deg": [j.limit[0] * _DEG, j.limit[1] * _DEG],
                        "over_deg": over * _DEG} for j, a, over, _ in bad],
        },
        repair=Repair(
            "pose.clamp",
            (Raw(ctx.name),),
            {"joints": Raw("[%s]" % ", ".join(j.name for j, _, _, _ in bad))},
            alternatives=(
                "widen the declaration if the limit is what is wrong: a range "
                "copied from another rig is the usual cause",
            ),
        ),
    )


# --------------------------------------------------------------------- E502


@rule(
    "E502",
    ERROR,
    "Declared reach target missed",
    needs=("emit.articulation", "emit.articulation.reach"),
    defaults={"max_report": 6},
)
def e502(ctx):
    """`assert reach(ankle, target) < 2mm`, checked on the emitted pose.

    The IK counterpart of E107: the program stated where a named tip ends up,
    so the gate measures whether it does. Which tip and which target are the
    program's business -- a foot, a fingertip, a tentacle end or a drill bit
    are all just a named joint here.
    """
    art = _articulation(ctx)
    if not art.reach:
        ctx.not_applicable("emit declares no reach targets")
    _, world = _world(art)

    misses = []
    for r in art.reach:
        if r.tip not in world:
            misses.append((r, None, float("inf")))
            continue
        got = world[r.tip][:3, 3]
        d = float(np.linalg.norm(got - np.asarray(r.target, dtype=np.float64)))
        if d > float(r.tolerance):
            misses.append((r, got, d))
    if not misses:
        return

    misses.sort(key=lambda t: -t[2])
    rows = []
    for r, got, d in misses[: ctx.p("max_report")]:
        if got is None:
            rows.append("%-12s no such joint in the declared rig" % r.tip)
        else:
            rows.append("%-12s off by %s mm (tolerance %s mm)"
                        % (r.tip, num(d * 1000.0), num(r.tolerance * 1000.0)))
    worst = misses[0]
    yield ctx.finding(
        summary="%d reach target%s missed, worst %s by %s mm"
        % (len(misses), "" if len(misses) == 1 else "s", worst[0].tip,
           num(worst[2] * 1000.0) if worst[1] is not None else "n/a"),
        entity="%s/%s" % (ctx.name, worst[0].tip),
        where=[("targets", rows)],
        evidence="%d of %d declared targets missed" % (len(misses), len(art.reach)),
        data={"count": len(misses),
              "targets": [{"tip": r.tip, "error_mm": None if g is None else d * 1000.0,
                           "tolerance_mm": r.tolerance * 1000.0}
                          for r, g, d in misses]},
        repair=Repair(
            "pose.solve",
            (Raw(ctx.name),),
            {"tip": Raw(worst[0].tip),
             "target": Raw("(%s)" % ", ".join(num(v) for v in worst[0].target))},
            alternatives=(
                "the chain may not reach: check the limits before moving the "
                "target, since a one-sided limit halves the workspace",
            ),
        ),
    )


# --------------------------------------------------------------------- W503


@rule(
    "W503",
    WARN,
    "Emitted pose rests against a declared joint limit",
    needs=("emit.articulation",),
    defaults={"max_report": 8, "tolerance_deg": 1e-6},
)
def w503(ctx):
    """A joint pinned exactly on its bound usually means the solver ran out of
    range, not that anyone wanted that precise angle.

    Not an error: a hinge deliberately posed fully open is legitimate, and only
    the author knows which it was. It is worth seeing because the alternative is
    discovering it when the pose is one degree from impossible.
    """
    art = _articulation(ctx)
    tol = float(ctx.p("tolerance_deg")) / _DEG

    pinned = []
    for j in art.joints:
        lo, hi = float(j.limit[0]), float(j.limit[1])
        if lo == hi:
            continue                              # welded, not pinned
        a = float(j.angle)
        if abs(a - lo) <= tol:
            pinned.append((j, a, "lower"))
        elif abs(a - hi) <= tol:
            pinned.append((j, a, "upper"))
    if not pinned:
        return

    rows = ["%-14s %9.3f°  on its %s bound" % (j.name, a * _DEG, side)
            for j, a, side in pinned[: ctx.p("max_report")]]
    yield ctx.finding(
        summary="%d joint%s resting on a declared limit"
        % (len(pinned), "" if len(pinned) == 1 else "s"),
        entity="%s/%s" % (ctx.name, pinned[0][0].name),
        where=[("joints", rows)],
        evidence="%d of %d declared joints pinned at a bound"
        % (len(pinned), len(art.joints)),
        data={"count": len(pinned),
              "joints": [{"joint": j.name, "angle_deg": a * _DEG, "bound": side}
                         for j, a, side in pinned]},
        repair=Repair(
            "pose.solve",
            (Raw(ctx.name),),
            {"seed": Raw("away from the bound")},
            alternatives=(
                "widen the limit if the range is the thing that is wrong",
                "leave it if the joint is meant to be fully open here",
            ),
        ),
    )


# --------------------------------------------------------------------- W504


@rule(
    "W504",
    WARN,
    "Root joint count differs from the declared expectation",
    needs=("emit.articulation", "emit.articulation.roots"),
    defaults={},
)
def w504(ctx):
    """A joint with no parent is a root, and the count is a body-plan fact.

    Deliberately compared rather than assumed to be 1. One root fits most
    creatures; several are correct for a floating orb, a detached jaw, a swarm,
    or two halves of a mechanism that never touch. The failure this catches is
    the silent one -- a limb whose parent went missing becomes its own root and
    then moves independently of the body, which no still frame will show.
    """
    art = _articulation(ctx)
    declared = art.roots
    if declared is None:
        ctx.not_applicable(
            "emit declares no root count — add `roots: n` to the articulation "
            "spec (one for most creatures, more for a detached or multi-part rig)"
        )
    roots = [j.name for j in art.joints if j.parent is None]
    declared = int(declared)
    if len(roots) == declared:
        return

    # Which root is the wrong one is genuinely unknowable here -- a root is just
    # a joint with no parent, and nothing marks the intended one. Say that,
    # rather than naming every root in a reparent call and telling the author to
    # reattach the body to something.
    if len(roots) > declared:
        repair = Repair(
            "rig.reparent",
            (Raw(ctx.name),),
            {"joints": Raw("[%s]" % ", ".join(roots))},
            alternatives=(
                "only %d of these %d should be parentless, and the rule cannot "
                "tell which — a root is just a joint with no parent"
                % (declared, len(roots)),
                "declare `roots: %d` if the rig really is in %d pieces"
                % (len(roots), len(roots)),
            ),
        )
    else:
        repair = Repair(
            "rig.detach",
            (Raw(ctx.name),),
            {"expect": declared},
            alternatives=(
                "a joint that was meant to be free has acquired a parent; %d "
                "were declared and %d are parentless"
                % (declared, len(roots)),
                "declare `roots: %d` if one body is what you meant" % len(roots),
            ),
        )

    yield ctx.finding(
        summary="Rig has %d root joint%s, declared %d"
        % (len(roots), "" if len(roots) == 1 else "s", declared),
        entity=ctx.name,
        where=[("roots", roots or ["none"]),
               ("declared", str(declared))],
        evidence="%d parentless joint%s of %d total: %s"
        % (len(roots), "" if len(roots) == 1 else "s", len(art.joints),
           ", ".join(roots) or "none"),
        data={"roots": roots, "declared": declared, "actual": len(roots)},
        repair=repair,
    )


# --------------------------------------------------------------------- W505


@rule(
    "W505",
    WARN,
    "Declared rig symmetry does not hold",
    needs=("emit.articulation", "emit.articulation.symmetry"),
    defaults={"max_report": 8},
    approximation="Checks joint *origins* map onto joint origins under the "
    "declared transform. It does not compare limits, axes or the geometry bound "
    "to each joint, so a rig can be positionally symmetric and still bend "
    "asymmetrically. It is the cheap half of the question and it is exact about "
    "the half it answers.",
)
def w505(ctx):
    """Symmetry, on the body plan the program declared rather than a mirror
    plane assumed from anatomy.

    `bilateral` reflects, `radial` rotates by 360/fold. A five-armed starfish,
    an eight-fold jellyfish and a bilaterally symmetric spider are each checked
    against their own plan; a snake or a lamp arm declares `none` and gets a
    *not checked* line rather than a warning it could never satisfy.
    """
    art = _articulation(ctx)
    sym = art.symmetry
    if sym is None:
        ctx.not_applicable(
            "emit declares no body plan — add `symmetry` to the articulation "
            "spec (bilateral, radial with a fold, or none)"
        )
    kind = (sym.kind or "none").lower()
    if kind == "none":
        ctx.not_applicable(
            "body plan is declared asymmetric, so there is nothing to mirror"
        )
    if kind not in ("bilateral", "radial"):
        ctx.not_applicable(
            "unknown body plan %r — known: bilateral, radial, none" % sym.kind
        )
    if kind == "radial" and int(sym.fold) < 2:
        ctx.not_applicable(
            "radial symmetry needs a fold of 2 or more, got %r" % sym.fold
        )

    _, world = _world(art)
    names = [j.name for j in art.joints]
    pts = np.array([world[n][:3, 3] for n in names], dtype=np.float64)
    mapped = _map_points(pts, sym, kind)
    tol = float(sym.tolerance)

    # Nearest declared joint to each mapped position. A joint that lies on the
    # mirror plane or the rotation axis maps to itself, which is a match.
    orphans = []
    for i, p in enumerate(mapped):
        d = float(np.min(np.linalg.norm(pts - p, axis=1)))
        if d > tol:
            orphans.append((names[i], d))
    if not orphans:
        return

    orphans.sort(key=lambda t: -t[1])
    plan = ("%d-fold radial about %s" % (int(sym.fold), _vec(sym.axis))
            if kind == "radial" else "bilateral across %s" % _vec(sym.normal))
    rows = ["%-14s nearest counterpart %s mm away" % (n, num(d * 1000.0))
            for n, d in orphans[: ctx.p("max_report")]]
    yield ctx.finding(
        summary="%d joint%s break the declared %s plan"
        % (len(orphans), "" if len(orphans) == 1 else "s", plan),
        entity="%s/%s" % (ctx.name, orphans[0][0]),
        where=[("plan", plan), ("joints", rows),
               ("within", "%s mm" % num(tol * 1000.0))],
        evidence="%d of %d joints have no counterpart under the declared "
        "transform; worst %s at %s mm against a %s mm tolerance"
        % (len(orphans), len(names), orphans[0][0], num(orphans[0][1] * 1000.0),
           num(tol * 1000.0)),
        data={"plan": kind, "fold": int(sym.fold) if kind == "radial" else None,
              "count": len(orphans),
              "joints": [{"joint": n, "gap_mm": d * 1000.0} for n, d in orphans]},
        repair=Repair(
            "rig.mirror" if kind == "bilateral" else "rig.radial_array",
            (Raw(ctx.name),),
            ({"across": Raw(_vec(sym.normal))} if kind == "bilateral"
             else {"axis": Raw(_vec(sym.axis)), "count": int(sym.fold)}),
            alternatives=(
                "declare `symmetry: none` if the rig is meant to be asymmetric "
                "— an asymmetric plan is a real answer, not a failure",
            ),
        ),
    )


def _map_points(pts: np.ndarray, sym, kind: str) -> np.ndarray:
    """Where each joint origin should land under the declared plan."""
    if kind == "bilateral":
        n = np.asarray(sym.normal, dtype=np.float64)
        n = n / (np.linalg.norm(n) or 1.0)
        # Reflect across {p . n == offset}.
        d = pts @ n - float(sym.offset)
        return pts - 2.0 * d[:, None] * n[None, :]

    from ...pose import rotation

    a = np.asarray(sym.axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) or 1.0)
    c = np.asarray(sym.center, dtype=np.float64)
    R = rotation(a, 2.0 * math.pi / int(sym.fold))
    return (pts - c) @ R.T + c


def _vec(v) -> str:
    return "(%s)" % ", ".join(num(float(x)) for x in v)
