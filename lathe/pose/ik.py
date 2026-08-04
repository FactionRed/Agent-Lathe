"""
Inverse kinematics -- the `solve { ... }` block DESIGN.md section 5.4 reserves.

> This is a lazy expression graph, **not** a general geometric constraint
> solver -- the deliberate scope limit keeps failures explicable. Add a real
> solver later, behind an explicit `solve { ... }` block, if it earns its keep.

This is that solver, kept to the one case that earns it: place a named tip at a
named target, by rotating the hinges between them.

**Cyclic coordinate descent**, not FABRIK. FABRIK is the faster and more famous
choice, and it is the wrong one here. FABRIK solves for joint *positions* and
then reconstructs orientations, so a hinge axis and a signed joint limit are
both things it has to be talked into afterwards. CCD works directly in the
angles the program actually declares: each step is "rotate this one hinge about
its own axis to point the tip at the target", which clamps against a limit
exactly and writes back as `param elbow = 35deg` with no reconstruction step in
between. The variable this system wants to report and diff is the angle, so the
solver works in angles.

**Determinism.** Principle 1 says the same program produces the same geometry,
byte for byte. So: a fixed joint order (tip to root), a fixed iteration cap, no
random restarts, and no early exit that depends on anything but the measured
error. Two runs of the same solve return identical floats.

**The straight-chain singularity, and why it needs handling rather than
reporting.** A fully extended chain aimed directly at a nearer target is a stall
CCD cannot climb out of on its own: seen from any hinge, the tip and the target
lie on the same ray, the angle between them is exactly zero, and so is every
correction. The pose is stationary and wrong. It is also the single most
ordinary thing a caller will ask for -- a rest pose is usually straight, and
"reach that point in front of you" is the first request anyone makes.

The usual fix is a random restart, which principle 1 forbids. Instead, a stalled
pass -- no joint moved, error still above tolerance -- triggers a fixed nudge of
every movable joint toward the middle of its own declared range. Deterministic,
so two runs still agree byte for byte, and bounded, so a genuinely blocked solve
(a limit in the way, a target out of reach) still terminates and still reports
the stall rather than pretending. `NUDGE` is small enough not to bias the answer
it converges to; it only breaks the tie that collinearity created.

**What it does not do.** No collision avoidance, no pole-vector control, and no
preference between the several poses that reach the same point. A two-link arm
has two solutions, elbow up and elbow down; CCD returns whichever the starting
pose is nearer, so the seed pose *is* the tie-breaker. Say so in the program by
seeding deliberately rather than expecting the solver to read your mind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .skeleton import Pose, RigError, Skeleton, chain_length, forward

_EPS = 1e-12

DEFAULT_TOLERANCE = 1e-5   # 0.01 mm -- below any modelling tolerance here
DEFAULT_MAX_PASSES = 64

# Desingularisation. NUDGE is ~0.06 degrees: enough to leave a collinear pose,
# far below any angle a caller would notice in the answer. MAX_NUDGES bounds the
# blocked case, where nudging and clamping would otherwise trade places until
# the pass cap ran out.
NUDGE = 1e-3
MAX_NUDGES = 4


@dataclass(frozen=True)
class IKResult:
    """The solved pose plus the numbers needed to judge it.

    Shaped like a verifier finding on purpose: what the answer is, how wrong it
    is, and which declaration got in the way. `pose` is always usable -- a
    failed solve returns the best pose found, not nothing, because a
    near-miss is still the most useful thing to show a human.
    """

    pose: Dict[str, float]
    error: float                       # metres from tip to target
    passes: int
    converged: bool
    reachable: bool
    tolerance: float = DEFAULT_TOLERANCE
    at_limit: Tuple[str, ...] = ()     # joints resting against a declared limit
    reach_deficit: float = 0.0         # metres the target sits beyond the chain

    @property
    def error_mm(self) -> float:
        return self.error * 1000.0

    def describe(self) -> str:
        """One line, in the section 7 register: measured, not adjectival."""
        head = "converged" if self.converged else (
            "unreachable" if not self.reachable else "stalled"
        )
        bits = ["%s after %d pass%s" % (head, self.passes,
                                        "" if self.passes == 1 else "es"),
                "error %.4f mm" % self.error_mm]
        if not self.reachable:
            bits.append("target %.4f mm beyond the chain" % (self.reach_deficit * 1000.0))
        if self.at_limit:
            bits.append("at limit: %s" % ", ".join(self.at_limit))
        return "; ".join(bits)


def solve(
    skeleton: Skeleton,
    root: str,
    tip: str,
    target,
    seed: Optional[Pose] = None,
    tolerance: float = DEFAULT_TOLERANCE,
    max_passes: int = DEFAULT_MAX_PASSES,
) -> IKResult:
    """Rotate the hinges between `root` and `tip` to put `tip` on `target`.

    `seed` is the starting pose and the tie-breaker between mirror solutions;
    it defaults to rest. Angles outside the chain are carried through untouched,
    so solving an arm does not quietly straighten a leg.
    """
    names = skeleton.chain(root, tip)
    if len(names) < 2:
        raise RigError(
            "chain %r -> %r has one joint, so there is nothing to solve\n"
            "  IK needs at least one hinge between the root and the tip."
            % (root, tip)
        )

    goal = np.asarray(target, dtype=np.float64).reshape(3)
    angles: Dict[str, float] = dict(skeleton.rest())
    if seed:
        angles.update({k: float(v) for k, v in seed.items()})
    # The seed is data from outside; a value outside its own limit is a bug
    # worth absorbing quietly rather than solving from an illegal start.
    for n in names:
        angles[n] = skeleton.joint(n).clamp(angles[n])

    # Every joint but the tip can move it. The tip's own rotation spins the
    # subtree below it and leaves its origin exactly where it was.
    movable = [n for n in names[:-1] if not skeleton.joint(n).locked]

    world = forward(skeleton, angles)
    span = chain_length(skeleton, names)
    root_origin = world[names[0]][:3, 3]
    distance = float(np.linalg.norm(goal - root_origin))
    deficit = max(0.0, distance - span)
    reachable = deficit <= tolerance

    best = dict(angles)
    best_err = float(np.linalg.norm(world[tip][:3, 3] - goal))
    passes = 0
    nudges = 0

    for passes in range(1, max_passes + 1):
        stationary = True
        for name in reversed(movable):
            joint = skeleton.joint(name)
            M = world[name]
            origin = M[:3, 3]
            axis = M[:3, :3] @ joint.unit_axis()

            to_tip = world[tip][:3, 3] - origin
            to_goal = goal - origin

            # Only the components perpendicular to the axis can rotate. A hinge
            # cannot move anything that lies along its own axis, and pretending
            # otherwise is how a solver starts violating the constraint it was
            # given.
            e = to_tip - np.dot(to_tip, axis) * axis
            t = to_goal - np.dot(to_goal, axis) * axis
            ne, nt = float(np.linalg.norm(e)), float(np.linalg.norm(t))
            if ne < _EPS or nt < _EPS:
                continue  # degenerate: tip or target sits on the axis

            e /= ne
            t /= nt
            step = math.atan2(float(np.dot(np.cross(e, t), axis)),
                              float(np.clip(np.dot(e, t), -1.0, 1.0)))
            moved = joint.clamp(angles[name] + step)
            if abs(moved - angles[name]) < _EPS:
                continue
            angles[name] = moved
            stationary = False
            world = forward(skeleton, angles)

        err = float(np.linalg.norm(world[tip][:3, 3] - goal))
        if err < best_err:
            best_err, best = err, dict(angles)
        if err <= tolerance:
            break

        if stationary:
            # Collinear, so every correction is exactly zero. Bend out of it.
            if nudges >= MAX_NUDGES:
                break  # genuinely blocked, not merely singular
            nudges += 1
            for name in movable:
                j = skeleton.joint(name)
                mid = 0.5 * (j.limit[0] + j.limit[1])
                toward = 1.0 if mid >= angles[name] else -1.0
                angles[name] = j.clamp(angles[name] + toward * NUDGE)
            world = forward(skeleton, angles)

    at_limit = tuple(
        n for n in movable
        if _on_limit(skeleton, n, best[n])
    )
    return IKResult(
        pose=best,
        error=best_err,
        passes=passes,
        converged=best_err <= tolerance,
        reachable=reachable,
        tolerance=tolerance,
        at_limit=at_limit,
        reach_deficit=deficit,
    )


def _on_limit(skeleton: Skeleton, name: str, angle: float, eps: float = 1e-9) -> bool:
    lo, hi = skeleton.joint(name).limit
    return abs(angle - lo) < eps or abs(angle - hi) < eps


def reach_error(skeleton: Skeleton, tip: str, target, pose: Optional[Pose] = None) -> float:
    """Metres between `tip` and `target` under `pose`. The assertable number.

    This is what `assert reach(ankle, target) < 2mm` compiles to, and it is a
    measurement rather than a solve -- so it is honest about a pose someone
    posed by hand, not only ones this module produced.
    """
    world = forward(skeleton, pose)
    if tip not in world:
        raise RigError("no joint named %r" % tip)
    goal = np.asarray(target, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(world[tip][:3, 3] - goal))


def violations(skeleton: Skeleton, pose: Pose) -> Tuple[Tuple[str, float, float, float], ...]:
    """Joints whose angle falls outside their declared limit.

    Returns (name, angle, lo, hi) per offender, in skeleton order. Empty is the
    pass case. A hand-authored or estimator-derived pose is exactly the input
    that needs this -- `solve` clamps, but nothing clamps a pose typed by a
    human or lifted off a photograph.
    """
    out = []
    for name in skeleton.names:
        if name not in pose:
            continue
        j = skeleton.joint(name)
        a = float(pose[name])
        lo, hi = j.limit
        if a < lo - 1e-12 or a > hi + 1e-12:
            out.append((name, a, lo, hi))
    return tuple(out)
