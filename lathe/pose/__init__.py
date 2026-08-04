"""
Articulation: joints, forward kinematics, IK, and rigid posing.

DESIGN.md section 5.4 reserves `solve { ... }` for "a real solver later, if it
earns its keep". This is that solver, plus the joint model it needs.

The through-line is section 10: a pose is a **parameter edit**, not a mesh
mutation. `Skeleton` holds structure, `Pose` is `{name: radians}`, and the two
never merge. That is what lets a human drag a limb in the viewer, or a pose
estimator read a photograph, and have either one come back as `param elbow =
35deg` -- text the agent can read, the DAG can diff, and the verifier can check.

    from lathe import pose as P
    import lathe.build as B

    rig = P.Rig(P.Skeleton((
        P.Joint("shoulder", offset=(0, 0, 1.4)),
        P.Joint("elbow", parent="shoulder", offset=(0, 0, -0.3),
                limit=(0.0, P.deg(150))),
        P.Joint("wrist", parent="elbow", offset=(0, 0, -0.28)),
    )))
    rig = rig.bind("shoulder", upper_arm).bind("elbow", forearm)

    hit = P.solve(rig.skeleton, "shoulder", "wrist", target=(0.25, 0.0, 1.1))
    solid = rig.posed(hit.pose)          # a normal Solid: verify, render, export

Rigid parts only -- see `rig.py` for why, and `models/README.md` for the
measurement behind it.
"""

from __future__ import annotations

import math

from .ik import (  # noqa: F401
    DEFAULT_MAX_PASSES,
    DEFAULT_TOLERANCE,
    IKResult,
    reach_error,
    solve,
    violations,
)
from .rig import Rig  # noqa: F401
from .skeleton import (  # noqa: F401
    Joint,
    Pose,
    RigError,
    Skeleton,
    chain_length,
    forward,
    positions,
    rotation,
)


def deg(x: float) -> float:
    """Degrees to radians. Poses are SI internally, like the rest of the system.

    Present so a hand-written Python rig reads in the same units the `.lathe`
    program does -- `60deg` there, `deg(60)` here -- rather than making the
    Python path the one place a bare number silently means radians.
    """
    return float(x) * math.pi / 180.0


def degrees(x: float) -> float:
    """Radians back to degrees, for reports and writeback."""
    return float(x) * 180.0 / math.pi


__all__ = [
    "Joint", "Skeleton", "Pose", "Rig", "RigError",
    "forward", "positions", "chain_length", "rotation",
    "solve", "reach_error", "violations", "IKResult",
    "DEFAULT_TOLERANCE", "DEFAULT_MAX_PASSES",
    "deg", "degrees",
]
