"""
Binding solids to joints, and evaluating a posed assembly.

**Rigid parts only, and that is the honest ceiling of this module.** Each bound
solid moves with its joint as one piece. There is no vertex weighting and no
deformation, because `models/README.md` measured why there could not be:

> **not riggable.** No edge loops at shoulder, elbow, hip or knee. A bone
> rotation would pinch and tear it.

Skinning a mesh that has no edge loops at the joint does not produce a bend, it
produces a crease and a self-intersection. So this module does the thing the
kernel is actually good at: hard-surface articulation. Hinged doors, lamp arms,
linkages, mechanisms, posable modular kit pieces. DESIGN.md section 12 defers
"quad remeshing to animation-grade topology" and "template basemesh + cage
deformation" to later, and until that lands, deformation stays out.

The useful consequence is that rigid articulation needs none of it. A robot arm
has no edge flow requirement at all.

**Overlap parts across the pivot.** `models/README.md` again, from the villager:

> Coincident contact is not connection. The villager's boots met the shins
> exactly on z=0.075 and came back as separate components.

Articulation makes that sharper, because **the component count is a function of
the pose**. Measured on the fixtures in `tests/`: two parts left 10 mm apart read
as two components at rest and weld into one at 25 degrees, when a corner swings
into contact. So a rig declaring `components: 1` is making a claim about its
whole range of motion, and checking the rest shape alone does not test it.

Overlapping the parts across the pivot is what makes the claim hold at every
angle, and it is the only arrangement that also avoids a gap opening at the
hinge. `posed()` returns an ordinary `Solid`, so `.components()` measures this
and E108 grades it against the program's own declaration -- at whichever poses
the program chooses to check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from ..build.solid import LatheError, Solid, union_all
from .skeleton import Pose, RigError, Skeleton, forward


@dataclass(frozen=True)
class Rig:
    """A skeleton plus the geometry bound to it.

    Solids are authored in **rest space** -- modelled where they sit when every
    angle is zero, in ordinary world coordinates. That means an existing model
    can be rigged without being rebuilt: bind the parts you already have, and
    the rest pose is by definition the shape you already verified.
    """

    skeleton: Skeleton
    bindings: Mapping[str, Solid] = field(default_factory=dict)
    name: str = "rig"

    def __post_init__(self) -> None:
        for joint_name in self.bindings:
            self.skeleton.joint(joint_name)  # raises, naming the known joints

    # -- authoring ---------------------------------------------------------

    def bind(self, joint: str, solid: Solid) -> "Rig":
        """Attach a solid to a joint. Returns a new rig.

        Binding twice to one joint unions the two, so a limb can be assembled
        from several primitives without inventing a grouping concept.
        """
        self.skeleton.joint(joint)
        merged = dict(self.bindings)
        if joint in merged:
            merged[joint] = merged[joint] + solid
        else:
            merged[joint] = solid
        return Rig(self.skeleton, merged, self.name)

    @property
    def bound(self) -> Tuple[str, ...]:
        return tuple(n for n in self.skeleton.names if n in self.bindings)

    @property
    def unbound(self) -> Tuple[str, ...]:
        """Joints carrying no geometry. Legal -- a tip marker is usually bare."""
        return tuple(n for n in self.skeleton.names if n not in self.bindings)

    # -- evaluation --------------------------------------------------------

    def part(self, joint: str, pose: Optional[Pose] = None) -> Solid:
        """One bound part, moved into `pose`."""
        if joint not in self.bindings:
            raise RigError(
                "joint %r carries no geometry\n  Bound joints: %s."
                % (joint, ", ".join(self.bound) or "none")
            )
        deltas = self._deltas(pose)
        return _apply(self.bindings[joint], deltas[joint])

    def posed(self, pose: Optional[Pose] = None) -> Solid:
        """The whole assembly under `pose`, as one solid.

        Pure: same rig, same pose, same geometry. The union is taken in
        skeleton order (parents before children) so the result is
        byte-identical across runs rather than merely equivalent.
        """
        if not self.bindings:
            raise RigError(
                "rig %r has no geometry bound to any joint\n"
                "  Use `rig.bind(\"elbow\", forearm)` before posing it."
                % self.name
            )
        deltas = self._deltas(pose)
        parts = [_apply(self.bindings[n], deltas[n]) for n in self.bound]
        out = parts[0] if len(parts) == 1 else union_all(parts)
        return out.rename(self.name)

    def at_rest(self) -> Solid:
        return self.posed(None)

    def _deltas(self, pose: Optional[Pose]) -> Dict[str, np.ndarray]:
        """Per joint, the 4x4 taking rest-space geometry into the pose.

        `world(pose) @ inverse(world(rest))`. Inverted from the rest matrix
        rather than assumed to be identity, because a joint's rest angle is its
        clamped zero, and a limit that excludes zero makes that non-trivial.
        """
        now = forward(self.skeleton, pose)
        rest = forward(self.skeleton, None)
        return {n: now[n] @ _rigid_inverse(rest[n]) for n in now}


# --------------------------------------------------------------------- maths


def _rigid_inverse(M: np.ndarray) -> np.ndarray:
    """Inverse of a rotation+translation 4x4, without a general solve.

    R is orthonormal here by construction -- every matrix in this module comes
    from Rodrigues and a translation -- so the transpose is the inverse exactly,
    and `np.linalg.inv` would only add float noise to a known-exact answer.
    """
    R = M[:3, :3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ M[:3, 3]
    return out


def _apply(solid: Solid, M: np.ndarray) -> Solid:
    """Push a solid through a 4x4, via Manifold's 3x4 affine.

    Manifold combines and applies transforms lazily, so a chain of these costs
    one pass over the vertices rather than one per joint.
    """
    affine = np.ascontiguousarray(M[:3, :4], dtype=np.float64)
    man = solid.man.transform(affine)
    out = Solid(man, dict(solid.tags), solid.name)
    if not out.status_ok():
        raise LatheError(
            "posing %s produced %s -- the transform is not rigid"
            % (solid.name, man.status())
        )
    return out
