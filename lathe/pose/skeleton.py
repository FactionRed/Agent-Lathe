"""
Joints and forward kinematics -- DESIGN.md sections 2, 5.1 and 5.4.

A rig here is two separate things, and keeping them separate is the whole point:

  * a **Skeleton** is structure -- named joints, rest offsets, hinge axes, limits;
  * a **Pose** is `{joint name: angle}`, and nothing else.

That split is what makes a pose a *parameter edit* rather than a mesh mutation.
Section 10 asks for exactly this: "the app writes back a parameter or constraint
edit, not an opaque mesh mutation ... Human correction becomes machine-legible
input." A dict of named angles diffs cleanly, survives a proportion change
upstream, and reads back into the program as `param elbow = 35deg`. Baked vertex
positions do none of that.

Joints are addressed by name, never by index (principle 2). A chain is resolved
by walking parent links at evaluation time, so inserting a joint upstream does
not silently renumber anything downstream.

Everything is SI internally -- metres and radians -- matching `lang/values.py`.
The DSL's angle units (`deg`, `rad`, `turn`) already convert on the way in.

**Scope limit, stated up front.** These are 1-DOF hinges. Real shoulders are
ball joints, and a ball joint is three hinges here. That is a deliberate
restriction rather than an oversight: a hinge has an unambiguous signed angle, so
its limit is checkable and its value is meaningful in a diff. Euler triples have
neither property -- the same orientation has several spellings, and clamping one
of three coupled angles is not a joint limit. Compose three hinges and the
program says which order it meant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..build.solid import LatheError

Vec3 = Tuple[float, float, float]

# A pose is data, not an object: {joint name -> angle in radians}.
Pose = Mapping[str, float]

_EPS = 1e-12


class RigError(LatheError):
    """A malformed skeleton, or a pose naming a joint that does not exist."""


# ----------------------------------------------------------------------- joint


@dataclass(frozen=True)
class Joint:
    """One hinge.

    `offset` is the rest translation from the parent's origin, expressed in the
    parent's frame. `axis` is the rotation axis in the same frame. `limit` is
    (min, max) in radians; the default is a full turn either way, which is the
    honest default for "nobody said".

    A joint with `limit=(0.0, 0.0)` is welded -- useful as a named tip so IK can
    target a toe or a fingertip without inventing a separate end-effector
    concept.
    """

    name: str
    parent: Optional[str] = None
    offset: Vec3 = (0.0, 0.0, 0.0)
    axis: Vec3 = (0.0, 1.0, 0.0)
    limit: Tuple[float, float] = (-2.0 * math.pi, 2.0 * math.pi)

    @property
    def locked(self) -> bool:
        return self.limit[0] == self.limit[1]

    def clamp(self, angle: float) -> float:
        lo, hi = self.limit
        return min(max(angle, lo), hi)

    def unit_axis(self) -> np.ndarray:
        a = np.asarray(self.axis, dtype=np.float64)
        n = float(np.linalg.norm(a))
        if n < _EPS:
            raise RigError(
                "joint %r has a zero-length axis %r\n"
                "  A hinge needs a direction. Use (0, 1, 0) for a pitch hinge."
                % (self.name, self.axis)
            )
        return a / n


# -------------------------------------------------------------------- skeleton


@dataclass(frozen=True)
class Skeleton:
    """A named joint tree. Frozen, so a rig is a value like everything else."""

    joints: Tuple[Joint, ...] = ()
    name: str = "rig"
    _by_name: Dict[str, Joint] = field(
        default_factory=dict, compare=False, repr=False
    )
    _order: Tuple[str, ...] = field(default=(), compare=False, repr=False)

    def __post_init__(self) -> None:
        by_name: Dict[str, Joint] = {}
        for j in self.joints:
            if j.name in by_name:
                raise RigError(
                    "duplicate joint name %r\n"
                    "  Joints are addressed by name, so names must be unique."
                    % j.name
                )
            by_name[j.name] = j
        for j in self.joints:
            if j.parent is not None and j.parent not in by_name:
                raise RigError(
                    "joint %r names parent %r, which does not exist\n"
                    "  Known joints: %s."
                    % (j.name, j.parent, ", ".join(sorted(by_name)) or "none")
                )
        object.__setattr__(self, "_by_name", by_name)
        object.__setattr__(self, "_order", _topological(self.joints, by_name))

    # -- lookup ------------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __len__(self) -> int:
        return len(self.joints)

    @property
    def names(self) -> Tuple[str, ...]:
        """Joint names in parents-before-children order."""
        return self._order

    def joint(self, name: str) -> Joint:
        try:
            return self._by_name[name]
        except KeyError:
            raise RigError(
                "no joint named %r\n  Known joints: %s."
                % (name, ", ".join(self._order))
            ) from None

    def roots(self) -> Tuple[str, ...]:
        return tuple(j.name for j in self.joints if j.parent is None)

    def children(self, name: str) -> Tuple[str, ...]:
        self.joint(name)
        return tuple(j.name for j in self.joints if j.parent == name)

    # -- chains ------------------------------------------------------------

    def chain(self, root: str, tip: str) -> Tuple[str, ...]:
        """The joint names from `root` to `tip` inclusive, in that order.

        Walks parent links upward from the tip, which is why an edit upstream of
        `root` cannot renumber this chain -- there are no numbers.
        """
        self.joint(root)
        walk, cur = [], tip
        while cur is not None:
            walk.append(cur)
            if cur == root:
                out = tuple(reversed(walk))
                return out
            cur = self.joint(cur).parent
        raise RigError(
            "%r is not an ancestor of %r\n"
            "  The chain walks parent links up from the tip. %r sits under %s."
            % (root, tip, tip, self.joint(tip).parent or "no parent (it is a root)")
        )

    def rest(self) -> Dict[str, float]:
        """The zero pose, with every angle clamped into its own limit.

        A joint whose limit excludes zero has no business reporting zero as its
        rest value, and silently doing so would put the rig outside its own
        declared range before anyone touched it.
        """
        return {j.name: j.clamp(0.0) for j in self.joints}

    def with_joint(self, joint: Joint) -> "Skeleton":
        """Add or replace one joint. Returns a new skeleton."""
        kept = tuple(j for j in self.joints if j.name != joint.name)
        return Skeleton(kept + (joint,), self.name)


def _topological(joints: Sequence[Joint], by_name: Mapping[str, Joint]) -> Tuple[str, ...]:
    """Parents before children. Raises on a cycle, naming the members."""
    order: list = []
    state: Dict[str, int] = {}  # 0 = visiting, 1 = done

    def visit(name: str, trail: Tuple[str, ...]) -> None:
        seen = state.get(name)
        if seen == 1:
            return
        if seen == 0:
            cycle = list(trail[trail.index(name):]) + [name]
            raise RigError(
                "joint cycle: %s\n"
                "  A skeleton is a tree. One of these parents must change."
                % " -> ".join(cycle)
            )
        state[name] = 0
        parent = by_name[name].parent
        if parent is not None:
            visit(parent, trail + (name,))
        state[name] = 1
        order.append(name)

    for j in joints:
        visit(j.name, ())
    return tuple(order)


# ---------------------------------------------------------------- kinematics


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """3x3 rotation about a unit axis, by Rodrigues.

    Built from the axis directly rather than from Euler angles, because the
    hinge *is* the axis -- converting to a triple and back would introduce an
    ordering convention the program never stated.
    """
    x, y, z = (float(v) for v in axis)
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def _local(joint: Joint, angle: float) -> np.ndarray:
    """The joint's 4x4: translate to its origin, then rotate about its axis."""
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = rotation(joint.unit_axis(), angle)
    M[:3, 3] = np.asarray(joint.offset, dtype=np.float64)
    return M


def forward(skeleton: Skeleton, pose: Optional[Pose] = None) -> Dict[str, np.ndarray]:
    """Forward kinematics: every joint's 4x4 world matrix, keyed by name.

    Pure -- same skeleton and same pose give the same matrices, every time.
    Unnamed joints take their rest angle rather than raising, so a partial pose
    is a legal thing to write and `{"elbow": 0.5}` means what it looks like.
    """
    angles = dict(skeleton.rest())
    if pose:
        unknown = sorted(set(pose) - set(skeleton.names))
        if unknown:
            raise RigError(
                "pose names %s, which %s not in the skeleton\n"
                "  Known joints: %s."
                % (
                    ", ".join(repr(u) for u in unknown),
                    "are" if len(unknown) > 1 else "is",
                    ", ".join(skeleton.names),
                )
            )
        angles.update({k: float(v) for k, v in pose.items()})

    out: Dict[str, np.ndarray] = {}
    for name in skeleton.names:
        j = skeleton.joint(name)
        local = _local(j, angles[name])
        out[name] = local if j.parent is None else out[j.parent] @ local
    return out


def positions(skeleton: Skeleton, pose: Optional[Pose] = None) -> Dict[str, np.ndarray]:
    """Just the world-space origin of each joint, in metres."""
    return {k: M[:3, 3].copy() for k, M in forward(skeleton, pose).items()}


def chain_length(skeleton: Skeleton, chain: Iterable[str]) -> float:
    """Summed rest offsets along a chain -- the furthest the tip can reach.

    Measured at rest, which is exact for hinges: a hinge rotates its subtree and
    never stretches it, so no pose can change these lengths.
    """
    names = list(chain)
    return float(
        sum(
            float(np.linalg.norm(np.asarray(skeleton.joint(n).offset, dtype=np.float64)))
            for n in names[1:]
        )
    )
