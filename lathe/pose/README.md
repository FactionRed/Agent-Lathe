# Articulation subsystem — DESIGN.md sections 5.4 and 10

**Status:** working. 3 modules, 45 tests, no Blender, no new dependency.

> This is a lazy expression graph, **not** a general geometric constraint
> solver — the deliberate scope limit keeps failures explicable. Add a real
> solver later, behind an explicit `solve { … }` block, if it earns its keep.
> — [DESIGN.md](../../DESIGN.md) §5.4

This is that solver, plus the joint model it needs.

```bash
python -m pytest lathe/pose/tests -q     # 45 tests, ~0.2 s
```

## The one idea

A **pose is a parameter edit, not a mesh mutation.** §10 asks for exactly this:

> When a human adjusts something in the viewer, the app **writes back a
> parameter or constraint edit**, not an opaque mesh mutation … Human
> correction becomes machine-legible input. That feedback loop is the actual
> product.

So the structure and the values never merge:

| | what it is | why separate |
|---|---|---|
| `Skeleton` | named joints, rest offsets, hinge axes, limits | the rig, versioned with the program |
| `Pose` | `{joint name: radians}`, a plain mapping | diffable, re-evaluable, writes back as `param elbow = 35deg` |

A human dragging a limb and a pose estimator reading a photograph produce the
same artifact: a handful of typed numbers the agent can read, the DAG can diff,
and the verifier can check. Baked vertex positions are none of those things, and
would reintroduce the opaque-mutation problem the whole system exists to avoid.

## What is here

```python
import lathe.build as B
from lathe import pose as P

sk = P.Skeleton((
    P.Joint("base",  axis=(0, 0, 1)),
    P.Joint("lower", parent="base",  offset=(0, 0, 0.08), axis=(0, 1, 0),
            limit=(-P.deg(90), P.deg(90))),
    P.Joint("upper", parent="lower", offset=(0, 0, 0.34), axis=(0, 1, 0),
            limit=(-P.deg(140), P.deg(140))),
    P.Joint("head",  parent="upper", offset=(0, 0, 0.30), limit=(0, 0)),
))

rig = (P.Rig(sk, name="SM_LampArm_Brass_LOD0")
       .bind("base",  B.cylinder(0.10, 0.09, center=False, tag="base"))
       .bind("lower", B.box(0.05, 0.05, 0.34, tag="lower").at(z=0.25))
       .bind("upper", B.box(0.045, 0.045, 0.32, tag="upper").at(z=0.57)))

hit = P.solve(sk, "lower", "head", target=(-0.28, 0.0, 0.46))
solid = rig.posed(hit.pose)
```

Real output from that program:

```
IK  : converged after 11 passes; error 0.0092 mm
pose: {'base': '0.0deg', 'lower': '2.9deg', 'upper': '-85.2deg', 'head': '0.0deg'}
solid: 164 tris, 1 component(s), watertight=True

check  SM_LampArm_Brass_LOD0   profile game_prop_pbr   PASS   0 error, 0 warn, 0 note
```

`posed()` returns an ordinary `Solid`. It goes straight into verify, perceive,
render and export with no special case — which is the point. A subsystem that
needed its own export path would have bought itself an island.

## Decisions worth defending

**Hinges, not ball joints.** A hinge has an unambiguous signed angle, so its
limit is checkable and its value means something in a diff. Euler triples have
neither property: the same orientation has several spellings, and clamping one
of three coupled angles is not a joint limit. A shoulder is three hinges here,
and the program says which order it meant.

**CCD, not FABRIK.** FABRIK is faster and better known, and it solves for joint
*positions*, then reconstructs orientations — so a hinge axis and a signed limit
are both things it has to be talked into afterwards. CCD works directly in the
angles the program declares. Each step is "rotate this one hinge about its own
axis to point the tip at the target", which clamps against a limit exactly and
writes back as `param elbow = 35deg` with no reconstruction in between. The
variable this system wants to report and diff is the angle, so the solver works
in angles.

**Determinism over the last 1% of convergence.** Fixed joint order, fixed pass
cap, no random restarts. Two runs return identical floats, per principle 1.

**The straight-chain singularity is handled, not reported.** A fully extended
chain aimed at a nearer target is a stall CCD cannot climb out of: seen from any
hinge, tip and target lie on the same ray, so every correction is exactly zero.
It is also the most ordinary request anyone makes — rest poses are straight. The
textbook fix is a random restart, which principle 1 forbids, so a stalled pass
triggers a *fixed* nudge toward the middle of each joint's declared range.
Deterministic, bounded, and it only breaks the tie collinearity created.

## What this does not do

- **No skinning, and that is the ceiling.** Parts move rigidly. `models/README.md`
  measured why there is no alternative yet: the villager has "no edge loops at
  shoulder, elbow, hip or knee. A bone rotation would pinch and tear it."
  Skinning a mesh with no edge loops at the joint produces a crease and a
  self-intersection, not a bend. §12 defers quad remeshing and cage deformation
  to later, and until that lands, deformation stays out.
  **Rigid articulation needs none of it** — hinged doors, lamp arms, linkages,
  mechanisms and posable kit pieces are exactly what this kernel is good at.
- **No `joint` / `solve` statements in `.lathe` yet.** This is the Python API.
  The DSL already has angle units, so the parser work is the remaining piece.
- **No verifier rules yet.** `violations()` and `reach_error()` return the
  numbers a §7 rule pack would report; nothing grades them into a profile.
- **No collision avoidance and no pole-vector control.** A two-link chain has two
  solutions, and CCD returns whichever branch the seed sits in. The seed has to
  be decisive — ±20° lands in the same basin either way, and only a seed past
  the fold selects the other branch. Measured, in `tests/test_pose.py`.

## Two things that will surprise you

**A one-sided limit halves the workspace.** A positive elbow angle folds the
forearm one specific way, so an elbow that may only open positively reaches
targets on one side and cannot reach their mirror. That is not a solver failure;
it is the declared limit doing what it says. The result reports `at_limit` and a
measured error rather than shrugging.

**Component count is a function of the pose.** Two parts left 10 mm apart read
as two components at rest and weld into one at 25°, when a corner swings into
contact. So a rig declaring `components: 1` is making a claim about its whole
range of motion, and checking the rest shape alone does not test it. Overlap
parts across the pivot — it is the only arrangement that holds at every angle
and also avoids a gap opening at the hinge.

## Files

| Path | Lines | Contents |
|---|---|---|
| `skeleton.py` | 303 | `Joint`, `Skeleton`, topological order, FK, Rodrigues |
| `ik.py` | 260 | CCD solver, `IKResult`, `reach_error`, `violations` |
| `rig.py` | 169 | binding solids to joints, rest→pose deltas, `posed()` |
| `tests/` | 434 | 45 tests against closed-form truth |

A two-link arm's elbow angle is `acos((a² + b² − d²) / 2ab)` by the law of
cosines, and the suite asserts the solver lands on it to 1e-9 rather than merely
near it. A test that only checked "the solver converged" would prove nothing —
a solver can converge confidently onto the wrong point.
