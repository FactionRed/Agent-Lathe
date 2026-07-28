"""
Lathe perception -- DESIGN.md section 6, tiers 1 and 2.

    from lathe.perceive import tier1, tier2, Mesh, from_manifold

    mesh = from_manifold(solid, name="SM_Crate_Wood")
    print(tier1.report(mesh))          # every numeric probe, one block
    print(tier2.report(mesh))          # topology / UV / material / part tables
    print(tier1.symmetry(mesh, "YZ"))  # one probe, with its method declared

Design commitments this package keeps:

  * probes are pure functions of an immutable Mesh -- no session, no selection,
    no active object (DESIGN.md 2, principle 1);
  * every length carries a unit, on the way out *and* on the way in
    (DESIGN.md 4.1);
  * headless, and Blender-optional: numpy is required, manifold3d is used for
    the exact overlap paths and is optional everywhere else (DESIGN.md 11);
  * text output, shaped for an agent to read instead of looking at pixels
    (DESIGN.md 6, and 13 "renders over-trusted").

Tier 3 (renders) and tier 4 (diffs) are deliberately not here.
"""

from . import fmt, kernel, mesh, tier1, tier2, units
from .mesh import Mesh, from_manifold, to_manifold
from .units import Angle, Area, Length, Point, Volume

__all__ = [
    "tier1",
    "tier2",
    "mesh",
    "kernel",
    "units",
    "fmt",
    "Mesh",
    "from_manifold",
    "to_manifold",
    "Length",
    "Area",
    "Volume",
    "Angle",
    "Point",
]
