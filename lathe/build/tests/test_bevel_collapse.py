"""
A bevel must not collapse a finely tessellated surface. Bug hunt B2.

The worker merges near-coplanar triangles back into logical faces before it
bevels, so that a triangulated cube is not rounded along its diagonals. It did so
pairwise: every adjacent pair within coplanar_deg merged. On a surface sampled
finer than that, every pair is within a degree, so the whole dome chained into
one polygon, and a dome revolved from an arc sampled every 0.9 degrees came out
of bevel() as a flat disc: 39% of its volume, 16 mm tall to 4 mm, reported
"delivered on all 96 edges", verifier PASS.

Regions now grow from a seed face and admit a neighbour only if it is within
coplanar_deg of the SEED, so a merged face can never bend more than that.
"""

import math
import unittest

import lathe.build as B
from lathe.build.bevel import blender_path

needs_blender = unittest.skipUnless(
    blender_path() is not None, "bevel needs Blender")


def dome(step_deg, R=0.05, rim_h=0.004, cap_h=0.012):
    """A disc with a spherical cap, the cap's arc sampled every `step_deg`."""
    Rc = (R * R + cap_h * cap_h) / (2 * cap_h)
    half = math.asin(R / Rc)
    k = max(2, int(math.degrees(half) / step_deg))
    pts = [(0.0, 0.0), (R, 0.0), (R, rim_h)]
    for i in range(1, k + 1):
        a = half * (1 - i / k)
        pts.append((Rc * math.sin(a), rim_h + (Rc * math.cos(a) - (Rc - cap_h))))
    pts[-1] = (0.0, rim_h + cap_h)
    return pts


@needs_blender
class TestFinelyTessellatedSurfaces(unittest.TestCase):
    def _bevel(self, step, **kw):
        s = B.revolve(dome(step), segments=48, tag="body", name="SM_Dome")
        return s, B.bevel(s, width=0.0012, segments=2, **kw)

    def test_a_dome_sampled_under_a_degree_keeps_its_height(self):
        for step in (0.5, 0.9):
            with self.subTest(step=step):
                s, b = self._bevel(step)
                self.assertGreater(b.volume() / s.volume(), 0.99)
                self.assertGreater(b.bbox()[1][2], 0.99 * s.bbox()[1][2])

    def test_a_dome_sampled_coarsely_still_does(self):
        for step in (2.0, 6.0):
            with self.subTest(step=step):
                s, b = self._bevel(step)
                self.assertGreater(b.volume() / s.volume(), 0.99)
                self.assertAlmostEqual(b.bbox()[1][2], s.bbox()[1][2], places=3)

    def test_a_cylinder_of_400_facets_delivers_its_width(self):
        """It delivered 0.39 mm of 5 mm before, for the same reason."""
        c = B.cylinder(0.1, 0.05, segments=400, center=False, tag="body",
                       name="SM_C")
        b = B.bevel(c, width=0.005, segments=3)
        self.assertTrue(b.bevel_report.ok, b.bevel_report.describe())
        self.assertGreater(b.volume() / c.volume(), 0.99)

    def test_a_plain_box_is_unchanged_by_the_new_merge(self):
        """Triangulated flat faces must still merge, or the diagonals are
        rounded too: 12 edges on a box, not 18."""
        r = B.bevel(B.box(0.1, 0.1, 0.1, tag="b", name="SM_B"), width=0.005,
                    segments=2).bevel_report
        self.assertEqual(r.edges, 12)

    def test_a_region_is_bounded_by_its_seed_not_its_neighbours(self):
        """With a generous tolerance the old pairwise merge flattened the dome
        completely. Bounded by the seed, the same setting bends it less than the
        tolerance allows, and past what the guard accepts it raises."""
        s = B.revolve(dome(2.0), segments=48, tag="body", name="SM_Dome")
        ok = B.bevel(s, width=0.0012, segments=2, coplanar_deg=5.0)
        self.assertGreater(ok.volume() / s.volume(), 0.99)
        with self.assertRaises(B.LatheError) as ctx:
            B.bevel(s, width=0.0012, segments=2, coplanar_deg=60.0)
        self.assertIn("collapsed the solid", str(ctx.exception))
        self.assertIn("coplanar_deg", str(ctx.exception))

    def test_a_maximum_rounding_is_not_mistaken_for_a_collapse(self):
        """A cube with an oversize width becomes a rounded block (41% of its
        volume), keeps its bounding box, and is a legitimate result: reported
        short, not refused."""
        cube = B.box(0.1, 0.1, 0.1, tag="b", name="SM_B")
        b = B.bevel(cube, width=0.12, segments=2)
        self.assertTrue(b.bevel_report.clamped)
        self.assertAlmostEqual(b.bbox()[1][0] - b.bbox()[0][0], 0.1, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=1)
