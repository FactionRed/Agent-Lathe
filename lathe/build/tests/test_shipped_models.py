"""
The shipped models' bevels deliver the width their parameters say.

bevel() used to return a valid, correctly tagged solid whatever it delivered, and
the chair asked for 2 mm and got 0.99 mm on every edge for as long as it existed.
Nothing could have told, so nothing was tested. bevel_report is what makes this
checkable, and these pin it for the models that bevel.
"""

import os
import sys
import unittest

from lathe.build.bevel import blender_path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

needs_blender = unittest.skipUnless(
    blender_path() is not None, "these models bevel, and bevel needs Blender")


@needs_blender
class TestChair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import models.chair as chair
        cls.chair = chair
        cls.built = chair.build()
        cls.low = chair.build(bevel=False)

    def test_the_two_millimetre_bevel_is_two_millimetres(self):
        r = self.built.bevel_report
        self.assertIsNotNone(r)
        self.assertEqual(r.requested, 0.002)
        self.assertFalse(r.clamped, r.describe())
        self.assertGreater(r.edges, 150)
        self.assertAlmostEqual(r.min_delivered / r.requested, 1.0, places=3)

    def test_a_flush_joint_is_refused_not_shipped_half_width(self):
        """The old geometry. The model's own guard must say so, loudly."""
        with self.assertRaises(Exception) as ctx:
            self.chair.build(stile_drop=0.0)
        self.assertIn("came out short", str(ctx.exception))

    def test_a_step_needs_two_bevel_widths(self):
        """Not a magic number: a drop under 2w makes the bevels at the step's two
        edges collide, so the guard fires. Both sides of that line are checked."""
        self.chair.build(bevel_w=0.001)                     # default drop = 2 mm
        with self.assertRaises(Exception):
            self.chair.build(bevel_w=0.002, stile_drop=0.002)

    def test_the_silhouette_did_not_move(self):
        lo, hi = self.low.bbox()
        self.assertAlmostEqual(hi[2], 0.90, places=6, msg="overall height is the rail")
        self.assertEqual(self.built.components(), 1)

    def test_the_low_poly_is_the_same_chair(self):
        self.assertEqual(self.low.components(), 1)
        self.assertLess(self.low.n_tris, self.built.n_tris)


@needs_blender
class TestBowl(unittest.TestCase):
    def test_the_bowl_bevel_delivers_in_full(self):
        import models.bowl as bowl
        r = bowl.build().bevel_report
        self.assertIsNotNone(r)
        self.assertFalse(r.clamped, r.describe())


if __name__ == "__main__":
    unittest.main(verbosity=1)
