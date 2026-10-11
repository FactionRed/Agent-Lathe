"""
bevel() says how much width it delivered. DESIGN.md section 14.7.

bmesh's clamp_overlap shrinks a bevel where it fears overlap and says nothing.
Measured on curved input an 8 mm request delivered 3.4 to 7.6 mm and came back a
valid, correctly tagged solid. The default was not changed -- clamp=False can
fold the surface over itself, which nothing can grade -- so the answer is that
the call reports what it did.

Two classes. The first feeds delivered_widths a hand-made mesh and needs no
Blender. The second runs the real bevel and needs it.
"""

import asyncio
import unittest

import numpy as np

import lathe.build as B
from lathe.build.bevel import (BevelReport, blender_path, collecting_reports,
                               delivered_widths)

needs_blender = unittest.skipUnless(
    blender_path() is not None, "bevel needs Blender")

N = 48


def _row(a, b, n1, u1, n2, u2):
    return np.array([a, b, n1, u1, n2, u2], dtype=np.float64)


# One picked edge along +X from the origin. The top face (+Z normal) lies on
# the +Y side of it; the front face (+Y normal) lies below, toward -Z.
EDGE = _row((0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 0), (0, 0, -1))


class TestDeliveredWidths(unittest.TestCase):
    """The measurement, on meshes small enough to check by eye."""

    def test_the_nearest_vertex_in_each_face_plane_is_the_boundary(self):
        verts = np.array([
            (0.2, 0.3, 0.0), (0.8, 0.3, 0.0),      # top face boundary, 0.3 in
            (0.2, 0.0, -0.3), (0.8, 0.0, -0.3),    # front face boundary, 0.3 in
            (0.5, 0.9, 0.0), (0.5, 0.0, -0.9),     # farther along both faces
        ])
        got = delivered_widths(verts, EDGE[None], width=0.3)
        self.assertAlmostEqual(float(got[0]), 0.3, places=9)

    def test_an_edge_is_as_wide_as_its_narrower_side(self):
        verts = np.array([(0.5, 0.3, 0.0), (0.5, 0.0, -0.12)])
        got = delivered_widths(verts, EDGE[None], width=0.3)
        self.assertAlmostEqual(float(got[0]), 0.12, places=9)

    def test_a_neighbours_vertex_does_not_lend_its_width(self):
        """The first estimator looked at every bevel vertex near an edge and read
        a 62.5% clamp as 88%. A vertex off both face planes must not count."""
        # (0.5, 0.2, 0.2) is 0.28 from the edge, inside the 0.3 request, and on
        # neither face's plane: a distance-only estimator takes it, 0.1 is right.
        verts = np.array([(0.5, 0.1, 0.0), (0.5, 0.0, -0.1),   # this edge: 0.1
                          (0.5, 0.2, 0.2)])
        got = delivered_widths(verts, EDGE[None], width=0.3)
        self.assertAlmostEqual(float(got[0]), 0.1, places=9)

    def test_vertices_beyond_the_ends_do_not_count(self):
        verts = np.array([(1.4, 0.05, 0.0), (1.4, 0.0, -0.05),
                          (0.5, 0.3, 0.0), (0.5, 0.0, -0.3)])
        got = delivered_widths(verts, EDGE[None], width=0.3)
        self.assertAlmostEqual(float(got[0]), 0.3, places=9)

    def test_nothing_near_the_edge_is_zero_not_the_request(self):
        got = delivered_widths(np.array([(5.0, 5.0, 5.0)]), EDGE[None], width=0.3)
        self.assertEqual(float(got[0]), 0.0)

    def test_no_edges_is_an_empty_result(self):
        got = delivered_widths(np.zeros((3, 3)), np.zeros((0, 6, 3)), width=0.3)
        self.assertEqual(got.shape, (0,))


class TestReportText(unittest.TestCase):
    def test_a_full_delivery_says_so_and_has_no_advice(self):
        r = BevelReport(0.005, 96, 0, 0.005, 0.005, True)
        self.assertFalse(r.clamped)
        self.assertIn("delivered on all 96 edges", r.describe())
        self.assertEqual(r.advice(), "")

    def test_a_short_delivery_names_the_worst_edge_and_the_way_out(self):
        r = BevelReport(0.008, 192, 192, 0.00467, 0.00467, True)
        self.assertTrue(r.clamped)
        self.assertIn("8.00 mm requested", r.describe())
        self.assertIn("4.67 mm (58%)", r.describe())
        self.assertIn("clamp=False", r.advice())

    def test_short_with_clamp_off_does_not_blame_the_clamp(self):
        r = BevelReport(0.008, 4, 4, 0.002, 0.002, False)
        self.assertIn("not bmesh's clamp", r.advice())


@needs_blender
class TestOnRealBevels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c = B.cylinder(0.2, 0.1, segments=N, center=False, tag="body", name="SM_C")
        cls.cyl = B.bevel(c, width=0.005, segments=4)

        lo = B.cylinder(0.05, 0.10, segments=N, center=False, tag="lower", name="SM_S")
        up = B.cylinder(0.08, 0.06, segments=N, center=False, tag="upper",
                        name="SM_S").at(z=0.05)
        cls.step_src = lo + up
        with collecting_reports() as sink:
            cls.step_clamped = B.bevel(cls.step_src, width=0.008, segments=3)
        cls.sink = sink
        cls.step_free = B.bevel(cls.step_src, width=0.008, segments=3, clamp=False)

    def test_an_exact_bevel_reports_no_shortfall(self):
        r = self.cyl.bevel_report
        self.assertIsNotNone(r)
        self.assertFalse(r.clamped)
        self.assertEqual(r.edges, 96)
        self.assertAlmostEqual(r.min_delivered / 0.005, 1.0, places=5)

    def test_a_clamped_bevel_reports_what_it_delivered(self):
        r = self.step_clamped.bevel_report
        self.assertTrue(r.clamped)
        self.assertEqual(r.short, r.edges)
        # 4.67 mm, measured independently from the output slice in the referee's
        # investigation.
        self.assertAlmostEqual(r.min_delivered, 0.00467, places=5)
        self.assertLess(r.min_delivered, 0.008 * 0.7)

    def test_the_report_is_the_width_clamp_false_would_need(self):
        """The strongest check available: bevel again, unclamped, at exactly the
        width that was reported as delivered. It must be the same solid."""
        r = self.step_clamped.bevel_report
        again = B.bevel(self.step_src, width=r.min_delivered, segments=3,
                        clamp=False)
        self.assertAlmostEqual(
            again.volume() / self.step_clamped.volume(), 1.0, places=6)

    def test_clamp_false_delivers_in_full(self):
        r = self.step_free.bevel_report
        self.assertFalse(r.clamped)
        self.assertAlmostEqual(r.min_delivered / 0.008, 1.0, places=5)

    def test_the_collector_sees_the_report_without_the_solid(self):
        self.assertEqual(len(self.sink), 1)
        self.assertIs(self.sink[0], self.step_clamped.bevel_report)

    def test_a_report_does_not_leak_between_collectors(self):
        with collecting_reports() as a:
            pass
        self.assertEqual(a, [])

    # -- through the surface ----------------------------------------------

    PROGRAM = (
        "lo = cylinder(0.05, 0.10, segments=48, center=False, tag='lower', "
        "name='SM_Rep_LOD0')\n"
        "up = cylinder(0.08, 0.06, segments=48, center=False, tag='upper', "
        "name='SM_Rep_LOD0').at(z=0.05)\n"
        "body = bevel(lo + up, width=0.008, segments=3)\n"
        "result = body - box(0.02, 0.02, 0.4, tag='hole', name='SM_Rep_LOD0')\n")

    def test_a_note_survives_the_boolean_that_follows_the_bevel(self):
        from lathe.mcp.session import Workspace
        c = Workspace().apply(self.PROGRAM, "report")[0]
        self.assertEqual(len(c.bevel_notes), 1)
        self.assertIn("came out short", c.bevel_notes[0])

    def test_lathe_apply_prints_it(self):
        from lathe.mcp import server
        out = asyncio.run(server.lathe_apply(server.ApplyInput(
            program=self.PROGRAM, message="report", branch="bevel_report_test")))
        self.assertIn("came out short", out)
        self.assertIn("clamp=False", out)

    def test_a_clean_bevel_adds_no_note(self):
        from lathe.mcp.session import Workspace
        prog = ("c = cylinder(0.2, 0.1, segments=48, center=False, tag='b', "
                "name='SM_Rep_LOD0')\nresult = bevel(c, width=0.005, segments=4)\n")
        self.assertEqual(Workspace().apply(prog, "clean")[0].bevel_notes, ())


if __name__ == "__main__":
    unittest.main(verbosity=1)
