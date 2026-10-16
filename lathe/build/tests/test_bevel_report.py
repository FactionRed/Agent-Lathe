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


def _measure(top, front, width, edge=EDGE):
    """delivered_widths for one edge, given each side's polygon (loop order)."""
    top = np.asarray(top, dtype=np.float64).reshape(-1, 3)
    front = np.asarray(front, dtype=np.float64).reshape(-1, 3)
    verts = np.vstack([top, front])
    off = np.array([0, len(top), len(top) + len(front)])
    return float(delivered_widths(edge[None], verts, off, width)[0])


# A face inset 0.3 from the edge, seen from above (z = 0), and the same on the
# front face (y = 0), going down.
TOP_03 = [(0.3, 0.3, 0), (0.7, 0.3, 0), (0.7, 0.9, 0), (0.3, 0.9, 0)]
FRONT_03 = [(0.3, 0, -0.3), (0.7, 0, -0.3), (0.7, 0, -0.9), (0.3, 0, -0.9)]


class TestDeliveredWidths(unittest.TestCase):
    """The measurement, on polygons small enough to check by eye."""

    def test_the_parallel_segment_is_the_boundary(self):
        self.assertAlmostEqual(_measure(TOP_03, FRONT_03, 0.3), 0.3, places=9)

    def test_an_edge_is_as_wide_as_its_narrower_side(self):
        front = [(0.3, 0, -0.12), (0.7, 0, -0.12), (0.7, 0, -0.9), (0.3, 0, -0.9)]
        self.assertAlmostEqual(_measure(TOP_03, front, 0.3), 0.12, places=9)

    def test_a_corner_arc_does_not_pass_for_the_boundary(self):
        """The outline at a corner follows the vertex patch arc, which comes
        within half a width of the edge end. The nearest outline point read the
        chair's exact 0.99 mm bevel as 0.41 mm; the parallel segment is the width."""
        top = [(0.2, 0.15, 0), (0.3, 0.3, 0), (0.7, 0.3, 0), (0.7, 0.9, 0),
               (0.2, 0.9, 0)]
        self.assertAlmostEqual(_measure(top, FRONT_03, 0.3), 0.3, places=9)

    def test_a_boundary_that_runs_past_the_ends_still_counts(self):
        """Where the face turns away (a ring inner rim) the boundary line is
        longer than the edge. Testing vertices against [0, L] lost these."""
        top = [(-0.2, 0.3, 0), (1.2, 0.3, 0), (1.2, 0.9, 0), (-0.2, 0.9, 0)]
        self.assertAlmostEqual(_measure(top, FRONT_03, 0.3), 0.3, places=9)

    def test_outline_still_on_the_old_edge_is_ignored(self):
        """An unbeveled end leaves a vertex, and a segment, on the edge itself."""
        top = [(0.0, 0.0, 0), (1.0, 0.0, 0), (1.0, 0.9, 0), (0.7, 0.3, 0),
               (0.3, 0.3, 0), (0.0, 0.9, 0)]
        self.assertAlmostEqual(_measure(top, FRONT_03, 0.3), 0.3, places=9)

    def test_a_tilted_boundary_on_a_merged_nonplanar_face_is_read_at_mid_edge(self):
        """bmesh merges faces across up to 1 degree, so the polygon is not planar
        and its boundary line is tilted against the edge. The bowl rim read 0.14%
        short from the segment ends, and went unmeasured under a tight parallel
        test. The offset at the middle of the edge is the width."""
        # Line from s=0.2994 at x=0 to s=0.3006 at x=1, lifted 0.01 off the plane
        # at one end (so the face is visibly not planar).
        top = [(0.0, 0.2994, 0.0), (1.0, 0.3006, 0.01), (1.0, 0.9, 0.0),
               (0.0, 0.9, 0.0)]
        self.assertAlmostEqual(_measure(top, FRONT_03, 0.3), 0.3, delta=2e-4)

    def test_a_planar_face_gets_no_slack_for_a_stray_segment(self):
        """The slack is for faces that are not planar. On a planar one, a long
        segment 1.5 degrees off the edge is not its boundary line and must not be
        measured as one (the chair read 0.51 mm for its 0.99 mm bevel)."""
        sl = np.tan(np.radians(1.5))
        top = [(0.05, 0.10, 0), (0.90, 0.10 + 0.85 * sl, 0), (0.7, 0.3, 0),
               (0.3, 0.3, 0)]
        self.assertAlmostEqual(_measure(top, FRONT_03, 0.3), 0.3, places=9)

    def test_a_tiny_edge_falls_back_to_the_corner_vertex(self):
        """No parallel segment exists when the edge is shorter than the bevel:
        the neighbours boundary lines meet in a point, and that is the width."""
        tiny = _row((0, 0, 0), (0.001, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 0),
                    (0, 0, -1))
        top = [(0.0005, 0.0003, 0), (0.5, 0.9, 0), (-0.5, 0.9, 0)]
        front = [(0.0005, 0, -0.0003), (0.5, 0, -0.9), (-0.5, 0, -0.9)]
        self.assertAlmostEqual(_measure(top, front, 0.005, tiny), 0.0003, places=9)

    def test_nothing_in_range_is_nan_not_the_request_and_not_zero(self):
        far = [(0.3, 0.1, 0), (0.7, 0.1, 0), (0.7, 0.9, 0), (0.3, 0.9, 0)]
        # Width asked: 0.005. The only outline is 0.1 away: that is not
        # "delivered in full", and not 0.1 either. It is unknown.
        self.assertTrue(np.isnan(_measure(far, FRONT_03, 0.005)))

    def test_a_face_that_is_gone_is_nan(self):
        self.assertTrue(np.isnan(_measure([], FRONT_03, 0.3)))

    def test_the_reported_width_cannot_exceed_the_request(self):
        """max_delivered used to read 99.94 mm for a 5 mm request."""
        wide = [(0.3, 0.8, 0), (0.7, 0.8, 0), (0.7, 0.9, 0), (0.3, 0.9, 0)]
        got = _measure(wide, FRONT_03, 0.3)
        self.assertTrue(np.isnan(got) or got <= 0.3 * 1.01)

    def test_no_edges_is_an_empty_result(self):
        got = delivered_widths(np.zeros((0, 6, 3)), np.zeros((0, 3)),
                               np.array([0]), 0.3)
        self.assertEqual(got.shape, (0,))

    def test_it_is_linear_in_the_number_of_edges(self):
        """52 s at 32k vertices before. The bound is generous so a slow runner
        does not flake, and far below what an O(V*E) scan would take."""
        import time
        n = 4000
        edges = np.repeat(EDGE[None], n, axis=0)
        poly = np.array(TOP_03 + FRONT_03, dtype=np.float64)
        packed = np.concatenate([poly + [0, 0, 0.01 * i] for i in range(n)])
        edges[:, :2, 2] += (np.arange(n) * 0.01)[:, None]
        off = np.arange(0, 8 * n + 1, 4, dtype=np.int64)
        t0 = time.perf_counter()
        got = delivered_widths(edges, packed, off, 0.3)
        self.assertLess(time.perf_counter() - t0, 30.0)
        self.assertTrue(np.allclose(got, 0.3))


class TestReportText(unittest.TestCase):
    def test_a_full_delivery_says_so_and_has_no_advice(self):
        r = BevelReport(0.005, 96, 0, 0.005, 0.005, True)
        self.assertTrue(r.ok)
        self.assertFalse(r.clamped)
        self.assertIn("delivered on all 96 edges", r.describe())
        self.assertEqual(r.advice(), "")

    def test_a_short_delivery_names_the_worst_edge_and_the_way_out(self):
        r = BevelReport(0.008, 192, 192, 0.00467, 0.00467, True)
        self.assertTrue(r.clamped)
        self.assertFalse(r.ok)
        self.assertIn("8 mm requested", r.describe())
        self.assertIn("4.67 mm (58%)", r.describe())
        self.assertIn("clamp=False", r.advice())

    def test_short_with_clamp_off_does_not_blame_the_clamp(self):
        r = BevelReport(0.008, 4, 4, 0.002, 0.002, False)
        self.assertIn("not bmesh's clamp", r.advice())

    def test_an_unmeasured_edge_is_not_a_pass(self):
        r = BevelReport(0.005, 10, 0, 0.005, 0.005, True, unmeasured=3)
        self.assertFalse(r.clamped)
        self.assertFalse(r.ok)
        self.assertIn("3 of 10 edges could not be measured", r.describe())
        self.assertNotIn("delivered on all", r.describe())
        self.assertIn("not a pass", r.advice())

    def test_nothing_measured_does_not_print_nan_as_a_width(self):
        r = BevelReport(0.005, 4, 0, float("nan"), float("nan"), True,
                        unmeasured=4)
        self.assertFalse(r.ok)
        self.assertNotIn("nan", r.describe().lower())


class TestArguments(unittest.TestCase):
    """Refused before Blender is looked for, so these run anywhere."""

    def setUp(self):
        self.cube = B.box(0.1, 0.1, 0.1, tag="b", name="SM_B")

    def test_a_non_positive_or_non_finite_width_is_refused(self):
        for w in (0.0, -0.001, float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(B.LatheError, msg=repr(w)) as ctx:
                B.bevel(self.cube, width=w)
            self.assertIn("positive, finite", str(ctx.exception))

    def test_a_width_that_is_not_a_number_is_refused(self):
        with self.assertRaises(B.LatheError):
            B.bevel(self.cube, width="2mm")

    def test_zero_segments_are_refused(self):
        with self.assertRaises(B.LatheError):
            B.bevel(self.cube, width=0.005, segments=0)


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

    # -- what the bug hunt found -------------------------------------------

    def test_a_tiny_edge_is_reported_short_not_delivered_in_full(self):
        """B1. A 0.05 mm chamfer edge asked for 5 mm delivers 0.025 mm; the first
        estimator said 99.94 mm delivered and lathe_apply printed nothing."""
        d = 0.05e-3 / np.sqrt(2)
        poly = [(d, 0), (0.1, 0), (0.1, 0.1), (0, 0.1), (0, d)]
        r = B.bevel(B.extrude(poly, 0.1), width=0.005, segments=2).bevel_report
        self.assertFalse(r.ok, r.describe())
        self.assertEqual(r.unmeasured, 0)
        self.assertAlmostEqual(r.min_delivered, 0.025e-3, delta=0.002e-3)
        self.assertLessEqual(r.max_delivered, 0.005 * 1.001)

    def test_high_segment_counts_are_not_reported_short(self):
        """B8. 32 segments, 1 mm, on a cube nothing can clamp: 12 of 12 short at
        95% before, because arc vertices within the old plane tolerance were
        taken for the boundary."""
        cube = B.box(0.1, 0.1, 0.1, tag="b", name="SM_B")
        for w, seg in ((0.001, 32), (0.0005, 64), (0.005, 64)):
            r = B.bevel(cube, width=w, segments=seg).bevel_report
            self.assertTrue(r.ok, "%g m x %d: %s" % (w, seg, r.describe()))

    def test_a_model_far_from_the_origin_bevels_like_one_at_it(self):
        """B8, float32. Blender holds float32, so 1 km out is 0.1 mm of noise
        before anything happens and 100 km loses the bevel (volume -20%). The
        model is recentred on the way in."""
        cube = B.box(0.1, 0.1, 0.1, tag="b", name="SM_B")
        ref = B.bevel(cube, width=0.005, segments=3)
        for off in (1000.0, 100000.0):
            moved = B.bevel(cube.translate((off, 0.0, 0.0)), width=0.005,
                            segments=3)
            self.assertTrue(moved.bevel_report.ok, moved.bevel_report.describe())
            self.assertAlmostEqual(moved.volume() / ref.volume(), 1.0, places=6)

    def test_a_ring_rim_is_measured_not_dropped(self):
        """The inner rim of a ring is a reflex corner for the tread: its boundary
        line runs past the edge ends. Every one of those edges went unmeasured in
        the first rewrite."""
        lo = B.cylinder(0.05, 0.10, segments=N, center=False, tag="lower",
                        name="SM_S")
        up = B.cylinder(0.08, 0.06, segments=N, center=False, tag="upper",
                        name="SM_S").at(z=0.05)
        r = B.bevel(lo + up, width=0.008, segments=3, clamp=False).bevel_report
        self.assertEqual(r.unmeasured, 0)
        self.assertTrue(r.ok, r.describe())

    def test_a_finely_faceted_sphere_is_fully_measured_and_quick(self):
        """B16. sphere(128) took 22 s of 24.6 s in the measurement alone."""
        import time
        sph = B.sphere(0.1, segments=96)
        t0 = time.perf_counter()
        r = B.bevel(sph, width=0.002, segments=2, min_dihedral=2.0,
                    clamp=False).bevel_report
        self.assertLess(time.perf_counter() - t0, 120.0)
        self.assertEqual(r.unmeasured, 0)

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
