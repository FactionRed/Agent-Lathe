"""
Bevel on curved input, graded against closed-form truth. DESIGN.md section 14.7.

Every fixture in the bevel spike was an axis-aligned box, because its referee
named reference planes by hand. That is the easy case, and section 11 chose
bmesh on it. These fixtures are faceted solids of revolution -- cylinders, a
cone, a bowl, two cylinders stacked into one body -- and bevel_truth.py gets
exact volume, per-tag area and per-triangle provenance for all of them from one
2D polygon.

Two classes. The first needs nothing but Manifold and checks the referee's own
arithmetic against the kernel, because a referee that is wrong grades nothing.
The second needs Blender, and says so by skipping.

What is pinned here is what was MEASURED, not what would be nice:

  * Provenance is exact on curved input, including a concave corner and a
    two-tag body. Zero triangles are mis-tagged. This is section 5.2's claim,
    and it held.
  * Geometry is exact to ~1e-7 whenever the requested width is delivered.
  * At a corner that is not a right angle, profile=0.5 is an ellipse, not a
    circular fillet. Same endpoints, different curve.
  * With clamp=True, the default, the width delivered is silently less than the
    width asked for, by an amount that depends on incidental geometry. The
    expectedFailure below is that gap, written down so it flips loudly when it
    is closed.
"""

import math
import unittest

import numpy as np

import lathe.build as B
from lathe.build.bevel import blender_path
from lathe.build.tests import bevel_truth as T

HAVE_BLENDER = blender_path() is not None
needs_blender = unittest.skipUnless(
    HAVE_BLENDER, "bevel needs Blender; the referee's own checks still run")

N = 48          # facets round the axis


def _grade(solid, profile, n, model="skew", **kw):
    """Bevel `solid` and grade the result against `profile` beveled the same way."""
    out = B.bevel(solid, **kw)
    expected = T.bevel_profile(
        profile, kw["width"], kw.get("segments", 2),
        kw.get("min_dihedral", 30.0), kw.get("max_dihedral", 180.0),
        kw.get("only_tags"), model=model)
    m = out.mesh()
    claimed = [out.tags[int(i)] for i in m.part_id]
    return out, T.grade(expected, n, m.verts, m.tris, claimed, out.volume())


def _stepped():
    lo = B.cylinder(0.05, 0.10, segments=N, center=False, tag="lower", name="SM_S")
    up = B.cylinder(0.08, 0.06, segments=N, center=False, tag="upper",
                    name="SM_S").at(z=0.05)
    prof = T.stacked_cylinders([(0.10, 0.05, "lower"), (0.06, 0.08, "upper")], N)
    return lo + up, prof


# --------------------------------------------------------------------------
# The referee, checked against the kernel. No Blender.
# --------------------------------------------------------------------------

class TestRefereeArithmetic(unittest.TestCase):
    """If these fail, nothing below means anything."""

    PTS = [(0, 0), (0.10, 0), (0.06, 0.15), (0, 0.15)]

    def setUp(self):
        self.solid = B.revolve(self.PTS, segments=N, tag="body", name="SM_Cone")
        self.prof = T.profile_from_revolve(self.PTS, N)

    def test_volume_matches_the_kernel(self):
        self.assertAlmostEqual(
            T.volume(self.prof, N) / self.solid.volume(), 1.0, places=7)

    def test_a_cylinder_is_the_closed_form(self):
        r, h = 0.1, 0.2
        c = B.cylinder(h, r, segments=N, center=False, tag="body", name="SM_C")
        prof = T.profile_from_revolve([(0, 0), (r, 0), (r, h), (0, h)], N)
        # Area of a regular n-gon of circumradius r, times height.
        want = 0.5 * N * r * r * math.sin(2 * math.pi / N) * h
        self.assertAlmostEqual(T.volume(prof, N) / want, 1.0, places=9)
        self.assertAlmostEqual(c.volume() / want, 1.0, places=6)

    def test_every_triangle_of_the_unbeveled_solid_is_on_the_profile(self):
        m = self.solid.mesh()
        best, dist, _a = T.classify(self.prof, N, m.verts, m.tris)
        self.assertEqual(int((best < 0).sum()), 0)
        self.assertLess(float(dist.max()), 1e-7)

    def test_area_by_tag_matches_the_mesh(self):
        m = self.solid.mesh()
        p0, p1, p2 = (m.verts[m.tris[:, i]] for i in range(3))
        mesh_area = float(np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1).sum() / 2)
        self.assertAlmostEqual(
            sum(T.area_by_tag(self.prof, N).values()) / mesh_area, 1.0, places=7)

    def test_the_models_meet_at_a_right_angle_and_part_elsewhere(self):
        square = T.profile_from_revolve([(0, 0), (0.1, 0), (0.1, 0.2), (0, 0.2)], N)
        a = T.bevel_profile(square, 0.005, 4, model="skew")
        b = T.bevel_profile(square, 0.005, 4, model="circle")
        self.assertTrue(np.allclose(a.pts, b.pts, atol=1e-12),
                        "at 90 degrees the two arcs are the same circle")
        c = T.bevel_profile(self.prof, 0.006, 4, model="skew")
        d = T.bevel_profile(self.prof, 0.006, 4, model="circle")
        self.assertGreater(float(np.abs(np.array(c.pts) - np.array(d.pts)).max()), 1e-5)

    def test_both_models_keep_the_endpoints_on_the_faces(self):
        for model in T.MODELS:
            out = T.bevel_profile(self.prof, 0.006, 4, model=model)
            # The first arc point is `width` from the corner along the face.
            corner = np.array(self.prof.pts[1])
            toward = np.array(self.prof.pts[0]) - corner
            toward /= np.linalg.norm(toward)
            want = corner + 0.006 * toward
            self.assertTrue(
                any(np.allclose(p, want, atol=1e-12) for p in out.pts), model)

    def test_an_arc_in_the_limit_removes_the_fillet_area(self):
        """w^2 (1 - pi/4): the textbook area between a right corner and its fillet."""
        w = 0.01
        # Bottom, wall, top, then back down the axis.
        sq = T.Profile(((0.0, 0.0), (0.1, 0.0), (0.1, 0.1), (0.0, 0.1)),
                       ("a", "a", "a", None))
        # Flat cross-section area, no revolve: Shoelace on the 2D polygon.
        def poly_area(p):
            x = np.array([q[0] for q in p.pts]); y = np.array([q[1] for q in p.pts])
            return abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        before = poly_area(sq)
        fine = T.bevel_profile(sq, w, 4096, model="circle")
        self.assertAlmostEqual(
            (before - poly_area(fine)) / (w * w * (1 - math.pi / 4)), 2.0, places=3,
            msg="two corners are beveled; the other two sit on the axis")

    def test_overlapping_bevels_are_refused_not_guessed(self):
        with self.assertRaises(ValueError):
            T.bevel_profile(T.profile_from_revolve(
                [(0, 0), (0.1, 0), (0.1, 0.01), (0, 0.01)], N), 0.006, 2)

    def test_only_tags_needs_both_faces(self):
        _solid, prof = _stepped()
        got = T.bevel_profile(prof, 0.004, 2, only_tags=["upper"])
        # Only the upper cylinder's own top rim has `upper` on both sides.
        self.assertEqual(len(got.pts), len(prof.pts) + 2)


# --------------------------------------------------------------------------
# bevel() on curved input. Needs Blender.
# --------------------------------------------------------------------------

@needs_blender
class TestBevelOnCurvedInput(unittest.TestCase):
    """The bevel runs once per fixture and the tests read the results."""

    @classmethod
    def setUpClass(cls):
        c = B.cylinder(0.2, 0.1, segments=N, center=False, tag="body", name="SM_C")
        cp = T.profile_from_revolve([(0, 0), (0.1, 0), (0.1, 0.2), (0, 0.2)], N)
        cls.cyl_solid = c
        cls.cyl = _grade(c, cp, N, width=0.005, segments=4)

        fp = [(0, 0), (0.10, 0), (0.06, 0.15), (0, 0.15)]
        frustum = B.revolve(fp, segments=N, tag="body", name="SM_Cone")
        fprof = T.profile_from_revolve(fp, N)
        cls.frustum = _grade(frustum, fprof, N, width=0.006, segments=4)
        cls.frustum_circle = _grade(frustum, fprof, N, model="circle",
                                    width=0.006, segments=4)

        bp = [(0, 0), (0.05, 0), (0.10, 0.06), (0.10, 0.07), (0.095, 0.07),
              (0.045, 0.01), (0, 0.01)]
        bowl = B.revolve(bp, segments=N, tag="body", name="SM_Bowl")
        bprof = T.profile_from_revolve(bp, N)
        cls.bowl_free = _grade(bowl, bprof, N, width=0.002, segments=4,
                               min_dihedral=20, clamp=False)
        cls.bowl_clamped = _grade(bowl, bprof, N, width=0.002, segments=4,
                                  min_dihedral=20)

        step, sprof = _stepped()
        cls.step_free = _grade(step, sprof, N, width=0.008, segments=3, clamp=False)
        cls.step_clamped = _grade(step, sprof, N, width=0.008, segments=3)
        cls.step_only_upper = _grade(step, sprof, N, width=0.004, segments=3,
                                     clamp=False, only_tags=["upper"])

    # -- provenance: section 5.2 on curved input ---------------------------

    def test_provenance_is_exact_on_every_fixture(self):
        for label in ("cyl", "frustum", "bowl_free", "bowl_clamped",
                      "step_free", "step_clamped", "step_only_upper"):
            _out, g = getattr(self, label)
            self.assertEqual(g.mis_tagged, 0,
                             "%s: %d triangles claim a tag the geometry denies "
                             "(%.3e m2)" % (label, g.mis_tagged, g.mis_tagged_area))

    def test_new_surface_is_tagged_bevel_and_nothing_else_is(self):
        for label in ("cyl", "step_free"):
            out, g = getattr(self, label)
            self.assertIn("bevel", set(out.tags.values()))
            self.assertAlmostEqual(
                g.area_measured["bevel"] / g.area_truth["bevel"], 1.0, places=5)

    def test_the_concave_corner_keeps_its_tags(self):
        """The step's inner corner fills material. Both tags must survive it."""
        out, g = self.step_free
        for tag in ("lower", "upper"):
            self.assertAlmostEqual(
                g.area_measured[tag] / g.area_truth[tag], 1.0, places=5, msg=tag)

    # -- geometry: exact when the width is delivered ------------------------

    def test_a_cylinder_is_beveled_exactly(self):
        out, g = self.cyl
        self.assertLess(abs(g.rel_volume_error()), 1e-6)
        self.assertLess(abs(g.rel_area_error("body")), 1e-6)
        self.assertLess(abs(g.rel_area_error("bevel")), 1e-5)
        self.assertEqual(g.unmatched, 0)
        self.assertEqual(out.components(), 1)

    def test_facet_edges_are_not_features_at_the_default_threshold(self):
        """48 facets means 7.5 degrees between neighbours, under min_dihedral=30.
        Beveling them would multiply the triangle count and change the volume."""
        out, g = self.cyl
        self.assertLess(out.n_tris, 6 * self.cyl_solid.n_tris)
        self.assertLess(abs(g.rel_volume_error()), 1e-6)

    def test_a_cone_matches_the_skew_profile(self):
        _out, g = self.frustum
        self.assertEqual(g.unmatched, 0)
        self.assertLess(abs(g.rel_volume_error()), 1e-6)
        self.assertLess(abs(g.rel_area_error("bevel")), 1e-5)

    def test_a_bowl_with_inside_and_outside_corners_is_exact_unclamped(self):
        _out, g = self.bowl_free
        self.assertEqual(g.unmatched, 0)
        self.assertLess(abs(g.rel_volume_error()), 1e-6)
        self.assertLess(abs(g.rel_area_error("bevel")), 1e-5)

    def test_a_two_tag_step_is_exact_unclamped(self):
        _out, g = self.step_free
        self.assertEqual(g.unmatched, 0)
        self.assertLess(abs(g.rel_volume_error()), 1e-6)
        for tag in ("lower", "upper", "bevel"):
            self.assertLess(abs(g.rel_area_error(tag)), 1e-5, tag)

    def test_only_tags_bevels_edges_with_both_faces_in_the_set(self):
        _out, g = self.step_only_upper
        self.assertEqual(g.unmatched, 0)
        self.assertLess(abs(g.rel_volume_error()), 1e-6)

    # -- what a requested width is not --------------------------------------

    def test_a_non_right_corner_is_an_ellipse_not_a_fillet(self):
        """Same endpoints as a circular fillet, a different curve between them.

        Not a defect to fix so much as a fact to know before quoting 'bevel
        radius': at 76 degrees and a 6 mm width the area is 0.9% off a true
        fillet and the volume 8e-5. Nothing to see at 90 degrees.
        """
        _o, skew = self.frustum
        _o, circle = self.frustum_circle
        self.assertLess(abs(skew.rel_area_error("bevel")), 1e-5)
        self.assertGreater(abs(circle.rel_area_error("bevel")), 1e-3)
        self.assertLess(abs(circle.rel_area_error("bevel")), 5e-2)
        self.assertGreater(circle.unmatched, 0)

    def test_default_clamp_under_delivers_the_width_on_a_step(self):
        """The characterisation of the gap: asked for 8 mm, got less, no signal.

        On this fixture the clamped bevel is valid, one component, correctly
        tagged -- and the area it removes is off by 41%. A caller has no way to
        see that from the returned Solid.
        """
        out, g = self.step_clamped
        self.assertTrue(out.status_ok())
        self.assertEqual(out.components(), 1)
        self.assertGreater(abs(g.rel_area_error("bevel")), 0.05)

    def test_default_clamp_under_delivers_on_a_bowl_rim_too(self):
        out, g = self.bowl_clamped
        self.assertTrue(out.status_ok())
        self.assertGreater(abs(g.rel_area_error("bevel")), 1e-3)

    @unittest.expectedFailure
    def test_default_clamp_delivers_the_requested_width(self):
        """The contract bevel(width=...) reads as. Fails today; when it passes,
        delete the two characterisation tests above and the expectedFailure."""
        _out, g = self.step_clamped
        self.assertLess(abs(g.rel_area_error("bevel")), 1e-5)


@needs_blender
class TestFacetEdgesAsFeatures(unittest.TestCase):
    """The hazard the default threshold exists to prevent: a min_dihedral under
    the facet angle makes every facet edge a bevel target. No closed form for the
    result, so this checks the properties that must survive anyway."""

    def test_beveling_every_facet_edge_is_still_a_valid_operand(self):
        c = B.cylinder(0.2, 0.1, segments=N, center=False, tag="body", name="SM_C")
        out = B.bevel(c, width=0.001, segments=2, min_dihedral=5.0)
        self.assertTrue(out.status_ok())
        self.assertEqual(out.components(), 1)
        self.assertLess(out.volume(), c.volume())
        self.assertGreater(out.volume(), 0.98 * c.volume())
        self.assertGreater(out.n_tris, 5 * c.n_tris,
                           "facet edges were beveled, so the mesh must have grown")
        # Still a boolean operand: the property that matters downstream.
        cut = out - B.box(0.05, 0.05, 0.3, tag="cut", name="SM_Cut")
        self.assertTrue(cut.status_ok())


if __name__ == "__main__":
    unittest.main(verbosity=1)
