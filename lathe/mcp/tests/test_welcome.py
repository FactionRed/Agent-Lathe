"""
The welcome piece is the first thing a new user sees, so a seed that produces a
broken program is a broken product. These tests build real geometry.

Every failure the generator shipped with was a rule already written down in
DESIGN.md and not applied here: coincident contact is not connection (13), and
logical face counts are tessellation-dependent on curved surfaces (5.3).
"""

import unittest

from lathe.build.bevel import blender_path
from lathe.mcp import welcome
from lathe.mcp.session import Workspace


needs_blender = unittest.skipUnless(
    blender_path() is not None, "this style bevels, and bevel needs Blender")


class TestVariation(unittest.TestCase):
    def test_seed_is_stable_and_seeds_differ(self):
        a = welcome.generate(1234)
        b = welcome.generate(1234)
        c = welcome.generate(1235)
        self.assertEqual(a["program"], b["program"])
        self.assertNotEqual(a["program"], c["program"])

    def test_default_seed_is_stable_within_a_machine(self):
        self.assertEqual(welcome.default_seed(), welcome.default_seed())

    def test_all_four_styles_are_reachable(self):
        seen = {welcome.generate(s)["style"] for s in range(80)}
        self.assertEqual(seen, set(welcome.STYLES))

    def test_palettes_vary(self):
        seen = {welcome.generate(s)["palette"] for s in range(80)}
        self.assertGreater(len(seen), 4)

    def test_style_can_be_forced(self):
        for style in welcome.STYLES:
            self.assertEqual(welcome.generate(7, style)["style"], style)

    def test_unknown_style_lists_the_valid_ones(self):
        with self.assertRaises(ValueError) as ctx:
            welcome.generate(1, "spaceship")
        self.assertIn("stool", str(ctx.exception))

    def test_program_carries_its_seed_in_a_comment(self):
        w = welcome.generate(99)
        self.assertIn(w["seed_label"], w["program"])

    @needs_blender
    def test_materials_cover_every_tag_the_program_uses(self):
        for s in range(24):
            w = welcome.generate(s)
            solid = Workspace().apply(w["program"], "x")[1]
            for tag in set(solid.tags.values()):
                self.assertIn(tag, w["materials"],
                              "seed %d (%s): tag %r has no material"
                              % (s, w["style"], tag))


class TestEveryStyleBuilds(unittest.TestCase):
    """Three seeds per style. Real booleans and a real bevel, so it is slow."""

    def _build(self, style, seed):
        w = welcome.generate(seed, style)
        solid = Workspace().apply(w["program"], "welcome")[1]
        self.assertTrue(solid.status_ok(),
                        "%s seed %d is not a valid solid" % (style, seed))
        self.assertEqual(solid.components(), 1,
                         "%s seed %d came out in %d pieces -- almost always "
                         "coincident contact rather than overlap"
                         % (style, seed, solid.components()))
        self.assertGreater(solid.n_tris, 20)
        self.assertGreater(solid.volume(), 0.0)
        lo, hi = solid.bbox()
        self.assertGreater(hi[2] - lo[2], 0.05, "suspiciously flat")
        return solid

    @needs_blender
    def test_stool(self):
        for seed in (2, 6, 19):
            self._build("stool", seed)

    def test_planter(self):
        for seed in (79, 100, 3):
            self._build("planter", seed)

    @needs_blender
    def test_lantern(self):
        for seed in (35, 5, 23):
            self._build("lantern", seed)

    @needs_blender
    def test_pedestal(self):
        for seed in (0, 9, 11):
            self._build("pedestal", seed)


class TestAudit(unittest.TestCase):
    def test_audit_reports_this_environment_as_ready(self):
        a = welcome.audit()
        self.assertTrue(a["ready_to_build"], a["missing"])
        self.assertIsNone(a["pip_command"])
        self.assertIn("manifold3d", a["present"])

    def test_audit_names_a_pip_command_when_something_is_missing(self):
        real = welcome.DEPS
        try:
            welcome.DEPS = real + (("definitely_not_installed_xyz", "test", "xyz"),)
            a = welcome.audit()
            self.assertFalse(a["ready_to_build"])
            self.assertIn("xyz", a["pip_command"])
        finally:
            welcome.DEPS = real


if __name__ == "__main__":
    unittest.main(verbosity=1)
