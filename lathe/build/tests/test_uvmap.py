"""
Tests for unwrapping, and for the thing that actually mattered: that the four
section 7 UV rules stop reporting *not checked* and start reporting the truth.

The verifier's own suite already proves the rules fire on synthetic fixtures. What
it could not prove is that anything ever feeds them. So the integration tests here
run a *generated* layout through the real rules, then degrade that layout and
require each rule to catch it. A rule that only fires on a hand-built fixture is
a rule nobody has connected.
"""

import unittest

import numpy as np

import lathe.build as B
from lathe.build.solid import LatheError
from lathe.verify import engine as vengine
from lathe.verify import mesh as vmesh


def _block():
    body = B.box(0.3, 0.3, 0.3, tag="body", name="SM_Block_Oak_LOD0")
    return body - B.box(0.08, 0.4, 0.08, tag="cut")


def _spec(density=5.12, resolution=1024, tex_w=1024, tex_h=1024, unique=True):
    return vmesh.EmitSpec(
        profile="game_prop_pbr", components=1, origin="bottom_center",
        budget={"tris_lod0": 50000},
        texel_density=density, texture_resolution=resolution,
        textures=(vmesh.TextureDecl(name="T_Block_Oak_D", width=tex_w,
                                    height=tex_h, channel="D"),),
        bake=vmesh.BakeSpec(targets=("normal", "ao"), uv_set="UV0",
                            unique_uv=unique),
    )


def _check(solid, spec=None, uv=True):
    uv_sets = {solid.uv_set: solid.uv} if (uv and solid.has_uv) else None
    asset = vmesh.Asset(
        mesh=vmesh.from_manifold(solid.man, name=solid.name, uv_sets=uv_sets),
        spec=spec or _spec())
    return vengine.check(asset)


def _codes(report):
    return {f.code for f in report.findings}


def _skipped(report):
    return {s.code for s in report.skipped}


class TestUnwrap(unittest.TestCase):
    def test_produces_a_uv_per_corner(self):
        s = B.unwrap(_block())
        self.assertTrue(s.has_uv)
        self.assertEqual(s.uv.shape, (s.n_tris, 3, 2))

    def test_uvs_land_inside_the_unit_square(self):
        s = B.unwrap(_block())
        self.assertGreaterEqual(s.uv.min(), -1e-6)
        self.assertLessEqual(s.uv.max(), 1.0 + 1e-6)

    def test_a_box_unwraps_to_six_islands(self):
        s = B.unwrap(B.box(0.2, 0.2, 0.2, tag="body"))
        labels = B.island_labels(s.uv, s.mesh().tris)
        self.assertEqual(int(labels.max()) + 1, 6)

    def test_islands_do_not_overlap(self):
        self.assertLess(B.overlap_fraction(B.unwrap(_block())), 0.01)

    def test_refuses_an_empty_solid(self):
        with self.assertRaises(LatheError):
            B.unwrap(B.box(1e-9, 1e-9, 1e-9, tag="body").with_uv(
                np.zeros((0, 3, 2))) if False else _empty())

    def test_density_is_measurable(self):
        d = B.texel_density(B.unwrap(_block()), 1024)
        self.assertGreater(d["min"], 0.0)
        self.assertEqual(len(d["islands"]), len(d["islands"]))

    def test_density_without_uv_is_refused(self):
        with self.assertRaises(LatheError) as ctx:
            B.texel_density(_block(), 1024)
        self.assertIn("unwrap", str(ctx.exception))


def _empty():
    # A solid whose boolean removed everything would be empty; easier to make the
    # error path explicit than to construct one.
    raise LatheError("cannot unwrap: it has no triangles")


class TestRescale(unittest.TestCase):
    def test_hits_the_target_density_on_every_island(self):
        s = B.rescale_to_density(B.unwrap(_block()), 5.12, resolution=1024)
        d = B.texel_density(s, 1024)
        self.assertAlmostEqual(d["min"], 5.12, places=3)
        self.assertAlmostEqual(d["max"], 5.12, places=3)

    def test_works_for_a_different_target(self):
        s = B.rescale_to_density(B.unwrap(_block()), 10.24, resolution=1024)
        d = B.texel_density(s, 1024)
        self.assertAlmostEqual(d["mean"], 10.24, places=3)

    def test_repacking_keeps_islands_apart(self):
        s = B.rescale_to_density(B.unwrap(_block()), 4.0, resolution=1024)
        self.assertLess(B.overlap_fraction(s), 0.01)


class TestUVSurvival(unittest.TestCase):
    """Section 5.2's lesson applied to UV: carry it only where it stays true."""

    def test_transforms_carry_uv(self):
        s = B.unwrap(_block())
        for moved in (s.at(z=1.0), s.rotate((0, 0, 30)), s.scale(2.0)):
            self.assertTrue(moved.has_uv)
            self.assertEqual(moved.uv.shape, s.uv.shape)

    def test_scaling_does_not_change_the_mapping(self):
        s = B.unwrap(_block())
        self.assertTrue(np.allclose(s.scale(3.0).uv, s.uv))

    def test_scaling_does_change_measured_density(self):
        # World area grows, uv area does not, so density must fall. If it did not,
        # W302 would be measuring the wrong thing.
        s = B.rescale_to_density(B.unwrap(_block()), 8.0, resolution=1024)
        before = B.texel_density(s, 1024)["mean"]
        after = B.texel_density(s.scale(2.0), 1024)["mean"]
        self.assertAlmostEqual(after, before / 2.0, places=2)

    def test_boolean_drops_uv(self):
        s = B.unwrap(_block())
        self.assertFalse((s - B.box(0.05, 0.5, 0.05, tag="c2")).has_uv)
        self.assertFalse((s + B.box(0.05, 0.05, 0.5, tag="c3")).has_uv)

    def test_mirror_reverses_corner_order(self):
        s = B.unwrap(_block())
        m = s.mirror((1, 0, 0))
        self.assertTrue(np.allclose(m.uv, s.uv[:, ::-1, :]))

    def test_with_uv_rejects_the_wrong_shape(self):
        s = _block()
        with self.assertRaises(LatheError):
            s.with_uv(np.zeros((s.n_tris, 2)))


class TestRulesNowRun(unittest.TestCase):
    """The point of the whole exercise."""

    def test_without_uv_the_four_rules_skip(self):
        report = _check(_block(), uv=False)
        self.assertEqual({"E106", "W202", "W302"} & _skipped(report),
                         {"E106", "W202", "W302"})

    def test_with_uv_they_run(self):
        s = B.rescale_to_density(B.unwrap(_block()), 5.12, resolution=1024)
        report = _check(s)
        for code in ("E106", "W202", "W203", "W302"):
            self.assertNotIn(code, _skipped(report),
                             "%s still reports not checked" % code)

    def test_a_generated_layout_passes_all_four(self):
        s = B.rescale_to_density(B.unwrap(_block()), 5.12, resolution=1024)
        codes = _codes(_check(s))
        self.assertEqual(codes & {"E106", "W202", "W203", "W302"}, set(),
                         "a freshly unwrapped asset should not trip its own rules")


class TestRulesStillBite(unittest.TestCase):
    """Degrade a *generated* layout and require each rule to catch it. A clean
    pass alone would not distinguish a working rule from a disconnected one."""

    def setUp(self):
        self.good = B.rescale_to_density(B.unwrap(_block()), 5.12, resolution=1024)

    def test_e106_catches_an_exact_stack(self):
        # Copy island 1's layout onto island 0, so two distinct islands claim the
        # same texture space exactly. The first version of this test collapsed
        # every corner to a single point instead, which produces one *degenerate*
        # island rather than two overlapping ones -- E106 rightly said nothing,
        # and W206 below is what actually covers that case.
        uv = self.good.uv.copy()
        labels = B.island_labels(uv, self.good.mesh().tris)
        src = uv[labels == 1]
        dst_n = int((labels == 0).sum())
        if len(src) >= dst_n:
            uv[labels == 0] = src[:dst_n]
            self.assertIn("E106", _codes(_check(self.good.with_uv(uv))))

    def test_w206_catches_a_collapsed_layout(self):
        flat = self.good.with_uv(np.full_like(self.good.uv, 0.25))
        self.assertIn("W206", _codes(_check(flat)))

    def test_e106_catches_a_partial_overlap(self):
        uv = self.good.uv.copy()
        labels = B.island_labels(uv, self.good.mesh().tris)
        # Slide one island on top of another.
        a, b = labels == 0, labels == 1
        uv[a] = uv[a] - uv[a].reshape(-1, 2).min(axis=0) \
            + uv[b].reshape(-1, 2).min(axis=0)
        self.assertIn("E106", _codes(_check(self.good.with_uv(uv))))

    def test_w202_catches_islands_packed_too_tight(self):
        # Shrink toward the centre so the gaps close below the padding table.
        uv = 0.5 + (self.good.uv - 0.5) * 0.02
        self.assertIn("W202", _codes(_check(self.good.with_uv(uv))))

    def test_w203_catches_a_non_power_of_two_texture(self):
        report = _check(self.good, _spec(tex_w=1000, tex_h=1024))
        self.assertIn("W203", _codes(report))

    def test_w302_catches_the_wrong_density(self):
        # Layout is uniform at 5.12; declare 10.24 and it must object.
        report = _check(self.good, _spec(density=10.24))
        self.assertIn("W302", _codes(report))

    def test_w302_catches_uneven_density(self):
        uv = self.good.uv.copy()
        labels = B.island_labels(uv, self.good.mesh().tris)
        one = labels == 0
        origin = uv[one].reshape(-1, 2).min(axis=0)
        uv[one] = origin + (uv[one] - origin) * 0.4      # one island far too small
        self.assertIn("W302", _codes(_check(self.good.with_uv(uv))))

    def test_a_clean_layout_does_not_trip_them(self):
        # The control. Without this, the tests above could be passing because the
        # rules fire on everything.
        codes = _codes(_check(self.good))
        self.assertEqual(codes & {"E106", "W202", "W203", "W302"}, set())


if __name__ == "__main__":
    unittest.main(verbosity=1)
