"""
Tests for high-to-low baking.

The ones that matter inspect the *pixels*. A bake that runs and writes a black or
empty image is worse than a bake that fails, because it looks like success — so
the checks below assert the map is a plausible tangent-space normal map, not
merely that a file appeared.

Anything touching Blender is skipped when Blender is absent, so the suite still
runs on a machine that only has the kernel.
"""

import os
import tempfile
import unittest

import numpy as np

import lathe.build as B
from lathe.bake import bake
from lathe.build.bevel import blender_path
from lathe.build.solid import LatheError

HAVE_BLENDER = blender_path() is not None
needs_blender = unittest.skipUnless(HAVE_BLENDER, "Blender not found")


def _pair():
    """A slab with a bevel: the bevel is the only difference, which is exactly
    the detail a normal map should carry."""
    base = B.box(0.30, 0.20, 0.05, tag="body", name="SM_Slab_Oak_LOD0")
    low = base
    high = B.bevel(base, width=0.004, segments=3, min_dihedral=55)
    return low, high


def _unbevelled_pair():
    """Two valid solids and nothing else. The guards below refuse on their
    arguments, so the high mesh only has to exist; building it with bevel() made
    these tests need the Blender they claim not to."""
    low = B.box(0.30, 0.20, 0.05, tag="body", name="SM_Slab_Oak_LOD0")
    high = B.box(0.30, 0.20, 0.05, tag="body", name="SM_Slab_Oak_High")
    return low, high


class TestGuards(unittest.TestCase):
    """These need no Blender: they refuse before spawning anything."""

    def test_low_without_uv_is_refused(self):
        low, high = _unbevelled_pair()
        with self.assertRaises(LatheError) as ctx:
            bake(low, high, tempfile.gettempdir())
        msg = str(ctx.exception)
        self.assertIn("no UV set", msg)
        self.assertIn("unwrap", msg)

    def test_unknown_target_is_named_with_the_alternatives(self):
        low, high = _unbevelled_pair()
        low = B.unwrap(low, resolution=256)
        with self.assertRaises(LatheError) as ctx:
            bake(low, high, tempfile.gettempdir(), targets=("gloss",))
        self.assertIn("gloss", str(ctx.exception))
        self.assertIn("normal", str(ctx.exception))


@needs_blender
class TestBake(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="lathe_bake_test_")
        low, high = _pair()
        cls.low = B.unwrap(low, resolution=256, texel_density=5.12)
        cls.high = high
        cls.info = bake(cls.low, cls.high, cls.tmp, stem="T_Slab_Oak",
                        targets=("normal", "ao"), resolution=256, samples=4)

    def test_writes_the_requested_maps(self):
        self.assertEqual(set(self.info["maps"]), {"normal", "ao"})
        for path in self.info["maps"].values():
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 200)

    def test_reports_the_poly_ratio(self):
        self.assertEqual(self.info["low_tris"], self.low.n_tris)
        self.assertEqual(self.info["high_tris"], self.high.n_tris)
        self.assertGreater(self.info["ratio"], 1.0)

    def test_cage_and_ray_scale_with_the_subject(self):
        # A fixed distance is wrong at both ends of the size range.
        lo, hi = self.low.bbox()
        diag = max(b - a for a, b in zip(lo, hi))
        self.assertAlmostEqual(self.info["cage_extrusion_m"], diag * 0.02, places=5)
        self.assertGreater(self.info["max_ray_distance_m"],
                           self.info["cage_extrusion_m"])

    def _pixels(self, target):
        from PIL import Image
        return np.asarray(Image.open(self.info["maps"][target]).convert("RGB"))

    def test_the_normal_map_is_not_blank(self):
        px = self._pixels("normal")
        self.assertGreater(px.std(), 1.0, "a flat image means the bake found nothing")

    def test_the_normal_map_is_mostly_neutral(self):
        # Tangent space: unbevelled surface is (128, 128, 255). Most of a slab is
        # flat, so most pixels should sit near that. A map that is mostly *not*
        # neutral means the tangent basis is wrong.
        px = self._pixels("normal").reshape(-1, 3).astype(np.float64)
        near = (np.abs(px - np.array([128.0, 128.0, 255.0])) < 12).all(axis=1)
        self.assertGreater(near.mean(), 0.5,
                           "only %.0f%% of pixels are near neutral" % (100 * near.mean()))

    def test_the_normal_map_carries_the_bevel(self):
        # The bevel must show up as pixels tilted away from neutral in R or G.
        px = self._pixels("normal").reshape(-1, 3).astype(np.float64)
        tilted = (np.abs(px[:, 0] - 128) > 25) | (np.abs(px[:, 1] - 128) > 25)
        self.assertGreater(tilted.sum(), 50,
                           "no tilted pixels: the bevel did not transfer")

    def test_blue_dominates_in_tangent_space(self):
        px = self._pixels("normal").reshape(-1, 3).astype(np.float64)
        self.assertGreater(px[:, 2].mean(), px[:, 0].mean())
        self.assertGreater(px[:, 2].mean(), px[:, 1].mean())

    def test_the_ao_map_has_range(self):
        px = self._pixels("ao")
        self.assertGreater(px.max(), px.min(), "AO with no range baked nothing")

    def test_flip_green_follows_the_target_engine(self):
        self.assertFalse(self.info["flip_green"])          # unreal
        info = bake(self.low, self.high, self.tmp, stem="T_Slab_Unity",
                    targets=("normal",), resolution=128, samples=4,
                    target_engine="unity")
        self.assertTrue(info["flip_green"])

    def test_unity_and_unreal_normals_differ_in_green(self):
        from PIL import Image
        a = np.asarray(Image.open(self.info["maps"]["normal"]).convert("RGB"))
        info = bake(self.low, self.high, self.tmp, stem="T_Slab_Cmp",
                    targets=("normal",), resolution=256, samples=4,
                    target_engine="unity")
        b = np.asarray(Image.open(info["maps"]["normal"]).convert("RGB"))
        # Same resolution, so the green channel should be mirrored about 128.
        self.assertGreater(np.abs(a[:, :, 1].astype(int)
                                  - b[:, :, 1].astype(int)).mean(), 1.0,
                           "flip_green changed nothing in the image")


if __name__ == "__main__":
    unittest.main(verbosity=1)
