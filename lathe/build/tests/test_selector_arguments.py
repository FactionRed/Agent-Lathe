"""Selector arguments that are not what they say. Bug hunt lows.

faces(normal=(0, 0)) and (0, 0, 1, 1) were a bare TypeError from inside numpy, and
(0, 0, 0) printed "normal ~ (nan, nan, nan)", matched nothing, and raised a numpy
RuntimeWarning on the way.
"""

import unittest
import warnings

import lathe.build as B


class TestFacesNormalArgument(unittest.TestCase):
    def setUp(self):
        self.box = B.box(0.1, 0.1, 0.1, tag="b", name="SM_B")

    def test_the_wrong_number_of_components_is_a_lathe_error(self):
        for bad in ((0, 0), (0, 0, 1, 1), 5, "up"):
            with self.subTest(normal=bad):
                with self.assertRaises(B.LatheError):
                    self.box.faces(normal=bad)

    def test_a_zero_vector_is_refused_without_a_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(B.LatheError) as ctx:
                self.box.faces(normal=(0, 0, 0))
        self.assertIn("direction", str(ctx.exception))

    def test_nan_is_refused(self):
        with self.assertRaises(B.LatheError):
            self.box.faces(normal=(0, float("nan"), 1))

    def test_a_good_normal_still_selects_and_is_normalised(self):
        self.assertEqual(self.box.faces(normal=(0, 0, 5)).count(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=1)
