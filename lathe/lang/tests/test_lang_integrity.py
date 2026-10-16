"""
The .lathe language does what it says, and says when it will not.

Bug hunt B5, B6, B10, B11 and the language-level lows. The thread through them is
the one DESIGN.md section 13 is about: input accepted and then not read. `bevel(
carved, rim, ...)` took a selection and used one number from it; `box(depth = 1mm)`
parsed; `components : 1mm` became 0; a helper `let` after `emit` shipped under the
asset's name. Every one built something, and what it built was wrong.
"""

import unittest

import lathe.build as B
from lathe.build.bevel import blender_path, collecting_reports
from lathe.lang.interp import LatheRuntimeError, run
from lathe.lang.syntax import LatheSyntaxError, parse

needs_blender = unittest.skipUnless(blender_path() is not None,
                                    "bevel needs Blender")

BOX = "let b = box(200mm, 100mm, 100mm)\n"
EMIT = 'emit "SM_T_LOD0" {}\n'


def fails(src):
    with unittest.TestCase().assertRaises(
            (LatheRuntimeError, LatheSyntaxError)) as ctx:
        run(src)
    return str(ctx.exception)


class TestArgumentsAreRead(unittest.TestCase):
    """Each of these built, and built the wrong thing."""

    def test_an_argument_a_function_does_not_take_is_an_error(self):
        for call in ("box(100mm, 100mm, 100mm, depth = 1mm)",
                     "box(100mm, 100mm, 100mm, 7mm)",
                     "cylinder(100mm, 50mm, stride = 2)",
                     "sphere(50mm, segmants = 9)",
                     "revolve([(0mm, 0mm), (50mm, 0mm), (50mm, 50mm)], tag = a, "
                     "twist = 3deg)"):
            with self.subTest(call=call):
                msg = fails("let a = %s\n%s" % (call, EMIT))
                self.assertIn("line 1", msg)
                self.assertTrue("has no argument" in msg or "at most" in msg, msg)

    def test_bevel_names_what_it_accepts(self):
        msg = fails(BOX + "let r = b.edges(dihedral > 60deg)\n"
                    "let c = bevel(b, r, width = 2mm, only_tags = seat)\n" + EMIT)
        self.assertIn("only_tags", msg)
        self.assertIn("width", msg)         # the accepted list
        self.assertIn("line 3", msg)

    def test_a_count_below_one_is_not_an_empty_list(self):
        for n in ("0", "-3"):
            with self.subTest(count=n):
                msg = fails(BOX + "let a = array(b, count = %s, axis = X)\n%s"
                            % (n, EMIT))
                self.assertIn("between 1 and", msg)

    def test_a_count_too_large_to_finish_is_refused(self):
        msg = fails(BOX + "let a = array(b, count = 20000, axis = X)\n" + EMIT)
        self.assertIn("between 1 and", msg)

    def test_an_axis_that_is_not_an_axis_is_an_error(self):
        msg = fails(BOX + "let a = array(b, count = 2, axis = 5)\n" + EMIT)
        self.assertIn("axis must be X, Y or Z", msg)
        msg = fails(BOX + 'let a = array(b, count = 2, axis = "W")\n' + EMIT)
        self.assertIn("axis must be X, Y or Z", msg)

    def test_bevel_clamp_and_profile_are_typed(self):
        msg = fails(BOX + "let c = bevel(b, width = 2mm, clamp = 3)\n" + EMIT)
        self.assertIn("clamp must be true or false", msg)
        msg = fails(BOX + "let c = bevel(b, width = 2mm, profile = 2)\n" + EMIT)
        self.assertIn("profile must be in (0, 1]", msg)

    def test_bevel_wants_an_edge_selection_not_faces(self):
        msg = fails(BOX + "let f = b.faces(all)\n"
                    "let c = bevel(b, f, width = 2mm)\n" + EMIT)
        self.assertIn("EDGE selection", msg)


class TestEmitIsDeclarative(unittest.TestCase):
    def test_a_helper_after_emit_does_not_ship_in_its_place(self):
        """B10. The last solid bound used to win, so a scratch solid after the
        emit shipped under the finished asset's name and spec."""
        src = ("let asset = box(200mm, 100mm, 100mm)\n"
               'emit "SM_T_LOD0" { components : 1 }\n'
               "let scratch = box(10mm, 10mm, 10mm)\n")
        solid, it = run(src)
        self.assertEqual(solid.name, "SM_T_LOD0")
        lo, hi = solid.bbox()
        self.assertAlmostEqual(hi[0] - lo[0], 0.2, places=6)

    def test_let_result_names_what_ships(self):
        src = ("let result = box(200mm, 100mm, 100mm)\n"
               "let scratch = box(10mm, 10mm, 10mm)\n")
        solid, _ = run(src)
        lo, hi = solid.bbox()
        self.assertAlmostEqual(hi[0] - lo[0], 0.2, places=6)

    def test_without_either_the_last_solid_still_wins(self):
        solid, _ = run("let a = box(200mm, 100mm, 100mm)\n"
                       "let b = box(10mm, 10mm, 10mm)\n")
        lo, hi = solid.bbox()
        self.assertAlmostEqual(hi[0] - lo[0], 0.01, places=6)

    def test_a_second_emit_is_refused_not_merged(self):
        msg = fails(BOX + 'emit "SM_A_LOD0" { components : 1 }\n'
                    'emit "SM_B_LOD0" {}\n')
        self.assertIn("one `emit`", msg)
        self.assertIn("line 3", msg)

    def test_a_key_given_twice_is_an_error_not_last_wins(self):
        msg = fails(BOX + 'emit "SM_A_LOD0" {\n  components : 1\n  components : 2\n}\n')
        self.assertIn("'components' is given twice", msg)
        self.assertIn("line 4", msg)

    def test_options_are_typed_and_carry_a_line(self):
        cases = (
            ("components : 1mm", "components is a count"),
            ("components : 1.9", "whole number"),
            ("components : -1", "whole number"),
            ("components : \"two\"", "must be a number"),
            ("texel_density : 10mm", "pixels per centimetre"),
            ("texel_density : 0", "must be positive"),
            ("texture_resolution : 0", "whole number"),
            ("origin : somewhere", "origin must be one of"),
            ("budget : { tris_lod0: 1.5 }", "whole number of triangles"),
        )
        for opt, want in cases:
            with self.subTest(opt=opt):
                msg = fails(BOX + 'emit "SM_A_LOD0" {\n  %s\n}\n' % opt)
                self.assertIn(want, msg)
                self.assertIn("line 3", msg)

    def test_good_options_are_typed_values(self):
        _, it = run(BOX + 'emit "SM_A_LOD0" {\n  components : 1\n'
                    "  texel_density : 5.12\n  texture_resolution : 1024\n"
                    "  budget : { tris_lod0: 200 }\n  origin : bottom_center\n"
                    "  profile : game_prop_pbr\n}\n")
        o = it.emit_options
        self.assertEqual(o["components"], 1)
        self.assertIsInstance(o["components"], int)
        self.assertEqual(o["texel_density"], 5.12)
        self.assertEqual(o["texture_resolution"], 1024)
        self.assertEqual(o["budget"], {"tris_lod0": 200})
        self.assertEqual(o["origin"], "bottom_center")


class TestSolveTolerance(unittest.TestCase):
    def test_a_non_positive_within_is_an_error_not_unreachable(self):
        from lathe.lang.tests.test_lang import ARM
        for tol in ("-1mm", "0mm"):
            with self.subTest(within=tol):
                msg = fails(ARM + "solve { reach head to (-280mm, 0mm, 460mm) "
                            "within %s }\n" % tol + EMIT)
                self.assertIn("within must be a positive length", msg)


class TestSourceText(unittest.TestCase):
    def test_a_byte_order_mark_is_not_a_syntax_error(self):
        solid, _ = run("﻿" + BOX + EMIT)
        self.assertEqual(solid.name, "SM_T_LOD0")

    def test_unicode_line_separators_do_not_shift_line_numbers(self):
        """str.splitlines() also splits on \\x0c, \\x85 and U+2028, so a stray one
        moved every later error's line number."""
        for sep in ("\x0c", "\x85", " "):
            with self.subTest(sep=repr(sep)):
                src = "param a = 1mm\n# note %s\nlet c = box(1, 1, 1)\n" % sep
                self.assertIn("line 3", fails(src))

    def test_a_separator_inside_a_comment_does_not_leak_code(self):
        src = BOX + "# comment  let b = box(1, 1, 1)\n" + EMIT
        solid, _ = run(src)
        self.assertEqual(solid.name, "SM_T_LOD0")


class TestAcceptedButUnused(unittest.TestCase):
    """`unit system`, `scene up` and `mat ... on` parse and nothing reads them.
    Refusing them would break a program written to DESIGN.md section 4, so they
    are accepted and *said*."""

    def test_the_interpreter_records_what_it_did_not_use(self):
        src = ("unit system = metric\nscene up = +Z, forward = -Y\n" + BOX
               + "mat oak on b.faces(all)\n" + EMIT)
        _, it = run(src)
        heads = " | ".join(h for h, _ in it.notes)
        self.assertIn("unit system", heads)
        self.assertIn("scene up", heads)
        self.assertIn("mat oak", heads)

    def test_the_session_surfaces_them_on_the_commit(self):
        from lathe.mcp.session import Workspace
        c = Workspace().apply("unit system = metric\n" + BOX + EMIT)[0]
        self.assertTrue(any("has no effect" in n for n in c.bevel_notes),
                        c.bevel_notes)


@needs_blender
class TestBevelTakesTheSelection(unittest.TestCase):
    """B5. `bevel(carved, rim, ...)` recovered only the dihedral threshold of
    `rim` and beveled every edge over it. `assert count(rim) == 8` passed and 12
    edges were beveled; `length > 500mm` selected none and beveled all 12."""

    def _bevel(self, selector, **kw):
        src = (BOX + "let rim = b.edges(%s)\n" % selector
               + "let c = bevel(b, rim, width = 2mm, segments = 2)\n" + EMIT)
        with collecting_reports() as reps:
            solid, it = run(src)
        return solid, it, reps

    def test_only_the_selected_edges_are_beveled(self):
        # The four 200 mm edges of a 200 x 100 x 100 box.
        solid, it, reps = self._bevel("length > 150mm")
        self.assertEqual(len(reps), 1)
        self.assertEqual(reps[0].edges, 4)
        full = B.bevel(B.box(0.2, 0.1, 0.1), width=0.002, segments=2)
        self.assertLess(solid.n_tris, full.n_tris)

    def test_an_empty_selection_is_an_error_not_a_full_bevel(self):
        with self.assertRaises(LatheRuntimeError) as ctx:
            self._bevel("length > 500mm")
        self.assertIn("selected no edges", str(ctx.exception))

    def test_every_edge_selected_is_every_edge_beveled(self):
        _, _, reps = self._bevel("dihedral > 60deg")
        self.assertEqual(reps[0].edges, 12)

    def test_a_selection_from_another_solid_is_refused(self):
        src = (BOX + "let other = box(300mm, 100mm, 100mm)\n"
               "let rim = other.edges(dihedral > 60deg)\n"
               "let c = bevel(b, rim, width = 2mm)\n" + EMIT)
        with self.assertRaises(LatheRuntimeError) as ctx:
            run(src)
        self.assertIn("different solid", str(ctx.exception))

    def test_the_program_is_deterministic(self):
        """B6. The dihedral rode in a side table keyed by id(Selection), never
        cleared, so a reused id inherited an unrelated threshold: 56 triangles
        fresh, 48 in 5 of 25 reruns."""
        src = (BOX + "let rim = b.edges(dihedral > 60deg)\n"
               "let c = bevel(b, rim, width = 2mm, segments = 2)\n" + EMIT)
        counts = {run(src)[0].n_tris for _ in range(4)}
        self.assertEqual(len(counts), 1, counts)

    def test_clamp_and_profile_are_passed_through(self):
        src = (BOX + "let c = bevel(b, width = 2mm, segments = 2, clamp = false)\n"
               + EMIT)
        with collecting_reports() as reps:
            run(src)
        self.assertFalse(reps[0].clamp)


if __name__ == "__main__":
    unittest.main(verbosity=1)
