"""
Tests for the .lathe language.

The point of the language is that a bare number is a type error, so most of these
assert that something is *rejected*. A DSL that accepts `box(600, 600, 600)` has
no reason to exist.
"""

import unittest

from lathe.lang import interp, syntax, values as V
from lathe.lang.interp import LatheRuntimeError, run
from lathe.lang.syntax import LatheSyntaxError, parse, tokenize
from lathe.lang.values import UnitError

CRATE = '''
# DESIGN.md section 4, as real source.
param size    = 600mm
param slat_w  = 80mm
param slat_n  = 5
param rail    = 60mm
param bevel_w = 4mm

let body   = box(size, size, size) @ origin(bottom_center)
let cutter = box(slat_w * 0.35, size * 1.1, size - rail * 2) @ origin(center)
let slots  = array(cutter, axis = X, count = slat_n, span = size * 0.9)
let carved = body - slots

assert components(carved) == 1

let rim   = carved.edges(dihedral > 60deg, length > 20mm)
let final = bevel(carved, rim, width = bevel_w, segments = 2)

emit "SM_Crate_Wood_LOD0" {
  profile : game_prop_pbr
  budget  : { tris_lod0: 20000 }
}
'''


class TestUnits(unittest.TestCase):
    def test_same_dimension_unifies(self):
        self.assertEqual(V.Length.of(600, "mm"), V.Length.of(0.6, "m"))
        self.assertEqual(V.Angle.of(180, "deg").rad, V.Angle.of(3.141592653589793,
                                                               "rad").rad)

    def test_length_plus_length(self):
        self.assertEqual(V.add(V.Length.of(1, "m"), V.Length.of(500, "mm")),
                         V.Length.of(1.5, "m"))

    def test_length_plus_bare_number_is_an_error(self):
        with self.assertRaises(UnitError) as ctx:
            V.add(V.Length.of(1, "m"), V.Scalar(2))
        self.assertIn("4.1", str(ctx.exception))

    def test_length_times_length_is_area(self):
        self.assertIsInstance(V.mul(V.Length.of(2, "m"), V.Length.of(3, "m")),
                              V.Area)

    def test_length_over_length_is_dimensionless(self):
        self.assertEqual(V.div(V.Length.of(1, "m"), V.Length.of(500, "mm")),
                         V.Scalar(2))

    def test_angle_and_length_do_not_mix(self):
        with self.assertRaises(UnitError):
            V.add(V.Angle.of(90, "deg"), V.Length.of(1, "m"))

    def test_comparing_different_dimensions_raises_rather_than_lying(self):
        # Returning False would let a nonsense predicate select nothing quietly.
        with self.assertRaises(UnitError):
            V.compare(">", V.Angle.of(60, "deg"), V.Length.of(20, "mm"))

    def test_as_metres_rejects_a_bare_number_by_name(self):
        with self.assertRaises(UnitError) as ctx:
            V.as_metres(V.Scalar(600), "box width")
        msg = str(ctx.exception)
        self.assertIn("box width", msg)
        self.assertIn("600mm", msg)

    def test_as_number_rejects_a_length(self):
        with self.assertRaises(UnitError) as ctx:
            V.as_number(V.Length.of(4, "mm"), "segments")
        self.assertIn("Drop the unit", str(ctx.exception))

    def test_division_by_zero_is_named(self):
        with self.assertRaises(UnitError):
            V.div(V.Length.of(1, "m"), V.Scalar(0))


class TestLexer(unittest.TestCase):
    def test_unit_suffixed_numbers(self):
        got = [(t.kind, t.value, t.unit) for t in tokenize("600mm 0.6m 60deg 5")
               if t.kind in ("number", "unit_number")]
        self.assertEqual(got, [("unit_number", 600.0, "mm"),
                               ("unit_number", 0.6, "m"),
                               ("unit_number", 60.0, "deg"),
                               ("number", 5.0, None)])

    def test_unknown_unit_is_a_lex_error_listing_the_real_ones(self):
        with self.assertRaises(LatheSyntaxError) as ctx:
            parse("param x = 600foo")
        self.assertIn("not a unit", str(ctx.exception))
        self.assertIn("mm", str(ctx.exception))

    def test_comments_and_blank_lines(self):
        p = parse("# just a comment\n\nparam a = 1mm  # trailing\n")
        self.assertEqual(len(p.statements), 1)

    def test_ge_does_not_lex_as_gt_then_eq(self):
        ops = [t.text for t in tokenize("a >= b") if t.kind == "op"]
        self.assertEqual(ops, [">="])

    def test_error_points_at_the_column(self):
        with self.assertRaises(LatheSyntaxError) as ctx:
            parse("param x = 1mm\nlet y = $\n")
        self.assertEqual(ctx.exception.line, 2)


class TestParser(unittest.TestCase):
    def test_statements(self):
        p = parse(CRATE)
        kinds = [type(s).__name__ for s in p.statements]
        self.assertEqual(kinds.count("Param"), 5)
        self.assertEqual(kinds.count("Let"), 6)
        self.assertIn("Assert", kinds)
        self.assertIn("Emit", kinds)

    def test_selector_predicates_parse_as_predicates(self):
        p = parse("let r = c.edges(dihedral > 60deg, length > 20mm)")
        method = p.statements[0].expr
        self.assertEqual([(a, o) for a, o, _ in method.predicates],
                         [("dihedral", ">"), ("length", ">")])

    def test_profile_lists_and_tuples(self):
        p = parse("let b = revolve([(0mm, 0mm), (110mm, 65mm)], segments = 48)")
        call = p.statements[0].expr
        self.assertIsInstance(call.args[0], syntax.ListLit)
        self.assertEqual(len(call.args[0].items), 2)

    def test_emit_options_and_nested_map(self):
        p = parse('emit "X" {\n profile: game_prop_pbr\n budget: { tris_lod0: 15 }\n}')
        emit = p.statements[0]
        self.assertEqual(emit.name, "X")
        self.assertIn("budget", emit.options)

    def test_missing_statement_keyword_is_explained(self):
        with self.assertRaises(LatheSyntaxError) as ctx:
            parse("size = 600mm")
        self.assertIn("param", str(ctx.exception))


class TestInterpreterRejections(unittest.TestCase):
    """Everything the language exists to refuse."""

    def bad(self, src):
        with self.assertRaises((LatheRuntimeError, LatheSyntaxError)) as ctx:
            run(src)
        return str(ctx.exception)

    def test_bare_number_as_a_length(self):
        msg = self.bad('let b = box(600, 600, 600)\nemit "X" {}')
        self.assertIn("bare number", msg)
        self.assertIn("600mm", msg)

    def test_bare_number_as_an_angle(self):
        msg = self.bad('let b = box(1m,1m,1m)\nlet e = b.edges(dihedral > 60)\n'
                       'emit "X" {}')
        self.assertIn("angle", msg)

    def test_length_where_a_count_belongs(self):
        msg = self.bad('let b = box(1m,1m,1m)\n'
                       'let c = cylinder(1m, 1m, segments = 8mm)\nemit "X" {}')
        self.assertIn("plain number", msg)

    def test_mixing_dimensions_in_arithmetic(self):
        msg = self.bad('param x = 1m + 2\nemit "X" {}')
        self.assertIn("4.1", msg)

    def test_unknown_name_lists_what_is_defined(self):
        msg = self.bad('param a = 1mm\nlet b = box(a, a, nope)\nemit "X" {}')
        self.assertIn("Defined so far", msg)

    def test_unknown_function_lists_the_real_ones(self):
        msg = self.bad('let b = frobnicate(1mm)\nemit "X" {}')
        self.assertIn("Available:", msg)

    def test_unknown_selector_predicate(self):
        msg = self.bad('let b = box(1m,1m,1m)\nlet e = b.edges(wibble > 1mm)\n'
                       'emit "X" {}')
        self.assertIn("no predicate", msg)

    def test_failed_assertion_names_the_expectation(self):
        msg = self.bad('let b = box(1m,1m,1m)\nassert components(b) == 7\n'
                       'emit "X" {}')
        self.assertIn("assert", msg.lower())

    def test_failed_cardinality_assertion_carries_the_selector(self):
        msg = self.bad('let b = box(1m,1m,1m)\n'
                       'let e = b.edges(dihedral > 60deg)\n'
                       'assert count(e) == 99\nemit "X" {}')
        self.assertIn("actual", msg)

    def test_uv_is_refused_rather_than_silently_ignored(self):
        msg = self.bad('let b = box(1m,1m,1m)\nuv b by angle_seams()\nemit "X" {}')
        self.assertIn("not implemented", msg)

    def test_program_with_no_solid(self):
        msg = self.bad("param a = 1mm")
        self.assertIn("never produced a solid", msg)

    def test_errors_carry_a_line_number(self):
        msg = self.bad('param a = 1mm\nparam b = 2mm\nlet c = box(1, 1, 1)')
        self.assertIn("line 3", msg)


class TestInterpreterBuilds(unittest.TestCase):
    def test_the_design_doc_crate(self):
        solid, it = run(CRATE)
        self.assertTrue(solid.status_ok())
        self.assertEqual(solid.name, "SM_Crate_Wood_LOD0")
        self.assertEqual(solid.components(), 1)
        self.assertEqual(it.emit_options["profile"], "game_prop_pbr")
        self.assertGreaterEqual(it.assertions, 1)
        # 600 mm cube, and the origin moved it to sit on z=0.
        lo, hi = solid.bbox()
        self.assertAlmostEqual(hi[0] - lo[0], 0.6, places=6)
        self.assertAlmostEqual(lo[2], 0.0, places=9)
        self.assertEqual(len(it.params), 5)

    def test_mm_and_m_are_interchangeable(self):
        a = run('let b = box(600mm, 600mm, 600mm)\nemit "X" {}')[0]
        b = run('let b = box(0.6m, 0.6m, 0.6m)\nemit "X" {}')[0]
        self.assertAlmostEqual(a.volume(), b.volume(), places=12)

    def test_inches_work_too(self):
        s = run('let b = box(1in, 1in, 1in)\nemit "X" {}')[0]
        self.assertAlmostEqual(s.volume(), 0.0254 ** 3, places=12)

    def test_revolve_from_a_profile(self):
        s = run('let bowl = revolve([(0mm,0mm), (110mm,0mm), (110mm,65mm),'
                ' (104mm,65mm), (0mm,6mm)], segments = 48)\nemit "X" {}')[0]
        self.assertTrue(s.status_ok())
        self.assertEqual(s.components(), 1)

    def test_solid_difference_and_tags(self):
        s = run('let a = box(1m,1m,1m, tag = "body")\n'
                'let b = box(0.2m,2m,0.2m, tag = "cut")\n'
                'let c = a - b\nemit "X" {}')[0]
        self.assertEqual(set(s.tags.values()), {"body", "cut"})

    def test_origin_bottom_center(self):
        s = run('let b = box(1m,1m,1m) @ origin(bottom_center)\nemit "X" {}')[0]
        lo, hi = s.bbox()
        self.assertAlmostEqual(lo[2], 0.0, places=9)
        self.assertAlmostEqual((lo[0] + hi[0]) / 2, 0.0, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=1)
