"""
Tests for the .lathe language.

The point of the language is that a bare number is a type error, so most of these
assert that something is *rejected*. A DSL that accepts `box(600, 600, 600)` has
no reason to exist.
"""

import unittest

from lathe.build.bevel import blender_path
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

    def test_unknown_uv_method_is_named(self):
        # `uv` used to be refused outright as unimplemented. It works now, so what
        # remains worth guarding is that a method nobody wrote is not silently
        # accepted.
        msg = self.bad('let b = box(1m,1m,1m)\nuv b by lightning()\nemit "X" {}')
        self.assertIn("angle_seams", msg)

    def test_uv_rejects_a_bare_number_for_the_threshold(self):
        msg = self.bad('let b = box(1m,1m,1m)\n'
                       'uv b by angle_seams(threshold = 66)\nemit "X" {}')
        self.assertIn("angle", msg)

    def test_program_with_no_solid(self):
        msg = self.bad("param a = 1mm")
        self.assertIn("never produced a solid", msg)

    def test_errors_carry_a_line_number(self):
        msg = self.bad('param a = 1mm\nparam b = 2mm\nlet c = box(1, 1, 1)')
        self.assertIn("line 3", msg)


class TestInterpreterBuilds(unittest.TestCase):
    @unittest.skipUnless(blender_path() is not None,
                         "the crate bevels, and bevel needs Blender")
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

    def test_uv_statement_produces_a_uv_set(self):
        s = run('let b = box(200mm, 200mm, 200mm)\n'
                'uv b by angle_seams(threshold = 66deg, resolution = 1024)\n'
                'emit "X" {}')[0]
        self.assertTrue(s.has_uv)
        self.assertEqual(s.uv.shape, (s.n_tris, 3, 2))

    def test_uv_statement_honours_a_density_target(self):
        import lathe.build as LB
        s = run('let b = box(200mm, 200mm, 200mm)\n'
                'uv b by angle_seams(resolution = 1024, texel_density = 5.12)\n'
                'emit "X" {}')[0]
        d = LB.texel_density(s, 1024)
        # Padding imposes a floor on small charts, so only the low end can be
        # expected to land on target -- see uvmap.unwrap's W302 note.
        self.assertAlmostEqual(d["min"], 5.12, delta=0.3)

    def test_origin_bottom_center(self):
        s = run('let b = box(1m,1m,1m) @ origin(bottom_center)\nemit "X" {}')[0]
        lo, hi = s.bbox()
        self.assertAlmostEqual(lo[2], 0.0, places=9)
        self.assertAlmostEqual((lo[0] + hi[0]) / 2, 0.0, places=9)


# Joint stations: base 0, lower 80, upper 420, head 720 mm up +Z. Each part is
# authored in rest space where its joint actually is, and the two overlap across
# the pivot at 420 mm on purpose -- `lathe/pose/README.md` measured why a rig
# whose parts merely touch comes apart, or welds, depending on the pose.
ARM = '''
joint base                     axis Z
joint lower from base  at (0mm, 0mm, 80mm)  axis Y limit (-90deg, 90deg)
joint upper from lower at (0mm, 0mm, 340mm) axis Y limit (-140deg, 140deg)
joint head  from upper at (0mm, 0mm, 300mm) limit (0deg, 0deg)

let arm_lo = box(50mm, 50mm, 360mm).at(z = 250mm)
let arm_hi = box(45mm, 45mm, 340mm).at(z = 570mm)
bind lower to arm_lo
bind upper to arm_hi
'''


class TestArticulation(unittest.TestCase):
    """`joint`, `bind` and the section 5.4 `solve { }` block."""

    def test_a_rig_is_declared_in_source_order(self):
        _, it = run(ARM + 'emit "SM_Arm_Steel_LOD0" {}')
        self.assertEqual([j.name for j in it.joints],
                         ["base", "lower", "upper", "head"])
        self.assertEqual(sorted(it.bindings), ["lower", "upper"])

    def test_the_emitted_solid_is_the_posed_rig(self):
        solid, it = run(ARM + 'emit "SM_Arm_Steel_LOD0" {}')
        self.assertEqual(solid.name, "SM_Arm_Steel_LOD0")
        self.assertGreater(solid.n_tris, 0)
        self.assertEqual(sorted(set(solid.tags.values())), ["body"])

    def test_angles_are_typed_so_a_bare_number_is_rejected(self):
        """The whole reason the language exists, applied to joints."""
        with self.assertRaises(UnitError):
            run('joint a limit (-90, 90)\nlet s = box(1mm,1mm,1mm)\nbind a to s')

    def test_a_param_drives_a_joint_angle(self):
        """The section 10 loop in one line: a viewer hands back
        `param tilt = 25deg` and the rig moves."""
        import math

        _, it = run('param tilt = 25deg\n'
                    'joint a axis Y limit (-90deg, 90deg) angle tilt\n'
                    'let s = box(1mm,1mm,1mm)\nbind a to s')
        self.assertAlmostEqual(math.degrees(it.pose["a"]), 25.0, places=9)

    def test_an_angle_outside_its_limit_is_clamped_not_accepted(self):
        import math

        _, it = run('joint a axis Y limit (-10deg, 10deg) angle 80deg\n'
                    'let s = box(1mm,1mm,1mm)\nbind a to s')
        self.assertAlmostEqual(math.degrees(it.pose["a"]), 10.0, places=9)

    def test_axis_takes_a_name_or_a_vector(self):
        _, by_name = run('joint a axis Y\nlet s = box(1mm,1mm,1mm)\nbind a to s')
        _, by_vec = run('joint a axis (0, 1, 0)\nlet s = box(1mm,1mm,1mm)\nbind a to s')
        self.assertEqual(by_name.joints[0].axis, by_vec.joints[0].axis)

    def test_binding_twice_unions_rather_than_replaces(self):
        """Placed clear of each other, so the added volume cannot hide inside
        geometry that was already there."""
        _, it = run('joint a\n'
                    'let s = box(10mm, 10mm, 10mm)\n'
                    'let t = box(10mm, 10mm, 10mm).at(x = 50mm)\n'
                    'bind a to s\nbind a to t')
        self.assertAlmostEqual(it.bindings["a"].volume(), 2 * 0.01 ** 3, places=12)

    # -- solve ------------------------------------------------------------

    def test_solve_places_the_tip_on_the_target(self):
        from lathe import pose as P

        solid, it = run(ARM + 'solve {\n'
                              '  reach head to (-280mm, 0mm, 460mm) within 1mm\n'
                              '}\nemit "SM_Arm_Steel_LOD0" {}')
        self.assertEqual(len(it.solved), 1)
        self.assertTrue(it.solved[0].converged, it.solved[0].describe())
        tip = P.positions(it.skeleton(), it.pose)["head"]
        for got, want in zip(tip, (-0.280, 0.0, 0.460)):
            self.assertAlmostEqual(got, want, delta=0.001)

    def test_solve_moves_the_geometry_too(self):
        """The solve is not just bookkeeping -- the emitted solid moves.

        Checked on the bounding box rather than the volume: the parts overlap
        across the pivot on purpose, so bending the joint changes how much of
        that overlap the union double-counts. The assembly's volume is
        *supposed* to move, which `lathe/pose/tests` pins per part instead.
        """
        rest = run(ARM + 'emit "X" {}')[0]
        posed = run(ARM + 'solve { reach head to (-280mm, 0mm, 460mm) }\n'
                          'emit "X" {}')[0]
        self.assertAlmostEqual(rest.bbox()[1][2], 0.740, places=3)   # stands up
        self.assertLess(posed.bbox()[0][0], -0.25)                   # swung to -X
        self.assertLess(posed.bbox()[1][2], rest.bbox()[1][2])       # and down

    def test_an_unreachable_goal_is_an_error_naming_the_deficit(self):
        with self.assertRaises(LatheRuntimeError) as cm:
            run(ARM + 'solve { reach head to (9m, 0mm, 0mm) }\nemit "X" {}')
        msg = str(cm.exception)
        self.assertIn("did not converge", msg)
        self.assertIn("beyond the chain", msg)

    def test_solving_is_deterministic(self):
        src = ARM + 'solve { reach head to (-280mm, 0mm, 460mm) }\nemit "X" {}'
        self.assertEqual(run(src)[1].pose, run(src)[1].pose)

    def test_solve_without_a_rig_says_so(self):
        with self.assertRaises(LatheRuntimeError) as cm:
            run('solve { reach x to (0mm, 0mm, 0mm) }')
        self.assertIn("solve needs a rig", str(cm.exception))

    # -- rejections -------------------------------------------------------

    def test_rejections(self):
        for label, src, want in (
            ("unknown clause", "joint a wobble 3", "no clause 'wobble'"),
            ("unknown parent", "joint a\njoint b from nope", "unknown parent"),
            ("duplicate joint", "joint a\njoint a", "declared twice"),
            ("reversed limit", "joint a limit (90deg, -90deg)", "is above max"),
            ("mistyped axis", "joint a axis Q", "unknown axis"),
            ("bind unknown", "let s = box(1mm,1mm,1mm)\nbind nope to s",
             "unknown joint"),
            ("bind non-solid", "joint a\nlet n = 3\nbind a to n", "needs a solid"),
            ("reach without to", "joint a\nsolve { reach a within 2mm }",
             "needs `to`"),
        ):
            with self.subTest(case=label):
                with self.assertRaises((LatheRuntimeError, LatheSyntaxError)) as cm:
                    run(src)
                self.assertIn(want, str(cm.exception))

    def test_every_rejection_names_a_line(self):
        """Section 7's shape: what broke, and where."""
        with self.assertRaises(LatheRuntimeError) as cm:
            run("joint a\njoint b from nope")
        self.assertIn("line 2", str(cm.exception))

    def test_a_program_without_joints_is_untouched(self):
        """The whole feature is additive; every existing program must be
        byte-identical through it."""
        solid, it = run('let b = box(1m, 1m, 1m)\nemit "X" {}')
        self.assertEqual(it.joints, [])
        self.assertEqual(it.bindings, {})
        self.assertAlmostEqual(solid.volume(), 1.0, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=1)
