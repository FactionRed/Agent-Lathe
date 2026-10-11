"""Workspace and program-execution tests. No Blender required."""

import unittest

from lathe.build.solid import CardinalityError, LatheError
from lathe.mcp import probes
from lathe.mcp.session import ProgramError, Workspace

CHAIR = '''
seat = box(0.42, 0.40, mm(35), tag="seat", name="SM_Test_Chair_LOD0").at(z=0.4325)
legs = [box(0.04, 0.04, 0.415, tag="leg").at(x=sx * 0.162, y=sy * 0.152, z=0.2075)
        for sx in (-1, 1) for sy in (-1, 1)]
chair = union_all([seat] + legs, name="SM_Test_Chair_LOD0")
assert_count(chair.faces(tag="seat"), 6, "seat faces")
spec = {"components": 1, "origin": "bottom_center", "budget": {"tris_lod0": 3000}}
result = chair
'''

CUBE = 'result = box(0.2, 0.2, 0.2, tag="body", name="SM_Cube_Plain_LOD0")'


class TestApply(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()

    def test_commits_and_reports_geometry(self):
        c, solid, diff, is_new = self.ws.apply(CHAIR, "first")
        self.assertTrue(is_new)
        self.assertEqual(c.n_components, 1)
        self.assertEqual(solid.name, "SM_Test_Chair_LOD0")
        self.assertEqual(set(c.tags), {"seat", "leg"})
        self.assertEqual(c.spec["components"], 1)
        self.assertEqual(diff, "")  # root commit has no parent to diff against

    def test_identical_program_is_not_a_new_commit(self):
        first = self.ws.apply(CHAIR, "a")[0]
        c, _, _, is_new = self.ws.apply(CHAIR, "b")
        self.assertFalse(is_new)
        self.assertEqual(c.id, first.id)
        self.assertEqual(len(self.ws.commits), 1)

    def test_edit_produces_a_diff_and_a_new_geometry_hash(self):
        a = self.ws.apply(CHAIR, "a")[0]
        wider = CHAIR.replace("0.42, 0.40", "0.50, 0.40")
        b, _, diff, is_new = self.ws.apply(wider, "wider seat")
        self.assertTrue(is_new)
        self.assertIn("0.50", diff)
        self.assertNotEqual(a.geom_hash, b.geom_hash)
        self.assertEqual(b.parent, a.id)

    def test_program_is_replayable_from_the_commit_alone(self):
        c = self.ws.apply(CHAIR, "a")[0]
        fresh = Workspace()
        again = fresh.apply(c.program, "replay")[0]
        # Purity claim from section 2: same program, same geometry, byte for byte.
        self.assertEqual(c.geom_hash, again.geom_hash)


class TestProgramErrors(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()

    def test_missing_result_is_explained(self):
        with self.assertRaises(ProgramError) as ctx:
            self.ws.apply('x = box(1, 1, 1)')
        self.assertIn("did not assign `result`", str(ctx.exception))

    def test_result_of_wrong_type_is_explained(self):
        with self.assertRaises(ProgramError) as ctx:
            self.ws.apply('result = box(1, 1, 1).volume()')
        self.assertIn("expected a Solid", str(ctx.exception))

    def test_syntax_error_names_the_line(self):
        with self.assertRaises(ProgramError):
            self.ws.apply('result = box(')

    def test_runtime_error_names_the_line_and_lists_the_api(self):
        with self.assertRaises(ProgramError) as ctx:
            self.ws.apply('result = frobnicate(3)\n')
        msg = str(ctx.exception)
        self.assertIn("line 1", msg)
        self.assertIn("Available:", msg)

    def test_cardinality_failure_propagates_intact(self):
        bad = CHAIR.replace('chair.faces(tag="seat"), 6', 'chair.faces(tag="seat"), 99')
        with self.assertRaises(CardinalityError) as ctx:
            self.ws.apply(bad)
        self.assertIn("actual", str(ctx.exception))

    def test_failed_program_leaves_no_commit(self):
        try:
            self.ws.apply('result = 1')
        except ProgramError:
            pass
        self.assertEqual(len(self.ws.commits), 0)
        self.assertIsNone(self.ws.tip())


class TestBranching(unittest.TestCase):
    def setUp(self):
        self.ws = Workspace()
        self.base = self.ws.apply(CHAIR, "base")[0]

    def test_branch_then_diverge(self):
        self.ws.new_branch("wider")
        wider = CHAIR.replace("0.42, 0.40", "0.52, 0.40")
        tip = self.ws.apply(wider, "wider")[0]
        self.assertEqual(self.ws.head, "wider")
        self.assertEqual(tip.parent, self.base.id)
        self.assertEqual(self.ws.branches["main"], self.base.id)
        self.assertNotEqual(self.ws.branches["wider"], self.ws.branches["main"])

    def test_checkout_by_branch_and_by_commit(self):
        self.ws.new_branch("side")
        self.ws.apply(CUBE, "cube")
        self.ws.checkout("main")
        self.assertEqual(self.ws.solid().name, "SM_Test_Chair_LOD0")
        self.ws.checkout(self.base.id[:6])
        self.assertEqual(self.ws.solid().name, "SM_Test_Chair_LOD0")

    def test_head_tilde_walks_back(self):
        second = self.ws.apply(CUBE, "cube")[0]
        self.assertEqual(self.ws.resolve("HEAD"), second.id)
        self.assertEqual(self.ws.resolve("HEAD~1"), self.base.id)

    def test_unknown_ref_lists_what_exists(self):
        with self.assertRaises(LatheError) as ctx:
            self.ws.resolve("nope")
        self.assertIn("Branches:", str(ctx.exception))

    def test_duplicate_branch_is_refused(self):
        self.ws.new_branch("dup")
        with self.assertRaises(LatheError):
            self.ws.new_branch("dup")

    def test_log_walks_parents_newest_first(self):
        second = self.ws.apply(CUBE, "cube")[0]
        log = self.ws.log()
        self.assertEqual([c.id for c in log], [second.id, self.base.id])


class TestQueryGrammar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()
        cls.solid = cls.ws.apply(CHAIR, "base")[1]

    def q(self, expr):
        return probes.evaluate(self.solid, expr)

    def test_scalars_answer_with_units(self):
        self.assertIn("m", self.q("bbox"))
        self.assertIn("m³", self.q("volume"))
        self.assertIn("true", self.q("watertight").lower())

    def test_tags_lists_part_ids(self):
        out = self.q("tags")
        self.assertIn("seat", out)
        self.assertIn("leg", out)

    def test_face_selector_reports_logical_faces(self):
        out = self.q("faces(tag=seat)")
        self.assertIn("6", out)
        self.assertIn("(part, face) keys", out)

    def test_edge_selector_accepts_two_predicates(self):
        out = self.q("edges(dihedral_gt=60, length_gt=0.02)")
        self.assertIn("matched", out)

    def test_symmetry_reports_its_method(self):
        out = self.q("symmetry(YZ)")
        self.assertIn("method", out)

    def test_unknown_expression_returns_the_grammar(self):
        with self.assertRaises(LatheError) as ctx:
            self.q("wat")
        self.assertIn("Valid query expressions", str(ctx.exception))

    def test_malformed_selector_args_are_explained(self):
        with self.assertRaises(LatheError) as ctx:
            self.q("faces(seat)")
        self.assertIn("name=value", str(ctx.exception))

    def test_unknown_tag_names_the_tags_that_exist(self):
        with self.assertRaises(LatheError) as ctx:
            self.q("faces(tag=nope)")
        self.assertIn("Tags present", str(ctx.exception))

    def test_raycast_supplies_a_unit(self):
        # The probe layer rejects an unmarked length (section 4.1). The query
        # layer must state the unit, not route around the check.
        out = self.q("raycast(from=0,-2,0.44, dir=0,1,0)")
        self.assertIn("raycast", out)

    def test_raycast_accepts_an_explicit_unit(self):
        out = self.q("raycast(from=0,-2000,440, dir=0,1,0, unit=mm)")
        self.assertIn("raycast", out)


class TestCheckSeesUv(unittest.TestCase):
    """The UV rules must be reachable from the MCP surface.

    They were not. `_check` built its AssetMesh from the manifold and never
    handed over the solid's UV, so a program that called `unwrap()` got a report
    saying "asset declares no uv set at all" for W202, W206 and W302. That is
    worse than silence: it reads as a fact about the asset rather than a gap in
    the caller, and it made an entire rule set unreachable from the only
    interface an agent uses. `models/build_all.py` passed uv_sets explicitly,
    which is exactly why the rules looked wired up.
    """

    def setUp(self):
        import lathe.build as B
        from lathe.build import unwrap
        self.solid = unwrap(
            B.box(0.4, 0.4, 0.05, tag="top", name="SM_UvReach_Test_LOD0"),
            resolution=1024, texel_density=10.24)

    def test_the_two_adapters_agree_on_triangle_order(self):
        """The invariant the fix rests on.

        `solid.uv` is indexed by the triangle order of `pmesh.from_manifold`
        (to_mesh64). The verifier reads `vmesh.from_manifold` (to_mesh). Handing
        the UV across is only correct while those two orders match -- and if they
        ever diverge, UVs attach to the wrong triangles and no rule can tell,
        because every triangle still has *a* UV. Silent, and worse than the bug
        it replaced.
        """
        import numpy as np
        from lathe.perceive import mesh as pmesh
        from lathe.verify import mesh as vmesh
        pm = pmesh.from_manifold(self.solid.man)
        vm = vmesh.from_manifold(self.solid.man, name="x")
        self.assertTrue(
            np.array_equal(pm.tris, np.array(vm.faces, dtype=np.int64)),
            "to_mesh and to_mesh64 disagree on triangle order -- _check must "
            "remap the UV instead of passing it straight through")
        self.assertTrue(np.array_equal(self.solid.mesh().tris, pm.tris))

    def test_uv_rules_stop_reporting_no_uv_set(self):
        from lathe.verify import engine as vengine
        from lathe.verify import mesh as vmesh
        spec = vmesh.EmitSpec(components=1, texel_density=10.24,
                              texture_resolution=1024)
        asset = vmesh.Asset(
            mesh=vmesh.from_manifold(self.solid.man, name=self.solid.name,
                                     uv_sets={self.solid.uv_set: self.solid.uv}),
            spec=spec)
        skipped = {s.code for s in vengine.check(asset, "game_prop_pbr").skipped}
        for code in ("W202", "W206", "W302"):
            self.assertNotIn(code, skipped,
                             "%s is still skipped, so the UV never arrived" % code)

    def test_the_server_check_path_carries_the_uv(self):
        """End to end through the function lathe_apply and lathe_check both use."""
        from lathe.mcp import server
        ws = server.WS
        program = (
            's = box(0.4, 0.4, 0.05, tag="top", name="SM_UvReach_Test_LOD0")\n'
            'result = unwrap(s, resolution=1024, texel_density=10.24)\n'
            'spec = {"components": 1, "texel_density": 10.24, '
            '"texture_resolution": 1024}\n')
        commit, solid, _d, _n = ws.apply(program, "uv reach", branch="uvreach_test")
        self.assertTrue(solid.has_uv, "the program's unwrap did not stick")
        report = server._check(commit.id)
        self.assertNotIn("no uv set at all", report,
                         "the check path is still blind to UVs")


LATHE_TABLE = '''
param top_w  = 600mm
param thick  = 40mm
let top = box(top_w, top_w, thick) @ origin(bottom_center)
assert components(top) == 1
emit "SM_Lathe_Slab_LOD0" {
  profile    : game_prop_pbr
  budget     : { tris_lod0: 100 }
  components : 1
  origin     : bottom_center
}
'''


class TestLatheLanguage(unittest.TestCase):
    """`.lathe` reachable through the workspace, with the same guarantees.

    The language existed, was tested, and was unreachable: lathe_apply took Python
    only, so the one place `600` is a type error was not on the path an agent
    uses. These pin the seam, not the language -- lathe/lang/tests covers that.
    """

    def setUp(self):
        self.ws = Workspace()

    def bad(self, src, **kw):
        with self.assertRaises(ProgramError) as ctx:
            self.ws.apply(src, **kw)
        return str(ctx.exception)

    # -- detection ---------------------------------------------------------

    def test_detects_the_openers_that_cannot_be_python(self):
        from lathe.mcp.session import detect_language
        for src in ("param a = 1mm", "let b = box(1m, 1m, 1m)",
                    '# note\n\n   emit "X" {}', "unit system = metric",
                    "mat oak on top", "solve { reach head to (0m, 0m, 1m) }"):
            self.assertEqual(detect_language(src), "lathe", src)

    def test_python_assignments_to_lathe_words_stay_python(self):
        from lathe.mcp.session import detect_language
        for src in ("uv = 3\nresult = box(1, 1, 1)", "mat = None", "let_x = 1",
                    CUBE, "# only a comment", ""):
            self.assertEqual(detect_language(src), "python", repr(src))

    # -- running -----------------------------------------------------------

    def test_a_lathe_program_commits_with_its_emit_as_the_spec(self):
        c, solid, _d, is_new = self.ws.apply(LATHE_TABLE, "slab")
        self.assertTrue(is_new)
        self.assertEqual(c.language, "lathe")
        self.assertEqual(solid.name, "SM_Lathe_Slab_LOD0")
        self.assertEqual(c.spec, {"profile": "game_prop_pbr",
                                  "budget": {"tris_lod0": 100},
                                  "components": 1, "origin": "bottom_center"})
        lo, hi = solid.bbox()
        self.assertAlmostEqual(hi[0] - lo[0], 0.6, places=6)

    def test_python_programs_are_still_python(self):
        c = self.ws.apply(CUBE)[0]
        self.assertEqual(c.language, "python")

    def test_repeating_a_lathe_program_is_a_no_op(self):
        self.ws.apply(LATHE_TABLE)
        self.assertFalse(self.ws.apply(LATHE_TABLE)[3])

    def test_a_lathe_commit_replays_from_its_text(self):
        c = self.ws.apply(LATHE_TABLE)[0]
        self.ws._solids.clear()
        self.assertEqual(self.ws.solid(c.id).name, "SM_Lathe_Slab_LOD0")
        self.assertEqual(self.ws.commit(c.id).language, "lathe")

    def test_language_can_be_forced_either_way(self):
        # A Python program forced through the .lathe parser is a syntax error,
        # and the converse -- neither silently falls back to the other.
        self.assertIn("syntax error in the .lathe program",
                      self.bad(CUBE, language="lathe"))
        self.assertIn("SyntaxError", self.bad(LATHE_TABLE, language="python"))

    def test_an_unknown_language_is_refused_before_any_branch_is_made(self):
        with self.assertRaises(LatheError):
            self.ws.apply(CUBE, branch="ghost", language="rust")
        self.assertNotIn("ghost", self.ws.branches)

    # -- what the language is for -------------------------------------------

    def test_a_bare_number_is_a_unit_error_with_a_line_number(self):
        msg = self.bad('param a = 1mm\nlet b = box(600, 600, 600)\n'
                       'emit "SM_X_LOD0" {}')
        self.assertIn("line 2", msg)
        self.assertNotIn("Traceback", msg)

    def test_a_syntax_error_names_the_line(self):
        msg = self.bad('param a = 1mm\nlet b = = box(1m, 1m, 1m)\n')
        self.assertIn("syntax error in the .lathe program", msg)
        self.assertIn("line 2", msg)

    def test_a_failed_assertion_is_a_program_error_not_a_crash(self):
        msg = self.bad('let b = box(1m, 1m, 1m)\n'
                       'assert components(b) == 2\nemit "SM_X_LOD0" {}')
        self.assertIn("assertion failed", msg)

    def test_a_failed_lathe_program_leaves_no_commit(self):
        self.bad('let b = box(600, 600, 600)\nemit "SM_X_LOD0" {}')
        self.assertEqual(self.ws.commits, {})

    # -- the emit block is a declaration, so it must not be dropped ---------

    def test_an_unknown_emit_key_is_refused_not_ignored(self):
        msg = self.bad('let b = box(1m, 1m, 1m)\n'
                       'emit "SM_X_LOD0" { profile: game_prop_pbr\n'
                       '  polycount: 100 }')
        self.assertIn("polycount", msg)
        self.assertIn("Accepted", msg)

    def test_lod_is_refused_with_the_reason(self):
        msg = self.bad('let b = box(1m, 1m, 1m)\n'
                       'emit "SM_X" { lod: 0 }')
        self.assertIn("LOD chain", msg)

    # -- end to end --------------------------------------------------------

    def test_the_check_path_reads_the_emit_spec(self):
        """E107 can only fire if the budget in `emit` reached the verifier.

        The assertion is on the finding's own text, not on the code. A rule that
        is skipped prints its code too ("E107  emit declares no triangle
        budget"), so `assertIn("E107", report)` passes whether the budget got
        through or was dropped -- dropping it was checked and that is exactly
        what it did.
        """
        from lathe.mcp import server
        src = LATHE_TABLE.replace("tris_lod0: 100", "tris_lod0: 4")
        commit = server.WS.apply(src, "tight", branch="lathe_budget_test")[0]
        self.assertEqual(commit.spec["budget"], {"tris_lod0": 4})
        report = server._check(commit.id)
        self.assertIn("over budget 4", report)
        self.assertNotIn("declares no triangle budget", report)

    def test_uv_declared_in_lathe_reaches_the_uv_rules(self):
        from lathe.mcp import server
        src = ('let s = box(400mm, 400mm, 50mm)\n'
               'uv s by angle_seams(resolution = 1024, texel_density = 10.24)\n'
               'emit "SM_LatheUv_Test_LOD0" {\n'
               '  profile: game_prop_pbr\n  components: 1\n'
               '  texel_density: 10.24\n  texture_resolution: 1024\n}\n')
        commit, solid, _d, _n = server.WS.apply(
            src, "lathe uv", branch="lathe_uv_test")
        self.assertTrue(solid.has_uv, "the uv statement did not stick")
        self.assertNotIn("no uv set at all", server._check(commit.id))


if __name__ == "__main__":
    unittest.main(verbosity=1)
