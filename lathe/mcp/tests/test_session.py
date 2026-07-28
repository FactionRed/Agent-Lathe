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


if __name__ == "__main__":
    unittest.main(verbosity=1)
