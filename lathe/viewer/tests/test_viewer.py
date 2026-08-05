"""
Tests for the interactive preview.

These check the page the generator emits, not the WebGL it runs -- a browser is
needed for that, and `lathe/mcp/tests/test_stdio.py` plus a manual load cover it.
What is checkable here is exactly what went wrong in practice: an unreplaced
placeholder, mesh data that does not round-trip, and statistics that disagree with
the geometry they claim to describe.
"""

import base64
import json
import os
import re
import tempfile
import unittest

import numpy as np

import lathe.build as B
from lathe import pose as P
from lathe.viewer import to_html, write


def _arm_rig():
    """A two-hinge arm, one part per joint, overlapping across each pivot."""
    sk = P.Skeleton((
        P.Joint("base", axis=(0.0, 0.0, 1.0)),
        P.Joint("lower", parent="base", offset=(0.0, 0.0, 0.08),
                axis=(0.0, 1.0, 0.0), limit=(-P.deg(90), P.deg(90))),
        P.Joint("upper", parent="lower", offset=(0.0, 0.0, 0.34),
                axis=(0.0, 1.0, 0.0), limit=(-P.deg(140), P.deg(140))),
        P.Joint("head", parent="upper", offset=(0.0, 0.0, 0.30), limit=(0.0, 0.0)),
    ))
    return (P.Rig(sk, name="SM_LampArm_Steel_LOD0")
            .bind("base", B.cylinder(0.10, 0.09, segments=16, center=False, tag="base"))
            .bind("lower", B.box(0.05, 0.05, 0.34, tag="lower").at(z=0.25))
            .bind("upper", B.box(0.045, 0.045, 0.32, tag="upper").at(z=0.57)))


def _cube(tag="body"):
    return B.box(0.2, 0.2, 0.2, tag=tag, name="SM_Test_Cube_LOD0")


def _two_part():
    a = B.box(0.4, 0.4, 0.04, tag="seat", name="SM_Test_Thing_LOD0").at(z=0.4)
    b = B.box(0.04, 0.04, 0.4, tag="leg").at(x=0.15, y=0.15, z=0.2)
    return a + b


def _payload(html):
    m = re.search(r'<script id="lathe-data" type="application/json">(.*?)</script>',
                  html, re.S)
    assert m, "the embedded data block is missing"
    return json.loads(m.group(1))


def _decode(b64, dtype):
    return np.frombuffer(base64.b64decode(b64), dtype=dtype)


class TestPageShape(unittest.TestCase):
    def setUp(self):
        self.html = to_html(_cube(), materials={"body": "wood_mid"})

    def test_every_python_placeholder_is_filled(self):
        left = {t for t in re.findall(r"__[A-Z][A-Z0-9_]*__", self.html)
                if t in ("__TITLE__", "__NAME__", "__DATA__")}
        self.assertEqual(left, set(), "unfilled placeholders: %s" % left)

    def test_the_shader_token_is_substituted_globally(self):
        # __NTAG__ is deliberately left for the page's own JS to fill. It appears
        # twice in the fragment shader -- the array size and the loop bound -- and
        # a plain string replace only swapped the first, which broke every page.
        self.assertGreaterEqual(self.html.count("__NTAG__"), 2)
        self.assertIn("/__NTAG__/g", self.html)

    def test_is_self_contained(self):
        # A strict CSP blocks every external host, so any absolute URL in a src
        # or href would silently fail to load.
        for m in re.finditer(r'(?:src|href)\s*=\s*"([^"]+)"', self.html):
            self.assertFalse(m.group(1).startswith(("http:", "https:", "//")),
                             "external reference: %s" % m.group(1))

    def test_omits_the_document_skeleton(self):
        # The Artifact pipeline supplies doctype, head and body itself.
        low = self.html.lower()
        for tag in ("<!doctype", "<html", "<head>", "<body"):
            self.assertNotIn(tag, low)

    def test_declares_a_title_and_names_the_asset(self):
        self.assertIn("<title>SM_Test_Cube_LOD0</title>", self.html)
        self.assertIn("SM_Test_Cube_LOD0", self.html)

    def test_defines_both_themes(self):
        # The viewer's toggle stamps data-theme, which must beat the media query.
        self.assertIn("prefers-color-scheme: dark", self.html)
        self.assertIn('[data-theme="light"]', self.html)
        self.assertIn('[data-theme="dark"]', self.html)

    def test_escapes_a_hostile_name(self):
        html = to_html(_cube().rename('<script>x</script>'))
        self.assertNotIn("<script>x</script>", html)
        self.assertIn("&lt;script&gt;", html)


class TestMeshPayload(unittest.TestCase):
    def setUp(self):
        self.solid = _two_part()
        self.data = _payload(to_html(self.solid))

    def test_positions_are_unindexed_triangles(self):
        pos = _decode(self.data["pos"], np.float32)
        self.assertEqual(len(pos), self.solid.n_tris * 9)

    def test_one_normal_and_one_tag_per_corner(self):
        n = len(_decode(self.data["nrm"], np.float32))
        t = len(_decode(self.data["tag"], np.float32))
        self.assertEqual(n, self.solid.n_tris * 9)
        self.assertEqual(t, self.solid.n_tris * 3)

    def test_normals_are_unit_length(self):
        nrm = _decode(self.data["nrm"], np.float32).reshape(-1, 3)
        lengths = np.linalg.norm(nrm, axis=1)
        self.assertTrue(np.allclose(lengths, 1.0, atol=1e-4))

    def test_normals_are_flat_per_triangle(self):
        # All three corners of a triangle share the face normal: smoothing them
        # would hide faceting the mesh actually has.
        nrm = _decode(self.data["nrm"], np.float32).reshape(-1, 3)
        for i in range(0, min(len(nrm), 90), 3):
            self.assertTrue(np.allclose(nrm[i], nrm[i + 1], atol=1e-6))
            self.assertTrue(np.allclose(nrm[i], nrm[i + 2], atol=1e-6))

    def test_tag_indices_stay_inside_the_palette(self):
        tags = _decode(self.data["tag"], np.float32)
        self.assertGreaterEqual(tags.min(), 0)
        self.assertLess(tags.max(), len(self.data["tags"]))

    def test_a_colour_per_tag_in_both_palettes(self):
        n = len(self.data["tags"])
        self.assertEqual(len(self.data["cols"]), n)
        self.assertEqual(len(self.data["tagCols"]), n)
        for c in self.data["cols"] + self.data["tagCols"]:
            self.assertEqual(len(c), 3)
            self.assertTrue(all(0.0 <= x <= 1.0 for x in c))

    def test_identity_palette_separates_two_wooden_parts(self):
        # Both tags map to the same material, so Material mode cannot tell them
        # apart. Tags mode must.
        data = _payload(to_html(_two_part(),
                                materials={"seat": "wood_mid", "leg": "wood_mid"}))
        self.assertEqual(data["cols"][0], data["cols"][1])
        self.assertNotEqual(data["tagCols"][0], data["tagCols"][1])

    def test_per_tag_triangles_sum_to_the_whole_mesh(self):
        self.assertEqual(sum(self.data["tagTris"]), self.solid.n_tris)

    def test_wire_edges_are_creases_not_every_edge(self):
        edge = _decode(self.data["edge"], np.float32)
        self.assertEqual(len(edge) % 6, 0)          # pairs of xyz
        drawn = len(edge) // 6
        total = self.solid.mesh().topology.n_edges
        self.assertGreater(drawn, 0)
        self.assertLess(drawn, total)

    def test_a_smooth_sphere_has_far_fewer_creases_than_a_box(self):
        sphere = _payload(to_html(B.sphere(0.1, segments=48, tag="body")))
        drawn = len(_decode(sphere["edge"], np.float32)) // 6
        total = B.sphere(0.1, segments=48).mesh().topology.n_edges
        self.assertLess(drawn / max(total, 1), 0.5)

    def test_bbox_matches_the_solid(self):
        lo, hi = self.solid.bbox()
        self.assertAlmostEqual(self.data["bbox"]["size"][2], hi[2] - lo[2], places=5)
        self.assertAlmostEqual(self.data["bbox"]["center"][2],
                               (lo[2] + hi[2]) / 2, places=5)


class TestStats(unittest.TestCase):
    def test_counts_agree_with_the_geometry(self):
        solid = _two_part()
        s = _payload(to_html(solid))["stats"]
        msh = solid.mesh()
        self.assertEqual(s["tris"], solid.n_tris)
        self.assertEqual(s["verts"], msh.n_verts)
        self.assertEqual(s["edges"], msh.topology.n_edges)
        self.assertEqual(s["components"], solid.components())

    def test_a_closed_solid_reports_sealed(self):
        self.assertTrue(_payload(to_html(_cube()))["stats"]["watertight"])

    def test_extent_is_reported_in_millimetres(self):
        s = _payload(to_html(_cube()))["stats"]
        self.assertEqual(s["size"], "200 x 200 x 200 mm")

    def test_logical_faces_beat_triangles_for_a_box(self):
        # A cube is 12 triangles but 6 faces. Reporting 12 would mislead.
        s = _payload(to_html(_cube()))["stats"]
        self.assertEqual(s["tris"], 12)
        self.assertEqual(s["faces"], 6)

    def test_genus_counts_a_through_hole(self):
        holed = B.box(0.2, 0.2, 0.2, tag="body") - B.box(0.05, 0.4, 0.05, tag="cut")
        self.assertEqual(_payload(to_html(holed))["stats"]["genus"], 1)


class TestRigPage(unittest.TestCase):
    """A Rig subject turns the preview into a posing surface.

    The kinematics themselves live in `lathe/pose/tests`, and the JavaScript
    needs a browser. What is checkable here is the contract between them: that
    the page is handed everything it needs to pose the model, and nothing that
    contradicts the kernel.
    """

    def setUp(self):
        self.rig = _arm_rig()
        self.data = _payload(to_html(self.rig))

    def test_an_ordinary_solid_carries_no_rig(self):
        """The posing payload is additive. Every existing page is unaffected."""
        plain = _payload(to_html(_cube()))
        for key in ("rig", "jnt", "edgeJnt"):
            self.assertNotIn(key, plain)

    def test_the_skeleton_ships_with_its_limits(self):
        """A slider is built from the declared limit, so a pose taken off this
        page is always one the rig allows. Ship the limit or that guarantee is
        just a hope."""
        joints = {j["name"]: j for j in self.data["rig"]["joints"]}
        self.assertEqual(sorted(joints), ["base", "head", "lower", "upper"])
        self.assertEqual(joints["lower"]["limit"], [-90.0, 90.0])
        self.assertEqual(joints["upper"]["limit"], [-140.0, 140.0])
        self.assertEqual(joints["lower"]["parent"], "base")

    def test_a_welded_joint_is_flagged_rather_than_omitted(self):
        """The tip still has to appear -- it is a named point the program can
        target -- but it must not get a slider that does nothing."""
        joints = {j["name"]: j for j in self.data["rig"]["joints"]}
        self.assertTrue(joints["head"]["locked"])
        self.assertFalse(joints["lower"]["locked"])

    def test_joints_arrive_parents_before_children(self):
        """The page walks this list once to accumulate world matrices, so a
        child ahead of its parent would read a matrix that is not there yet."""
        order = [j["name"] for j in self.data["rig"]["joints"]]
        seen = set()
        for j in self.data["rig"]["joints"]:
            if j["parent"] is not None:
                self.assertIn(j["parent"], seen, "%s precedes its parent" % j["name"])
            seen.add(j["name"])
        self.assertEqual(order[0], "base")

    def test_rest_world_positions_match_the_kernel(self):
        """These are what the page checks its own kinematics against, so they
        have to be Python's answer and not a restatement of the offsets."""
        world = P.positions(self.rig.skeleton)
        for j in self.data["rig"]["joints"]:
            np.testing.assert_allclose(j["restWorld"], world[j["name"]], atol=1e-9)

    def test_every_corner_names_the_joint_that_drives_it(self):
        """One bone per vertex. A corner with no joint could not be posed, and a
        corner naming a joint that carries no geometry could not be drawn."""
        jnt = _decode(self.data["jnt"], np.float32)
        pos = _decode(self.data["pos"], np.float32)
        self.assertEqual(len(jnt), len(pos) // 3)
        bound = self.data["rig"]["bound"]
        self.assertEqual(sorted(set(jnt.tolist())),
                         [float(i) for i in range(len(bound))])

    def test_wire_edges_are_driven_too(self):
        """The wireframe moves with the mesh or it detaches on the first hinge."""
        ej = _decode(self.data["edgeJnt"], np.float32)
        edge = _decode(self.data["edge"], np.float32)
        self.assertEqual(len(ej), len(edge) // 3)

    def test_parts_stay_separable_rather_than_welded(self):
        """`Rig.posed()` unions, which welds the parts into one vertex set and
        makes a hinge impossible. The drawn geometry must be the parts."""
        drawn = len(_decode(self.data["pos"], np.float32)) // 9
        self.assertEqual(drawn, sum(p.n_tris for p in self.rig.bindings.values()))

    def test_statistics_describe_the_assembled_asset(self):
        """The HUD reports what gets budgeted, verified and exported -- the
        union -- not the loose parts the viewer happens to draw. Those are
        different numbers, and the panel must not mix them."""
        assembled = self.rig.at_rest()
        self.assertEqual(self.data["stats"]["tris"], assembled.n_tris)
        self.assertEqual(sum(self.data["tagTris"]), assembled.n_tris)
        self.assertNotEqual(self.data["stats"]["tris"],
                            len(_decode(self.data["pos"], np.float32)) // 9)

    def test_a_starting_pose_becomes_the_baseline(self):
        """Reset returns here and the write-back diffs against it: the user
        wants to send the change they made, not the change since the joint
        tree's zero."""
        data = _payload(to_html(self.rig, pose={"lower": P.deg(-35)}))
        joints = {j["name"]: j for j in data["rig"]["joints"]}
        self.assertAlmostEqual(joints["lower"]["start"], -35.0, places=4)
        self.assertAlmostEqual(joints["upper"]["start"], 0.0, places=4)
        # The structural rest is a separate number and must not have moved.
        np.testing.assert_allclose(joints["lower"]["restWorld"], [0, 0, 0.08], atol=1e-9)

    def test_the_page_carries_the_pose_panel_and_a_writeback_box(self):
        html = to_html(self.rig)
        self.assertIn('id="posesect"', html)
        self.assertIn('id="writeback"', html)
        self.assertIn("pose = {", html)          # the emitted form, in the page

    def test_a_rig_page_is_still_self_contained(self):
        for m in re.finditer(r'(?:src|href)\s*=\s*"([^"]+)"', to_html(self.rig)):
            self.assertFalse(m.group(1).startswith(("http:", "https:", "//")))

    def test_writing_a_rig_reports_the_assembled_triangles(self):
        with tempfile.TemporaryDirectory(prefix="lathe_rig_") as tmp:
            dest = os.path.join(tmp, "arm.html")
            info = write(self.rig, dest, pose={"lower": P.deg(20)})
            self.assertTrue(os.path.exists(dest))
            self.assertEqual(info["tris"], self.rig.at_rest().n_tris)
            self.assertTrue(info["self_contained"])


class TestParameters(unittest.TestCase):
    """Geometry parameters, which the page can request but cannot preview.

    Changing `seat_h` needs Manifold and a re-evaluation, which a file with no
    network cannot do. The page offers the edit and says so; what it must never
    do is imply the model on screen reflects it.
    """

    def _params(self):
        from lathe.lang import values as LV
        return {
            "plinth_r": LV.Length.of(90, "mm"),
            "arm_len": LV.Length.of(0.34, "m"),
            "segments": LV.Scalar(32),
            "tilt": LV.Angle.of(15, "deg"),
        }

    def test_values_arrive_in_the_unit_the_source_wrote(self):
        """`450mm` must come back as `480mm`, not `0.48m`. The write-back is
        pasted over the original line, and silently changing the unit would make
        a correct edit look like a rewrite."""
        data = _payload(to_html(_cube(), params=self._params()))
        by_name = {p["name"]: p for p in data["params"]}
        self.assertEqual((by_name["plinth_r"]["value"], by_name["plinth_r"]["unit"]),
                         (90.0, "mm"))
        self.assertEqual((by_name["arm_len"]["value"], by_name["arm_len"]["unit"]),
                         (0.34, "m"))
        self.assertEqual((by_name["tilt"]["value"], by_name["tilt"]["unit"]),
                         (15.0, "deg"))

    def test_a_dimensionless_param_carries_no_unit(self):
        data = _payload(to_html(_cube(), params=self._params()))
        seg = [p for p in data["params"] if p["name"] == "segments"][0]
        self.assertIsNone(seg["unit"])
        self.assertEqual(seg["kind"], "number")

    def test_plain_numbers_are_accepted(self):
        """The viewer keeps no import of `lathe.lang`, so the bridge is duck
        typed and a bare float has to work."""
        data = _payload(to_html(_cube(), params={"count": 4}))
        self.assertEqual(data["params"][0], {"name": "count", "value": 4.0,
                                             "unit": None, "kind": "number"})

    def test_values_that_are_not_numbers_are_dropped(self):
        """A solid or a string is not something the page can offer an edit for.
        Showing an inert row would be worse than showing none."""
        data = _payload(to_html(_cube(), params={"n": 3, "body": _cube(),
                                                 "label": "wood"}))
        self.assertEqual([p["name"] for p in data["params"]], ["n"])

    def test_a_page_without_params_carries_none(self):
        self.assertNotIn("params", _payload(to_html(_cube())))

    def test_params_need_no_rig(self):
        """A static prop has parameters worth requesting even with no joints."""
        html = to_html(_cube(), params={"size": 3})
        data = _payload(html)
        self.assertIn("params", data)
        self.assertNotIn("rig", data)
        self.assertIn('id="paramsect"', html)

    def test_the_page_says_the_edit_is_not_previewed(self):
        """The load-bearing sentence. Without it, a control that changes nothing
        on screen reads as broken rather than as honest."""
        html = to_html(_cube(), params={"size": 3})
        self.assertIn("rebuilding needs the kernel", html)
        self.assertIn("needs a rebuild, not previewed", html)

    def test_identifiers_are_not_uppercased_by_the_row_style(self):
        """`plinth_r` shown as `PLINTH_R`, or `mm` as `MM`, is a name nobody can
        paste back. The row style uppercases its labels, so these opt out."""
        html = to_html(_cube(), params={"size": 3})
        self.assertIn("span.ident", html)
        self.assertIn("text-transform: none", html)

    def test_params_and_a_rig_coexist(self):
        data = _payload(to_html(_arm_rig(), params=self._params()))
        self.assertIn("rig", data)
        self.assertEqual(len(data["params"]), 4)


class TestWrite(unittest.TestCase):
    def test_writes_a_file_and_reports_it(self):
        with tempfile.TemporaryDirectory(prefix="lathe_view_") as tmp:
            dest = os.path.join(tmp, "nested", "p.html")
            info = write(_cube(), dest, materials={"body": "wood_mid"})
            self.assertTrue(os.path.exists(dest))
            self.assertEqual(info["bytes"], os.path.getsize(dest))
            self.assertTrue(info["self_contained"])
            self.assertEqual(info["tris"], 12)


if __name__ == "__main__":
    unittest.main(verbosity=1)
