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
from lathe.viewer import to_html, write


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
