"""
Tests for the native glTF reader.

Grouped by what would actually go wrong. The container tests are cheap and
obvious; the ones worth reading are TestAxes (an importer that skips the Y-up
conversion produces a model on its face and *no error*), TestWelding (unwelded, no
real .glb ever becomes a solid) and TestTransforms (a mirrored node read without
rewinding comes back inside out, and every normal-facing check on it then lies).

Nothing here needs Blender. The round-trip class does, and skips without it --
that one closes the loop: our own exporter writes a .glb, this reads it back, and
the volume has to survive.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import tempfile
import unittest

import numpy as np

import lathe.build as B
from lathe.build.bevel import blender_path
from lathe.io import glb
from lathe.io.tests import gltf_fixtures as F

HAVE_BLENDER = blender_path() is not None
needs_blender = unittest.skipUnless(HAVE_BLENDER, "Blender not found")


class GlbCase(unittest.TestCase):
    """Writes fixtures into a per-class temp directory."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="lathe_glb_test_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _write(self, name, meshes, **kw):
        b = kw.pop("builder")
        doc = F.document(b, meshes, **kw)
        return F.write_glb(os.path.join(self.tmp, name), doc, bytes(b.bin))

    def _cube(self, name="cube.glb", **doc_kw):
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        return self._write(name, [[F.primitive(b, pos, tris)]], builder=b, **doc_kw)


# ------------------------------------------------------------------ container


class TestContainer(GlbCase):
    def test_reads_a_hand_built_glb(self):
        imp = glb.read(self._cube("basic.glb"))
        self.assertEqual(imp.mesh.n_tris, 12)
        self.assertEqual(imp.mesh.n_verts, 8, "the 24 stored corners should weld")

    def test_bad_magic_is_named(self):
        path = os.path.join(self.tmp, "bad.glb")
        with open(path, "wb") as fh:
            fh.write(b"NOPE" + b"\x00" * 32)
        with self.assertRaises(glb.GlbError) as ctx:
            glb.read(path)
        # Not 'glTF' and not JSON either, so it should say both.
        self.assertIn("glTF", str(ctx.exception))

    def test_truncation_is_reported_as_truncation(self):
        full = open(self._cube("trunc_src.glb"), "rb").read()
        path = os.path.join(self.tmp, "trunc.glb")
        with open(path, "wb") as fh:
            fh.write(full[:len(full) // 2])
        with self.assertRaises(glb.GlbError) as ctx:
            glb.read(path)
        self.assertIn("truncated", str(ctx.exception))

    def test_a_short_file_is_not_an_index_error(self):
        path = os.path.join(self.tmp, "tiny.glb")
        with open(path, "wb") as fh:
            fh.write(b"glTF")
        with self.assertRaises(glb.GlbError):
            glb.read(path)

    def test_unknown_chunk_types_are_skipped(self):
        # The spec requires a reader to ignore chunks it does not know.
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        doc = F.document(b, [[F.primitive(b, pos, tris)]])
        js = json.dumps(doc).encode("utf-8")
        js += b" " * (-len(js) % 4)
        bn = bytes(b.bin)
        bn += b"\x00" * (-len(bn) % 4)
        junk = b"XXXX"
        body = (struct.pack("<II", len(js), 0x4E4F534A) + js
                + struct.pack("<II", len(junk), 0x12345678) + junk
                + struct.pack("<II", len(bn), 0x004E4942) + bn)
        path = os.path.join(self.tmp, "extra_chunk.glb")
        with open(path, "wb") as fh:
            fh.write(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)
        self.assertEqual(glb.read(path).mesh.n_tris, 12)

    def test_version_one_is_refused_with_a_reason(self):
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        doc = F.document(b, [[F.primitive(b, pos, tris)]])
        doc["asset"]["version"] = "1.0"
        path = F.write_glb(os.path.join(self.tmp, "v1.glb"), doc, bytes(b.bin))
        with self.assertRaises(glb.GlbError) as ctx:
            glb.read(path)
        self.assertIn("2.0", str(ctx.exception))

    def test_records_the_content_hash(self):
        # A program that imports is only reproducible while the file is unchanged,
        # so the hash has to be reported for that to be checkable.
        imp = glb.read(self._cube("hash.glb"))
        self.assertRegex(imp.report["file"]["sha256"], r"^[0-9a-f]{12}$")

    def test_reports_the_generator(self):
        imp = glb.read(self._cube("gen.glb"))
        self.assertIn("lathe test fixture", imp.report["file"]["generator"])

    def test_a_file_with_no_meshes_says_so(self):
        b = F.Builder()
        doc = F.document(b, [])
        doc["meshes"] = []
        path = F.write_glb(os.path.join(self.tmp, "empty.glb"), doc, b"\x00\x00\x00\x00")
        with self.assertRaises(glb.GlbError) as ctx:
            glb.read(path)
        self.assertIn("no meshes", str(ctx.exception))

    def test_missing_file(self):
        with self.assertRaises(glb.GlbError) as ctx:
            glb.read(os.path.join(self.tmp, "nope.glb"))
        self.assertIn("no such file", str(ctx.exception))


class TestGltfJson(GlbCase):
    def test_reads_a_gltf_with_an_embedded_data_uri(self):
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        doc = F.document(b, [[F.primitive(b, pos, tris)]])
        path = F.write_gltf(os.path.join(self.tmp, "embed.gltf"), doc, bytes(b.bin))
        imp = glb.read(path)
        self.assertEqual(imp.mesh.n_tris, 12)
        self.assertEqual(imp.report["file"]["container"], "gltf")

    def test_reads_a_gltf_with_a_sidecar_bin(self):
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        doc = F.document(b, [[F.primitive(b, pos, tris)]], buffer_uri="side.bin")
        path = os.path.join(self.tmp, "side.gltf")
        F.write_gltf(path, doc, bytes(b.bin), embed=False)
        self.assertEqual(glb.read(path).mesh.n_tris, 12)

    def test_a_missing_sidecar_explains_the_difference_from_glb(self):
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        doc = F.document(b, [[F.primitive(b, pos, tris)]], buffer_uri="gone.bin")
        path = os.path.join(self.tmp, "missing_side.gltf")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        with self.assertRaises(glb.GlbError) as ctx:
            glb.read(path)
        self.assertIn("not next to", str(ctx.exception))


class TestUriSafety(GlbCase):
    """A URI in a .gltf is attacker-controlled text. Reading a model must not
    become an arbitrary-file-read primitive."""

    def _with_uri(self, name, uri):
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        doc = F.document(b, [[F.primitive(b, pos, tris)]], buffer_uri=uri)
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        return path

    def test_parent_directory_escape_is_refused(self):
        with self.assertRaises(glb.GlbError) as ctx:
            glb.read(self._with_uri("esc.gltf", "../../../secret.bin"))
        self.assertIn("outside the directory", str(ctx.exception))

    def test_percent_encoded_escape_is_refused(self):
        # Unquoting has to happen before the check, or %2e%2e walks straight out.
        with self.assertRaises(glb.GlbError) as ctx:
            glb.read(self._with_uri("pct.gltf", "%2e%2e/%2e%2e/secret.bin"))
        self.assertIn("outside the directory", str(ctx.exception))

    def test_absolute_path_is_refused(self):
        with self.assertRaises(glb.GlbError) as ctx:
            glb.read(self._with_uri("abs.gltf", "/etc/passwd"))
        self.assertIn("absolute", str(ctx.exception))

    def test_windows_drive_path_is_refused(self):
        with self.assertRaises(glb.GlbError) as ctx:
            glb.read(self._with_uri("drive.gltf", "C:/Windows/win.ini"))
        self.assertIn("absolute", str(ctx.exception))

    def test_http_is_not_fetched(self):
        with self.assertRaises(glb.GlbError) as ctx:
            glb.read(self._with_uri("net.gltf", "https://example.invalid/b.bin"))
        self.assertIn("network", str(ctx.exception))


class TestRefusals(GlbCase):
    def test_draco_is_refused_by_name_with_a_way_out(self):
        path = self._cube("draco.glb", required=["KHR_draco_mesh_compression"])
        with self.assertRaises(glb.GlbError) as ctx:
            glb.read(path)
        msg = str(ctx.exception)
        self.assertIn("KHR_draco_mesh_compression", msg)
        self.assertIn("gltf-transform", msg, "a refusal should say what to do")

    def test_meshopt_is_refused(self):
        path = self._cube("meshopt.glb", used=["EXT_meshopt_compression"])
        with self.assertRaises(glb.GlbError) as ctx:
            glb.read(path)
        self.assertIn("EXT_meshopt_compression", str(ctx.exception))

    def test_an_unrelated_required_extension_still_loads_but_is_noted(self):
        # Refusing to read vertices because of a *material* extension would be
        # following a rendering contract into a tool that does not render.
        path = self._cube("unlit.glb", required=["KHR_materials_unlit"])
        imp = glb.read(path)
        self.assertEqual(imp.mesh.n_tris, 12)
        self.assertTrue(any("KHR_materials_unlit" in n for n in imp.report["notes"]))


# ---------------------------------------------------------------------- axes


class TestAxes(GlbCase):
    def test_gltf_up_becomes_lathe_up(self):
        """glTF +Y is up. Lathe +Z is up. Skipping this lays the asset down and
        raises no error at all, which is why it gets its own test."""
        b = F.Builder()
        pos = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=np.float32)
        tris = np.array([(0, 1, 2)], dtype=np.int64)
        path = self._write("axis.glb", [[F.primitive(b, pos, tris)]], builder=b)
        v = glb.read(path).mesh.verts
        np.testing.assert_allclose(v[np.argmax(v[:, 2])], (0, 0, 1), atol=1e-6)

    def test_gltf_forward_becomes_plus_y(self):
        b = F.Builder()
        pos = np.array([(0, 0, 0), (1, 0, 0), (0, 0, -1)], dtype=np.float32)
        tris = np.array([(0, 1, 2)], dtype=np.int64)
        path = self._write("fwd.glb", [[F.primitive(b, pos, tris)]], builder=b)
        v = glb.read(path).mesh.verts
        np.testing.assert_allclose(v[np.argmax(v[:, 1])], (0, 1, 0), atol=1e-6)

    def test_the_conversion_preserves_winding(self):
        # A rotation with determinant +1 cannot flip a normal. If it did, every
        # face-facing check downstream would be inverted.
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        path = self._write("wind.glb", [[F.primitive(b, pos, tris)]], builder=b)
        from lathe.perceive import mesh as pmesh
        self.assertGreater(pmesh.signed_volume(glb.read(path).mesh), 0.0)

    def test_up_z_skips_the_conversion(self):
        b = F.Builder()
        pos = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=np.float32)
        tris = np.array([(0, 1, 2)], dtype=np.int64)
        path = self._write("asz.glb", [[F.primitive(b, pos, tris)]], builder=b)
        v = glb.read(path, up="z").mesh.verts
        self.assertAlmostEqual(float(v[:, 2].max()), 0.0, places=6)
        self.assertAlmostEqual(float(v[:, 1].max()), 1.0, places=6)


# ------------------------------------------------------------------- welding


class TestWelding(GlbCase):
    def test_a_split_cube_welds_and_becomes_a_solid(self):
        """The load-bearing test. A glTF vertex carries its own normal, so a cube
        arrives as 24 vertices and every edge is a boundary. Without welding no
        real .glb would ever promote."""
        imp = glb.read(self._cube("weld.glb"))
        self.assertEqual(imp.report["weld"]["verts_before"], 24)
        self.assertEqual(imp.report["weld"]["verts_after"], 8)
        self.assertEqual(imp.report["weld"]["merged"], 16)
        self.assertTrue(imp.is_solid, imp.reason)

    def test_without_welding_the_same_cube_is_not_a_solid(self):
        imp = glb.read(self._cube("noweld.glb"), weld=False)
        self.assertFalse(imp.is_solid)
        self.assertEqual(imp.mesh.n_verts, 24)
        self.assertIn("boundary edge", imp.reason)

    def test_a_vertex_on_the_axis_welds(self):
        # -0.0 and 0.0 are distinct bytes. A row-wise byte comparison would leave
        # a vertex sitting exactly on an axis unwelded against its own twin.
        b = F.Builder()
        pos = np.array([(0.0, 0.0, 0.0), (1, 0, 0), (0, 1, 0),
                        (-0.0, -0.0, -0.0), (0, 1, 0), (0, 0, 1)],
                       dtype=np.float32)
        tris = np.array([(0, 1, 2), (3, 4, 5)], dtype=np.int64)
        path = self._write("negzero.glb", [[F.primitive(b, pos, tris)]], builder=b)
        imp = glb.read(path)
        self.assertEqual(imp.mesh.n_verts, 4,
                         "the two origins and the shared (0,1,0) should merge")

    def test_the_tolerance_is_reported(self):
        imp = glb.read(self._cube("tol.glb"), weld_tol=1e-6)
        self.assertEqual(imp.report["weld"]["tol_m"], 1e-6)

    def test_a_coarse_tolerance_does_not_collapse_real_geometry(self):
        # 1 mm apart, welded at 1 um: must stay distinct.
        b = F.Builder()
        pos = np.array([(0, 0, 0), (0.001, 0, 0), (0, 0.001, 0)], dtype=np.float32)
        tris = np.array([(0, 1, 2)], dtype=np.int64)
        path = self._write("coarse.glb", [[F.primitive(b, pos, tris)]], builder=b)
        self.assertEqual(glb.read(path, weld_tol=1e-6).mesh.n_verts, 3)

    def test_degenerate_triangles_from_welding_are_dropped(self):
        # Two corners a hair apart collapse onto one vertex, leaving a triangle
        # that uses it twice. Manifold rejects any mesh holding one.
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        pos = np.vstack([pos, np.array([(0, 0, 0), (1e-12, 0, 0), (0, 0.5, 0)],
                                       dtype=np.float32)])
        tris = np.vstack([tris, np.array([(24, 25, 26)], dtype=np.int64)])
        path = self._write("degen.glb", [[F.primitive(b, pos, tris)]], builder=b)
        imp = glb.read(path)
        self.assertEqual(imp.report["clean"]["degenerate_tris_dropped"], 1)
        self.assertEqual(imp.mesh.n_tris, 12)

    def test_dropping_a_triangle_keeps_the_channels_in_step(self):
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        pos = np.vstack([pos, np.array([(0, 0, 0), (1e-12, 0, 0), (0, 0.5, 0)],
                                       dtype=np.float32)])
        tris = np.vstack([tris, np.array([(24, 25, 26)], dtype=np.int64)])
        uv = np.tile(np.array([[0.25, 0.75]], dtype=np.float32), (len(pos), 1))
        path = self._write("degen_uv.glb", [[F.primitive(b, pos, tris, uv=uv)]],
                           builder=b)
        imp = glb.read(path)
        # Mesh construction validates uv against the triangle count, so a channel
        # left out of step would raise rather than merely disagree.
        self.assertEqual(len(imp.mesh.uv), imp.mesh.n_tris)
        self.assertEqual(len(imp.mesh.part_id), imp.mesh.n_tris)


# ---------------------------------------------------------------- transforms


class TestTransforms(GlbCase):
    def _one_tri(self, node):
        b = F.Builder()
        pos = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=np.float32)
        tris = np.array([(0, 1, 2)], dtype=np.int64)
        node = dict(node)
        node["mesh"] = 0
        return self._write("tf_%d.glb" % abs(hash(json.dumps(node, sort_keys=True))),
                           [[F.primitive(b, pos, tris)]], builder=b, nodes=[node])

    def test_translation_is_baked(self):
        path = self._one_tri({"translation": [2.0, 0.0, 0.0]})
        v = glb.read(path).mesh.verts
        self.assertAlmostEqual(float(v[:, 0].min()), 2.0, places=6)

    def test_scale_is_baked_and_the_deviation_is_reported(self):
        path = self._one_tri({"scale": [3.0, 1.0, 1.0]})
        imp = glb.read(path)
        self.assertAlmostEqual(float(imp.mesh.verts[:, 0].max()), 3.0, places=6)
        # Baking destroys the E104 signal, so the report has to state it.
        self.assertAlmostEqual(imp.report["transform"]["max_scale_deviation"], 2.0,
                               places=6)

    def test_a_node_matrix_is_read_column_major(self):
        # glTF stores matrices column-major. Reading it row-major transposes the
        # rotation, which for a translation puts the offset in the wrong place.
        m = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1]  # translate x=5
        path = self._one_tri({"matrix": m})
        v = glb.read(path).mesh.verts
        self.assertAlmostEqual(float(v[:, 0].min()), 5.0, places=6)

    def test_quaternions_are_xyzw_not_wxyz(self):
        """90 degrees about glTF +Y sends +X to -Z and +Y to +Y. After the
        up-conversion that is +X -> +Y and +Y -> +Z. Read as wxyz the same four
        numbers are a different rotation entirely.

        Asserted on the whole sorted vertex set rather than on an index: welding
        runs the positions through a row-wise unique, which sorts them, so no
        vertex keeps the slot it arrived in.
        """
        s = 2 ** -0.5
        path = self._one_tri({"rotation": [0.0, s, 0.0, s]})
        v = glb.read(path).mesh.verts
        got = np.array(sorted(map(tuple, np.round(v, 6))))
        want = np.array(sorted([(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]))
        np.testing.assert_allclose(got, want, atol=1e-6)

    def test_nested_nodes_compose(self):
        b = F.Builder()
        pos = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=np.float32)
        tris = np.array([(0, 1, 2)], dtype=np.int64)
        nodes = [{"children": [1], "translation": [1.0, 0.0, 0.0]},
                 {"mesh": 0, "translation": [2.0, 0.0, 0.0]}]
        path = self._write("nest.glb", [[F.primitive(b, pos, tris)]], builder=b,
                           nodes=nodes, scene_nodes=[0])
        v = glb.read(path).mesh.verts
        self.assertAlmostEqual(float(v[:, 0].min()), 3.0, places=6)

    def test_one_mesh_on_two_nodes_is_instanced_twice(self):
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        nodes = [{"mesh": 0, "translation": [0.0, 0.0, 0.0]},
                 {"mesh": 0, "translation": [10.0, 0.0, 0.0]}]
        path = self._write("inst.glb", [[F.primitive(b, pos, tris)]], builder=b,
                           nodes=nodes)
        imp = glb.read(path)
        self.assertEqual(imp.report["scene"]["instances"], 2)
        self.assertEqual(imp.mesh.n_tris, 24)
        self.assertEqual(imp.mesh.topology.n_components, 2)

    def test_a_mirrored_node_is_rewound(self):
        """A negative determinant reverses winding. Left uncorrected the part comes
        back inside out, and volume, normals and every facing check invert."""
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        path = self._write("mirror.glb", [[F.primitive(b, pos, tris)]], builder=b,
                           nodes=[{"mesh": 0, "scale": [-1.0, 1.0, 1.0]}])
        imp = glb.read(path)
        self.assertEqual(imp.report["transform"]["mirrored_instances"], 1)
        from lathe.perceive import mesh as pmesh
        self.assertGreater(pmesh.signed_volume(imp.mesh), 0.0,
                           "mirrored geometry came back inside out")
        self.assertTrue(imp.is_solid, imp.reason)

    def test_a_node_cycle_does_not_hang(self):
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        nodes = [{"mesh": 0, "children": [1]}, {"children": [0]}]
        path = self._write("cycle.glb", [[F.primitive(b, pos, tris)]], builder=b,
                           nodes=nodes, scene_nodes=[0])
        self.assertGreaterEqual(glb.read(path).mesh.n_tris, 12)

    def test_a_file_with_no_scene_falls_back_to_the_nodes(self):
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        doc = F.document(b, [[F.primitive(b, pos, tris)]])
        doc.pop("scenes")
        doc.pop("scene")
        path = F.write_glb(os.path.join(self.tmp, "noscene.glb"), doc, bytes(b.bin))
        self.assertEqual(glb.read(path).mesh.n_tris, 12)


# ------------------------------------------------------------------ accessors


class TestAccessors(GlbCase):
    def test_interleaved_positions(self):
        """byteStride larger than the element: position and UV share one view.
        Plenty of tools write this; reading it as tight would shear the mesh."""
        pos, tris = F.cube_split(1.0)
        uv = np.tile(np.array([[0.5, 0.5]], dtype=np.float32), (len(pos), 1))
        packed = np.zeros((len(pos), 5), dtype=np.float32)
        packed[:, :3] = pos
        packed[:, 3:] = uv
        b = F.Builder()
        view = b.view(packed.tobytes(), stride=20)
        p_acc = b.accessor(pos, F.FLOAT, "VEC3", view_index=view, count=len(pos))
        uv_acc = b.accessor(uv, F.FLOAT, "VEC2", view_index=view, offset=12,
                            count=len(uv))
        prim = {"attributes": {"POSITION": p_acc, "TEXCOORD_0": uv_acc},
                "indices": b.accessor(tris.reshape(-1), F.USHORT, "SCALAR")}
        path = self._write("interleaved.glb", [[prim]], builder=b)
        imp = glb.read(path)
        self.assertEqual(imp.mesh.n_tris, 12)
        self.assertTrue(imp.is_solid, imp.reason)
        np.testing.assert_allclose(
            np.abs(imp.mesh.verts).max(), 0.5, atol=1e-6,
            err_msg="interleaved positions were read at the wrong stride")

    def test_unsigned_byte_indices(self):
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        path = self._write("ubyte.glb",
                           [[F.primitive(b, pos, tris, indices_comp=F.UBYTE)]],
                           builder=b)
        self.assertEqual(glb.read(path).mesh.n_tris, 12)

    def test_unsigned_int_indices(self):
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        path = self._write("uint.glb",
                           [[F.primitive(b, pos, tris, indices_comp=F.UINT)]],
                           builder=b)
        self.assertEqual(glb.read(path).mesh.n_tris, 12)

    def test_a_non_indexed_primitive_is_sequential(self):
        b = F.Builder()
        pos, _ = F.cube_split(1.0)
        prim = F.primitive(b, pos, None)
        path = self._write("noidx.glb", [[prim]], builder=b)
        # 24 vertices taken three at a time is 8 triangles, not the cube's 12 --
        # the point is only that it reads without indices at all.
        self.assertEqual(glb.read(path).mesh.n_tris, 8)

    def test_normalized_ushort_uvs_are_scaled_to_unit(self):
        # KHR_mesh_quantization writes UVs as unsigned shorts. Read raw they come
        # back in the tens of thousands.
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        raw = np.full((len(pos), 2), 65535, dtype=np.uint16)
        p_acc = b.accessor(np.asarray(pos, dtype=np.float32), F.FLOAT, "VEC3")
        uv_acc = b.accessor(raw, F.USHORT, "VEC2", normalized=True)
        prim = {"attributes": {"POSITION": p_acc, "TEXCOORD_0": uv_acc},
                "indices": b.accessor(tris.reshape(-1), F.USHORT, "SCALAR")}
        path = self._write("quant.glb", [[prim]], builder=b,
                           required=["KHR_mesh_quantization"])
        uv = glb.read(path).mesh.uv
        # 1.0 in, then V flipped for the bottom-left convention -> (1, 0).
        np.testing.assert_allclose(uv.reshape(-1, 2)[0], (1.0, 0.0), atol=1e-4)

    def test_an_out_of_range_index_is_named(self):
        b = F.Builder()
        pos, _ = F.cube_split(1.0)
        bad = np.array([[0, 1, 999]], dtype=np.int64)
        path = self._write("oob.glb", [[F.primitive(b, pos, bad)]], builder=b)
        with self.assertRaises(glb.GlbError) as ctx:
            glb.read(path)
        self.assertIn("999", str(ctx.exception))

    def test_a_sparse_accessor_overrides_its_base(self):
        b = F.Builder()
        pos = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=np.float32)
        tris = np.array([(0, 1, 2)], dtype=np.int64)
        p_acc = b.accessor(pos, F.FLOAT, "VEC3")
        idx_view = b.view(np.array([2], dtype=np.uint16).tobytes())
        val_view = b.view(np.array([(0, 7, 0)], dtype=np.float32).tobytes())
        b.accessors[p_acc]["sparse"] = {
            "count": 1,
            "indices": {"bufferView": idx_view, "componentType": F.USHORT},
            "values": {"bufferView": val_view},
        }
        prim = {"attributes": {"POSITION": p_acc},
                "indices": b.accessor(tris.reshape(-1), F.USHORT, "SCALAR")}
        path = self._write("sparse.glb", [[prim]], builder=b)
        v = glb.read(path).mesh.verts
        # glTF y=7 becomes Lathe z=7.
        self.assertAlmostEqual(float(v[:, 2].max()), 7.0, places=5)


class TestPrimitiveModes(GlbCase):
    def test_a_triangle_strip_alternates_winding(self):
        # A strip whose odd triangles are not reversed has half its faces
        # backwards, which reads as a mesh with holes in the shading.
        b = F.Builder()
        pos = np.array([(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)],
                       dtype=np.float32)
        strip = np.array([0, 1, 2, 3], dtype=np.int64)
        path = self._write("strip.glb",
                           [[F.primitive(b, pos, strip, mode=5)]], builder=b)
        imp = glb.read(path)
        self.assertEqual(imp.mesh.n_tris, 2)
        from lathe.perceive import mesh as pmesh
        n = pmesh.tri_normals(imp.mesh)
        np.testing.assert_allclose(n[0], n[1], atol=1e-6,
                                   err_msg="strip winding was not alternated")

    def test_a_triangle_fan(self):
        b = F.Builder()
        pos = np.array([(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
                       dtype=np.float32)
        fan = np.array([0, 1, 2, 3], dtype=np.int64)
        path = self._write("fan.glb", [[F.primitive(b, pos, fan, mode=6)]],
                           builder=b)
        self.assertEqual(glb.read(path).mesh.n_tris, 2)

    def test_points_are_skipped_rather_than_misread(self):
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        pts = np.array([(9, 9, 9), (8, 8, 8), (7, 7, 7)], dtype=np.float32)
        meshes = [[F.primitive(b, pos, tris),
                   F.primitive(b, pts, np.array([0, 1, 2]), mode=0)]]
        path = self._write("points.glb", meshes, builder=b)
        imp = glb.read(path)
        self.assertEqual(imp.mesh.n_tris, 12)
        self.assertEqual(imp.report["scene"]["primitives_skipped"], 1)
        self.assertTrue(any("POINTS" in s for s in imp.report["skipped"]))
        # A point cloud read as triangles would blow the bounding box out to 9 m.
        self.assertLess(float(np.abs(imp.mesh.verts).max()), 1.0)

    def test_a_primitive_without_position_is_skipped(self):
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        good = F.primitive(b, pos, tris)
        path = self._write("nopos.glb", [[good, {"attributes": {}}]], builder=b)
        imp = glb.read(path)
        self.assertEqual(imp.mesh.n_tris, 12)
        self.assertTrue(any("POSITION" in s for s in imp.report["skipped"]))


# --------------------------------------------------------------- parts and UV


class TestPartsAndUv(GlbCase):
    def test_material_names_become_parts(self):
        b = F.Builder()
        pos, tris = F.cube_split(1.0, centre=(0, 0, 0))
        pos2, tris2 = F.cube_split(1.0, centre=(3, 0, 0))
        meshes = [[F.primitive(b, pos, tris, material=0),
                   F.primitive(b, pos2, tris2, material=1)]]
        path = self._write("mats.glb", meshes, builder=b,
                           materials=[{"name": "M_oak"}, {"name": "M_iron"}])
        imp = glb.read(path)
        self.assertEqual(set(imp.report["parts"]), {"M_oak", "M_iron"})
        self.assertEqual(imp.report["parts"]["M_oak"], 12)
        self.assertEqual(set(imp.mesh.parts.values()), {"M_oak", "M_iron"})

    def test_a_primitive_with_no_material_falls_back_to_the_mesh_name(self):
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        path = self._write("nomat.glb", [[F.primitive(b, pos, tris)]], builder=b)
        self.assertEqual(list(glb.read(path).report["parts"]), ["m0"])

    def test_markup_in_a_material_name_is_stripped(self):
        """Names reach report text, tags and the HTML preview. A name is a name."""
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        path = self._write("xss.glb", [[F.primitive(b, pos, tris, material=0)]],
                           builder=b,
                           materials=[{"name": "</script><script>alert(1)</script>"}])
        name = list(glb.read(path).report["parts"])[0]
        for ch in "<>/":
            self.assertNotIn(ch, name)

    def test_an_absurdly_long_name_is_bounded(self):
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        path = self._write("longname.glb",
                           [[F.primitive(b, pos, tris, material=0)]], builder=b,
                           materials=[{"name": "A" * 5000}])
        self.assertLessEqual(len(list(glb.read(path).report["parts"])[0]), 64)

    def test_uv_v_is_flipped_to_the_bottom_left_convention(self):
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        uv = np.tile(np.array([[0.25, 0.0]], dtype=np.float32), (len(pos), 1))
        path = self._write("uvflip.glb", [[F.primitive(b, pos, tris, uv=uv)]],
                           builder=b)
        got = glb.read(path).mesh.uv.reshape(-1, 2)[0]
        np.testing.assert_allclose(got, (0.25, 1.0), atol=1e-6)

    def test_a_second_uv_set_is_kept_separately(self):
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        uv0 = np.tile(np.array([[0.1, 0.1]], dtype=np.float32), (len(pos), 1))
        uv1 = np.tile(np.array([[0.9, 0.9]], dtype=np.float32), (len(pos), 1))
        path = self._write("uv2.glb",
                           [[F.primitive(b, pos, tris, uv=uv0, uv1=uv1)]],
                           builder=b)
        imp = glb.read(path)
        self.assertEqual(imp.report["uv"]["sets"], ["UV0", "UV1"])
        np.testing.assert_allclose(imp.uv_sets["UV1"].reshape(-1, 2)[0],
                                   (0.9, 0.1), atol=1e-6)

    def test_uv_on_only_one_of_two_primitives_lands_on_the_right_triangles(self):
        """The bug this catches: accumulating a cursor per UV slot. A primitive
        with no TEXCOORD_0 adds nothing to that slot, so a running cursor hands
        its rows to the next primitive that does have one."""
        b = F.Builder()
        plain, plain_t = F.cube_split(1.0, centre=(0, 0, 0))
        mapped, mapped_t = F.cube_split(1.0, centre=(3, 0, 0))
        uv = np.tile(np.array([[0.5, 0.5]], dtype=np.float32), (len(mapped), 1))
        meshes = [[F.primitive(b, plain, plain_t),
                   F.primitive(b, mapped, mapped_t, uv=uv)]]
        path = self._write("uvgap.glb", meshes, builder=b)
        imp = glb.read(path)
        uvs = imp.mesh.uv
        self.assertEqual(len(uvs), 24)
        # First cube: no UV, so zeros. Second: the 0.5 it declared.
        self.assertTrue(np.allclose(uvs[:12], 0.0),
                        "UVs leaked onto the primitive that had none")
        self.assertTrue(np.allclose(uvs[12:], 0.5),
                        "the mapped primitive lost its UVs")
        self.assertEqual(imp.report["uv"]["tris_without_uv"], 12)


# ------------------------------------------------------------------ promotion


class TestPromotion(GlbCase):
    def test_an_open_surface_stays_a_mesh_and_says_why(self):
        b = F.Builder()
        pos, tris = F.open_quad()
        path = self._write("open.glb", [[F.primitive(b, pos, tris)]], builder=b)
        imp = glb.read(path)
        self.assertFalse(imp.is_solid)
        self.assertIsNotNone(imp.mesh)
        self.assertEqual(imp.mesh.n_tris, 2)
        self.assertIn("boundary edge", imp.reason)
        self.assertIn("encloses no volume", imp.reason)
        # The reason must point at what still works, or an agent gives up here.
        self.assertIn("Tier 1", imp.reason)

    def test_a_nonmanifold_edge_is_named_as_such(self):
        # Three triangles on one edge: a surface that branches.
        b = F.Builder()
        pos = np.array([(0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0), (0, -1, 0)],
                       dtype=np.float32)
        tris = np.array([(0, 1, 2), (0, 1, 3), (0, 1, 4)], dtype=np.int64)
        path = self._write("branch.glb", [[F.primitive(b, pos, tris)]], builder=b)
        imp = glb.read(path)
        self.assertFalse(imp.is_solid)
        self.assertIn("more than two triangles", imp.reason)

    def test_a_promoted_solid_carries_one_tag_not_one_per_material(self):
        # Manifold gives a rebuilt solid a single original id. Claiming the
        # material split survives a later boolean would be provenance drift.
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        pos2, tris2 = F.cube_split(1.0, centre=(3, 0, 0))
        meshes = [[F.primitive(b, pos, tris, material=0),
                   F.primitive(b, pos2, tris2, material=1)]]
        path = self._write("tags.glb", meshes, builder=b,
                           materials=[{"name": "M_a"}, {"name": "M_b"}])
        imp = glb.read(path, tag="barrel")
        self.assertTrue(imp.is_solid, imp.reason)
        self.assertEqual(set(imp.solid.tags.values()), {"barrel"})
        # while the Mesh keeps both
        self.assertEqual(set(imp.mesh.parts.values()), {"M_a", "M_b"})

    def test_a_promoted_solid_can_be_booleaned(self):
        imp = glb.read(self._cube("bool.glb"))
        cut = imp.solid - B.box(0.4, 0.4, 4.0, tag="cutter")
        self.assertLess(cut.volume(), imp.solid.volume())
        self.assertTrue(cut.status_ok())

    def test_import_glb_raises_for_a_non_solid(self):
        b = F.Builder()
        pos, tris = F.open_quad()
        path = self._write("open2.glb", [[F.primitive(b, pos, tris)]], builder=b)
        with self.assertRaises(glb.GlbError) as ctx:
            glb.import_glb(path)
        self.assertIn("read()", str(ctx.exception))

    def test_import_glb_returns_a_solid(self):
        s = glb.import_glb(self._cube("prog.glb"), tag="crate", name="SM_Crate_LOD0")
        self.assertEqual(s.name, "SM_Crate_LOD0")
        self.assertAlmostEqual(s.volume(), 1.0, places=5)


# --------------------------------------------------------- downstream wiring


class TestDownstream(GlbCase):
    def test_tier1_probes_read_an_import(self):
        from lathe.perceive import tier1
        imp = glb.read(self._cube("probe.glb"))
        self.assertEqual(tier1.counts(imp.mesh).fields["components"], 1)
        self.assertAlmostEqual(tier1.volume(imp.mesh).fields["volume_m3"], 1.0,
                               places=5)

    def test_tier1_reports_an_open_import_without_pretending(self):
        from lathe.perceive import tier1
        b = F.Builder()
        pos, tris = F.open_quad()
        path = self._write("openprobe.glb", [[F.primitive(b, pos, tris)]], builder=b)
        vol = tier1.volume(glb.read(path).mesh)
        self.assertFalse(vol.fields["closed"])
        self.assertIn("UNRELIABLE", vol.value)

    def test_the_verifier_runs_on_an_import(self):
        from lathe.verify import engine as vengine
        from lathe.verify import mesh as vmesh
        imp = glb.read(self._cube("verify.glb"))
        asset = vmesh.Asset(mesh=vmesh.from_mesh(imp.mesh),
                            spec=vmesh.EmitSpec(components=1))
        report = vengine.check(asset)
        self.assertEqual(len(report.errors), 0, str(report))

    def test_verifier_findings_name_the_material(self):
        from lathe.verify import mesh as vmesh
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        path = self._write("named.glb",
                           [[F.primitive(b, pos, tris, material=0)]], builder=b,
                           materials=[{"name": "M_brass"}])
        am = vmesh.from_mesh(glb.read(path).mesh)
        self.assertIn("M_brass", am.face_groups)
        self.assertIn("M_brass", am.entity(0))

    def test_the_verifier_catches_an_open_import(self):
        from lathe.verify import engine as vengine
        from lathe.verify import mesh as vmesh
        b = F.Builder()
        pos, tris = F.open_quad()
        path = self._write("openverify.glb", [[F.primitive(b, pos, tris)]],
                           builder=b)
        report = vengine.check(vmesh.Asset(mesh=vmesh.from_mesh(glb.read(path).mesh)))
        self.assertTrue(any(f.code == "E101" for f in report.errors), str(report))

    def test_uv_sets_reach_the_verifier(self):
        from lathe.verify import mesh as vmesh
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        uv = np.tile(np.array([[0.3, 0.4]], dtype=np.float32), (len(pos), 1))
        path = self._write("uvverify.glb", [[F.primitive(b, pos, tris, uv=uv)]],
                           builder=b)
        imp = glb.read(path)
        am = vmesh.from_mesh(imp.mesh, uv_sets=imp.uv_sets)
        self.assertIn("UV0", am.uv_sets)
        self.assertEqual(am.uv_sets["UV0"].shape, (am.num_loops, 2))

    def test_the_viewer_writes_from_a_mesh_that_is_not_a_solid(self):
        from lathe.viewer import write as write_view
        b = F.Builder()
        pos, tris = F.open_quad()
        path = self._write("openview.glb", [[F.primitive(b, pos, tris)]], builder=b)
        imp = glb.read(path)
        self.assertFalse(imp.is_solid)
        dest = os.path.join(self.tmp, "open.html")
        info = write_view(imp.mesh, dest)
        self.assertTrue(os.path.exists(dest))
        self.assertEqual(info["tris"], 2)
        html = open(dest, encoding="utf-8").read()
        self.assertIn("n/a (open surface)", html,
                      "an open mesh must not report a confident volume")

    def test_the_viewer_shows_imported_material_parts(self):
        from lathe.viewer import to_html
        b = F.Builder()
        pos, tris = F.cube_split(1.0)
        pos2, tris2 = F.cube_split(1.0, centre=(3, 0, 0))
        meshes = [[F.primitive(b, pos, tris, material=0),
                   F.primitive(b, pos2, tris2, material=1)]]
        path = self._write("viewparts.glb", meshes, builder=b,
                           materials=[{"name": "M_oak"}, {"name": "M_iron"}])
        html = to_html(glb.read(path).mesh)
        self.assertIn("M_oak", html)
        self.assertIn("M_iron", html)

    def test_the_viewer_payload_cannot_break_out_of_its_script_block(self):
        from lathe.viewer import to_html
        s = B.box(0.1, 0.1, 0.1, tag="a</script><script>alert(1)</script>")
        html = to_html(s)
        self.assertNotIn("</script><script>alert", html)
        self.assertIn("\\u003c", html)


# ----------------------------------------------------------------- round trip


@needs_blender
class TestRoundTrip(unittest.TestCase):
    """Export with our own exporter, read it back, and require the geometry to
    survive. This is the only test that exercises a real exporter's output --
    every fixture above is one this project wrote for itself."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="lathe_glb_rt_")
        from lathe.io import export
        cls.source = B.box(0.6, 0.4, 0.3, tag="body", name="SM_Slab_Oak_LOD0")
        cls.path = os.path.join(cls.tmp, "slab.glb")
        export(cls.source, cls.path, origin="bottom_center")
        cls.imp = glb.read(cls.path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_it_comes_back_as_a_solid(self):
        self.assertTrue(self.imp.is_solid, self.imp.reason)

    def test_the_triangle_count_survives(self):
        self.assertEqual(self.imp.mesh.n_tris, self.source.n_tris)

    def test_the_volume_survives_to_float32(self):
        # glTF stores POSITION as float32, so 0.3 m round trips as 0.30000001.
        # That caps the achievable agreement at about 1e-7 relative.
        self.assertAlmostEqual(self.imp.solid.volume(), self.source.volume(),
                               places=6)

    def test_the_bounding_box_survives_including_the_up_axis(self):
        lo, hi = self.imp.solid.bbox()
        np.testing.assert_allclose(lo, (-0.3, -0.2, 0.0), atol=1e-6)
        np.testing.assert_allclose(hi, (0.3, 0.2, 0.3), atol=1e-6)

    def test_a_real_exporter_splits_vertices_as_predicted(self):
        # 24 stored for 8 real. If this ever stops being true the welding step
        # stops being load-bearing, and it is worth knowing.
        self.assertEqual(self.imp.report["weld"]["verts_before"], 24)
        self.assertEqual(self.imp.report["weld"]["verts_after"], 8)

    def test_the_obj_group_became_a_material_part(self):
        self.assertTrue(any("body" in k for k in self.imp.report["parts"]),
                        self.imp.report["parts"])

    def test_a_bevelled_asset_round_trips(self):
        from lathe.io import export
        high = B.bevel(self.source, width=0.004, segments=2, min_dihedral=55)
        path = os.path.join(self.tmp, "bev.glb")
        export(high, path, origin="as_built")
        back = glb.read(path)
        self.assertTrue(back.is_solid, back.reason)
        self.assertEqual(back.mesh.n_tris, high.n_tris)
        self.assertAlmostEqual(back.solid.volume(), high.volume(), places=6)


HIERARCHY_SCRIPT = r'''
import bpy, sys
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=0.2, location=(0, 0, 0))
p = bpy.context.active_object
p.name = "Parent"; p.location = (0.0, 0.0, 1.0); p.scale = (2.0, 2.0, 2.0)
bpy.ops.mesh.primitive_cube_add(size=0.2, location=(0, 0, 0))
c = bpy.context.active_object
c.name = "Child"; c.parent = p; c.location = (3.0, 0.0, 0.0)
c.scale = (-1.0, 1.0, 1.0)
# export_apply=False keeps the object transforms as glTF *node* transforms, which
# is the path a hand-built fixture cannot prove.
bpy.ops.export_scene.gltf(filepath=sys.argv[-1], export_format='GLB',
                          export_apply=False, export_yup=True)
'''


@needs_blender
class TestRealHierarchy(unittest.TestCase):
    """A parented, scaled, mirrored hierarchy from a real exporter.

    The fixtures above prove the transform maths against the spec. This proves it
    against what an exporter actually writes -- Blender leaves object transforms
    as node transforms, so this is the only test where the composition, the
    parent-scale inheritance and the mirrored winding all arrive from outside.
    """

    @classmethod
    def setUpClass(cls):
        import subprocess
        cls.tmp = tempfile.mkdtemp(prefix="lathe_glb_hier_")
        script = os.path.join(cls.tmp, "mk.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(HIERARCHY_SCRIPT)
        cls.path = os.path.join(cls.tmp, "hier.glb")
        subprocess.run([blender_path(), "-b", "--factory-startup", "-noaudio",
                        "-P", script, "--", cls.path],
                       capture_output=True, text=True, timeout=600,
                       stdin=subprocess.DEVNULL)
        if not os.path.exists(cls.path):
            raise unittest.SkipTest("Blender did not write the hierarchy fixture")
        cls.imp = glb.read(cls.path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_both_instances_are_found(self):
        self.assertEqual(self.imp.report["scene"]["instances"], 2)
        self.assertEqual(self.imp.mesh.n_tris, 24)

    def test_the_parent_scale_reaches_the_child(self):
        # Child is at local x=3 under a parent scaled 2x, so world x=6. A reader
        # that took the child's local transform alone would put it at x=3.
        from lathe.perceive import mesh as pmesh
        lo, hi = pmesh.bounds(self.imp.mesh)
        np.testing.assert_allclose(lo, (-0.2, -0.2, 0.8), atol=1e-5)
        np.testing.assert_allclose(hi, (6.2, 0.2, 1.2), atol=1e-5)

    def test_the_mirrored_child_is_not_inside_out(self):
        """Each cube is 0.4 m on a side, so 0.064 m3 each and 0.128 together. A
        mirrored child whose winding was not flipped contributes *negative*
        volume, and the total would come out at 0."""
        from lathe.perceive import mesh as pmesh
        self.assertEqual(self.imp.report["transform"]["mirrored_instances"], 1)
        self.assertAlmostEqual(pmesh.signed_volume(self.imp.mesh), 0.128, places=5)

    def test_it_promotes_despite_the_hierarchy(self):
        self.assertTrue(self.imp.is_solid, self.imp.reason)
        self.assertEqual(self.imp.mesh.topology.n_components, 2)


if __name__ == "__main__":
    unittest.main(verbosity=1)
