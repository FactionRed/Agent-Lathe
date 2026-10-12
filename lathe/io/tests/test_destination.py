"""
Where an export or preview may be written. Bug hunt lows.

`NUL.obj` reported success (the null device), `a:b.obj` wrote an NTFS alternate
stream and then failed on the .mtl, a relative path wrote into whatever directory
the client had started the server from, and a path the filesystem refused came
back as "UNEXPECTED PermissionError". Nothing here needs Blender.
"""

import asyncio
import os
import tempfile
import unittest

from lathe.build.solid import LatheError
from lathe.io.export import safe_destination


class TestSafeDestination(unittest.TestCase):
    def setUp(self):
        self.base = tempfile.mkdtemp(prefix="lathe_dest_")

    def test_a_relative_path_lands_under_the_base_not_the_cwd(self):
        got = safe_destination("sub/x.obj", self.base)
        self.assertEqual(got, os.path.abspath(os.path.join(self.base, "sub", "x.obj")))

    def test_an_absolute_path_is_kept(self):
        p = os.path.join(self.base, "ok.stl")
        self.assertEqual(safe_destination(p, "elsewhere"), os.path.abspath(p))

    def test_reserved_device_names_are_refused_with_or_without_extension(self):
        for name in ("NUL.obj", "nul.stl", "CON.obj", "aux.glb", "COM1.obj",
                     "lpt9.stl", "PRN"):
            with self.subTest(name=name):
                with self.assertRaises(LatheError) as ctx:
                    safe_destination(name, self.base)
                self.assertIn("reserved device name", str(ctx.exception))

    def test_a_reserved_name_in_a_directory_position_is_refused_too(self):
        with self.assertRaises(LatheError):
            safe_destination(os.path.join("CON", "x.obj"), self.base)

    def test_characters_a_file_name_cannot_hold_are_refused(self):
        for name in ("a:b.obj", "q?.obj", "x*y.stl", 'a"b.obj', "a|b.obj",
                     "a<b.obj", "tab\there.obj"):
            with self.subTest(name=name):
                with self.assertRaises(LatheError):
                    safe_destination(name, self.base)

    def test_a_trailing_space_or_dot_is_refused(self):
        for name in ("x.obj ", "x.obj.", "dir./x.obj"):
            with self.subTest(name=name):
                with self.assertRaises(LatheError):
                    safe_destination(name, self.base)

    def test_empty_and_nul_byte_paths_are_refused(self):
        for bad in ("", "   ", "a\x00b.obj"):
            with self.subTest(path=repr(bad)):
                with self.assertRaises(LatheError):
                    safe_destination(bad, self.base)

    def test_an_overlong_component_is_refused(self):
        with self.assertRaises(LatheError):
            safe_destination("l" * 250 + ".obj", self.base)

    def test_a_drive_colon_is_not_mistaken_for_a_bad_character(self):
        if os.name != "nt":
            self.skipTest("drive letters are Windows")
        p = os.path.join(os.path.abspath(self.base), "ok.obj")
        self.assertEqual(safe_destination(p), os.path.abspath(p))


class TestServerSurfaces(unittest.TestCase):
    def setUp(self):
        from lathe.mcp import server
        from lathe.mcp.session import Workspace
        server.WS = Workspace()
        server.WS.apply("result = box(0.1, 0.1, 0.1, tag='b', name='SM_D_LOD0')",
                        "box")
        self.server = server
        self.out = tempfile.mkdtemp(prefix="lathe_out_")
        self._old = server.OUT_DIR
        server.OUT_DIR = self.out

    def tearDown(self):
        self.server.OUT_DIR = self._old

    def _export(self, path):
        S = self.server
        return asyncio.run(S.lathe_export(S.ExportInput(path=path)))

    def test_a_reserved_name_is_an_error_not_a_silent_success(self):
        out = self._export("NUL.obj")
        self.assertTrue(out.startswith("ERROR"), out)
        self.assertIn("reserved device name", out)

    def test_a_relative_export_goes_to_the_output_directory(self):
        out = self._export("rel_box.obj")
        self.assertIn(os.path.basename(self.out), out)
        self.assertTrue(os.path.exists(os.path.join(self.out, "rel_box.obj")))
        self.assertFalse(os.path.exists(os.path.join(os.getcwd(), "rel_box.obj")))

    def test_an_unwritable_path_is_an_error_not_unexpected(self):
        blocker = os.path.join(self.out, "a_file")
        open(blocker, "w").close()
        out = self._export(os.path.join(blocker, "inside.obj"))   # a file as a dir
        self.assertTrue(out.startswith("ERROR"), out)
        self.assertNotIn("UNEXPECTED", out)


if __name__ == "__main__":
    unittest.main(verbosity=1)
