"""
The report format IS the product, so it gets its own tests.

DESIGN.md section 7 shows the shape:

    ✗ E106  Overlapping UVs on unique bake
            object   : SM_Crate_Wood
            islands  : side_L overlaps side_R (91% area)
            evidence : bake target = normal, uv_set = UV0
            repair   : uv.unwrap(final, split_mirrored = true)
                       — or emit with bake_uv_set = UV1

These tests assert that shape structurally, and then assert it for *every*
finding the fixture suite can produce -- a format that holds for the sample and
drifts everywhere else is not a format.
"""

from __future__ import annotations

import json
import re
import unittest

from .. import fixtures
from ..engine import run
from ..findings import ERROR, Finding, Raw, Repair, Report, Severity, fold_ascii

KEY_LINE = re.compile(r"^ {8}(?P<key>[a-z][a-z_ ]{0,7}) : (?P<value>.+)$")
CONT_LINE = re.compile(r"^ {19}\S.*$")
HEAD_LINE = re.compile(r"^[✗⚠•] (?P<code>[EWN]\d{3})  (?P<summary>\S.*)$")


def all_reports():
    for name in fixtures.EXPECTED:
        yield name, run(getattr(fixtures, name)())
    yield "multi_fault", run(fixtures.multi_fault())


class TestSectionSevenShape(unittest.TestCase):
    def test_e106_matches_the_documented_sample(self):
        """Line for line against the section 7 sample, modulo the actual
        numbers: glyph and code, then object / islands / evidence / repair, then
        an indented alternative."""
        report = run(fixtures.overlapping_uvs())
        block = report.one("E106").render().split("\n")
        self.assertEqual(block[0], "✗ E106  Overlapping UVs on unique bake")
        self.assertEqual(block[1], "        object   : SM_Crate_Wood_LOD0.side_L")
        self.assertTrue(block[2].startswith("        islands  : side_L overlaps side_R ("))
        self.assertTrue(block[2].endswith("% area)"))
        # The sample's evidence line is the bake configuration. This
        # implementation appends the measured overlap to it as well, so the
        # text channel carries the same numbers the JSON channel does.
        evidence = [line for line in block if line.startswith("        evidence :")]
        self.assertEqual(len(evidence), 1)
        self.assertTrue(
            evidence[0].startswith("        evidence : bake target = normal, uv_set = UV0"),
            evidence[0],
        )
        repair = [i for i, line in enumerate(block) if line.startswith("        repair   :")]
        self.assertEqual(len(repair), 1)
        self.assertEqual(
            block[repair[0]],
            "        repair   : uv.unwrap(SM_Crate_Wood_LOD0, split_mirrored = true)",
        )
        self.assertEqual(block[repair[0] + 1], " " * 19 + "— or emit with bake_uv_set = UV1")

    def test_w302_matches_the_documented_sample(self):
        """The section 7 W302 sample carries its numbers in the headline and its
        offenders in an `islands` line."""
        f = run(fixtures.texel_variance()).one("W302")
        block = f.render().split("\n")
        self.assertRegex(
            block[0],
            r"^⚠ W302  Texel density variance [\d.]+–[\d.]+ px/cm "
            r"\(target [\d.]+ ±\d+%\)$",
        )
        self.assertIn("        islands  : top", block)
        self.assertIn("        repair   : uv.rescale(islands = [top], to = 5.12)", block)

    def test_every_finding_has_the_four_required_parts(self):
        """Named code, named entity, numeric evidence, executable repair."""
        seen = 0
        for name, report in all_reports():
            for f in report.findings:
                seen += 1
                with self.subTest(fixture=name, code=f.code):
                    self.assertRegex(f.code, r"^[EWN]\d{3}$")
                    self.assertTrue(f.entity.startswith(report.asset))
                    self.assertRegex(f.evidence, r"\d", "evidence must carry a number")
                    self.assertTrue(f.repair.text())
                    self.assertTrue(f.data, "no machine-readable evidence")
        self.assertGreater(seen, 15)

    def test_every_block_keeps_the_column_layout(self):
        for name, report in all_reports():
            for f in report.findings:
                lines = f.render().split("\n")
                with self.subTest(fixture=name, code=f.code):
                    self.assertRegex(lines[0], HEAD_LINE.pattern)
                    keys = []
                    for line in lines[1:]:
                        m = KEY_LINE.match(line)
                        if m:
                            keys.append(m.group("key").strip())
                        else:
                            self.assertRegex(
                                line, CONT_LINE.pattern, "stray line: %r" % line
                            )
                    self.assertEqual(keys[0], "object")
                    self.assertEqual(keys[-2:], ["evidence", "repair"])

    def test_repairs_look_executable(self):
        """A repair is a call or an assertion, never prose. Prose is what the
        design doc rejects with 'they never fix boolean failed'."""
        pattern = re.compile(r"^(assert [a-z]|[a-z][a-z_]*(\.[a-z][a-z_]*)?\()")
        for name, report in all_reports():
            for f in report.findings:
                with self.subTest(fixture=name, code=f.code):
                    self.assertRegex(f.repair.text(), pattern)
                    self.assertNotIn("TODO", f.repair.text())
                    self.assertNotIn("...", f.repair.text())

    def test_bounded_output(self):
        """A finding lists at most a handful of offenders; an agent handed 400
        edge locations fixes none of them."""
        report = run(fixtures.multi_fault())
        for f in report.findings:
            for line in f.render().split("\n"):
                self.assertLessEqual(len(line), 200, line)
            self.assertLessEqual(len(f.render().split("\n")), 24)


class TestRenderModes(unittest.TestCase):
    def test_ascii_mode_is_pure_ascii(self):
        """Windows consoles are cp1252. Losing a build gate to
        UnicodeEncodeError would be a silly way to fail."""
        for name, report in all_reports():
            text = report.render(ascii_only=True)
            with self.subTest(fixture=name):
                self.assertTrue(text.isascii(), text)
                text.encode("cp1252")  # raises if it would break a PowerShell pipe

    def test_ascii_fold_keeps_the_meaning(self):
        self.assertEqual(fold_ascii("scale ≠ (1,1,1)"), "scale != (1,1,1)")
        self.assertEqual(fold_ascii("0.36 m²"), "0.36 m^2")
        self.assertEqual(fold_ascii("§7"), "sec 7")
        self.assertEqual(fold_ascii("5.12 ±10%"), "5.12 +/-10%")

    def test_header_carries_the_gate_and_the_counts(self):
        report = run(fixtures.multi_fault())
        head = report.render().split("\n")[0]
        self.assertIn("profile game_prop_pbr", head)
        self.assertIn("FAIL", head)
        self.assertRegex(head, r"\d+ error, \d+ warn, \d+ note")

    def test_clean_report_says_so(self):
        report = run(fixtures.clean_prop())
        self.assertIn("PASS", report.render())
        self.assertIn("no findings", report.render())

    def test_not_checked_section(self):
        text = run(fixtures.clean_solid()).render()
        self.assertIn("not checked", text)
        self.assertIn("E106", text)
        text_off = run(fixtures.clean_solid()).render(show_skipped=False)
        self.assertNotIn("not checked", text_off)


class TestJson(unittest.TestCase):
    def test_round_trip(self):
        for name, report in all_reports():
            with self.subTest(fixture=name):
                data = json.loads(report.to_json())
                self.assertEqual(data["asset"], report.asset)
                self.assertEqual(data["passed"], report.passed)
                self.assertEqual(len(data["findings"]), len(report.findings))
                for f in data["findings"]:
                    for key in ("code", "severity", "entity", "evidence", "repair", "data"):
                        self.assertIn(key, f)
                    self.assertTrue(f["repair"]["text"])
                    self.assertIn("call", f["repair"])

    def test_numbers_survive_as_numbers(self):
        """The JSON form exists so tooling can compare values, not parse prose."""
        data = json.loads(run(fixtures.over_budget()).to_json())
        e107 = [f for f in data["findings"] if f["code"] == "E107"][0]
        self.assertIsInstance(e107["data"]["tri_count"], int)
        self.assertIsInstance(e107["data"]["over_pct"], float)
        self.assertEqual(e107["repair"]["kwargs"]["target_tris"], 1500)

    def test_not_checked_is_reported_in_json_too(self):
        data = json.loads(run(fixtures.clean_solid()).to_json())
        codes = [s["code"] for s in data["not_checked"]]
        self.assertEqual(sorted(codes), ["E106", "W202", "W203", "W302"])


class TestFindingContract(unittest.TestCase):
    """The four required fields are enforced by the constructor, not by review."""

    def _kw(self, **over):
        kw = dict(
            code="E999",
            severity=ERROR,
            summary="s",
            entity="SM_X",
            evidence="1 thing",
            repair=Repair("mesh.weld", (Raw("SM_X"),)),
        )
        kw.update(over)
        return kw

    def test_valid(self):
        self.assertTrue(Finding(**self._kw()).render())

    def test_missing_entity_rejected(self):
        with self.assertRaises(ValueError):
            Finding(**self._kw(entity=""))

    def test_missing_evidence_rejected(self):
        with self.assertRaises(ValueError):
            Finding(**self._kw(evidence=""))

    def test_overlong_detail_key_rejected(self):
        """The fixed-width key column is a contract, so it is enforced where the
        finding is built rather than discovered in a misaligned report."""
        with self.assertRaises(ValueError):
            Finding(**self._kw(where=(("suggested", "x"),)))
        self.assertTrue(Finding(**self._kw(where=(("suggest", "x"),))).render())

    def test_repair_rendering(self):
        self.assertEqual(
            Repair("uv.unwrap", (Raw("final"),), {"split_mirrored": True}).text(),
            "uv.unwrap(final, split_mirrored = true)",
        )
        self.assertEqual(
            Repair("uv.rescale", (), {"islands": [Raw("a"), Raw("b")], "to": 5.12}).text(),
            "uv.rescale(islands = [a, b], to = 5.12)",
        )
        self.assertEqual(
            Repair("assert", (Raw("components(x) == 1"),), statement=True).text(),
            "assert components(x) == 1",
        )
        self.assertEqual(
            Repair("texture.resize", (Raw("T_X_D"), 1024, 1024)).text(),
            "texture.resize(T_X_D, 1024, 1024)",
        )

    def test_gate_semantics(self):
        err = Finding(**self._kw(severity=Severity.ERROR))
        warn = Finding(**self._kw(code="W999", severity=Severity.WARN))
        self.assertFalse(Report("a", "p", (err,)).passed)
        self.assertTrue(Report("a", "p", (warn,)).passed)
        self.assertFalse(Report("a", "p", (warn,)).clean)
        self.assertTrue(Report("a", "p", ()).clean)


if __name__ == "__main__":
    unittest.main()
