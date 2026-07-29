"""
Every rule fires on a broken fixture with the right numbers, and no rule fires
on a clean one.

The second half matters as much as the first. DESIGN.md section 7 makes the
report the primary agent feedback channel; a channel with false positives
teaches the agent to ignore it, and an agent that ignores the report is worse
off than one that never had it, because it now also believes it was checked.
"""

from __future__ import annotations

import unittest

from .. import fixtures
from ..engine import RULES, get_profile, run
from ..findings import ERROR, NOTE, WARN
from ..topology import Topology


def report_for(name: str, profile=None):
    return run(getattr(fixtures, name)(), profile)


class TestFixtureTable(unittest.TestCase):
    """The table in fixtures.EXPECTED is the contract: each broken asset trips
    exactly the codes it is built to trip, and nothing else."""

    def test_expected_codes_exactly(self):
        for name, expected in fixtures.EXPECTED.items():
            with self.subTest(fixture=name):
                report = report_for(name)
                self.assertEqual(
                    sorted(set(report.codes)),
                    sorted(set(expected)),
                    "\n" + report.render(ascii_only=True),
                )

    def test_every_profile_rule_has_a_fixture(self):
        """A rule with no fixture is a rule nobody has run."""
        covered = set()
        for name in fixtures.EXPECTED:
            covered.update(report_for(name).codes)
        selected = set(get_profile("game_prop_pbr").rules)
        self.assertEqual(
            selected - covered, set(), "profile rules never exercised by a fixture"
        )

    def test_severity_grades(self):
        for code, want in (
            ("E101", ERROR), ("E102", ERROR), ("E103", ERROR), ("E104", ERROR),
            ("E105", ERROR), ("E106", ERROR), ("E107", ERROR), ("E108", ERROR),
            ("W201", WARN), ("W202", WARN), ("W203", WARN), ("W302", WARN),
            ("N401", NOTE),
        ):
            with self.subTest(code=code):
                self.assertEqual(RULES[code].severity, want)


class TestCleanAssets(unittest.TestCase):
    """Zero findings, and -- for the fully declared asset -- zero skips."""

    def test_clean_prop_has_no_findings(self):
        report = report_for("clean_prop")
        self.assertEqual(report.findings, (), report.render(ascii_only=True))
        self.assertTrue(report.passed)
        self.assertTrue(report.clean)

    def test_clean_prop_runs_every_rule(self):
        """A clean report is only meaningful if every rule actually ran."""
        report = report_for("clean_prop")
        self.assertEqual([s.code for s in report.skipped], [])

    def test_clean_prop_stays_clean_when_subdivided(self):
        for n in (1, 2, 5, 13):
            with self.subTest(subdivisions=n):
                asset = fixtures.clean_prop(subdivisions=n)
                # 13 subdivisions is over the declared budget on purpose
                # elsewhere; give this variant the budget it needs.
                if asset.mesh.triangle_count() > 1500:
                    asset = fixtures.Asset(
                        asset.mesh, fixtures.prop_spec(budget={"tris_lod0": 4000})
                    )
                report = run(asset)
                self.assertEqual(report.findings, (), report.render(ascii_only=True))

    def test_clean_solid_reports_what_it_could_not_check(self):
        report = report_for("clean_solid")
        self.assertEqual(report.findings, ())
        self.assertEqual(
            sorted(s.code for s in report.skipped),
            ["E106", "W202", "W203", "W206", "W302"]
        )
        for skip in report.skipped:
            self.assertTrue(skip.reason, "a skip without a reason is a silent pass")

    def test_fixed_crate_is_clean(self):
        """prototype/README.md: `size-rail*2 (fixed) components=1 OK`."""
        report = report_for("fixed_crate")
        self.assertEqual(report.findings, (), report.render(ascii_only=True))


class TestE101NonManifold(unittest.TestCase):
    def test_edge_with_four_faces(self):
        f = report_for("nonmanifold_edge").one("E101")
        self.assertEqual(f.data["nonmanifold_edge_count"], 1)
        self.assertEqual(f.data["worst_incident_faces"], 4)
        self.assertEqual(len(f.data["locations"]), 1)
        self.assertEqual(f.repair.call, "mesh.weld")
        self.assertIn("more than 2 faces", f.summary)

    def test_bowtie_vertex(self):
        f = report_for("bowtie_vertex").one("E101")
        self.assertEqual(f.data["bowtie_vert_count"], 1)
        self.assertEqual(f.repair.call, "mesh.split_vertex")
        self.assertNotIn("boundary", f.summary.lower())

    def test_open_shell(self):
        f = report_for("open_shell").one("E101")
        self.assertEqual(f.data["boundary_edge_count"], 4)
        self.assertEqual(f.data["open_components"], ["component_00"])
        self.assertEqual(f.repair.call, "mesh.fill_holes")

    def test_boundary_is_legal_when_the_profile_says_so(self):
        """mobile_vr allows open shells: foliage cards are not broken props."""
        report = report_for("open_shell", "mobile_vr")
        self.assertNotIn("E101", report.codes)


class TestE102Ngons(unittest.TestCase):
    def test_hexagonal_caps(self):
        f = report_for("ngon_prism").one("E102")
        self.assertEqual(f.data["ngon_count"], 2)
        self.assertEqual(f.data["size_histogram"], {"6": 2})
        self.assertEqual(f.data["max_face_verts"], 4)
        self.assertIn("faces(verts > 4)", f.repair.text())

    def test_quads_are_ngons_on_a_triangulated_target(self):
        """mobile_vr sets max_face_verts = 3, so the clean prop's quads fail --
        the same geometry, a different contract."""
        f = report_for("clean_prop", "mobile_vr").one("E102")
        self.assertEqual(f.data["ngon_count"], 6)
        self.assertEqual(f.data["max_face_verts"], 3)


class TestE103Normals(unittest.TestCase):
    def test_one_flipped_face_is_inconsistent(self):
        f = report_for("flipped_face").one("E103")
        self.assertEqual(f.data["inconsistent_edge_count"], 4)
        self.assertEqual(len(f.data["affected_faces"]), 5)
        self.assertEqual(f.repair.call, "mesh.recalculate_normals")

    def test_whole_shell_inverted_is_caught_by_volume(self):
        f = report_for("inside_out").one("E103")
        self.assertAlmostEqual(f.data["signed_volume_m3"], -0.216, places=6)
        self.assertEqual(f.data["component"], "component_00")
        self.assertEqual(f.repair.call, "mesh.flip_normals")

    def test_inverted_check_waits_for_consistency(self):
        """A shell with mixed winding has no meaningful inside, so the volume
        branch must stay quiet and let the winding branch report."""
        report = report_for("flipped_face")
        self.assertEqual(len(report.by_code("E103")), 1)
        self.assertIn("Inconsistent", report.one("E103").summary)


class TestE104Transform(unittest.TestCase):
    def test_scale_and_rotation(self):
        f = report_for("unapplied_transform").one("E104")
        self.assertEqual(f.data["scale"], [0.01, 0.01, 0.01])
        self.assertAlmostEqual(f.data["rotation_deg"][2], 45.0, places=6)
        self.assertAlmostEqual(f.data["bbox_local_m"][2], 0.6, places=9)
        self.assertAlmostEqual(f.data["bbox_world_m"][2], 0.006, places=9)
        self.assertEqual(f.repair.call, "transform.apply")
        self.assertEqual(f.repair.kwargs, {"scale": True, "rotation": True})

    def test_identity_transform_passes(self):
        self.assertNotIn("E104", report_for("clean_solid").codes)


class TestE105Loose(unittest.TestCase):
    def test_loose_vert_and_wire_edge(self):
        f = report_for("loose_geometry").one("E105")
        self.assertEqual(f.data["loose_vert_count"], 1)
        self.assertEqual(f.data["wire_edge_count"], 1)
        self.assertEqual(f.data["zero_area_face_count"], 0)
        self.assertEqual(f.repair.call, "mesh.delete_loose")

    def test_zero_area_face(self):
        f = report_for("zero_area_face").one("E105")
        self.assertEqual(f.data["zero_area_face_count"], 1)
        self.assertEqual(f.data["degenerate_faces"][0]["area_m2"], 0.0)
        self.assertEqual(f.data["degenerate_faces"][0]["reason"], "zero area")
        # delete_loose would be a no-op here, and a no-op repair is a lie
        self.assertEqual(f.repair.call, "mesh.dissolve_degenerate")


class TestE106OverlappingUVs(unittest.TestCase):
    def test_mirrored_islands_stacked(self):
        f = report_for("overlapping_uvs").one("E106")
        self.assertEqual(f.data["islands"], ["side_L", "side_R"])
        self.assertAlmostEqual(f.data["overlap_fraction"], 1.0, places=6)
        self.assertAlmostEqual(f.data["overlap_uv_area"], 0.09, places=6)
        self.assertTrue(f.data["mirrored"])
        self.assertEqual(f.data["uv_set"], "UV0")
        self.assertEqual(f.data["bake_targets"], ["normal"])
        self.assertEqual(
            f.repair.text(), "uv.unwrap(SM_Crate_Wood_LOD0, split_mirrored = true)"
        )
        self.assertIn("emit with bake_uv_set = UV1", f.repair.alternatives)

    def test_entity_names_the_island_not_the_object(self):
        f = report_for("overlapping_uvs").one("E106")
        self.assertEqual(f.entity, "SM_Crate_Wood_LOD0.side_L")

    def test_not_checked_without_a_declared_bake(self):
        report = report_for("clean_solid")
        reasons = {s.code: s.reason for s in report.skipped}
        self.assertIn("E106", reasons)
        self.assertIn("unique-uv bake", reasons["E106"])

    def test_tiling_uvs_may_overlap(self):
        """Overlap is only a defect against a declared unique bake."""
        asset = fixtures.overlapping_uvs()
        tiling = fixtures.Asset(
            asset.mesh, fixtures.prop_spec(bake=None, texel_density=None)
        )
        self.assertNotIn("E106", run(tiling).codes)


class TestE107Budget(unittest.TestCase):
    def test_over_budget(self):
        f = report_for("over_budget").one("E107")
        self.assertEqual(f.data["tri_count"], 2028)
        self.assertEqual(f.data["budget"], 1500)
        self.assertEqual(f.data["over_by"], 528)
        self.assertAlmostEqual(f.data["over_pct"], 35.2, places=6)
        self.assertEqual(len(f.data["by_group"]), 6)
        self.assertEqual({g["tris"] for g in f.data["by_group"]}, {338})
        self.assertEqual(f.repair.call, "lod.decimate")
        self.assertEqual(f.repair.kwargs["target_tris"], 1500)

    def test_counts_triangles_not_faces(self):
        """1,014 quads pass a face count and fail a triangle budget. The engine
        importer triangulates, so triangles are the honest unit."""
        asset = fixtures.over_budget()
        self.assertEqual(asset.mesh.num_faces, 1014)
        self.assertEqual(asset.mesh.triangle_count(), 2028)

    def test_not_checked_without_a_budget(self):
        asset = fixtures.over_budget()
        no_budget = fixtures.Asset(asset.mesh, fixtures.prop_spec(budget={}))
        report = run(no_budget)
        self.assertIn("E107", [s.code for s in report.skipped])
        self.assertNotIn("E107", report.codes)


class TestE108Components(unittest.TestCase):
    """The rule that exists because the doc's own example was severed."""

    def test_severed_crate_is_six_slabs(self):
        f = report_for("severed_crate").one("E108")
        self.assertEqual(f.data["components"], 6)
        self.assertEqual(f.data["declared"], 1)
        self.assertEqual(len(f.data["tri_counts"]), 6)
        self.assertEqual(f.repair.text(), "assert components(SM_Crate_Wood_LOD0) == 1")
        self.assertTrue(
            any("size - rail * 2" in a for a in f.repair.alternatives),
            "the repair should point at the fix the prototype actually landed",
        )

    def test_matches_the_prototype_measurement(self):
        """prototype/README.md measured 6 components severed, 1 when fixed."""
        severed = Topology(fixtures.severed_crate().mesh)
        fixed = Topology(fixtures.fixed_crate().mesh)
        self.assertEqual(len(severed.components), 6)
        self.assertEqual(len(fixed.components), 1)

    def test_debris_gets_a_different_repair(self):
        """One dominant shell plus specks is a different defect from a severed
        body, and the repair follows the evidence."""
        asset = fixtures.zero_area_face()  # cube + one stray triangle
        f = run(fixtures.Asset(asset.mesh, fixtures.solid_spec(components=1))).one("E108")
        self.assertEqual(f.data["components"], 2)
        self.assertGreater(f.data["largest_share"], 0.8)
        self.assertEqual(f.repair.call, "solid.keep_largest_component")

    def test_nags_when_nothing_is_declared(self):
        asset = fixtures.severed_crate()
        undeclared = fixtures.Asset(asset.mesh, fixtures.solid_spec(components=None))
        report = run(undeclared)
        reasons = {s.code: s.reason for s in report.skipped}
        self.assertIn("assert components", reasons["E108"])


class TestWarningsAndNotes(unittest.TestCase):
    def test_w201_origin_offset(self):
        f = report_for("origin_offset").one("W201")
        self.assertEqual(f.data["declared"], "bottom_center")
        self.assertAlmostEqual(f.data["distance_mm"], 300.0, places=6)
        self.assertAlmostEqual(f.data["offset_mm"][2], 300.0, places=6)
        self.assertEqual(f.repair.call, "transform.set_origin")

    def test_w201_not_checked_for_an_unknown_reference(self):
        asset = fixtures.clean_solid()
        odd = fixtures.Asset(asset.mesh, fixtures.solid_spec(origin="left_handle"))
        reasons = {s.code: s.reason for s in run(odd).skipped}
        self.assertIn("unknown origin reference", reasons["W201"])

    def test_w202_padding(self):
        f = report_for("tight_padding").one("W202")
        self.assertEqual(f.data["required_px"], 4.0)
        self.assertEqual(f.data["resolution"], 1024)
        self.assertLess(f.data["worst_gap_px"], 4.0)
        self.assertEqual(f.repair.call, "uv.pack")
        self.assertEqual(f.repair.kwargs["margin_px"], 4)

    def test_w202_defers_to_e106_on_overlap(self):
        """A zero gap is an overlap. Reporting it twice would have the agent
        repair one defect two ways."""
        report = report_for("overlapping_uvs")
        self.assertIn("E106", report.codes)
        self.assertNotIn("W202", report.codes)

    def test_w203_npot(self):
        f = report_for("npot_textures").one("W203")
        self.assertEqual(f.data["texture"], "T_Crate_Wood_D")
        self.assertEqual(f.data["width"], 1000)
        self.assertEqual(f.data["nearest_pot"], [1024, 1024])
        self.assertEqual(f.entity, "SM_Crate_Wood_LOD0/T_Crate_Wood_D")
        self.assertEqual(f.repair.call, "texture.resize")
        self.assertEqual(f.repair.args[1:], (1024, 1024))

    def test_w302_texel_density(self):
        f = report_for("texel_variance").one("W302")
        self.assertAlmostEqual(f.data["target_px_cm"], 5.12, places=6)
        self.assertEqual(len(f.data["islands"]), 1)
        self.assertEqual(f.data["islands"][0]["island"], "top")
        self.assertAlmostEqual(f.data["islands"][0]["px_cm"], 4.096, places=3)
        self.assertAlmostEqual(f.data["islands"][0]["off_target_pct"], -20.0, places=3)
        self.assertEqual(f.repair.text(), "uv.rescale(islands = [top], to = 5.12)")

    def test_w302_summary_carries_the_range(self):
        """Section 7 prints the measured range in the headline, not just a
        pass/fail verdict."""
        f = report_for("texel_variance").one("W302")
        self.assertIn("4.096", f.summary)
        self.assertIn("5.12", f.summary)
        self.assertIn("±10%", f.summary)

    def test_clean_prop_density_is_exactly_on_target(self):
        """The fixture is arithmetic, not luck: a 0.6 m face on a 0.3 uv island
        at 1024 is 5.12 px/cm exactly."""
        from ..uvmap import UVLayout

        layout = UVLayout(fixtures.clean_prop().mesh, "UV0")
        for island in layout.islands:
            self.assertAlmostEqual(island.texel_density(1024), 5.12, places=9)

    def test_n401_naming(self):
        f = report_for("bad_name").one("N401")
        self.assertEqual(f.data["actual"], "crate_final")
        self.assertEqual(f.data["suggested"], "SM_Crate_Final_LOD0")
        self.assertEqual(f.data["segments"], ["crate", "final"])
        self.assertEqual(f.repair.call, "emit.rename")
        self.assertEqual(f.repair.kwargs["to"], "SM_Crate_Final_LOD0")

    def test_n401_suggestions_are_always_valid(self):
        """The suggestion is the repair, so it has to satisfy the rule it came
        from -- including for names with nothing usable in them."""
        for name in ("crate_final", "Cube.001", "x", "SM_Chair_LOD3", "sm_chair_wood",
                     "SM_Chair_Wood_Old_LOD0"):
            with self.subTest(name=name):
                asset = fixtures.Asset(
                    fixtures.box_mesh(name, with_uv=False), fixtures.solid_spec()
                )
                report = run(asset)
                if not report.by_code("N401"):
                    continue
                suggested = report.by_code("N401")[0].data["suggested"]
                self.assertNotEqual(suggested, name, "a rename to the same name is a no-op")
                fixed = fixtures.Asset(
                    fixtures.box_mesh(suggested, with_uv=False), fixtures.solid_spec()
                )
                self.assertEqual(
                    run(fixed).by_code("N401"),
                    (),
                    "suggested %r still fails the rule it came from" % suggested,
                )

    def test_n401_checks_texture_names_too(self):
        from ..mesh import TextureDecl

        asset = fixtures.clean_prop()
        spec = fixtures.prop_spec(
            textures=(TextureDecl("crate_diffuse", 1024, 1024, "D"),)
        )
        findings = run(fixtures.Asset(asset.mesh, spec)).by_code("N401")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].data["kind"], "texture")
        self.assertEqual(findings[0].data["suggested"], "T_Crate_Diffuse_D")


class TestMultiFault(unittest.TestCase):
    """What an agent hand-off actually looks like: several defects at once."""

    def test_orders_by_severity_then_code(self):
        report = report_for("multi_fault")
        severities = [int(f.severity) for f in report.findings]
        self.assertEqual(severities, sorted(severities, reverse=True))
        errors = [f.code for f in report.findings if f.severity == ERROR]
        self.assertEqual(errors, sorted(errors))
        self.assertFalse(report.passed)

    def test_every_finding_still_names_an_entity_and_a_repair(self):
        report = report_for("multi_fault")
        self.assertGreater(len(report.findings), 3)
        for f in report.findings:
            with self.subTest(code=f.code):
                self.assertTrue(f.entity)
                self.assertTrue(f.evidence)
                self.assertTrue(f.repair.text())


if __name__ == "__main__":
    unittest.main()
