"""
Rules are data, not control flow.

The brief: "profiles select rule sets, and adding a profile (character_skinned,
mobile_vr) must not require touching the engine." The strongest test of that is
to define a profile here, in the test file, and run it -- if the engine needs to
know about a profile in advance, this file cannot pass.
"""

from __future__ import annotations

import unittest

from .. import fixtures
from ..engine import (
    PROFILES,
    RULES,
    Profile,
    RuleNotApplicable,
    get_profile,
    register_profile,
    rule,
    rule_table,
    run,
)
from ..findings import ERROR, NOTE, WARN


class TestRegistry(unittest.TestCase):
    def test_rules_are_registered_by_import(self):
        """No `load_rules()` call here, deliberately.

        This test used to call it, and every other test in the suite silently
        depended on that -- it passed under unittest only because class names sort
        `TestRegistry` after four classes that populate the registry as a side
        effect of calling `run()`. Under pytest, which keeps definition order, this
        class runs first and 241 tests failed in cascade. The package `__init__`
        loads the registry now, so this asserts the guarantee instead of creating
        it, and fails if that ever regresses.

        Asserted as an equality, not a series of `assertIn`: a missing code is the
        failure this catches, and W206 was absent from the old list.
        """
        self.assertEqual(
            set(RULES),
            {"E101", "E102", "E103", "E104", "E105", "E106", "E107", "E108",
             "W201", "W202", "W203", "W206", "W302", "N401"})

    def test_every_rule_has_a_title_and_a_docstring(self):
        for code, spec in RULES.items():
            with self.subTest(code=code):
                self.assertTrue(spec.title)
                self.assertTrue(spec.doc, "%s has no docstring to explain itself" % code)

    def test_approximate_rules_declare_it(self):
        """A rule that only estimates has to say so, or the report overclaims."""
        approximate = {c for c, s in RULES.items() if s.is_approximate}
        # W206 cannot tell a deliberately unmapped surface from an accidental
        # collapse, so it declares itself approximate too.
        self.assertEqual(approximate, {"W202", "W206", "W302"})
        for code in approximate:
            self.assertGreater(len(RULES[code].approximation), 40)

    def test_duplicate_code_is_rejected(self):
        with self.assertRaises(ValueError):

            @rule("E101", ERROR, "duplicate")
            def _dupe(ctx):  # pragma: no cover
                return ()

    def test_registered_profiles(self):
        """Every profile section 7 names, and nothing invented beyond them."""
        self.assertEqual(
            sorted(PROFILES),
            [
                "character_skinned",
                "film_usd",
                "game_prop_pbr",
                "hard_surface_subdiv",
                "mobile_vr",
            ],
        )

    def test_every_profile_describes_itself(self):
        """A profile is a policy. One that cannot say what it is for cannot be
        chosen on purpose."""
        for name, prof in PROFILES.items():
            with self.subTest(profile=name):
                self.assertGreater(len(prof.description), 40)

    def test_quad_profiles_grade_the_clean_prop_as_a_pass(self):
        """The clean fixture is a watertight quad mesh at the declared density.
        Every profile that accepts quads must pass it; a new profile that fails
        it is asserting something the fixture never promised.

        `mobile_vr` is excluded on purpose -- it demands triangles, so the
        quads are a genuine finding there. See test_rules.
        """
        for name in sorted(set(PROFILES) - {"mobile_vr"}):
            with self.subTest(profile=name):
                report = run(fixtures.clean_prop(), name)
                self.assertTrue(report.passed, report.render(ascii_only=True))

    def test_profiles_only_select_registered_rules(self):
        for name, prof in PROFILES.items():
            for code in prof.rules:
                with self.subTest(profile=name, code=code):
                    self.assertIn(code, RULES)

    def test_profile_params_target_rules_they_select(self):
        for name, prof in PROFILES.items():
            for code in list(prof.params) + list(prof.severity):
                with self.subTest(profile=name, code=code):
                    self.assertIn(code, prof.rules)

    def test_profile_params_are_known_rule_parameters(self):
        """A typo in a profile parameter would otherwise be silently ignored."""
        for name, prof in PROFILES.items():
            for code, params in prof.params.items():
                for key in params:
                    with self.subTest(profile=name, code=code, param=key):
                        self.assertIn(key, RULES[code].defaults)

    def test_unknown_profile_raises(self):
        with self.assertRaises(KeyError):
            get_profile("no_such_profile")

    def test_unknown_rule_in_profile_raises_at_run(self):
        bogus = Profile(name="bogus", rules=("E101", "E999"))
        with self.assertRaises(KeyError):
            run(fixtures.clean_solid(), bogus)


class TestProfileSelection(unittest.TestCase):
    def test_profile_comes_from_the_emit_spec_by_default(self):
        """Section 7: profiles are 'selected by emit'."""
        report = run(fixtures.clean_prop())
        self.assertEqual(report.profile, "game_prop_pbr")

    def test_explicit_profile_wins(self):
        report = run(fixtures.clean_prop(), "mobile_vr")
        self.assertEqual(report.profile, "mobile_vr")

    def test_a_profile_defined_outside_the_package_works(self):
        """The whole claim, in one test: a new profile is data."""
        lite = Profile(
            name="prop_lite_test",
            rules=("E107", "N401"),
            params={"E107": {"tolerance_pct": 50.0}},
            severity={"N401": WARN},
        )
        report = run(fixtures.multi_fault(), lite)
        self.assertEqual(report.profile, "prop_lite_test")
        self.assertEqual(sorted(set(report.codes)), ["E107", "N401"])
        self.assertEqual(report.one("N401").severity, WARN)
        # tolerance_pct = 50 is a parameter the base profile never sets
        self.assertEqual(report.one("E107").data["budget"], 500)

    def test_registering_a_profile_needs_no_engine_change(self):
        register_profile(Profile(name="scratch_test", rules=("N401",)))
        try:
            report = run(fixtures.bad_name(), "scratch_test")
            self.assertEqual(set(report.codes), {"N401"})
        finally:
            PROFILES.pop("scratch_test", None)

    def test_severity_override_changes_the_gate(self):
        """mobile_vr grades texel density as an error: the same measurement,
        a different consequence."""
        prop = run(fixtures.texel_variance())
        vr = run(fixtures.texel_variance(), "mobile_vr")
        self.assertEqual(prop.one("W302").severity, WARN)
        self.assertTrue(prop.passed)
        self.assertEqual(vr.one("W302").severity, ERROR)
        self.assertFalse(vr.passed)

    def test_param_override_changes_the_verdict(self):
        """Same geometry, two profiles, opposite answers -- because the contract
        differs, not because the code branches."""
        self.assertIn("E101", run(fixtures.open_shell(), "game_prop_pbr").codes)
        self.assertNotIn("E101", run(fixtures.open_shell(), "mobile_vr").codes)

    def test_call_site_overrides_beat_the_profile(self):
        report = run(
            fixtures.open_shell(), "game_prop_pbr",
            overrides={"E101": {"allow_boundary": True}},
        )
        self.assertNotIn("E101", report.codes)

    def test_every_profile_checking_uv_checks_all_of_it(self):
        """E106, W202 and W206 partition UV layout defects between them.

        rules/uv.py divides the space explicitly: E106 owns islands that
        intersect, W202 owns islands merely too close, W206 owns UV area that has
        collapsed. A profile that selects two thirds of a partition has a hole in
        it, and the hole is invisible -- a collapsed island simply goes unreported
        and the profile still says PASS.

        This is a real defect that reached main, not a hypothetical. W206 and the
        two newest profiles were written on separate branches; git merged both
        cleanly because they touch different lines, and the *combination* was
        incomplete in a way neither side could have caught. A structural
        invariant is the only thing that catches that class of gap, so it is
        asserted over every registered profile rather than per profile.
        """
        partition = {"E106", "W202", "W206"}
        for name in sorted(PROFILES):
            codes = {r.code for r in rule_table(name)}
            present = codes & partition
            if not present:
                continue  # a profile may legitimately ignore UVs entirely
            self.assertEqual(
                present, partition,
                "profile %r selects %s but not %s -- UV defects are partitioned "
                "between all three, so the missing code is a silent hole"
                % (name, sorted(present), sorted(partition - present)))

    def test_character_profile_tightens_origin_tolerance(self):
        asset = fixtures.clean_solid()
        nudged = fixtures.Asset(
            asset.mesh.evolve(verts=asset.mesh.verts + [0.0, 0.0, 0.0005]),
            fixtures.solid_spec(origin="bottom_center"),
        )
        self.assertNotIn("W201", run(nudged, "game_prop_pbr").codes)  # 0.5 mm < 1 mm
        self.assertIn("W201", run(nudged, "character_skinned").codes)  # > 0.1 mm


class TestSubdivProfile(unittest.TestCase):
    """`hard_surface_subdiv`: the emitted faces are a cage, not the surface."""

    def test_an_ngon_cage_is_still_an_error(self):
        """Catmull-Clark on a 6-gon cap is an extraordinary vertex and a visible
        pinch, so the quad limit is not relaxed the way film relaxes it."""
        report = run(fixtures.ngon_prism(), "hard_surface_subdiv")
        self.assertEqual(report.one("E102").severity, ERROR)
        self.assertFalse(report.passed)

    def test_a_triangulated_cage_is_an_error_too(self):
        """The other half of "a cage is quads", and the half that was missing.

        `max_face_verts: 4` bounds only the top, so a triangle satisfied it and
        this profile returned an affirmative PASS on a fully triangulated cage --
        not a skip, a pass, on the exact input its own comment calls a bug. A
        triangle is also the non-quad that actually turns up: it is what every
        triangle kernel emits, including this project's.
        """
        tri = fixtures.clean_prop()
        # Split each quad on a diagonal, keeping the same surface.
        faces = []
        for f in tri.mesh.faces:
            if len(f) == 4:
                faces += [(f[0], f[1], f[2]), (f[0], f[2], f[3])]
            else:
                faces.append(f)
        tri = fixtures.Asset(tri.mesh.evolve(faces=tuple(faces), uv_sets={}),
                             tri.spec)
        self.assertNotIn("E102", run(tri, "game_prop_pbr").codes)
        report = run(tri, "hard_surface_subdiv")
        self.assertEqual(report.one("E102").severity, ERROR)
        self.assertIn("under 4 verts", report.one("E102").summary)
        self.assertFalse(report.passed)

    def test_the_lower_bound_defaults_to_a_no_op(self):
        """min_face_verts is new, so every profile that does not ask for it must
        behave exactly as before. A triangulated mesh is the deliverable for four
        of the five profiles."""
        for name in sorted(set(PROFILES) - {"hard_surface_subdiv"}):
            with self.subTest(profile=name):
                self.assertEqual(
                    get_profile(name).params.get("E102", {}).get("min_face_verts", 3),
                    3)

    def test_an_unapplied_transform_is_graded_more_tightly_than_a_prop(self):
        """A subdivision modifier evaluates in object space, so the transform
        changes the resulting surface rather than only its interpretation."""
        asset = fixtures.clean_solid()
        skewed = fixtures.Asset(
            asset.mesh.evolve(
                transform=fixtures.Transform(scale=(1.0, 1.0, 1.0 + 1e-7))
            ),
            fixtures.solid_spec(),
        )
        self.assertNotIn("E104", run(skewed, "game_prop_pbr").codes)  # 1e-7 < 1e-6
        self.assertIn("E104", run(skewed, "hard_surface_subdiv").codes)  # > 1e-9

    def test_texel_density_is_reported_but_does_not_block(self):
        """Density is measured on the cage and the renderer shades the limit
        surface, so the number is real and the gate would be wrong."""
        report = run(fixtures.texel_variance(), "hard_surface_subdiv")
        self.assertEqual(report.one("W302").severity, NOTE)
        self.assertTrue(report.passed)


class TestFilmProfile(unittest.TestCase):
    """`film_usd`: an offline renderer, not an engine importer, reads this."""

    def test_ngons_are_reported_as_a_note_not_an_error(self):
        report = run(fixtures.ngon_prism(), "film_usd")
        self.assertEqual(report.one("E102").severity, NOTE)
        self.assertTrue(report.passed)

    def test_an_unapplied_transform_warns_because_usd_preserves_it(self):
        """The section 1 complaint is still worth naming, but a composed
        xformOp survives as authored, so it is not the silent FBX bug."""
        report = run(fixtures.unapplied_transform(), "film_usd")
        self.assertEqual(report.one("E104").severity, WARN)
        self.assertTrue(report.passed)

    def test_open_shells_are_legal_as_they_are_on_mobile(self):
        """Cards, cloth and hair meshes are open by design."""
        self.assertNotIn("E101", run(fixtures.open_shell(), "film_usd").codes)

    def test_npot_textures_are_an_observation(self):
        report = run(fixtures.npot_textures(), "film_usd")
        self.assertEqual(report.one("W203").severity, NOTE)

    def test_no_triangle_budget_is_selected_at_all(self):
        """Film does not budget triangles. E107 is absent rather than parked in
        the *not checked* list, where it would nag forever about a declaration
        this pipeline is never going to make."""
        report = run(fixtures.over_budget(), "film_usd")
        self.assertNotIn("E107", report.codes)
        self.assertNotIn("E107", [s.code for s in report.skipped])

    def test_udim_texture_names_are_not_held_to_the_mesh_convention(self):
        """`T_Crate_Wood_D` passes either way; the point is that the object name
        is still checked while the texture segments are not."""
        report = run(fixtures.bad_name(), "film_usd")
        self.assertEqual([f.entity for f in report.findings if f.code == "N401"],
                         ["crate_final"])


class TestContextAndCaching(unittest.TestCase):
    def test_topology_is_computed_once_per_asset(self):
        """Ten rules asking for the edge map ten times is the one performance
        mistake that would matter here."""
        asset = fixtures.multi_fault()
        seen = []

        original = RULES["E101"].check

        def spy(ctx):
            seen.append(id(ctx.topology))
            return original(ctx)

        RULES["E101"] = RULES["E101"].__class__(
            **{**RULES["E101"].__dict__, "check": spy}
        )
        try:
            report = run(asset)
            ids = {id_ for id_ in seen}
            self.assertEqual(len(ids), 1)
            self.assertTrue(report.findings)
        finally:
            RULES["E101"] = RULES["E101"].__class__(
                **{**RULES["E101"].__dict__, "check": original}
            )

    def test_not_applicable_becomes_a_skip_not_a_pass(self):
        report = run(fixtures.clean_solid())
        self.assertIn("E106", [s.code for s in report.skipped])
        self.assertNotIn("E106", report.codes)

    def test_rule_not_applicable_is_public(self):
        self.assertTrue(issubclass(RuleNotApplicable, Exception))

    def test_uv_lookup_prefers_the_declared_bake_set(self):
        from ..engine import RuleContext

        asset = fixtures.clean_prop()
        ctx = RuleContext(asset, RULES["W302"], {}, WARN)
        self.assertEqual(ctx.uv().set_name, "UV0")
        with self.assertRaises(RuleNotApplicable):
            ctx.uv("UV3")


class TestDeterminism(unittest.TestCase):
    """Section 8 stores a validation report per commit. Reports that differ
    between identical runs cannot be diffed across a history DAG."""

    def test_same_asset_same_text(self):
        for name in fixtures.EXPECTED:
            with self.subTest(fixture=name):
                a = run(getattr(fixtures, name)()).render(ascii_only=True)
                b = run(getattr(fixtures, name)()).render(ascii_only=True)
                self.assertEqual(a, b)

    def test_same_asset_same_json(self):
        a = run(fixtures.multi_fault()).to_json()
        b = run(fixtures.multi_fault()).to_json()
        self.assertEqual(a, b)

    def test_findings_are_sorted_by_severity_then_code(self):
        report = run(fixtures.multi_fault())
        keys = [(-int(f.severity), f.code, f.entity, f.summary) for f in report.findings]
        self.assertEqual(keys, sorted(keys))


class TestPublicSurface(unittest.TestCase):
    def test_check_is_an_alias_for_run(self):
        from .. import check

        self.assertEqual(
            check(fixtures.bad_name()).codes, run(fixtures.bad_name()).codes
        )

    def test_severities_are_ordered(self):
        self.assertGreater(ERROR, WARN)
        self.assertGreater(WARN, NOTE)
        self.assertEqual(ERROR.label, "error")


if __name__ == "__main__":
    unittest.main()
