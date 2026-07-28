"""
Tier 1 probes against closed-form truth.

Every assertion here is a number someone can derive on paper: a unit cube
encloses exactly 1.0, a sphere of radius r encloses 4/3 pi r^3, a cube milled
with one pocket has symmetry (V-2v)/V about that pocket's mirror plane. Where a
probe can only approximate, the test pins the size of the error and the
convergence order rather than accepting whatever came out.
"""

import math

import numpy as np
import pytest

from lathe.perceive import kernel
from lathe.perceive import mesh as M
from lathe.perceive import tier1
from lathe.perceive.tests import fixtures as fx
from lathe.perceive.units import Length, Point

EXACT = 1e-15  # float64 round-off on a handful of operations


# --------------------------------------------------------------------- bbox


def test_bbox_of_a_unit_cube():
    p = tier1.bbox(fx.unit_cube())
    assert p.fields["size_m"] == (1.0, 1.0, 1.0)
    assert p.fields["min_m"] == (0.0, 0.0, 0.0)
    assert p.fields["center_m"] == (0.5, 0.5, 0.5)
    assert p.fields["diagonal_m"] == pytest.approx(math.sqrt(3.0), abs=EXACT)
    assert p.value == "1.000 × 1.000 × 1.000 m"


def test_bbox_matches_the_design_example_shape():
    """DESIGN.md 6: measure(final.bbox) -> 0.600 x 0.600 x 0.600 m"""
    assert tier1.bbox(fx.centered_cube(0.6)).value == "0.600 × 0.600 × 0.600 m"


def test_bbox_of_a_named_part():
    p = tier1.bbox(fx.stool(), part="seat")
    assert p.probe == "measure(seat.bbox)"
    assert p.fields["size_m"] == pytest.approx((0.4, 0.4, 0.04), abs=1e-15)


def test_bbox_carries_a_unit_in_every_rendering():
    for unit in ("m", "mm", "cm", "in"):
        assert tier1.bbox(fx.unit_cube(), unit=unit).value.endswith(unit)


# ------------------------------------------------------------------- volume


def test_unit_cube_volume_is_exactly_one():
    p = tier1.volume(fx.unit_cube())
    assert p.fields["volume_m3"] == 1.0  # not approx: exactly
    assert p.value == "1.000 m³"
    assert p.ok


def test_cube_volume_scales_as_the_cube_of_the_edge():
    for size in (0.001, 0.6, 2.5, 100.0):
        got = tier1.volume(fx.centered_cube(size)).fields["volume_m3"]
        assert got == pytest.approx(size**3, rel=1e-14)


def test_tetrahedron_volume_is_eight_thirds():
    got = tier1.volume(fx.tetrahedron()).fields["volume_m3"]
    assert got == pytest.approx(8.0 / 3.0, rel=1e-15)


def test_sphere_volume_converges_second_order_from_below():
    """An inscribed polyhedron always under-reports, and halving the edge length
    should quarter the error. That is the check that the maths is right rather
    than merely close."""
    r = 1.0
    exact = fx.sphere_volume(r)
    errors = []
    for segments in (32, 64, 128, 256):
        got = tier1.volume(fx.sphere(r, segments)).fields["volume_m3"]
        assert got < exact  # inscribed
        errors.append(exact - got)
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    for ratio in ratios:
        assert 3.8 < ratio < 4.2, "expected 2nd order convergence, got %r" % ratios
    assert errors[-1] / exact < 4e-4  # 0.04% at 256 segments


def test_cylinder_volume_matches_pi_r_squared_h():
    got = tier1.volume(fx.cylinder(0.5, 2.0, 512)).fields["volume_m3"]
    assert got == pytest.approx(fx.cylinder_volume(0.5, 2.0), rel=3e-5)


def test_torus_volume_matches_the_closed_form():
    got = tier1.volume(fx.torus(1.0, 0.3, 128)).fields["volume_m3"]
    assert got == pytest.approx(fx.torus_volume(1.0, 0.3), rel=1e-3)


def test_crate_volume_matches_the_design_example_arithmetic():
    got = tier1.volume(fx.crate()).fields["volume_m3"]
    assert got == pytest.approx(fx.crate_analytic_volume(), rel=1e-14)


def test_volume_on_an_open_mesh_is_reported_as_unreliable():
    p = tier1.volume(fx.open_box())
    assert not p.ok
    assert "UNRELIABLE" in p.value
    assert "undefined on an open surface" in p.problems[0]
    assert p.fields["closed"] is False


def test_inverted_normals_are_caught_by_the_sign():
    p = tier1.volume(fx.inverted_cube())
    assert p.fields["signed_volume_m3"] == -1.0
    assert p.fields["inverted"] is True
    assert any("E103" in x for x in p.problems)


# -------------------------------------------------------------- area / com


def test_cube_surface_area_is_exactly_six():
    assert tier1.surface_area(fx.unit_cube()).fields["area_m2"] == 6.0


def test_sphere_area_converges_from_below():
    exact = fx.sphere_area(1.0)
    coarse = tier1.surface_area(fx.sphere(1.0, 64)).fields["area_m2"]
    fine = tier1.surface_area(fx.sphere(1.0, 256)).fields["area_m2"]
    assert coarse < fine < exact
    assert (exact - fine) / exact < 6e-4


def test_torus_area_matches_the_closed_form():
    got = tier1.surface_area(fx.torus(1.0, 0.3, 128)).fields["area_m2"]
    assert got == pytest.approx(fx.torus_area(1.0, 0.3), rel=1e-3)


def test_center_of_mass_of_a_unit_cube():
    p = tier1.center_of_mass(fx.unit_cube())
    assert p.fields["center_m"] == pytest.approx((0.5, 0.5, 0.5), abs=EXACT)
    assert p.fields["method"].startswith("volume_centroid")


def test_center_of_mass_of_a_cylinder_is_its_axis_midpoint():
    got = tier1.center_of_mass(fx.cylinder(0.5, 2.0, 256)).fields["center_m"]
    assert got == pytest.approx((0.0, 0.0, 0.0), abs=1e-15)


def test_center_of_mass_is_volume_weighted_not_vertex_averaged():
    """A stool's mass sits in the seat, well above the mid-height of its bounds.
    A vertex average would land somewhere else entirely."""
    stool = fx.stool()
    com = tier1.center_of_mass(stool).fields["center_m"]
    lo, hi = M.bounds(stool)
    assert com[2] > (lo[2] + hi[2]) / 2.0
    vertex_mean = stool.verts[np.unique(stool.tris)].mean(axis=0)
    assert abs(com[2] - vertex_mean[2]) > 0.05


def test_center_of_mass_of_a_two_box_assembly_is_the_lever_rule():
    """Boxes of volume 1 at x=0 and volume 8 at x=3 -> com at 8*3/9 = 2.667."""
    a = fx.centered_cube(1.0)
    import manifold3d as m3

    big = m3.Manifold.cube((2.0, 2.0, 2.0), True).translate((3.0, 0.0, 0.0))
    solid = a.source + big
    mesh = M.from_manifold(solid, name="lever")
    com = tier1.center_of_mass(mesh).fields["center_m"]
    assert com[0] == pytest.approx(8.0 * 3.0 / 9.0, rel=1e-14)


def test_center_of_mass_on_an_open_mesh_says_which_centre_it_is():
    p = tier1.center_of_mass(fx.open_box())
    assert p.fields["method"].startswith("area_centroid")
    assert not p.ok


# ------------------------------------------------------------------- counts


def test_cube_counts_are_six_quads():
    f = tier1.counts(fx.unit_cube()).fields
    assert (f["tris"], f["verts"], f["edges"]) == (12, 8, 18)
    assert f["faces_logical"] == 6
    assert f["face_degrees"] == {4: 6}
    assert f["components"] == 1 and f["genus"] == 0


def test_crate_topology_matches_the_prototype_finding():
    """prototype/README.md: the fixed crate is one component with genus 5."""
    f = tier1.counts(fx.crate()).fields
    assert f["components"] == 1
    assert f["genus"] == 5
    # 6 body faces + 5 slots x 4 walls, and the two faces the slots pierce are
    # quads with five rectangular holes -> 24 outline edges each
    assert f["faces_logical"] == 26
    assert f["face_degrees"] == {4: 24, 24: 2}


def test_torus_genus_is_one():
    assert tier1.counts(fx.torus()).fields["genus"] == 1


def test_genus_is_refused_on_an_open_surface():
    f = tier1.counts(fx.open_box()).fields
    assert f["genus"] is None
    assert "n/a" in tier1.counts(fx.open_box()).text()


def test_counts_flag_non_manifold_edges():
    f = tier1.counts(fx.nonmanifold_fin()).fields
    assert f["nonmanifold_edges"] == 1
    assert f["boundary_edges"] == 6
    assert any("E101" in p for p in tier1.counts(fx.nonmanifold_fin()).problems)


# ----------------------------------------------------------------- distance


def test_stool_leg_spacing_matches_the_design_example():
    """DESIGN.md 6: measure(distance(leg_01, leg_02)) -> 0.320 m, from a 400 mm
    seat with legs inset 40 mm. The surface gap is that minus two leg radii."""
    stool = fx.stool()
    p = tier1.distance(stool.part("leg_01"), stool.part("leg_02"))
    assert p.fields["center_distance_m"] == pytest.approx(0.320, abs=1e-15)
    assert p.value == "0.320 m"
    assert p.fields["surface_gap_m"] == pytest.approx(0.280, abs=1e-9)


def test_distance_axis_deltas_expose_a_misplacement():
    stool = fx.stool()
    p = tier1.distance(stool.part("leg_01"), stool.part("leg_03"))
    dx, dy, dz = p.fields["delta_m"]
    assert (dx, dy) == pytest.approx((0.32, 0.32), abs=1e-15)
    assert dz == pytest.approx(0.0, abs=1e-15)


def test_exact_gap_beats_the_vertex_fallback_on_a_face_to_face_case():
    """The honest limit of the kernel-free path, measured.

    Two crossing bars 200 mm apart: the closest points are in the middle of two
    faces, nowhere near a vertex, so a vertex-to-surface search overshoots by
    ~5x. probe_distance uses the kernel and says so; this test is what keeps the
    README's accuracy note true.
    """
    a, b = fx.crossed_bars(gap=0.2)
    p = tier1.distance(a, b)
    assert p.fields["surface_gap_m"] == pytest.approx(0.2, abs=1e-12)
    assert "exact" in p.fields["gap_method"]
    fallback = kernel.vertex_to_surface_gap(a, b)
    assert fallback > 4.0 * 0.2  # documented failure mode, not a silent one


def test_touching_solids_report_zero_gap_as_a_problem():
    import manifold3d as m3

    a = M.from_manifold(m3.Manifold.cube((1.0, 1, 1), True), name="a")
    b = M.from_manifold(
        m3.Manifold.cube((1.0, 1, 1), True).translate((1.0, 0, 0)), name="b"
    )
    p = tier1.distance(a, b)
    assert p.fields["surface_gap_m"] == pytest.approx(0.0, abs=1e-12)
    assert not p.ok


# ----------------------------------------------------------------- raycast


def test_raycast_hits_at_the_exact_distance():
    p = tier1.raycast(fx.unit_cube(), Point.m(0.5, 0.5, -1.0), "+Z")
    assert p.fields["hit"] is True
    assert p.fields["distance_m"] == pytest.approx(1.0, abs=EXACT)
    assert p.fields["position_m"] == pytest.approx((0.5, 0.5, 0.0), abs=EXACT)
    assert p.fields["normal"] == pytest.approx((0.0, 0.0, -1.0), abs=EXACT)
    assert p.fields["front_face"] is True
    assert p.fields["hits"] == 2  # in the near face, out the far one
    assert p.fields["all_distances_m"][1] == pytest.approx(2.0, abs=EXACT)


def test_raycast_reports_the_provenance_tag_it_hit():
    """DESIGN.md 6: 'hit final #cut_face at 1.71 m'."""
    p = tier1.raycast(fx.crate(), Point.m(0.0, -2.0, 0.29), "+Y")
    assert p.fields["distance_m"] == pytest.approx(1.7, abs=1e-15)
    assert p.fields["part"] == "body"
    assert p.fields["material"] == "oak"
    assert p.value.startswith("hit SM_Crate_Wood #body at 1.700 m")


def test_raycast_through_a_slot_misses_and_says_by_how_much():
    """The crate's middle slot is at x=0 and 480 mm tall, so a ray down the
    centre line passes through the gap. 'miss' alone would be useless; the
    nearest-approach number is the slot's half height."""
    p = tier1.raycast(fx.crate(), Point.m(0.0, -2.0, 0.0), "+Y")
    assert p.fields["hit"] is False
    # the nearest geometry is the slot's inner corner: half the cutter width
    # (0.08 * 0.35 / 2) across, and half the slot height (0.6 - 2*0.06)/2 up
    assert p.fields["nearest_approach_m"] == pytest.approx(
        math.hypot(0.08 * 0.35 / 2.0, 0.24), abs=1e-12
    )
    assert not p.ok


def test_raycast_respects_max_distance():
    p = tier1.raycast(
        fx.unit_cube(), Point.m(0.5, 0.5, -1.0), "+Z", max_distance=Length.mm(500)
    )
    assert p.fields["hit"] is False
    assert p.fields["searched_m"] == pytest.approx(0.5)


def test_raycast_odd_crossing_count_flags_the_hole():
    """Fired down through the missing lid, the ray leaves the solid without ever
    having entered it."""
    p = tier1.raycast(fx.open_box(), Point.m(0.5, 0.5, 2.0), "-Z")
    assert p.fields["hits"] == 1
    assert p.fields["front_face"] is False
    assert any("odd crossing count" in x for x in p.problems)


def test_raycast_agrees_with_the_kernel():
    """Cross-check against manifold3d's own ray_cast, whose `distance` is a
    parametric t along the segment, not a length -- see README footguns."""
    for mesh in (fx.unit_cube(), fx.crate(), fx.sphere(0.5, 64)):
        man = mesh.source
        for origin, direction in (
            ((0.0, -3.0, 0.05), (0.0, 1.0, 0.0)),
            ((-3.0, 0.11, 0.07), (1.0, 0.0, 0.0)),
            ((0.13, 0.02, -3.0), (0.0, 0.0, 1.0)),
        ):
            mine = tier1.raycast(mesh, Point.of(origin, "m"), direction)
            length = 20.0
            end = tuple(np.array(origin) + np.array(direction) * length)
            theirs = [h.distance * length for h in man.ray_cast(origin, end)]
            if not theirs:
                assert mine.fields["hit"] is False
                continue
            assert mine.fields["distance_m"] == pytest.approx(min(theirs), abs=1e-9)


def test_raycast_requires_a_unit_on_a_bare_tuple():
    with pytest.raises(TypeError):
        tier1.raycast(fx.unit_cube(), (0.5, 0.5, -1.0), "+Z")
    ok = tier1.raycast(fx.unit_cube(), (0.5, 0.5, -1.0), "+Z", unit="m")
    assert ok.fields["distance_m"] == pytest.approx(1.0)


# -------------------------------------------------------------- watertight


def test_a_cube_is_watertight():
    p = tier1.is_watertight(fx.unit_cube())
    assert bool(p) is True
    assert p.value == "true"
    assert p.ok


def test_an_open_box_is_not_watertight_and_says_where():
    p = tier1.is_watertight(fx.open_box())
    assert bool(p) is False
    assert p.fields["boundary_edges"] == 4
    assert p.fields["holes"] == 1
    assert p.fields["edge_manifold"] is True
    assert "E101" in p.problems[0]
    assert "4 boundary edges in 1 hole" in p.value


def test_a_non_manifold_fin_is_reported_in_full():
    p = tier1.is_watertight(fx.nonmanifold_fin())
    assert bool(p) is False
    assert p.fields["nonmanifold_edges"] == 1
    assert p.fields["duplicate_halfedges"] == 2
    assert p.fields["oriented"] is False


def test_an_inverted_cube_is_closed_but_flagged():
    p = tier1.is_watertight(fx.inverted_cube())
    assert bool(p) is True  # it really is closed
    assert p.fields["outward"] is False
    assert any("E103" in x for x in p.problems)  # and still wrong


# ------------------------------------------------------------- degenerates


def test_a_clean_cube_has_no_degenerates():
    p = tier1.degenerates(fx.unit_cube())
    assert bool(p) is True
    assert p.value == "none"


def test_degenerates_finds_the_planted_faults():
    p = tier1.degenerates(fx.loose_and_degenerate())
    assert bool(p) is False
    assert p.fields["zero_area_tris"] == 1
    assert p.fields["orphan_verts"] == 1
    assert p.fields["coincident_verts"] == 1
    assert "E105" in p.problems[0]


# ------------------------------------------------------------------ origin


def test_origin_labels():
    assert tier1.origin_placement(fx.unit_cube()).fields["label"] == "min_corner"
    assert tier1.origin_placement(fx.centered_cube()).fields["label"] == "center"
    assert tier1.origin_placement(fx.stool()).fields["label"] == "bottom_center"


def test_off_origin_is_a_warning_with_evidence():
    import manifold3d as m3

    mesh = M.from_manifold(
        m3.Manifold.cube((1.0, 1, 1), True).translate((0.37, 0, 0)), name="shifted"
    )
    p = tier1.origin_placement(mesh)
    assert p.fields["label"] == "off"
    assert "W201" in p.problems[0]
    assert p.fields["offset_from_bbox_center_m"][0] == pytest.approx(-0.37, abs=1e-12)


# ---------------------------------------------------------------- symmetry


def test_a_symmetric_solid_scores_one_on_every_principal_plane():
    for plane in ("YZ", "XZ", "XY"):
        p = tier1.symmetry(fx.centered_cube(0.6), plane)
        assert float(p) == pytest.approx(1.0, abs=1e-12)
        assert p.fields["method"] == "volume_iou"


def test_symmetry_of_a_pocketed_cube_is_analytic():
    """One pocket on +X only. The pocket and its mirror image are disjoint, so
    intersection = V - 2v and union = V exactly."""
    size, pocket, depth = 0.6, 0.1, 0.05
    V = size**3
    v = depth * pocket**2
    mesh = fx.notched_cube(size, pocket, depth)
    yz = tier1.symmetry(mesh, "YZ")
    assert float(yz) == pytest.approx((V - 2 * v) / V, abs=1e-12)
    assert yz.fields["asymmetric_m3"] == pytest.approx(2 * v, rel=1e-9)
    # the pocket is centred in Y and Z, so those planes stay perfect
    assert float(tier1.symmetry(mesh, "XZ")) == pytest.approx(1.0, abs=1e-12)
    assert float(tier1.symmetry(mesh, "XY")) == pytest.approx(1.0, abs=1e-12)


def test_symmetry_about_the_origin_differs_from_about_the_bbox_centre():
    """A cube centred at x = 0.15 is perfectly symmetric about its own middle and
    exactly 1/3 symmetric about the world YZ plane. Reporting one when the agent
    meant the other is a real bug, so the plane position is always printed."""
    import manifold3d as m3

    mesh = M.from_manifold(
        m3.Manifold.cube((0.6, 0.6, 0.6), True).translate((0.15, 0, 0)), name="offset"
    )
    own = tier1.symmetry(mesh, "YZ", about="bbox_center")
    world = tier1.symmetry(mesh, "YZ", about="origin")
    assert float(own) == pytest.approx(1.0, abs=1e-12)
    assert float(world) == pytest.approx(1.0 / 3.0, rel=1e-9)
    assert "x = 0.150 m" in own.text()
    assert "x = 0.000 m" in world.text()


def test_symmetry_about_an_explicit_plane_needs_a_length():
    mesh = fx.centered_cube(0.6)
    with pytest.raises(TypeError):
        tier1.symmetry(mesh, "YZ", about=0.15)
    assert float(tier1.symmetry(mesh, "YZ", about=Length.m(0.0))) == pytest.approx(1.0)


def test_asymmetric_solid_scores_low_on_the_plane_it_breaks():
    scores = tier1.symmetry_planes(fx.stool()).fields["scores"]
    assert scores["YZ"] == pytest.approx(1.0, abs=1e-9)
    assert scores["XZ"] == pytest.approx(1.0, abs=1e-9)
    assert scores["XY"] < 0.3  # a stool is not symmetric top to bottom


def test_surface_sample_method_agrees_on_the_symmetric_cases():
    """The fallback must not disagree about *whether* a shape is symmetric, even
    though its score is a normalised deviation rather than an overlap ratio."""
    cube = tier1.symmetry(fx.centered_cube(0.6), "YZ", method="surface_sample")
    assert float(cube) == pytest.approx(1.0, abs=1e-12)
    assert cube.fields["rms_deviation_m"] == pytest.approx(0.0, abs=1e-12)
    notched = tier1.symmetry(fx.notched_cube(), "YZ", method="surface_sample")
    assert float(notched) < 1.0
    assert notched.fields["max_deviation_m"] == pytest.approx(0.05, abs=1e-9)


def test_symmetry_falls_back_automatically_on_an_open_mesh():
    """The open box lost its lid, so it is still symmetric left-right but no
    longer top-bottom -- and no boolean is possible at all."""
    box = fx.open_box()
    yz = tier1.symmetry(box, "YZ")
    assert yz.fields["method"] == "surface_sample"
    assert float(yz) == pytest.approx(1.0, abs=1e-12)
    assert float(tier1.symmetry(box, "XY")) < 0.99


def test_volume_iou_is_refused_rather_than_faked():
    with pytest.raises(ValueError):
        tier1.symmetry(fx.open_box(), "YZ", method="volume_iou")


# ------------------------------------------------------------------ report


def test_report_is_one_block_with_every_probe():
    text = tier1.report(fx.crate())
    for expected in (
        "measure(bbox)",
        "count(tris)",
        "count(components)",
        "genus",
        "volume(",
        "surface_area(",
        "center_of_mass(",
        "is_watertight(",
        "degenerates(",
        "origin_placement(",
        "symmetry(",
    ):
        assert expected in text
    assert "clean" in text.splitlines()[0]


def test_report_surfaces_problems_at_the_end():
    text = tier1.report(fx.open_box())
    assert "problem(s)" in text.splitlines()[0]
    assert "E101" in text


def test_report_is_deterministic():
    assert tier1.report(fx.stool()) == tier1.report(fx.stool())


def test_every_reported_length_carries_a_unit():
    """A number without a unit is the bug DESIGN.md 4.1 exists to prevent.

    Checked on the probes that report lengths, areas and volumes; counts, scores
    and ratios are dimensionless and correctly carry no unit.
    """
    mesh = fx.crate()
    stool = fx.stool()
    cases = [
        tier1.bbox(mesh),
        tier1.volume(mesh),
        tier1.surface_area(mesh),
        tier1.center_of_mass(mesh),
        tier1.distance(stool.part("leg_01"), stool.part("leg_02")),
        tier1.raycast(mesh, Point.m(0.0, -2.0, 0.29), "+Y"),
        tier1.origin_placement(mesh),
    ]
    import re

    has_unit = re.compile(r"[\d)]\s?(mm|cm|km|in|ft|m)\b|m[²³]|mm[²³]|°|%|px")
    dimensionless = ("of size", "of bounds", "face group", "hits along ray",
                     "origin inside")
    for probe in cases:
        for line in probe.lines():
            payload = line.split("→")[-1] if "→" in line else line.split(":", 1)[-1]
            if not any(ch.isdigit() for ch in payload):
                continue
            if any(tag in line for tag in dimensionless):
                continue  # counts and fractions of the bounds carry no unit
            assert has_unit.search(payload), "no unit in %r" % line
