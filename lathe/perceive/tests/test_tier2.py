"""
Tier 2 summaries against closed-form truth.

The UV fixture is built so the answers are exact: six 0.2 x 0.2 islands on a 1 m
cube at 1024 px is 2.048 px/cm on every island, 24% coverage, and 51.2 px of
padding. The LOD fixture is a real `simplify` chain, so the ratios are whatever
the kernel produced -- there the tests check the *relationships* (monotone
reduction, silhouette deviation growing with reduction, warnings firing at the
declared thresholds) rather than invented constants.
"""

import numpy as np
import pytest

from lathe.perceive import tier2
from lathe.perceive.tests import fixtures as fx


# --------------------------------------------------------------- topology


def test_cube_topology_is_six_quads():
    s = tier2.topology_summary(fx.unit_cube())
    assert s.fields["faces_logical"] == 6
    assert s.fields["face_degrees"] == {4: 6}
    assert s.fields["components"] == 1
    assert s.fields["genus"] == 0
    assert s.fields["holes"] == 0
    assert not s.warnings


def test_split_quad_aspect_ratio_is_root_three():
    """Each face of a cube is two right isoceles triangles. Normalised so that
    an equilateral triangle is 1.0, that shape is exactly sqrt(3)."""
    s = tier2.topology_summary(fx.unit_cube())
    p50, p95, mx = s.fields["aspect_ratio"]
    assert p50 == pytest.approx(np.sqrt(3.0), abs=1e-12)
    assert mx == pytest.approx(np.sqrt(3.0), abs=1e-12)
    assert s.fields["slivers_over_8"] == 0


def test_cube_tri_areas_are_all_one_half():
    lo, mid, hi = tier2.topology_summary(fx.unit_cube()).fields["tri_area_m2"]
    assert (lo, mid, hi) == pytest.approx((0.5, 0.5, 0.5), abs=1e-15)


def test_cube_dihedral_histogram_is_six_flat_and_twelve_hard():
    """18 interior edges: 6 face diagonals at 0 degrees, 12 cube edges at 90."""
    hist = tier2.topology_summary(fx.unit_cube()).fields["dihedral_hist"]
    assert hist["flat<1°"] == 6
    assert hist["hard>60°"] == 12
    assert sum(hist.values()) == 18


def test_crate_topology_reports_the_pierced_faces_as_n_gons():
    s = tier2.topology_summary(fx.crate())
    assert s.fields["face_degrees"] == {4: 24, 24: 2}
    assert any("E102" in w for w in s.warnings)
    assert s.fields["genus"] == 5


def test_open_box_topology_reports_the_hole():
    s = tier2.topology_summary(fx.open_box())
    assert s.fields["holes"] == 1
    assert s.fields["genus"] is None


def test_valence_histogram_totals_the_vertices():
    s = tier2.topology_summary(fx.sphere(1.0, 64))
    assert sum(s.fields["valence_hist"].values()) == s.fields["verts"]


# --------------------------------------------------------------------- UV


def test_box_projection_gives_six_islands():
    mesh = fx.uv_cube()
    labels = tier2.uv_islands(mesh)
    assert len(set(labels.tolist())) == 6
    assert tier2.uv_summary(mesh).fields["islands"] == 6


def test_texel_density_is_the_analytic_value():
    """1024 px over a 0.2 UV span on a 1 m face = 2.048 px/cm, on every island."""
    mesh = fx.uv_cube(texture_px=1024)
    s = tier2.uv_summary(mesh, texture_px=1024, target_density_px_cm=2.048)
    want = fx.uv_cube_analytic_density()
    lo, mid, hi = s.fields["density_px_cm"]
    assert lo == pytest.approx(want, rel=1e-12)
    assert hi == pytest.approx(want, rel=1e-12)
    assert s.fields["worst_density_deviation"] == pytest.approx(0.0, abs=1e-12)
    assert not s.warnings


def test_texel_density_formula_matches_by_hand():
    assert tier2.texel_density_px_cm(0.04, 1.0, 1024) == pytest.approx(2.048, rel=1e-12)
    # halve the UV area -> density falls by sqrt(2)
    assert tier2.texel_density_px_cm(0.02, 1.0, 1024) == pytest.approx(
        2.048 / np.sqrt(2.0), rel=1e-12
    )


def test_coverage_and_padding_are_exact():
    s = tier2.uv_summary(fx.uv_cube(), texture_px=1024)
    assert s.fields["coverage"] == pytest.approx(6 * 0.2 * 0.2, rel=1e-12)
    assert s.fields["padding_px"] == pytest.approx(0.05 * 1024, rel=1e-12)
    assert s.fields["uv_outside_unit_square"] == 0


def test_clean_uv_has_no_flipped_islands_and_no_overlap():
    s = tier2.uv_summary(fx.uv_cube())
    assert s.fields["flipped_tris"] == 0
    assert s.fields["overlap_cells"] == 0
    assert s.fields["overlap_fraction"] == 0.0


def test_stacked_islands_are_detected_as_overlap():
    """Two faces mapped onto the same rect. They share no mesh edge, so they must
    stay two islands -- and be reported, not merged."""
    s = tier2.uv_summary(fx.uv_cube(overlap_face=5), texture_px=1024)
    assert s.fields["islands"] == 6
    assert s.fields["overlap_cells"] > 0
    worst = s.fields["overlap_pairs"][0]
    assert worst[0] == pytest.approx(1.0, abs=0.02)  # 100% of the smaller island
    assert any("E106" in w for w in s.warnings)


def test_shrunken_island_trips_the_density_warning():
    s = tier2.uv_summary(
        fx.uv_cube(shrink_face=4), texture_px=1024, target_density_px_cm=2.048
    )
    lo, _, hi = s.fields["density_px_cm"]
    assert lo == pytest.approx(2.048 / 2.0, rel=1e-12)  # half the rect side
    assert hi == pytest.approx(2.048, rel=1e-12)
    assert s.fields["worst_density_deviation"] == pytest.approx(0.5, rel=1e-12)
    assert any("W302" in w for w in s.warnings)


def test_zero_padding_trips_W202():
    s = tier2.uv_summary(fx.uv_cube(overlap_face=5), texture_px=1024)
    assert s.fields["padding_px"] == pytest.approx(0.0, abs=1e-9)
    assert any("W202" in w for w in s.warnings)


def test_uv_summary_says_so_when_there_is_no_uv_set():
    s = tier2.uv_summary(fx.unit_cube())
    assert s.fields["has_uv"] is False
    assert "absent" in s.text()


# -------------------------------------------------------------- materials


def test_material_areas_are_exact_on_a_unit_cube():
    """Four sides oak (4 m2), top and bottom steel (2 m2)."""
    s = tier2.material_table(fx.uv_cube())
    by_name = {r["name"]: r for r in s.fields["materials"]}
    assert by_name["oak"]["area_m2"] == pytest.approx(4.0, abs=1e-15)
    assert by_name["steel_brushed"]["area_m2"] == pytest.approx(2.0, abs=1e-15)
    assert by_name["oak"]["area_fraction"] == pytest.approx(2.0 / 3.0, rel=1e-15)
    assert by_name["oak"]["tris"] == 8


def test_material_table_lists_the_parts_it_covers():
    s = tier2.material_table(fx.crate())
    by_name = {r["name"]: r for r in s.fields["materials"]}
    assert by_name["oak"]["parts"] == ("body",)
    assert set(by_name["steel_brushed"]["parts"]) == {
        "slot_1", "slot_2", "slot_3", "slot_4", "slot_5"
    }


def test_material_shares_sum_to_one():
    s = tier2.material_table(fx.stool())
    assert sum(r["tri_fraction"] for r in s.fields["materials"]) == pytest.approx(1.0)
    assert sum(r["area_fraction"] for r in s.fields["materials"]) == pytest.approx(1.0)


# ------------------------------------------------------------------ parts


def test_part_summary_covers_every_triangle_once():
    s = tier2.part_summary(fx.stool())
    parts = s.fields["parts"]
    assert len(parts) == 5
    assert sum(p["tris"] for p in parts) == fx.stool().n_tris
    assert sum(p["tri_fraction"] for p in parts) == pytest.approx(1.0)


def test_part_bbox_is_the_declared_geometry():
    s = tier2.part_summary(fx.stool())
    seat = next(p for p in s.fields["parts"] if p["name"] == "seat")
    assert seat["size_m"] == pytest.approx((0.4, 0.4, 0.04), abs=1e-12)


def test_part_volume_is_withheld_when_the_part_is_not_closed():
    """Every part of a unioned solid is open at its seam, so no part may claim a
    volume. Withholding it is the point."""
    for p in tier2.part_summary(fx.stool()).fields["parts"]:
        assert p["closed"] is False
        assert p["volume_m3"] is None
    assert " -  " in tier2.part_summary(fx.stool()).text() or "-" in tier2.part_summary(
        fx.stool()
    ).text()


def test_crate_parts_are_the_body_and_five_slots():
    names = [p["name"] for p in tier2.part_summary(fx.crate()).fields["parts"]]
    assert names[0] == "body"
    assert sorted(names[1:]) == ["slot_1", "slot_2", "slot_3", "slot_4", "slot_5"]


# -------------------------------------------------------------------- LOD


def test_lod_chain_ratios_are_monotone_and_consistent():
    s = tier2.lod_summary(fx.lod_chain())
    levels = s.fields["levels"]
    assert len(levels) == 4
    assert levels[0]["ratio_base"] == 1.0
    tris = [lv["tris"] for lv in levels]
    assert tris == sorted(tris, reverse=True)
    for i in range(1, len(levels)):
        assert levels[i]["ratio_base"] == pytest.approx(tris[i] / tris[0], rel=1e-12)
        assert levels[i]["ratio_prev"] == pytest.approx(tris[i] / tris[i - 1], rel=1e-12)


def test_lod_silhouette_deviation_grows_with_reduction():
    s = tier2.lod_summary(fx.lod_chain())
    devs = [lv["silhouette_deviation"] for lv in s.fields["levels"]]
    assert devs[0] == 0.0
    assert all(devs[i] < devs[i + 1] for i in range(len(devs) - 1))
    assert s.fields["method"] == ("outline_iou (exact)",)


def test_lod_identical_levels_have_zero_deviation():
    """A level that is bit-identical to LOD0 must score exactly 0 deviation, or
    the metric has a bias."""
    base = fx.lod_chain(tolerances=(0.0,))[0]
    s = tier2.lod_summary([("SM_Twin_LOD0", base[1]), ("SM_Twin_LOD1", base[1])])
    assert s.fields["levels"][1]["silhouette_deviation"] == pytest.approx(0.0, abs=1e-12)
    assert s.fields["levels"][1]["volume_delta"] == 0.0


def test_lod_warnings_fire_at_the_declared_thresholds():
    s = tier2.lod_summary(fx.lod_chain(), target_ratio=0.5, ratio_tol=0.15,
                          silhouette_max=0.02)
    assert any("W205" in w for w in s.warnings)
    assert any("W204" in w for w in s.warnings)
    # and a permissive threshold silences W205
    relaxed = tier2.lod_summary(fx.lod_chain(), silhouette_max=0.5, ratio_tol=0.9)
    assert not relaxed.warnings


def test_raster_silhouette_agrees_with_the_exact_one():
    """The kernel-free fallback must land within a fraction of a percent of the
    exact 2D boolean, or the fallback is not usable."""
    from lathe.perceive import kernel

    chain = fx.lod_chain()
    base, coarse = chain[0][1], chain[-1][1]
    for axis in (0, 1, 2):
        exact = kernel.outline_overlap_exact(base.source, coarse.source, axis)["iou"]
        rough = kernel.outline_overlap_raster(base, coarse, axis, res=512)["iou"]
        assert abs(exact - rough) < 0.005


# ----------------------------------------------------------------- report


def test_report_includes_only_the_applicable_blocks():
    plain = tier2.report(fx.unit_cube())
    assert "topology" in plain
    assert "uv " not in plain
    full = tier2.report(fx.uv_cube(), texture_px=1024, target_density_px_cm=2.048)
    for block in ("topology", "materials", "parts", "uv "):
        assert block in full


def test_report_is_deterministic():
    assert tier2.report(fx.crate()) == tier2.report(fx.crate())
