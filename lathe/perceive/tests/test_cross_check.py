"""
Independent-implementation cross-checks.

The probes compute volume, area, genus, components and ray hits from the neutral
arrays with numpy, so that they work on meshes the kernel cannot represent. That
independence is only worth anything if the two implementations agree where both
are defined -- so this file compares every quantity against manifold3d's own
answer, and pins the size of the disagreement where one is approximate.
"""

import numpy as np
import pytest

import manifold3d as m3

from lathe.perceive import kernel
from lathe.perceive import mesh as M
from lathe.perceive import tier1
from lathe.perceive.tests import fixtures as fx

MESHES = [
    ("unit_cube", fx.unit_cube),
    ("centered_cube", fx.centered_cube),
    ("sphere", lambda: fx.sphere(0.7, 64)),
    ("cylinder", lambda: fx.cylinder(0.3, 1.1, 128)),
    ("torus", lambda: fx.torus(1.0, 0.3, 64)),
    ("tetrahedron", fx.tetrahedron),
    ("crate", fx.crate),
    ("stool", fx.stool),
    ("notched_cube", fx.notched_cube),
]


@pytest.mark.parametrize("name,build", MESHES, ids=[n for n, _ in MESHES])
def test_volume_agrees_with_the_kernel(name, build):
    mesh = build()
    mine = tier1.volume(mesh).fields["volume_m3"]
    assert mine == pytest.approx(mesh.source.volume(), rel=1e-12)


@pytest.mark.parametrize("name,build", MESHES, ids=[n for n, _ in MESHES])
def test_surface_area_agrees_with_the_kernel(name, build):
    mesh = build()
    mine = tier1.surface_area(mesh).fields["area_m2"]
    assert mine == pytest.approx(mesh.source.surface_area(), rel=1e-12)


@pytest.mark.parametrize("name,build", MESHES, ids=[n for n, _ in MESHES])
def test_bbox_agrees_with_the_kernel(name, build):
    mesh = build()
    f = tier1.bbox(mesh).fields
    lo = np.array(f["min_m"])
    hi = np.array(f["max_m"])
    theirs = np.array(mesh.source.bounding_box())
    assert np.allclose(np.concatenate([lo, hi]), theirs, atol=1e-12)


@pytest.mark.parametrize("name,build", MESHES, ids=[n for n, _ in MESHES])
def test_genus_and_components_agree_with_the_kernel(name, build):
    mesh = build()
    f = tier1.counts(mesh).fields
    assert f["components"] == len(mesh.source.decompose())
    assert f["genus"] == mesh.source.genus()


def test_genus_agrees_on_a_multi_component_case():
    """Euler characteristic per component, against Manifold's own per-solid genus."""
    solid = m3.Manifold.cube((1.0, 1, 1), True) + fx.torus(0.4, 0.1, 48).source.translate(
        (4.0, 0, 0)
    )
    mesh = M.from_manifold(solid, name="cube_and_torus")
    f = tier1.counts(mesh).fields
    assert f["components"] == 2
    assert sorted(f["genus_per_component"]) == sorted(
        [part.genus() for part in solid.decompose()]
    )
    assert f["genus"] == 1


def test_min_gap_agrees_where_the_fallback_is_exact():
    """Face-to-face along an axis: the closest pair involves vertices, so the
    kernel-free bound is tight."""
    a = M.from_manifold(m3.Manifold.cube((1.0, 1, 1), True), name="a")
    b = M.from_manifold(
        m3.Manifold.cube((1.0, 1, 1), True).translate((1.75, 0, 0)), name="b"
    )
    exact = kernel.min_gap_exact(a.source, b.source, 10.0)
    rough = kernel.vertex_to_surface_gap(a, b)
    assert exact == pytest.approx(0.75, abs=1e-12)
    assert rough == pytest.approx(exact, abs=1e-12)


def test_min_gap_fallback_overestimates_on_the_documented_case():
    a, b = fx.crossed_bars(gap=0.2)
    exact = kernel.min_gap_exact(a.source, b.source, 10.0)
    rough = kernel.vertex_to_surface_gap(a, b)
    assert exact == pytest.approx(0.2, abs=1e-12)
    assert rough > exact  # never under-reports, so it can only be conservative
    assert rough == pytest.approx(0.9708, abs=1e-3)  # the number in the README


def test_mirroring_twice_is_the_identity():
    """A property check on the symmetry machinery itself: mirror(mirror(A)) == A,
    so its IoU against A must be exactly 1."""
    mesh = fx.notched_cube()
    once = kernel.mirror_about(mesh.source, 0, 0.0)
    twice = kernel.mirror_about(once, 0, 0.0)
    assert kernel.volume_overlap(mesh.source, twice)["iou"] == pytest.approx(1.0, abs=1e-12)
    assert twice.volume() == pytest.approx(mesh.source.volume(), rel=1e-12)


def test_symmetry_methods_agree_on_the_verdict():
    """The two symmetry methods are different measures, so their scores differ --
    but they must never disagree about whether a shape is symmetric."""
    for build, plane, symmetric in (
        (fx.centered_cube, "YZ", True),
        (fx.crate, "YZ", True),
        (fx.crate, "XY", True),
        (fx.notched_cube, "YZ", False),
        (fx.stool, "XY", False),
    ):
        mesh = build()
        exact = float(tier1.symmetry(mesh, plane, method="volume_iou"))
        sampled = float(tier1.symmetry(mesh, plane, method="surface_sample"))
        if symmetric:
            assert exact == pytest.approx(1.0, abs=1e-9)
            assert sampled == pytest.approx(1.0, abs=1e-9)
        else:
            assert exact < 0.999
            assert sampled < 0.999


def test_surface_sampling_is_deterministic():
    """No RNG anywhere: the same mesh must give byte-identical samples."""
    mesh = fx.sphere(0.5, 32)
    a = kernel.sample_surface(mesh)
    b = kernel.sample_surface(mesh)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert a[1].sum() == pytest.approx(mesh.source.surface_area(), rel=1e-12)


def test_outline_area_agrees_with_a_hand_computed_silhouette():
    """A 1 x 2 x 3 box: silhouettes of 2, 3 and 6 m2 down X, Y and Z."""
    mesh = M.from_manifold(m3.Manifold.cube((1.0, 2.0, 3.0), True), name="brick")
    want = {0: 2.0 * 3.0, 1: 1.0 * 3.0, 2: 1.0 * 2.0}
    for axis, area in want.items():
        assert kernel.outline(mesh.source, axis).area() == pytest.approx(area, rel=1e-12)
        rough = kernel.outline_overlap_raster(mesh, mesh, axis, res=256)
        assert rough["iou"] == pytest.approx(1.0, abs=1e-12)
        assert rough["area_a"] == pytest.approx(area, rel=0.02)


def test_float32_round_trip_would_have_cost_real_precision():
    """Why from_manifold reads to_mesh64: the float32 path loses ~1e-7 relative,
    which is enough to break a 1 um tolerance on a 600 mm part."""
    man = m3.Manifold.cube((0.6, 0.6, 0.6), True).translate((0.1, 0.0, 0.0))
    exact = np.array(man.to_mesh64().vert_properties, dtype=np.float64)[:, :3]
    lossy = np.array(man.to_mesh().vert_properties, dtype=np.float64)[:, :3]
    error = np.abs(exact - lossy).max()
    assert error > 1e-9  # the float32 mesh really is that far off
    mesh = M.from_manifold(man, name="probe_me")
    assert np.abs(mesh.verts - exact).max() == 0.0  # and we do not pay it
