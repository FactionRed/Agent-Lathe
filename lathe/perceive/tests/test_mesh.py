"""The neutral mesh type: immutability, purity, and the topology bookkeeping
every probe builds on."""

import numpy as np
import pytest

from lathe.perceive import mesh as M
from lathe.perceive import tier1
from lathe.perceive.mesh import Mesh
from lathe.perceive.tests import fixtures as fx


# ------------------------------------------------------- purity, principle 1


def test_arrays_are_read_only():
    cube = fx.unit_cube()
    with pytest.raises(ValueError):
        cube.verts[0, 0] = 99.0
    with pytest.raises(ValueError):
        cube.tris[0, 0] = 3


def test_probes_do_not_mutate_their_input():
    cube = fx.unit_cube()
    before_v = cube.verts.copy()
    before_t = cube.tris.copy()
    tier1.report(cube)
    assert np.array_equal(cube.verts, before_v)
    assert np.array_equal(cube.tris, before_t)


def test_output_is_identical_cold_and_warm():
    """The topology memo must not be able to change an answer."""
    cold = fx.crate()
    warm = fx.crate()
    _ = warm.topology  # warm the memo before probing
    assert tier1.report(cold, detail=True) == tier1.report(warm, detail=True)
    # and twice in a row on the same value
    assert tier1.report(cold, detail=True) == tier1.report(cold, detail=True)


def test_same_arrays_give_the_same_answer_without_the_kernel_handle():
    """A Mesh rebuilt from the arrays alone agrees with the kernel-backed one on
    everything that does not need a boolean."""
    solid = fx.crate()
    plain = Mesh(verts=solid.verts, tris=solid.tris, name=solid.name)
    for probe in (tier1.bbox, tier1.volume, tier1.surface_area, tier1.center_of_mass,
                  tier1.is_watertight, tier1.degenerates):
        assert probe(solid).text() == probe(plain).text()


# ------------------------------------------------------------------ topology


def test_cube_topology_counts():
    topo = fx.unit_cube().topology
    assert topo.n_edges == 18  # 12 cube edges + 6 face diagonals
    assert topo.is_closed and topo.is_edge_manifold and topo.is_oriented
    assert topo.n_components == 1
    assert topo.boundary_loops() == 0
    assert int(topo.valence.sum()) == 2 * 18


def test_open_box_has_four_boundary_edges_in_one_loop():
    topo = fx.open_box().topology
    assert len(topo.boundary_edges) == 4
    assert topo.boundary_loops() == 1
    assert not topo.is_closed
    assert topo.is_edge_manifold  # a hole is not a non-manifold edge


def test_fin_is_non_manifold_and_unoriented():
    topo = fx.nonmanifold_fin().topology
    assert len(topo.nonmanifold_edges) == 1
    assert topo.duplicate_halfedges == 2  # three triangles all traverse A->B
    assert not topo.is_edge_manifold


def _reference_components(edges, n_nodes):
    """Textbook union-find, kept here only to check the vectorised one."""
    parent = list(range(n_nodes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges.tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return [find(i) for i in range(n_nodes)]


@pytest.mark.parametrize("seed", range(8))
def test_vectorised_components_match_a_reference_union_find(seed):
    """The label-propagation implementation is clever, so it gets checked against
    the obvious one on random graphs, including sparse and near-path shapes."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 60))
    m = int(rng.integers(0, 2 * n))
    edges = rng.integers(0, n, size=(m, 2))
    edges = edges[edges[:, 0] != edges[:, 1]]
    if len(edges) == 0:
        edges = np.zeros((0, 2), dtype=np.int64)
    mine = M.connected_labels(edges, n)
    want = _reference_components(edges, n)
    # labels need not be equal, but the partition must be
    assert [int(x) for x in mine] == want


def test_vectorised_components_handle_a_long_path():
    """A path is the worst case for label propagation: check it still converges
    to one component."""
    n = 500
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    labels = M.connected_labels(edges, n)
    assert len(set(labels.tolist())) == 1


def test_two_solids_are_two_components():
    a, b = fx.crossed_bars()
    joined = Mesh(
        verts=np.vstack([a.verts, b.verts]),
        tris=np.vstack([a.tris, b.tris + len(a.verts)]),
        name="two_bars",
    )
    assert joined.topology.n_components == 2
    assert tier1.counts(joined).fields["components"] == 2


# ------------------------------------------------------------ exact geometry


def test_signed_volume_of_a_unit_cube_is_exactly_one():
    assert M.signed_volume(fx.unit_cube()) == 1.0


def test_reversed_winding_flips_the_sign_exactly():
    assert M.signed_volume(fx.inverted_cube()) == -1.0


def test_point_triangle_distance_against_closed_form():
    """Distances to a single triangle in the z=0 plane, by hand."""
    tri = Mesh(verts=np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]]),
               tris=np.array([[0, 1, 2]]), name="tri")
    points = np.array([
        [0.25, 0.25, 2.0],   # above the interior -> plane distance 2
        [-1.0, 0.0, 0.0],    # beyond vertex a    -> 1
        [2.0, 0.0, 0.0],     # beyond vertex b    -> 1
        [1.0, 1.0, 0.0],     # off the hypotenuse -> sqrt(2)/2
        [0.5, -0.5, 0.0],    # off edge ab        -> 0.5
    ])
    got = np.sqrt(M.point_triangle_distance2(points, tri))
    want = np.array([2.0, 1.0, 1.0, np.sqrt(2) / 2, 0.5])
    assert np.allclose(got, want, atol=1e-12)


def test_bounds_ignore_unreferenced_vertices():
    sloppy = fx.loose_and_degenerate()  # carries a vertex at (2,2,2) used by no face
    lo, hi = M.bounds(sloppy)
    assert np.allclose(hi, [1, 1, 1])
    assert np.allclose(lo, [0, 0, 0])


# ------------------------------------------------------------------ adapters


def test_manifold_round_trip_is_lossless():
    """Mesh64 keeps float64, so verts survive a round trip bit for bit."""
    crate = fx.crate()
    plain = Mesh(verts=crate.verts, tris=crate.tris, name="rebuilt")
    man, why = M.to_manifold(plain)
    assert why is None and man is not None
    back = M.from_manifold(man, name="rebuilt")
    assert M.signed_volume(back) == pytest.approx(M.signed_volume(crate), rel=1e-15)


def test_open_mesh_has_no_kernel_representation():
    man, why = M.to_manifold(fx.open_box())
    assert man is None
    assert "NotManifold" in why


def test_provenance_survives_the_boolean():
    """DESIGN.md 5.2 on the kernel: run_original_id attributes every triangle."""
    crate = fx.crate()
    assert crate.part_id is not None
    names = set(crate.part_names())
    assert "body" in names and "slot_1" in names and "slot_5" in names
    assert len(names) == 6


def test_part_scoping_restricts_geometry():
    stool = fx.stool()
    seat = stool.part("seat")
    lo, hi = M.bounds(seat)
    assert np.allclose(hi - lo, [0.4, 0.4, 0.04], atol=1e-12)
    assert seat.source is None  # a part of a solid is not itself a solid
    with pytest.raises(KeyError):
        stool.part("leg_99")
