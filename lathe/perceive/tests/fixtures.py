"""
Solids whose properties are known in closed form, plus a few deliberately broken
ones.

Every fixture documents the analytic truth it carries, because the point of the
test suite is numerical accuracy, not coverage: a unit cube encloses exactly
1.0 m3, a sphere of radius r encloses 4/3 pi r^3 in the limit of refinement, the
crate from DESIGN.md 4 encloses 0.216 - 5 * 0.028 * 0.6 * 0.48 m3, and a mesh
with a hole in it must fail is_watertight.
"""

from __future__ import annotations

import math

import numpy as np

from ..mesh import Mesh, from_manifold, tri_normals

try:
    import manifold3d as m3
except ImportError:  # pragma: no cover
    m3 = None


# --------------------------------------------------------------- primitives


def unit_cube(name: str = "unit_cube") -> Mesh:
    """1 m cube in the first octant. volume 1.0 m3, com (0.5,0.5,0.5) m."""
    return from_manifold(m3.Manifold.cube((1.0, 1.0, 1.0), False), name=name)


def centered_cube(size: float = 0.6, name: str = "cube") -> Mesh:
    """Cube centred on the origin. volume size^3, com (0,0,0)."""
    return from_manifold(m3.Manifold.cube((size, size, size), True), name=name)


def sphere(radius: float = 1.0, segments: int = 128, name: str = "sphere") -> Mesh:
    """Geodesic sphere. Volume tends to 4/3 pi r^3 from below as segments grow."""
    return from_manifold(m3.Manifold.sphere(radius, segments), name=name)


def cylinder(radius: float = 0.5, height: float = 2.0, segments: int = 512,
             name: str = "cylinder") -> Mesh:
    """Right cylinder centred on the origin. Volume tends to pi r^2 h."""
    return from_manifold(
        m3.Manifold.cylinder(height, radius, radius, segments, True), name=name
    )


def torus(major: float = 1.0, minor: float = 0.3, segments: int = 128,
          name: str = "torus") -> Mesh:
    """Genus 1. Volume tends to 2 pi^2 R r^2, area to 4 pi^2 R r."""
    profile = m3.CrossSection.circle(minor, segments).translate((major, 0.0))
    return from_manifold(profile.revolve(circular_segments=segments), name=name)


def tetrahedron(name: str = "tetrahedron") -> Mesh:
    """Manifold's regular tetrahedron: corners (1,1,1) (-1,-1,1) (1,-1,-1)
    (-1,1,-1), so its volume is 8/3 m3 exactly."""
    return from_manifold(m3.Manifold.tetrahedron(), name=name)


# ----------------------------------------------------------- broken on purpose


def open_box(name: str = "open_box") -> Mesh:
    """Unit cube with the +Z face deleted: 4 boundary edges, one hole.

    This is the fixture that proves a probe *reports* a problem rather than
    returning a confident number. It cannot be a Manifold at all -- the kernel
    answers Error.NotManifold -- so it also exercises every kernel-free path.
    """
    cube = unit_cube()
    normals = tri_normals(cube)
    keep = ~(normals[:, 2] > 0.999)
    return cube.select_tris(keep, name=name)


def inverted_cube(name: str = "inverted_cube") -> Mesh:
    """Unit cube with every triangle wound backwards: signed volume -1.0 m3."""
    cube = unit_cube()
    return Mesh(
        verts=cube.verts,
        tris=cube.tris[:, ::-1],
        name=name,
        face_id=cube.face_id,
    )


def nonmanifold_fin(name: str = "nonmanifold_fin") -> Mesh:
    """Three triangles sharing one edge: 1 non-manifold edge, 2 duplicate
    half-edges, 5 boundary edges (E101)."""
    verts = np.array(
        [[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1], [0.0, -1, 0]]
    )
    tris = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]])
    return Mesh(verts=verts, tris=tris, name=name)


def loose_and_degenerate(name: str = "sloppy_cube") -> Mesh:
    """Unit cube plus an orphan vertex, a duplicated vertex and a sliver tri."""
    cube = unit_cube()
    verts = np.vstack([cube.verts, [[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]]])
    n = len(cube.verts)
    zero_area = np.array([[0, 1, n + 1]])  # (0,0,0) duplicated -> zero area
    tris = np.vstack([cube.tris, zero_area])
    return Mesh(verts=verts, tris=tris, name=name)


# ------------------------------------------------------------- named examples


def notched_cube(size: float = 0.6, pocket: float = 0.1, depth: float = 0.05,
                 name: str = "notched_cube") -> Mesh:
    """Centred cube with a shallow pocket milled into its +X face only.

    Analytic symmetry: the pocket and its mirror image are disjoint, so
        vol(A and mirror(A)) = V - 2v ,  vol(A or mirror(A)) = V
        symmetry(YZ) = (V - 2v) / V   with V = size^3, v = depth*pocket^2
    and the pocket is centred in Y and Z, so symmetry(XZ) = symmetry(XY) = 1.
    The bounding box is unchanged, so the mirror plane sits at x = 0 either way.
    """
    body = m3.Manifold.cube((size, size, size), True).as_original()
    cut = (
        m3.Manifold.cube((depth * 2.0, pocket, pocket), True)
        .translate((size / 2.0, 0.0, 0.0))
        .as_original()
    )
    solid = body - cut
    return from_manifold(
        solid,
        name=name,
        parts={body.original_id(): "body", cut.original_id(): "pocket"},
    )


def crate(size: float = 0.6, slat_w: float = 0.08, slat_n: int = 5,
          rail: float = 0.06, name: str = "SM_Crate_Wood") -> Mesh:
    """The DESIGN.md 4 crate, with the section 5 fix to the cutter height.

    Analytic volume: size^3 - slat_n * (slat_w*0.35) * size * (size - 2*rail)
    -- the slots run right through in Y and never overlap in X. Topology:
    one component, genus slat_n. Symmetric about all three principal planes.
    Parts come from Manifold run_original_id, so `body` and each `slot_i` stay
    distinguishable exactly as DESIGN.md 5.2 requires.
    """
    body = m3.Manifold.cube((size, size, size), True).as_original()
    parts = {body.original_id(): "body"}
    solid = body
    span = size * 0.9
    for i in range(slat_n):
        x = -span / 2.0 + (i / (slat_n - 1.0)) * span
        cutter = (
            m3.Manifold.cube((slat_w * 0.35, size * 1.1, size - rail * 2.0), True)
            .translate((x, 0.0, 0.0))
            .as_original()
        )
        parts[cutter.original_id()] = "slot_%d" % (i + 1)
        solid = solid - cutter
    mesh = from_manifold(solid, name=name, parts=parts)
    # oak on the body, steel on every surface the boolean cut (#cut_face)
    body_id = body.original_id()
    material_id = np.where(mesh.part_id == body_id, 0, 1)
    return Mesh(
        verts=mesh.verts,
        tris=mesh.tris,
        name=mesh.name,
        face_id=mesh.face_id,
        part_id=mesh.part_id,
        parts=mesh.parts,
        material_id=material_id,
        materials={0: "oak", 1: "steel_brushed"},
        source=solid,
    )


def crate_analytic_volume(size: float = 0.6, slat_w: float = 0.08,
                          slat_n: int = 5, rail: float = 0.06) -> float:
    return size**3 - slat_n * (slat_w * 0.35) * size * (size - 2.0 * rail)


def stool(seat: float = 0.4, seat_h: float = 0.45, thick: float = 0.04,
          inset: float = 0.04, leg_r: float = 0.02, segments: int = 64,
          name: str = "SM_Stool") -> Mesh:
    """DESIGN.md 5.4's worked example: a seat and four legs, by relationship.

    Leg centres sit `inset` from the seat edge, so distance(leg_01, leg_02) is
    seat - 2*inset = 0.320 m centre to centre and 0.320 - 2*leg_r = 0.280 m
    surface to surface. Symmetric about YZ and XZ, not about XY.
    """
    top = (
        m3.Manifold.cube((seat, seat, thick), True)
        .translate((0.0, 0.0, seat_h - thick / 2.0))
        .as_original()
    )
    parts = {top.original_id(): "seat"}
    materials = {0: "oak"}
    solid = top
    material_of_part = {top.original_id(): 0}
    offset = seat / 2.0 - inset
    order = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for i, (sx, sy) in enumerate(order):
        leg = (
            m3.Manifold.cylinder(seat_h - thick, leg_r, leg_r, segments, False)
            .translate((sx * offset, sy * offset, 0.0))
            .as_original()
        )
        parts[leg.original_id()] = "leg_%02d" % (i + 1)
        material_of_part[leg.original_id()] = 1
        solid = solid + leg
    materials[1] = "steel_brushed"
    mesh = from_manifold(solid, name=name, parts=parts)
    material_id = np.array(
        [material_of_part.get(int(p), 0) for p in mesh.part_id], dtype=np.int64
    )
    return Mesh(
        verts=mesh.verts,
        tris=mesh.tris,
        name=mesh.name,
        face_id=mesh.face_id,
        part_id=mesh.part_id,
        parts=mesh.parts,
        material_id=material_id,
        materials=materials,
        source=solid,
    )


def crossed_bars(gap: float = 0.2, bar: float = 0.1, length: float = 2.0):
    """Two long bars crossing at right angles, `gap` apart in Z.

    The closest pair of points lies in the *interior* of two faces, far from any
    vertex: the true gap is `gap`, but a vertex-to-surface search has to reach
    all the way to a bar end and reports roughly length/2 instead. This is the
    fixture that pins down where the kernel-free distance fallback is wrong, and
    by how much.
    """
    a = m3.Manifold.cube((length, bar, bar), True)
    b = m3.Manifold.cube((bar, length, bar), True).translate((0.0, 0.0, gap + bar))
    return from_manifold(a, name="bar_X"), from_manifold(b, name="bar_Y")


# ------------------------------------------------------------------ UV sets

_FACE_KEYS = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
_FACE_NAMES = ["side_+X", "side_-X", "side_+Y", "side_-Y", "top_+Z", "bottom_-Z"]


def _box_face_groups(mesh: Mesh) -> np.ndarray:
    """Group index 0..5 per triangle, by which cube face it lies on."""
    n = tri_normals(mesh)
    axis = np.argmax(np.abs(n), axis=1)
    sign = np.sign(n[np.arange(len(n)), axis])
    out = np.zeros(len(n), dtype=np.int64)
    for gi, (ax, sg) in enumerate(_FACE_KEYS):
        out[(axis == ax) & (sign == sg)] = gi
    return out


def uv_cube(size: float = 1.0, rect: float = 0.2, pad: float = 0.05,
            texture_px: int = 1024, name: str = "SM_UVBox",
            shrink_face: int | None = None, overlap_face: int | None = None) -> Mesh:
    """Box-projected cube: six square UV islands of known area.

    Each face of area size^2 maps to a rect^2 UV island, so every island has the
    same texel density:
        px/m  = texture_px * rect / size
        px/cm = px/m / 100          -> 1024 * 0.2 / 1.0 / 100 = 2.048 px/cm
    Coverage is 6 * rect^2 = 24% at the defaults, and neighbouring islands are
    `pad` apart, i.e. 51.2 px at 1024.

    shrink_face  halves that island's rect, so density drops by 2x (W302 bait)
    overlap_face stacks that island exactly on island 0 (E106 bait)
    """
    cube = from_manifold(m3.Manifold.cube((size, size, size), True), name=name)
    group = _box_face_groups(cube)
    uv = np.zeros((cube.n_tris, 3, 2), dtype=np.float64)
    for gi in range(6):
        rows = np.flatnonzero(group == gi)
        side = rect * (0.5 if shrink_face == gi else 1.0)
        col, row = gi % 4, gi // 4
        if overlap_face == gi:
            col, row = 0, 0
        u0 = pad / 2.0 + col * (rect + pad)
        v0 = pad / 2.0 + row * (rect + pad)
        ax = _FACE_KEYS[gi][0]
        keep = [i for i in range(3) if i != ax]
        corners = cube.verts[cube.tris[rows]]  # (n,3,3)
        local = (corners[:, :, keep] + size / 2.0) / size  # 0..1 across the face
        # a naive box projection mirrors the faces whose normal runs backwards
        # against the kept axis pair; flip u there so no island is wound
        # negatively (a negative signed UV area is a flipped island, E106)
        d1 = local[:, 1] - local[:, 0]
        d2 = local[:, 2] - local[:, 0]
        signed = 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        if signed.sum() < 0:
            local[:, :, 0] = 1.0 - local[:, :, 0]
        uv[rows, :, 0] = u0 + local[:, :, 0] * side
        uv[rows, :, 1] = v0 + local[:, :, 1] * side
    material_id = np.where(group >= 4, 1, 0)  # top/bottom get the second material
    return Mesh(
        verts=cube.verts,
        tris=cube.tris,
        name=name,
        face_id=group,
        part_id=group,
        parts={i: _FACE_NAMES[i] for i in range(6)},
        material_id=material_id,
        materials={0: "oak", 1: "steel_brushed"},
        uv=uv,
        source=cube.source,
    )


def uv_cube_analytic_density(size: float = 1.0, rect: float = 0.2,
                             texture_px: int = 1024) -> float:
    """px per cm for the clean uv_cube."""
    return texture_px * rect / size / 100.0


# ------------------------------------------------------------------- LOD set


def lod_chain(radius: float = 0.5, segments: int = 96,
              tolerances=(0.0, 0.004, 0.012, 0.035), name: str = "SM_Rock"):
    """A real reduction chain: Manifold `simplify` at growing tolerance.

    Returns [(level_name, Mesh)]. Every level stays a closed solid, so the
    silhouette comparison can use the exact kernel path.
    """
    base = m3.Manifold.sphere(radius, segments)
    out = []
    for i, tol in enumerate(tolerances):
        solid = base if tol <= 0.0 else base.simplify(tol)
        out.append(("%s_LOD%d" % (name, i), from_manifold(solid, name="%s_LOD%d" % (name, i))))
    return out


def sphere_volume(radius: float) -> float:
    return 4.0 / 3.0 * math.pi * radius**3


def sphere_area(radius: float) -> float:
    return 4.0 * math.pi * radius**2


def torus_volume(major: float, minor: float) -> float:
    return 2.0 * math.pi**2 * major * minor**2


def torus_area(major: float, minor: float) -> float:
    return 4.0 * math.pi**2 * major * minor


def cylinder_volume(radius: float, height: float) -> float:
    return math.pi * radius**2 * height
