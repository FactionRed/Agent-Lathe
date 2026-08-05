"""
Deliberately broken assets, one per failure mode.

DESIGN.md section 12, Phase 2 demo: "an agent is given a deliberately broken
asset and repairs every violation from the report alone, with no human hints."
This module is that asset set. Each builder documents which codes it must
trigger, and `tests/test_rules.py` asserts the exact set -- a fixture that
trips a second rule by accident is a bug in the fixture or a false positive in
the rule, and both matter.

Two builders are load-bearing beyond their own rule:

  * `clean_prop()` must produce ZERO findings. A verifier with false positives
    is worse than no verifier, because the agent learns to ignore the report.
  * `severed_crate()` is DESIGN.md's own section 4 example with the cutter span
    it originally shipped. prototype/README.md measured it as six loose slabs.
    It is the reason E108 exists, so it is a real regression test, not a toy.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .mesh import (
    ArticulationSpec,
    Asset,
    AssetMesh,
    BakeSpec,
    EmitSpec,
    JointDecl,
    ReachDecl,
    SymmetryDecl,
    TextureDecl,
    Transform,
)

# The six sides of a box, each as four corners in counter-clockwise order seen
# from outside, expressed as (x, y, z) selectors over (x0, x1, y0, y1, z0, z1).
_SIDES: Tuple[Tuple[str, Tuple[Tuple[int, int, int], ...]], ...] = (
    ("side_L", ((0, 1, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1))),  # -X
    ("side_R", ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),  # +X
    ("front", ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),  # -Y
    ("back", ((1, 1, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1))),  # +Y
    ("top", ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),  # +Z
    ("bottom", ((0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0))),  # -Z
)

# Where each side's island sits in a 3 x 2 uv grid.
_UV_CELLS = {
    "side_L": (0, 0),
    "side_R": (1, 0),
    "front": (2, 0),
    "back": (0, 1),
    "top": (1, 1),
    "bottom": (2, 1),
}

DEFAULT_SIZE = 0.6  # m -- the section 4 crate
DEFAULT_ISLAND = 0.3  # uv side; with 0.6 m faces at 1024 this is exactly 5.12 px/cm
TEXEL_TARGET = 5.12  # px/cm, from the section 4 emit block
RESOLUTION = 1024


# ---------------------------------------------------------------- geometry


def box_mesh(
    name: str,
    boxes: Optional[Sequence[Tuple[float, float, float, float, float, float]]] = None,
    size: float = DEFAULT_SIZE,
    origin: str = "bottom_center",
    subdivisions: int = 1,
    with_uv: bool = True,
    island: float = DEFAULT_ISLAND,
    transform: Transform = Transform(),
) -> AssetMesh:
    """A welded, closed, outward-wound box (or several boxes welded together).

    Vertices are shared across sides, so the result is watertight; UVs are per
    corner, so the six sides are still six islands. That combination is what a
    correct game prop looks like, and it is why the clean fixture can be clean.
    """
    if boxes is None:
        half = size / 2.0
        z0 = 0.0 if origin == "bottom_center" else -half
        boxes = [(-half, half, -half, half, z0, z0 + size)]

    verts: List[Tuple[float, float, float]] = []
    index: Dict[Tuple[float, float, float], int] = {}
    faces: List[Tuple[int, ...]] = []
    uvs: List[Tuple[float, float]] = []
    groups: Dict[str, List[int]] = {}

    def vert(p) -> int:
        key = (round(p[0], 9), round(p[1], 9), round(p[2], 9))
        if key not in index:
            index[key] = len(verts)
            verts.append((float(p[0]), float(p[1]), float(p[2])))
        return index[key]

    n = max(1, int(subdivisions))
    for bi, (x0, x1, y0, y1, z0, z1) in enumerate(boxes):
        axes = ((x0, x1), (y0, y1), (z0, z1))
        for side, corners in _SIDES:
            p = [np.array([axes[a][c[a]] for a in range(3)], dtype=np.float64)
                 for c in corners]
            du, dv = p[1] - p[0], p[3] - p[0]
            gname = side if len(boxes) == 1 else "%s_%02d" % (side, bi)
            cell = _UV_CELLS[side]
            for i in range(n):
                for j in range(n):
                    u0, u1 = i / n, (i + 1) / n
                    v0, v1 = j / n, (j + 1) / n
                    quad = [
                        p[0] + du * u0 + dv * v0,
                        p[0] + du * u1 + dv * v0,
                        p[0] + du * u1 + dv * v1,
                        p[0] + du * u0 + dv * v1,
                    ]
                    faces.append(tuple(vert(q) for q in quad))
                    groups.setdefault(gname, []).append(len(faces) - 1)
                    if with_uv:
                        ox = cell[0] / 3.0 + (1.0 / 3.0 - island) / 2.0
                        oy = cell[1] / 2.0 + (1.0 / 2.0 - island) / 2.0
                        for uu, vv in ((u0, v0), (u1, v0), (u1, v1), (u0, v1)):
                            uvs.append((ox + uu * island, oy + vv * island))

    return AssetMesh(
        name=name,
        verts=np.array(verts, dtype=np.float64),
        faces=tuple(faces),
        transform=transform,
        uv_sets={"UV0": np.array(uvs, dtype=np.float64)} if with_uv else {},
        face_groups={k: tuple(v) for k, v in groups.items()},
    )


# -------------------------------------------------------------------- specs


def prop_spec(**kw) -> EmitSpec:
    """The section 4 emit block, as a declaration the verifier can check."""
    base = dict(
        profile="game_prop_pbr",
        target="unreal",
        lod=0,
        budget={"tris_lod0": 1500},
        components=1,
        origin="bottom_center",
        texel_density=TEXEL_TARGET,
        texture_resolution=RESOLUTION,
        textures=(
            TextureDecl("T_Crate_Wood_D", RESOLUTION, RESOLUTION, "D"),
            TextureDecl("T_Crate_Wood_N", RESOLUTION, RESOLUTION, "N"),
        ),
        bake=BakeSpec(targets=("normal",), uv_set="UV0", unique_uv=True),
    )
    base.update(kw)
    return EmitSpec(**base)


def solid_spec(**kw) -> EmitSpec:
    """A declaration with no UV or texture promises, for the topology fixtures.

    Deliberate: a rule with nothing to check must report *not checked* rather
    than pass, and the topology fixtures are how that is tested.
    """
    base = dict(
        profile="game_prop_pbr",
        target="unreal",
        budget={"tris_lod0": 1500},
        components=1,
        origin="bottom_center",
        texel_density=None,
        textures=(),
        bake=None,
    )
    base.update(kw)
    return EmitSpec(**base)


# ------------------------------------------------------------------- clean


def clean_prop(subdivisions: int = 1, name: str = "SM_Crate_Wood_LOD0") -> Asset:
    """MUST produce zero findings. Watertight, quads only, outward normals,
    identity transform, origin at bottom_center, six non-overlapping islands at
    exactly 5.12 px/cm, power-of-two maps, conventional name."""
    return Asset(box_mesh(name, subdivisions=subdivisions), prop_spec())


def clean_solid(name: str = "SM_Crate_Wood_LOD0", **spec_kw) -> Asset:
    """MUST produce zero findings and four *not checked* entries (E106, W202,
    W302 have no UVs; W203 has no textures)."""
    return Asset(box_mesh(name, with_uv=False), solid_spec(**spec_kw))


# ------------------------------------------------------- E101 non-manifold


def nonmanifold_edge() -> Asset:
    """E101 only. Two boxes welded along one vertical edge: that edge carries
    four faces. Everything else stays closed, so this isolates the sub-rule."""
    s = DEFAULT_SIZE
    mesh = box_mesh(
        "SM_Crate_Wood_LOD0",
        boxes=[(0.0, s, 0.0, s, 0.0, s), (s, 2 * s, s, 2 * s, 0.0, s)],
        with_uv=False,
    )
    return Asset(mesh, solid_spec(origin=None))


def bowtie_vertex() -> Asset:
    """E101 only. Two boxes welded at a single corner vertex: every edge is
    manifold, one vertex joins two fans."""
    s = DEFAULT_SIZE
    mesh = box_mesh(
        "SM_Crate_Wood_LOD0",
        boxes=[(0.0, s, 0.0, s, 0.0, s), (s, 2 * s, s, 2 * s, s, 2 * s)],
        with_uv=False,
    )
    return Asset(mesh, solid_spec(origin=None))


def open_shell() -> Asset:
    """E101 only. A box with the top face deleted -- the classic 'boolean left a
    hole' shape."""
    mesh = box_mesh("SM_Crate_Wood_LOD0", with_uv=False)
    top = mesh.face_groups["top"][0]
    return Asset(drop_face(mesh, top), solid_spec())


# --------------------------------------------------------------- E102 n-gons


def ngon_prism(sides: int = 6, height: float = 0.6, radius: float = 0.3) -> Asset:
    """E102 only. A closed prism with two n-gon caps and quad walls: manifold,
    consistently wound, origin at bottom_center."""
    verts: List[Tuple[float, float, float]] = []
    for z in (0.0, height):
        for i in range(sides):
            a = 2.0 * math.pi * i / sides
            verts.append((radius * math.cos(a), radius * math.sin(a), z))
    faces: List[Tuple[int, ...]] = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.append((i, j, sides + j, sides + i))  # outward wall
    faces.append(tuple(range(sides - 1, -1, -1)))  # bottom cap, wound -Z
    faces.append(tuple(range(sides, 2 * sides)))  # top cap, wound +Z
    groups = {
        "wall": tuple(range(sides)),
        "cap_bottom": (sides,),
        "cap_top": (sides + 1,),
    }
    mesh = AssetMesh(
        name="SM_Column_Stone_LOD0",
        verts=np.array(verts, dtype=np.float64),
        faces=tuple(faces),
        face_groups=groups,
    )
    return Asset(mesh, solid_spec())


# -------------------------------------------------------------- E103 normals


def flipped_face() -> Asset:
    """E103 only, inconsistent branch: one face wound the wrong way."""
    mesh = box_mesh("SM_Crate_Wood_LOD0", with_uv=False)
    fi = mesh.face_groups["top"][0]
    faces = list(mesh.faces)
    faces[fi] = tuple(reversed(faces[fi]))
    return Asset(mesh.evolve(faces=tuple(faces)), solid_spec())


def inside_out() -> Asset:
    """E103 only, inverted branch: the whole shell wound inward. Consistent, so
    only the signed volume can see it."""
    mesh = box_mesh("SM_Crate_Wood_LOD0", with_uv=False)
    faces = tuple(tuple(reversed(f)) for f in mesh.faces)
    return Asset(mesh.evolve(faces=faces), solid_spec())


# ------------------------------------------------------------ E104 transform


def unapplied_transform() -> Asset:
    """E104 only. The section 1 complaint, verbatim: "the object is 100x too
    large" -- here 100x too small, plus a 45 degree yaw."""
    mesh = box_mesh(
        "SM_Crate_Wood_LOD0",
        with_uv=False,
        transform=Transform(scale=(0.01, 0.01, 0.01), rotation_euler=(0.0, 0.0, math.pi / 4)),
    )
    return Asset(mesh, solid_spec())


# ---------------------------------------------------------------- E105 loose


def loose_geometry() -> Asset:
    """E105 only. One orphan vertex and one wire edge, both of which survive an
    FBX round trip and neither of which is visible in a viewport.

    The strays sit inside the box's x/y footprint on purpose. Put them off to
    one side and they drag the bounding box with them, which moves
    `bottom_center` and legitimately trips W201 as well -- see
    `zero_area_face`, which keeps that cascade deliberately.
    """
    mesh = box_mesh("SM_Crate_Wood_LOD0", with_uv=False)
    extra = np.array([[0.0, 0.0, 0.9], [0.1, 0.0, 0.9], [0.1, 0.1, 0.9]], dtype=np.float64)
    base = mesh.num_verts
    return Asset(
        mesh.evolve(
            verts=np.vstack([mesh.verts, extra]),
            wire_edges=((base + 1, base + 2),),
        ),
        solid_spec(),
    )


def zero_area_face() -> Asset:
    """E105 plus E101 plus W201, and all three are true.

    A collinear triangle off to one side is degenerate (E105), opens three
    boundary edges (E101), and drags the bounding box far enough that the
    declared `bottom_center` origin is 450 mm off (W201). Kept as a cascade on
    purpose: one defect can produce several correct findings, and the report has
    to stay legible when it does. A fixture that quietly trips extra rules is
    how false positives get normalised, so the expectation is written down.
    """
    mesh = box_mesh("SM_Crate_Wood_LOD0", with_uv=False)
    extra = np.array([[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=np.float64)
    base = mesh.num_verts
    faces = mesh.faces + ((base, base + 1, base + 2),)
    return Asset(
        mesh.evolve(verts=np.vstack([mesh.verts, extra]), faces=faces),
        solid_spec(components=2),
    )


# ------------------------------------------------------------------ E106 uv


def overlapping_uvs() -> Asset:
    """E106 only. side_R is mirrored in U and moved exactly on top of side_L --
    the mirrored-unwrap case from the section 7 sample report, where the two
    halves of a symmetrical asset were unwrapped onto one island."""
    mesh = box_mesh("SM_Crate_Wood_LOD0")
    delta = (
        (_UV_CELLS["side_L"][0] - _UV_CELLS["side_R"][0]) / 3.0,
        (_UV_CELLS["side_L"][1] - _UV_CELLS["side_R"][1]) / 2.0,
    )
    return Asset(shift_island(mirror_island(mesh, "side_R"), "side_R", delta), prop_spec())


def tight_padding() -> Asset:
    """W202 only. Islands grown to 0.331 uv, leaving 2.4 px between them at
    1024 where the table wants 4. texel_density is left undeclared so W302
    reports *not checked* instead of firing on the same defect."""
    mesh = box_mesh("SM_Crate_Wood_LOD0", island=0.331)
    return Asset(mesh, prop_spec(texel_density=None))


def collapsed_uv() -> Asset:
    """W206 only. The `top` island is scaled to effectively nothing, so real
    surface samples a single texel.

    This case has an owner because testing the unwrapper found it did not: an
    island with no uv area does not *overlap* anything, so E106 correctly stayed
    silent and no other rule spoke. texel_density is left undeclared so W302 does
    not fire on the same defect.
    """
    mesh = box_mesh("SM_Crate_Wood_LOD0")
    return Asset(scale_island(mesh, "top", 1e-7), prop_spec(texel_density=None))


def texel_variance() -> Asset:
    """W302 only. The top island is scaled to 0.8, so it sits at 4.1 px/cm
    against a 5.12 target -- outside the +/-10% band, inside the padding table."""
    mesh = box_mesh("SM_Crate_Wood_LOD0")
    return Asset(scale_island(mesh, "top", 0.8), prop_spec())


# -------------------------------------------------------------- E107 budget


def over_budget(subdivisions: int = 13) -> Asset:
    """E107 only. 2,028 triangles against the declared 1,500, with the whole
    overage spread evenly across six named groups."""
    return Asset(box_mesh("SM_Crate_Wood_LOD0", subdivisions=subdivisions), prop_spec())


# ---------------------------------------------- E108 the doc's own example


def _crate(cutter_z: float, slat_n: int = 5, size: float = DEFAULT_SIZE,
           slat_w: float = 0.08, name: str = "SM_Crate_Wood_LOD0"):
    """DESIGN.md section 4's crate, built on the kernel section 11 selects."""
    import manifold3d as m

    body = m.Manifold.cube((size, size, size), True)
    span = size * 0.9
    for i in range(slat_n):
        t = 0.5 if slat_n == 1 else i / (slat_n - 1.0)
        x = -span / 2 + t * span
        body -= m.Manifold.cube((slat_w * 0.35, size * 1.1, cutter_z), True).translate(
            (x, 0.0, 0.0)
        )
    from .mesh import from_manifold

    return from_manifold(body, name)


def severed_crate() -> Asset:
    """E108 only. The section 4 crate with the cutter span the doc originally
    shipped (`size * 1.1` in Z), which severs the body into six slabs.
    prototype/README.md: "Six loose slabs, not a crate. Nothing in the program
    said otherwise, which is the whole argument for section 7."
    """
    mesh = _crate(cutter_z=DEFAULT_SIZE * 1.1)
    return Asset(mesh, solid_spec(components=1, origin="center"))


def fixed_crate() -> Asset:
    """MUST produce zero findings. The same program with the fix the prototype
    landed: the cutter stops short of the rails, so the body stays one solid."""
    rail = 0.06
    mesh = _crate(cutter_z=DEFAULT_SIZE - rail * 2)
    return Asset(mesh, solid_spec(components=1, origin="center"))


# ------------------------------------------------------- warnings and notes


def origin_offset() -> Asset:
    """W201 only. Geometry centred on the origin while emit declares
    bottom_center -- 300 mm out, which is a floating crate in the level."""
    mesh = box_mesh("SM_Crate_Wood_LOD0", origin="center", with_uv=False)
    return Asset(mesh, solid_spec(origin="bottom_center"))


def npot_textures() -> Asset:
    """W203 only. One 1000 x 1024 map among power-of-two maps."""
    spec = prop_spec(
        textures=(
            TextureDecl("T_Crate_Wood_D", 1000, RESOLUTION, "D"),
            TextureDecl("T_Crate_Wood_N", RESOLUTION, RESOLUTION, "N"),
        )
    )
    return Asset(box_mesh("SM_Crate_Wood_LOD0"), spec)


def bad_name() -> Asset:
    """N401 only. `crate_final` -- no prefix, no variant, no LOD, lowercase."""
    return Asset(box_mesh("crate_final"), prop_spec())


# ------------------------------------------------------------ the kitchen sink


def multi_fault() -> Asset:
    """Several faults at once, which is what a real agent hand-off looks like:
    an n-gon lid, a flipped face, an unapplied transform, an over-budget mesh
    and a bad name. Used to prove the report stays readable and ordered."""
    mesh = box_mesh("Cube.001", subdivisions=13, with_uv=False)
    faces = list(mesh.faces)
    fi = mesh.face_groups["top"][0]
    faces[fi] = tuple(reversed(faces[fi]))
    merged = faces[mesh.face_groups["bottom"][0]] + faces[mesh.face_groups["bottom"][1]][:2]
    faces.append(merged)  # a 6-gon glued onto the bottom
    groups = dict(mesh.face_groups)
    groups["lid"] = (len(faces) - 1,)
    mesh = mesh.evolve(
        faces=tuple(faces),
        face_groups=groups,
        transform=Transform(scale=(2.0, 2.0, 2.0)),
    )
    return Asset(mesh, solid_spec(budget={"tris_lod0": 500}))


# ------------------------------------------------------------------- helpers


def drop_face(mesh: AssetMesh, fi: int) -> AssetMesh:
    """Remove one face, keeping uv loops and face groups consistent."""
    starts = mesh.loop_starts
    n = len(mesh.faces[fi])
    keep = [i for i in range(mesh.num_loops) if not (starts[fi] <= i < starts[fi] + n)]
    faces = tuple(f for i, f in enumerate(mesh.faces) if i != fi)
    groups = {
        k: tuple((i if i < fi else i - 1) for i in v if i != fi)
        for k, v in mesh.face_groups.items()
    }
    groups = {k: v for k, v in groups.items() if v}
    uv_sets = {k: uv[keep] for k, uv in mesh.uv_sets.items()}
    return replace(mesh, faces=faces, face_groups=groups, uv_sets=uv_sets)


def _island_loops(mesh: AssetMesh, group: str) -> List[int]:
    starts = mesh.loop_starts
    out: List[int] = []
    for fi in mesh.face_groups[group]:
        out.extend(range(starts[fi], starts[fi] + len(mesh.faces[fi])))
    return out


def shift_island(mesh: AssetMesh, group: str, delta, uv_set: str = "UV0") -> AssetMesh:
    uv = mesh.uv_sets[uv_set].copy()
    loops = _island_loops(mesh, group)
    uv[loops] += np.asarray(delta, dtype=np.float64)
    sets = dict(mesh.uv_sets)
    sets[uv_set] = uv
    return replace(mesh, uv_sets=sets)


def mirror_island(mesh: AssetMesh, group: str, axis: int = 0, uv_set: str = "UV0") -> AssetMesh:
    """Flip an island about its own centre, which reverses its uv winding. This
    is what a mirrored unwrap produces, and what E106 keys its repair on."""
    uv = mesh.uv_sets[uv_set].copy()
    loops = _island_loops(mesh, group)
    centre = uv[loops].mean(axis=0)
    uv[loops, axis] = 2.0 * centre[axis] - uv[loops, axis]
    sets = dict(mesh.uv_sets)
    sets[uv_set] = uv
    return replace(mesh, uv_sets=sets)


def scale_island(mesh: AssetMesh, group: str, factor: float, uv_set: str = "UV0") -> AssetMesh:
    uv = mesh.uv_sets[uv_set].copy()
    loops = _island_loops(mesh, group)
    centre = uv[loops].mean(axis=0)
    uv[loops] = centre + (uv[loops] - centre) * factor
    sets = dict(mesh.uv_sets)
    sets[uv_set] = uv
    return replace(mesh, uv_sets=sets)


# What each fixture must produce. The test suite reads this table, so the
# expectation lives next to the geometry that is supposed to violate it.
# ------------------------------------------------------- E501-W505 rigs
#
# Deliberately not a humanoid among them. A rule that quietly assumed two arms,
# one root, or a mirror plane would pass every test written around a person and
# fail the first time anyone rigged a starfish, so the fixtures are a radial
# echinoderm, a serpent, a hexapod and a lamp arm.


def _hinge(name, parent=None, offset=(0.0, 0.0, 0.0), axis=(0.0, 1.0, 0.0),
           limit=(-math.pi / 2, math.pi / 2), angle=0.0) -> JointDecl:
    return JointDecl(name=name, parent=parent, offset=offset, axis=axis,
                     limit=limit, angle=angle)


def _radial_arms(fold: int = 5, reach: float = 0.25, angle: float = 0.0):
    """A starfish: `fold` identical arms about +Z, no mirror plane anywhere.

    Bilateral symmetry is undefined for an odd fold -- reflecting a five-armed
    disc lands every arm between two others -- which is exactly why the plan has
    to be declared rather than guessed.
    """
    joints = [_hinge("disc", axis=(0.0, 0.0, 1.0), limit=(0.0, 0.0))]
    for i in range(fold):
        a = 2.0 * math.pi * i / fold
        joints.append(_hinge("arm_%d" % i, parent="disc",
                             offset=(reach * math.cos(a), reach * math.sin(a), 0.0),
                             axis=(-math.sin(a), math.cos(a), 0.0),
                             limit=(-math.pi / 4, math.pi / 4), angle=angle))
    return tuple(joints)


def starfish_rig(fold: int = 5, **kw) -> Asset:
    """Clean. Five-fold radial, declared as such, and it holds."""
    art = ArticulationSpec(
        joints=_radial_arms(fold),
        symmetry=SymmetryDecl(kind="radial", axis=(0.0, 0.0, 1.0), fold=fold),
        roots=1, **kw)
    return Asset(box_mesh("SK_Starfish_Coral_LOD0", with_uv=False),
                 solid_spec(profile="character_skinned", articulation=art))


def starfish_broken_arm() -> Asset:
    """W505 only. One arm nudged 30 mm off its radial station.

    A bilateral check would have said nothing useful about this rig at all.
    """
    joints = list(_radial_arms(5))
    j = joints[3]
    joints[3] = replace(j, offset=(j.offset[0] + 0.03, j.offset[1], j.offset[2]))
    art = ArticulationSpec(
        joints=tuple(joints),
        symmetry=SymmetryDecl(kind="radial", axis=(0.0, 0.0, 1.0), fold=5),
        roots=1)
    return Asset(box_mesh("SK_Starfish_Coral_LOD0", with_uv=False),
                 solid_spec(profile="character_skinned", articulation=art))


def serpent_rig(n: int = 24, angle: float = 0.0) -> Asset:
    """A long unbranched chain: no limbs, no symmetry worth checking.

    Declares `none`, which must produce a *not checked* line rather than a
    warning it could never satisfy. Also the depth case -- nothing may assume a
    chain is short.
    """
    joints = [_hinge("spine_00", axis=(0.0, 0.0, 1.0), limit=(0.0, 0.0))]
    for i in range(1, n):
        joints.append(_hinge("spine_%02d" % i, parent="spine_%02d" % (i - 1),
                             offset=(0.0, 0.0, -0.04), axis=(0.0, 0.0, 1.0),
                             limit=(-math.pi / 6, math.pi / 6), angle=angle))
    art = ArticulationSpec(joints=tuple(joints),
                           symmetry=SymmetryDecl(kind="none"), roots=1)
    return Asset(box_mesh("SK_Serpent_Scaled_LOD0", with_uv=False),
                 solid_spec(profile="character_skinned", articulation=art))


def hexapod_rig(legs: int = 6, angle: float = 0.0) -> Asset:
    """Bilateral, but with six limbs rather than four.

    The count is never checked against anything, which is the point: a rule that
    knew how many legs a body has would be wrong here and wrong again for an
    octopus.
    """
    joints = [_hinge("thorax", axis=(0.0, 0.0, 1.0), limit=(0.0, 0.0))]
    for i in range(legs // 2):
        y = 0.06 - 0.06 * i
        for side, x in (("L", 0.09), ("R", -0.09)):
            joints.append(_hinge("leg_%s%d" % (side, i), parent="thorax",
                                 offset=(x, y, 0.0), axis=(0.0, 1.0, 0.0),
                                 limit=(-math.pi / 3, math.pi / 3), angle=angle))
    art = ArticulationSpec(
        joints=tuple(joints),
        symmetry=SymmetryDecl(kind="bilateral", normal=(1.0, 0.0, 0.0)),
        roots=1)
    return Asset(box_mesh("SK_Hexapod_Chitin_LOD0", with_uv=False),
                 solid_spec(profile="character_skinned", articulation=art))


def joint_over_limit() -> Asset:
    """E501 only. The elbow-equivalent driven 20 degrees past its upper bound,
    the way a retarget or a hand-typed pose does it."""
    joints = list(_radial_arms(5))
    joints[2] = replace(joints[2], angle=math.pi / 4 + math.radians(20))
    art = ArticulationSpec(joints=tuple(joints),
                           symmetry=SymmetryDecl(kind="radial",
                                                 axis=(0.0, 0.0, 1.0), fold=5),
                           roots=1)
    return Asset(box_mesh("SK_Starfish_Coral_LOD0", with_uv=False),
                 solid_spec(profile="character_skinned", articulation=art))


def joint_pinned_at_limit() -> Asset:
    """W503 only. Two arms resting exactly on their bound, which is what an IK
    solve that ran out of range leaves behind."""
    joints = list(_radial_arms(5))
    joints[1] = replace(joints[1], angle=math.pi / 4)
    joints[4] = replace(joints[4], angle=-math.pi / 4)
    art = ArticulationSpec(joints=tuple(joints),
                           symmetry=SymmetryDecl(kind="radial",
                                                 axis=(0.0, 0.0, 1.0), fold=5),
                           roots=1)
    return Asset(box_mesh("SK_Starfish_Coral_LOD0", with_uv=False),
                 solid_spec(profile="character_skinned", articulation=art))


def severed_limb() -> Asset:
    """W504 only. One arm lost its parent and became a second root.

    The silent failure this exists for: a rig that looks identical at rest and
    moves independently of the body the moment anything is posed.
    """
    joints = list(_radial_arms(5))
    joints[2] = replace(joints[2], parent=None)
    art = ArticulationSpec(joints=tuple(joints),
                           symmetry=SymmetryDecl(kind="radial",
                                                 axis=(0.0, 0.0, 1.0), fold=5),
                           roots=1)
    return Asset(box_mesh("SK_Starfish_Coral_LOD0", with_uv=False),
                 solid_spec(profile="character_skinned", articulation=art))


def reach_missed() -> Asset:
    """E502 only. A declared target the emitted pose does not arrive at."""
    art = ArticulationSpec(
        joints=_radial_arms(5),
        symmetry=SymmetryDecl(kind="radial", axis=(0.0, 0.0, 1.0), fold=5),
        roots=1,
        reach=(ReachDecl(tip="arm_0", target=(0.40, 0.0, 0.0), tolerance=0.002),))
    return Asset(box_mesh("SK_Starfish_Coral_LOD0", with_uv=False),
                 solid_spec(profile="character_skinned", articulation=art))


EXPECTED: Dict[str, Tuple[str, ...]] = {
    "starfish_rig": (),
    "serpent_rig": (),
    "hexapod_rig": (),
    "starfish_broken_arm": ("W505",),
    "joint_over_limit": ("E501",),
    "joint_pinned_at_limit": ("W503",),
    "severed_limb": ("W504",),
    "reach_missed": ("E502",),
    "clean_prop": (),
    "clean_solid": (),
    "fixed_crate": (),
    "nonmanifold_edge": ("E101",),
    "bowtie_vertex": ("E101",),
    "open_shell": ("E101",),
    "ngon_prism": ("E102",),
    "flipped_face": ("E103",),
    "inside_out": ("E103",),
    "unapplied_transform": ("E104",),
    "loose_geometry": ("E105",),
    # A degenerate face off to one side is three true findings, not one: it is
    # degenerate, it is an open boundary, and it moved the declared origin.
    "zero_area_face": ("E101", "E105", "W201"),
    "overlapping_uvs": ("E106",),
    "over_budget": ("E107",),
    "severed_crate": ("E108",),
    "origin_offset": ("W201",),
    "tight_padding": ("W202",),
    "npot_textures": ("W203",),
    "texel_variance": ("W302",),
    "collapsed_uv": ("W206",),
    "bad_name": ("N401",),
}
