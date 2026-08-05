"""
Lathe DESIGN.md section 7 -- the neutral asset representation the verifier reads.

DESIGN DECISION: rules run on this structure, never on `bpy` data.

DESIGN.md section 11 states that "Blender is a backend and a viewer, not the
foundation", and section 7 wants the verifier to be the cheapest and most
portable subsystem. Three consequences settle the question:

  * The emit path is Manifold-backed (section 11), so the verifier must be able
    to read a triangle soup from a kernel that has never heard of Blender.
  * `bmesh` is reachable only from inside a Blender process. Binding the rule
    engine to it would make `check()` cost an interpreter start, and would put
    the highest-leverage subsystem behind the exact hidden global state that
    section 1 indicts.
  * A rule that reads plain arrays is testable with hand-built broken fixtures.
    A rule that reads `bmesh` needs Blender to run its own unit test.

So adapters convert (Manifold mesh | bpy Mesh | glTF | FBX) into `AssetMesh`,
and every rule is a pure function of `AssetMesh` + `EmitSpec`. `from_manifold`
lives here because the kernel is not optional. A bpy adapter is a short
function that belongs in the Blender backend, not in the engine.

Faces are variable-length index tuples, not triangles. That is deliberate:
E102 (n-gons) cannot exist as a rule if the representation cannot hold an
n-gon. The verifier must see what the exporter would see.

Derived data lives next door: `topology.py` for connectivity, `uvmap.py` for
UV islands, `geom.py` for the coordinate maths. The split keeps this file about
what an asset *is*, not what can be computed from it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .geom import EdgeKey, edge_key, newell_normal, polygon_area_3d

DEG = math.pi / 180.0

Vec3 = Tuple[float, float, float]
Face = Tuple[int, ...]


# --------------------------------------------------------------- declarations


@dataclass(frozen=True)
class Transform:
    """Object-level transform, exactly as an exporter would write it.

    Kept separate from the vertex array on purpose: E104 exists because the
    transform and the geometry disagree about what "1 unit" means, and you
    cannot see that disagreement once the transform is baked in.
    """

    translation: Vec3 = (0.0, 0.0, 0.0)
    rotation_euler: Vec3 = (0.0, 0.0, 0.0)  # radians, XYZ order
    scale: Vec3 = (1.0, 1.0, 1.0)

    def matrix3(self) -> np.ndarray:
        """Rotation * scale, without translation."""
        rx, ry, rz = self.rotation_euler
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
        my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
        mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
        return (mz @ my @ mx) @ np.diag(np.asarray(self.scale, dtype=np.float64))

    def is_identity_rotation(self, tol: float = 1e-6) -> bool:
        return all(abs(a) <= tol for a in self.rotation_euler)

    def is_unit_scale(self, tol: float = 1e-6) -> bool:
        return all(abs(s - 1.0) <= tol for s in self.scale)


@dataclass(frozen=True)
class TextureDecl:
    """A texture the emit spec promises to write."""

    name: str
    width: int
    height: int
    channel: str = "D"  # D, N, ORM, ...


@dataclass(frozen=True)
class BakeSpec:
    """What `emit` says about baking. Drives E106 applicability."""

    targets: Tuple[str, ...] = ()  # ("normal", "ao")
    uv_set: str = "UV0"
    unique_uv: bool = True  # a unique (non-tiling, non-overlapping) map


@dataclass(frozen=True)
class JointDecl:
    """One hinge, as the emitted asset declares it.

    Plain data, like every other declaration here -- deliberately not a
    `pose.Joint`. `EmitSpec` is the contract an asset carries, and it holds
    numbers rather than kernel objects so the verifier can grade an asset that
    arrived from somewhere else entirely. Angles are radians, matching the rest
    of the system.
    """

    name: str
    limit: Tuple[float, float] = (-2.0 * math.pi, 2.0 * math.pi)
    angle: float = 0.0                       # the pose this asset was emitted in
    parent: Optional[str] = None
    offset: Vec3 = (0.0, 0.0, 0.0)           # rest translation, in the parent frame
    axis: Vec3 = (0.0, 1.0, 0.0)


@dataclass(frozen=True)
class ReachDecl:
    """`assert reach(ankle, target) < 2mm`, carried forward to emit.

    The IK counterpart of a triangle budget: the program stated where a named
    tip was supposed to end up, so the gate can check that it did.
    """

    tip: str
    target: Vec3
    tolerance: float = 0.002                 # metres


@dataclass(frozen=True)
class SymmetryDecl:
    """The body plan, declared rather than assumed.

    **This exists because "symmetry" silently means "bilateral" to almost every
    rig tool, and that is a statement about vertebrates, not about geometry.** A
    starfish is five-fold radial, a jellyfish eight-fold, a spider bilateral
    with eight limbs rather than four, a snake has no limb symmetry worth
    checking at all, and a lamp arm is not symmetric in any direction. Measuring
    any of those against a mirror plane through the origin produces a confident
    wrong answer, which is the one output this subsystem must never emit.

    So the program says which plan it built, and the rule checks *that*:

        `bilateral` — reflect across the plane {p . normal == offset}
        `radial`    — rotate by 360/fold about `axis` through `center`
        `none`      — checked as not applicable, and nothing is inferred

    `none` is a real answer, not an opt-out. An asymmetric mechanism declaring
    it gets a *not checked* line instead of a warning it can never satisfy.
    """

    kind: str = "none"                       # bilateral | radial | none
    axis: Vec3 = (0.0, 0.0, 1.0)             # radial: the rotation axis
    normal: Vec3 = (1.0, 0.0, 0.0)           # bilateral: the mirror plane normal
    offset: float = 0.0                      # plane offset along `normal`, metres
    center: Vec3 = (0.0, 0.0, 0.0)           # radial: a point on the axis
    fold: int = 2                            # radial: how many repeats
    tolerance: float = 1e-4                  # metres


@dataclass(frozen=True)
class ArticulationSpec:
    """What `emit` declares about a posed asset: the rig, and the pose used."""

    joints: Tuple[JointDecl, ...] = ()
    reach: Tuple[ReachDecl, ...] = ()
    symmetry: Optional[SymmetryDecl] = None
    # How many joints have no parent. One for most creatures, but a rig can
    # legitimately hold several -- a floating orb, a detached jaw, a swarm --
    # so this is declared and compared, never assumed to be 1.
    roots: Optional[int] = None

    def pose(self) -> Dict[str, float]:
        return {j.name: float(j.angle) for j in self.joints}

    def joint(self, name: str) -> Optional[JointDecl]:
        for j in self.joints:
            if j.name == name:
                return j
        return None


@dataclass(frozen=True)
class EmitSpec:
    """The declared expectation -- the other half of every assertion.

    Section 7 rules are comparisons between what the program declared and what
    the geometry actually is. With no declaration there is no rule, only taste,
    which is the thing this subsystem exists to avoid. Rules that need a
    declaration and do not find one report themselves as *not checked* rather
    than passing silently.
    """

    profile: str = "game_prop_pbr"
    target: Optional[str] = None  # "unreal", "unity", ...
    lod: int = 0
    budget: Mapping[str, int] = field(default_factory=dict)  # {"tris_lod0": 1500}
    components: Optional[int] = None  # `assert components(x) == n`
    origin: Optional[str] = None  # "bottom_center", "center", ...
    texel_density: Optional[float] = None  # px/cm
    texture_resolution: int = 1024  # the map the density targets
    textures: Tuple[TextureDecl, ...] = ()
    bake: Optional[BakeSpec] = None
    unit_scale: float = 1.0  # metres per mesh unit
    articulation: Optional[ArticulationSpec] = None

    def budget_for_lod(self) -> Optional[int]:
        key = "tris_lod%d" % self.lod
        if key in self.budget:
            return int(self.budget[key])
        if "tris" in self.budget:
            return int(self.budget["tris"])
        return None


# ------------------------------------------------------------------- the mesh


@dataclass(frozen=True)
class AssetMesh:
    """A polygon mesh plus the names the report needs to be actionable.

    `face_groups` is how section 7's "offending named entity" survives into the
    report. It is the verifier-side shadow of section 5.2 provenance: whatever
    the kernel calls a logical face (Manifold `face_id`, a bpy attribute, a
    Lathe tag) is flattened to a name here, so a finding can say `side_L`
    instead of `face 4471`.
    """

    name: str
    verts: np.ndarray  # (V, 3) float64, object-local space
    faces: Tuple[Face, ...] = ()
    wire_edges: Tuple[EdgeKey, ...] = ()  # edges belonging to no face
    transform: Transform = Transform()
    uv_sets: Mapping[str, np.ndarray] = field(default_factory=dict)  # name -> (L, 2)
    face_groups: Mapping[str, Tuple[int, ...]] = field(default_factory=dict)

    # -- basics ------------------------------------------------------------

    def __post_init__(self) -> None:
        v = np.asarray(self.verts, dtype=np.float64)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError("verts must be (V, 3), got %s" % (v.shape,))
        object.__setattr__(self, "verts", v)
        object.__setattr__(self, "faces", tuple(tuple(int(i) for i in f) for f in self.faces))
        object.__setattr__(
            self, "wire_edges", tuple((int(a), int(b)) for a, b in self.wire_edges)
        )
        object.__setattr__(
            self,
            "uv_sets",
            {k: np.asarray(v2, dtype=np.float64) for k, v2 in dict(self.uv_sets).items()},
        )
        object.__setattr__(
            self,
            "face_groups",
            {k: tuple(int(i) for i in v2) for k, v2 in dict(self.face_groups).items()},
        )
        n_loops = sum(len(f) for f in self.faces)
        for k, uv in self.uv_sets.items():
            if uv.shape != (n_loops, 2):
                raise ValueError(
                    "uv set %r must be (%d, 2) -- one entry per face corner, got %s"
                    % (k, n_loops, uv.shape)
                )

    @property
    def num_verts(self) -> int:
        return int(self.verts.shape[0])

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    @property
    def loop_starts(self) -> Tuple[int, ...]:
        out, n = [], 0
        for f in self.faces:
            out.append(n)
            n += len(f)
        return tuple(out)

    @property
    def num_loops(self) -> int:
        return sum(len(f) for f in self.faces)

    def triangle_count(self) -> int:
        """Triangles after fan triangulation -- what the engine importer counts."""
        return sum(max(len(f) - 2, 0) for f in self.faces)

    def face_size_histogram(self) -> Dict[int, int]:
        hist: Dict[int, int] = {}
        for f in self.faces:
            hist[len(f)] = hist.get(len(f), 0) + 1
        return dict(sorted(hist.items()))

    # -- geometry ----------------------------------------------------------

    def world_verts(self) -> np.ndarray:
        """Cached: a rule asking for face areas would otherwise re-transform the
        whole vertex array once per face."""
        cached = getattr(self, "_world_cache", None)
        if cached is None:
            m = self.transform.matrix3()
            t = np.asarray(self.transform.translation, dtype=np.float64)
            cached = self.verts @ m.T + t
            object.__setattr__(self, "_world_cache", cached)
        return cached

    def bbox(self, world: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        v = self.world_verts() if world else self.verts
        if v.shape[0] == 0:
            z = np.zeros(3)
            return z, z
        return v.min(axis=0), v.max(axis=0)

    def face_centroid(self, i: int, world: bool = False) -> np.ndarray:
        v = self.world_verts() if world else self.verts
        return v[list(self.faces[i])].mean(axis=0)

    def face_area(self, i: int, world: bool = True) -> float:
        if world:
            return float(self.face_areas()[i])
        return polygon_area_3d(self.verts[list(self.faces[i])])

    def face_areas(self, world: bool = True) -> np.ndarray:
        """Cached for world space, which is what every rule asks for."""
        if not world:
            v = self.verts
            return np.array(
                [polygon_area_3d(v[list(f)]) for f in self.faces], dtype=np.float64
            )
        cached = getattr(self, "_area_cache", None)
        if cached is None:
            v = self.world_verts()
            cached = np.array(
                [polygon_area_3d(v[list(f)]) for f in self.faces], dtype=np.float64
            )
            object.__setattr__(self, "_area_cache", cached)
        return cached

    def face_normal(self, i: int, world: bool = True) -> np.ndarray:
        v = self.world_verts() if world else self.verts
        return newell_normal(v[list(self.faces[i])])

    # -- names -------------------------------------------------------------

    def group_of_face(self, i: int) -> Optional[str]:
        index = getattr(self, "_group_cache", None)
        if index is None:
            index = {}
            for name, faces in self.face_groups.items():
                for fi in faces:
                    index.setdefault(fi, name)
            object.__setattr__(self, "_group_cache", index)
        return index.get(i)

    def entity(self, face_index: Optional[int] = None) -> str:
        """`SM_Crate_Wood_LOD0` or `SM_Crate_Wood_LOD0.side_L`."""
        if face_index is None:
            return self.name
        g = self.group_of_face(face_index)
        return "%s.%s" % (self.name, g) if g else self.name

    def evolve(self, **kw) -> "AssetMesh":
        return replace(self, **kw)


@dataclass(frozen=True)
class Asset:
    """What `check(profile)` takes: one mesh plus what emit declared about it."""

    mesh: AssetMesh
    spec: EmitSpec = EmitSpec()

    @property
    def name(self) -> str:
        return self.mesh.name


def articulation_from_rig(rig, pose=None, reach=(), symmetry=None,
                          roots: Optional[int] = None) -> ArticulationSpec:
    """Flatten a `pose.Rig` into the declaration the verifier grades.

    The bridge lives here rather than in `lathe.pose` because it is the
    verifier's contract being satisfied, not the rig's. `lathe.pose` has no idea
    this subsystem exists and should keep it that way.
    """
    angles = dict(rig.skeleton.rest())
    if pose:
        angles.update({k: float(v) for k, v in pose.items()})
    joints = tuple(
        JointDecl(
            name=name,
            limit=tuple(float(v) for v in rig.skeleton.joint(name).limit),
            angle=float(angles[name]),
            parent=rig.skeleton.joint(name).parent,
            offset=tuple(float(v) for v in rig.skeleton.joint(name).offset),
            axis=tuple(float(v) for v in rig.skeleton.joint(name).unit_axis()),
        )
        for name in rig.skeleton.names
    )
    return ArticulationSpec(joints=joints, reach=tuple(reach),
                            symmetry=symmetry, roots=roots)


# --------------------------------------------------------------- adapters


def from_manifold(man, name: str, transform: Transform = Transform(),
                  face_groups: Optional[Mapping[str, Sequence[int]]] = None,
                  uv_sets: Optional[Mapping[str, Sequence]] = None) -> AssetMesh:
    """Adapt a `manifold3d.Manifold` into an AssetMesh.

    `to_mesh()` hands back read-only views and `np.ascontiguousarray` is a
    no-op on them, so the copies below are explicit -- see prototype/README.md.

    `uv_sets` accepts either the per-corner `(T, 3, 2)` shape the authoring layer
    carries or the flat `(L, 2)` this class stores; both are the same numbers, and
    accepting the former means the caller does not have to remember which. The
    verifier stays independent of the authoring layer either way -- it takes UVs,
    it does not know how to make them.
    """
    mesh = man.to_mesh()
    vp = np.array(mesh.vert_properties, dtype=np.float64, copy=True)
    tv = np.array(mesh.tri_verts, dtype=np.int64, copy=True)
    verts = vp[:, :3]
    faces = tuple(tuple(int(i) for i in tri) for tri in tv)

    flat: dict = {}
    for key, uv in dict(uv_sets or {}).items():
        arr = np.asarray(uv, dtype=np.float64)
        if arr.ndim == 3 and arr.shape[1:] == (3, 2):
            arr = arr.reshape(-1, 2)
        flat[key] = arr

    return AssetMesh(
        name=name,
        verts=verts,
        faces=faces,
        transform=transform,
        face_groups=dict(face_groups or {}),
        uv_sets=flat,
    )


def from_mesh(mesh, name: Optional[str] = None, transform: Transform = Transform(),
              uv_sets: Optional[Mapping[str, Sequence]] = None) -> AssetMesh:
    """Adapt a neutral triangle mesh -- the shape an imported glTF arrives in.

    Duck-typed on purpose. This reads `.verts`, `.tris` and optionally
    `.part_id`, `.parts` and `.uv`, which is the whole contract; it does not
    import `perceive` or `io`, so the dependency arrow keeps pointing one way and
    the rule engine stays the cheapest, most portable subsystem (see the module
    docstring). Anything that can produce those arrays can be verified.

    Face groups come from part ids, so a finding on an imported asset names the
    material it came from -- `barrel.M_metal_bands` rather than `face 4471`.
    Several parts may share a name (an exporter that splits one material across
    primitives), and their triangles merge under it, which is what you want when
    reading a report.
    """
    verts = np.asarray(mesh.verts, dtype=np.float64)
    tris = np.asarray(mesh.tris, dtype=np.int64)
    faces = tuple(tuple(int(i) for i in tri) for tri in tris)

    groups: Dict[str, List[int]] = {}
    part_id = getattr(mesh, "part_id", None)
    if part_id is not None:
        names = dict(getattr(mesh, "parts", {}) or {})
        for i, pid in enumerate(np.asarray(part_id).reshape(-1).tolist()):
            groups.setdefault(names.get(int(pid), "part_%d" % int(pid)), []).append(i)

    sets = dict(uv_sets or {})
    if not sets and getattr(mesh, "uv", None) is not None:
        sets[getattr(mesh, "uv_set", "UV0")] = mesh.uv
    flat: dict = {}
    for key, uv in sets.items():
        arr = np.asarray(uv, dtype=np.float64)
        if arr.ndim == 3 and arr.shape[1:] == (3, 2):
            arr = arr.reshape(-1, 2)
        flat[key] = arr

    return AssetMesh(
        name=name or getattr(mesh, "name", "mesh"),
        verts=verts,
        faces=faces,
        transform=transform,
        face_groups={k: tuple(v) for k, v in groups.items()},
        uv_sets=flat,
    )


def from_bpy_mesh(obj) -> AssetMesh:  # pragma: no cover - exercised by blender_smoke.py
    """Adapt a `bpy.types.Object` holding a mesh.

    Kept deliberately short: this function is the entire Blender surface area of
    the verifier. The engine never imports bpy, so Blender is a source of assets
    and a viewer, not a dependency (DESIGN.md section 11).

    Face groups come from material slots, which is the closest thing bpy has to
    a named logical face. A Lathe-native asset would supply section 5.2
    provenance tags instead, and the report would name `#cut_face` rather than
    a material.
    """
    me = obj.data
    verts = np.array([tuple(v.co) for v in me.vertices], dtype=np.float64)
    faces = tuple(tuple(p.vertices) for p in me.polygons)

    face_edges = set()
    for f in faces:
        for j in range(len(f)):
            a, b = f[j], f[(j + 1) % len(f)]
            if a != b:
                face_edges.add(edge_key(a, b))
    wire = tuple(
        edge_key(*e.vertices) for e in me.edges if edge_key(*e.vertices) not in face_edges
    )

    uv_sets = {
        layer.name: np.array([tuple(d.uv) for d in layer.data], dtype=np.float64)
        for layer in me.uv_layers
    }

    groups: Dict[str, List[int]] = {}
    if me.materials:
        for i, p in enumerate(me.polygons):
            mat = me.materials[p.material_index] if p.material_index < len(me.materials) else None
            if mat is not None:
                groups.setdefault(mat.name, []).append(i)

    return AssetMesh(
        name=obj.name,
        verts=verts,
        faces=faces,
        wire_edges=wire,
        transform=Transform(
            translation=tuple(obj.location),
            rotation_euler=tuple(obj.rotation_euler),
            scale=tuple(obj.scale),
        ),
        uv_sets=uv_sets,
        face_groups={k: tuple(v) for k, v in groups.items()},
    )


# --------------------------------------------------------------- topology


def reference_point(mesh: AssetMesh, which: str) -> Optional[np.ndarray]:
    """Named reference points for W201. Unknown names return None so the rule
    reports 'not checked' instead of inventing an answer."""
    lo, hi = mesh.bbox(world=False)
    mid = (lo + hi) * 0.5
    table = {
        "bottom_center": np.array([mid[0], mid[1], lo[2]]),
        "top_center": np.array([mid[0], mid[1], hi[2]]),
        "center": mid,
        "centroid": mid,
        "min_corner": lo,
        "max_corner": hi,
        "world_origin": np.zeros(3),
    }
    return table.get(which)
