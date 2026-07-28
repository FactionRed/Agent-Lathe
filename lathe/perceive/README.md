# Perception — Tier 1 and Tier 2

This package implements [DESIGN.md](../../DESIGN.md) section 6, tiers 1 and 2. Tier 3
(renders) and tier 4 (diffs) stay out of scope. Nothing here draws pixels.

DESIGN.md §13 lists "renders over-trusted" as a risk. It names the mitigation: *make Tier 1
output genuinely excellent*. That is the brief this package answers. An agent can read every
number below in the same turn it made the edit. Every number that is not exact says so.

```bash
python -m pytest lathe/perceive/tests -q          # 172 tests, ~3 s
```

```python
from lathe.perceive import tier1, tier2, from_manifold
from lathe.perceive.units import Point, Length

mesh = from_manifold(solid, name="SM_Crate_Wood")
print(tier1.report(mesh))                      # every numeric probe, one block
print(tier1.symmetry(mesh, "YZ"))              # one probe, with its method declared
print(tier2.report(mesh, texture_px=2048))     # topology / material / part / UV tables
tier1.raycast(mesh, Point.m(0, -2, 0.29), "+Y").fields["distance_m"]   # numbers, not text
```

`lathe` is a namespace package. No install step. Run from the repo root.

## Contents

| File | What |
|---|---|
| `units.py` | `Length` `Area` `Volume` `Angle` `Point`, and the coercions that reject a bare number |
| `mesh.py` | the neutral `Mesh` value, topology bookkeeping, exact closed-form geometry, Manifold adapters |
| `kernel.py` | overlap measures: exact through manifold3d, approximate without it |
| `tier1.py` | 11 numeric probes + `report()` |
| `tier2.py` | 5 structural summaries + `report()` |
| `tests/` | the analytic battery and the fixtures it measures |

## Design commitments

**Pure functions, no hidden state** (§2 principle 1). A probe takes an immutable `Mesh` and
returns a value. No session, no selection, no active object, no global config, no RNG. The
symmetry sampler uses fixed barycentric coordinates, so it repeats without a seed. The `Mesh`
constructor marks its arrays read-only, so a probe cannot mutate its input even by accident.
Each `Mesh` memoises its derived topology. The tests assert that a warm memo and a cold memo
produce byte-identical output.

**Every length carries a unit** (§4.1). On the way out, and also on the way in:

```python
tier1.raycast(mesh, (0, -2, 0.3), "+Y")             # TypeError, and the message cites §4.1
tier1.raycast(mesh, (0, -2, 0.3), "+Y", unit="m")   # fine
tier1.raycast(mesh, Point.mm(0, -2000, 300), "+Y")  # also fine, and identical
```

`600mm` and `0.6m` unify. `Length` has no `__float__`, so a dimensioned value cannot leak
into arithmetic as a bare number. Directions carry no dimension and take `"+Y"` or a
3-vector.

**Headless, Blender-optional, kernel-preferred** (§11). This package needs numpy. It uses
manifold3d for the two measures that genuinely need booleans, and treats it as optional
everywhere else. Nothing imports `bpy`. `test_no_kernel.py` proves it: the suite runs the
probes in a subprocess with both `manifold3d` and `bpy` blocked at import, and the exact
answers do not move.

The neutral `Mesh` exists because Manifold *cannot represent* a non-watertight mesh.
`Manifold(open_mesh)` returns empty with `Error.NotManifold`. Part of this layer's job is to
report broken geometry, so it cannot rest on a type that refuses to hold it.

**Method is part of the answer.** A probe with both an exact path and an approximate one
prints which path ran. An agent over-trusts exactly those numbers whose provenance it cannot
see.

---

# Tier 1 — numeric probes

`tier1.report(mesh)` is the one call to make after every edit.

```
probe SM_Crate_Wood                    tier 1 · 112 tris · clean
  measure(bbox)                  → 0.600 × 0.600 × 0.600 m
  count(tris)                    → 112
  count(verts) / count(edges)    → 48 / 168
  count(faces)                   → 26 logical   quad:24  24-gon:2
  count(components)              → 1
  genus                          → 5
  volume(SM_Crate_Wood)          → 0.176 m³
  surface_area(SM_Crate_Wood)    → 5.074 m²
  center_of_mass(SM_Crate_Wood)  → (0.000, 0.000, 0.000) m
  is_watertight(SM_Crate_Wood)   → true
  degenerates(SM_Crate_Wood)     → none
  origin_placement(SM_Crate_Wood) → center   at (0.500, 0.500, 0.500) of bounds
  symmetry(SM_Crate_Wood, all planes) → YZ 1.000  XZ 1.000  XY 1.000
```

The same call on a mesh with a hole in it. Each affected probe says so in its own words. The
block repeats the problems at the end with verifier codes from §7:

```
probe open_box                   tier 1 · 10 tris · 4 problem(s)
  measure(bbox)                  → 1.000 × 1.000 × 1.000 m
  count(tris)                    → 10
  count(verts) / count(edges)    → 8 / 17
  count(faces)                   → 5 logical   quad:5
  count(components)              → 1
  genus                          → n/a (not a closed orientable surface)
    problem            : E101 4 boundary edges -- the surface is open
  volume(open_box)               → 0.667 m³  UNRELIABLE
    problem            : volume is undefined on an open surface (4 boundary edges); the number above is the divergence-theorem sum, not an enclosed volume
  surface_area(open_box)         → 5.000 m²
  center_of_mass(open_box)       → (0.500, 0.500, 0.400) m
    problem            : no enclosed solid, so there is no volumetric centre of mass
  is_watertight(open_box)        → false  (4 boundary edges in 1 hole)
    problem            : E101 not watertight: 4 boundary edges in 1 hole
  degenerates(open_box)          → none
  origin_placement(open_box)     → min_corner   at (0.000, 0.000, 0.000) of bounds
  symmetry(open_box, all planes) → YZ 1.000  XZ 1.000  XY 0.943

  ! E101 4 boundary edges -- the surface is open
  ! volume is undefined on an open surface (4 boundary edges); the number above is the divergence-theorem sum, not an enclosed volume
  ! no enclosed solid, so there is no volumetric centre of mass
  ! E101 not watertight: 4 boundary edges in 1 hole
```

Every probe also returns `.fields`, the same content as data (`distance_m`, `volume_m3`,
`face_degrees`, …). `query` can hand back structure instead of prose.

## measure(bbox)

```
  measure(bbox)                  → 0.600 × 0.600 × 0.600 m
    min / max          : (-0.300, -0.300, -0.300) m / (0.300, 0.300, 0.300) m
    center             : (0.000, 0.000, 0.000) m
    diagonal           : 1.039 m
  measure(seat.bbox)             → 0.400 × 0.400 × 0.040 m
    min / max          : (-0.200, -0.200, 0.410) m / (0.200, 0.200, 0.450) m
    center             : (0.000, 0.000, 0.430) m
    diagonal           : 0.567 m
```

Exact. `part="seat"` scopes any probe to a named part. It resolves through the provenance
channel (§5.2), never through indices. The probe ignores unreferenced vertices on purpose.
One stray vertex left 100 m away must not multiply the reported size of the asset.

## measure(distance(a, b))

```
  measure(distance(SM_Stool.leg_01, SM_Stool.leg_02)) → 0.320 m
    centers            : (-0.160, -0.160, 0.205) m -> (0.160, -0.160, 0.205) m
    axis delta         : dx 0.320 m  dy 0.000 m  dz 0.000 m
    surface gap        : 0.280 m   method=vertex_to_surface (lower bound; not a solid (NotManifold))
    bbox gap           : 0.280 m
```

The headline is centre-to-centre of the two bounding boxes. That reproduces DESIGN.md's own
example: a 400 mm seat with legs inset 40 mm gives 0.320 m. It is also what a program that
placed parts *by relationship* (§5.4) asked for.

The surface gap answers "do these collide?". It sits on a detail line with its method named,
because the kernel-free path can overestimate it. See Accuracy.

## count(...)

```
  count(tris)                    → 112
  count(verts) / count(edges)    → 48 / 168
  count(faces)                   → 26 logical   quad:24  24-gon:2
  count(components)              → 1
  genus                          → 5
```

Exact. `count(faces)` counts *logical* faces, not triangles: coplanar triangle groups keyed
by `(run_original_id, face_id)`. Each group's polygon degree comes from the edges used once
inside it. The crate has 6 body faces plus 5 slots of 4 walls each. The two faces the slots
pierce are quads with five rectangular holes, hence 24 outline edges each.

`count(components)` is E108 evidence. `genus` is the handle count. Both are the numbers that
caught DESIGN.md's own crate example severing into six slabs (`prototype/README.md`). The
probe refuses genus rather than guessing it when the surface is not closed and orientable.

## volume / surface_area / center_of_mass

```
  volume(SM_Crate_Wood)          → 0.176 m³
  surface_area(SM_Crate_Wood)    → 5.074 m²
  center_of_mass(SM_Stool)       → (0.000, 0.000, 0.375) m
    method             : volume_centroid (exact)
    relative to bbox   : (0.500, 0.500, 0.834) of size
```

Volume uses the divergence theorem over signed tetrahedra. A unit cube returns exactly `1.0`,
not `0.9999999`. Centre of mass uses the volume-weighted tetrahedron decomposition, so it
describes a uniform-density solid rather than a vertex average. The stool's mass sits at 83%
of its height. A vertex average would put it somewhere else entirely.

On an open mesh both degrade loudly rather than quietly:

```
  volume(open_box)               → 0.667 m³  UNRELIABLE
    problem            : volume is undefined on an open surface (4 boundary edges); the number above is the divergence-theorem sum, not an enclosed volume
  center_of_mass(open_box)       → (0.500, 0.500, 0.400) m
    method             : area_centroid (mesh is not closed; this is the shell centre)
```

DESIGN.md does not list `surface_area`. It is here for two reasons. Texel density (W302)
divides by it, and area is the cheapest way to notice that a boolean left an interior shell
behind.

## raycast(from, dir)

```
  raycast(from = (0.000, -2.000, 0.290) m, dir = +Y) → hit SM_Crate_Wood #body at 1.700 m
    position           : (0.000, -0.300, 0.290) m
    normal             : (0.000, -1.000, 0.000)   incidence 0.0°   front side
    part / face        : body / face group 0
    material           : oak
    hits along ray     : 2   first 1.700 m  last 2.300 m  span 0.600 m
    searched           : 4.157 m
```

Möller–Trumbore over every triangle, with no acceleration structure. The probe merges hits
closer together than one part in 10⁹ of the bounds, so a ray crossing a shared edge counts
once rather than twice. It reports the crossing count because that count is diagnostic. Two
crossings from outside means solid. Anything odd means a hole.

A miss reports the aim error, because "miss" alone is not actionable:

```
  raycast(from = (0.000, -2.000, 0.000) m, dir = +Y) → miss
    searched           : 4.157 m along +Y
    nearest approach   : 0.240 m at 1.700 m along the ray
    problem            : no geometry along that ray -- check the aim, not the mesh
```

That is the crate's middle slot. The ray went straight through the gap, and 0.240 m is the
slot's half-height. Fired into a mesh with a missing lid, the probe names both faults:

```
  raycast(from = (0.500, 0.500, 2.000) m, dir = -Z) → hit open_box at 2.000 m
    normal             : (0.000, 0.000, -1.000)   incidence 0.0°   back side
    hits along ray     : 1
    problem            : E103 first hit is a back face -- the ray starts inside the solid, or that normal is inverted, or the front surface is missing
    problem            : odd crossing count (1) -- the surface is not closed along this ray (a hole), or the ray started inside it
```

## is_watertight

```
  is_watertight(SM_Crate_Wood)   → true
    closed             : true
    edge manifold      : true
    oriented           : true
    components         : 1
    orientation        : outward

  is_watertight(open_box)        → false  (4 boundary edges in 1 hole)
    closed             : false
    holes              : 1  (4 boundary edges)
    problem            : E101 not watertight: 4 boundary edges in 1 hole

  is_watertight(nonmanifold_fin) → false  (6 boundary edges in 1 hole; 1 non-manifold edge; 2 duplicate half-edges)
    closed             : false
    edge manifold      : false
    oriented           : false
    problem            : E101 not watertight: 6 boundary edges in 1 hole; 1 non-manifold edge; 2 duplicate half-edges
```

Three independent conditions, reported separately because the repairs differ. *Closed*: every
edge has exactly two triangles. *Edge-manifold*: no edge has three or more. *Oriented*: no
edge runs the same way twice, which would be a winding clash.

The result is truthy, so `assert is_watertight(mesh)` reads correctly. `orientation` catches
the closed-but-inside-out case:

```
  is_watertight(inverted_cube)   → true
    orientation        : inward (inverted)
    problem            : E103 closed but inside out (negative signed volume)
```

## degenerates and origin_placement

Two additions to DESIGN.md's list. §7 makes both of them errors, and an agent cannot see
either one in a render.

```
  degenerates(SM_Crate_Wood)     → none
  degenerates(sloppy_cube)       → zero_area_tris:1  orphan_verts:1  coincident_verts:1
    weld tolerance     : 0.002 mm
    problem            : E105 degenerate geometry: 1 zero area tris, 1 orphan verts, 1 coincident verts

  origin_placement(SM_Stool)     → bottom_center   at (0.500, 0.500, 0.000) of bounds
    offset from center : (0.0, 0.0, -225.0) mm
    origin inside bbox : true
```

`degenerates` covers E105 (loose vertices, zero-area faces) plus coincident-but-unwelded
vertices and duplicate faces. `origin_placement` covers W201. DESIGN.md §4 declares the origin
in the program (`@ origin(bottom_center)`) and the emitter must honour it, so the check costs
two subtractions and a name.

## symmetry(plane) — read this one carefully

```
  symmetry(SM_Crate_Wood, plane = YZ) → 1.000
    method             : volume_iou (exact, manifold kernel)
    mirror plane       : x = 0.000 m (bbox_center)
    asymmetric vol     : 0.000000 m³ of 0.175680 m³ union   0.00%

  symmetry(notched_cube, plane = YZ) → 0.995
    method             : volume_iou (exact, manifold kernel)
    mirror plane       : x = 0.000 m (bbox_center)
    asymmetric vol     : 0.001000 m³ of 0.216000 m³ union   0.46%

  symmetry(SM_Stool, plane = XY) → 0.154
    method             : volume_iou (exact, manifold kernel)
    mirror plane       : z = 0.225 m (bbox_center)
    asymmetric vol     : 0.012399 m³ of 0.014657 m³ union   84.59%
```

**Method 1 — `volume_iou`. Default for a closed solid. Exact.** Mirror the solid across the
plane, then take the volumetric intersection over union:

```
score = vol(A ∩ mirror(A)) / vol(A ∪ mirror(A))
```

This is the fraction of the object that maps onto itself. It is a true ratio in 0..1. It
returns exactly 1.000 for an exactly symmetric solid. Tessellation cannot fake it, because
re-triangulating a shape does not move its volume. The cost is two kernel booleans. This is
the one Tier 1 probe that costs milliseconds rather than microseconds.

**Method 2 — `surface_sample`. Fallback. Approximate.** The probe picks this automatically
when the mesh is not a closed solid, because then no boolean exists to run. It mirrors
deterministic area-weighted samples of the surface. For each one it measures the distance back
to the nearest *triangle*. Point-to-surface rather than point-to-vertex means the number does
not move when the tessellation changes.

```
deviation_rms = sqrt( Σ wᵢ dᵢ² / Σ wᵢ )        wᵢ = triangle area / samples per triangle
score         = clamp(1 − deviation_rms / bbox_diagonal, 0, 1)
```

```
  symmetry(open_box, plane = XY) → 0.943
    method             : surface_sample (approximate; mesh is not a closed solid)
    mirror plane       : z = 0.500 m (bbox_center)
    deviation          : rms 98.60 mm   max 333.33 mm at (0.667, 0.667, 1.000) m
    score formula      : 1 - rms / bbox_diagonal (1.732 m); not an IoU
    samples            : 40
```

**The two scores are not comparable.** Method 1 gives an overlap fraction. Method 2 gives a
normalised deviation. Both read 1.000 for an exactly symmetric shape, and both fall as
asymmetry grows. But the same object scores 0.995 under method 1 and 0.996 under method 2,
measured on `notched_cube`.

When the method line says `surface_sample`, read the **deviation lengths** and not the score.
"Worst deviation 50 mm at (x, y, z)" tells an agent where to look. A normalised score does
not. The tests assert that the two methods never disagree about *whether* a shape is
symmetric, which is the property that matters.

**The output always prints the mirror plane position.** "Symmetric about its own middle" and
"symmetric about the world YZ plane" are different questions, and confusing them is a real
bug. `about=` accepts `"bbox_center"` (default), `"origin"`, `"centroid"`, or an explicit
`Length`. A cube centred at x = 0.15 m scores 1.000 about its own centre and exactly 1/3
about the world plane. Both numbers answer different questions correctly.

`symmetry_planes(mesh)` runs all three principal planes, which is the question agents
actually ask:

```
  symmetry(SM_Stool, all planes) → YZ 1.000  XZ 1.000  XY 0.154
```

---

# Tier 2 — structural summaries

Each summary returns a `Summary`. `str(s)` gives the agent text, `s.fields` gives the caller
data, `s.warnings` gives the verifier codes.

## Topology histograms

```
topology SM_Crate_Wood                                    tier 2
  tris / verts / edges           → 112 / 48 / 168
  faces (logical)                → 26   quad:24  24-gon:2   E102 n-gons present
  components / genus / holes     → 1 / 5 / 0
  tri area                       → min 840.0 mm²  p50 8400.0 mm²  max 180000.0 mm²   max/min 214.3
  edge length                    → min 28.0 mm  p50 480.0 mm  max 848.5 mm
  aspect ratio                   → p50 4.08  p95 32.91  max 52.68   slivers>8: 52
  vertex valence                 → 5:4  6:18  7:16  8:2  9+:8   of 48
  dihedral (normal deviation)    → flat<1°:96  hard>60°:72   of 168

  ! E102 2 logical face(s) have more than 4 sides (largest 24)
```

```
topology sphere                                           tier 2
  tris / verts / edges           → 4,608 / 2,306 / 6,912
  faces (logical)                → 4,608   tri:4,608
  components / genus / holes     → 1 / 0 / 0
  tri area                       → min 535.3 mm²  p50 657.9 mm²  max 933.7 mm²   max/min 1.7
  edge length                    → min 32.7 mm  p50 40.1 mm  max 48.0 mm
  aspect ratio                   → p50 1.29  p95 1.67  max 1.73   slivers>8: 0
  vertex valence                 → 4:6  6:2,300   of 2,306
  dihedral (normal deviation)    → flat<1°:672  1-30°:6,240   of 6,912
```

That contrast is the point of the block. The crate's kernel triangulation is 46% slivers with
a 214× area spread. The sphere is uniform. An agent reading a wireframe render cannot tell
those apart reliably. Two numbers here can.

Aspect ratio is longest edge over shortest altitude, normalised so an equilateral triangle
reads 1.0 and a split quad (right isoceles) reads √3 ≈ 1.73. Dihedral is the angle between
adjacent face normals, so 0° is flat and a cube edge is 90°.

## Per-island UV stats

```
uv SM_UVBox                                               tier 2
  uv set / texture               → UV0 / 1024 px
  islands                        → 6   coverage 24.0% of the sheet
  texel density px/cm            → min 2.048  p50 2.048  max 2.048   spread 0.0%
  vs target 2.048 px/cm          → worst deviation 0.0%   within ±10%
  overlap                        → none detected   raster 512²
  island padding                 → 51.2 px (bbox lower bound)   OK

    island     tris  uv area   3d area  px/cm  stretch  uv bbox
    ---------  ----  -------  --------  -----  -------  ---------------------------
    side_-Y       2   0.0400  1.000 m²  2.048     1.00  (0.775,0.025)-(0.975,0.225)
    side_+Y       2   0.0400  1.000 m²  2.048     1.00  (0.525,0.025)-(0.725,0.225)
    bottom_-Z     2   0.0400  1.000 m²  2.048     1.00  (0.275,0.275)-(0.475,0.475)
    side_-X       2   0.0400  1.000 m²  2.048     1.00  (0.275,0.025)-(0.475,0.225)
    top_+Z        2   0.0400  1.000 m²  2.048     1.00  (0.025,0.275)-(0.225,0.475)
    side_+X       2   0.0400  1.000 m²  2.048     1.00  (0.025,0.025)-(0.225,0.225)
```

The same sheet, with one island half-scaled and one stacked on top of another:

```
  texel density px/cm            → min 1.024  p50 2.048  max 2.048   spread 50.0%
  vs target 2.048 px/cm          → worst deviation 50.0%   W302
  overlap                        → 23.5% of covered area on 10,404 cells   E106
    overlaps           : bottom_-Z ∩ side_+X  100.0% of the smaller island
  island padding                 → 0.0 px (bbox lower bound)   W202 below 4 px

  ! W302 texel density 1.024-2.048 px/cm against a 2.048 ±10% target
  ! E106 UV islands overlap: bottom_-Z ∩ side_+X at 100.0% of the smaller island
  ! W202 island padding 0.0 px below the 4 px table value for 1024 px
```

Texel density is `texture_px * sqrt(uv_area / surface_area) / 100`, in pixels per centimetre
of surface. That is the quantity W302 constrains. `stretch` is the ratio of worst to best
density inside one island. It measures the distortion an agent cannot otherwise see without a
checker render.

The island walk follows mesh adjacency and splits where UVs disagree across a shared edge. It
does *not* cluster UV positions. That distinction keeps two islands that sit exactly on top of
each other apart, which is the E106 case this block hunts for.

## Material assignment table

```
materials SM_Crate_Wood                                   tier 2
  materials                      → 2 assigned   slots: oak, steel_brushed

    material       tris  %tris      area  %area  parts
    -------------  ----  -----  --------  -----  -------------------------
    oak              72  64.3%  2.026 m²  39.9%  body
    steel_brushed    40  35.7%  3.048 m²  60.1%  slot_1, slot_2, slot_3 +2
```

Triangle share drives draw calls. Area share drives texture budget. They disagree here by 24
points, so the table prints both.

## Polygon distribution by named part

```
parts SM_Stool                                            tier 2
  parts                          → 5
  largest                        → seat (26.6% of tris)

    part    tris  %tris  verts  bbox                         area  volume  closed
    ------  ----  -----  -----  -----------------------  --------  ------  ------
    seat     276  26.6%    264  0.400 × 0.400 × 0.040 m  0.379 m²       -      no
    leg_01   190  18.3%    128  0.040 × 0.040 × 0.410 m  0.053 m²       -      no
    leg_02   190  18.3%    128  0.040 × 0.040 × 0.410 m  0.053 m²       -      no
    leg_03   190  18.3%    128  0.040 × 0.040 × 0.410 m  0.053 m²       -      no
    leg_04   190  18.3%    128  0.040 × 0.040 × 0.410 m  0.053 m²       -      no
```

The `volume` column stays empty on purpose. A union leaves every part open at its seam, which
the `closed` column reports. No part may claim an enclosed volume. A plausible-looking number
there would be exactly the quiet lie this subsystem exists to prevent.

## LOD chain summary

```
lod chain SM_Rock                                         tier 2
  levels                         → 4
  silhouette method              → outline_iou (exact)   deviation = 1 - outline IoU, worst of X/Y/Z

    level          tris  vs prev  vs LOD0  Δvolume   Δarea    Δbbox  silhouette
    ------------  -----  -------  -------  -------  ------  -------  ----------
    SM_Rock_LOD0  4,608        -    1.000        -       -        -           -
    SM_Rock_LOD1  2,604    0.565    0.565   -0.36%  -0.15%  1.07 mm       0.19%
    SM_Rock_LOD2  1,044    0.401    0.227   -1.73%  -0.67%  2.16 mm       0.54%
    SM_Rock_LOD3    300    0.287    0.065   -7.56%  -3.11%  8.89 mm       2.72%

  ! W204 SM_Rock_LOD3 reduces 0.287 from the previous level (target 0.500 ±0.150)
  ! W205 SM_Rock_LOD3 silhouette deviates 2.72% from LOD0 (limit 2.00%)
```

W205 is "silhouette deviation between LODn and LOD0 above threshold". That sounds like it
needs a renderer. It does not. Project both solids down each axis and compare the outlines.
The kernel does it exactly (`project()` plus a 2D boolean) when both levels are closed solids,
and a raster does it otherwise.

**This is the one place where a Tier 3 render was the obvious implementation and a Tier
1-style number beats it.** The number is exact, it takes 8 ms for the whole chain, and it has
no camera to get wrong.

---

# Accuracy — what is exact, what is not

### Exact to float64 round-off

`bbox`, `volume`, `surface_area`, `center_of_mass`, all counts, `genus`, `is_watertight`,
`degenerates`, `origin_placement`, `raycast` distances, per-island UV areas, and material and
part area shares.

These use closed forms. A unit cube returns volume exactly `1.0` and area exactly `6.0`.
Manifold's own tetrahedron returns exactly `8/3`. The DESIGN.md §4 crate returns
`0.216 − 5 × 0.028 × 0.6 × 0.48 = 0.17568` to a relative 1e-14. The tests also check every one
of these against manifold3d's independent implementation — `volume()`, `surface_area()`,
`bounding_box()`, `genus()`, `decompose()`, `ray_cast()` — to a relative 1e-12 across nine
fixtures.

### Discretisation, not error

A probe measures the mesh it holds, not the ideal shape that mesh approximates. A geodesic
sphere is an inscribed polyhedron, so its volume falls *correctly* below `4/3 πr³`:

| segments | tris | volume error | ratio |
|---|---|---|---|
| 32 | 512 | 9.36e-02 | — |
| 64 | 2,048 | 2.37e-02 | 3.96 |
| 128 | 8,192 | 5.93e-03 | 3.99 |
| 256 | 32,768 | 1.48e-03 | 4.00 |

Second-order convergence, exactly as it should be. The tests assert the ratio stays in
3.8–4.2. That is a much stronger statement than "close to π". An agent that needs the ideal
volume of a sphere should compute it, not measure it.

### Approximations, named

**`distance` surface gap.** Exact when a kernel handle exists (`min_gap`). Without one it
falls back to a vertex-to-surface search. That search is exact whenever the closest pair
involves a vertex, and it **overestimates** when it does not. Measured worst case: two 2 m
bars crossing 200 mm apart, closest points mid-face.

| | value |
|---|---|
| true gap (`min_gap`) | 0.200 m |
| `vertex_to_surface` fallback | 0.971 m |

A 4.9× overestimate. It never *under*-reports, so it is safe as a clearance check and useless
as a contact check. `probe_distance` prefers the kernel and prints the method it used. Part
sub-meshes are open surfaces, so a probe between two named parts of one solid always takes the
fallback path. The `leg_01`/`leg_02` line above shows that.

**`symmetry` fallback score.** Covered in full above. In short: the fallback score is a
normalised deviation, not an overlap fraction, and it does not compare numerically with the
exact method. Read the deviation lengths.

**`symmetry` sampling budget.** The fallback caps work at 1 M point-triangle pairs. Above
about 250 triangles it keeps every k-th sample and prints the stride. At 32k triangles that
leaves 30 samples, enough to see a gross mismatch and not enough to locate a 1 mm one. The
exact path has no such limit. Close the mesh if you need precision on a big one.

**UV overlap.** Measured on a 512² raster, configurable. It detects any overlap bigger than
one cell, about 0.002 of the sheet per side, and it cannot see anything smaller. The output is
a fraction of covered area plus the worst pair, never a bare boolean.

**UV padding.** Measured between island UV *bounding boxes*. That is a conservative lower
bound on the true gap between islands, so it can warn early and never late. A tight L-shaped
island pair will report less padding than it really has.

**LOD silhouette deviation.** Exact through the kernel. The rasterised fallback lands within
0.5% absolute IoU of the exact value at 512², which the tests assert. That is adequate against
a 2% W205 threshold and not against a 0.5% one.

**Boundary loop count.** Two boundary loops that share one pinch vertex count as a single
loop. The boundary *edge* count stays exact. Only the loop count suffers.

**Component count.** Vertex connectivity, so two solids touching at a single vertex count as
one component. `Manifold.decompose()` gives the same answer, and the tests check equality
across all fixtures including a two-component case.

**N-gon degree.** A coplanar region with holes reports its full outline edge count, so the
crate's pierced faces read as "24-gons". The edge count is correct, but "24-gon" is a poor name
for "quad with five rectangular holes". See the proposed DESIGN.md change below.

### Cost, measured

Per call on this machine: Python 3.12, numpy 2.5, manifold3d 3.x, Windows. Each mesh builds
its topology once and memoises it, so the first probe on a mesh pays for it.

| probe | 112 tris | 1,036 tris | 8,192 tris | 32,768 tris |
|---|---|---|---|---|
| `bbox` | 0.034 ms | 0.056 ms | 0.261 ms | 0.917 ms |
| `volume` | 0.052 ms | 0.083 ms | 0.422 ms | 2.49 ms |
| `center_of_mass` | 0.072 ms | 0.154 ms | 0.920 ms | 4.17 ms |
| `is_watertight` | 0.055 ms | 0.097 ms | 0.390 ms | 2.42 ms |
| `counts` | 0.125 ms | 0.366 ms | 2.81 ms | 12.0 ms |
| `raycast` | 0.157 ms | 0.313 ms | 2.31 ms | 9.07 ms |
| `degenerates` | 0.119 ms | 0.320 ms | 2.16 ms | 9.44 ms |
| `report` (symmetry off) | 0.60 ms | 1.64 ms | 7.6 ms | 29.8 ms |
| `symmetry` (volume_iou) | 6.2 ms | 19.9 ms | 87 ms | 316 ms |
| `symmetry` (surface_sample) | 8.3 ms | 158 ms | 157 ms | 164 ms |
| `tier2.topology_summary` | 0.84 ms | 1.78 ms | 7.5 ms | 30.0 ms |

DESIGN.md §6 calls Tier 1 "microseconds". That holds for the closed-form probes on prop-sized
geometry, at tens to hundreds of microseconds. It does not hold for `symmetry`, which needs
two kernel booleans. `report(mesh, symmetry_method="off")` is the sub-millisecond form. Even
the expensive case costs two to three orders of magnitude less than a render, and it needs no
camera, no light, and no interpretation.

Two fixes got the cost there, both worth recording. `np.unique(..., axis=0)` costs about 8× a
1-D unique on the same pairs, 39 ms against 4.8 ms on a 32k-triangle sphere, so all edge and
face bookkeeping now encodes each pair as one `int64`. The textbook Python union-find for
components cost 56 ms on that mesh, so components now use vectorised min-label propagation
with pointer jumping. The tests check it against the textbook version on random graphs.

---

# Kernel footguns found

Additions to the two that `prototype/README.md` already records.

**`to_mesh()` returns float32 positions.** Manifold works in double internally, but its
default `Mesh` interchange type is single precision. Reading it turns a 600 mm cube into
600.00002 mm and caps every downstream probe at ~1e-7 relative accuracy. That is enough to
break a 1 µm tolerance, and enough to make `symmetry` report 0.9999999 for an exactly
symmetric solid. `from_manifold` reads `to_mesh64()` instead. This cost about twenty minutes
and a confusing set of failing tests that all looked like different bugs.

**`RayHit.distance` is a parametric t, not a length.** `ray_cast(origin, endpoint)` returns
hits whose `distance` is the fraction along the segment. A hit 4.5 m along a 10 m segment
reports `0.45`. Multiply by the segment length. Code that treats it as metres is wrong by a
factor of the search range, silently, and it gets worse the further you search.

**`face_id` is scoped per original solid, not globally.** Two different ancestors reuse the
same small ids. Grouping triangles by `face_id` alone therefore merges unrelated faces from
different originals. On the crate that collapsed 26 logical faces into 6. The logical face key
is `(run_original_id, face_id)`.

**`as_original()` before a union gives queryable ids.** `original_id()` returns −1 for a
transformed primitive, so building an assembly and then asking each part for its id does not
work. Call `.as_original()` on each part first. That assigns stable ids, which
`run_original_id` then reports per triangle, and which give the part tables above their names.
It also drops ancestor relations, so it is only safe on freshly built parts. §5.2 already warns
about that.

**`Manifold(open_mesh)` is empty, not an error you can ignore.** It returns an empty solid with
`status() == Error.NotManifold`. Check both, because an empty result with `NoError` is also
possible, for example a non-overlapping intersection.

---

# Proposed DESIGN.md changes

Not applied. This package does not touch DESIGN.md.

1. **§6, symmetry cost.** The doc lists `symmetry()` under "microseconds". An exact volumetric
   method needs two kernel booleans: 6 ms at 112 triangles, 87 ms at 8k. Either move it to
   Tier 2, or annotate it as the one Tier 1 probe with kernel cost.
2. **§6, the symmetry example value.** `symmetry(final, plane = YZ) → 0.998` for an exactly
   symmetric crate implies a sampled method. An exact volumetric IoU returns `1.000`. An agent
   that sees 0.998 on a perfect part will hunt for a fault that is not there. Suggest changing
   the example to `1.000` and stating the method.
3. **§6, units in the example.** `center_of_mass(final) → (0, 0, 0.287)` carries no unit, which
   contradicts §4.1. Suggest `(0.000, 0.000, 0.287) m`.
4. **§6, add a Tier 1 principle.** "Any probe that is not exact reports its method and its
   error bound." That rule keeps the cheap tiers trustworthy. §13's "renders over-trusted" risk
   needs it on the numeric side too.
5. **§5.2, add the `face_id` scoping footgun** to the two already listed. State that selectors
   must key on `(run_original_id, face_id)`.
6. **§7, E102 wording.** "N-gons in emitted geometry" should separate a convex n-gon from a
   coplanar region with holes. The crate's pierced faces legitimately carry 24 outline edges
   and are not what the rule aims at.
7. **§7, add E103 evidence.** A negative signed volume is a one-line exact test for inverted
   normals on a closed mesh. Naming it in the rule makes the repair obvious.
8. **§6 Tier 2, part volumes.** Note that a part of a unioned solid is not itself a solid, so a
   per-part table must withhold volume rather than compute it on an open surface.
9. **§11, kernel adapter.** Specify `to_mesh64()`. The float32 default silently caps the
   precision of the whole perception layer.
10. **§12 Phase 3.** Tiers 1 and 2 are about 3,400 lines of numpy and need no renderer. The LOD
    silhouette check, the one item that looks like it needs Tier 3, is exact and cheaper
    without one. Phase 3 could ship the numeric half first and let render work follow demand.

---

# Tests

172 tests, about 3 s. The deliverable is the numbers, not the coverage.

| File | What it pins down |
|---|---|
| `test_units.py` | 600 mm = 0.6 m, bare numbers rejected, dimension mismatches rejected, the DESIGN.md §6 output shapes, ASCII fallback |
| `test_mesh.py` | read-only arrays, no mutation, cold memo = warm memo, exact cube volume ±0 and inverted −1, hand-computed point-triangle distances, vectorised components against a textbook union-find on random graphs |
| `test_tier1.py` | the analytic battery: cube 1.0, tetrahedron 8/3, sphere second-order convergence, cylinder πr²h, torus 2π²Rr² and genus 1, crate 0.17568 and genus 5, stool legs 0.320/0.280, symmetry (V−2v)/V, off-centre symmetry exactly 1/3, raycast distances against the kernel, and every failure fixture reporting its fault |
| `test_tier2.py` | 6 islands, 2.048 px/cm exactly, 24% coverage, 51.2 px padding, stacked islands detected and not merged, oak 4 m² and steel 2 m², part shares summing to 1, LOD ratio consistency and threshold warnings |
| `test_cross_check.py` | volume, area, bbox, genus, components, ray hits and min-gap against manifold3d across nine fixtures, mirror-twice identity, raster against exact silhouette, and the float32 precision the adapter avoids |
| `test_no_kernel.py` | the whole package in a subprocess with `manifold3d` and `bpy` blocked at import: same exact volume, area, centroid, genus and ray distance, the open mesh still diagnosed, `symmetry` falling back and naming the reason |

The suite exercises the "probe reports a problem" path as hard as the happy path. Failure
fixtures: `open_box` (4 boundary edges, one hole), `nonmanifold_fin` (1 non-manifold edge, 2
duplicate half-edges), `inverted_cube` (closed but inside out), `loose_and_degenerate` (orphan
vertex, coincident vertex, zero-area triangle), `crossed_bars` (the distance-fallback failure
case), and the broken UV sheet (E106 with W302 and W202).

# Not built

Tier 3 renders and Tier 4 diffs, both out of scope. No BVH, so `raycast` and the distance
fallback cost O(triangles) per call. That is fine to about 50k triangles and wrong for a
million. No selector language: probes take a mesh and an optional part name, and the §5
selector layer would sit above them. No UV *authoring*, only measurement. No texture or bake
inspection. `Length` covers metric plus inch, foot and yard, and no unit systems beyond that.
