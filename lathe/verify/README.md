# Verification subsystem — DESIGN.md section 7

**Status:** working. 13 rules, 3 profiles, 92 tests, 21 fixtures. Headless.

> Convert taste into assertions. An agent cannot have judgment, but it can
> satisfy a spec. — [DESIGN.md](../../DESIGN.md) §7

```bash
python lathe/verify/tests/run.py            # 92 tests, ~5 s
python -m lathe.verify                      # every fixture, one report each
python -m lathe.verify severed_crate        # one asset
python -m lathe.verify --json clean_prop    # the machine form
python -m lathe.verify --profile mobile_vr clean_prop
python -m lathe.verify --table              # rules and profiles

"/c/Program Files/Blender Foundation/Blender 5.2/blender.exe" \
    -b -P lathe/verify/tests/blender_smoke.py     # the bpy adapter, 10 checks
```

`lathe` is an implicit namespace package. This subsystem adds no
`lathe/__init__.py`, so it owns nothing outside `lathe/verify/`. Run it from the
repository root, or put the root on `PYTHONPATH`.

```python
from lathe.verify import check, fixtures

report = check(fixtures.severed_crate())
report.passed        # False: the export gate. Errors block, warnings do not.
report.render()      # the agent-facing text report
report.to_dict()     # the same findings as JSON
```

## The rules

`game_prop_pbr` covers every code in the §7 table that this pass implements.
Severity is the profile's decision, not the rule's. The column below shows the
`game_prop_pbr` grading.

| Code | Sev | Fires when | Numeric evidence it reports |
|---|---|---|---|
| E101 | error | An edge has >2 faces, a vertex joins >1 face fan, or (unless `allow_boundary`) an edge has 1 face | offending edge count, worst incident-face count, fan count, world-space locations |
| E102 | error | A face has more verts than `max_face_verts` | n-gon count / face count, size histogram, largest n-gon area |
| E103 | error | Two faces traverse a shared edge the same way, **or** a closed consistently-wound shell has negative volume | inconsistent edge count of total, affected face count, signed volume in m³ |
| E104 | error | Object scale ≠ (1,1,1) or rotation ≠ 0 beyond tolerance | scale, rotation in degrees, mesh bbox vs exported bbox |
| E105 | error | Verts in no face or edge, wire edges, faces under `min_face_area`, faces repeating a vertex | count per kind, area per offending face, locations |
| E106 | error | Two UV islands intersect, or one folds onto itself, while `emit` declares a unique bake | overlap area in uv², as a % of the smaller island, mirrored flag, bake target and uv set |
| E107 | error | Triangles after fan triangulation exceed `budget.tris_lod<n>` | tris, budget, overage and %, triangles per named group |
| E108 | error | Component count ≠ `emit.components` | actual vs declared, tri count / volume / extents per shell, largest shell's share |
| W201 | warn | Origin sits further than `tolerance_mm` from the declared reference point | offset per axis in mm, distance, the reference the geometry *does* match |
| W202 | warn | An island-to-island or island-to-border gap falls under the padding table for the declared resolution | worst gap in px, required px, per-pair gaps |
| W203 | warn | A declared texture dimension is not a power of two | actual size, nearest power of two, pixel-count delta |
| W302 | warn | An island's texel density leaves target ±`tolerance_pct` | px/cm per island, % off target, observed range, allowed band |
| N401 | note | An object or texture name does not match `[Prefix]_[Name]_[Variant]_[LOD]` | which segment failed and why, plus a corrected name |

Rule parameters live in the profile, not in the rule body:

| Rule | Parameters (defaults) |
|---|---|
| E101 | `allow_boundary=False`, `max_report=8` |
| E102 | `max_face_verts=4` |
| E103 | `max_report=8` |
| E104 | `scale_tolerance=1e-6`, `rotation_tolerance_deg=1e-4` |
| E105 | `min_face_area=1e-9` |
| E106 | `min_overlap_fraction=0.005`, `max_pairs=6` |
| E107 | `tolerance_pct=0.0` |
| E108 | `debris_ratio=0.8` |
| W201 | `tolerance_mm=1.0` |
| W202 | `padding_table={256:2, 512:2, 1024:4, 2048:8, 4096:16, 8192:32}` |
| W203 | `require_square=False` |
| W302 | `tolerance_pct=10.0` |
| N401 | `object_segments`, `texture_segments`, `default_prefix="SM"`, `check_textures=True` |

Three profiles ship. `character_skinned` tightens origin and transform
tolerance, loosens texel density, and downgrades E108 to a warning. Faces get
more texels on purpose, which is why the density band widens. `mobile_vr` allows
open shells for foliage cards. It also needs triangles rather than quads, and it
grades texel density and NPOT textures as errors. Both profiles are pure data in
[`profiles.py`](profiles.py). The engine has never heard of either.

## Example report

Real output. `overlapping_uvs` and `texel_variance`, folded into one asset:

```
check  SM_Crate_Wood_LOD0   profile game_prop_pbr   FAIL   1 error, 1 warn, 0 note

✗ E106  Overlapping UVs on unique bake
        object   : SM_Crate_Wood_LOD0.side_L
        islands  : side_L overlaps side_R (100% area)
        uv       : side_L bbox (0.0167, 0.1000)→(0.3167, 0.4000)   side_R bbox (0.0167, 0.1000)→(0.3167, 0.4000)
        evidence : bake target = normal, uv_set = UV0; overlap = 0.09 uv² of 0.09 uv² (100%); islands are mirrored
        repair   : uv.unwrap(SM_Crate_Wood_LOD0, split_mirrored = true)
                   — or emit with bake_uv_set = UV1
                   — or uv.pack(SM_Crate_Wood_LOD0, islands = [side_L, side_R], margin_px = 4, resolution = 1024)

⚠ W302  Texel density variance 4.096–5.12 px/cm (target 5.12 ±10%)
        object   : SM_Crate_Wood_LOD0
        islands  : top
        measured : top            4.096 px/cm  (-20% off target)
        evidence : target 5.12 px/cm at 1024², allowed 4.608–5.632; 1 of 6 islands outside, observed 4.096–5.12 px/cm
        repair   : uv.rescale(islands = [top], to = 5.12)
                   — or emit with texel_density = 4.949 px/cm if the current layout is intended
```

Now the rule that exists because the doc's own example was broken. This is
`python -m lathe.verify severed_crate`:

```
✗ E108  Component count 6, declared 1
        object   : SM_Crate_Wood_LOD0
        shells   : component_00  12 tris  0.00576 m³  extents 0.016 × 0.6 × 0.6 m
                   component_01  12 tris  0.03852 m³  extents 0.107 × 0.6 × 0.6 m
                   component_02  12 tris  0.03852 m³  extents 0.107 × 0.6 × 0.6 m
                   component_03  12 tris  0.03852 m³  extents 0.107 × 0.6 × 0.6 m
                   component_04  12 tris  0.03852 m³  extents 0.107 × 0.6 × 0.6 m
                   component_05  12 tris  0.00576 m³  extents 0.016 × 0.6 × 0.6 m
        evidence : components = 6, declared = 1; 72 tris split 12/12/12/12/12/12; largest holds 16.667% of the triangles
        repair   : assert components(SM_Crate_Wood_LOD0) == 1
                   — or clamp the cutter extent: a cutter spanning the full body severs it. DESIGN.md §4 uses `size - rail * 2`, not `size * 1.1`
                   — the pieces are of comparable size, so this is a severed body, not debris

not checked
        E106  emit declares no unique-uv bake, so overlap is legal
        W202  asset declares no uv set at all
        W203  emit declares no textures
        W302  emit declares no texel_density target
```

Four fields carry the weight. The `Finding` constructor rejects a finding that
lacks any of them:

| Field | Why |
|---|---|
| **code** | stable, citable, greppable in a studio wiki |
| **entity** | the offending *named* thing — `SM_Crate_Wood_LOD0.side_L`, never `face 4471` |
| **evidence** | numbers with units, so the agent can tell whether its edit helped |
| **repair** | a call it can paste, with alternatives when the geometry and the declaration are both candidates |

`report.to_dict()` carries the same data with the numbers as numbers. Locators
are world-space coordinates, not indices. An index does not survive the next
evaluation of the program (§5.1). A position does.

### The "not checked" section

The report lists the rules that had nothing to compare against. A rule with no
declaration reports itself as unchecked. It never passes silently. A silent pass
and a real pass look the same in a report. That is how "we checked the UVs"
becomes "there were no UVs". On the fully declared `clean_prop` fixture, `check`
skips nothing, and the suite asserts it. A clean report means nothing unless
every rule ran.

### Repair vocabulary

The rules write repairs in the §4 IR dialect. This is the full set they emit.
Treat it as a proposal for the phase-2 repair API, not as an existing surface:

```
mesh.weld · mesh.delete_faces · mesh.delete_loose · mesh.dissolve_degenerate
mesh.triangulate · mesh.quadrangulate · mesh.recalculate_normals
mesh.flip_normals · mesh.split_vertex · mesh.fill_holes
solid.keep_largest_component · solid.union · solid.split
transform.apply · transform.set_origin
uv.unwrap · uv.pack · uv.rescale
lod.decimate · texture.resize · texture.rename · emit.rename
assert components(x) == n
```

The evidence picks the repair, not the code path. E106 suggests `split_mirrored`
when the islands measure as mirrored, and `uv.pack` when they do not. E108
suggests `keep_largest_component` when one shell holds over 80% of the
triangles. Otherwise it points at the cutter extent. E105 leads with
`dissolve_degenerate` when nothing loose exists to delete, because
`delete_loose(verts = false, edges = false)` is a no-op. An agent that runs a
no-op repair reports the finding as fixed.

## Neutral mesh, not bpy

Rules read [`AssetMesh`](mesh.py): vertices, variable-length faces, per-corner
UVs, an object transform, and named face groups. They never touch `bpy`.

- The emit path uses Manifold (§11). So the verifier must read a triangle soup
  from a kernel that has never heard of Blender.
- `bmesh` only works inside a Blender process. Binding the rule engine to it
  would hide the most valuable subsystem behind the global state §1 indicts. It
  would also make `check()` cost an interpreter start.
- A rule over plain arrays takes hand-built broken fixtures. A rule over `bmesh`
  needs Blender to run its own unit test. This suite runs 92 tests in 5 s.

Blender stays a supported *source*. `from_bpy_mesh` is 40 lines, and it maps
material slots to face groups. [`blender_smoke.py`](tests/blender_smoke.py) runs
this engine inside Blender against a bpy-sourced asset. All 10 checks pass,
including zero findings on a clean bpy asset.

Faces are variable-length index tuples rather than triangles, on purpose.
**E102 cannot exist as a rule if the representation cannot hold an n-gon.** The
verifier has to see what the exporter sees. Here the verifier and the perception
layer diverge. `lathe/perceive/mesh.py` is triangles-only. That is right for
probes, and it could not host E102. See proposed change 8.

## Files

| File | Lines | What |
|---|---|---|
| `engine.py` | 299 | registry, `Profile`, `RuleContext`, `run()`. Knows no rule names |
| `findings.py` | 379 | `Severity`, `Repair`, `Finding`, `Report`, text and JSON rendering |
| `mesh.py` | 393 | `AssetMesh`, `EmitSpec`, `Transform`, Manifold and bpy adapters |
| `topology.py` | 244 | edge maps, manifoldness, bowties, components, degeneracy |
| `uvmap.py` | 183 | UV islands, island naming, per-island area and texel density |
| `geom.py` | 144 | polygon area, Newell normals, convex clip, segment distance |
| `profiles.py` | 150 | `game_prop_pbr`, `character_skinned`, `mobile_vr`. Data only |
| `rules/*.py` | 1590 | the 13 rules, grouped by what they read |
| `fixtures.py` | 544 | 21 assets: 3 clean, 18 broken, one per failure mode |
| `tests/*.py` | 1131 | 92 tests plus the Blender bridge script |

A new rule is a decorated function. A new profile is a tuple of codes. Neither
touches `engine.py`. `test_engine.py` proves it: the test file defines a profile
and runs it.

## Fixtures

Each broken asset trips exactly the codes it targets. The expectation table
lives in `fixtures.EXPECTED`, beside the geometry. The suite asserts the *exact*
code set. So a rule that starts over-reporting fails a test. It does not quietly
train the agent to ignore findings.

| Fixture | Codes | How it is broken |
|---|---|---|
| `clean_prop` | *none* | watertight quads, identity transform, 6 islands at exactly 5.12 px/cm, POT maps |
| `clean_solid` | *none* | the same without UVs or textures — 4 rules report *not checked* |
| `fixed_crate` | *none* | the §4 crate with the prototype's cutter fix |
| `nonmanifold_edge` | E101 | two boxes welded along one edge: 4 faces on it, nothing else open |
| `bowtie_vertex` | E101 | two boxes welded at one corner: every edge manifold, one vertex two fans |
| `open_shell` | E101 | top face deleted |
| `ngon_prism` | E102 | closed hexagonal prism, n-gon caps |
| `flipped_face` | E103 | one face reversed |
| `inside_out` | E103 | every face reversed — consistent, so only volume sees it |
| `unapplied_transform` | E104 | scale 0.01, yaw 45° |
| `loose_geometry` | E105 | one orphan vertex, one wire edge |
| `zero_area_face` | E101 E105 W201 | a collinear triangle: degenerate, open, *and* it moves the declared origin |
| `overlapping_uvs` | E106 | `side_R` mirrored and stacked on `side_L` |
| `over_budget` | E107 | 2,028 tris against 1,500 |
| `severed_crate` | E108 | **the §4 crate with `size * 1.1` cutters** |
| `origin_offset` | W201 | centred geometry, `bottom_center` declared |
| `tight_padding` | W202 | islands grown to 0.331 uv — a 2.4 px gap where the table wants 4 |
| `npot_textures` | W203 | one 1000 × 1024 map |
| `texel_variance` | W302 | `top` island at 0.8 scale, so 4.096 px/cm |
| `bad_name` | N401 | `crate_final` |
| `multi_fault` | 8 findings | several faults at once, to check the report stays ordered |

`severed_crate` is not a toy. `prototype/README.md` measured DESIGN.md's own §4
crate as six loose slabs. The cutters spanned `size * 1.1` in Z as well as Y.
That measurement produced E108. This fixture rebuilds the case through
`manifold3d`. The suite asserts 6 components when severed and 1 when fixed. Both
numbers match the prototype.

## Honest notes

**Two rules are approximations, and they say so in code.** `RuleSpec.approximation`
holds the text, and `--table` prints the flag.

- **W202 (padding)** measures the distance between island outlines in UV space,
  then multiplies by the declared resolution. It does not rasterise. So it
  cannot see bleed that appears at one mip level only. It also assumes square,
  non-tiling textures. The padding table is a convention, not a measurement.
  That is why it is a profile parameter.
- **W302 (texel density)** averages per island: `sqrt(uv area) × resolution /
  sqrt(world area in cm)`. An island that is stretched inside itself can average
  onto target and pass. A per-face measure would catch that. It would also flag
  every legitimately foreshortened face. The average is the deliberate choice.
  Real texel density wants a per-face histogram with a stretch metric. That is a
  bigger rule than this one.

**The other rules measure exactly, within a limited scope.** Worth stating
plainly:

- **E101** treats a boundary edge as a defect by convention, not by topology. An
  open shell is perfectly manifold. So `allow_boundary` is a profile parameter,
  and `mobile_vr` sets it true. Edges with more than 2 faces and bowtie vertices
  stay unconditional.
- **E103** can only tell inside from outside on a shell that is closed *and*
  consistently wound. It cannot see an inverted open shell, such as a card that
  faces away. It checks winding consistency unconditionally. The volume test
  waits for consistency.
- **E106** fans n-gon UVs into triangles before the clip. That is exact for
  convex polygons and approximate for concave ones. The `min_overlap_fraction`
  threshold hides a sliver overlap under 0.5% of the smaller island.
- **E108** counts vertex-connected components. Two shells welded at one vertex
  count as one. The `bowtie_vertex` fixture is that case, and E101 reports it
  instead. Manifold's severed output has separate vertex sets, so the crate case
  still works.
- **E105**'s `min_face_area` is absolute in m², not relative to asset scale. A
  1 km asset has legitimately small faces. A profile for one should raise the
  threshold.
- **E107** counts triangles after *fan* triangulation. An engine importer may use
  a different triangulator, and on concave n-gons it can land 1-2 triangles
  apart. Ship triangles, as `mobile_vr` needs, and the ambiguity disappears.
- **N401** is a regex over segments. It cannot know that `Chair` is the right
  name for a chair. It checks shape only. Its suggestion always satisfies the
  rule, and the suite asserts that for six adversarial names. `Cube.001` is one
  of them, because the first version produced an invalid suggestion for it.

**Not implemented from the §7 table.** W204 (LOD chain steps) and W205
(silhouette deviation) need a LOD chain and a rasteriser. The asset model here
holds one mesh. Both rules belong with §6 perception. N402 (dense topology in
low-curvature non-deforming regions) needs a curvature estimate and a
deformation map. Without those it would flag every bevel. Reserved, not stubbed.

**Performance.** The engine computes topology once per asset and shares it
across rules. Measured on this machine: 2 ms for a 12-triangle asset, and 143 ms
for 2,028 triangles with UVs. The UV overlap broad phase dominates the second
number. It cost 1,054 ms before a bbox pre-filter and scalar area maths. That is
fine for props and wrong for a 100k-triangle character. The overlap pass wants a
real spatial index, and the topology wants to be incremental. Nobody has run any
of this at production density. `prototype/README.md` records the same caveat.

## Proposed DESIGN.md changes

Not applied. This pass writes only inside `lathe/verify/`.

1. **§4 and §7 name the example asset `SM_Crate_Wood`, and §7's own N401 rule
   rejects that name.** Verified against the implementation:

   ```
   • N401  Naming: 'SM_Crate_Wood' does not match [Prefix]_[Name]_[Variant]_[LOD]
           problems : 3 segments, expected 4 ([Prefix]_[Name]_[Variant]_[LOD])
                      lod missing — LOD0 for the base mesh
           suggest  : SM_Crate_Wood_LOD0
   ```

   Rename the example to `SM_Crate_Wood_LOD0`, or drop the `[LOD]` segment
   requirement for LOD0. This is the same class of finding as the prototype's
   E108 discovery. The doc's example does not satisfy the doc's rules.

2. **E108 has no home in the `emit` block.** §4 puts `assert components(carved)
   == 1` in the program body. The verifier runs on the emitted asset, and it
   needs that number at emit time. Either the emitter carries the last assertion
   on the emitted solid forward, or `emit` grows `components: 1`. This
   implementation reads `EmitSpec.components`. When it is absent, the report says
   *not checked* and nags.

3. **Severity belongs to the profile, not the rule.** `mobile_vr` wants W302 and
   W203 as errors, and W201 as a note. `character_skinned` wants E108 as a
   warning. Read §7's Sev column as the `game_prop_pbr` grading, and say so.

4. **Add *not checked* to the report specification.** §7 describes findings, but
   not applicability. Without it, a profile skips every rule whose declaration is
   missing, and the report still reads as a pass. It costs three lines in the
   format. It is what makes the gate honest.

5. **State the gate.** Errors block export. Warnings and notes do not. The
   severity names imply this, and the doc never says it.

6. **§11 says Manifold "eliminates E101 by construction". That holds for the
   boolean output, not for the pipeline.** Imported geometry can reintroduce
   non-manifold edges. So can whatever implements bevel, since §14 records that
   Manifold has none. So can a hand edit in Blender, or an exporter weld. Say
   "eliminates E101 from boolean output". The rule stays cheap, and it stays in
   the profile.

7. **E104's wording should exclude translation.** "Object scale ≠ (1,1,1) or
   rotation ≠ 0" is exactly right. But "unapplied transform" colloquially
   includes location, and location is W201's job. One clause fixes it. Otherwise
   an agent reading the table expects E104 to catch a misplaced origin.

8. **Two subsystems, two neutral mesh types.** §6 and §7 both need one. Another
   process built `lathe/perceive/mesh.py` beside this package. Both reached the
   same decision: a neutral value type, Manifold as an adapter, headless. The two
   shapes are incompatible. Perceive's mesh is triangles-only, which suits probes
   and cannot hold an n-gon. Have §11 specify **one** shared neutral mesh:
   polygonal, with a triangle fast path. Do it before the two drift. The verifier
   needs polygons for E102, and the probes lose nothing by having them.

9. **Three codes the fixtures wanted and the table lacks.** First, a unique bake
   declared with no UV set at all. The report says *not checked* today, which is
   honest but weak. Second, UVs outside 0–1 on a unique bake. Third, a face that
   repeats a vertex, which E105 absorbs here. The first two are silent failures
   of exactly the kind §7 exists to catch.
