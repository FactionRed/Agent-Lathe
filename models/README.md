# Test assets

Three assets built with the tool, to find out whether it is usable rather than
merely designed.

```bash
python models/build_all.py              # build, verify, probe, render all three
python models/build_all.py chair        # one asset
python models/build_all.py --no-render  # skip Blender, keep the checks
```

## Results

| Asset | Tris | Components | Verifier | Build |
|---|---|---|---|---|
| `SM_Chair_Oak_LOD0` | 2,324 | 1 | PASS — 0 error, 0 warn, 0 note | 2.6 s |
| `SM_BowlFruit_Ceramic_LOD0` | 5,848 | 1 | PASS — 0 error, 0 warn, 0 note | 2.8 s |
| `SM_Villager_Stylized_LOD0` | 2,446 | 1 | PASS — 0 error, 0 warn, 0 note | <0.1 s |

All three are watertight, manifold, stand on z=0, and are centred on their
declared origin. The chair measures 1.000 symmetry about YZ by exact volumetric
IoU. Build time is dominated by the bevel, which pays 2.5 s of Blender process
startup per call (DESIGN.md §14.8).

## The villager is not a character

DESIGN.md §10 says procedural geometry does not have animation-grade edge flow,
and that the real path for deforming assets is template basemeshes with cage
deformation. That limit held. The villager is a **stylized figure assembled from
primitives** at correct 1.7 m proportions, and it is:

- **not riggable.** No edge loops at shoulder, elbow, hip or knee. A bone
  rotation would pinch and tear it.
- **not a face.** The eyes are placed spheres. There is no brow, nose, mouth or
  ear topology, and no boolean work would produce any.
- **not anatomically continuous.** Limbs meet the torso by intersection, so the
  silhouette carries seams a sculpt would not.

Read it as prop-grade geometry in the shape of a person. It would serve as a
distant background villager in a stylized game. It would not survive a close-up
or a rig.

The chair and the bowl have no such asterisk. Hard surface and revolved forms
are what this kernel is actually good at.

## UVs

All three are unwrapped with xatlas and packed to a declared texel density, so E106, W202,
W203, W206 and W302 run against real layouts instead of reporting *not checked*. All three
carry **0 errors**. They also carry warnings that are correct, and worth reading rather than
silencing:

| Warning | Why it is right |
|---|---|
| W302 density variance | Padding and block alignment impose a minimum chart size, so a sliver gets more texels than its area asks for. These meshes carry their bevels in *geometry*, which makes slivers numerous. The production fix is to bake the bevel into a normal map, not to widen the tolerance. |
| W202 padding | xatlas pads chart bounding *rectangles*; the verifier measures outline to outline. A concave chart nests into a neighbour's notch, so outlines come closer than the rectangle gap. Measured at 4, 6, 8 and 12 px padding — reported at all of them. A polygon packer would fix it; a bigger margin does not. |
| W206 collapsed islands | Tiny bevel charts whose UV area rounds to nothing. Same root cause as W302. |

E106 *was* firing on the chair: one chart wrapped far enough around a bevel to overlap
itself. A chart that folds is an under-split chart, so the fix was a tighter chart cost —
0.35 instead of the looser default, 101 charts instead of 84, fold gone. Raising padding did
not help, which is what pointed at chart splitting rather than packing.

## Baking, and where it helps

```bash
python models/bake_all.py     # build the pair, unwrap the low, bake the high onto it
```

`build_all.py` emits the bevelled mesh. `bake_all.py` emits the low poly plus maps, which is
what ships in an engine. Two scripts because they answer different questions.

| Asset | Low | High | Ratio | Effect |
|---|---|---|---|---|
| chair | **308** tris | 2,324 | 7.55:1 | W302 from 202% off target to 17% |
| bowl | 4,822 tris | 5,848 | 1.21:1 | essentially none |

**Baking helps in proportion to how much of the mesh the detail was.** The chair's bevel was
87% of its triangles, so moving it into a normal map is transformative. The bowl's was 18%, so
it barely moves — and the bowl's density problems were never the bevel. They come from the
pole triangles of its spheres and revolves, which no amount of baking addresses.

That is worth knowing before reaching for a bake as a general fix. The verified normal map
shows it plainly: a neutral lavender field with a bright and dark lip on each chart, which is
the arris the low poly no longer has.

## What building these found

**Assertions earn their place immediately.** Two of the three assets failed a
§5.3 cardinality assertion on the first run, and both times the assertion was
right and the expectation was wrong. `slat_n * 5` should have been `slat_n * 4`,
because both ends of every slat are buried in the stiles, not one.

**Logical face counts are tessellation-independent only for planar faces.**
`assert count(faces(#bowl)) == 1` failed with an actual of **322**. `face_id`
groups *coplanar* triangles, so a 64-segment revolve reports one facet per
segment. The count scales with segment count. Assert on logical faces for planar
hard surface; on curved surfaces assert the invariants that hold instead.
`Selection.count()` carries a comment claiming tessellation independence — true
for the planar case only, and now qualified.

**Coincident contact is not connection.** The villager's boots met the shins
exactly on z=0.075 and came back as separate components. A boolean treats
measure-zero contact as disjoint. Overlap deliberately, or accept the split.

**Cutter tags reach the result, and that is correct.** Trimming the hair volume
leaves a cut surface whose triangles carry the *cutter's* tag, per §5.2. Left
unmapped they shaded default grey. The fix is a material for the cut tag, not
the removal of the tag — the cut face genuinely is a distinct surface, and being
able to address it is the point.

**A declaration copied from a measurement checks nothing.** The bowl first
declared `components=None` and filled it from the geometry, which made E108
vacuous. It now declares 1.

## Renders

`out/*.png`. Contact sheets, never single views, per §6. `chair_modes_*.png`
shows the two diagnostic modes: `matte` for silhouette, `wire` for topology
density. On merged-coplanar hard surface the wireframe is sparse, because flat
faces really are single quads — the emptiness is the reading.
