# Selector layer prototype

Tests section 5 of [DESIGN.md](../DESIGN.md) against both candidate geometry kernels, to
find out whether provenance tagging needs a forked backend. It does not.

```bash
"/c/Program Files/Blender Foundation/Blender 5.2/blender.exe" -b -P prototype/run_experiments.py
python prototype/manifold_provenance.py
```

`lathe_proto.py` is a working slice of the design: pure functions, no `bpy.ops`, no mode
switches, no selection state. Tags live in an integer face attribute as a bitmask.

## Blender 5.2.0 LTS

| Claim | Result |
|---|---|
| 5.2 Tags survive EXACT boolean | PASS — 0 untagged faces leaked |
| 5.2 `preserve_all_data_layers` needed? | No — identical either way |
| 5.2 Tags survive `bmesh.ops.bevel` | PASS |
| 5.1 Tag-filtered selectors resolve | PASS |
| 5.1 `#cut_face` stable across bevel, `inherit=True` | **FAIL — 2 to 10 matches** |
| 5.1 `#cut_face` stable across bevel, `inherit=False` | PASS — 2 to 2 |
| 5.3 Cardinality assert fires on param change | PASS — rim 64 to 88 edges |

## Manifold (`manifold3d`)

| Claim | Result |
|---|---|
| 5.2 Tags survive boolean via `run_original_id` | PASS — exact split, 20 + 4 tris |
| 5.2 Three ancestors through chained booleans | PASS — all distinguishable |
| 5.1 ID tracking drifts? | No — exact attribution, not interpolated |
| 5.1 `face_id` sub-original granularity | 6 logical faces from 20 triangles |
| 7 Fixed crate is one solid | PASS — see below |

Manifold is the better kernel for this. `reserve_ids` exists precisely to mark triangle
sets that stay queryable after later operations, `run_original_id` attributes every output
triangle to its source solid, and `face_id` groups coplanar triangles into logical faces —
which is what `seat.top_face()` needs.

Two footguns. `as_original()` deliberately drops ancestor relations, so it destroys
provenance. And Manifold ships no bevel or fillet, so the drift risk returns in whatever
implements those.

## The two findings that changed the design

**1. Attribute inheritance causes silent selector drift.** `bmesh` interpolates ancestor
attributes onto new geometry, so bevelling a cut edge makes the new bevel faces answer to
`#cut_face` as well — 2 matches became 10, with no error. The selector kept resolving. It
just quietly meant something else.

The fix is a default, not a check: operations tag new geometry with their own tag only,
and a program wanting ancestry must ask for it. A cardinality assertion would not have
caught this, because nobody asserts on a selector they did not know had moved. DESIGN.md
§13 previously claimed assertions were the primary mitigation. They are the backstop.

**2. The doc's own worked example was broken.** The section 4 crate used cutters spanning
`size * 1.1` in both Y and Z, which severs the body instead of slotting it:

```
size*1.1 (as first written)    components=6   genus=-5   SEVERED
size-rail*2 (fixed)            components=1   genus=5    OK
```

Six loose slabs, not a crate. Nothing in the program said otherwise, which is the whole
argument for §7 — so this produced a new rule (E108, unexpected component count) and an
`assert components(carved) == 1` line in the example.

## Caveats

- M3 does not convict property channels. The test sets the channel uniformly per solid, so
  interpolation lands on an identical value and hides itself. Prefer IDs on first
  principles, not on that measurement.
- Everything here is eager resolution on trivial meshes. It says nothing about lazy
  selectors or performance at production density.
- `to_mesh()` returns read-only arrays and `np.ascontiguousarray` is a no-op on them, so
  rebuilding a tagged `Mesh` needs an explicit `copy=True`. Cost an hour.
