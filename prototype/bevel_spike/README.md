# Bevel / fillet spike

Manifold ships no bevel and no fillet. [DESIGN.md](../../DESIGN.md) section 13 names that
the top risk. [prototype/README.md](../README.md) already proved why: bevel is where
provenance breaks. `bmesh.ops.bevel` interpolated ancestor attributes onto new faces. That
widened `faces(#cut_face)` from 2 matches to 10, and raised no error.

This spike builds a minimal working bevel on three candidate kernels. It measures each one
against that failure.

```powershell
python run_all.py                 # every candidate available here
python run_all.py --only C
python run_all.py --json out.json

python approach_b_bmesh.py        # one at a time
python approach_c_sdf.py
<occt-venv>\Scripts\python approach_a_occt.py
```

| file | what it is |
|---|---|
| `common.py` | fixture, geometric referee, validity audit, provenance metrics |
| `approach_a_occt.py` | A — OpenCASCADE `BRepFilletAPI_MakeFillet` via `cadquery-ocp` |
| `approach_b_bmesh.py` | B — driver; spawns `blender -b -P blender_bevel.py` |
| `blender_bevel.py` | B — the in-Blender half, under Blender's own Python |
| `approach_c_sdf.py` | C — `Manifold.level_set`, skimage marching cubes, Minkowski |
| `run_all.py` | runner; picks the interpreter per approach, prints the headline |

## Method

**Fixture.** A 600 mm cube. A 150 mm slot runs across the top, 150 mm deep. Manifold cuts
it with `reserve_ids`, so the body and the cutter arrive as distinct `run_original_id` runs.
One connected component, per the `m5()` lesson in `manifold_provenance.py`. A cutter that
spans the full height severs the body instead of slotting it. 28 triangles, 10 logical
faces, genus 0, volume 0.202500 m³.

The cut surface is 3 logical faces. Two slot walls at x = ±75 mm, and the floor at
z = +150 mm. **`faces(#cut_face)` must equal 3 before the bevel and 3 after it.** That one
number is the drift test.

**The referee.** No approach grades itself. Each one returns `(verts, tris, tri_tag)`.
`tri_tag` is *its own claim* about provenance, a bitmask of `#body`, `#cut_face` and
`#bevel`. Then `common.referee()` recomputes the truth from geometry alone. A triangle lies
on a reference plane when its normal is parallel to that plane and its centroid sits on it.
Everything else is new bevel surface. Body and cut planes share axes, so the offset
separates them. x = ±300 mm is body. x = ±75 mm is slot wall. Scoring the claim against the
referee gives false positives and false negatives. A false positive is bevel surface that
answers to `#cut_face`. A false negative is real cut surface that lost its tag.

**Validity.** `common.check_mesh()` audits every result independently of the kernel that
made it. It checks half-edge pairing, degenerate and duplicate triangles, component count,
Euler genus, divergence volume and coincident vertex positions. It then re-imports into
Manifold and runs one more boolean. "Can this still be a boolean operand" is the property
that matters downstream. It reports the audit twice, on the producer's own indices and
after a 1e-7 weld. Welding a surface that legitimately touches itself manufactures a fake
non-manifold edge. This spike made that mistake once. See the notes below.

Radius 20 mm throughout. `segments=3` where segments are a parameter.

## Headline

`faces(#cut_face)`, which must stay at 3.

| approach | before | after | wrong tris | tris | time |
|---|---|---|---|---|---|
| **A** OCCT fillet, B-Rep history | 3 | **3** | **0** | 5 952 | **0.10 s** |
| **B** bmesh, `tag_mode=raw` (bmesh's own default) | 3 | **78** | 97 | 380 | 3.07 s |
| **B** bmesh, `tag_mode=inherit` | 3 | **78** | 97 | 380 | 3.20 s |
| **B** bmesh, `tag_mode=exclusive` (DESIGN.md §5.2) | 3 | **3** | **0** | **380** | 3.19 s |
| **C** `level_set` L=30 mm | 3 | **429** | n/a | 10 356 | 1.05 s |
| **C** `level_set` L=20 mm | 3 | **317** | n/a | 23 944 | 3.47 s |
| **C** `level_set` L=10 mm | 3 | **1 301** | n/a | 94 448 | 26.03 s |
| **C** Minkowski open, 16 seg (convex only) | 3 | 4 | 0 | 1 360 | 0.23 s |
| **C** Minkowski close, 16 seg (concave only) | 3 | **3** | 0 | 566 | 0.35 s |
| **C** Minkowski open+close, 16 seg | 3 | 3 | 0 | 16 406 | 2.06 s |

"Wrong tris" counts triangles that an approach labels `#cut_face` and the referee places on
new bevel surface. The `level_set` rows say n/a because there is no claim to score. Those
routes destroy provenance, so it must be re-derived from geometry first. The "after" column
already shows what that re-derivation returns.

## Full results

| | A — OCCT fillet | B — bmesh bevel | C — implicit / SDF |
|---|---|---|---|
| **1. Correct, manifold geometry** | Yes. Watertight, 1 component, genus 0, volume 0.201840 m³. Valid Manifold operand. | Yes. Watertight, 1 component, genus 0, volume 0.201736 m³. Valid Manifold operand. | `level_set` yes at every resolution (marching tetrahedra, manifold by construction). skimage MC watertight with 0 non-manifold edges here, but no general guarantee. Minkowski `open` and `close` are clean. Composed `open+close` leaves 52 coincident vertex positions, so a weld turns it into 13 non-manifold edges. |
| **2. Provenance survives** | **Exactly.** `Modified()` and `Generated()` attribute all 10 post-boolean faces (7 body, 3 cut). They attribute all 50 post-fillet faces: 10 tracked, 24 fillet strips from `Generated(edge)`, 16 corner patches. 0 false positives, 0 false negatives, 0 untagged. | **Only if the wrapper forces it.** `run_original_id` maps to a face int attribute that survives the bevel. But bmesh copies ancestor attributes onto 184 new faces by itself: 133 got `#body`, 51 got `#cut_face`. With `exclusive` tagging: 0 false positives. | **No.** Every route returns a mesh with exactly **1** `run_original_id`. All ancestor information is gone. Only geometric re-attribution remains. Manifold also exposes no signed-distance or closest-point query, so `level_set` cannot round an arbitrary boolean result at all. |
| **3. Drift** | **None.** 3 → 3. | **97 triangles drift by default.** `faces(#cut_face)` goes 3 → 78. The §5.2 non-inheriting default fixes it: 3 → 3. `inherit` mode keeps `faces(#cut_face)` at 3 and sends `faces(~cut_face)` to 78. That is the generation semantics §5.2 asks for. | **Severe.** `level_set` gives 3 → 429, 317 and 1 301 logical faces at L = 30, 20 and 10 mm. That is not monotonic, and it is worst at the finest resolution. The flat faces are no longer flat, so they shatter into per-triangle groups. Minkowski `close` is exact (3 → 3). `open` gives 4. |
| **4. Triangles, runtime, density** | 908 / 1 444 / 5 952 / 51 324 tris at angular deflection 0.7 / 0.5 / 0.35 / 0.2 rad. Whole path (build, select, fillet, mesh) takes 0.09 s for 1 slot, 0.21 s for 3, 0.34 s for 5. | 380 tris. The bevel itself takes 0.6 ms. A **2.5 s process startup floor** dominates, giving ~2.9 s wall per call. Dissolve normalises input density: 55 970 in, 10 logical faces, 8 492 out, bevel 58 ms, wall 6.2 s. | `level_set` takes 26 s for 94 448 tris at L=10 mm, over 731 159 SDF calls through a Python callback. Vectorised numpy plus skimage MC takes 0.55 s for 178 816 tris at L=5 mm, about 100× faster. Minkowski `open` takes 0.18 / 0.41 / 0.72 s for 1 / 3 / 5 slots, giving 1 372 / 3 132 / 4 942 tris. |
| **5. Per-edge width, segments, predicate** | **All three.** `Add(radius, edge)` sets one radius per edge. A mixed 30 / 8 mm run gives a distinct valid solid. `Add(r1, r2, edge)` varies radius along one edge: measured fillet area 0.01648 m² sits between the constant-radius bounds 0.00471 and 0.02827, so the law is linear. Segments are not a parameter, because arcs stay exact until emit. Predicate: 24 sharp edges, split 22 convex / 2 concave. | Segments yes. Predicate yes: 24 sharp edges, 22 convex / 2 concave via `calc_face_angle_signed`. **Per-edge width no.** `bmesh.ops.bevel` takes one scalar `offset`. Two passes work: convex at 30 mm, then concave at 8 mm, valid and provenance-exact. But the second pass re-resolves its predicate against topology the first already changed, and the concave count moved 2 → 6. | **None.** One radius per primitive plus one blend radius per boolean. No per-edge width, no predicate, no segment count. Tessellation density is a single global grid spacing. |
| **6. Install and platform risk** | Heavy. `pythonocc-core` has **no PyPI distribution at all**. pip answers `No matching distribution found`, so conda is the only route. `cadquery-ocp` 7.9.3.1.1 does install: 52 s, **626 MB** venv. OCP is 159 MB of that. **VTK is 328 MB, and the fillet path never touches it.** VTK is unavoidable, because `OCP/__init__.py` calls `os.add_dll_directory(vtk.libs)` at import. `--no-deps` therefore raises `FileNotFoundError`. Import costs 1.5 s. | Blender is already installed. DESIGN.md §11 already needs it as viewer and exporter. No pip dependency. The cost is a subprocess boundary and 2.5 s per call. | **Zero for Minkowski and `level_set`.** Both already ship in `manifold3d` 3.5.2, which is already the kernel. `scikit-image` is optional, for vectorised marching cubes: 33 s, and it left numpy untouched. |

### Details worth keeping

**A — OCCT.** The B-Rep history API has the right shape for §5.2. One trap: OCCT omits an
unmodified face from `Modified()`. A face that survived untouched needs a lookup by
identity. Missing that made 3 of a box's 10 faces look like new geometry.

Radius accuracy is exact. Filleting one 600 mm edge at 30 mm gives a measured new surface of
0.028260 m². The analytic quarter cylinder is 0.028274 m², so the error is 0.05%. Flat faces
stay on their nominal planes to 0.0.

Failures are loud and localised. Filleting a 5-slot crate at r=20 mm with 25 mm walls fails
with `IsDone=False`, `NbFaultyContours=1` and `FaultyContour=[78]`. At 5 mm walls it reports
24 faulty contours and `ChFiDS_StartsolFailure`. Both cases are genuinely infeasible, since
wall < 2r. OCCT names the offending contour. That is enough to build the bevel forensics
diagnostic DESIGN.md §7 wants.

**A5 is the architectural catch.** OCCT fillets a *feature history*, not a mesh. Sewing
Manifold's 28 triangles into a B-Rep gives 28 one-triangle faces and 42 edges. The dihedral
predicate still finds 24 of them. But each logical cube edge now arrives as several
collinear segments. `BRepFilletAPI_MakeFillet` then **reports `IsDone` and returns garbage**:
74 faces, non-watertight, `NotManifold` on re-import, in 0.68 s. That is 7× the whole native
path. So adopting OCCT fillets means building the whole solid in OCCT. That is a second
boolean kernel, with the opaque-failure property DESIGN.md §11 already flags. It is not a
post-process on a Manifold result.

**A6 — the bridge works.** The labelled OCCT result re-enters Manifold's native channels.
`reserve_ids` plus `run_index` and `run_original_id` carry the tags. `face_id` carries the
B-Rep face index. All of it survives another boolean. The census reads
`{body: 98, cut_face: 6, bevel: 5860, probe: 8}`, with 0 unattributed triangles and 51
distinct `face_id` values for 50 B-Rep faces plus the probe. One step is mandatory first:
weld the vertices. OCCT emits a separate vertex array per face, so the raw mesh is a pile of
disconnected patches, and Manifold rejects it as `NotManifold`.

**B — bmesh.** The drift belongs to bmesh, not to the wrapper. `tag_mode=raw` sets the face
attribute before the bevel and then touches nothing. The 184 new faces still come back
carrying ancestor masks: 133 `#body` and 51 `#cut_face`. They carry no `#bevel` tag at all.
So a naive port both corrupts `#cut_face` and loses the ability to select the bevel. The
false positives sit 2.0 mm (median) to 8.7 mm from the cut plane. That is inside the 20 mm
bevel band, so they are genuinely new surface wearing an ancestor's tag.

`dissolve_limit` before the bevel is load-bearing and free. It gives 28 → 10, 92 → 26 and
55 970 → 10 logical faces, with 0 untagged. It recovers Manifold's `face_id` granularity
from an arbitrary triangulation.

Two footguns cost real time here. Faces made with `bm.faces.new()` carry a **zero normal**
until `bm.normal_update()`. So `calc_face_angle` returns 0 for every edge, and
`edges(dihedral > 60deg)` silently matches nothing. `dissolve_limit` silently dissolves
nothing too. This is hard to notice, because `dissolve_limit` updates normals as a side
effect, so the next read looks correct.

The second footgun is in the metrics, not the kernel. Refined input appeared to fail
provenance: 4 and 13 wrong triangles at 14 724 and 55 970 input triangles. Those are not
provenance failures. They are 24 to 29 zero-area slivers whose normals are numerical noise,
so the referee cannot place them. Split by area, `false_positive_real_tris` is 0 in every
case. The slivers are still a real defect, but an E105 one.

Where OCCT refuses, bmesh **succeeds quietly**. The 5-slot, 100 mm-slot case has 5 mm walls
and a 20 mm radius. OCCT rejects it with 24 named faulty contours. bmesh returns a
watertight `NoError` mesh carrying 42 degenerate triangles.

**C — implicit.** The problem is not accuracy. It is that there is nothing left to carry.
Every route rebuilds the mesh from scratch and returns 1 `run_original_id`.
`Manifold.level_set` takes a *callable* SDF. The Manifold API has no signed-distance or
closest-point query. This spike checked that against the full API listing. So an implicit
round cannot reach an arbitrary boolean result. The whole model must be an SDF. That is a
different IR, not a bevel operation.

Even the unrounded baseline degrades the shape. `level_set` at L=30 mm on the *exact* SDF
already moves flat faces up to 1.9 mm off their planes. Rounded, the smooth subtraction
pulls the top face **5.04 mm inward** against a requested 20 mm blend. Nothing bulges
outward, since the maximum outward deviation is −5.6e-14 mm. So the smooth-min parameter is
a blend strength, not a width. OCCT's error on the same job is 0.05%.

**Minkowski morphology** is the one genuinely interesting result in this family. It needs no
new dependency. Erode-then-dilate with a 16-segment sphere rounds every convex edge in
0.18 s and produces 1 360 triangles. It leaves the flat faces **exactly** on their nominal
planes, with a worst offset error of 1.2e-8, which is float32 noise. Dilate-then-erode does
the concave edges: 566 triangles, exact planes, and `faces(#cut_face)` re-attributes to
exactly 3. Each direction handles only one sign of curvature. Composing both degrades badly:
16 406 triangles, 52 coincident vertex positions, 13 non-manifold edges after welding, and
68 sharp edges still left. Raising sphere resolution makes it worse, not better. `open` at
32 segments moves the flat faces 0.38 mm and produces 21 wrong triangles. 16 segments
produced 0.

## Recommendation

**Do the bevel in Blender's bmesh, behind a headless subprocess. Enforce §5.2's
non-inheriting tag default in the wrapper. Do not adopt OCCT for Phase 2.**

The reasoning, heaviest first.

1. **A default fully mitigates the provenance risk, and the real kernel now shows it.**
   `tag_mode=exclusive` gives 3 → 3, with 0 false positives, 0 false negatives and 0
   untagged triangles against an independent geometric referee. The tags then survive
   re-entry into Manifold and a further boolean, with 0 unattributed triangles. So the drift
   in DESIGN.md §13 is a wrapper policy question, not a kernel defect. Reproducing it from
   bmesh's own default also confirms the design's diagnosis. The `inherit` run shows the
   `#tag` and `~tag` generation split in §5.2 behaving exactly as specified.
2. **OCCT is better at bevel and worse at adoption.** It offers exact provenance, exact
   radius, per-edge and variable radius, loud localised failures, and 30× the speed. None of
   it is reachable without rebuilding the model in OCCT, because A5 shows the mesh-to-B-Rep
   bridge returns `IsDone` on garbage. The price is a second boolean kernel with opaque
   failures (DESIGN.md §11), a Manifold-to-OCCT provenance bridge, and 626 MB of install
   including 328 MB of unavoidable VTK. The alternative binding has no PyPI path at all.
   That is a Phase 4 decision about exact CAD interop, not a Phase 2 bevel.
3. **Blender costs nothing extra.** DESIGN.md §11 already ships a `.blend` exporter and
   already uses `blender -b` for Cycles. Bevel adds no dependency, no second kernel and no
   new platform risk.
4. **The implicit family cannot do this job.** Accuracy is not the reason. It destroys the
   addressing scheme the whole system rests on. The drift test goes 3 → 429, 317 and 1 301
   as the grid tightens, so refining does not help. It also cannot reach a Manifold result
   at all.

Design consequences to accept.

- **Per-edge width needs multiple passes.** `bmesh.ops.bevel` takes one scalar offset.
  Group the selection by width, then emit one call per group and re-resolve the predicate
  each time. This works and stays provenance-exact. But the concave count moved 2 → 6
  between passes on this fixture. So the IR must either forbid overlapping width groups or
  demand a cardinality assertion per group. That is §5.3 earning its keep for a second
  reason.
- **Budget one long-lived Blender process, not one per call.** 2.5 s of the 2.9 s is process
  start. The bevel is 0.6 ms. A persistent worker speaking the same npz protocol removes
  about 99% of the cost. This spike did not build one.
- **Run `dissolve_limit` first, always.** It is free. It is also what makes the bevel see
  logical faces instead of triangulation.
- **Add a sliver check to the verifier.** bmesh's `clamp_overlap` degrades silently where
  OCCT refuses. The infeasible case produced 42 zero-area triangles and imported as
  `NoError`. E105 should be a hard gate after every bevel. Add a feasibility precheck too,
  comparing bevel width against local wall thickness. `Manifold.min_gap` can measure that.
  Then the system fails as loudly as OCCT does.
- **Keep Minkowski `open` and `close` as a cheap secondary path.** They suit whole-object
  rounding where selectors do not matter: LOD silhouette softening, draft studies, quick
  variants. 0.2 s, 1 360 triangles, flat faces preserved exactly, no new dependency. Do not
  compose the two directions.

## What could not be tested, and why

- **Curved and freeform input.** Every fixture is axis-aligned planar boxes. The referee
  works by matching triangles to nominal planes. It cannot grade a fillet on a cylinder, a
  revolve or a lofted surface. That is where OCCT fillets usually fail in practice, and
  where bmesh bevel usually looks worst. **This is the largest gap in the spike.**
- **OCCT's B-Rep provenance through OCCT's own booleans on hard cases.** Only box-minus-box
  cuts ran here, and all of them succeeded. DESIGN.md §11 claims OCCT boolean failures are
  opaque. This spike measured only the *fillet* failure diagnostics.
- **A persistent Blender worker.** This spike measured the 2.5 s startup floor. It asserts the
  fix, and does not build it.
- **`harden_normals`, `miter_outer`, `miter_inner`, custom profiles.** All runs used
  `harden_normals=False`. DESIGN.md §4's worked example passes `harden_normals = true`. Its
  interaction with the exporter's normal handling stays unmeasured.
- **UV and material behaviour through a bevel.** Untouched. Both approaches would have to
  carry a UV channel across the npz boundary. `bmesh.ops.bevel` interpolates UVs onto new
  faces. That is the same inheritance mechanism that caused the tag drift, so it deserves
  the same scrutiny.
- **Scale.** The largest input is 55 970 triangles, and the largest output 178 816. Nothing
  here says anything about a 500k-triangle environment kit, or about memory.
- **Manifold's `smooth_out` plus `refine` path.** `smooth_out(min_sharp_angle,
  min_smoothness)` claims that values above 0 give a small fillet on sharp edges. It is a
  fourth candidate. It already ships in the kernel. It went unmeasured. It has no width
  parameter and no edge predicate, so it looked like a subset of C's control problem. That
  is a judgement, not a measurement.
- **`skimage.marching_cubes` manifoldness in general.** It measured clean here, with 0
  non-manifold edges at three resolutions. Marching cubes has genuinely ambiguous
  configurations. Three passing cases are not a guarantee.
- **Determinism across runs and platforms.** DESIGN.md §2 principle 1 needs byte-identical output. This
  spike checked no approach for it. The Blender path routes through a subprocess and a temp
  directory, so it is the most suspect.
- **Anything other than Windows.** Everything ran on Windows 10, Python 3.12.6,
  `manifold3d` 3.5.2, Blender 5.2.0 LTS and OCP 7.9.3.1.1. The OCCT venv lives in a session
  scratchpad, so `run_all.py` will not find it on a fresh checkout. Set `LATHE_OCCT_PYTHON`,
  or create `prototype/bevel_spike/occtvenv`. Approach A then reports itself skipped rather
  than failing.

## Proposed DESIGN.md changes

This spike edited nothing outside `prototype/bevel_spike/`. These are the deltas the
measurements support.

1. **§13 "Selector instability" — promote the mitigation from policy to measured result.**
   The non-inheriting default now has evidence on the kernel that will ship it.
   `faces(#cut_face)` holds at 3 → 3, with zero false positives against an independent
   referee. The drift also reproduced from bmesh's *unmodified* default, where 184 new faces
   arrived pre-tagged with 133 `#body` and 51 `#cut_face`. State plainly that the drift is a
   wrapper policy failure, not a kernel defect. Note that §5.2's `#tag` and `~tag` split
   behaves as written.

2. **§11 "Geometry kernels" — replace "It brings no bevel or fillet, so budget for that
   separately" with the answer.** Bevel comes from `bmesh` through headless Blender, which
   §11 already lists as a backend. Move OCCT from "only if exact fillets and CAD interop
   become requirements" to a named Phase 4 decision, with the measured cost. OCCT cannot
   post-process a Manifold mesh, because `BRepFilletAPI_MakeFillet` returns `IsDone` on
   sewn-triangle input and emits non-manifold output. Adopting it therefore means a second
   boolean kernel. `cadquery-ocp` is also the only pip route, at 626 MB including 328 MB of
   unavoidable VTK.

3. **§11 — add a persistent Blender worker to the build strategy.** §11 currently describes
   Blender as `blender -b` for final renders only. Bevel makes it a per-operation
   dependency, at a measured 2.5 s process floor against a 0.6 ms operation. A long-lived
   worker over the npz protocol in `blender_bevel.py` is a Phase 2 requirement, not an
   optimisation.

4. **§4 and §4.1 — make `width` per-selection, and lower it to one bmesh pass per distinct
   width.** The IR line `bevel(carved, rim, width = bevel_w, segments = 2,
   harden_normals = true)` implies a uniform width. Multi-pass lowering works, but it
   changes topology between passes. So the type system should either reject overlapping
   width groups or demand a cardinality assertion per group.

5. **§7 verification profile — add two rules.**
   - `E109 Degenerate faces introduced by a bevel.` bmesh's `clamp_overlap` silently
     produced 42 zero-area triangles on geometry OCCT refused outright. Manifold imports the
     result as `NoError`.
   - `W206 Bevel width exceeds local wall thickness.` A precheck, because the failure is
     predictable. Every observed failure had wall < 2 × radius. `Manifold.min_gap` can
     measure it.

6. **§14 open question 2 — answer it for bevel. No.** `level_set` returns a single
   `run_original_id`, so it destroys all ancestor provenance. `faces(#cut_face)` goes
   3 → 429, 317 and 1 301 as the grid tightens. Flat faces move up to 5 mm against a 20 mm
   requested blend. Manifold exposes no signed-distance query, so implicit rounding cannot
   reach a boolean result at all. It needs the whole model as an SDF. Keep SDF for
   form-finding, where DESIGN.md already puts it. Keep it off the selector-bearing path.

7. **§14 — add a new open question.** Minkowski `open` and `close` ship in `manifold3d`
   today. They preserve flat faces exactly, cost 0.2 s and 1 360 triangles, and need no
   dependency. They also handle only one sign of curvature, and degrade when composed. Is
   whole-object rounding worth its own IR verb, say `round(solid, radius, only = convex)`,
   distinct from `bevel(solid, edges, width)`?
