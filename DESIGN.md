# Lathe — an agent-native 3D asset authoring system

**Status:** design proposal, v0
**Working name:** Lathe (placeholder)

---

## 1. The problem

Agents can already drive Blender through `bpy`. It mostly does not work. The reasons are
structural, not a matter of prompt quality.

**Hidden global state.** `bpy.ops` is a function of mode, selection, active object, 3D
cursor, active collection, and view layer — none of which appear in the call. Every
operator mutates that state as a side effect, so a correct call becomes incorrect when
preceded by another correct call. The agent is writing into a machine whose registers it
cannot see.

**Indices instead of names.** Real modeling work is "bevel the top rim." The API wants
vertex 4,471. Agents cannot track integer indices across topology-changing operations,
and topology-changing operations are the entire job.

**No perception.** The script runs, exits 0, and produces garbage. There is no cheap,
structured way for the agent to find out that the normals inverted, the boolean left a
non-manifold edge, or the object is 100× too large.

**No contract.** Nothing in the API distinguishes a good asset from a bad one. Every
failure mode in a production checklist — n-gons, non-manifold edges, unapplied scale,
inverted normals, overlapping UVs, texel density drift — is silent. An agent will
cheerfully ship all of them, because nothing told it not to.

**Coordinates instead of relationships.** Agents reason well symbolically ("four legs at
the corners of the seat, inset 40 mm") and badly numerically ("vertex at 0.2341,
−0.1883, 0.0"). The API only accepts the second form, so the model's strongest capability
goes unused and its weakest one carries the work.

The common thread: `bpy` is a remote control for a GUI. An agent is not a mouse.

## 2. Design thesis

> **The agent's output is a program, not a mesh. The system's output is a verified
> asset, not a viewport.**

Seven principles follow.

1. **Pure and deterministic.** Every operation is a function of its explicit arguments.
   No mode, no selection state, no cursor. The same program always produces the same
   geometry, byte for byte.
2. **Names and queries, never indices.** Semantic selectors address geometry, and they
   re-resolve on every evaluation.
3. **Relationships over coordinates.** Positions are expressions over named parameters.
   Changing a parameter propagates.
4. **Valid by construction where possible.** Use a geometry kernel that cannot produce
   non-manifold output, so an entire failure class stops existing.
5. **Checkable by assertion where not.** Everything a senior artist would catch in review
   becomes a named, machine-evaluated rule with a suggested repair.
6. **Perception is a first-class subsystem.** The agent must be able to look at, measure,
   and diff what it built — cheaply, and mostly in text.
7. **Exploration is cheap.** History is a branching DAG. Twenty variants is a normal
   thing to ask for, not an ordeal.

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Agent (Claude, via MCP)                                     │
└────────────────────────┬─────────────────────────────────────┘
                         │  ~10 tools; the DSL carries expressiveness
┌────────────────────────▼─────────────────────────────────────┐
│  Session / MCP server                                        │
│  apply · query · observe · check · branch · export           │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│  Lathe IR  —  declarative, pure, text                        │
│  params · expressions · selectors · assertions · emit specs  │
└───┬──────────────┬──────────────┬───────────────┬────────────┘
    │              │              │               │
┌───▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐ ┌──────▼─────────┐
│ Evaluator│ │ Perception │ │ Verifier   │ │ History (DAG)  │
│ DAG,     │ │ probes,    │ │ rule packs │ │ content-hashed │
│ cached   │ │ renders,   │ │ + profiles │ │ commits,       │
│          │ │ diffs      │ │            │ │ branches       │
└───┬──────┘ └─────┬──────┘ └────────────┘ └────────────────┘
    │              │
┌───▼──────────────▼───────────────────────────────────────────┐
│  Geometry kernels                                            │
│  Manifold (booleans) · OpenVDB/SDF (form) · OCCT (B-Rep,     │
│  optional) · quad remesher · UV · baker · renderer           │
└──────────────────────────────────────────────────────────────┘
```

Nothing above the kernel row is off-the-shelf. That is where the value is. Borrow
everything in the kernel row.

## 4. The IR

A small typed language. Text, so it diffs, reviews, and version-controls. Pure, so it
replays.

```
# SM_Crate_Wood.lathe

param size     = 600mm          # outer cube edge
param slat_w   = 80mm
param slat_n   = 5
param rail     = 60mm           # material left above and below the slots
param bevel_w  = 4mm

unit  system  = metric
scene up = +Z, forward = -Y      # engine target set at emit, not here

let body   = box(size, size, size) @ origin(bottom_center)

let cutter = box(slat_w * 0.35, size * 1.1, size - rail * 2)
let slots  = array(cutter, axis = X, count = slat_n, span = size * 0.9, align = center)

let carved = body - slots                    # manifold-guaranteed difference
assert components(carved) == 1               # slots must not sever the body

let rim = carved.edges(dihedral > 60deg, length > 20mm)
assert count(rim) == 96                       # cardinality guard — see §5.3

let final = bevel(carved, rim, width = bevel_w, segments = 2, harden_normals = true)

uv  final by angle_seams(threshold = 66deg, pack = square)
mat oak on final.faces(all)

emit "SM_Crate_Wood" {                        # a stem, not a filename
  profile        : game_prop_pbr
  target         : unreal
  lod            : auto(4)                    # fans out to _LOD0 .. _LOD3
  budget         : { tris_lod0: 1500 }
  texel_density  : 5.12 px/cm
}
```

**`emit` names a stem.** One `emit` produces a *set* of artifacts, and the emitter appends
the suffixes. `lod: auto(4)` yields `SM_Crate_Wood_LOD0` through `_LOD3`; a texture set
yields `T_Crate_Wood_D` and siblings. Naming rules therefore apply to the emitted artifact
names, never to the stem — a stem carrying its own `_LOD0` would produce
`SM_Crate_Wood_LOD0_LOD0`.

This was underspecified in v0, and the first implementation of §7 read `lod` as *this
asset's LOD index* rather than *how many to generate*. Both readings are defensible from
the v0 text, which is the problem. Under the index reading, the verifier then reported the
example above as violating N401 for the missing `_LOD0` — a false positive produced
entirely by the ambiguity, in the one subsystem whose value depends on not emitting those.

Notes on what is deliberately absent: no mode switches, no selection calls, no
`transform_apply`. The program declares origin and scale, and the emitter must honor
them. This designs out the most common export bug instead of relying on memory.

### 4.1 Type system

Small and total. `Solid`, `Surface`, `Curve`, `Selection<Vert|Edge|Face>`, `Frame`,
`Length`, `Angle`, `Count`, `Material`, `UVSet`. Units are part of the type: `600mm` and
`0.6m` unify. `600` alone is a type error. Scale bugs die here.

## 5. Selectors — the core mechanism

This is the load-bearing idea. If it works, the rest follows. If it fails, the system
degenerates into `bpy` with extra steps.

### 5.1 Queries, not indices

A selector is a predicate over geometric and topological attributes, resolved at
evaluation time:

```
carved.faces(normal ≈ +Z, area > 0.01m²)
carved.edges(dihedral > 60deg, on_boundary = false)
carved.faces(material = oak).grow(1).boundary()
seat.top_face()
leg_01.face(facing = ground)
```

Because they re-resolve, an edit upstream does not invalidate a selector downstream. This
is what makes parametric editing survivable and is precisely what index-based APIs cannot
offer.

### 5.2 Provenance tags

Operations tag what they produce. A bevel tags its new faces `#bevel`, a boolean tags the
cut surface `#cut_face`, an array tags each instance `#instance[i]`. Selectors can filter
on tags, which gives stable handles on features that have no clean geometric predicate:

```
final.faces(#cut_face).edges(#bevel)
```

**Validated on both candidate kernels.** `prototype/` measures this.

On Blender 5.2, integer face attributes survive the Exact boolean intact — zero untagged
faces leaked, and `preserve_all_data_layers` made no difference. Tags survive
`bmesh.ops.bevel` too.

On Manifold, provenance is a **native feature and the better mechanism**. `reserve_ids`
exists specifically to mark triangle sets that remain queryable after later operations.
`run_original_id` attributes every output triangle to the input solid it came from. Three
chained booleans kept all three ancestors exactly distinguishable. Prefer ID tracking over
property channels: IDs are exact attribution, while property channels interpolate at new
cut vertices and therefore inherit the drift problem below.

**`face_id` needs a composite key.** An earlier revision of this section claimed `face_id`
groups coplanar triangles into logical faces, full stop. That is wrong in a way that
matters. `face_id` is scoped **per original solid**, so values collide across solids:

```
run_original_id 3  ->  face_ids [0, 1, 2, 3, 5, 6]
run_original_id 5  ->  face_ids [0, 2]            # 0 and 2 collide
distinct face_id alone        : 6
distinct (run_original_id, face_id) : 8
```

A selector keyed on `face_id` alone silently conflates faces from different source solids
and under-counts. The logical face key is the pair `(run_original_id, face_id)`. Note the
shape of this bug: a selector matching the wrong set with no error raised — the same class
as the bevel drift, hiding inside the mechanism recommended to fix the bevel drift.

**But inheritance is not free.** `bmesh` interpolates ancestor attributes onto newly
created faces. After bevelling a cut edge, the new bevel faces carry `#cut_face` too, and
`faces(#cut_face)` silently went from 2 matches to 10. The selector kept resolving. It
just stopped meaning what the program said.

So tags need **generation semantics**: a face that *is* a cut surface and a face that
merely *descends from* one are different sets, and the language must distinguish them.

```
final.faces(#cut_face)            # primary only -- the default
final.faces(~cut_face)            # derived: anything descended from a cut face
final.faces(#cut_face, #bevel)    # explicit intersection
```

Operations default to **not** propagating inherited tags to new geometry; they tag new
faces with their own tag only. The prototype confirms this restores stability (2 matches
before the bevel, 2 after). Inheritance stays available, but the program must ask for it.
Defaulting the other way produces exactly the invisible drift this section exists to
prevent.

### 5.3 Cardinality assertions

Queries silently matching the wrong set is the obvious failure mode, and it is dangerous
precisely because it is silent. The mitigation is to make the agent state its
expectation:

```
assert count(rim) == 96
assert count(legs) == 4
assert unique(seat.top_face())
```

Changing `slat_n` from 5 to 7 changes the edge count, the assertion fires, and the agent
is told *exactly* which assumption its edit broke.

**Assert logical faces on planar geometry only.** A logical face count is stable under
retessellation for flat faces and *not* for curved ones. `face_id` groups coplanar
triangles, so a 64-segment revolve reports one facet per segment — building the bowl in
`models/` produced 322 where the program asserted 1. On curved surfaces the count is a
function of segment count, so assert the invariants that actually hold instead: volume,
watertightness, component count, bounding box.

Two of the three test assets tripped a cardinality assertion on their first run, and both
times the assertion was right and the author's expectation was wrong. That is the mechanism
working. Cheap to write, and it converts a
class of silent corruption into a loud, localized error. Assertions should be
strongly encouraged by the tool's error messages and emitted by default in agent
scaffolding.

### 5.4 Constraints

Positions are expressions, evaluated over a dependency DAG:

```
param seat_h  = 450mm
param inset   = 40mm

let seat = box(400mm, 400mm, 40mm) @ z(seat_h)
let legs = at(seat.bottom_face().corners().inset(inset),
              cylinder(r = 20mm, h = seat_h))
```

Change `seat_h` and the legs follow. This is a lazy expression graph, **not** a general
geometric constraint solver — the deliberate scope limit keeps failures explicable. A
Add a real solver later, behind an explicit `solve { … }` block, if it earns its keep.

## 6. Perception

Ordered by cost. The system steers the agent toward cheap signals first, because renders
are slow, ambiguous, and easy to over-trust.

**Tier 1 — numeric probes (text).** This is where most verification should happen.

Two cost classes inside this tier, and the distinction is measured, not assumed.
Closed-form probes — `bbox`, `volume`, `center_of_mass`, `counts`, `is_watertight`,
`surface_area` — run in **30–90 µs** on prop-sized meshes. `symmetry` needs kernel
booleans and costs **0.8 ms on 20 triangles, 6–90 ms at production density**: three to four
orders of magnitude more. An earlier revision filed the whole tier under "microseconds",
which was wrong and would have mislead an agent budgeting a verification loop.

**Every probe declares its method.** `symmetry` has an exact path (volumetric IoU, immune
to tessellation, exactly 1.000 when symmetric) and a sampling fallback for open meshes
whose score is a normalized deviation and is **not comparable** to an IoU. A bare number
that silently means two different things is worse than no number, so the method and the
mirror plane print alongside the score.

```
measure(final.bbox)              → 0.600 × 0.600 × 0.600 m
measure(distance(leg_01, leg_02))→ 0.320 m
count(final.tris)                → 1,284
volume(final)                    → 0.164 m³
center_of_mass(final)            → (0, 0, 0.287)
raycast(from = (0,-2,0.3), dir = +Y) → hit final #cut_face at 1.71 m
is_watertight(final)             → true
symmetry(final, plane = YZ)      → 0.875   method volume_iou (exact)
                                           mirror x = 0.000 m (bbox_center)
```

**Tier 2 — structural summaries (milliseconds, text).** Topology histograms, per-island
UV stats, material assignment table, LOD chain summary, poly distribution by named part.
Compact, and agents read them far more reliably than pixels.

**Tier 3 — renders (seconds, images).** Always as a **contact sheet**, never a single
view: front / side / top / three-quarter in one image, with a metric grid and a 1.8 m
human silhouette for scale.

Two render modes matter more than beauty shots:
- **Silhouette matte** — flat black on white. Agents read shape from silhouette far more
  accurately than from shaded renders, where lighting confounds form.
- **Wireframe-over-shaded** — makes topology density legible.

**Tier 4 — diffs.** Before/after contact sheets plus a per-vertex displacement heat map,
and a textual op-diff from the history DAG. This directly answers "did my edit do what I
intended?", which is the question the agent actually has and currently cannot ask.

## 7. Verification

The highest-leverage subsystem, and the cheapest to build. **Convert taste into
assertions.** An agent cannot have judgment, but it can satisfy a spec — so push as much
of "good asset" as possible into machine-checkable rules, and reserve humans for the
residue.

Rules are grouped into **profiles** selected by `emit`. Each rule has a stable code, a
severity, the offending named entity, numeric evidence, and a suggested repair
expression.

`game_prop_pbr` profile, abbreviated:

| Code | Sev | Rule |
|------|-----|------|
| E101 | error | Non-manifold edges or vertices |
| E102 | error | N-gons in emitted geometry |
| E103 | error | Inverted / inconsistent face normals |
| E104 | error | Object scale ≠ (1,1,1) or rotation ≠ 0 at emit |
| E105 | error | Loose vertices / edges, zero-area faces |
| E106 | error | Overlapping UVs when unique maps are baked |
| E107 | error | Triangle count over declared budget |
| E108 | error | Component count differs from the declared expectation |
| W201 | warn  | Origin not at declared reference point |
| W202 | warn  | UV padding below the table for target resolution |
| W203 | warn  | Non-power-of-two texture dimensions |
| W204 | warn  | LOD chain missing a step, or reduction ratio off target |
| W205 | warn  | Silhouette deviation between LOD*n* and LOD0 above threshold |
| W302 | warn  | Texel density variance outside target ±10% |
| N401 | note  | Emitted artifact name does not match `[Prefix]_[Name]_[Variant]_[LOD]` (see §4 — the rule reads emitted names, not the `emit` stem) |
| N402 | note  | Dense topology in low-curvature, non-deforming regions |

Additional profiles: `character_skinned` (adds edge-loop counts at joints, pole
placement, quad ratio, symmetry), `hard_surface_subdiv`, `film_usd`, `mobile_vr`.

The report is the primary agent feedback channel. Optimize it for repair, not for
reading:

```
✗ E106  Overlapping UVs on unique bake
        object   : SM_Crate_Wood
        islands  : side_L overlaps side_R (91% area)
        evidence : bake target = normal, uv_set = UV0
        repair   : uv.unwrap(final, split_mirrored = true)
                   — or emit with bake_uv_set = UV1

⚠ W302  Texel density variance 4.9–6.3 px/cm (target 5.12 ±10%)
        islands  : side_L, side_R, lid_top
        repair   : uv.rescale(islands = [side_L, side_R, lid_top], to = 5.12)
```

Named code, named entity, numeric evidence, executable repair. Agents fix errors of this
shape reliably. They never fix "boolean failed."

## 8. History

Content-addressed DAG, git-shaped, because the operations are pure and therefore
replayable.

- Every `apply` is a commit: program delta, resulting geometry hash, validation report.
- `branch` / `checkout` / `merge` on the IR text, with geometry as derived output.
- **Contact-sheet compare across branch tips** — render every variant into one grid.

This is what makes cheap exploration real. "Give me eight crate proportions" becomes
eight branches, one contact sheet, one pick — instead of eight sequential
build-look-undo cycles. Breadth is the thing agents are actually good at, and current
tools tax it heavily.

## 9. Agent interface

Deliberately small. A large tool surface drowns the model. A small tool surface plus a
rich language does not.

| Tool | Returns |
|------|---------|
| `apply(program, branch?)` | commit id, op diff, validation report |
| `query(expr)` | structured facts (Tier 1/2 probes) |
| `observe(views, mode, diff_from?)` | contact-sheet image(s) |
| `check(profile)` | full validation report |
| `branch(from) / checkout(ref) / log()` | history ops |
| `compare(refs)` | contact-sheet grid across commits |
| `library(query)` | parametric component search (§11) |
| `export(target, path)` | file + final gate report |

`apply` returning the validation report inline is important: the agent sees consequences
in the same turn it makes the change, rather than needing to remember to look.

## 10. What stays human

Worth stating plainly, so the system is not oversold.

Agents will be strong at: structure and layout, parametric variation, spec compliance,
LOD chains, UV/naming/export hygiene, batch and modular work, and iterating against
explicit numeric targets.

Agents will be weak at: proportion and appeal, silhouette *quality* as opposed to
silhouette *accuracy*, matching visual reference, stylization, and anything where the
target is taste rather than a measurement.

The design accommodates this rather than pretending otherwise. When a human adjusts
something in the viewer, the app **writes back a parameter or constraint edit**, not an
opaque mesh mutation — `param inset: 40mm → 60mm` appears in the program, the agent reads
it, and subsequent generation respects it. Human correction becomes machine-legible
input. That feedback loop is the actual product.

Character topology deserves explicit honesty: procedurally generated meshes will not have
animation-grade edge flow. The realistic path for deforming assets is template basemeshes
plus cage deformation and blendshape-safe edits — not generation from scratch. Phase
this in later, and do not promise it early.

## 11. Build strategy

**Do not build a DCC.** A competitive 3D UI is a decade of work and is not the
differentiator. The differentiators are the IR, the verifier, the perception layer, and
the history model — all of which are headless.

- **Core:** Rust engine (determinism, speed, clean FFI) with a Python binding. Or
  Python-first if velocity matters more than eventual performance — a defensible
  trade at Phase 1, painful by Phase 4.
- **Geometry kernels:** [Manifold](https://github.com/elalish/manifold) for booleans —
  guaranteed-manifold output eliminates E101 by construction, which is worth a great deal.
  Its native ID tracking also carries §5.2 provenance for free, which settles the kernel
  choice. It brings no bevel or fillet, so budget for that separately. Read geometry with
  `to_mesh64()`, never `to_mesh()` — the latter silently returns float32, so a 600 mm cube
  reads back as 600.000012 mm. The absolute error is negligible; the broken exact
  comparison against a declared `600mm` is not.
  OpenVDB for SDF/implicit form-finding and smooth blends. Instant Meshes or QuadWild for
  quad remeshing. xatlas for UV packing. OpenCASCADE only if exact fillets and CAD
  interop become requirements — it is heavy and its boolean failures are opaque.
- **Bevel and fillet:** bmesh behind a headless subprocess, with the §5.2 non-inheriting
  tag default enforced in the wrapper rather than left to the caller. Chosen on measured
  evidence over two alternatives. OCCT is the better bevel — exact radii, per-edge and
  variable width, failures that name the offending contour — but it fillets a *feature
  history*, not a mesh: it reports `IsDone` while emitting non-manifold garbage on a sewn
  mesh, so adopting it means carrying a second kernel plus 626 MB of install. Defer it to a
  Phase 4 CAD-interop decision. The implicit routes are out: every one collapses provenance
  to a single `run_original_id`, Manifold exposes no signed-distance query so `level_set`
  cannot round an arbitrary boolean result, and the smooth-min parameter is not a width —
  it pulled a face 5.04 mm inward against a requested 20 mm. Manifold's Minkowski
  `open`/`close` is worth keeping for single-curvature-sign rounding at ~1000 triangles in
  0.2 s.
- **Renderer:** headless wgpu for probes and matte/wireframe passes; Cycles via
  `blender -b` only for final beauty renders.
- **Blender is a backend and a viewer, not the foundation.** Keep an exporter to `.blend`
  so humans can finish work in familiar tools, and optionally an importer that lifts
  simple meshes back into IR. Do not build the system inside Blender — its global state
  is the problem this system solves.
- **Human UI:** a thin web viewer (three.js) showing model, DAG, program text, and
  validation panel. A **review** surface first; direct manipulation that writes back to
  the IR comes later.

### 11.1 Component library

Much agent work is assembly rather than invention. A curated library of parametric
components — fasteners, hinges, moldings, modular wall/floor kits on a grid, primitive
furniture, foliage cards — with typed parameters and pre-verified topology, queryable via
the `library` tool. This raises the floor faster than any improvement to generation. Do not defer it
to the end.

## 12. Milestones

Each phase ends with a demo that a skeptic can evaluate.

**Phase 1 — Kernel and IR (6–8 weeks).**
Parser, type system with units, evaluator DAG, Manifold-backed primitives and booleans,
transforms, arrays, `emit` to glTF/FBX with correct scale and origin.
*Demo: an agent writes a parametric crate from a text prompt; it exports into Unreal at
correct scale, first try.*

**Phase 2 — Verification (4–6 weeks).**
Rule engine, `game_prop_pbr` profile covering E101–E107 and W201–W205, repair
suggestions, `check` and `apply` reports. Bevel, and UV unwrap/pack with texel density.
*Demo: an agent is given a deliberately broken asset and repairs every violation from
the report alone, with no human hints.*

**Phase 3 — Perception (4–6 weeks).**
Tier 1 and 2 probes, contact-sheet renders, silhouette matte, wireframe overlay, diff
renders and heat maps.
*Demo: an agent catches and self-corrects a proportion error it introduced, without a
human noticing it first.*

**Phase 4 — History and exploration (3–4 weeks).**
Commit DAG, branching, `compare` contact sheets, merge on IR text.
*Demo: "twelve variations of this lamp" → one grid → human picks two → agent refines
both in parallel.*

**Phase 5 — Human loop and library (6–8 weeks).**
Web viewer with DAG and validation panel, direct manipulation writing back to params,
component library with ~100 verified parts, LOD generation with silhouette-preservation
checks.
*Demo: a mixed human/agent session producing a small modular environment kit that passes
a studio's real import checks.*

**Later:** SDF/implicit form-finding, quad remeshing to animation-grade topology, template
basemesh + cage deformation for characters, USD export, baking pipeline, material graph.

## 13. Risks

**Selector instability.** A query silently matching the wrong set is the most dangerous
failure mode, because it is invisible. The prototype reproduced it on the second try: a
bevel widened `faces(#cut_face)` from 2 matches to 10 without any error, because the
kernel interpolated ancestor tags onto the new faces.

The bevel spike then measured it properly, against a geometric referee that recomputes
provenance from plane geometry so no implementation grades itself. The drift is **bmesh's
own default**, not a wrapper artifact: `faces(#cut_face)` goes 3 → 78 with 97 wrongly
attributed triangles, and 184 new faces arrive already carrying ancestor masks and no
`#bevel` tag at all. The §5.2 non-inheriting default holds at 3 — zero false positives,
zero false negatives, zero untagged. The mitigation is validated in the kernel that will
ship it. Two mitigations, both now
load-bearing. First, tag inheritance is opt-in, not default (§5.2). Second, cardinality
assertions (§5.3), enforced culturally — the tool should nag when a selector feeding a
destructive op carries no assertion. Note that the assertion alone would **not** have
caught the drift case, since nobody asserts on a selector they did not know had changed.
Defaults do the real work here.

**Topology quality ceiling.** SDF and boolean output does not have artist edge flow. This
caps the system at static-prop grade until remeshing matures. Accept the cap, state it,
and do not sell character work in Phase 1–4.

**Coincident contact is not connection.** Two solids meeting exactly on a shared plane come
back as separate components, because the contact has measure zero. The villager's boots met
its shins precisely on z=0.075 and the figure assembled in four pieces. Assembly helpers
should overlap deliberately rather than abut, and the E108 component check is what catches
it — which is the second time that rule has earned its place.

**Boolean fragility shrinks, but does not vanish.** Manifold removes non-manifold output.
It does not remove coincident-face degeneracy or the need to think about cut placement.
Budget for a "boolean forensics" diagnostic that explains failures in terms of named
inputs.

**The constraint DAG becoming the new opaque state.** Keep it lazy and explicit. Resist a
general solver until there is evidence it is needed; solvers fail in ways that are very
hard to explain to an agent.

**Renders over-trusted.** Agents will reach for `observe` when `query` would be faster and
more reliable. Mitigate in tool descriptions and by making Tier 1 output genuinely
excellent.

**Adoption.** Studios have pipelines. The wedge is the verifier and the export hygiene —
those provide value even to teams that never let an agent model anything, which makes the
system adoptable before it is trusted.

## 14. Open questions

1. Rust core or Python-first? Velocity now versus a painful port at Phase 4.
2. Does the SDF path earn its complexity in Phase 1, or is boolean-plus-bevel enough to
   reach a compelling demo?
3. Should the IR be a bespoke language or an embedded Python DSL? Bespoke gives a clean
   type system with units and better errors; embedded gives an escape hatch and zero
   parser work. *Leaning bespoke, with an `escape { python … }` block.*
4. ~~How much does provenance tagging cost at kernel level, and does Manifold expose
   enough to implement it without forking?~~ **Answered on both kernels — see §5.2 and
   `prototype/`.** Manifold is the stronger of the two: `reserve_ids` /
   `run_original_id` / `face_id` give exact, non-interpolated provenance as a first-class
   feature. Two footguns to record. `as_original()` deliberately drops all ancestor
   relations, so calling it destroys provenance. And Manifold has no bevel or fillet — the
   drift risk returns in whatever implements those, so the §5.2 non-inheriting default
   still earns its place.
5. What is the minimum component library that makes assembly genuinely faster than
   generation?
6. **One mesh type, and it must be polygonal.** §7 and §6 were built in parallel and each
   independently chose a neutral mesh, then diverged: the verifier uses variable-length
   faces, perception is triangles-only. The verifier's reasoning decides it — *E102 cannot
   exist if the representation cannot hold an n-gon*. Both packages import together today,
   so this is a scheduled refactor rather than a bug, but it should land before either
   grows a third consumer.
7. Does the bevel decision survive curved input? Every fixture in the spike is
   axis-aligned planar boxes, because the geometric referee cannot grade a fillet on a
   cylinder or revolve. That is exactly where OCCT fillets fail in practice and where
   bmesh bevel looks worst, so the §11 choice rests on the easy case. Extend the referee
   before trusting it further.
8. Bevel currently costs 2.9 s, of which 2.5 s is Blender process startup. A persistent
   worker is the obvious fix and is untested.
