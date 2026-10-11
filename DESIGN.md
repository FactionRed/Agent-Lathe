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

**Tier 3b — the interactive preview.** Built: `lathe/viewer/`. A contact sheet cannot be
turned over, and some questions only an orbit answers. One self-contained HTML file with the
mesh embedded and a WebGL renderer inline — no network, no library, and it survives a CSP
that blocks every external host. Full orbit camera, an adjustable lighting rig, four shading
modes, a crease-only wireframe, and a live readout including a per-tag triangle breakdown.

Hand-rolled WebGL was the cheaper choice, not the purist one: a CDN import cannot load under
the CSP at all, and inlining three.js would ship ~600 KB per preview against ~300 lines for
a Lambert-plus-Blinn shader and an orbit camera.

Two portability notes worth keeping. GLSL ES 1.0 **forbids indexing a uniform array with a
non-constant expression**, so `palette[int(vTag)]` does not compile in WebGL 1 — the legal
form is a constant-bounded loop, which also keeps the per-vertex payload at one float rather
than three. And the canvas reads its clear colour and grid tint from the CSS custom
properties, so the render cannot drift out of step with the panels when the theme flips.

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
| W206 | warn  | Collapsed UV area — real surface sampling a single texel |
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
build-look-undo cycles.

**Built** — `lathe/mcp/session.py`. A commit stores the *whole* program rather than a delta,
since a pure program is already a complete description of its geometry and replaying it is
what checkout means. One correction from building it: content-addressing on
`(parent, program)` does **not** make a repeated program idempotent, because after a commit
the tip is its own parent, so identical text addresses differently. The no-op check has to
compare against the tip's text directly. A test caught that. Breadth is the thing agents are actually good at, and current
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
| `import(path)` | *added in build* — read a .glb/.gltf, report and validate it |
| `view(path?)` | *added in build* — write the self-contained interactive preview (§6 Tier 3b) |
| `setup()` | *added in build* — audit the install, build a seeded welcome piece as editable program text |

`apply` returning the validation report inline is important: the agent sees consequences
in the same turn it makes the change, rather than needing to remember to look.

**Built** — `lathe/mcp/`, twelve tools over stdio, 51 wire tests. Four notes from building it.

`library()` is **omitted rather than stubbed**, because the parametric component library it
searches does not exist. A tool that always returns nothing is worse than a tool the agent
knows is absent.

`apply` took **Python** only at first — §14.3's `escape { python … }` promoted to the only
mode — and that cost the unit types: a bare `600` in the build layer is a silent
metres-versus-millimetres bug rather than a compile error. The language existed and was
tested, and was not reachable from the only interface an agent uses. That seam is closed:
`apply` reads `.lathe` too, chosen by `language`, defaulting to a read of the first
statement. The `emit` block becomes the verifier's declared spec, and a key it does not
understand is an error rather than a no-op — the same rule as §13's "a rule that never
runs is not a rule", applied to a declaration that is never read. Python stays for what
`.lathe` cannot say (loops, functions, `import_glb`), and a bare `600` there is still a
silent bug.

`apply` runs `exec` in a namespace preloaded with the API. That is a **restricted namespace,
not a sandbox** — builtins are present. Acceptable for a local stdio server whose caller
could already run Python, and not acceptable over a network transport. Stated in the tool
description rather than left for someone to discover.

`import` was **not in this table originally**, and its absence was a real gap: a tool that
can only make assets it authored cannot look at the asset you already have. Two decisions
came out of building it, and both were forced by measurement rather than chosen.

**An import is a Mesh first and a Solid only if it earns it.** A `.glb` in the wild is
usually not watertight — an open shell is the *normal* state of anything scanned or
sculpted. §3 already made the neutral polygonal `Mesh` the type that can hold geometry
Manifold cannot represent, and import is the case that vindicates that choice rather than
merely justifying it. Every Tier 1 probe and every §7 rule reads the mesh whether or not it
is a solid; only booleans, bevel and volume need the kernel. The reason text names *which*
property failed, because `Error.NotManifold` is true and tells an agent nothing it can act
on.

**A promoted import carries one tag, not one per material.** Tempting to map each glTF
material onto a §5.2 part and keep the split. It would be a lie: rebuilding a Manifold from
a triangle soup gives it a single original id, so the material structure would evaporate at
the first boolean — exactly the silent provenance drift §13 records. So the Mesh keeps the
material parts and the Solid does not pretend to. The verifier reads the *mesh*, which is
how a finding on an imported asset can still say `barrel.M_metal_bands` instead of
`face 4471`.

`import` also breaks the purity rule in §2, and is the only thing here that does. Every
other operation is a function of its explicit arguments; this one reads a file. A commit
that imports therefore replays identically only while that file is unchanged, so the report
carries the file's content hash and the generated program text records it in a comment.
Naming the exception is better than quietly having it.

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

**Baking scales with how much of the mesh the detail was.** `lathe/bake/` moves a modelled
bevel into a normal map, which is the production answer to the W302 spread above. It worked
spectacularly on one asset and did nothing for another, and the difference is instructive: the
chair's bevel was 87% of its triangles, so the low poly is 308 against 2,324 and the density
spread falls from 418% off target to 17%. The bowl's bevel was 18%, so its low poly is 82% of
the original and the warning is unchanged — because that asset's density problem was never the
bevel. It is the pole triangles of its spheres and revolves.

The general lesson: a bake is not a fix for bad UV density, it is a fix for *detail carried in
geometry*. Diagnose which one you have first. The verifier's per-island evidence is what makes
that distinguishable.

**A rule that never runs is not a rule.** E106, W202, W203 and W302 shipped with the verifier
and reported *not checked* on every asset for as long as nothing produced a UV set. They were
tested — against fixtures built by hand for the purpose — and that was enough to prove the
logic and not enough to prove anything was connected to it. Wiring an unwrapper in
immediately produced a genuine E106 on the chair, a W302 spread of 5.1 to 26.5 px/cm, and a
gap with no owner at all: a *collapsed* UV island does not overlap anything, so E106 stayed
silent and nothing else spoke. That became W206.

The lesson generalises past UVs. `needs=` makes a rule skip safely, which is the right
behaviour and also a very quiet failure mode. A profile where a third of the rules have never
seen production data reads as passing.

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

**A rule with only one bound cannot state a two-sided contract.** `hard_surface_subdiv`
documented "a cage is quads or it is a bug" and enforced it with `max_face_verts: 4`, which
admits a triangle as readily as a quad. So it returned an affirmative PASS on a fully
triangulated cage — not a skip, a pass, on the exact input its own comment names as the bug.
The single test covering it used a 6-gon, so it exercised the bound that worked.

Two things made it survive review. `max = 3` *does* express "triangles only", so `mobile_vr`
reads correctly and the parameter looks expressive enough. And the failure mode is a false
pass, which no amount of running the suite surfaces — a green report is what you expect.
E102 has a `min_face_verts` now, defaulting to 3 so nothing else moved.

The consequence is worth stating: since Manifold emits triangles, **no asset this project can
currently build satisfies `hard_surface_subdiv`**. That is the correct answer, and it was
hidden by the missing bound. A profile that passes everything is indistinguishable from a
profile that checks nothing, and this one was closer to the second than anyone believed.

**A clean textual merge can still be a semantic conflict.** W206 and the two newest profiles
(`hard_surface_subdiv`, `film_usd`) were written on separate branches. Both touched
`profiles.py`, in different places, so git merged them without a murmur — and the result was
wrong. The new profiles listed `E106` and `W202` but not `W206`, because `W206` did not exist
when they were written. `rules/uv.py` divides UV defects into exactly three: islands that
intersect, islands merely too close, and UV area that has collapsed. A profile holding two
thirds of a partition has a hole, and the hole is *silent* — a collapsed island simply goes
unreported and the profile still says PASS.

Nothing in either branch's tests could have caught it, because neither branch was wrong. Only
the combination was. Review would not reliably catch it either: the diff shows two additive
changes to a list of string literals.

The fix is not to add the missing code, or not only that. It is an invariant asserted over
*every registered profile* — any profile selecting part of the UV partition must select all of
it — so the next rule or profile added on a branch cannot reopen the gap. That generalises: a
rule set that grows on one axis while profiles grow on another needs structural tests over the
cross product, not per-item tests on each side. §7's claim that "a new profile is data" holds,
but data still has invariants.

**A test suite that runs one way and collapses the other has never been run the other way.**
`RULES` is populated lazily, and the verifier suite depended on
`TestRegistry.test_rules_are_registered_by_import` calling `load_rules()` and mutating the
shared dict in place — so every other test needed that one to have run first. Under `unittest`
it passes, by luck: class names sort alphabetically, `TestRegistry` sorts sixth, and four
classes ahead of it call `run()` and populate the registry as a side effect. `pytest` keeps
*definition* order, where `TestRegistry` is first. 241 tests then fail in cascade against an
empty registry.

Nothing was wrong with the tests or the engine. The suite simply had an initialisation order it
never declared, and one runner happened to satisfy it. Loading the registry in the test
package's `__init__` fixes it, and the test that used to create the guarantee now asserts it
instead — where it also turned out to be checking 13 of the 14 rules, having never been updated
for W206.

**Reading a file is a new trust boundary, and it arrived late.** Everything before import
consumed only what this system generated. A `.glb` from a marketplace is attacker-controlled
data, and three things in it reach places that were never designed for hostile input.

*Buffer URIs.* A `.gltf` may reference external buffers by path. Resolved naively, opening a
model becomes an arbitrary-file-read primitive — so absolute paths, drive paths, `..`, and
percent-encoded `..` are refused rather than resolved. The percent-encoded case is the one
worth calling out: unquoting after the containment check instead of before it defeats the
check entirely, and both orders look equally reasonable in the diff.

*Names.* Material and node names flow into report text, §5.2 tags, and the HTML viewer.
Sanitising at each use would eventually miss one, so names are stripped to a safe character
set at the reader, once.

*The viewer's payload.* Auditing that path found a latent hole that predated import:
`json.dumps` does not escape `<`, and the viewer drops its data straight into a `<script>`
block. A tag named `</script><script>…` would have executed. Harmless while every tag came
from the agent's own program, and a live injection the moment tags came from a stranger's
file. The fix is one line, but the *lesson* is that a data path's safety depends on where its
input comes from, so widening the input can break code that has not changed. Nothing in the
test suite would have caught it; there is now a test that would.

*Compressed geometry.* Draco and meshopt are refused by name. A reader that ignores an
unsupported compression extension does not error, it emits plausible-looking rubbish, which
is precisely the class of silent failure §1 indicts.

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

   **Import made the cost concrete.** Reading a `.glb` produces a perception `Mesh`, and
   verifying it needs an `AssetMesh`, so the work needed a *third* adapter — `from_mesh`,
   alongside `from_manifold` and `from_bpy_mesh` — whose entire job is to restate the same
   arrays in the other shape. It is short and duck-typed, so `verify` still imports nothing
   from `perceive` and the dependency arrow keeps pointing one way. But an adapter written
   only because two internal types disagree is the clearest evidence yet that this is worth
   fixing, and the count of them is now the metric. Import also confirms the polygonal
   choice is the right target: a glTF primitive is triangles, so a merged type loses
   nothing on the way in, while the verifier keeps the n-gon it needs on the way out.
7. ~~Does the bevel decision survive curved input?~~ **Mostly, with one defect and one
   caveat — see below.** Every fixture in the spike was an axis-aligned box, because its
   referee named reference planes by hand. `lathe/build/tests/bevel_truth.py` replaces that
   for any faceted solid of revolution. A solid revolved in *n* segments is a stack of regular
   *n*-gons, so in (apothem, z) space it is a plain 2D polygon and volume, per-tag area and
   per-triangle provenance are closed forms over it, independent of anything bmesh does.
   Cylinders, a cone, a bowl and a two-tag stepped body (which has a concave corner) were
   graded; `lathe/build/tests/test_bevel_curved.py` pins what was measured.

   *Held.* Provenance is exact: zero mis-tagged triangles on every fixture, including the
   concave corner and the tag boundary — §5.2's claim survives curvature. Geometry is exact
   to ~1e-7 in volume and area whenever the requested width is delivered, and facet edges
   (7.5° at 48 segments) are correctly not treated as features under the default
   `min_dihedral=30`.

   *Caveat — `profile=0.5` is not a circle off a right angle.* bmesh draws the quadrant in
   the frame whose axes are the two faces, so at 90° it is a circle and at any other angle an
   ellipse with the same endpoints. At 76° and 6 mm that is 0.9% off a true fillet in bevel
   area and 8e-5 in volume. Harmless for a game asset; wrong to quote as a radius.

   *Defect — the default `clamp=True` silently under-delivers width.* On the stepped fixture
   an 8 mm request delivered 3.4 to 7.6 mm depending on the step's radius, was non-monotonic
   in it, and returned a valid, correctly tagged, single-component solid: the bevel area is
   41% off and nothing reports it. On a bowl rim a 2 mm request under-delivered by 1%. With
   `clamp=False` the same requests are exact. This is §13's silent-selector failure in a new
   place — a parameter that means one thing and does another with no error — and it is
   pinned as an `expectedFailure` so it flips loudly when closed. Not yet decided: whether
   the default should change, or `bevel()` should measure and report the delivered width.
   `clamp=False` can produce overlapping geometry on a genuinely too-large width, and the
   referee has no closed form for that case.

   *Not done.* OCCT was not re-run on curved input, so the *comparison* that §11 rests on is
   still one-sided: bmesh is now known to be good on curved input, not known to be better
   than the alternative there. Solids that are not revolutions (a swept or lofted body) have
   no 2D reduction, so this referee does not reach them.
8. Bevel currently costs 2.9 s, of which 2.5 s is Blender process startup. A persistent
   worker is the obvious fix and is untested.
