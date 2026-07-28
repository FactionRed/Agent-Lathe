# Lathe

An agent-native 3D asset authoring system. The agent writes a program. The system returns a
verified asset.

![Three test assets built by the tool](out/chair.png)

## Why

Agents can already drive Blender through `bpy`. It mostly does not work, for structural
reasons rather than prompt-quality ones.

- **Hidden global state.** `bpy.ops` depends on mode, selection, active object and cursor —
  none of which appear in the call. A correct call becomes incorrect when preceded by another
  correct call.
- **Indices instead of names.** The job is "bevel the top rim". The API wants vertex 4,471.
- **No perception.** The script exits 0 and produces garbage. Nothing reports the inverted
  normals or the non-manifold edge.
- **No contract.** N-gons, unapplied scale, overlapping UVs, texel drift — every production
  failure mode is silent, so an agent ships all of them.

## Approach

- **Pure and deterministic.** Every operation is a function of its explicit arguments. Same
  program, same geometry, every time.
- **Selectors, not indices.** `carved.edges(dihedral > 60deg)` re-resolves on every
  evaluation, so an upstream edit does not invalidate a downstream reference.
- **Provenance that survives.** Tags ride Manifold's native ID channels. A bevelled solid put
  through a further boolean returns 0 unattributed triangles.
- **Cardinality assertions.** `assert count(rim) == 96` turns a silent selector mismatch into
  a loud error naming the assumption that broke.
- **Taste converted to assertions.** An agent cannot have judgement, but it can satisfy a
  spec. 13 rules, each with a named code, the offending entity, numeric evidence, and an
  executable repair.
- **Perception as a subsystem.** Numeric probes first at 30–90 µs. Renders as contact sheets,
  never one flattering angle.

## What works today

| Asset | Tris | Verifier |
|---|---|---|
| `SM_Chair_Oak_LOD0` | 2,324 | PASS — 0 error, 0 warn, 0 note |
| `SM_BowlFruit_Ceramic_LOD0` | 5,848 | PASS — 0 error, 0 warn, 0 note |
| `SM_Villager_Stylized_LOD0` | 2,446 | PASS — 0 error, 0 warn, 0 note |

- Authoring kernel: primitives, booleans, revolve, extrude, arrays, transforms.
- Bevel through headless bmesh, with the non-inheriting tag default enforced in the wrapper.
- Verifier: E101–E108 plus W201, W202, W203, W302, N401, across three profiles.
- Perception: Tier 1 probes and Tier 2 structural summaries, both Blender-optional.
- Renderer: shaded contact sheets, silhouette mattes, wireframe overlays.
- 264 tests pass. Checked against closed-form truth, not smoke.

## What is missing

Stated plainly, because the gap decides whether this is useful to you.

- **No DSL.** The `.lathe` language in [DESIGN.md](DESIGN.md) §4 has no parser. Models are
  Python against the API.
- **No unit types.** §4.1 wants a bare `600` to be a type error. Today it is float metres plus
  an `mm()` helper.
- **No UVs, textures or baking.** Four UV and texture rules report *not checked*, so about a
  third of `game_prop_pbr` is untested against real data.
- **No MCP surface.** §9's tool layer does not exist, so an agent cannot drive this yet. Only
  someone writing Python can.
- **No animation-grade topology.** Procedural geometry has no edge flow. The villager is a
  stylized figure, not a riggable character, and [models/README.md](models/README.md) says so
  in detail.

The DSL and the MCP surface are the two that matter for the premise. Without them this is a
good library rather than an agent-facing tool.

## Quick start

Needs Python 3.12, `manifold3d`, `numpy`, `Pillow`, and Blender 5.2 for bevel and rendering.

```bash
pip install manifold3d numpy pillow
python models/build_all.py               # build, verify, probe and render all three
python models/build_all.py --no-render   # skip Blender, keep the checks
python -m lathe.verify.tests             # 92 tests
python -m pytest lathe/perceive/tests -q # 172 tests
```

Set `LATHE_BLENDER` if Blender is not on `PATH`.

## Repo map

| Path | Contents |
|---|---|
| [DESIGN.md](DESIGN.md) | The design, and every place a measurement corrected it |
| `lathe/build/` | Authoring kernel, selectors, assertions, bevel |
| `lathe/verify/` | Rule engine and profiles |
| `lathe/perceive/` | Numeric probes and structural summaries |
| `lathe/render/` | Contact sheets, mattes, wireframes |
| `models/` | The three test assets and the build runner |
| `prototype/` | Kernel spikes that settled the design questions |

## Status

Working prototype, not a product. `DESIGN.md` records the open questions, including the two
that could still move the design: whether the bevel choice survives curved input, and whether
one shared polygonal mesh type should replace the two that exist.
