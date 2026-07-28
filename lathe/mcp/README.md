# MCP surface

DESIGN.md §9. The layer that lets an agent drive Lathe instead of a human writing
Python.

Ten tools, deliberately. A large tool surface drowns the model. The expressive
weight lives in the program text `lathe_apply` takes and the expression language
`lathe_query` speaks.

## Setup

```bash
pip install "mcp[cli]" manifold3d numpy pillow
claude mcp add lathe -- python -m lathe.mcp.server
```

Run it from the repo root, or put the repo on `PYTHONPATH`. Set `LATHE_BLENDER` if
Blender is not on `PATH` — `lathe_observe`, `lathe_compare` and glTF/FBX export need
it. `LATHE_OUT` sets where sheets land (default `~/.lathe/out`).

Check it directly first:

```bash
python -m pytest lathe/mcp/tests -q        # 26 tests, no Blender needed
python lathe/mcp/tests/smoke.py            # drives every tool
python lathe/mcp/tests/smoke.py --render   # includes render and glb
```

## Tools

| Tool | Does | Read-only |
|---|---|---|
| `lathe_setup` | Audit the install, check the project page, build a welcome piece | no |
| `lathe_apply` | Run a program, commit it, return the validation report inline | no |
| `lathe_query` | Measure geometry — 13 scalars, selectors, raycast, symmetry | yes |
| `lathe_observe` | Multi-view contact sheet: shaded, matte or wire | yes |
| `lathe_check` | Full rule report against a profile | yes |
| `lathe_export` | Write obj/stl natively, glb/gltf/fbx/usd/ply via Blender | no |
| `lathe_log` | Commits, branches, current head | yes |
| `lathe_branch` | Branch and switch, for trying a variant | no |
| `lathe_checkout` | Switch to a branch or commit | no |
| `lathe_compare` | Render several commits into one grid | yes |

`library()` from §9 is **not** implemented. It searches a parametric component
library, and no such library exists yet. A tool that returns nothing is worse than
a tool an agent knows is absent.

## First run

`lathe_setup` answers three questions: can this environment build and render, is
the local copy current, and what does a working Lathe program look like.

It is **two-step by design**. With `confirm=false` it reports and asks. The agent
relays anything missing, the user answers, and only then does `confirm=true`
install the Python packages. A tool call cannot pause to ask mid-flight, and
installing packages unasked is not acceptable. Blender is never installed for you —
it is a large third-party installer, so the report points at blender.org.

The welcome piece is generated as **program text**, seeded from the machine and
account name. Two users get different pieces; the same user gets the same piece
until they pass a new seed. Four styles — stool, planter, lantern, pedestal — and
eight palettes, with proportions, segment counts and detail all varying.

It hands back the program rather than only the geometry on purpose. A first
artefact should be something the user can read and edit, which is the premise of
the whole system.

**Every seed must build.** The generator shipped with three bugs, all of them rules
already written down in DESIGN.md and not applied here:

- **Coincident contact is not connection** (§13). Stacked parts placed flush came
  out as separate components — 7 of 8 pedestals, then intermittently on lanterns
  and planters. Whether a boolean joins a shared plane depends on the exact
  coordinates, which makes it worse than a consistent failure. Every assembly now
  laps deliberately.
- **Logical face counts are tessellation-dependent on curved surfaces** (§5.3). A
  round stool seat is a cylinder, so `assert_count(faces("top"), 3)` was wrong; its
  side is one facet per segment.
- **Window openings sized against the radius severed the corner posts** once the
  side count rose. They are sized against the face chord now.

150 consecutive seeds build cleanly. `test_welcome.py` keeps three seeds per style
as a regression guard, including the ones that used to fail.

## Why `apply` returns the report

§9 is explicit about this: the agent should see the consequences of a change in the
same turn it makes the change, rather than needing to remember to look. So
`lathe_apply` returns the commit headline, the program diff against its parent, and
the whole validation report together.

## The query language

One tool, many expressions. Send an invalid one and the full grammar comes back.

```
bbox  volume  surface_area  counts  watertight  degenerates
center_of_mass  topology  parts  materials  uv  tags
symmetry(YZ)
faces(tag=seat)    faces(normal=0,0,1)    faces(z_gt=0.4)
edges(dihedral_gt=60, length_gt=0.02)    edges(tag=seat)
raycast(from=0,-2,0.3, dir=0,1,0)    raycast(from=0,-2000,300, dir=0,1,0, unit=mm)
```

Lengths are metres unless `unit=` says otherwise. The probe layer refuses an
unmarked length on purpose — §4.1 makes a bare number a type error for a length —
so the query layer states the unit rather than routing around the check.

## History

Each commit stores the **whole** program, not a delta. Operations are pure, so a
program is a complete description of its geometry and checking one out means
replaying it. Branching is therefore alternative program text, and comparing tips
is comparing two programs' output.

Re-sending the program that produced the current tip is a no-op. That is checked
against the tip's text directly, because content-addressing on `(parent, program)`
cannot catch it: after a commit the tip is its own parent, so identical text
addresses differently. The first version of this got it wrong and a test caught it.

## Security

`lathe_apply` runs the program through `exec` in a namespace preloaded with the
Lathe API. **That is a restricted namespace, not a sandbox.** Builtins are present,
so a program that wants to reach the filesystem can.

The threat model is that the caller could already run Python — which on a local
stdio server it can. Do not expose this over a network transport, and do not point
it at programs from untrusted sources.

## Still not the DSL

Programs are Python against the `lathe.build` API, not the `.lathe` language in §4.
So an agent writes `assert_count(rim, 96)`, not `assert count(rim) == 96`, and
`mm(35)` rather than `35mm`. §14.3 keeps the question open and leans bespoke with an
`escape { python … }` block. This surface is that escape hatch, promoted to the only
mode.

What that costs: no unit types in the build layer, so a bare `600` is a silent
metres-versus-millimetres bug rather than a compile error. The perception layer does
enforce units, which is why `raycast` needs one and `box` does not.
