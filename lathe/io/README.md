# Asset I/O

**Out.** OBJ and STL written natively. glTF, FBX, USD and PLY go through headless
Blender. Every path applies the transform first. Unapplied scale (E104) is the most
common export bug there is, and an exporter that *can* emit it eventually will.

**In.** `.glb` and `.gltf` read natively — no Blender, no library, no subprocess.

```python
from lathe.io import read, import_glb

imp = read("barrel.glb")          # always works: Mesh + report
print(imp.summary())
if imp.is_solid:
    cut = imp.solid - box(0.1, 0.1, 1.0)   # booleans need a real solid

solid = import_glb("barrel.glb")  # raises, naming which property failed
```

## What an import becomes

Two things, because a `.glb` in the wild is usually not a solid.

| | always | when the geometry really is one |
|---|---|---|
| type | `perceive.Mesh` | `build.Solid` |
| provenance | one part per material | one tag for the whole import |
| works | Tier 1 probes, every verifier rule, the viewer, export | booleans, bevel, volume, genus |

An open shell is the **normal** state of anything scanned or sculpted. The reader
does not treat it as a failure. `imp.mesh` stays fully usable, and `imp.reason`
names which property failed: open boundary, branching surface, or disagreeing
winding. `Error.NotManifold` on its own is true and useless.

Manifold *cannot represent* a non-watertight mesh. That is exactly why the neutral
`Mesh` type exists, and import is the case that vindicates it.

## The four things that decide whether an import is usable

Not "successful" — successful is easy. Get any of these wrong and you get no error
at all.

**Axes.** glTF is Y-up. Lathe is Z-up, because assets stand on z=0. One global
rotation of +90° about X converts it. Skip it and the model lies on its face,
silently. The matrix has determinant +1, so winding survives. The suite asserts
that rather than assuming it.

**Welding.** A glTF vertex carries its own normal and UV. So an exporter splits one
corner into as many vertices as it has distinct normals. **A cube arrives with 24
vertices, not 8.** Every seam is then a *boundary* edge, which means not closed,
which means Manifold refuses it. Unwelded, essentially no real `.glb` would ever
promote. Welding is not tidying. It decides whether you can boolean the import at
all.

Measured against this project's own exporter: 24 → 8 for a cube, 5,498 → 1,148 for
the chair.

The weld quantises to 1 nm and compares integers. `-0.0` and `0.0` are distinct
*bytes*, and numpy's row-wise unique compares bytes, so a vertex sitting exactly on
an axis would otherwise fail to weld against its own twin.

Welding sorts the vertices. It leaves triangle *order* alone, which is what keeps
the per-triangle channels in step, but nothing may assume a vertex kept its slot.
My own test asserted on `verts[1]` and was wrong — the reader was not.

**Transforms.** Node transforms compose down the scene graph, and the reader bakes
them into the vertices. It has to: several nodes can share one mesh at different
placements, and an `AssetMesh` holds one transform. Baking destroys the E104
signal, so the report states the worst axis scale deviation instead. A node whose
composed matrix has a negative determinant is a mirror, and the reader flips its
winding back. Without that the part returns inside out, and every facing check on
it lies.

**Refusal.** Draco and meshopt change how the vertex bytes decode. A reader that
ignores them does not fail — it produces **plausible-looking rubbish**. This one
names them, and says which command re-exports without them.

## Deliberate spec deviation

glTF says a client MUST NOT load a file whose `extensionsRequired` it does not
implement. This reader loads geometry anyway when the unknown extension cannot
affect geometry decoding — a required *material* extension, say. It records the
deviation in `report["notes"]`. Refusing to read vertices over a texture extension
would follow the letter of a rendering contract into a tool that does not render.

## Security

A `.gltf` may point at external buffers by URI, and that URI comes from an
untrusted file. The reader refuses absolute paths, Windows drive paths, `..`
escapes, percent-encoded `..` escapes, and network schemes. Otherwise opening a
model becomes an arbitrary-file-read primitive. It does decode `data:` URIs.

Names from the file reach report text, tags, and the HTML viewer. The reader strips
them to `[A-Za-z0-9_.- ]` and caps them at 64 characters at the boundary. A
material called `</script><script>alert(1)</script>` is a name, not markup. The
viewer now escapes its JSON payload too — defence in depth, and a latent hole
before import existed.

## Also handled

Interleaved accessors (`byteStride` > element). Normalized integer components
(`KHR_mesh_quantization`). Sparse accessors. Unsigned byte, short and int indices.
Non-indexed primitives. Triangle strips and fans, including the strip's alternating
winding. `TEXCOORD_0..n` as separate UV sets, with V flipped to the bottom-left
convention. Unknown GLB chunks skipped per spec. Files with no scene. Node cycles.
One mesh instanced onto several nodes.

Not handled: skins, morph targets, animation, cameras, lights, textures. This reads
*geometry*.

## Precision

glTF stores POSITION as float32. A 0.3 m coordinate round trips as 0.30000001, so
volume agreement caps at about 1e-7 relative. The round-trip test asserts to 6
places for that reason, not because the reader is sloppy.

## Tests

```bash
python -m pytest lathe/io/tests -q      # 85 tests; 74 need no Blender
```

`gltf_fixtures.py` builds containers by hand. They are written against the spec
rather than against the reader, so the two can disagree. Every fixture is
deliberately awkward — interleaved, quantised, column-major, mirrored, stripped.
An exporter only ever writes the one shape it likes, and those are the shapes a
reader already handles.

Two classes need Blender. `TestRoundTrip` closes the loop: this project's exporter
writes a `.glb`, the reader reads it back, and the volume has to survive.
`TestRealHierarchy` exports a parented, scaled, **mirrored** hierarchy with
`export_apply=False`, so the node transforms arrive unbaked. Each cube is 0.064 m³,
so the pair must measure 0.128. A mirrored child whose winding nobody flipped
contributes negative volume, and the total comes out at 0.
