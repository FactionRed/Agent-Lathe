"""
Read binary glTF (.glb) and JSON glTF (.gltf), natively.

DESIGN.md section 11 makes Blender a backend rather than a foundation, and
verify/mesh.py's own docstring already promised that "adapters convert (Manifold
mesh | bpy Mesh | glTF | FBX) into AssetMesh". This is the glTF half of that
promise, written against the container directly: a .glb is a 12-byte header plus
length-prefixed JSON and BIN chunks, and decoding it needs no library and no
subprocess. Reading it through Blender would cost 2.5 s of process startup and
would make the *importer* -- the thing you reach for when you want to look at
someone else's asset -- the one part of the tool that cannot run without a
1 GB dependency.

Four things decide whether an import is useful rather than merely successful.

**Axes.** glTF is Y-up, right-handed, metres. Lathe is Z-up, because assets
stand on z=0. Every importer that skips this produces a model lying on its face,
so the conversion is applied once, globally, and named in the report. The matrix
is a proper rotation (determinant +1), so triangle winding survives it.

**Welding.** A glTF vertex carries its normal and UV, so an exporter splits one
corner into as many vertices as it has distinct normals -- a cube arrives with 24
vertices, not 8. Every seam is then a *boundary* edge, and a boundary edge means
the mesh is not closed, which means Manifold refuses it. Unwelded, essentially no
real .glb would ever promote to a Solid. Welding by position is therefore not
tidying: it is the step that decides whether an import can be booleaned.

**Transforms.** Node transforms compose down the scene graph and are baked into
the vertices here, because several nodes can share one mesh at different
placements and a single AssetMesh has only one transform to hold. Baking loses
the E104 signal (unapplied scale), so the report states what was baked instead. A
node whose composed matrix has a negative determinant is mirrored, and its
triangle winding is flipped to compensate -- without that the part comes back
inside out and every normal-facing check on it is wrong.

**Refusal.** Draco and meshopt compression change how the vertex bytes are
decoded. A reader that ignores them does not fail, it produces *garbage*, which
is the failure mode this project exists to eliminate. They are named and refused.

Deliberate spec deviation: glTF says a client MUST NOT load a file whose
`extensionsRequired` it does not implement. This reader loads geometry anyway
when the unknown extension cannot affect geometry decoding (a required *material*
extension, say), and records the deviation in the report. Refusing to read
vertices because of a texture extension would be following the letter of a
contract about rendering into a tool that does not render.

Security: a .gltf may point at external buffers by URI. Those URIs come from an
untrusted file, so `..` and absolute paths are rejected rather than resolved --
otherwise reading a model would be an arbitrary-file-read primitive. Names from
the file reach report text, tags, and the HTML viewer, so they are stripped to a
safe character set at the boundary.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import struct
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..build.solid import LatheError, Solid
from ..perceive import mesh as pmesh

IMPORT_FORMATS = ("glb", "gltf")

_MAGIC = 0x46546C67          # 'glTF'
_CHUNK_JSON = 0x4E4F534A     # 'JSON'
_CHUNK_BIN = 0x004E4942      # 'BIN\0'

# Accessor component types -> (numpy dtype, byte size, is integer)
_COMPONENT = {
    5120: ("<i1", 1, True),
    5121: ("<u1", 1, True),
    5122: ("<i2", 2, True),
    5123: ("<u2", 2, True),
    5125: ("<u4", 4, True),
    5126: ("<f4", 4, False),
}
_COMPONENT_NAME = {
    5120: "BYTE", 5121: "UNSIGNED_BYTE", 5122: "SHORT",
    5123: "UNSIGNED_SHORT", 5125: "UNSIGNED_INT", 5126: "FLOAT",
}
_TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
               "MAT2": 4, "MAT3": 9, "MAT4": 16}

_MODE_NAME = {0: "POINTS", 1: "LINES", 2: "LINE_LOOP", 3: "LINE_STRIP",
              4: "TRIANGLES", 5: "TRIANGLE_STRIP", 6: "TRIANGLE_FAN"}

# Extensions that change how geometry bytes are decoded. Anything here must be
# refused: silently mis-decoding a compressed buffer yields plausible-looking
# rubbish, which is worse than an error.
_GEOMETRY_EXTENSIONS = {
    "KHR_draco_mesh_compression":
        "Draco-compressed geometry. Re-export without Draco, or decompress "
        "first: `gltf-transform dedup --no-compress in.glb out.glb`.",
    "EXT_meshopt_compression":
        "meshopt-compressed buffers. Re-export without meshopt compression, or "
        "run `gltf-transform meshopt --no-compress in.glb out.glb`.",
}

# Handled, so their presence in extensionsRequired is not a deviation.
_HANDLED_EXTENSIONS = {"KHR_mesh_quantization"}

# glTF Y-up, -Z forward -> Lathe Z-up, +Y forward. A +90 degree rotation about X:
# (x, y, z) -> (x, -z, y). Determinant is +1, so winding is preserved.
YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0],
                       [0.0, 1.0, 0.0]], dtype=np.float64)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.\- ]+")

# 1 nanometre. Far below any modelling tolerance, so it never merges distinct
# geometry, while corners an exporter split from one source vertex stay
# bit-identical through transform baking and collapse back together.
DEFAULT_WELD_TOL = 1e-9


class GlbError(LatheError):
    """The file could not be read as glTF, or could not be read faithfully."""


# --------------------------------------------------------------- the result


@dataclass(frozen=True)
class GlbImport:
    """What an import produced, and what it could not.

    `mesh` is always present: the neutral Mesh can hold geometry Manifold cannot
    represent, which is the entire reason it exists (perceive/mesh.py). Every
    Tier 1 probe and every verifier rule therefore runs on an import whether or
    not it is a solid.

    `solid` is present only when the geometry genuinely is a closed, oriented,
    manifold solid. When it is None, `reason` says which of those it failed, and
    `report["topology"]` carries the counts that say how far off it is.
    """

    path: str
    mesh: pmesh.Mesh
    solid: Optional[Solid]
    reason: Optional[str]
    report: Dict[str, Any]
    uv_sets: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def is_solid(self) -> bool:
        return self.solid is not None

    def summary(self) -> str:
        """The report as text, in the shape the rest of the tool uses."""
        return _format_report(self)


# ------------------------------------------------------------------ container


def _read_container(path: str) -> Tuple[dict, Dict[int, bytes], str, str]:
    """(gltf json, buffer index -> bytes, container kind, sha256 prefix)."""
    if not os.path.isfile(path):
        raise GlbError("no such file: %s" % path)
    with open(path, "rb") as fh:
        blob = fh.read()
    if len(blob) < 12:
        raise GlbError("%s is %d bytes -- too short to be glTF" % (path, len(blob)))
    digest = hashlib.sha256(blob).hexdigest()[:12]

    if blob[:4] == b"glTF":
        gltf, chunk_bin = _parse_glb(blob, path)
        kind = "glb"
    else:
        try:
            gltf = json.loads(blob.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GlbError(
                "%s is neither binary glTF (it does not start with 'glTF') nor "
                "valid JSON glTF: %s" % (os.path.basename(path), exc))
        chunk_bin = None
        kind = "gltf"
    if not isinstance(gltf, dict):
        raise GlbError("the glTF JSON chunk is %s, expected an object"
                       % type(gltf).__name__)

    version = str((gltf.get("asset") or {}).get("version", ""))
    if version and not version.startswith("2"):
        raise GlbError(
            "glTF asset version %r. This reader implements glTF 2.0; a 1.0 file "
            "uses a different material and shader model entirely. Convert it "
            "first." % version)

    buffers = _load_buffers(gltf, chunk_bin, path)
    return gltf, buffers, kind, digest


def _parse_glb(blob: bytes, path: str) -> Tuple[dict, Optional[bytes]]:
    magic, version, length = struct.unpack_from("<III", blob, 0)
    if magic != _MAGIC:
        raise GlbError("bad GLB magic 0x%08x in %s" % (magic, os.path.basename(path)))
    if version != 2:
        raise GlbError("GLB container version %d, expected 2" % version)
    if length > len(blob):
        # Truncated download is the common cause, so say that rather than
        # failing later on a short buffer read.
        raise GlbError(
            "GLB header declares %d bytes but the file is %d. The file is "
            "truncated." % (length, len(blob)))

    gltf: Optional[dict] = None
    chunk_bin: Optional[bytes] = None
    offset = 12
    while offset + 8 <= length:
        clen, ctype = struct.unpack_from("<II", blob, offset)
        start = offset + 8
        end = start + clen
        if end > length:
            raise GlbError(
                "chunk at byte %d declares %d bytes, past the end of the file"
                % (offset, clen))
        if ctype == _CHUNK_JSON and gltf is None:
            text = blob[start:end].decode("utf-8", "replace").rstrip("\x00 ")
            try:
                gltf = json.loads(text)
            except json.JSONDecodeError as exc:
                raise GlbError("the GLB JSON chunk is not valid JSON: %s" % exc)
        elif ctype == _CHUNK_BIN and chunk_bin is None:
            chunk_bin = blob[start:end]
        # Unknown chunk types are skipped, which the spec requires.
        offset = end + (-clen % 4)  # chunks are 4-byte aligned

    if gltf is None:
        raise GlbError("no JSON chunk in %s" % os.path.basename(path))
    return gltf, chunk_bin


def _load_buffers(gltf: dict, chunk_bin: Optional[bytes], path: str) -> Dict[int, bytes]:
    """Resolve every buffer to bytes: the BIN chunk, a data: URI, or a file.

    External URIs come from an untrusted file. Anything absolute, or anything
    that resolves outside the directory holding the glTF, is refused -- without
    that check, opening a model is an arbitrary-file-read primitive.
    """
    base = os.path.dirname(os.path.abspath(path))
    out: Dict[int, bytes] = {}
    for i, buf in enumerate(gltf.get("buffers") or []):
        uri = buf.get("uri")
        if uri is None:
            if chunk_bin is None:
                raise GlbError(
                    "buffer %d has no uri, which means it is the GLB binary "
                    "chunk -- but there is no binary chunk in this file." % i)
            out[i] = chunk_bin
            continue
        if uri.startswith("data:"):
            out[i] = _decode_data_uri(uri, i)
            continue
        out[i] = _read_external(uri, base, i)
    return out


def _decode_data_uri(uri: str, index: int) -> bytes:
    head, _, payload = uri.partition(",")
    if not payload:
        raise GlbError("buffer %d has an empty data: URI" % index)
    if "base64" not in head:
        return urllib.parse.unquote_to_bytes(payload)
    try:
        return base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise GlbError("buffer %d has an undecodable base64 data: URI: %s"
                       % (index, exc))


def _read_external(uri: str, base: str, index: int) -> bytes:
    split = urllib.parse.urlsplit(uri)
    # `urlsplit` reads the drive letter of "C:/Windows/win.ini" as a scheme, so a
    # one-character scheme is a Windows absolute path, not a protocol. Checked
    # before the network branch, or a local absolute path is refused with the
    # wrong explanation.
    is_drive = len(split.scheme) == 1
    if split.scheme and not is_drive and split.scheme != "file":
        raise GlbError(
            "buffer %d points at %r. This reader does not fetch buffers over "
            "the network -- download the file and its .bin next to each other."
            % (index, uri[:120]))
    # Unquote after: a percent-encoded '..' is still a '..'.
    rel = uri if is_drive else urllib.parse.unquote(split.path or uri)
    if is_drive or os.path.isabs(rel):
        raise GlbError(
            "buffer %d uses the absolute path %r. Refused: a glTF file may only "
            "reference buffers beside it." % (index, rel[:120]))
    target = os.path.normpath(os.path.join(base, rel.lstrip("/\\")))
    if os.path.commonpath([os.path.abspath(target), base]) != base:
        raise GlbError(
            "buffer %d resolves to %r, outside the directory holding the glTF. "
            "Refused." % (index, rel[:120]))
    if not os.path.isfile(target):
        raise GlbError(
            "buffer %d needs the external file %r, which is not next to the "
            ".gltf. A .glb keeps its buffer inside the file; a .gltf does not."
            % (index, rel[:120]))
    with open(target, "rb") as fh:
        return fh.read()


# ------------------------------------------------------------------ accessors

# A glTF can name a count it has no bytes for (an accessor with no bufferView is
# all zeros, and a sparse block sets some of them), and the reader then allocates
# the count. 769 bytes of JSON asked for 2.7 GB that way, and a count of 4e8 for
# 68 GB. Nothing real reads a million vertices from nothing.
MAX_VIRTUAL_COUNT = 2_000_000
# Deeper than the interpreter's own recursion limit allows the walk to follow,
# and deeper than any scene a person made.
MAX_NODE_DEPTH = 400


def _nat(value, what: str, lo: int = 0, hi: Optional[int] = None) -> int:
    """A JSON number that must be a whole number in range. `int()` alone turned
    "abc" into a bare ValueError, 3.9 into 3, and -3 into a garbage vertex."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GlbError("%s is %r, expected a whole number" % (what, value))
    if isinstance(value, float) and (value != value or value != int(value)):
        raise GlbError("%s is %r, expected a whole number" % (what, value))
    n = int(value)
    if n < lo or (hi is not None and n > hi):
        raise GlbError("%s is %d, outside %d..%s"
                       % (what, n, lo, "inf" if hi is None else hi))
    return n


def _accessor(gltf: dict, buffers: Dict[int, bytes], index: int,
              want: Optional[str] = None) -> np.ndarray:
    """Read accessor `index` as an (N, components) float64 or int64 array.

    Handles the three things that make accessors more than a memcpy: an
    interleaved bufferView (`byteStride` larger than one element), normalized
    integer components (KHR_mesh_quantization writes UVs as unsigned shorts),
    and sparse overrides.
    """
    accessors = gltf.get("accessors") or []
    if not 0 <= index < len(accessors):
        raise GlbError("accessor %d does not exist (%d accessors)"
                       % (index, len(accessors)))
    acc = accessors[index]
    if not isinstance(acc, dict):
        raise GlbError("accessor %d is not an object" % index)
    ctype = _nat(acc.get("componentType", 0), "accessor %d componentType" % index)
    if ctype not in _COMPONENT:
        raise GlbError("accessor %d has unknown componentType %d" % (index, ctype))
    kind = str(acc.get("type", ""))
    if kind not in _TYPE_COUNT:
        raise GlbError("accessor %d has unknown type %r" % (index, kind))
    if want and kind != want:
        raise GlbError("accessor %d is %s, expected %s" % (index, kind, want))

    dtype, csize, is_int = _COMPONENT[ctype]
    ncomp = _TYPE_COUNT[kind]
    count = _nat(acc.get("count", 0), "accessor %d count" % index)

    if "bufferView" not in acc:
        # The spec allows an accessor with no bufferView: every value is zero,
        # and a sparse block supplies the non-zero ones.
        if count > MAX_VIRTUAL_COUNT:
            raise GlbError(
                "accessor %d declares %d elements and has no bufferView to hold "
                "them; this reader allocates at most %d for an accessor with no "
                "data. The count is not the file's own."
                % (index, count, MAX_VIRTUAL_COUNT))
        data = np.zeros((count, ncomp), dtype=np.float64)
    else:
        data = _read_view(
            gltf, buffers, _nat(acc["bufferView"], "accessor %d bufferView" % index),
            _nat(acc.get("byteOffset", 0), "accessor %d byteOffset" % index),
            count, ncomp, dtype, csize, index)

    if "sparse" in acc:
        data = _apply_sparse(gltf, buffers, acc["sparse"], data, ncomp, dtype,
                             csize, index)

    if is_int and acc.get("normalized"):
        # Normalized integers encode [0,1] for unsigned and [-1,1] for signed.
        limit = float(np.iinfo(np.dtype(dtype)).max)
        if np.dtype(dtype).kind == "i":
            data = np.maximum(data / limit, -1.0)
        else:
            data = data / limit
    return data


def _read_view(gltf: dict, buffers: Dict[int, bytes], view_index: int,
               acc_offset: int, count: int, ncomp: int, dtype: str, csize: int,
               acc_index: int) -> np.ndarray:
    views = gltf.get("bufferViews") or []
    if not 0 <= view_index < len(views):
        raise GlbError("accessor %d points at bufferView %d, which does not exist"
                       % (acc_index, view_index))
    view = views[view_index]
    if not isinstance(view, dict):
        raise GlbError("bufferView %d is not an object" % view_index)
    buf_index = _nat(view.get("buffer", 0), "bufferView %d buffer" % view_index)
    if buf_index not in buffers:
        raise GlbError("bufferView %d points at buffer %d, which was not loaded"
                       % (view_index, buf_index))
    blob = buffers[buf_index]

    elem = ncomp * csize
    raw_stride = _nat(view.get("byteStride", 0), "bufferView %d byteStride"
                      % view_index, 0, 65536)
    if raw_stride and raw_stride < elem:
        raise GlbError(
            "bufferView %d has byteStride %d, smaller than the %d bytes of one "
            "element" % (view_index, raw_stride, elem))
    stride = raw_stride or elem
    start = _nat(view.get("byteOffset", 0), "bufferView %d byteOffset"
                 % view_index) + acc_offset
    if count == 0:
        return np.zeros((0, ncomp), dtype=np.float64)
    need = stride * (count - 1) + elem
    if start < 0 or start + need > len(blob):
        raise GlbError(
            "accessor %d reads bytes %d..%d of a %d-byte buffer. The file is "
            "truncated or its offsets are wrong."
            % (acc_index, start, start + need, len(blob)))

    raw = np.frombuffer(blob, dtype=np.uint8, count=need, offset=start)
    if stride == elem:
        vals = raw.view(dtype).reshape(count, ncomp)
    else:
        # Interleaved: gather each element's bytes, then reinterpret. copy()
        # first because .view() needs the result contiguous.
        picks = (np.arange(count, dtype=np.int64)[:, None] * stride
                 + np.arange(elem, dtype=np.int64)[None, :])
        vals = raw[picks].copy().view(dtype).reshape(count, ncomp)
    return vals.astype(np.float64)


def _apply_sparse(gltf: dict, buffers: Dict[int, bytes], sparse: dict,
                  data: np.ndarray, ncomp: int, dtype: str, csize: int,
                  acc_index: int) -> np.ndarray:
    """Overwrite the rows a sparse block names. Values carry the accessor's own
    component type; only the *indices* have a type of their own."""
    n = _nat(sparse.get("count", 0), "accessor %d sparse count" % acc_index)
    if n <= 0:
        return data
    if n > len(data):
        raise GlbError("accessor %d has a sparse block of %d overrides for %d "
                       "elements" % (acc_index, n, len(data)))
    idx_spec = sparse.get("indices") or {}
    val_spec = sparse.get("values") or {}
    if "bufferView" not in idx_spec or "bufferView" not in val_spec:
        raise GlbError("accessor %d has a sparse block missing a bufferView"
                       % acc_index)
    ictype = int(idx_spec.get("componentType", 5125))
    if ictype not in _COMPONENT:
        raise GlbError("accessor %d has a sparse index componentType of %d"
                       % (acc_index, ictype))
    idtype, icsize, _ = _COMPONENT[ictype]
    idx = _read_view(gltf, buffers, int(idx_spec["bufferView"]),
                     int(idx_spec.get("byteOffset", 0)), n, 1, idtype, icsize,
                     acc_index).reshape(-1).astype(np.int64)
    vals = _read_view(gltf, buffers, int(val_spec["bufferView"]),
                      int(val_spec.get("byteOffset", 0)), n, ncomp, dtype,
                      csize, acc_index)
    if idx.size and (idx.min() < 0 or idx.max() >= len(data)):
        raise GlbError("accessor %d has a sparse index outside the accessor"
                       % acc_index)
    out = data.copy()
    out[idx] = vals
    return out


# ------------------------------------------------------------------ transforms


def _node_matrix(node: dict) -> np.ndarray:
    """A node's local 4x4. glTF stores `matrix` column-major."""
    if "matrix" in node:
        m = np.asarray(node["matrix"], dtype=np.float64)
        if m.size != 16:
            raise GlbError("a node matrix has %d entries, expected 16" % m.size)
        return m.reshape(4, 4).T
    out = np.eye(4, dtype=np.float64)
    scale = np.asarray(node.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64)
    rot = np.asarray(node.get("rotation", (0.0, 0.0, 0.0, 1.0)), dtype=np.float64)
    trans = np.asarray(node.get("translation", (0.0, 0.0, 0.0)), dtype=np.float64)
    out[:3, :3] = _quat_matrix(rot) @ np.diag(scale)
    out[:3, 3] = trans
    return out


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    """glTF quaternions are (x, y, z, w) -- not (w, x, y, z)."""
    x, y, z, w = (float(v) for v in q[:4])
    n = x * x + y * y + z * z + w * w
    if n <= 0.0:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ], dtype=np.float64)


def _walk(gltf: dict) -> List[Tuple[int, np.ndarray, str]]:
    """Every mesh instance in the default scene: (mesh index, world 4x4, path).

    Falls back to every node, then to every mesh, because plenty of files in the
    wild have no scene at all and refusing them would be pedantry.
    """
    nodes = gltf.get("nodes") or []
    scenes = gltf.get("scenes") or []
    roots: List[int]
    if scenes:
        s = int(gltf.get("scene", 0))
        s = s if 0 <= s < len(scenes) else 0
        roots = [int(i) for i in (scenes[s].get("nodes") or [])]
    else:
        child_of = {int(c) for n in nodes for c in (n.get("children") or [])}
        roots = [i for i in range(len(nodes)) if i not in child_of]

    out: List[Tuple[int, np.ndarray, str]] = []
    if not nodes:
        for mi in range(len(gltf.get("meshes") or [])):
            out.append((mi, np.eye(4), "mesh[%d]" % mi))
        return out

    seen: set = set()

    def visit(index: int, parent: np.ndarray, trail: str, depth: int = 0) -> None:
        if index in seen or not 0 <= index < len(nodes):
            return  # a cycle is malformed, but must not hang the reader
        if depth > MAX_NODE_DEPTH:
            raise GlbError(
                "the node tree is more than %d levels deep, which no modelling "
                "tool writes and the reader will not recurse into" % MAX_NODE_DEPTH)
        seen.add(index)
        node = nodes[index]
        world = parent @ _node_matrix(node)
        label = _clean_name(node.get("name"), "node%d" % index)
        path = "%s/%s" % (trail, label) if trail else label
        if "mesh" in node:
            out.append((int(node["mesh"]), world, path))
        for child in node.get("children") or []:
            visit(int(child), world, path, depth + 1)
        seen.discard(index)  # a node may legitimately be reached once per parent

    for r in roots:
        visit(r, np.eye(4), "")
    return out


# ------------------------------------------------------------------ primitives


def _triangles(prim: dict, gltf: dict, buffers: Dict[int, bytes],
               n_verts: int) -> Tuple[np.ndarray, str]:
    """(T, 3) int64 triangle indices for one primitive, plus a note."""
    mode = int(prim.get("mode", 4))
    if "indices" in prim:
        idx = _accessor(gltf, buffers, int(prim["indices"]),
                        want="SCALAR").reshape(-1).astype(np.int64)
    else:
        idx = np.arange(n_verts, dtype=np.int64)

    if idx.size and (idx.min() < 0 or idx.max() >= n_verts):
        raise GlbError(
            "a primitive indexes vertex %d but its POSITION accessor has %d"
            % (int(idx.max()), n_verts))

    if mode == 4:
        return idx[:(len(idx) // 3) * 3].reshape(-1, 3), ""
    if mode == 5:  # TRIANGLE_STRIP: winding alternates
        if len(idx) < 3:
            return np.zeros((0, 3), np.int64), ""
        a, b, c = idx[:-2], idx[1:-1], idx[2:]
        tris = np.stack([a, b, c], axis=1)
        odd = np.arange(len(tris)) % 2 == 1
        tris[odd] = tris[odd][:, [0, 2, 1]]
        return tris, "converted a TRIANGLE_STRIP"
    if mode == 6:  # TRIANGLE_FAN
        if len(idx) < 3:
            return np.zeros((0, 3), np.int64), ""
        hub = np.full(len(idx) - 2, idx[0], dtype=np.int64)
        return np.stack([hub, idx[1:-1], idx[2:]], axis=1), "converted a TRIANGLE_FAN"
    return np.zeros((0, 3), np.int64), "skipped: mode %s carries no surface" % (
        _MODE_NAME.get(mode, str(mode)))


def _clean_name(raw: Any, fallback: str) -> str:
    """Names from the file reach tags, report text and the HTML viewer.

    Stripped to a safe set at the boundary rather than escaped at each use: a
    material called `</script><script>` would otherwise be an injection into
    every preview page written from an imported asset.
    """
    text = str(raw or "").strip()
    text = _SAFE_NAME.sub("_", text)[:64].strip("_ ")
    return text or fallback


# ------------------------------------------------------------------- assembly


def _weld(verts: np.ndarray, tris: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """Merge vertices that share a position. Returns (verts, tris, merged).

    Quantised to `tol` and compared as integers rather than compared as floats:
    -0.0 and 0.0 are distinct *bytes*, and numpy's row-wise unique compares
    bytes, so a vertex sitting exactly on an axis would fail to weld against its
    own twin. At the 1 nm default this never merges geometry that is genuinely
    distinct.

    Vertices come back in sorted order, not source order -- the row-wise unique
    sorts them. Triangle *order* is untouched, which is what lets the per-triangle
    channels stay in step; but nothing may assume a vertex kept its slot.
    """
    if len(verts) == 0:
        return verts, tris, 0
    quantum = max(float(tol), 1e-12)
    keys = np.rint(verts / quantum).astype(np.int64)
    keys += 0  # normalises -0 -> 0 after rint, so an on-axis vertex welds
    _uniq, first, inverse = np.unique(keys, axis=0, return_index=True,
                                      return_inverse=True)
    # `first` is ordered like the unique rows and `inverse` indexes into that
    # same ordering, so the two agree without any further permutation. The
    # surviving vertex keeps its *original* coordinates -- averaging would move
    # geometry by up to the tolerance for no benefit.
    return verts[first], inverse.reshape(-1)[tris], int(len(verts) - len(first))


def _degenerate_mask(tris: np.ndarray) -> np.ndarray:
    """True where a triangle uses the same vertex twice, so it has no area.

    Welding creates these: two corners an exporter split apart collapse onto one
    vertex. They carry no surface, and Manifold rejects any mesh holding one.
    """
    if len(tris) == 0:
        return np.zeros(0, dtype=bool)
    return ((tris[:, 0] == tris[:, 1]) | (tris[:, 1] == tris[:, 2])
            | (tris[:, 0] == tris[:, 2]))


def read(path: str, weld: bool = True, weld_tol: float = DEFAULT_WELD_TOL,
         up: str = "y", name: Optional[str] = None,
         tag: Optional[str] = None) -> GlbImport:
    """Read a .glb or .gltf into a neutral Mesh, and a Solid where possible.

    Anything the file's own structure does wrong -- a field of the wrong type, a
    missing array, a reference to nothing -- is a GlbError naming the file. The
    reader is handed attacker-controlled data (DESIGN.md section 13), and a bare
    KeyError or ValueError from inside it tells the caller nothing.
    """
    try:
        return _read(path, weld, weld_tol, up, name, tag)
    except GlbError:
        raise
    except (KeyError, IndexError, TypeError, ValueError, AttributeError,
            RecursionError, OverflowError) as exc:
        raise GlbError(
            "%s is not a glTF this reader can follow: %s: %s. A field has the "
            "wrong type, or a reference points at nothing."
            % (os.path.basename(path), type(exc).__name__, str(exc)[:160])) from None
    except MemoryError:
        raise GlbError(
            "%s asks for more memory than this reader will allocate"
            % os.path.basename(path)) from None


def _read(path: str, weld: bool, weld_tol: float, up: str,
          name: Optional[str], tag: Optional[str]) -> GlbImport:
    """The body of read(). See there.


    up: "y" for the glTF convention (converted to Lathe's Z-up), or "z" for a
        file already authored Z-up, which some exporters emit in defiance of the
        spec. Getting this wrong is visible immediately -- the asset lies down.
    """
    gltf, buffers, kind, digest = _read_container(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    asset_name = _clean_name(name or stem, "imported")

    refusals: List[str] = []
    notes: List[str] = []

    used = list(gltf.get("extensionsUsed") or [])
    required = list(gltf.get("extensionsRequired") or [])
    for ext in sorted(set(used) | set(required)):
        if ext in _GEOMETRY_EXTENSIONS:
            refusals.append("%s -- %s" % (ext, _GEOMETRY_EXTENSIONS[ext]))
    if refusals:
        raise GlbError(
            "%s uses geometry compression this reader does not decode:\n    %s\n"
            "  Decoding it wrongly would produce plausible-looking rubbish, so "
            "it is refused rather than guessed at."
            % (os.path.basename(path), "\n    ".join(refusals)))
    unknown_required = [e for e in required
                        if e not in _HANDLED_EXTENSIONS
                        and e not in _GEOMETRY_EXTENSIONS]
    if unknown_required:
        notes.append(
            "loaded geometry despite required extension(s) %s, which this reader "
            "does not implement -- they cannot affect vertex decoding"
            % ", ".join(sorted(unknown_required)))

    meshes = gltf.get("meshes") or []
    if not meshes:
        raise GlbError(
            "%s contains no meshes. It may be a materials-only or animation-only "
            "file." % os.path.basename(path))

    instances = _walk(gltf)
    if not instances:
        raise GlbError(
            "%s has %d mesh(es) but no scene node references any of them, so "
            "there is nothing to place." % (os.path.basename(path), len(meshes)))

    materials = gltf.get("materials") or []
    to_zup = YUP_TO_ZUP if str(up).lower() == "y" else np.eye(3)

    v_chunks: List[np.ndarray] = []
    t_chunks: List[np.ndarray] = []
    part_chunks: List[np.ndarray] = []
    # slot -> [(per-vertex uv, primitive-local tris, first triangle row)]. The
    # triangle row is recorded rather than accumulated, because a primitive with
    # no TEXCOORD_1 contributes nothing to that slot's list and a running cursor
    # would then hand its rows to the next primitive that does.
    uv_chunks: Dict[str, List[Tuple[np.ndarray, np.ndarray, int]]] = {}
    parts: Dict[int, str] = {}
    part_tris: Dict[str, int] = {}
    skipped: List[str] = []
    mirrored = 0
    max_scale_dev = 0.0
    n_uv_missing = 0
    base = 0        # vertices appended so far
    tri_base = 0    # triangles appended so far
    next_part = 0

    for mesh_index, world, trail in instances:
        if not 0 <= mesh_index < len(meshes):
            skipped.append("%s references mesh %d, which does not exist"
                           % (trail, mesh_index))
            continue
        linear = world[:3, :3]
        det = float(np.linalg.det(linear))
        # Column norms are the per-axis scale; report the worst deviation so the
        # E104 signal that baking destroys is at least *stated*.
        for col in range(3):
            max_scale_dev = max(max_scale_dev,
                                abs(float(np.linalg.norm(linear[:, col])) - 1.0))
        if det < 0:
            mirrored += 1

        mesh_label = _clean_name(meshes[mesh_index].get("name"),
                                 "mesh%d" % mesh_index)
        for pi, prim in enumerate(meshes[mesh_index].get("primitives") or []):
            attrs = prim.get("attributes") or {}
            if "POSITION" not in attrs:
                skipped.append("%s primitive %d has no POSITION" % (trail, pi))
                continue
            pos = _accessor(gltf, buffers, int(attrs["POSITION"]), want="VEC3")
            tris, note = _triangles(prim, gltf, buffers, len(pos))
            if note.startswith("skipped"):
                skipped.append("%s primitive %d %s" % (trail, pi, note))
                continue
            if note:
                notes.append("%s primitive %d: %s" % (trail, pi, note))
            if len(tris) == 0:
                skipped.append("%s primitive %d has no triangles" % (trail, pi))
                continue

            # Bake: node transform, then the global up-axis conversion.
            world_pos = (pos @ linear.T) + world[:3, 3]
            world_pos = world_pos @ to_zup.T
            if det < 0:
                tris = tris[:, [0, 2, 1]]  # mirroring reverses winding

            mat = prim.get("material")
            if mat is not None and 0 <= int(mat) < len(materials):
                label = _clean_name(materials[int(mat)].get("name"),
                                    "material%d" % int(mat))
            else:
                label = mesh_label
            parts[next_part] = label
            part_tris[label] = part_tris.get(label, 0) + len(tris)

            v_chunks.append(world_pos)
            t_chunks.append(tris + base)
            part_chunks.append(np.full(len(tris), next_part, dtype=np.int64))

            found_uv = False
            for key, acc_index in attrs.items():
                if not key.startswith("TEXCOORD_"):
                    continue
                slot = "UV%s" % key.split("_", 1)[1]
                uv = _accessor(gltf, buffers, int(acc_index), want="VEC2")
                if len(uv) != len(pos):
                    skipped.append("%s primitive %d has %d %s for %d vertices"
                                   % (trail, pi, len(uv), key, len(pos)))
                    continue
                uv_chunks.setdefault(slot, []).append((uv, tris, tri_base))
                found_uv = True
            if not found_uv:
                n_uv_missing += len(tris)

            base += len(pos)
            tri_base += len(tris)
            next_part += 1

    if not v_chunks:
        raise GlbError(
            "%s yielded no triangles.%s"
            % (os.path.basename(path),
               ("\n  " + "\n  ".join(skipped[:8])) if skipped else ""))

    verts = np.concatenate(v_chunks, axis=0)
    tris = np.concatenate(t_chunks, axis=0)
    part_id = np.concatenate(part_chunks, axis=0)

    # Per-corner UV, built before welding: the whole point of the per-corner
    # shape is that a welded vertex can still carry two UVs at a seam.
    uv_sets: Dict[str, np.ndarray] = {}
    for slot, chunks in uv_chunks.items():
        rows = np.zeros((len(tris), 3, 2), dtype=np.float64)
        for uv, local_tris, first in chunks:
            # glTF UV origin is top-left; the rest of this pipeline, like
            # Blender and OBJ, is bottom-left.
            flipped = np.stack([uv[:, 0], 1.0 - uv[:, 1]], axis=1)
            rows[first:first + len(local_tris)] = flipped[local_tris]
        uv_sets[slot] = rows

    verts_in = len(verts)
    merged = 0
    if weld:
        verts, tris, merged = _weld(verts, tris, weld_tol)

    # One mask, applied to every per-triangle channel at once. Filtering the
    # triangles and then rebuilding the mask separately for the others is how
    # channels drift out of step with the geometry they describe.
    bad = _degenerate_mask(tris)
    n_degenerate = int(np.count_nonzero(bad))
    if n_degenerate:
        keep = ~bad
        tris = tris[keep]
        part_id = part_id[keep]
        for slot in list(uv_sets):
            uv_sets[slot] = uv_sets[slot][keep]

    msh = pmesh.Mesh(
        verts=verts, tris=tris, name=asset_name,
        part_id=part_id, parts=parts,
        uv=uv_sets.get("UV0"), uv_set="UV0",
        source=None,
    )

    topo = msh.topology
    solid, reason = pmesh.to_manifold(msh)
    lathe_solid: Optional[Solid] = None
    if solid is not None:
        # One tag, not one per material. `as_original()` gives the rebuilt solid
        # a single id -- safe here, and only here, because an import has no
        # ancestry to destroy (see solid._original). Claiming the material split
        # survives a later boolean would be exactly the provenance drift
        # DESIGN.md section 13 records: the Mesh keeps the material parts, and
        # the Solid does not pretend to.
        man = solid.as_original()
        lathe_solid = Solid(man, {int(man.original_id()):
                                  _clean_name(tag, "imported")}, asset_name)
    else:
        reason = _promotion_reason(topo, reason)

    report = {
        "file": {
            "path": os.path.abspath(path),
            "container": kind,
            "bytes": os.path.getsize(path),
            "sha256": digest,
            "generator": _clean_name((gltf.get("asset") or {}).get("generator"),
                                     "unknown"),
            "gltf_version": str((gltf.get("asset") or {}).get("version", "?")),
        },
        "scene": {
            "meshes": len(meshes),
            "instances": len(instances),
            "primitives_merged": len(v_chunks),
            "primitives_skipped": len(skipped),
            "materials": len(materials),
        },
        "axes": ("glTF Y-up -> Lathe Z-up (+90 about X)" if str(up).lower() == "y"
                 else "taken as Z-up already, no conversion"),
        "transform": {
            "baked": True,
            "max_scale_deviation": round(max_scale_dev, 9),
            "mirrored_instances": mirrored,
        },
        "weld": {
            "enabled": bool(weld),
            "tol_m": weld_tol if weld else None,
            "verts_before": verts_in,
            "verts_after": int(len(verts)),
            "merged": merged,
        },
        "clean": {"degenerate_tris_dropped": n_degenerate},
        "topology": {
            "tris": int(msh.n_tris),
            "verts": int(msh.n_verts),
            "edges": int(topo.n_edges),
            "components": int(topo.n_components),
            "boundary_edges": int(len(topo.boundary_edges)),
            "nonmanifold_edges": int(len(topo.nonmanifold_edges)),
            "duplicate_halfedges": int(topo.duplicate_halfedges),
            "closed": bool(topo.is_closed),
            "oriented": bool(topo.is_oriented),
        },
        "uv": {
            "sets": sorted(uv_sets),
            "tris_without_uv": n_uv_missing,
        },
        "parts": dict(sorted(part_tris.items(), key=lambda kv: -kv[1])),
        "extensions": {"used": sorted(used), "required": sorted(required)},
        "notes": notes,
        "skipped": skipped[:20],
        "promoted_to_solid": lathe_solid is not None,
    }
    return GlbImport(path=os.path.abspath(path), mesh=msh, solid=lathe_solid,
                     reason=None if lathe_solid is not None else reason,
                     report=report, uv_sets=uv_sets)


def _promotion_reason(topo, kernel_reason: Optional[str]) -> str:
    """Say which property failed, not just that the kernel said no.

    `Error.NotManifold` is true and useless. An agent needs to know whether it is
    holding an open shell (bake it, do not boolean it), a soup with reversed
    faces (recalculate normals), or several shells that never met.
    """
    bits = []
    if len(topo.boundary_edges):
        bits.append(
            "%d boundary edge(s): the surface is open, so it encloses no volume. "
            "This is normal for a scanned or sculpted asset, and for anything "
            "exported as a shell." % len(topo.boundary_edges))
    if len(topo.nonmanifold_edges):
        bits.append(
            "%d edge(s) shared by more than two triangles: the surface branches, "
            "which no solid can do." % len(topo.nonmanifold_edges))
    if topo.duplicate_halfedges:
        bits.append(
            "%d duplicated directed edge(s): triangle winding disagrees, or a "
            "triangle appears twice." % topo.duplicate_halfedges)
    if not bits:
        bits.append(kernel_reason or "the kernel rejected the mesh")
    return (
        "not a solid.\n    " + "\n    ".join(bits)
        + "\n  The Mesh is still fully usable: every Tier 1 probe and every "
          "verifier rule reads it. Booleans, bevel and volume do not."
    )


# ------------------------------------------------------------------ formatting


def _format_report(imp: GlbImport) -> str:
    r = imp.report
    f, s, t, w = r["file"], r["scene"], r["topology"], r["weld"]
    lines = [
        "  %s   %s   %d bytes   sha256 %s"
        % (os.path.basename(f["path"]), f["container"], f["bytes"], f["sha256"]),
        "  written by %s (glTF %s)" % (f["generator"], f["gltf_version"]),
        "",
        "  scene      %d mesh(es), %d instance(s), %d primitive(s) merged, "
        "%d material(s)" % (s["meshes"], s["instances"], s["primitives_merged"],
                            s["materials"]),
        "  axes       %s" % r["axes"],
        "  transform  baked; worst axis scale deviates %.4g from 1%s"
        % (r["transform"]["max_scale_deviation"],
           ", %d mirrored instance(s) rewound" % r["transform"]["mirrored_instances"]
           if r["transform"]["mirrored_instances"] else ""),
    ]
    if w["enabled"]:
        lines.append("  weld       %d -> %d verts (%d merged at %g m)"
                     % (w["verts_before"], w["verts_after"], w["merged"],
                        w["tol_m"]))
    else:
        lines.append("  weld       off -- %d verts as stored" % w["verts_after"])
    if r["clean"]["degenerate_tris_dropped"]:
        lines.append("  clean      dropped %d degenerate triangle(s)"
                     % r["clean"]["degenerate_tris_dropped"])
    lines += [
        "  topology   %d tris, %d verts, %d edges, %d component(s)"
        % (t["tris"], t["verts"], t["edges"], t["components"]),
        "             closed %s   oriented %s   boundary %d   non-manifold %d"
        % (t["closed"], t["oriented"], t["boundary_edges"], t["nonmanifold_edges"]),
        "  uv         %s"
        % (", ".join(r["uv"]["sets"]) if r["uv"]["sets"] else "none"),
    ]
    if r["parts"]:
        lines.append("  parts      " + "; ".join(
            "%s %d" % (k, v) for k, v in list(r["parts"].items())[:8]))
    if imp.is_solid:
        lines += ["", "  PROMOTED TO SOLID -- booleans, bevel and volume all work."]
    else:
        lines += ["", "  MESH ONLY -- %s" % imp.reason]
    for note in r["notes"][:6]:
        lines.append("  note       %s" % note)
    for skip in r["skipped"][:6]:
        lines.append("  skipped    %s" % skip)
    return "\n".join(lines)


# ------------------------------------------------------------ program-facing


def import_glb(path: str, weld: bool = True, up: str = "y",
               tag: str = "imported", name: Optional[str] = None) -> Solid:
    """Read a .glb as a Solid, or raise saying which property failed.

    This is the form a Lathe program uses, so an import can be the first step of
    a program and the commit replays like any other. It raises rather than
    returning None because a program that carries on with no geometry produces a
    worse error later, further from the cause.

    Purity caveat: every other operation in this kernel is a function of its
    arguments alone. This one reads a file, so a program that imports is only
    reproducible while that file is unchanged. `read()` reports the content hash
    for exactly that reason.
    """
    imp = read(path, weld=weld, up=up, tag=tag, name=name)
    if imp.solid is None:
        raise GlbError(
            "%s is %s\n  Use lathe.io.glb.read() to get the Mesh, which can "
            "still be measured, validated and viewed."
            % (os.path.basename(path), imp.reason))
    return imp.solid
