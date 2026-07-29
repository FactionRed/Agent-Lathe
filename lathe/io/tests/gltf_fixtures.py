"""
Build .glb and .gltf files by hand, so the reader can be tested without Blender.

Every fixture here is a *deliberately awkward* container: interleaved accessors,
quantised UVs, column-major node matrices, mirrored scales, triangle strips. Those
are the parts of glTF a reader gets wrong, and a fixture produced by an exporter
would never exercise them -- an exporter writes the one shape it likes.

Written against the spec rather than against the reader, so the two can disagree.
"""

from __future__ import annotations

import base64
import json
import struct
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

FLOAT = 5126
USHORT = 5123
UBYTE = 5121
UINT = 5125


def _pad4(data: bytes, fill: bytes = b"\x00") -> bytes:
    return data + fill * (-len(data) % 4)


class Builder:
    """Accumulates buffer views and accessors into a glTF JSON document."""

    def __init__(self) -> None:
        self.bin = bytearray()
        self.views: List[dict] = []
        self.accessors: List[dict] = []

    def view(self, data: bytes, stride: Optional[int] = None) -> int:
        # Views start 4-byte aligned so every component type stays aligned.
        while len(self.bin) % 4:
            self.bin.append(0)
        view: Dict[str, Any] = {"buffer": 0, "byteOffset": len(self.bin),
                                "byteLength": len(data)}
        if stride:
            view["byteStride"] = stride
        self.bin.extend(data)
        self.views.append(view)
        return len(self.views) - 1

    def accessor(self, arr, comp: int, kind: str, normalized: bool = False,
                 stride: Optional[int] = None, offset: int = 0,
                 view_index: Optional[int] = None, count: Optional[int] = None) -> int:
        dtype = {FLOAT: "<f4", USHORT: "<u2", UBYTE: "<u1", UINT: "<u4"}[comp]
        arr = np.asarray(arr)
        n = count if count is not None else len(arr)
        if view_index is None:
            view_index = self.view(_pad4(np.ascontiguousarray(arr, dtype=dtype).tobytes()),
                                   stride)
        acc: Dict[str, Any] = {"bufferView": view_index, "componentType": comp,
                               "count": int(n), "type": kind}
        if offset:
            acc["byteOffset"] = offset
        if normalized:
            acc["normalized"] = True
        if kind == "VEC3" and comp == FLOAT:
            a = np.asarray(arr, dtype=np.float64).reshape(-1, 3)
            acc["min"] = [float(x) for x in a.min(axis=0)]
            acc["max"] = [float(x) for x in a.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1


def primitive(b: Builder, pos, tris=None, uv=None, uv1=None, material=None,
              mode: Optional[int] = None, indices_comp: int = USHORT,
              indexed: bool = True) -> dict:
    """One glTF primitive. `tris` may be an (N,) index list for strips and fans."""
    prim: Dict[str, Any] = {"attributes": {
        "POSITION": b.accessor(np.asarray(pos, dtype=np.float32), FLOAT, "VEC3")}}
    if uv is not None:
        prim["attributes"]["TEXCOORD_0"] = b.accessor(
            np.asarray(uv, dtype=np.float32), FLOAT, "VEC2")
    if uv1 is not None:
        prim["attributes"]["TEXCOORD_1"] = b.accessor(
            np.asarray(uv1, dtype=np.float32), FLOAT, "VEC2")
    if tris is not None and indexed:
        flat = np.asarray(tris).reshape(-1)
        prim["indices"] = b.accessor(flat, indices_comp, "SCALAR")
    if material is not None:
        prim["material"] = material
    if mode is not None:
        prim["mode"] = mode
    return prim


def document(b: Builder, meshes: Sequence[Sequence[dict]],
             nodes: Optional[List[dict]] = None,
             materials: Optional[List[dict]] = None,
             scene_nodes: Optional[List[int]] = None,
             required: Optional[List[str]] = None,
             used: Optional[List[str]] = None,
             buffer_uri: Optional[str] = None,
             generator: str = "lathe test fixture") -> dict:
    if nodes is None:
        nodes = [{"mesh": i, "name": "n%d" % i} for i in range(len(meshes))]
    doc: Dict[str, Any] = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": scene_nodes if scene_nodes is not None
                    else list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": [{"name": "m%d" % i, "primitives": list(p)}
                   for i, p in enumerate(meshes)],
        "accessors": b.accessors,
        "bufferViews": b.views,
        "buffers": [{"byteLength": len(b.bin)}],
    }
    if buffer_uri is not None:
        doc["buffers"][0]["uri"] = buffer_uri
    if materials:
        doc["materials"] = materials
    if required:
        doc["extensionsRequired"] = required
    if used or required:
        doc["extensionsUsed"] = sorted(set((used or []) + (required or [])))
    return doc


def glb_bytes(doc: dict, blob: bytes) -> bytes:
    """Wrap JSON and BIN into the 12-byte-header container."""
    js = _pad4(json.dumps(doc).encode("utf-8"), b" ")
    bn = _pad4(bytes(blob))
    length = 12 + 8 + len(js) + (8 + len(bn) if bn else 0)
    out = struct.pack("<III", 0x46546C67, 2, length)
    out += struct.pack("<II", len(js), 0x4E4F534A) + js
    if bn:
        out += struct.pack("<II", len(bn), 0x004E4942) + bn
    return out


def write_glb(path: str, doc: dict, blob: bytes) -> str:
    with open(path, "wb") as fh:
        fh.write(glb_bytes(doc, blob))
    return path


def write_gltf(path: str, doc: dict, blob: bytes, embed: bool = True) -> str:
    """A .gltf, with its buffer either base64-embedded or in a sidecar .bin."""
    import os
    if embed:
        doc = dict(doc)
        doc["buffers"] = [{"byteLength": len(blob),
                           "uri": "data:application/octet-stream;base64,"
                                  + base64.b64encode(bytes(blob)).decode("ascii")}]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh)
    if not embed:
        side = os.path.splitext(path)[0] + ".bin"
        with open(side, "wb") as fh:
            fh.write(bytes(blob))
    return path


# --------------------------------------------------------------- shapes

def cube_split(size: float = 1.0, centre=(0.0, 0.0, 0.0)):
    """A cube with 24 vertices -- four per face, the way an exporter writes it.

    This is the fixture that matters most. Nothing welds it back to 8 vertices
    for you, and until it is welded every one of its 12 edges is a boundary.
    Coordinates are glTF-convention (Y-up).
    """
    h = size / 2.0
    cx, cy, cz = centre
    faces = [
        ((-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)),        # +Z
        ((h, -h, -h), (-h, -h, -h), (-h, h, -h), (h, h, -h)),    # -Z
        ((h, -h, h), (h, -h, -h), (h, h, -h), (h, h, h)),        # +X
        ((-h, -h, -h), (-h, -h, h), (-h, h, h), (-h, h, -h)),    # -X
        ((-h, h, h), (h, h, h), (h, h, -h), (-h, h, -h)),        # +Y
        ((-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)),    # -Y
    ]
    pos, tris = [], []
    for quad in faces:
        base = len(pos)
        pos.extend([(x + cx, y + cy, z + cz) for x, y, z in quad])
        tris.append((base, base + 1, base + 2))
        tris.append((base, base + 2, base + 3))
    return (np.array(pos, dtype=np.float32),
            np.array(tris, dtype=np.int64))


def open_quad():
    """Two triangles: a surface with a boundary, so never a solid."""
    pos = np.array([(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], dtype=np.float32)
    tris = np.array([(0, 1, 2), (0, 2, 3)], dtype=np.int64)
    return pos, tris
