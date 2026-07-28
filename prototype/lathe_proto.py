"""
Lathe DESIGN.md section 5 -- selector layer prototype.

Tests three claims from the design doc:
  5.1  selectors as queries (predicates over attributes), not indices
  5.2  provenance tags that survive topology-changing operations
  5.3  cardinality assertions that turn silent mismatch into a loud error

Runs inside Blender (bpy + bmesh). No bpy.ops, no mode switches, no selection
state -- everything is a function of its explicit arguments, per the design
principle in section 2.
"""

import bmesh
import bpy
from mathutils import Vector

TAG_ATTR = "lathe_tag"


class LatheError(Exception):
    pass


class CardinalityError(LatheError):
    pass


# ---------------------------------------------------------------- tags (5.2)

_TAG_BITS = {}


def tag_bit(name):
    """Stable bit for a tag name. Tags are a bitmask so a face can carry several."""
    if name not in _TAG_BITS:
        if len(_TAG_BITS) >= 31:
            raise LatheError("prototype supports at most 31 distinct tags")
        _TAG_BITS[name] = 1 << len(_TAG_BITS)
    return _TAG_BITS[name]


def mask_of(tags):
    m = 0
    for t in tags:
        m |= tag_bit(t)
    return m


def tags_of(mask):
    return sorted(n for n, b in _TAG_BITS.items() if mask & b)


# ------------------------------------------------------------- solid + query

class Sel:
    """A resolved selection. The real system keeps these lazy; the prototype
    resolves eagerly so failures are easy to read."""

    def __init__(self, solid, domain, indices, expr):
        self.solid = solid
        self.domain = domain
        self.indices = list(indices)
        self.expr = expr

    def count(self):
        return len(self.indices)

    def __repr__(self):
        return "<Sel %s %s n=%d>" % (self.domain, self.expr, len(self.indices))


class Solid:
    def __init__(self, obj):
        self.obj = obj

    @property
    def name(self):
        return self.obj.name

    def open(self):
        bm = bmesh.new()
        bm.from_mesh(self.obj.data)
        bm.faces.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        return bm

    def commit(self, bm):
        bm.to_mesh(self.obj.data)
        bm.free()
        self.obj.data.update()

    def face_layer(self, bm):
        lay = bm.faces.layers.int.get(TAG_ATTR)
        if lay is None:
            lay = bm.faces.layers.int.new(TAG_ATTR)
        return lay

    # -- 5.1 selectors -----------------------------------------------------

    def faces(self, normal=None, normal_tol=0.15, area_gt=None, area_lt=None,
              tag=None, all=False):
        bm = self.open()
        lay = self.face_layer(bm)
        terms, hits = [], []
        if all:
            terms.append("all")
        if normal is not None:
            terms.append("normal ~ %s" % (tuple(normal),))
        if area_gt is not None:
            terms.append("area > %g" % area_gt)
        if area_lt is not None:
            terms.append("area < %g" % area_lt)
        if tag is not None:
            terms.append("#%s" % tag)
        want = Vector(normal).normalized() if normal is not None else None
        bit = tag_bit(tag) if tag is not None else None
        for f in bm.faces:
            if want is not None and (f.normal.normalized() - want).length > normal_tol:
                continue
            if area_gt is not None and f.calc_area() <= area_gt:
                continue
            if area_lt is not None and f.calc_area() >= area_lt:
                continue
            if bit is not None and not (f[lay] & bit):
                continue
            hits.append(f.index)
        bm.free()
        return Sel(self, "faces", hits, "faces(%s)" % ", ".join(terms) or "faces()")

    def edges(self, dihedral_gt=None, dihedral_lt=None, length_gt=None,
              tag=None, all=False):
        import math
        bm = self.open()
        lay = self.face_layer(bm)
        terms, hits = [], []
        if all:
            terms.append("all")
        if dihedral_gt is not None:
            terms.append("dihedral > %gdeg" % dihedral_gt)
        if dihedral_lt is not None:
            terms.append("dihedral < %gdeg" % dihedral_lt)
        if length_gt is not None:
            terms.append("length > %g" % length_gt)
        if tag is not None:
            terms.append("#%s on both faces" % tag)
        bit = tag_bit(tag) if tag is not None else None
        for e in bm.edges:
            if len(e.link_faces) != 2:
                continue
            ang = math.degrees(e.calc_face_angle(0.0))
            if dihedral_gt is not None and ang <= dihedral_gt:
                continue
            if dihedral_lt is not None and ang >= dihedral_lt:
                continue
            if length_gt is not None and e.calc_length() <= length_gt:
                continue
            if bit is not None and not all((f[lay] & bit) for f in e.link_faces):
                continue
            hits.append(e.index)
        bm.free()
        return Sel(self, "edges", hits, "edges(%s)" % ", ".join(terms) or "edges()")

    # -- provenance readout -------------------------------------------------

    def tag_census(self):
        """Face count per tag combination -- the evidence for 5.2."""
        bm = self.open()
        lay = self.face_layer(bm)
        census = {}
        for f in bm.faces:
            key = tuple(tags_of(f[lay])) or ("<untagged>",)
            census[key] = census.get(key, 0) + 1
        bm.free()
        return census

    def counts(self):
        me = self.obj.data
        return {"verts": len(me.vertices), "edges": len(me.edges),
                "faces": len(me.polygons)}


# ----------------------------------------------------------- 5.3 assertions

def assert_count(sel, expected, label=None):
    if sel.count() != expected:
        raise CardinalityError(
            "assert count(%s) == %d\n"
            "  actual   : %d\n"
            "  selector : %s\n"
            "  on       : %s (%s)\n"
            "  meaning  : the selector matched a different set than the program\n"
            "             assumed. An upstream parameter probably changed."
            % (label or sel.expr, expected, sel.count(), sel.expr,
               sel.solid.name, sel.solid.counts())
        )
    return sel


# ------------------------------------------------------------------- ops

def _link(name, bm):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return Solid(obj)


def box(name, sx, sy, sz, location=(0, 0, 0), tags=()):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector((sx, sy, sz)), verts=bm.verts)
    bmesh.ops.translate(bm, vec=Vector(location), verts=bm.verts)
    lay = bm.faces.layers.int.new(TAG_ATTR)
    m = mask_of(tags)
    for f in bm.faces:
        f[lay] = m
    return _link(name, bm)


def difference(a, cutters, cut_tag="cut_face", preserve_layers=True):
    """a - cutters, via the Exact solver. Cutter faces carry the tag BEFORE the
    boolean. If the tag shows up on the result, provenance survived.

    preserve_layers maps to new_from_object(preserve_all_data_layers=...),
    which defaults to False in bpy -- experiment 1b measures what that costs."""
    for c in cutters:
        bm = c.open()
        lay = c.face_layer(bm)
        for f in bm.faces:
            f[lay] = mask_of([cut_tag])
        c.commit(bm)

    for c in cutters:
        mod = a.obj.modifiers.new(name="bool_%s" % c.name, type="BOOLEAN")
        mod.operation = "DIFFERENCE"
        mod.solver = "EXACT"
        mod.object = c.obj

    dg = bpy.context.evaluated_depsgraph_get()
    evaluated = a.obj.evaluated_get(dg)
    baked = bpy.data.meshes.new_from_object(
        evaluated, preserve_all_data_layers=preserve_layers, depsgraph=dg)
    a.obj.modifiers.clear()
    old = a.obj.data
    a.obj.data = baked
    bpy.data.meshes.remove(old)
    return a


def bevel(solid, sel, width, segments=2, new_tag="bevel", inherit=True):
    """inherit=True lets bmesh's customdata interpolation carry ancestor tags
    onto the new faces. That is what makes faces(#cut_face) drift after a
    bevel -- see EXP 4. inherit=False gives new faces the new tag only."""
    if sel.domain != "edges":
        raise LatheError("bevel needs an edge selection, got %s" % sel.domain)
    bm = solid.open()
    lay = solid.face_layer(bm)
    edges = [bm.edges[i] for i in sel.indices]
    res = bmesh.ops.bevel(bm, geom=edges, offset=width, segments=segments,
                          affect="EDGES", clamp_overlap=True)
    bit = mask_of([new_tag])
    for f in res.get("faces", []):
        f[lay] = (f[lay] | bit) if inherit else bit
    solid.commit(bm)
    return solid


def delete(solid):
    me = solid.obj.data
    bpy.data.objects.remove(solid.obj)
    bpy.data.meshes.remove(me)


def reset_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for m in list(bpy.data.meshes):
        bpy.data.meshes.remove(m)
