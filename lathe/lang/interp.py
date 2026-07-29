"""
Evaluator: .lathe AST -> geometry, via lathe.build.

The language is a thin, *typed* skin over the authoring kernel. Its job is not
expressive power -- it has no loops and no user functions -- but to make the two
things DESIGN.md keeps insisting on unmissable:

  * a bare number is not a length (4.1), so `box(600, 600, 600)` fails to compile
    instead of silently building a 600-metre crate;
  * a selector reads as a predicate (5.1), so `edges(dihedral > 60deg)` says what
    it means and the angle carries its unit.

Errors follow the section 7 shape: what broke, where, and what to do about it.

SECURITY -- and this is the part that matters beyond ergonomics. `Interp.eval` is
an **AST walker**, not Python's `eval`. Nothing here compiles or executes source:
a program can only reach the handful of `fn_*` methods below, there is no
attribute access into host objects, no imports, and no builtins in scope.

That makes this path strictly safer than the Python escape hatch in
`mcp/session.py`, which runs `exec` in a namespace that still has builtins and is
therefore a restricted namespace rather than a sandbox. A `.lathe` program is
inert text; a Python program is code. If the server ever needs to accept programs
from somewhere less trusted than the operator's own keyboard, this is the path
that can be allowed and that one is not.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .. import build as B
from ..build.solid import CardinalityError, LatheError, Selection, Solid
from . import syntax as S
from . import values as V
from .values import UnitError


class LatheRuntimeError(LatheError):
    """A failure while evaluating, located in the source."""

    def __init__(self, msg: str, node: Optional[S.Node] = None,
                 src_line: str = "", hint: str = ""):
        self.node = node
        where = ""
        if node is not None and node.line:
            where = "line %d, column %d: " % (node.line, node.col)
        body = where + msg
        if src_line:
            body += "\n    %s" % src_line.rstrip()
        if hint:
            body += "\n  %s" % hint
        super().__init__(body)


AXES = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
ORIGINS = ("bottom_center", "center", "as_built")

# Selector predicate attribute -> (domain, kwarg builder)
_EDGE_PREDS = {
    "dihedral": {">": "dihedral_gt", "<": "dihedral_lt"},
    "length": {">": "length_gt"},
}
_FACE_PREDS = {
    "area": {">": "area_gt", "<": "area_lt"},
    "z": {">": "z_gt", "<": "z_lt"},
}


class Interp:
    def __init__(self, src: str):
        self.src = src
        self.lines = src.splitlines()
        self.env: Dict[str, V.Value] = {}
        self.params: Dict[str, V.Value] = {}
        self.settings: Dict[str, Dict[str, str]] = {}
        self.materials: Dict[str, Any] = {}   # tag -> Selection
        self.uv_requests: List[Tuple[Any, str, Dict[str, Any]]] = []
        self.emit_name: Optional[str] = None
        self.emit_options: Dict[str, Any] = {}
        self.result: Optional[Solid] = None
        self.assertions = 0

    # -- helpers ----------------------------------------------------------

    def line_of(self, node: S.Node) -> str:
        if node is not None and 0 < node.line <= len(self.lines):
            return self.lines[node.line - 1]
        return ""

    def die(self, msg: str, node: S.Node = None, hint: str = ""):
        raise LatheRuntimeError(msg, node, self.line_of(node), hint)

    # -- entry ------------------------------------------------------------

    def run(self, prog: S.Program) -> Solid:
        for stmt in prog.statements:
            self.statement(stmt)
        if self.result is None:
            self.die(
                "the program never produced a solid",
                hint=("End with `emit \"NAME\" { ... }`, or bind the finished "
                      "solid to a name called `result`."))
        return self.result

    def statement(self, node: S.Node) -> None:
        if isinstance(node, S.Param):
            self.params[node.name] = self.env[node.name] = self.eval(node.expr)
        elif isinstance(node, S.Let):
            val = self.eval(node.expr)
            self.env[node.name] = val
            if isinstance(val, V.Opaque) and isinstance(val.obj, Solid):
                # The last solid bound wins, so a program without `emit` still
                # has an obvious answer.
                self.result = val.obj
                if node.name == "result":
                    self.result = val.obj
        elif isinstance(node, S.Assert):
            self.do_assert(node)
        elif isinstance(node, S.Setting):
            self.settings.setdefault(node.group, {}).update(node.items)
        elif isinstance(node, S.Material):
            self.do_material(node)
        elif isinstance(node, S.UVSet):
            self.do_uv(node)
        elif isinstance(node, S.Emit):
            self.do_emit(node)
        else:
            self.die("cannot execute %s" % type(node).__name__, node)

    # -- statements -------------------------------------------------------

    def do_assert(self, node: S.Assert) -> None:
        # `count(sel) == n` is routed to assert_count so the failure carries the
        # selector and the actual set, not just "false".
        e = node.expr
        if (isinstance(e, S.Binary) and e.op == "==" and
                isinstance(e.left, S.Call) and e.left.func == "count" and
                len(e.left.args) == 1):
            sel = self.eval(e.left.args[0])
            want = self.eval(e.right)
            if not isinstance(sel, V.Opaque) or not isinstance(sel.obj, Selection):
                self.die("count() needs a selection, got %s" % sel.describe(), node)
            n = V.as_int(want, "the expected count")
            try:
                B.assert_count(sel.obj, n, node.text)
            except CardinalityError as exc:
                raise LatheRuntimeError(
                    "assertion failed\n  %s" % str(exc).replace("\n", "\n  "),
                    node, self.line_of(node)) from None
            self.assertions += 1
            return

        val = self.eval(e)
        if not isinstance(val, V.Bool):
            self.die(
                "assert needs a true/false expression, got %s" % val.describe(),
                node, "Write a comparison, e.g. `assert components(x) == 1`.")
        self.assertions += 1
        if not val.b:
            self.die("assertion failed: %s" % node.text, node,
                     "The program's own stated expectation does not hold.")

    def do_material(self, node: S.Material) -> None:
        sel = self.eval(node.target)
        if not isinstance(sel, V.Opaque):
            self.die("`mat ... on` needs a face selection", node)
        self.materials[node.name] = sel.obj

    def do_uv(self, node: S.UVSet) -> None:
        """`uv final by angle_seams(threshold = 66deg, resolution = 1024,
        texel_density = 5.12)`"""
        target = self.eval(node.target)
        if not (isinstance(target, V.Opaque) and isinstance(target.obj, Solid)):
            self.die("`uv ... by` needs a solid", node)
        method = node.method
        name = method.func if isinstance(method, S.Call) else "angle_seams"
        if name not in ("angle_seams", "unwrap"):
            self.die("unknown UV method %r" % name, node,
                     "Only angle_seams is implemented.")
        kw = {k: self.eval(v) for k, v in getattr(method, "kwargs", {}).items()}

        thr = kw.get("threshold")
        res = kw.get("resolution")
        dens = kw.get("texel_density")
        pad = kw.get("padding")
        resolution = 1024 if res is None else V.as_int(res, "resolution")

        try:
            solid = B.unwrap(
                target.obj,
                max_angle=66.0 if thr is None else V.as_degrees(thr, "threshold"),
                resolution=resolution,
                padding_px=None if pad is None else V.as_int(pad, "padding"))
            if dens is not None:
                # Declared density means uniform density, and an unwrap alone does
                # not give it: the packer stretches whatever it must to fill the
                # atlas. Rescaling here is what makes W302 pass by construction.
                solid = B.rescale_to_density(
                    solid, V.as_number(dens, "texel_density"),
                    resolution=resolution)
        except (UnitError, LatheError) as exc:
            raise LatheRuntimeError(str(exc), node, self.line_of(node)) from None

        self.result = solid
        # Rebind the name the program used, so later statements see the UVs.
        for key, val in list(self.env.items()):
            if isinstance(val, V.Opaque) and val.obj is target.obj:
                self.env[key] = V.Opaque(solid, "solid")
        self.uv_requests.append((solid.uv_set, name, {
            "resolution": resolution,
            "texel_density": None if dens is None else V.as_number(dens, "d"),
        }))

    def do_emit(self, node: S.Emit) -> None:
        if self.result is None:
            self.die("emit has no solid to write", node,
                     "Bind one with `let` before emitting.")
        self.emit_name = node.name
        self.result = self.result.rename(node.name)
        for key, expr in node.options.items():
            self.emit_options[key] = self.emit_value(key, expr)

    def emit_value(self, key: str, expr: S.Node) -> Any:
        if isinstance(expr, S.Name):
            return expr.name           # profile: game_prop_pbr
        if isinstance(expr, S.MapLit):
            return {k: V.as_number(self.eval(v), "%s.%s" % (key, k))
                    for k, v in expr.items.items()}
        val = self.eval(expr)
        if isinstance(val, V.Text):
            return val.s
        if isinstance(val, V.Bool):
            return val.b
        if isinstance(val, V.Length):
            return val.m
        if isinstance(val, V.Scalar):
            return val.v
        if isinstance(val, V.Opaque):
            return val.obj
        return str(val)

    # -- expressions ------------------------------------------------------

    def eval(self, node: S.Node) -> V.Value:
        if isinstance(node, S.Num):
            if node.unit is None:
                return V.Scalar(node.value)
            if node.unit in V.LENGTH_UNITS:
                return V.Length.of(node.value, node.unit)
            return V.Angle.of(node.value, node.unit)
        if isinstance(node, S.Str):
            return V.Text(node.value)
        if isinstance(node, S.BoolLit):
            return V.Bool(node.value)
        if isinstance(node, S.Name):
            return self.lookup(node)
        if isinstance(node, S.Unary):
            return self.guard(node, V.neg, self.eval(node.operand))
        if isinstance(node, S.Binary):
            return self.binary(node)
        if isinstance(node, S.ListLit):
            return V.Opaque([self.eval(x) for x in node.items], "list")
        if isinstance(node, S.TupleLit):
            return V.Opaque(tuple(self.eval(x) for x in node.items), "tuple")
        if isinstance(node, S.MapLit):
            return V.Opaque({k: self.eval(v) for k, v in node.items.items()}, "map")
        if isinstance(node, S.Call):
            return self.call(node)
        if isinstance(node, S.Method):
            return self.method(node)
        if isinstance(node, S.Placed):
            return self.placed(node)
        self.die("cannot evaluate %s" % type(node).__name__, node)

    def lookup(self, node: S.Name) -> V.Value:
        if node.name in self.env:
            return self.env[node.name]
        if node.name in AXES or node.name in ORIGINS:
            return V.Text(node.name)
        known = sorted(self.env)
        self.die("unknown name %r" % node.name, node,
                 ("Defined so far: %s." % ", ".join(known)) if known else
                 "Nothing is defined yet -- start with `param` or `let`.")

    def guard(self, node: S.Node, fn, *args) -> V.Value:
        try:
            return fn(*args)
        except UnitError as exc:
            raise LatheRuntimeError(str(exc), node, self.line_of(node)) from None

    def binary(self, node: S.Binary) -> V.Value:
        left = self.eval(node.left)
        right = self.eval(node.right)
        op = node.op

        # Solid algebra: a - b, a + b, a & b
        if isinstance(left, V.Opaque) and isinstance(left.obj, Solid):
            if op in ("-", "+"):
                other = right
                if isinstance(other, V.Opaque) and isinstance(other.obj, list):
                    # `body - slots` where slots is an array of cutters
                    out = left.obj
                    for item in other.obj:
                        out = self.apply_solid_op(node, out, item, op)
                    return V.Opaque(out, "solid")
                return V.Opaque(
                    self.apply_solid_op(node, left.obj, other, op), "solid")
            self.die("cannot use %r on solids" % op, node,
                     "Solids support - (difference) and + (union).")

        if op in ("<", "<=", ">", ">=", "==", "!="):
            return self.guard(node, V.compare, op, left, right)
        fn = {"+": V.add, "-": V.sub, "*": V.mul, "/": V.div}.get(op)
        if fn is None:
            self.die("unknown operator %r" % op, node)
        return self.guard(node, fn, left, right)

    def apply_solid_op(self, node, a: Solid, b, op: str) -> Solid:
        # b may be a wrapped value or a raw Solid: array() hands back a plain
        # Python list of Solids, and the loop above passes its items straight in.
        rhs = b.obj if isinstance(b, V.Opaque) else b
        if not isinstance(rhs, Solid):
            desc = b.describe() if isinstance(b, V.Value) else type(rhs).__name__
            self.die("the right side of %r must be a solid, got %s" % (op, desc),
                     node)
        try:
            return (a - rhs) if op == "-" else (a + rhs)
        except LatheError as exc:
            raise LatheRuntimeError(str(exc), node, self.line_of(node)) from None

    def placed(self, node: S.Placed) -> V.Value:
        target = self.eval(node.target)
        if not (isinstance(target, V.Opaque) and isinstance(target.obj, Solid)):
            self.die("`@` needs a solid on the left", node)
        spec = node.spec
        if not (isinstance(spec, S.Call) and spec.func == "origin"):
            self.die("`@` expects origin(...)", node,
                     "Write: box(...) @ origin(bottom_center)")
        if len(spec.args) != 1 or not isinstance(spec.args[0], S.Name):
            self.die("origin() takes one of %s" % ", ".join(ORIGINS), node)
        where = spec.args[0].name
        if where not in ORIGINS:
            self.die("unknown origin %r" % where, node,
                     "Use one of %s." % ", ".join(ORIGINS))
        solid = target.obj
        lo, hi = solid.bbox()
        if where == "bottom_center":
            solid = solid.at(x=-(lo[0] + hi[0]) / 2, y=-(lo[1] + hi[1]) / 2, z=-lo[2])
        elif where == "center":
            solid = solid.at(x=-(lo[0] + hi[0]) / 2, y=-(lo[1] + hi[1]) / 2,
                             z=-(lo[2] + hi[2]) / 2)
        return V.Opaque(solid, "solid")

    # -- calls ------------------------------------------------------------

    def call(self, node: S.Call) -> V.Value:
        fn = getattr(self, "fn_" + node.func, None)
        if fn is None:
            names = sorted(n[3:] for n in dir(self) if n.startswith("fn_"))
            self.die("unknown function %r" % node.func, node,
                     "Available: %s." % ", ".join(names))
        if node.predicates:
            self.die("%s() does not take predicates" % node.func, node,
                     "Predicates like `dihedral > 60deg` belong in .edges() "
                     "and .faces().")
        try:
            return fn(node)
        except (UnitError, CardinalityError) as exc:
            raise LatheRuntimeError(str(exc), node, self.line_of(node)) from None
        except LatheRuntimeError:
            raise
        except LatheError as exc:
            raise LatheRuntimeError(str(exc), node, self.line_of(node)) from None

    def arg(self, node, i: int, what: str) -> V.Value:
        if i >= len(node.args):
            self.die("%s() needs %s as argument %d" % (node.func, what, i + 1), node)
        return self.eval(node.args[i])

    def kw(self, node, name: str, default=None) -> Optional[V.Value]:
        if name in node.kwargs:
            return self.eval(node.kwargs[name])
        return default

    def solid_arg(self, node, i: int) -> Solid:
        v = self.arg(node, i, "a solid")
        if not (isinstance(v, V.Opaque) and isinstance(v.obj, Solid)):
            self.die("%s() argument %d must be a solid, got %s"
                     % (node.func, i + 1, v.describe()), node)
        return v.obj

    def tag_of(self, node, default: str) -> str:
        t = self.kw(node, "tag")
        if t is None:
            return default
        if isinstance(t, V.Text):
            return t.s
        self.die("tag must be a name or string", node)

    # -- primitives

    def fn_box(self, node) -> V.Value:
        w = V.as_metres(self.arg(node, 0, "a width"), "box width")
        d = V.as_metres(self.arg(node, 1, "a depth"), "box depth")
        h = V.as_metres(self.arg(node, 2, "a height"), "box height")
        return V.Opaque(B.box(w, d, h, tag=self.tag_of(node, "body"),
                              name=self.emit_name or "solid"), "solid")

    def fn_cylinder(self, node) -> V.Value:
        h = V.as_metres(self.arg(node, 0, "a height"), "cylinder height")
        r = V.as_metres(self.arg(node, 1, "a radius"), "cylinder radius")
        rt = self.kw(node, "radius_top")
        segs = self.kw(node, "segments")
        return V.Opaque(B.cylinder(
            h, r,
            None if rt is None else V.as_metres(rt, "radius_top"),
            segments=32 if segs is None else V.as_int(segs, "segments"),
            center=False, tag=self.tag_of(node, "body")), "solid")

    def fn_sphere(self, node) -> V.Value:
        r = V.as_metres(self.arg(node, 0, "a radius"), "sphere radius")
        segs = self.kw(node, "segments")
        return V.Opaque(B.sphere(
            r, segments=32 if segs is None else V.as_int(segs, "segments"),
            tag=self.tag_of(node, "body")), "solid")

    def _profile(self, node, i: int) -> List[Tuple[float, float]]:
        v = self.arg(node, i, "a profile")
        if not (isinstance(v, V.Opaque) and isinstance(v.obj, list)):
            self.die("%s() needs a profile list like [(0mm, 0mm), (110mm, 65mm)]"
                     % node.func, node)
        pts = []
        for item in v.obj:
            if not (isinstance(item, V.Opaque) and isinstance(item.obj, tuple)
                    and len(item.obj) == 2):
                self.die("every profile point must be a (radius, height) pair", node)
            r, h = item.obj
            pts.append((V.as_metres(r, "profile radius"),
                        V.as_metres(h, "profile height")))
        return pts

    def fn_revolve(self, node) -> V.Value:
        pts = self._profile(node, 0)
        segs = self.kw(node, "segments")
        deg = self.kw(node, "degrees")
        return V.Opaque(B.revolve(
            pts, segments=48 if segs is None else V.as_int(segs, "segments"),
            degrees=360.0 if deg is None else V.as_degrees(deg, "degrees"),
            tag=self.tag_of(node, "body")), "solid")

    def fn_extrude(self, node) -> V.Value:
        pts = self._profile(node, 0)
        h = V.as_metres(self.arg(node, 1, "a height"), "extrude height")
        return V.Opaque(B.extrude(pts, h, tag=self.tag_of(node, "body")), "solid")

    # -- assembly

    def fn_array(self, node) -> V.Value:
        base = self.solid_arg(node, 0)
        count = V.as_int(self.kw(node, "count", V.Scalar(1)), "count")
        span = self.kw(node, "span")
        span_m = 0.0 if span is None else V.as_metres(span, "span")
        axis = self.kw(node, "axis", V.Text("X"))
        ax = V.as_text(axis, "axis").lower() if isinstance(axis, V.Text) else "x"
        if ax not in ("x", "y", "z"):
            self.die("axis must be X, Y or Z", node)
        out = B.array(lambda i: base, count, axis=ax, span=span_m, center=True)
        return V.Opaque(out, "list")

    def fn_union(self, node) -> V.Value:
        items = [self.solid_arg(node, i) for i in range(len(node.args))]
        return V.Opaque(B.union_all(items), "solid")

    def fn_bevel(self, node) -> V.Value:
        solid = self.solid_arg(node, 0)
        sel = None
        if len(node.args) > 1:
            v = self.eval(node.args[1])
            if isinstance(v, V.Opaque) and isinstance(v.obj, Selection):
                sel = v.obj
            else:
                self.die("bevel()'s second argument must be an edge selection", node)
        width = self.kw(node, "width")
        if width is None:
            self.die("bevel() needs width =", node,
                     "e.g. bevel(carved, rim, width = 4mm, segments = 2)")
        segs = self.kw(node, "segments")
        min_d = 30.0
        if sel is not None:
            # Reuse the predicate the selection already carries, so the worker
            # picks the same edges the program asked for.
            min_d = _min_dihedral_of(sel)
        return V.Opaque(B.bevel(
            solid, V.as_metres(width, "bevel width"),
            segments=2 if segs is None else V.as_int(segs, "segments"),
            min_dihedral=min_d), "solid")

    # -- measures

    def fn_count(self, node) -> V.Value:
        v = self.arg(node, 0, "a selection")
        if isinstance(v, V.Opaque) and isinstance(v.obj, Selection):
            return V.Scalar(v.obj.count())
        if isinstance(v, V.Opaque) and isinstance(v.obj, list):
            return V.Scalar(len(v.obj))
        self.die("count() needs a selection or a list, got %s" % v.describe(), node)

    def fn_components(self, node) -> V.Value:
        return V.Scalar(self.solid_arg(node, 0).components())

    def fn_volume(self, node) -> V.Value:
        return V.Volume(self.solid_arg(node, 0).volume())

    def fn_tris(self, node) -> V.Value:
        return V.Scalar(self.solid_arg(node, 0).n_tris)

    def fn_origin(self, node) -> V.Value:
        # Only reachable as `@ origin(...)`, handled in placed().
        self.die("origin() is only valid after `@`", node,
                 "Write: box(...) @ origin(bottom_center)")

    # -- methods ----------------------------------------------------------

    def method(self, node: S.Method) -> V.Value:
        target = self.eval(node.target)
        if not isinstance(target, V.Opaque):
            self.die("%s has no method %r" % (target.describe(), node.name), node)
        obj = target.obj
        if isinstance(obj, Solid):
            if node.name in ("faces", "edges"):
                # Unit failures inside a predicate must surface located, like
                # every other error, rather than as a raw TypeError.
                try:
                    return self.selector(node, obj)
                except UnitError as exc:
                    raise LatheRuntimeError(
                        str(exc), node, self.line_of(node)) from None
            handler = {
                "translate": self._m_translate, "at": self._m_at,
                "rotate": self._m_rotate, "scale": self._m_scale,
                "mirror": self._m_mirror, "rename": self._m_rename,
            }.get(node.name)
            if handler is None:
                self.die("solids have no method %r" % node.name, node,
                         "Available: faces, edges, at, translate, rotate, scale, "
                         "mirror, rename.")
            return handler(node, obj)
        self.die("%s has no method %r" % (target.label, node.name), node)

    def selector(self, node: S.Method, solid: Solid) -> V.Value:
        domain = node.name
        kw: Dict[str, Any] = {}
        table = _EDGE_PREDS if domain == "edges" else _FACE_PREDS
        for attr, op, expr in node.predicates:
            if attr not in table:
                self.die("%s() has no predicate %r" % (domain, attr), node,
                         "Available: %s." % ", ".join(sorted(table)))
            if op not in table[attr]:
                self.die("%s %s ... is not supported" % (attr, op), node,
                         "Use %s." % " or ".join("%s %s" % (attr, o)
                                                 for o in sorted(table[attr])))
            val = self.eval(expr)
            kwarg = table[attr][op]
            if attr == "dihedral":
                kw[kwarg] = V.as_degrees(val, "%s %s" % (attr, op))
            elif attr in ("length", "z"):
                kw[kwarg] = V.as_metres(val, "%s %s" % (attr, op))
            else:
                kw[kwarg] = V.as_number(val, "%s %s" % (attr, op))
        for key, expr in node.kwargs.items():
            val = self.eval(expr)
            if key == "tag":
                kw["tag"] = V.as_text(val, "tag") if isinstance(val, V.Text) \
                    else str(val)
            else:
                self.die("%s() has no argument %r" % (domain, key), node,
                         "Use predicates: dihedral > 60deg, length > 20mm.")
        for a in node.args:
            if isinstance(a, S.Name) and a.name == "all":
                kw["all"] = True
            else:
                self.die("%s() takes predicates or `all`" % domain, node)
        try:
            sel = solid.faces(**kw) if domain == "faces" else solid.edges(**kw)
        except LatheError as exc:
            raise LatheRuntimeError(str(exc), node, self.line_of(node)) from None
        if domain == "edges":
            _remember_dihedral(sel, kw.get("dihedral_gt", 30.0))
        return V.Opaque(sel, "selection")

    def _xyz(self, node) -> Tuple[float, float, float]:
        out = [0.0, 0.0, 0.0]
        for i, key in enumerate("xyz"):
            if key in node.kwargs:
                out[i] = V.as_metres(self.eval(node.kwargs[key]), key)
        return tuple(out)

    def _m_at(self, node, solid):
        return V.Opaque(solid.at(*self._xyz(node)), "solid")

    def _m_translate(self, node, solid):
        return V.Opaque(solid.at(*self._xyz(node)), "solid")

    def _m_rotate(self, node, solid):
        degs = [0.0, 0.0, 0.0]
        for i, key in enumerate("xyz"):
            if key in node.kwargs:
                degs[i] = V.as_degrees(self.eval(node.kwargs[key]), key)
        return V.Opaque(solid.rotate(tuple(degs)), "solid")

    def _m_scale(self, node, solid):
        if node.args:
            f = V.as_number(self.eval(node.args[0]), "scale factor")
            return V.Opaque(solid.scale(f), "solid")
        vals = [1.0, 1.0, 1.0]
        for i, key in enumerate("xyz"):
            if key in node.kwargs:
                vals[i] = V.as_number(self.eval(node.kwargs[key]), key)
        return V.Opaque(solid.scale(tuple(vals)), "solid")

    def _m_mirror(self, node, solid):
        axis = "X"
        if node.args and isinstance(node.args[0], S.Name):
            axis = node.args[0].name.upper()
        if axis not in AXES:
            self.die("mirror() needs X, Y or Z", node)
        return V.Opaque(solid.mirror(AXES[axis]), "solid")

    def _m_rename(self, node, solid):
        if not node.args:
            self.die("rename() needs a name", node)
        return V.Opaque(solid.rename(V.as_text(self.eval(node.args[0]), "name")),
                        "solid")


# Selection is frozen, so the dihedral used to build it rides in a side table
# rather than on the object. bevel() needs it to pick the same edges.
_DIHEDRAL_MEMO: "dict[int, float]" = {}


def _remember_dihedral(sel: Selection, deg: float) -> None:
    _DIHEDRAL_MEMO[id(sel)] = deg


def _min_dihedral_of(sel: Selection) -> float:
    return _DIHEDRAL_MEMO.get(id(sel), 30.0)


def run(src: str):
    """Parse and evaluate. Returns (solid, interpreter)."""
    prog = S.parse(src)
    interp = Interp(src)
    solid = interp.run(prog)
    return solid, interp
