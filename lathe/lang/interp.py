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

SOLVE -- the one place the language stops being a pure expression graph. Section
5.4 asked for exactly that boundary:

    Add a real solver later, behind an explicit `solve { ... }` block, if it
    earns its keep.

So it is explicit and it is a block. Everything outside it is still evaluated;
only inside is a value *searched for*. Determinism survives, because the solver
it calls has no random restarts and a fixed pass cap, so the same program still
produces the same geometry byte for byte. A goal that does not converge raises
rather than shrugging -- the program stated where the tip goes, and a pose that
misses is a failed assertion in exactly the way `assert count(rim) == 96` is.

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

import math
from typing import Any, Dict, List, Optional, Tuple

from .. import build as B
from ..build.solid import CardinalityError, LatheError, Selection, Solid
from . import syntax as S
from .syntax import split_lines
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

# A boolean over this many instances is minutes of work with no way to stop it.
MAX_ARRAY_COUNT = 1024

# function -> (most positional arguments, accepted keyword names)
_CALL_SHAPES = {
    "box": (3, {"tag"}),
    "cylinder": (2, {"radius_top", "segments", "tag"}),
    "sphere": (1, {"segments", "tag"}),
    "revolve": (1, {"segments", "degrees", "tag"}),
    "extrude": (2, {"tag"}),
    "array": (1, {"count", "span", "axis"}),
    "union": (None, set()),
    "bevel": (2, {"width", "segments", "clamp", "profile"}),
    "count": (1, set()),
    "components": (1, set()),
    "volume": (1, set()),
    "tris": (1, set()),
}
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
        self.lines = split_lines(src)
        self.env: Dict[str, V.Value] = {}
        self.params: Dict[str, V.Value] = {}
        self.settings: Dict[str, Dict[str, str]] = {}
        self.materials: Dict[str, Any] = {}   # tag -> Selection
        self.uv_requests: List[Tuple[Any, str, Dict[str, Any]]] = []
        # Articulation. `joints` is the declared skeleton in source order,
        # `bindings` the geometry hung off it, and `pose` the angles -- which
        # start at whatever each joint declared and are then overwritten by any
        # `solve` block. Keeping the pose as a plain mapping is what lets a
        # viewer hand one back as `param elbow = 35deg` (DESIGN.md section 10).
        self.joints: List[Any] = []
        self.bindings: Dict[str, Solid] = {}
        self.pose: Dict[str, float] = {}
        self.solved: List[Any] = []           # IKResult per goal, in order
        self.emit_name: Optional[str] = None
        self.emit_options: Dict[str, Any] = {}
        self.result: Optional[Solid] = None
        # Once `emit` or `let result` has named what ships, a later `let` is a
        # helper and does not replace it. Before this the last solid bound won,
        # so a scratch solid after the emit shipped under the asset's name.
        self.pinned = False
        self.pinned_by_emit = False
        self.assertions = 0
        # Things the program said that nothing reads. (head, advice) pairs.
        self.notes: List[Tuple[str, str]] = []

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
        # A rig assembles last. Every `bind` names geometry that a `let` already
        # produced, and `solve` may still be adjusting the pose on the final
        # line, so the posed solid cannot be built until the program is done.
        if self.bindings:
            self.result = self.rig().posed(self.pose)
            if self.emit_name:
                self.result = self.result.rename(self.emit_name)
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
                if node.name == "result":
                    self.result = val.obj
                    self.pinned = True
                elif not self.pinned:
                    # With no `emit` and no `result`, the last solid bound is the
                    # answer, so a short program still has an obvious one.
                    self.result = val.obj
        elif isinstance(node, S.Assert):
            self.do_assert(node)
        elif isinstance(node, S.Setting):
            self.settings.setdefault(node.group, {}).update(node.items)
            self.notes.append((
                "`%s %s` is accepted and has no effect" % (
                    node.group, " ".join(node.items) or ""),
                "Lengths are metres inside Lathe and the number carries its "
                "unit, so there is nothing here to set. The line is kept so a "
                "program written to DESIGN.md section 4 still reads."))
        elif isinstance(node, S.Material):
            self.do_material(node)
            self.notes.append((
                "`mat %s on ...` is recorded and nothing reads it yet" % node.name,
                "Pass materials to lathe_observe or lathe_view to colour the "
                "preview; they do not reach the verifier or an export."))
        elif isinstance(node, S.UVSet):
            self.do_uv(node)
        elif isinstance(node, S.Joint):
            self.do_joint(node)
        elif isinstance(node, S.Bind):
            self.do_bind(node)
        elif isinstance(node, S.Solve):
            self.do_solve(node)
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

        if not self.pinned or target.obj is self.result:
            self.result = solid
        # Rebind the name the program used, so later statements see the UVs.
        for key, val in list(self.env.items()):
            if isinstance(val, V.Opaque) and val.obj is target.obj:
                self.env[key] = V.Opaque(solid, "solid")
        self.uv_requests.append((solid.uv_set, name, {
            "resolution": resolution,
            "texel_density": None if dens is None else V.as_number(dens, "d"),
        }))

    # -- articulation (section 5.4) ---------------------------------------

    _JOINT_CLAUSES = ("from", "at", "axis", "limit", "angle")

    def do_joint(self, node: S.Joint) -> None:
        """`joint lower from base at (0mm, 0mm, 80mm) axis Y limit (...)`.

        A joint is a declaration, not a value: it does not bind a name in the
        environment, because `lower` names a hinge in the rig rather than
        something an expression can be written about.
        """
        from ..pose import Joint

        for key in node.clauses:
            if key not in self._JOINT_CLAUSES:
                self.die("joint has no clause %r" % key, node,
                         "Available: %s." % ", ".join(self._JOINT_CLAUSES))
        if any(j.name == node.name for j in self.joints):
            self.die("joint %r is declared twice" % node.name, node,
                     "Joints are addressed by name, so names must be unique.")

        parent = None
        if "from" in node.clauses:
            expr = node.clauses["from"]
            if not isinstance(expr, S.Name):
                self.die("`from` needs a joint name", node,
                         "Write: joint elbow from shoulder at (...)")
            parent = expr.name
            if not any(j.name == parent for j in self.joints):
                known = ", ".join(j.name for j in self.joints)
                self.die("unknown parent joint %r" % parent, node,
                         ("Declared so far: %s." % known) if known else
                         "No joint is declared yet — this one is the root.")

        offset = (0.0, 0.0, 0.0)
        if "at" in node.clauses:
            offset = self.vec3(node.clauses["at"], "at", node)
        axis = (0.0, 1.0, 0.0)
        if "axis" in node.clauses:
            axis = self.axis_of(node.clauses["axis"], node)
        limit = (-2.0 * math.pi, 2.0 * math.pi)
        if "limit" in node.clauses:
            limit = self.limit_of(node.clauses["limit"], node)
        angle = 0.0
        if "angle" in node.clauses:
            angle = math.radians(
                V.as_degrees(self.eval(node.clauses["angle"]), "joint angle"))

        try:
            joint = Joint(name=node.name, parent=parent, offset=offset,
                          axis=axis, limit=limit)
            joint.unit_axis()                       # rejects a zero-length axis
        except LatheError as exc:
            raise LatheRuntimeError(str(exc), node, self.line_of(node)) from None
        self.joints.append(joint)
        self.pose[node.name] = joint.clamp(angle)

    def do_bind(self, node: S.Bind) -> None:
        if not any(j.name == node.joint for j in self.joints):
            known = ", ".join(j.name for j in self.joints)
            self.die("unknown joint %r" % node.joint, node,
                     ("Declared joints: %s." % known) if known else
                     "Declare the joint before binding geometry to it.")
        val = self.eval(node.target)
        if not (isinstance(val, V.Opaque) and isinstance(val.obj, Solid)):
            self.die("`bind ... to` needs a solid, got %s" % val.describe(), node)
        if node.joint in self.bindings:
            self.bindings[node.joint] = self.bindings[node.joint] + val.obj
        else:
            self.bindings[node.joint] = val.obj

    def do_solve(self, node: S.Solve) -> None:
        """The one place a value is searched for rather than evaluated.

        Section 5.4 asked for it to be explicit and it is: outside this block
        the language is still a pure expression graph, and inside it every
        result is deterministic, so the program stays reproducible.
        """
        from ..pose import DEFAULT_TOLERANCE, solve

        if not self.joints:
            self.die("solve needs a rig", node,
                     "Declare joints first: joint base ... / joint tip from ...")
        sk = self.skeleton(node)
        for goal in node.goals:
            if goal.tip not in sk:
                self.die("unknown joint %r" % goal.tip, goal,
                         "Declared joints: %s." % ", ".join(sk.names))
            target = self.vec3(goal.target, "reach target", goal)
            tol = DEFAULT_TOLERANCE
            if goal.within is not None:
                tol = V.as_metres(self.eval(goal.within), "within")
                if not tol > 0.0:
                    self.die("within must be a positive length, got %g mm"
                             % (tol * 1000.0), goal.within,
                             "e.g. `within 2mm`. A negative or zero tolerance is "
                             "unreachable by construction and was reported as "
                             "'unreachable ... 0.0000 mm'.")
            root = self.chain_root(sk, goal.tip)
            try:
                hit = solve(sk, root, goal.tip, target, seed=self.pose,
                            tolerance=tol)
            except LatheError as exc:
                raise LatheRuntimeError(str(exc), goal, self.line_of(goal)) from None
            self.pose.update(hit.pose)
            self.solved.append(hit)
            if not hit.converged:
                # A missed goal is a failed assertion, not a warning: the
                # program said where the tip goes, and it does not go there.
                self.die(
                    "reach %s did not converge — %s" % (goal.tip, hit.describe()),
                    goal,
                    "Widen a limit, move the target, or seed the pose with an "
                    "`angle` clause: a one-sided limit halves the workspace.")

    @staticmethod
    def chain_root(sk, tip: str) -> str:
        """The topmost ancestor of `tip`.

        Solving the whole chain to the root is the predictable default: the
        program named the tip and the target, and nothing in the source says
        which joints should be held still. A narrower chain is what `limit`
        exists for.
        """
        cur = tip
        while sk.joint(cur).parent is not None:
            cur = sk.joint(cur).parent
        return cur

    def skeleton(self, node: S.Node = None):
        from ..pose import Skeleton

        try:
            return Skeleton(tuple(self.joints), name=self.emit_name or "rig")
        except LatheError as exc:
            raise LatheRuntimeError(str(exc), node, self.line_of(node)) from None

    def rig(self, node: S.Node = None):
        from ..pose import Rig

        try:
            return Rig(self.skeleton(node), dict(self.bindings),
                       name=self.emit_name or "rig")
        except LatheError as exc:
            raise LatheRuntimeError(str(exc), node, self.line_of(node)) from None

    # -- clause helpers

    def vec3(self, expr: S.Node, what: str, node: S.Node) -> Tuple[float, float, float]:
        val = self.eval(expr)
        if not (isinstance(val, V.Opaque) and isinstance(val.obj, tuple)
                and len(val.obj) == 3):
            self.die("%s needs three lengths" % what, node,
                     "Write: %s (0mm, 0mm, 80mm)" % what.split()[0])
        return tuple(V.as_metres(v, what) for v in val.obj)

    def axis_of(self, expr: S.Node, node: S.Node) -> Tuple[float, float, float]:
        """`axis Y`, or an explicit `(0, 1, 0)` for a hinge off the cardinals."""
        if isinstance(expr, S.Name) and expr.name in AXES:
            return AXES[expr.name]
        if isinstance(expr, S.Name) and expr.name not in self.env:
            # Caught before `eval`, which would report a bare "unknown name" and
            # send the reader off to define a param. In an `axis` clause the
            # likely mistake is a mistyped axis, so say that instead.
            self.die("unknown axis %r" % expr.name, node,
                     "Use X, Y or Z, or a three-number vector like (0, 1, 0).")
        val = self.eval(expr)
        if isinstance(val, V.Text) and val.s in AXES:
            return AXES[val.s]
        if isinstance(val, V.Opaque) and isinstance(val.obj, tuple) and len(val.obj) == 3:
            return tuple(V.as_number(v, "axis") for v in val.obj)
        self.die("axis must be X, Y, Z or a three-number vector", node,
                 "Write: axis Y, or axis (0, 0.7071, 0.7071)")

    def limit_of(self, expr: S.Node, node: S.Node) -> Tuple[float, float]:
        val = self.eval(expr)
        if not (isinstance(val, V.Opaque) and isinstance(val.obj, tuple)
                and len(val.obj) == 2):
            self.die("limit needs a (min, max) pair of angles", node,
                     "Write: limit (-90deg, 90deg)")
        lo, hi = (math.radians(V.as_degrees(v, "limit")) for v in val.obj)
        if lo > hi:
            self.die("limit min %s is above max %s"
                     % (math.degrees(lo), math.degrees(hi)), node,
                     "Write the smaller bound first: limit (-90deg, 90deg)")
        return (lo, hi)

    def do_emit(self, node: S.Emit) -> None:
        if self.result is None:
            self.die("emit has no solid to write", node,
                     "Bind one with `let` before emitting.")
        if self.pinned_by_emit:
            self.die("a program has one `emit`; this is the second", node,
                     "Put the options in one block. The first emit named %r."
                     % self.emit_name)
        self.pinned_by_emit = True
        self.pinned = True
        self.emit_name = node.name
        self.result = self.result.rename(node.name)
        self.emit_options = {}
        for key, expr in node.options.items():
            self.emit_options[key] = self.typed_emit(key, expr)

    def typed_emit(self, key: str, expr: S.Node) -> Any:
        """An emit option, checked against what it is for.

        components : 1mm read as 0, 1.9 as 1, texel_density : 10mm as 0.01 and
        "two" raised a bare ValueError with no line, because every value was
        coerced and none was checked.
        """
        val = self.emit_value(key, expr)
        if key in ("components", "texture_resolution", "texel_density"):
            ev = self.eval(expr)
            if isinstance(ev, (V.Length, V.Angle)):
                self.die(
                    "%s is a %s, not a length or an angle: `%s` carries a unit"
                    % (key, "count" if key != "texel_density" else
                       "number of pixels per centimetre", self.src_of(expr)),
                    expr, "Write the plain number, e.g. `%s: %s`" % (
                        key, {"components": "1", "texture_resolution": "1024",
                              "texel_density": "5.12"}[key]))
            if not isinstance(ev, V.Scalar) or isinstance(val, bool):
                self.die("%s must be a number, got %s" % (key, ev.describe()), expr)
            num = float(ev.v)
            if key == "texel_density":
                if not num > 0:
                    self.die("texel_density must be positive, got %g" % num, expr)
                return num
            if num != int(num) or num < (0 if key == "components" else 1):
                self.die("%s must be a whole number%s, got %g" % (
                    key, "" if key == "components" else " of pixels", num), expr)
            return int(num)
        if key == "budget":
            if not isinstance(val, dict):
                self.die("budget must be a block, e.g. `budget: { tris_lod0: 1500 }`",
                         expr)
            for name, n in val.items():
                if isinstance(n, bool) or n != int(n) or n < 0:
                    self.die("budget %s must be a whole number of triangles, got %r"
                             % (name, n), expr)
            return {k: int(v) for k, v in val.items()}
        if key in ("profile", "target", "origin"):
            if not isinstance(val, str):
                self.die("%s is a name, e.g. `%s: %s`" % (
                    key, key, {"profile": "game_prop_pbr", "target": "unreal",
                               "origin": "bottom_center"}[key]), expr)
            if key == "origin" and val not in ORIGINS:
                self.die("origin must be one of %s, got %r"
                         % (", ".join(ORIGINS), val), expr)
            return val
        return val

    def src_of(self, expr) -> str:
        line = self.line_of(expr).strip()
        return line if len(line) < 60 else line[:57] + "..."

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
        shape = _CALL_SHAPES.get(node.func)
        if shape is not None:
            max_args, allowed = shape
            if max_args is not None and len(node.args) > max_args:
                self.die("%s() takes at most %d positional argument(s), got %d"
                         % (node.func, max_args, len(node.args)), node)
            stray = sorted(set(node.kwargs) - allowed)
            if stray:
                # Accepted and ignored is how `bevel(only_tags = ...)`,
                # `box(depth = ...)` and `bevel(segmants = 9)` all built, and
                # built the wrong thing.
                self.die("%s() has no argument %s" % (
                    node.func, ", ".join(repr(k) for k in stray)), node,
                    "Arguments: %s." % (", ".join(sorted(allowed)) or "none"))
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
        if not 1 <= count <= MAX_ARRAY_COUNT:
            self.die("array count must be between 1 and %d, got %d"
                     % (MAX_ARRAY_COUNT, count), node,
                     "A union of this many instances takes minutes and cannot "
                     "be interrupted; build it in pieces." if count > 1 else
                     "count = 0 or less made an empty list that every later "
                     "statement then failed on.")
        span = self.kw(node, "span")
        span_m = 0.0 if span is None else V.as_metres(span, "span")
        axis = self.kw(node, "axis", V.Text("X"))
        if not isinstance(axis, V.Text):
            self.die("axis must be X, Y or Z, got %s" % axis.describe(), node)
        ax = V.as_text(axis, "axis").lower()
        if ax not in ("x", "y", "z"):
            self.die("axis must be X, Y or Z, got %r" % axis.s, node)
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
        clamp = self.kw(node, "clamp")
        if clamp is not None and not isinstance(clamp, V.Bool):
            self.die("bevel() clamp must be true or false, got %s" % clamp.describe(),
                     node)
        profile = self.kw(node, "profile")
        prof = 0.5 if profile is None else V.as_number(profile, "profile")
        if not 0.0 < prof <= 1.0:
            self.die("bevel() profile must be in (0, 1], got %g" % prof, node,
                     "0.5 is a circular arc; 1 is a square corner pulled out.")
        if sel is not None and sel.domain != "edges":
            self.die("bevel()'s second argument must be an EDGE selection, got "
                     "faces", node, "Select edges: carved.edges(dihedral > 60deg)")
        # The selection is the edge set: it is handed over whole, not reduced to
        # its dihedral threshold (which beveled every edge over it, whatever else
        # the selection said).
        return V.Opaque(B.bevel(
            solid, V.as_metres(width, "bevel width"),
            segments=2 if segs is None else V.as_int(segs, "segments"),
            clamp=True if clamp is None else clamp.b, profile=prof,
            edges=sel), "solid")

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


def run(src: str):
    """Parse and evaluate. Returns (solid, interpreter)."""
    prog = S.parse(src)
    interp = Interp(src)
    solid = interp.run(prog)
    return solid, interp
