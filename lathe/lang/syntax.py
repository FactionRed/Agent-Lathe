"""
Lexer, AST and parser for the .lathe language -- DESIGN.md section 4.

The grammar is small on purpose. Section 2 wants programs that are pure,
declarative and diffable, not a general-purpose language: there are no loops, no
user functions and no mutation. If a program needs those, it belongs in the Python
escape hatch, and section 14.3 already says so.

    unit  system  = metric
    scene up = +Z, forward = -Y

    param size    = 600mm
    param slat_n  = 5

    let body   = box(size, size, size) @ origin(bottom_center)
    let cutter = box(slat_w * 0.35, size * 1.1, size - rail * 2)
    let slots  = array(cutter, axis = X, count = slat_n, span = size * 0.9)
    let carved = body - slots

    assert components(carved) == 1
    let rim = carved.edges(dihedral > 60deg, length > 20mm)
    assert count(rim) == 96

    let final = bevel(carved, rim, width = 4mm, segments = 2)
    mat oak on final.faces(all)

    emit "SM_Crate_Wood" {
      profile       : game_prop_pbr
      budget        : { tris_lod0: 1500 }
      texel_density : 5.12
    }

Articulation is three more statements, and section 5.4's reserved block:

    param tilt = 25deg

    joint base                     axis Z
    joint lower from base  at (0mm, 0mm, 80mm)  axis Y limit (-90deg, 90deg)
    joint upper from lower at (0mm, 0mm, 340mm) axis Y limit (-140deg, 140deg) angle tilt
    joint head  from upper at (0mm, 0mm, 300mm) limit (0deg, 0deg)

    bind lower to arm_lower
    bind upper to arm_upper

    solve {
      reach head to (-280mm, 0mm, 460mm) within 1mm
    }

`joint` clauses are `<ident> <expression>` pairs, which is why `from`, `at`,
`axis`, `limit` and `angle` are not reserved words -- they stay usable as
parameter names, and a new clause costs nothing in the lexer. A joint declares
a hinge rather than binding a name, so `lower` addresses the rig and not the
environment. A welded joint is `limit (0deg, 0deg)` rather than a keyword,
because the range already says it.

Angles are typed like everything else: `limit (-90, 90)` is a unit error, for
the same reason `box(600, 600, 600)` is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .values import ALL_UNITS

KEYWORDS = {"param", "let", "assert", "emit", "unit", "scene", "mat", "uv",
            "joint", "bind", "solve",
            "on", "by", "true", "false", "and", "or", "not"}

# Longest first, so >= never lexes as > then =.
OPERATORS = ("<=", ">=", "==", "!=", "->", "+", "-", "*", "/", "<", ">", "=",
             "(", ")", "{", "}", "[", "]", ",", ".", ":", "@")

_NUM = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_UNIT_SUFFIX = re.compile(r"[A-Za-z]+")


class LatheSyntaxError(SyntaxError):
    """A parse or lex failure, carrying line, column and the offending line."""

    def __init__(self, msg: str, line: int, col: int, src_line: str = "",
                 hint: str = ""):
        self.msg, self.line, self.col = msg, line, col
        self.src_line, self.hint = src_line, hint
        body = "line %d, column %d: %s" % (line, col, msg)
        if src_line:
            body += "\n    %s\n    %s^" % (src_line.rstrip(), " " * max(col - 1, 0))
        if hint:
            body += "\n  %s" % hint
        super().__init__(body)


@dataclass(frozen=True)
class Token:
    kind: str          # number, unit_number, ident, keyword, string, op, newline, eof
    text: str
    line: int
    col: int
    unit: Optional[str] = None
    value: Optional[float] = None


def split_lines(src: str) -> List[str]:
    """Source lines, split on \\n, \\r\\n and \\r only, with a BOM dropped.

    str.splitlines() also splits on \\x0b, \\x0c, \\x1c-\\x1e, \\x85 and U+2028/9,
    which shifted every later line number and let a `# comment<U+2028>code`
    leak its tail out of the comment.
    """
    text = src.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def tokenize(src: str) -> List[Token]:
    toks: List[Token] = []
    lines = split_lines(src)
    for lineno, raw in enumerate(lines, start=1):
        i, n = 0, len(raw)
        while i < n:
            ch = raw[i]
            if ch in " \t":
                i += 1
                continue
            if ch == "#":
                break  # comment to end of line
            col = i + 1
            m = _NUM.match(raw, i)
            if m:
                num = float(m.group(0))
                i = m.end()
                um = _UNIT_SUFFIX.match(raw, i)
                if um and um.group(0) in ALL_UNITS:
                    toks.append(Token("unit_number", m.group(0) + um.group(0),
                                      lineno, col, unit=um.group(0), value=num))
                    i = um.end()
                else:
                    if um and len(um.group(0)) <= 4:
                        raise LatheSyntaxError(
                            "%r is not a unit" % um.group(0), lineno, i + 1, raw,
                            "Length: %s. Angle: deg, rad, turn."
                            % ", ".join(sorted(u for u in ALL_UNITS
                                               if u not in ("deg", "rad", "turn"))))
                    toks.append(Token("number", m.group(0), lineno, col, value=num))
                continue
            m = _IDENT.match(raw, i)
            if m:
                word = m.group(0)
                kind = "keyword" if word in KEYWORDS else "ident"
                toks.append(Token(kind, word, lineno, col))
                i = m.end()
                continue
            if ch in "\"'":
                j = raw.find(ch, i + 1)
                if j < 0:
                    raise LatheSyntaxError("unterminated string", lineno, col, raw)
                toks.append(Token("string", raw[i + 1:j], lineno, col))
                i = j + 1
                continue
            for op in OPERATORS:
                if raw.startswith(op, i):
                    toks.append(Token("op", op, lineno, col))
                    i += len(op)
                    break
            else:
                raise LatheSyntaxError(
                    "unexpected character %r" % ch, lineno, col, raw)
        toks.append(Token("newline", "", lineno, len(raw) + 1))
    toks.append(Token("eof", "", len(lines) + 1, 1))
    return toks


# ---------------------------------------------------------------------- AST

@dataclass
class Node:
    line: int = 0
    col: int = 0


@dataclass
class Num(Node):
    value: float = 0.0
    unit: Optional[str] = None


@dataclass
class Str(Node):
    value: str = ""


@dataclass
class BoolLit(Node):
    value: bool = False


@dataclass
class Name(Node):
    name: str = ""


@dataclass
class Unary(Node):
    op: str = "-"
    operand: Optional[Node] = None


@dataclass
class Binary(Node):
    op: str = "+"
    left: Optional[Node] = None
    right: Optional[Node] = None


@dataclass
class Call(Node):
    func: str = ""
    args: List[Node] = field(default_factory=list)
    kwargs: Dict[str, Node] = field(default_factory=dict)
    # (attr, op, value) -- `dihedral > 60deg` inside a selector call
    predicates: List[Tuple[str, str, Node]] = field(default_factory=list)


@dataclass
class Method(Node):
    target: Optional[Node] = None
    name: str = ""
    args: List[Node] = field(default_factory=list)
    kwargs: Dict[str, Node] = field(default_factory=dict)
    predicates: List[Tuple[str, str, Node]] = field(default_factory=list)


@dataclass
class Placed(Node):
    """`expr @ origin(bottom_center)`"""
    target: Optional[Node] = None
    spec: Optional[Node] = None


@dataclass
class MapLit(Node):
    items: Dict[str, Node] = field(default_factory=dict)


@dataclass
class ListLit(Node):
    """`[(0mm, 0mm), (110mm, 65mm)]` -- revolve and extrude profiles."""
    items: List[Node] = field(default_factory=list)


@dataclass
class TupleLit(Node):
    items: List[Node] = field(default_factory=list)


# -- statements

@dataclass
class Param(Node):
    name: str = ""
    expr: Optional[Node] = None


@dataclass
class Let(Node):
    name: str = ""
    expr: Optional[Node] = None


@dataclass
class Assert(Node):
    expr: Optional[Node] = None
    text: str = ""


@dataclass
class Setting(Node):
    """`unit system = metric` / `scene up = +Z, forward = -Y`"""
    group: str = ""
    items: Dict[str, str] = field(default_factory=dict)


@dataclass
class Material(Node):
    """`mat oak on final.faces(all)`"""
    name: str = ""
    target: Optional[Node] = None


@dataclass
class UVSet(Node):
    """`uv final by angle_seams(threshold = 66deg)`"""
    target: Optional[Node] = None
    method: Optional[Node] = None


@dataclass
class Joint(Node):
    """`joint lower from base at (0mm, 0mm, 80mm) axis Y limit (-90deg, 90deg)`

    The clauses are an open mapping rather than fixed fields on purpose. Every
    one is optional, order does not matter, and a rig gains attributes over time
    -- so an unknown clause is a runtime error naming the ones that exist, which
    beats a parse error that cannot say what it wanted.
    """

    name: str = ""
    clauses: Dict[str, Node] = field(default_factory=dict)


@dataclass
class Bind(Node):
    """`bind lower to lower_arm` -- geometry, authored in rest space."""

    joint: str = ""
    target: Optional[Node] = None


@dataclass
class Reach(Node):
    """`reach head to (-280mm, 0mm, 460mm) within 2mm`"""

    tip: str = ""
    target: Optional[Node] = None
    within: Optional[Node] = None


@dataclass
class Solve(Node):
    """DESIGN.md section 5.4's reserved block, finally spent.

    > Add a real solver later, behind an explicit `solve { ... }` block, if it
    > earns its keep.

    Explicit, so the one place in the language where a value is *searched for*
    rather than evaluated is visible in the source. Everything outside this
    block is still a pure expression graph.
    """

    goals: List[Node] = field(default_factory=list)


@dataclass
class Emit(Node):
    name: str = ""
    options: Dict[str, Node] = field(default_factory=dict)


@dataclass
class Program(Node):
    statements: List[Node] = field(default_factory=list)


# ------------------------------------------------------------------ parser

_PRED_OPS = ("<", "<=", ">", ">=", "==", "!=")


class Parser:
    def __init__(self, src: str):
        self.src = src
        self.lines = split_lines(src)
        self.toks = tokenize(src)
        self.i = 0

    # -- token helpers ----------------------------------------------------

    def peek(self, ahead: int = 0) -> Token:
        return self.toks[min(self.i + ahead, len(self.toks) - 1)]

    def next(self) -> Token:
        t = self.toks[self.i]
        self.i += 1
        return t

    def at_op(self, *ops: str) -> bool:
        t = self.peek()
        return t.kind == "op" and t.text in ops

    def at_kw(self, *words: str) -> bool:
        t = self.peek()
        return t.kind == "keyword" and t.text in words

    def eat_op(self, op: str) -> Token:
        if not self.at_op(op):
            self.fail("expected %r" % op)
        return self.next()

    def eat_newlines(self) -> None:
        while self.peek().kind == "newline":
            self.next()

    def fail(self, msg: str, hint: str = "", tok: Optional[Token] = None):
        t = tok or self.peek()
        got = ("end of file" if t.kind == "eof" else
               "end of line" if t.kind == "newline" else repr(t.text))
        src = self.lines[t.line - 1] if 0 < t.line <= len(self.lines) else ""
        raise LatheSyntaxError("%s, got %s" % (msg, got), t.line, t.col, src, hint)

    # -- program ----------------------------------------------------------

    def parse(self) -> Program:
        prog = Program(1, 1)
        self.eat_newlines()
        while self.peek().kind != "eof":
            prog.statements.append(self.statement())
            self.eat_newlines()
        return prog

    def statement(self) -> Node:
        t = self.peek()
        if t.kind == "keyword":
            if t.text in ("param", "let"):
                return self.binding(t.text)
            if t.text == "assert":
                return self.assertion()
            if t.text == "emit":
                return self.emit()
            if t.text in ("unit", "scene"):
                return self.setting()
            if t.text == "mat":
                return self.material()
            if t.text == "uv":
                return self.uvset()
            if t.text == "joint":
                return self.joint()
            if t.text == "bind":
                return self.bind()
            if t.text == "solve":
                return self.solve()
        self.fail("expected a statement",
                  "Statements start with param, let, assert, mat, uv, joint, "
                  "bind, solve, emit, unit or scene.")

    def binding(self, which: str) -> Node:
        kw = self.next()
        name = self.peek()
        if name.kind != "ident":
            self.fail("expected a name after %r" % which)
        self.next()
        self.eat_op("=")
        expr = self.expression()
        node = (Param if which == "param" else Let)(kw.line, kw.col)
        node.name, node.expr = name.text, expr
        return node

    def assertion(self) -> Assert:
        kw = self.next()
        start = self.i
        expr = self.expression()
        text = " ".join(t.text for t in self.toks[start:self.i])
        node = Assert(kw.line, kw.col)
        node.expr, node.text = expr, text
        return node

    def setting(self) -> Setting:
        kw = self.next()
        node = Setting(kw.line, kw.col)
        node.group = kw.text
        while True:
            key = self.peek()
            if key.kind not in ("ident", "keyword"):
                self.fail("expected a setting name")
            self.next()
            self.eat_op("=")
            val = self.next()
            sign = ""
            if val.kind == "op" and val.text in ("+", "-"):
                sign, val = val.text, self.next()
            node.items[key.text] = sign + val.text
            if self.at_op(","):
                self.next()
                continue
            break
        return node

    def material(self) -> Material:
        kw = self.next()
        name = self.peek()
        if name.kind != "ident":
            self.fail("expected a material name after `mat`")
        self.next()
        if not self.at_kw("on"):
            self.fail("expected `on`", "Write: mat oak on final.faces(all)")
        self.next()
        node = Material(kw.line, kw.col)
        node.name, node.target = name.text, self.expression()
        return node

    def uvset(self) -> UVSet:
        kw = self.next()
        target = self.expression()
        if not self.at_kw("by"):
            self.fail("expected `by`",
                      "Write: uv final by angle_seams(threshold = 66deg)")
        self.next()
        node = UVSet(kw.line, kw.col)
        node.target, node.method = target, self.expression()
        return node

    def joint(self) -> Joint:
        kw = self.next()
        name = self.peek()
        if name.kind != "ident":
            self.fail("expected a joint name after `joint`",
                      "Write: joint elbow from shoulder at (0mm, 0mm, -300mm)")
        self.next()
        node = Joint(kw.line, kw.col)
        node.name = name.text
        node.clauses = self.clauses()
        return node

    def clauses(self) -> Dict[str, Node]:
        """`<ident> <expression>` pairs to the end of the line.

        Reading the key as an ordinary identifier rather than a keyword is what
        keeps `from`, `at`, `axis` and `limit` out of the reserved set -- they
        stay usable as parameter names, and a new clause costs nothing in the
        lexer.
        """
        out: Dict[str, Node] = {}
        while self.peek().kind == "ident":
            key = self.next()
            if key.text in out:
                self.fail("clause %r given twice" % key.text, tok=key)
            out[key.text] = self.expression()
        return out

    def bind(self) -> Bind:
        kw = self.next()
        name = self.peek()
        if name.kind != "ident":
            self.fail("expected a joint name after `bind`",
                      "Write: bind elbow to forearm")
        self.next()
        if not (self.peek().kind == "ident" and self.peek().text == "to"):
            self.fail("expected `to`", "Write: bind elbow to forearm")
        self.next()
        node = Bind(kw.line, kw.col)
        node.joint, node.target = name.text, self.expression()
        return node

    def solve(self) -> Solve:
        kw = self.next()
        node = Solve(kw.line, kw.col)
        self.eat_op("{")
        self.eat_newlines()
        while not self.at_op("}"):
            node.goals.append(self.goal())
            self.eat_newlines()
        self.eat_op("}")
        return node

    def goal(self) -> Reach:
        t = self.peek()
        if not (t.kind == "ident" and t.text == "reach"):
            self.fail("expected a goal",
                      "A solve block holds goals: "
                      "reach head to (0mm, 0mm, 1m) within 2mm")
        self.next()
        tip = self.peek()
        if tip.kind != "ident":
            self.fail("expected the joint to place after `reach`")
        self.next()
        node = Reach(t.line, t.col)
        node.tip = tip.text
        cl = self.clauses()
        for key in cl:
            if key not in ("to", "within"):
                self.fail("`reach` has no clause %r" % key,
                          "Write: reach head to (0mm, 0mm, 1m) within 2mm", tok=t)
        if "to" not in cl:
            self.fail("`reach` needs `to`",
                      "Write: reach head to (0mm, 0mm, 1m)", tok=t)
        node.target, node.within = cl["to"], cl.get("within")
        return node

    def emit(self) -> Emit:
        kw = self.next()
        name = self.peek()
        if name.kind != "string":
            self.fail("expected a quoted asset name after `emit`",
                      'Write: emit "SM_Crate_Wood_LOD0" { ... }')
        self.next()
        node = Emit(kw.line, kw.col)
        node.name = name.text
        if self.at_op("{"):
            node.options = self.map_body()
        return node

    def map_body(self) -> Dict[str, Node]:
        self.eat_op("{")
        items: Dict[str, Node] = {}
        self.eat_newlines()
        while not self.at_op("}"):
            key = self.peek()
            if key.kind not in ("ident", "keyword", "string"):
                self.fail("expected a key")
            self.next()
            if key.text in items:
                # A dict kept the last, and the first was dropped without a word.
                self.fail("%r is given twice in this block; keep one" % key.text,
                          tok=key)
            if self.at_op(":"):
                self.next()
            elif self.at_op("="):
                self.next()
            else:
                self.fail("expected ':' after %r" % key.text)
            items[key.text] = self.expression()
            if self.at_op(","):
                self.next()
            self.eat_newlines()
        self.eat_op("}")
        return items

    # -- expressions ------------------------------------------------------

    def expression(self) -> Node:
        return self.comparison()

    def comparison(self) -> Node:
        left = self.additive()
        if self.at_op(*_PRED_OPS):
            op = self.next()
            right = self.additive()
            node = Binary(op.line, op.col)
            node.op, node.left, node.right = op.text, left, right
            return node
        return left

    def additive(self) -> Node:
        node = self.multiplicative()
        while self.at_op("+", "-"):
            op = self.next()
            rhs = self.multiplicative()
            b = Binary(op.line, op.col)
            b.op, b.left, b.right = op.text, node, rhs
            node = b
        return node

    def multiplicative(self) -> Node:
        node = self.unary()
        while self.at_op("*", "/"):
            op = self.next()
            rhs = self.unary()
            b = Binary(op.line, op.col)
            b.op, b.left, b.right = op.text, node, rhs
            node = b
        return node

    def unary(self) -> Node:
        if self.at_op("-"):
            op = self.next()
            u = Unary(op.line, op.col)
            u.op, u.operand = "-", self.unary()
            return u
        if self.at_op("+"):
            self.next()
            return self.unary()
        return self.postfix()

    def postfix(self) -> Node:
        node = self.primary()
        while True:
            if self.at_op("."):
                dot = self.next()
                name = self.peek()
                if name.kind not in ("ident", "keyword"):
                    self.fail("expected a method or property name after '.'")
                self.next()
                m = Method(dot.line, dot.col)
                m.target, m.name = node, name.text
                if self.at_op("("):
                    m.args, m.kwargs, m.predicates = self.arguments()
                node = m
                continue
            if self.at_op("@"):
                at = self.next()
                p = Placed(at.line, at.col)
                p.target, p.spec = node, self.postfix()
                node = p
                continue
            break
        return node

    def primary(self) -> Node:
        t = self.peek()
        if t.kind in ("number", "unit_number"):
            self.next()
            n = Num(t.line, t.col)
            n.value, n.unit = t.value, t.unit
            return n
        if t.kind == "string":
            self.next()
            s = Str(t.line, t.col)
            s.value = t.text
            return s
        if t.kind == "keyword" and t.text in ("true", "false"):
            self.next()
            b = BoolLit(t.line, t.col)
            b.value = (t.text == "true")
            return b
        if t.kind == "op" and t.text == "{":
            m = MapLit(t.line, t.col)
            m.items = self.map_body()
            return m
        if t.kind == "op" and t.text == "[":
            self.next()
            lst = ListLit(t.line, t.col)
            self.eat_newlines()
            while not self.at_op("]"):
                lst.items.append(self.expression())
                if self.at_op(","):
                    self.next()
                self.eat_newlines()
            self.eat_op("]")
            return lst
        if t.kind == "op" and t.text == "(":
            self.next()
            self.eat_newlines()
            inner = self.expression()
            # A comma makes it a tuple, which is how profile points are written.
            if self.at_op(","):
                tup = TupleLit(t.line, t.col)
                tup.items.append(inner)
                while self.at_op(","):
                    self.next()
                    self.eat_newlines()
                    if self.at_op(")"):
                        break
                    tup.items.append(self.expression())
                self.eat_op(")")
                return tup
            self.eat_op(")")
            return inner
        if t.kind == "ident":
            self.next()
            if self.at_op("("):
                c = Call(t.line, t.col)
                c.func = t.text
                c.args, c.kwargs, c.predicates = self.arguments()
                return c
            nm = Name(t.line, t.col)
            nm.name = t.text
            return nm
        self.fail("expected a value")

    def arguments(self):
        """`( a, b, k = v, dihedral > 60deg )` in any mix."""
        self.eat_op("(")
        args: List[Node] = []
        kwargs: Dict[str, Node] = {}
        preds: List[Tuple[str, str, Node]] = []
        self.eat_newlines()
        while not self.at_op(")"):
            # name = value
            if (self.peek().kind == "ident" and self.peek(1).kind == "op"
                    and self.peek(1).text == "="):
                key = self.next().text
                self.next()
                kwargs[key] = self.expression()
            # attr <op> value  -- a selector predicate
            elif (self.peek().kind == "ident" and self.peek(1).kind == "op"
                  and self.peek(1).text in _PRED_OPS):
                attr = self.next().text
                op = self.next().text
                preds.append((attr, op, self.expression()))
            else:
                args.append(self.expression())
            if self.at_op(","):
                self.next()
            self.eat_newlines()
        self.eat_op(")")
        return args, kwargs, preds


def parse(src: str) -> Program:
    return Parser(src).parse()
