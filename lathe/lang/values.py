"""
Dimensioned values -- DESIGN.md section 4.1.

    "Units are part of the type: `600mm` and `0.6m` unify; `600` alone is a type
     error. Scale bugs die here."

That sentence is the whole reason this module exists. The build layer works in
float metres, so nothing there can tell 600 millimetres from 600 metres. Here a
bare number simply is not a length, and asking for one where a length belongs
raises rather than guessing.

Arithmetic follows dimension, not convenience:

    Length + Length -> Length          Length + Scalar -> TypeError
    Length * Scalar -> Length          Length * Length -> Area
    Length / Length -> Scalar          Length / Scalar -> Length
    Angle  + Angle  -> Angle           Angle  + Length -> TypeError

Everything is stored in SI -- metres, radians -- and the unit a value was written
in is kept only so an error message can quote the user's own numbers back.
"""

from __future__ import annotations

import math
from typing import Union

# unit -> metres
LENGTH_UNITS = {
    "mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
    "in": 0.0254, "ft": 0.3048, "thou": 0.0000254,
}
ANGLE_UNITS = {"deg": math.pi / 180.0, "rad": 1.0, "turn": 2 * math.pi}
ALL_UNITS = dict(LENGTH_UNITS)
ALL_UNITS.update(ANGLE_UNITS)


class UnitError(TypeError):
    """A dimension mismatch. Names both operands and what would have worked."""


class Value:
    """Base for every runtime value. `kind` is what error messages call it."""

    kind = "value"

    def describe(self) -> str:
        return "%s %s" % (self.kind, self)


class Scalar(Value):
    """A dimensionless number: a count, a ratio, a segment count."""

    kind = "number"
    __slots__ = ("v",)

    def __init__(self, v: float):
        self.v = float(v)

    def __str__(self) -> str:
        return _fmt(self.v)

    def __repr__(self) -> str:
        return "Scalar(%r)" % self.v

    def __eq__(self, other):
        return isinstance(other, Scalar) and abs(self.v - other.v) < 1e-12

    def __hash__(self):
        return hash(("scalar", round(self.v, 12)))


class Length(Value):
    """A distance. Stored in metres."""

    kind = "length"
    __slots__ = ("m", "unit")

    def __init__(self, metres: float, unit: str = "m"):
        self.m = float(metres)
        self.unit = unit if unit in LENGTH_UNITS else "m"

    @classmethod
    def of(cls, n: float, unit: str) -> "Length":
        if unit not in LENGTH_UNITS:
            raise UnitError(
                "%r is not a length unit. Use one of %s."
                % (unit, ", ".join(sorted(LENGTH_UNITS))))
        return cls(n * LENGTH_UNITS[unit], unit)

    def to(self, unit: str) -> float:
        if unit not in LENGTH_UNITS:
            raise UnitError("%r is not a length unit" % unit)
        return self.m / LENGTH_UNITS[unit]

    def __str__(self) -> str:
        return "%s%s" % (_fmt(self.to(self.unit)), self.unit)

    def __repr__(self) -> str:
        return "Length(%r m)" % self.m

    def __eq__(self, other):
        return isinstance(other, Length) and abs(self.m - other.m) < 1e-12

    def __hash__(self):
        return hash(("length", round(self.m, 12)))


class Angle(Value):
    """A rotation. Stored in radians."""

    kind = "angle"
    __slots__ = ("rad", "unit")

    def __init__(self, radians: float, unit: str = "deg"):
        self.rad = float(radians)
        self.unit = unit if unit in ANGLE_UNITS else "deg"

    @classmethod
    def of(cls, n: float, unit: str) -> "Angle":
        if unit not in ANGLE_UNITS:
            raise UnitError(
                "%r is not an angle unit. Use one of %s."
                % (unit, ", ".join(sorted(ANGLE_UNITS))))
        return cls(n * ANGLE_UNITS[unit], unit)

    def to(self, unit: str) -> float:
        return self.rad / ANGLE_UNITS[unit]

    @property
    def deg(self) -> float:
        return self.to("deg")

    def __str__(self) -> str:
        return "%s%s" % (_fmt(self.to(self.unit)), self.unit)

    def __repr__(self) -> str:
        return "Angle(%r rad)" % self.rad

    def __eq__(self, other):
        return isinstance(other, Angle) and abs(self.rad - other.rad) < 1e-12

    def __hash__(self):
        return hash(("angle", round(self.rad, 12)))


class Area(Value):
    """Length squared. Falls out of `w * h`, and W302 needs it."""

    kind = "area"
    __slots__ = ("m2",)

    def __init__(self, m2: float):
        self.m2 = float(m2)

    def __str__(self) -> str:
        return "%sm2" % _fmt(self.m2)

    def __eq__(self, other):
        return isinstance(other, Area) and abs(self.m2 - other.m2) < 1e-15

    def __hash__(self):
        return hash(("area", round(self.m2, 15)))


class Volume(Value):
    kind = "volume"
    __slots__ = ("m3",)

    def __init__(self, m3: float):
        self.m3 = float(m3)

    def __str__(self) -> str:
        return "%sm3" % _fmt(self.m3)

    def __eq__(self, other):
        return isinstance(other, Volume) and abs(self.m3 - other.m3) < 1e-18

    def __hash__(self):
        return hash(("volume", round(self.m3, 18)))


class Text(Value):
    kind = "text"
    __slots__ = ("s",)

    def __init__(self, s: str):
        self.s = str(s)

    def __str__(self) -> str:
        return '"%s"' % self.s

    def __eq__(self, other):
        return isinstance(other, Text) and self.s == other.s

    def __hash__(self):
        return hash(("text", self.s))


class Bool(Value):
    kind = "boolean"
    __slots__ = ("b",)

    def __init__(self, b: bool):
        self.b = bool(b)

    def __str__(self) -> str:
        return "true" if self.b else "false"

    def __eq__(self, other):
        return isinstance(other, Bool) and self.b == other.b

    def __hash__(self):
        return hash(("bool", self.b))


class Opaque(Value):
    """A host object -- a Solid, a Selection. Carried, never arithmetic."""

    kind = "object"
    __slots__ = ("obj", "label")

    def __init__(self, obj, label: str = "object"):
        self.obj = obj
        self.label = label

    def __str__(self) -> str:
        return self.label


def _fmt(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return "%d" % round(x)
    return ("%.6f" % x).rstrip("0").rstrip(".")


# ------------------------------------------------------------------ maths

_NUMERIC = (Scalar, Length, Angle, Area, Volume)


def _bad(op: str, a: Value, b: Value, hint: str = "") -> UnitError:
    msg = ("cannot %s %s and %s" % (op, a.describe(), b.describe()))
    if hint:
        msg += "\n  %s" % hint
    return UnitError(msg)


_MIX_HINT = ("Units are part of the type (DESIGN.md 4.1). Write the unit on the "
             "number -- 600mm, 0.6m, 60deg -- rather than leaving it bare.")


def add(a: Value, b: Value) -> Value:
    if type(a) is not type(b) or not isinstance(a, _NUMERIC):
        raise _bad("add", a, b, _MIX_HINT if isinstance(a, _NUMERIC) and
                   isinstance(b, _NUMERIC) else "")
    if isinstance(a, Scalar):
        return Scalar(a.v + b.v)
    if isinstance(a, Length):
        return Length(a.m + b.m, a.unit)
    if isinstance(a, Angle):
        return Angle(a.rad + b.rad, a.unit)
    if isinstance(a, Area):
        return Area(a.m2 + b.m2)
    return Volume(a.m3 + b.m3)


def sub(a: Value, b: Value) -> Value:
    return add(a, neg(b))


def neg(a: Value) -> Value:
    if isinstance(a, Scalar):
        return Scalar(-a.v)
    if isinstance(a, Length):
        return Length(-a.m, a.unit)
    if isinstance(a, Angle):
        return Angle(-a.rad, a.unit)
    if isinstance(a, Area):
        return Area(-a.m2)
    if isinstance(a, Volume):
        return Volume(-a.m3)
    raise UnitError("cannot negate %s" % a.describe())


def mul(a: Value, b: Value) -> Value:
    if isinstance(a, Scalar) and isinstance(b, Scalar):
        return Scalar(a.v * b.v)
    # scalar scaling, either order
    for x, y in ((a, b), (b, a)):
        if isinstance(y, Scalar):
            if isinstance(x, Length):
                return Length(x.m * y.v, x.unit)
            if isinstance(x, Angle):
                return Angle(x.rad * y.v, x.unit)
            if isinstance(x, Area):
                return Area(x.m2 * y.v)
            if isinstance(x, Volume):
                return Volume(x.m3 * y.v)
    if isinstance(a, Length) and isinstance(b, Length):
        return Area(a.m * b.m)
    if (isinstance(a, Area) and isinstance(b, Length)) or \
       (isinstance(a, Length) and isinstance(b, Area)):
        m2 = a.m2 if isinstance(a, Area) else b.m2
        m = b.m if isinstance(a, Area) else a.m
        return Volume(m2 * m)
    raise _bad("multiply", a, b)


def div(a: Value, b: Value) -> Value:
    zero = ((isinstance(b, Scalar) and b.v == 0) or
            (isinstance(b, Length) and b.m == 0) or
            (isinstance(b, Angle) and b.rad == 0))
    if zero:
        raise UnitError("division by zero: %s / %s" % (a, b))
    if isinstance(b, Scalar):
        if isinstance(a, Scalar):
            return Scalar(a.v / b.v)
        if isinstance(a, Length):
            return Length(a.m / b.v, a.unit)
        if isinstance(a, Angle):
            return Angle(a.rad / b.v, a.unit)
        if isinstance(a, Area):
            return Area(a.m2 / b.v)
        if isinstance(a, Volume):
            return Volume(a.m3 / b.v)
    if isinstance(a, Length) and isinstance(b, Length):
        return Scalar(a.m / b.m)
    if isinstance(a, Angle) and isinstance(b, Angle):
        return Scalar(a.rad / b.rad)
    if isinstance(a, Area) and isinstance(b, Length):
        return Length(a.m2 / b.m)
    if isinstance(a, Volume) and isinstance(b, Area):
        return Length(a.m3 / b.m2)
    raise _bad("divide", a, b)


def compare(op: str, a: Value, b: Value) -> Bool:
    """Ordering and equality. Mixed dimensions are an error, not False.

    Returning False for `60deg > 20mm` would let a nonsense predicate silently
    select nothing, which is the class of bug DESIGN.md 5 exists to prevent.
    """
    if op in ("==", "!="):
        if isinstance(a, (Text, Bool)) or isinstance(b, (Text, Bool)):
            same = type(a) is type(b) and a == b
            return Bool(same if op == "==" else not same)
    if type(a) is not type(b):
        raise _bad("compare", a, b, _MIX_HINT)
    x, y = _raw(a), _raw(b)
    return Bool({
        "<": x < y, "<=": x <= y, ">": x > y, ">=": x >= y,
        "==": abs(x - y) < 1e-12, "!=": abs(x - y) >= 1e-12,
    }[op])


def _raw(v: Value) -> float:
    if isinstance(v, Scalar):
        return v.v
    if isinstance(v, Length):
        return v.m
    if isinstance(v, Angle):
        return v.rad
    if isinstance(v, Area):
        return v.m2
    if isinstance(v, Volume):
        return v.m3
    raise UnitError("%s has no numeric value" % v.describe())


# ------------------------------------------------------- host conversions

def as_metres(v: Value, what: str) -> float:
    """A Length, in metres, for the build layer. This is the gate."""
    if isinstance(v, Length):
        return v.m
    if isinstance(v, Scalar):
        raise UnitError(
            "%s needs a length but got the bare number %s.\n"
            "  A number without a unit is not a length (DESIGN.md 4.1) -- it is "
            "the metres-versus-millimetres bug waiting to happen.\n"
            "  Write %smm, or %sm, or %scm." % (what, v, v, v, v))
    raise UnitError("%s needs a length, got %s" % (what, v.describe()))


def as_degrees(v: Value, what: str) -> float:
    if isinstance(v, Angle):
        return v.deg
    if isinstance(v, Scalar):
        raise UnitError(
            "%s needs an angle but got the bare number %s.\n"
            "  Write %sdeg (or %srad)." % (what, v, v, v))
    raise UnitError("%s needs an angle, got %s" % (what, v.describe()))


def as_number(v: Value, what: str) -> float:
    """A dimensionless number. A Length here is just as wrong as the reverse."""
    if isinstance(v, Scalar):
        return v.v
    if isinstance(v, (Length, Angle)):
        raise UnitError(
            "%s needs a plain number but got %s. Drop the unit -- a count or a "
            "ratio has no dimension." % (what, v.describe()))
    raise UnitError("%s needs a number, got %s" % (what, v.describe()))


def as_int(v: Value, what: str) -> int:
    n = as_number(v, what)
    if abs(n - round(n)) > 1e-9:
        raise UnitError("%s needs a whole number, got %s" % (what, _fmt(n)))
    return int(round(n))


def as_text(v: Value, what: str) -> str:
    if isinstance(v, Text):
        return v.s
    raise UnitError('%s needs a quoted string, got %s' % (what, v.describe()))


def as_bool(v: Value, what: str) -> bool:
    if isinstance(v, Bool):
        return v.b
    raise UnitError("%s needs true or false, got %s" % (what, v.describe()))


def wrap(x) -> Value:
    """Host value -> DSL value, for results coming back out of the kernel."""
    if isinstance(x, Value):
        return x
    if isinstance(x, bool):
        return Bool(x)
    if isinstance(x, (int, float)):
        return Scalar(x)
    if isinstance(x, str):
        return Text(x)
    return Opaque(x, type(x).__name__)
