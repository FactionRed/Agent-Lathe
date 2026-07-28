"""
Quantities with units, for perception output.

DESIGN.md 4.1 makes units part of the type system: ``600mm`` and ``0.6m`` unify,
and a bare ``600`` is a type error. This module is the smallest thing that keeps
that promise at the perception boundary:

  * every length a probe *reports* is a ``Length`` and prints with a unit;
  * every length a probe *accepts* must arrive as a ``Length``/``Point``, or as
    a bare number together with an explicit ``unit=``. A bare number alone
    raises ``TypeError``.

Storage is always SI (metre, m^2, m^3, radian). The display unit is a
formatting choice, never a stored piece of state.
"""

from __future__ import annotations

import math
from typing import Sequence

# metres per unit
_LENGTH_UNITS = {
    "km": 1e3,
    "m": 1.0,
    "dm": 1e-1,
    "cm": 1e-2,
    "mm": 1e-3,
    "um": 1e-6,
    "in": 0.0254,
    "ft": 0.3048,
    "yd": 0.9144,
}

_SUPERSCRIPT = {1: "", 2: "²", 3: "³"}


def _split_unit(unit: str) -> tuple[str, int]:
    """'mm^2' | 'mm2' | 'mm²' -> ('mm', 2)."""
    if not isinstance(unit, str):
        raise TypeError("unit must be a string, got %r" % (unit,))
    u = unit.strip().replace("²", "^2").replace("³", "^3")
    if "^" in u:
        base, _, power = u.partition("^")
        try:
            return base.strip(), int(power)
        except ValueError:
            raise ValueError("cannot read exponent in unit %r" % unit) from None
    if u[-1:].isdigit() and len(u) > 1:
        return u[:-1], int(u[-1])
    return u, 1


def _unit_factor(unit: str, power: int) -> float:
    """SI value of one `unit`, checking that its dimension is `power`."""
    base, p = _split_unit(unit)
    if base not in _LENGTH_UNITS:
        raise ValueError(
            "unknown length unit %r; known: %s"
            % (base, ", ".join(sorted(_LENGTH_UNITS)))
        )
    if p != power:
        raise ValueError(
            "unit %r has dimension L^%d but a L^%d quantity was asked for"
            % (unit, p, power)
        )
    return _LENGTH_UNITS[base] ** power


def _unit_label(unit: str, power: int) -> str:
    base, _ = _split_unit(unit)
    return base + _SUPERSCRIPT[power]


class _Quantity:
    """Base for length-dimensioned scalars. Immutable, SI-stored."""

    POWER = 1
    DEFAULT_UNIT = "m"
    __slots__ = ("si",)

    def __init__(self, si_value):
        if isinstance(si_value, _Quantity):
            raise TypeError(
                "%s() takes a plain SI number; got %r. Use .to(unit) to convert."
                % (type(self).__name__, si_value)
            )
        self.si = float(si_value)

    # -- construction ------------------------------------------------------
    @classmethod
    def of(cls, value, unit: str):
        """Value expressed in `unit`. This is the only way in for bare numbers."""
        return cls(float(value) * _unit_factor(unit, cls.POWER))

    # -- conversion / formatting -------------------------------------------
    def to(self, unit: str) -> float:
        """Bare float in `unit`. Deliberately explicit: no implicit __float__."""
        return self.si / _unit_factor(unit, self.POWER)

    def format(self, unit: str | None = None, dp: int = 3, unit_suffix: bool = True) -> str:
        unit = unit or self.DEFAULT_UNIT
        if unit == "auto":
            unit = self._auto_unit()
        txt = "%.*f" % (dp, self.to(unit))
        if txt.startswith("-0.") and float(txt) == 0.0:
            txt = txt[1:]  # kill "-0.000"
        return txt + (" " + _unit_label(unit, self.POWER) if unit_suffix else "")

    def _auto_unit(self) -> str:
        mag = abs(self.si) ** (1.0 / self.POWER) if self.si else 1.0
        if mag >= 1e3:
            return "km" + ("^%d" % self.POWER if self.POWER > 1 else "")
        if mag >= 0.1:
            return "m" + ("^%d" % self.POWER if self.POWER > 1 else "")
        return "mm" + ("^%d" % self.POWER if self.POWER > 1 else "")

    def __str__(self) -> str:
        return self.format()

    def __repr__(self) -> str:
        return "%s(%s)" % (type(self).__name__, self.format())

    def __format__(self, spec: str) -> str:
        return self.format(spec) if spec else self.format()

    # -- arithmetic --------------------------------------------------------
    def _same(self, other, op):
        if type(other) is not type(self):
            raise TypeError(
                "cannot %s %s and %s -- dimensions differ"
                % (op, type(self).__name__, type(other).__name__)
            )
        return other

    def __add__(self, other):
        return type(self)(self.si + self._same(other, "add").si)

    def __sub__(self, other):
        return type(self)(self.si - self._same(other, "subtract").si)

    def __neg__(self):
        return type(self)(-self.si)

    def __abs__(self):
        return type(self)(abs(self.si))

    def __mul__(self, k):
        if isinstance(k, _Quantity):
            raise TypeError("multiplying two dimensioned values is not supported")
        return type(self)(self.si * float(k))

    __rmul__ = __mul__

    def __truediv__(self, other):
        if type(other) is type(self):
            return self.si / other.si  # dimensionless ratio
        if isinstance(other, _Quantity):
            raise TypeError("mixed-dimension division is not supported")
        return type(self)(self.si / float(other))

    def __eq__(self, other):
        return type(other) is type(self) and other.si == self.si

    def __hash__(self):
        return hash((type(self).__name__, self.si))

    def _cmp(self, other):
        return self._same(other, "compare").si

    def __lt__(self, other):
        return self.si < self._cmp(other)

    def __le__(self, other):
        return self.si <= self._cmp(other)

    def __gt__(self, other):
        return self.si > self._cmp(other)

    def __ge__(self, other):
        return self.si >= self._cmp(other)


class Length(_Quantity):
    POWER = 1
    DEFAULT_UNIT = "m"
    __slots__ = ()

    @classmethod
    def mm(cls, v):
        return cls.of(v, "mm")

    @classmethod
    def cm(cls, v):
        return cls.of(v, "cm")

    @classmethod
    def m(cls, v):
        return cls(v)

    @classmethod
    def inch(cls, v):
        return cls.of(v, "in")

    @classmethod
    def ft(cls, v):
        return cls.of(v, "ft")


class Area(_Quantity):
    POWER = 2
    DEFAULT_UNIT = "m^2"
    __slots__ = ()


class Volume(_Quantity):
    POWER = 3
    DEFAULT_UNIT = "m^3"
    __slots__ = ()


class Angle:
    """Stored in radians, printed in degrees."""

    __slots__ = ("rad",)

    def __init__(self, radians):
        self.rad = float(radians)

    @classmethod
    def deg(cls, value):
        return cls(math.radians(float(value)))

    @property
    def deg_value(self) -> float:
        return math.degrees(self.rad)

    def format(self, dp: int = 1) -> str:
        return "%.*f°" % (dp, self.deg_value)

    def __str__(self):
        return self.format()

    def __repr__(self):
        return "Angle(%s)" % self.format()

    def __eq__(self, other):
        return isinstance(other, Angle) and other.rad == self.rad

    def __hash__(self):
        return hash(("Angle", self.rad))


class Point:
    """Three length-dimensioned components: a position, a size or an offset."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)  # metres

    # -- construction ------------------------------------------------------
    @classmethod
    def of(cls, seq: Sequence[float], unit: str):
        v = tuple(float(c) for c in seq)
        if len(v) != 3:
            raise ValueError("a Point needs 3 components, got %d" % len(v))
        f = _unit_factor(unit, 1)
        return cls(v[0] * f, v[1] * f, v[2] * f)

    @classmethod
    def m(cls, x, y, z):
        return cls(x, y, z)

    @classmethod
    def mm(cls, x, y, z):
        return cls.of((x, y, z), "mm")

    @classmethod
    def cm(cls, x, y, z):
        return cls.of((x, y, z), "cm")

    @classmethod
    def from_lengths(cls, x: Length, y: Length, z: Length):
        return cls(x.si, y.si, z.si)

    # -- access ------------------------------------------------------------
    @property
    def si(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def lengths(self) -> tuple[Length, Length, Length]:
        return (Length(self.x), Length(self.y), Length(self.z))

    def to(self, unit: str) -> tuple[float, float, float]:
        f = _unit_factor(unit, 1)
        return (self.x / f, self.y / f, self.z / f)

    def norm(self) -> Length:
        return Length(math.sqrt(self.x**2 + self.y**2 + self.z**2))

    def __sub__(self, other):
        if not isinstance(other, Point):
            raise TypeError("Point - %s" % type(other).__name__)
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other):
        if not isinstance(other, Point):
            raise TypeError("Point + %s" % type(other).__name__)
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    # -- formatting --------------------------------------------------------
    def format(self, unit: str = "m", dp: int = 3, unit_suffix: bool = True) -> str:
        vals = []
        for c in self.to(unit):
            txt = "%.*f" % (dp, c)
            if float(txt) == 0.0:
                txt = "%.*f" % (dp, 0.0)  # normalise -0.000
            vals.append(txt)
        body = "(%s)" % ", ".join(vals)
        return body + (" " + _unit_label(unit, 1) if unit_suffix else "")

    def format_size(self, unit: str = "m", dp: int = 3) -> str:
        """'0.600 x 0.600 x 0.600 m' -- the DESIGN.md 6 bbox shape."""
        vals = ["%.*f" % (dp, c) for c in self.to(unit)]
        return " × ".join(vals) + " " + _unit_label(unit, 1)

    def __str__(self):
        return self.format()

    def __repr__(self):
        return "Point%s" % self.format()

    def __eq__(self, other):
        return isinstance(other, Point) and other.si == self.si

    def __hash__(self):
        return hash(("Point",) + self.si)


# --------------------------------------------------------------- coercions


_NO_UNIT_HINT = (
    "DESIGN.md 4.1: a bare number is a type error for a length. "
    "Pass %s, or pass the number together with unit=\"mm\"/\"m\"/\"in\"."
)


def as_length(value, unit: str | None = None, what: str = "length") -> Length:
    """Coerce to Length. Bare numbers require an explicit unit."""
    if isinstance(value, Length):
        if unit is not None:
            raise TypeError("%s is already a Length; drop unit=%r" % (what, unit))
        return value
    if isinstance(value, _Quantity):
        raise TypeError("%s must be a Length, got %s" % (what, type(value).__name__))
    if unit is None:
        raise TypeError(
            ("%s=%r has no unit. " % (what, value)) + _NO_UNIT_HINT % "Length.mm(600)"
        )
    return Length.of(value, unit)


def as_point(value, unit: str | None = None, what: str = "point") -> Point:
    """Coerce to Point. Bare tuples require an explicit unit."""
    if isinstance(value, Point):
        if unit is not None:
            raise TypeError("%s is already a Point; drop unit=%r" % (what, unit))
        return value
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(
        isinstance(c, Length) for c in value
    ):
        return Point.from_lengths(*value)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise TypeError("%s must be a Point or a 3-sequence, got %r" % (what, value))
    if unit is None:
        raise TypeError(
            ("%s=%r has no unit. " % (what, tuple(value)))
            + _NO_UNIT_HINT % "Point.m(0, -2, 0.3)"
        )
    return Point.of(value, unit)


def as_direction(value) -> tuple[float, float, float]:
    """Directions are dimensionless. Accepts '+Y', '-z' or a 3-sequence."""
    if isinstance(value, str):
        txt = value.strip().lower().replace(" ", "")
        sign = -1.0 if txt.startswith("-") else 1.0
        axis = txt.lstrip("+-")
        if axis not in ("x", "y", "z"):
            raise ValueError("direction %r is not an axis like '+Y' or '-Z'" % value)
        v = [0.0, 0.0, 0.0]
        v["xyz".index(axis)] = sign
        return tuple(v)
    if isinstance(value, Point):
        raise TypeError(
            "a direction is dimensionless -- pass a plain 3-tuple or '+Y', not a Point"
        )
    v = tuple(float(c) for c in value)
    if len(v) != 3:
        raise ValueError("direction needs 3 components, got %d" % len(v))
    n = math.sqrt(sum(c * c for c in v))
    if n == 0.0:
        raise ValueError("direction must not be the zero vector")
    return (v[0] / n, v[1] / n, v[2] / n)


def axis_index(axis) -> int:
    """'X'|'x'|0 -> 0."""
    if isinstance(axis, int):
        if axis not in (0, 1, 2):
            raise ValueError("axis index must be 0, 1 or 2")
        return axis
    a = str(axis).strip().lower()
    if a not in ("x", "y", "z"):
        raise ValueError("axis must be X, Y or Z (or 0/1/2), got %r" % (axis,))
    return "xyz".index(a)


AXIS_NAMES = ("X", "Y", "Z")
PLANE_NORMAL = {"YZ": 0, "ZY": 0, "XZ": 1, "ZX": 1, "XY": 2, "YX": 2}


def plane_normal_axis(plane) -> int:
    """'YZ' -> 0 (the plane whose normal is X)."""
    if isinstance(plane, str):
        key = plane.strip().upper().replace("-", "").replace("_", "")
        if key in PLANE_NORMAL:
            return PLANE_NORMAL[key]
        raise ValueError(
            "plane must be YZ, XZ or XY (named by the axes that span it), got %r" % plane
        )
    return axis_index(plane)


def plane_name(normal_axis: int) -> str:
    return {0: "YZ", 1: "XZ", 2: "XY"}[normal_axis]


__all__ = [
    "Length",
    "Area",
    "Volume",
    "Angle",
    "Point",
    "as_length",
    "as_point",
    "as_direction",
    "axis_index",
    "plane_normal_axis",
    "plane_name",
    "AXIS_NAMES",
]
