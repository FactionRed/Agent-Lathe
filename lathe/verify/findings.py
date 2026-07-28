"""
Lathe DESIGN.md section 7 -- the report format.

The report format IS the product. Section 7:

    Named code, named entity, numeric evidence, executable repair. Agents fix
    errors of this shape reliably. They never fix "boolean failed."

So a Finding is not a message. It is four required fields, and the type system
refuses to build one without them:

    code      stable, citable, greppable in a studio wiki   -> Finding.code
    entity    the offending NAMED thing, never an index     -> Finding.entity
    evidence  numbers, with units, that the agent can diff  -> Finding.evidence
    repair    a call the agent can paste and run            -> Finding.repair

Rendering matches the section 7 sample byte for byte:

    x E106  Overlapping UVs on unique bake
            object   : SM_Crate_Wood
            islands  : side_L overlaps side_R (91% area)
            evidence : bake target = normal, uv_set = UV0
            repair   : uv.unwrap(final, split_mirrored = true)
                       -- or emit with bake_uv_set = UV1

Two output forms, one source of truth. `render()` is for a human reading a
terminal and for the agent's text channel. `to_dict()` is the same data as JSON
for tooling that wants to route findings mechanically. Neither is a
reformatting of the other.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


class Severity(IntEnum):
    NOTE = 0
    WARN = 1
    ERROR = 2

    @property
    def label(self) -> str:
        return {0: "note", 1: "warn", 2: "error"}[int(self)]


ERROR, WARN, NOTE = Severity.ERROR, Severity.WARN, Severity.NOTE

_GLYPH_UNICODE = {ERROR: "✗", WARN: "⚠", NOTE: "•"}
_GLYPH_ASCII = {ERROR: "x", WARN: "!", NOTE: "-"}

KEY_WIDTH = 8
INDENT = " " * 8
CONT = " " * (8 + KEY_WIDTH + 3)
MAX_LIST_ITEMS = 6  # bounded output: an agent reading 400 edge ids fixes nothing
MAX_JSON_ITEMS = 64


# ------------------------------------------------------------------- repair


class Raw(str):
    """A repair argument that is an identifier or expression, not a string
    literal: `Raw("final")` renders as `final`, `"final"` renders as `"final"`."""


@dataclass(frozen=True)
class Repair:
    """An executable repair suggestion.

    `call` plus `args` plus `kwargs` is the machine form; `text` is the same
    thing rendered in the section 4 IR dialect. Alternatives exist because the
    honest answer is sometimes "fix the geometry, or change the declaration" --
    and the agent should be told both rather than guessing.
    """

    call: str
    args: Tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    alternatives: Tuple[str, ...] = ()
    note: Optional[str] = None
    statement: bool = False  # `assert components(x) == 1`, not `assert(...)`

    def text(self) -> str:
        parts = [_render_arg(a) for a in self.args]
        parts += ["%s = %s" % (k, _render_arg(v)) for k, v in self.kwargs.items()]
        if self.statement:
            return "%s %s" % (self.call, " ".join(parts))
        return "%s(%s)" % (self.call, ", ".join(parts))

    def lines(self, ascii_only: bool = False) -> Tuple[str, ...]:
        dash = "--" if ascii_only else "—"
        out = [self.text()]
        out += ["%s or %s" % (dash, a) for a in self.alternatives]
        if self.note:
            out.append("%s %s" % (dash, self.note))
        return tuple(out)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call": self.call,
            "args": [_jsonable(a) for a in self.args],
            "kwargs": {k: _jsonable(v) for k, v in self.kwargs.items()},
            "text": self.text(),
            "statement": self.statement,
            "alternatives": list(self.alternatives),
            "note": self.note,
        }


def _render_arg(v: Any) -> str:
    if isinstance(v, Raw):
        return str(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return _num(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, (list, tuple)):
        return "[%s]" % ", ".join(_render_arg(x) for x in v)
    if v is None:
        return "none"
    return '"%s"' % v


def _num(v: float) -> str:
    """Compact but honest: never scientific notation for readable magnitudes,
    never more digits than the measurement supports."""
    if isinstance(v, int):
        return str(v)
    if v != v or math.isinf(v):
        return str(v)
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1000:
        return "%.0f" % v
    if a >= 1:
        return ("%.3f" % v).rstrip("0").rstrip(".")
    if a >= 1e-4:
        return ("%.6f" % v).rstrip("0").rstrip(".")
    return "%.2e" % v


def _jsonable(v: Any) -> Any:
    if isinstance(v, Raw):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v][:MAX_JSON_ITEMS]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if hasattr(v, "item") and hasattr(v, "dtype"):  # numpy scalar
        return v.item()
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


# ------------------------------------------------------------------ finding


@dataclass(frozen=True)
class Finding:
    """One violation. Four required fields, by construction."""

    code: str
    severity: Severity
    summary: str  # headline; may carry numbers, as W302 does in section 7
    entity: str  # the offending NAMED entity
    evidence: str  # numeric evidence, rendered
    repair: Repair
    where: Tuple[Tuple[str, Union[str, Sequence[str]]], ...] = ()  # extra keyed lines
    data: Mapping[str, Any] = field(default_factory=dict)  # machine-readable evidence
    rule: str = ""  # static rule title, for grouping in dashboards

    def __post_init__(self) -> None:
        for name in ("code", "summary", "entity", "evidence"):
            if not getattr(self, name):
                raise ValueError(
                    "Finding.%s is required -- section 7 findings without it are "
                    "the 'boolean failed' message this design rejects" % name
                )
        for key, _value in self.where:
            # The key column is fixed width. A longer key would shift the colon
            # and break the one thing every consumer of this report relies on.
            if len(key) > KEY_WIDTH:
                raise ValueError(
                    "detail key %r is %d chars; the report column is %d"
                    % (key, len(key), KEY_WIDTH)
                )

    # -- text form ---------------------------------------------------------

    def render(self, ascii_only: bool = False) -> str:
        glyph = (_GLYPH_ASCII if ascii_only else _GLYPH_UNICODE)[self.severity]
        lines = ["%s %s  %s" % (glyph, self.code, self.summary)]
        rows: List[Tuple[str, Union[str, Sequence[str]]]] = [("object", self.entity)]
        rows.extend(self.where)
        rows.append(("evidence", self.evidence))
        rows.append(("repair", list(self.repair.lines(ascii_only=ascii_only))))
        for key, value in rows:
            values = [value] if isinstance(value, str) else list(value)
            if not values:
                continue
            shown, extra = _bound(values)
            lines.append(INDENT + key.ljust(KEY_WIDTH) + " : " + shown[0])
            for v in shown[1:]:
                lines.append(CONT + v)
            if extra:
                lines.append(CONT + "%s and %d more" % ("..." if ascii_only else "…", extra))
        text = "\n".join(lines)
        return fold_ascii(text) if ascii_only else text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.label,
            "rule": self.rule or self.summary,
            "summary": self.summary,
            "entity": self.entity,
            "evidence": self.evidence,
            "where": {k: _jsonable(v) for k, v in self.where},
            "data": {k: _jsonable(v) for k, v in self.data.items()},
            "repair": self.repair.to_dict(),
        }


# Rules write the typographically correct thing; a cp1252 console gets a fold.
# Losing a report to UnicodeEncodeError is a silly way to fail a build gate.
_ASCII_FOLD = {
    "≠": "!=",
    "≈": "~",
    "×": "x",
    "²": "^2",
    "³": "^3",
    "±": "+/-",
    "—": "--",
    "–": "-",
    "…": "...",
    "°": "deg",
    "µ": "u",
    "→": "->",
    "↔": "<->",
    "§": "sec ",
}


def fold_ascii(text: str) -> str:
    for a, b in _ASCII_FOLD.items():
        text = text.replace(a, b)
    return text.encode("ascii", "replace").decode("ascii")


def _bound(values: Sequence[str]) -> Tuple[List[str], int]:
    if len(values) <= MAX_LIST_ITEMS:
        return list(values), 0
    return list(values[:MAX_LIST_ITEMS]), len(values) - MAX_LIST_ITEMS


# -------------------------------------------------------------------- report


@dataclass(frozen=True)
class Skip:
    """A rule that did not run, and why.

    Silence and a pass look identical in most linters, which is how "we checked
    the UVs" turns into "there were no UVs". The report says which rules had
    nothing to check.
    """

    code: str
    reason: str


@dataclass(frozen=True)
class Report:
    asset: str
    profile: str
    findings: Tuple[Finding, ...] = ()
    skipped: Tuple[Skip, ...] = ()

    # -- gate --------------------------------------------------------------

    @property
    def errors(self) -> Tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == ERROR)

    @property
    def warnings(self) -> Tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == WARN)

    @property
    def notes(self) -> Tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == NOTE)

    @property
    def passed(self) -> bool:
        """The export gate: errors block, warnings and notes do not."""
        return not self.errors

    @property
    def clean(self) -> bool:
        """Zero findings of any severity. What a clean fixture must produce."""
        return not self.findings

    @property
    def codes(self) -> Tuple[str, ...]:
        return tuple(f.code for f in self.findings)

    def by_code(self, code: str) -> Tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.code == code)

    def one(self, code: str) -> Finding:
        hits = self.by_code(code)
        if len(hits) != 1:
            raise AssertionError("expected exactly one %s, got %d" % (code, len(hits)))
        return hits[0]

    # -- output ------------------------------------------------------------

    def render(self, ascii_only: Optional[bool] = None, show_skipped: bool = True) -> str:
        if ascii_only is None:
            ascii_only = not _stdout_handles_unicode()
        head = "check  %s   profile %s   %s   %d error, %d warn, %d note" % (
            self.asset,
            self.profile,
            "PASS" if self.passed else "FAIL",
            len(self.errors),
            len(self.warnings),
            len(self.notes),
        )
        blocks = [head]
        if self.clean:
            blocks.append(INDENT + "no findings")
        for f in self.findings:
            blocks.append(f.render(ascii_only=ascii_only))
        if show_skipped and self.skipped:
            lines = ["not checked"]
            for s in self.skipped:
                lines.append(INDENT + s.code.ljust(6) + s.reason)
            blocks.append("\n".join(lines))
        text = "\n\n".join(blocks)
        return fold_ascii(text) if ascii_only else text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "profile": self.profile,
            "passed": self.passed,
            "counts": {
                "error": len(self.errors),
                "warn": len(self.warnings),
                "note": len(self.notes),
            },
            "findings": [f.to_dict() for f in self.findings],
            "not_checked": [{"code": s.code, "reason": s.reason} for s in self.skipped],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.render()


def _stdout_handles_unicode() -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "✗⚠—".encode(enc)
    except (UnicodeEncodeError, LookupError):
        return False
    return True
