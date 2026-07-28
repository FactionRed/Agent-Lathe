"""
Text shaping for probe output.

The output format *is* the product: these blocks are read by an LLM, not by a
person squinting at a viewport. Rules followed here:

  * one fact per line, ``label -> value``, label column padded so values align;
  * tables only when there is a repeating record (islands, parts, LOD levels);
  * counts get thousands separators, ratios get a fixed number of decimals,
    lengths carry a unit (see units.py);
  * no ANSI colour, no box-drawing, no art. Tokens are not free.

Everything here is a pure function of its arguments.
"""

from __future__ import annotations

from typing import Iterable, Sequence

ARROW = "→"
LABEL_WIDTH = 30

_ASCII_MAP = {
    "→": "->",
    "×": "x",
    "²": "^2",
    "³": "^3",
    "°": "deg",
    "±": "+/-",
    "·": "-",
    "≈": "~",
    "⊂": "in",
    "Δ": "d",
    "✗": "FAIL",
    "⚠": "WARN",
    "✓": "OK",
}


def to_ascii(text: str) -> str:
    """Transliterate the handful of non-ASCII glyphs used in probe output.

    Probe text is UTF-8 by default because it reaches the agent over MCP. Call
    this when the sink is a legacy console codepage.
    """
    for k, v in _ASCII_MAP.items():
        text = text.replace(k, v)
    return text


def count(n: int) -> str:
    """1284 -> '1,284' (DESIGN.md 6 prints counts this way)."""
    return format(int(n), ",")


def pct(fraction: float, dp: int = 1) -> str:
    return "%.*f%%" % (dp, 100.0 * fraction)


def ratio(value: float, dp: int = 3) -> str:
    return "%.*f" % (dp, value)


def score(value: float) -> str:
    """A 0..1 score, always 3 decimals so drift is visible."""
    return "%.3f" % value


def boolean(value: bool) -> str:
    return "true" if value else "false"


def probe_line(label: str, value: str, note: str | None = None,
               width: int = LABEL_WIDTH, indent: str = "  ") -> str:
    """'  measure(bbox)               -> 0.600 x 0.600 x 0.600 m'"""
    head = "%s%-*s %s %s" % (indent, width, label, ARROW, value)
    if note:
        head += "   " + note
    return head.rstrip()


def field_line(label: str, value: str, width: int = 18, indent: str = "    ") -> str:
    """Sub-detail under a probe line: '    method : volume_iou'."""
    return ("%s%-*s : %s" % (indent, width, label, value)).rstrip()


def header(title: str, right: str | None = None, width: int = 64) -> str:
    if not right:
        return title
    pad = max(1, width - len(title) - len(right))
    return title + " " * pad + right


def table(headers: Sequence[str], rows: Iterable[Sequence[object]],
          align: str | None = None, indent: str = "    ",
          gap: str = "  ") -> list[str]:
    """Aligned plain-text table. `align` is one char per column: 'l' or 'r'."""
    body = [[("" if c is None else str(c)) for c in row] for row in rows]
    ncol = len(headers)
    align = align or ("l" + "r" * (ncol - 1))
    if len(align) < ncol:
        align = align + "r" * (ncol - len(align))
    widths = [len(h) for h in headers]
    for row in body:
        for i, cell in enumerate(row[:ncol]):
            widths[i] = max(widths[i], len(cell))

    def render(cells: Sequence[str]) -> str:
        out = []
        for i in range(ncol):
            cell = cells[i] if i < len(cells) else ""
            out.append(cell.rjust(widths[i]) if align[i] == "r" else cell.ljust(widths[i]))
        return (indent + gap.join(out)).rstrip()

    lines = [render(list(headers)), indent + gap.join("-" * w for w in widths)]
    lines.extend(render(r) for r in body)
    return lines


def histogram_line(label: str, buckets: Sequence[tuple[str, int]],
                   width: int = LABEL_WIDTH, indent: str = "  ",
                   total: int | None = None) -> str:
    """'  valence                      -> 3:8  4:512  5:96  6:30'"""
    parts = ["%s:%s" % (name, count(n)) for name, n in buckets if n]
    value = "  ".join(parts) if parts else "(empty)"
    note = "of %s" % count(total) if total is not None else None
    return probe_line(label, value, note=note, width=width, indent=indent)


def verdict(ok: bool, warn: bool = False) -> str:
    if not ok:
        return "FAIL"
    return "WARN" if warn else "OK"


def block(title: str, lines: Iterable[str], right: str | None = None) -> str:
    return "\n".join([header(title, right)] + list(lines))


__all__ = [
    "ARROW",
    "to_ascii",
    "count",
    "pct",
    "ratio",
    "score",
    "boolean",
    "probe_line",
    "field_line",
    "header",
    "table",
    "histogram_line",
    "verdict",
    "block",
]
