"""A legacy console codepage must not be able to kill a build.

`models/build_all.py` prints probe output straight to stdout. On a default
Windows console that stream is cp1252, so one `→` from `fmt.probe_line` raised
UnicodeEncodeError and took the run down before the verifier ever ran -- a
build gate lost to a formatting choice.

These tests drive the real print path through a strict cp1252 stream rather
than reading the developer's own console, so they fail everywhere or nowhere.
"""

import contextlib
import io
import re
from pathlib import Path

import pytest

from lathe.perceive import fmt, tier1, tier2
from lathe.perceive.tests import fixtures as fx

PERCEIVE = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def console(encoding):
    """stdout as a console of that codepage, strict about what it cannot encode."""
    stream = io.TextIOWrapper(
        io.BytesIO(), encoding=encoding, errors="strict", write_through=True)
    with contextlib.redirect_stdout(stream):
        yield stream
    stream.flush()


def printed(stream) -> str:
    stream.flush()
    return stream.buffer.getvalue().decode(stream.encoding)


# ------------------------------------------------------- the reported crash


def test_build_all_probe_loop_survives_a_cp1252_console():
    """The exact loop at models/build_all.py:61-64, on a cp1252 stdout."""
    msh = fx.crate()
    with console("cp1252") as stream:
        for probe in (tier1.bbox, tier1.volume, tier1.counts, tier1.is_watertight,
                      tier1.center_of_mass):
            print("  " + str(probe(msh)).strip().replace("\n", "\n  "))
        print("  " + str(tier1.symmetry(msh, "YZ")).strip().replace("\n", "\n  "))
    out = printed(stream)
    assert "measure(bbox) " in out and "->" in out
    assert "→" not in out


@pytest.mark.parametrize("report", [
    lambda m: tier1.report(m),
    lambda m: tier1.report(m, detail=True),
    lambda m: tier2.report(m, texture_px=2048, target_density_px_cm=2.048),
])
def test_reports_survive_a_cp1252_console(report):
    mesh = fx.uv_cube()  # carries UVs, materials and parts: every tier 2 block
    with console("cp1252") as stream:
        print(report(mesh))
    assert "->" in printed(stream)


def test_probe_text_survives_every_codepage_we_claim_to_support():
    """cp1252 is the one that bites, but ascii is the honest lower bound."""
    for encoding in ("cp1252", "ascii", "cp437", "latin-1"):
        with console(encoding) as stream:
            print(tier1.report(fx.stool()))
        assert "->" in printed(stream)


# ------------------------------------------------- and no cost to a good one


def test_a_utf8_console_still_gets_the_typographic_form():
    with console("utf-8") as stream:
        print(tier1.bbox(fx.unit_cube()))
    out = printed(stream)
    assert "→" in out and "×" in out


def test_mcp_output_stays_utf8_behind_a_cp1252_console():
    """The fold follows the sink, not the process.

    Probe text reaching an agent over MCP is UTF-8 whatever console the server
    was launched from, so `lathe_query bbox` must not degrade just because the
    operator started it in cmd.exe.
    """
    from lathe.build.solid import box

    solid = box(0.4, 0.4, 0.4, name="SM_Box_LOD0")
    with console("cp1252"):
        from lathe.mcp import probes

        assert "→" in probes.evaluate(solid, "bbox")
        assert "→" in probes.evaluate(solid, "counts")
        assert "→" in probes.evaluate(solid, "symmetry(YZ)")


def test_lines_stay_utf8_because_they_are_structure_not_output():
    """`lines()` is the structural accessor; only `text()` meets a sink.

    Without this the probe values a caller parses would change under it
    depending on where stdout happened to point.
    """
    with console("cp1252"):
        assert any("→" in line for line in tier1.bbox(fx.unit_cube()).lines())


# ------------------------------------------------------------- the guardrail


def test_every_glyph_the_layer_can_emit_has_an_ascii_fold():
    """Catch the next `∩` at the moment it is added, not on a Windows build.

    A glyph missing from the table would still not raise -- `to_ascii` replaces
    the unknown with '?' -- but '?' is not an answer, and the fix belongs in the
    table.
    """
    missing = {}
    for path in sorted(PERCEIVE.glob("*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for ch in line:
                if ord(ch) > 127 and ch not in fmt._ASCII_MAP:
                    missing.setdefault(ch, "%s:%d" % (path.name, n))
    assert not missing, "no ASCII fold for %s" % missing


def test_to_ascii_leaves_nothing_a_seven_bit_sink_can_refuse():
    folded = fmt.to_ascii("".join(fmt._ASCII_MAP) + "☃ tail")
    folded.encode("ascii")  # raises if anything survived
    assert folded.endswith("? tail")  # the unknown degrades, it does not escape


def test_the_fold_is_all_or_nothing():
    """cp1252 can encode `×` and `°` but not `→`. Half-folded output reads
    worse than either form, so the layer commits to one."""
    with console("cp1252"):
        text = tier1.bbox(fx.unit_cube()).text()
    assert re.search(r"\d x \d", text) and "×" not in text
