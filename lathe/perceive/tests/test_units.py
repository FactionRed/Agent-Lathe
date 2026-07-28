"""Units are part of the type system (DESIGN.md 4.1). These tests hold that line."""

import math

import pytest

from lathe.perceive import fmt
from lathe.perceive.units import (
    Angle,
    Area,
    Length,
    Point,
    Volume,
    as_direction,
    as_length,
    as_point,
    plane_normal_axis,
)


def test_600mm_and_0_6m_unify():
    assert Length.mm(600) == Length.m(0.6)
    assert Length.mm(600).to("m") == pytest.approx(0.6, abs=0.0)
    assert Length.of(600, "mm").si == pytest.approx(0.6)


def test_imperial_round_trip():
    assert Length.inch(1).to("mm") == pytest.approx(25.4, abs=1e-12)
    assert Length.ft(1).to("m") == pytest.approx(0.3048, abs=1e-15)


def test_bare_number_is_a_type_error():
    with pytest.raises(TypeError) as err:
        as_length(600)
    assert "4.1" in str(err.value)  # the error cites the design rule
    assert as_length(600, "mm") == Length.mm(600)


def test_bare_point_is_a_type_error():
    with pytest.raises(TypeError):
        as_point((0, -2, 0.3))
    assert as_point((0, -2, 0.3), "m") == Point.m(0, -2, 0.3)
    assert as_point(Point.mm(0, -2000, 300)) == Point.m(0, -2, 0.3)


def test_dimension_mismatch_is_rejected():
    with pytest.raises(ValueError):
        Length(1.0).to("m^2")
    with pytest.raises(ValueError):
        Area(1.0).to("m")
    with pytest.raises(TypeError):
        Length(1.0) + Area(1.0)


def test_no_implicit_float_conversion():
    """A Length must not silently become a bare number."""
    with pytest.raises(TypeError):
        float(Length.mm(600))


def test_length_arithmetic():
    assert (Length.mm(600) + Length.mm(400)).to("m") == pytest.approx(1.0)
    assert (Length.m(1) * 3).to("m") == pytest.approx(3.0)
    assert Length.m(2) / Length.m(4) == pytest.approx(0.5)
    assert Length.mm(1) < Length.cm(1) < Length.m(1)


def test_formatting_matches_the_design_examples():
    assert Point.m(0.6, 0.6, 0.6).format_size("m") == "0.600 × 0.600 × 0.600 m"
    assert Length.m(0.32).format() == "0.320 m"
    assert Volume(0.164).format() == "0.164 m³"
    assert Point.m(0, 0, 0.287).format() == "(0.000, 0.000, 0.287) m"
    assert Length.m(1.71).format(dp=2) == "1.71 m"


def test_negative_zero_never_prints():
    assert Length(-1e-18).format() == "0.000 m"
    assert "-0.000" not in Point.m(-1e-18, 0, 0).format()


def test_area_and_volume_units():
    assert Area(1.0).to("mm^2") == pytest.approx(1e6)
    assert Volume(1.0).to("mm^3") == pytest.approx(1e9)
    assert Area(2.5).format("m²") == "2.500 m²"
    assert Volume(1e-6).format("mm^3", dp=0) == "1000 mm³"


def test_angle_prints_degrees():
    assert Angle.deg(66).format() == "66.0°"
    assert Angle.deg(90).rad == pytest.approx(math.pi / 2)


def test_directions_are_dimensionless():
    assert as_direction("+Y") == (0.0, 1.0, 0.0)
    assert as_direction("-z") == (0.0, 0.0, -1.0)
    assert as_direction((0, 2, 0)) == (0.0, 1.0, 0.0)  # normalised
    with pytest.raises(TypeError):
        as_direction(Point.m(0, 1, 0))  # a direction is not a position
    with pytest.raises(ValueError):
        as_direction((0, 0, 0))


def test_plane_names():
    assert plane_normal_axis("YZ") == 0
    assert plane_normal_axis("XZ") == 1
    assert plane_normal_axis("xy") == 2
    with pytest.raises(ValueError):
        plane_normal_axis("QQ")


def test_count_and_pct_formatting():
    assert fmt.count(1284) == "1,284"
    assert fmt.pct(0.913) == "91.3%"
    assert fmt.score(0.9982) == "0.998"


def test_ascii_fallback_leaves_no_high_bytes():
    text = fmt.probe_line("measure(bbox)", Point.m(0.6, 0.6, 0.6).format_size("m"))
    assert "→" in text and "×" in text
    ascii_text = fmt.to_ascii(text + Volume(0.164).format())
    ascii_text.encode("ascii")  # raises if anything is left un-transliterated
    assert "->" in ascii_text and " x " in ascii_text and "m^3" in ascii_text
