"""Lathe authoring layer. DESIGN.md sections 2, 4, 5 and the section 11 bevel."""

from .solid import (  # noqa: F401
    CardinalityError,
    LatheError,
    Selection,
    Solid,
    array,
    assert_count,
    bowl_profile,
    box,
    cylinder,
    extrude,
    mm,
    revolve,
    sphere,
    union_all,
)
from .bevel import bevel, bevel_available  # noqa: F401
from .uvmap import (  # noqa: F401
    PADDING_TABLE,
    island_labels,
    overlap_fraction,
    rescale_to_density,
    texel_density,
    unwrap,
)

__all__ = [
    "Solid", "Selection", "LatheError", "CardinalityError",
    "box", "cylinder", "sphere", "revolve", "extrude",
    "array", "union_all", "assert_count", "mm", "bowl_profile",
    "bevel", "bevel_available",
    "unwrap", "rescale_to_density", "texel_density", "island_labels",
    "overlap_fraction", "PADDING_TABLE",
]
