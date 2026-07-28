"""
Lathe DESIGN.md section 7 -- profiles.

"Rules are grouped into profiles selected by `emit`."

A profile is three pieces of data: which codes run, what parameters they run
with, and any severity change. Nothing else. The test suite asserts that a
profile defined outside this module works identically, which is the real proof
that `character_skinned` and `mobile_vr` will not require engine surgery.

Severity is a profile decision, not a rule decision. The same measurement is an
error on a mobile target and a note on a film asset; that is a policy, and
policy belongs with the profile.
"""

from __future__ import annotations

from .engine import Profile, register_profile
from .findings import ERROR, NOTE, WARN

# --------------------------------------------------------- game_prop_pbr

GAME_PROP_PBR = register_profile(
    Profile(
        name="game_prop_pbr",
        description=(
            "Static prop for a PBR game pipeline. The section 7 table, with the "
            "codes that are implemented."
        ),
        rules=(
            "E101",  # non-manifold edges or vertices
            "E102",  # n-gons
            "E103",  # inverted / inconsistent normals
            "E104",  # unapplied scale or rotation
            "E105",  # loose verts / edges, zero-area faces
            "E106",  # overlapping UVs on unique bake
            "E107",  # triangle budget
            "E108",  # component count
            "W201",  # origin not at declared reference
            "W202",  # uv padding
            "W203",  # non-power-of-two textures
            "W302",  # texel density variance
            "N401",  # naming convention
        ),
        params={
            # A prop is a watertight solid. Cards and decals are not props.
            "E101": {"allow_boundary": False},
            # Quads survive an engine import predictably; n-gons do not.
            "E102": {"max_face_verts": 4},
            "E105": {"min_face_area": 1e-9},
            "E107": {"tolerance_pct": 0.0},
            "W201": {"tolerance_mm": 1.0},
            "W302": {"tolerance_pct": 10.0},
            "N401": {"default_prefix": "SM"},
        },
    )
)

# ------------------------------------------------------- character_skinned

CHARACTER_SKINNED = register_profile(
    Profile(
        name="character_skinned",
        description=(
            "Deforming character mesh. Section 7 promises edge-loop counts at "
            "joints, pole placement, quad ratio and symmetry; those rules are not "
            "implemented, so this profile is the honest subset plus the parameter "
            "changes a skinned asset needs."
        ),
        rules=(
            "E101",
            "E102",
            "E103",
            "E104",
            "E105",
            "E106",
            "E107",
            "E108",
            "W201",
            "W202",
            "W203",
            "W302",
            "N401",
        ),
        params={
            "E101": {"allow_boundary": False},
            # Triangles in a deforming surface pin the diagonal and pop under
            # skinning, so a character is quads-only and quads are the maximum.
            "E102": {"max_face_verts": 4},
            "E104": {"scale_tolerance": 1e-9, "rotation_tolerance_deg": 1e-6},
            "W201": {"tolerance_mm": 0.1},  # rigs are built on the origin
            "W302": {"tolerance_pct": 15.0},  # faces get more texels on purpose
            "N401": {"default_prefix": "SK"},
        },
        severity={
            # A skinned mesh with a stray component is usually a lost limb, but
            # the count is often genuinely unknown while blocking out.
            "E108": WARN,
        },
    )
)

# --------------------------------------------------------------- mobile_vr

MOBILE_VR = register_profile(
    Profile(
        name="mobile_vr",
        description=(
            "Mobile and standalone VR. Open shells are legitimate here (foliage "
            "cards, decals), the triangle budget is the hard constraint, and "
            "texel density drift shows up as visible seams on a low-resolution "
            "atlas, so it is an error rather than a warning."
        ),
        rules=(
            "E101",
            "E102",
            "E103",
            "E104",
            "E105",
            "E106",
            "E107",
            "E108",
            "W201",
            "W202",
            "W203",
            "W302",
            "N401",
        ),
        params={
            "E101": {"allow_boundary": True},  # cards are open by design
            "E102": {"max_face_verts": 3},  # ship triangulated, no import surprises
            "E107": {"tolerance_pct": 0.0},
            "W202": {"padding_table": {256: 2, 512: 4, 1024: 8, 2048: 16}},
            "W302": {"tolerance_pct": 8.0},
        },
        severity={
            "W302": ERROR,
            "W203": ERROR,  # NPOT costs the mip chain on tiled GPUs
            "W201": NOTE,
        },
    )
)


def all_profiles():
    from .engine import PROFILES

    return dict(PROFILES)
