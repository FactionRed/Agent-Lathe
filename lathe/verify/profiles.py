"""
Lathe DESIGN.md section 7 -- profiles.

"Rules are grouped into profiles selected by `emit`."

A profile is three pieces of data: which codes run, what parameters they run
with, and any severity change. Nothing else. The test suite asserts that a
profile defined outside this module works identically, which is the real proof
that `character_skinned` and `mobile_vr` will not require engine surgery.

All four profiles section 7 names now ship. `hard_surface_subdiv` and `film_usd`
were the pair still outstanding, and adding them cost exactly what the claim
above predicted: two literals in this file and no engine change at all.

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
            "W206",  # collapsed uv area
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
            "An asset with a rig. Section 7 promises edge-loop counts at joints, "
            "pole placement and quad ratio; those are not implemented, so this "
            "profile is the honest subset, the parameter changes a deforming "
            "asset needs, and the articulation rules. Symmetry is implemented, "
            "but as a declared body plan rather than the bilateral one section 7 "
            "assumed -- see W505."
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
            "W206",
            "W203",
            "W302",
            "N401",
            # Articulation. Selected here and nowhere else: this is the profile
            # for an asset that has a rig, and a static prop would only collect
            # five identical "declares no articulation" lines forever -- the
            # same nagging film_usd drops E107 to avoid. An articulated prop
            # that is not a character selects this profile, or derives one: a
            # profile is a tuple of codes, so that costs a literal.
            #
            # Nothing in this block assumes a body plan. E501/W503 are per-hinge,
            # W504 compares a declared root count rather than assuming one, and
            # W505 checks the symmetry the program declared -- radial, bilateral
            # or none. A starfish, a serpent and a hexapod are all gradeable, and
            # none of them is measured against a human.
            "E501",  # joint outside its declared limit
            "E502",  # declared reach target missed
            "W503",  # pose resting on a limit
            "W504",  # root joint count
            "W505",  # declared body plan does not hold
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
            "W206",
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


# ------------------------------------------------------- hard_surface_subdiv

HARD_SURFACE_SUBDIV = register_profile(
    Profile(
        name="hard_surface_subdiv",
        description=(
            "Hard-surface model emitted as a subdivision cage. The shipped "
            "surface is the limit surface, not these faces, and that single "
            "fact sets every parameter below: the cage must be all quads, the "
            "declared budget is the cage, and texel density measured on the "
            "cage does not describe what the renderer shades."
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
            "W206",
            "W203",
            "W302",
            "N401",
        ),
        params={
            "E101": {"allow_boundary": False},
            # Catmull-Clark turns every non-quad into an extraordinary vertex,
            # and the pinch it produces is visible on the limit surface. A cage
            # is quads or it is a bug.
            #
            # Both bounds, because a maximum alone does not say "quads". With
            # only `max_face_verts: 4` this profile passed a fully triangulated
            # cage -- an affirmative PASS on the exact input the comment above
            # calls a bug, and a triangle is the non-quad that actually turns up.
            "E102": {"max_face_verts": 4, "min_face_verts": 4},
            # A subdivision modifier evaluates in object space, so an unapplied
            # transform changes the resulting surface rather than merely being
            # reinterpreted on import. Tighter than the prop default for the
            # same reason character_skinned is.
            "E104": {"scale_tolerance": 1e-9, "rotation_tolerance_deg": 1e-6},
            # A cage has few, large faces. A sliver that would vanish in a dense
            # mesh is a pinch in the limit surface here, so the threshold is
            # raised rather than left at the dense-mesh default.
            "E105": {"min_face_area": 1e-7},
            # The budget declares the cage. What the renderer dices it to is a
            # render setting, not an emit-time fact.
            "E107": {"tolerance_pct": 0.0},
            "W201": {"tolerance_mm": 0.1},  # kit pieces snap to a grid
            # Subdivision pulls the surface in toward the limit, so a cage reads
            # systematically low. The band widens for that, and stops there --
            # the severity below carries the rest of the argument.
            "W302": {"tolerance_pct": 15.0},
            "N401": {"default_prefix": "SM"},
        },
        severity={
            # Density is measured on the cage, and subdivision pulls the surface
            # in toward the limit, so the number reads low by an amount this
            # rule cannot know. Reporting it is useful; blocking an export on it
            # would be blocking on a measurement of the wrong surface.
            #
            # W206 is deliberately *not* demoted alongside it, though the instinct
            # is to treat the two as one policy. They are not the same kind of
            # claim: W302 is a measurement of the wrong surface, while a collapsed
            # island is a topological fact that subdivision cannot undo.
            # Subdivision interpolates UVs -- it does not invent UV area where
            # there is none, so an island with no area on the cage has no area on
            # the limit surface either.
            "W302": NOTE,
        },
    )
)

# ---------------------------------------------------------------- film_usd

FILM_USD = register_profile(
    Profile(
        name="film_usd",
        description=(
            "Film asset headed for a USD pipeline. Offline renderers dice and "
            "subdivide at render time and composed xformOps survive as authored, "
            "so two of the game profile's hard errors are not defects here. "
            "There is no triangle budget in film, so E107 is not selected at "
            "all rather than nagging as *not checked* forever."
        ),
        rules=(
            "E101",
            "E102",
            "E103",
            "E104",
            "E105",
            "E106",
            "E108",
            "W201",
            "W202",
            # A wider density band is the point of UDIM authoring. Zero density is
            # not a wide band -- a collapsed island samples one texel whatever
            # tile it sits in, so this stays selected at its default severity.
            "W206",
            "W203",
            "W302",
            "N401",
        ),
        params={
            # Cards, cloth and hair meshes are open by design, as they are on
            # mobile_vr. Edges with more than 2 faces stay unconditional.
            "E101": {"allow_boundary": True},
            "E102": {"max_face_verts": 4},
            # UDIM tiles are authored per-region at different resolutions, so a
            # wider band is the layout working, not drifting.
            "W302": {"tolerance_pct": 25.0},
            # UDIM maps are named `<name>.<udim>.<ext>`, which cannot satisfy
            # [T]_[Name]_[Variant]_[Channel]. Checking them would emit a note per
            # tile for a convention this pipeline does not use.
            "N401": {"check_textures": False},
        },
        severity={
            # Quads still matter where the asset is subdivided, so the n-gons
            # are still worth naming -- but a renderer that accepts arbitrary
            # polygons makes this advice, not a gate.
            "E102": NOTE,
            # USD composes xformOps onto the prim and preserves them exactly.
            # Unlike an FBX round trip, nothing silently reinterprets the
            # numbers, so a non-identity transform is worth flagging and not
            # worth blocking.
            "E104": WARN,
            # A film texture system mips arbitrary dimensions. Non-power-of-two
            # costs nothing here, so it is an observation.
            "W203": NOTE,
        },
    )
)


def all_profiles():
    from .engine import PROFILES

    return dict(PROFILES)
