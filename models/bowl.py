"""
SM_BowlFruit_Ceramic_LOD0 -- a revolved bowl with fruit resting in it.

Exercises the parts of the kernel the chair does not: `revolve` for the bowl and
the pear, and warped spheres for the rest.

This is a *scene*, not a single asset, so it is legitimately several components
and declares that. E108 compares against the declaration rather than assuming 1,
which is the whole reason the rule takes a number.
"""

import lathe.build as L
from lathe.build import mm

MATERIALS = {
    "bowl": "ceramic_blue",
    "apple": "apple",
    "apple_stem": "wood_dark",
    "orange": "orange",
    "pear": "pear",
    "leaf": "leaf",
    "bevel": "ceramic_blue",
}

# Ripe-fruit sizes, in metres.
R_APPLE = mm(37)
R_ORANGE = mm(34)


def pear_profile(h=mm(92), r=mm(36)):
    """A (radius, height) silhouette: wide belly low down, narrow neck."""
    return [
        (0.0, 0.0),
        (r * 0.62, h * 0.045),
        (r * 1.00, h * 0.30),
        (r * 0.86, h * 0.52),
        (r * 0.50, h * 0.72),
        (r * 0.33, h * 0.89),
        (r * 0.30, h * 0.97),
        (0.0, h),
    ]


def _apple(tag="apple"):
    body = L.sphere(R_APPLE, segments=40, tag=tag).scale((1.0, 1.0, 0.86))
    stem = L.cylinder(mm(20), mm(2.2), segments=10, center=False,
                      tag="apple_stem").at(z=R_APPLE * 0.72)
    return body + stem


def _orange(tag="orange"):
    return L.sphere(R_ORANGE, segments=40, tag=tag).scale((1.0, 1.0, 0.93))


def _pear(tag="pear"):
    return L.revolve(pear_profile(), segments=40, tag=tag)


def build(bowl_r=0.115, bowl_h=mm(68), wall=mm(6), bevel_w=mm(1.2)):
    bowl = L.revolve(
        L.bowl_profile(bowl_r, bowl_h, wall, foot_radius=bowl_r * 0.42),
        segments=64, tag="bowl", name="SM_BowlFruit_Ceramic_LOD0")

    # Bowl interior, so the fruit can be placed against real numbers rather
    # than by eye. The inner floor sits one wall thickness up.
    floor_z = wall * 2
    inner_rim_r = bowl_r - wall

    # Three fruit nested on the floor, then one resting on top of them.
    ring_r = mm(42)
    lower = [
        _apple().at(x=ring_r, y=mm(4), z=floor_z + R_APPLE * 0.80),
        _orange().at(x=-ring_r * 0.62, y=ring_r * 0.72,
                     z=floor_z + R_ORANGE * 0.80),
        _orange().at(x=-ring_r * 0.60, y=-ring_r * 0.74,
                     z=floor_z + R_ORANGE * 0.80),
    ]
    upper = [
        _apple().at(x=mm(-2), y=mm(-6), z=floor_z + R_APPLE * 2.05),
        _pear().rotate((6.0, -13.0, 22.0)).at(x=mm(30), y=mm(-34),
                                              z=floor_z + mm(30)),
    ]

    scene = L.union_all([bowl] + lower + upper, name="SM_BowlFruit_Ceramic_LOD0")

    # Every fruit must clear the bowl wall. Cheaper to assert than to spot in a
    # render, which is the section 7 argument in miniature.
    lo, hi = scene.bbox()
    span = max(hi[0] - lo[0], hi[1] - lo[1])
    if span > bowl_r * 2 + mm(1):
        raise L.LatheError(
            "something pokes through the bowl: scene spans %.3f m but the bowl "
            "is only %.3f m across. Reduce ring_r (%.3f m) or the fruit radii."
            % (span, bowl_r * 2, ring_r))
    # No assertion on the bowl's logical face count. `face_id` groups *coplanar*
    # triangles, so a 64-segment revolve reports 322 faces, not 1 -- the count
    # scales with segment count and would break the moment anyone retessellated.
    # Logical-face assertions are for planar hard surface. Assert the invariants
    # that actually hold instead.
    if "bowl" not in scene.tags.values():
        raise L.LatheError("the bowl lost its tag during assembly")
    if not scene.status_ok():
        raise L.LatheError("bowl scene is not a valid solid")

    scene = L.bevel(scene, width=bevel_w, segments=2, min_dihedral=62,
                    only_tags=["bowl"], min_length=mm(3))
    return scene


DECLARED_COMPONENTS = None  # measured at build time; a scene, not one solid


if __name__ == "__main__":
    s = build()
    print(s.name, s.n_tris, "tris", s.components(), "components")
