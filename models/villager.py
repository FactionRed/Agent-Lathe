"""
SM_Villager_Stylized_LOD0 -- and an honest account of what this is not.

DESIGN.md section 10 states the limit plainly: procedurally generated meshes do
not have animation-grade edge flow, and the realistic path for deforming assets
is template basemeshes plus cage deformation, not generation from primitives.
That limit is real and this file does not beat it.

What this IS: a stylized low-poly figure assembled from tapered revolves and
spheres, at correct human proportions for 1.7 m. Good enough for a background
villager seen at distance in a stylized game.

What this is NOT, and cannot be from primitives:
  * riggable. There are no edge loops at the shoulder, elbow, hip or knee, so
    it would pinch and tear the moment a bone rotated it.
  * a face. Eyes are placed spheres. There is no brow, nose, mouth or ear
    topology, and no amount of boolean work would produce any.
  * anatomically continuous. Limbs meet the torso by intersection, not by a
    shared surface, so the silhouette has seams a sculpt would not.

Read the poly budget accordingly: this is prop-grade geometry in the shape of a
person, not a character asset.
"""

import lathe.build as L
from lathe.build import mm

MATERIALS = {
    "skin": "skin",
    "hair": "hair_brown",
    "eye": "wood_dark",
    "tunic": "cloth_tan",
    "belt": "leather",
    "trouser": "cloth_blue",
    "boot": "leather",
    "sleeve": "cloth_tan",
    # The cut surface left by a trim is still hair, so it shades as hair. The
    # tag stays distinct from "hair" on purpose -- section 5.2 wants a cut face
    # to remain addressable -- but it must not fall through to grey default.
    "hair_cut": "hair_brown",
}

HEIGHT = 1.70


def pivot(solid, degrees, about):
    """Rotate about an arbitrary point. Manifold rotates about the origin."""
    ax, ay, az = about
    return solid.at(x=-ax, y=-ay, z=-az).rotate(degrees).at(x=ax, y=ay, z=az)


def _limb(length, r_far, r_near, segments=16, tag="skin"):
    """A tapered limb segment growing in +z, `r_near` at the top."""
    return L.cylinder(length, r_far, r_near, segments=segments, center=False,
                      tag=tag)


def build(height=HEIGHT):
    s = height / 1.70  # single scale factor, so proportions stay fixed

    hip_z = 0.94 * s
    shoulder_z = 1.38 * s
    neck_z = 1.45 * s
    head_c = 1.575 * s

    parts = []

    # -- head, neck, hair, eyes ------------------------------------------
    head = (L.sphere(0.106 * s, segments=36, tag="skin")
            .scale((0.90, 1.00, 1.10)).at(z=head_c))
    # The neck must run *into* the head, not up to it. At 0.10 m it stopped
    # 11 mm short and the head came off as its own component.
    neck = _limb(0.15 * s, 0.046 * s, 0.050 * s, 14, "skin").at(z=neck_z - 0.10 * s)

    # Hair: a slightly larger dome, trimmed flat so it sits like a cap rather
    # than a full sphere.
    hair_ball = (L.sphere(0.112 * s, segments=32, tag="hair")
                 .scale((0.92, 1.00, 1.04)).at(z=head_c + 0.016 * s))
    # Trim higher than before: the old cap reached the brow and hid the face.
    hair = hair_ball - L.box(0.5 * s, 0.5 * s, 0.5 * s, tag="hair_cut").at(
        z=head_c + 0.022 * s - 0.25 * s)
    # And cut the front back, so it reads as a hairline rather than a helmet.
    hair = hair - L.box(0.5 * s, 0.16 * s, 0.10 * s, tag="hair_cut").at(
        y=-0.115 * s, z=head_c + 0.050 * s)

    eye_r = 0.0130 * s
    eyes = [
        L.sphere(eye_r, segments=12, tag="eye").scale((1.15, 0.75, 0.9)).at(
            x=sx * 0.038 * s, y=-0.078 * s, z=head_c + 0.006 * s)
        for sx in (-1, 1)
    ]
    parts += [head, neck, hair] + eyes

    # -- torso: a tunic, wider at the shoulder than the waist ------------
    torso_h = shoulder_z - hip_z + 0.06 * s
    tunic = (_limb(torso_h, 0.148 * s, 0.166 * s, 28, "tunic")
             .scale((1.0, 0.70, 1.0)).at(z=hip_z - 0.02 * s))
    # Skirt of the tunic flares below the belt. Kept short so the forearms read
    # as arms instead of merging with the hem.
    skirt = (_limb(0.115 * s, 0.180 * s, 0.150 * s, 28, "tunic")
             .scale((1.0, 0.72, 1.0)).at(z=hip_z - 0.095 * s))
    belt = (L.cylinder(0.035 * s, 0.158 * s, 0.158 * s, segments=28,
                       center=False, tag="belt")
            .scale((1.0, 0.72, 1.0)).at(z=hip_z + 0.03 * s))
    parts += [tunic, skirt, belt]

    # -- arms: sleeve to the elbow, bare forearm, hand -------------------
    for sx in (-1, 1):
        sh = (sx * 0.150 * s, 0.0, shoulder_z)
        upper_len = 0.29 * s
        upper = _limb(upper_len, 0.044 * s, 0.056 * s, 16, "sleeve").at(
            x=sh[0], z=shoulder_z - upper_len)
        # 5 degrees, not 9. At 9 the figure measured 0.532 m across, which is
        # far too broad for 1.7 m -- arms at rest hang close to the body.
        upper = pivot(upper, (0, sx * 5.0, 0), sh)

        elbow_z = shoulder_z - upper_len * 0.99
        el = (sx * (0.150 * s + 0.026 * s), 0.0, elbow_z)
        fore_len = 0.25 * s
        fore = _limb(fore_len, 0.037 * s, 0.043 * s, 16, "skin").at(
            x=el[0], z=elbow_z - fore_len)
        fore = pivot(fore, (0, sx * 2.0, 0), el)

        hand = (L.sphere(0.044 * s, segments=16, tag="skin")
                .scale((0.70, 0.92, 1.10))
                .at(x=el[0] + sx * 0.012 * s, z=elbow_z - fore_len - 0.010 * s))
        parts += [upper, fore, hand]

    # -- legs and boots --------------------------------------------------
    for sx in (-1, 1):
        hip = (sx * 0.076 * s, 0.0, hip_z)
        thigh_len = 0.47 * s
        thigh = _limb(thigh_len, 0.062 * s, 0.088 * s, 18, "trouser").at(
            x=hip[0], z=hip_z - thigh_len)
        thigh = pivot(thigh, (0, sx * 3.0, 0), hip)

        knee_z = hip_z - thigh_len
        # Run the shin down *inside* the boot. Ending it exactly on the boot's
        # top face is coincident contact, which a boolean treats as disconnected
        # -- both boots came off as separate components that way.
        shin_len = knee_z - 0.040 * s
        # Shin top matches thigh bottom (0.062) and sits on the same axis. The
        # earlier 0.060 radius plus a 12 mm offset showed as a step at the knee.
        shin = _limb(shin_len, 0.049 * s, 0.062 * s, 18, "trouser").at(
            x=hip[0] + sx * 0.004 * s, z=knee_z - shin_len)

        boot = L.box(0.092 * s, 0.215 * s, 0.075 * s, tag="boot").at(
            x=hip[0] + sx * 0.004 * s, y=0.034 * s, z=0.0375 * s)
        parts += [thigh, shin, boot]

    figure = L.union_all(parts, name="SM_Villager_Stylized_LOD0")

    # Recentre in XY on the bounding box. The boots reach further forward than
    # the heels reach back, which left the origin 5.95 mm off the declared
    # bottom_center and tripped W201.
    lo0, hi0 = figure.bbox()
    figure = figure.at(x=-(lo0[0] + hi0[0]) / 2, y=-(lo0[1] + hi0[1]) / 2)

    # -- invariants worth guarding ---------------------------------------
    lo, hi = figure.bbox()
    got_h = hi[2] - lo[2]
    if abs(got_h - height) > 0.02:
        raise L.LatheError(
            "villager is %.3f m tall, expected %.3f m. A limb probably escaped "
            "the intended proportions." % (got_h, height))
    if abs(lo[2]) > 1e-6:
        raise L.LatheError(
            "villager does not stand on z=0 (lowest point %.4f m); boots are "
            "floating or sunk." % lo[2])
    n = figure.components()
    if n != 1:
        # Say *which* pieces are loose and where. "A limb is not touching the
        # torso" is not a repair; a bounding box is.
        lines = []
        for i, part in enumerate(sorted(figure.man.decompose(),
                                        key=lambda p: -p.volume())):
            b = part.bounding_box()
            lines.append(
                "    piece %d  vol %.6f m3  x %.3f..%.3f  y %.3f..%.3f  z %.3f..%.3f"
                % (i, part.volume(), b[0], b[3], b[1], b[4], b[2], b[5]))
        raise L.LatheError(
            "villager is in %d pieces, expected 1.\n%s\n"
            "  The largest piece is the body. Any small piece is a limb or "
            "feature whose placement missed what it should intersect."
            % (n, "\n".join(lines)))
    return figure


if __name__ == "__main__":
    v = build()
    print(v.name, v.n_tris, "tris", v.components(), "components")
