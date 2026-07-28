"""
SM_Chair_Oak_LOD0 -- a dining chair, modelled at real-world scale.

Written the way DESIGN.md section 4 argues an agent should work: named
parameters, positions as expressions over them, and section 5.3 assertions on
anything a later edit could silently break.

Dimensions come from furniture practice, not from taste: 450 mm seat height,
900 mm overall, 40 mm leg section.
"""

import lathe.build as L
from lathe.build import mm

MATERIALS = {
    "seat": "wood_mid",
    "leg": "wood_dark",
    "stile": "wood_dark",
    "slat": "wood_light",
    "rail": "wood_dark",
    "bevel": "wood_mid",
}


def build(seat_h=0.45, seat_w=0.42, seat_d=0.40, leg=0.040, slat_n=3,
          back_h=0.90, bevel_w=mm(2.0)):
    # -- parameters as expressions, so one edit moves everything that depends
    seat_t = mm(35)
    inset = mm(28)
    lx = seat_w / 2 - inset - leg / 2
    ly = seat_d / 2 - inset - leg / 2
    seat_top = seat_h
    seat_mid = seat_top - seat_t / 2

    seat = L.box(seat_w, seat_d, seat_t, tag="seat", name="SM_Chair_Oak_LOD0").at(z=seat_mid)

    # Front legs stop under the seat. Back legs continue up and become the
    # stiles the backrest attaches to -- one piece, as a joiner would cut it.
    front_h = seat_top - seat_t
    front = [
        L.box(leg, leg, front_h, tag="leg").at(x=sx * lx, y=-ly, z=front_h / 2)
        for sx in (-1, 1)
    ]
    stile_h = back_h
    stiles = [
        L.box(leg, leg, stile_h, tag="stile").at(x=sx * lx, y=ly, z=stile_h / 2)
        for sx in (-1, 1)
    ]

    # Side rails brace front to back near the floor.
    rail_z = mm(120)
    rails = [
        L.box(mm(22), seat_d - inset * 2 - leg, mm(45), tag="rail")
        .at(x=sx * lx, y=0.0, z=rail_z)
        for sx in (-1, 1)
    ]
    # Back and front rails between the legs.
    rails.append(
        L.box(seat_w - inset * 2 - leg, mm(22), mm(45), tag="rail")
        .at(y=ly, z=rail_z))
    rails.append(
        L.box(seat_w - inset * 2 - leg, mm(22), mm(45), tag="rail")
        .at(y=-ly, z=rail_z))

    # Top rail, flush with the tops of the stiles. Without it the stiles end in
    # mid-air, which the first render made obvious from the side.
    top_t = mm(55)
    rails.append(
        L.box(seat_w - inset * 2 + leg, mm(20), top_t, tag="rail")
        .at(y=ly, z=back_h - top_t / 2))

    # Backrest slats, spread between the seat and the underside of the top rail.
    slat_w = seat_w - inset * 2 - leg
    span_lo = seat_top + mm(120)
    span_hi = back_h - top_t - mm(70)
    slats = []
    for i in range(slat_n):
        t = 0.5 if slat_n == 1 else i / (slat_n - 1.0)
        z = span_lo + t * (span_hi - span_lo)
        slats.append(
            L.box(slat_w, mm(16), mm(70), tag="slat").at(y=ly, z=z))

    chair = L.union_all([seat] + front + stiles + rails + slats,
                        name="SM_Chair_Oak_LOD0")

    # -- section 5.3 guards. Change slat_n or inset and these say so.
    L.assert_count(chair.faces(tag="seat"), 6, "seat faces")
    # Four visible faces per slat, not six: both ends are buried in the stiles.
    # The first version of this line guessed five and the assertion said so.
    L.assert_count(chair.faces(tag="slat"), slat_n * 4, "slat faces")
    if chair.components() != 1:
        raise L.LatheError(
            "chair is in %d pieces, expected 1. A leg or slat is not touching "
            "the seat -- check inset (%.3f m) against seat size."
            % (chair.components(), inset))

    # Bevel every sharp edge. Real timber has no zero-radius arris, and the
    # wrapper enforces the non-inheriting tag default, so the assertions above
    # still hold afterwards.
    chair = L.bevel(chair, width=bevel_w, segments=2, min_dihedral=55,
                    min_length=mm(4))
    return chair


if __name__ == "__main__":
    c = build()
    print(c.name, c.n_tris, "tris", c.components(), "component(s)")
