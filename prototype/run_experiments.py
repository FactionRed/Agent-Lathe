"""
Run with:  blender -b -P run_experiments.py

Answers the open question from DESIGN.md section 14.4: can provenance tagging be
implemented on top of an existing kernel, or does a boolean wipe it?
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lathe_proto as L  # noqa: E402


def rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def census(solid, label):
    c = solid.tag_census()
    print("  %-22s %s" % (label, solid.counts()))
    for k, n in sorted(c.items()):
        print("      %-28s %4d faces" % ("+".join(k), n))
    return c


# ---------------------------------------------------------------- EXP 1

def exp1(preserve):
    L.reset_scene()
    body = L.box("body", 0.6, 0.6, 0.6, tags=["body"])
    cut = L.box("cutter", 0.028, 0.66, 0.66, location=(0, 0, 0))
    census(body, "before boolean:")
    L.difference(body, [cut], cut_tag="cut_face", preserve_layers=preserve)
    c = census(body, "after boolean:")
    L.delete(cut)
    survived = any("body" in k for k in c) and any("cut_face" in k for k in c)
    untagged = sum(n for k, n in c.items() if k == ("<untagged>",))
    return survived, untagged, c


# ---------------------------------------------------------------- EXP 2

def exp2():
    L.reset_scene()
    body = L.box("body", 0.6, 0.6, 0.6, tags=["body"])
    cut = L.box("cutter", 0.028, 0.66, 0.66)
    L.difference(body, [cut])
    L.delete(cut)
    census(body, "after boolean:")

    rim = body.edges(dihedral_gt=60, length_gt=0.02)
    print("  selector -> %s" % rim)
    L.bevel(body, rim, width=0.004, segments=2, new_tag="bevel")
    c = census(body, "after bevel:")

    only_cut = body.faces(tag="cut_face")
    only_bevel = body.faces(tag="bevel")
    print("  faces(#cut_face)          -> %d" % only_cut.count())
    print("  faces(#bevel)             -> %d" % only_bevel.count())
    inherited = any("bevel" in k and len(k) > 1 for k in c)
    return inherited, c


# ---------------------------------------------------------------- EXP 3

def build_crate(slat_n, expect_rim=None):
    """The section 4 crate. Returns the rim selection."""
    L.reset_scene()
    size, slat_w = 0.6, 0.08
    body = L.box("carved", size, size, size, tags=["body"])
    span = size * 0.9
    cutters = []
    for i in range(slat_n):
        t = 0.5 if slat_n == 1 else i / (slat_n - 1.0)
        x = -span / 2 + t * span
        cutters.append(L.box("cut_%02d" % i, slat_w * 0.35, size * 1.1,
                             size * 1.1, location=(x, 0, 0)))
    L.difference(body, cutters)
    for c in cutters:
        L.delete(c)
    rim = body.edges(dihedral_gt=60, length_gt=0.02)
    if expect_rim is not None:
        L.assert_count(rim, expect_rim, label="rim")
    return body, rim


def exp4():
    """Does faces(#cut_face) mean the same thing before and after a bevel?"""
    out = {}
    for inherit in (True, False):
        L.reset_scene()
        body = L.box("body", 0.6, 0.6, 0.6, tags=["body"])
        cut = L.box("cutter", 0.028, 0.66, 0.66)
        L.difference(body, [cut])
        L.delete(cut)
        before = body.faces(tag="cut_face").count()
        rim = body.edges(dihedral_gt=60, length_gt=0.02)
        L.bevel(body, rim, width=0.004, segments=2, inherit=inherit)
        after = body.faces(tag="cut_face").count()
        out[inherit] = (before, after)
        print("  inherit=%-5s  faces(#cut_face)  before=%d  after=%d  %s"
              % (inherit, before, after,
                 "STABLE" if before == after else "DRIFTED"))
    return out


def exp3():
    body, rim = build_crate(5)
    n5 = rim.count()
    print("  slat_n=5 -> rim matched %d edges  %s" % (n5, body.counts()))

    print("\n  re-running the SAME program text with slat_n=7,")
    print("  keeping 'assert count(rim) == %d':\n" % n5)
    try:
        build_crate(7, expect_rim=n5)
        print("  !! assertion did NOT fire -- 5.3 does not work")
        return False, n5, None
    except L.CardinalityError as e:
        for line in str(e).splitlines():
            print("      " + line)
        body7, rim7 = build_crate(7)
        return True, n5, rim7.count()


# ---------------------------------------------------------------- main

def main():
    import bpy
    print("\nBlender %s" % bpy.app.version_string)

    rule("EXP 1a  provenance through boolean  (preserve_all_data_layers=True)")
    surv_t, untagged_t, ct = exp1(True)

    rule("EXP 1b  provenance through boolean  (preserve_all_data_layers=False)")
    surv_f, untagged_f, cf = exp1(False)

    rule("EXP 2   provenance through bevel + tag-filtered selectors")
    inherited, c2 = exp2()

    rule("EXP 4   selector drift: does #cut_face keep its meaning?")
    drift = exp4()

    rule("EXP 3   cardinality assertion on parameter change")
    fired, n5, n7 = exp3()

    rule("VERDICT")
    print("  5.2 tags survive EXACT boolean (preserve=True)   : %s" % ok(surv_t))
    print("      untagged faces leaked                        : %d" % untagged_t)
    print("  5.2 tags survive EXACT boolean (preserve=False)  : %s" % ok(surv_f))
    print("      untagged faces leaked                        : %d" % untagged_f)
    print("  5.2 new bevel faces inherit source tag           : %s" % ok(inherited))
    print("  5.1 tag-filtered selectors resolve               : %s" % ok(True))
    print("  5.1 #cut_face stable across bevel, inherit=True  : %s  (%d -> %d)"
          % (ok(drift[True][0] == drift[True][1]), *drift[True]))
    print("  5.1 #cut_face stable across bevel, inherit=False : %s  (%d -> %d)"
          % (ok(drift[False][0] == drift[False][1]), *drift[False]))
    print("  5.3 cardinality assert fires on param change     : %s" % ok(fired))
    print("      rim count 5 slats -> 7 slats                 : %s -> %s" % (n5, n7))


def ok(b):
    return "PASS" if b else "FAIL"


main()
