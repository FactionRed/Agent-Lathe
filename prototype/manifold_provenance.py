"""
Does Manifold support DESIGN.md section 5.2 provenance tagging?

DESIGN.md section 11 commits to Manifold as the boolean kernel, because it
guarantees manifold output. The Blender prototype proved tagging works there,
but Blender is not the kernel the plan builds on. This closes that gap.

Manifold offers two candidate mechanisms, and they behave differently:
  A. vertex property channels   (set_properties / vert_properties)
  B. native mesh ID tracking    (reserve_ids / run_original_id / face_id)

Run:  python prototype/manifold_provenance.py
"""

import numpy as np
import manifold3d as m

TAG_NAMES = {}


def rule(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def tagged_cube(size, center, tag_id, tag_name):
    """A cube whose triangles all carry tag_id in run_original_id."""
    TAG_NAMES[tag_id] = tag_name
    base = m.Manifold.cube(size, True).translate(center)
    mesh = base.to_mesh()
    # to_mesh() hands back read-only views. nanobind requires writable arrays,
    # and ascontiguousarray is a no-op on them, so copy explicitly.
    tv = np.array(mesh.tri_verts, dtype=np.uint32, order="C", copy=True)
    vp = np.array(mesh.vert_properties, dtype=np.float32, order="C", copy=True)
    retagged = m.Mesh(
        vert_properties=vp,
        tri_verts=tv,
        run_index=np.array([0, tv.shape[0] * 3], dtype=np.uint32),
        run_original_id=np.array([tag_id], dtype=np.uint32),
    )
    return m.Manifold(retagged)


def census(man, label):
    """Triangles per ancestor tag -- the section 5.2 evidence."""
    mesh = man.to_mesh()
    run_idx = np.asarray(mesh.run_index, dtype=np.int64)
    run_ids = np.asarray(mesh.run_original_id, dtype=np.int64)
    out = {}
    for i, rid in enumerate(run_ids):
        tris = (run_idx[i + 1] - run_idx[i]) // 3
        name = TAG_NAMES.get(int(rid), "id:%d" % rid)
        out[name] = out.get(name, 0) + int(tris)
    print("  %-24s tris=%-4d verts=%-4d genus=%d"
          % (label, man.num_tri(), man.num_vert(), man.genus()))
    for k in sorted(out):
        print("      %-24s %4d tris" % (k, out[k]))
    return out


# ------------------------------------------------------------------ M1

def m1():
    ids = m.Manifold.reserve_ids(2)
    body = tagged_cube((0.6, 0.6, 0.6), (0, 0, 0), ids, "body")
    cut = tagged_cube((0.028, 0.66, 0.66), (0, 0, 0), ids + 1, "cut_face")
    census(body, "body before:")
    result = body - cut
    c = census(result, "after difference:")
    return ("body" in c and "cut_face" in c), c


# ------------------------------------------------------------------ M2

def m2():
    """Chained booleans: do all three ancestors stay distinguishable?"""
    ids = m.Manifold.reserve_ids(3)
    body = tagged_cube((0.6, 0.6, 0.6), (0, 0, 0), ids, "body")
    c1 = tagged_cube((0.028, 0.7, 0.7), (-0.2, 0, 0), ids + 1, "slot_A")
    c2 = tagged_cube((0.028, 0.7, 0.7), (0.2, 0, 0), ids + 2, "slot_B")
    r = (body - c1) - c2
    c = census(r, "body - A - B:")
    return len(c) == 3, c


# ------------------------------------------------------------------ M3

def m3():
    """The drift test. Blender's attribute interpolation silently widened
    faces(#cut_face) from 2 to 10. Do property channels do the same here,
    and does ID tracking avoid it?"""
    ids = m.Manifold.reserve_ids(2)
    body = tagged_cube((0.6, 0.6, 0.6), (0, 0, 0), ids, "body")
    cut = tagged_cube((0.2, 0.7, 0.7), (0.35, 0, 0), ids + 1, "cut_face")

    # -- mechanism A: property channel carrying the tag as a number
    body_p = body.set_properties(1, lambda pos, prop: [1.0])
    cut_p = cut.set_properties(1, lambda pos, prop: [2.0])
    res_p = body_p - cut_p
    vp = np.asarray(res_p.to_mesh().vert_properties, dtype=np.float64)
    chan = vp[:, 3] if vp.shape[1] > 3 else np.array([])
    uniq = sorted(set(np.round(chan, 4).tolist()))
    intermediate = [v for v in uniq if v not in (1.0, 2.0)]
    print("  A  property channel values present : %s" % uniq[:12])
    print("     values that are neither 1 nor 2 : %d  %s"
          % (len(intermediate), "<- interpolated, tag is now meaningless"
             if intermediate else "<- clean HERE, but see caveat"))
    print("     caveat: this test sets the channel uniformly per solid, so any"
          "\n             interpolation lands on an identical value and hides"
          "\n             itself. It does not clear property channels -- it only"
          "\n             fails to convict them. Prefer IDs on first principles.")

    # -- mechanism B: native ID tracking
    res_id = body - cut
    c = census(res_id, "  B  run_original_id:")
    ids_clean = set(c) <= {"body", "cut_face"}
    print("     only declared tags present      : %s" % ids_clean)

    return len(intermediate) > 0, ids_clean


# ------------------------------------------------------------------ M4

def m4():
    """face_id granularity: can a selector address one face of one original,
    the way seat.top_face() needs to?"""
    ids = m.Manifold.reserve_ids(2)
    body = tagged_cube((0.6, 0.6, 0.6), (0, 0, 0), ids, "body")
    cut = tagged_cube((0.2, 0.2, 0.7), (0.3, 0.3, 0), ids + 1, "cut_face")
    r = body - cut
    fid = np.asarray(r.to_mesh().face_id, dtype=np.int64)
    print("  triangles                 : %d" % r.num_tri())
    print("  distinct face_id values   : %d" % len(set(fid.tolist())))
    print("  tris per face_id (top 8)  : %s"
          % sorted(np.bincount(fid - fid.min()).tolist(), reverse=True)[:8])
    return len(set(fid.tolist()))


def m5():
    """The section 4 crate example, before and after the doc fix. The original
    cutter spanned size*1.1 in Z as well as Y, which severs the body."""
    size, slat_w, slat_n, rail = 0.6, 0.08, 5, 0.06
    for label, cz, cz_off in [("size*1.1 (as first written)", size * 1.1, 0),
                              ("size-rail*2 (fixed)", size - rail * 2, 0)]:
        body = m.Manifold.cube((size, size, size), True)
        span = size * 0.9
        for i in range(slat_n):
            x = -span / 2 + (i / (slat_n - 1.0)) * span
            body -= m.Manifold.cube(
                (slat_w * 0.35, size * 1.1, cz), True).translate((x, cz_off, 0))
        parts = len(body.decompose())
        print("  %-30s components=%-3d genus=%-3d tris=%-4d %s"
              % (label, parts, body.genus(), body.num_tri(),
                 "OK" if parts == 1 else "SEVERED - E108"))
    return parts


def ok(b):
    return "PASS" if b else "FAIL"


rule("M1  run_original_id survives a difference")
m1_ok, _ = m1()

rule("M2  three ancestors through chained booleans")
m2_ok, m2_c = m2()

rule("M3  drift: property channels vs native ID tracking")
prop_drifted, id_clean = m3()

rule("M4  face_id granularity")
n_faces = m4()

rule("M5  the doc's own crate example, checked")
crate_parts = m5()

rule("VERDICT")
print("  5.2 tags survive Manifold boolean (IDs)      : %s" % ok(m1_ok))
print("  5.2 chained ops keep ancestors distinct      : %s  (%d tags)"
      % (ok(m2_ok), len(m2_c)))
print("  5.1 property channel drifts on cut verts     : %s"
      % ("YES - do not use for tags" if prop_drifted else "no"))
print("  5.1 ID tracking stays exact, no drift        : %s" % ok(id_clean))
print("  5.1 face_id gives sub-original granularity   : %d distinct faces" % n_faces)
print("  7   fixed crate is one solid (E108 clean)    : %s" % ok(crate_parts == 1))
