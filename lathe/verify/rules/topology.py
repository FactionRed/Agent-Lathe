"""
Topology rules: E101, E102, E103, E105, E108.

These are the rules DESIGN.md section 1 calls silent: "The script runs, exits 0,
and produces garbage." Every one of them is a comparison a senior artist makes
by reflex in review, and every one of them is arithmetic.

Note on E101: section 11 claims Manifold "eliminates E101 by construction".
That is true of Manifold's own boolean output and false of the pipeline, because
a mesh can acquire non-manifold geometry after the kernel is done with it --
imported geometry, a bevel implementation (Manifold ships none, see section 14),
a hand edit in Blender, an exporter that welds. The rule stays, and it is cheap.
"""

from __future__ import annotations

import numpy as np

from ..engine import rule
from ..findings import ERROR, Raw, Repair
from ._common import entity_for, face_locator, group_summary, loc, num, plural

# --------------------------------------------------------------------- E101


@rule(
    "E101",
    ERROR,
    "Non-manifold edges or vertices",
    needs=("topology",),
    defaults={"allow_boundary": False, "max_report": 8},
)
def e101(ctx):
    """Edges with more than two incident faces, open boundary edges, and
    vertices where two shells are pinched together (bowties)."""
    mesh, topo = ctx.mesh, ctx.topology

    # -- (a) edges shared by more than two faces
    if topo.nonmanifold_edges:
        worst = max(len(topo.edge_faces[e]) for e in topo.nonmanifold_edges)
        rows = []
        for e in topo.nonmanifold_edges[: ctx.p("max_report")]:
            faces = topo.edge_faces[e]
            rows.append(
                "%s shared by %d faces [%s]"
                % (loc(topo.edge_midpoint(e)), len(faces), group_summary(mesh, faces))
            )
        yield ctx.finding(
            summary="Non-manifold edges: %s with more than 2 faces"
            % plural(len(topo.nonmanifold_edges), "edge"),
            entity=entity_for(mesh),
            where=[("edges", rows)],
            evidence="%s non-manifold, worst has %d incident faces "
            "(manifold requires exactly 2)"
            % (plural(len(topo.nonmanifold_edges), "edge"), worst),
            data={
                "nonmanifold_edge_count": len(topo.nonmanifold_edges),
                "worst_incident_faces": worst,
                "locations": [list(map(float, topo.edge_midpoint(e)))
                              for e in topo.nonmanifold_edges],
            },
            repair=Repair(
                "mesh.weld",
                (Raw(mesh.name),),
                {"tolerance": Raw("1e-5")},
                alternatives=(
                    "mesh.delete_faces(%s, at = [%s])  — when a stray face is doubled"
                    % (mesh.name, loc(topo.edge_midpoint(topo.nonmanifold_edges[0]))),
                    "rebuild the op through the Manifold kernel: its boolean output is "
                    "manifold by construction (DESIGN.md §11)",
                ),
            ),
        )

    # -- (b) bowtie vertices: one point, two fans
    if topo.bowtie_verts:
        wv = mesh.world_verts()
        rows = [
            "%s joins %d disconnected face fans" % (loc(wv[v]), topo.fan_count(v))
            for v in topo.bowtie_verts[: ctx.p("max_report")]
        ]
        yield ctx.finding(
            summary="Non-manifold vertices: %s"
            % plural(len(topo.bowtie_verts), "bowtie vertex", "es"),
            entity=entity_for(mesh),
            where=[("verts", rows)],
            evidence="%s whose incident faces form more than one fan; "
            "every edge at those vertices is otherwise manifold"
            % plural(len(topo.bowtie_verts), "vertex", "es"),
            data={
                "bowtie_vert_count": len(topo.bowtie_verts),
                "locations": [list(map(float, wv[v])) for v in topo.bowtie_verts],
            },
            repair=Repair(
                "mesh.split_vertex",
                (Raw(mesh.name),),
                {"at": [Raw(loc(wv[v])) for v in topo.bowtie_verts[:4]]},
                alternatives=(
                    "solid.union(parts) — a bowtie usually means two shells were "
                    "merged by proximity instead of by a boolean",
                ),
            ),
        )

    # -- (c) open boundary. An error for a solid prop, legitimate for cards.
    if topo.boundary_edges and not ctx.p("allow_boundary"):
        rows = [
            "%s [%s]" % (loc(topo.edge_midpoint(e)), group_summary(mesh, topo.edge_faces[e]))
            for e in topo.boundary_edges[: ctx.p("max_report")]
        ]
        open_comps = [c.name for c in topo.components if not c.closed]
        yield ctx.finding(
            summary="Open shell: %s with a single face"
            % plural(len(topo.boundary_edges), "boundary edge"),
            entity=entity_for(mesh),
            where=[("edges", rows), ("shells", ", ".join(open_comps) or "—")],
            evidence="%s across %d of %d shells; profile requires watertight "
            "geometry (E101.allow_boundary = false)"
            % (
                plural(len(topo.boundary_edges), "boundary edge"),
                len(open_comps),
                len(topo.components),
            ),
            data={
                "boundary_edge_count": len(topo.boundary_edges),
                "open_components": open_comps,
                "locations": [list(map(float, topo.edge_midpoint(e)))
                              for e in topo.boundary_edges],
            },
            repair=Repair(
                "mesh.fill_holes",
                (Raw(mesh.name),),
                {"max_boundary_edges": len(topo.boundary_edges)},
                alternatives=(
                    "set E101.allow_boundary = true in the profile when the asset is "
                    "deliberately an open shell (foliage cards, decals)",
                ),
            ),
        )


# --------------------------------------------------------------------- E102


@rule(
    "E102",
    ERROR,
    "Face size outside the profile's range",
    needs=("topology",),
    defaults={"max_face_verts": 4, "min_face_verts": 3, "max_report": 8},
)
def e102(ctx):
    """Faces the profile does not allow, in either direction.

    The upper bound is the classic n-gon check: engine importers triangulate
    n-gons with an algorithm nobody chose, so shading and the triangle count both
    stop being predictable at emit.

    The lower bound exists because a maximum alone cannot say "quads". `max = 3`
    happens to mean triangles-only, so `mobile_vr` reads correctly by accident,
    but `max = 4` admits triangles as readily as quads -- and a triangle *is* the
    non-quad that matters. `hard_surface_subdiv` documents "a cage is quads or it
    is a bug", and with only a maximum it certified a fully triangulated cage as
    a PASS: not a skip, an affirmative pass on the exact input its own comment
    calls a bug. Catmull-Clark makes every triangle corner an extraordinary
    vertex, which is a visible pinch on the limit surface.

    Default 3, so no profile that does not ask for a lower bound changes.
    """
    mesh = ctx.mesh
    limit = int(ctx.p("max_face_verts"))
    floor = int(ctx.p("min_face_verts"))
    over = [i for i, f in enumerate(mesh.faces) if len(f) > limit]
    under = [i for i, f in enumerate(mesh.faces) if len(f) < floor]
    offenders = sorted(over + under)
    if not offenders:
        return

    hist = {n: c for n, c in mesh.face_size_histogram().items()
            if n > limit or n < floor}
    # Rank by distance from the allowed range, so the report leads with the worst.
    def _excess(i):
        n = len(mesh.faces[i])
        return max(n - limit, floor - n)

    worst = max(offenders, key=_excess)
    rows = [
        "%d-gon %s" % (len(mesh.faces[i]), face_locator(mesh, i))
        for i in sorted(offenders, key=lambda i: -_excess(i))[: ctx.p("max_report")]
    ]
    if over and under:
        what = "%s outside %d-%d verts" % (plural(len(offenders), "face"), floor, limit)
    elif over:
        what = "%s over %d verts" % (plural(len(over), "face"), limit)
    else:
        what = "%s under %d verts" % (plural(len(under), "face"), floor)

    # A face with too few sides is fixed by joining, not by cutting.
    if under and not over:
        repair = Repair(
            "mesh.quadrangulate",
            (Raw(mesh.name),),
            {"faces": Raw("faces(verts < %d)" % floor)},
            alternatives=(
                "relax E102.min_face_verts to 3 in the profile when a triangulated "
                "mesh is genuinely the deliverable -- but not for a subdivision "
                "cage, where every triangle is an extraordinary vertex",
            ),
        )
    else:
        repair = Repair(
            "mesh.triangulate",
            (Raw(mesh.name),),
            {"faces": Raw("faces(verts > %d)" % limit), "method": Raw("beauty")},
            alternatives=(
                "mesh.quadrangulate(%s, faces = faces(verts > %d))  — when the target "
                "wants quads, not tris" % (mesh.name, limit),
            ),
        )

    yield ctx.finding(
        summary="Face size outside the profile's range: %s" % what,
        entity=entity_for(mesh, offenders),
        where=[("faces", rows)],
        evidence="%d/%d faces outside %d-%d verts (%s); furthest is a %d-gon of %s m²"
        % (
            len(offenders),
            mesh.num_faces,
            floor,
            limit,
            ", ".join("%d-gon x%d" % (k, v) for k, v in sorted(hist.items())),
            len(mesh.faces[worst]),
            num(mesh.face_area(worst)),
        ),
        data={
            "ngon_count": len(over),
            "undersized_count": len(under),
            "face_count": mesh.num_faces,
            "max_face_verts": limit,
            "min_face_verts": floor,
            "size_histogram": {str(k): v for k, v in hist.items()},
            "locations": [list(map(float, mesh.face_centroid(i, world=True)))
                          for i in offenders],
        },
        repair=repair,
    )


# --------------------------------------------------------------------- E103


@rule(
    "E103",
    ERROR,
    "Inverted / inconsistent face normals",
    needs=("topology",),
    defaults={"max_report": 8},
)
def e103(ctx):
    """Two separate defects with one code, because they have one cause and one
    repair: neighbouring faces wound in opposite directions (inconsistent), and
    a closed shell wound inward (inverted)."""
    mesh, topo = ctx.mesh, ctx.topology

    if topo.inconsistent_edges:
        rows = []
        for e in topo.inconsistent_edges[: ctx.p("max_report")]:
            faces = topo.edge_faces[e]
            rows.append(
                "%s between [%s]" % (loc(topo.edge_midpoint(e)), group_summary(mesh, faces))
            )
        affected = sorted({fi for e in topo.inconsistent_edges for fi in topo.edge_faces[e]})
        yield ctx.finding(
            summary="Inconsistent face normals: %s wound the same way by both faces"
            % plural(len(topo.inconsistent_edges), "edge"),
            entity=entity_for(mesh, affected),
            where=[("edges", rows), ("faces", group_summary(mesh, affected))],
            evidence="%d of %d two-face edges are traversed in the same direction by "
            "both neighbours; %s involved"
            % (
                len(topo.inconsistent_edges),
                len(topo.edge_faces),
                plural(len(affected), "face"),
            ),
            data={
                "inconsistent_edge_count": len(topo.inconsistent_edges),
                "edge_count": len(topo.edge_faces),
                "affected_faces": affected,
                "locations": [list(map(float, topo.edge_midpoint(e)))
                              for e in topo.inconsistent_edges],
            },
            repair=Repair(
                "mesh.recalculate_normals",
                (Raw(mesh.name),),
                {"outside": True},
                alternatives=(
                    "mesh.flip_normals(%s, faces = faces(normal_inconsistent))  — when "
                    "only a patch is wrong and the rest is authored" % mesh.name,
                ),
            ),
        )

    for comp in topo.components:
        if not (comp.closed and comp.consistent):
            continue  # inside/outside is undefined until winding is consistent
        if comp.signed_volume >= 0:
            continue
        yield ctx.finding(
            summary="Inverted face normals: %s is wound inward" % comp.name,
            entity="%s.%s" % (mesh.name, comp.name),
            where=[
                ("shell", "%s, %d faces, %d tris" % (comp.name, len(comp.faces), comp.tri_count)),
                ("extents", "%s m" % " × ".join(num(e) for e in comp.extents)),
            ],
            evidence="signed volume = %s m³ (negative means the shell faces inward); "
            "shell is closed and consistently wound, so this is a whole-shell flip"
            % num(comp.signed_volume),
            data={
                "component": comp.name,
                "signed_volume_m3": comp.signed_volume,
                "tri_count": comp.tri_count,
            },
            repair=Repair(
                "mesh.flip_normals",
                (Raw(mesh.name),),
                {"faces": Raw("faces(component = %s)" % comp.name)},
                alternatives=(
                    "mesh.recalculate_normals(%s, outside = true)" % mesh.name,
                ),
            ),
        )


# --------------------------------------------------------------------- E105


@rule(
    "E105",
    ERROR,
    "Loose vertices / edges, zero-area faces",
    needs=("topology",),
    defaults={"min_face_area": 1e-9, "max_report": 8},
)
def e105(ctx):
    """Geometry that contributes nothing and breaks things downstream: loose
    verts inflate the bbox, wire edges survive FBX export as renderable
    curves, zero-area faces produce NaN tangents in a normal bake."""
    mesh, topo = ctx.mesh, ctx.topology
    min_area = float(ctx.p("min_face_area"))
    degenerate = topo.degenerate_faces(min_area)
    if not (topo.loose_verts or topo.loose_wire_edges or degenerate):
        return

    wv = mesh.world_verts()
    rows = []
    for v in topo.loose_verts[: ctx.p("max_report")]:
        rows.append("loose vert at %s" % loc(wv[v]))
    for e in topo.loose_wire_edges[: ctx.p("max_report")]:
        rows.append(
            "wire edge %s → %s (length %s m)"
            % (loc(wv[e[0]]), loc(wv[e[1]]), num(float(np.linalg.norm(wv[e[1]] - wv[e[0]]))))
        )
    for fi, area, reason in degenerate[: ctx.p("max_report")]:
        rows.append(
            "degenerate face %s — %s (area %s m²)"
            % (face_locator(mesh, fi), reason, num(area))
        )

    zero_area = [d for d in degenerate if "zero area" in d[2]]
    malformed = [d for d in degenerate if "zero area" not in d[2]]
    parts = []
    if topo.loose_verts:
        parts.append(plural(len(topo.loose_verts), "loose vert"))
    if topo.loose_wire_edges:
        parts.append(plural(len(topo.loose_wire_edges), "wire edge"))
    if zero_area:
        parts.append(
            "%s under %s m²" % (plural(len(zero_area), "face"), num(min_area))
        )
    if malformed:
        parts.append(
            "%s repeating a vertex" % plural(len(malformed), "face")
        )

    # The primary repair has to be the one that fixes the majority of what was
    # found. `delete_loose(verts = false, edges = false)` is a no-op, and an
    # agent that runs a no-op repair reports the finding fixed.
    loose = bool(topo.loose_verts or topo.loose_wire_edges)
    dissolve = Repair(
        "mesh.dissolve_degenerate",
        (Raw(mesh.name),),
        {"threshold": Raw(num(min_area))},
        alternatives=(
            ("mesh.delete_loose(%s, verts = true, edges = true)" % mesh.name,)
            if loose
            else ()
        ),
    )
    if not loose:
        repair = dissolve
    else:
        repair = Repair(
            "mesh.delete_loose",
            (Raw(mesh.name),),
            {"verts": bool(topo.loose_verts), "edges": bool(topo.loose_wire_edges)},
            alternatives=(
                ("mesh.dissolve_degenerate(%s, threshold = %s)" % (mesh.name, num(min_area)),)
                if degenerate
                else ()
            ),
            note=(
                "delete_loose does not touch the zero-area faces; they need the dissolve"
                if degenerate
                else None
            ),
        )

    yield ctx.finding(
        summary="Loose or degenerate geometry: " + ", ".join(parts),
        entity=entity_for(mesh, [fi for fi, _, _ in degenerate]),
        where=[("items", rows)],
        evidence="%s; mesh has %d verts, %d faces%s"
        % (
            "; ".join(parts),
            mesh.num_verts,
            mesh.num_faces,
            "; smallest offending area %s m²" % num(min(a for _, a, _ in degenerate))
            if degenerate
            else "",
        ),
        data={
            "loose_vert_count": len(topo.loose_verts),
            "wire_edge_count": len(topo.loose_wire_edges),
            "zero_area_face_count": len(zero_area),
            "malformed_face_count": len(malformed),
            "min_face_area": min_area,
            "loose_vert_locations": [list(map(float, wv[v])) for v in topo.loose_verts],
            "degenerate_faces": [
                {
                    "location": list(map(float, mesh.face_centroid(fi, world=True))),
                    "area_m2": a,
                    "reason": r,
                }
                for fi, a, r in degenerate
            ],
        },
        repair=repair,
    )


# --------------------------------------------------------------------- E108


@rule(
    "E108",
    ERROR,
    "Component count differs from the declared expectation",
    needs=("emit.components",),
    defaults={"debris_ratio": 0.8},
)
def e108(ctx):
    """The rule that exists because DESIGN.md's own worked example was broken.

    prototype/README.md: the section 4 crate used cutters spanning `size * 1.1`
    in Z as well as Y, which severs the body instead of slotting it. Six loose
    slabs, no error, nothing in the program said otherwise. This is the rule
    that says otherwise.
    """
    mesh, topo = ctx.mesh, ctx.topology
    declared = ctx.emit.components
    if declared is None:
        ctx.not_applicable(
            "emit declares no component count — add `assert components(x) == n` "
            "(DESIGN.md §5.3)"
        )
    actual = len(topo.components)
    if actual == declared:
        return

    total_tris = sum(c.tri_count for c in topo.components) or 1
    biggest = max(topo.components, key=lambda c: c.tri_count)
    share = biggest.tri_count / total_tris
    rows = [
        "%s  %d tris  %s m³  extents %s m"
        % (
            c.name,
            c.tri_count,
            num(abs(c.signed_volume)),
            " × ".join(num(e) for e in c.extents),
        )
        for c in topo.components
    ]

    if actual > declared and share >= float(ctx.p("debris_ratio")):
        repair = Repair(
            "solid.keep_largest_component",
            (Raw(mesh.name),),
            alternatives=(
                "solid.union(components(%s))  — when every piece belongs to the asset"
                % mesh.name,
            ),
            note="%s holds %s%% of the triangles; the rest look like boolean debris"
            % (biggest.name, num(share * 100.0)),
        )
    elif actual > declared:
        repair = Repair(
            "assert",
            (Raw("components(%s) == %d" % (mesh.name, declared)),),
            statement=True,
            alternatives=(
                "clamp the cutter extent: a cutter spanning the full body severs it. "
                "DESIGN.md §4 uses `size - rail * 2`, not `size * 1.1`",
            ),
            note="the pieces are of comparable size, so this is a severed body, "
            "not debris",
        )
    else:
        repair = Repair(
            "assert",
            (Raw("components(%s) == %d" % (mesh.name, actual)),),
            statement=True,
            alternatives=(
                "solid.split(%s) — when the parts were meant to stay separate"
                % mesh.name,
            ),
            note="fewer components than declared: an intended cut did not go through",
        )

    yield ctx.finding(
        summary="Component count %d, declared %d" % (actual, declared),
        entity=entity_for(mesh),
        where=[("shells", rows)],
        evidence="components = %d, declared = %d; %d tris split %s; largest holds "
        "%s%% of the triangles"
        % (
            actual,
            declared,
            total_tris,
            "/".join(str(c.tri_count) for c in topo.components[:8]),
            num(share * 100.0),
        ),
        data={
            "components": actual,
            "declared": declared,
            "tri_counts": [c.tri_count for c in topo.components],
            "largest_share": share,
            "extents": [list(map(float, c.extents)) for c in topo.components],
        },
        repair=repair,
    )
