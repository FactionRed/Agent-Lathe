"""
Budget rule: E107.

DESIGN.md section 4 declares `budget : { tris_lod0: 1500 }`. This is the rule
that reads it. The evidence includes the per-group distribution, because
"you are 528 tris over" is a fact and "the bevel on the lid holds 61% of your
triangles" is a repair.
"""

from __future__ import annotations

from ..engine import rule
from ..findings import ERROR, Raw, Repair
from ._common import num

# --------------------------------------------------------------------- E107


@rule(
    "E107",
    ERROR,
    "Triangle count over declared budget",
    needs=("emit.budget",),
    defaults={"tolerance_pct": 0.0, "max_report": 6},
)
def e107(ctx):
    """Triangles counted after fan triangulation, which is what an engine
    importer does to n-gons. A mesh can pass this rule as quads and fail it as
    triangles, so counting quads would be counting the wrong thing."""
    mesh = ctx.mesh
    budget = ctx.emit.budget_for_lod()
    if budget is None:
        ctx.not_applicable(
            "emit declares no triangle budget for LOD%d — add "
            "`budget : { tris_lod%d: n }`" % (ctx.emit.lod, ctx.emit.lod)
        )
    actual = mesh.triangle_count()
    allowed = budget * (1.0 + float(ctx.p("tolerance_pct")) / 100.0)
    if actual <= allowed:
        return

    over = actual - budget
    contributions = _by_group(mesh)
    rows = [
        "%-14s %6d tris  %s%% of total" % (name, tris, num(100.0 * tris / max(actual, 1)))
        for name, tris in contributions[: ctx.p("max_report")]
    ]
    yield ctx.finding(
        summary="Triangle count %d over budget %d (+%s%%)"
        % (actual, budget, num(100.0 * over / budget)),
        entity=mesh.name,
        where=[("budget", "tris_lod%d = %d" % (ctx.emit.lod, budget)),
               ("actual", "%d tris in %d faces" % (actual, mesh.num_faces)),
               ("heaviest", rows)],
        evidence="tris = %d, budget = %d, over by %d (%s%%); face sizes %s"
        % (
            actual,
            budget,
            over,
            num(100.0 * over / budget),
            ", ".join("%d-gon x%d" % (k, v) for k, v in mesh.face_size_histogram().items()),
        ),
        data={
            "tri_count": actual,
            "budget": budget,
            "over_by": over,
            "over_pct": 100.0 * over / budget,
            "by_group": [{"group": g, "tris": t} for g, t in contributions],
        },
        repair=Repair(
            "lod.decimate",
            (Raw(mesh.name),),
            {"target_tris": budget, "preserve": Raw("silhouette")},
            alternatives=(
                (
                    "reduce the heaviest part first: %s holds %s%% of the triangles"
                    % (contributions[0][0], num(100.0 * contributions[0][1] / actual))
                )
                if contributions
                else "raise the declared budget if the asset genuinely needs it",
                "emit with budget.tris_lod%d = %d if the budget is wrong"
                % (ctx.emit.lod, actual),
            ),
        ),
    )


def _by_group(mesh):
    """Triangles per named face group, heaviest first. Ungrouped faces are
    reported as one bucket rather than dropped."""
    tally = {}
    for fi, f in enumerate(mesh.faces):
        name = mesh.group_of_face(fi) or "<ungrouped>"
        tally[name] = tally.get(name, 0) + max(len(f) - 2, 0)
    return sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
