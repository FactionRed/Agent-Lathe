"""
Texture rule: W203.

Non-power-of-two dimensions cost real memory and real frames: no block
compression on some targets, no mip chain on others, and a silent rescale on
import that quietly desynchronises the bake from the UVs.
"""

from __future__ import annotations

from ..engine import rule
from ..findings import WARN, Raw, Repair

# --------------------------------------------------------------------- W203


@rule(
    "W203",
    WARN,
    "Non-power-of-two texture dimensions",
    needs=("emit.textures",),
    defaults={"require_square": False},
)
def w203(ctx):
    """Checked per declared texture, so the finding names the texture rather
    than the mesh -- the offending entity is the map, not the model."""
    textures = ctx.emit.textures
    if not textures:
        ctx.not_applicable("emit declares no textures")

    for tex in textures:
        bad_w, bad_h = not _is_pot(tex.width), not _is_pot(tex.height)
        non_square = ctx.p("require_square") and tex.width != tex.height
        if not (bad_w or bad_h or non_square):
            continue
        near_w, near_h = _nearest_pot(tex.width), _nearest_pot(tex.height)
        axes = ", ".join(
            n for n, bad in (("width", bad_w), ("height", bad_h)) if bad
        ) or "aspect"
        yield ctx.finding(
            summary="Non-power-of-two texture: %s is %d × %d" % (tex.name, tex.width, tex.height),
            entity="%s/%s" % (ctx.name, tex.name),
            where=[
                ("texture", "%s (%s channel)" % (tex.name, tex.channel)),
                ("size", "%d × %d  →  nearest power of two %d × %d"
                 % (tex.width, tex.height, near_w, near_h)),
            ],
            evidence="%s not a power of two (%d × %d); nearest %d × %d, "
            "%s%% pixel-count change"
            % (
                axes,
                tex.width,
                tex.height,
                near_w,
                near_h,
                _pct(tex.width * tex.height, near_w * near_h),
            ),
            data={
                "texture": tex.name,
                "channel": tex.channel,
                "width": tex.width,
                "height": tex.height,
                "nearest_pot": [near_w, near_h],
                "non_square": bool(non_square),
            },
            repair=Repair(
                "texture.resize",
                (Raw(tex.name), near_w, near_h),
                {"filter": Raw("lanczos")},
                alternatives=(
                    "re-bake at %d × %d instead of resampling, which keeps the bake "
                    "aligned to the uvs" % (near_w, near_h),
                ),
            ),
        )


def _is_pot(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


def _nearest_pot(v: int) -> int:
    if v < 1:
        return 1
    lower = 1 << (v.bit_length() - 1)
    upper = lower << 1
    return lower if (v - lower) <= (upper - v) else upper


def _pct(a: int, b: int) -> str:
    if a == 0:
        return "0"
    return "%+.1f" % (100.0 * (b - a) / a)
