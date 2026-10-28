"""Scene regions for the Ghost AnimateDiff lanes: the beat's own objects, each in its own part of the frame.

THE PROBLEM THIS ANSWERS. A Ghost beat is drawn from ONE short prompt (320 characters at most) in which
the style takes the front and the back and the beat's object gets one clause in the middle
(``ghost_signal_prompt.compose_ghost_prompt_v3``). The dialogue often names a lantern, a ledger, a
harbour, and the picture shows the console or a generic room. Regional text conditioning gives each
named object its own text and its own region, so it appears instead of competing inside one prompt.

THE RULING THIS KEEPS. ``resolve_crux_kernel`` ranks the episode's own ``key_objects`` by which one the
beat's words refer to and never lifts a noun out of the dialogue (operator ruling 2026-09-03: "beat
reference RANKS candidates; it is NOT the source"). This module follows it exactly: the pool is
``ghost_subject_candidates(meta)`` and nothing else, the beat text only decides which pool members the
beat refers to, and a noun the beat negates is never a region.

THE CHOICE OF OBJECTS. The kernel's own object first (so the region reinforces what the global prompt
already says), then the other pool members the beat cleanly refers to, at most three. A bookend beat has
no dialogue and gets no regions. A beat whose dialogue names no pool object and whose kernel is not a
pool object gets none either, which leaves its picture byte-identical to the base lane.

MASKS ARE GENERATED, NEVER PAINTED. One object: a soft centred ellipse. Two: left and right halves that
overlap at the seam. Three: thirds. The recipe this technique comes from drives its regions from a mask
video; a static layout is enough for a first arm, and drifting masks are a later refinement.

Cold-import clean: numpy and the author module are resolved inside the functions.
"""
from __future__ import annotations

import hashlib

#: The most regions a beat gets. Three is the recipe's own count and the most a 512x288 frame can hold
#: without each region becoming a sliver.
SCENE_REGION_MAX_OBJECTS = 3

#: Edge softness of every mask, as a fraction of the frame width. A hard edge draws a visible seam.
SCENE_REGION_FEATHER = 0.12

#: Layout names, by object count. Stamped into the receipt.
SCENE_REGION_LAYOUTS = {1: "centre", 2: "halves", 3: "thirds"}

_CHARACTER_ROLE = "character_video"


def resolve_scene_objects(meta, *, ordinal=0, role="", mode="", beat_text="",
                          authored_subject="") -> dict:
    """``{"objects": tuple, "source": str}`` -- the pool objects this beat gets regions for.

    PURE. Never raises, never returns a noun that is not a member of the episode's own key-object pool.
    ``source`` is the kernel tier that supplied the first object (``key_object_in_beat``,
    ``authored_subject``, ``key_object``) or ``dialogue`` when only the beat's references supplied
    them, or ``none`` with an empty tuple.
    """
    from . import ghost_signal_author as _gsa

    empty = {"objects": (), "source": "none"}
    if str(role or "") != _CHARACTER_ROLE:
        return empty
    pool = list(_gsa.ghost_subject_candidates(meta))
    if not pool:
        return empty

    referenced = [obj for obj in pool if _gsa._beat_mentions_object(beat_text, obj)]
    kernel, kernel_source = _gsa.resolve_crux_kernel(
        meta, ordinal=ordinal, role=role, mode=mode, beat_text=beat_text,
        authored_subject=authored_subject)

    # The kernel is "<subject>" or "<subject> <connector> <place>"; the subject is the longest pool
    # member the kernel starts with, so a pool of ["pen", "pen case"] cannot confuse the match.
    kernel_text = " ".join(str(kernel or "").split())
    kernel_object = ""
    for obj in sorted(pool, key=len, reverse=True):
        spelled = " ".join(str(obj).split())
        if kernel_text == spelled or kernel_text.startswith(spelled + " "):
            kernel_object = obj
            break

    ordered = []
    if kernel_object and _gsa._beat_object_stance(beat_text, kernel_object) != "negated":
        ordered.append(kernel_object)
    for obj in referenced:
        if obj not in ordered:
            ordered.append(obj)
    ordered = ordered[:SCENE_REGION_MAX_OBJECTS]
    if not ordered:
        return empty
    source = kernel_source if ordered[0] == kernel_object else "dialogue"
    return {"objects": tuple(ordered), "source": str(source)}


def region_layout_name(count) -> str:
    """The layout name for ``count`` objects, or raise ``ValueError`` for a count outside 1..3."""
    try:
        return SCENE_REGION_LAYOUTS[int(count)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "scene regions support 1 to %d objects, got %r"
            % (SCENE_REGION_MAX_OBJECTS, count)) from None


def _smoothstep(edge0, edge1, x):
    import numpy as np
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def region_masks(count, width, height, feather=SCENE_REGION_FEATHER) -> list:
    """``count`` float32 masks of shape (height, width), values in [0, 1], one per object.

    Deterministic: the same arguments give the same bytes on every host.
    """
    import numpy as np

    layout = region_layout_name(count)
    width = int(width)
    height = int(height)
    if width < 8 or height < 8:
        raise ValueError("scene region masks need at least an 8x8 frame, got %dx%d" % (width, height))
    feather = max(float(feather), 1e-3)
    xs = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
    ys = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    grid_x, grid_y = np.meshgrid(xs, ys)

    if layout == "centre":
        # An ellipse about the middle of the frame, a little below centre where a held object sits.
        cx, cy, rx, ry = 0.5, 0.55, 0.30, 0.40
        dist = np.sqrt(((grid_x - cx) / rx) ** 2 + ((grid_y - cy) / ry) ** 2)
        edge = feather / rx
        mask = 1.0 - _smoothstep(1.0 - edge, 1.0 + edge, dist)
        masks = [mask]
    elif layout == "halves":
        left = 1.0 - _smoothstep(0.5 - feather, 0.5 + feather, grid_x)
        masks = [left, 1.0 - left]
    else:
        masks = []
        for index in range(3):
            lo = index / 3.0
            hi = (index + 1) / 3.0
            rise = _smoothstep(lo - feather, lo + feather, grid_x) if index > 0 else np.ones_like(grid_x)
            fall = 1.0 - _smoothstep(hi - feather, hi + feather, grid_x) if index < 2 else np.ones_like(grid_x)
            masks.append(rise * fall)
    return [np.clip(m, 0.0, 1.0).astype(np.float32) for m in masks]


def mask_digest(masks) -> str:
    """An 8-hex digest of the masks, for the receipt. Quantised so it is stable across hosts."""
    import numpy as np
    digest = hashlib.sha256()
    for mask in masks:
        digest.update(np.round(np.asarray(mask, dtype=np.float32) * 255.0).astype(np.uint8).tobytes())
    return digest.hexdigest()[:8]


__all__ = [
    "SCENE_REGION_MAX_OBJECTS", "SCENE_REGION_FEATHER", "SCENE_REGION_LAYOUTS",
    "resolve_scene_objects", "region_layout_name", "region_masks", "mask_digest",
]
