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

THE CHOICE OF OBJECTS. Only objects that belong to THIS beat: the pool members the beat's own words
refer to as whole words and do not negate, plus the object the batched author ranked for the beat from
its intent. The kernel's positional pick (the odometer tier, chosen by where the beat sits in the episode
and unrelated to what it says) never gets a region: reinforcing it would make an off-topic picture more
sure of itself. The kernel's own object leads when it qualifies, so the region reinforces what the global
prompt already says. At most three. A bookend beat has no dialogue and gets no regions. A beat with no
qualifying object gets none, which leaves its picture byte-identical to the base lane.

STRICTER THAN THE KERNEL'S OWN MENTION TEST, on purpose. ``_beat_mentions_object`` matches by substring
and its negation cues omit the bare "no", "not", "without" and "don't" (a gap that predates this
module), which is tolerable for ONE kernel noun but would turn "during" into a mention of "ring" and
"there is no ledger" into a region here. A region is a stronger claim than a kernel clause, so this
module also requires a whole-word mention and checks the few words before it for a bare negator.

MASKS ARE GENERATED, NEVER PAINTED. One object: a soft centred ellipse. Two: left and right halves that
overlap at the seam. Three: thirds. The recipe this technique comes from drives its regions from a mask
video; a static layout is enough for a first arm, and drifting masks are a later refinement.

Cold-import clean: numpy and the author module are resolved inside the functions.
"""
from __future__ import annotations

import hashlib
import re

#: The most regions a beat gets. Three is the recipe's own count and the most a 512x288 frame can hold
#: without each region becoming a sliver.
SCENE_REGION_MAX_OBJECTS = 3

#: Edge softness of every mask, as a fraction of the frame width. A hard edge draws a visible seam.
SCENE_REGION_FEATHER = 0.12

#: Layout names, by object count. Stamped into the receipt.
SCENE_REGION_LAYOUTS = {1: "centre", 2: "halves", 3: "thirds"}

_CHARACTER_ROLE = "character_video"

#: Kernel tiers whose object belongs to this beat. The positional tier (``key_object``) does not.
_BEAT_OWNED_SOURCES = frozenset({"key_object_in_beat", "authored_subject"})

_WORD = re.compile(r"[a-z0-9']+")
_SENTENCE_END = re.compile(r"[.!?;]+")

#: Words that, shortly before a mention, make it a mention of something ABSENT.
_NEGATORS = frozenset({
    "no", "not", "without", "never", "nobody", "nothing", "neither", "nor", "none", "instead",
    "rather", "hardly",
    "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt", "weren't", "werent", "don't", "dont",
    "doesn't", "doesnt", "didn't", "didnt", "can't", "cant", "won't", "wont", "couldn't",
    "couldnt", "wouldn't", "wouldnt", "haven't", "havent", "hasn't", "hasnt",
})

#: How many words before a mention a negator may sit and still reach it.
_NEGATION_WINDOW = 4


def _head_word(obj) -> str:
    """The object's most specific word: its last long word, as the kernel's own mention test reads it."""
    words = [w for w in _WORD.findall(str(obj or "").lower()) if w]
    long_words = [w for w in words if len(w) > 3]
    return (long_words or words or [""])[-1]


def _whole_word_stance(beat_text, obj) -> str:
    """``"clean"``, ``"negated"`` or ``"absent"``: how the beat treats ``obj`` as a WHOLE WORD.

    ``clean`` needs one whole-word mention with no negator in the few words before it, within its own
    sentence. ``negated`` means every whole-word mention is negated. ``absent`` means the beat never
    says the word (a substring such as "ring" inside "during" is not a mention).
    """
    head = _head_word(obj)
    if not head:
        return "absent"
    forms = {head, head + "s", head + "es"}
    mentioned = False
    for sentence in _SENTENCE_END.split(str(beat_text or "").lower()):
        tokens = _WORD.findall(sentence)
        for index, token in enumerate(tokens):
            if token not in forms:
                continue
            mentioned = True
            if not any(t in _NEGATORS for t in tokens[max(0, index - _NEGATION_WINDOW):index]):
                return "clean"
    return "negated" if mentioned else "absent"


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

    referenced = [obj for obj in pool
                  if _gsa._beat_mentions_object(beat_text, obj)
                  and _whole_word_stance(beat_text, obj) == "clean"]
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
    if (kernel_object and kernel_source in _BEAT_OWNED_SOURCES
            and _gsa._beat_object_stance(beat_text, kernel_object) != "negated"
            and _whole_word_stance(beat_text, kernel_object) != "negated"
            and (kernel_source != "key_object_in_beat"
                 or _whole_word_stance(beat_text, kernel_object) == "clean")):
        ordered.append(kernel_object)
    for obj in referenced:
        if obj not in ordered:
            ordered.append(obj)
    # One region per thing: two pool entries that end in the same word ("ledger", "handwritten
    # ledger") are one object, and the first stays.
    unique, heads = [], set()
    for obj in ordered:
        head = _head_word(obj)
        if head in heads:
            continue
        heads.add(head)
        unique.append(obj)
    ordered = unique[:SCENE_REGION_MAX_OBJECTS]
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
