"""The PLATE PROMPT of the still-in lab peer (campaign item 2, 2026-09-02).

One subject-free picture of the episode's WORLD in the pack's MEDIUM, for the
in-family plate the peer samples before the motion pass. Composed by
``render_driver`` (which holds the resolved style and the ledger meta) onto the
declared request field ``plate_prompt``; the engine never sees a pack.

THE HEAD IS THE POINT. The video prompt on this lane carries the style as a
two-word cue (``compact_style_cue``) -- the defect the whole campaign is about --
so this composer bypasses that authority entirely and leads with the pack's
FULL ``positive_tail``. That clause is PROTECTED: it is never trimmed, and a
pack whose positive_tail alone exceeds the author target is refused by name
rather than quietly truncated. After it, in order: the pack's ``plate_look``
(the world-set geometry), the story's setting (top two terms), the
still-profile era tail (palette, lighting, atmosphere, capped at 120 chars by
its own helper), the pack's ``image_grade_tail``, the pack's ``era_tail``.
Those five are DROPPABLE, last first, until the whole prompt sits inside ONE
measured CLIP window and the lane's 320-char ceiling; ``plate_look`` goes
last, and on the two longest packs (recur_frac, video_art) the char ceiling is
what bites -- the receipt's ``plate_prompt`` shows exactly what survived.

NEVER carried: ``scene_instruction_look`` (an instruction sentence), the
broadcast tail, any motion register (camera words; some carry damping words),
open subjects / announcer / portrait looks (subject and identity), the leaf,
the mode law, the no-text clause (this lane puts exclusions in the negative).
Measured with the installed SD1 tokenizer through the same
``resolve_token_measure`` seam the video prompt uses; under ``OTR_TEST_MODE``
with no tokenizer the gate is skipped OUT LOUD, as there.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Optional

_LOG = logging.getLogger("OTR.video.ghost_plate_prompt")

#: The two protected clauses' author target and the one-window ceiling, from
#: the lane's own author module (single-sourced there).
#: Droppable clauses, LAST FIRST: the pack era tail, the grade tail, the
#: still-profile era tail (palette / lighting / atmosphere), the setting, and
#: only then the pack's plate_look. The positive_tail is never dropped.
PLATE_DROP_ORDER = ("era_tail", "image_grade_tail", "still_era_tail", "setting",
                    "plate_look")


def _attr(obj: Any, name: str) -> str:
    val = getattr(obj, name, None)
    if val is None and isinstance(obj, dict):
        val = obj.get(name)
    return str(val or "").strip().rstrip(",.").strip()


def _join(parts) -> str:
    return ", ".join(p for p in parts if p)


def compose_plate_prompt(vstyle: Any, ledger_meta: Any, *,
                         token_measure_fn: Optional[Callable] = None) -> dict:
    """``{"positive", "head", "dropped", "clip_tokens", "clip_windows",
    "sha8", "measured"}`` for the plate of ``vstyle`` in the world of
    ``ledger_meta``. Pure; raises ``GhostBudgetError`` (named) when the
    protected head alone does not fit the author target."""
    from . import ghost_signal_author as _gsa
    try:
        from ..otr_meta_brief_image_prompt import _read_setting  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from otr_meta_brief_image_prompt import _read_setting  # type: ignore
    try:
        from .._otr_story_brief_helpers import get_era_tail  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_story_brief_helpers import get_era_tail  # type: ignore

    meta = ledger_meta if isinstance(ledger_meta, dict) else {}
    # THE PROTECTED HEAD is the pack's positive_tail -- the medium. plate_look
    # (the world-set geometry) rides next and is the LAST clause to drop:
    # recur_frac's positive_tail + plate_look alone are 356 chars, so a head
    # that protected both would refuse a registered style by construction.
    head = _attr(vstyle, "positive_tail")
    if not head:
        raise _gsa.GhostBudgetError(
            "plate prompt: the style pack carries no positive_tail -- there is "
            "no medium to paint the world in")

    droppable = {
        "plate_look": _attr(vstyle, "plate_look"),
        "setting": _read_setting(meta),
        "still_era_tail": str(get_era_tail(meta, profile="still",
                                           style=vstyle) or "").strip(),
        "image_grade_tail": _attr(vstyle, "image_grade_tail"),
        "era_tail": _attr(vstyle, "era_tail"),
    }
    # The still era tail can only repeat the pack era tail; keep one.
    if droppable["era_tail"] and droppable["era_tail"] in droppable["still_era_tail"]:
        droppable["era_tail"] = ""

    from . import ghost_signal_prompt as _gsp
    measure = _gsa.resolve_token_measure(token_measure_fn)
    window = int(_gsa.GHOST_CLIP_WINDOW_TOKENS)
    target = int(_gsa.GHOST_AUTHOR_TOKEN_TARGET)
    max_chars = int(_gsp.GHOST_PROMPT_MAX_CHARS)

    head_tokens = None
    if measure is not None:
        head_tokens, head_windows = measure(head)
        if head_windows > 1 or head_tokens > target:
            raise _gsa.GhostBudgetError(
                "plate prompt: the protected head (positive_tail + plate_look) "
                "measures %d SD1 tokens in %d window(s); the author target is "
                "%d in one window. The pack's own language does not fit the "
                "plate -- shorten the pack, never the head."
                % (head_tokens, head_windows, target))
    if len(head) > max_chars:
        raise _gsa.GhostBudgetError(
            "plate prompt: the protected head is %d chars, over the %d-char "
            "ceiling" % (len(head), max_chars))

    order = ("plate_look", "setting", "still_era_tail", "image_grade_tail",
             "era_tail")
    kept = {k: droppable[k] for k in order if droppable[k]}
    dropped: list = []

    def _assemble() -> str:
        return _join([head] + [kept[k] for k in order if k in kept])

    positive = _assemble()
    tokens = windows = None
    if measure is not None:
        tokens, windows = measure(positive)
    for key in PLATE_DROP_ORDER:
        fits_tokens = (measure is None) or (windows <= 1 and tokens <= window)
        fits_chars = len(positive) <= max_chars
        if fits_tokens and fits_chars:
            break
        if key in kept:
            dropped.append(key)
            del kept[key]
            positive = _assemble()
            if measure is not None:
                tokens, windows = measure(positive)
    if measure is not None and (windows > 1 or tokens > window):
        raise _gsa.GhostBudgetError(
            "plate prompt: %d tokens in %d window(s) after every droppable "
            "clause is gone" % (tokens, windows))
    if len(positive) > max_chars:
        raise _gsa.GhostBudgetError(
            "plate prompt: %d chars after every droppable clause is gone"
            % len(positive))
    if dropped:
        _LOG.info("[ghost_plate_prompt] dropped %s to fit one window",
                  ", ".join(dropped))
    return {
        "positive": positive,
        "head": head,
        "dropped": dropped,
        "clip_tokens": tokens,
        "clip_windows": windows,
        "head_clip_tokens": head_tokens,
        "measured": measure is not None,
        "sha8": hashlib.sha256(positive.encode("utf-8")).hexdigest()[:8],
    }
