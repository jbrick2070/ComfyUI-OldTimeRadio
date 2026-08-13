"""Deterministic hero-title-card planning, shared by the renderer and the burn.

WHY THIS MODULE EXISTS
----------------------
The hero title used to be drawn straight into the procgen CRT frame, which is
then composited by ``otr_post_upscale_procgen_blend`` with ``screen`` +
``green_only`` -- a LIGHTEN-ONLY blend. ``screen(A, 0) = A``, so black is the
blend's identity and an outline drawn there is a mathematical no-op. Measured on
a published episode the title ran 8.96:1 over a dark ceiling and 1.13:1 over a
lit monitor. Bible 07.32.

The fix moves the title DOWNSTREAM of that blend, into the ASS text layer
``OTR_CaptionBurn`` already burns, where an outline is real. That means two
different modules must agree, frame for frame, on what the title looks like --
so neither of them owns the arithmetic. This module does.

``video_engine`` has the PIL fonts and can MEASURE text; ``_otr_captions`` has
the ASS writer and cannot. So measurement is injected (``measure_text`` /
``fit_hero``) and everything downstream of it -- the reveal ramp, the seeded
scramble, the POP, the dock lerp, the frame clamps -- lives here and is computed
ONCE. The renderer hands the finished plan to the burn as serializable data.

WHY THE PLAN IS PER-FRAME
-------------------------
It is not a stylistic choice. The scramble RNG is seeded PER FRAME, the POP is a
1-2 frame colour shift, and the dock rescales the font every frame. None of that
is expressible as ASS tag interpolation, and ``\\pos`` combined with ``\\t()``
does not animate position at all (``\\move`` does, and mixing them is invalid).
One frame-bounded event per visible line is the only shape that reproduces the
Matrix decode deterministically. Density is ~250 events for a 5 s card, which is
trivial for libass.

Deliberately dependency-light: numpy only, no PIL, no ffmpeg, and it imports
NOTHING from ``video_engine`` or ``_otr_captions`` -- both import it, so any
edge between them would be a cycle. UTF-8, no BOM.
"""
from __future__ import annotations

import hashlib

import numpy as np

# The plan travels between two nodes as JSON. Bump this when the shape changes
# in a way a consumer must notice; the consumer REFUSES an unknown version
# rather than guessing at a field, because a silently-misread plan ships an
# episode with a wrong or missing title and a clean log.
PLAN_VERSION = 1

# Scramble alphabet. Kept byte-identical to the renderer's original so a frame
# planned here and a frame drawn there produce the same glyphs.
DECODE_GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#%*+=-"

# Phosphor green, and the 1-2 frame POP bloom right before lock.
HERO_RGB = (0, 255, 65)
POP_RGB = (200, 255, 200)

# The reveal completes in the first fraction of the window and then HOLDS solid
# until the POP/dock. Before BUG-LOCAL-409 it stretched across the whole window,
# so the title only resolved on the final frame and read as scramble throughout.
DEFAULT_REVEAL_FRACTION = 0.4

# A real sentinel, because ``None`` is a LEGITIMATE seed title: the renderer
# hashes ``f"{self.title}|..."``, so a None title seeds from the string "None".
# Defaulting on ``is None`` would quietly seed the plan from "SIGNAL" instead
# and scramble different glyphs than the frames it is reproducing.
_UNSET = object()


def decode_rng(title, fi, salt):
    """A stable-hash-seeded Generator: same (title, frame, salt) -> same frame.

    Stable hashing, not ``hash()`` -- PYTHONHASHSEED would otherwise make the
    scramble differ between the render process and any process that re-plans it.
    """
    seed = int.from_bytes(
        hashlib.blake2s(f"{title}|{int(fi)}|{salt}".encode()).digest()[:8],
        "big",
    )
    return np.random.default_rng(seed)


def decode_line(text, reveal_n, rng):
    """Left-to-right decode: the first ``reveal_n`` chars are solid, the rest are
    seeded scramble. Spaces stay spaces so word shape survives the scramble."""
    out = []
    for i, ch in enumerate(text):
        if ch == " " or i < reveal_n:
            out.append(ch)
        else:
            out.append(DECODE_GLYPHS[int(rng.integers(0, len(DECODE_GLYPHS)))])
    return "".join(out)


def title_reveal_progress(fi, w0, me, in_dock, reveal_frac=DEFAULT_REVEAL_FRACTION):
    """BUG-LOCAL-409: decode/reveal progress for the hero title card.

    The reveal COMPLETES in the first ``reveal_frac`` of the ``[w0, me)`` window
    and then HOLDS solid (p == 1.0) until the POP/dock. Returns p in [0.0, 1.0].
    """
    if in_dock:
        return 1.0
    try:
        rf = float(reveal_frac)
    except (TypeError, ValueError):
        rf = DEFAULT_REVEAL_FRACTION
    rf = min(1.0, max(0.05, rf))
    span = max(1, int((int(me) - int(w0)) * rf))
    return min(1.0, max(0.0, (int(fi) - int(w0)) / span))


def _ass_colour(rgb):
    """PIL (r, g, b) -> the ASS ``&HBBGGRR&`` inline-override form."""
    r, g, b = (int(max(0, min(255, c))) for c in rgb)
    return f"&H{b:02X}{g:02X}{r:02X}&"


def plan_card_frame(*, title, card, fi, frame_size, ident_xy, title_size,
                    measure_text, fit_hero, reveal_frac=DEFAULT_REVEAL_FRACTION,
                    rng_title=_UNSET):
    """Plan ONE frame of ONE card. Returns a list of draw items.

    This is the arithmetic ``_CRTRenderer._draw_title_card`` used to carry
    inline, lifted out verbatim so the ASS emitter and the procgen draw cannot
    disagree. Items are ``{"kind": "text", ...}`` and ``{"kind": "rect", ...}``
    in draw order, with pixel geometry in the frame's own coordinate space.

    ``measure_text(text, size) -> (width, height)`` and
    ``fit_hero(title) -> (lines, size)`` are injected because measuring needs
    PIL font state the burn side does not have.

    ``rng_title`` seeds the scramble and defaults to ``title``. They are NOT
    always the same string: the renderer draws the UPPERCASED title but seeds
    from the raw one, so a plan that seeded from the uppercased text would
    scramble different glyphs than the frames it is meant to reproduce. Passing
    both explicitly is what keeps the two sides identical.
    """
    w, h = int(frame_size[0]), int(frame_size[1])
    ident_x, ident_y = int(ident_xy[0]), int(ident_xy[1])
    me = int(card["music_end_f"])
    w0 = int(card["start_f"])
    dock_frames = int(card.get("dock_frames") or 0)
    in_dock = fi >= me
    p = title_reveal_progress(fi, w0, me, in_dock, reveal_frac)

    lines, hero_size = fit_hero(title)

    # POP: a 1-2 frame brightness bloom right before lock (music_end).
    pop = (not in_dock) and (1 <= (me - fi) <= 2)
    colour = POP_RGB if pop else HERO_RGB

    # Decoded-fragment reveal across the wrapped lines (integer-frame).
    total_chars = sum(len(ln) for ln in lines)
    revealed = total_chars if in_dock else int(p * total_chars)
    # A FRESH generator PER LINE, not one stream shared across them. The draw
    # calls `self._rng(fi, "hero")` inside its own per-line loop, and `_rng`
    # builds a new Generator from (title, frame, salt) every call -- none of
    # which vary by line -- so each line restarts the SAME stream. Hoisting this
    # out of the loop looks like an optimisation and silently desynchronises
    # every line after the first: a wrapped title scrambled correctly on line 1
    # and differently on line 2, in a way no single-line test can see.
    seed_title = title if rng_title is _UNSET else rng_title
    shown, consumed = [], 0
    for ln in lines:
        take = max(0, min(len(ln), revealed - consumed))
        shown.append(decode_line(ln, take, decode_rng(seed_title, fi, "hero")))
        consumed += len(ln)

    # Block metrics at the (possibly docked) size. The dock shrinks the hero to
    # the ident's own size and lerps it onto the ident anchor -- a ~0.4 s
    # handoff, not a state.
    size = hero_size
    if in_dock and dock_frames > 0:
        d = min(1.0, max(0.0, (fi - me) / dock_frames))
        size = int(round(hero_size + (int(title_size) - hero_size) * d))
        size = max(8, size)
        lines = fit_hero(title, size=size)[0]
        shown = lines  # fully decoded while docking
    else:
        d = 0.0

    line_h = max(measure_text(s or "M", size)[1] for s in shown)
    block_h = line_h * len(shown) + (len(shown) - 1) * (line_h // 4)
    cx_centre, cy_centre = w // 2, h // 2
    top_centre = cy_centre - block_h // 2

    items = []
    for li, s in enumerate(shown):
        tw = measure_text(s, size)[0]
        x_centre = cx_centre - tw // 2
        y_centre = top_centre + li * (line_h + line_h // 4)
        x = int(round(x_centre + (ident_x - x_centre) * d))
        y = int(round(y_centre + (ident_y - y_centre) * d))
        # Clamp inside the frame -- no black edge from any drift or dock.
        x = max(0, min(x, w - tw))
        y = max(0, min(y, h - line_h))
        items.append({
            "kind": "text", "text": s, "x": x, "y": y,
            "size": int(size), "colour": _ass_colour(colour),
            "line_h": int(line_h),
        })
        # Block cursor after the last revealed char (during the reveal only).
        if not in_dock and li == len(shown) - 1 and p < 1.0:
            cur_w = max(4, line_h // 2)
            items.append({
                "kind": "rect", "x": x + tw + 2, "y": y,
                "w": cur_w, "h": int(line_h), "colour": _ass_colour(colour),
            })
    return items


def build_title_plan(*, title, cards, fps, frame_size, ident_xy, title_size,
                     main_frames, measure_text, fit_hero,
                     reveal_frac=DEFAULT_REVEAL_FRACTION, rng_title=_UNSET):
    """Plan EVERY frame of EVERY card. Returns the serializable plan dict.

    ``main_frames`` is the MAIN video's frame count, deliberately NOT the
    encoded total: the encode appends the telemetry HUD post-roll, and a closing
    card clamped to the encoded length would hang the title over the post-roll
    and into the credits. The synthesized close sets its end to ``total``, which
    is main-video frames, so this is the number that agrees with it.
    """
    total = int(main_frames)
    frames = []
    for card in cards or []:
        start_f = max(0, int(card["start_f"]))
        end_f = min(total, int(card["end_f"]))
        for fi in range(start_f, end_f):
            items = plan_card_frame(
                title=title, card=card, fi=fi, frame_size=frame_size,
                ident_xy=ident_xy, title_size=title_size,
                measure_text=measure_text, fit_hero=fit_hero,
                reveal_frac=reveal_frac, rng_title=rng_title,
            )
            if items:
                frames.append({"f": int(fi), "kind": str(card.get("kind") or ""),
                               "items": items})
    frames.sort(key=lambda fr: fr["f"])
    return {
        "version": PLAN_VERSION,
        "title": str(title or ""),
        "fps": int(fps),
        "play_res": [int(frame_size[0]), int(frame_size[1])],
        "main_frames": total,
        "reveal_fraction": float(reveal_frac),
        "cards": [{"kind": str(c.get("kind") or ""),
                   "start_f": int(c["start_f"]),
                   "music_end_f": int(c["music_end_f"]),
                   "dock_frames": int(c.get("dock_frames") or 0),
                   "end_f": int(c["end_f"])} for c in (cards or [])],
        "frames": frames,
    }


__all__ = [
    "PLAN_VERSION", "DECODE_GLYPHS", "HERO_RGB", "POP_RGB",
    "DEFAULT_REVEAL_FRACTION", "decode_rng", "decode_line",
    "title_reveal_progress", "plan_card_frame", "build_title_plan",
]
