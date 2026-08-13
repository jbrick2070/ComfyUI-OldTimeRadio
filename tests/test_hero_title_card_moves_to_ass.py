"""The hero title left the procgen frame -- prove what stayed and what went.

The title was drawn into the procgen CRT layer, which
``otr_post_upscale_procgen_blend`` composites with ``screen`` + ``green_only``.
``screen`` can only LIGHTEN and ``screen(A, 0) = A``, so black is the blend's
identity: the operator's own suggested outline draws literally nothing there.
Measured on a published episode the hero ran 8.96:1 over a dark ceiling and
1.13:1 over a lit monitor (Bible 07.32). It now renders as ASS through
``OTR_CaptionBurn``, which runs AFTER that blend.

**No test anywhere constructed ``_CRTRenderer`` or called ``render()`` before
this file.** Title coverage was ``_title_reveal_progress`` (a pure function),
``_draw_crt_overstrike_text`` on a standalone canvas, and ``_resolve_title_timing``
window arithmetic -- nothing asserted a single pixel reached a frame. So
suppressing the hero could equally have suppressed the ident, the carrier
decode or the meter and the whole suite would have stayed green.

Headless PIL only: no ffmpeg, no GPU, no model.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("PIL")

from nodes import _otr_title_card as OTRTC  # noqa: E402
from nodes.video_engine import _CRTRenderer  # noqa: E402

W, H, FPS = 1920, 1080, 25
TOTAL = 200
# The open card runs [10, 60) reveal then docks to 72; the close card is
# [150, 180) with no dock. Both are exercised -- "start AND end" was the ask.
TIMING = {
    "music_open_start_f": 10, "music_open_end_f": 60, "first_dialogue_f": 80,
    "music_close_start_f": 150, "music_close_end_f": 180,
}


def _renderer(title="The Probe"):
    vol = np.linspace(0.0, 1.0, TOTAL).astype("float32")
    freqs = [np.zeros(32, dtype="float32") for _ in range(TOTAL)]
    waves = [np.zeros(64, dtype="float32") for _ in range(TOTAL)]
    return _CRTRenderer(W, H, title, vol, freqs, waves, FPS, timing=TIMING)


def _ink(img, box):
    """Count phosphor-bright pixels in ``box`` = (x0, y0, x1, y1).

    Threshold 90 on green. ``draw_scopes=False`` keeps the ring, particles and
    waveform out of the frame, and the background grid's green is capped at
    ``15 + vol * 25`` -- far below 90 -- so anything counted here is TEXT.
    """
    arr = np.asarray(img.crop(box), dtype=np.int16)
    return int((arr[:, :, 1] > 90).sum())


def _centre_box():
    return (int(W * 0.2), int(H * 0.3), int(W * 0.8), int(H * 0.7))


def _ident_box():
    return (0, 0, int(W * 0.55), int(H * 0.10))


# ---------------------------------------------------------------------------
# 1. What must SURVIVE the suppression
# ---------------------------------------------------------------------------

def test_the_ident_slot_keeps_its_ink_during_a_card_and_after_it():
    """The decode-scrambled carrier line and the meter live ONLY on card frames
    and are part of the kept Matrix look; the solid ident returns after.

    This is the assertion that would have caught an early return from
    ``_draw_title_card`` -- the tempting version of the suppression, which takes
    the carrier decode and the meter with it.
    """
    r = _renderer()
    during = r.render(30, draw_scopes=False)          # mid-reveal, open card
    assert _ink(during, _ident_box()) > 0, \
        "carrier line vanished on a card frame -- the suppression cut too deep"

    after = r.render(100, draw_scopes=False)          # past end_f=72, no card
    assert _ink(after, _ident_box()) > 0, \
        "the section-1 ident vanished on a non-card frame"

    closing = r.render(160, draw_scopes=False)        # inside the CLOSE card
    assert _ink(closing, _ident_box()) > 0, \
        "carrier line vanished on the closing card"


# ---------------------------------------------------------------------------
# 2. What must be GONE -- and must come back for the control render
# ---------------------------------------------------------------------------

def test_the_hero_is_absent_by_default_and_returns_under_the_control_flag(monkeypatch):
    """Centre ink is the hero. Zero by default; non-zero under the escape hatch.

    Both halves are asserted deliberately. A test that only checked "the hero is
    gone" would still pass if ``_draw_title_card`` stopped being called at all,
    which is a different and much worse bug.
    """
    monkeypatch.delenv("OTR_HERO_TITLE_IN_PROCGEN", raising=False)
    r = _renderer()
    frame = r.render(40, draw_scopes=False)
    assert _ink(frame, _centre_box()) == 0, \
        "hero glyphs are still burned into the procgen layer"

    monkeypatch.setenv("OTR_HERO_TITLE_IN_PROCGEN", "1")
    r2 = _renderer()
    control = r2.render(40, draw_scopes=False)
    assert _ink(control, _centre_box()) > 0, \
        "the control arm draws no hero -- the acceptance measurement has no baseline"
    # ...and the legacy arm must emit NO plan, or the control render carries the
    # title TWICE (baked in AND burned on top) and compares the new title
    # against itself. Found in QA.
    assert r2.title_card_plan(main_frames=TOTAL) is None, \
        "the legacy arm still emits an ASS plan -- the control is not a control"


def test_the_closing_card_hero_is_suppressed_too(monkeypatch):
    """'start AND end' was the operator's ask; a fix that only covered the
    opening card would leave a second illegible title at the back of every
    episode."""
    monkeypatch.delenv("OTR_HERO_TITLE_IN_PROCGEN", raising=False)
    r = _renderer()
    assert _ink(r.render(160, draw_scopes=False), _centre_box()) == 0


# ---------------------------------------------------------------------------
# 3. The plan that replaced those pixels
# ---------------------------------------------------------------------------

def test_the_plan_covers_both_cards_and_never_runs_past_the_main_video():
    r = _renderer()
    plan = r.title_card_plan(main_frames=TOTAL)
    assert plan is not None and plan["version"] == OTRTC.PLAN_VERSION
    assert sorted({c["kind"] for c in plan["cards"]}) == ["close", "open"]
    assert sorted({f["kind"] for f in plan["frames"]}) == ["close", "open"]
    # NEVER past the MAIN video. The encode appends the telemetry HUD post-roll,
    # and a closing card clamped to the ENCODED length would hang the title over
    # the post-roll and into the credits.
    assert max(f["f"] for f in plan["frames"]) < TOTAL


def test_the_plan_is_deterministic():
    """The scramble is seeded per frame from a stable hash, not `hash()`.

    If it were not deterministic the ASS could not reproduce frames it never
    rendered, and PYTHONHASHSEED would make every run differ.
    """
    a = _renderer().title_card_plan(main_frames=TOTAL)
    b = _renderer().title_card_plan(main_frames=TOTAL)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_the_plan_scrambles_early_and_resolves_before_the_dock():
    """The Matrix decode is KEPT -- that was explicit. Prove it is in the plan
    rather than merely absent from the frame."""
    r = _renderer()
    plan = r.title_card_plan(main_frames=TOTAL)
    by_frame = {f["f"]: f for f in plan["frames"]}
    early = "".join(i["text"] for i in by_frame[11]["items"] if i["kind"] == "text")
    late = "".join(i["text"] for i in by_frame[55]["items"] if i["kind"] == "text")
    assert late.replace(" ", "") == "THEPROBE", late
    assert early.replace(" ", "") != "THEPROBE", "no scramble -- the decode was lost"


def test_a_MULTI_LINE_title_scrambles_identically_in_the_plan_and_the_draw():
    """The plan must reproduce the draw LINE BY LINE, not just on line one.

    Caught in QA. ``_draw_title_card`` calls ``self._rng(fi, "hero")`` INSIDE its
    per-line loop, and `_rng` builds a fresh Generator from (title, frame, salt)
    -- none of which vary by line -- so every line restarts the same stream. The
    planner had hoisted that call out of the loop, which looks like an
    optimisation and silently gave line 2 onward different glyphs.

    Every other test in this file uses a title short enough to fit one line, so
    all of them passed while wrapped titles were wrong. Episode titles wrap at
    hero size routinely, and this asserts against the REAL draw rather than a
    remembered expectation.
    """
    from PIL import Image, ImageDraw

    title = "The Mystery Of The Vanishing Broadcast Tower At Midnight"
    r = _renderer(title)

    # What the draw would produce for this frame, line by line.
    img = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    upper = title.upper()
    _font, lines, _size = r._fit_hero(draw, upper)
    assert len(lines) > 1, "fixture no longer wraps -- pick a longer title"

    fi = 20
    p = OTRTC.title_reveal_progress(fi, 10, 60, False, 0.4)
    total_chars = sum(len(ln) for ln in lines)
    revealed = int(p * total_chars)
    expected, consumed = [], 0
    for ln in lines:
        take = max(0, min(len(ln), revealed - consumed))
        expected.append(r._decode_line(ln, take, r._rng(fi, "hero")))
        consumed += len(ln)

    plan = r.title_card_plan(main_frames=TOTAL)
    frame = next(f for f in plan["frames"] if f["f"] == fi)
    planned = [i["text"] for i in frame["items"] if i["kind"] == "text"]
    assert planned == expected, (
        "the plan and the draw scramble differently:\n"
        f"  draw: {expected}\n  plan: {planned}")


def test_the_plan_matches_the_REAL_DRAW_on_every_frame_including_the_dock(
        monkeypatch):
    """Compare the plan against what the renderer ACTUALLY draws, frame by frame.

    The dock is the least-tested surface in this change: both sides re-wrap at a
    shrinking font size and lerp the block onto the ident anchor, and three
    review passes reasoned about it without anything executing it. Reading two
    implementations and agreeing they match is exactly how the per-line RNG
    divergence survived its first review.

    So this does not re-derive the draw's arithmetic -- duplicating it in the
    test is the same mistake one level up. It SPIES on the real draw call,
    capturing the position, text, colour and font size handed to
    ``_draw_crt_overstrike_text`` for every frame of the card, and requires the
    plan to have predicted all four.
    """
    from nodes import video_engine as VE

    title = "The Mystery Of The Vanishing Broadcast Tower At Midnight"

    # The plan is built with the procgen draw OFF (its shipping state).
    plan = _renderer(title).title_card_plan(main_frames=TOTAL)
    planned = {}
    for fr in plan["frames"]:
        planned[fr["f"]] = [i for i in fr["items"] if i["kind"] == "text"]

    # The draw is exercised with the legacy flag ON, so it actually runs.
    monkeypatch.setenv("OTR_HERO_TITLE_IN_PROCGEN", "1")
    drawn = {}
    real = VE._draw_crt_overstrike_text

    def spy(draw, xy, text, fill=None, font=None, **kw):
        drawn.setdefault(spy.fi, []).append(
            {"x": xy[0], "y": xy[1], "text": text, "fill": tuple(fill),
             "size": getattr(font, "size", None)})
        return real(draw, xy, text, fill=fill, font=font, **kw)

    monkeypatch.setattr(VE, "_draw_crt_overstrike_text", spy)

    r = _renderer(title)
    card = next(c for c in r._cards if c["kind"] == "open")
    assert card["dock_frames"] > 0, "fixture no longer docks -- pick other timing"

    for fi in range(card["start_f"], card["end_f"]):
        spy.fi = fi
        r.render(fi, draw_scopes=False)

    assert drawn, "the spy captured nothing -- the draw never ran"
    dock_frames = [f for f in drawn if f >= card["music_end_f"]]
    assert dock_frames, "no dock frames were exercised"

    for fi in sorted(drawn):
        got = drawn[fi]
        want = planned.get(fi, [])
        assert len(got) == len(want), (
            f"frame {fi}: draw made {len(got)} text calls, plan has {len(want)}")
        for gi, (g, w) in enumerate(zip(got, want)):
            assert g["text"] == w["text"], (
                f"frame {fi} line {gi}: draw {g['text']!r} != plan {w['text']!r}")
            assert (g["x"], g["y"]) == (w["x"], w["y"]), (
                f"frame {fi} line {gi}: draw at {(g['x'], g['y'])}, "
                f"plan at {(w['x'], w['y'])}")
            assert g["size"] == w["size"], (
                f"frame {fi} line {gi}: draw size {g['size']}, plan {w['size']}")
            assert OTRTC._ass_colour(g["fill"]) == w["colour"], (
                f"frame {fi} line {gi}: draw colour {g['fill']} != plan "
                f"{w['colour']} -- the POP window disagrees")


def test_an_episode_with_no_card_plans_nothing_rather_than_failing():
    """No resolved music window is a legitimate state, and it must produce None
    -- an EMPTY plan, not a failed one. The burn refuses an empty plan on
    purpose, so conflating the two would refuse every card-less episode."""
    vol = np.zeros(TOTAL, dtype="float32")
    freqs = [np.zeros(32, dtype="float32") for _ in range(TOTAL)]
    waves = [np.zeros(64, dtype="float32") for _ in range(TOTAL)]
    r = _CRTRenderer(W, H, "No Card", vol, freqs, waves, FPS, timing={})
    assert r.title_card_plan(main_frames=TOTAL) is None
