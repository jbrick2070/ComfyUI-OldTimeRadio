"""Sprint 3 (2026-05-28) -- arc-aware line composer prompt (Path A).

Pinned invariants:
  * The Sprint 3 fields on LineRequest default empty; legacy callers
    that do not thread them produce the EXACT pre-Sprint-3 prompt
    (byte-identical / KV-cache stable).
  * When any dramatic field is set, the prompt grows a DRAMATIC
    FRAME block above the LAST SPOKEN rolling window AND grows a
    no-restate output constraint above "Speak now."
  * The block order does NOT touch the static prefix (STYLE / THEME
    / EPISODE CONTEXT / NAMED ENTITIES / CAST / OUTLINE) so the
    KV-reusable prefix stays intact.
  * No-restate heuristic helper rejects lines that verbatim-quote
    objective or turn text.
"""
from __future__ import annotations

import pytest

from nodes._otr_line_composer import (
    LineRequest,
    _build_user_prompt,
)


def _minimal_legacy_request(**overrides) -> LineRequest:
    """Pre-Sprint-3 shape: only the original required + minimum
    Phase-1 fields. The result must NOT carry the DRAMATIC FRAME
    block or the no-restate constraint -- proves KV stability."""
    base = dict(
        speaker="ALICE",
        intent="press maeve to admit the omission",
        mood="dry",
        canon_header="TITLE: TBD\nPREMISE: test premise\nSETTING: lab\nTIME: midnight",
        last_lines=[("ANNOUNCER", "Welcome back to Night Forensics.")],
    )
    base.update(overrides)
    return LineRequest(**base)


def _sprint3_request(**overrides) -> LineRequest:
    """A LineRequest with all Sprint 3 fields populated."""
    base = dict(
        speaker="ALICE",
        intent="press maeve to admit the omission",
        mood="dry",
        canon_header="TITLE: TBD\nPREMISE: test premise\nSETTING: lab\nTIME: midnight",
        last_lines=[("ANNOUNCER", "Welcome back to Night Forensics.")],
        dramatic_question="Will Maeve sign the confession before the audit closes?",
        beat_objective="press maeve to name her own forged signature",
        beat_obstacle="maeve has rehearsed her denials",
        beat_turn="maeve names the omission aloud",
        beat_subtext="alice already knows the answer",
        beat_tension=4,
        next_turn="bob enters with the audit clock",
    )
    base.update(overrides)
    return LineRequest(**base)


# ---------------------------------------------------------------------------
# KV-stability invariant: legacy callers produce the pre-Sprint-3 prompt
# ---------------------------------------------------------------------------


def test_legacy_request_has_no_dramatic_frame():
    """When Sprint 3 fields are empty, the prompt must NOT mention the
    DRAMATIC FRAME (DRAMATIC QUESTION / THIS BEAT / NEXT BEAT MUST REVEAL).

    F6 (story-engine v1, SPLIT): the indirect-performance rider is now
    UNCONDITIONAL, so it IS present even on a legacy beat -- but the
    situation-change clause stays gated to turn beats (no beat_turn here).
    """
    prompt = _build_user_prompt(_minimal_legacy_request())
    assert "DRAMATIC QUESTION" not in prompt
    assert "THIS BEAT:" not in prompt
    assert "NEXT BEAT MUST REVEAL" not in prompt
    # F6: indirect-performance rider is unconditional now.
    assert "Do not summarize the objective" in prompt
    assert "Perform the objective indirectly" in prompt
    # ...but the situation-change clause only fires on turn/costly beats.
    assert "The situation must be different after this line." not in prompt


def test_legacy_request_still_carries_write_line_block():
    """Sanity: the pre-Sprint-3 prompt's WRITE LINE block survived."""
    prompt = _build_user_prompt(_minimal_legacy_request())
    assert "WRITE LINE" in prompt
    assert "Speak now." in prompt


# ---------------------------------------------------------------------------
# Sprint 3 enrichment: DRAMATIC FRAME + no-restate constraint
# ---------------------------------------------------------------------------


def test_sprint3_request_renders_dramatic_frame_above_last_spoken():
    prompt = _build_user_prompt(_sprint3_request())
    # Block exists.
    assert "DRAMATIC QUESTION: Will Maeve sign" in prompt
    assert "THIS BEAT:" in prompt
    # story-quality v2 (baked in): at tension >= OBJECTIVE_DEFLECTION_TENSION_MIN
    # (=4) with subtext on a character beat, the L2 authoring contract WITHHOLDS
    # the bald Objective line and injects the deflection directive instead.
    assert "Objective: press maeve to name her own forged signature" not in prompt
    assert "this line IS the deflection" in prompt
    assert "Obstacle:  maeve has rehearsed her denials" in prompt
    assert "Turn:      maeve names the omission aloud" in prompt
    assert "Subtext:   alice already knows the answer" in prompt
    assert "Tension:   4/5" in prompt
    assert "NEXT BEAT MUST REVEAL: bob enters with the audit clock" in prompt
    # Block sits above the rolling window (magnetic pole position).
    next_idx = prompt.index("NEXT BEAT MUST REVEAL")
    last_spoken_idx = prompt.index("LAST SPOKEN (this scene):")
    assert next_idx < last_spoken_idx, (
        "DRAMATIC FRAME must sit ABOVE LAST SPOKEN so next_turn is "
        "the magnetic pole closest to the generation slot."
    )


def test_sprint3_request_renders_no_restate_constraint():
    prompt = _build_user_prompt(_sprint3_request())
    assert "Do not summarize the objective" in prompt
    assert "Perform the objective indirectly" in prompt
    assert "The situation must be different after this line." in prompt
    # Constraint must land BEFORE Speak now. so the model reads it
    # as its final directive.
    constraint_idx = prompt.index("Perform the objective indirectly")
    speak_idx = prompt.index("Speak now.")
    assert constraint_idx < speak_idx


def test_partial_sprint3_fields_render_only_those_lines():
    """Only set dramatic_question + next_turn; THIS BEAT block has
    nothing to render and must be omitted entirely."""
    req = _sprint3_request(
        beat_objective="",
        beat_obstacle="",
        beat_turn="",
        beat_subtext="",
        beat_tension=0,
    )
    prompt = _build_user_prompt(req)
    assert "DRAMATIC QUESTION: Will Maeve sign" in prompt
    assert "NEXT BEAT MUST REVEAL: bob enters" in prompt
    # No empty `THIS BEAT:` block.
    assert "THIS BEAT:" not in prompt
    # Constraint still lands because dramatic_question / next_turn
    # are present.
    assert "Perform the objective indirectly" in prompt


def test_static_prefix_unchanged_between_legacy_and_sprint3():
    """KV-cache stability: the static prefix (STYLE / EPISODE
    CONTEXT / CAST / OUTLINE block) must be byte-identical between
    a legacy and a Sprint 3 request when the static fields are
    identical. The Sprint 3 fields only add to the PER-BEAT TAIL."""
    legacy = _minimal_legacy_request(
        style_descriptor="late-night-anthology",
        outline_spine="OUTLINE\nb001: ANNOUNCER -- open\nb002: ALICE -- press",
    )
    s3 = _sprint3_request(
        style_descriptor="late-night-anthology",
        outline_spine="OUTLINE\nb001: ANNOUNCER -- open\nb002: ALICE -- press",
    )
    legacy_prompt = _build_user_prompt(legacy)
    s3_prompt = _build_user_prompt(s3)
    # Both prompts begin with the same static-prefix slice up through
    # OUTLINE (which is the last static-prefix block before the
    # per-beat tail).
    cut = "OUTLINE\nb001: ANNOUNCER -- open\nb002: ALICE -- press"
    assert cut in legacy_prompt
    assert cut in s3_prompt
    legacy_prefix = legacy_prompt[: legacy_prompt.index(cut) + len(cut)]
    s3_prefix = s3_prompt[: s3_prompt.index(cut) + len(cut)]
    assert legacy_prefix == s3_prefix, (
        "Static prefix must be byte-identical between legacy and "
        "Sprint 3 requests; Sprint 3 enrichment lives in the per-"
        "beat tail only."
    )


def test_tension_out_of_range_silently_dropped():
    """Tension 0 or 6+ does not render -- the per-line guard skips
    rather than emitting `Tension: 0/5` (which would look like a
    deliberate signal)."""
    req = _sprint3_request(beat_tension=0)
    prompt = _build_user_prompt(req)
    assert "Tension:" not in prompt

    req2 = _sprint3_request(beat_tension=9)
    prompt2 = _build_user_prompt(req2)
    assert "Tension:" not in prompt2
