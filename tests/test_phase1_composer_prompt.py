"""tests/test_phase1_composer_prompt.py — composer prompt enrichment.

Covers (per script-writing-architecture synthesis §3 Phase 1 + §6.D):

  * LineRequest schema extension (style_descriptor / outline_spine /
    character_voice_card) with frozen-dataclass defaults
  * render_outline_spine    — flat one-line-per-beat spine, both
                              pydantic Outline and plain-dict input
  * build_voice_card        — cast row → compact `name (gender, traits)`
  * _build_user_prompt      — static-first ordering for KV cache,
                              optional blocks dropped when empty
  * sliding-window N=5      — confirmed via the dataclass plumbing,
                              not via a live LLM call (window cap
                              lives in the orchestrator)

Pure-Python. No GPU. No LLM. Unit-scope only; soak is Jeffrey's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_composer import (  # noqa: E402
    LineRequest,
    _build_user_prompt,
    build_voice_card,
    render_outline_spine,
)


# ---------------------------------------------------------------------------
# render_outline_spine
# ---------------------------------------------------------------------------


def _beat(beat_id: str, speaker: str, role: str, intent: str, mood: str = "") -> dict:
    return {
        "beat_id": beat_id,
        "speaker": speaker,
        "speaker_role": role,
        "intent": intent,
        "mood": mood,
    }


class TestRenderOutlineSpine:

    def test_empty_outline_returns_empty(self):
        assert render_outline_spine(None) == ""
        assert render_outline_spine([]) == ""

    def test_single_voiced_beat(self):
        spine = render_outline_spine([
            _beat("b002", "ALICE", "character",
                  "hears unusual signal in lab", "curious"),
        ])
        assert spine.startswith("OUTLINE:")
        assert "b002 ALICE (curious): hears unusual signal in lab" in spine

    def test_announcer_beat_rendered_like_character(self):
        spine = render_outline_spine([
            _beat("b001", "ANNOUNCER", "announcer",
                  "introduce the episode", "steady"),
        ])
        assert "b001 ANNOUNCER (steady): introduce the episode" in spine

    def test_music_beats_rendered_with_role_label(self):
        spine = render_outline_spine([
            _beat("b001", "NARRATOR", "music_open", "cold open"),
            _beat("b020", "NARRATOR", "music_close", "fade out"),
        ])
        assert "b001 [music_open]: cold open" in spine
        assert "b020 [music_close]: fade out" in spine
        # No speaker name on music beats.
        assert "NARRATOR" not in spine

    def test_mood_omitted_when_empty(self):
        spine = render_outline_spine([
            _beat("b002", "ALICE", "character", "speak", mood=""),
        ])
        assert "b002 ALICE: speak" in spine
        # No empty parens.
        assert "()" not in spine

    def test_accepts_object_with_beats_attribute(self):
        class FakeOutline:
            beats = [_beat("b001", "ALICE", "character", "speak", "tense")]
        spine = render_outline_spine(FakeOutline())
        assert "b001 ALICE (tense): speak" in spine

    def test_never_raises_on_garbage_beats(self):
        # Garbage entry should not raise.
        spine = render_outline_spine([{}, _beat("b001", "A", "character", "x")])
        assert "OUTLINE:" in spine

    def test_full_episode_arc_renders_in_order(self):
        beats = [
            _beat("b001", "NARRATOR", "music_open", "cold open"),
            _beat("b002", "ANNOUNCER", "announcer", "intro", "steady"),
            _beat("b003", "ALICE", "character", "hears signal", "curious"),
            _beat("b004", "BOB", "character", "warns", "worried"),
            _beat("b005", "NARRATOR", "music_inter", "transition"),
            _beat("b006", "ALICE", "character", "decides", "determined"),
        ]
        spine = render_outline_spine(beats)
        # Lines should appear in given order.
        order = ["b001", "b002", "b003", "b004", "b005", "b006"]
        positions = [spine.find(bid) for bid in order]
        assert all(p > 0 for p in positions), \
            f"some beat_ids missing from spine: {positions}"
        for i in range(len(positions) - 1):
            assert positions[i] < positions[i + 1], (
                f"spine order broken at {order[i]} vs {order[i+1]}"
            )


# ---------------------------------------------------------------------------
# build_voice_card
# ---------------------------------------------------------------------------


class TestBuildVoiceCard:

    def test_full_row(self):
        card = build_voice_card({
            "name": "ALICE",
            "gender": "female",
            "character_description": "weary forensic engineer in her 40s",
        })
        assert card == "ALICE (female, weary forensic engineer in her 40s)"

    def test_missing_description(self):
        card = build_voice_card({"name": "ALICE", "gender": "female"})
        assert card == "ALICE (female)"

    def test_missing_gender(self):
        card = build_voice_card({
            "name": "ALICE",
            "character_description": "lone caretaker",
        })
        assert card == "ALICE (lone caretaker)"

    def test_bare_name_only(self):
        card = build_voice_card({"name": "ALICE"})
        assert card == "ALICE"

    def test_announcer_stub(self):
        card = build_voice_card({"name": "ANNOUNCER"})
        assert card == "ANNOUNCER (omniscient narrator)"

    def test_announcer_with_description_uses_description(self):
        card = build_voice_card({
            "name": "ANNOUNCER",
            "character_description": "midcentury news anchor",
        })
        # The ANNOUNCER stub only fires when description is empty;
        # populated description wins.
        assert card == "ANNOUNCER (midcentury news anchor)"

    def test_empty_row(self):
        assert build_voice_card({}) == ""
        assert build_voice_card(None) == ""

    def test_strips_whitespace(self):
        card = build_voice_card({
            "name": "  ALICE  ",
            "gender": "  female  ",
            "character_description": "  trait  ",
        })
        assert card == "ALICE (female, trait)"


# ---------------------------------------------------------------------------
# LineRequest schema
# ---------------------------------------------------------------------------


class TestLineRequestSchema:

    def test_phase1_fields_default_empty_strings(self):
        req = LineRequest(
            speaker="ALICE", intent="reveal", mood="tense", target_words=15,
            canon_header="x", last_lines=[],
        )
        assert req.style_descriptor == ""
        assert req.outline_spine == ""
        assert req.character_voice_card == ""

    def test_phase1_fields_round_trip(self):
        req = LineRequest(
            speaker="ALICE", intent="reveal", mood="tense", target_words=15,
            canon_header="x", last_lines=[],
            style_descriptor="closed_room_suspense",
            outline_spine="OUTLINE:\n  b001 ALICE (tense): speak",
            character_voice_card="ALICE (female, weary)",
        )
        assert req.style_descriptor == "closed_room_suspense"
        assert "OUTLINE:" in req.outline_spine
        assert req.character_voice_card == "ALICE (female, weary)"


# ---------------------------------------------------------------------------
# _build_user_prompt -- static-first ordering for KV cache
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:

    def test_bare_request_omits_optional_blocks(self):
        req = LineRequest(
            speaker="ALICE", intent="reveal", mood="tense", target_words=15,
            canon_header="TITLE: x\nSETTING: y", last_lines=[],
        )
        prompt = _build_user_prompt(req)
        assert "STYLE:" not in prompt
        assert "THEME:" not in prompt
        assert "OUTLINE:" not in prompt
        assert "ALLOWED NAMES" not in prompt
        assert "NAMED ENTITIES" not in prompt
        assert "CHARACTER:" not in prompt
        assert "CAST" not in prompt
        assert "CURRENT BEAT" not in prompt
        assert "POSITION:" not in prompt
        assert "SOUND IN THE ROOM" not in prompt
        # Required blocks always present.
        assert "EPISODE CONTEXT" in prompt
        assert "LAST SPOKEN (this scene):" in prompt
        assert "WRITE LINE" in prompt
        # v4: role induction replaces "Speaker: ALICE" label.
        assert "You are ALICE." in prompt
        assert "Speak now." in prompt

    def test_style_block_renders_when_set(self):
        req = LineRequest(
            speaker="ALICE", intent="reveal", mood="tense", target_words=15,
            canon_header="x", last_lines=[],
            style_descriptor="noir_interrogation",
        )
        prompt = _build_user_prompt(req)
        assert "STYLE: noir_interrogation" in prompt

    def test_outline_block_renders_when_set(self):
        req = LineRequest(
            speaker="ALICE", intent="reveal", mood="tense", target_words=15,
            canon_header="x", last_lines=[],
            outline_spine="OUTLINE:\n  b001 ALICE (tense): speak",
        )
        prompt = _build_user_prompt(req)
        assert "OUTLINE:" in prompt
        assert "b001 ALICE (tense): speak" in prompt

    def test_allowed_roster_block_renders_when_set(self):
        req = LineRequest(
            speaker="ALICE", intent="reveal", mood="tense", target_words=15,
            canon_header="x", last_lines=[],
            allowed_roster=frozenset({"ALICE", "BOB", "ANNOUNCER"}),
        )
        prompt = _build_user_prompt(req)
        assert "ALLOWED NAMES" in prompt
        # Sorted for KV-cache stability across calls.
        assert "ALICE, ANNOUNCER, BOB" in prompt

    def test_character_voice_card_block_renders_when_set(self):
        req = LineRequest(
            speaker="ALICE", intent="reveal", mood="tense", target_words=15,
            canon_header="x", last_lines=[],
            character_voice_card="ALICE (female, weary forensic engineer)",
        )
        prompt = _build_user_prompt(req)
        # v4: legacy CHARACTER block still emits when all_voice_cards
        # is empty. Full-cast CAST block (Commit 2) supersedes it.
        assert "CHARACTER: ALICE (female, weary forensic engineer)" in prompt

    def test_cast_block_replaces_character_when_all_voice_cards_set(self):
        # v4 Commit 2: full-cast block joined from voice_card_by_name.
        req = LineRequest(
            speaker="ALICE", intent="reveal", mood="tense", target_words=15,
            canon_header="x", last_lines=[],
            character_voice_card="ALICE (female, weary)",
            all_voice_cards=(
                "ALICE (female, weary)\nBOB (male, anxious)"
            ),
        )
        prompt = _build_user_prompt(req)
        assert "CAST" in prompt
        # When CAST renders, the single-speaker CHARACTER block is
        # suppressed.
        assert "CHARACTER:" not in prompt
        assert "ALICE (female, weary)" in prompt
        assert "BOB (male, anxious)" in prompt

    def test_named_entities_block_replaces_allowed_names_when_split_set(self):
        # v4 Commit 1: when allowed_people / allowed_things are
        # populated, render NAMED ENTITIES split blocks instead of
        # the legacy ALLOWED NAMES line.
        req = LineRequest(
            speaker="ALICE", intent="reveal", mood="tense", target_words=15,
            canon_header="x", last_lines=[],
            allowed_people=frozenset({"ALICE", "BOB"}),
            allowed_things=frozenset({"CERN", "JPL"}),
            allowed_roster=frozenset({"ALICE", "BOB", "CERN", "JPL", "ANNOUNCER"}),
        )
        prompt = _build_user_prompt(req)
        assert "NAMED ENTITIES IN THIS WORLD" in prompt
        assert "People: ALICE, BOB" in prompt
        assert "Places, agencies, things: CERN, JPL" in prompt
        assert "Generic roles" in prompt
        # Legacy ALLOWED NAMES line does NOT also render.
        assert "ALLOWED NAMES" not in prompt

    def test_current_beat_block_renders_when_set(self):
        req = LineRequest(
            speaker="ALICE", intent="reveal", mood="tense", target_words=15,
            canon_header="x", last_lines=[],
            current_beat_block="CURRENT BEAT\n  b003 ALICE (tense): reveal",
        )
        prompt = _build_user_prompt(req)
        assert "CURRENT BEAT" in prompt
        assert "b003 ALICE (tense): reveal" in prompt

    def test_position_block_renders_when_set(self):
        # v4 Commit 4: POSITION supersedes ARC PHASE.
        req = LineRequest(
            speaker="ALICE", intent="reveal", mood="tense", target_words=15,
            canon_header="x", last_lines=[],
            position="complication, beat 2 of 4. Next phase: climax.",
        )
        prompt = _build_user_prompt(req)
        assert "POSITION: complication, beat 2 of 4. Next phase: climax." in prompt
        # Legacy ARC PHASE block does NOT also render when position is set.
        assert "ARC PHASE" not in prompt

    def test_position_falls_back_to_arc_phase_when_only_arc_phase_set(self):
        # Back-compat: arc_phase set + position empty still renders
        # the legacy ARC PHASE block.
        req = LineRequest(
            speaker="ALICE", intent="reveal", mood="tense", target_words=15,
            canon_header="x", last_lines=[],
            arc_phase="complication",
        )
        prompt = _build_user_prompt(req)
        assert "ARC PHASE: complication" in prompt
        assert "POSITION:" not in prompt

    def test_theme_block_renders_when_set(self):
        # v4 Commit 2: one-sentence theme from meta.news.script_brief.
        req = LineRequest(
            speaker="ALICE", intent="reveal", mood="tense", target_words=15,
            canon_header="x", last_lines=[],
            theme="A signal from the void answers back.",
        )
        prompt = _build_user_prompt(req)
        assert "THEME: A signal from the void answers back." in prompt

    def test_sfx_cue_block_renders_when_set(self):
        # v4 Commit 2: beat.sfx_cue threaded as SOUND IN THE ROOM.
        req = LineRequest(
            speaker="ALICE", intent="reveal", mood="tense", target_words=15,
            canon_header="x", last_lines=[],
            sfx_cue="distant klaxon",
        )
        prompt = _build_user_prompt(req)
        assert "SOUND IN THE ROOM: distant klaxon" in prompt

    def test_role_induction_responds_to_when_prev_speaker_set(self):
        req = LineRequest(
            speaker="ALICE", intent="reveal", mood="tense", target_words=15,
            canon_header="x", last_lines=[("BOB", "Hi.")],
            prev_speaker="BOB",
        )
        prompt = _build_user_prompt(req)
        assert "You are ALICE. You are responding to BOB." in prompt

    def test_role_induction_no_responding_clause_when_prev_speaker_empty(self):
        req = LineRequest(
            speaker="ALICE", intent="reveal", mood="tense", target_words=15,
            canon_header="x", last_lines=[],
            prev_speaker="",
        )
        prompt = _build_user_prompt(req)
        assert "You are ALICE." in prompt
        assert "responding to" not in prompt

    def test_role_induction_drops_responding_clause_when_prev_is_self(self):
        # Edge case: rolling window's last entry is the same speaker
        # (two-line monologue). Drop the "responding to" clause to
        # avoid "You are ALICE. You are responding to ALICE."
        req = LineRequest(
            speaker="ALICE", intent="reveal", mood="tense", target_words=15,
            canon_header="x", last_lines=[("ALICE", "I started.")],
            prev_speaker="ALICE",
        )
        prompt = _build_user_prompt(req)
        assert "You are ALICE." in prompt
        assert "responding to" not in prompt

    def test_static_blocks_precede_variable_blocks_for_kv_cache(self):
        """STATIC = style + theme + canon + entities + cast + spine (cached prefix).
        VARIABLE = current_beat + position + sfx + last_spoken + write_line
                   (changes per call).

        For KV-cache reuse to hit, every STATIC element must appear
        BEFORE every VARIABLE element. This test pins that ordering
        so a future refactor can't quietly reshuffle the prompt and
        destroy the cache hit.
        """
        req = LineRequest(
            speaker="ALICE", intent="reveal", mood="tense", target_words=15,
            canon_header="TITLE: x", last_lines=[("BOB", "Hi.")],
            style_descriptor="noir_interrogation",
            outline_spine="OUTLINE:\n  b001 BOB (worried): warn\n  b002 ALICE (tense): reveal",
            character_voice_card="ALICE (female, weary)",
            allowed_roster=frozenset({"ALICE", "BOB", "ANNOUNCER"}),
            allowed_people=frozenset({"ALICE", "BOB"}),
            allowed_things=frozenset({"CERN"}),
            all_voice_cards="ALICE (female, weary)\nBOB (male, anxious)",
            theme="The voice on the wire is not who it claims to be.",
            current_beat_block="CURRENT BEAT\n  b002 ALICE (tense): reveal",
            position="complication, beat 1 of 2. Next phase: resolution.",
            sfx_cue="distant klaxon",
            prev_speaker="BOB",
        )
        prompt = _build_user_prompt(req)
        static_blocks = ["STYLE:", "THEME:", "EPISODE CONTEXT",
                         "NAMED ENTITIES", "CAST", "OUTLINE:"]
        variable_blocks = ["CURRENT BEAT", "POSITION:",
                           "SOUND IN THE ROOM",
                           "LAST SPOKEN (this scene):", "WRITE LINE"]
        static_positions = [prompt.find(b) for b in static_blocks]
        variable_positions = [prompt.find(b) for b in variable_blocks]
        assert all(p > -1 for p in static_positions), (
            f"missing static blocks: "
            f"{dict(zip(static_blocks, static_positions))}"
        )
        assert all(p > -1 for p in variable_positions), (
            f"missing variable blocks: "
            f"{dict(zip(variable_blocks, variable_positions))}"
        )
        max_static = max(static_positions)
        min_variable = min(variable_positions)
        assert max_static < min_variable, (
            f"prompt ordering violated KV-cache layout: last static "
            f"block at {max_static}, first variable block at "
            f"{min_variable}"
        )

    def test_allowed_roster_sorted_for_byte_stable_prefix(self):
        # Same roster contents, different construction order -> same
        # prompt string. Otherwise the cached prefix would diverge
        # per call (and per run) on tail-end variations.
        prompt_a = _build_user_prompt(LineRequest(
            speaker="ALICE", intent="x", mood="x", target_words=10,
            canon_header="x", last_lines=[],
            allowed_roster=frozenset(["ALICE", "BOB", "CERN"]),
        ))
        prompt_b = _build_user_prompt(LineRequest(
            speaker="ALICE", intent="x", mood="x", target_words=10,
            canon_header="x", last_lines=[],
            allowed_roster=frozenset(["CERN", "ALICE", "BOB"]),
        ))
        assert prompt_a == prompt_b

    def test_last_lines_renders_window(self):
        five_lines = [
            ("ALICE", f"line {i}") for i in range(5)
        ]
        req = LineRequest(
            speaker="ALICE", intent="x", mood="x", target_words=10,
            canon_header="x", last_lines=five_lines,
        )
        prompt = _build_user_prompt(req)
        for i in range(5):
            assert f"line {i}" in prompt

    def test_empty_last_lines_uses_placeholder(self):
        req = LineRequest(
            speaker="ALICE", intent="x", mood="x", target_words=10,
            canon_header="x", last_lines=[],
        )
        prompt = _build_user_prompt(req)
        # v4: placeholder phrasing updated to "scene just opened".
        assert "scene just opened" in prompt


# ---------------------------------------------------------------------------
# Sliding-window cap (orchestrator-side; mirrored here as a contract test)
# ---------------------------------------------------------------------------


class TestSlidingWindowConstant:

    def test_writer_uses_window_of_5(self):
        """Synthesis §6.D requires N=5 in the writer. This pins the
        constant so a future refactor that drops it to 3 (or removes
        it) fails this test rather than silently shrinking the
        composer's context window."""
        from nodes import OTR_LedgerScriptWriter as W  # noqa: N812
        assert W.LAST_LINES_WINDOW == 5
