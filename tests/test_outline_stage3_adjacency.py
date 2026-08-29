"""tests/test_outline_stage3_adjacency.py -- Sprint 3B outline Stage 3
hardening (2026-05-25).

Locks the two Sprint 3B changes to nodes/_otr_outline.py:

  (1) Adjacency context in the per-beat fleshout prompt. The Stage 3
      `_build_beat_user_prompt` used to be fully beat-localized ("NO
      other beat context"). It now carries a 1-beat adjacency window:
        * the previous beat's intent (real, already-generated),
        * the next beat's speaker (the available forward signal --
          Stage 3 is sequential, so the next beat's intent does not
          exist yet),
        * a one-line phase summary from ARC_PHASE_GUIDANCE.
      The first voiced beat has no previous; the last has no next.
      A missing neighbour produces NO line at all (no empty / "None"
      placeholder).

  (2) target_words dropped from the _BeatFleshout LLM schema. Python's
      _allocate_phase_target_words already owned the per-beat word
      count and overrode whatever the LLM emitted, so removing the
      field is a behaviour-neutral token-budget cleanup. The combiner
      now stamps Python's allocation onto Beat.target_words via the
      `beat_allocations` channel; phase word totals are unchanged.

Pure-Python. No GPU. No real LLM -- generate_fn is a stage-aware mock.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_episode_budget import (  # noqa: E402
    ARC_PHASE_GUIDANCE,
    compute_episode_budget,
)
from nodes._otr_outline import (  # noqa: E402
    _BeatFleshout,
    Outline,
    OutlineRequest,
    _build_beat_user_prompt,
    _MacroShape,
    _phase_summary,
    generate_outline,
    validate_outline_against_budget,
)


# ---------------------------------------------------------------------------
# Shared fixtures -- a satisfiable 350-word / 3-act budget shape
# ---------------------------------------------------------------------------

_TARGET_WORDS = 350
# 2026-08-14: the act count replaced `_TARGET_WORDS` as the thing that
# shapes an episode. Three acts is the default shape.
_ACT_COUNT = 3


def _budget(num_characters: int):
    return compute_episode_budget(
        _ACT_COUNT, True,
        num_characters,
    )


def _request(cast: tuple[str, ...]) -> OutlineRequest:
    return OutlineRequest(
        news_seed="a real science seed", style="noir",
        character_cast=cast,
        budget=_budget(len(cast)),
    )


_MACRO = _MacroShape(
    title="Adjacency Test",
    premise="A premise about a faint science signal and what it costs.",
    setting="A quiet observatory lab",
    time_of_day="midnight",
)


# ---------------------------------------------------------------------------
# Change 2 -- _BeatFleshout no longer carries target_words
# ---------------------------------------------------------------------------
class TestBeatFleshoutSchema:

    def test_target_words_field_removed(self):
        assert "target_words" not in _BeatFleshout.model_fields, (
            "Sprint 3B: target_words must be dropped from the "
            "_BeatFleshout LLM schema"
        )

    def test_remaining_fields_are_intent_and_mood(self):
        assert set(_BeatFleshout.model_fields) == {"intent", "mood"}

    def test_constructs_from_intent_and_mood_only(self):
        fleshout = _BeatFleshout(intent="advance the scene", mood="tense")
        assert fleshout.intent == "advance the scene"
        assert fleshout.mood == "tense"

    def test_stray_target_words_key_is_ignored_not_rejected(self):
        # A transitional LLM (or an old fixture) that still emits a
        # target_words key must parse cleanly: pydantic's default
        # extra="ignore" drops the stray key rather than raising.
        fleshout = _BeatFleshout.model_validate(
            {"intent": "advance the scene", "mood": "tense",
             "target_words": 999}
        )
        assert not hasattr(fleshout, "target_words")
        assert fleshout.intent == "advance the scene"

    def test_beat_system_prompt_drops_target_words(self):
        from nodes._otr_outline import _BEAT_SYSTEM_PROMPT
        assert "target_words" not in _BEAT_SYSTEM_PROMPT, (
            "the Stage 3 system prompt must no longer ask for "
            "target_words"
        )
        assert "intent" in _BEAT_SYSTEM_PROMPT
        assert "mood" in _BEAT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Change 1 -- _phase_summary helper
# ---------------------------------------------------------------------------
class TestPhaseSummary:

    @pytest.mark.parametrize("phase", sorted(ARC_PHASE_GUIDANCE))
    def test_known_phase_returns_guidance_line(self, phase):
        assert _phase_summary(phase) == ARC_PHASE_GUIDANCE[phase]

    def test_unknown_phase_falls_back_to_bare_name(self):
        # No exception, no empty string -- a bare phase name is the
        # graceful fallback for a label absent from the table.
        assert _phase_summary("a_phase_not_in_the_table") == (
            "a_phase_not_in_the_table"
        )


# ---------------------------------------------------------------------------
# Change 1 -- adjacency rendering in _build_beat_user_prompt
# ---------------------------------------------------------------------------
class TestBeatPromptAdjacency:

    def test_all_three_adjacency_lines_render_for_a_middle_beat(self):
        prompt = _build_beat_user_prompt(
            _request(("ALICE", "BOB")), _MACRO, "complication", "ALICE",
            (1, 4),
            previous_beat_intent="establish the strange reading",
            next_beat_speaker="BOB",
            phase_summary=ARC_PHASE_GUIDANCE["complication"],
        )
        assert "Previous beat intent: establish the strange reading" in prompt
        assert "Next beat is spoken by: BOB" in prompt
        assert (
            f"Phase focus: {ARC_PHASE_GUIDANCE['complication']}" in prompt
        )

    def test_first_beat_omits_previous_line_entirely(self):
        # No previous neighbour -> NO "Previous beat intent" line at
        # all. Never an empty value, never the literal "None".
        prompt = _build_beat_user_prompt(
            _request(("ALICE", "BOB")), _MACRO, "setup", "ALICE",
            (0, 4),
            previous_beat_intent=None,
            next_beat_speaker="BOB",
            phase_summary=ARC_PHASE_GUIDANCE["setup"],
        )
        assert "Previous beat intent" not in prompt
        assert "None" not in prompt
        # The forward neighbour is still present.
        assert "Next beat is spoken by: BOB" in prompt

    def test_last_beat_omits_next_line_entirely(self):
        prompt = _build_beat_user_prompt(
            _request(("ALICE", "BOB")), _MACRO, "resolution", "BOB",
            (3, 4),
            previous_beat_intent="land the consequence",
            next_beat_speaker=None,
            phase_summary=ARC_PHASE_GUIDANCE["resolution"],
        )
        assert "Next beat is spoken by" not in prompt
        assert "None" not in prompt
        assert "Previous beat intent: land the consequence" in prompt

    def test_lone_beat_omits_both_neighbour_lines(self):
        # A single-beat outline (no previous, no next) renders neither
        # adjacency line -- the phase summary still appears.
        prompt = _build_beat_user_prompt(
            _request(("ALICE",)), _MACRO, "scene", "ALICE",
            (0, 1),
            previous_beat_intent=None,
            next_beat_speaker=None,
            phase_summary=ARC_PHASE_GUIDANCE["scene"],
        )
        assert "Previous beat intent" not in prompt
        assert "Next beat is spoken by" not in prompt
        assert "None" not in prompt
        assert f"Phase focus: {ARC_PHASE_GUIDANCE['scene']}" in prompt

    def test_prompt_no_longer_requests_target_words(self):
        prompt = _build_beat_user_prompt(
            _request(("ALICE", "BOB")), _MACRO, "setup", "ALICE",
            (0, 4),
            previous_beat_intent=None,
            next_beat_speaker="BOB",
            phase_summary=ARC_PHASE_GUIDANCE["setup"],
        )
        assert "target_words" not in prompt


# ---------------------------------------------------------------------------
# Change 1 -- adjacency wired end-to-end through generate_outline
# ---------------------------------------------------------------------------

_MACRO_JSON = json.dumps({
    "title": "Adjacency Wiring Test",
    "premise": "A premise about a faint science signal and what it costs.",
    "setting": "A quiet observatory lab",
    "time_of_day": "midnight",
})


def _stage_of(messages):
    system = messages[0]["content"]
    if "You plan one phase" in system:
        return "phase"
    if "You flesh out one beat" in system:
        return "beat"
    return "macro"


def _make_recording_gen(beat_prompts: list[str]):
    """Stage-aware mock generate_fn that records every Stage 3 user
    prompt into `beat_prompts` and emits a distinct intent per beat so
    the previous-beat-intent wiring is observable in the next prompt.
    """
    counter = {"n": 0}

    def _gen(messages, *, temperature, max_new_tokens):
        stage = _stage_of(messages)
        if stage == "phase":
            user = messages[1]["content"]
            m = re.search(r"Beats to plan in this phase: (\d+)", user)
            n = int(m.group(1)) if m else 1
            return json.dumps(
                {"beats": [{"speaker": "ALICE"} for _ in range(n)]}
            )
        if stage == "beat":
            beat_prompts.append(messages[1]["content"])
            counter["n"] += 1
            return json.dumps({
                "intent": f"distinct intent number {counter['n']}",
                "mood": "tense",
            })
        return _MACRO_JSON

    return _gen


class TestAdjacencyWiredEndToEnd:

    def test_phase_summary_reaches_every_beat_prompt(self):
        beat_prompts: list[str] = []
        generate_outline(
            _make_recording_gen(beat_prompts),
            _request(("ALICE", "BOB")),
        )
        assert beat_prompts, "no Stage 3 beat prompts were recorded"
        for prompt in beat_prompts:
            assert "Phase focus:" in prompt, (
                f"every beat prompt must carry a phase summary; "
                f"missing in:\n{prompt}"
            )

    def test_first_beat_prompt_has_no_previous_intent(self):
        beat_prompts: list[str] = []
        generate_outline(
            _make_recording_gen(beat_prompts),
            _request(("ALICE", "BOB")),
        )
        # The very first Stage 3 call is the outline's first voiced
        # beat -- it has no predecessor.
        assert "Previous beat intent" not in beat_prompts[0], (
            "the outline's first voiced beat must omit the "
            "previous-beat line"
        )

    def test_last_beat_prompt_has_no_next_speaker(self):
        beat_prompts: list[str] = []
        generate_outline(
            _make_recording_gen(beat_prompts),
            _request(("ALICE", "BOB")),
        )
        assert "Next beat is spoken by" not in beat_prompts[-1], (
            "the outline's last voiced beat must omit the "
            "next-beat line"
        )

    def test_later_beat_carries_the_real_previous_intent(self):
        # The mock emits "distinct intent number K" for the K-th beat.
        # The second beat's prompt must quote the first beat's intent
        # verbatim -- proof the adjacency window carries the real,
        # already-generated intent (not a placeholder).
        beat_prompts: list[str] = []
        generate_outline(
            _make_recording_gen(beat_prompts),
            _request(("ALICE", "BOB")),
        )
        assert len(beat_prompts) >= 2
        assert (
            "Previous beat intent: distinct intent number 1"
            in beat_prompts[1]
        ), (
            "the second beat prompt must carry beat 1's real intent; "
            f"got:\n{beat_prompts[1]}"
        )

    def test_no_none_placeholder_leaks_into_any_beat_prompt(self):
        beat_prompts: list[str] = []
        generate_outline(
            _make_recording_gen(beat_prompts),
            _request(("ALICE", "BOB")),
        )
        for prompt in beat_prompts:
            for line in prompt.splitlines():
                if line.startswith(("Previous beat intent:",
                                    "Next beat is spoken by:")):
                    payload = line.split(":", 1)[1].strip()
                    assert payload and payload != "None", (
                        f"adjacency line carried an empty / None "
                        f"payload: {line!r}"
                    )


# ---------------------------------------------------------------------------
# Change 2 -- phase word allocation is unchanged after the schema trim
# ---------------------------------------------------------------------------

# `TestPhaseWordAllocationUnchanged` was DELETED 2026-08-14, not repaired.
# Its two tests existed to prove that Python's per-beat word allocation beat
# whatever number the LLM emitted -- `test_combiner_stamps_python_allocation_not_llm_number` and `test_phase_word_totals_match_the_python_allocation`.
# There is no per-beat word allocation any more: `_allocate_phase_target_words`,
# `Beat.target_words`, `budget.per_phase_words` and `budget.words_per_beat_range`
# were all removed with the word authority. A repaired version of these tests
# would have had to assert something they were never written to say.

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
