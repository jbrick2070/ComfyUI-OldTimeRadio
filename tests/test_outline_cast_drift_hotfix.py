"""tests/test_outline_cast_drift_hotfix.py -- BUG-LOCAL-259 outline
cast-drift crash HOTFIX (2026-05-23).

Locks the ROADMAP "## HOTFIX -- Outline cast-drift crash" contract for
nodes/_otr_outline.py:

  Step 1  singleton-cast bypass -- a 1-character locked cast skips the
          Stage 2 speaker LLM call entirely; the sole name is stamped
          on every beat deterministically.
  Step 2  deterministic no-crash fallback -- when the Stage 2 retry
          budget is exhausted for a multi-character cast,
          generate_outline builds a round-robin skeleton instead of
          raising OutlineFailedError. A cast-membership miss never
          crashes a run.
  Step 3  the Stage 2 LLM call samples low -- a low base plus a lower
          structural retry via the structured_call ladder, not the
          legacy rising schedule.
  Step 5  emitted speakers are normalized (uppercase, strip stray
          surrounding punctuation) before the cast-membership check;
          this is NOT broad edit-distance fuzzy matching.

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
    compute_episode_budget,
)
from nodes._otr_outline import (  # noqa: E402
    Outline,
    OutlineRequest,
    _STAGE2_BASE_TEMPERATURE,
    _STAGE2_STRUCTURAL_RETRY_TEMPERATURE,
    _deterministic_phase_skeleton,
    _normalize_speaker,
    generate_outline,
)


# ---------------------------------------------------------------------------
# Stage-aware mock generate_fn
# ---------------------------------------------------------------------------

_MACRO_JSON = json.dumps({
    "title": "Cast Drift Hotfix Test",
    "premise": "A premise about a faint science signal and what it costs.",
    "setting": "A quiet observatory lab",
    "time_of_day": "midnight",
})
_BEAT_JSON = json.dumps({
    "intent": "advance the scene toward the next turn",
    "mood": "tense",
})


def _stage_of(messages):
    """Return 'macro' | 'phase' | 'beat' from the system prompt."""
    system = messages[0]["content"]
    if "You plan one phase" in system:
        return "phase"
    if "You flesh out one beat" in system:
        return "beat"
    return "macro"


def _requested_beat_count(messages):
    """Parse the per-phase beat count from a Stage 2 user prompt."""
    user = messages[1]["content"]
    m = re.search(r"Beats to plan in this phase: (\d+)", user)
    return int(m.group(1)) if m else 1


def _make_gen(phase_speaker_fn, *, record=None):
    """Build a stage-aware mock generate_fn.

    phase_speaker_fn(n) -> list[str]: the speaker names the Stage 2
    call emits for an n-beat phase. record (optional list) collects
    (stage, temperature) per call.
    """
    def _gen(messages, *, temperature, max_new_tokens):
        stage = _stage_of(messages)
        if record is not None:
            record.append((stage, temperature))
        if stage == "phase":
            n = _requested_beat_count(messages)
            speakers = phase_speaker_fn(n)
            return json.dumps(
                {"beats": [{"speaker": s} for s in speakers]}
            )
        if stage == "beat":
            return _BEAT_JSON
        return _MACRO_JSON
    return _gen


# 350 words at the matching default act count (3) is the proven-
# satisfiable budget shape -- see tests/test_phase2a_episode_budget.py
# test_clean_outline_passes. Pairing target_words with
# default_act_count is exactly what OTR_LedgerScriptWriter does in
# production; a mismatched pair (e.g. 150 words forced into 3 acts)
# yields an internally unsatisfiable per-phase word / beat budget.
_TARGET_WORDS = 350
# 2026-08-14: the act count replaced `_TARGET_WORDS` as the thing that
# shapes an episode. Three acts is the default shape.
_ACT_COUNT = 3


def _budget(num_characters):
    return compute_episode_budget(
        _ACT_COUNT, True,
        num_characters,
    )


def _request(cast):
    return OutlineRequest(
        news_seed="a real science seed", style="noir",
        character_cast=cast,
        budget=_budget(len(cast)),
    )


# ---------------------------------------------------------------------------
# Step 5 -- speaker normalization (unit)
# ---------------------------------------------------------------------------
class TestNormalizeSpeaker:

    @pytest.mark.parametrize("raw,expected", [
        ("LEMMY", "LEMMY"),
        (" LEMMY ", "LEMMY"),
        ("LEMMY:", "LEMMY"),
        ('"LEMMY"', "LEMMY"),
        ("'LEMMY'", "LEMMY"),
        ("(LEMMY)", "LEMMY"),
        ("[LEMMY]", "LEMMY"),
        ("LEMMY.", "LEMMY"),
        ("- LEMMY -", "LEMMY"),
        ("lemmy", "LEMMY"),
        ("  alice:  ", "ALICE"),
    ])
    def test_strips_edge_punctuation_and_uppercases(self, raw, expected):
        assert _normalize_speaker(raw) == expected

    def test_internal_punctuation_preserved(self):
        # Only EDGE punctuation is stripped; internal punctuation stays.
        assert _normalize_speaker("DR. LEMMY") == "DR. LEMMY"

    def test_hallucination_not_snapped(self):
        # Normalization is NOT fuzzy matching: a genuinely wrong name
        # (including the bug's 'LEMMEY' typo) normalizes to itself and
        # still misses the cast. The typo class is recovered by the
        # step 1 bypass / step 2 fallback, not by normalization.
        assert _normalize_speaker("CAPTAIN") == "CAPTAIN"
        assert _normalize_speaker("LEMMEY") == "LEMMEY"


# ---------------------------------------------------------------------------
# Step 3 -- Stage 2 temperature (structured_call ladder, Sprint 2A/2D)
# ---------------------------------------------------------------------------
class TestStage2Temperature:

    def test_stage2_temperatures_are_low_and_retry_lowers(self):
        # Stage 2 is structured constraint-routing, not creative prose:
        # both the base and the structural-retry temperature stay low,
        # and the structural retry LOWERS temperature (the 2B
        # principle), never raises it.
        assert _STAGE2_BASE_TEMPERATURE <= 0.5
        assert _STAGE2_STRUCTURAL_RETRY_TEMPERATURE <= 0.5
        assert (
            _STAGE2_STRUCTURAL_RETRY_TEMPERATURE < _STAGE2_BASE_TEMPERATURE
        ), "the structural retry must sample more conservatively, not less"

    def test_generate_outline_routes_stage2_at_low_base(self):
        # End-to-end: generate_outline runs the Stage 2 phase call at
        # _STAGE2_BASE_TEMPERATURE via the structured_call ladder, not
        # the 0.70 creative base_temperature.
        record = []
        generate_outline(
            _make_gen(lambda n: ["ALICE"] * n, record=record),
            _request(("ALICE", "BOB")),
        )
        phase_temps = [t for s, t in record if s == "phase"]
        assert phase_temps, "no Stage 2 phase call was recorded"
        assert phase_temps[0] == pytest.approx(_STAGE2_BASE_TEMPERATURE)


# ---------------------------------------------------------------------------
# Step 1 -- singleton-cast bypass
# ---------------------------------------------------------------------------
class TestSingletonBypass:

    def test_singleton_cast_skips_stage2_and_uses_sole_name(self):
        record = []
        outline = generate_outline(
            _make_gen(lambda n: ["LEMMY"] * n, record=record),
            _request(("LEMMY",)),
        )
        stages = [s for s, _ in record]
        assert "phase" not in stages, (
            f"a singleton cast must skip the Stage 2 phase LLM call; "
            f"stages invoked: {stages}"
        )
        assert "macro" in stages and "beat" in stages, (
            f"macro + beat stages should still run; stages: {stages}"
        )
        assert isinstance(outline, Outline)
        char_speakers = {
            b.speaker for b in outline.beats
            if b.speaker_role == "character"
        }
        assert char_speakers == {"LEMMY"}

    def test_deterministic_skeleton_singleton_uses_sole_name(self):
        skel = _deterministic_phase_skeleton(3, ("LEMMY",))
        assert [b.speaker for b in skel.beats] == ["LEMMY"] * 3


# ---------------------------------------------------------------------------
# Step 2 -- deterministic no-crash fallback
# ---------------------------------------------------------------------------
class TestDeterministicFallback:

    def test_persistent_cast_drift_does_not_raise(self):
        # Every Stage 2 attempt emits an off-cast speaker (CAROL).
        # Pre-hotfix this raised OutlineFailedError; now it must fall
        # back to a deterministic skeleton and complete the outline.
        outline = generate_outline(
            _make_gen(lambda n: ["CAROL"] * n),
            _request(("ALICE", "BOB")),
            max_attempts=3,
        )
        assert isinstance(outline, Outline)
        char_speakers = {
            b.speaker for b in outline.beats
            if b.speaker_role == "character"
        }
        assert char_speakers, "outline has no character beats"
        assert char_speakers <= {"ALICE", "BOB"}, (
            f"deterministic fallback leaked off-cast speakers: "
            f"{char_speakers!r}"
        )

    def test_fallback_round_robins_the_sorted_cast(self):
        # The deterministic skeleton cycles sorted(locked_cast) by beat
        # position -- input order does not matter.
        skel = _deterministic_phase_skeleton(4, ("BOB", "ALICE"))
        assert [b.speaker for b in skel.beats] == [
            "ALICE", "BOB", "ALICE", "BOB",
        ]

    def test_deterministic_skeleton_is_seed_stable(self):
        a = _deterministic_phase_skeleton(5, ("ALICE", "BOB", "CARL"))
        b = _deterministic_phase_skeleton(5, ("ALICE", "BOB", "CARL"))
        assert [x.speaker for x in a.beats] == [
            x.speaker for x in b.beats
        ]

    def test_deterministic_skeleton_beat_count(self):
        skel = _deterministic_phase_skeleton(7, ("ALICE", "BOB"))
        assert len(skel.beats) == 7


# ---------------------------------------------------------------------------
# Step 5 -- speaker normalization (integration)
# ---------------------------------------------------------------------------
class TestSpeakerNormalizationIntegration:

    def test_punctuated_speaker_accepted_without_retry(self):
        # Stage 2 emits punctuated names ('ALICE:', 'BOB:'). They are
        # not literally in the cast but normalize cleanly into it, so
        # each phase passes on attempt 1 -- one Stage 2 call per phase,
        # no retries, no deterministic fallback.
        record = []
        req = _request(("ALICE", "BOB"))

        def _punctuated(n):
            pool = ["ALICE:", "BOB:"]
            return [pool[i % 2] for i in range(n)]

        outline = generate_outline(
            _make_gen(_punctuated, record=record), req,
        )
        n_phases = len(req.budget.arc_phases)
        phase_calls = sum(1 for s, _ in record if s == "phase")
        assert phase_calls == n_phases, (
            f"punctuated speakers should normalize and pass on "
            f"attempt 1 -- expected {n_phases} Stage 2 calls (one per "
            f"phase), saw {phase_calls}"
        )
        for b in outline.beats:
            assert ":" not in b.speaker, (
                f"normalization must strip the stray colon; outline "
                f"carried {b.speaker!r}"
            )
        char_speakers = {
            b.speaker for b in outline.beats
            if b.speaker_role == "character"
        }
        assert char_speakers <= {"ALICE", "BOB"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
