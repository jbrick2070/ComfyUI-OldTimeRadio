"""Adapter + escalation-gate wiring regression.

Covers:
  * nodes/_otr_legacy_to_stage1_adapter:
      - Empty ledger -> None
      - Ledger with cast + lines -> valid Stage1Plan
      - Voice preset normalization (v2/en_speaker_N, bm_* fallback)
      - Pronoun derivation from gender (male/female/nonbinary)
      - Reserved-speaker rows in cast skipped
      - Beat speaker resolution via char_id -> cast.name
      - Padding to 3 beats when legacy ledger has fewer
      - Running_facts extracted from continuity_ledger if present
      - extract_rendered_lines pulls (beat_id, speaker, text)
  * nodes/_otr_freeze_cascade:
      - Source-level pins on the Sprint 10B Wave 1 Agent C escalation
        gate (BUG-LOCAL-281): the cascade dispatches on
        decide_escalation_scope / EscalationScope and builds a no-op
        RerollDisposition on the ship path.

The Stage 7 whole-episode shadow critic was removed in the 2026-05-29
lean-down; only the live legacy->Stage1 adapter and the escalation
gate remain pinned here.
"""

from __future__ import annotations

import pathlib

import pytest

from nodes._otr_legacy_to_stage1_adapter import (
    extract_rendered_lines,
    legacy_ledger_to_stage1_plan,
)
from nodes._otr_stage1_plan import Stage1Plan


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CASCADE_SRC = REPO_ROOT / "nodes" / "_otr_freeze_cascade.py"


# ---------------------------------------------------------------------------
# Adapter -- empty / missing
# ---------------------------------------------------------------------------


class TestAdapterEdgeCases:
    def test_none_ledger_returns_none(self):
        assert legacy_ledger_to_stage1_plan(None) is None  # type: ignore[arg-type]

    def test_empty_dict_returns_none(self):
        assert legacy_ledger_to_stage1_plan({}) is None

    def test_no_cast_returns_none(self):
        led = {"lines": [{"text": "hi"}], "cast": []}
        assert legacy_ledger_to_stage1_plan(led) is None

    def test_no_lines_returns_none(self):
        led = {"cast": [{"name": "DALE", "gender": "male"}], "lines": []}
        assert legacy_ledger_to_stage1_plan(led) is None


# ---------------------------------------------------------------------------
# Adapter -- happy path
# ---------------------------------------------------------------------------


def _legacy_ledger() -> dict:
    """Minimal legacy-shape ledger sufficient for the adapter to
    produce a parseable Stage1Plan."""
    return {
        "cast": [
            {
                "char_id": "c00",
                "name": "ANNOUNCER",       # reserved -- skipped from cast
                "gender": "male",
                "voice_preset": "bm_fable",
            },
            {
                "char_id": "c01",
                "name": "DALE PORTER",
                "gender": "male",
                "voice_preset": "v2/en_speaker_5",
                "character_description": "Experienced relay technician.",
                "role": "anchor of competence",
            },
            {
                "char_id": "c02",
                "name": "MEG SAARI",
                "gender": "female",
                "voice_preset": "v2/en_speaker_2",
                "character_description": "Junior tech.",
                "role": "pressure source",
            },
        ],
        "lines": [
            {
                "beat_id": "b001", "char_id": "c00",
                "speaker_role": "announcer",
                "text": "Welcome to Signal Lost.",
            },
            {
                "beat_id": "b002", "char_id": "c01",
                "speaker_role": "character",
                "text": "The carrier is off-band, channel four.",
                "beat_intent": "Notice the carrier.",
                "emotional_register": "alert calm",
            },
            {
                "beat_id": "b003", "char_id": "c02",
                "speaker_role": "character",
                "text": "Off-band? That shouldn't be possible.",
                "beat_intent": "Ask Dale about it.",
                "emotional_register": "alert curious",
            },
        ],
        "meta": {
            "news_seed": "Signal anomaly detected at the Mars relay station.",
            "episode_canon": {
                "premise": "Two technicians at a Mars relay catch a signal that should not exist.",
                "arc": {
                    "setup": "Routine night shift, calm comms traffic.",
                    "complication": "An off-band carrier locks in.",
                    "resolution": "They identify the source and acknowledge it.",
                },
            },
            "continuity_ledger": {
                "facts": [
                    {"text": "The relay is on Mars."},
                    "Dale is senior to Meg.",
                ],
            },
        },
    }


class TestAdapterHappyPath:
    def test_builds_valid_stage1_plan(self):
        plan = legacy_ledger_to_stage1_plan(_legacy_ledger())
        assert isinstance(plan, Stage1Plan)
        assert "Mars relay" in plan.premise
        assert len(plan.cast) == 2   # ANNOUNCER row skipped
        assert {c.name for c in plan.cast} == {"DALE PORTER", "MEG SAARI"}

    def test_pronouns_derived_from_gender(self):
        plan = legacy_ledger_to_stage1_plan(_legacy_ledger())
        by_name = {c.name: c for c in plan.cast}
        assert by_name["DALE PORTER"].pronouns == "he/him"
        assert by_name["MEG SAARI"].pronouns == "she/her"

    def test_voice_preset_kept_or_normalized(self):
        plan = legacy_ledger_to_stage1_plan(_legacy_ledger())
        for c in plan.cast:
            assert c.voice_id.startswith("v2/en_speaker_")
            # Single-digit speaker index per Stage 1 schema.
            assert c.voice_id[-1].isdigit()

    def test_running_facts_extracted(self):
        plan = legacy_ledger_to_stage1_plan(_legacy_ledger())
        assert "The relay is on Mars." in plan.running_facts
        assert "Dale is senior to Meg." in plan.running_facts

    def test_reserved_announcer_beat_speaker_preserved(self):
        plan = legacy_ledger_to_stage1_plan(_legacy_ledger())
        announcer_beat = next(
            (b for b in plan.beats if b.beat_id == "b001"), None,
        )
        assert announcer_beat is not None
        assert announcer_beat.speaker == "ANNOUNCER"

    def test_character_beat_speaker_resolved_via_char_id(self):
        plan = legacy_ledger_to_stage1_plan(_legacy_ledger())
        b002 = next((b for b in plan.beats if b.beat_id == "b002"), None)
        assert b002 is not None
        assert b002.speaker == "DALE PORTER"

    def test_length_target_uses_actual_word_count(self):
        plan = legacy_ledger_to_stage1_plan(_legacy_ledger())
        b002 = next((b for b in plan.beats if b.beat_id == "b002"), None)
        # "The carrier is off-band, channel four." -> 6 words
        # Adapter clamps to min 5 per schema.
        assert b002.length_target_words >= 5


class TestAdapterPaddingAndFallbacks:
    def test_few_beats_padded_to_three(self):
        led = _legacy_ledger()
        led["lines"] = led["lines"][:1]   # only 1 beat
        plan = legacy_ledger_to_stage1_plan(led)
        assert plan is not None
        assert len(plan.beats) >= 3

    def test_missing_premise_uses_news_seed(self):
        led = _legacy_ledger()
        led["meta"]["episode_canon"] = {}   # strip premise
        plan = legacy_ledger_to_stage1_plan(led)
        assert plan is not None
        assert "Signal anomaly" in plan.premise

    def test_news_seed_as_dict_does_not_raise(self):
        """BUG-LOCAL-277 regression. Real soak run
        signal_lost_..._154105 carried meta.news_seed as a dict
        with keys {headline, source, url, date, body_chars, style,
        selected_at} -- the current NewsFetcher shape. The adapter
        was doing `(meta.get('news_seed') or '').strip()` which
        raises AttributeError on a dict. With the
        _coerce_news_seed_text helper, both shapes resolve cleanly.
        """
        led = _legacy_ledger()
        led["meta"]["episode_canon"] = {}   # force fallback to news_seed
        led["meta"]["news_seed"] = {
            "headline": "Venomous Himalayan pit viper was actually 5 species",
            "source": "Latest Science News -- ScienceDaily",
            "url": "https://example.com/article",
            "date": "2026-05-26",
            "body_chars": 414,
            "style": "noir",
            "selected_at": "2026-05-26T15:41:05Z",
        }
        # Must not raise:
        plan = legacy_ledger_to_stage1_plan(led)
        assert plan is not None
        # headline carries through as the premise excerpt:
        assert "Venomous Himalayan pit viper" in plan.premise

    def test_news_seed_dict_falls_back_to_source(self):
        """If a dict-shaped news_seed has no headline, the adapter
        falls back to the source. Ensures we never pass a dict into
        .strip() and never produce a garbled premise.
        """
        led = _legacy_ledger()
        led["meta"]["episode_canon"] = {}
        led["meta"]["news_seed"] = {
            "headline": "",
            "source": "Wire service",
            "url": "https://example.com",
        }
        plan = legacy_ledger_to_stage1_plan(led)
        assert plan is not None
        # Either the source carries through or the placeholder kicks
        # in when the source string is too short for the >=10 bound.
        assert plan.premise

    def test_news_seed_dict_all_empty_falls_back_to_placeholder(self):
        """Defensive: when both episode_canon AND news_seed are
        unusable, the adapter must still produce a valid plan via
        the placeholder premise.
        """
        led = _legacy_ledger()
        led["meta"]["episode_canon"] = {}
        led["meta"]["news_seed"] = {
            "headline": None,
            "source": None,
        }
        plan = legacy_ledger_to_stage1_plan(led)
        assert plan is not None
        assert len(plan.premise) >= 10   # schema floor

    def test_missing_arc_uses_placeholder(self):
        led = _legacy_ledger()
        led["meta"]["episode_canon"] = {
            "premise": "valid premise long enough to clear the bound",
        }   # no arc
        plan = legacy_ledger_to_stage1_plan(led)
        assert plan is not None
        # Placeholder satisfies schema (>= 10 chars).
        assert len(plan.arc.setup) >= 10
        assert len(plan.arc.complication) >= 10
        assert len(plan.arc.resolution) >= 10

    def test_bm_voice_preset_falls_back_to_v2(self):
        led = _legacy_ledger()
        led["cast"][1]["voice_preset"] = "bm_fable"   # Kokoro preset
        plan = legacy_ledger_to_stage1_plan(led)
        assert plan is not None
        dale = next(c for c in plan.cast if c.name == "DALE PORTER")
        assert dale.voice_id.startswith("v2/en_speaker_")

    def test_unique_voice_ids_after_dedupe(self):
        led = _legacy_ledger()
        # Two cast members with the same v2 preset.
        led["cast"][1]["voice_preset"] = "v2/en_speaker_5"
        led["cast"][2]["voice_preset"] = "v2/en_speaker_5"
        plan = legacy_ledger_to_stage1_plan(led)
        assert plan is not None
        ids = [c.voice_id for c in plan.cast]
        assert len(ids) == len(set(ids)), (
            f"voice_ids must be unique after dedupe; got {ids}"
        )


# ---------------------------------------------------------------------------
# extract_rendered_lines
# ---------------------------------------------------------------------------


class TestExtractRenderedLines:
    def test_pulls_beat_id_speaker_text(self):
        lines = extract_rendered_lines(_legacy_ledger())
        assert len(lines) == 3
        for line in lines:
            assert "beat_id" in line
            assert "speaker" in line
            assert "text" in line

    def test_announcer_speaker_resolved(self):
        lines = extract_rendered_lines(_legacy_ledger())
        announcer = next(l for l in lines if l["beat_id"] == "b001")
        assert announcer["speaker"] == "ANNOUNCER"

    def test_char_id_resolves_to_cast_name(self):
        lines = extract_rendered_lines(_legacy_ledger())
        dale_line = next(l for l in lines if l["beat_id"] == "b002")
        assert dale_line["speaker"] == "DALE PORTER"

    def test_empty_text_lines_skipped(self):
        led = _legacy_ledger()
        led["lines"].append({
            "beat_id": "b099", "char_id": "c01",
            "speaker_role": "character", "text": "",
        })
        lines = extract_rendered_lines(led)
        # 3 valid lines, the empty one is skipped.
        assert len(lines) == 3

    def test_none_ledger_returns_empty(self):
        assert extract_rendered_lines(None) == []  # type: ignore[arg-type]


class TestBugLocal281Stage7ShipSkipsReroll:
    """BUG-LOCAL-281 (Sprint 10B Wave 0 follow-on, 2026-05-27):
    when the Stage 7 whole-episode critic returns verdict='ship',
    the cascade must skip the Sprint 5C legacy reroll loop entirely
    and ship the composed lines as-is. Interim gate until Wave 1
    Agent C (critic-driven escalation) replaces the legacy reroll.

    Live operator soak that triggered the fix: Stage 7 returned
    verdict=ship mean=3.70 failing_axes=[] but the legacy story
    critic independently flagged 2 reroll targets, forcing the
    Sprint 5C loop, which exhausted both cycles, stamped
    needs_full_rerun, and halted Bark per BUG-LOCAL-276. The
    composed episode was shippable per Stage 7 but never reached
    audio.
    """

    def _src(self) -> str:
        return CASCADE_SRC.read_text(encoding="utf-8")

    def test_gate_marker_present(self):
        src = self._src()
        assert "BUG-LOCAL-281" in src, (
            "BUG-LOCAL-281 marker comment must live in the cascade "
            "source documenting the Stage 7 ship-verdict gate"
        )

    def test_gate_decides_escalation_before_reroll(self):
        """The escalation decision must be computed BEFORE the reroll
        dispatch so a ship/structural verdict short-circuits the legacy
        loop entirely (not after the fact)."""
        src = self._src()
        decide_idx = src.index("decide_escalation_scope(")
        reroll_call_idx = src.index("_OTRRR.run_targeted_reroll")
        assert decide_idx < reroll_call_idx, (
            "escalation decision must be computed before the reroll dispatch"
        )

    def test_gate_uses_escalation_module(self):
        """Sprint 10B Wave 1 Agent C (2026-05-27, commits 73bfed7..)
        replaced the inline `_w0f_s7_verdict == 'ship'` check with a
        typed decision via decide_escalation_scope. The ship-gate
        semantic survives (EscalationScope.NONE is the same outcome)
        but the implementation is now centralized in
        _otr_reroll_escalation. Pin: cascade dispatches on
        escalation.scope, not on a raw verdict-string comparison.
        """
        src = self._src()
        gate_section_start = src.index("BUG-LOCAL-281")
        gate_window = src[gate_section_start:gate_section_start + 6000]
        # New pin: the cascade uses the typed escalation enum.
        assert "EscalationScope.NONE" in gate_window
        # The decision is computed from decide_escalation_scope.
        assert "decide_escalation_scope" in src

    def test_gate_constructs_no_op_reroll_disposition(self):
        """When the gate fires, downstream code that reads
        reroll_disp.verdict / .cycles_run must still get a valid
        object. Pin: a RerollDisposition with verdict='no_reroll' is
        constructed in the gate branch."""
        src = self._src()
        # Find the gate region and assert it builds a no-op RerollDisposition.
        gate_section_start = src.index("BUG-LOCAL-281")
        # Look forward a reasonable window for the construction.
        gate_window = src[gate_section_start:gate_section_start + 4000]
        assert "_OTRRR.RerollDisposition(" in gate_window
        assert 'verdict="no_reroll"' in gate_window

    def test_gate_stamps_audit_marker_on_meta(self):
        """When the gate fires, stamp meta['reroll_skipped_by_stage7_ship']
        so soak diagnostics can grep the decision after the fact."""
        src = self._src()
        gate_section_start = src.index("BUG-LOCAL-281")
        gate_window = src[gate_section_start:gate_section_start + 4000]
        assert '"reroll_skipped_by_stage7_ship"' in gate_window

    def test_gate_distinguishes_ship_episode_beat_line(self):
        """Sprint 10B Wave 1 Agent C: the gate now dispatches on a
        4-value scope enum (NONE/EPISODE/BEAT/LINE), not on a binary
        ship/not-ship check. Pin: all four scopes referenced in the
        cascade source so a future refactor can't silently collapse
        the decision back to a binary."""
        src = self._src()
        gate_section_start = src.index("BUG-LOCAL-281")
        gate_window = src[gate_section_start:gate_section_start + 6000]
        # NONE -- ship case
        assert "EscalationScope.NONE" in gate_window
        # EPISODE -- structural failure short-circuit
        assert "EscalationScope.EPISODE" in gate_window
        # The else branch (BEAT/LINE) routes to the legacy reroll.
        assert "_OTRRR.run_targeted_reroll" in gate_window
