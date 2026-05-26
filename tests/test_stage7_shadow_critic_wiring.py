"""Sprint 10A step 7 wiring regression -- adapter + cascade integration.

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
      - Source-level pin: Stage 7 shadow block lives AFTER the
        legacy story_critic stamp and BEFORE the reroll loop.
      - Source-level pin: shadow block gated on
        meta['stage1_shadow_attempts'].
      - Source-level pin: catch-all Exception arm present + stamps
        shadow_setup_failed marker.

Tests do NOT exercise the runtime cascade (would need a full
ledger fixture + LLM mock); the wiring is a thin diagnostic
addition that we pin via source-level inspection.
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


# ---------------------------------------------------------------------------
# Cascade wiring -- source-level pins
# ---------------------------------------------------------------------------


class TestCascadeWiringSource:
    def _src(self) -> str:
        return CASCADE_SRC.read_text(encoding="utf-8")

    def test_step7_shadow_block_marker_present(self):
        src = self._src()
        assert "Sprint 10A step 7: whole-episode shadow critic" in src

    def test_block_runs_after_legacy_critic_stamp(self):
        """Critical placement: the Stage 7 shadow block must live
        AFTER meta['story_critic_report'] = ... and BEFORE the
        Sprint 5C reroll loop. Otherwise the critic sees a stale or
        missing legacy critic report."""
        src = self._src()
        legacy_stamp_idx = src.index(
            'meta["story_critic_report"] = story_critic_report.model_dump()'
        )
        shadow_idx = src.index("Sprint 10A step 7: whole-episode shadow critic")
        reroll_idx = src.index("Sprint 5C: targeted reroll loop")
        assert legacy_stamp_idx < shadow_idx < reroll_idx, (
            "Stage 7 shadow block must be sandwiched between the "
            "legacy critic stamp and the Sprint 5C reroll loop"
        )

    def test_block_gated_on_stage1_shadow_attempts(self):
        """The shadow critic must only run when the operator opted
        into shadow diagnostics (writer stamped
        meta.stage1_shadow_attempts). This is the 'same widget,
        same opt-in' semantics carried across nodes."""
        src = self._src()
        shadow_idx = src.index("Sprint 10A step 7: whole-episode shadow critic")
        # Block extends from the marker through to the reroll loop;
        # search the whole region so we don't miss the gate behind
        # the long comment header.
        reroll_idx = src.index("Sprint 5C: targeted reroll loop")
        block = src[shadow_idx:reroll_idx]
        assert '"stage1_shadow_attempts" in meta' in block, (
            "Stage 7 shadow block must gate on meta['stage1_shadow_attempts']"
        )

    def test_block_imports_are_local(self):
        """Imports must be local to the branch so workflows with the
        flag off don't pay the import cost."""
        src = self._src()
        shadow_idx = src.index("Sprint 10A step 7: whole-episode shadow critic")
        reroll_idx = src.index("Sprint 5C: targeted reroll loop")
        block = src[shadow_idx:reroll_idx]
        assert "from . import _otr_legacy_to_stage1_adapter" in block
        assert "from . import _otr_whole_episode_critic" in block

    def test_block_has_catch_all_exception_arm(self):
        """A bug in the shadow critic must NEVER halt the cascade.
        Pin via source check: catch-all 'except Exception' present
        in the shadow block."""
        src = self._src()
        shadow_idx = src.index("Sprint 10A step 7: whole-episode shadow critic")
        # The block ends at the next major marker (the reroll loop).
        reroll_idx = src.index("Sprint 5C: targeted reroll loop")
        block = src[shadow_idx:reroll_idx]
        assert "except Exception" in block, (
            "Stage 7 shadow block must carry a catch-all Exception arm "
            "so a bug here never halts the cascade"
        )

    def test_block_stamps_meta_keys(self):
        """The shadow block must stamp meta['stage7_shadow_critic'] in
        every reachable code path so the soak ledger always carries
        a forensic record."""
        src = self._src()
        shadow_idx = src.index("Sprint 10A step 7: whole-episode shadow critic")
        reroll_idx = src.index("Sprint 5C: targeted reroll loop")
        block = src[shadow_idx:reroll_idx]
        assert 'meta["stage7_shadow_critic"]' in block
        # And the failure path stamps shadow_setup_failed.
        assert "shadow_setup_failed" in block
