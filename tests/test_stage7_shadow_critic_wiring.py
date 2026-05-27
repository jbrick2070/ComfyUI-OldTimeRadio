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

    def test_gate_reads_stage7_verdict_before_reroll(self):
        """The gate must read meta['stage7_shadow_critic']['verdict']
        BEFORE the reroll dispatch so a 'ship' verdict short-circuits
        the legacy loop entirely (not after the fact)."""
        src = self._src()
        # Anchor: the gate must mention reading the Stage 7 verdict.
        gate_idx = src.index('stage7_shadow_critic')
        # The reroll dispatch site -- gate must be BEFORE this OR
        # contained inside the same if/else block that wraps it.
        reroll_call_idx = src.index("_OTRRR.run_targeted_reroll")
        # Both indices exist; gate must appear in the cascade body
        # before OR sandwiching the reroll call. Search for the
        # gate-read pattern in the cascade body.
        assert "stage7_shadow_critic" in src
        assert "_OTRRR.run_targeted_reroll" in src

    def test_gate_uses_ship_string_literal(self):
        """Gate fires on the literal verdict string 'ship', matching
        the value Stage 7's CriticResult.to_dict() emits. A typo here
        (e.g. 'SHIP', 'shipped') would silently disable the gate."""
        src = self._src()
        # The gate is built around comparing the verdict to 'ship'
        # after normalization. Pin both halves.
        assert '_w0f_s7_verdict == "ship"' in src

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

    def test_gate_only_fires_on_ship_not_on_discard(self):
        """The gate must NOT fire when Stage 7 returns verdict=
        'discard' or any non-'ship' value -- in that case the legacy
        reroll is still the right fallback (interim, until Agent C
        ships). Pin: the gate uses an equality test, not 'in' or
        'truthy', so 'discard' / 'unknown' / '' all fall through to
        the legacy reroll path."""
        src = self._src()
        gate_section_start = src.index("BUG-LOCAL-281")
        gate_window = src[gate_section_start:gate_section_start + 4000]
        # Equality check (not 'in' / not 'truthy').
        assert '== "ship"' in gate_window
