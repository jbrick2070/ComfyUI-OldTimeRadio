"""Story-Quality R3 L7 -- dialogue|action split (AUGMENT the scrub).

split_stage_business (_otr_line_hygiene) extracts a leaked trailing/leading
3rd-person action from a quoted line; scrub_ledger records it on the row's
compose_flags (action_split:{json}) under meta.story_quality_v2_enabled and
keeps ONLY the spoken dialogue, then STILL runs the existing strip chain.
get_line_action reads it back. Flag OFF => byte-identical (no split). Pure /
CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_hygiene import split_stage_business  # noqa: E402
from nodes._otr_ledger_scrub import (  # noqa: E402
    scrub_ledger,
    get_line_action,
    append_compose_flag,
)

# A quoted character line with a leaked, verb-led trailing stage direction
# ("taps" is in the extended _NARRATION_VERBS family).
_DIRTY = '"The relay is dead." taps the console nervously.'
_ACTION = "taps the console nervously."


class TestSplitStageBusiness:
    def test_trailing_action_after_quote(self):
        dialogue, action, reason = split_stage_business(_DIRTY)
        assert action == _ACTION
        assert reason == "trailing_after_quote"
        assert "taps" not in dialogue
        assert "relay is dead" in dialogue

    def test_clean_quoted_dialogue_not_split(self):
        clean = '"The relay is dead. We have minutes, not hours."'
        dialogue, action, reason = split_stage_business(clean)
        assert action == ""
        assert reason == ""

    def test_no_quotes_not_split(self):
        dialogue, action, reason = split_stage_business("The relay is dead.")
        assert action == ""

    def test_unbalanced_quotes_not_split(self):
        dialogue, action, reason = split_stage_business('"The relay is dead.')
        assert action == ""

    def test_never_raises_on_garbage(self):
        assert split_stage_business(None) == ("", "", "")
        assert split_stage_business(12345)[1] == ""


class TestGetLineAction:
    def test_reads_latest_valid(self):
        row = {"compose_flags": []}
        append_compose_flag(row, 'action_split:{"action":"first","reason":"x"}')
        append_compose_flag(row, "stage_dir_stripped:bare")
        append_compose_flag(row, 'action_split:{"action":"second","reason":"y"}')
        assert get_line_action(row) == "second"   # latest valid wins

    def test_ignores_malformed(self):
        row = {"compose_flags": ["action_split:not-json", "other:flag"]}
        assert get_line_action(row) == ""

    def test_empty_when_none(self):
        assert get_line_action({"compose_flags": []}) == ""
        assert get_line_action({}) == ""
        assert get_line_action(None) == ""


def _ledger(flag_on: bool) -> dict:
    meta = {"story_quality_v2_enabled": True} if flag_on else {}
    return {
        "schema_version": "l3",
        "episode_id": "r3_l7",
        "meta": meta,
        "cast": [
            {"char_id": "c01", "name": "ANNOUNCER", "gender": "male",
             "tts_model": "kokoro", "voice_preset": "bm_george"},
            {"char_id": "c02", "name": "EDNA REEVES", "gender": "female",
             "tts_model": "bark", "voice_preset": "v2/en_speaker_4"},
        ],
        "lines": [
            {"line_id": "b001", "beat_id": "b001", "char_id": "announcer",
             "speaker_role": "announcer",
             "text": "Tonight, a signal from the dark side of the moon."},
            {"line_id": "b002", "beat_id": "b002", "char_id": "c02",
             "speaker_role": "character", "compose_flags": [],
             "text": _DIRTY,
             "word_count": len(_DIRTY.split()),
             "char_count": len(_DIRTY)},
        ],
    }


class TestScrubIntegration:
    def test_flag_on_splits_and_records(self):
        led = _ledger(flag_on=True)
        scrub_ledger(led, repair_available=True)
        row = led["lines"][1]
        assert get_line_action(row) == _ACTION
        assert "taps" not in row["text"]
        assert "relay is dead" in row["text"]
        # word_count restamped to the dialogue, not the original
        assert row["word_count"] == len(row["text"].split())

    def test_flag_off_byte_identical(self):
        led = _ledger(flag_on=False)
        scrub_ledger(led, repair_available=True)
        row = led["lines"][1]
        # no split recorded; the action text is NOT extracted under flag-off
        assert get_line_action(row) == ""
        assert not any(
            f.startswith("action_split:") for f in row.get("compose_flags", [])
        )

    def test_idempotent_no_double_split(self):
        led = _ledger(flag_on=True)
        scrub_ledger(led, repair_available=True)
        text_after_first = led["lines"][1]["text"]
        scrub_ledger(led, repair_available=True)
        row = led["lines"][1]
        splits = [
            f for f in row.get("compose_flags", [])
            if f.startswith("action_split:")
        ]
        assert len(splits) == 1                       # only one record
        assert row["text"] == text_after_first        # stable


class TestTelemetry:
    def _ledger_with_flags(self, flag_on: bool) -> dict:
        meta = {"story_quality_v2_enabled": True} if flag_on else {}
        return {
            "schema_version": "l3", "episode_id": "tel", "meta": meta,
            "cast": [
                {"char_id": "c02", "name": "EDNA REEVES", "gender": "female",
                 "tts_model": "bark", "voice_preset": "v2/en_speaker_4"},
            ],
            "lines": [
                {"line_id": "b001", "beat_id": "b001", "char_id": "c02",
                 "speaker_role": "character",
                 "compose_flags": ["objective_literal_retry"],
                 "text": "The relay is dead.",
                 "word_count": 4, "char_count": 18},
                {"line_id": "b002", "beat_id": "b002", "char_id": "c02",
                 "speaker_role": "character",
                 "compose_flags": [
                     'action_split:{"action":"taps","reason":"x"}',
                     "action_split_failed:ValueError",
                 ],
                 "text": "We have minutes.",
                 "word_count": 3, "char_count": 16},
            ],
        }

    def test_flag_on_aggregates_counts(self):
        led = self._ledger_with_flags(flag_on=True)
        scrub_ledger(led, repair_available=True)
        sq = led["meta"]["story_quality"]
        assert sq["l1_rerolls"] == 1
        assert sq["l7_splits"] == 1
        assert sq["l7_split_failures"] == 1

    def test_flag_off_writes_no_summary(self):
        led = self._ledger_with_flags(flag_on=False)
        scrub_ledger(led, repair_available=True)
        assert "story_quality" not in led.get("meta", {})
