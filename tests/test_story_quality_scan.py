"""tests/test_story_quality_scan.py -- story-engine v1 measurement harness.

Covers the shared resolved/hedge helper (nodes/_otr_dramatic_state) and the
pure metric functions in scripts/story_quality_scan. No LLM, no I/O beyond a
tmp ledger. Deterministic.
"""
from __future__ import annotations

import importlib.util
import json
import os

import pytest

from nodes._otr_dramatic_state import HEDGE_LIST, is_resolved_ending_change

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCAN = os.path.join(_REPO, "scripts", "story_quality_scan.py")
_spec = importlib.util.spec_from_file_location("otr_story_quality_scan", _SCAN)
scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan)  # type: ignore


# --- shared helper -------------------------------------------------------

def test_hedge_list_nonempty_strings():
    assert HEDGE_LIST
    assert all(isinstance(p, str) and p for p in HEDGE_LIST)


@pytest.mark.parametrize("ending,expected", [
    ("The vial is destroyed and the lab is sealed for good.", True),
    ("Control of the data passes to the rival team by dawn.", True),
    ("", False),
    ("   ", False),
    ("Whether the truth comes out remains to be seen.", False),
    ("The outcome is yet to be seen.", False),
    ("It stays an open question who was right.", False),
    ("Its DNA is forever unread.", False),
])
def test_is_resolved_ending_change(ending, expected):
    assert is_resolved_ending_change(ending) is expected


# --- ledger fixtures -----------------------------------------------------

def _ledger(**meta_over):
    meta = {
        "target_words": 100,
        "character_word_count": 70,
        "announcer_word_count": 15,
        "freeze_verdict": "frozen_with_warns",
        "slot_drama_contracts_audit": {"ok": True},
        "dramatic_state": {"ending_change": "The lab is sealed for good."},
        "length_pass_report": {"fired": False},
    }
    meta.update(meta_over)
    return {
        "episode_id": "ep_test",
        "meta": meta,
        "lines": [
            {"speaker_role": "announcer", "char_name": "ANNOUNCER",
             "text": "Tonight, a sealed lab and a buried secret."},
            {"speaker_role": "character", "char_name": "EDNA",
             "text": "I will not open that vial, not for anyone."},
            {"speaker_role": "music_open", "text": "Musical interlude."},
            {"speaker_role": "announcer", "char_name": "ANNOUNCER",
             "text": "And so the lab was sealed for good. Goodnight."},
        ],
    }


def test_length_ratio_uses_voiced_counts():
    led = _ledger()
    assert scan.voiced_word_total(led) == 85
    assert scan.length_ratio(led, None) == pytest.approx(0.85)


def test_length_ratio_target_override():
    led = _ledger()
    # override target to 170 -> ratio halves
    assert scan.length_ratio(led, 170) == pytest.approx(0.5)


def test_length_ratio_sums_lines_when_counts_absent():
    led = _ledger()
    led["meta"].pop("character_word_count")
    led["meta"].pop("announcer_word_count")
    # voiced = announcer(8) + character(10) + announcer(8) = 26 words
    assert scan.voiced_word_total(led) == 26


def test_length_pass_fired_reads_report():
    assert scan.length_pass_fired(_ledger(length_pass_report={"fired": True})) is True
    assert scan.length_pass_fired(_ledger(length_pass_report={"fired": False})) is False
    assert scan.length_pass_fired(_ledger(length_pass_report={"words_added": 12})) is True
    led = _ledger()
    led["meta"].pop("length_pass_report")
    assert scan.length_pass_fired(led) is None


def test_episode_valid_requires_freeze_and_dramatic():
    assert scan.episode_valid(_ledger())[0] is True
    assert scan.episode_valid(_ledger(freeze_verdict="refused"))[0] is False
    assert scan.episode_valid(
        _ledger(slot_drama_contracts_audit={"ok": False}))[0] is False


def test_find_outro_is_last_announcer_line():
    assert "sealed for good" in scan.find_outro_text(_ledger())


def test_outro_hedge_vs_resolved_flags_contradiction():
    # resolved ending + hedging outro = contradiction
    led = _ledger()
    led["lines"][-1]["text"] = "Whether she was right remains to be seen."
    assert scan.outro_hedge_vs_resolved(led) is True


def test_outro_hedge_ok_when_ending_unresolved():
    led = _ledger(dramatic_state={"ending_change": "The truth remains unknown."})
    led["lines"][-1]["text"] = "It remains to be seen who was right."
    # ending is unresolved -> hedging is allowed -> not flagged
    assert scan.outro_hedge_vs_resolved(led) is False


def test_outro_no_hedge_resolved_not_flagged():
    assert scan.outro_hedge_vs_resolved(_ledger()) is False


def test_narration_self_address_detector():
    # first-person allowed
    assert scan.detect_narration_self_address("I cannot open it.", "EDNA") is False
    # legitimate 3rd-person reference to ANOTHER (no narration verb on self lead)
    assert scan.detect_narration_self_address("They know the code.", "EDNA") is False
    # true self-narration in third person
    assert scan.detect_narration_self_address("She paces the dark lab slowly.", "EDNA") is True


def test_narration_count_only_character_lines():
    led = _ledger()
    led["lines"][1]["text"] = "He stops at the sealed door and stares."
    assert scan.narration_self_address_lines(led) == 1
    # announcer narration is not counted (only character role)
    led["lines"][0]["text"] = "She paces the studio."
    assert scan.narration_self_address_lines(led) == 1


def test_scan_ledger_roundtrip(tmp_path):
    p = tmp_path / "ep_ledger.json"
    p.write_text(json.dumps(_ledger()), encoding="utf-8")
    row = scan.scan_ledger(str(p), None)
    assert row["length_ratio"] == pytest.approx(0.85)
    assert row["episode_valid"] is True
    assert row["outro_hedge_vs_resolved"] is False


def test_aggregate_counts():
    rows = [
        {"length_ratio": 0.9, "length_pass_fired": False, "episode_valid": True,
         "outro_hedge_vs_resolved": False, "narration_self_address_lines": 0,
         "arc_shape": "heist"},
        {"length_ratio": 0.8, "length_pass_fired": True, "episode_valid": False,
         "outro_hedge_vs_resolved": True, "narration_self_address_lines": 2,
         "arc_shape": "slow_dread"},
    ]
    agg = scan.aggregate(rows)
    assert agg["n_legs"] == 2
    assert agg["length_ratio_mean"] == pytest.approx(0.85)
    assert agg["length_pass_fired_count"] == 1
    assert agg["episode_valid_count"] == 1
    assert agg["outro_hedge_vs_resolved_count"] == 1
    assert agg["narration_self_address_total"] == 2
    assert agg["arc_shapes"] == ["heist", "slow_dread"]
