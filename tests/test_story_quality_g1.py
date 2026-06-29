"""Story-quality G1 (C2) -- gate de-compression locks.

Pins the v2-gated behavior: the collapse hint switches to the non-compressing V2
variant, the defect-score ordering, and the budget-derived one-breath cap threaded
through the per-line quality scorer. v2-OFF stays the legacy 28-word cap. Pure / CPU.
UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import json
from pathlib import Path

from nodes._otr_line_composer import (
    LineRequest,
    _QUALITY_COLLAPSE_HINT,
    _QUALITY_COLLAPSE_HINT_V2,
    _quality_flags_for_line,
    _quality_reroll_hint,
    line_quality_defect_score,
)

_CORRECTED = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "golden_story_quality.json")
    .read_text(encoding="utf-8"))["rows"][0]["corrected_text"]  # ~32-word clean line


def _char_req(**kw) -> LineRequest:
    base = dict(speaker="ANYA", intent="", mood="", target_words=50,
                canon_header="", last_lines=(),
                speaker_role="character", story_quality_v2_enabled=True)
    base.update(kw)
    return LineRequest(**base)


# --- the V2 collapse hint -----------------------------------------------------

def _collapse_flags():
    return [("one_breath", "r", "one_breath"),
            ("anchor_stuffing", "r", "anchor_stuffing")]


def test_collapse_hint_v2_on_v2_path():
    assert _quality_reroll_hint(_collapse_flags(), True) == _QUALITY_COLLAPSE_HINT_V2[:240]
    assert "do not pad" in _quality_reroll_hint(_collapse_flags(), True)


def test_collapse_hint_legacy_when_v2_off():
    assert _quality_reroll_hint(_collapse_flags(), False) == _QUALITY_COLLAPSE_HINT[:240]
    # default arg (no flag) is also the legacy constant -> byte-identical caller
    assert _quality_reroll_hint(_collapse_flags()) == _QUALITY_COLLAPSE_HINT[:240]


# --- the defect-score ordering ------------------------------------------------

def test_defect_score_penalizes_truncation_over_clean():
    req = _char_req(words_per_beat_range=(28, 50))
    clean = line_quality_defect_score(_CORRECTED, req)
    trunc = line_quality_defect_score("I will hold the line until the relief column and", req)
    assert trunc > clean  # +2 for is_truncated dominates


# --- the budget-derived one-breath cap (the heart of G1) ----------------------

def _has_one_breath(req) -> bool:
    return any(c == "one_breath" for c, _r, _f in _quality_flags_for_line(_CORRECTED, req))


def test_budget_cap_admits_fuller_line():
    """The ~32-word corrected line: flagged one_breath at the legacy 28 cap (no
    range), NOT flagged once the per-beat budget raises the cap to 50."""
    assert _has_one_breath(_char_req(words_per_beat_range=(0, 0))) is True
    assert _has_one_breath(_char_req(words_per_beat_range=(28, 50))) is False
