"""tests/test_freeze_warn_taxonomy.py -- story-ledger DRIFT chunk 4.

The deterministic freeze WARN taxonomy (nodes/_otr_freeze_cascade.py):
structural_error (unrenderable -> blocks) / story_accuracy_warning (continuity /
canon-divergence / unverified critic / consistency -> ships NON-clean + visible)
/ cosmetic_warning (clean-with-warns). Stops a SHIPPED arc/continuity failure
from being mislabeled "structural". Pure-Python; no GPU; no LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_freeze_cascade import (  # noqa: E402
    FREEZE_WARN_TIERS,
    build_freeze_warn_taxonomy,
    classify_freeze_warning,
)


def test_tiers_are_the_three_named():
    assert FREEZE_WARN_TIERS == (
        "structural_error", "story_accuracy_warning", "cosmetic_warning",
    )


def test_classify_structural_unrenderable():
    for w in (
        "line l003 has no voiced text (unrenderable)",
        "missing speaker for beat b004",
        "empty line at l007",
    ):
        assert classify_freeze_warning(w) == "structural_error", w


def test_classify_story_accuracy():
    for w in (
        "continuity break: the door was open then shut",
        "canon divergence on setting",
        "timeline contradiction in act 2",
        "sound_palette empty despite a selected style",
    ):
        assert classify_freeze_warning(w) == "story_accuracy_warning", w


def test_classify_cosmetic_default():
    assert classify_freeze_warning("title casing looks a little off") == "cosmetic_warning"
    assert classify_freeze_warning("") == "cosmetic_warning"
    assert classify_freeze_warning(None) == "cosmetic_warning"


def test_structural_wins_over_accuracy():
    # a finding that is BOTH unrenderable and a continuity note is structural.
    assert classify_freeze_warning(
        "continuity: beat b004 has no voiced text"
    ) == "structural_error"


def test_build_taxonomy_buckets_every_source():
    critic = SimpleNamespace(continuity_issues=[{"issue": "the lamp moved by itself"}])
    tax = build_freeze_warn_taxonomy(
        gap_warnings=["minor spacing nit", "a continuity hiccup", "no voiced text on l002"],
        critic_report=critic,
        critic_status={"ran": True, "validated": False, "failure": "exhausted"},
        consistency_status={"clean": False, "defects": [{"field": "sound_palette"}]},
    )
    assert "no voiced text on l002" in " ".join(tax["structural_error"])
    acc = " ".join(tax["story_accuracy_warning"])
    assert "continuity hiccup" in acc        # gap warn -> accuracy
    assert "unverified" in acc                # unverified critic -> accuracy
    assert "lamp moved" in acc                # critic continuity issue -> accuracy
    assert "sound_palette" in acc             # consistency defect -> accuracy
    assert "minor spacing nit" in " ".join(tax["cosmetic_warning"])


def test_build_taxonomy_clean_when_no_findings():
    tax = build_freeze_warn_taxonomy(
        gap_warnings=[], critic_report=None,
        critic_status={"ran": True, "validated": True, "failure": ""},
        consistency_status={"clean": True},
    )
    assert all(tax[t] == [] for t in FREEZE_WARN_TIERS)


def test_build_taxonomy_never_raises_on_garbage():
    tax = build_freeze_warn_taxonomy(
        gap_warnings=None, critic_report=object(),
        critic_status="not a dict", consistency_status=123,
    )
    assert set(tax.keys()) == set(FREEZE_WARN_TIERS)


if __name__ == "__main__":  # pragma: no cover
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
