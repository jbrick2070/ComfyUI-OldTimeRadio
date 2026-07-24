"""G9 narrow spoken-safety ship stop shared by every story bank."""

import pytest

from nodes import _otr_ledger_freeze as freeze
from nodes._otr_content_safety import (
    EXPLICIT_NUDITY_TERMS,
    EXPLICIT_WEAPON_TERMS,
    PROFANITY_TERMS,
    find_text_hits,
)


def _ledger(*texts):
    return {
        "schema_version": freeze.EXPECTED_SCHEMA_VERSION,
        "meta": {},
        "cast": [],
        "beats": [],
        "scenes": [],
        "shots": [],
        "music": [],
        "clips": [],
        "lines": [
            {
                "line_id": f"line_{index:03d}",
                "speaker_role": "character",
                "char_id": "c01",
                "text": text,
            }
            for index, text in enumerate(texts, 1)
        ],
    }


def _g9_errors(ledger):
    report = freeze.phase_0_gap_audit_pre(ledger)
    return [error for error in report.errors if error.startswith("G9")]


def test_ordinary_visual_language_smoking_and_begun_pass():
    assert not _g9_errors(
        _ledger(
            "The velvet chair stood beside a smoking chimney.",
            "She had begun the long walk past the silver machine.",
        )
    )


@pytest.mark.parametrize("term", ["damn", "gun", "knife", "naked"])
def test_authorized_terminal_categories_are_critical_gaps(term):
    errors = _g9_errors(_ledger("Fine.", f"The report says {term} tonight."))
    assert errors
    assert "line_002" in errors[0]
    assert term in errors[0]


def test_policy_is_whole_word_only():
    assert not _g9_errors(
        _ledger("Hello, class; the parcel passed after the work had begun.")
    )


def test_shared_live_vocabulary_is_detected():
    for term in (*PROFANITY_TERMS, *EXPLICIT_WEAPON_TERMS, *EXPLICIT_NUDITY_TERMS):
        assert find_text_hits(f"and then {term} happened"), term


def test_phase_10_refuses_an_unsafe_episode():
    ledger = _ledger("Welcome.", "What the hell was that noise?")
    with pytest.raises(freeze.FreezeAssertionError):
        freeze.phase_10_gap_audit_post_and_freeze(ledger)
    assert ledger["meta"]["freeze_verdict"] == "needs_full_rerun"
