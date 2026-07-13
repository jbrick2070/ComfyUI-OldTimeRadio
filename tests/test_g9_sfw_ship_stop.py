"""G9: the deterministic SFW ship-stop on spoken text.

PD4 ("Safe for work. No profanity.") had no enforcement on the episode that
ships: the ledger scrub runs only on `run_story_spine=True` lanes and nothing
reads its verdict, and the lanes that checked at all did it with an LLM opinion.

G9 lives in the gap audit -- the one path every lane crosses -- so Phase 10
raises on it like any other critical gap.
"""
import pytest

from nodes import _otr_ledger_freeze as freeze
from nodes._otr_stage3_validators import DEFAULT_PROFANITY_TERMS


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
            {"line_id": f"line_{i:03d}", "text": text}
            for i, text in enumerate(texts, 1)
        ],
    }


def _g9_errors(led):
    report = freeze.phase_0_gap_audit_pre(led)
    return [e for e in report.errors if e.startswith("G9")]


def test_clean_spoken_text_passes_g9():
    assert not _g9_errors(_ledger(
        "Good evening, and welcome to the broadcast.",
        "Hello there -- the shellfish stand is closed.",
        "Michelle heard the grille repeat every word.",
    ))


@pytest.mark.parametrize("term", ["damn", "hell", "shit", "screw you"])
def test_profanity_on_the_microphone_is_a_critical_gap(term):
    g9 = _g9_errors(_ledger("Fine.", f"Oh {term}, the tape is gone."))
    assert g9, f"{term!r} must be a critical gap"
    assert "line_002" in g9[0]
    assert term in g9[0]


def test_g9_is_whole_word_only():
    # "hell" inside "hello"/"shelf", "ass" inside "passed"/"class".
    assert not _g9_errors(
        _ledger("Hello, class -- the parcel passed the shelf."),
    )


def test_g9_matches_every_term_in_the_live_list():
    pattern = freeze._sfw_pattern()
    for term in DEFAULT_PROFANITY_TERMS:
        assert pattern.search(f"and then {term} happened"), term


def test_phase_10_refuses_to_freeze_a_profane_episode():
    led = _ledger("Welcome.", "What the hell was that noise?")
    with pytest.raises(freeze.FreezeAssertionError):
        freeze.phase_10_gap_audit_post_and_freeze(led)
    assert led["meta"]["freeze_verdict"] == "needs_full_rerun"
