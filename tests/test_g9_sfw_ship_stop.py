"""G9 is RETIRED. These tests prove the ship stop is gone and stays gone.

G9 was a terminal spoken-safety gate: a delivered character/announcer row whose
words matched the profanity / weapon / sexual list failed the freeze and killed
the episode. It was DELETED 2026-08-05 by operator directive -- no content
guardrails on generated episodes -- because on the adaptation lanes it rejected
the source's own language. "Is this a dagger which I see before me" is MACBETH,
and `dagger` was in the terminal list.

This file used to assert the opposite of everything below. It is kept, inverted,
rather than deleted: a deleted test file is silence, and silence is how a
guardrail creeps back. If one of these fails, someone re-armed content
enforcement in the freeze path and that is an operator decision, not a fix.
"""

import pytest

from nodes import _otr_ledger_freeze as freeze


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


def test_the_gate_function_is_gone():
    """The symbol itself must not come back under the same name."""
    assert not hasattr(freeze, "_check_g9_sfw_spoken_text"), (
        "the G9 spoken-safety gate was reinstated -- operator directive "
        "2026-08-05 removed content enforcement from the freeze path")


def test_ordinary_visual_language_still_passes():
    assert not _g9_errors(
        _ledger(
            "The velvet chair stood beside a smoking chimney.",
            "She had begun the long walk past the silver machine.",
        )
    )


@pytest.mark.parametrize("term", ["damn", "gun", "knife", "dagger", "naked"])
def test_formerly_terminal_words_no_longer_stop_the_ship(term):
    """The exact words that used to kill an episode now pass the audit."""
    assert not _g9_errors(_ledger("Fine.", f"The report says {term} tonight.")), (
        "%r still raises a G9 error" % term)


def test_phase_10_raises_nothing_about_content():
    """A line that used to be a terminal CONTENT failure no longer is one.

    This fixture ledger is deliberately minimal, so it can still fail the
    freeze on STRUCTURE -- that is correct and untouched by this change. The
    claim under test is narrower and exact: whatever the freeze objects to, it
    is never the words.
    """
    ledger = _ledger("Welcome.", "What the hell was that noise?")
    try:
        freeze.phase_10_gap_audit_post_and_freeze(ledger)
    except freeze.FreezeAssertionError as exc:
        offending = [e for e in exc.errors if str(e).startswith("G9")]
        assert not offending, "content is still terminal at freeze: %r" % offending


def test_macbeth_keeps_its_dagger():
    """The line this whole change exists for."""
    ledger = _ledger(
        "Is this a dagger which I see before me, the handle toward my hand?",
    )
    assert not _g9_errors(ledger)
