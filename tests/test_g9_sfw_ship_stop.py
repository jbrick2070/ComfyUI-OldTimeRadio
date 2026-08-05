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


def _content_errors(ledger):
    """EVERY audit error, not just G9-prefixed ones."""
    report = freeze.phase_0_gap_audit_pre(ledger)
    return list(report.errors)


def _errors_for(text):
    """Audit errors for a one-line ledger carrying `text`."""
    return _content_errors(_ledger("Fine.", text))


#: The audit result for a line whose vocabulary is unremarkable. Every
#: content-word case below must match this EXACTLY.
def _baseline():
    return _errors_for("The report says nothing unusual tonight.")


def test_the_gate_function_is_gone():
    """The symbol itself must not come back under the same name."""
    assert not hasattr(freeze, "_check_g9_sfw_spoken_text"), (
        "the G9 spoken-safety gate was reinstated -- operator directive "
        "2026-08-05 removed content enforcement from the freeze path")


def test_a_word_never_changes_the_audit_result():
    """DIFFERENTIAL, so a renamed content check cannot hide behind a prefix.

    r4 (Codex) caught the earlier version filtering errors on the "G9" prefix:
    enforcement reintroduced under any other name would have passed silently.
    These fixtures are deliberately minimal and DO carry structural errors --
    that is fine and unrelated. The claim is that swapping one word for another
    changes NOTHING about the audit. Any content policy, under any name, breaks
    this the moment it fires.
    """
    assert _errors_for("The velvet chair stood beside a smoking chimney.") == _baseline()


@pytest.mark.parametrize("term", ["damn", "gun", "knife", "dagger", "naked"])
def test_formerly_terminal_words_no_longer_stop_the_ship(term):
    """The exact words that used to kill an episode are now inert."""
    assert _errors_for(f"The report says {term} tonight.") == _baseline(), (
        "%r still changes the audit -- content enforcement is back" % term)


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
    assert _errors_for(
        "Is this a dagger which I see before me, the handle toward my hand?"
    ) == _baseline()
