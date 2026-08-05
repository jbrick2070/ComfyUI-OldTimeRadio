"""Behavioral contract for the final deterministic inline-ledger scrub."""

from __future__ import annotations

from nodes import _otr_story_spine
from nodes._otr_ledger_scrub import scrub_ledger


def _ledger(text: str = "WARNING: The transmission has begun.") -> dict:
    return {
        "schema_version": "l3",
        "episode_id": "scrub-contract",
        "cast": [
            {"char_id": "c01", "name": "ALICE VALE"},
        ],
        "lines": [
            {
                "line_id": "b001",
                "beat_id": "b001",
                "char_id": "c01",
                "speaker_role": "character",
                "text": text,
            },
        ],
    }


def test_clean_ledger_passes_without_reauthoring():
    ledger = _ledger()
    result = scrub_ledger(ledger)

    assert result.status == "PASS"
    assert ledger["lines"][0]["text"] == (
        "WARNING: The transmission has begun."
    )


def test_exact_roster_transport_is_removed_without_changing_story_words():
    ledger = _ledger("ALICE VALE: The transmission has begun.")
    result = scrub_ledger(ledger)

    assert result.status == "PASS"
    assert result.normalized
    assert ledger["lines"][0]["text"] == "The transmission has begun."


def test_arbitrary_length_vocabulary_and_smoking_survive_scrub():
    text = (
        " ".join(["lantern"] * 500)
        + " beside a velvet chair in the smoking room."
    )
    ledger = _ledger(text)
    result = scrub_ledger(ledger)

    assert result.status == "PASS"
    assert ledger["lines"][0]["text"] == text


def test_empty_spoken_row_fails_structural_integrity():
    result = scrub_ledger(_ledger("   "))

    assert result.status == "FAIL"
    assert any(row.code == "empty_spoken_line" for row in result.findings)


def test_content_no_longer_fails_the_scrub():
    """Inverted 2026-08-05: content is not a scrub failure any more.

    "safety_violation" was a BLOCKING finding code, so a faithful adaptation
    failed the scrub on its own author's vocabulary. The directive removed
    content enforcement; STRUCTURAL blocking codes are untouched (see the
    empty_spoken_line test above, which still fails closed).

    safety_violations stays on ScrubResult as a permanently-empty list so the
    dataclass and its to_dict() shape do not change for any existing reader --
    a ripped pass may not orphan a field.
    """
    result = scrub_ledger(_ledger("The gun is on the table."))

    assert result.status == "PASS"
    assert result.safety_violations == []
    assert not any(row.code == "safety_violation" for row in result.findings)


def test_story_spine_records_clean_scrub_without_reauthoring(monkeypatch):
    ledger = _ledger()
    meta = {}
    monkeypatch.setattr(
        _otr_story_spine,
        "_unload_writer_llm",
        lambda receipt: receipt.update({"writer_llm_unload": "test"}),
    )

    _otr_story_spine.run_post_script_spine(ledger, meta)

    assert meta["story_spine_status"] == "clean"
    assert meta["ledger_scrub"]["status"] == "PASS"
    assert ledger["lines"][0]["text"] == (
        "WARNING: The transmission has begun."
    )
