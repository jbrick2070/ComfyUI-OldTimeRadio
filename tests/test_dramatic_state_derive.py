"""Sprint 2.1 (2026-05-28) -- DramaticState derivation wire-up.

Pins the helper that OTR_LedgerScriptWriter calls after news +
cast + Stage 1 to stamp meta["dramatic_state"]. The helper must
always produce a schema-valid DramaticState; any failure short of
that risks crashing the writer.
"""
from __future__ import annotations

import pytest

from nodes._otr_dramatic_state import (
    DramaticState,
    derive_dramatic_state_from_meta,
    pick_costly_choice_slot,
)


# ---------------------------------------------------------------------------
# pick_costly_choice_slot
# ---------------------------------------------------------------------------


def test_pick_costly_choice_slot_long_list_picks_second_to_last():
    """>=4 slots: pick the penultimate (one before the close)."""
    assert pick_costly_choice_slot(
        ["d001", "d002", "d003", "d004", "d005"]
    ) == "d004"


def test_pick_costly_choice_slot_short_list_picks_last():
    assert pick_costly_choice_slot(["d001", "d002"]) == "d002"


def test_pick_costly_choice_slot_empty_defaults_to_d001():
    """DramaticState requires a non-empty d-id; helper falls back
    rather than producing an unmappable slot."""
    assert pick_costly_choice_slot([]) == "d001"
    assert pick_costly_choice_slot(None) == "d001"


# ---------------------------------------------------------------------------
# derive_dramatic_state_from_meta -- happy paths
# ---------------------------------------------------------------------------


def test_derive_with_full_inputs():
    ds = derive_dramatic_state_from_meta(
        script_brief=(
            "An audit team races to find a falsified ledger before "
            "midnight. Will the forensic engineer sign the confession "
            "in time?"
        ),
        cast_rows=[
            {"name": "ALICE"},
            {"name": "BOB"},
        ],
        voice_slot_ids=["d001", "d002", "d003", "d004", "d005"],
        ending_change="the falsified ledger becomes public record",
    )
    assert isinstance(ds, DramaticState)
    assert ds.dramatic_question.endswith("?")
    assert "ALICE" in ds.character_a_wants
    assert "BOB" in ds.character_b_wants
    assert ds.costly_choice_beat == "d004"
    assert "public record" in ds.ending_change


def test_derive_with_empty_script_brief_uses_default_question():
    ds = derive_dramatic_state_from_meta(
        script_brief="",
        cast_rows=[{"name": "ALICE"}, {"name": "BOB"}],
        voice_slot_ids=["d001", "d002"],
    )
    assert "principals" in ds.dramatic_question
    assert ds.costly_choice_beat == "d002"


def test_derive_with_brief_without_question_mark_wraps_into_question():
    """A declarative brief gets wrapped into a How-does-this-resolve
    question template so the dramatic_question field always ends
    with `?`."""
    ds = derive_dramatic_state_from_meta(
        script_brief="An audit team must verify a falsified ledger.",
        cast_rows=[{"name": "ALICE"}, {"name": "BOB"}],
        voice_slot_ids=["d001", "d002", "d003"],
    )
    assert ds.dramatic_question.endswith("?")


def test_derive_with_single_cast_member():
    """Helper invents an OPPOSITION placeholder so the wants are
    still distinct (the DramaticState wants-must-oppose validator
    would otherwise reject)."""
    ds = derive_dramatic_state_from_meta(
        script_brief="solo voice piece",
        cast_rows=[{"name": "ALICE"}],
        voice_slot_ids=["d001"],
    )
    assert "ALICE" in ds.character_a_wants
    assert "OPPOSITION" in ds.character_b_wants


def test_derive_with_no_cast_uses_placeholders():
    ds = derive_dramatic_state_from_meta(
        script_brief="placeholder brief",
        cast_rows=[],
        voice_slot_ids=["d001"],
    )
    assert "PROTAGONIST" in ds.character_a_wants
    assert "ANTAGONIST" in ds.character_b_wants


def test_derive_accepts_attribute_style_cast_rows():
    """Helper duck-types name attribute / 'name' key alike."""
    class Row:
        def __init__(self, name): self.name = name

    ds = derive_dramatic_state_from_meta(
        script_brief="brief",
        cast_rows=[Row("ALICE"), Row("BOB")],
        voice_slot_ids=["d001", "d002"],
    )
    assert "ALICE" in ds.character_a_wants
    assert "BOB" in ds.character_b_wants


def test_derive_truncates_long_brief_into_question_under_240():
    """The pattern validator caps the question at 240 chars."""
    long_brief = (
        "An audit team races against the clock to verify a falsified "
        "ledger that may implicate multiple senior officials in a "
        "decade-long pattern of misrepresentation that touches three "
        "different agencies and four jurisdictions while the auditor "
        "considers whether to sign the confession by midnight, "
        "knowing the consequences for the team and the agency overall."
    )
    ds = derive_dramatic_state_from_meta(
        script_brief=long_brief,
        cast_rows=[{"name": "ALICE"}, {"name": "BOB"}],
        voice_slot_ids=["d001", "d002", "d003", "d004"],
    )
    assert 10 <= len(ds.dramatic_question) <= 240
