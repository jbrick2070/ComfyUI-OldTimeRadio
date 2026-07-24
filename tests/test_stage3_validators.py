"""Behavioral coverage for the narrow Stage-3 validation boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nodes._otr_stage3_validators import (
    CODE_EMPTY_SPOKEN_LINE,
    CODE_SFW_VIOLATION,
    CODE_SPEAKER_MARKUP,
    ValidationFinding,
    ValidationResult,
    validate_line,
    validate_sfw,
    validate_spoken_structure,
)


def _beat(
    beat_id: str = "b002",
    speaker: str = "DALE PORTER",
    role: str = "character",
):
    return SimpleNamespace(
        beat_id=beat_id,
        speaker=speaker,
        speaker_role=role,
    )


@pytest.mark.parametrize(
    "text",
    [
        "Go.",
        " ".join(["lantern"] * 600),
        "A crimson lighthouse, brass telescope, and silver tram filled the square.",
        "She lit a cigarette beside the tobacco tin while pipe smoke curled.",
        "The midnight transmission has begun.",
        "WARNING: Do not touch the relay.",
        "[REDACTED] remains in the station record.",
        "She knew the descent was a forfeit to the void.",
    ],
)
def test_ordinary_authored_prose_is_not_scored_or_rejected(text):
    result = validate_line(_beat(), text)

    assert result.ok
    assert result.findings == []


@pytest.mark.parametrize("text", [None, "", "   "])
def test_empty_spoken_rows_are_structural_errors(text):
    finding = validate_spoken_structure(_beat(), text)

    assert finding is not None
    assert finding.code == CODE_EMPTY_SPOKEN_LINE


@pytest.mark.parametrize(
    "text",
    [
        "DALE PORTER: The carrier is steady.",
        "[DALE PORTER] The carrier is steady.",
        "[VOICE: DALE PORTER] The carrier is steady.",
        "The carrier is steady.\nThe bearing is north.",
    ],
)
def test_exact_speaker_transport_is_structural_markup(text):
    finding = validate_spoken_structure(_beat(), text)

    assert finding is not None
    assert finding.code == CODE_SPEAKER_MARKUP


def test_nonspoken_rows_are_outside_spoken_validation():
    music = _beat("b004", "MUSIC", "music")

    assert validate_spoken_structure(music, None) is None
    assert validate_sfw(music, "Damn, the gun exposed nudity.") is None


@pytest.mark.parametrize(
    ("text", "expected_term"),
    [
        ("Damn, the relay is dark.", "damn"),
        ("Two guns were on the table.", "guns"),
        ("The knives remained in the drawer.", "knives"),
        ("No weapons belong in this story.", "weapons"),
        ("The report described sexual intercourse.", "sexual intercourse"),
        ("The gallery displayed nudity.", "nudity"),
    ],
)
def test_shared_terminal_safety_categories_are_reported(text, expected_term):
    finding = validate_sfw(_beat(), text)

    assert finding is not None
    assert finding.code == CODE_SFW_VIOLATION
    assert finding.got == expected_term


def test_begun_is_not_a_weapon_match():
    assert validate_line(
        _beat(), "The midnight transmission has begun."
    ).ok



def test_validation_result_serializes_structural_findings():
    result = ValidationResult(findings=[
        ValidationFinding(
            severity="error",
            code=CODE_EMPTY_SPOKEN_LINE,
            beat_id="b002",
            speaker="DALE PORTER",
            message="Spoken ledger row is empty.",
            expected="nonempty spoken text",
            got="",
        )
    ])

    payload = result.to_dict()
    assert payload["ok"] is False
    assert payload["error_count"] == 1
    assert payload["warn_count"] == 0
    assert payload["findings"][0]["code"] == CODE_EMPTY_SPOKEN_LINE
