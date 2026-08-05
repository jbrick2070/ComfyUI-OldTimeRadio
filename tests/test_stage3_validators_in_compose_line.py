"""Compose-line integration for the narrow, telemetry-only Stage-3 boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nodes._otr_line_composer import LineRequest, compose_line


def _req(speaker: str = "REN BLACK") -> LineRequest:
    return LineRequest(
        speaker=speaker,
        intent="Answer the call.",
        mood="alert",
        canon_header="The relay is on Mars.",
        last_lines=[],
        allowed_people=frozenset({"REN BLACK", "DR. MAEVE COLE"}),
    )


def _beat(speaker: str = "REN BLACK"):
    return SimpleNamespace(
        beat_id="b002",
        speaker=speaker,
        speaker_role="character",
    )


def _creative(text: str, calls: list | None = None):
    def generate(messages, *, temperature, max_new_tokens, stop=None):
        if calls is not None:
            calls.append(messages)
        return text
    return generate


@pytest.mark.parametrize(
    "text",
    [
        "Go.",
        " ".join(["lantern"] * 600),
        "She lit a cigarette while pipe smoke curled above the velvet chair.",
        "WARNING: The midnight transmission has begun.",
        "[REDACTED] remains part of the station record.",
        "A crimson lighthouse and a silver tram crossed the flooded square.",
    ],
)
def test_authored_length_vocabulary_and_smoking_ship_unchanged(text):
    calls = []
    result = compose_line(
        creative_fn=_creative(text, calls),
        req=_req(),
        enable_stage3_validators=True,
        stage3_beat=_beat(),
        max_attempts=1,
    )

    assert result.text == text
    assert result.validation_findings == ()
    assert len(calls) == 1


def test_exact_locked_speaker_label_is_transport_not_prose():
    result = compose_line(
        creative_fn=_creative("REN BLACK: The transmission has begun."),
        req=_req(),
        enable_stage3_validators=True,
        stage3_beat=_beat(),
        max_attempts=1,
    )

    assert result.text == "The transmission has begun."
    assert result.validation_findings == ()


def test_no_safety_finding_is_stamped_and_the_model_is_not_recalled():
    """Inverted 2026-08-05: content is no longer a stage3 finding.

    This asserted that "The gun is on the table." stamped an `sfw_violation`
    row onto the composed line. The policy is retired, so the SAME line now
    composes clean. The half of the original contract that still matters --
    a finding never costs a second model call -- is kept: one call, exact text.
    """
    calls = []
    text = "The gun is on the table."
    result = compose_line(
        creative_fn=_creative(text, calls),
        req=_req(),
        enable_stage3_validators=True,
        stage3_beat=_beat(),
        max_attempts=1,
    )

    assert result.text == text
    assert len(calls) == 1
    assert [row["code"] for row in result.validation_findings] == []


def test_disabled_stage3_preserves_one_call_and_exact_text():
    calls = []
    text = "WARNING: The transmission has begun."
    result = compose_line(
        creative_fn=_creative(text, calls),
        req=_req(),
        enable_stage3_validators=False,
        stage3_beat=_beat(),
        max_attempts=1,
    )

    assert result.text == text
    assert result.validation_findings == ()
    assert len(calls) == 1


def test_missing_beat_skips_stage3_telemetry():
    calls = []
    text = "The gun is on the table."
    result = compose_line(
        creative_fn=_creative(text, calls),
        req=_req(),
        enable_stage3_validators=True,
        stage3_beat=None,
        max_attempts=1,
    )

    assert result.text == text
    assert result.validation_findings == ()
    assert len(calls) == 1
