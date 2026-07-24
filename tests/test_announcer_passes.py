"""Positive behavior for non-gating announcer and source-coda surfaces."""

from __future__ import annotations

import pytest

from nodes._otr_line_composer import (
    compose_announcer_intro,
    compose_announcer_outro,
    compose_news_coda,
    clean_one_line,
    fallback_announcer_intro,
    fallback_announcer_outro,
    validate_announcer_line,
    validate_news_coda_bridge,
)


def _creative(text=None, *, error=None, calls=None):
    def generate(messages, *, temperature, max_new_tokens, stop=None):
        if calls is not None:
            calls.append(messages)
        if error is not None:
            raise error
        return text
    return generate


@pytest.mark.parametrize(
    "text",
    [
        "Go.",
        " ".join(["lantern"] * 600),
        "WARNING: The transmission has begun.",
        "She smoked a cigarette beside the velvet chair.",
    ],
)
def test_announcer_length_vocabulary_and_smoking_pass(text):
    ok, cleaned = validate_announcer_line(text)

    assert ok
    assert cleaned == text


def test_clean_one_line_normalizes_transport_whitespace_without_truncation():
    text = "  \"The   transmission\nremains   steady.\"  "
    assert clean_one_line(text, max_chars=5) == (
        "The transmission remains steady."
    )


def test_intro_strips_fixed_announcer_label_with_one_call():
    calls = []
    result = compose_announcer_intro(
        creative_fn=_creative(
            "ANNOUNCER: WARNING: The transmission has begun.",
            calls=calls,
        ),
        script_brief="A relay receives an impossible signal.",
    )

    assert result.text == "WARNING: The transmission has begun."
    assert result.compose_flags == ("announcer_intro",)
    assert len(calls) == 1


def test_intro_provider_failure_uses_structural_fallback():
    result = compose_announcer_intro(
        creative_fn=_creative(error=RuntimeError("provider unavailable")),
        script_brief="A relay receives an impossible signal.",
    )

    assert result.text == fallback_announcer_intro(
        "A relay receives an impossible signal."
    )
    assert result.compose_flags == (
        "announcer_intro_structural_fallback",
    )


def test_outro_arbitrary_length_ships_with_one_call():
    calls = []
    text = " ".join(["goodnight"] * 500)
    result = compose_announcer_outro(
        creative_fn=_creative(text, calls=calls),
        script_brief="The relay remains open.",
        news_close_brief="The source archive survives.",
        intro_text="Welcome.",
    )

    assert result.text == text
    assert result.compose_flags == ("announcer_outro",)
    assert len(calls) == 1


def test_outro_provider_failure_uses_structural_fallback():
    result = compose_announcer_outro(
        creative_fn=_creative(error=RuntimeError("provider unavailable")),
        script_brief="The relay remains open.",
        news_close_brief="The source archive survives.",
        intro_text="Welcome.",
    )

    assert result.text == fallback_announcer_outro(
        "The source archive survives."
    )
    assert result.compose_flags == (
        "announcer_outro_structural_fallback",
    )


@pytest.mark.parametrize("text", ["Go.", "WARNING: Continue.", "A " * 500])
def test_news_coda_bridge_has_no_length_or_style_gate(text):
    ok, cleaned = validate_news_coda_bridge(text)

    assert ok
    assert cleaned == text.strip().rstrip(" :;,-")


def test_news_coda_appends_source_fact_unchanged():
    calls = []
    fact = "The archive dates the recording to 1938."
    result = compose_news_coda(
        creative_fn=_creative("And the record remains", calls=calls),
        news_close_brief=fact,
        premise="A flooded transmitter is restored.",
    )

    assert result.text == "And the record remains: " + fact
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"script_brief": "", "safe_open_brief": None},
    ],
)
def test_intro_requires_nonempty_structural_context(kwargs):
    with pytest.raises(RuntimeError):
        compose_announcer_intro(
            creative_fn=_creative("Welcome."),
            **kwargs,
        )
