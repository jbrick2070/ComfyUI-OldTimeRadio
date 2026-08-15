"""The announcer gets TOLD what was wrong before Python writes its line.

These three sites used to be reject-and-substitute: one model call, and if
the line held a single bracket Python threw it away and shipped a hardcoded
sentence of its own -- "Good evening. This is SIGNAL LOST." -- on air, with
no reroll and no repair.

That is Python AUTHORING BROADCAST PROSE, which is the furthest any call site
in this repo sat from the standing law that only a model may write a spoken
line (operator, 2026-08-14: *"I hate shims ... I like LLM calls to ask it to
fix things"*).

The fix is not to delete the fallback -- a render is never killed, and a
silent opening is worse than a plain one. The fix is that the model is given
the one thing it never got: the complaint. The Python line becomes the last
resort AFTER the model was actually asked, rather than the second thing tried.
"""
from __future__ import annotations

from nodes import _otr_line_composer as lc


class _Announcer:
    """Replies in order; records what it was told between attempts."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls = 0
        self.prompts: "list[str]" = []

    def __call__(self, messages, **kwargs):
        self.calls += 1
        self.prompts.append(
            "\n".join(m.get("content", "") for m in messages))
        return self.replies[min(self.calls, len(self.replies)) - 1]


BRACKETED = "[MUSIC UP] Good evening, and welcome to the lighthouse."
CLEAN = "Good evening, and welcome to the lighthouse."


def test_a_bracketed_opening_is_handed_back_not_replaced():
    said = _Announcer(BRACKETED, CLEAN)
    result = lc.compose_announcer_intro(
        creative_fn=said,
        script_brief="a keeper alone on the night the lamp fails",
    )

    assert said.calls == 2, "the model must be asked again, not replaced"
    # It was TOLD what was wrong -- that is the whole point.
    assert "brackets or braces" in said.prompts[1]
    assert BRACKETED in said.prompts[1], "it must see its own bad line"
    # The model's second line ships. No Python prose anywhere.
    assert result.text == CLEAN
    assert "announcer_intro_after_retry" in result.compose_flags


def test_a_good_opening_costs_no_second_call():
    said = _Announcer(CLEAN)
    result = lc.compose_announcer_intro(
        creative_fn=said,
        script_brief="a keeper alone on the night the lamp fails",
    )
    assert said.calls == 1
    assert result.text == CLEAN
    assert "announcer_intro" in result.compose_flags


def test_python_prose_ships_only_after_the_model_was_asked_twice():
    """The fallback stays -- a render is never killed for this -- but it is
    now the LAST resort and the flag says a Python line went to air."""
    said = _Announcer(BRACKETED, "[STILL BRACKETED] nope")
    result = lc.compose_announcer_intro(
        creative_fn=said,
        script_brief="a keeper alone on the night the lamp fails",
    )
    assert said.calls == 2, "bounded -- it does not grind"
    assert "announcer_intro_structural_fallback" in result.compose_flags
    assert result.text, "silence is worse than a plain opening"


def test_the_closing_is_asked_twice_too():
    said = _Announcer("[SFX] That was the lighthouse.",
                      "That was the lighthouse.")
    result = lc.compose_announcer_outro(
        creative_fn=said,
        script_brief="a keeper alone on the night the lamp fails",
        news_close_brief="",
        intro_text="Good evening.",
    )
    assert said.calls == 2
    assert result.text == "That was the lighthouse."
    assert "announcer_outro_after_retry" in result.compose_flags


def test_the_retry_runs_cooler_than_the_first_ask():
    """A correction, not a fresh invention -- the same 2B principle the
    structured-call ladder uses for its own retry."""
    seen: "list[float]" = []

    class _Temps:
        calls = 0

        def __call__(self, messages, *, temperature=None, **kwargs):
            type(self).calls += 1
            seen.append(temperature)
            return BRACKETED if type(self).calls == 1 else CLEAN

    lc.compose_announcer_intro(
        creative_fn=_Temps(),
        script_brief="a keeper alone on the night the lamp fails",
    )
    assert len(seen) == 2
    assert seen[1] < seen[0], f"retry should be cooler: {seen}"
    assert seen[1] == lc._STRUCTURAL_RETRY_TEMPERATURE
