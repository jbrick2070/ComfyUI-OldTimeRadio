"""`story_brief_attempts` must report the ladder's REAL cost on success.

WHY THIS FILE EXISTS. The failure path always reported honestly --
`delta["story_brief_attempts"] = exc.attempts` when the ladder was exhausted.
The SUCCESS path set nothing, so the writer's
`_brief_delta.get("story_brief_attempts", 1)` fell through to its default and
`meta["visual_style_receipt"]["attempts"]` read 1 forever, even when
structured_call had retried twice before succeeding.

That is the BUG-12.86 shape one level along: a receipt that can only ever print
one value is not a measurement, and a test asserting `attempts == 1` would have
passed against both the broken and the fixed code. So these tests assert the
value MOVES with the number of attempts, which is the only assertion the old
behaviour could not satisfy.
"""
from __future__ import annotations

import pytest

from nodes import _otr_story_brief as sb


class _Card:
    """Minimal stand-in for the pydantic model the reflection returns."""

    def __init__(self):
        self.visual_card = {"medium_short": "test"}

    def model_dump(self):
        return {"visual_card": self.visual_card}


@pytest.mark.parametrize("attempts", [1, 2, 3])
def test_success_reports_the_real_attempt_count(monkeypatch, attempts):
    """The assertion the old code could not pass: the number must TRACK.

    Parameterised deliberately -- a single `== 1` case is exactly what let the
    hardcoded default survive.
    """
    seen = {}

    def fake_structured_call(**kwargs):
        hook = kwargs.get("on_attempt_complete")
        seen["hook_passed"] = callable(hook)
        if callable(hook):
            # Mirror structured_call: fire once per attempt, cumulative count.
            for i in range(1, attempts + 1):
                hook(i, None, None)
        return _Card()

    monkeypatch.setattr(sb, "structured_call", fake_structured_call)
    monkeypatch.setattr(sb, "_build_reflection_input", lambda _led: "INPUT")
    monkeypatch.setattr(
        sb, "_success_delta",
        lambda **kw: {"story_brief_status": "ok"})

    delta = sb.run_story_brief_reflection(
        object(), lambda *a, **k: "", technical_model_id="m",
        is_visual_storybased=True)

    assert seen.get("hook_passed"), (
        "on_attempt_complete was not passed to structured_call, so the count "
        "can never be observed"
    )
    assert delta["story_brief_attempts"] == attempts, (
        f"reported {delta.get('story_brief_attempts')!r} after {attempts} "
        f"attempt(s) -- the receipt is not tracking the ladder"
    )


def test_hook_never_breaks_the_pass_on_a_bad_value(monkeypatch):
    """Telemetry must not be able to fail a story pass."""
    def fake_structured_call(**kwargs):
        hook = kwargs.get("on_attempt_complete")
        hook("not-an-int", None, None)      # must be swallowed
        return _Card()

    monkeypatch.setattr(sb, "structured_call", fake_structured_call)
    monkeypatch.setattr(sb, "_build_reflection_input", lambda _led: "INPUT")
    monkeypatch.setattr(
        sb, "_success_delta",
        lambda **kw: {"story_brief_status": "ok"})

    delta = sb.run_story_brief_reflection(
        object(), lambda *a, **k: "", technical_model_id="m",
        is_visual_storybased=True)
    assert delta["story_brief_attempts"] == 1


def test_the_writer_still_reads_the_key_this_sets():
    """Both halves of the contract, so a rename cannot silently desync them --
    the exact failure mode BUG-12.86 describes."""
    from tests.fixtures.writer_family import family_source
    writer = family_source()
    assert 'get("story_brief_attempts"' in writer, (
        "the writer no longer reads story_brief_attempts -- either it was "
        "renamed on one side only, or the receipt lost its source"
    )
