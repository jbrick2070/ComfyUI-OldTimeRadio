"""Story-Quality R2 C4+C5 -- per-act escalation prompt + on-the-nose subtext gate.

C4: the outline beat prompt asks each beat to RAISE the stake over the prior one.
C5: flag_on_the_nose flags a line that NAMES an emotion/the stakes flatly; it
feeds the existing compose_line quality-reroll. (Grounded scope: the composer
does not receive the beat's arc-phase, so the gate applies to all character
lines, not turn-beats-only -- a strong writer avoids on-the-nose emotion anywhere.)
Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_hygiene import flag_on_the_nose  # noqa: E402
from nodes._otr_line_composer import LineRequest, compose_line  # noqa: E402

_NODES = Path(__file__).resolve().parents[1] / "nodes"
_CLEAN = "Pinky, the decay curve gives us ten months, and Skip has not finished the rig yet."
_NOSE = "I'm so scared, Pinky, and this is dangerous beyond anything we ever once planned."


class TestFlagOnTheNose:
    @pytest.mark.parametrize("text", [
        "I'm scared, Skip.",
        "I am terrified of what comes next.",
        "This is dangerous and you know it.",
        "I feel afraid, honestly.",
    ])
    def test_flags(self, text):
        assert flag_on_the_nose(text)[0] is True

    @pytest.mark.parametrize("text", [
        _CLEAN,
        "The numbers don't lie, and neither do you when you go quiet like that.",
    ])
    def test_clean_passes(self, text):
        assert flag_on_the_nose(text)[0] is False


def _req():
    return LineRequest(
        speaker="ALICE", intent="reveal", mood="tense", target_words=15,
        canon_header="TITLE: x\nSETTING: y\nTIME: z\nPREMISE: w",
        last_lines=[("BOB", "What did you find?")],
    )


class TestRerollOnNose:
    def test_on_the_nose_draft_rerolls(self):
        calls = {"n": 0}

        def mock(messages, *, temperature, max_new_tokens):
            calls["n"] += 1
            return _NOSE if calls["n"] == 1 else _CLEAN

        res = compose_line(creative_fn=mock, req=_req())
        assert calls["n"] == 2
        assert res.text == _CLEAN

    def test_clean_no_reroll(self):
        calls = {"n": 0}

        def mock(messages, *, temperature, max_new_tokens):
            calls["n"] += 1
            return _CLEAN

        res = compose_line(creative_fn=mock, req=_req())
        assert calls["n"] == 1


class TestEscalationPrompt:
    def test_beat_prompt_has_escalation(self):
        src = " ".join((_NODES / "_otr_outline.py").read_text(
            encoding="utf-8").split())
        assert "RAISE THE STAKE" in src


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
