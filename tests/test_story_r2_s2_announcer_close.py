"""Story-Quality R2 S2 -- announcer close = concrete final image, not thesis.

flag_thesis_close detects a moral/lesson/news-summary close; compose_announcer_outro
recomposes ONCE for an image (mirrors the F3 hedge recompose). Pure / CPU.
UTF-8 no BOM, SFW.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_hygiene import flag_thesis_close  # noqa: E402
from nodes._otr_line_composer import compose_announcer_outro  # noqa: E402

_NODES = Path(__file__).resolve().parents[1] / "nodes"


class TestFlagThesisClose:
    @pytest.mark.parametrize("text", [
        "Tonight's revelation is the cost of speed.",
        "Tonight’s revelation is the cost of speed.",   # curly apostrophe
        "And so the lesson is that haste makes waste.",
        "A story reminding us that integrity matters.",
        "It ends with Chris proving Skip right after all.",
        "The secret is now shared across the whole lab.",
        "This shows the price of ambition.",
    ])
    def test_flags_thesis(self, text):
        flagged, reason = flag_thesis_close(text)
        assert flagged is True
        assert reason

    @pytest.mark.parametrize("text", [
        "The pencil lies still on the empty console as the lights wink out.",
        "Swift climbs back to a safe orbit while the control room exhales.",
        "",
    ])
    def test_keeps_image_close(self, text):
        assert flag_thesis_close(text)[0] is False


class TestOutroRecompose:
    def test_thesis_close_recomposes_to_image(self):
        calls = {"n": 0}

        def mock(messages, *, temperature, max_new_tokens, stop=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return "And so the lesson is that haste makes waste tonight."
            return "The pencil lies still on the empty console as the lights wink out."

        res = compose_announcer_outro(
            creative_fn=mock,
            script_brief="A lab races a deadline to clear a contamination.",
            news_close_brief="The lab cleared its name before the games.",
            intro_text="Tonight, a lab races the clock.",
        )
        assert calls["n"] == 2  # one recompose
        assert "lesson is" not in res.text
        assert res.compose_flags == ("announcer_outro_image_recomposed",)

    def test_image_close_not_recomposed(self):
        calls = {"n": 0}

        def mock(messages, *, temperature, max_new_tokens, stop=None):
            calls["n"] += 1
            return "The pencil lies still on the empty console as the lights wink out."

        res = compose_announcer_outro(
            creative_fn=mock,
            script_brief="A lab races a deadline to clear a contamination.",
            news_close_brief="The lab cleared its name before the games.",
            intro_text="Tonight, a lab races the clock.",
        )
        assert calls["n"] == 1  # no recompose
        assert res.compose_flags == ("announcer_outro",)


class TestSourcePins:
    def test_outline_close_intent(self):
        src = " ".join((_NODES / "_otr_outline.py").read_text(
            encoding="utf-8").split())
        assert "Close on a concrete final image showing what changed" in src

    def test_outro_system_prompt(self):
        src = " ".join((_NODES / "_otr_line_composer.py").read_text(
            encoding="utf-8").split())
        assert "CLOSE ON A CONCRETE FINAL IMAGE" in src


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
