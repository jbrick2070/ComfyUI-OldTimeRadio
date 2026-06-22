"""Story-Quality R2 S3 -- cliche + flat stage-business reject gate.

flag_cliche / flag_stage_business in _otr_line_hygiene (FLAGS ONLY); a flagged
voiced draft triggers ONE reroll via the existing compose_line repair path.
Pure / CPU (mock creative_fn). UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_hygiene import flag_cliche, flag_stage_business  # noqa: E402
from nodes._otr_line_composer import LineRequest, compose_line  # noqa: E402

_CLEAN = "Pinky, the decay curve gives us ten months, and Skip has not finished the rig yet."
_CLICHE = "Listen Pinky, you're playing with fire, and we must act before it is far too late."
_BUSINESS = "Alright everyone, I'll go check the readings and report straight back to all of you."


class TestFlagCliche:
    @pytest.mark.parametrize("text", [
        "You're playing with fire, Skip.",
        "This changes everything for the mission.",
        "We're not leaving anything to chance tonight.",
        "There's no turning back now.",
    ])
    def test_flags(self, text):
        assert flag_cliche(text)[0] is True

    def test_clean_passes(self):
        assert flag_cliche(_CLEAN)[0] is False


class TestFlagStageBusiness:
    @pytest.mark.parametrize("text", [
        "I'll go check the panel.",
        "I'll double-check the seals.",
        "I'll lock the lab down.",
        "Let me handle this.",
    ])
    def test_flags(self, text):
        assert flag_stage_business(text)[0] is True

    def test_clean_passes(self):
        assert flag_stage_business(_CLEAN)[0] is False


def _req():
    return LineRequest(
        speaker="ALICE", intent="reveal", mood="tense", target_words=15,
        canon_header="TITLE: x\nSETTING: y\nTIME: z\nPREMISE: w",
        last_lines=[("BOB", "What did you find?")],
    )


class TestReroll:
    @pytest.mark.parametrize("dirty", [_CLICHE, _BUSINESS])
    def test_flagged_draft_rerolls(self, dirty):
        calls = {"n": 0}

        def mock(messages, *, temperature, max_new_tokens):
            calls["n"] += 1
            return dirty if calls["n"] == 1 else _CLEAN

        res = compose_line(creative_fn=mock, req=_req())
        assert calls["n"] == 2          # one reroll
        assert res.text == _CLEAN

    def test_clean_draft_no_reroll(self):
        calls = {"n": 0}

        def mock(messages, *, temperature, max_new_tokens):
            calls["n"] += 1
            return _CLEAN

        res = compose_line(creative_fn=mock, req=_req())
        assert calls["n"] == 1
        assert res.text == _CLEAN


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
