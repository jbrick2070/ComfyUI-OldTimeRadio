"""Story-Quality R3 L1 -- objective-literal floor.

flag_objective_literal in _otr_line_hygiene (NARROW deterministic matcher,
FLAG ONLY) + the composer gate: under meta.story_quality_v2_enabled, a SHORT
character line that baldly restates the beat objective recomposes ONCE via the
existing <=1-reroll path and stamps `objective_literal_retry`. Flag OFF =>
byte-identical (no detection, no reroll). Pure / CPU (mock creative_fn).
UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_hygiene import flag_objective_literal  # noqa: E402
from nodes._otr_line_composer import LineRequest, compose_line  # noqa: E402

_OBJ = "get Marlowe to confess he leaked the codes"
_BALD = "Marlowe, confess that you leaked the codes."
_SUBTEXT = "Funny -- the safe was already open when I walked in this morning."


class TestFlagObjectiveLiteral:
    def test_bald_restatement_flagged(self):
        assert flag_objective_literal(_BALD, _OBJ)[0] is True

    def test_subtextual_line_not_flagged(self):
        assert flag_objective_literal(_SUBTEXT, _OBJ)[0] is False

    def test_long_elaborated_line_not_flagged(self):
        """A long line has room to play the objective indirectly -- the
        false-positive guard must never flag it even if it reuses the words."""
        long_line = (
            "You know, I keep turning it over -- Marlowe at the console, the "
            "codes blinking, and the way nobody will say out loud who finally "
            "leaked them when the lights went down that night."
        )
        assert flag_objective_literal(long_line, _OBJ)[0] is False

    def test_thin_objective_not_flagged(self):
        """An objective with < 2 distinctive content words is too thin to
        judge -- never flag."""
        assert flag_objective_literal("Go now.", "go")[0] is False

    def test_empty_inputs_safe(self):
        assert flag_objective_literal("", _OBJ) == (False, "")
        assert flag_objective_literal(_BALD, "") == (False, "")
        assert flag_objective_literal(None, None) == (False, "")


def _req(**over):
    base = dict(
        speaker="EDNA", intent="confront", mood="tense", target_words=15,
        canon_header="TITLE: x\nSETTING: y",
        last_lines=[("MARLOWE", "You wanted to see me?")],
        speaker_role="character",
        beat_objective=_OBJ,
        beat_subtext="she dreads he will deny it",
        beat_tension=5,
    )
    base.update(over)
    return LineRequest(**base)


class TestComposerGate:
    def test_flag_on_bald_draft_rerolls_and_stamps(self):
        calls = {"n": 0}

        def mock(messages, *, temperature, max_new_tokens):
            calls["n"] += 1
            return _BALD if calls["n"] == 1 else _SUBTEXT

        res = compose_line(
            creative_fn=mock, req=_req(),
        )
        assert calls["n"] == 2                       # exactly one reroll
        assert res.text == _SUBTEXT
        assert "objective_literal_retry" in res.compose_flags

    def test_flag_on_subtextual_draft_no_reroll(self):
        calls = {"n": 0}

        def mock(messages, *, temperature, max_new_tokens):
            calls["n"] += 1
            return _SUBTEXT

        res = compose_line(
            creative_fn=mock, req=_req(),
        )
        assert calls["n"] == 1
        assert "objective_literal_retry" not in res.compose_flags

    def test_flag_on_no_objective_skips_gate(self):
        """Flag on but no beat_objective frame -> gate skips (no reroll)."""
        calls = {"n": 0}

        def mock(messages, *, temperature, max_new_tokens):
            calls["n"] += 1
            return _BALD

        res = compose_line(
            creative_fn=mock,
            req=_req(beat_objective=""),
        )
        assert calls["n"] == 1
        assert "objective_literal_retry" not in res.compose_flags


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
