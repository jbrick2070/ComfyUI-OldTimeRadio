"""tests/test_bug_local_292_callback_coercion.py

BUG-LOCAL-292 (2026-05-27): coerce LLM-emitted no-callback synonyms
to None on Stage1Beat.callback_to.

Live evidence: pending_20260527_212428 Story Room Extract attempt 1
returned `17 validation errors for StoryRoomExtractionSchema` with
each beat carrying `callback_to='none'` (literal string). The
schema gate was correct in rejecting -- the LLM was wrong --
but optional fields shouldn't require the LLM to emit a precise
JSON null.

The fix accepts any case-insensitive synonym for "no callback"
({'none', 'null', 'n/a', 'na', 'no', 'nil', 'nothing', '-', ''})
and coerces to None. Real beat_id values (bNNN) still validate;
genuine garbage still fails the regex check.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from nodes._otr_stage1_plan import Stage1Beat


def _beat(**overrides) -> dict:
    defaults = dict(
        beat_id="b002",
        speaker="REN BLACK",
        intent="State the refusal concretely.",
        length_target_words=30,
        emotional_register="guarded",
    )
    defaults.update(overrides)
    return defaults


class TestCallbackCoercion:
    def test_none_value_passes(self):
        beat = Stage1Beat(**_beat(callback_to=None))
        assert beat.callback_to is None

    def test_empty_string_passes(self):
        beat = Stage1Beat(**_beat(callback_to=""))
        assert beat.callback_to is None

    def test_whitespace_passes(self):
        beat = Stage1Beat(**_beat(callback_to="   "))
        assert beat.callback_to is None

    @pytest.mark.parametrize("synonym", [
        "none", "None", "NONE", "NoNe",
        "null", "NULL", "Null",
        "n/a", "N/A", "na", "NA",
        "no", "NO", "No",
        "nil", "NIL",
        "nothing", "Nothing",
        "-",
    ])
    def test_no_callback_synonyms_coerce_to_none(self, synonym):
        beat = Stage1Beat(**_beat(callback_to=synonym))
        assert beat.callback_to is None, (
            f"synonym {synonym!r} must coerce to None"
        )

    def test_valid_beat_id_round_trips(self):
        beat = Stage1Beat(**_beat(callback_to="b002"))
        assert beat.callback_to == "b002"

    def test_valid_beat_id_with_whitespace_stripped(self):
        beat = Stage1Beat(**_beat(callback_to="  b003  "))
        assert beat.callback_to == "b003"

    def test_garbage_string_still_rejected(self):
        with pytest.raises(ValidationError):
            Stage1Beat(**_beat(callback_to="not a beat id"))

    def test_non_string_non_none_rejected(self):
        with pytest.raises(ValidationError):
            Stage1Beat(**_beat(callback_to=42))
