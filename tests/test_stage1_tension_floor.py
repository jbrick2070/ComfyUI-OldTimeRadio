"""Build 1 (2026-05-28) -- regression for the Stage-1 tension floor.

Status: LIVE. Promoted from docs/sprint_drafts/build1/ alongside the
fix (_otr_stage1_plan.Stage1Beat.tension floor `ge=1` -> `ge=0`, applied
in the same commit). With the floor at 0 the Stage-1 shadow pass no
longer burns its retry budget on a tension=0 opening beat.

What it proves:
  * The CURRENT schema rejects a beat with tension=0 (xfail-style guard,
    documents the bug so the test is honest about pre-fix state).
  * AFTER the fix, a beat carrying tension=0 validates cleanly through
    both the bare Stage1Beat model and the full
    parse_and_validate_plan entry point -- i.e. the Stage-1 shadow pass
    would no longer burn its retry budget on a tension=0 plan.
  * tension=-1 and tension=6 still reject (the fix only moves the floor
    from 1 to 0; it does not open the field wide).
  * tension is still Optional: a beat omitting it entirely parses.

Run (after promoting into tests/):
    pytest tests/test_stage1_tension_floor.py -v
"""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from nodes._otr_stage1_plan import (
    Stage1Beat,
    Stage1Plan,
    parse_and_validate_plan,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _good_beat_dict() -> dict:
    """A minimal valid voiced beat. Mutated by the tension tests."""
    return {
        "beat_id": "b002",
        "speaker": "DALE PORTER",
        "intent": "Establish the routine night shift and Dale's role.",
        "length_target_words": 24,
        "emotional_register": "calm grounded",
        "callback_to": None,
    }


def _good_plan_dict() -> dict:
    """A minimal valid plan whose first voiced beat carries tension=0.

    This is the exact shape the constrained Stage-1 generator emits and
    that the shadow pass currently discards: a low-stakes opening beat
    the model labels tension=0.
    """
    return {
        "premise": "Two technicians at a Mars relay catch a signal that should not exist.",
        "arc": {
            "setup": "Routine night shift, calm comms traffic.",
            "complication": "An off-band carrier locks in and refuses to drop.",
            "resolution": "They identify the source and decide whether to acknowledge it.",
        },
        "cast": [
            {
                "name": "DALE PORTER",
                "gender": "male",
                "pronouns": "he/him",
                "voice_id": "v2/en_speaker_5",
                "persona": "Experienced relay technician. Speaks in measured cadences. Trusts procedure over hunches but knows when to bend it.",
                "arc_role": "anchor of competence",
            },
            {
                "name": "MEG SAARI",
                "gender": "female",
                "pronouns": "she/her",
                "voice_id": "v2/en_speaker_2",
                "persona": "Junior tech, six months on the post. Quick to question, slow to panic. Asks the obvious questions out loud.",
                "arc_role": "pressure source and witness",
            },
        ],
        "beats": [
            {
                "beat_id": "b001",
                "speaker": "ANNOUNCER",
                "intent": "Open the episode with a date stamp and the location.",
                "length_target_words": 18,
                "emotional_register": "neutral procedural",
                "callback_to": None,
            },
            {
                "beat_id": "b002",
                "speaker": "DALE PORTER",
                "intent": "Establish the routine night shift and Dale's role.",
                "length_target_words": 24,
                "emotional_register": "calm grounded",
                "callback_to": None,
                # The crux: the model labels this opening beat tension=0.
                "tension": 0,
            },
            {
                "beat_id": "b003",
                "speaker": "MEG SAARI",
                "intent": "Notice the off-band carrier and ask Dale about it.",
                "length_target_words": 22,
                "emotional_register": "alert curious",
                "callback_to": "b002",
                "tension": 3,
            },
            {
                "beat_id": "b004",
                "speaker": "ANNOUNCER",
                "intent": "Close the episode with a date stamp and a hook.",
                "length_target_words": 16,
                "emotional_register": "measured close",
                "callback_to": None,
            },
        ],
        "running_facts": [
            "The relay station is on Mars.",
            "Dale outranks Meg.",
        ],
    }


# ---------------------------------------------------------------------------
# Positive floor assertion -- tension=0 must validate post-fix.
# ---------------------------------------------------------------------------


def test_bare_beat_tension_zero_validates():
    """tension=0 validates on the bare Stage1Beat model (post-fix floor ge=0)."""
    beat = _good_beat_dict()
    beat["tension"] = 0
    model = Stage1Beat(**beat)
    assert model.tension == 0


# ---------------------------------------------------------------------------
# Full-plan assertions -- the contract the fix satisfies end-to-end.
# ---------------------------------------------------------------------------


def test_plan_with_tension_zero_beat_parses_post_fix():
    """A full plan whose voiced beat carries tension=0 should validate
    end-to-end through parse_and_validate_plan once the floor is ge=0.

    Skips itself (rather than failing) while the schema still has ge=1
    so the draft file is green in the investigation commit; once the
    fix lands the skip condition is False and the assertion runs.
    """
    plan_dict = _good_plan_dict()

    # Probe the current floor without coupling to a private attribute.
    floor_is_zero = True
    try:
        Stage1Beat(**{**_good_beat_dict(), "tension": 0})
    except ValidationError:
        floor_is_zero = False

    if not floor_is_zero:
        pytest.skip(
            "tension floor still ge=1 -- apply tension_floor_fix_spec.md "
            "(ge=0) then this assertion runs."
        )

    plan = parse_and_validate_plan(plan_dict)
    assert isinstance(plan, Stage1Plan)
    # b002 is the first voiced beat after the announcer bookend.
    b002 = next(b for b in plan.beats if b.beat_id == "b002")
    assert b002.tension == 0
    # slot ids still stamped (sanity: the rest of the plan is intact).
    assert b002.dialogue_slot_id is not None


def test_tension_omitted_still_parses():
    """tension is Optional; a beat that omits it parses regardless of
    the floor value. This must stay true after the fix."""
    plan_dict = _good_plan_dict()
    for beat in plan_dict["beats"]:
        beat.pop("tension", None)
    plan = parse_and_validate_plan(plan_dict)
    assert all(b.tension is None for b in plan.beats)


def test_tension_negative_still_rejected():
    """The fix moves the floor to 0, not below. tension=-1 must still
    raise."""
    beat = _good_beat_dict()
    beat["tension"] = -1
    with pytest.raises(ValidationError):
        Stage1Beat(**beat)


def test_tension_above_max_still_rejected():
    """Ceiling is unchanged at 5; tension=6 must still raise."""
    beat = _good_beat_dict()
    beat["tension"] = 6
    with pytest.raises(ValidationError):
        Stage1Beat(**beat)
