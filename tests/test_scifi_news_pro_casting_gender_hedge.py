"""A hedged casting gender is answered by the voice the model picked.

THE DEFECT (PBUG-20260824-03, live TWICE in one evening on the
`scifi_news_pro` fast-iteration loop). `CastVoice.gender` is
`Literal["male", "female"]` -- there is no third option, because every voice
in stock is one or the other. A model that will not commit writes a hedge,
and it does not repeat itself:

  * first failure : `gender='both'`
  * second failure: `gender='n/a'` -- hours later, WITH the targeted repair
    prompt already deployed, so the prompt fix is provably not sufficient.
    (`n/a` is a legal `age_band` value, so the model is confusing the fields.)

Both died after 2 attempts and killed a finished episode, which THE LAW
forbids: an audit may improve a story, it may never fail one.

THE FIX IS A TWO-RUNG LADDER, and neither rung guesses:

  1. `CastVoice._canonicalize_gender_synonyms` -- `Male`, `M`, `woman` are
     correct answers in a refused spelling. Canonicalized through the voice
     bank's own synonym authority, which PASSES THROUGH what it does not
     recognise, so a genuine hedge is NOT resolved here.
  2. `_make_casting_gender_repair` -- the row already carries `timbre`, the
     menu id the model CHOSE, and every menu entry has a gender. In this lane
     the voice IS the gender (that is exactly what `_make_casting_validator`
     enforces), so the model answered the question when it picked the larynx.

Deliberately NO name inference: a four-tier name->gender ladder was specced
and refused twice (r2 and r3 both NO). Unfixable rows fail closed to the LLM
repair rung rather than taking a coin flip.

CPU-only: no model, no GPU.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from nodes import _otr_scifi_news_pro as F2


# --------------------------------------------------------------------------- #
# a menu whose entries carry the ground-truth gender
# --------------------------------------------------------------------------- #
def _menu():
    """The real VoiceMenu shape, built from the module's own frozen
    dataclasses -- not a stub, so it cannot drift from what production
    passes to the repair."""
    return F2.VoiceMenu(entries=(
        F2.VoiceMenuEntry(menu_id="v1", gender="male",
                          description="gravelly", preset="v2/en_speaker_1"),
        F2.VoiceMenuEntry(menu_id="v2", gender="female",
                          description="clipped", preset="v2/en_speaker_2"),
    ))


def _payload(gender, timbre="v1"):
    return json.dumps({"cast": [{
        "name": "Dr. Chen", "role": "lead",
        "character_description": "a tired physicist",
        "gender": gender, "timbre": timbre,
    }]})


def _gender_error(gender, timbre="v1") -> ValidationError:
    """The REAL rejection, raised by the real schema -- not a hand-built stub.
    A stub could drift from what production actually raises."""
    with pytest.raises(ValidationError) as caught:
        F2.CastingVoices.model_validate(json.loads(_payload(gender, timbre)))
    return caught.value


# --------------------------------------------------------------------------- #
# rung 1 -- spellings, not decisions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ("male", "male"), ("female", "female"),
    ("Male", "male"), ("  FEMALE  ", "female"),
    ("m", "male"), ("f", "female"),
    ("man", "male"), ("woman", "female"),
])
def test_a_correct_answer_in_a_refused_spelling_is_accepted(raw, expected):
    row = F2.CastVoice(
        name="X", role="r", character_description="d",
        gender=raw, timbre="v1",
    )
    assert row.gender == expected


@pytest.mark.parametrize("hedge", ["both", "n/a", "nonbinary", "unspecified"])
def test_a_genuine_hedge_is_NOT_silently_resolved_at_rung_1(hedge):
    """THE NEGATIVE CONTROL FOR RUNG 1. If the synonym map ever grew a
    hedge->male entry, every hedge would become a silent coin flip and rung 2
    would never run. A hedge must still fail here."""
    with pytest.raises(ValidationError):
        F2.CastVoice(
            name="X", role="r", character_description="d",
            gender=hedge, timbre="v1",
        )


# --------------------------------------------------------------------------- #
# rung 2 -- the voice answers it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hedge", ["both", "n/a", "nonbinary", ""])
def test_a_hedged_gender_is_resolved_from_the_chosen_voice(hedge):
    """THE FIX, against the exact live shapes. `v1` is male, so the row is
    male -- read off the artifact, never guessed."""
    repair = F2._make_casting_gender_repair(_menu())
    fixed = repair(_payload(hedge, "v1"), _gender_error(hedge, "v1"))

    assert fixed is not None, f"{hedge!r} should have been resolved from v1"
    assert fixed.cast[0].gender == "male"
    assert fixed.cast[0].timbre == "v1"


def test_the_resolved_gender_follows_the_VOICE_not_a_default():
    """Proves it is reading the menu rather than defaulting to male: the same
    hedge with a female timbre must come back female."""
    repair = F2._make_casting_gender_repair(_menu())
    fixed = repair(_payload("both", "v2"), _gender_error("both", "v2"))

    assert fixed is not None
    assert fixed.cast[0].gender == "female"


def test_an_unknown_timbre_fails_CLOSED_to_the_model():
    """No voice, no answer. Returning a coin flip here would be worse than
    the failure it replaces -- the LLM repair rung runs instead."""
    repair = F2._make_casting_gender_repair(_menu())
    assert repair(_payload("both", "nope"), _gender_error("both", "nope")) is None


def test_an_unrelated_validation_error_is_left_alone():
    """Scoped to OUR failure class -- another error must reach its own typed
    repair prompt rather than being swallowed here."""
    repair = F2._make_casting_gender_repair(_menu())
    with pytest.raises(ValidationError) as caught:
        F2.CastingVoices.model_validate({"cast": [{"name": "X"}]})

    assert repair(json.dumps({"cast": [{"name": "X"}]}), caught.value) is None


def test_unparseable_output_fails_closed():
    repair = F2._make_casting_gender_repair(_menu())
    assert repair("not json at all", _gender_error("both")) is None


def test_a_row_that_was_already_valid_is_not_rewritten():
    """The repair only touches rows it must. If nothing needed fixing it
    returns None so the real error still surfaces."""
    repair = F2._make_casting_gender_repair(_menu())
    assert repair(_payload("male"), _gender_error("both")) is None


# --------------------------------------------------------------------------- #
# wiring -- an unwired repair is inert, and the suite would still be green
# --------------------------------------------------------------------------- #
def test_the_repair_is_actually_wired_into_the_casting_pass():
    """The defect this guards against is the fix existing and never running.
    `_pass_casting` must hand it to the dispatching factory."""
    import inspect

    source = inspect.getsource(F2._pass_casting)
    assert "deterministic_repair=_make_casting_gender_repair(menu)" in source


def test_the_deterministic_rung_runs_before_the_prompt():
    """The shared factory tries a supplied deterministic repair on any
    ValidationError BEFORE building a repair prompt -- that ordering is what
    makes this cost zero model calls."""
    import inspect

    from nodes import _otr_repair_prompts as rp

    source = inspect.getsource(rp.make_dispatching_repair_factory)
    det = source.index("deterministic_repair is not None")
    prompt = source.index("_is_gender_literal_validation_error")
    assert det < prompt, (
        "the deterministic rung must be attempted before the typed prompt")
