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
  2. `_make_casting_repair` -- the row already carries `timbre`, the
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
    repair = F2._make_casting_repair(_menu(), ["Dr. Chen"])
    fixed = repair(_payload(hedge, "v1"), _gender_error(hedge, "v1"))

    assert fixed is not None, f"{hedge!r} should have been resolved from v1"
    assert fixed.cast[0].gender == "male"
    assert fixed.cast[0].timbre == "v1"


def test_the_resolved_gender_follows_the_VOICE_not_a_default():
    """Proves it is reading the menu rather than defaulting to male: the same
    hedge with a female timbre must come back female."""
    repair = F2._make_casting_repair(_menu(), ["Dr. Chen"])
    fixed = repair(_payload("both", "v2"), _gender_error("both", "v2"))

    assert fixed is not None
    assert fixed.cast[0].gender == "female"


def test_an_unknown_timbre_fails_CLOSED_to_the_model():
    """No voice, no answer. Returning a coin flip here would be worse than
    the failure it replaces -- the LLM repair rung runs instead."""
    repair = F2._make_casting_repair(_menu(), ["Dr. Chen"])
    assert repair(_payload("both", "nope"), _gender_error("both", "nope")) is None


def test_an_unrelated_validation_error_is_left_alone():
    """Scoped to OUR failure class -- another error must reach its own typed
    repair prompt rather than being swallowed here."""
    repair = F2._make_casting_repair(_menu(), ["Dr. Chen"])
    with pytest.raises(ValidationError) as caught:
        F2.CastingVoices.model_validate({"cast": [{"name": "X"}]})

    assert repair(json.dumps({"cast": [{"name": "X"}]}), caught.value) is None


def test_unparseable_output_fails_closed():
    repair = F2._make_casting_repair(_menu(), ["Dr. Chen"])
    assert repair("not json at all", _gender_error("both")) is None


def test_a_row_that_was_already_valid_is_not_rewritten():
    """The repair only touches rows it must. If nothing needed fixing it
    returns None so the real error still surfaces."""
    repair = F2._make_casting_repair(_menu(), ["Dr. Chen"])
    assert repair(_payload("male"), _gender_error("both")) is None


# --------------------------------------------------------------------------- #
# wiring -- an unwired repair is inert, and the suite would still be green
# --------------------------------------------------------------------------- #
def test_the_repair_is_actually_wired_into_the_casting_pass():
    """The defect this guards against is the fix existing and never running.
    `_pass_casting` must hand it to the dispatching factory."""
    import inspect

    source = inspect.getsource(F2._pass_casting)
    assert "deterministic_repair=_make_casting_repair(menu, speakers)" in source


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


# --------------------------------------------------------------------------- #
# Bug Bible 10.08 -- the name pool gets the final word, and LEMMY is exempt
# --------------------------------------------------------------------------- #
#
# Operator, 2026-08-24: "it has to pull from the pool and pick the gender
# first". 10.08 is the law: two correlated attributes of one entity from two
# independent generators are each correct alone and incoherent together (the
# MALIK-HIBBERT-voiced-female defect). Here the NAME is sealed in the script,
# so our pool CLASSIFIES it and the voice follows -- the inverse of the legacy
# lane, which names the cast before the script exists and may rename instead.
def _mixed_menu():
    """A menu with a spare voice of each gender, so a corrected gender has
    somewhere to go."""
    return F2.VoiceMenu(entries=(
        F2.VoiceMenuEntry(menu_id="m1", gender="male",
                          description="gravelly", preset="v2/en_speaker_1"),
        F2.VoiceMenuEntry(menu_id="m2", gender="male",
                          description="dry", preset="v2/en_speaker_3"),
        F2.VoiceMenuEntry(menu_id="f1", gender="female",
                          description="clipped", preset="v2/en_speaker_2"),
        F2.VoiceMenuEntry(menu_id="f2", gender="female",
                          description="warm", preset="v2/en_speaker_4"),
    ))


def _named_payload(name, gender, timbre):
    return json.dumps({"cast": [{
        "name": name, "role": "lead",
        "character_description": "a tired physicist",
        "gender": gender, "timbre": timbre,
    }]})


def _named_error(name, gender, timbre) -> ValidationError:
    with pytest.raises(ValidationError) as caught:
        F2.CastingVoices.model_validate(
            json.loads(_named_payload(name, gender, timbre)))
    return caught.value


def _pool_name(gender: str) -> str:
    """A first name our OWN pool confidently classifies, so the test cannot
    drift from the curated vocabulary."""
    from config import cast_pools

    for candidate in cast_pools.FIRST_NAMES_BY_GENDER[gender]:
        if cast_pools.gender_of_first_name(candidate) == gender:
            return candidate
    raise AssertionError(f"no confidently-{gender} name in the pool")


def test_the_name_pool_overrules_a_voice_that_disagrees_with_it():
    """A hedge resolved from a MALE voice, on a confidently-female name, must
    come back female -- and the voice must move with it, or the validator
    would refuse the very pair this repaired."""
    female_name = _pool_name("female")
    repair = F2._make_casting_repair(_mixed_menu(), [])

    fixed = repair(_named_payload(female_name, "both", "m1"),
                   _named_error(female_name, "both", "m1"))

    assert fixed is not None
    assert fixed.cast[0].gender == "female"
    assert fixed.cast[0].timbre.startswith("f"), (
        "the voice must follow the corrected gender, not stay male")


def test_a_name_the_pool_cannot_classify_keeps_the_voice_answer():
    """`gender_of_first_name` returns 'unknown' outside the curated pool and
    callers must never force a repair on that -- so an invented sci-fi name
    keeps the gender its chosen voice implies."""
    repair = F2._make_casting_repair(_mixed_menu(), [])

    fixed = repair(_named_payload("Zyrelle-9", "both", "m1"),
                   _named_error("Zyrelle-9", "both", "m1"))

    assert fixed is not None
    assert fixed.cast[0].gender == "male"
    assert fixed.cast[0].timbre == "m1", "no voice change was warranted"


def test_LEMMY_is_EXEMPT_from_the_name_pool_reclassification():
    """THE CAMEO IS PINNED. His name, gender and audition-proven Cockney voice
    come from LEMMY_PROFILE, and 10.08 tells us to exempt an explicitly named
    entity. Reclassifying the recurring cameo from a name pool is the one
    repair that could regress a settled character."""
    from config import cast_pools

    lemmy = str(cast_pools.LEMMY_PROFILE["name"])
    repair = F2._make_casting_repair(_mixed_menu(), [])

    fixed = repair(_named_payload(lemmy, "both", "m1"),
                   _named_error(lemmy, "both", "m1"))

    assert fixed is not None
    assert fixed.cast[0].gender == "male", "LEMMY is male, from his profile"
    assert fixed.cast[0].timbre == "m1", (
        "LEMMY's voice must not be reshuffled by the name-pool rung")


def test_the_classifier_is_the_legacy_lane_s_not_a_second_copy():
    """One classifier, not two -- the same `gender_of_first_name` the legacy
    casting path uses. A private copy here is the defect class fixed three
    times today."""
    import inspect

    # The delegation now sits one indirection away, in the helper that
    # strips the honorific first -- but it is still the LEGACY LANE's
    # classifier and still not a second copy.
    source = inspect.getsource(F2._pool_gender_for_label)
    assert "_POOLS.gender_of_first_name(" in source


# --------------------------------------------------------------------------- #
# THE SAFETY NET -- a finished episode is completed, never discarded
# --------------------------------------------------------------------------- #
#
# Operator: "I don't want fails ... it has to pull from the pool and pick the
# gender first". The kibitz r1 panel refuted the obvious move (loosening
# `_make_casting_validator`): `_assign_voices` runs AFTER the retry ladder and
# raises on a speaker it cannot find, so accepting a subset relocates the kill
# one seam later with the audio work already spent. Equality stays strict; the
# CAST is completed to satisfy it.
from nodes._otr_structured_call import PostValidationError  # noqa: E402


def _stocked_menu():
    return F2.VoiceMenu(entries=tuple(
        F2.VoiceMenuEntry(menu_id=f"{g[0]}{i}", gender=g,
                          description="d", preset=f"v2/en_speaker_{g[0]}{i}")
        for g in ("male", "female") for i in (1, 2, 3)
    ))


_COVERAGE_ERROR = PostValidationError(
    "cast names ['A'] != script speakers ['A', 'B'] -- cast EXACTLY the "
    "script's speakers")


def _one_row(name="Dr. Domitilla Del Vecchio", gender="female", timbre="f1"):
    return {"cast": [{"name": name, "role": "r",
                      "character_description": "d",
                      "gender": gender, "timbre": timbre}]}


def test_an_uncast_speaker_is_minted_so_the_episode_ships():
    speakers = ["Dr. Domitilla Del Vecchio", "DR. MICHAEL ELOTWIZ"]
    repair = F2._make_casting_repair(_stocked_menu(), speakers)

    fixed = repair(json.dumps(_one_row()), _COVERAGE_ERROR)

    assert fixed is not None, "a finished episode must not be discarded"
    assert [c.name for c in fixed.cast] == speakers


def test_the_minted_row_picks_GENDER_FIRST_from_our_own_name_pool():
    """Bug Bible 10.08: one attribute authoritative, the other reconciled to
    it. MICHAEL is male in our pool, so the row is male and only THEN is a
    male voice taken -- never a voice first with a gender rationalised after."""
    speakers = ["Dr. Domitilla Del Vecchio", "DR. MICHAEL ELOTWIZ"]
    repair = F2._make_casting_repair(_stocked_menu(), speakers)

    fixed = repair(json.dumps(_one_row()), _COVERAGE_ERROR)
    minted = [c for c in fixed.cast if c.name == "DR. MICHAEL ELOTWIZ"][0]

    assert minted.gender == "male"
    assert minted.timbre.startswith("m"), "the voice must follow the gender"


def test_an_honorific_does_not_hide_the_given_name_from_the_pool():
    """CAUGHT BY THIS FILE'S OWN TEST before it shipped. `gender_of_first_name`
    reads only the FIRST token by design, so a script label like
    `DR. MICHAEL ELOTWIZ` handed it `DR.` -- unknown -- and MICHAEL was minted
    FEMALE on the first live-shaped run. Script labels routinely carry an
    honorific; treatment names do not."""
    assert F2._pool_gender_for_label("DR. MICHAEL ELOTWIZ") == "male"
    assert F2._pool_gender_for_label("MICHAEL ELOTWIZ") == "male"
    # The raw classifier is what it is; the helper is what fixes it.
    from config import cast_pools
    assert cast_pools.gender_of_first_name("DR. MICHAEL ELOTWIZ") == "unknown"


def test_a_name_outside_the_pool_is_stable_not_defaulted():
    """10.08 forbids forcing a repair on a name we cannot confidently gender,
    so an invented sci-fi label takes an ISOLATED-rng coin -- but the SAME
    coin every run, or the episode would not reproduce."""
    speakers = ["Dr. Domitilla Del Vecchio", "Zyrelle-9"]
    first = F2._make_casting_repair(_stocked_menu(), speakers)(
        json.dumps(_one_row()), _COVERAGE_ERROR)
    second = F2._make_casting_repair(_stocked_menu(), speakers)(
        json.dumps(_one_row()), _COVERAGE_ERROR)

    a = [c for c in first.cast if c.name == "Zyrelle-9"][0]
    b = [c for c in second.cast if c.name == "Zyrelle-9"][0]
    assert (a.gender, a.timbre) == (b.gender, b.timbre)


def test_two_characters_never_share_a_voice_even_when_minting():
    speakers = ["Dr. Domitilla Del Vecchio", "DR. MICHAEL ELOTWIZ", "Zyrelle-9"]
    repair = F2._make_casting_repair(_stocked_menu(), speakers)

    fixed = repair(json.dumps(_one_row()), _COVERAGE_ERROR)
    timbres = [c.timbre for c in fixed.cast]

    assert len(set(timbres)) == len(timbres), timbres


def test_exhausted_stock_fails_CLOSED_rather_than_reusing_a_larynx():
    """Two characters sharing one voice is the invariant `_assign_voices` and
    `_assert_unique_bark_voices` both exist to protect. A confusing episode is
    worse than a retry, so the rung declines."""
    thin = F2.VoiceMenu(entries=(
        F2.VoiceMenuEntry(menu_id="f1", gender="female",
                          description="d", preset="v2/en_speaker_f1"),
    ))
    speakers = ["Dr. Domitilla Del Vecchio", "DR. MICHAEL ELOTWIZ"]

    assert F2._make_casting_repair(thin, speakers)(
        json.dumps(_one_row()), _COVERAGE_ERROR) is None


def test_a_row_naming_nobody_in_the_sealed_script_is_dropped():
    """The model returning the TREATMENT's people is the live failure. A cast
    row is not dialogue, so dropping it loses no spoken word."""
    speakers = ["Dr. Domitilla Del Vecchio"]
    repair = F2._make_casting_repair(_stocked_menu(), speakers)
    payload = _one_row()
    payload["cast"].append({"name": "Dr. Michael Elowitz", "role": "r",
                            "character_description": "d",
                            "gender": "male", "timbre": "m1"})

    fixed = repair(json.dumps(payload), _COVERAGE_ERROR)
    assert [c.name for c in fixed.cast] == speakers


def test_an_already_complete_cast_is_left_alone():
    """A true no-op when nothing is wrong -- 10.08's verify demands it."""
    speakers = ["Dr. Domitilla Del Vecchio"]
    repair = F2._make_casting_repair(_stocked_menu(), speakers)
    assert repair(json.dumps(_one_row()), _COVERAGE_ERROR) is None


def test_an_unrelated_post_validation_error_is_not_hijacked():
    speakers = ["Dr. Domitilla Del Vecchio"]
    repair = F2._make_casting_repair(_stocked_menu(), speakers)
    assert repair(json.dumps(_one_row()),
                  PostValidationError("timbre f1 already taken")) is None


# --------------------------------------------------------------------------- #
# the casting prompt must not contradict itself
# --------------------------------------------------------------------------- #
def test_the_prompt_keys_cast_shapes_by_the_SCRIPT_label():
    """THE ROOT OF THE LIVE FAILURE (kibitz r1, Cursor and Fable). The prompt
    dumped treatment names above an instruction to return script speakers; on
    a salvaged episode those disagree, the model sensibly answered with the
    treatment's people, and the validator refused it."""
    import inspect

    source = inspect.getsource(F2._pass_casting)
    assert "_shapes_for_speakers(treatment, speakers, final_draft)" in source
    assert "TREATMENT CAST SHAPES" not in source, (
        "the shapes block must be keyed by the label the model must return")
