"""`num_characters` is a REQUEST, not a cap.

OPERATOR DIRECTIVE 2026-08-12, ALL BANKS: *"remove the cap, the story can have
as many characters as it wants."*

WHY IT CAME UP. The `scifi_news_pro` leg of the cross-bank writer gate died
after four attempts with `UNKNOWN_SPEAKER: INMATE #347`. The model had NOT
ignored the roster -- the roster is enumerated in every attempt's prompt. The
profile requested a cast of 2, the rolled seed was rising deaths in LA County
Jail, and the module offers no bit-part or incidental-speaker channel. A premise
that needs an inmate voice, given two slots, leaves the model a choice between
inventing a speaker and distorting the story. It invented, four times.

So the request stopped being enforced. This mirrors the standing
no-word-count-chasing rule exactly: `target_words` is a request the pipeline
tries to honour and never refuses over, and `num_characters` now behaves the
same way.

WHAT STILL BINDS, AND WHY IT IS NOT A POLICY CAP. `_make_casting_validator`
enforces *"two characters never share a voice"*, and `_deal_voice_menu` refuses
when the pool is short. So the cast genuinely cannot exceed the number of
distinct voices in stock -- the alternative is two characters speaking with one
voice, which the operator names as a correctness defect. `MAX_SPEAKING_CAST`
gates that EARLY, in the pitch and treatment validators, rather than letting a
whole script be written and then failing at casting.
"""
from __future__ import annotations

import inspect

import pytest

from nodes import _otr_scifi_fable2 as fable2
from nodes import OTR_LedgerScriptWriter as writer_node


def voice_stock_size():
    """The live pool, the same way `_deal_voice_menu` counts it."""
    from config import cast_pools

    return len(cast_pools.open_voice_pool(set()))


# ---------------------------------------------------------------------------
# The ceiling is the VOICE STOCK, and the constants must not drift from it
# ---------------------------------------------------------------------------
def test_MAX_SPEAKING_CAST_equals_the_live_voice_stock():
    """The constant is a literal because pydantic needs one at class-build
    time. This is what keeps that literal honest: grow or shrink the pool and
    this reports the drift instead of letting the cap quietly mismatch."""
    assert fable2.MAX_SPEAKING_CAST == voice_stock_size(), (
        "MAX_SPEAKING_CAST (%d) no longer matches the voice stock (%d) -- "
        "update the constant, or the cast can be validated to a size casting "
        "cannot fill" % (fable2.MAX_SPEAKING_CAST, voice_stock_size()))


def test_the_node_widget_bound_agrees_with_the_writer():
    """`INPUT_TYPES` cannot import the writer module (registration-time import
    order), so the bound is duplicated. This is the check that makes the
    duplicate safe."""
    assert writer_node._FABLE2_MAX_CAST == fable2.MAX_SPEAKING_CAST


def test_the_widget_advertises_the_real_ceiling_not_the_old_cap():
    spec = writer_node.OTR_LedgerScriptWriter.INPUT_TYPES()
    for section in ("required", "optional"):
        entry = spec.get(section, {}).get("num_characters")
        if entry:
            opts = entry[1]
            assert opts["max"] == fable2.MAX_SPEAKING_CAST, (
                "the widget still caps at %r" % opts["max"])
            assert opts["min"] == 1
            return
    pytest.fail("num_characters widget not found")


# ---------------------------------------------------------------------------
# The REQUEST is no longer enforced
# ---------------------------------------------------------------------------
def test_a_treatment_may_EXCEED_the_requested_cast_size():
    """THE DIRECTIVE. A request of 2 must not refuse a cast of 4."""
    check = fable2._make_treatment_validator(
        dossier=None, n_max=2, provenance={}, digest="")
    treatment = fable2.Treatment(
        title="T", dramatic_question="Q?", setting="S",
        cast_shapes=[fable2.CastShape(name=n, role="r", want="w",
                                      pressure="p", register="dry")
                     for n in ("Ada", "Bo", "Cy", "Di")],
        turn="turn",
        priced_ending={"choice": "she stays", "cost_paid": "the archive burns"},
        news_thread="n")
    assert check(treatment) is None, (
        "a story that wants 4 voices was refused because 2 were requested")


def test_a_pitch_may_EXCEED_the_requested_cast_size():
    check = fable2._make_pitch_validator([{"name": "card"}], n_max=2)
    pitch = fable2.Pitch.model_construct(cast_size=5, frame_card="card")
    assert check(pitch) is None


# ---------------------------------------------------------------------------
# The REAL ceiling still binds, and fails EARLY
# ---------------------------------------------------------------------------
def test_a_treatment_beyond_the_VOICE_STOCK_is_refused_by_the_validator():
    """Refused in the treatment pass, not after a whole script is written."""
    check = fable2._make_treatment_validator(
        dossier=None, n_max=2, provenance={}, digest="")
    oversized = fable2.Treatment.model_construct(
        cast_shapes=[object()] * (fable2.MAX_SPEAKING_CAST + 1))
    verdict = check(oversized)
    assert verdict and "voices in stock" in verdict


def test_a_pitch_beyond_the_VOICE_STOCK_is_refused():
    check = fable2._make_pitch_validator([{"name": "card"}], n_max=2)
    pitch = fable2.Pitch.model_construct(
        cast_size=fable2.MAX_SPEAKING_CAST + 1, frame_card="card")
    verdict = check(pitch)
    assert verdict and "voices in stock" in verdict


def test_the_schema_admits_a_cast_up_to_the_voice_stock():
    """The pydantic bound moved with the policy -- an 8-item literal would now
    refuse two legal voices."""
    for model, field in ((fable2.Treatment, "cast_shapes"),
                         (fable2.CastingVoices, "cast")):
        meta = model.model_fields[field].metadata
        maxes = [m.max_length for m in meta if hasattr(m, "max_length")]
        assert maxes and maxes[0] == fable2.MAX_SPEAKING_CAST, (
            "%s.%s caps at %r" % (model.__name__, field, maxes))


def test_two_characters_still_never_share_a_voice():
    """The invariant the ceiling exists to protect. If this ever stops being
    enforced, the ceiling is arbitrary and should be reconsidered -- not
    silently kept."""
    source = inspect.getsource(fable2._make_casting_validator)
    assert "already taken" in source and "never share a voice" in source


# ---------------------------------------------------------------------------
# The prompt asks rather than commands
# ---------------------------------------------------------------------------
def test_the_prompt_no_longer_calls_the_request_a_CEILING():
    """The model was being told N_MAX was a ceiling while the code no longer
    treats it as one -- the prompt and the validator must not disagree."""
    source = inspect.getsource(fable2)
    assert "N_MAX (speaking-character ceiling)" not in source
    assert "REQUESTED cast size" in source
    assert "a REQUEST, not a limit" in source
