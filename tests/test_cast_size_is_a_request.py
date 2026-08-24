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

from nodes import _otr_scifi_news_pro as scifi_news_pro
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
    assert scifi_news_pro.MAX_SPEAKING_CAST == voice_stock_size(), (
        "MAX_SPEAKING_CAST (%d) no longer matches the voice stock (%d) -- "
        "update the constant, or the cast can be validated to a size casting "
        "cannot fill" % (scifi_news_pro.MAX_SPEAKING_CAST, voice_stock_size()))


def test_the_node_widget_bound_agrees_with_the_writer():
    """`INPUT_TYPES` cannot import the writer module (registration-time import
    order), so the bound is duplicated. This is the check that makes the
    duplicate safe."""
    assert writer_node._FABLE2_MAX_CAST == scifi_news_pro.MAX_SPEAKING_CAST


def test_the_widget_advertises_the_real_ceiling_not_the_old_cap():
    spec = writer_node.OTR_LedgerScriptWriter.INPUT_TYPES()
    for section in ("required", "optional"):
        entry = spec.get(section, {}).get("num_characters")
        if entry:
            opts = entry[1]
            assert opts["max"] == scifi_news_pro.MAX_SPEAKING_CAST, (
                "the widget still caps at %r" % opts["max"])
            assert opts["min"] == 1
            return
    pytest.fail("num_characters widget not found")


# ---------------------------------------------------------------------------
# The REQUEST is no longer enforced
# ---------------------------------------------------------------------------
def test_a_treatment_may_EXCEED_the_requested_cast_size():
    """THE DIRECTIVE. A request of 2 must not refuse a cast of 4."""
    check = scifi_news_pro._make_treatment_validator(
        dossier=None, n_max=2, provenance={}, digest="")
    treatment = scifi_news_pro.Treatment(
        title="T", dramatic_question="Q?", setting="S",
        cast_shapes=[scifi_news_pro.CastShape(name=n, role="r", want="w",
                                      pressure="p", register="dry")
                     for n in ("Ada", "Bo", "Cy", "Di")],
        turn="turn",
        priced_ending={"choice": "she stays", "cost_paid": "the archive burns"},
        news_thread="n")
    assert check(treatment) is None, (
        "a story that wants 4 voices was refused because 2 were requested")


def test_a_pitch_may_EXCEED_the_requested_cast_size():
    check = scifi_news_pro._make_pitch_validator([{"name": "card"}], n_max=2)
    pitch = scifi_news_pro.Pitch.model_construct(cast_size=5, frame_card="card")
    assert check(pitch) is None


# ---------------------------------------------------------------------------
# The REAL ceiling still binds, and fails EARLY
# ---------------------------------------------------------------------------
def test_a_treatment_beyond_the_VOICE_STOCK_is_refused_by_the_validator():
    """Refused in the treatment pass, not after a whole script is written."""
    check = scifi_news_pro._make_treatment_validator(
        dossier=None, n_max=2, provenance={}, digest="")
    oversized = scifi_news_pro.Treatment.model_construct(
        cast_shapes=[object()] * (scifi_news_pro.MAX_SPEAKING_CAST + 1))
    verdict = check(oversized)
    assert verdict and "voices in stock" in verdict


def test_a_pitch_beyond_the_VOICE_STOCK_is_refused():
    check = scifi_news_pro._make_pitch_validator([{"name": "card"}], n_max=2)
    pitch = scifi_news_pro.Pitch.model_construct(
        cast_size=scifi_news_pro.MAX_SPEAKING_CAST + 1, frame_card="card")
    verdict = check(pitch)
    assert verdict and "voices in stock" in verdict


def test_the_schema_admits_a_cast_up_to_the_voice_stock():
    """The pydantic bound moved with the policy -- an 8-item literal would now
    refuse two legal voices."""
    for model, field in ((scifi_news_pro.Treatment, "cast_shapes"),
                         (scifi_news_pro.CastingVoices, "cast")):
        meta = model.model_fields[field].metadata
        maxes = [m.max_length for m in meta if hasattr(m, "max_length")]
        assert maxes and maxes[0] == scifi_news_pro.MAX_SPEAKING_CAST, (
            "%s.%s caps at %r" % (model.__name__, field, maxes))


def test_two_characters_still_never_share_a_voice():
    """The invariant the ceiling exists to protect. If this ever stops being
    enforced, the ceiling is arbitrary and should be reconsidered -- not
    silently kept."""
    source = inspect.getsource(scifi_news_pro._make_casting_validator)
    assert "already taken" in source and "never share a voice" in source


# ---------------------------------------------------------------------------
# The prompt asks rather than commands
# ---------------------------------------------------------------------------
def test_the_prompt_no_longer_calls_the_request_a_CEILING():
    """The model was being told N_MAX was a ceiling while the code no longer
    treats it as one -- the prompt and the validator must not disagree."""
    source = inspect.getsource(scifi_news_pro)
    assert "N_MAX (speaking-character ceiling)" not in source
    assert "REQUESTED cast size" in source
    assert "a REQUEST, not a limit" in source


# ---------------------------------------------------------------------------
# The LEGACY lane's smaller stock degrades the request; it must never raise
# ---------------------------------------------------------------------------
#
# THE DEFECT (found 2026-08-24 by the character-selection trace). The widget
# spans BOTH lanes and advertises the dispatched lane's ceiling (10), but the
# legacy assembler seats only what its Bark stock can voice (6) -- and it
# enforced that by RAISING. An operator moving the slider to 7 on `original`
# or `media_archive` killed the run with an uncaught ValueError out of
# `assemble_pre_locked_rows`, after the RSS fetch, the bank roll and the
# story-contract build had already spent minutes, producing no episode.
def test_the_widget_can_out_ask_the_legacy_lane_and_that_is_expected():
    """The premise. These two bounds are SUPPOSED to differ -- the lanes draw
    from different voice stocks -- which is exactly why an over-request has to
    degrade rather than refuse."""
    from nodes import _otr_casting

    assert writer_node._FABLE2_MAX_CAST > _otr_casting._LEGACY_MAX_SPEAKING_CAST


def test_an_over_request_is_CLAMPED_by_lock_cast_never_raised():
    """THE FIX. `lock_cast` reduces an over-request to the largest cast this
    lane can actually voice, and says so at WARNING -- the operator asked for
    8 and is getting 6, and a silent clamp is how that becomes a mystery
    instead of a decision."""
    from nodes import _otr_casting

    source = inspect.getsource(_otr_casting.lock_cast)

    assert "if num_characters > _LEGACY_MAX_SPEAKING_CAST:" in source, (
        "lock_cast must clamp an over-request before the assembler sees it")
    assert "log.warning(" in source, (
        "a clamp the operator cannot see is a mystery, not a decision")


def test_the_assembler_still_REFUSES_a_programming_error():
    """The clamp must not become a blanket swallow. A 0 or a negative is a
    caller bug, not an operator request, and still raises -- CastLock's replay
    detonating with 'num_characters must be 1-6, got 0' is a real diagnostic
    the content-owned contract comments cite by name."""
    from nodes import _otr_casting

    for bad in (0, -1):
        with pytest.raises(ValueError):
            _otr_casting.assemble_pre_locked_rows(num_characters=bad)


def test_the_bound_and_the_clamp_read_ONE_constant():
    """Named rather than a bare 6 in two places: the assembler's bound and
    lock_cast's clamp have to move together, or an over-request starts
    raising again."""
    from nodes import _otr_casting

    assert _otr_casting._LEGACY_MAX_SPEAKING_CAST == 6
    bound = inspect.getsource(_otr_casting.assemble_pre_locked_rows)
    assert "1 <= num_characters <= _LEGACY_MAX_SPEAKING_CAST" in bound, (
        "the assembler's bound must read the shared constant, not a literal")
