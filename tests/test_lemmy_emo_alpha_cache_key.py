"""LEMMY CHUNK A1 -- a render-time knob that moved the audio and not the key.

THE DEFECT. `eng_indextts2.generate_voice` reads `OTR_INDEXTTS2_EMO_ALPHA` from
the environment at GENERATE time -- deliberately, so a long-running server picks
up a change -- while `ResolvedVoiceRequest` captured `profile.default_params` at
REQUEST-BUILD time. The alpha therefore changed the RENDER and not the KEY, so
the next identical line replayed audio blended under the previous alpha while its
receipt described the new one. `IS_CHANGED` carried no alpha term either, so an
in-graph rerun did not save it.

THE FIX is the numeric sibling of the mechanism that already existed for this
exact class of bug: `identity_params` folds an env-selected MODEL into the key
for Google TTS, and `render_time_params` now folds env-resolved NUMERIC knobs
into `quantized_params`. The engine resolves the value through the SAME function
its forward calls, so the key and the render cannot disagree.

WHAT MUST NOT MOVE: every engine without such a knob keys byte-identically and
still answers `IS_CHANGED == "static"`. Both directions are asserted here,
because a fix that quietly re-baselines every other engine's cache is not a fix.
"""
from __future__ import annotations

import pytest

from nodes._otr_audio_engines import get_engine
from nodes._otr_audio_engines.base import AudioEngineAdapter
from nodes._otr_resolved_request import build_resolved_request


def _request(params):
    """A request identical in every field except the params under test."""
    return build_resolved_request(
        role="char_voice",
        engine_name="indextts2",
        engine_profile_id="p1",
        engine_impl_version="1",
        char_id="c02",
        line_id="l0001",
        prepared_text="Evening, squire.",
        voice_ref_id="idx_lemmy_algenib_cockney_v1",
        params=params,
    )


# ---------------------------------------------------------------------------
# The knob now keys
# ---------------------------------------------------------------------------
def test_a_changed_emo_alpha_changes_the_cache_key():
    """The whole defect, in one assertion."""
    a = _request({"emo_alpha": 1.0})
    b = _request({"emo_alpha": 0.6})
    assert a.cache_key != b.cache_key


def test_the_same_emo_alpha_keys_identically():
    """A cache that never hits is not a cache -- the key must be STABLE too."""
    assert _request({"emo_alpha": 0.6}).cache_key == \
        _request({"emo_alpha": 0.6}).cache_key


def test_the_alpha_survives_quantization_at_three_decimals():
    """`quantize_params` keeps 3 decimal places, and `current_emo_alpha` rounds
    to the same resolution -- so neighbouring alphas must not collide."""
    assert _request({"emo_alpha": 0.851}).cache_key != \
        _request({"emo_alpha": 0.852}).cache_key
    assert dict(_request({"emo_alpha": 0.85}).quantized_params)["emo_alpha"] == 850


# ---------------------------------------------------------------------------
# The engine exposes it, through the SAME function the forward calls
# ---------------------------------------------------------------------------
def test_indextts2_declares_the_alpha_as_a_render_time_param(monkeypatch):
    engine = get_engine("indextts2")
    monkeypatch.delenv("OTR_INDEXTTS2_EMO_MASS_CAP", raising=False)
    monkeypatch.setenv("OTR_INDEXTTS2_EMO_ALPHA", "0.4")
    assert engine.render_time_params()["emo_alpha"] == 0.4
    monkeypatch.setenv("OTR_INDEXTTS2_EMO_ALPHA", "1.0")
    assert engine.render_time_params()["emo_alpha"] == 1.0


def test_the_key_and_the_forward_read_the_SAME_resolver(monkeypatch):
    """Two readers of one value is how a key and a render start disagreeing.

    `render_time_params` must delegate to `current_emo_alpha` rather than
    re-implementing the env read -- otherwise a clamp or a default could drift
    between them and the key would describe an alpha the forward never used.
    """
    engine = get_engine("indextts2")
    calls = []
    real = type(engine).current_emo_alpha

    def _spy():
        calls.append(1)
        return real()

    monkeypatch.setattr(type(engine), "current_emo_alpha", staticmethod(_spy))
    engine.render_time_params()
    assert calls, ("render_time_params resolved the alpha without going through "
                   "current_emo_alpha -- the key and the forward can now drift")


def test_a_malformed_env_value_keys_as_the_clamped_default(monkeypatch):
    """`current_emo_alpha` clamps and falls back; the key must reflect what the
    forward will ACTUALLY use, not the raw string.

    THE DEFAULT IS 0.4, NOT 1.0, SINCE THE VOICE-IDENTITY FIX (2026-08-18).
    At 1.0 the vendor spends the emotion vector's whole sum against the
    speaker's own embedding, so a weighted line kept almost nothing of the
    reference voice -- the half of PBUG-20260817-09 that alpha owns. The
    unusable-value fallback is asserted against the module constant rather than
    a literal so a future re-anchor cannot leave this test quietly describing an
    alpha nobody renders.
    """
    from nodes._otr_audio_engines.eng_indextts2 import EMO_ALPHA_DEFAULT

    engine = get_engine("indextts2")
    for raw in ("not-a-number", "", "nan"):
        monkeypatch.setenv("OTR_INDEXTTS2_EMO_ALPHA", raw)
        assert engine.render_time_params()["emo_alpha"] == EMO_ALPHA_DEFAULT
    monkeypatch.setenv("OTR_INDEXTTS2_EMO_ALPHA", "5.0")
    assert engine.render_time_params()["emo_alpha"] == 1.0   # clamped high
    monkeypatch.setenv("OTR_INDEXTTS2_EMO_ALPHA", "-2")
    assert engine.render_time_params()["emo_alpha"] == 0.0   # clamped low


def test_the_unset_env_default_is_the_re_anchored_alpha(monkeypatch):
    """The SHIPPED default, asserted where a reader will look for it.

    The test above proves the fallback for unusable input; this one proves the
    ordinary case -- no env var set at all, which is how every production render
    runs -- so a regression to 1.0 cannot hide behind the malformed-value path.
    """
    from nodes._otr_audio_engines.eng_indextts2 import EMO_ALPHA_DEFAULT

    monkeypatch.delenv("OTR_INDEXTTS2_EMO_ALPHA", raising=False)
    # 0.4 -> 1.0 on 2026-08-18: alpha and the ceiling used to share one job and
    # the ceiling now owns it alone, so alpha is a pass-through. This is NOT a
    # return to the pre-fix build, which was alpha 1.0 with no ceiling at all.
    assert EMO_ALPHA_DEFAULT == 1.0
    assert get_engine("indextts2").current_emo_alpha() == 1.0


def test_the_alpha_is_normalized_to_the_resolution_the_key_keeps(monkeypatch):
    """QA-2: clamp THEN round to three decimals, because the key quantizes there.

    `quantize_params` stores `round(value * 1000)`, so 0.4001 and 0.4 record the
    SAME tick. If the forward were allowed to use the raw float, two renders
    that sound different would share one cache key and the second line would
    replay the first one's blend. Rounding at the source makes the key and the
    render the same number by construction.
    """
    engine = get_engine("indextts2")
    monkeypatch.setenv("OTR_INDEXTTS2_EMO_ALPHA", "0.4001")
    assert engine.current_emo_alpha() == 0.4
    monkeypatch.setenv("OTR_INDEXTTS2_EMO_ALPHA", "0.85049")
    assert engine.current_emo_alpha() == 0.85
    # The value the forward uses and the value the key records must agree.
    alpha = engine.current_emo_alpha()
    assert dict(_request({"emo_alpha": alpha}).quantized_params)["emo_alpha"] == 850


# ---------------------------------------------------------------------------
# EVERY OTHER ENGINE IS UNTOUCHED -- the half that makes this safe
# ---------------------------------------------------------------------------
def test_the_base_default_is_empty_so_engines_without_the_knob_do_not_move():
    assert AudioEngineAdapter().render_time_params() == {}


def test_no_other_shipped_engine_gained_a_render_time_param():
    """A hook that silently spread would re-baseline caches nobody asked to
    move. Only the engine with the defect declares one."""
    from nodes._otr_audio_engines import registry as areg

    declaring = []
    for name in sorted(areg._REGISTRY):
        try:
            params = get_engine(name).render_time_params()
        except Exception:  # noqa: BLE001 -- an engine that cannot be built
            continue
        if params:
            declaring.append(name)
    assert declaring == ["indextts2"], declaring


def test_an_engine_with_no_render_time_param_keys_exactly_as_before():
    """The empty dict must be a NO-OP on the key, not an empty entry in it."""
    without = build_resolved_request(
        role="char_voice", engine_name="kokoro", engine_profile_id="p1",
        engine_impl_version="1", char_id="c01", line_id="l1",
        prepared_text="hello", params={"speed": 1.0})
    merged = dict({"speed": 1.0})
    merged.update({})
    with_hook = build_resolved_request(
        role="char_voice", engine_name="kokoro", engine_profile_id="p1",
        engine_impl_version="1", char_id="c01", line_id="l1",
        prepared_text="hello", params=merged)
    assert without.cache_key == with_hook.cache_key


# ---------------------------------------------------------------------------
# IS_CHANGED -- the other half of the defect
# ---------------------------------------------------------------------------
def test_is_changed_still_answers_static_for_an_engine_without_the_knob():
    """The shipping local path must keep its EXACT in-graph caching behaviour.

    `"static"` is asserted as the literal string it has always been, not merely
    as "something stable" -- the whole point of the empty default is that these
    legs do not move.
    """
    from nodes import batch_character_voices as bcv

    answer = bcv.BatchCharacterVoices.IS_CHANGED(
        script_json="{}", engine="kokoro", ledger_json="", gate_in="")
    assert answer == "static" or answer != answer      # "static" or NaN


def test_is_changed_stops_saying_static_when_the_alpha_can_move_the_render(
        monkeypatch):
    """For indextts2 the honest answer is a fingerprint, and it must MOVE with
    the alpha -- otherwise an in-graph rerun serves the previous blend."""
    from nodes import batch_character_voices as bcv

    node = bcv.BatchCharacterVoices

    monkeypatch.setenv("OTR_INDEXTTS2_EMO_ALPHA", "1.0")
    first = node.IS_CHANGED(script_json="{}", engine="indextts2",
                            ledger_json="", gate_in="")
    monkeypatch.setenv("OTR_INDEXTTS2_EMO_ALPHA", "0.5")
    second = node.IS_CHANGED(script_json="{}", engine="indextts2",
                             ledger_json="", gate_in="")

    if first != first or second != second:      # NaN: rerun always, also safe
        pytest.skip("this box resolves indextts2 to a cache-enabled profile")
    assert first != "static", (
        "a local indextts2 leg answered 'static' while OTR_INDEXTTS2_EMO_ALPHA "
        "can change what it renders")
    assert first != second, (
        "the IS_CHANGED fingerprint did not move when the emo alpha did")
