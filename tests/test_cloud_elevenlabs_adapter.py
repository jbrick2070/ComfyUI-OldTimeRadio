"""ElevenLabs cloud TTS adapter -- unit tests (cloud-audio S1, 2026-07-03).

Proves the four properties the C3/C4/C7 grounding fixed:

* C4 -- the ELEVENLABS_VOICE socket is a plain STRING = the stable voice_id, and
  the adapter feeds ``provider_voice_id`` straight in as ``voice`` (no selector
  node, no display-label round-trip).
* fail-loud / no-fallback -- a blank line OR a missing provider_voice_id RAISES;
  the adapter NEVER inherits another voice or degrades to a local engine.
* C7 -- the per-LINE cost estimate is per-line scale (chars * $0.24/1K), not an
  episode-total that would trip the budget cap around line 9.
* delivery-vector -> stability is the only per-line knob, clamped to [0, 1].

invoke_partner_node + canonicalize_audio are the cloud/ffmpeg seams; both are
monkeypatched so the test runs OFFLINE with no credits and no ffmpeg.
"""
import numpy as np
import pytest
import soundfile as sf

from nodes._otr_audio_engines import registry as areg
from nodes._otr_audio_engines import eng_cloud_elevenlabs as EL


@pytest.fixture(autouse=True)
def _stable_elevenlabs_env(monkeypatch):
    monkeypatch.delenv("OTR_ELEVENLABS_MODEL_ID", raising=False)
    monkeypatch.delenv("OTR_ELEVENLABS_OUTPUT_FORMAT", raising=False)
    monkeypatch.delenv("OTR_ELEVENLABS_TEXT_NORMALIZATION", raising=False)


def _install_seams(monkeypatch, tmp_path, capture):
    """Patch the cloud invoke + canonicalize seams; capture the emitted kwargs."""
    import nodes._otr_shared.cloud_media_invoke as inv
    import nodes._otr_shared.cloud_media_canonical as canon

    def fake_invoke(node_key, inputs, timeout_s=None, estimated_usd=None):
        capture["node_key"] = node_key
        capture["inputs"] = dict(inputs)
        capture["estimated_usd"] = estimated_usd
        return {"audio_path": str(tmp_path / "cloud_raw.wav")}

    wav = tmp_path / "canon.wav"
    t = np.linspace(0, 0.5, 8000, dtype="float32")
    sf.write(str(wav), np.stack([0.1 * np.sin(2 * 3.14159 * 200 * t)] * 2, axis=1), 16000)

    class _Canon:
        path = str(wav)

    def fake_canon(result, spec, session=None):
        capture["canon_spec"] = dict(spec)
        return _Canon()

    monkeypatch.setattr(inv, "invoke_partner_node", fake_invoke)
    monkeypatch.setattr(canon, "canonicalize_audio", fake_canon)


def test_registered_as_elevenlabs():
    eng = areg.get_engine("elevenlabs")
    assert eng.name == "elevenlabs"
    assert set(eng.roles) == {"char_voice", "announcer_voice"}
    # C1/C2: cloud voice is opt-in ONLY -- never a byte-identical default.
    assert eng.default_roles == ()
    # C3: identity is adapter metadata -- no local ref clip, no local fallback.
    assert eng.requires_voice_ref is False
    assert eng.voice_ref_field == "provider_voice_id"
    assert eng.missing_ref_fallback is None


def test_voice_id_passed_straight_through_as_voice(monkeypatch, tmp_path):
    """C4: provider_voice_id (the id) IS the `voice` kwarg -- no selector node."""
    cap = {}
    _install_seams(monkeypatch, tmp_path, cap)
    eng = areg.get_engine("elevenlabs")
    out = eng.generate_voice("Signal lost. Stand by.", "EXAVITQu4vr4xnSDxMaL",
                             {"expressiveness": 0.8}, seed=7)
    assert cap["node_key"] == "cloud_elevenlabs_tts"
    assert cap["inputs"]["voice"] == "EXAVITQu4vr4xnSDxMaL"
    assert cap["inputs"]["text"] == "Signal lost. Stand by."
    assert cap["inputs"]["seed"] == 7
    assert cap["inputs"]["apply_text_normalization"] == "auto"
    assert cap["inputs"]["output_format"] == "mp3_44100_192"
    assert cap["inputs"]["model"] == {
        "model": "eleven_multilingual_v2",
        "similarity_boost": 0.75,
        "speed": 1.0,
        "use_speaker_boost": False,
        "style": 0.0,
    }
    # expressive delivery => LOW stability (1 - 0.8 = 0.2), clamped to [0,1].
    assert cap["inputs"]["stability"] == pytest.approx(0.2)
    # returns the Bark AUDIO contract from the canonical WAV.
    assert set(out) == {"waveform", "sample_rate"}
    assert out["waveform"].shape[0] == 1        # (1, C, T)


def test_emitted_kwargs_match_partner_inputs(monkeypatch, tmp_path):
    """The kwargs actually emitted == the declared _partner_inputs contract."""
    cap = {}
    _install_seams(monkeypatch, tmp_path, cap)
    eng = areg.get_engine("elevenlabs")
    eng.generate_voice("x", "vid123", None, seed=0)
    assert set(cap["inputs"]) == set(eng._partner_inputs(None))


def test_seed_is_reduced_to_partner_int_range(monkeypatch, tmp_path):
    cap = {}
    _install_seams(monkeypatch, tmp_path, cap)
    eng = areg.get_engine("elevenlabs")
    eng.generate_voice("x", "vid123", None, seed=2**31 + 13)
    assert cap["inputs"]["seed"] == 13


def test_eleven_v3_env_uses_partner_tts_shape(monkeypatch, tmp_path):
    cap = {}
    _install_seams(monkeypatch, tmp_path, cap)
    monkeypatch.setenv("OTR_ELEVENLABS_MODEL_ID", "eleven_v3")
    eng = areg.get_engine("elevenlabs")
    eng.generate_voice("x", "vid123", None, seed=0)
    assert cap["inputs"]["model"] == {
        "model": "eleven_v3",
        "similarity_boost": 0.75,
        "speed": 1.0,
    }


def test_invalid_partner_model_env_fails_before_invoke(monkeypatch, tmp_path):
    cap = {}
    _install_seams(monkeypatch, tmp_path, cap)
    monkeypatch.setenv("OTR_ELEVENLABS_MODEL_ID", "scribe_v2")
    eng = areg.get_engine("elevenlabs")
    with pytest.raises(ValueError, match="unsupported model"):
        eng.generate_voice("x", "vid123", None, seed=0)
    assert "node_key" not in cap


def test_invalid_partner_output_format_env_fails_before_invoke(monkeypatch, tmp_path):
    cap = {}
    _install_seams(monkeypatch, tmp_path, cap)
    monkeypatch.setenv("OTR_ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
    eng = areg.get_engine("elevenlabs")
    with pytest.raises(ValueError, match="unsupported output_format"):
        eng.generate_voice("x", "vid123", None, seed=0)
    assert "node_key" not in cap


def test_cost_estimate_is_per_line_scale():
    # C7: a 40-char line ~= $0.0096, NOT an episode-total. Cap trips ~line 9 only
    # if this were wrongly episode-scale.
    line = "x" * 40
    assert EL.estimate_tts_usd(line) == pytest.approx(40 * 0.24 / 1000.0)
    assert EL.estimate_tts_usd("") == pytest.approx(1 * 0.24 / 1000.0)  # min 1 char


def test_estimated_usd_is_forwarded_to_invoke(monkeypatch, tmp_path):
    cap = {}
    _install_seams(monkeypatch, tmp_path, cap)
    eng = areg.get_engine("elevenlabs")
    text = "The rails are molten and the timetable is a lie."
    eng.generate_voice(text, "vid123", None, seed=1)
    assert cap["estimated_usd"] == pytest.approx(EL.estimate_tts_usd(text))


def test_blank_line_raises_no_fallback(monkeypatch, tmp_path):
    _install_seams(monkeypatch, tmp_path, {})
    eng = areg.get_engine("elevenlabs")
    with pytest.raises(ValueError, match="blank line"):
        eng.generate_voice("   ", "vid123", None, seed=0)


def test_missing_voice_id_raises_no_fallback(monkeypatch, tmp_path):
    _install_seams(monkeypatch, tmp_path, {})
    eng = areg.get_engine("elevenlabs")
    with pytest.raises(ValueError, match="no provider_voice_id"):
        eng.generate_voice("a real line", "", None, seed=0)


def test_stability_defaults_and_clamps():
    assert EL._stability_from_delivery(None) == pytest.approx(0.5)
    assert EL._stability_from_delivery({"stability": 5.0}) == pytest.approx(1.0)  # clamp hi
    assert EL._stability_from_delivery({"expressiveness": 2.0}) == pytest.approx(0.0)  # clamp lo
    assert EL._stability_from_delivery({"stability": 0.3}) == pytest.approx(0.3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
