"""Sonilo cloud MUSIC adapter -- unit tests (cloud-audio S5, 2026-07-03).

Proves the properties the cloud-music lane needs:

* clip contract -- ``interface == "clip"``, ``roles == ("music",)``, and
  ``generate_clip(prompt, duration_s, seed)`` returns the Bark AUDIO dict.
* prompt/duration/seed are forwarded to the pinned ``cloud_sonilo_music`` row
  with provider-native INT ``duration`` (rounded).
* fail-loud / no-fallback -- a blank prompt OR a non-numeric duration RAISES; the
  adapter NEVER swaps to a local music engine.
* C7 -- the per-cue cost estimate is per-cue scale (duration * $0.15/60s).

invoke_partner_node + canonicalize_audio are the cloud/ffmpeg seams; both are
monkeypatched so the test runs OFFLINE with no credits and no ffmpeg.
"""
import numpy as np
import pytest
import soundfile as sf

from nodes._otr_audio_engines import registry as areg
from nodes._otr_audio_engines import eng_cloud_sonilo as SON


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
    sf.write(str(wav), np.stack([0.1 * np.sin(2 * 3.14159 * 200 * t)] * 2, axis=1), 44100)

    class _Canon:
        path = str(wav)

    def fake_canon(result, spec, session=None):
        capture["canon_spec"] = dict(spec)
        return _Canon()

    monkeypatch.setattr(inv, "invoke_partner_node", fake_invoke)
    monkeypatch.setattr(canon, "canonicalize_audio", fake_canon)


def test_registered_as_sonilo():
    eng = areg.get_engine("sonilo")
    assert eng.name == "sonilo"
    assert set(eng.roles) == {"music"}
    assert eng.interface == "clip"
    # C1/C2: cloud music is opt-in ONLY -- never a byte-identical default.
    assert eng.default_roles == ()
    assert eng.node_key == "cloud_sonilo_music"


def test_prompt_duration_seed_forwarded(monkeypatch, tmp_path):
    cap = {}
    # isolate RAW forwarding from the min-duration floor (floor is tested in
    # test_cloud_sonilo_floor.py); floor=0 => the cue duration passes through.
    monkeypatch.setenv("OTR_SONILO_MIN_DURATION_S", "0")
    _install_seams(monkeypatch, tmp_path, cap)
    eng = areg.get_engine("sonilo")
    out = eng.generate_clip("melancholy analog synth, 1962 radio drama", 11.6, seed=42)
    assert cap["node_key"] == "cloud_sonilo_music"
    assert cap["inputs"]["prompt"] == "melancholy analog synth, 1962 radio drama"
    # provider-native duration is a ROUNDED INT.
    assert cap["inputs"]["duration"] == 12
    assert isinstance(cap["inputs"]["duration"], int)
    assert cap["inputs"]["seed"] == 42
    # returns the Bark AUDIO contract from the canonical WAV.
    assert set(out) == {"waveform", "sample_rate"}
    assert out["waveform"].shape[0] == 1        # (1, C, T)
    assert out["sample_rate"] == 44100


def test_emitted_kwargs_match_partner_inputs(monkeypatch, tmp_path):
    """The kwargs actually emitted == the declared _partner_inputs contract."""
    cap = {}
    _install_seams(monkeypatch, tmp_path, cap)
    eng = areg.get_engine("sonilo")
    eng.generate_clip("a cue", 8, seed=0)
    assert set(cap["inputs"]) == set(eng._partner_inputs(None))


def test_cost_estimate_is_per_cue_scale():
    # C7: a 12s cue ~= $0.03, NOT an episode-total.
    assert SON.estimate_music_usd(12) == pytest.approx(12 * 0.15 / 60.0)
    assert SON.estimate_music_usd(0) == pytest.approx(1 * 0.15 / 60.0)  # min 1s


def test_estimated_usd_is_forwarded_to_invoke(monkeypatch, tmp_path):
    cap = {}
    monkeypatch.setenv("OTR_SONILO_MIN_DURATION_S", "0")   # isolate from the floor
    _install_seams(monkeypatch, tmp_path, cap)
    eng = areg.get_engine("sonilo")
    eng.generate_clip("interstitial sting", 6, seed=1)
    assert cap["estimated_usd"] == pytest.approx(SON.estimate_music_usd(6))


def test_blank_prompt_raises_no_fallback(monkeypatch, tmp_path):
    _install_seams(monkeypatch, tmp_path, {})
    eng = areg.get_engine("sonilo")
    with pytest.raises(ValueError, match="blank music prompt"):
        eng.generate_clip("   ", 10, seed=0)


def test_bad_duration_raises_no_fallback(monkeypatch, tmp_path):
    _install_seams(monkeypatch, tmp_path, {})
    eng = areg.get_engine("sonilo")
    with pytest.raises(ValueError, match="non-numeric duration"):
        eng.generate_clip("a real cue", "not-a-number", seed=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
