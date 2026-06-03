"""G1 -- the opt-in audio forwards run inside ``deterministic_inference``.

Headless (CUDA masked by conftest). The legacy byte-identical BATCH path is
never wrapped (it delegates verbatim to the legacy node); only the per-line
voice path and the per-cue theme path -- both opt-in, default-off -- scope
strict-determinism + RNG seed/restore around the single engine forward.

These tests inject a spy adapter + a stub resolver so no engine library is
needed: the spy records whether ``torch.are_deterministic_algorithms_enabled()``
was True at the moment its forward ran, and the test asserts the flags are
restored afterwards. A source guard pins the wrap at both dispatch sites.
"""
from __future__ import annotations

import json
import pathlib

import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class _SpyVoiceAdapter:
    name = "spy_voice"
    interface = "per_line"
    sample_rate = 24000

    def __init__(self):
        self.calls = []
        self.det_enabled_during_call = None

    def prepare_text(self, text, delivery_vector=None):
        return text

    def generate_voice(self, text, ref_clip_path, delivery_vector, seed):
        self.det_enabled_during_call = torch.are_deterministic_algorithms_enabled()
        self.calls.append((text, ref_clip_path, seed))
        return {"waveform": torch.zeros(1, 1, 8, dtype=torch.float32),
                "sample_rate": self.sample_rate}

    def unload(self):
        pass


class _SpyClipAdapter:
    name = "spy_music"
    interface = "clip"
    sample_rate = 44100

    def __init__(self):
        self.calls = []
        self.det_enabled_during_call = None

    def generate_clip(self, prompt, duration_s, seed):
        self.det_enabled_during_call = torch.are_deterministic_algorithms_enabled()
        self.calls.append((prompt, duration_s, seed))
        return {"waveform": torch.zeros(1, 2, 8, dtype=torch.float32),
                "sample_rate": self.sample_rate}

    def unload(self):
        pass


class _FakeProfile:
    def __init__(self, sample_rate):
        self.profile_id = "spy_profile_v1"
        self.engine_impl_version = "test"
        self.sample_rate = sample_rate
        self.default_params = {}
        self.commercial_clean = True


class _FakeResolver:
    def __init__(self, sample_rate):
        self._sr = sample_rate

    def resolve_casting_plan(self, **kwargs):
        return _FakeProfile(self._sr)


def _stub_dispatch(monkeypatch, adapter, sample_rate):
    import nodes._otr_audio_engines as AE
    import nodes._otr_engine_profiles as EP

    monkeypatch.setattr(AE, "assert_usable", lambda name, role: name)
    monkeypatch.setattr(AE, "get_engine", lambda name: adapter)
    monkeypatch.setattr(EP, "require_resolver", lambda: _FakeResolver(sample_rate))
    monkeypatch.setattr(EP, "assert_token_for_profile", lambda p: None)
    monkeypatch.setattr(EP, "assert_model_available", lambda p: None)


# --------------------------------------------------------------------------- #
# Mechanism
# --------------------------------------------------------------------------- #
def test_deterministic_inference_enables_and_restores():
    from nodes._otr_determinism import deterministic_inference

    before = torch.are_deterministic_algorithms_enabled()
    with deterministic_inference(123, warn_only=True):
        assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.are_deterministic_algorithms_enabled() == before


# --------------------------------------------------------------------------- #
# Behavioral: the forward runs INSIDE the CM, flags restored after
# --------------------------------------------------------------------------- #
def test_per_line_voice_forward_runs_inside_deterministic_inference(monkeypatch):
    from nodes._otr_resolved_request import assert_audio_batch_contract
    from nodes.batch_character_voices import BatchCharacterVoices

    spy = _SpyVoiceAdapter()
    _stub_dispatch(monkeypatch, spy, sample_rate=24000)

    before = torch.are_deterministic_algorithms_enabled()
    ledger = json.dumps({
        "lines": [{"line_id": "l1", "char_id": "c1",
                   "speaker_role": "character", "text": "Hello there."}],
        "cast": [{"char_id": "c1", "voice_ref_id": "vr1", "voice_preset": "vp1"}],
        "meta": {"episode_seed": 7, "cast_lock_revision": 0},
    })
    out = BatchCharacterVoices().generate(script_json=ledger, engine="spy_voice")

    assert spy.calls, "the per-line forward must have been called"
    assert spy.det_enabled_during_call is True, (
        "generate_voice must run inside deterministic_inference"
    )
    assert torch.are_deterministic_algorithms_enabled() == before, (
        "determinism flags must be restored after the forward"
    )
    assert_audio_batch_contract(out[0], where="test.voice")


def test_theme_clip_forward_runs_inside_deterministic_inference(monkeypatch):
    from nodes._otr_resolved_request import assert_audio_batch_contract
    from nodes.stable_audio_theme import StableAudioTheme

    spy = _SpyClipAdapter()
    _stub_dispatch(monkeypatch, spy, sample_rate=44100)

    before = torch.are_deterministic_algorithms_enabled()
    ledger = json.dumps({"meta": {"episode_seed": 7}})
    out = StableAudioTheme().generate(script_json=ledger, engine="spy_music")

    assert len(spy.calls) == 3, "all three theme cues must render"
    assert spy.det_enabled_during_call is True, (
        "generate_clip must run inside deterministic_inference"
    )
    assert torch.are_deterministic_algorithms_enabled() == before
    for i in range(3):
        assert_audio_batch_contract(out[i], where=f"test.cue{i}")


# --------------------------------------------------------------------------- #
# Source guard: the wrap is pinned at BOTH dispatch sites
# --------------------------------------------------------------------------- #
def test_voice_dispatch_wraps_forward_in_cm():
    src = (REPO_ROOT / "nodes" / "_otr_voice_node_common.py").read_text(encoding="utf-8")
    assert "from ._otr_determinism import deterministic_inference" in src
    i_with = src.find("with deterministic_inference(engine_seed")
    i_call = src.find("adapter.generate_voice(", i_with)
    assert 0 <= i_with < i_call, "generate_voice must sit inside the CM"


def test_theme_dispatch_wraps_forward_in_cm():
    src = (REPO_ROOT / "nodes" / "stable_audio_theme.py").read_text(encoding="utf-8")
    assert "from ._otr_determinism import deterministic_inference" in src
    i_with = src.find("with deterministic_inference(engine_seed")
    i_call = src.find("adapter.generate_clip(", i_with)
    assert 0 <= i_with < i_call, "generate_clip must sit inside the CM"


# test_legacy_batch_path_is_not_wrapped deleted in the audio clean-break (1c):
# the batch-delegation path (_delegate_batch) was retired entirely -- every audio
# engine is now self-contained per_line / clip, each wrapped in the determinism CM.


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
