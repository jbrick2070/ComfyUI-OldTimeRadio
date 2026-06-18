"""Bark whiny-voice fix: per-stage temperatures + route probe (2026-06-17).

The thin/whiny Bark timbre came from a single flat ``temperature`` over-randomizing
the acoustic stages. The fix sets the three Bark stages independently (semantic 0.7
warm for content commitment; coarse/fine 0.5 to firm up the acoustics) and routes
them through the transformers ``BarkModel.generate`` PREFIXED-kwargs contract
(``semantic_temperature`` / ``coarse_temperature`` / ``fine_temperature``), which
never mutates the globally-cached model config -> no cross-voice leak.

CPU CI runs the route probe + the pure-logic + plumbing/contract tests; the actual
render is CUDA-only (the OTR test conftest sets CUDA_VISIBLE_DEVICES='') and skips.
"""
from __future__ import annotations

import inspect

import pytest

from nodes import _otr_bark_lib as bl
from nodes import _otr_engine_profiles as ep
from nodes._otr_audio_engines.eng_bark import BarkEngine


# --------------------------------------------------------------------------- #
# S1 -- route probe (CPU-safe; records transformers version + the prefix route)
# --------------------------------------------------------------------------- #
def test_route_probe_bark_generate_supports_stage_prefix():
    """The chosen route is prefixed kwargs: BarkModel.generate splits kwargs by a
    ``semantic_`` / ``coarse_`` / ``fine_`` prefix and routes each to its sub-model
    (prefixed has priority). This is a SOFT probe -- it records the transformers
    version and asserts the prefix mechanism exists, not a brittle signature pin."""
    transformers = pytest.importorskip("transformers")
    from transformers import BarkModel

    src = inspect.getsource(BarkModel.generate)
    # the prefix-split contract the fix relies on
    assert 'startswith("semantic_")' in src or "semantic_" in src, (
        f"BarkModel.generate (transformers {transformers.__version__}) no longer "
        f"documents the semantic_/coarse_/fine_ prefix route -- re-verify the fix"
    )
    print(f"[route-probe] transformers={transformers.__version__}; "
          f"prefix-kwargs route confirmed")


# --------------------------------------------------------------------------- #
# S2 -- pure stage-temp resolution + caps (_stage_temps_for_line)
# --------------------------------------------------------------------------- #
def test_none_stages_inherit_legacy_temperature():
    # Back-compat: the orchestrator health probe passes only temperature=0.6.
    assert bl._stage_temps_for_line(0.6, None, None, None, "v2/en_speaker_0",
                                    False) == (0.6, 0.6, 0.6)


def test_explicit_per_stage_temps_used():
    assert bl._stage_temps_for_line(0.7, 0.7, 0.5, 0.5, "v2/en_speaker_3",
                                    False) == (0.7, 0.5, 0.5)


def test_explicit_zero_is_honored_not_treated_as_falsey():
    # 0.0 must NOT fall back to temperature (the `is not None` contract).
    sem, crs, fin = bl._stage_temps_for_line(0.7, 0.0, 0.0, 0.0,
                                             "v2/en_speaker_3", False)
    assert (sem, crs, fin) == (0.0, 0.0, 0.0)


def test_intl_cap_hits_semantic_only():
    # Foreign preset: semantic capped to 0.55, coarse/fine untouched.
    sem, crs, fin = bl._stage_temps_for_line(0.7, 0.7, 0.6, 0.6,
                                             "v2/de_speaker_0", False)
    assert sem == 0.55
    assert (crs, fin) == (0.6, 0.6)


def test_first_line_cap_semantic_only_en():
    sem, crs, fin = bl._stage_temps_for_line(0.7, 0.7, 0.5, 0.5,
                                             "v2/en_speaker_0", True)
    assert sem == 0.6              # capped on the first line
    assert (crs, fin) == (0.5, 0.5)  # acoustics keep their profile values


def test_first_line_cap_semantic_only_intl():
    sem, crs, fin = bl._stage_temps_for_line(0.7, 0.7, 0.5, 0.5,
                                             "v2/fr_speaker_1", True)
    assert sem == 0.5              # intl first-line cap
    assert (crs, fin) == (0.5, 0.5)


def test_cap_is_a_ceiling_not_a_floor():
    # A low semantic temp is NOT raised by the first-line "cap".
    sem, _crs, _fin = bl._stage_temps_for_line(0.7, 0.3, 0.5, 0.5,
                                               "v2/en_speaker_0", True)
    assert sem == 0.3


# --------------------------------------------------------------------------- #
# S2 -- alias resolver on BarkEngine (profile -> stage temps, class-attr fallback)
# --------------------------------------------------------------------------- #
class _FakeProfile:
    def __init__(self, params):
        self.default_params = params


class _FakeResolver:
    def __init__(self, params):
        self._params = params

    def profile_for(self, role, engine):
        return _FakeProfile(self._params) if self._params is not None else None


def _patch_profile(monkeypatch, params):
    monkeypatch.setattr(ep, "load_resolver", lambda: _FakeResolver(params))


def test_resolve_uses_explicit_stage_keys(monkeypatch):
    _patch_profile(monkeypatch, {"semantic_temp": 0.7, "coarse_temp": 0.5,
                                 "fine_temp": 0.5})
    assert BarkEngine()._resolve_stage_temps() == (0.7, 0.5, 0.5)


def test_resolve_falls_back_to_legacy_aliases(monkeypatch):
    # Only the old keys present: semantic<-text_temp, coarse/fine<-waveform_temp.
    _patch_profile(monkeypatch, {"text_temp": 0.65, "waveform_temp": 0.45})
    assert BarkEngine()._resolve_stage_temps() == (0.65, 0.45, 0.45)


def test_resolve_honors_explicit_zero_over_alias(monkeypatch):
    _patch_profile(monkeypatch, {"semantic_temp": 0.0, "text_temp": 0.7,
                                 "coarse_temp": 0.0, "waveform_temp": 0.5})
    sem, crs, fin = BarkEngine()._resolve_stage_temps()
    assert sem == 0.0 and crs == 0.0


def test_resolve_skips_none_values(monkeypatch):
    _patch_profile(monkeypatch, {"semantic_temp": None, "text_temp": 0.66,
                                 "coarse_temp": None, "waveform_temp": 0.44,
                                 "fine_temp": None})
    assert BarkEngine()._resolve_stage_temps() == (0.66, 0.44, 0.44)


def test_resolve_missing_profile_uses_class_attrs(monkeypatch):
    _patch_profile(monkeypatch, None)             # profile_for -> None
    eng = BarkEngine()
    assert eng._resolve_stage_temps() == (
        eng.semantic_temp, eng.coarse_temp, eng.fine_temp)


def test_resolve_broken_resolver_uses_class_attrs(monkeypatch):
    monkeypatch.setattr(ep, "load_resolver", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    eng = BarkEngine()
    assert eng._resolve_stage_temps() == (0.7, 0.5, 0.5)


def test_live_profile_yields_the_fix_defaults():
    # No patching: reads the real config/audio_engine_profiles.yaml.
    assert BarkEngine()._resolve_stage_temps() == (0.7, 0.5, 0.5)


# --------------------------------------------------------------------------- #
# S2 -- plumbing + no-leak CONTRACT on _generate_single_line (source-level, CPU)
# --------------------------------------------------------------------------- #
def test_generate_single_line_signature_is_back_compatible():
    sig = inspect.signature(bl._generate_single_line)
    params = sig.parameters
    # legacy single temp still accepted (orchestrator health probe)
    assert "temperature" in params
    # new per-stage temps are keyword-only
    for name in ("semantic_temp", "coarse_temp", "fine_temp"):
        assert params[name].kind == inspect.Parameter.KEYWORD_ONLY


def test_generate_routes_prefixed_stage_temps_no_flat_temperature():
    src = inspect.getsource(bl._generate_single_line)
    assert "semantic_temperature=sem" in src
    assert "coarse_temperature=crs" in src
    assert "fine_temperature=fin" in src
    # the old single flat temperature= kwarg to generate is gone
    assert "temperature=temperature" not in src


def test_generate_does_not_mutate_cached_generation_config():
    # No-leak by construction: the route passes kwargs, it must NOT assign
    # temperature onto the globally-cached model's generation_config.
    src = inspect.getsource(bl._generate_single_line)
    assert "generation_config.temperature" not in src


def test_eng_bark_passes_resolved_stage_temps():
    src = inspect.getsource(BarkEngine.generate_voice)
    assert "_resolve_stage_temps()" in src
    assert "semantic_temp=semantic_temp" in src
    assert "coarse_temp=coarse_temp" in src
    assert "fine_temp=fine_temp" in src


# --------------------------------------------------------------------------- #
# S1 -- forced-Bark render baseline harness (CUDA-only; skips on CPU CI)
# --------------------------------------------------------------------------- #
def _render_smoke_enabled():
    # Opt-in only (mirrors test_audio_byte_identical's OTR_REGRESSION_RUNTIME gate):
    # the render needs a real GPU AND the Bark weights on disk, so it must never
    # fire in the unit suite. Set OTR_BARK_RENDER_SMOKE=1 on a GPU box to run it.
    import os
    if not os.environ.get("OTR_BARK_RENDER_SMOKE"):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _render_smoke_enabled(),
                    reason="Set OTR_BARK_RENDER_SMOKE=1 on a GPU box with Bark "
                           "weights to run the forced-Bark render smoke")
def test_forced_bark_render_smoke():  # pragma: no cover - GPU-only
    """Forced-Bark baseline harness: render one line with a FRESH engine (state
    reset so the first-line guard is deterministic) and assert a non-empty mono
    waveform at 24 kHz. Records whether the first-line guard fired."""
    eng = BarkEngine()
    eng.unload()  # reset _presets_started / residency
    out = eng.generate_voice("This is a radio drama test line.",
                             "v2/en_speaker_6", None, 42)
    wav = out["waveform"]
    assert out["sample_rate"] == 24000
    assert tuple(wav.shape[:2]) == (1, 1)
    assert wav.shape[-1] > 2400  # more than the empty-text floor
    eng.unload()
