"""Bark artifact B2 -- thread the existing per-line seed (determinism).

generate_voice already RECEIVES a `seed` but dropped it (Bark.generate ran
unseeded -> non-deterministic). B2 threads it into _generate_single_line and
seeds the global torch RNG before model.generate (Bark binds no external
Generator: eng_bark.supports_external_generator=False).

Unit suite = signature + source contract (CPU). The behavioral "same seed ->
identical clip / different seed -> different clip" render is GPU+weights only,
gated on OTR_BARK_RENDER_SMOKE (mirrors test_bark_voice_stage_temps).

UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nodes._otr_bark_lib as bl  # noqa: E402
from nodes._otr_audio_engines.eng_bark import BarkEngine  # noqa: E402


# ---------------------------------------------------------------------------
# Plumbing contract (CPU)
# ---------------------------------------------------------------------------


class TestSeedPlumbing:
    def test_seed_is_keyword_only_default_none(self):
        p = inspect.signature(bl._generate_single_line).parameters["seed"]
        assert p.kind == inspect.Parameter.KEYWORD_ONLY
        assert p.default is None

    def test_seed_applied_before_generate(self):
        src = inspect.getsource(bl._generate_single_line)
        assert "if seed is not None:" in src
        assert "torch.manual_seed(int(seed))" in src
        assert "torch.cuda.manual_seed_all(int(seed))" in src
        # seeding must precede the generate call (before the chunk loop)
        assert src.index("torch.manual_seed(int(seed))") < src.index("model.generate(")

    def test_eng_bark_threads_seed(self):
        src = inspect.getsource(BarkEngine.generate_voice)
        assert "seed=seed" in src

    def test_engine_declares_no_external_generator(self):
        # The reason we seed the global RNG rather than pass a Generator.
        assert BarkEngine.supports_external_generator is False


# ---------------------------------------------------------------------------
# GPU determinism smoke (opt-in: OTR_BARK_RENDER_SMOKE=1 on a GPU box)
# ---------------------------------------------------------------------------


def _render_smoke_enabled():
    if not os.environ.get("OTR_BARK_RENDER_SMOKE"):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _render_smoke_enabled(),
                    reason="Set OTR_BARK_RENDER_SMOKE=1 on a GPU box with Bark "
                           "weights to run the seed-determinism render smoke")
def test_seed_determinism_render_smoke():  # pragma: no cover - GPU-only
    """Same seed -> identical clip; different seed -> different clip."""
    line = "This is a radio drama determinism test line."
    preset = "v2/en_speaker_6"

    def _render(seed):
        eng = BarkEngine()
        eng.unload()  # reset _presets_started so is_first is deterministic
        out = eng.generate_voice(line, preset, None, seed)
        eng.unload()
        return out["waveform"].cpu().numpy().reshape(-1)

    a = _render(1234)
    b = _render(1234)
    c = _render(9999)

    assert a.shape == b.shape
    assert np.array_equal(a, b), "same seed must produce a byte-identical clip"
    # different seed should diverge (shape may differ; if equal length, content differs)
    assert a.shape != c.shape or not np.array_equal(a, c), \
        "different seed should produce a different clip"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
