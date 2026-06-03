"""tests/test_bark_cast_contract.py

Gate 3 (voice-path cleanbreak) -- re-homed into the per_line bark engine.

The retired OTR_BatchBarkGenerator hard-raised on an empty / non-v2/*
``cast.voice_preset`` as the last renderability net after Gate 1 (writer) and
Gate 2 (FreezeCascade Phase 0/10). In the audio clean-break (1a) bark became a
per_line registry engine (nodes/_otr_audio_engines/eng_bark.py); the same net
now lives in ``BarkEngine.generate_voice``, which fails closed with a named
``EngineUnusable(MALFORMED_CONFIG)`` BEFORE any model load -- so the contract is
exercised headless (CUDA masked by tests/conftest.py).
"""
from __future__ import annotations

import pytest

from nodes._otr_audio_engines.eng_bark import BarkEngine
from nodes._otr_audio_engines.registry import (
    EngineUnusable,
    EngineUsabilityReason,
)


def _bark():
    return BarkEngine()


def test_bark_raises_on_empty_voice_preset():
    """Empty preset is a writer cast-lock contract violation."""
    with pytest.raises(EngineUnusable) as exc:
        _bark().generate_voice("Hello.", "", None, 7)
    assert exc.value.reason is EngineUsabilityReason.MALFORMED_CONFIG


def test_bark_raises_on_none_voice_preset():
    with pytest.raises(EngineUnusable) as exc:
        _bark().generate_voice("Hello.", None, None, 7)
    assert exc.value.reason is EngineUsabilityReason.MALFORMED_CONFIG


def test_bark_raises_on_non_v2_voice_preset():
    """A Kokoro id on a bark line is a contract violation."""
    with pytest.raises(EngineUnusable) as exc:
        _bark().generate_voice("Hello.", "bm_george", None, 7)
    assert exc.value.reason is EngineUsabilityReason.MALFORMED_CONFIG


def test_bark_preset_error_names_the_offender():
    """The fail-closed message names the v2/* requirement so the operator can
    trace the bad preset without reading source."""
    with pytest.raises(EngineUnusable, match=r"v2/"):
        _bark().generate_voice("Hello.", "bm_george", None, 7)


def test_bark_engine_declares_per_line_preset_routing():
    """The dispatch routes cast.voice_preset into the ref slot via these attrs
    (appendix A). Pins the contract the routing depends on."""
    eng = _bark()
    assert eng.interface == "per_line"
    assert eng.voice_ref_field == "voice_preset"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
