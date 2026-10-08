"""Kokoro engine contracts (clean-break 1b).

Kokoro is a per_line registry engine (nodes/_otr_audio_engines/eng_kokoro.py)
whose hardcoded speed must equal the curated profile default.
"""
from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_kokoro_is_per_line_registry_engine():
    from nodes._otr_audio_engines import get_engine

    kokoro = get_engine("kokoro")
    assert getattr(kokoro, "interface", None) == "per_line"
    assert getattr(kokoro, "voice_ref_field", None) == "voice_ref_id"


def test_kokoro_speed_matches_profile_ssot():
    """The hardcoded eng_kokoro.speed must equal the announcer_kokoro_v1 profile
    default so the constant cannot silently drift from the curated SSOT (D5)."""
    from nodes._otr_audio_engines import get_engine
    from nodes._otr_engine_profiles import load_resolver

    kokoro = get_engine("kokoro")
    prof = load_resolver().profile_for("announcer_voice", "kokoro")
    assert prof is not None
    assert float(kokoro.speed) == float(prof.default_params["speed"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
