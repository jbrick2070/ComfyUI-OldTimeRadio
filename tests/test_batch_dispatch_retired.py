"""Audio engine contracts left after the clean-break 1c (batch dispatch retired).

After 1a/1b/1c every audio engine is self-contained (per_line / clip). The
musicgen engine is a clip engine, and its hardcoded guidance scale must equal
the curated profile default.
"""
from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_musicgen_is_clip_engine():
    from nodes._otr_audio_engines import get_engine

    mg = get_engine("musicgen")
    assert mg.interface == "clip"


def test_musicgen_guidance_matches_profile_ssot():
    """eng_musicgen.guidance_scale must equal the music_musicgen_v1 profile
    default so the hardcoded constant cannot drift from the curated SSOT (D5)."""
    from nodes._otr_audio_engines import get_engine
    from nodes._otr_engine_profiles import load_resolver

    mg = get_engine("musicgen")
    prof = load_resolver().profile_for("music", "musicgen")
    assert prof is not None
    assert float(mg.guidance_scale) == float(prof.default_params["guidance_scale"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
