"""Declarative profile metadata: runtime and license_state.
Headless; the only IO is loading the shipped YAML.

These cover the metadata surface only; the resolver behaviour stays pinned by
test_engine_profiles.py. The live byte-identical dispatch is NOT exercised here.
"""
from __future__ import annotations

import pytest

from nodes import _otr_engine_profiles as EP
from nodes._otr_audio_engines import EngineUnusable, EngineUsabilityReason


def _resolver():
    r = EP.load_resolver()
    assert r is not None
    return r


def test_every_profile_has_sprint1_metadata():
    r = _resolver()
    for pid in r.profile_ids():
        p = r.get(pid)
        assert p.runtime in EP._VALID_RUNTIMES
        assert p.license_state in EP._VALID_LICENSE_STATES


def test_effective_license_state_blank_derivation():
    p_clean = EP.EngineProfile(
        profile_id="x", role="music", engine="stable_audio_3",
        commercial_clean=True,
    )
    assert EP.effective_license_state(p_clean) == "clean"
    p_gated = EP.EngineProfile(
        profile_id="y", role="music", engine="musicgen",
        commercial_clean=False,
    )
    assert EP.effective_license_state(p_gated) == "gated"


def test_license_state_mirrors_commercial_clean_for_all_rows():
    r = _resolver()
    for pid in r.profile_ids():
        p = r.get(pid)
        assert EP.effective_license_state(p) == p.license_state
        if p.license_state == "clean":
            assert p.commercial_clean is True
        elif p.license_state == "gated":
            assert p.commercial_clean is False


def test_bad_runtime_rejected():
    with pytest.raises(Exception):
        EP.EngineProfile(
            profile_id="x", role="music", engine="musicgen",
            commercial_clean=False, runtime="cloud",
        )


def test_bad_license_state_rejected():
    with pytest.raises(Exception):
        EP.EngineProfile(
            profile_id="x", role="music", engine="musicgen",
            commercial_clean=False, license_state="maybe",
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
