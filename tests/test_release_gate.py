"""Wave 1 / 1e -- delivery profiles (E.3) + release gate (E.5 / I-8).

Headless. The release gate scans heterogeneous items (dicts + AudioCacheRecord)
for the three-state commercial rule, reusing the audio-engine usability enum.
"""
from __future__ import annotations

import pathlib

import pytest

from nodes._otr_audio_engines import EngineUnusable, EngineUsabilityReason
from nodes._otr_delivery_profiles import DELIVERY_PROFILE_VERSION, DELIVERY_PROJECTION_VERSION, UnknownDeliveryProfile, available_delivery_profiles, get_delivery_profile
from nodes._otr_release_gate import ReleaseReport

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------
# Delivery profiles (E.3)
# ----------------------------------------------------------------------------


def test_only_neutral_ships():
    assert available_delivery_profiles() == ["neutral"]


def test_unknown_delivery_profile_raises():
    with pytest.raises(UnknownDeliveryProfile):
        get_delivery_profile("dramatic")


def test_delivery_versions_pinned():
    assert DELIVERY_PROFILE_VERSION == "1"
    assert DELIVERY_PROJECTION_VERSION == "1"


# ----------------------------------------------------------------------------
# Release gate (E.5 / I-8)
# ----------------------------------------------------------------------------


def test_source_is_ascii_no_em_dash():
    for name in ("_otr_release_gate.py", "_otr_delivery_profiles.py"):
        src = (REPO_ROOT / "nodes" / name).read_text(encoding="utf-8")
        assert "—" not in src, name
        src.encode("ascii")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
