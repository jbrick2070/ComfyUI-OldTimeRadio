"""Wave 1 / 1e -- delivery profiles (E.3) + release gate (E.5 / I-8).

Headless. The release gate scans heterogeneous items (dicts + AudioCacheRecord)
for the three-state commercial rule, reusing the audio-engine usability enum.
"""
from __future__ import annotations

import pathlib

import pytest

from nodes._otr_audio_engines import EngineUnusable, EngineUsabilityReason
from nodes._otr_delivery_profiles import (
    DELIVERY_PROFILE_VERSION,
    DELIVERY_PROJECTION_VERSION,
    UnknownDeliveryProfile,
    apply_delivery,
    available_delivery_profiles,
    get_delivery_profile,
)
from nodes._otr_release_gate import (
    ReleaseReport,
    assert_release_clean,
    mangle_release_filename,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------
# Delivery profiles (E.3)
# ----------------------------------------------------------------------------
def test_neutral_is_identity():
    profile = get_delivery_profile("neutral")
    text, params = apply_delivery(profile, "Hello, world.", {"temperature": 0.6})
    assert text == "Hello, world."          # no text projection
    assert params == {"temperature": 0.6}   # engine defaults unchanged


def test_neutral_overlay_returns_a_copy():
    profile = get_delivery_profile("neutral")
    src = {"a": 1}
    _text, params = apply_delivery(profile, "x", src)
    assert params == src and params is not src  # copy, not the same object


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
def test_all_clean_passes_silently():
    items = [
        {"role": "char_voice", "commercial_clean": True},
        {"voice_ref_id": "cc_male_warm", "commercial_clean": True},
    ]
    report = assert_release_clean(items)
    assert isinstance(report, ReleaseReport)
    assert report.clean is True
    assert report.gated == 0 and report.scanned == 2
    assert report.warnings == ()


def test_gated_warns_but_does_not_block():
    items = [
        {"engine_name": "indextts2", "commercial_clean": False},
        {"engine_name": "kokoro", "commercial_clean": True},
    ]
    report = assert_release_clean(items)
    assert report.clean is False
    assert report.gated == 1
    assert len(report.warnings) == 1
    assert "known-gated" in report.warnings[0]


def test_missing_commercial_fails_closed():
    with pytest.raises(EngineUnusable) as exc:
        assert_release_clean([{"role": "music"}])  # no commercial_clean
    assert exc.value.reason is EngineUsabilityReason.MALFORMED_CONFIG


def test_null_commercial_fails_closed():
    with pytest.raises(EngineUnusable) as exc:
        assert_release_clean([{"role": "music", "commercial_clean": None}])
    assert exc.value.reason is EngineUsabilityReason.MALFORMED_CONFIG


def test_non_boolean_commercial_fails_closed():
    with pytest.raises(EngineUnusable) as exc:
        assert_release_clean([{"commercial_clean": "true"}])  # string, not bool
    assert exc.value.reason is EngineUsabilityReason.MALFORMED_CONFIG


def test_strict_commercial_blocks_gated():
    with pytest.raises(EngineUnusable) as exc:
        assert_release_clean(
            [{"engine_name": "musicgen", "commercial_clean": False}],
            strict_commercial=True,
        )
    assert exc.value.reason is EngineUsabilityReason.NONCOMMERCIAL_BLOCKED


def test_require_allowed_for_release_blocks_unreleasable():
    rec = {"cache_key": "k1", "commercial_clean": True, "allowed_for_release": False}
    with pytest.raises(EngineUnusable) as exc:
        assert_release_clean([rec], require_allowed_for_release=True)
    assert exc.value.reason is EngineUsabilityReason.MALFORMED_CONFIG
    # ...and passes when the record is releasable.
    rec["allowed_for_release"] = True
    report = assert_release_clean([rec], require_allowed_for_release=True)
    assert report.clean is True


def test_scans_audio_cache_record_objects():
    from nodes._otr_audio_cache import AudioCacheRecord

    ok = AudioCacheRecord(cache_key="k", commercial_clean=True, allowed_for_release=True)
    report = assert_release_clean([ok])
    assert report.scanned == 1 and report.clean is True

    bad = AudioCacheRecord(cache_key="k2")  # commercial_clean defaults to None
    with pytest.raises(EngineUnusable) as exc:
        assert_release_clean([bad])
    assert exc.value.reason is EngineUsabilityReason.MALFORMED_CONFIG


def test_filename_mangle_clean_unchanged():
    assert mangle_release_filename("opening_theme.wav", True) == "opening_theme.wav"


def test_filename_mangle_gated_hashes_stem():
    name = "facebook_musicgen_medium_opening.wav"
    out = mangle_release_filename(name, False)
    assert out.startswith("gated_") and out.endswith(".wav")
    assert "musicgen" not in out  # gated identifier does not leak
    assert mangle_release_filename(name, False) == out  # deterministic


def test_filename_mangle_unknown_commercial_is_mangled():
    # None / non-True is treated as not-clean (fail-closed at the filename layer).
    assert mangle_release_filename("x.wav", None).startswith("gated_")


def test_source_is_ascii_no_em_dash():
    for name in ("_otr_release_gate.py", "_otr_delivery_profiles.py"):
        src = (REPO_ROOT / "nodes" / name).read_text(encoding="utf-8")
        assert "—" not in src, name
        src.encode("ascii")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
