"""tests/test_voice_two_lane.py -- VC chunk 2 (two-lane identity).

Covers:
  - bark_preset_gender: reads cast_pools.VOICE_PROFILES (known male/female
    presets, '' for unknown).
  - same_gender_voice_ref_for_preset: deterministic lowest-id same-gender
    pick, engine-scoped, reject-skipping, fail-soft on no match.
  - CastingResponse.voice_preset now accepts a verbose (>80 char) id.
  - CastLock STEP-3 fail-soft two-lane: a repaired character row (missing
    voice_preset) gets a bark fallback AND, when a cloner engine resolves,
    a same-gender voice_ref_id; rows that already carry a preset are
    untouched (byte-safe).

Hermetic: no GPU, no network. The helper tests use a synthetic bank.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODES_DIR = _REPO_ROOT / "nodes"
for p in (_REPO_ROOT, _NODES_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Import via the `nodes` package (matches conftest) so the relative imports
# inside cast_lock (`from ._otr_voice_bank import ...`) resolve and the bank
# path is genuinely exercised, not silently swallowed by a fail-soft except.
from nodes import _otr_voice_bank as VB  # noqa: E402
from nodes._otr_casting import CastingResponse  # noqa: E402


def _entry(vrid, engine="indextts2", gender="male", tier="", roles=("char_voice",)):
    return VB.VoiceBankEntry(
        voice_ref_id=vrid, engine=engine, gender=gender,
        timbre=("warm",), roles=tuple(roles), age_band="adult",
        ref_path=f"models/TTS/refs/{vrid}.wav", ref_sha256="deadbeef",
        commercial_clean=True, quality_tier=tier, style_tags=(),
    )


# --------------------------------------------------------------------------- #
# bark_preset_gender
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("preset,expected", [
    ("v2/en_speaker_0", "male"),
    ("v2/en_speaker_8", "male"),
    ("v2/en_speaker_2", "female"),
    ("v2/en_speaker_7", "female"),  # reclassified female in cast_pools
])
def test_bark_preset_gender_known(preset, expected):
    assert VB.bark_preset_gender(preset) == expected


def test_bark_preset_gender_unknown_is_empty():
    assert VB.bark_preset_gender("v2/en_speaker_999") == ""
    assert VB.bark_preset_gender("") == ""
    assert VB.bark_preset_gender(None) == ""


# --------------------------------------------------------------------------- #
# same_gender_voice_ref_for_preset
# --------------------------------------------------------------------------- #
def test_same_gender_pick_is_lowest_id_and_deterministic():
    bank = (
        _entry("vz_charlie", gender="male"),
        _entry("vz_alpha", gender="male"),
        _entry("vz_bravo", gender="male"),
        _entry("vz_fem", gender="female"),
    )
    # v2/en_speaker_0 is male -> lowest male id == 'vz_alpha'.
    got = VB.same_gender_voice_ref_for_preset("v2/en_speaker_0", "indextts2", bank=bank)
    assert got == "vz_alpha"
    # Deterministic across calls.
    assert got == VB.same_gender_voice_ref_for_preset(
        "v2/en_speaker_0", "indextts2", bank=bank)


def test_same_gender_respects_engine_and_gender():
    bank = (
        _entry("vz_m1", engine="indextts2", gender="male"),
        _entry("cb_f1", engine="chatterbox", gender="female"),
    )
    # female preset, indextts2 -> no female indextts2 ref -> ''.
    assert VB.same_gender_voice_ref_for_preset(
        "v2/en_speaker_2", "indextts2", bank=bank) == ""
    # gender_hint overrides preset-derived gender.
    assert VB.same_gender_voice_ref_for_preset(
        "v2/en_speaker_0", "chatterbox", bank=bank, gender_hint="female") == "cb_f1"


def test_same_gender_skips_reject_tier():
    bank = (
        _entry("vz_aaa", gender="male", tier="reject"),
        _entry("vz_bbb", gender="male", tier="a"),
    )
    assert VB.same_gender_voice_ref_for_preset(
        "v2/en_speaker_0", "indextts2", bank=bank) == "vz_bbb"


def test_same_gender_fail_soft_on_empty():
    assert VB.same_gender_voice_ref_for_preset("", "indextts2", bank=()) == ""
    assert VB.same_gender_voice_ref_for_preset("v2/en_speaker_0", "", bank=()) == ""


# --------------------------------------------------------------------------- #
# CastingResponse verbose id (max_length 80 -> 255)
# --------------------------------------------------------------------------- #
def test_casting_response_accepts_verbose_voice_preset():
    long_id = "indextts2/" + ("a" * 200)  # > 80 chars
    resp = CastingResponse(
        character_description="A weary engineer who fixes the relay.",
        gender="male", voice_preset=long_id,
    )
    assert resp.voice_preset == long_id
    assert len(long_id) > 80


# --------------------------------------------------------------------------- #
# CastLock STEP-3 two-lane refine
# --------------------------------------------------------------------------- #
def test_castlock_missing_preset_row_fails_loud():
    # NO-FALLBACK (2026-07-03): a character row missing voice_preset is a casting
    # defect -> CastLock RAISES VoiceCastingError. The old two-lane repair (stamp a
    # v2/en_speaker_* fallback identity + a same-gender clone id) is RETIRED.
    import pytest

    from nodes.cast_lock import CastLock
    from nodes._otr_voice_bank import VoiceCastingError

    cast = [
        {"char_id": "c01", "name": "ANNOUNCER", "voice_preset": "bf_emma"},
        {"char_id": "c02", "name": "ALICE", "gender": "female"},  # no preset
    ]
    with pytest.raises(VoiceCastingError, match="no voice_preset"):
        CastLock._resolve_character_voices_fail_soft(
            cast, lines=[], voice_bank="default")


def test_castlock_two_lane_leaves_voiced_rows_untouched():
    from nodes.cast_lock import CastLock

    cast = [
        {"char_id": "c01", "name": "ANNOUNCER", "voice_preset": "bf_emma"},
        {"char_id": "c02", "name": "BOB", "gender": "male",
         "voice_preset": "v2/en_speaker_5"},  # already voiced
    ]
    CastLock._resolve_character_voices_fail_soft(
        cast, lines=[], voice_bank="default")
    bob = cast[1]
    # Untouched: preset preserved, no clone id force-stamped on a normal row.
    assert bob["voice_preset"] == "v2/en_speaker_5"
    assert "voice_ref_id" not in bob
