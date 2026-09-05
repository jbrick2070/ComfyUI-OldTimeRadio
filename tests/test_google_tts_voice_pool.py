"""The google_tts castable pool must stay wide enough to cast a real cast.

This exists because the pool silently sat at FOUR castable char voices while
the engine accepted thirty, and nothing caught it: ``google_tts`` was missing
from ``APPROVED_VOICE_ENGINES``, so the coverage floor -- the gate whose whole
job is to stop an engine going thin -- never looked at the cloud lane. A
nine-speaker Shakespeare roster failed casting on three characters, and
CastLock RE-RAISES for google_tts with no fallback, so that is a dead render
rather than a duplicated voice.

The gender labels here are measured, not asserted: every row carries the
median F0 it was classified from.
"""
from __future__ import annotations

import pytest

from nodes._otr_voice_bank import APPROVED_VOICE_ENGINES, VoiceCastingError, assign_voice_for_slot, compute_bank_coverage, load_voice_bank

#: One speaking part per Macbeth II.ii-era roster. Nine is not a stress test;
#: it is a normal Shakespeare cast.
ROSTER = [
    ("macbeth", "male"), ("lady_macbeth", "female"), ("duncan", "male"),
    ("banquo", "male"), ("macduff", "male"), ("porter", "male"),
    ("witch_one", "female"), ("witch_two", "female"), ("witch_three", "female"),
]


def _google_rows():
    bank, _ = load_voice_bank()
    return [e for e in bank if e.engine == "google_tts"]


def test_google_tts_is_inside_the_coverage_floor():
    """It was absent, which is precisely why the pool went thin unnoticed."""
    assert "google_tts" in APPROVED_VOICE_ENGINES
    cov = compute_bank_coverage(cast_size=5)
    assert "google_tts" in cov
    assert cov["google_tts"]["meets_floor"], cov["google_tts"]


def test_every_supported_engine_voice_is_castable():
    """The bank must map every voice the engine will actually accept -- a
    supported voice with no bank row can never be cast."""
    from nodes._otr_audio_engines.eng_google_tts import _SUPPORTED_VOICES

    mapped = {e.provider_voice_id for e in _google_rows()}
    missing = sorted(set(_SUPPORTED_VOICES) - mapped)
    assert not missing, "engine accepts these but the bank cannot cast them: %s" % missing


def test_nine_speaker_roster_casts_with_no_reuse():
    """The regression itself: this failed 3 of 9 characters at four voices."""
    bank, _ = load_voice_bank()
    bank = tuple(bank)
    used: set = set()
    picked = []
    for char_id, gender in ROSTER:
        ref = assign_voice_for_slot(
            role="char_voice", engine="google_tts", char_id=char_id,
            gender=gender, timbre=(), age_band="adult",
            episode_seed=20260808, allow_voice_reuse=False,
            used_voice_ref_ids=used, bank=bank,
        )
        used.add(ref.voice_ref_id)
        picked.append(ref.provider_voice_id)
    assert len(set(picked)) == len(ROSTER), picked


def test_exhausting_char_voices_falls_to_the_announcer_row_before_raising():
    """Documented as OBSERVED, not as desired.

    With every male char_voice row taken, casting hands a character the
    ANNOUNCER's voice (gt_charon_announcer) rather than raising. That is
    pre-existing behaviour and it is why the thin pool was dangerous: at the
    old FOUR castable voices it was reachable almost immediately, so the
    announcer could start speaking as a character in a normal-sized cast.
    Widening the pool makes it remote; it does not remove it. Pinned here so
    the day someone decides announcer separation must be absolute, this test
    is the thing that has to change on purpose.
    """
    bank, _ = load_voice_bank()
    bank = tuple(bank)
    char_males = {e.voice_ref_id for e in bank
                  if e.engine == "google_tts" and e.gender == "male"
                  and "char_voice" in tuple(e.roles or ())}
    ref = assign_voice_for_slot(
        role="char_voice", engine="google_tts", char_id="one_too_many",
        gender="male", timbre=(), age_band="adult", episode_seed=1,
        allow_voice_reuse=False, used_voice_ref_ids=char_males, bank=bank,
    )
    assert "announcer_voice" in tuple(ref.roles or ())


def test_true_exhaustion_raises_rather_than_reusing():
    """google_tts has a no-fallback contract: once the announcer rung is gone
    too, casting must be LOUD, never a silently duplicated voice."""
    bank, _ = load_voice_bank()
    bank = tuple(bank)
    all_male = {e.voice_ref_id for e in bank
                if e.engine == "google_tts" and e.gender == "male"}
    with pytest.raises(VoiceCastingError):
        assign_voice_for_slot(
            role="char_voice", engine="google_tts", char_id="one_too_many",
            gender="male", timbre=(), age_band="adult", episode_seed=1,
            allow_voice_reuse=False, used_voice_ref_ids=all_male, bank=bank,
        )


def _google_rows_on_disk():
    """Read the bank JSON directly.

    ``VoiceBankEntry`` is a frozen dataclass with fixed fields, so the
    evidence keys are dropped at load time -- they are on-disk PROVENANCE, not
    runtime state, and nothing in the render path reads them. That is fine and
    deliberate, but it means these assertions have to go to the file.
    """
    import json
    import pathlib

    bank_path = (pathlib.Path(__file__).resolve().parents[1]
                 / "config" / "voice_reference_bank.json")
    data = json.loads(bank_path.read_text(encoding="utf-8"))
    return [v for v in data["voices"] if v.get("engine") == "google_tts"]


@pytest.mark.parametrize("field", ["measured_median_f0_hz", "gender_source"])
def test_every_google_row_carries_its_gender_evidence(field):
    """Google publishes no gender and the schema requires one, so each row
    records what it was measured from. A row without evidence is an assertion,
    and an asserted gender is the 'character speaks with the wrong voice'
    defect waiting to happen."""
    for row in _google_rows_on_disk():
        assert row.get(field), "%s missing %s" % (row.get("voice_ref_id"), field)


def test_measured_pitch_agrees_with_the_recorded_gender():
    """The split (134.2 Hz) was derived from the six pre-existing rows and all
    six reproduced. A row whose stored pitch contradicts its stored gender
    means the bank was hand-edited away from the measurement -- which is
    exactly how a guessed gender would sneak back in."""
    split = 134.2
    for row in _google_rows_on_disk():
        f0 = row.get("measured_median_f0_hz")
        if not f0:
            continue
        expected = "male" if float(f0) < split else "female"
        assert row["gender"] == expected, (
            "%s: %.1f Hz implies %s but the bank says %s"
            % (row["voice_ref_id"], f0, expected, row["gender"]))
