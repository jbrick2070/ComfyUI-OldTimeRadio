"""S-C C1 -- audio_motion_profile EXTRACTION core tests (analysis only).

Proves the 8-field profile is deterministic, correctly ordered on the obvious
axes (loud>quiet, bright>dark, onset timing), and -- the load-bearing invariant
-- that extraction is READ-ONLY: the source WAV is byte-identical before/after.
No consumers are tested (C2 deferred); no ComfyUI import.
"""
import hashlib
import math

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")

from nodes._otr_audio_motion import (  # noqa: E402
    PROFILE_FIELDS,
    PROFILE_VERSION,
    analyze_samples,
    analyze_wav,
    build_audio_motion_profiles,
    stamp_ledger_audio_motion,
)

_SR = 44100


def _tone(freq, secs, amp=0.5, lead_silence_s=0.0, sr=_SR):
    lead = np.zeros(int(lead_silence_s * sr), dtype="float32")
    t = np.arange(int(secs * sr), dtype="float32") / sr
    wav = (amp * np.sin(2 * math.pi * freq * t)).astype("float32")
    return np.concatenate([lead, wav])


def _write(tmp_path, name, samples, sr=_SR):
    p = tmp_path / name
    soundfile.write(str(p), samples, sr)
    return str(p)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# schema / determinism
# --------------------------------------------------------------------------- #
def test_profile_has_exactly_the_field_contract(tmp_path):
    prof = analyze_wav(_write(tmp_path, "a.wav", _tone(440, 1.0)))
    assert prof is not None
    assert set(prof) == set(PROFILE_FIELDS)
    for k in PROFILE_FIELDS:
        if k == "speech_vs_music":
            assert prof[k] in ("speech", "music", "mixed")
        else:
            assert isinstance(prof[k], (int, float))


def test_extraction_is_deterministic(tmp_path):
    p = _write(tmp_path, "d.wav", _tone(330, 0.8, amp=0.4))
    assert analyze_wav(p) == analyze_wav(p)


def test_duration_matches(tmp_path):
    prof = analyze_wav(_write(tmp_path, "dur.wav", _tone(200, 1.5)))
    assert prof["duration_s"] == pytest.approx(1.5, abs=0.02)


# --------------------------------------------------------------------------- #
# feature axes
# --------------------------------------------------------------------------- #
def test_louder_clip_has_higher_rms(tmp_path):
    loud = analyze_wav(_write(tmp_path, "loud.wav", _tone(300, 1.0, amp=0.8)))
    quiet = analyze_wav(_write(tmp_path, "quiet.wav", _tone(300, 1.0, amp=0.05)))
    assert loud["rms_dbfs"] > quiet["rms_dbfs"]
    assert loud["peak_dbfs"] > quiet["peak_dbfs"]


def test_high_freq_is_brighter(tmp_path):
    dark = analyze_wav(_write(tmp_path, "dark.wav", _tone(120, 1.0)))
    bright = analyze_wav(_write(tmp_path, "bright.wav", _tone(9000, 1.0)))
    assert bright["brightness"] > dark["brightness"]


def test_silence_clip_is_mostly_silent_and_never_onsets(tmp_path):
    sil = analyze_wav(_write(tmp_path, "sil.wav", np.zeros(_SR, dtype="float32")))
    assert sil["silence_ratio"] >= 0.95
    assert sil["onset_s"] == -1.0


def test_onset_after_leading_silence(tmp_path):
    prof = analyze_wav(_write(tmp_path, "onset.wav",
                              _tone(440, 1.0, amp=0.6, lead_silence_s=0.5)))
    assert prof["onset_s"] == pytest.approx(0.5, abs=0.05)


def test_analyze_empty_samples_is_safe():
    prof = analyze_samples(np.zeros(0, dtype="float32"), _SR)
    assert prof["duration_s"] == 0.0
    assert prof["onset_s"] == -1.0


# --------------------------------------------------------------------------- #
# READ-ONLY invariant (the whole point of C1)
# --------------------------------------------------------------------------- #
def test_extraction_never_mutates_the_wav(tmp_path):
    p = _write(tmp_path, "frozen.wav", _tone(440, 1.0, amp=0.7))
    before = _sha(p)
    analyze_wav(p)
    analyze_wav(p)
    assert _sha(p) == before


def test_bad_path_returns_none():
    assert analyze_wav("C:/nope/does_not_exist_amp.wav") is None


# --------------------------------------------------------------------------- #
# ledger orchestration (extraction + stamp), resolver-driven, side-effect-free
# --------------------------------------------------------------------------- #
def test_build_profiles_over_rows_and_resolver(tmp_path):
    good = _write(tmp_path, "b1.wav", _tone(440, 0.6))
    rows = [
        {"beat_id": "b1", "start_s": 0.0, "dur_s": 0.6},
        {"beat_id": "b2", "start_s": 0.6, "dur_s": 0.5},   # resolver -> None
        {"line_id": "l3", "start_s": 1.1, "dur_s": 0.4},   # unreadable path
    ]

    def resolver(row):
        rid = row.get("beat_id") or row.get("line_id")
        if rid == "b1":
            return good
        if rid == "b2":
            return None
        return "C:/nope/missing.wav"

    profs = build_audio_motion_profiles(rows, resolver)
    assert [p["id"] for p in profs] == ["b1", "b2", "l3"]
    assert profs[0]["ok"] is True
    assert set(PROFILE_FIELDS).issubset(profs[0])
    assert profs[0]["profile_version"] == PROFILE_VERSION
    assert profs[1]["ok"] is False and "resolved" in profs[1]["reason"]
    assert profs[2]["ok"] is False and "unreadable" in profs[2]["reason"]


def test_resolver_exception_is_captured_not_raised():
    def boom(_row):
        raise RuntimeError("kaboom")

    profs = build_audio_motion_profiles([{"beat_id": "x"}], boom)
    assert profs[0]["ok"] is False
    assert "resolver error" in profs[0]["reason"]


def test_stamp_touches_only_its_section():
    ledger = {"audio": {"master_audio_sha256": "FROZEN"}, "video": {"a": 1}}
    profs = [{"id": "b1", "ok": True}]
    out = stamp_ledger_audio_motion(ledger, profs)
    assert out is ledger
    assert out["audio"] == {"master_audio_sha256": "FROZEN"}   # untouched
    assert out["video"] == {"a": 1}                            # untouched
    assert out["audio_motion_profiles"] == profs
    assert out["meta"]["audio_motion_profile"]["version"] == PROFILE_VERSION
    assert out["meta"]["audio_motion_profile"]["count"] == 1
    assert out["meta"]["audio_motion_profile"]["ok_count"] == 1


def test_stamp_rejects_non_dict():
    with pytest.raises(TypeError):
        stamp_ledger_audio_motion(["not", "a", "dict"], [])
