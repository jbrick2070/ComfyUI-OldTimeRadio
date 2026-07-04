"""Sonilo cloud-music min-duration FLOOR + trim (2026-07-04).

Sonilo 422-rejects the OTR theme cues (opening 12 / closing 8 / interstitial 4 s)
as too short. eng_cloud_sonilo.generate_clip REQUESTS at least the service floor
(OTR_SONILO_MIN_DURATION_S, default 30) and TRIMS the returned audio back to the
cue's canonical length so cue identity + the mix are unchanged. Proven live
2026-07-04 (all three cues cleared the 422 at floor=30).
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OTR_TEST_MODE", "1")


def _run(monkeypatch, tmp_path, floor, cue_s):
    import numpy as np
    import soundfile as sf
    from nodes._otr_audio_engines.eng_cloud_sonilo import SoniloCloudMusic
    import nodes._otr_shared.cloud_media_invoke as inv
    import nodes._otr_shared.cloud_media_canonical as canon_mod

    sr = 44100
    # the provider returns audio at the REQUESTED (floored) length
    req = max(cue_s, floor) if floor else cue_s
    sf.write(str(tmp_path / "m.wav"),
             np.zeros((int(req * sr), 2), dtype="float32"), sr)
    seen = {}

    def fake_invoke(row, inputs, **kw):
        seen["duration"] = inputs["duration"]
        return object()

    class _Canon:
        path = str(tmp_path / "m.wav")

    monkeypatch.setattr(inv, "invoke_partner_node", fake_invoke)
    monkeypatch.setattr(canon_mod, "canonicalize_audio", lambda *a, **k: _Canon())
    if floor is None:
        monkeypatch.delenv("OTR_SONILO_MIN_DURATION_S", raising=False)
    else:
        monkeypatch.setenv("OTR_SONILO_MIN_DURATION_S", str(floor))
    audio = SoniloCloudMusic().generate_clip("gentle warm piano", cue_s, 0)
    return seen, audio, sr


def test_sonilo_floors_request_and_trims_back(monkeypatch, tmp_path):
    seen, audio, sr = _run(monkeypatch, tmp_path, 30, 12)
    assert seen["duration"] == 30, "sub-floor cue must be REQUESTED at the floor"
    # delivered clip trimmed back to the 12s cue length (sample-accurate-ish)
    assert abs(audio["waveform"].shape[-1] - 12 * sr) < sr, \
        "returned cue must be trimmed to its canonical length, not the floor"


def test_sonilo_floor_zero_disables(monkeypatch, tmp_path):
    seen, audio, sr = _run(monkeypatch, tmp_path, 0, 12)
    assert seen["duration"] == 12, "floor=0 must request the raw cue duration"


def test_sonilo_above_floor_passthrough(monkeypatch, tmp_path):
    seen, audio, sr = _run(monkeypatch, tmp_path, 30, 45)
    assert seen["duration"] == 45, "a cue >= floor is requested unchanged"
    assert abs(audio["waveform"].shape[-1] - 45 * sr) < sr, "no trim above the floor"
