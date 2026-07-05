"""S-C C1 fold-in tests: run_episode collects per-shot conditioning-WAV rows,
and OTR_VideoRenderBatch._stamp_audio_motion_profiles durably stamps them
(fail-soft). Read-only; no GPU, no ffmpeg. UTF-8, no BOM, SFW.
"""
import os
import sys
import unittest.mock as mock

import numpy as np
import pytest

soundfile = pytest.importorskip("soundfile")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes._otr_video_engines import cheap_families  # noqa: F401,E402 (register floor)
from nodes._otr_video_engines import render_driver as rd  # noqa: E402
from nodes import otr_video_render_batch as vrb  # noqa: E402
from nodes import _otr_ledger as OTRL  # noqa: E402


def _wav(tmp_path, name="c.wav"):
    p = tmp_path / name
    t = np.arange(8000, dtype="float32") / 16000.0
    soundfile.write(str(p), (0.4 * np.sin(2 * 3.14159 * 300 * t)).astype("float32"), 16000)
    return str(p)


# --------------------------------------------------------------------------- #
# run_episode collects audio_motion_rows from each shot's resolved audio_ref
# --------------------------------------------------------------------------- #
def test_run_episode_collects_audio_motion_rows(tmp_path):
    wav = _wav(tmp_path)
    ledger = {"video": {"shots": [
        {"shot_id": "shot_b1", "engine_id": "still_flat", "target_frame_count": 25}]}}

    def _fake_render_shot(shot, request, **kw):
        return ("clip.mp4", dict(shot), 1, 0)

    with mock.patch.object(rd, "render_shot", side_effect=_fake_render_shot):
        ep = rd.run_episode(ledger, assets={"audio_ref": wav}, frame_count=25)

    rows = ep.get("audio_motion_rows")
    assert rows and rows[0]["id"] == "shot_b1"
    assert rows[0]["_wav"] == wav


def test_run_episode_no_audio_ref_yields_no_rows(tmp_path):
    ledger = {"video": {"shots": [
        {"shot_id": "shot_b1", "engine_id": "still_flat", "target_frame_count": 25}]}}

    def _fake_render_shot(shot, request, **kw):
        return ("clip.mp4", dict(shot), 1, 0)

    with mock.patch.object(rd, "render_shot", side_effect=_fake_render_shot):
        ep = rd.run_episode(ledger, assets={}, frame_count=25)  # no audio_ref
    assert ep.get("audio_motion_rows") == []


# --------------------------------------------------------------------------- #
# _stamp_audio_motion_profiles -- durable stamp + fail-soft
# --------------------------------------------------------------------------- #
def test_stamp_empty_rows_is_noop():
    vrb._stamp_audio_motion_profiles([])          # must not raise


def test_stamp_soft_when_no_inflight_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(OTRL, "in_flight_ledger_path", lambda: None)
    # must not raise even though rows are present
    vrb._stamp_audio_motion_profiles([{"id": "b1", "_wav": _wav(tmp_path)}])


def test_stamp_writes_profiles_when_ledger_present(tmp_path, monkeypatch):
    from pathlib import Path
    saved = {}
    fake_path = Path(str(tmp_path / "ledger.json"))
    monkeypatch.setattr(OTRL, "in_flight_ledger_path", lambda: fake_path)
    monkeypatch.setattr(OTRL, "load_ledger_safe", lambda p: {"video": {}})

    def _save(p, led):
        saved["led"] = led
        return True

    monkeypatch.setattr(OTRL, "save_ledger_safe", _save)

    vrb._stamp_audio_motion_profiles([{"id": "b1", "_wav": _wav(tmp_path)}])
    led = saved["led"]
    assert "audio_motion_profiles" in led
    assert led["audio_motion_profiles"][0]["id"] == "b1"
    assert led["audio_motion_profiles"][0]["ok"] is True
    assert led["meta"]["audio_motion_profile"]["ok_count"] == 1


def test_stamp_never_raises_on_bad_wav(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(OTRL, "in_flight_ledger_path", lambda: Path(str(tmp_path / "l.json")))
    monkeypatch.setattr(OTRL, "load_ledger_safe", lambda p: {"video": {}})
    monkeypatch.setattr(OTRL, "save_ledger_safe", lambda p, led: True)
    # unreadable wav -> profile ok=False, but the stamp still succeeds soft
    vrb._stamp_audio_motion_profiles([{"id": "b1", "_wav": "C:/nope/missing.wav"}])
