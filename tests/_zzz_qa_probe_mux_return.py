"""THROWAWAY QA probe -- diagnoses whether OTRMasterAudioMux().mux()'s new
dict return breaks tests/test_video_render_path_cw4.py's real-ffmpeg call
site. Deleted immediately after use; not part of the suite."""
import sys
import types

from tests.test_video_render_path_cw4 import _sine, _silent_video, _publishable_episode, needs_ffmpeg
from nodes.otr_master_audio_mux import OTRMasterAudioMux


@needs_ffmpeg
def test_zzz_probe_mux_return_shape(tmp_path, monkeypatch):
    master = tmp_path / "master.wav"
    silent = tmp_path / "silent.mp4"
    _sine(master)
    _silent_video(silent)
    fake_fp = types.SimpleNamespace(get_output_directory=lambda: str(tmp_path / "out"))
    monkeypatch.setitem(sys.modules, "folder_paths", fake_fp)
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.delenv("OTR_OBS_DIR", raising=False)
    _publishable_episode(tmp_path, monkeypatch, "silent")
    node = OTRMasterAudioMux()
    raw = node.mux(str(silent), str(master))
    raw_type = type(raw).__name__
    final, status = raw
    assert False, (
        "PROBE RESULT :: raw_type=%s raw_keys=%s final=%r status=%r"
        % (raw_type,
           list(raw.keys()) if isinstance(raw, dict) else "n/a",
           final, status)
    )
