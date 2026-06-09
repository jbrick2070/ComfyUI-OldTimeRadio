"""CPU tests for the per-beat audio slice path in render_driver (Subproject-C #2).

Tests verify:
- _slice_master_audio: caches by hash, degrades LOUD on ffmpeg failure, skips
  on empty master path.
- build_request_from_shot: uses sliced audio when ledger has no *_wav_path AND
  master+timing are valid; falls back to no audio (LOUD warning) when timing is
  absent; uses existing per-line wav when present (no slice).
- run_real_episode: master_audio_path kwarg threads through partial into
  build_request_from_shot.
- OTR_VideoRenderBatch: INPUT_TYPES has master_audio_path forceInput; render()
  accepts it and passes to _render_episode.

No GPU, no ffmpeg binary required (subprocess.run is patched). UTF-8, no BOM,
ASCII-only, SFW.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest.mock as mock

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes._otr_video_engines import cheap_families  # noqa: F401 (register floor)
from nodes._otr_video_engines import render_driver as rd


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _ledger_with_line(beat_id, *, char_id="c01", wav_path="",
                      start_s=1.5, dur_s=3.0,
                      portrait_path=""):
    """Minimal ledger with one line entry and an images section."""
    line = {
        "line_id": beat_id,
        "char_id": char_id,
        "start_s": start_s,
        "dur_s": dur_s,
        "start_s_space": "master_mix",
    }
    if wav_path:
        line["indextts2_wav_path"] = wav_path
    images = []
    if portrait_path:
        images = [{"object_id": char_id, "path": portrait_path}]
    return {
        "audio": {"master_audio_sha256": rd.FROZEN_AUDIO_SHA,
                  "ledger_frozen": True},
        "video": {"video_revision": 1, "shots": []},
        "lines": [line],
        "images": {"images": images},
    }


def _shot(beat_id, *, engine="humo"):
    sid = "shot_%s" % beat_id
    return {
        "shot_id": sid, "beat_id": beat_id,
        "engine_id": engine, "family": "audio_driven_face",
        "target_frame_count": 25, "degradation_trail": [],
        "source_line_ids": [beat_id],
        "creative": {"text_prompt": "a radio studio", "request_hash": "aabbcc"},
    }


# --------------------------------------------------------------------------- #
# _slice_master_audio
# --------------------------------------------------------------------------- #

class TestSliceMasterAudio:
    def test_returns_empty_for_missing_master(self, tmp_path):
        out = rd._slice_master_audio("/no/such/file.mp4", 0.0, 2.0)
        assert out == ""

    def test_returns_empty_when_ffmpeg_fails(self, tmp_path):
        # Create a fake master file so the is-file check passes; patch subprocess
        master = tmp_path / "master.mp4"
        master.write_bytes(b"fake")
        with mock.patch("nodes._otr_video_engines.render_driver.subprocess.run",
                        side_effect=Exception("ffmpeg not found")):
            out = rd._slice_master_audio(str(master), 1.0, 2.0)
        assert out == ""

    def test_caches_hit_skips_ffmpeg(self, tmp_path, monkeypatch):
        """If the output file already exists and is non-empty, ffmpeg is not called."""
        master = tmp_path / "master.mp4"
        master.write_bytes(b"fake")
        # Pre-create the expected cache file
        import hashlib, tempfile as _tf
        key = hashlib.sha256(
            ("%.6f|%.6f|%s" % (1.5, 3.0, str(master))).encode("utf-8")
        ).hexdigest()[:16]
        cache_dir = os.path.join(_tf.gettempdir(), "otr_audio_slices")
        os.makedirs(cache_dir, exist_ok=True)
        cached = os.path.join(cache_dir, "slice_%s.wav" % key)
        with open(cached, "wb") as f:
            f.write(b"RIFF fake wav")
        run_calls = []
        with mock.patch("nodes._otr_video_engines.render_driver.subprocess.run",
                        side_effect=lambda *a, **kw: run_calls.append(a)):
            out = rd._slice_master_audio(str(master), 1.5, 3.0)
        assert out == cached
        assert len(run_calls) == 0, "ffmpeg must not be called on cache hit"

    def test_returns_path_on_success(self, tmp_path):
        """When ffmpeg succeeds and writes a non-empty file, the path is returned."""
        master = tmp_path / "master.mp4"
        master.write_bytes(b"fake")

        def _fake_run(cmd, **kwargs):
            # Write a minimal fake WAV to the output path
            out_path = cmd[-1]
            with open(out_path, "wb") as f:
                f.write(b"RIFF\x24\x00\x00\x00WAVEfmt ")
            return mock.Mock(returncode=0)

        with mock.patch("nodes._otr_video_engines.render_driver.subprocess.run",
                        side_effect=_fake_run):
            out = rd._slice_master_audio(str(master), 0.5, 2.5)
        assert out != ""
        assert os.path.exists(out)

    def test_returns_empty_when_ffmpeg_writes_empty_file(self, tmp_path):
        master = tmp_path / "master.mp4"
        master.write_bytes(b"fake")

        def _fake_run(cmd, **kwargs):
            # Create the file but leave it empty
            out_path = cmd[-1]
            open(out_path, "wb").close()
            return mock.Mock(returncode=0)

        with mock.patch("nodes._otr_video_engines.render_driver.subprocess.run",
                        side_effect=_fake_run):
            out = rd._slice_master_audio(str(master), 0.5, 2.5)
        assert out == ""


# --------------------------------------------------------------------------- #
# build_request_from_shot -- per-beat audio path
# --------------------------------------------------------------------------- #

class TestBuildRequestFromShotPerBeatAudio:
    def test_uses_existing_wav_path_no_slice(self, tmp_path):
        """If ledger line has a *_wav_path, use it directly (no slice)."""
        wav = tmp_path / "line.wav"
        wav.write_bytes(b"RIFF fake")
        ledger = _ledger_with_line("b001", wav_path=str(wav))
        shot = _shot("b001")
        with mock.patch("nodes._otr_video_engines.render_driver._slice_master_audio") as m:
            req = rd.build_request_from_shot(shot, ledger,
                                             master_audio_path="/any/master.mp4")
        m.assert_not_called()
        assert req["audio_ref"] == {"path": str(wav)}

    def test_slices_from_master_when_no_wav(self, tmp_path):
        """If ledger line has no *_wav_path + master + timing -> slice is called."""
        ledger = _ledger_with_line("b002", start_s=2.0, dur_s=4.5)
        shot = _shot("b002")
        sliced = str(tmp_path / "slice.wav")
        master = str(tmp_path / "master.mp4")
        open(master, "wb").close()  # must exist for os.path.isfile

        with mock.patch("nodes._otr_video_engines.render_driver._slice_master_audio",
                        return_value=sliced) as m:
            # Also patch os.path.isfile so master "exists"
            orig_isfile = os.path.isfile
            with mock.patch("nodes._otr_video_engines.render_driver.os.path.isfile",
                            side_effect=lambda p: p == master or orig_isfile(p)):
                req = rd.build_request_from_shot(shot, ledger,
                                                 master_audio_path=master)
        m.assert_called_once_with(master, 2.0, 4.5)
        assert req["audio_ref"] == {"path": sliced}

    def test_no_audio_when_master_empty(self):
        """If master_audio_path is '', no slice is attempted; audio_ref is None."""
        ledger = _ledger_with_line("b003", start_s=1.0, dur_s=3.0)
        shot = _shot("b003")
        with mock.patch("nodes._otr_video_engines.render_driver._slice_master_audio") as m:
            req = rd.build_request_from_shot(shot, ledger, master_audio_path="")
        m.assert_not_called()
        assert req["audio_ref"] is None

    def test_no_audio_when_timing_missing(self, tmp_path):
        """If line has no start_s/dur_s, slice is NOT called (LOUD warning)."""
        # patch isfile so the master "exists" without creating a real file
        ledger = {
            "audio": {"master_audio_sha256": rd.FROZEN_AUDIO_SHA, "ledger_frozen": True},
            "video": {"video_revision": 1, "shots": []},
            "lines": [{"line_id": "b004", "char_id": "c01"}],
            "images": {"images": []},
        }
        shot = _shot("b004")
        with mock.patch("nodes._otr_video_engines.render_driver._slice_master_audio") as m:
            with mock.patch("nodes._otr_video_engines.render_driver.os.path.isfile",
                            return_value=True):
                req = rd.build_request_from_shot(shot, ledger,
                                                 master_audio_path="/fake/master.mp4")
        m.assert_not_called()
        assert req["audio_ref"] is None

    def test_no_audio_when_dur_s_zero(self, tmp_path):
        """If dur_s == 0, slice is NOT called."""
        ledger = _ledger_with_line("b005", start_s=1.0, dur_s=0.0)
        shot = _shot("b005")
        with mock.patch("nodes._otr_video_engines.render_driver._slice_master_audio") as m:
            with mock.patch("nodes._otr_video_engines.render_driver.os.path.isfile",
                            return_value=True):
                req = rd.build_request_from_shot(shot, ledger,
                                                 master_audio_path="/fake/master.mp4")
        m.assert_not_called()
        assert req["audio_ref"] is None

    def test_degrades_to_no_audio_when_slice_fails(self, tmp_path):
        """If _slice_master_audio returns '', audio_ref is None (LOUD warn, no crash)."""
        ledger = _ledger_with_line("b006", start_s=1.0, dur_s=2.0)
        shot = _shot("b006")
        with mock.patch("nodes._otr_video_engines.render_driver._slice_master_audio",
                        return_value=""):
            with mock.patch("nodes._otr_video_engines.render_driver.os.path.isfile",
                            return_value=True):
                req = rd.build_request_from_shot(shot, ledger,
                                                 master_audio_path="/fake/master.mp4")
        assert req["audio_ref"] is None

    def test_timing_carried_onto_request(self):
        """start_s / dur_s from the ledger line are carried onto req['timing']."""
        ledger = _ledger_with_line("b007", start_s=5.0, dur_s=7.5)
        shot = _shot("b007")
        req = rd.build_request_from_shot(shot, ledger)
        assert req["timing"]["start_s"] == 5.0
        assert req["timing"]["dur_s"] == 7.5

    def test_audio_frozen_section_untouched(self):
        """The frozen audio section in the ledger must not be modified."""
        ledger = _ledger_with_line("b008", start_s=1.0, dur_s=2.0)
        original_sha = ledger["audio"]["master_audio_sha256"]
        shot = _shot("b008")
        rd.build_request_from_shot(shot, ledger)
        assert ledger["audio"]["master_audio_sha256"] == original_sha
        assert ledger["audio"]["ledger_frozen"] is True


# --------------------------------------------------------------------------- #
# run_real_episode: master_audio_path threads through
# --------------------------------------------------------------------------- #

class TestRunRealEpisodeMasterAudioPath:
    def test_master_audio_path_threaded_to_request_builder(self):
        """run_real_episode passes master_audio_path into build_request_from_shot
        via functools.partial; the partial carries the keyword so every per-shot
        call automatically uses the right master path."""
        import functools

        ledger = {
            "audio": {"master_audio_sha256": rd.FROZEN_AUDIO_SHA,
                      "ledger_frozen": True},
            "video": {"video_revision": 1, "shots": []},
            "lines": [{"line_id": "b001", "start_s": 1.0, "dur_s": 2.0}],
        }
        fake_master = "/fake/master.mp4"

        captured = {}

        def _fake_run_episode(led, *, fallback_of=None,
                              request_builder=None, canvas=None):
            captured["request_builder"] = request_builder
            return {"ledger": led, "clips": {}, "trace": [], "vram_peak_mb": 0}

        # Don't touch functools.partial itself -- just spy on run_episode
        with mock.patch("nodes._otr_video_engines.render_driver.run_episode",
                        side_effect=_fake_run_episode):
            rd.run_real_episode(ledger, master_audio_path=fake_master)

        rb = captured.get("request_builder")
        assert rb is not None, "request_builder was not passed to run_episode"
        assert isinstance(rb, functools.partial), \
            "request_builder must be a functools.partial"
        assert rb.keywords.get("master_audio_path") == fake_master, (
            "master_audio_path not threaded into partial; "
            "got keywords: %r" % rb.keywords
        )


# --------------------------------------------------------------------------- #
# OTR_VideoRenderBatch node interface
# --------------------------------------------------------------------------- #

class TestVideoRenderBatchMasterAudioInput:
    def test_master_audio_path_in_input_types(self):
        from nodes.otr_video_render_batch import OTRVideoRenderBatch
        it = OTRVideoRenderBatch.INPUT_TYPES()
        opt = it.get("optional", {})
        assert "master_audio_path" in opt, \
            "master_audio_path must be an optional input on OTR_VideoRenderBatch"
        spec = opt["master_audio_path"]
        assert spec[0] == "STRING"
        assert spec[1].get("forceInput") is True, \
            "master_audio_path must be forceInput=True (not a widget)"

    def test_render_method_accepts_master_audio_path(self):
        """render() must accept master_audio_path kwarg without TypeError."""
        from nodes.otr_video_render_batch import OTRVideoRenderBatch
        import inspect
        sig = inspect.signature(OTRVideoRenderBatch.render)
        assert "master_audio_path" in sig.parameters

    def test_render_episode_receives_master_audio_path(self):
        """_render_episode receives master_audio_path from render()."""
        from nodes.otr_video_render_batch import OTRVideoRenderBatch

        received = []

        def _fake_render_episode(rd, ledger_json, master_audio_path=""):
            received.append(master_audio_path)
            return ({"ok": False, "error": "stub"}, "", "stub.json")

        node = OTRVideoRenderBatch()
        with mock.patch.object(OTRVideoRenderBatch, "_render_episode",
                               staticmethod(_fake_render_episode)):
            node.render("episode", 1, 0, 25,
                        master_audio_path="/my/master.mp4")

        assert received == ["/my/master.mp4"], \
            "master_audio_path must reach _render_episode"

    def test_widgets_values_unchanged_seven_slots(self):
        """forceInput master_audio_path must NOT add a widgets_values slot."""
        import json, os
        wf_path = os.path.join(_REPO, "workflows", "otr_scifi_16gb_full.json")
        with open(wf_path, encoding="utf-8") as f:
            wf = json.load(f)
        n92 = next(n for n in wf["nodes"] if n["id"] == 92)
        wv = n92.get("widgets_values", [])
        # 7 widget slots: mode, beats, oom_index, frame_count, engine,
        # portrait_path, audio_path.  patched_ledger_json + master_audio_path
        # are forceInput so they are NOT in widgets_values.
        assert len(wv) == 7, (
            "widgets_values must have exactly 7 entries (forceInput fields "
            "excluded); got %d: %r" % (len(wv), wv))

    def test_workflow_link_262_wired(self):
        """Link 262 (node12[0] -> node92.master_audio_path) must exist."""
        import json, os
        wf_path = os.path.join(_REPO, "workflows", "otr_scifi_16gb_full.json")
        with open(wf_path, encoding="utf-8") as f:
            wf = json.load(f)
        lnk = next((l for l in wf["links"] if l[0] == 262), None)
        assert lnk is not None, "link 262 missing from workflow"
        assert lnk[1] == 12 and lnk[2] == 0, \
            "link 262 must originate from node 12 slot 0"
        assert lnk[3] == 92, "link 262 must target node 92"
        assert wf["last_link_id"] == 262
