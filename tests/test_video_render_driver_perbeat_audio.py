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
        """If the output file already exists and is non-empty, ffmpeg is not
        called (keyed via slice_cache_key -- the 7.3 split)."""
        master = tmp_path / "master.mp4"
        master.write_bytes(b"fake")
        import tempfile as _tf
        from nodes._otr_video_engines._tmp import _in_tree_tmp_dir
        key = rd.slice_cache_key("", 1.5, 3.0, master_path=str(master))
        # R1: slices now land in the in-tree tmp tier (audio_slices leaf), with
        # the system temp dir only as a fallback -- mirror the code's resolution.
        _base = _in_tree_tmp_dir() or _tf.gettempdir()
        cache_dir = os.path.join(_base, "audio_slices")
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
# 2026-06-22: ambient-audio (ltx_av_music / visualizer) no-timing music-beat gap
# --------------------------------------------------------------------------- #
class TestAmbientAudioMusicBeatGap:
    def test_uses_ambient_master_audio_scope(self):
        # ltx_av_music's family + the visualizer engine = ambient master lanes.
        assert rd._uses_ambient_master_audio("ltx_av_music", "audio_conditioned_video") is True
        assert rd._uses_ambient_master_audio("visualizer", "abstract") is True
        # HuMo / talk (audio_driven_face) is EXCLUDED -- it needs the OWN voice,
        # never a master-mix slice (would lip-sync to the wrong audio).
        assert rd._uses_ambient_master_audio("humo", "audio_driven_face") is False
        assert rd._uses_ambient_master_audio("ltx_video", "text_to_video") is False

    def test_cumulative_beat_start_sums_preceding(self):
        ledger = {"video": {"fps": 25, "shots": [
            {"shot_id": "shot_a", "target_frame_count": 25},   # 1.0s
            {"shot_id": "shot_b", "target_frame_count": 50},   # 2.0s
            {"shot_id": "shot_c", "target_frame_count": 99},
        ]}}
        assert rd._cumulative_beat_start(ledger, {"shot_id": "shot_a"}, 25) == 0.0
        assert rd._cumulative_beat_start(ledger, {"shot_id": "shot_b"}, 25) == 1.0
        assert rd._cumulative_beat_start(ledger, {"shot_id": "shot_c"}, 25) == 3.0
        # unknown shot -> head (0.0), still bounded, never the whole master
        assert rd._cumulative_beat_start(ledger, {"shot_id": "nope"}, 25) == 0.0

    def test_music_beat_no_timing_gets_bounded_slice(self, tmp_path):
        """A music beat (ambient lane) with NO per-line/shot timing synthesizes a
        BOUNDED master slice (dur = target_frame_count/fps), so audio_ref is
        satisfied instead of FamilyInputGap-crashing. Never the whole master."""
        master = tmp_path / "master.mp4"
        master.write_bytes(b"fake-master")
        # a music line with NO start_s/dur_s (the inter-music-beat case)
        ledger = {
            "audio": {"master_audio_sha256": "deadbeef"},
            "video": {"fps": 25, "shots": [
                {"shot_id": "shot_pre", "target_frame_count": 25},   # 1.0s before
                {"shot_id": "shot_b006", "target_frame_count": 50},  # this beat, 2.0s
            ]},
            "lines": [{"line_id": "b006"}],   # no start_s / dur_s
            "images": {"images": []},
        }
        shot = {"shot_id": "shot_b006", "beat_id": "b006",
                "engine_id": "visualizer", "family": "abstract",
                "target_frame_count": 50, "source_line_ids": ["b006"],
                "creative": {"text_prompt": "scope", "request_hash": "h"}}
        captured = {}
        slice_path = str(tmp_path / "synth_slice.wav")

        def _fake_slice(path, start, dur, master_hash=""):
            captured["start"] = start
            captured["dur"] = dur
            with open(slice_path, "wb") as f:        # must EXIST (build_request filters)
                f.write(b"RIFF fake wav")
            return slice_path

        with mock.patch.object(rd, "_slice_master_audio", side_effect=_fake_slice):
            req = rd.build_request_from_shot(shot, ledger, master_audio_path=str(master))
        # audio_ref satisfied (would otherwise FamilyInputGap-crash ltx_av_music):
        # build_request puts it at the top-level req["audio_ref"], and
        # _present_request_tokens then reports "audio_ref" -> the family gate passes.
        assert (req.get("audio_ref") or {}).get("path") == slice_path
        assert "audio_ref" in rd._present_request_tokens(req)
        # BOUNDED: dur == the beat's own length (50/25 = 2.0s), NOT the whole master
        assert captured["dur"] == pytest.approx(2.0)
        # start == cumulative position of preceding beats (1.0s)
        assert captured["start"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 7.3 slice/curve cache-key SPLIT (3D plan; don't over-key the cheap WAV)
# --------------------------------------------------------------------------- #

class TestSliceCurveCacheKeySplit:
    def test_master_content_hash_is_the_slice_identity(self):
        """A NEW master at the SAME path invalidates (the shipped path-only
        key under-invalidated); the path is ignored once a hash is given."""
        a = rd.slice_cache_key("hash_aaa", 1.0, 2.0, master_path="X:/m.mp4")
        b = rd.slice_cache_key("hash_bbb", 1.0, 2.0, master_path="X:/m.mp4")
        assert a != b
        # path does not participate when the content hash is present
        c = rd.slice_cache_key("hash_aaa", 1.0, 2.0, master_path="Y:/other.mp4")
        assert a == c

    def test_hashless_fallback_is_path_keyed(self):
        a = rd.slice_cache_key("", 1.0, 2.0, master_path="X:/m.mp4")
        b = rd.slice_cache_key("", 1.0, 2.0, master_path="Y:/other.mp4")
        assert a != b          # legacy callers do not all collide on one key
        assert a == rd.slice_cache_key("", 1.0, 2.0, master_path="X:/m.mp4")

    def test_slice_key_binds_timing_rate_channels_version(self):
        base = dict(sample_rate=44100, channels=1, slicer_version="2")
        k = rd.slice_cache_key("h", 1.0, 2.0, **base)
        assert k != rd.slice_cache_key("h", 1.5, 2.0, **base)
        assert k != rd.slice_cache_key("h", 1.0, 2.5, **base)
        assert k != rd.slice_cache_key("h", 1.0, 2.0,
                                       **dict(base, sample_rate=16000))
        assert k != rd.slice_cache_key("h", 1.0, 2.0,
                                       **dict(base, channels=2))
        assert k != rd.slice_cache_key("h", 1.0, 2.0,
                                       **dict(base, slicer_version="3"))

    def test_curve_key_binds_driver_inputs_slice_key_does_not(self):
        """The SPLIT: driver-side inputs (line_id / fps / driver version /
        mapping hash / onset policy) churn ONLY the curve key -- the cheap
        44.1k WAV key never moves."""
        sk = rd.slice_cache_key("h", 1.0, 2.0)
        base = dict(fps=25, driver_version="rhubarb-1.13",
                    mapping_hash="map_v1", onset_policy="onset_in_clip_v1")
        ck = rd.curve_cache_key(sk, "b001", **base)
        assert ck != rd.curve_cache_key(sk, "b002", **base)
        assert ck != rd.curve_cache_key(sk, "b001", **dict(base, fps=24))
        assert ck != rd.curve_cache_key(
            sk, "b001", **dict(base, driver_version="rhubarb-1.14"))
        assert ck != rd.curve_cache_key(
            sk, "b001", **dict(base, mapping_hash="map_v2"))
        assert ck != rd.curve_cache_key(
            sk, "b001", **dict(base, onset_policy="other"))
        # the curve key DERIVES from the slice key (audio identity flows in)
        sk2 = rd.slice_cache_key("h2", 1.0, 2.0)
        assert ck != rd.curve_cache_key(sk2, "b001", **base)
        # and the slice key itself is untouched by any driver input
        assert sk == rd.slice_cache_key("h", 1.0, 2.0)

    def test_build_request_from_shot_threads_the_ledger_hash(self):
        """build_request_from_shot feeds master_audio_sha256 into the slicer
        (the 7.3 mechanics: the signature gained the hash)."""
        ledger = _ledger_with_line("b009", wav_path="", start_s=1.0, dur_s=2.0)
        shot = _shot("b009")
        with mock.patch("nodes._otr_video_engines.render_driver."
                        "_slice_master_audio", return_value="") as m:
            with mock.patch("nodes._otr_video_engines.render_driver."
                            "os.path.isfile", return_value=True):
                rd.build_request_from_shot(shot, ledger,
                                           master_audio_path="/fake/m.mp4")
        assert m.call_count == 1
        assert m.call_args.kwargs.get("master_hash") == rd.FROZEN_AUDIO_SHA


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
        m.assert_called_once_with(master, 2.0, 4.5,
                                  master_hash=rd.FROZEN_AUDIO_SHA)
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
        """start_s / dur_s from the ledger line are carried onto req['timing']
        (the schema spelling: dur_s -> target_duration_s, W7-pre migration)."""
        ledger = _ledger_with_line("b007", start_s=5.0, dur_s=7.5)
        shot = _shot("b007")
        req = rd.build_request_from_shot(shot, ledger)
        assert req["timing"]["start_s"] == 5.0
        assert req["timing"]["target_duration_s"] == 7.5
        assert "dur_s" not in req["timing"]

    def test_audio_frozen_section_untouched(self):
        """The frozen audio section in the ledger must not be modified."""
        ledger = _ledger_with_line("b008", start_s=1.0, dur_s=2.0)
        original_sha = ledger["audio"]["master_audio_sha256"]
        shot = _shot("b008")
        rd.build_request_from_shot(shot, ledger)
        assert ledger["audio"]["master_audio_sha256"] == original_sha
        assert ledger["audio"]["ledger_frozen"] is True

    def test_announcer_beat_resolves_init_image_and_slice(self, tmp_path):
        """The b001/b005 keystone gap, CPU-proven end to end: an ANNOUNCER beat
        whose ledger carries (a) a minted announcer portrait in
        ledger['images'] (the radio-style image; object_id='announcer') and
        (b) start_s/dur_s line timing resolves BOTH init_image AND a sliced
        audio_ref -- the two inputs eng_humo's instant guard requires. Root
        cause of the starvation was the MISSING announcer portrait, not the
        slice (forensics 2026-06-09: all 5 slice files existed, keys matched
        the re-resolved master WAV; the humo attempt failed in <1s on
        init_image='')."""
        portrait = tmp_path / "announcer_radio.png"
        portrait.write_bytes(b"\x89PNG\r\n\x1a\n fake")
        ledger = _ledger_with_line("b001", char_id="announcer",
                                   start_s=9.5, dur_s=9.40375,
                                   portrait_path=str(portrait))
        ledger["lines"][0]["speaker_role"] = "announcer"
        shot = _shot("b001")
        sliced = str(tmp_path / "slice_b001.wav")
        master = str(tmp_path / "master.wav")
        open(master, "wb").close()
        with mock.patch("nodes._otr_video_engines.render_driver._slice_master_audio",
                        return_value=sliced) as m:
            req = rd.build_request_from_shot(shot, ledger,
                                             master_audio_path=master)
        m.assert_called_once_with(master, 9.5, 9.40375,
                                  master_hash=rd.FROZEN_AUDIO_SHA)
        assert req["audio_ref"] == {"path": sliced}
        assert req["asset_refs"]["init_image"] == str(portrait), (
            "announcer beat must resolve the radio-style portrait as "
            "init_image -- without it HuMo fails its instant guard and the "
            "intro/outro starve to the still floor")


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

    def test_workflow_link_264_wired(self):
        """Chunk E cleanbreak: link 264 (node7[1] -> node92.master_audio_path)
        routes master audio from EpisodeAssembler.output_path, replacing the
        legacy link 262 (procgen-mp4 -> render_batch). Link 262 must be gone."""
        import json, os
        wf_path = os.path.join(_REPO, "workflows", "otr_scifi_16gb_full.json")
        with open(wf_path, encoding="utf-8") as f:
            wf = json.load(f)
        # Legacy link 262 (procgen-mp4 source) must be cut
        lnk262 = next((l for l in wf["links"] if l[0] == 262), None)
        assert lnk262 is None, (
            "link 262 (procgen-mp4 -> render_batch.master_audio_path) must be "
            "cut in Chunk E cleanbreak"
        )
        # New link 264 (assembler.output_path -> render_batch) must be present
        lnk = next((l for l in wf["links"] if l[0] == 264), None)
        assert lnk is not None, "link 264 (assembler -> render_batch.master_audio_path) missing"
        assert lnk[1] == 7,  f"link 264 src must be node 7 (EpisodeAssembler), got {lnk[1]}"
        assert lnk[2] == 1,  f"link 264 src slot must be 1 (output_path), got {lnk[2]}"
        assert lnk[3] == 92, f"link 264 dst must be node 92 (VideoRenderBatch), got {lnk[3]}"
        assert lnk[4] == 1,  f"link 264 dst slot must be 1 (master_audio_path), got {lnk[4]}"
        # Floor pin (later graph work appends links -- 2026-06-09 blend stage).
        assert wf["last_link_id"] >= 264
