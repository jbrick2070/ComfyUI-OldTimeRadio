"""The canvas preview must never be able to hurt a finished episode.

`_canvas_preview` draws the poster frame that tells a user their run worked --
added 2026-09-13 after the first person outside this project reported a green
run with an empty canvas and went through the filesystem to find out whether it
had produced anything.

It had no test at all when it was written, which is this repo's most repeated
defect in the other direction: correct code nothing exercises. Two independent
reviewers asked for these, and the ones that matter are not the happy path --
they are the guards, because the helper runs ON THE COMFYUI SERVER THREAD after
the episode is already on disk. Every failure here must degrade, never raise and
never block:

  * a missing ffmpeg, a missing ComfyUI, a failed extract -> text, no image
  * the ffmpeg spawn carries `-nostdin` AND a timeout, or a hung decoder takes
    the whole queue down with it and no `except` can reach it
  * a withheld episode says it was withheld, rather than showing a bare
    archival path beside a poster frame, which reads as a normal publish

No real ffmpeg, no real ComfyUI: every boundary is stubbed, so these run
anywhere the suite runs.
"""
from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nodes.otr_master_audio_mux as MUX  # noqa: E402


@pytest.fixture
def episode(tmp_path):
    """A file standing in for the finished mp4. Only its existence is read."""
    final = tmp_path / "the_shivering_gauge_20260913_103436__sbke__final.mp4"
    final.write_bytes(b"\x00" * 64)
    return final


@pytest.fixture
def comfy_temp(tmp_path, monkeypatch):
    """A stub `folder_paths` module, the way ComfyUI supplies one at runtime."""
    temp = tmp_path / "comfy_temp"
    mod = types.ModuleType("folder_paths")
    mod.get_temp_directory = lambda: str(temp)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", mod)
    return temp


def _stub_ffmpeg(monkeypatch, path="ffmpeg"):
    import nodes._otr_shared.ffmpeg as F
    monkeypatch.setattr(F, "resolve_ffmpeg", lambda *a, **k: path)


class TestItDrawsWhenItCan:
    def test_the_payload_matches_what_core_preview_nodes_emit(
            self, episode, comfy_temp, monkeypatch):
        """filename/subfolder/type, type == temp -- the SaveImage shape.

        If these keys drift, ComfyUI serves nothing and the canvas is empty
        again, which is the whole defect this helper exists to close.
        """
        _stub_ffmpeg(monkeypatch)
        monkeypatch.setattr(MUX, "_probe_float", lambda *a, **k: 90.0)

        def fake_run(argv, **kw):
            Path(argv[argv.index("-vf") + 2]).parent.mkdir(parents=True, exist_ok=True)
            Path(argv[-1]).write_bytes(b"PNG")
            return subprocess.CompletedProcess(argv, 0, "", "")
        monkeypatch.setattr(MUX.otr_proc, "run", fake_run)

        ui = MUX._canvas_preview(str(episode), "/obs/copy.mp4")
        assert list(ui["images"][0]) == ["filename", "subfolder", "type"]
        assert ui["images"][0]["type"] == "temp"
        assert ui["images"][0]["filename"].startswith("otr_preview_")
        assert ui["images"][0]["filename"].endswith(".png")

    def test_two_runs_of_one_episode_do_not_share_a_filename(
            self, episode, comfy_temp, monkeypatch):
        """A deterministic name served the PREVIOUS frame from the browser
        cache, because the /view URL was byte-identical. Core PreviewImage
        appends a random suffix for the same reason."""
        _stub_ffmpeg(monkeypatch)
        monkeypatch.setattr(MUX, "_probe_float", lambda *a, **k: 90.0)

        def fake_run(argv, **kw):
            Path(argv[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(argv[-1]).write_bytes(b"PNG")
            return subprocess.CompletedProcess(argv, 0, "", "")
        monkeypatch.setattr(MUX.otr_proc, "run", fake_run)

        first = MUX._canvas_preview(str(episode), "")["images"][0]["filename"]
        second = MUX._canvas_preview(str(episode), "")["images"][0]["filename"]
        assert first != second


class TestItCannotHangTheQueue:
    """The guard that matters. This runs on the ComfyUI server thread with the
    episode already written, so an ffmpeg that waits on stdin or deadlocks on a
    bad container would stall every later prompt -- and `except Exception`
    cannot catch a process that never returns."""

    def test_the_spawn_is_bounded_and_never_waits_on_stdin(
            self, episode, comfy_temp, monkeypatch):
        seen = {}
        _stub_ffmpeg(monkeypatch)
        monkeypatch.setattr(MUX, "_probe_float", lambda *a, **k: 90.0)

        def fake_run(argv, **kw):
            seen["argv"], seen["kw"] = argv, kw
            return subprocess.CompletedProcess(argv, 1, "", "")
        monkeypatch.setattr(MUX.otr_proc, "run", fake_run)

        MUX._canvas_preview(str(episode), "")
        assert "-nostdin" in seen["argv"], seen["argv"]
        assert seen["kw"].get("timeout"), "an unbounded preview can hang the queue"

    def test_a_timeout_degrades_instead_of_raising(
            self, episode, comfy_temp, monkeypatch):
        _stub_ffmpeg(monkeypatch)
        monkeypatch.setattr(MUX, "_probe_float", lambda *a, **k: 90.0)

        def boom(argv, **kw):
            raise subprocess.TimeoutExpired(argv, 20)
        monkeypatch.setattr(MUX.otr_proc, "run", boom)

        ui = MUX._canvas_preview(str(episode), "/obs/copy.mp4")
        assert "images" not in ui
        assert ui["text"] == ["/obs/copy.mp4"]


class TestItDegradesInSteps:
    def test_no_ffmpeg_still_names_the_file(
            self, episode, comfy_temp, monkeypatch):
        _stub_ffmpeg(monkeypatch, path="")
        ui = MUX._canvas_preview(str(episode), "/obs/copy.mp4")
        assert ui == {"text": ["/obs/copy.mp4"]}

    def test_no_comfyui_still_names_the_file(self, episode, monkeypatch):
        """Under pytest there is no `folder_paths`; the helper must not care."""
        monkeypatch.setitem(sys.modules, "folder_paths", None)
        ui = MUX._canvas_preview(str(episode), "/obs/copy.mp4")
        assert ui == {"text": ["/obs/copy.mp4"]}

    def test_a_missing_episode_is_not_an_error(self, tmp_path):
        ui = MUX._canvas_preview(str(tmp_path / "gone.mp4"), "")
        assert isinstance(ui, dict)


class TestTheWithheldEpisodeSaysSo:
    def test_a_withheld_episode_is_named_as_withheld(
            self, episode, comfy_temp, monkeypatch):
        """obs_copy is None when the rights receipt does not clear. A poster
        frame beside a bare archival path reads as a normal publish -- the
        exact confusion this preview exists to remove."""
        _stub_ffmpeg(monkeypatch, path="")
        ui = MUX._canvas_preview(str(episode), None)
        text = ui["text"][0]
        assert "not published" in text
        assert str(episode) in text

    def test_a_published_episode_names_the_obs_copy_only(
            self, episode, comfy_temp, monkeypatch):
        _stub_ffmpeg(monkeypatch, path="")
        ui = MUX._canvas_preview(str(episode), "/obs/copy.mp4")
        assert ui["text"] == ["/obs/copy.mp4"]
        assert "not published" not in ui["text"][0]


class TestTheAppPlaysTheEpisode:
    """Plan 0e: ComfyUI's app pane draws every list of {filename, subfolder,
    type} the output node returns and plays a .mp4 by its suffix. /view
    serves only files under the output directory, so only those are offered."""

    @pytest.fixture
    def comfy_output(self, tmp_path, monkeypatch):
        out = tmp_path / "output"
        mod = types.ModuleType("folder_paths")
        mod.get_output_directory = lambda: str(out)  # type: ignore[attr-defined]
        mod.get_temp_directory = lambda: str(tmp_path / "temp")  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "folder_paths", mod)
        _stub_ffmpeg(monkeypatch, path=None)   # no poster; the video still comes
        return out

    def test_the_published_copy_is_offered_as_a_playable_video(
            self, episode, comfy_output):
        obs = comfy_output / "otr" / "obs"
        obs.mkdir(parents=True)
        published = obs / "gauge_20260913__sbke_final.mp4"
        published.write_bytes(b"\x00" * 8)
        ui = MUX._canvas_preview(str(episode), str(published))
        assert ui["video"] == [{"filename": published.name,
                                "subfolder": "otr/obs", "type": "output"}]

    def test_a_file_outside_the_output_tree_is_not_offered(
            self, episode, comfy_output):
        ui = MUX._canvas_preview(str(episode), str(episode))
        assert "video" not in ui

    def test_without_comfyui_nothing_is_offered(self, episode, monkeypatch):
        monkeypatch.setitem(sys.modules, "folder_paths", None)
        assert MUX._served_output_ref(str(episode)) is None
