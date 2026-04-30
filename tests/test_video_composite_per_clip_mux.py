"""
test_video_composite_per_clip_mux.py
=====================================

Unit coverage for ROADMAP P0 Step 6: VideoComposite's
``master_mix_per_clip_mux`` mode (C7 byte-perfect audio path).

Strategy: mock ``subprocess.run`` so we verify the SHAPE of the
ffmpeg invocations -- specifically that:
  - pillarbox pass uses ``-an`` (audio stripped)
  - pillarbox pass re-encodes video with libx264 (acceptable: Step 6
    explicitly accepts one video re-encode in exchange for zero
    audio re-encodes)
  - concat-demux pass uses ``-c copy`` (no re-encode)
  - mux pass uses ``-c:v copy -c:a copy -shortest`` (zero re-encodes)
  - audio source for the mux pass is the procgen mp4 (master mix lives
    there)
  - timeline is sorted by start_s ascending
  - missing clips on disk are skipped, not fatal
  - empty timeline raises a clean RuntimeError so the caller can fall
    back to humo_concat

This is a NO-FFMPEG test: we never actually shell out.  Real-ffmpeg
verification happens during a ComfyUI test run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_NODES_DIR = os.path.join(_REPO_ROOT, "nodes")
for p in (_REPO_ROOT, _NODES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# video_composite imports `from . import _otr_ledger` etc.  When loaded
# directly we need the parent package (`nodes`) findable.  Use absolute
# import via importlib.
import importlib.util


def _load_video_composite_module():
    spec = importlib.util.spec_from_file_location(
        "video_composite", os.path.join(_NODES_DIR, "video_composite.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


VC = _load_video_composite_module()


@pytest.fixture()
def fake_clip(tmp_path: Path) -> Path:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"fake mp4")
    return p


@pytest.fixture()
def fake_procgen(tmp_path: Path) -> Path:
    p = tmp_path / "procgen.mp4"
    p.write_bytes(b"fake procgen mp4")
    return p


# ---------------------------------------------------------------------------
# _pillarbox_humo_silent
# ---------------------------------------------------------------------------

class TestPillarboxHumoSilent:
    """The new helper that strips audio from a HuMo clip while
    pillarboxing video."""

    def test_command_strips_audio(self, fake_clip, tmp_path):
        out = tmp_path / "out.mp4"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=b"", stderr=b"",
            )
            VC._pillarbox_humo_silent(
                clip=fake_clip,
                canvas_w=1920, canvas_h=1080, canvas_fps=25,
                humo_target_h=1080,
                out_path=out, ffmpeg="ffmpeg",
            )
            cmd = mock_run.call_args[0][0]
            assert "-an" in cmd, (
                "pillarbox helper must use -an to strip audio so the "
                "downstream mux step can attach master mix audio "
                "with -c:a copy (zero audio re-encode)"
            )
            # Audio codec should NOT be specified at all (since -an
            # drops the stream entirely).
            assert "-c:a" not in cmd, (
                "with -an, audio codec args are spurious"
            )

    def test_command_re_encodes_video_to_libx264(self, fake_clip, tmp_path):
        out = tmp_path / "out.mp4"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=b"", stderr=b"",
            )
            VC._pillarbox_humo_silent(
                clip=fake_clip,
                canvas_w=1920, canvas_h=1080, canvas_fps=25,
                humo_target_h=1080,
                out_path=out, ffmpeg="ffmpeg",
            )
            cmd = mock_run.call_args[0][0]
            assert "-c:v" in cmd
            cv_idx = cmd.index("-c:v")
            assert cmd[cv_idx + 1] == "libx264"
            # Step 6 accepts ONE video re-encode in exchange for zero
            # audio re-encodes.  The commit message documents this.

    def test_canvas_dims_in_filter(self, fake_clip, tmp_path):
        out = tmp_path / "out.mp4"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=b"", stderr=b"",
            )
            VC._pillarbox_humo_silent(
                clip=fake_clip,
                canvas_w=1920, canvas_h=1080, canvas_fps=25,
                humo_target_h=1080,
                out_path=out, ffmpeg="ffmpeg",
            )
            cmd = mock_run.call_args[0][0]
            vf_idx = cmd.index("-vf")
            vf = cmd[vf_idx + 1]
            assert "scale=-2:1080" in vf
            assert "pad=1920:1080" in vf
            assert "fps=25" in vf


# ---------------------------------------------------------------------------
# _render_master_mix_per_clip_mux_mode
# ---------------------------------------------------------------------------

@pytest.fixture()
def populated_clips_dir(tmp_path: Path) -> Path:
    """Three fake HuMo clips with stable names matching the ledger
    line_ids below.  Content doesn't matter -- subprocess.run is
    mocked."""
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    for line_id in ("char_l001", "music_open_001", "music_close_001"):
        (clips_dir / f"{line_id}.mp4").write_bytes(b"x")
    return clips_dir


@pytest.fixture()
def populated_ledger() -> dict:
    """Realistic ledger shape post-Step 4b/4c: lines[] covers
    music_open + dialogue + music_close in master_mix space."""
    return {
        "episode_id": "test_ep",
        "lines": [
            {
                "line_id": "music_open_001",
                "speaker_role": "music_open",
                "start_s": 0.0,
                "dur_s": 12.0,
                "start_s_space": "master_mix",
            },
            {
                "line_id": "char_l001",
                "speaker_role": "character",
                "start_s": 11.5,   # crossfade overlap with music_open
                "dur_s":   5.0,
                "start_s_space": "master_mix",
            },
            {
                "line_id": "music_close_001",
                "speaker_role": "music_close",
                "start_s": 16.0,
                "dur_s":   8.0,
                "start_s_space": "master_mix",
            },
        ],
    }


class TestRenderMasterMixPerClipMux:
    """Top-level helper: pillarbox -> concat -> mux."""

    def test_empty_lines_raises(self, tmp_path, fake_procgen):
        out = tmp_path / "out.mp4"
        with pytest.raises(RuntimeError, match="no usable HuMo clips"):
            VC._render_master_mix_per_clip_mux_mode(
                ledger={"lines": []},
                clips_dir=tmp_path,
                procgen=fake_procgen,
                out_mp4=out,
                canvas_w=1920, canvas_h=1080, canvas_fps=25,
                humo_target_h=1080,
                fallback_clip_length=7.0,
                ffmpeg="ffmpeg", ffprobe="ffprobe",
            )

    def test_lines_sorted_by_start_s(
        self, populated_ledger, populated_clips_dir, fake_procgen, tmp_path
    ):
        # Reverse the ledger's order to prove sorting happens.
        led = dict(populated_ledger)
        led["lines"] = list(reversed(populated_ledger["lines"]))
        out = tmp_path / "out.mp4"
        seen_pillarbox_inputs = []

        def _record_run(cmd, **_kw):
            # Detect pillarbox calls by the -an flag, capture the input clip.
            if "-an" in cmd and "-i" in cmd:
                i = cmd.index("-i")
                seen_pillarbox_inputs.append(cmd[i + 1])
            # Also touch the output file so existence checks pass.
            if str(cmd[-1]).endswith(".mp4"):
                Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
                Path(cmd[-1]).write_bytes(b"x")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=b"", stderr=b"",
            )

        with patch("subprocess.run", side_effect=_record_run):
            VC._render_master_mix_per_clip_mux_mode(
                ledger=led,
                clips_dir=populated_clips_dir,
                procgen=fake_procgen,
                out_mp4=out,
                canvas_w=1920, canvas_h=1080, canvas_fps=25,
                humo_target_h=1080,
                fallback_clip_length=7.0,
                ffmpeg="ffmpeg", ffprobe="ffprobe",
            )

        # Pillarbox order must match start_s ascending (music_open ->
        # char -> music_close), regardless of the ledger ordering.
        names = [Path(p).stem for p in seen_pillarbox_inputs]
        assert names == ["music_open_001", "char_l001", "music_close_001"]

    def test_final_mux_uses_c_copy_for_audio(
        self, populated_ledger, populated_clips_dir, fake_procgen, tmp_path
    ):
        """The whole point of Step 6: -c:a copy at the mux stage."""
        out = tmp_path / "out.mp4"
        all_cmds = []

        def _record_run(cmd, **_kw):
            all_cmds.append(list(cmd))
            if str(cmd[-1]).endswith(".mp4"):
                Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
                Path(cmd[-1]).write_bytes(b"x")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=b"", stderr=b"",
            )

        with patch("subprocess.run", side_effect=_record_run):
            VC._render_master_mix_per_clip_mux_mode(
                ledger=populated_ledger,
                clips_dir=populated_clips_dir,
                procgen=fake_procgen,
                out_mp4=out,
                canvas_w=1920, canvas_h=1080, canvas_fps=25,
                humo_target_h=1080,
                fallback_clip_length=7.0,
                ffmpeg="ffmpeg", ffprobe="ffprobe",
            )

        # The final mux command should reference procgen as input,
        # use -c:a copy AND -c:v copy AND -shortest, and write to
        # out_mp4.
        mux_cmd = next(
            c for c in all_cmds
            if str(fake_procgen) in c and str(out) in c
        )
        # Audio is the C7 critical path -- must be copy.
        ca_idx = mux_cmd.index("-c:a")
        assert mux_cmd[ca_idx + 1] == "copy"
        # Video is also copy at the mux stage (concat already produced
        # the right-dim video in the previous pass).
        cv_idx = mux_cmd.index("-c:v")
        assert mux_cmd[cv_idx + 1] == "copy"
        # -shortest trims any audio overhang.
        assert "-shortest" in mux_cmd

    def test_concat_pass_uses_c_copy(
        self, populated_ledger, populated_clips_dir, fake_procgen, tmp_path
    ):
        out = tmp_path / "out.mp4"
        all_cmds = []

        def _record_run(cmd, **_kw):
            all_cmds.append(list(cmd))
            if str(cmd[-1]).endswith(".mp4"):
                Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
                Path(cmd[-1]).write_bytes(b"x")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=b"", stderr=b"",
            )

        with patch("subprocess.run", side_effect=_record_run):
            VC._render_master_mix_per_clip_mux_mode(
                ledger=populated_ledger,
                clips_dir=populated_clips_dir,
                procgen=fake_procgen,
                out_mp4=out,
                canvas_w=1920, canvas_h=1080, canvas_fps=25,
                humo_target_h=1080,
                fallback_clip_length=7.0,
                ffmpeg="ffmpeg", ffprobe="ffprobe",
            )

        # Find the concat-demuxer call (uses -f concat).
        concat_cmd = next(c for c in all_cmds if "-f" in c and "concat" in c)
        # Should use -c copy for everything (no re-encode).
        c_idx = concat_cmd.index("-c")
        assert concat_cmd[c_idx + 1] == "copy"

    def test_skips_lines_with_missing_clips(
        self, populated_ledger, fake_procgen, tmp_path
    ):
        """If a line's HuMo clip isn't on disk, skip it without
        raising.  Crash only if NO clips survive."""
        clips_dir = tmp_path / "partial"
        clips_dir.mkdir()
        # Only one of the three exists.
        (clips_dir / "char_l001.mp4").write_bytes(b"x")
        out = tmp_path / "out.mp4"
        seen_pillarbox_inputs = []

        def _record_run(cmd, **_kw):
            if "-an" in cmd and "-i" in cmd:
                seen_pillarbox_inputs.append(cmd[cmd.index("-i") + 1])
            if str(cmd[-1]).endswith(".mp4"):
                Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
                Path(cmd[-1]).write_bytes(b"x")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=b"", stderr=b"",
            )

        with patch("subprocess.run", side_effect=_record_run):
            final_path, report = VC._render_master_mix_per_clip_mux_mode(
                ledger=populated_ledger,
                clips_dir=clips_dir,
                procgen=fake_procgen,
                out_mp4=out,
                canvas_w=1920, canvas_h=1080, canvas_fps=25,
                humo_target_h=1080,
                fallback_clip_length=7.0,
                ffmpeg="ffmpeg", ffprobe="ffprobe",
            )

        # Only the one existing clip should have been pillarboxed.
        assert len(seen_pillarbox_inputs) == 1
        assert "char_l001" in seen_pillarbox_inputs[0]

    def test_skips_lines_with_invalid_timing(self, fake_procgen, tmp_path):
        """Lines missing start_s or with non-positive dur_s get
        filtered before pillarbox attempts."""
        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        (clips_dir / "ok.mp4").write_bytes(b"x")
        (clips_dir / "bad_no_start.mp4").write_bytes(b"x")
        (clips_dir / "bad_zero_dur.mp4").write_bytes(b"x")
        led = {
            "lines": [
                {"line_id": "ok",            "start_s": 0.0,  "dur_s": 5.0,
                 "speaker_role": "character"},
                {"line_id": "bad_no_start",                   "dur_s": 5.0},
                {"line_id": "bad_zero_dur",  "start_s": 5.0,  "dur_s": 0.0},
            ],
        }
        out = tmp_path / "out.mp4"
        seen_pillarbox_inputs = []

        def _record_run(cmd, **_kw):
            if "-an" in cmd and "-i" in cmd:
                seen_pillarbox_inputs.append(cmd[cmd.index("-i") + 1])
            if str(cmd[-1]).endswith(".mp4"):
                Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
                Path(cmd[-1]).write_bytes(b"x")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=b"", stderr=b"",
            )

        with patch("subprocess.run", side_effect=_record_run):
            VC._render_master_mix_per_clip_mux_mode(
                ledger=led, clips_dir=clips_dir, procgen=fake_procgen,
                out_mp4=out,
                canvas_w=1920, canvas_h=1080, canvas_fps=25,
                humo_target_h=1080,
                fallback_clip_length=7.0,
                ffmpeg="ffmpeg", ffprobe="ffprobe",
            )

        # Only "ok" should have been pillarboxed.
        names = [Path(p).stem for p in seen_pillarbox_inputs]
        assert names == ["ok"]


class TestC7Contract:
    """The whole reason this mode exists: ZERO audio re-encodes
    downstream of SignalLostVideo.  Verify there is NO `-c:a aac`
    or `-b:a` (audio bitrate, implies re-encode) anywhere in any
    ffmpeg invocation in the per_clip_mux pipeline."""

    def test_no_audio_re_encode_anywhere(
        self, populated_ledger, populated_clips_dir, fake_procgen, tmp_path
    ):
        out = tmp_path / "out.mp4"
        all_cmds = []

        def _record_run(cmd, **_kw):
            all_cmds.append(list(cmd))
            if str(cmd[-1]).endswith(".mp4"):
                Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
                Path(cmd[-1]).write_bytes(b"x")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=b"", stderr=b"",
            )

        with patch("subprocess.run", side_effect=_record_run):
            VC._render_master_mix_per_clip_mux_mode(
                ledger=populated_ledger,
                clips_dir=populated_clips_dir,
                procgen=fake_procgen,
                out_mp4=out,
                canvas_w=1920, canvas_h=1080, canvas_fps=25,
                humo_target_h=1080,
                fallback_clip_length=7.0,
                ffmpeg="ffmpeg", ffprobe="ffprobe",
            )

        for cmd in all_cmds:
            # No AAC re-encode.
            for tok in ("aac",):
                # Acceptable: -c:a copy, -an, -map 0:a etc -- those
                # don't contain "aac".  We forbid the literal token.
                assert tok not in cmd, (
                    f"Found AAC re-encode token {tok!r} in ffmpeg cmd: "
                    f"{cmd}"
                )
            # No audio bitrate flag (implies re-encode).
            assert "-b:a" not in cmd, (
                f"Found -b:a (audio bitrate) in ffmpeg cmd, implies "
                f"re-encode: {cmd}"
            )
            # No audio sample rate flag (implies re-encode).
            assert "-ar" not in cmd, (
                f"Found -ar (sample rate) in ffmpeg cmd, implies "
                f"re-encode: {cmd}"
            )
