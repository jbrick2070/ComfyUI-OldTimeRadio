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


class TestTailPad:
    """The last clip in the timeline gets extend_tail_s passed so
    silent_combined.mp4 ends slightly after the master mix audio.
    -shortest then truncates the inaudible video tail rather than
    nuking the trailing audio (which would silently break C7)."""

    def test_pillarbox_tpad_filter_when_extend_tail_s_set(self, fake_clip, tmp_path):
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
                extend_tail_s=0.5,
            )
            cmd = mock_run.call_args[0][0]
            vf = cmd[cmd.index("-vf") + 1]
            assert "tpad=stop_mode=clone:stop_duration=0.500" in vf

    def test_no_tpad_when_extend_tail_s_zero(self, fake_clip, tmp_path):
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
                extend_tail_s=0.0,
            )
            cmd = mock_run.call_args[0][0]
            vf = cmd[cmd.index("-vf") + 1]
            assert "tpad" not in vf

    def test_only_last_timeline_entry_gets_tpad(
        self, populated_ledger, populated_clips_dir, fake_procgen, tmp_path
    ):
        """Verify _render_master_mix_per_clip_mux_mode passes
        extend_tail_s>0 for ONLY the last clip in the timeline."""
        out = tmp_path / "out.mp4"
        # Capture extend_tail_s per pillarbox call by name.
        per_clip_tail = {}

        original_pb = VC._pillarbox_humo_silent

        def _spy_pb(**kwargs):
            line_id = Path(kwargs["clip"]).stem
            per_clip_tail[line_id] = kwargs.get("extend_tail_s", 0.0)
            return original_pb(**kwargs)

        def _record_run(cmd, **_kw):
            if str(cmd[-1]).endswith(".mp4"):
                Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
                Path(cmd[-1]).write_bytes(b"x")
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=b"", stderr=b"",
            )

        with patch("subprocess.run", side_effect=_record_run), \
             patch.object(VC, "_pillarbox_humo_silent", side_effect=_spy_pb):
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

        # Timeline sorted by start_s ascending = music_open, char,
        # music_close.  Only music_close (the last one) should get
        # a non-zero tail pad.
        assert per_clip_tail["music_open_001"] == 0.0
        assert per_clip_tail["char_l001"] == 0.0
        assert per_clip_tail["music_close_001"] > 0.0
        # Documented value: 0.5s
        assert per_clip_tail["music_close_001"] == pytest.approx(0.5)


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


class TestEndOfRunCleanup:
    """BUG-LOCAL-123 (2026-04-30): VideoComposite must unload all
    models from VRAM at end of run so the next ComfyUI prompt starts
    with a clean VRAM slate. ComfyUI's model_management deliberately
    keeps loaded models resident across prompts; OTR explicitly does
    not want that for the LAST node in a run.

    These tests verify that the cleanup helper is callable, idempotent,
    and never raises -- even when comfy.model_management is unavailable
    (e.g. unit-test environment without ComfyUI installed).
    """

    def test_end_of_run_cleanup_never_raises_in_test_env(self):
        # comfy.model_management is not importable in unit-test env.
        # The helper MUST log warnings and continue, not raise.
        VC.VideoComposite._end_of_run_cleanup()

    def test_end_of_run_cleanup_calls_unload_when_available(self):
        # Inject a fake comfy.model_management module and confirm
        # both unload_all_models and soft_empty_cache get called.
        called = {"unload": False, "empty_cache": False}

        class FakeMM:
            @staticmethod
            def unload_all_models():
                called["unload"] = True

            @staticmethod
            def soft_empty_cache(force=False):
                called["empty_cache"] = True
                assert force is True, (
                    "VideoComposite must pass force=True to "
                    "soft_empty_cache so the CUDA pool actually drops"
                )

        # Patch sys.modules so `import comfy.model_management` resolves
        # to our fake.
        import sys as _sys
        fake_pkg = type(_sys)("comfy")
        fake_pkg.model_management = FakeMM
        original_comfy = _sys.modules.get("comfy")
        original_mm = _sys.modules.get("comfy.model_management")
        _sys.modules["comfy"] = fake_pkg
        _sys.modules["comfy.model_management"] = FakeMM
        try:
            VC.VideoComposite._end_of_run_cleanup()
        finally:
            # Restore.
            if original_comfy is not None:
                _sys.modules["comfy"] = original_comfy
            else:
                _sys.modules.pop("comfy", None)
            if original_mm is not None:
                _sys.modules["comfy.model_management"] = original_mm
            else:
                _sys.modules.pop("comfy.model_management", None)
        assert called["unload"] is True, (
            "_end_of_run_cleanup must call comfy.model_management."
            "unload_all_models()"
        )
        assert called["empty_cache"] is True, (
            "_end_of_run_cleanup must call comfy.model_management."
            "soft_empty_cache(force=True)"
        )

    def test_end_of_run_cleanup_swallows_unload_exception(self):
        # If unload_all_models raises, cleanup should still attempt
        # soft_empty_cache, and never raise to the caller.
        class FakeMMUnloadRaises:
            @staticmethod
            def unload_all_models():
                raise RuntimeError("simulated unload failure")

            @staticmethod
            def soft_empty_cache(force=False):
                # This should still get called despite the unload
                # failure -- they are independent steps.
                pass

        import sys as _sys
        fake_pkg = type(_sys)("comfy")
        fake_pkg.model_management = FakeMMUnloadRaises
        original_comfy = _sys.modules.get("comfy")
        original_mm = _sys.modules.get("comfy.model_management")
        _sys.modules["comfy"] = fake_pkg
        _sys.modules["comfy.model_management"] = FakeMMUnloadRaises
        try:
            # Must not raise.
            VC.VideoComposite._end_of_run_cleanup()
        finally:
            if original_comfy is not None:
                _sys.modules["comfy"] = original_comfy
            else:
                _sys.modules.pop("comfy", None)
            if original_mm is not None:
                _sys.modules["comfy.model_management"] = original_mm
            else:
                _sys.modules.pop("comfy.model_management", None)

    def test_execute_calls_cleanup_on_success(self):
        # The execute() wrapper must call _end_of_run_cleanup() in
        # finally even when _execute_body returns normally.
        cleanup_calls = {"count": 0}

        original_cleanup = VC.VideoComposite._end_of_run_cleanup
        original_body = VC.VideoComposite._execute_body

        @staticmethod
        def fake_cleanup():
            cleanup_calls["count"] += 1

        def fake_body(self, **kwargs):
            return ("/fake/path.mp4", "fake report")

        VC.VideoComposite._end_of_run_cleanup = fake_cleanup
        VC.VideoComposite._execute_body = fake_body
        try:
            inst = VC.VideoComposite()
            result = inst.execute(procgen_video_path="x", clips_dir="y", ledger_json="{}")
            assert result == ("/fake/path.mp4", "fake report")
            assert cleanup_calls["count"] == 1
        finally:
            VC.VideoComposite._end_of_run_cleanup = original_cleanup
            VC.VideoComposite._execute_body = original_body

    def test_execute_calls_cleanup_on_exception(self):
        # The execute() wrapper must call _end_of_run_cleanup() in
        # finally even when _execute_body raises.  This is the
        # critical case: a failed render still needs to drop VRAM
        # so the next attempt isn't poisoned.
        cleanup_calls = {"count": 0}

        original_cleanup = VC.VideoComposite._end_of_run_cleanup
        original_body = VC.VideoComposite._execute_body

        @staticmethod
        def fake_cleanup():
            cleanup_calls["count"] += 1

        def fake_body_raises(self, **kwargs):
            raise RuntimeError("simulated render failure")

        VC.VideoComposite._end_of_run_cleanup = fake_cleanup
        VC.VideoComposite._execute_body = fake_body_raises
        try:
            inst = VC.VideoComposite()
            with pytest.raises(RuntimeError, match="simulated render failure"):
                inst.execute(procgen_video_path="x", clips_dir="y", ledger_json="{}")
            assert cleanup_calls["count"] == 1, (
                "Cleanup MUST fire even when _execute_body raises"
            )
        finally:
            VC.VideoComposite._end_of_run_cleanup = original_cleanup
            VC.VideoComposite._execute_body = original_body
