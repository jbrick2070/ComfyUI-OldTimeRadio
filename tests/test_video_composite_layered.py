"""BUG-LOCAL-030 regression: layered per-clip composite (1472x832 canvas).

Tests the new helpers added 2026-05-03 EVENING per Jeffrey's spec:
  - _resolve_episode_background: picks FLUX env still or radio bookend
    from the per-episode workspace.
  - _layered_per_clip_silent: builds an ffmpeg cmd that either
    (a) overlays a HuMo center pillar onto a scene-still backdrop for
        character clips, OR
    (b) scale-crops a non-character clip to fill the full canvas.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OTR_TEST_MODE", "1")


def _import_vc():
    """Late-import video_composite to honor conftest CUDA mask."""
    from nodes import video_composite as VC  # noqa: WPS433

    return VC


# ---------------------------------------------------------------------------
# (1) _resolve_episode_background
# ---------------------------------------------------------------------------

def test_resolve_picks_freshest_env_still(tmp_path):
    VC = _import_vc()
    ep = tmp_path / "ep_a"
    stills = ep / "stills"
    stills.mkdir(parents=True)
    older = stills / "full_env_00001_.png"
    newer = stills / "full_env_00002_.png"
    older.write_bytes(b"old")
    newer.write_bytes(b"new")
    # Force mtimes so "newer" really is newer.
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))
    pick = VC._resolve_episode_background(ep)
    assert pick == newer, f"expected freshest env still, got {pick}"


def test_resolve_falls_back_to_radio_bookend_if_no_env_still(tmp_path):
    VC = _import_vc()
    ep = tmp_path / "ep_b"
    stills = ep / "stills"
    stills.mkdir(parents=True)
    bookend = stills / "radio_bookend_signal_lost_test.png"
    bookend.write_bytes(b"radio")
    pick = VC._resolve_episode_background(ep)
    assert pick == bookend


def test_resolve_returns_none_when_no_stills(tmp_path):
    VC = _import_vc()
    ep = tmp_path / "ep_c"
    ep.mkdir()
    # No stills/ subdir at all.
    assert VC._resolve_episode_background(ep) is None


def test_resolve_returns_none_when_stills_empty(tmp_path):
    VC = _import_vc()
    ep = tmp_path / "ep_d"
    (ep / "stills").mkdir(parents=True)
    assert VC._resolve_episode_background(ep) is None


# ---------------------------------------------------------------------------
# (2) _layered_per_clip_silent — ffmpeg cmd shape
# ---------------------------------------------------------------------------

def _capture_cmd():
    """Mock subprocess.run, return the captured cmd list."""
    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs

        class _R:
            returncode = 0

        return _R()

    return captured, _fake_run


def test_character_clip_with_background_uses_filter_complex(tmp_path):
    VC = _import_vc()
    clip = tmp_path / "l003.mp4"
    clip.write_bytes(b"clip")
    bg = tmp_path / "full_env_00001_.png"
    bg.write_bytes(b"bg")
    out = tmp_path / "pb_l003.mp4"
    captured, fake_run = _capture_cmd()
    with mock.patch.object(VC.subprocess, "run", side_effect=fake_run):
        VC._layered_per_clip_silent(
            clip=clip, speaker_role="character",
            canvas_w=1472, canvas_h=832, canvas_fps=25,
            humo_pillar_w=512, humo_pillar_h=832,
            background_png=bg,
            out_path=out, ffmpeg="ffmpeg",
        )
    cmd = captured["cmd"]
    # Layered path uses -filter_complex
    assert "-filter_complex" in cmd, f"layered path must use -filter_complex; got {cmd}"
    fc_idx = cmd.index("-filter_complex")
    fc = cmd[fc_idx + 1]
    assert "[0:v]" in fc and "[1:v]" in fc, f"layered must reference both inputs; got {fc}"
    assert "scale=1472:832" in fc, f"backdrop must scale-cover canvas: {fc}"
    assert "crop=1472:832" in fc, f"backdrop must crop to canvas: {fc}"
    assert "crop=512:832" in fc, f"pillar must center-crop to 512x832: {fc}"
    assert "overlay=x=(W-w)/2:y=0" in fc, f"pillar must center-overlay: {fc}"
    # Background loop input is present
    assert str(bg) in cmd, f"background image must be a -i input: {cmd}"
    assert "-loop" in cmd
    # Audio dropped
    assert "-an" in cmd
    # Output mapped from filter
    assert "-map" in cmd
    map_idx = cmd.index("-map")
    assert cmd[map_idx + 1] == "[v]"


def test_announcer_clip_uses_simple_scale_crop(tmp_path):
    VC = _import_vc()
    clip = tmp_path / "l001.mp4"
    clip.write_bytes(b"clip")
    out = tmp_path / "pb_l001.mp4"
    captured, fake_run = _capture_cmd()
    with mock.patch.object(VC.subprocess, "run", side_effect=fake_run):
        VC._layered_per_clip_silent(
            clip=clip, speaker_role="announcer",
            canvas_w=1472, canvas_h=832, canvas_fps=25,
            humo_pillar_w=512, humo_pillar_h=832,
            background_png=None,
            out_path=out, ffmpeg="ffmpeg",
        )
    cmd = captured["cmd"]
    # Single-source fill path uses -vf, NOT -filter_complex
    assert "-vf" in cmd, f"non-character path uses -vf; got {cmd}"
    assert "-filter_complex" not in cmd, "non-character must not use -filter_complex"
    vf_idx = cmd.index("-vf")
    vf = cmd[vf_idx + 1]
    assert "scale=1472:832:force_original_aspect_ratio=increase" in vf, vf
    assert "crop=1472:832" in vf, vf
    assert "-an" in cmd


def test_character_with_no_background_falls_back_to_simple_fill(tmp_path):
    """If a character clip has no backdrop available, gracefully degrade
    to the simple canvas-fill so the timeline still renders."""
    VC = _import_vc()
    clip = tmp_path / "l003.mp4"
    clip.write_bytes(b"clip")
    out = tmp_path / "pb_l003.mp4"
    captured, fake_run = _capture_cmd()
    with mock.patch.object(VC.subprocess, "run", side_effect=fake_run):
        VC._layered_per_clip_silent(
            clip=clip, speaker_role="character",
            canvas_w=1472, canvas_h=832, canvas_fps=25,
            humo_pillar_w=512, humo_pillar_h=832,
            background_png=None,
            out_path=out, ffmpeg="ffmpeg",
        )
    cmd = captured["cmd"]
    assert "-filter_complex" not in cmd, (
        "character + no bg must fall back to simple -vf fill"
    )
    assert "-vf" in cmd


def test_extend_tail_appended_to_layered_chain(tmp_path):
    VC = _import_vc()
    clip = tmp_path / "l_last.mp4"
    clip.write_bytes(b"clip")
    bg = tmp_path / "bg.png"
    bg.write_bytes(b"bg")
    out = tmp_path / "pb_l_last.mp4"
    captured, fake_run = _capture_cmd()
    with mock.patch.object(VC.subprocess, "run", side_effect=fake_run):
        VC._layered_per_clip_silent(
            clip=clip, speaker_role="character",
            canvas_w=1472, canvas_h=832, canvas_fps=25,
            humo_pillar_w=512, humo_pillar_h=832,
            background_png=bg,
            out_path=out, ffmpeg="ffmpeg",
            extend_tail_s=0.5,
        )
    cmd = captured["cmd"]
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "tpad=stop_mode=clone:stop_duration=0.500" in fc, (
        f"tail-pad must be appended to overlay chain in layered mode: {fc}"
    )


def test_extend_tail_appended_to_simple_chain(tmp_path):
    VC = _import_vc()
    clip = tmp_path / "l_last.mp4"
    clip.write_bytes(b"clip")
    out = tmp_path / "pb_l_last.mp4"
    captured, fake_run = _capture_cmd()
    with mock.patch.object(VC.subprocess, "run", side_effect=fake_run):
        VC._layered_per_clip_silent(
            clip=clip, speaker_role="announcer",
            canvas_w=1472, canvas_h=832, canvas_fps=25,
            humo_pillar_w=512, humo_pillar_h=832,
            background_png=None,
            out_path=out, ffmpeg="ffmpeg",
            extend_tail_s=0.5,
        )
    cmd = captured["cmd"]
    vf = cmd[cmd.index("-vf") + 1]
    assert "tpad=stop_mode=clone:stop_duration=0.500" in vf


# ---------------------------------------------------------------------------
# (3) INPUT_TYPES + module constants
# ---------------------------------------------------------------------------

def test_video_composite_input_types_have_new_defaults():
    VC = _import_vc()
    spec = VC.VideoComposite.INPUT_TYPES()
    opt = spec.get("optional", {})
    assert opt["canvas_width"][1]["default"] == 1472
    assert opt["canvas_height"][1]["default"] == 832
    assert opt["humo_target_height"][1]["default"] == 832
    assert opt["humo_pillar_width"][1]["default"] == 512


def test_humo_render_widget_defaults_are_landscape():
    """BatchHumoRender INPUT_TYPES width=1280, height=720 per BUG-030 spec."""
    src = (_REPO_ROOT / "nodes" / "batch_humo_render.py").read_text(encoding="utf-8")
    import re
    w = re.search(r'"width":\s*\("INT",\s*\{"default":\s*(\d+)', src)
    h = re.search(r'"height":\s*\("INT",\s*\{"default":\s*(\d+)', src)
    assert w and w.group(1) == "1280", f"expected width=1280, got {w.group(1) if w else None}"
    assert h and h.group(1) == "720", f"expected height=720, got {h.group(1) if h else None}"


def test_ltx_render_constants_match_spec():
    """LTX_WIDTH=1216, LTX_HEIGHT=704, LTX_FPS=25 per BUG-030 spec.

    Reads via file parse instead of `import nodes.batch_ltx_render` because
    that module pulls in `folder_paths` (a ComfyUI-runtime-only module not
    available in the bare pytest environment).
    """
    import re
    src = (_REPO_ROOT / "nodes" / "batch_ltx_render.py").read_text(encoding="utf-8")
    w = re.search(r'^LTX_WIDTH\s*=\s*(\d+)', src, re.MULTILINE)
    h = re.search(r'^LTX_HEIGHT\s*=\s*(\d+)', src, re.MULTILINE)
    f = re.search(r'^LTX_FPS\s*=\s*(\d+)', src, re.MULTILINE)
    assert w and w.group(1) == "1216", f"expected LTX_WIDTH=1216, got {w.group(1) if w else None}"
    assert h and h.group(1) == "704", f"expected LTX_HEIGHT=704, got {h.group(1) if h else None}"
    assert f and f.group(1) == "25", f"expected LTX_FPS=25, got {f.group(1) if f else None}"
