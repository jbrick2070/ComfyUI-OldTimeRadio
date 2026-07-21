"""CW-4 -- OTR_CaptionBurn: the SDH open-caption burn MIGRATED off the legacy
blend node into the new render path. CPU tests.

Proves subtitles survive the legacy teardown: default-OFF passthrough (clean
master), the .ass builds from a timed ledger via the SURVIVING _otr_captions
builder, and (libass-gated) a real burn produces a still-SILENT video. The audio
spine is never touched (audio is added downstream by MasterAudioMux).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import pathlib

import pytest

from nodes.otr_caption_burn import (
    OTRCaptionBurn, burn_captions_on_video, _ass_filter_arg, _DEFAULT_CAPTION_STYLE,
)

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _has_libass() -> bool:
    if not HAVE_FFMPEG:
        return False
    p = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return " ass " in (p.stdout or "") or " subtitles " in (p.stdout or "")


needs_libass = pytest.mark.skipif(not _has_libass(),
                                  reason="ffmpeg libass (ass filter) required")


def _timed_ledger(path):
    led = {
        "episode_id": "testep",
        "cast": [{"char_id": "c1", "name": "BABA"}],
        "lines": [
            {"line_id": "l1", "char_id": "c1", "speaker_role": "character",
             "text": "we have to leave now", "start_s": 0.0, "dur_s": 2.0},
            {"line_id": "l2", "char_id": "c1", "speaker_role": "character",
             "text": "the station is failing", "start_s": 2.0, "dur_s": 2.0},
        ],
    }
    pathlib.Path(path).write_text(json.dumps(led), encoding="utf-8")
    return str(path)


def test_caption_burn_default_off_passthrough():
    # default OFF -> input passes through unchanged, NO ffmpeg, video need not exist
    out, report = OTRCaptionBurn().burn("C:/somewhere/ep_silent.mp4", burn_captions=False)
    assert out == "C:/somewhere/ep_silent.mp4"
    assert "passthrough" in report.lower()


def test_caption_burn_env_does_not_enable(monkeypatch, tmp_path):
    # 2026-07-04 widget-audit Batch 3: the OTR_BURN_CAPTIONS env-only enable path
    # was CUT -- captions are a widget + capability-profile on/off feature only.
    # With the widget OFF, the env must NOT force captions on: it stays a
    # clean-master passthrough (no ffmpeg, no ledger resolution attempted).
    monkeypatch.setenv("OTR_BURN_CAPTIONS", "1")
    vid = tmp_path / "ep_silent.mp4"
    vid.write_bytes(b"\x00\x00")            # not a real mp4
    out, report = OTRCaptionBurn().burn(str(vid), burn_captions=False, ledger_path="")
    assert out == str(vid)
    # "clean master" proves the env did NOT force the burn on (an env-forced run
    # would instead report "no timed ledger").
    assert "clean master" in report.lower()
    assert "passthrough" in report.lower()


def test_caption_burn_widgets_default_off():
    it = OTRCaptionBurn.INPUT_TYPES()
    assert "video_path" in it["required"]
    assert it["optional"]["burn_captions"][1]["default"] is False   # clean master default
    assert "caption_style" in it["optional"]
    assert OTRCaptionBurn.OUTPUT_NODE is False


def test_ass_filter_arg_basename_and_cwd(tmp_path):
    ass = tmp_path / "ep_captions.ass"
    name, cwd = _ass_filter_arg(str(ass))
    assert name == "ep_captions.ass" and cwd == str(tmp_path)


def test_ass_builds_from_timed_ledger(tmp_path):
    """The migration reuses the SURVIVING _otr_captions builder -> a real .ass."""
    from nodes._otr_captions import build_ass_from_ledger
    led = _timed_ledger(tmp_path / "testep_ledger.json")
    out, report = build_ass_from_ledger(led, style=_DEFAULT_CAPTION_STYLE)
    assert out and pathlib.Path(out).is_file()
    text = pathlib.Path(out).read_text(encoding="utf-8")
    assert "[Events]" in text and "Dialogue:" in text and "leave now" in text


def test_ass_uses_alias_aware_speaker_and_excludes_skipped_rows_from_clamp(tmp_path):
    """A content-owned announcer sentinel resolves through the canonical cast,
    while a muted row can neither render nor shorten the preceding cue."""
    from nodes._otr_captions import build_ass_from_ledger

    path = tmp_path / "alias_ledger.json"
    path.write_text(json.dumps({
        "episode_id": "alias",
        "cast": [
            {"char_id": "c01", "name": "ANNOUNCER"},
            {"char_id": "c02", "name": "MARA"},
        ],
        "lines": [
            {"line_id": "b001", "char_id": "announcer",
             "speaker_role": "announcer", "text": "The signal holds.",
             "start_s": 0.0, "dur_s": 4.0},
            {"line_id": "b002", "char_id": "c02",
             "speaker_role": "character", "text": "MUTED ROW MUST NOT SHIP",
             "start_s": 1.0, "dur_s": 1.0, "skip": True},
            {"line_id": "b003", "char_id": "c02",
             "speaker_role": "character", "text": "I can hear it.",
             "start_s": 4.0, "dur_s": 2.0},
        ],
    }), encoding="utf-8")

    out, report = build_ass_from_ledger(path, style=_DEFAULT_CAPTION_STYLE)
    assert out, report
    ass = pathlib.Path(out).read_text(encoding="utf-8")
    assert "ANNOUNCER:" in ass
    assert "MUTED ROW MUST NOT SHIP" not in ass
    assert "Dialogue: 0,0:00:00.00,0:00:04.00" in ass
    assert "from 2 speech lines" in report


def test_caption_burn_no_shortest_in_source():
    src = (pathlib.Path(__file__).resolve().parent.parent
           / "nodes" / "otr_caption_burn.py").read_text(encoding="utf-8")
    # the only '-shortest' token is the V-2 guard assert, never a cmd arg
    assert src.count('"-shortest"') == 1
    assert 'assert "-shortest" not in cmd' in src


@needs_libass
def test_caption_burn_real_keeps_video_silent(tmp_path):
    """Operator/CI with libass: a real burn produces a captioned video that is
    STILL SILENT (audio added later by MasterAudioMux)."""
    from nodes.otr_silent_composite import count_audio_streams
    vid = tmp_path / "testep_silent.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "color=c=black:s=320x240:d=4", "-r", "25", "-pix_fmt",
                    "yuv420p", "-an", str(vid)], check=True, capture_output=True)
    led = _timed_ledger(tmp_path / "testep_ledger.json")
    out = tmp_path / "testep_captioned.mp4"
    final, report = burn_captions_on_video(str(vid), led, str(out), fps=25)
    assert pathlib.Path(final).is_file()
    assert count_audio_streams(final) == 0   # V-1: still silent after the burn
