"""Optional green audio-reactive bottom-bars overlay (CODER_KICKOFF_BARS_OVERLAY).

CPU coverage for the additive landscape_bars feature on OTR_SceneAwareScopes:
  - off (DEFAULT) == the pre-overlay behavior (landscape clips suppress -> None);
  - bottom -> a real landscape clip gets the ('bars', ...) bottom-strip mode,
    while portrait gutters / gap centres / un-probeable suppression are UNCHANGED;
  - the strip sits ABOVE the caption safe-area (captions never occluded);
  - the shared freq-bar routine is GREEN-ONLY (no amber/red).

No GPU, no ffmpeg -- the plan + geometry + draw are pure.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from nodes import otr_scene_aware_scopes as sc
from nodes._otr_shared import scope_draw as sd


def _manifest():
    # Two back-to-back beats: a landscape clip then a portrait clip (sequential
    # mode -- no start_s). total_target_frames == the concat length (no tail).
    return {
        "fps": 25, "total_target_frames": 20,
        "clips": [
            {"shot_id": "b001", "engine_id": "ltx_video", "path": "land.mp4",
             "exists": True, "target_frame_count": 10, "start_s": None},
            {"shot_id": "b002", "engine_id": "humo", "path": "port.mp4",
             "exists": True, "target_frame_count": 10, "start_s": None},
        ],
    }


def _patch_probe(monkeypatch, land=False, port=True):
    aspect = {"land.mp4": land, "port.mp4": port}

    def fake(path, ffprobe, cache):
        return aspect.get(path)
    monkeypatch.setattr(sc, "_probe_is_portrait", fake)


def test_bars_off_landscape_suppressed(monkeypatch):
    # DEFAULT off == today: a landscape clip draws nothing (None), portrait gutters.
    _patch_probe(monkeypatch)
    plan, total = sc.plan_scope_frames(_manifest(), 1920, 1080,
                                       landscape_bars="off")
    assert total == 20
    assert plan[0] is None                    # landscape clip frame -> suppressed
    assert plan[12][0] == "gutters"           # portrait clip frame -> gutters


def test_bars_bottom_landscape_gets_bars(monkeypatch):
    _patch_probe(monkeypatch)
    plan, total = sc.plan_scope_frames(_manifest(), 1920, 1080,
                                       landscape_bars="bottom")
    assert plan[0][0] == "bars"               # landscape clip -> bottom strip
    assert plan[5][0] == "bars"
    assert plan[12][0] == "gutters"           # portrait UNCHANGED


def test_bars_default_param_matches_off(monkeypatch):
    # The default value of landscape_bars is 'off' (regression-lock).
    _patch_probe(monkeypatch)
    plan_default, _ = sc.plan_scope_frames(_manifest(), 1920, 1080)
    plan_off, _ = sc.plan_scope_frames(_manifest(), 1920, 1080,
                                       landscape_bars="off")
    assert plan_default == plan_off


def test_bars_unprobeable_still_suppressed(monkeypatch):
    # An un-probeable clip (aspect None) suppresses even with bars on (unchanged).
    monkeypatch.setattr(sc, "_probe_is_portrait",
                        lambda path, ffprobe, cache: None)
    plan, _ = sc.plan_scope_frames(_manifest(), 1920, 1080,
                                   landscape_bars="bottom")
    assert plan[0] is None


def test_bars_geom_above_caption_safe_area():
    # The strip must end ABOVE the lower ~15% reserved for SDH captions.
    x, y, w, h = sc._bars_geom(1920, 1080)
    assert y + h <= int(1080 * (1.0 - sc._BARS_CAPTION_SAFE_FRAC))
    assert x > 0 and w > 0 and h > 0 and y > 0


def test_freq_bars_green_is_green_only():
    img = Image.new("RGB", (320, 80), (0, 0, 0))
    d = ImageDraw.Draw(img)
    sd.freq_bars_green(d, np.ones(32, dtype=np.float32), 0, 0, 320, 80)
    arr = np.asarray(img)
    assert arr[..., 0].max() == 0             # GREEN-ONLY: no red/amber channel
    assert arr[..., 1].max() > 0              # some green actually drawn


def test_freq_bars_green_empty_is_noop():
    img = Image.new("RGB", (320, 80), (0, 0, 0))
    d = ImageDraw.Draw(img)
    sd.freq_bars_green(d, np.zeros(0, dtype=np.float32), 0, 0, 320, 80)
    assert np.asarray(img).max() == 0         # nothing to draw -> black
