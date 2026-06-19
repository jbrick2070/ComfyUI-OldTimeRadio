"""CLIP-FILL / dynamic-VRAM frame budget (GO_FORWARD 2026-06-18).

CPU coverage for the five clip-fill pieces:
  1. motion_common.compute_real_frame_budget -- PREDICT the VRAM-affordable frame
     count (never react-to-OOM); free_vram_mb() the zero-cost probe.
  3. wrapper_bridge.extend_frames_to_target -- ping-pong/mirror-extend a short
     render up to the beat target (seamless, no hard loop seam).
  4. render_driver.persist_episode_clips -- move clips out of the swept _shared/tmp
     scratch tier into the durable episodes/<ep>/clips/ workspace.
  5. otr_silent_composite._warn_clip_underrun -- LOUD-warn (never raise) when a
     real clip is far shorter than its beat target.

No GPU, no model load. (wan_ti2v's _floor_length wiring is covered in test_wan_ti2v.)
"""
from __future__ import annotations

import logging
import os

from nodes._otr_video_engines import motion_common as mc
from nodes._otr_video_engines import wrapper_bridge as wb


# --------------------------------------------------------------------------- #
# Piece 1 -- compute_real_frame_budget + free_vram_mb
# --------------------------------------------------------------------------- #
def _clear_cost_env(monkeypatch):
    for k in ("OTR_VIDEO_COST_OVERHEAD_MB", "OTR_VIDEO_COST_PER_FRAME_MB",
              "OTR_VIDEO_BUDGET_MARGIN", "OTR_VRAM_CEILING_MB"):
        monkeypatch.delenv(k, raising=False)


def test_budget_none_free_trusts_target(monkeypatch):
    # No live VRAM read (CPU box) -> trust the audio-derived target, snapped to a
    # valid 4n+1 <= target (the render-window NVML probe still guards at render).
    _clear_cost_env(monkeypatch)
    assert mc.compute_real_frame_budget(None, 280, 1472, 832, "wan_ti2v") == 277
    assert mc.compute_real_frame_budget(0, 33, 1472, 832, "wan_ti2v") == 33


def test_budget_predicts_fewer_frames_under_pressure(monkeypatch):
    # overhead 7000 + 185/frame @1472x832; budget = min(free,14500)*0.85.
    # free 14775 -> 12325 budget -> (12325-7000)/185 = 28.7 -> 28 -> 4n+1 snap 29.
    _clear_cost_env(monkeypatch)
    assert mc.compute_real_frame_budget(14775.0, 280, 1472, 832, "wan_ti2v") == 29


def test_budget_motion_floor_wins_when_starved(monkeypatch):
    # 8 GB free cannot fit a frame past the 7000 MB overhead -> the 17-frame motion
    # floor wins (never 0 -- a beat always carries motion).
    _clear_cost_env(monkeypatch)
    assert mc.compute_real_frame_budget(8000.0, 280, 1472, 832, "wan_ti2v") == 17


def test_budget_never_exceeds_target(monkeypatch):
    # Abundant VRAM never renders MORE than the beat needs.
    _clear_cost_env(monkeypatch)
    out = mc.compute_real_frame_budget(60000.0, 49, 1472, 832, "wan_ti2v")
    assert out <= 49 and (out - 1) % 4 == 0


def test_budget_scales_cost_with_pixel_area(monkeypatch):
    # A smaller canvas affords MORE frames (per-frame cost scales with pixel area).
    _clear_cost_env(monkeypatch)
    big = mc.compute_real_frame_budget(14775.0, 280, 1472, 832, "wan_ti2v")
    small = mc.compute_real_frame_budget(14775.0, 280, 832, 480, "wan_ti2v")
    assert small > big


def test_budget_env_overrides_cost_model(monkeypatch):
    _clear_cost_env(monkeypatch)
    monkeypatch.setenv("OTR_VIDEO_COST_OVERHEAD_MB", "1000")
    monkeypatch.setenv("OTR_VIDEO_COST_PER_FRAME_MB", "10")
    # (min(14775,14500)*0.85 - 1000)/10 = (12325-1000)/10 = 1132 -> capped at target.
    assert mc.compute_real_frame_budget(14775.0, 81, 1472, 832, "wan_ti2v") == 81


def test_free_vram_mb_is_none_or_positive():
    # The zero-cost probe returns None off-GPU (CI CPU box) or a positive float on
    # a live-GPU box (the headless suite box) -- never a crash, never <= 0.
    free = mc.free_vram_mb()
    assert free is None or (isinstance(free, float) and free > 0)


def test_budget_exposed_on_motion_base():
    assert mc.MotionEngineBase.compute_real_frame_budget(
        None, 33, 832, 480, "wan_ti2v") == 33


# --------------------------------------------------------------------------- #
# Piece 3 -- extend_frames_to_target (ping-pong)
# --------------------------------------------------------------------------- #
def _frames(n):
    import numpy as np
    # n distinct 1x1x3 frames so identity is checkable by value.
    return np.stack([np.full((1, 1, 3), i, dtype=np.uint8) for i in range(n)])


def test_extend_noop_when_already_long_enough():
    import numpy as np
    f = _frames(10)
    out = wb.extend_frames_to_target(f, 5)
    assert np.array_equal(out, f)            # target <= n -> unchanged
    out2 = wb.extend_frames_to_target(f, 10)
    assert np.array_equal(out2, f)


def test_extend_reaches_exact_target():
    out = wb.extend_frames_to_target(_frames(7), 280)
    assert len(out) == 280


def test_extend_is_seamless_pingpong():
    # cycle [0,1,2,3,2,1] (period 2N-2=6) tiled -> indices ping-pong, no jump > 1.
    import numpy as np
    out = wb.extend_frames_to_target(_frames(4), 20)
    vals = [int(x.flat[0]) for x in out]
    assert len(out) == 20
    assert vals[:6] == [0, 1, 2, 3, 2, 1]
    assert max(abs(a - b) for a, b in zip(vals, vals[1:])) == 1   # seamless
    # the join wraps cleanly (…1 -> 0 -> 1…), still a step of 1.
    assert vals[5] == 1 and vals[6] == 0


def test_extend_single_frame_repeats():
    out = wb.extend_frames_to_target(_frames(1), 5)
    assert len(out) == 5                     # a still cannot mirror -> repeated


# --------------------------------------------------------------------------- #
# Piece 4 -- persist_episode_clips
# --------------------------------------------------------------------------- #
def test_persist_moves_clip_to_episode_clips_dir(monkeypatch, tmp_path):
    from nodes._otr_video_engines import render_driver as rd
    import nodes._otr_paths as paths

    src_dir = tmp_path / "scratch"
    src_dir.mkdir()
    src = src_dir / "otr_wan_ti2v_abc.mp4"
    src.write_bytes(b"fake-mp4-bytes")
    clips_dir = tmp_path / "ep" / "clips"
    monkeypatch.setattr(paths, "otr_clips_dir", lambda eid: clips_dir)

    result = {
        "clips": {"shot_b001": {"type": "video", "path": str(src),
                                 "engine_id": "wan_ti2v"}},
        "ledger": {"video": {"shots": [
            {"shot_id": "shot_b001", "role": "scene_broll",
             "engine_id": "wan_ti2v"}]}},
    }
    rd.persist_episode_clips(result, "ep")
    new_path = result["clips"]["shot_b001"]["path"]
    assert os.path.dirname(new_path) == str(clips_dir)
    assert os.path.isfile(new_path)
    assert not src.exists()                  # moved, not copied
    assert "scene_broll" in os.path.basename(new_path)
    assert "wan_ti2v" in os.path.basename(new_path)


def test_persist_noop_without_episode_id(tmp_path):
    from nodes._otr_video_engines import render_driver as rd
    src = tmp_path / "x.mp4"
    src.write_bytes(b"y")
    result = {"clips": {"s": {"type": "video", "path": str(src)}}}
    rd.persist_episode_clips(result, "")     # no episode id -> untouched
    assert result["clips"]["s"]["path"] == str(src)
    assert src.exists()


def test_persist_skips_directory_clips(monkeypatch, tmp_path):
    from nodes._otr_video_engines import render_driver as rd
    import nodes._otr_paths as paths
    monkeypatch.setattr(paths, "otr_clips_dir", lambda eid: tmp_path / "clips")
    result = {"clips": {"s": {"type": "directory", "path": str(tmp_path)}},
              "ledger": {"video": {"shots": []}}}
    rd.persist_episode_clips(result, "ep")
    assert result["clips"]["s"]["path"] == str(tmp_path)   # dir clip untouched


# --------------------------------------------------------------------------- #
# Piece 5 -- composite underrun guard
# --------------------------------------------------------------------------- #
def _manifest(rows, fps=25):
    return {"fps": fps, "clips": rows}


def test_underrun_warns_on_short_clip(monkeypatch, caplog):
    from nodes import otr_silent_composite as sc
    monkeypatch.delenv("OTR_CLIP_UNDERRUN_FRAC", raising=False)
    rows = [{"shot_id": "shot_b001", "engine_id": "wan_ti2v", "path": "x.mp4",
             "exists": True, "frame_count": 17, "target_frame_count": 280,
             "start_s": None}]
    with caplog.at_level(logging.WARNING, logger="OTR"):
        segs, total = sc.plan_timeline_segments(_manifest(rows))
    assert any("CLIP UNDERRUN" in r.message for r in caplog.records)
    assert total == 280                      # the segment still spans the beat


def test_no_underrun_warn_when_clip_fills_target(monkeypatch, caplog):
    from nodes import otr_silent_composite as sc
    monkeypatch.delenv("OTR_CLIP_UNDERRUN_FRAC", raising=False)
    rows = [{"shot_id": "shot_b001", "engine_id": "wan_ti2v", "path": "x.mp4",
             "exists": True, "frame_count": 277, "target_frame_count": 280,
             "start_s": None}]
    with caplog.at_level(logging.WARNING, logger="OTR"):
        sc.plan_timeline_segments(_manifest(rows))
    assert not any("CLIP UNDERRUN" in r.message for r in caplog.records)


def test_underrun_guard_disabled_by_env(monkeypatch, caplog):
    from nodes import otr_silent_composite as sc
    monkeypatch.setenv("OTR_CLIP_UNDERRUN_FRAC", "0")
    rows = [{"shot_id": "shot_b001", "engine_id": "wan_ti2v", "path": "x.mp4",
             "exists": True, "frame_count": 1, "target_frame_count": 280,
             "start_s": None}]
    with caplog.at_level(logging.WARNING, logger="OTR"):
        sc.plan_timeline_segments(_manifest(rows))
    assert not any("CLIP UNDERRUN" in r.message for r in caplog.records)
