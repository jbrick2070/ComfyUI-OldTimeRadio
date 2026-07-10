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

import pytest

from nodes._otr_video_engines import motion_common as mc
from nodes._otr_video_engines import wrapper_bridge as wb


# --------------------------------------------------------------------------- #
# Piece 1 -- compute_real_frame_budget + free_vram_mb
# --------------------------------------------------------------------------- #
def _clear_cost_env(monkeypatch):
    for k in ("OTR_VIDEO_COST_OVERHEAD_MB", "OTR_VIDEO_COST_PER_FRAME_MB",
              "OTR_VIDEO_BUDGET_MARGIN"):
        monkeypatch.delenv(k, raising=False)


def test_budget_none_free_trusts_target(monkeypatch):
    # No live VRAM read (CPU box) -> trust the audio-derived target, snapped to a
    # valid 4n+1 <= target (the render-window NVML probe still guards at render).
    _clear_cost_env(monkeypatch)
    assert mc.compute_real_frame_budget(None, 280, 1472, 832, "wan_ti2v") == 277
    assert mc.compute_real_frame_budget(0, 33, 1472, 832, "wan_ti2v") == 33


def test_budget_raises_under_pressure(monkeypatch):
    """S4 platform-portability rewrite (2026-07-10): the frame budget is now a
    STATIC 4n+1-snapped target (geometry only) -- it NEVER shrinks to fit live
    VRAM anymore. When the cost model predicts the snapped target will not
    fit, compute_real_frame_budget RAISES MotionBudgetError instead of
    returning a smaller frame count. Same pressure inputs as the pre-S4 test.
    # overhead 7000 + 185/frame @1472x832; budget = free*0.85 (no policy ceiling
    # post-VRAM-rip -- the operator's tier JSON owns the OOM budget).
    # free 14775 -> 12558.75 budget -> (12558.75-7000)/185 = 30 affordable, which
    # cannot fit the snapped target 277 (4n+1 <= 280) -> raises.
    """
    _clear_cost_env(monkeypatch)
    with pytest.raises(mc.MotionBudgetError):
        mc.compute_real_frame_budget(14775.0, 280, 1472, 832, "wan_ti2v")


def test_budget_raises_when_starved(monkeypatch):
    """S4 rewrite: the old expectation was floor-wins-shrink (the 17-frame
    motion floor). NEW: static budget, raise-never-resize -- starved VRAM
    that cannot even cover the per-engine overhead makes the affordable frame
    count negative, which is always below the snapped target -> raises,
    never silently shrinks to the motion floor.
    # 8 GB free cannot fit a frame past the 7000 MB overhead -> affordable < 0
    # < snapped(277) -> MotionBudgetError.
    """
    _clear_cost_env(monkeypatch)
    with pytest.raises(mc.MotionBudgetError):
        mc.compute_real_frame_budget(8000.0, 280, 1472, 832, "wan_ti2v")


def test_budget_never_exceeds_target(monkeypatch):
    # Abundant VRAM never renders MORE than the beat needs.
    _clear_cost_env(monkeypatch)
    out = mc.compute_real_frame_budget(60000.0, 49, 1472, 832, "wan_ti2v")
    assert out <= 49 and (out - 1) % 4 == 0


def test_budget_scales_cost_with_pixel_area(monkeypatch):
    """S4 rewrite: old expectation was bigger-canvas -> fewer predicted frames
    (a shrink). NEW: the snapped target (277 for target=280 on wan_ti2v) is
    canvas-independent -- per-frame VRAM cost still scales with pixel area, so
    at a free-VRAM level where the smaller canvas's cheaper per-frame cost
    affords the full snapped target, the SAME free level's larger canvas
    (costlier per frame) cannot afford it and raises instead of shrinking.
    # overhead 7000 + 185/frame @1472x832 (big); 185*480/1472=60.33/frame @832x480 (small).
    # free 40000 -> budget 34000: big affordable (34000-7000)/185=145 < 277 -> raises;
    # small affordable (34000-7000)/60.33=447 >= 277 -> returns the snapped target 277.
    """
    _clear_cost_env(monkeypatch)
    small = mc.compute_real_frame_budget(40000.0, 280, 832, 480, "wan_ti2v")
    assert small == 277
    with pytest.raises(mc.MotionBudgetError):
        mc.compute_real_frame_budget(40000.0, 280, 1472, 832, "wan_ti2v")


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
            {"shot_id": "shot_b001", "role": "retired_role_a",
             "engine_id": "wan_ti2v"}]}},
    }
    rd.persist_episode_clips(result, "ep")
    new_path = result["clips"]["shot_b001"]["path"]
    assert os.path.dirname(new_path) == str(clips_dir)
    assert os.path.isfile(new_path)
    assert not src.exists()                  # moved, not copied
    assert "retired_role_a" in os.path.basename(new_path)
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


def test_resolve_stale_pending_clip_episode_to_renamed_dir(monkeypatch, tmp_path):
    from nodes._otr_video_engines import render_driver as rd
    import nodes._otr_paths as paths

    monkeypatch.delenv("OTR_TEST_MODE", raising=False)
    root = tmp_path / "otr" / "episodes"
    final = root / "signal_lost_final"
    audio = final / "audio"
    audio.mkdir(parents=True)
    (audio / "signal_lost_final_ledger.json").write_text(
        '{"episode_id":"signal_lost_final"}', encoding="utf-8")
    monkeypatch.setattr(paths, "otr_episodes_root", lambda: root)

    assert rd.resolve_episode_id_for_clip_persistence(
        "pending_20260708_010101") == "signal_lost_final"


def test_persist_rekeys_sfx_to_renamed_episode_clips(monkeypatch, tmp_path):
    from nodes._otr_video_engines import render_driver as rd
    import nodes._otr_paths as paths

    monkeypatch.delenv("OTR_TEST_MODE", raising=False)
    root = tmp_path / "otr" / "episodes"
    final = root / "signal_lost_final"
    audio = final / "audio"
    audio.mkdir(parents=True)
    (audio / "signal_lost_final_ledger.json").write_text(
        '{"episode_id":"signal_lost_final"}', encoding="utf-8")
    monkeypatch.setattr(paths, "otr_episodes_root", lambda: root)
    monkeypatch.setattr(
        paths, "otr_clips_dir",
        lambda eid: root / eid / "clips")

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    clip_src = scratch / "shot.mp4"
    sfx_src = scratch / "shot.sfx.wav"
    clip_src.write_bytes(b"mp4")
    sfx_src.write_bytes(b"wav")
    result = {
        "clips": {
            "shot_b001": {
                "type": "video",
                "path": str(clip_src),
                "sfx_stem_path": str(sfx_src),
                "engine_id": "google_vid_sfx_veo_fast",
            }
        },
        "ledger": {"video": {"shots": [{
            "shot_id": "shot_b001",
            "role": "character_video",
            "engine_id": "google_vid_sfx_veo_fast",
        }]}},
    }

    rd.persist_episode_clips(result, "pending_20260708_010101")

    clip = result["clips"]["shot_b001"]
    assert str(final / "clips") in clip["path"]
    assert str(final / "clips") in clip["sfx_stem_path"]
    assert (final / "clips").is_dir()
    assert not (root / "pending_20260708_010101").exists()
    assert not clip_src.exists()
    assert not sfx_src.exists()


# --------------------------------------------------------------------------- #
# Piece 5 -- composite underrun guard
# --------------------------------------------------------------------------- #
def _manifest(rows, fps=25):
    return {"fps": fps, "clips": rows}


def test_underrun_loops_and_suppresses_warn(monkeypatch, caplog):
    # S-A: default clip-fill ON -> a short clip LOOPS to fill the beat (never
    # holds the last frame), and the LOUD underrun warning is SILENCED because
    # the fill has handled it.
    from nodes import otr_silent_composite as sc
    monkeypatch.delenv("OTR_CLIP_UNDERRUN_FRAC", raising=False)
    monkeypatch.delenv("OTR_CLIP_FILL", raising=False)
    rows = [{"shot_id": "shot_b001", "engine_id": "wan_ti2v", "path": "x.mp4",
             "exists": True, "frame_count": 17, "target_frame_count": 280,
             "start_s": None}]
    with caplog.at_level(logging.WARNING, logger="OTR"):
        segs, total = sc.plan_timeline_segments(_manifest(rows))
    assert total == 280                      # the segment still spans the beat
    clip_segs = [s for s in segs if s["source"] == "clip"]
    assert clip_segs and all(s["loop"] for s in clip_segs)   # loop-filled
    assert not any("CLIP UNDERRUN" in r.message for r in caplog.records)


def test_underrun_warns_when_fill_disabled(monkeypatch, caplog):
    # OTR_CLIP_FILL=0 restores the legacy held-last-frame behavior -> the LOUD
    # underrun warning fires again and the segment does NOT loop.
    from nodes import otr_silent_composite as sc
    monkeypatch.delenv("OTR_CLIP_UNDERRUN_FRAC", raising=False)
    monkeypatch.setenv("OTR_CLIP_FILL", "0")
    rows = [{"shot_id": "shot_b001", "engine_id": "wan_ti2v", "path": "x.mp4",
             "exists": True, "frame_count": 17, "target_frame_count": 280,
             "start_s": None}]
    with caplog.at_level(logging.WARNING, logger="OTR"):
        segs, total = sc.plan_timeline_segments(_manifest(rows))
    assert any("CLIP UNDERRUN" in r.message for r in caplog.records)
    assert not any(s.get("loop") for s in segs if s["source"] == "clip")


def test_no_underrun_warn_when_clip_fills_target(monkeypatch, caplog):
    # 277/280 is above the 50% underrun threshold -> no LOUD warn either way.
    from nodes import otr_silent_composite as sc
    monkeypatch.delenv("OTR_CLIP_UNDERRUN_FRAC", raising=False)
    monkeypatch.delenv("OTR_CLIP_FILL", raising=False)
    rows = [{"shot_id": "shot_b001", "engine_id": "wan_ti2v", "path": "x.mp4",
             "exists": True, "frame_count": 277, "target_frame_count": 280,
             "start_s": None}]
    with caplog.at_level(logging.WARNING, logger="OTR"):
        sc.plan_timeline_segments(_manifest(rows))
    assert not any("CLIP UNDERRUN" in r.message for r in caplog.records)


def test_underrun_guard_disabled_by_env(monkeypatch, caplog):
    from nodes import otr_silent_composite as sc
    monkeypatch.setenv("OTR_CLIP_UNDERRUN_FRAC", "0")
    monkeypatch.delenv("OTR_CLIP_FILL", raising=False)
    rows = [{"shot_id": "shot_b001", "engine_id": "wan_ti2v", "path": "x.mp4",
             "exists": True, "frame_count": 1, "target_frame_count": 280,
             "start_s": None}]
    with caplog.at_level(logging.WARNING, logger="OTR"):
        sc.plan_timeline_segments(_manifest(rows))
    assert not any("CLIP UNDERRUN" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# S-A -- clip-fill loop decision + delivered-frames QA + freezedetect parse
# --------------------------------------------------------------------------- #
def test_full_clip_does_not_loop(monkeypatch):
    from nodes import otr_silent_composite as sc
    monkeypatch.delenv("OTR_CLIP_FILL", raising=False)
    rows = [{"shot_id": "s", "engine_id": "ltx_video", "path": "x.mp4",
             "exists": True, "frame_count": 280, "target_frame_count": 280,
             "start_s": None}]
    segs, _ = sc.plan_timeline_segments(_manifest(rows))
    assert not any(s.get("loop") for s in segs if s["source"] == "clip")


def test_dir_clip_not_loop_filled(monkeypatch, tmp_path):
    # A frame-DIRECTORY clip (3D alpha handoff) has its own fill -> never looped.
    from nodes import otr_silent_composite as sc
    monkeypatch.delenv("OTR_CLIP_FILL", raising=False)
    d = tmp_path / "frames"
    d.mkdir()
    rows = [{"shot_id": "s", "engine_id": "character_3d", "path": str(d),
             "exists": True, "frame_count": 10, "target_frame_count": 280,
             "start_s": None}]
    segs, _ = sc.plan_timeline_segments(_manifest(rows))
    assert not any(s.get("loop") for s in segs if s["source"] == "clip")


def test_timeline_quality_report_statuses(monkeypatch):
    # NOTE 2026-06-30: row "b" (ltx / no family / not audio_driven_face) is the
    # generic loop-fill case; row "a" (humo) is now EXCLUDED from loop-fill (the
    # audio_driven_face lip-sync guard below), so it holds instead.
    from nodes import otr_silent_composite as sc
    monkeypatch.delenv("OTR_CLIP_FILL", raising=False)
    rows = [
        {"shot_id": "a", "engine_id": "humo", "family": "audio_driven_face",
         "path": "a.mp4", "exists": True,
         "frame_count": 100, "target_frame_count": 280, "start_s": None},
        {"shot_id": "b", "engine_id": "ltx", "path": "b.mp4", "exists": True,
         "frame_count": 280, "target_frame_count": 280, "start_s": None},
        {"shot_id": "c", "engine_id": "still", "path": "", "exists": False,
         "frame_count": 0, "target_frame_count": 120, "start_s": None},
    ]
    segs, _ = sc.plan_timeline_segments(_manifest(rows))
    qa = sc.timeline_quality_report(_manifest(rows), segs)
    st = {b["shot_id"]: b["quality_status"] for b in qa["beats"]}
    assert st["a"] == "held_last_frame"        # audio_driven_face: never loop
    assert st["b"] == "ok"
    assert st["c"] == "no_clip_segment"
    assert not qa["delivered_frames_ok"]        # row "a" IS a legibility failure
    da = next(b for b in qa["beats"] if b["shot_id"] == "a")
    # the segment plan still spans the FULL target either way (loop vs hold is
    # an ENCODING-time distinction, not a segment-length one) -- only the raw
    # engine-reported frame_count differs from the target.
    assert da["delivered_frame_count"] == 280 and da["frame_count"] == 100


def test_audio_driven_face_excluded_from_loop_fill(monkeypatch):
    # 2026-06-30 HuMo-improve plan: looping a talking/lip-synced face desyncs
    # the mouth from its own audio (the loop replays an earlier mouth shape
    # against later audio) -- worse than a static hold. audio_driven_face
    # (HuMo) rows are excluded from loop-fill even with OTR_CLIP_FILL=1
    # (default) -- they fall to the pre-existing held-last-frame path, which
    # the LOUD legibility guard (timeline_quality_report) still flags.
    from nodes import otr_silent_composite as sc
    monkeypatch.delenv("OTR_CLIP_FILL", raising=False)
    rows = [{"shot_id": "s", "engine_id": "humo_1.7B",
             "family": "audio_driven_face", "path": "x.mp4", "exists": True,
             "frame_count": 100, "target_frame_count": 280, "start_s": None}]
    segs, _ = sc.plan_timeline_segments(_manifest(rows))
    assert not any(s.get("loop") for s in segs if s["source"] == "clip")


def test_non_face_engine_still_loop_fills_alongside_excluded_face(monkeypatch):
    # The exclusion is PER-ROW family, not global: a non-audio_driven_face
    # engine in the SAME manifest still gets the loop-fill benefit.
    from nodes import otr_silent_composite as sc
    monkeypatch.delenv("OTR_CLIP_FILL", raising=False)
    rows = [
        {"shot_id": "face", "engine_id": "humo", "family": "audio_driven_face",
         "path": "a.mp4", "exists": True, "frame_count": 100,
         "target_frame_count": 280, "start_s": None},
        {"shot_id": "broll", "engine_id": "wan_ti2v", "family": "image_to_video",
         "path": "b.mp4", "exists": True, "frame_count": 100,
         "target_frame_count": 280, "start_s": None},
    ]
    segs, _ = sc.plan_timeline_segments(_manifest(rows))
    by_shot = {s["shot_id"]: s for s in segs if s["source"] == "clip"}
    assert by_shot["face"]["loop"] is False
    assert by_shot["broll"]["loop"] is True


def test_should_loop_fill_reads_family_not_engine_name(monkeypatch):
    # Direct unit coverage of the new gate: keyed on "family", not on any
    # engine-name substring match (robust to a future audio_driven_face engine
    # that isn't literally named "humo*").
    from nodes import otr_silent_composite as sc
    monkeypatch.delenv("OTR_CLIP_FILL", raising=False)
    assert sc._should_loop_fill(
        {"frame_count": 10, "family": "audio_driven_face"}, 100) is False
    assert sc._should_loop_fill(
        {"frame_count": 10, "family": "image_to_video"}, 100) is True
    # no family key at all (an incomplete/legacy row) -> not excluded, matches
    # the pre-2026-06-30 behavior for rows that predate the family stamp.
    assert sc._should_loop_fill({"frame_count": 10}, 100) is True


def test_timeline_quality_report_held_when_fill_off(monkeypatch):
    from nodes import otr_silent_composite as sc
    monkeypatch.setenv("OTR_CLIP_FILL", "0")
    rows = [{"shot_id": "a", "engine_id": "humo", "path": "a.mp4",
             "exists": True, "frame_count": 100, "target_frame_count": 280,
             "start_s": None}]
    segs, _ = sc.plan_timeline_segments(_manifest(rows))
    qa = sc.timeline_quality_report(_manifest(rows), segs)
    assert qa["beats"][0]["quality_status"] == "held_last_frame"
    assert not qa["delivered_frames_ok"]


def test_parse_freezedetect():
    from nodes import otr_silent_composite as sc
    stderr = (
        "[Parsed_freezedetect_0 @ 0x1] lavfi.freezedetect.freeze_start: 1.5\n"
        "[Parsed_freezedetect_0 @ 0x1] lavfi.freezedetect.freeze_duration: 1.5\n"
        "[Parsed_freezedetect_0 @ 0x1] lavfi.freezedetect.freeze_end: 3.0\n"
        "[Parsed_freezedetect_0 @ 0x1] lavfi.freezedetect.freeze_start: 7.2\n")
    spans = sc.parse_freezedetect(stderr)
    assert spans == [{"start": 1.5, "end": 3.0}, {"start": 7.2, "end": None}]
