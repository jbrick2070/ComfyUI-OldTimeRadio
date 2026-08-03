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
import json

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


#: PIECE 3 IS GONE (2026-08-02). ``extend_frames_to_target`` -- the
#: ping-pong/mirror extender these four tests covered -- was DELETED under the
#: operator's directive: "kill mirrors and ping-pong, true video for every second
#: of audio." The tests are replaced rather than removed, because what they
#: proved (a short render always ends up covering its beat) still has to be true;
#: only the mechanism changed. Coverage planning now splits a beat the engine
#: cannot afford into native forward-rendered segments, and a render that still
#: falls short is terminal.
def test_the_pingpong_extender_is_gone():
    assert not hasattr(wb, "extend_frames_to_target")


def test_a_short_render_no_longer_fills_itself():
    """The exact case the deleted extender existed for: 7 frames against a
    280-frame beat. It used to tile a mirror cycle to 280. It now refuses, and
    the beat is covered by splitting rather than by reusing frames."""
    with pytest.raises(wb.MirrorExtensionForbidden):
        wb.fit_frames_to_target(_frames(7), 280)


def test_a_single_frame_cannot_become_motion():
    """A still repeated five times was the extender's degenerate branch -- five
    frames of a frozen image standing in for five frames of a beat. That is the
    freeze this whole area was built to remove, so it refuses too."""
    with pytest.raises(wb.MirrorExtensionForbidden):
        wb.fit_frames_to_target(_frames(1), 5)


def test_over_length_renders_still_trim():
    """Trimming never reverses time and is the normal path for any engine whose
    ladder overshoots the target, so it is untouched by the rip."""
    import numpy as np
    f = _frames(10)
    assert np.array_equal(wb.fit_frames_to_target(f, 10), f)
    assert len(wb.fit_frames_to_target(f, 5)) == 5


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
    ledger_path = audio / "signal_lost_final_ledger.json"
    ledger_path.write_text(json.dumps({
        "episode_id": "signal_lost_final",
        "meta": {"freeze_timestamp": "freeze-clips"},
    }), encoding="utf-8")
    monkeypatch.setattr(paths, "otr_episodes_root", lambda: root)
    monkeypatch.setattr(
        "nodes._otr_ledger.in_flight_ledger_path", lambda: ledger_path,
    )

    assert rd.resolve_episode_id_for_clip_persistence(
        "pending_20260708_010101",
        freeze_timestamp="freeze-clips",
    ) == "signal_lost_final"


def test_persist_rekeys_sfx_to_renamed_episode_clips(monkeypatch, tmp_path):
    from nodes._otr_video_engines import render_driver as rd
    import nodes._otr_paths as paths

    monkeypatch.delenv("OTR_TEST_MODE", raising=False)
    root = tmp_path / "otr" / "episodes"
    final = root / "signal_lost_final"
    audio = final / "audio"
    audio.mkdir(parents=True)
    ledger_path = audio / "signal_lost_final_ledger.json"
    ledger_path.write_text(json.dumps({
        "episode_id": "signal_lost_final",
        "meta": {"freeze_timestamp": "freeze-clips"},
    }), encoding="utf-8")
    monkeypatch.setattr(paths, "otr_episodes_root", lambda: root)
    monkeypatch.setattr(
        "nodes._otr_ledger.in_flight_ledger_path", lambda: ledger_path,
    )
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


def test_resolve_stale_pending_clip_rejects_foreign_freeze(monkeypatch, tmp_path):
    from nodes._otr_video_engines import render_driver as rd
    import nodes._otr_paths as paths

    monkeypatch.delenv("OTR_TEST_MODE", raising=False)
    root = tmp_path / "otr" / "episodes"
    final = root / "signal_lost_foreign"
    audio = final / "audio"
    audio.mkdir(parents=True)
    ledger_path = audio / "signal_lost_foreign_ledger.json"
    ledger_path.write_text(json.dumps({
        "episode_id": "signal_lost_foreign",
        "meta": {"freeze_timestamp": "freeze-foreign"},
    }), encoding="utf-8")
    monkeypatch.setattr(paths, "otr_episodes_root", lambda: root)
    monkeypatch.setattr(
        "nodes._otr_ledger.in_flight_ledger_path", lambda: ledger_path,
    )

    assert rd.resolve_episode_id_for_clip_persistence(
        "pending_20260708_020202",
        freeze_timestamp="freeze-current",
    ) == "pending_20260708_020202"


# --------------------------------------------------------------------------- #
# Piece 5 -- composite underrun guard
# --------------------------------------------------------------------------- #
def _manifest(rows, fps=25):
    return {"fps": fps, "clips": rows}

# --------------------------------------------------------------------------- #
# Piece 5 -- THE COMPOSITE'S OWN COVERAGE MECHANISMS, RETIRED 2026-08-02
#
# This block used to pin loop-fill and held-last-frame: a real clip shorter than
# its beat was ffmpeg stream-looped to fill (every family except HuMo's
# audio_driven_face, which held its last frame instead, because looping a mouth
# desyncs it from its own audio).
#
# Both were frame REUSE, and they lived in the assembler rather than in any
# adapter -- which is exactly why they survived the engine-layer mirror rip:
# `extend_frames_to_target` was deleted, `eng_ltx_video`'s boomerang retired,
# and the composite went on looping the same short clip afterwards. Two
# independent review lanes found it in the same pass, and GO_FORWARD_PLAN had
# tracked it as chunk 7c "still open" since 2026-07-27.
#
# `_should_loop_fill`'s own docstring named the replacement and called itself
# interim: "the real fix is phrase-chunking -- render the beat's correct
# duration so it never underruns -- tracked as a follow-up". Coverage planning
# IS that follow-up and it is live, so a shortfall is now terminal.
# --------------------------------------------------------------------------- #
def test_a_short_clip_is_terminal_at_composite_time(monkeypatch):
    """The headline inversion: 17 frames against a 280-frame beat used to LOOP."""
    from nodes import otr_silent_composite as sc
    monkeypatch.delenv("OTR_CLIP_FILL", raising=False)
    rows = [{"shot_id": "shot_b001", "engine_id": "wan_ti2v", "path": "x.mp4",
             "exists": True, "frame_count": 17, "target_frame_count": 280,
             "start_s": None}]
    with pytest.raises(sc.ClipUnderrunsItsBeat) as exc:
        sc.plan_timeline_segments(_manifest(rows))
    assert exc.value.real == 17 and exc.value.target == 280
    assert "shot_b001" in str(exc.value) and "wan_ti2v" in str(exc.value)
    # The message must name the REMEDY, not just the number: the fix is always
    # in the render, never in the timeline.
    assert "coverage planning" in str(exc.value)


def test_the_fill_env_switch_can_no_longer_bring_looping_back(monkeypatch):
    """`OTR_CLIP_FILL=0` used to select held-last-frame instead of looping.

    Neither outcome is legal now, so the knob is inert -- the same reasoning
    that removed `allow_mirror` rather than defaulting it off.
    """
    from nodes import otr_silent_composite as sc
    rows = [{"shot_id": "shot_b001", "engine_id": "wan_ti2v", "path": "x.mp4",
             "exists": True, "frame_count": 17, "target_frame_count": 280,
             "start_s": None}]
    for val in ("0", "1"):
        monkeypatch.setenv("OTR_CLIP_FILL", val)
        with pytest.raises(sc.ClipUnderrunsItsBeat):
            sc.plan_timeline_segments(_manifest(rows))


def test_a_ONE_frame_shortfall_still_raises(monkeypatch):
    """The old guard only warned below a FRACTION of the target
    (`OTR_CLIP_UNDERRUN_FRAC`) -- sensible when the question was "is this bad
    enough to look at", but 40 ms of audio with no original video behind it is
    the whole thing being forbidden."""
    from nodes import otr_silent_composite as sc
    monkeypatch.setenv("OTR_CLIP_UNDERRUN_FRAC", "0.5")
    rows = [{"shot_id": "shot_b002", "engine_id": "ltx_8gb", "path": "x.mp4",
             "exists": True, "frame_count": 279, "target_frame_count": 280,
             "start_s": None}]
    with pytest.raises(sc.ClipUnderrunsItsBeat):
        sc.plan_timeline_segments(_manifest(rows))


def test_a_face_lane_is_no_longer_exempt(monkeypatch):
    """`audio_driven_face` was exempt from LOOPING because a looped mouth
    desyncs -- and then held its last frame instead, which covers the same audio
    with a frozen picture. The exemption bought the honest failure; the rule now
    forbids both outcomes, so the family no longer changes the answer."""
    from nodes import otr_silent_composite as sc
    monkeypatch.delenv("OTR_CLIP_FILL", raising=False)
    rows = [{"shot_id": "shot_b003", "engine_id": "humo", "path": "x.mp4",
             "family": "audio_driven_face", "exists": True,
             "frame_count": 49, "target_frame_count": 280, "start_s": None}]
    with pytest.raises(sc.ClipUnderrunsItsBeat):
        sc.plan_timeline_segments(_manifest(rows))


def test_should_loop_fill_is_a_named_no_op():
    """Kept as a no-op rather than deleted at its call sites, so the retirement
    is visible where the decision used to be made."""
    from nodes import otr_silent_composite as sc
    assert sc._should_loop_fill(
        {"frame_count": 1, "target_frame_count": 999, "path": "x.mp4"},
        999) is False
    assert sc._should_loop_fill({}, 0) is False


def test_a_clip_that_covers_its_beat_is_untouched(monkeypatch):
    """The normal path. Exact coverage plans one clip segment, no loop flag."""
    from nodes import otr_silent_composite as sc
    monkeypatch.delenv("OTR_CLIP_FILL", raising=False)
    rows = [{"shot_id": "shot_b001", "engine_id": "wan_ti2v", "path": "x.mp4",
             "exists": True, "frame_count": 280, "target_frame_count": 280,
             "start_s": None}]
    segs, total = sc.plan_timeline_segments(_manifest(rows))
    assert total == 280
    clip_segs = [s for s in segs if s["source"] == "clip"]
    assert clip_segs and not any(s.get("loop") for s in clip_segs)


def test_an_OVER_long_clip_is_fine(monkeypatch):
    """Rendering MORE than the beat needs is normal -- a ladder rung overshoots
    and the assembler trims. Only a shortfall is a coverage failure."""
    from nodes import otr_silent_composite as sc
    rows = [{"shot_id": "shot_b001", "engine_id": "ltx_audio_in", "path": "x.mp4",
             "exists": True, "frame_count": 449, "target_frame_count": 442,
             "start_s": None}]
    segs, total = sc.plan_timeline_segments(_manifest(rows))
    assert total == 442


def test_a_frame_DIRECTORY_clip_is_still_exempt(monkeypatch, tmp_path):
    """The 3D alpha handoff counts its frames with its own dir encoder, so this
    row's `frame_count` is not the authority for it."""
    from nodes import otr_silent_composite as sc
    d = tmp_path / "frames"
    d.mkdir()
    rows = [{"shot_id": "shot_b001", "engine_id": "mesh_stage", "path": str(d),
             "exists": True, "frame_count": 3, "target_frame_count": 280,
             "start_s": None}]
    segs, total = sc.plan_timeline_segments(_manifest(rows))
    assert total == 280


def test_parse_freezedetect():
    from nodes import otr_silent_composite as sc
    stderr = (
        "[Parsed_freezedetect_0 @ 0x1] lavfi.freezedetect.freeze_start: 1.5\n"
        "[Parsed_freezedetect_0 @ 0x1] lavfi.freezedetect.freeze_duration: 1.5\n"
        "[Parsed_freezedetect_0 @ 0x1] lavfi.freezedetect.freeze_end: 3.0\n"
        "[Parsed_freezedetect_0 @ 0x1] lavfi.freezedetect.freeze_start: 7.2\n")
    spans = sc.parse_freezedetect(stderr)
    assert spans == [{"start": 1.5, "end": 3.0}, {"start": 7.2, "end": None}]
