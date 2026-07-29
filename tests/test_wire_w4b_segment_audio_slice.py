"""WIRE-W4b -- a lip-synced SEGMENT is driven by its OWN audio.

THE DEFECT, stated plainly. Every segment of a multi-clip beat was handed the
WHOLE beat's audio slice. On ``audio_driven_face`` -- HuMo, whose entire output
is a mouth driven by this beat's speech -- a 3-segment beat therefore rendered
three clips all lip-syncing to the same waveform FROM THE TOP, and the
assembled beat said the opening of the line three times. Nothing caught it,
because every clip carried the right frame count and the right init image; only
the sound was wrong, and no gate listens.

The arithmetic is ``coverage_plan.segment_render_window`` -- pure, so it is
tested here on its own before anything about ffmpeg or the ledger is involved.
``render_driver``'s only job is to add the beat's own ``start_s``.
"""

from __future__ import annotations

import os
import sys
import unittest.mock as mock

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes._otr_video_engines import coverage_plan as cp
from nodes._otr_video_engines import render_driver as rd

FPS = 25


def _plan(join, *segments):
    """(render_frames, drop_head, trim_tail) triples -> a CoveragePlan."""
    segs = tuple(
        cp.CoverageSegment(index=i, render_frames=r, drop_head=d, trim_tail=t)
        for i, (r, d, t) in enumerate(segments))
    return cp.CoveragePlan(
        target_visible_frames=sum(s.visible_frames for s in segs),
        join_mode=join, segments=segs)


# ---------------------------------------------------------------------------
# The arithmetic, on its own
# ---------------------------------------------------------------------------

def test_a_JUMP_beats_segments_TILE_the_beat_end_to_end():
    """No drop, no trim: three 33-frame clips cover 99 frames, and each one's
    audio starts exactly where the previous one's ended. Any gap or overlap
    here is a gap or overlap in the finished beat's lip sync."""
    plan = _plan(cp.JOIN_JUMP, (33, 0, 0), (33, 0, 0), (33, 0, 0))
    windows = [cp.segment_render_window(plan, i, FPS) for i in range(3)]
    assert windows == [(0.0, 33 / FPS), (33 / FPS, 33 / FPS),
                       (66 / FPS, 33 / FPS)]
    for (start, dur), (nxt, _d) in zip(windows, windows[1:]):
        assert start + dur == pytest.approx(nxt)


def test_a_CHAINED_successor_gets_audio_for_the_frame_it_DROPS():
    """THE off-by-one this helper exists to get right. A chained successor
    renders one frame EARLIER than it contributes, because its first frame
    duplicates its predecessor's terminal frame and is dropped at assembly.
    Hand it the VISIBLE window and its mouth runs a frame ahead of its own
    audio for the whole clip -- on every segment but the first."""
    plan = _plan(cp.JOIN_CHAIN, (33, 0, 0), (33, 1, 0), (33, 1, 0))
    assert cp.segment_render_window(plan, 0, FPS) == (0.0, 33 / FPS)
    # segment 1 contributes from visible frame 33 but RENDERS from 32
    assert cp.segment_render_window(plan, 1, FPS) == (32 / FPS, 33 / FPS)
    # segment 2 contributes from 65 (33 + 32) and RENDERS from 64
    assert cp.segment_render_window(plan, 2, FPS) == (64 / FPS, 33 / FPS)


def test_the_TRIMMED_tail_still_gets_audio_because_it_is_still_RENDERED():
    """``trim_tail`` is an ASSEMBLY cut, not a render one -- the adapter is
    asked for those frames and needs audio under them like any others. The
    caller slices from the FROZEN MASTER rather than from a per-beat file, so
    the samples exist; they are discarded at assembly, which is the only reason
    it is safe for them to carry the next beat's sound."""
    plan = _plan(cp.JOIN_JUMP, (33, 0, 0), (33, 0, 2))
    assert cp.segment_render_window(plan, 1, FPS) == (33 / FPS, 33 / FPS)
    assert plan.total_visible_frames == 64          # the beat is 2 frames shorter


def test_the_WINDOWS_COVER_the_beat_for_every_shipped_ladder():
    """A property rather than an example: for any plan the partitioner emits,
    segment 0 starts at 0 and every later segment starts where the previous
    one's VISIBLE frames ended -- which is what makes the assembled audio
    continuous."""
    from nodes._otr_video_engines import frame_contract as fc
    from nodes._otr_video_engines import registry as vreg
    import nodes._otr_video_engines  # noqa: F401  -- populate the registry

    checked = 0
    for name in sorted(vreg.all_engine_names()):
        contract = fc.frame_contract_for(vreg.get_engine(name))
        for target in (150, 240, 400, 700):
            try:
                plan = cp.partition_beat(target, contract)
            except cp.CoveragePlanError:
                continue
            if not plan.is_multi_clip:
                continue
            visible = 0
            for segment in plan.segments:
                start, dur = cp.segment_render_window(
                    plan, segment.index, FPS)
                assert start == pytest.approx(
                    max(0, visible - segment.drop_head) / FPS)
                assert dur == pytest.approx(segment.render_frames / FPS)
                visible += segment.visible_frames
                checked += 1
    assert checked >= 30, "only %d segment windows checked" % checked


def test_an_OUT_OF_PLAN_index_is_refused_rather_than_read_as_segment_zero():
    """Indices are the plan's OWN. A plan replayed through ``from_dict`` is
    never re-checked for dense indices, and positional trust is what made
    ``jump_still_requests`` mint one still for two segments."""
    plan = _plan(cp.JOIN_JUMP, (33, 0, 0), (33, 0, 0))
    with pytest.raises(cp.CoveragePlanError) as caught:
        cp.segment_render_window(plan, 7, FPS)
    assert "NO FALLBACK" in str(caught.value)


def test_a_ZERO_FPS_is_refused_rather_than_dividing_by_it():
    plan = _plan(cp.JOIN_JUMP, (33, 0, 0), (33, 0, 0))
    for bad in (0, 0.0, None, -25):
        with pytest.raises(cp.CoveragePlanError):
            cp.segment_render_window(plan, 0, bad)


def test_a_SINGLE_CLIP_beat_must_not_ask_for_a_segment_window():
    """It takes the BEAT's window, unchanged. Answering here would give the
    caller a second, subtly different derivation of the same number."""
    with pytest.raises(cp.CoveragePlanError):
        cp.segment_render_window(None, 0, FPS)


# ---------------------------------------------------------------------------
# The seam: render_driver adds the beat's start_s and nothing else
# ---------------------------------------------------------------------------

def _ledger(master_hash="deadbeef"):
    return {
        "audio": {"master_audio_sha256": master_hash},
        "video": {"fps": FPS, "shots": [
            {"shot_id": "shot_b001", "target_frame_count": 99}]},
        "lines": [{"line_id": "b001", "start_s": 10.0, "dur_s": 99 / FPS}],
        "images": {"images": []},
    }


def _shot(plan=None):
    shot = {"shot_id": "shot_b001", "beat_id": "b001",
            "engine_id": "humo", "family": "audio_driven_face",
            "role": "character_video", "char_id": "c1",
            "target_frame_count": 99, "source_line_ids": ["b001"],
            "creative": {"text_prompt": "a line", "request_hash": "h"}}
    if plan is not None:
        shot["coverage_plan"] = plan.to_dict()
    return shot


def _capture_slices(tmp_path, shot, ledger, indices):
    """Build one request per segment index and return the (start, dur) each
    one asked the slicer for."""
    master = tmp_path / "master.mp4"
    master.write_bytes(b"fake-master")
    out = tmp_path / "slice.wav"
    out.write_bytes(b"RIFF fake wav")
    seen = []

    def _fake_slice(path, start, dur, master_hash=""):
        seen.append((start, dur))
        return str(out)

    with mock.patch.object(rd, "_slice_master_audio", side_effect=_fake_slice):
        for index in indices:
            rd.build_request_from_shot(dict(shot), ledger,
                                       master_audio_path=str(master),
                                       segment_index=index)
    return seen


def test_EVERY_SEGMENT_of_a_multi_clip_beat_asks_for_a_DIFFERENT_slice(
        tmp_path):
    """THE test this chunk exists to pass. Three segments, three windows, each
    offset by the beat's own start_s. Before this they were three identical
    requests for the whole beat."""
    plan = _plan(cp.JOIN_JUMP, (33, 0, 0), (33, 0, 0), (33, 0, 0))
    seen = _capture_slices(tmp_path, _shot(plan), _ledger(), (0, 1, 2))
    assert seen == [
        (pytest.approx(10.0), pytest.approx(33 / FPS)),
        (pytest.approx(10.0 + 33 / FPS), pytest.approx(33 / FPS)),
        (pytest.approx(10.0 + 66 / FPS), pytest.approx(33 / FPS)),
    ]
    assert len({round(s, 6) for s, _d in seen}) == 3, (
        "two segments asking for the same window is the defect itself")


def test_a_SINGLE_CLIP_beat_asks_for_exactly_what_it_always_did(tmp_path):
    """The majority path. A beat with no coverage plan -- and a beat whose plan
    is one segment -- must produce a byte-identical slice request, because that
    is what production renders today."""
    unplanned = _capture_slices(tmp_path, _shot(None), _ledger(), (0,))
    one_seg = _capture_slices(
        tmp_path, _shot(_plan(cp.JOIN_SINGLE, (99, 0, 0))), _ledger(), (0,))
    assert unplanned == one_seg
    assert unplanned == [(pytest.approx(10.0), pytest.approx(99 / FPS))]


def test_the_SLICE_CACHE_KEY_separates_the_segments(tmp_path):
    """The cache is keyed by (master hash, start, dur). Two segments asking for
    different windows must therefore get different cached files -- if the key
    ever collapsed, segment 2 would silently be served segment 1's wav and the
    defect would be back with a cache in front of it."""
    plan = _plan(cp.JOIN_JUMP, (33, 0, 0), (33, 0, 0), (33, 0, 0))
    seen = _capture_slices(tmp_path, _shot(plan), _ledger(), (0, 1, 2))
    keys = {rd.slice_cache_key("deadbeef", start, dur) for start, dur in seen}
    assert len(keys) == 3


def test_a_CHAINED_beats_slices_OVERLAP_by_exactly_the_dropped_frame(tmp_path):
    """Chained successors overlap their predecessor by one frame of audio, and
    that is correct: the overlapping frame is the one they drop."""
    plan = _plan(cp.JOIN_CHAIN, (33, 0, 0), (33, 1, 0))
    seen = _capture_slices(tmp_path, _shot(plan), _ledger(), (0, 1))
    (s0, d0), (s1, _d1) = seen
    assert s0 + d0 - s1 == pytest.approx(1 / FPS)
