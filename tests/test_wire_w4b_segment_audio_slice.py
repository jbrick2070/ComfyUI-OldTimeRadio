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

def _win(plan, index):
    """(offset, copy, pad) as plain floats, for readable comparisons."""
    w = cp.segment_render_window(plan, index, FPS)
    return (w.offset_s, w.copy_s, w.pad_s)


def test_a_JUMP_beats_segments_TILE_the_beat_end_to_end():
    """No drop, no trim: three 33-frame clips cover 99 frames, and each one's
    audio starts exactly where the previous one's ended. Any gap or overlap
    here is a gap or overlap in the finished beat's lip sync."""
    plan = _plan(cp.JOIN_JUMP, (33, 0, 0), (33, 0, 0), (33, 0, 0))
    windows = [cp.segment_render_window(plan, i, FPS) for i in range(3)]
    assert [_win(plan, i) for i in range(3)] == [
        (0.0, 33 / FPS, 0.0), (33 / FPS, 33 / FPS, 0.0),
        (66 / FPS, 33 / FPS, 0.0)]
    for cur, nxt in zip(windows, windows[1:]):
        assert cur.offset_s + cur.total_s == pytest.approx(nxt.offset_s)


def test_a_CHAINED_successor_gets_audio_for_the_frame_it_DROPS():
    """THE off-by-one this helper exists to get right. A chained successor
    renders one frame EARLIER than it contributes, because its first frame
    duplicates its predecessor's terminal frame and is dropped at assembly.
    Hand it the VISIBLE window and its mouth runs a frame ahead of its own
    audio for the whole clip -- on every segment but the first."""
    plan = _plan(cp.JOIN_CHAIN, (33, 0, 0), (33, 1, 0), (33, 1, 0))
    assert _win(plan, 0) == (0.0, 33 / FPS, 0.0)
    # segment 1 contributes from visible frame 33 but RENDERS from 32
    assert _win(plan, 1) == (32 / FPS, 33 / FPS, 0.0)
    # segment 2 contributes from 65 (33 + 32) and RENDERS from 64
    assert _win(plan, 2) == (64 / FPS, 33 / FPS, 0.0)


def test_the_TRIMMED_tail_is_SILENCE_and_never_the_next_beats_SPEECH():
    """r4/A4, and W4b got this wrong: it copied ``render_frames`` straight off
    the master, so the trimmed tail carried whatever came next in the episode.

    That looked harmless because those frames are discarded at assembly. It is
    not: the audio encoder sees the WHOLE waveform before a single frame is
    sampled, so speech sitting in the tail conditions the frames that DO
    survive. The window stays ``render_frames`` long -- that is the generation
    length -- but only ``render_frames - trim_tail`` is COPIED."""
    plan = _plan(cp.JOIN_JUMP, (33, 0, 0), (33, 0, 2))
    offset, copy, pad = _win(plan, 1)
    assert offset == pytest.approx(33 / FPS)
    assert copy == pytest.approx(31 / FPS)          # the real speech
    assert pad == pytest.approx(2 / FPS)            # silence, not the next beat
    window = cp.segment_render_window(plan, 1, FPS)
    assert window.total_s == pytest.approx(33 / FPS), (
        "the conditioning WAV duration EQUALS render_frames")
    assert plan.total_visible_frames == 64          # the beat is 2 frames shorter


def test_the_OPERATORS_PINNED_184_CASE_is_the_contracts_own_example():
    """r4/A4 states it in numbers: the pinned 184-frame beat plans [153, 33]
    with trim 2, so the last segment is "31 audio frames against a 33-frame
    render". Nothing else in this file would notice if the two stopped
    agreeing."""
    plan = _plan(cp.JOIN_JUMP, (153, 0, 0), (33, 0, 2))
    assert plan.total_visible_frames == 184
    window = cp.segment_render_window(plan, 1, FPS)
    assert window.copy_s * FPS == pytest.approx(31)
    assert window.pad_s * FPS == pytest.approx(2)
    assert window.total_s * FPS == pytest.approx(33)


def test_a_segment_with_NO_TRIM_owes_NO_SILENCE():
    """The pad is not a constant sprinkled on every segment -- only the one
    the ladder could not land on exactly carries any."""
    plan = _plan(cp.JOIN_JUMP, (33, 0, 0), (33, 0, 2))
    assert _win(plan, 0)[2] == 0.0
    assert _win(plan, 1)[2] > 0.0


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
                window = cp.segment_render_window(plan, segment.index, FPS)
                assert window.offset_s == pytest.approx(
                    max(0, visible - segment.drop_head) / FPS)
                assert window.total_s == pytest.approx(
                    segment.render_frames / FPS), (
                    "the conditioning WAV is always the GENERATION length")
                assert window.pad_s == pytest.approx(
                    segment.trim_tail / FPS), (
                    "and only the trimmed frames are silence")
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


def _capture_slices(tmp_path, shot, ledger, indices, with_pad=False):
    """Build one request per segment index and return what each one asked the
    slicer for -- (start, dur) by default, (start, dur, pad) when asked."""
    master = tmp_path / "master.mp4"
    master.write_bytes(b"fake-master")
    out = tmp_path / "slice.wav"
    out.write_bytes(b"RIFF fake wav")
    seen = []

    def _fake_slice(path, start, dur, master_hash="", pad_tail_s=0.0):
        seen.append((start, dur, pad_tail_s) if with_pad else (start, dur))
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


# ---------------------------------------------------------------------------
# WIRE-W4c -- the conditioning WAV contract reaches ffmpeg
# ---------------------------------------------------------------------------

def test_the_TRIM_reaches_the_SLICER_as_a_SILENCE_PAD(tmp_path):
    """The arithmetic being right is half of it -- the pad has to arrive at the
    process that writes the WAV. W4b passed only (start, dur), so the correct
    numbers were computed and the wrong file was written."""
    plan = _plan(cp.JOIN_JUMP, (153, 0, 0), (33, 0, 2))
    ledger = _ledger()
    ledger["lines"][0]["dur_s"] = 184 / FPS
    ledger["video"]["shots"][0]["target_frame_count"] = 184
    shot = _shot(plan)
    shot["target_frame_count"] = 184
    seen = _capture_slices(tmp_path, shot, ledger, (0, 1), with_pad=True)
    assert seen[0] == (pytest.approx(10.0), pytest.approx(153 / FPS),
                       pytest.approx(0.0))
    assert seen[1] == (pytest.approx(10.0 + 153 / FPS),
                       pytest.approx(31 / FPS), pytest.approx(2 / FPS))


def test_a_SINGLE_CLIP_beat_asks_for_NO_PAD(tmp_path):
    """The majority path stays byte-identical: no plan, no pad."""
    seen = _capture_slices(tmp_path, _shot(None), _ledger(), (0,),
                           with_pad=True)
    assert seen == [(pytest.approx(10.0), pytest.approx(99 / FPS), 0.0)]


def test_the_PAD_is_IN_the_cache_key(tmp_path):
    """Two segments can copy the IDENTICAL source interval and owe different
    silence. A key that ignored the pad would serve the first one's WAV to the
    second -- an under-length conditioning track wearing a cache hit."""
    bare = rd.slice_cache_key("deadbeef", 1.0, 2.0)
    padded = rd.slice_cache_key("deadbeef", 1.0, 2.0, pad_tail_s=0.08)
    assert bare != padded
    assert rd.slice_cache_key("deadbeef", 1.0, 2.0, pad_tail_s=0.0) == bare


def test_the_SLICER_VERSION_moved_so_old_cached_WAVs_cannot_be_served():
    """Every WAV on disk from before this chunk describes a different contract
    -- the same (master, start, dur) now means "and pad the tail". Serving one
    would hand a segment a conditioning track built to the old rule."""
    assert rd.SLICER_VERSION != "2"


def _slicer_argv(tmp_path, monkeypatch, **kwargs):
    """Run the slicer with ffmpeg faked and the cache ISOLATED to tmp_path,
    and return the argv it built.

    The isolation is load-bearing: the slice cache lives under the shared
    episode tmp dir, so a second run of this test would take a CACHE HIT, skip
    ffmpeg entirely and assert against an argv that was never built."""
    from nodes._otr_video_engines import _tmp as _otr_tmp
    monkeypatch.setattr(_otr_tmp, "_in_tree_tmp_dir", lambda: str(tmp_path))
    master = tmp_path / "master.wav"
    master.write_bytes(b"RIFF fake")
    seen = {}

    def _fake_run(cmd, **_kw):
        seen["cmd"] = list(cmd)
        with open(cmd[-1], "wb") as fh:          # ffmpeg's job, faked
            fh.write(b"RIFF fake out")
        return mock.Mock(returncode=0)

    monkeypatch.setattr(rd.otr_proc, "run", _fake_run)
    out = rd._slice_master_audio(str(master), master_hash="deadbeef", **kwargs)
    assert out, "the slicer reported failure, so there is no argv to judge"
    return seen["cmd"]


def test_the_FFMPEG_COMMAND_pads_with_SILENCE_and_truncates_to_length(
        tmp_path, monkeypatch):
    """Down to the argv, because this is the layer where "the tail is silence"
    either happens or does not. ``apad`` alone never terminates and a bare
    output ``-t`` would just re-cut the source, so the pair is the contract."""
    cmd = _slicer_argv(tmp_path, monkeypatch, start_s=10.0, dur_s=31 / FPS,
                       pad_tail_s=2 / FPS)
    assert "-af" in cmd and cmd[cmd.index("-af") + 1] == "apad"
    # the OUTPUT -t is the TOTAL contracted length, not the source interval
    assert cmd[-3:-1] == ["-t", "%.6f" % (31 / FPS + 2 / FPS)]
    # ...and the INPUT -t still reads only the real speech
    assert cmd[cmd.index("-i") - 1] == "%.6f" % (31 / FPS)


def test_an_UNPADDED_slice_keeps_the_shipped_ffmpeg_command(tmp_path,
                                                            monkeypatch):
    """CONTROL. Every single-clip beat in production takes this path, so the
    argv must not grow a filter it never had."""
    cmd = _slicer_argv(tmp_path, monkeypatch, start_s=10.0, dur_s=2.0)
    assert "-af" not in cmd
    assert cmd.count("-t") == 1, "only the INPUT read-duration"


# ---------------------------------------------------------------------------
# WIRE-W4e -- a PER-LINE voice wav is sliced per segment too
# ---------------------------------------------------------------------------

def _ledger_with_line_wav(wav_path):
    led = _ledger()
    led["lines"][0]["char_wav_path"] = str(wav_path)
    return led


def test_a_PER_LINE_VOICE_WAV_is_sliced_per_SEGMENT(tmp_path):
    """THE HOLE W4b LEFT, and the one that blocked the whole audio-driven lane.

    W4b/W4c narrowed the FROZEN-MASTER slice. A beat whose line carries its own
    clean voice wav never reaches that code -- it takes the per-line branch and
    skips the slicer entirely -- so every segment of a multi-clip beat got the
    WHOLE line from its start. ``otr_shot_lock`` refused such beats outright
    for exactly this reason, which is why every HuMo lane came back unbuildable
    on the first 45-word campaign leg: "beat l001 needs 2 clips on humo (185
    frames, cap 177)".

    The window authority is the SAME one the master path uses; only the ORIGIN
    differs -- a per-line wav starts at its own zero, so no ``start_s`` is
    added."""
    wav = tmp_path / "line.wav"
    wav.write_bytes(b"RIFF fake voice")
    plan = _plan(cp.JOIN_JUMP, (153, 0, 0), (33, 0, 2))
    shot = _shot(plan)
    shot["target_frame_count"] = 184
    ledger = _ledger_with_line_wav(wav)
    ledger["lines"][0]["dur_s"] = 184 / FPS
    seen = []

    def _fake_slice(path, start, dur, master_hash="", pad_tail_s=0.0):
        seen.append((os.path.basename(str(path)), start, dur, pad_tail_s))
        out = tmp_path / ("cut_%d.wav" % len(seen))
        out.write_bytes(b"RIFF cut")
        return str(out)

    with mock.patch.object(rd, "_slice_master_audio", side_effect=_fake_slice):
        for index in (0, 1):
            rd.build_request_from_shot(dict(shot), ledger, segment_index=index)

    assert [s[0] for s in seen] == ["line.wav", "line.wav"], (
        "the LINE's own wav is what gets cut, not the master")
    # offsets are from the wav's OWN zero -- no start_s added
    assert seen[0][1] == pytest.approx(0.0)
    assert seen[1][1] == pytest.approx(153 / FPS)
    # and the trimmed tail is silence, exactly as on the master path
    assert seen[1][2] == pytest.approx(31 / FPS)
    assert seen[1][3] == pytest.approx(2 / FPS)


def test_a_SINGLE_CLIP_beat_keeps_its_LINE_WAV_UNCUT(tmp_path):
    """CONTROL, and it is the majority path: every ordinary beat must hand the
    engine the line's wav exactly as the voice phase produced it."""
    wav = tmp_path / "line.wav"
    wav.write_bytes(b"RIFF fake voice")
    shot = _shot(_plan(cp.JOIN_SINGLE, (99, 0, 0)))
    with mock.patch.object(rd, "_slice_master_audio") as m:
        req = rd.build_request_from_shot(
            dict(shot), _ledger_with_line_wav(wav), segment_index=0)
    m.assert_not_called()
    assert (req.get("audio_ref") or {}).get("path") == str(wav)


def test_a_FAILED_per_segment_voice_slice_REFUSES_rather_than_using_the_LINE(
        tmp_path):
    """NO FALLBACK. Handing the segment the whole line is the sync defect this
    exists to remove, and it would ship as a finished episode -- so a slicer
    that comes back empty is terminal, not a degrade."""
    wav = tmp_path / "line.wav"
    wav.write_bytes(b"RIFF fake voice")
    shot = _shot(_plan(cp.JOIN_JUMP, (99, 0, 0), (99, 0, 0)))
    shot["target_frame_count"] = 198
    ledger = _ledger_with_line_wav(wav)
    ledger["lines"][0]["dur_s"] = 198 / FPS
    with mock.patch.object(rd, "_slice_master_audio", return_value=""):
        with pytest.raises(rd.RenderError) as caught:
            rd.build_request_from_shot(dict(shot), ledger, segment_index=1)
    assert "NO FALLBACK" in str(caught.value)


def test_the_SLICER_honours_the_CONFIGURED_ffmpeg(tmp_path, monkeypatch):
    """otr_credits_roll already honoured OTR_FFMPEG while this module used the
    bare literal, so on a box where ffmpeg is configured but not on PATH the
    credits rendered and the slice silently returned "" -- which reads
    downstream as "this beat has no voice line", not as "this box cannot
    slice"."""
    configured = tmp_path / "my-ffmpeg.exe"
    configured.write_bytes(b"#!/bin/sh\n")
    monkeypatch.setenv("OTR_FFMPEG", str(configured))
    assert rd._slicer_ffmpeg_bin() == str(configured)
    monkeypatch.delenv("OTR_FFMPEG", raising=False)
    assert "ffmpeg" in rd._slicer_ffmpeg_bin()
