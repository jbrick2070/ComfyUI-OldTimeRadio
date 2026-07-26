"""ONE BEAT, several renders: the loop and the assembly.

Multi-clip coverage chunks 6c + 6d. The stub engine here writes REAL mp4s
through the shared encoder, so the beat session, the per-segment requests, the
terminal-frame handoff and the assembly all run against actual ffmpeg rather
than against a mock that agrees with itself. Each segment paints a distinct
grey, which is what makes "did the right frames end up in the right order"
checkable instead of assumed.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

import nodes._otr_video_engines  # noqa: F401  -- populate the registry
from nodes._otr_video_engines import coverage_plan as cp
from nodes._otr_video_engines import render_driver as rd
from nodes._otr_video_engines import wan_shared as ws
from nodes._otr_video_engines import wrapper_bridge as wb

_HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
pytestmark = pytest.mark.skipif(not _HAS_FFMPEG, reason="needs ffmpeg + ffprobe")


class _BeatStub:
    """Writes a real clip per segment; records the lifecycle and the inits."""

    name = "stub_beat_engine"
    family = "abstract"

    def __init__(self, tmpdir):
        self.tmpdir = tmpdir
        self.load_calls = 0
        self.teardown_calls = 0
        self.seen_inits = []
        self.seen_frames = []

    def session_identity(self):
        return ("stub_beat_engine", "recipe_a", "weights_a")

    def load(self):
        self.load_calls += 1

    def prepare(self, host_caps, profile, session_ctx):
        self.load()
        return {"engine_id": self.name}

    def assert_usable(self, host_caps, profile, request_template=None):
        return True

    def render_clip(self, request, prepared):
        import numpy as np
        index = int(request["segment_index"])
        frames = int(request["frames"])
        self.seen_inits.append(request.get("init_image") or "")
        self.seen_frames.append(frames)
        # A distinct grey per segment, so the assembled beat is readable.
        level = 40 + index * 60
        imgs = [np.full((64, 64, 3), level, dtype=np.uint8) for _ in range(frames)]
        out = os.path.join(self.tmpdir, "seg%d.mp4" % index)
        path, n = wb.encode_frames_to_silent_mp4(imgs, out, 25)
        return {"path": path, "frame_count": n, "fps": 25}

    def canonicalize(self, raw, request, profile):
        return dict(raw, type="video")

    def teardown(self, prepared):
        self.teardown_calls += 1


def _plan(join_mode, segments, target):
    return {"target_visible_frames": target, "join_mode": join_mode,
            "segments": [dict(s) for s in segments]}


def _shot(plan):
    return {"shot_id": "shot_b001", "beat_id": "b001",
            "engine_id": "stub_beat_engine", "family": "abstract",
            "role": "character_video", "target_frame_count":
                plan["target_visible_frames"],
            "coverage_plan": plan}


def _request_builder(shot, ledger, *, canvas=None, segment_index=0):
    """Stands in for build_request_from_shot: gives each segment its own
    length off the stamped plan, which is the seam chunk 6b/QA6 landed."""
    return {"shot_id": shot["shot_id"], "segment_index": int(segment_index),
            "frames": rd.segment_render_frames(shot, segment_index),
            "init_image": "seg%d_still.png" % int(segment_index),
            "observability": {}}


def _install(monkeypatch, engine):
    monkeypatch.setattr(rd._vreg, "is_registered",
                        lambda name: name == engine.name)
    monkeypatch.setattr(rd._vreg, "get_engine", lambda name: engine)


# ---------------------------------------------------------------------------
# A JUMP beat: independent segments, each from its own still
# ---------------------------------------------------------------------------

def test_a_JUMP_beat_loads_ONCE_and_assembles_to_the_planned_length(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        eng = _BeatStub(tmp)
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_JUMP, [
            {"index": 0, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
        ], 50)
        clip, _out_shot, _att, _used = rd.render_beat_coverage(
            _shot(plan), {}, request_builder=_request_builder)

        assert eng.load_calls == 1, "the beat reloaded its weights per segment"
        assert eng.teardown_calls == 1
        assert eng.seen_frames == [25, 25]
        # Each segment conditioned on its OWN still, not the beat's.
        assert eng.seen_inits == ["seg0_still.png", "seg1_still.png"]
        # And the beat is exactly as long as the plan promised, counted by
        # decoding rather than read off a header.
        assert ws.ffprobe_counted_frames(clip["path"]) == 50
        assert clip["frame_count"] == 50
        assert clip["segment_count"] == 2
        assert clip["join_mode"] == cp.JOIN_JUMP


def test_the_assembled_beat_plays_the_segments_IN_ORDER(monkeypatch):
    """Segment 0 is dark, segment 1 is light. The first half of the assembled
    beat must be the dark one -- an assembler that concatenated backwards would
    still produce the right frame COUNT."""
    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        eng = _BeatStub(tmp)
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_JUMP, [
            {"index": 0, "render_frames": 10, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 10, "drop_head": 0, "trim_tail": 0},
        ], 20)
        clip, _s, _a, _u = rd.render_beat_coverage(
            _shot(plan), {}, request_builder=_request_builder)

        first = os.path.join(tmp, "first.png")
        wb.run_ffmpeg([
            "ffmpeg", "-y", "-i", clip["path"], "-frames:v", "1", first])
        head = float(np.asarray(Image.open(first).convert("RGB")).mean())
        assert abs(head - 40.0) < 12.0, (
            "the beat does not OPEN on segment 0 (got mean %.1f)" % head)


def test_a_TRIMMED_tail_is_honoured_frame_for_frame(monkeypatch):
    """``allow_tail_trim`` exists so a beat whose target has no exact cover can
    still be covered exactly. The assembler is where that promise is kept."""
    with tempfile.TemporaryDirectory() as tmp:
        eng = _BeatStub(tmp)
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_JUMP, [
            {"index": 0, "render_frames": 17, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 17, "drop_head": 0, "trim_tail": 5},
        ], 29)
        clip, _s, _a, _u = rd.render_beat_coverage(
            _shot(plan), {}, request_builder=_request_builder)
        assert ws.ffprobe_counted_frames(clip["path"]) == 29


# ---------------------------------------------------------------------------
# A CHAIN beat: the terminal transaction, inside the loop
# ---------------------------------------------------------------------------

def test_a_CHAIN_successor_begins_on_its_PREDECESSORS_terminal_frame(monkeypatch):
    """The whole reason the transaction is inside the loop: segment 1 cannot
    wait for a post-episode pass to learn where it starts."""
    with tempfile.TemporaryDirectory() as tmp:
        eng = _BeatStub(tmp)
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_CHAIN, [
            {"index": 0, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 25, "drop_head": 1, "trim_tail": 0},
        ], 49)
        clip, _s, _a, _u = rd.render_beat_coverage(
            _shot(plan), {}, request_builder=_request_builder)

        assert eng.seen_inits[0] == "seg0_still.png"
        terminal = eng.seen_inits[1]
        assert terminal.endswith(".png") and terminal != "seg1_still.png", (
            "segment 1 used its minted still instead of the chain terminal "
            "frame -- that is a jump cut wearing a chain's name")
        assert os.path.exists(terminal), "the terminal frame must be on disk"
        # The successor's duplicated first frame is dropped, so the beat is 49.
        assert ws.ffprobe_counted_frames(clip["path"]) == 49
        assert eng.load_calls == 1


def test_a_JUMP_beat_never_extracts_a_terminal_frame(monkeypatch):
    """A jump segment has its own still; extracting a terminal frame for it
    would be work that changes nothing."""
    calls = []
    real = wb.extract_terminal_frame
    with tempfile.TemporaryDirectory() as tmp:
        eng = _BeatStub(tmp)
        _install(monkeypatch, eng)
        monkeypatch.setattr(
            rd_wb_module(), "extract_terminal_frame",
            lambda *a, **k: calls.append(a) or real(*a, **k))
        plan = _plan(cp.JOIN_JUMP, [
            {"index": 0, "render_frames": 10, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 10, "drop_head": 0, "trim_tail": 0},
        ], 20)
        rd.render_beat_coverage(_shot(plan), {},
                                request_builder=_request_builder)
    assert calls == []


def rd_wb_module():
    from nodes._otr_video_engines import wrapper_bridge as _m
    return _m


# ---------------------------------------------------------------------------
# The single-clip path is untouched
# ---------------------------------------------------------------------------

def test_a_single_clip_beat_takes_the_HISTORICAL_path(monkeypatch):
    """Every beat today. One request -- the one the caller already built --
    one render, no assembly."""
    with tempfile.TemporaryDirectory() as tmp:
        eng = _BeatStub(tmp)
        _install(monkeypatch, eng)
        shot = {"shot_id": "shot_b001", "engine_id": "stub_beat_engine",
                "family": "abstract", "target_frame_count": 10}
        prebuilt = {"shot_id": "shot_b001", "segment_index": 0, "frames": 10,
                    "init_image": "beat_still.png", "observability": {}}
        clip, _s, _a, _u = rd.render_beat_coverage(
            shot, {}, request=prebuilt,
            request_builder=lambda *a, **k: pytest.fail(
                "the prebuilt request was rebuilt"))
        assert eng.seen_inits == ["beat_still.png"]
        assert "segment_count" not in clip, "a single-clip beat was assembled"
        assert eng.load_calls == 1 and eng.teardown_calls == 1


def test_a_multi_segment_beat_with_no_request_builder_is_TERMINAL(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        eng = _BeatStub(tmp)
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_JUMP, [
            {"index": 0, "render_frames": 10, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 10, "drop_head": 0, "trim_tail": 0},
        ], 20)
        with pytest.raises(rd.RenderError, match="no request builder"):
            rd.render_beat_coverage(_shot(plan), {}, request={"x": 1})


# ---------------------------------------------------------------------------
# The assembly is TRANSACTIONAL
# ---------------------------------------------------------------------------

def test_an_assembled_beat_of_the_WRONG_LENGTH_is_refused_and_removed():
    """The other end of the partitioner's promise. A header would have said
    this was fine."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        a = os.path.join(tmp, "a.mp4")
        wb.encode_frames_to_silent_mp4(
            [np.zeros((64, 64, 3), dtype=np.uint8)] * 10, a, 25)
        out = os.path.join(tmp, "beat.mp4")
        with pytest.raises(wb.GraphExecutionError, match="promised 99"):
            ws.assemble_beat_segments([(a, 0, 10)], out,
                                      expect_frames=99, expect_fps=25)
        assert not os.path.exists(out), (
            "a beat that failed verification survived on disk, where the next "
            "pass would find it and trust it")


def test_segments_with_MIXED_canvases_are_refused():
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        a = os.path.join(tmp, "a.mp4")
        b = os.path.join(tmp, "b.mp4")
        wb.encode_frames_to_silent_mp4(
            [np.zeros((64, 64, 3), dtype=np.uint8)] * 5, a, 25)
        wb.encode_frames_to_silent_mp4(
            [np.zeros((64, 96, 3), dtype=np.uint8)] * 5, b, 25)
        out = os.path.join(tmp, "beat.mp4")
        with pytest.raises(wb.GraphExecutionError, match="MIXED canvases"):
            ws.assemble_beat_segments([(a, 0, 5), (b, 0, 5)], out,
                                      expect_frames=10, expect_fps=25)


def test_the_assembled_beat_is_a_CanonicalClip_like_any_other():
    """bt709 / yuv420p / one video stream / no audio -- the mux is entitled to
    assume it."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        a = os.path.join(tmp, "a.mp4")
        wb.encode_frames_to_silent_mp4(
            [np.zeros((64, 64, 3), dtype=np.uint8)] * 6, a, 25)
        out = os.path.join(tmp, "beat.mp4")
        ws.assemble_beat_segments([(a, 0, 6)], out,
                                  expect_frames=6, expect_fps=25)
        ws.validate_silent_clip_contract(ws.ffprobe_clip_fields(out), 25)
