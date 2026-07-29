"""WIRE-W4d -- the segment requests are built BEFORE the GPU lease is taken.

r3, in as many words: *"Prebuild and validate all segment requests and audio
slices BEFORE entering BeatSession; only terminal-image chaining stays in the
render loop."*

WHY IT MATTERS RATHER THAN BEING TIDY. The request builder is neither cheap nor
pure: it resolves stills off the ledger and SHELLS OUT TO FFMPEG to cut each
segment's conditioning WAV out of the frozen master. Run inside the session,
that filesystem and subprocess work happened while the cross-process GPU lease
was held and a 14B UNET sat resident -- once per segment, between renders. The
lease is the scarcest thing this build owns, and every heavy render on the box
blocks its full 120 s timeout behind it.

It is also where a bad request SHOULD surface. A builder that raises on segment
2 used to do it after two renders and a 6 GiB load.

THE CHAIN'S TERMINAL FRAME IS THE ONE EXCEPTION, and it is not an oversight:
segment N's init image is segment N-1's last RENDERED frame, which does not
exist until segment N-1 has run.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes._otr_video_engines import coverage_plan as cp
from nodes._otr_video_engines import render_driver as rd
from nodes._otr_video_engines import wrapper_bridge as wb


class _OrderedStub:
    """A beat engine that RECORDS THE ORDER of every lifecycle event, which is
    the only thing this chunk changes."""

    name = "w4d_stub_engine"
    family = "abstract"
    accepts_still = True
    render_aspect = "wide"
    target_fps = 25

    def __init__(self, tmpdir, log):
        self.tmpdir = tmpdir
        self.log = log

    def session_identity(self):
        return ("w4d_stub_engine", "recipe", "weights")

    def load(self):
        self.log.append("load")

    def prepare(self, host_caps, profile, session_ctx):
        self.log.append("prepare")
        return {"engine_id": self.name}

    def assert_usable(self, host_caps, profile, request_template=None):
        return True

    def render_clip(self, request, prepared):
        import numpy as np
        self.log.append("render:%d" % int(request["segment_index"]))
        frames = int(request["frames"])
        imgs = [np.full((32, 32, 3), 60, dtype=np.uint8) for _ in range(frames)]
        out = os.path.join(self.tmpdir,
                           "seg%d.mp4" % int(request["segment_index"]))
        path, n = wb.encode_frames_to_silent_mp4(imgs, out, 25)
        return {"path": path, "frame_count": n, "fps": 25}

    def canonicalize(self, raw, request, profile):
        return dict(raw, type="video")

    def teardown(self, prepared):
        self.log.append("teardown")


def _plan(join_mode, segments, target):
    return {"target_visible_frames": target, "join_mode": join_mode,
            "segments": [dict(s) for s in segments]}


def _shot(plan):
    return {"shot_id": "shot_b001", "beat_id": "b001",
            "engine_id": "w4d_stub_engine", "family": "abstract",
            "role": "character_video",
            "target_frame_count": plan["target_visible_frames"],
            "coverage_plan": plan}


def _install(monkeypatch, engine):
    monkeypatch.setattr(rd._vreg, "is_registered",
                        lambda name: name == engine.name)
    monkeypatch.setattr(rd._vreg, "get_engine", lambda name: engine)


def _builder(log):
    def _build(shot, ledger, *, canvas=None, segment_index=0):
        log.append("build:%d" % int(segment_index))
        return {"shot_id": shot["shot_id"],
                "segment_index": int(segment_index),
                "frames": rd.segment_render_frames(shot, segment_index),
                "asset_refs": {"init_image": "seg%d.png" % int(segment_index)},
                "observability": {}}
    return _build


_THREE = [
    {"index": 0, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
    {"index": 1, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
    {"index": 2, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
]


def test_EVERY_request_is_built_BEFORE_prepare_takes_the_LEASE(monkeypatch):
    """THE test this chunk exists to pass. All three builds, then one prepare,
    then three renders -- never build/render interleaved, which is what put
    ffmpeg inside the lease."""
    log = []
    with tempfile.TemporaryDirectory() as tmp:
        eng = _OrderedStub(tmp, log)
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_JUMP, _THREE, 75)
        rd.render_beat_coverage(_shot(plan), {},
                                request_builder=_builder(log))
    assert log == ["build:0", "build:1", "build:2", "prepare",
                   "render:0", "render:1", "render:2", "teardown"], log


def test_a_BUILDER_THAT_RAISES_never_reaches_prepare(monkeypatch):
    """A builder that fails on segment 2 used to do it after two renders and a
    6 GiB load, with the lease held. Now the beat refuses before anything is
    taken -- and the proof is that ``prepare`` never ran."""
    log = []

    def _build(shot, ledger, *, canvas=None, segment_index=0):
        log.append("build:%d" % int(segment_index))
        if int(segment_index) == 2:
            raise RuntimeError("segment 2 has no still")
        return {"shot_id": shot["shot_id"],
                "segment_index": int(segment_index), "frames": 25,
                "asset_refs": {"init_image": "x.png"}, "observability": {}}

    with tempfile.TemporaryDirectory() as tmp:
        eng = _OrderedStub(tmp, log)
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_JUMP, _THREE, 75)
        with pytest.raises(RuntimeError):
            rd.render_beat_coverage(_shot(plan), {}, request_builder=_build)
    assert log == ["build:0", "build:1", "build:2"], log
    assert "prepare" not in log, "the lease was taken for a beat that refused"
    assert "teardown" not in log


def test_a_CHAINED_beat_still_gets_its_TERMINAL_FRAME_in_the_loop(monkeypatch):
    """The one exception, and it is not an oversight: segment N's init image is
    segment N-1's last RENDERED frame, which does not exist until N-1 has run.
    So the prebuilt request is still REWRITTEN inside the loop -- prebuilding
    the rest must not have frozen that."""
    log = []
    seen = []

    def _build(shot, ledger, *, canvas=None, segment_index=0):
        log.append("build:%d" % int(segment_index))
        return {"shot_id": shot["shot_id"],
                "segment_index": int(segment_index),
                "frames": rd.segment_render_frames(shot, segment_index),
                "asset_refs": {"init_image": "minted%d.png" % int(segment_index)},
                "observability": {}}

    with tempfile.TemporaryDirectory() as tmp:
        eng = _OrderedStub(tmp, log)
        real_render = eng.render_clip

        def _spy(request, prepared):
            seen.append((request.get("asset_refs") or {}).get("init_image"))
            return real_render(request, prepared)

        eng.render_clip = _spy
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_CHAIN, [
            {"index": 0, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 25, "drop_head": 1, "trim_tail": 0},
        ], 49)
        rd.render_beat_coverage(_shot(plan), {}, request_builder=_build)

    assert log[:2] == ["build:0", "build:1"], "still prebuilt"
    assert seen[0] == "minted0.png"
    assert seen[1] != "minted1.png", (
        "a chained successor must begin on its predecessor's REAL terminal "
        "frame, not on a minted still")
    assert seen[1] and seen[1].endswith(".png")


def test_the_SEGMENT_ORDER_survives_the_prebuild(monkeypatch):
    """The loop used to read ``segment`` straight off ``plan.segments``; it now
    reads a prebuilt pair. A plan whose indices are not 0,1,2 in order is
    exactly where a positional shortcut would show up -- and the frame counts
    come off the SEGMENT, so a mismatched pairing renders the wrong lengths."""
    log = []
    with tempfile.TemporaryDirectory() as tmp:
        eng = _OrderedStub(tmp, log)
        frames_seen = []
        real_render = eng.render_clip

        def _spy(request, prepared):
            frames_seen.append(int(request["frames"]))
            return real_render(request, prepared)

        eng.render_clip = _spy
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_JUMP, [
            {"index": 0, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 49, "drop_head": 0, "trim_tail": 0},
            {"index": 2, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
        ], 99)
        rd.render_beat_coverage(_shot(plan), {},
                                request_builder=_builder(log))
    assert frames_seen == [25, 49, 25], (
        "each segment must render ITS OWN planned length after the prebuild")


def test_a_SINGLE_CLIP_beat_never_reaches_this_code_at_all(monkeypatch):
    """CONTROL. The prebuild sits inside the multi-clip branch; a single-clip
    beat returns through ``render_shot`` well before it, which is what keeps
    production's majority path byte-identical."""
    log = []
    with tempfile.TemporaryDirectory() as tmp:
        eng = _OrderedStub(tmp, log)
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_SINGLE, [
            {"index": 0, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
        ], 25)
        shot = _shot(plan)
        request = _builder(log)(shot, {}, segment_index=0)
        log.clear()
        rd.render_beat_coverage(shot, {}, request=request,
                                request_builder=_builder(log))
    assert "build:0" not in log, (
        "a single-clip beat uses the request the caller already built")
