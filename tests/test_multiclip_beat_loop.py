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
        # READ IT WHERE A REAL ADAPTER READS IT. ``build_request`` puts the init
        # image at ``asset_refs["init_image"]``; an earlier version of this stub
        # read a top-level key, which meant the stub agreed with a driver bug
        # that wrote the terminal frame somewhere no engine looks.
        self.seen_inits.append(
            (request.get("asset_refs") or {}).get("init_image") or "")
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
            # The SAME shape build_request emits -- asset_refs, not a top-level
            # key -- so the stub cannot agree with a driver that writes to the
            # wrong place.
            "asset_refs": {"init_image": "seg%d_still.png" % int(segment_index)},
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
        # THE REAL SHAPE: ShotLock stamps a one-segment JOIN_SINGLE plan on
        # EVERY beat, so the live branch is `is_multi_clip == False` with a real
        # plan -- not an absent key. Testing only the absent-key half would have
        # left the branch every beat actually takes uncovered (QA panel).
        shot = {"shot_id": "shot_b001", "engine_id": "stub_beat_engine",
                "family": "abstract", "target_frame_count": 10,
                "coverage_plan": _plan(cp.JOIN_SINGLE, [
                    {"index": 0, "render_frames": 10, "drop_head": 0,
                     "trim_tail": 0}], 10)}
        prebuilt = {"shot_id": "shot_b001", "segment_index": 0, "frames": 10,
                    "asset_refs": {"init_image": "beat_still.png"},
                    "observability": {}}
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
        with pytest.raises(wb.GraphExecutionError, match="MIXED shape"):
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


# ---------------------------------------------------------------------------
# The failure paths (neither panel found these covered -- they were right)
# ---------------------------------------------------------------------------

def test_a_segment_that_RAISES_mid_beat_still_closes_the_session(monkeypatch):
    """The session is a `with`, but nothing asserted it. A stranded GPU lease
    is the next episode's mystery hang."""
    with tempfile.TemporaryDirectory() as tmp:
        eng = _BeatStub(tmp)
        real_render = eng.render_clip

        def _boom(request, prepared):
            if int(request["segment_index"]) == 1:
                raise RuntimeError("segment 1 exploded")
            return real_render(request, prepared)

        eng.render_clip = _boom
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_JUMP, [
            {"index": 0, "render_frames": 10, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 10, "drop_head": 0, "trim_tail": 0},
        ], 20)
        with pytest.raises(rd.RenderError, match="fallbacks are disabled"):
            rd.render_beat_coverage(_shot(plan), {},
                                    request_builder=_request_builder)
        assert eng.teardown_calls == 1, "the beat session leaked its handles"
        assert eng.load_calls == 1


def test_a_segment_that_returns_NO_PATH_is_TERMINAL(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        eng = _BeatStub(tmp)
        eng.render_clip = lambda request, prepared: {"frame_count": 10,
                                                     "fps": 25}
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_JUMP, [
            {"index": 0, "render_frames": 10, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 10, "drop_head": 0, "trim_tail": 0},
        ], 20)
        with pytest.raises(rd.RenderError, match="rendered no clip path"):
            rd.render_beat_coverage(_shot(plan), {},
                                    request_builder=_request_builder)
        assert eng.teardown_calls == 1


def test_a_SHORT_segment_is_named_at_the_segment_not_at_the_assembly(monkeypatch):
    """An engine that returns fewer frames than it was asked for used to
    surface much later as an assembly count mismatch, which reads as an
    assembler bug."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        eng = _BeatStub(tmp)

        def _short(request, prepared):
            imgs = [np.zeros((64, 64, 3), dtype=np.uint8)] * 4
            out = os.path.join(tmp, "short.mp4")
            path, n = wb.encode_frames_to_silent_mp4(imgs, out, 25)
            return {"path": path, "frame_count": n, "fps": 25}

        eng.render_clip = _short
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_JUMP, [
            {"index": 0, "render_frames": 10, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 10, "drop_head": 0, "trim_tail": 0},
        ], 20)
        with pytest.raises(rd.RenderError, match="rendered 4 frame"):
            rd.render_beat_coverage(_shot(plan), {},
                                    request_builder=_request_builder)
        assert eng.teardown_calls == 1


def test_the_beats_VRAM_peak_is_the_max_across_segments_not_the_last(monkeypatch):
    """Taking whatever the final segment reported under-reports a beat whose
    heaviest render was segment 0."""
    with tempfile.TemporaryDirectory() as tmp:
        eng = _BeatStub(tmp)
        _install(monkeypatch, eng)
        peaks = iter([900, 100])
        real = eng.render_clip
        eng.render_clip = lambda rq, pr: dict(real(rq, pr),
                                              vram_peak_mb=next(peaks))
        plan = _plan(cp.JOIN_JUMP, [
            {"index": 0, "render_frames": 10, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 10, "drop_head": 0, "trim_tail": 0},
        ], 20)
        _clip, _s, _a, used = rd.render_beat_coverage(
            _shot(plan), {}, request_builder=_request_builder)
        assert used == 900, "the beat reported its LAST segment's peak"


def test_a_FAILED_CONCAT_leaves_nothing_behind(monkeypatch):
    """The transaction covers the ffmpeg call itself, not only the checks
    after it -- a failing concat is the failure most likely to leave a partial
    file (QA panel)."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        a = os.path.join(tmp, "a.mp4")
        wb.encode_frames_to_silent_mp4(
            [np.zeros((64, 64, 3), dtype=np.uint8)] * 5, a, 25)
        out = os.path.join(tmp, "beat.mp4")

        def _fake_run(cmd):
            open(out, "wb").write(b"partial garbage")
            raise wb.GraphExecutionError("ffmpeg failed rc=1: synthetic")

        monkeypatch.setattr(wb, "run_ffmpeg", _fake_run)
        with pytest.raises(wb.GraphExecutionError, match="synthetic"):
            ws.assemble_beat_segments([(a, 0, 5)], out,
                                      expect_frames=5, expect_fps=25)
        assert not os.path.exists(out), (
            "a partial file from a failed concat survived on disk")


def test_segments_at_DIFFERENT_FPS_are_refused():
    """Same canvas, different frame rate -- as invisible in a header as a size
    change and as visible on screen."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        a = os.path.join(tmp, "a.mp4")
        b = os.path.join(tmp, "b.mp4")
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)] * 5
        wb.encode_frames_to_silent_mp4(frames, a, 25)
        wb.encode_frames_to_silent_mp4(frames, b, 30)
        out = os.path.join(tmp, "beat.mp4")
        with pytest.raises(wb.GraphExecutionError, match="MIXED shape"):
            ws.assemble_beat_segments([(a, 0, 5), (b, 0, 5)], out,
                                      expect_frames=10, expect_fps=25)


def test_a_segment_that_keeps_NO_frames_is_refused_not_clamped():
    with pytest.raises(wb.GraphExecutionError, match="NO CLAMP"):
        wb.ffmpeg_concat_segments_cmd([("a.mp4", 0, 0)], "out.mp4")
    with pytest.raises(wb.GraphExecutionError, match="NO CLAMP"):
        wb.ffmpeg_concat_segments_cmd([("a.mp4", -1, 5)], "out.mp4")


# ---------------------------------------------------------------------------
# C2 (2026-07-27): a segment that cannot SAY what it produced is not assemblable
# ---------------------------------------------------------------------------
#
# The per-segment length check read
#     got = int((clip or {}).get("frame_count") or 0)
#     if got and got != int(segment.render_frames):
# so a clip reporting 0 -- or omitting the field -- skipped the check entirely
# and was assembled anyway. `CanonicalClip.frame_count` DEFAULTS to 0
# (schemas.py:233), so "absent" and "zero" are the same value and neither is a
# length. A fail-OPEN guard inside a fail-closed function.
#
# This matters most for the lanes nobody can smoke-test cheaply: four provider
# canonicalizers derive frame_count as round(duration_s * fps), so a provider
# returning a zero/absent duration produced a zero count that sailed straight
# through the proof and into the assembly.


class _BadCountStub(_BeatStub):
    """Writes a REAL clip of the right length but MISREPORTS the count.

    Deliberately still writes the correct number of frames: the point is that
    the driver must not trust an unreadable *report*, even when the file behind
    it happens to be fine. Otherwise the test would be proving the assembler,
    not the guard.
    """

    def __init__(self, tmpdir, bad_value, omit=False):
        super().__init__(tmpdir)
        self.bad_value = bad_value
        self.omit = omit

    def render_clip(self, request, prepared):
        out = super().render_clip(request, prepared)
        if self.omit:
            out.pop("frame_count", None)
        else:
            out["frame_count"] = self.bad_value
        return out


@pytest.mark.parametrize("bad_value,omit,label", [
    (0, False, "zero"),
    (None, False, "None"),
    ("", False, "empty string"),
    ("not-a-number", False, "non-numeric"),
    (-3, False, "negative"),
    (None, True, "field absent entirely"),
])
def test_an_unreadable_segment_frame_count_is_TERMINAL(monkeypatch, bad_value,
                                                       omit, label):
    """THE MUTATION TARGET. Restore `if got and got != ...` and every one of
    these assembles silently instead of raising."""
    with tempfile.TemporaryDirectory() as tmp:
        eng = _BadCountStub(tmp, bad_value, omit=omit)
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_JUMP, [
            {"index": 0, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
        ], 50)
        with pytest.raises(rd.RenderError) as exc:
            rd.render_beat_coverage(_shot(plan), {},
                                    request_builder=_request_builder)
        msg = str(exc.value)
        assert "is not a length" in msg, (
            "the refusal for %s must name the unreadable count, not fall "
            "through to the generic mismatch message: %s" % (label, msg))
        assert "shot_b001" in msg and "segment 0" in msg, msg


def test_a_HONEST_count_still_passes_after_the_C2_tightening(monkeypatch):
    """The guard must not have become a blanket refusal.

    Pairs with the parametrized test above: without this, tightening the
    predicate to `raise` unconditionally would also go green.
    """
    with tempfile.TemporaryDirectory() as tmp:
        eng = _BeatStub(tmp)
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_JUMP, [
            {"index": 0, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
        ], 50)
        clip, _shot_out, _att, _used = rd.render_beat_coverage(
            _shot(plan), {}, request_builder=_request_builder)
        assert clip["frame_count"] == 50
        assert eng.seen_frames == [25, 25]


def test_a_WRONG_but_readable_count_still_names_the_mismatch(monkeypatch):
    """The pre-existing mismatch path must survive the tightening -- a readable
    count that disagrees still gets the 'asked for' message, not the new one."""
    with tempfile.TemporaryDirectory() as tmp:
        eng = _BadCountStub(tmp, 24)          # readable, but one short
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_JUMP, [
            {"index": 0, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
        ], 50)
        with pytest.raises(rd.RenderError) as exc:
            rd.render_beat_coverage(_shot(plan), {},
                                    request_builder=_request_builder)
        msg = str(exc.value)
        assert "rendered 24 frame(s) but its plan asked for 25" in msg, msg
        assert "is not a length" not in msg, (
            "a readable-but-wrong count must take the mismatch path, not the "
            "unreadable-count path: %s" % msg)


# ---------------------------------------------------------------------------
# PER-SEGMENT IDENTITY (operator ask, 2026-08-01)
#
# A split beat used to record only its aggregate, while the real renders behind
# it landed on disk under random hex. These tests pin the receipt that makes a
# split beat auditable: one row per render, in order, each naming itself and
# where its first frame came from.
# ---------------------------------------------------------------------------

def test_every_segment_of_a_split_beat_gets_an_ID_and_the_frames_ADD_UP(monkeypatch):
    """The receipt has to be checkable, not merely present.

    ``sum(visible_frames)`` over the rows must equal the beat's target, which is
    the arithmetic behind "a fresh clip for every second of audio" -- if the
    rows summed to less, some of the beat's audio would be playing over frames
    no row accounts for.
    """
    with tempfile.TemporaryDirectory() as tmp:
        eng = _BeatStub(tmp)
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_JUMP, [
            {"index": 0, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
            {"index": 2, "render_frames": 25, "drop_head": 0, "trim_tail": 5},
        ], 70)
        clip, _s, _a, _u = rd.render_beat_coverage(
            _shot(plan), {}, request_builder=_request_builder)

        rows = clip["segments"]
        assert len(rows) == 3
        assert [r["segment_index"] for r in rows] == [0, 1, 2]
        # Ids are stable, ordered, and derived from the beat -- not random hex.
        assert [r["segment_id"] for r in rows] == [
            "b001_seg00", "b001_seg01", "b001_seg02"]
        assert all(r["beat_id"] == "b001" for r in rows)
        assert all(r["segment_count"] == 3 for r in rows)
        # The whole point: the rows account for the beat, frame for frame.
        assert sum(r["visible_frames"] for r in rows) == clip["frame_count"] == 70
        # Each row names a real file on disk, and the names are distinct.
        assert len({r["path"] for r in rows}) == 3


def test_a_CHAIN_beats_rows_say_WHERE_each_first_frame_CAME_FROM(monkeypatch):
    """Only segment 0 is minted from a still; every successor chains.

    Without this the ledger could not distinguish a real chain from a beat that
    silently re-minted a still at each cut.
    """
    with tempfile.TemporaryDirectory() as tmp:
        eng = _BeatStub(tmp)
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_CHAIN, [
            {"index": 0, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 25, "drop_head": 1, "trim_tail": 0},
            {"index": 2, "render_frames": 25, "drop_head": 1, "trim_tail": 0},
        ], 73)
        clip, _s, _a, _u = rd.render_beat_coverage(
            _shot(plan), {}, request_builder=_request_builder)

        rows = clip["segments"]
        assert [r["init_source"] for r in rows] == [
            "still", "chain_terminal_frame", "chain_terminal_frame"]
        assert sum(r["visible_frames"] for r in rows) == clip["frame_count"] == 73


def test_the_FEAR_CAPE_is_stamped_on_the_segment_that_WEARS_it(monkeypatch):
    """The cape is painted on the still feeding the LAST segment, so the row
    that carries the flag must be the last one -- not the one that produced the
    frame. A receipt naming the wrong clip is worse than no receipt.
    """
    monkeypatch.delenv("OTR_FEAR_CAPE", raising=False)
    monkeypatch.delenv("OTR_FEAR_CAPE_MIN_SEGMENTS", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        eng = _BeatStub(tmp)
        _install(monkeypatch, eng)
        # 4 segments == the default fear-cape threshold.
        plan = _plan(cp.JOIN_CHAIN, [
            {"index": 0, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 25, "drop_head": 1, "trim_tail": 0},
            {"index": 2, "render_frames": 25, "drop_head": 1, "trim_tail": 0},
            {"index": 3, "render_frames": 25, "drop_head": 1, "trim_tail": 0},
        ], 97)
        clip, _s, _a, _u = rd.render_beat_coverage(
            _shot(plan), {}, request_builder=_request_builder)

        rows = clip["segments"]
        assert [r["fear_cape"] for r in rows] == [False, False, False, True], (
            "the cape must be stamped on the FINAL segment, the one rendered "
            "from the inverted still")


def test_a_SHORT_beat_stays_below_the_cape_threshold(monkeypatch):
    """Three segments is under the 4-segment rule of thumb, so no cape."""
    monkeypatch.delenv("OTR_FEAR_CAPE", raising=False)
    monkeypatch.delenv("OTR_FEAR_CAPE_MIN_SEGMENTS", raising=False)
    with tempfile.TemporaryDirectory() as tmp:
        eng = _BeatStub(tmp)
        _install(monkeypatch, eng)
        plan = _plan(cp.JOIN_CHAIN, [
            {"index": 0, "render_frames": 25, "drop_head": 0, "trim_tail": 0},
            {"index": 1, "render_frames": 25, "drop_head": 1, "trim_tail": 0},
            {"index": 2, "render_frames": 25, "drop_head": 1, "trim_tail": 0},
        ], 73)
        clip, _s, _a, _u = rd.render_beat_coverage(
            _shot(plan), {}, request_builder=_request_builder)
        assert not any(r["fear_cape"] for r in clip["segments"])
