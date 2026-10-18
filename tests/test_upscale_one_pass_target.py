"""A target-size engine is asked for the composite's own size (2026-10-18).

rtx_vsr was a fixed x2: a 512x288 frame got x2 from VSR and the remaining
1.875x to 1080p from a bicubic resize. `_run_model_pipeline` now hands an
engine that has `upscale_frames_to` the exact canvas on a full-frame row, or
the aspect-fit size on a padded row, and falls back to x2 when it answers
None. These drive the real loop against fake ffmpeg pipes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402


class TargetEngine:
    """Nearest-neighbour to any size; records what it was asked for."""

    name = "target_test_engine"
    intrinsic_scale = 2

    def __init__(self, answer=True):
        self.answer = answer
        self.asked = []
        self.x2_calls = 0

    @property
    def device(self):
        return torch.device("cpu")

    def upscale_frames(self, frames):
        self.x2_calls += 1
        return frames.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

    def upscale_frames_to(self, frames, out_w, out_h):
        self.asked.append((out_w, out_h))
        if not self.answer:
            return None
        x = frames.permute(0, 3, 1, 2)
        x = torch.nn.functional.interpolate(x, size=(out_h, out_w), mode="nearest")
        return x.permute(0, 2, 3, 1).contiguous()


def _run(monkeypatch, engine, *, n, src_w, src_h, out_w, out_h, full_frame):
    import nodes.otr_silent_composite as sc

    frame = np.full((src_h, src_w, 3), 90, dtype=np.uint8)
    frame[0, 0] = (255, 0, 0)
    payload = b"".join(frame.tobytes() for _ in range(n))

    class Out:
        def __init__(self, data):
            self.b, self.p = memoryview(data), 0

        def read(self, k):
            c = bytes(self.b[self.p:self.p + k])
            self.p += len(c)
            return c

    class In:
        def __init__(self):
            self.chunks = []

        def write(self, b):
            self.chunks.append(bytes(b))

        def close(self):
            pass

    class Proc:
        def __init__(self, stdout=None, stdin=None):
            self.stdout, self.stdin, self.returncode = stdout, stdin, 0

        def wait(self, timeout=None):
            return 0

    dec, enc = Proc(stdout=Out(payload)), Proc(stdin=In())
    calls = {"n": 0}

    def popen(args, **kw):
        calls["n"] += 1
        return dec if calls["n"] == 1 else enc

    monkeypatch.setattr(sc.otr_proc, "popen", popen)
    monkeypatch.setattr(sc, "count_video_frames", lambda p: int(n))
    sc._run_model_pipeline(
        fb="ffmpeg", src="src.mp4", seg_path="seg_0000.mp4", engine=engine,
        w=out_w, h=out_h, fps=25, n_frames=n, start_frame=0, loop=False,
        src_w=src_w, src_h=src_h, full_frame=full_frame)
    return b"".join(enc.stdin.chunks)


def test_a_full_frame_row_asks_for_the_exact_canvas(monkeypatch):
    eng = TargetEngine()
    out = _run(monkeypatch, eng, n=3, src_w=8, src_h=4, out_w=48, out_h=24,
               full_frame=True)
    assert eng.asked == [(48, 24)]           # held-frame reuse: one call
    assert eng.x2_calls == 0
    assert len(out) == 3 * 48 * 24 * 3


def test_a_padded_row_asks_for_the_aspect_fit_size(monkeypatch):
    eng = TargetEngine()
    # 8x4 (2:1) into a 48x36 canvas: the fit is 48x24, padded to 48x36.
    out = _run(monkeypatch, eng, n=2, src_w=8, src_h=4, out_w=48, out_h=36,
               full_frame=False)
    assert eng.asked == [(48, 24)]
    assert len(out) == 2 * 48 * 36 * 3


def test_none_falls_back_to_the_fixed_x2_path(monkeypatch):
    eng = TargetEngine(answer=False)
    out = _run(monkeypatch, eng, n=2, src_w=8, src_h=4, out_w=48, out_h=24,
               full_frame=True)
    assert eng.asked == [(48, 24)] and eng.x2_calls == 1
    assert len(out) == 2 * 48 * 24 * 3


def test_rtx_vsr_declines_what_one_pass_cannot_reach():
    """Below the frame or past ONE_PASS_MAX_SCALE (2x, measured) -> None."""
    from nodes._otr_upscale_engines.eng_rtx_vsr import ONE_PASS_MAX_SCALE as MAX_SCALE, RtxVsr
    eng = RtxVsr()
    eng._nvvfx = object()                    # loaded, but never reaches the SDK
    f = torch.zeros((1, 100, 200, 3))
    assert eng.upscale_frames_to(f, 150, 100) is None          # narrower
    assert eng.upscale_frames_to(f, 200 * MAX_SCALE + 8, 400) is None
    assert eng.upscale_frames_to(f, 400, 100 * MAX_SCALE + 8) is None


def test_each_segment_logs_which_branch_ran(monkeypatch, caplog):
    """The MODEL PATH line reads the same for both branches; the 8 GB leg on
    c7c3e375 could not tell them apart (2026-10-18). One INFO line names it."""
    import logging
    with caplog.at_level(logging.INFO, logger="OTR"):
        _run(monkeypatch, TargetEngine(), n=3, src_w=8, src_h=4,
             out_w=48, out_h=24, full_frame=True)
    lines = [r.getMessage() for r in caplog.records if "one pass to 48x24" in r.getMessage()]
    assert len(lines) == 1, [r.getMessage() for r in caplog.records]
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="OTR"):
        _run(monkeypatch, TargetEngine(answer=False), n=2, src_w=8, src_h=4,
             out_w=48, out_h=24, full_frame=True)
    assert any("x2 then resize" in r.getMessage() for r in caplog.records)
