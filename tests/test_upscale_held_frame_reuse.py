"""Held-frame memoization in `_run_model_pipeline` (operator, 2026-08-25).

A `still_*` beat is ONE image held across the whole segment, but the upscale
runs on the DECODED VIDEO, so Real-ESRGAN was re-upscaling the same picture
once per frame -- measured at 3-4 minutes per segment across 18+ segments,
turning a 10-20 minute episode into 105+.

The engine is deterministic (`.eval()` + `torch.inference_mode()`, a plain
conv net) and `_fit_and_pad_bhwc` is pure geometry, so identical input bytes
must produce identical output bytes. Reusing the previous result is therefore
MEMOIZATION, not an approximation.

These tests are BEHAVIOURAL: they drive the real `_run_model_pipeline` loop
with a fake engine that COUNTS model calls, and assert both halves of the
contract -- fewer calls AND byte-identical output. A source-grep test would
pass against a cache that silently never fires, which is the failure mode
this file exists to catch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

pytest.importorskip("numpy")
pytest.importorskip("torch")

import numpy as np  # noqa: E402
import torch  # noqa: E402


class CountingEngine:
    """A deterministic 2x nearest-neighbour 'model' that counts its calls.

    Deterministic on purpose: the whole correctness argument for the cache is
    that the same input yields the same output, so the stand-in must have that
    property too or the test proves nothing about reuse.
    """

    name = "counting_test_engine"
    intrinsic_scale = 2

    def __init__(self):
        self.calls = 0

    @property
    def device(self):
        return torch.device("cpu")

    def upscale_frames(self, frames):
        self.calls += 1
        # BHWC nearest-neighbour x2 -- deterministic, no learned weights.
        return frames.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)


def _run(monkeypatch, frames_bgr, *, n_frames, src_w, src_h, out_w, out_h):
    """Drive the real pipeline loop against fake ffmpeg pipes.

    Returns (written_bytes, engine.calls).
    """
    import nodes.otr_silent_composite as sc

    engine = CountingEngine()
    payload = b"".join(f.tobytes() for f in frames_bgr)

    class FakeStdout:
        def __init__(self, data):
            self._buf = memoryview(data)
            self._pos = 0

        def read(self, n):
            chunk = bytes(self._buf[self._pos:self._pos + n])
            self._pos += len(chunk)
            return chunk

    class FakeStdin:
        def __init__(self):
            self.chunks = []
            self.closed_flag = False

        def write(self, b):
            self.chunks.append(bytes(b))

        def close(self):
            self.closed_flag = True

    class FakeProc:
        def __init__(self, *, stdout=None, stdin=None):
            self.stdout = stdout
            self.stdin = stdin
            self.returncode = 0

        def wait(self, timeout=None):
            return 0

    dec = FakeProc(stdout=FakeStdout(payload))
    enc = FakeProc(stdin=FakeStdin())

    popen_calls = {"n": 0}

    def fake_popen(args, **kwargs):
        popen_calls["n"] += 1
        return dec if popen_calls["n"] == 1 else enc

    monkeypatch.setattr(sc.subprocess, "Popen", fake_popen)
    # The pipeline asserts the encoded segment's frame count; there is no real
    # file here, so return exactly what the loop was asked to write.
    monkeypatch.setattr(sc, "count_video_frames", lambda p: int(n_frames))
    # _validate_engine_output is imported INSIDE the function, so it is not a
    # module attribute to patch -- and it should run for real anyway: the
    # CountingEngine returns a true 2x upscale, so the genuine validator is a
    # free extra assertion that the cache did not corrupt the tensor shape.
    # _unlink_if_exists is also a function-local import, not a module
    # attribute; the success path never reaches it anyway.

    sc._run_model_pipeline(
        fb="ffmpeg", src="src.mp4", seg_path="seg_0000.mp4",
        engine=engine, w=out_w, h=out_h, fps=25,
        n_frames=n_frames, start_frame=0, loop=False,
        src_w=src_w, src_h=src_h,
    )
    return b"".join(enc.stdin.chunks), engine.calls


def _solid(w, h, value):
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_identical_held_frames_collapse_to_one_model_call(monkeypatch):
    """The still_flat case: one held image, N frames. THE point of the change."""
    n = 12
    frames = [_solid(8, 4, 77) for _ in range(n)]
    out, calls = _run(monkeypatch, frames, n_frames=n,
                      src_w=8, src_h=4, out_w=16, out_h=8)
    assert calls == 1, (
        f"held-frame reuse did not fire: {calls} model calls for {n} identical "
        "frames (expected exactly 1). A cache that never hits is the silent "
        "failure this test exists to catch.")
    assert len(out) == n * 16 * 8 * 3, "wrong number of frames written"


def test_reused_frames_are_byte_identical_to_recomputing_them(monkeypatch):
    """Memoization, not approximation: the cached bytes must equal the bytes a
    fresh model call would have produced for that same frame."""
    n = 6
    frames = [_solid(8, 4, 123) for _ in range(n)]
    cached_out, cached_calls = _run(monkeypatch, frames, n_frames=n,
                                    src_w=8, src_h=4, out_w=16, out_h=8)
    # Same content, but every frame differs by one pixel so the cache cannot
    # fire -- forcing a full model call per frame.
    forced = []
    for i in range(n):
        f = _solid(8, 4, 123)
        f[0, 0, 0] = i  # unique per frame
        forced.append(f)
    _, forced_calls = _run(monkeypatch, forced, n_frames=n,
                           src_w=8, src_h=4, out_w=16, out_h=8)
    assert cached_calls == 1 and forced_calls == n, (
        "control failed: expected 1 cached call vs n forced calls, got "
        f"{cached_calls} vs {forced_calls}")
    # Recompute frame 0 alone through the same path; every cached frame must
    # equal it exactly.
    single_out, _ = _run(monkeypatch, [_solid(8, 4, 123)], n_frames=1,
                         src_w=8, src_h=4, out_w=16, out_h=8)
    frame_bytes = 16 * 8 * 3
    for i in range(n):
        assert cached_out[i * frame_bytes:(i + 1) * frame_bytes] == single_out, (
            f"reused frame {i} is not byte-identical to a fresh computation -- "
            "the cache changed output, which it must never do")


def test_moving_frames_never_reuse(monkeypatch):
    """A genuinely moving lane (pan/motion) must still call the model per
    frame. The cache must not fire on merely SIMILAR frames."""
    n = 5
    frames = []
    for i in range(n):
        f = _solid(8, 4, 40)
        f[0, 0, 0] = i  # one pixel differs -> a different picture
        frames.append(f)
    _, calls = _run(monkeypatch, frames, n_frames=n,
                    src_w=8, src_h=4, out_w=16, out_h=8)
    assert calls == n, (
        f"cache fired on non-identical frames ({calls} calls for {n} distinct "
        "frames) -- that would emit the wrong picture")


def test_reuse_resumes_after_an_interruption(monkeypatch):
    """A->A->B->A: the slot holds only the PREVIOUS frame, so the trailing A
    must be recomputed rather than served from the stale first A."""
    frames = [_solid(8, 4, 10), _solid(8, 4, 10),
              _solid(8, 4, 20), _solid(8, 4, 10)]
    out, calls = _run(monkeypatch, frames, n_frames=4,
                      src_w=8, src_h=4, out_w=16, out_h=8)
    assert calls == 3, (
        f"expected 3 model calls for A,A,B,A (one reuse), got {calls}")
    frame_bytes = 16 * 8 * 3
    assert out[0:frame_bytes] == out[frame_bytes:2 * frame_bytes], (
        "the two adjacent A frames should be identical")
    assert out[0:frame_bytes] == out[3 * frame_bytes:4 * frame_bytes], (
        "the trailing A must equal the leading A (recomputed, same input)")
