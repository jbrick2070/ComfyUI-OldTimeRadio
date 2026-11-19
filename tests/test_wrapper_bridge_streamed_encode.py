"""Clip encode streams frames instead of holding extra full-size copies (O6).

images_to_uint8 built two full float32 copies of the clip before casting, and
the encoder copied the whole uint8 clip into one bytes object for
communicate(). Both now work a frame at a time and must produce the SAME bytes.
UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from nodes._otr_video_engines import wrapper_bridge as wb


def _reference_uint8(arr):
    """The conversion as it was, whole-batch, for the byte-identity check."""
    arr = np.clip(arr, 0.0, 1.0)
    return np.ascontiguousarray((arr * 255.0 + 0.5).astype(np.uint8))


def test_the_per_frame_conversion_is_byte_identical():
    rng = np.random.default_rng(7)
    clip = rng.uniform(-0.2, 1.2, size=(5, 6, 8, 3)).astype(np.float32)
    got = wb.images_to_uint8(clip)
    assert got.dtype == np.uint8 and got.flags["C_CONTIGUOUS"]
    assert np.array_equal(got, _reference_uint8(clip))


def test_a_uint8_clip_and_a_single_frame_pass_through():
    frame = np.full((4, 4, 3), 0.5, dtype=np.float32)
    assert wb.images_to_uint8(frame).shape == (1, 4, 4, 3)
    u8 = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    assert np.array_equal(wb.images_to_uint8(u8), u8)


def test_the_encoder_sends_the_same_bytes_frame_by_frame(monkeypatch, tmp_path):
    sent = bytearray()

    class _Stdin:
        def write(self, data):
            sent.extend(bytes(data))

        def close(self):
            pass

    class _Proc:
        returncode = 0
        stdin = _Stdin()

        def wait(self):
            (tmp_path / "out.mp4").write_bytes(b"x")
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(wb.otr_proc, "popen", lambda cmd, **kw: _Proc())
    monkeypatch.setattr(wb, "proven_frame_count", lambda path, n, ffprobe=None: n)
    frames = np.arange(3 * 4 * 6 * 3, dtype=np.uint8).reshape(3, 4, 6, 3)
    path, n = wb.encode_frames_to_silent_mp4(frames, str(tmp_path / "out.mp4"), 25)
    assert n == 3
    assert bytes(sent) == frames.tobytes()


def test_a_real_encode_still_proves_its_frame_count(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe not on this box")
    frames = wb.images_to_uint8(np.linspace(0, 1, 6 * 16 * 16 * 3,
                                            dtype=np.float32).reshape(6, 16, 16, 3))
    path, n = wb.encode_frames_to_silent_mp4(frames, str(tmp_path / "c.mp4"), 25)
    assert n == 6
