"""The base-video encoder cleans up what IT owns, on every exit (plan item 17).

``_encode_mp4`` pipes raw frames into an ffmpeg child. A raise from the pipe,
the frame iterator, or a cancellation used to skip the close and the stderr
read: the child lived until garbage collection and the log leaked. The caller's
audio WAV was ONE shared name in the system temp, removed only after success.

Mocked process boundary -- no ffmpeg, no GPU.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from nodes import video_engine as ve


class _Stdin:
    def __init__(self, fail_after=None, fail_on_close=False):
        self.writes, self.closed = 0, False
        self._fail_after, self._fail_on_close = fail_after, fail_on_close

    def write(self, data):
        if self._fail_after is not None and self.writes >= self._fail_after:
            raise BrokenPipeError("ffmpeg went away")
        self.writes += 1

    def close(self):
        self.closed = True
        if self._fail_on_close:
            raise OSError("close failed")


class _Proc:
    def __init__(self, stdin, returncode=0):
        self.stdin = stdin
        self._rc = returncode
        self.returncode = None
        self.killed = False
        self.waited = False

    def poll(self):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.waited = True
        if self.returncode is None:
            self.returncode = self._rc
        return self.returncode


@pytest.fixture
def harness(monkeypatch, tmp_path):
    state = {"procs": [], "logs": []}
    monkeypatch.setattr(ve, "_find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(ve, "_check_nvenc", lambda _ff: False)

    def popen(cmd, stdin=None, stdout=None, stderr=None):
        state["logs"].append(stderr.name)
        stderr.write(b"frame=  12 fps=0.0 q=0.0 size=0kB error: something\n")
        stderr.flush()
        proc = state["make"]()
        state["procs"].append(proc)
        return proc

    monkeypatch.setattr(ve.otr_proc, "popen", popen)
    state["make"] = lambda: _Proc(_Stdin())
    state["out"] = str(tmp_path / "out.mp4")
    return state


def _frames(n, closed):
    try:
        for _ in range(n):
            yield np.zeros((2, 2, 3), dtype=np.uint8)
    finally:
        closed.append(True)


def _encode(state, frames, total=4):
    return ve._encode_mp4(frames, total, "audio.wav", state["out"], 2, 2, 25)


def test_success_closes_and_reaps_and_leaves_no_log(harness):
    closed = []
    assert _encode(harness, _frames(4, closed)) == harness["out"]
    proc = harness["procs"][0]
    assert proc.stdin.closed and proc.waited and not proc.killed
    assert closed == [True]
    assert not any(os.path.exists(p) for p in harness["logs"])


@pytest.mark.parametrize("make,frames,exc", [
    # the pipe breaks mid-stream
    (lambda: _Proc(_Stdin(fail_after=2)), None, BrokenPipeError),
    # closing the pipe fails
    (lambda: _Proc(_Stdin(fail_on_close=True)), None, OSError),
])
def test_a_pipe_failure_stops_the_owned_child_and_keeps_the_cause(
        harness, make, frames, exc, caplog):
    harness["make"] = make
    closed = []
    with pytest.raises(exc):
        _encode(harness, _frames(4, closed))
    proc = harness["procs"][0]
    assert proc.killed or proc.returncode is not None
    assert proc.waited
    assert closed == [True], "the frame source must be closed so its pool unwinds"
    assert not any(os.path.exists(p) for p in harness["logs"])
    assert any("stderr tail" in r.message and "something" in r.message
               for r in caplog.records)


def test_a_frame_source_failure_or_cancel_stops_the_child(harness):
    class Interrupted(BaseException):
        """Stands in for a ComfyUI cancel."""

    for boom in (RuntimeError("frame render failed"), Interrupted("cancel")):
        def frames(boom=boom):
            yield np.zeros((2, 2, 3), dtype=np.uint8)
            raise boom
        with pytest.raises(type(boom)):
            _encode(harness, frames())
        proc = harness["procs"][-1]
        assert proc.killed and proc.waited
    assert not any(os.path.exists(p) for p in harness["logs"])


def test_a_spawn_failure_leaves_no_log(harness, monkeypatch):
    seen = []

    def popen(cmd, stdin=None, stdout=None, stderr=None):
        seen.append(stderr.name)
        raise OSError("cannot spawn ffmpeg")

    monkeypatch.setattr(ve.otr_proc, "popen", popen)
    with pytest.raises(OSError, match="cannot spawn"):
        _encode(harness, iter([]))
    assert seen and not os.path.exists(seen[0])


def test_a_nonzero_exit_still_quotes_stderr(harness):
    harness["make"] = lambda: _Proc(_Stdin(), returncode=1)
    with pytest.raises(RuntimeError, match="ffmpeg exited with code 1 :: .*something"):
        _encode(harness, _frames(2, []))


def test_each_encode_gets_its_own_wav_and_it_is_removed_on_every_exit():
    audio = np.zeros(2400, dtype=np.float32)
    with ve._owned_audio_wav(audio, 24000) as a, ve._owned_audio_wav(audio, 24000) as b:
        assert a != b and os.path.isfile(a) and os.path.isfile(b)
        assert os.path.basename(a).startswith("otr_video_audio_")
    assert not os.path.exists(a) and not os.path.exists(b)
    with pytest.raises(RuntimeError):
        with ve._owned_audio_wav(audio, 24000) as c:
            raise RuntimeError("encode failed")
    assert not os.path.exists(c)


def test_render_video_writes_the_wav_through_the_owner():
    """The caller no longer builds a shared temp name; the owned WAV is the only
    way the base encode gets its audio."""
    import inspect
    src = inspect.getsource(ve)
    assert '"otr_video_audio.wav"' not in src
    assert "with _owned_audio_wav(audio_np, sr) as tmp_wav:" in src
