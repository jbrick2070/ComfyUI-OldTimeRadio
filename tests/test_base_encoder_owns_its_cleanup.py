"""The base-video encoder cleans up what IT owns, on every exit (plan item 17).

``_encode_mp4`` pipes raw frames into an ffmpeg child. A raise from the pipe,
the frame iterator, or a cancellation used to skip the close and the stderr
read: the child lived until garbage collection and the log leaked. The caller's
audio WAV was ONE shared name in the system temp, removed only after success.

On a failure the full stderr log is now KEPT (evidence) and named, its tail is
logged and attached to the original exception; on success it is removed.

Mocked process boundary -- no ffmpeg, no GPU.
"""
from __future__ import annotations

import inspect
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
def harness(monkeypatch):
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
    state["out"] = "out.mp4"
    yield state
    for path in state["logs"]:          # kept failure logs: tidy up after
        try:
            os.remove(path)
        except OSError:
            pass


def _frames(n):
    for _ in range(n):
        yield np.zeros((2, 2, 3), dtype=np.uint8)


def _encode(state, frames, total=4):
    return ve._encode_mp4(frames, total, "audio.wav", state["out"], 2, 2, 25)


def test_success_closes_and_reaps_and_removes_the_log(harness):
    gen = _frames(4)
    assert _encode(harness, gen) == harness["out"]
    proc = harness["procs"][0]
    assert proc.stdin.closed and proc.waited and not proc.killed
    assert inspect.getgeneratorstate(gen) == inspect.GEN_CLOSED
    assert not any(os.path.exists(p) for p in harness["logs"])


@pytest.mark.parametrize("make,cause", [
    (lambda: _Proc(_Stdin(fail_after=2)), BrokenPipeError),   # pipe breaks mid-stream
    (lambda: _Proc(_Stdin(fail_on_close=True)), OSError),     # closing the pipe fails
])
def test_a_dead_ffmpeg_surfaces_its_own_reason_and_keeps_the_log(
        harness, make, cause, caplog):
    """ComfyUI's error panel shows str(exception), not notes: a broken pipe
    means ffmpeg died, so the error the user sees quotes ffmpeg's stderr,
    chained to the pipe error, and names the kept log."""
    harness["make"] = make
    gen = _frames(4)     # held here, so only an EXPLICIT close can close it
    with pytest.raises(RuntimeError, match="ffmpeg stopped mid-encode .*something") as raised:
        _encode(harness, gen)
    assert isinstance(raised.value.__cause__, cause)
    assert harness["logs"][0] in str(raised.value)
    proc = harness["procs"][0]
    assert proc.waited and (proc.killed or proc.returncode is not None)
    assert inspect.getgeneratorstate(gen) == inspect.GEN_CLOSED, \
        "the frame source must be closed so its worker pool unwinds"
    assert all(os.path.exists(p) for p in harness["logs"]), "the failure log is evidence"
    assert any("stderr tail" in r.message for r in caplog.records)


def test_a_frame_source_failure_keeps_its_own_exception_with_a_note(harness):
    def frames():
        yield np.zeros((2, 2, 3), dtype=np.uint8)
        raise ValueError("frame render failed")
    with pytest.raises(ValueError, match="frame render failed") as raised:
        _encode(harness, frames())
    assert "something" in " ".join(getattr(raised.value, "__notes__", []))


def test_the_failure_log_lives_in_the_swept_scratch_tier(harness, monkeypatch, tmp_path):
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(ve, "_scratch_dir_or_none", lambda: str(scratch.mkdir() or scratch))
    harness["make"] = lambda: _Proc(_Stdin(), returncode=1)
    with pytest.raises(RuntimeError):
        _encode(harness, _frames(2))
    assert os.path.dirname(harness["logs"][-1]) == str(scratch)


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


def test_a_spawn_failure_keeps_its_log_and_raises_the_cause(harness, monkeypatch):
    seen = []

    def popen(cmd, stdin=None, stdout=None, stderr=None):
        seen.append(stderr.name)
        harness["logs"].append(stderr.name)
        raise OSError("cannot spawn ffmpeg")

    monkeypatch.setattr(ve.otr_proc, "popen", popen)
    with pytest.raises(OSError, match="cannot spawn"):
        _encode(harness, iter([]))
    assert seen


def test_a_nonzero_exit_quotes_stderr_and_names_the_kept_log(harness):
    harness["make"] = lambda: _Proc(_Stdin(), returncode=1)
    with pytest.raises(RuntimeError, match="ffmpeg exited with code 1 :: .*something") as raised:
        _encode(harness, _frames(2))
    assert harness["logs"][0] in str(raised.value)
    assert os.path.exists(harness["logs"][0])


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
    src = inspect.getsource(ve)
    assert '"otr_video_audio.wav"' not in src
    assert "with _owned_audio_wav(audio_np, sr) as tmp_wav:" in src
