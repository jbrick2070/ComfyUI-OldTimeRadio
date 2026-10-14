"""IndexTTS2's sidecar worker is reaped on every failure branch.

A request timeout used to drop the worker handle without killing it, so a hung
worker kept its VRAM, ``unload()`` returned early, its stderr handle stayed
open and the next line spawned a second worker beside it. These drive the real
adapter against fake workers -- no venv, no subprocess, no GPU. UTF-8, no BOM.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nodes._otr_audio_engines import _otr_sidecar as SC  # noqa: E402
from nodes._otr_audio_engines import eng_indextts2 as idx  # noqa: E402


class _Stdin(io.StringIO):
    def __init__(self, broken=False):
        super().__init__()
        self.broken = broken

    def write(self, s):
        if self.broken:
            raise BrokenPipeError("worker gone")
        return super().write(s)


class _Stdout:
    def __init__(self, line="", block=False):
        self.line, self.block, self.closed = line, block, False

    def readline(self):
        if self.block:
            time.sleep(3)
            return ""
        return self.line

    def close(self):
        self.closed = True


class _Worker:
    def __init__(self, reply="", block=False, broken=False, alive=True,
                 write_partial=None):
        self.stdin = _Stdin(broken)
        self.stdout = _Stdout(reply, block)
        self.alive = alive
        self.killed = False
        self.write_partial = write_partial

    def poll(self):
        return None if self.alive else 0

    def kill(self):
        self.killed = True
        self.alive = False

    def wait(self, timeout=None):
        return 0


class _Stderr:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def engine(tmp_path, monkeypatch):
    eng = idx.IndexTTS2Engine()
    out = tmp_path / "otr_idx2_line.wav"
    monkeypatch.setattr(idx.tempfile, "mktemp", lambda **_kw: str(out))
    monkeypatch.setattr(eng, "_resolve_ref", lambda ref: str(ref))
    monkeypatch.setattr(SC, "request_timeout", lambda: 0.3)
    eng.out_path = out
    return eng


def _attach(eng, worker):
    eng._proc = worker
    eng._stderr = _Stderr()
    return eng._stderr


def _generate(eng):
    return eng.generate_voice("Hello.", "ref.wav", None, 7)


def test_a_timeout_kills_the_worker_and_closes_its_stderr(engine):
    worker = _Worker(block=True)
    stderr = _attach(engine, worker)
    engine.out_path.write_bytes(b"partial")  # the worker began writing
    with pytest.raises(RuntimeError, match="timed out"):
        _generate(engine)
    assert worker.killed, "a hung worker left alive keeps its VRAM"
    assert stderr.closed
    assert engine._proc is None and engine._stderr is None
    assert not engine.out_path.exists()


@pytest.mark.parametrize("worker_kwargs, needle", [
    ({"reply": ""}, "closed stdout"),
    ({"reply": "not json\n"}, "Expecting value"),
    ({"reply": "[1, 2]\n"}, "not a JSON object"),
    ({"reply": json.dumps({"ok": True}) + "\n"}, "missing out_path"),
    ({"broken": True}, "worker gone"),
])
def test_every_broken_stream_reaps_the_worker(engine, worker_kwargs, needle):
    worker = _Worker(**worker_kwargs)
    stderr = _attach(engine, worker)
    engine.out_path.write_bytes(b"partial")
    with pytest.raises(RuntimeError, match=needle):
        _generate(engine)
    assert worker.killed and stderr.closed
    assert engine._proc is None
    assert not engine.out_path.exists()


def test_a_render_failure_drops_the_partial_file_and_keeps_the_worker(engine):
    """ok:false is a clean answer on a healthy stream: the worker stays, the
    half-written file does not (infer can raise after it began writing)."""
    worker = _Worker(reply=json.dumps({"ok": False, "error": "boom"}) + "\n")
    stderr = _attach(engine, worker)
    engine.out_path.write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="render failed: boom"):
        _generate(engine)
    assert not worker.killed and not stderr.closed
    assert engine._proc is worker
    assert not engine.out_path.exists()


def test_a_successful_line_returns_audio_and_leaves_no_temp_file(engine):
    sf = pytest.importorskip("soundfile")
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    sf.write(str(engine.out_path), np.zeros(2205, dtype="float32"), 22050)
    worker = _Worker(reply=json.dumps(
        {"ok": True, "out_path": str(engine.out_path), "sample_rate": 22050})
        + "\n")
    _attach(engine, worker)
    audio = _generate(engine)
    assert audio["sample_rate"] == 22050
    assert tuple(audio["waveform"].shape) == (1, 2205)
    assert not engine.out_path.exists()
    assert engine._proc is worker and not worker.killed


def test_an_unreadable_wav_is_still_removed(engine):
    engine.out_path.write_bytes(b"not a wav")
    worker = _Worker(reply=json.dumps(
        {"ok": True, "out_path": str(engine.out_path)}) + "\n")
    _attach(engine, worker)
    with pytest.raises(Exception):
        _generate(engine)
    assert not engine.out_path.exists()


def test_unload_closes_stderr_even_after_a_failed_request(engine):
    stderr = _Stderr()
    engine._proc, engine._stderr = None, stderr
    engine.unload()
    assert stderr.closed


def test_load_reaps_a_worker_that_died_while_idle(engine, monkeypatch):
    dead = _Worker(alive=False)
    stderr = _attach(engine, dead)
    monkeypatch.setattr(engine, "install_gaps",
                        lambda: [("worker script", "/nope/worker.py")])
    with pytest.raises(RuntimeError, match="not installed"):
        engine.load()
    assert stderr.closed and dead.stdout.closed
    assert engine._proc is None and engine._stderr is None


def test_a_refused_start_reaps_the_worker_and_its_stderr(engine, tmp_path,
                                                         monkeypatch):
    spawned = {}

    def _popen(args, **kw):
        w = _Worker(reply=json.dumps({"ready": False, "error": "no cuda"})
                    + "\n")
        spawned["worker"], spawned["stderr"] = w, kw.get("stderr")
        return w

    monkeypatch.setattr(engine, "install_gaps", lambda: [])
    monkeypatch.setattr(idx.subprocess, "Popen", _popen)
    from nodes import _otr_paths
    monkeypatch.setattr(_otr_paths, "otr_sidecar_stderr_path",
                        lambda name: tmp_path / name)
    with pytest.raises(RuntimeError, match="failed to start: no cuda"):
        engine.load()
    assert spawned["worker"].killed
    assert spawned["stderr"].closed
    assert engine._proc is None
