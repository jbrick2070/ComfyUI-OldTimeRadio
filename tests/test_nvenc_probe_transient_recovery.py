"""VF-01: an NVENC probe that could not DECIDE is not cached as "no NVENC".

`encode_sink.has_nvenc` used to cache every failure as False for the life of
the server, so one probe that timed out -- or that found VRAM full of model
weights on an 8 GB card -- kept every later encode on CPU until a restart.

The contract these tests hold, all with a mocked subprocess (no GPU):
  * FINAL answers are cached after ONE probe: success, and every refusal that
    waiting cannot change. A box without NVENC is never probed twice.
  * A timeout, an OpenEncodeSessionEx "incompatible client key" (the GeForce
    session cap, measured on MRKT) or "encoder busy", any memory exhaustion,
    or a spawn OSError other than missing/denied, answers False for THIS call
    (the caller uses libx264, as always) and is re-probed after a doubling
    cooldown. A caller that arrives during a re-probe gets CPU at once.
  * Timeouts cost twenty seconds each, so three make the binary FINAL False.
    Fast refusals are never capped.
  * No probe outcome ever raises into a render (operator rule, 2026-10-14:
    no probe may fail a workflow; only a true OOM may).
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from nodes._otr_shared import encode_sink as es

BIN = "/opt/otr/ffmpeg"
OTHER = "/opt/other/ffmpeg"

#: Captured verbatim on MRKT (RTX 4060 Laptop, driver 617.42, ffmpeg 9.0,
#: 2026-10-14) with twelve live h264_nvenc encodes holding sessions: this is
#: how the GeForce concurrent-session cap refuses the thirteenth.
SESSION_CAP = (
    "[h264_nvenc @ 000001f7d5211a00] OpenEncodeSessionEx failed: incompatible "
    "client key (21): (no details)\n"
    "[h264_nvenc @ 000001f7d5211a00] No capable devices found\n"
    "[vost#0:0/h264_nvenc @ 000001f7d5211780] [enc:h264_nvenc @ "
    "000001f7d5210580] Error while opening encoder - maybe incorrect "
    "parameters such as bit_rate, rate, width or height.")
OOM = ("[h264_nvenc @ 0000021f] OpenEncodeSessionEx failed: out of memory "
       "(10): (no details)\n[vost#0:0/h264_nvenc] Error while opening encoder")
BUSY = "[h264_nvenc @ 0000021f] OpenEncodeSessionEx failed: encoder busy (19)"


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    monkeypatch.setattr(es, "_NVENC_PROBE", {})
    monkeypatch.setattr(es, "_NVENC_KEY_LOCKS", {})
    c = _Clock()
    monkeypatch.setattr(es, "_now", c)
    return c


def _script(monkeypatch, outcomes):
    """Each probe takes the next outcome: an int returncode, (rc, stderr), or
    an exception instance to raise. The last one repeats."""
    calls = []

    def run(cmd, **_kw):
        calls.append(cmd[0])
        o = outcomes[min(len(calls), len(outcomes)) - 1]
        if isinstance(o, BaseException):
            raise o
        rc, err = o if isinstance(o, tuple) else (o, "")
        return SimpleNamespace(returncode=rc, stderr=err)

    monkeypatch.setattr(es.otr_proc, "run", run)
    return calls


def _timeout():
    return subprocess.TimeoutExpired(cmd="ffmpeg", timeout=20)


# --------------------------------------------------------------------------- #
# Inconclusive -> CPU now, recover later
# --------------------------------------------------------------------------- #
def test_a_timeout_is_cpu_now_and_recovers_after_the_cooldown(monkeypatch, clock):
    calls = _script(monkeypatch, [_timeout(), 0])
    assert es.has_nvenc(BIN) is False
    assert es.has_nvenc(BIN) is False, "inside the cooldown: CPU, no spawn"
    assert len(calls) == 1
    clock.t += es._NVENC_RETRY_FIRST_S + 1
    assert es.has_nvenc(BIN) is True
    assert es.has_nvenc(BIN) is True
    assert len(calls) == 2, "a recovered binary is FINAL: no more probes"


@pytest.mark.parametrize("stderr", [
    SESSION_CAP, OOM, BUSY,
    "[h264_nvenc @ 0000021f] CreateInputBuffer failed: out of memory (10)",
    "[AVHWDeviceContext] cu->cuCtxCreate(...) failed -> "
    "CUDA_ERROR_OUT_OF_MEMORY: out of memory",
    "Error allocating frame: Cannot allocate memory",
], ids=["session_cap_measured", "out_of_memory", "busy", "surface_oom",
        "cuda_oom", "host_enomem"])
def test_a_refusal_that_can_clear_recovers(monkeypatch, clock, stderr):
    calls = _script(monkeypatch, [(1, stderr), 0])
    assert es.has_nvenc(BIN) is False
    clock.t += es._NVENC_RETRY_FIRST_S + 1
    assert es.has_nvenc(BIN) is True
    assert len(calls) == 2


@pytest.mark.parametrize("exc", [
    OSError(12, "Cannot allocate memory"),
    OSError(24, "Too many open files"),
    OSError(22, "The paging file is too small", None, 1455),
], ids=["enomem", "emfile", "winerror_1455"])
def test_a_spawn_that_fails_for_the_moment_recovers(monkeypatch, clock, exc):
    calls = _script(monkeypatch, [exc, 0])
    assert es.has_nvenc(BIN) is False
    clock.t += es._NVENC_RETRY_FIRST_S + 1
    assert es.has_nvenc(BIN) is True
    assert len(calls) == 2


# --------------------------------------------------------------------------- #
# Final refusals: a box without NVENC is probed exactly once
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stderr", [
    "Unknown encoder 'h264_nvenc'",
    "[h264_nvenc] Cannot load libnvidia-encode.so.1",
    "[h264_nvenc] OpenEncodeSessionEx failed: unsupported device (2)\n"
    "[h264_nvenc] No capable devices found",
    "Unknown input format: 'lavfi'",
    "",
], ids=["no_encoder", "no_library", "no_device", "no_lavfi", "silent"])
def test_a_refusal_that_waiting_cannot_change_is_probed_once(monkeypatch, clock, stderr):
    calls = _script(monkeypatch, [(1, stderr), 0])
    assert es.has_nvenc(BIN) is False
    clock.t += 10 * es._NVENC_RETRY_MAX_S
    assert es.has_nvenc(BIN) is False
    assert len(calls) == 1
    assert es._NVENC_PROBE[es.os.path.normcase(es.os.path.abspath(BIN))] is False


@pytest.mark.parametrize("exc", [
    FileNotFoundError(2, "No such file"),
    PermissionError(13, "Access is denied"),
    NotADirectoryError(20, "Not a directory"),
    es.otr_proc.ExecutableNotAllowed("not an allowed executable"),
], ids=["missing", "denied", "not_a_dir", "not_allowed"])
def test_a_binary_that_cannot_run_is_final(monkeypatch, clock, exc):
    calls = _script(monkeypatch, [exc, 0])
    assert es.has_nvenc(BIN) is False
    clock.t += 10 * es._NVENC_RETRY_MAX_S
    assert es.has_nvenc(BIN) is False
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# Bounds
# --------------------------------------------------------------------------- #
def test_timeouts_are_capped_then_final(monkeypatch, clock):
    calls = _script(monkeypatch, [_timeout()])
    for _ in range(es._NVENC_MAX_TIMEOUTS):
        assert es.has_nvenc(BIN) is False
        clock.t += es._NVENC_RETRY_MAX_S + 1
    assert len(calls) == es._NVENC_MAX_TIMEOUTS
    clock.t += 100 * es._NVENC_RETRY_MAX_S
    assert es.has_nvenc(BIN) is False
    assert len(calls) == es._NVENC_MAX_TIMEOUTS, "no fourth twenty-second probe"


def test_fast_refusals_back_off_doubling_to_the_ceiling_and_never_give_up(monkeypatch, clock):
    calls = _script(monkeypatch, [(1, OOM)])
    waits = []
    for _ in range(12):
        before = len(calls)
        assert es.has_nvenc(BIN) is False
        assert len(calls) == before + 1
        entry = es._NVENC_PROBE[es.os.path.normcase(es.os.path.abspath(BIN))]
        waits.append(entry.not_before - clock.t)
        clock.t += 1
        assert es.has_nvenc(BIN) is False and len(calls) == before + 1
        clock.t = entry.not_before
    assert waits[:5] == [120.0, 240.0, 480.0, 960.0, 1800.0], waits
    assert set(waits[4:]) == {es._NVENC_RETRY_MAX_S}
    assert not isinstance(
        es._NVENC_PROBE[es.os.path.normcase(es.os.path.abspath(BIN))], bool)


def test_a_very_long_run_of_refusals_never_raises(monkeypatch, clock):
    """The backoff exponent is capped: 2 ** 1024 overflows a float, and a probe
    must never be what fails a workflow."""
    _script(monkeypatch, [(1, OOM)])
    key = es.os.path.normcase(es.os.path.abspath(BIN))
    es._NVENC_PROBE[key] = es._Pending(5000, 0, clock.t)
    assert es.has_nvenc(BIN) is False
    assert es._NVENC_PROBE[key].not_before - clock.t == es._NVENC_RETRY_MAX_S


# --------------------------------------------------------------------------- #
# Isolation and concurrency
# --------------------------------------------------------------------------- #
def test_one_binary_in_cooldown_does_not_block_another(monkeypatch, clock):
    calls = []

    def run(cmd, **_kw):
        calls.append(cmd[0])
        if cmd[0] == BIN:
            raise _timeout()
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(es.otr_proc, "run", run)
    assert es.has_nvenc(BIN) is False
    assert es.has_nvenc(OTHER) is True
    assert calls == [BIN, OTHER]


def test_a_long_probe_does_not_stall_a_cached_answer_for_another_binary(monkeypatch, clock):
    es._NVENC_PROBE[es.os.path.normcase(es.os.path.abspath(OTHER))] = True
    started, release = threading.Event(), threading.Event()

    def run(cmd, **_kw):
        started.set()
        release.wait(5)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(es.otr_proc, "run", run)
    worker = threading.Thread(target=es.has_nvenc, args=(BIN,))
    worker.start()
    try:
        assert started.wait(5)
        t0 = time.monotonic()
        assert es.has_nvenc(OTHER) is True
        assert time.monotonic() - t0 < 1.0, "cached answer waited on another probe"
    finally:
        release.set()
        worker.join(5)


class _CountingLock:
    """A per-binary lock that counts acquisition attempts, so a test can prove
    every racer reached it WHILE the probe was in flight (not merely after)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cv = threading.Condition()
        self.attempts = 0

    def acquire(self, blocking=True, timeout=-1):
        with self._cv:
            self.attempts += 1
            self._cv.notify_all()
        return self._lock.acquire(blocking, timeout)

    def release(self):
        self._lock.release()

    def wait_for_attempts(self, n, timeout=5.0):
        with self._cv:
            return self._cv.wait_for(lambda: self.attempts >= n, timeout)


@pytest.mark.parametrize("outcome", [0, "timeout"], ids=["success", "timeout"])
def test_racing_first_callers_spawn_one_probe(monkeypatch, clock, outcome):
    lock = _CountingLock()
    es._NVENC_KEY_LOCKS[es.os.path.normcase(es.os.path.abspath(BIN))] = lock
    calls, contended = [], []

    def run(cmd, **_kw):
        calls.append(cmd[0])
        # Hold the probe until all four callers have tried the lock.
        contended.append(lock.wait_for_attempts(4))
        if outcome == "timeout":
            raise _timeout()
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(es.otr_proc, "run", run)
    results = []
    threads = [threading.Thread(target=lambda: results.append(es.has_nvenc(BIN)))
               for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert contended == [True], "the other callers never reached the lock mid-probe"
    assert len(calls) == 1, calls
    assert results == [outcome == 0] * 4


def test_a_caller_arriving_during_a_reprobe_gets_cpu_at_once(monkeypatch, clock):
    """A re-probe's answer is already "CPU for now"; nobody waits up to twenty
    seconds behind it."""
    key = es.os.path.normcase(es.os.path.abspath(BIN))
    es._NVENC_PROBE[key] = es._Pending(1, 0, clock.t)      # cooldown just ended
    started, release = threading.Event(), threading.Event()

    def run(cmd, **_kw):
        started.set()
        release.wait(5)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(es.otr_proc, "run", run)
    worker = threading.Thread(target=es.has_nvenc, args=(BIN,))
    worker.start()
    try:
        assert started.wait(5)
        t0 = time.monotonic()
        assert es.has_nvenc(BIN) is False
        assert time.monotonic() - t0 < 1.0, "a caller stalled behind a re-probe"
    finally:
        release.set()
        worker.join(5)
    assert es.has_nvenc(BIN) is True, "the re-probe itself still recovered"


# --------------------------------------------------------------------------- #
# Never raises (operator rule: no probe may fail a workflow)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("result", [None, SimpleNamespace(stderr="")],
                         ids=["none", "no_returncode"])
def test_a_malformed_probe_result_is_a_final_no(monkeypatch, clock, result):
    monkeypatch.setattr(es.otr_proc, "run", lambda cmd, **_kw: result)
    assert es.has_nvenc(BIN) is False


def test_has_nvenc_never_raises_whatever_breaks_inside(monkeypatch, clock):
    def broken(_ffmpeg):
        raise RuntimeError("bookkeeping blew up")

    monkeypatch.setattr(es, "_decide", broken)
    assert es.has_nvenc(BIN) is False


def test_a_broken_log_handler_cannot_change_the_answer(monkeypatch, clock):
    class Exploding(logging.Handler):
        def emit(self, record):
            raise OSError("log sink failed")

        def handleError(self, record):          # surface it, do not swallow
            raise OSError("log sink failed")

    handler = Exploding()
    es.log.addHandler(handler)
    try:
        _script(monkeypatch, [_timeout(), 0])
        assert es.has_nvenc(BIN) is False       # logs a warning: must not raise
        clock.t += es._NVENC_RETRY_FIRST_S + 1
        assert es.has_nvenc(BIN) is True        # logs a recovery: must not raise
    finally:
        es.log.removeHandler(handler)


# --------------------------------------------------------------------------- #
# The consumer: never raises, and announces a recovery
# --------------------------------------------------------------------------- #
def test_video_engine_announces_cpu_then_nvidia_once_each(monkeypatch, clock, caplog):
    from nodes import video_engine as ve
    monkeypatch.setattr(ve, "_NVENC_ANNOUNCED", set())
    _script(monkeypatch, [_timeout(), 0])
    with caplog.at_level(logging.INFO, logger="OTR"):
        assert ve._check_nvenc(BIN) is False
        assert ve._check_nvenc(BIN) is False
        clock.t += es._NVENC_RETRY_FIRST_S + 1
        assert ve._check_nvenc(BIN) is True
        assert ve._check_nvenc(BIN) is True
    lines = [r.getMessage() for r in caplog.records if "[Video] Encoder:" in r.getMessage()]
    assert lines == ["[Video] Encoder: CPU libx264",
                     "[Video] Encoder: NVIDIA h264_nvenc"], lines


def test_video_engine_never_raises_when_the_owner_does(monkeypatch):
    from nodes import video_engine as ve
    monkeypatch.setattr(ve, "_NVENC_ANNOUNCED", set())

    def explode(_ffmpeg):
        raise RuntimeError("owner blew up")

    monkeypatch.setattr(es, "has_nvenc", explode)
    assert ve._check_nvenc(BIN) is False


def test_video_engine_keeps_its_answer_when_logging_breaks(monkeypatch):
    from nodes import video_engine as ve
    monkeypatch.setattr(ve, "_NVENC_ANNOUNCED", set())
    monkeypatch.setattr(es, "has_nvenc", lambda _ffmpeg: True)

    class Broken:
        def info(self, *a, **k):
            raise OSError("log sink failed")

        warning = info

    monkeypatch.setattr(ve, "log", Broken())
    assert ve._check_nvenc(BIN) is True
    assert ve._check_nvenc(None) is False
