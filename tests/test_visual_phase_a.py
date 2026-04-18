"""
test_visual_phase_a.py  --  Phase A regression tests
=====================================================
Covers the Phase A robustness work landed on v2.0-alpha:

  - atomic_write_json / atomic_write_text never leave a partial file
    visible to a concurrent reader (`_atomic.py`).
  - VRAMCoordinator acquire / release / dead-PID reclaim / timeout
    semantics behave correctly (`vram_coordinator.py`).
  - VisualPoll detects a dead sidecar PID without waiting for the
    full poll-timeout (`poll.py` integration).
  - VisualBridge rejects malformed script_json with a PARSE_ERROR
    job_id (`bridge.py` schema validation).

These tests are pure-Python: no torch, no ComfyUI, no GPU.  They run
in the main venv and finish in well under a second total.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

# Make `otr_v2.visual.*` importable when pytest is run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from otr_v2.visual._atomic import atomic_write_json, atomic_write_text
from otr_v2.visual.vram_coordinator import VRAMCoordinator, _pid_alive


# ---------------------------------------------------------------------------
# atomic_write_*
# ---------------------------------------------------------------------------

class TestAtomicWrites:
    def test_atomic_write_text_creates_file(self, tmp_path: Path):
        target = tmp_path / "hello.txt"
        atomic_write_text(target, "hello world")
        assert target.read_text(encoding="utf-8") == "hello world"

    def test_atomic_write_text_creates_parents(self, tmp_path: Path):
        target = tmp_path / "deep" / "nested" / "file.txt"
        atomic_write_text(target, "ok")
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "ok"

    def test_atomic_write_json_round_trips(self, tmp_path: Path):
        target = tmp_path / "data.json"
        payload = {"status": "READY", "shots": [1, 2, 3], "title": "Sig\u2014nal"}
        atomic_write_json(target, payload)
        loaded = json.loads(target.read_text(encoding="utf-8"))
        assert loaded == payload

    def test_no_orphan_temp_files_on_success(self, tmp_path: Path):
        target = tmp_path / "out.json"
        atomic_write_json(target, {"x": 1})
        # Only the target file should exist; no .tmp leftovers.
        siblings = sorted(p.name for p in tmp_path.iterdir())
        assert siblings == ["out.json"], siblings

    def test_concurrent_reader_never_sees_partial_json(self, tmp_path: Path):
        """Hammer atomic_write_json from one thread while another reads.

        Without atomic writes the reader would occasionally see an
        empty or partial file and raise json.JSONDecodeError.  With
        atomic writes the rename is the only state transition, so
        every successful read MUST parse cleanly to one of the two
        states the writer alternates between.
        """
        target = tmp_path / "status.json"
        atomic_write_json(target, {"status": "RUNNING"})

        stop = threading.Event()
        states = [{"status": "RUNNING", "n": i} for i in range(50)]

        def writer():
            for s in states:
                if stop.is_set():
                    return
                atomic_write_json(target, s)
                # No sleep -- maximum hammering.

        decode_failures = 0
        successful_reads = 0

        def reader():
            nonlocal decode_failures, successful_reads
            for _ in range(2000):
                if stop.is_set():
                    return
                try:
                    data = json.loads(target.read_text(encoding="utf-8"))
                    # Sanity: must be one of our writes (status==RUNNING).
                    assert data.get("status") == "RUNNING"
                    successful_reads += 1
                except json.JSONDecodeError:
                    decode_failures += 1
                except (FileNotFoundError, PermissionError):
                    # Acceptable Windows-only transients during rename:
                    #   FileNotFoundError -> opened between unlink+create
                    #   PermissionError   -> opened mid-replace, OS lock
                    # Production readers in poll.py catch OSError, which
                    # is the parent class of both -- this matches.
                    pass

        t_w = threading.Thread(target=writer)
        t_r = threading.Thread(target=reader)
        t_r.start()
        t_w.start()
        t_w.join(timeout=10)
        stop.set()
        t_r.join(timeout=10)

        assert decode_failures == 0, (
            f"reader saw {decode_failures} partial-JSON failures "
            f"({successful_reads} successful reads)"
        )
        assert successful_reads > 0, "reader never managed to read the file"

    def test_concurrent_writers_never_crash(self, tmp_path: Path):
        """Two+ writers racing to the same file must not raise.

        Regression for a Windows-only bug surfaced 2026-04-16: ``os.replace``
        could hit ``PermissionError [WinError 5]`` when threads/processes
        raced to rename onto the same target because the OS briefly held
        an exclusive lock on the destination.  Fixed by retrying
        ``os.replace`` with backoff in ``_atomic.py``.
        """
        target = tmp_path / "status.json"
        atomic_write_json(target, {"status": "RUNNING"})

        errors: list[BaseException] = []
        stop = threading.Event()

        def writer(tag: str):
            for i in range(100):
                if stop.is_set():
                    return
                try:
                    atomic_write_json(target, {"status": "RUNNING", "tag": tag, "n": i})
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)
                    return

        threads = [
            threading.Thread(target=writer, args=("alpha",)),
            threading.Thread(target=writer, args=("bravo",)),
            threading.Thread(target=writer, args=("charlie",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)
        stop.set()

        assert not errors, (
            "concurrent writers crashed: "
            f"{[type(e).__name__ + ': ' + str(e) for e in errors]}"
        )
        # Final file must still be valid JSON written by one of the writers.
        final = json.loads(target.read_text(encoding="utf-8"))
        assert final.get("status") == "RUNNING"
        assert final.get("tag") in {"alpha", "bravo", "charlie"}


# ---------------------------------------------------------------------------
# VRAMCoordinator
# ---------------------------------------------------------------------------

class TestVRAMCoordinator:
    def test_acquire_release_basic(self, tmp_path: Path):
        coord = VRAMCoordinator(lock_path=tmp_path / "vram.lock")
        with coord.acquire(owner="placeholder", job_id="job1"):
            assert coord.is_held()
            info = coord.status()
            assert info is not None
            assert info.owner == "placeholder"
            assert info.job_id == "job1"
            assert info.pid == os.getpid()
        # After exit, lock should be released
        assert not coord.is_held()
        assert coord.status() is None

    def test_status_when_unlocked(self, tmp_path: Path):
        coord = VRAMCoordinator(lock_path=tmp_path / "vram.lock")
        assert coord.status() is None
        assert not coord.is_held()

    def test_dead_pid_reclaim(self, tmp_path: Path):
        """A lock left by a dead PID must be reclaimable."""
        lock_path = tmp_path / "vram.lock"
        # Simulate a stale lock: write a JSON payload referencing a PID
        # that will not exist.  PID 0 is "kernel" on POSIX and never
        # corresponds to a user process; on Windows we use a high
        # nonsense value.  Both fail _pid_alive().
        stale_pid = 0 if os.name != "nt" else 0xFFFFFFF0
        lock_path.write_text(json.dumps({
            "pid": stale_pid,
            "owner": "ghost_worker",
            "job_id": "stale_job",
            "acquired_at": time.time() - 7200,
        }), encoding="utf-8")

        coord = VRAMCoordinator(lock_path=lock_path, poll_interval_sec=0.05)
        with coord.acquire(owner="reclaimer", job_id="fresh", timeout=5.0):
            info = coord.status()
            assert info is not None
            assert info.owner == "reclaimer"
            assert info.pid == os.getpid()

    def test_timeout_when_held_by_live_pid(self, tmp_path: Path):
        """If a live PID holds the lock, acquire() must time out."""
        lock_path = tmp_path / "vram.lock"
        # Plant a lock owned by THIS process (so liveness check passes).
        lock_path.write_text(json.dumps({
            "pid": os.getpid(),
            "owner": "ourself",
            "job_id": "blocking",
            "acquired_at": time.time(),
        }), encoding="utf-8")

        coord = VRAMCoordinator(lock_path=lock_path, poll_interval_sec=0.05)
        with pytest.raises(TimeoutError):
            with coord.acquire(owner="other", job_id="x", timeout=0.5):
                pass

    def test_release_when_not_held_is_safe(self, tmp_path: Path):
        coord = VRAMCoordinator(lock_path=tmp_path / "vram.lock")
        # Should not raise.
        result = coord.release(owner="nobody")
        assert result is False

    def test_pid_alive_self(self):
        assert _pid_alive(os.getpid()) is True

    def test_pid_alive_invalid(self):
        assert _pid_alive(0) is False
        assert _pid_alive(-1) is False


# ---------------------------------------------------------------------------
# VisualBridge schema validation
# ---------------------------------------------------------------------------

class TestBridgeValidation:
    def _validator(self):
        from otr_v2.visual.bridge import _validate_script_lines
        return _validate_script_lines

    def test_valid_script(self):
        _validate_script_lines = self._validator()
        ok, reason = _validate_script_lines([
            {"type": "title", "value": "Test Episode"},
            {"type": "scene_break", "value": "Scene 1"},
            {"type": "dialogue", "speaker": "ALICE", "value": "Hello."},
        ])
        assert ok, reason

    def test_empty_list(self):
        _validate_script_lines = self._validator()
        ok, reason = _validate_script_lines([])
        assert not ok
        assert "empty" in reason.lower()

    def test_not_a_list(self):
        _validate_script_lines = self._validator()
        ok, reason = _validate_script_lines({"not": "a list"})
        assert not ok
        assert "array" in reason.lower()

    def test_entry_missing_type(self):
        _validate_script_lines = self._validator()
        ok, reason = _validate_script_lines([
            {"speaker": "ALICE", "value": "Hello"},  # no type
        ])
        assert not ok
        assert "type" in reason.lower()

    def test_no_canonical_token_types(self):
        _validate_script_lines = self._validator()
        ok, reason = _validate_script_lines([
            {"type": "garbage"},
            {"type": "also_garbage"},
        ])
        assert not ok
        assert "canonical" in reason.lower()

    def test_extra_token_types_allowed_with_canonical(self):
        """Forward-compat: unknown types are fine as long as at least
        one canonical type is present."""
        _validate_script_lines = self._validator()
        ok, reason = _validate_script_lines([
            {"type": "future_token_type"},
            {"type": "dialogue", "speaker": "X", "value": "ok"},
        ])
        assert ok, reason


# ---------------------------------------------------------------------------
# VisualPoll WORKER_DEAD detection
# ---------------------------------------------------------------------------

class TestPollWorkerDead:
    def test_pid_helpers_match_coordinator(self):
        """poll._pid_alive must give the same answer as the coordinator
        version for the canonical inputs (we keep the implementations
        independent for resilience but they must agree on the basics).
        """
        from otr_v2.visual.poll import _pid_alive as poll_alive
        from otr_v2.visual.vram_coordinator import _pid_alive as coord_alive
        assert poll_alive(os.getpid()) == coord_alive(os.getpid()) == True
        assert poll_alive(0) == coord_alive(0) == False

    def test_read_sidecar_pid_missing(self, tmp_path: Path, monkeypatch):
        """Missing PID file returns 0 (PID-tracking-unavailable sentinel)."""
        from otr_v2.visual import poll as poll_mod
        monkeypatch.setattr(poll_mod, "_IO_IN", tmp_path)
        assert poll_mod._read_sidecar_pid("nonexistent_job") == 0

    def test_read_sidecar_pid_valid(self, tmp_path: Path, monkeypatch):
        from otr_v2.visual import poll as poll_mod
        monkeypatch.setattr(poll_mod, "_IO_IN", tmp_path)
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        (job_dir / "sidecar_pid.txt").write_text("12345", encoding="utf-8")
        assert poll_mod._read_sidecar_pid("test_job") == 12345

    def test_read_sidecar_pid_garbage(self, tmp_path: Path, monkeypatch):
        from otr_v2.visual import poll as poll_mod
        monkeypatch.setattr(poll_mod, "_IO_IN", tmp_path)
        job_dir = tmp_path / "garbage_job"
        job_dir.mkdir()
        (job_dir / "sidecar_pid.txt").write_text("not a number", encoding="utf-8")
        assert poll_mod._read_sidecar_pid("garbage_job") == 0

    def test_worker_dead_terminal_status(self):
        """WORKER_DEAD must be in the terminal-status set so the poll
        loop exits cleanly when surfaced."""
        from otr_v2.visual.poll import _TERMINAL_STATUSES
        assert "WORKER_DEAD" in _TERMINAL_STATUSES
