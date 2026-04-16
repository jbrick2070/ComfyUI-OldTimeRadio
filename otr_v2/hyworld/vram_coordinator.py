"""
vram_coordinator.py  --  File-lock GPU gate for HyWorld worker / Bark TTS.
==========================================================================
Phase A robustness: when the HyWorld worker eventually does GPU work
(Phase B SDXL anchor, Phase C splat lift), it MUST NOT collide with
Bark TTS, which holds the GPU for ~12-18 minutes per episode.  The
audio path is byte-identical (C7) and is the king; if Bark gets OOM'd
mid-render, the episode is lost.

This module provides a single-writer file lock in the shared
``io/vram.lock`` directory.  Exactly one process can hold the lock
at a time.  The lock file contains the owning PID + owner_id +
timestamp so a stale lock from a crashed process can be detected
and reclaimed.

Today (Phase A) the worker still runs CPU-only (ffmpeg zoompan), so
no caller actually invokes ``acquire``.  This module is the scaffold:
landing it now means Phase B can wire in a single ``with coord:``
block instead of inventing a new gate at that point.

Cross-process atomicity
-----------------------
The lock acquisition uses ``os.open(..., O_CREAT | O_EXCL)`` on the
lock file's payload itself.  This is atomic on both POSIX and
Windows -- if the file already exists the open fails with FileExistsError
and we know someone else holds it.  No fcntl / msvcrt advisory locking
is used, so the file works identically across the two OS families
even with the worker spawned from a different conda env.

Liveness
--------
``acquire`` checks the existing owner's PID.  If the PID is no longer
running, the lock is forcibly released and reclaimed.  This prevents
a hard kill of either Bark or the worker from wedging the next run.

Usage
-----
    coord = VRAMCoordinator()
    with coord.acquire(owner="hyworld_worker", job_id="hw_xxx", timeout=1800):
        # ... do GPU work safely ...
        pass
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger("OTR.hyworld.vram_coordinator")

# Repo-root-relative location for the lock file.  Lives in io/ alongside
# the rest of the HyWorld exchange dirs so it gets picked up by the
# Phase E disk-hygiene sweep automatically.
_OTR_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_LOCK_PATH = _OTR_ROOT / "io" / "vram.lock"

# How long a held lock can stay before we consider it suspicious and
# log a warning (not an automatic break).  This is roughly Bark's worst
# observed wall time + a buffer.
_STALE_WARN_SEC = 30 * 60  # 30 minutes


def _pid_alive(pid: int) -> bool:
    """Return True if the given PID is currently a running process.

    Cross-platform: os.kill(pid, 0) on POSIX, OpenProcess on Windows.
    Falsey for invalid PIDs (<=0).
    """
    if pid <= 0:
        return False

    if os.name == "nt":
        # On Windows, signal 0 isn't supported, so probe via OpenProcess.
        try:
            import ctypes  # local import keeps module load fast on POSIX

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if not handle:
                return False
            # Check if it has already exited.
            STILL_ACTIVE = 259
            exit_code = ctypes.c_ulong(0)
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            if not ok:
                return False
            return exit_code.value == STILL_ACTIVE
        except Exception:
            # If we can't tell, err on the side of "alive" so we don't
            # break a real running process's lock.
            return True

    # POSIX
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but isn't ours -- still alive.
        return True
    except OSError:
        return False
    return True


@dataclass
class LockInfo:
    """Snapshot of an existing lock file's contents."""

    pid: int = 0
    owner: str = ""
    job_id: str = ""
    acquired_at: float = 0.0

    @classmethod
    def from_path(cls, lock_path: Path) -> Optional["LockInfo"]:
        """Read the lock file; returns None if missing or unparseable."""
        try:
            text = lock_path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Lock file is being written right now (rare race because
            # we use atomic create).  Treat as held by an unknown owner.
            return cls()
        return cls(
            pid=int(data.get("pid", 0) or 0),
            owner=str(data.get("owner", "")),
            job_id=str(data.get("job_id", "")),
            acquired_at=float(data.get("acquired_at", 0) or 0),
        )

    def is_alive(self) -> bool:
        return _pid_alive(self.pid)


@dataclass
class VRAMCoordinator:
    """Process-coordinator for the single GPU on this workstation.

    Today only the HyWorld worker is wired to this.  Bark TTS does not
    yet acquire the lock (it predates this module), but adding a single
    ``with coord.acquire(owner="bark"):`` block in the audio path is
    a one-line follow-up when we're ready.  Until then the worker
    still gets value from the dead-PID reclaim and from refusing to
    start GPU work if a previous worker is wedged.

    Attributes:
        lock_path: Filesystem path of the lock file.
        poll_interval_sec: Seconds between retry attempts during acquire.
        verbose_log: If True, log each retry; otherwise only log
            acquire/release/reclaim events.
    """

    lock_path: Path = field(default_factory=lambda: _DEFAULT_LOCK_PATH)
    poll_interval_sec: float = 1.0
    verbose_log: bool = False

    # ---- low-level operations -------------------------------------------------

    def _write_lock_payload(self, owner: str, job_id: str) -> None:
        """Write the JSON payload INTO an already-created lock file.

        The exclusive create has already succeeded; this is just the
        bookkeeping write.  Failure here is logged but not raised --
        the lock is still held even with an empty payload.
        """
        payload = {
            "pid": os.getpid(),
            "owner": owner,
            "job_id": job_id,
            "acquired_at": time.time(),
        }
        try:
            self.lock_path.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            log.warning(
                "[VRAMCoordinator] payload write failed (lock still held): %s", e
            )

    def _try_create(self) -> bool:
        """Atomically create the lock file with no contents.  True on success."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(
                str(self.lock_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
            os.close(fd)
            return True
        except FileExistsError:
            return False

    def _force_release(self, reason: str) -> None:
        """Forcibly remove the lock file.  Used for dead-PID reclaim."""
        try:
            self.lock_path.unlink()
            log.info("[VRAMCoordinator] lock force-released: %s", reason)
        except FileNotFoundError:
            pass
        except OSError as e:
            log.warning("[VRAMCoordinator] force-release failed: %s", e)

    # ---- public API -----------------------------------------------------------

    def status(self) -> Optional[LockInfo]:
        """Return current lock holder info, or None if unlocked."""
        if not self.lock_path.exists():
            return None
        return LockInfo.from_path(self.lock_path)

    def is_held(self) -> bool:
        """True if a live process holds the lock right now."""
        info = self.status()
        if info is None:
            return False
        return info.is_alive()

    @contextmanager
    def acquire(
        self,
        owner: str,
        job_id: str = "",
        timeout: float = 1800.0,
    ) -> Iterator["VRAMCoordinator"]:
        """Block until the GPU lock is held by us, or raise on timeout.

        Args:
            owner: short label like "hyworld_worker" or "bark".  Visible
                in the lock payload for debugging.
            job_id: optional job id (e.g. ``hw_xxx``) to embed in the payload.
            timeout: max wall-clock seconds to wait.  TimeoutError raised
                if we never get the lock.

        Reclaim policy: if the existing holder's PID is no longer alive,
        the lock is force-released and we try again immediately.  No
        sleep for that path.
        """
        start = time.monotonic()
        attempt = 0
        while True:
            attempt += 1
            if self._try_create():
                self._write_lock_payload(owner, job_id)
                log.info(
                    "[VRAMCoordinator] acquired by %s (job=%s, pid=%d)",
                    owner, job_id, os.getpid(),
                )
                try:
                    yield self
                finally:
                    self.release(owner)
                return

            # Already locked.  Inspect.
            info = self.status()
            if info is None:
                # Lock vanished between our create attempt and our read.
                # Loop without sleeping to retry immediately.
                continue

            if not info.is_alive():
                self._force_release(
                    f"prior owner pid={info.pid} ({info.owner}) is dead"
                )
                continue

            held_for = time.time() - info.acquired_at if info.acquired_at else 0
            if held_for > _STALE_WARN_SEC:
                log.warning(
                    "[VRAMCoordinator] lock held for %.0fs by %s (pid=%d) "
                    "-- still alive, not reclaiming",
                    held_for, info.owner, info.pid,
                )

            if (time.monotonic() - start) >= timeout:
                raise TimeoutError(
                    f"VRAMCoordinator: timed out after {timeout:.0f}s "
                    f"waiting for {info.owner} (pid={info.pid}, job={info.job_id})"
                )

            if self.verbose_log:
                log.debug(
                    "[VRAMCoordinator] attempt %d: held by %s pid=%d, retrying in %.1fs",
                    attempt, info.owner, info.pid, self.poll_interval_sec,
                )
            time.sleep(self.poll_interval_sec)

    def release(self, owner: str = "") -> bool:
        """Release the lock if we hold it.  Returns True if released.

        Safe to call when not held -- becomes a no-op.
        """
        info = self.status()
        if info is None:
            return False
        if info.pid != os.getpid():
            log.warning(
                "[VRAMCoordinator] release skipped: lock owned by pid=%d, not us (pid=%d)",
                info.pid, os.getpid(),
            )
            return False
        try:
            self.lock_path.unlink()
            log.info("[VRAMCoordinator] released by %s (pid=%d)", owner, os.getpid())
            return True
        except FileNotFoundError:
            return False
        except OSError as e:
            log.warning("[VRAMCoordinator] release failed: %s", e)
            return False
