"""Cross-process single-heavy-engine GPU residency lease (A-Seam AS-3).

The model-agnostic video platform keeps ONE heavy engine resident at a time
(invariant: 14.5 GB peak / 14.0 GB 3D sub-ceiling). Heavy work runs either in
the main ComfyUI process or in a cu128 subprocess sidecar (V-12 dependency
isolation). Those are SEPARATE OS processes, so an in-process lock cannot
serialise them. This module is the shared, cross-process mutex they all take
before loading weights -- promoted out of A so C's image sidecars and B's 3D
sidecars import the SAME lease (one lock, one resident heavy engine).

Primitive: an atomic lock DIRECTORY (``os.mkdir`` is atomic on Windows and
POSIX -- it fails if the directory already exists). The holder writes its PID +
a unique token inside. A waiter that finds the lock held checks whether the
owning PID is still alive; a lock left by a crashed process (dead PID) is
reclaimed, so a Cancel/crash never wedges the platform forever.

VRAM is probed MACHINE-WIDE via pynvml (the nvidia-smi truth -- every process
on the device), NEVER ComfyUI's ``get_free_memory()`` (this-process allocator
view only; it cannot see a sidecar's allocation). ``probe_used_mb`` mirrors the
``_nvml_used_bytes`` pattern in ``nodes/vram_context_test.py``.

Import-time is side-effect-free and dependency-free at module scope (stdlib
only -- no torch, no pynvml, no psutil). pynvml / psutil / ctypes are imported
lazily inside the functions that need them, so importing this module pulls in
nothing heavy (invariant V-12, the cold-import test).
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

log = logging.getLogger("OTR.gpu_residency")

#: Lock-directory name. The PARENT dir defaults to the OS temp root (a stable,
#: machine-wide, NON-OneDrive location); override the parent with
#: ``OTR_GPU_LEASE_DIR`` for tests or a multi-GPU split.
_LOCKDIR_NAME = "otr_gpu_residency.lockdir"
_OWNER_FILE = "owner"

#: bytes -> MB divisor (NVML reports bytes).
_MB = 1024 * 1024


def _default_lockdir() -> Path:
    parent = os.getenv("OTR_GPU_LEASE_DIR") or tempfile.gettempdir()
    return Path(parent) / _LOCKDIR_NAME


class LeaseTimeout(TimeoutError):
    """Raised by :func:`acquire` when the lease is still held by a LIVE owner
    after the timeout. FAIL CLOSED -- the caller must NOT load a heavy engine."""


class LeaseError(RuntimeError):
    """Raised when the lock directory cannot be created for a non-contention
    reason (e.g. the parent path is unwritable). FAIL CLOSED."""


def _pid_alive(pid: int) -> bool:
    """True if process ``pid`` is still running. Fail-SAFE: when liveness is
    uncertain (access-denied, unknown error) treat the owner as ALIVE so a live
    lock is never stolen. Only a definitively-absent PID returns False."""
    if not pid or int(pid) <= 0:
        return False
    pid = int(pid)
    try:
        import psutil  # type: ignore
        return bool(psutil.pid_exists(pid))
    except Exception:  # noqa: BLE001  (psutil absent/erroring -> manual check)
        pass
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            SYNCHRONIZE = 0x00100000
            handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            err = ctypes.get_last_error()
            ERROR_INVALID_PARAMETER = 87  # only this reliably means "no such pid"
            return err != ERROR_INVALID_PARAMETER
        except Exception:  # noqa: BLE001
            return True  # uncertain -> do not steal a possibly-live lock
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:  # noqa: BLE001
        return True
    return True


def _read_owner(lockdir: Path):
    """Return ``(pid, token)`` of the lock holder, or ``None`` if unreadable.
    A half-written / unreadable owner file is treated as held-by-unknown."""
    try:
        raw = (lockdir / _OWNER_FILE).read_text(encoding="utf-8").splitlines()
        pid = int(raw[0].strip())
        token = raw[1].strip() if len(raw) > 1 else ""
        return pid, token
    except Exception:  # noqa: BLE001
        return None


def read_owner(lockdir: Optional[Path] = None):
    """Public: current ``(pid, token)`` holder of the lease, or ``None``."""
    return _read_owner(Path(lockdir) if lockdir else _default_lockdir())


def is_held(lockdir: Optional[Path] = None) -> bool:
    """True if the lease directory currently exists (held or stale)."""
    d = Path(lockdir) if lockdir else _default_lockdir()
    return d.is_dir()


class LeaseContext:
    """A held GPU-residency lease. Use as a context manager (releases on exit)
    or call :meth:`release` explicitly. ``release`` is idempotent and only
    removes the lock if THIS context still owns it (token match) -- if another
    process reclaimed it after a crash, the reclaimer's lock is left alone."""

    def __init__(self, lockdir: Path, pid: int, token: str, acquired_at: float):
        self.lockdir = lockdir
        self.pid = pid
        self.token = token
        self.acquired_at = acquired_at
        self._released = False

    def __enter__(self) -> "LeaseContext":
        return self

    def __exit__(self, *exc) -> bool:
        self.release()
        return False

    @property
    def held(self) -> bool:
        return not self._released

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        owner = _read_owner(self.lockdir)
        if owner is not None and owner[1] != self.token:
            log.warning(
                "[gpu_residency] lease reclaimed by pid=%s; release is a no-op",
                owner[0],
            )
            return
        shutil.rmtree(self.lockdir, ignore_errors=True)


def acquire(timeout_s: float = 120.0, poll_s: float = 0.25,
            lockdir: Optional[Path] = None) -> LeaseContext:
    """Take the single-heavy-engine lease; FAIL CLOSED.

    Atomically creates the lock directory (``os.mkdir``). If it already exists,
    waits -- unless the owning PID is dead (crash/Cancel), in which case the
    stale lock is reclaimed and re-taken. Raises :class:`LeaseTimeout` if the
    lease is still held by a LIVE owner after ``timeout_s``; raises
    :class:`LeaseError` if the directory cannot be created at all.
    """
    d = Path(lockdir) if lockdir else _default_lockdir()
    d.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    token = uuid.uuid4().hex
    pid = os.getpid()
    while True:
        try:
            os.mkdir(d)
        except FileExistsError:
            owner = _read_owner(d)
            if owner is not None and not _pid_alive(owner[0]):
                log.warning(
                    "[gpu_residency] reclaiming stale lease from dead pid=%s",
                    owner[0],
                )
                shutil.rmtree(d, ignore_errors=True)
                continue
            if time.monotonic() >= deadline:
                held_pid = owner[0] if owner else "?"
                raise LeaseTimeout(
                    f"gpu residency lease held by live pid={held_pid}; "
                    f"timed out after {timeout_s:g}s"
                )
            time.sleep(max(0.01, float(poll_s)))
            continue
        except OSError as exc:
            raise LeaseError(f"cannot create lease dir {d}: {exc}") from exc
        # mkdir won the race -> stamp ownership.
        try:
            (d / _OWNER_FILE).write_text(
                f"{pid}\n{token}\n{time.time()}\n", encoding="utf-8",
            )
        except OSError as exc:
            shutil.rmtree(d, ignore_errors=True)
            raise LeaseError(f"cannot stamp lease owner in {d}: {exc}") from exc
        log.info("[gpu_residency] lease acquired pid=%s token=%s", pid, token[:8])
        return LeaseContext(d, pid, token, time.monotonic())


def release(lease: Optional[LeaseContext]) -> None:
    """Release a lease acquired by :func:`acquire` (idempotent)."""
    if lease is not None:
        lease.release()


def probe_used_mb(device_index: int = 0) -> int:
    """MACHINE-WIDE VRAM used on the device, in MB (nvidia-smi truth: every
    process on the device). Returns 0 if NVML is unavailable -- callers that
    gate on a floor MUST distinguish that case (see :func:`nvml_available`);
    NEVER use ComfyUI ``get_free_memory()`` here (this-process allocator view
    only -- it cannot see a sidecar's allocation)."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(int(device_index))
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            return int(info.used) // _MB
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        return 0


def nvml_available(device_index: int = 0) -> bool:
    """True if pynvml can read the device, so a 0 from :func:`probe_used_mb`
    means 'empty' rather than 'no NVML'. Lets a floor-gate fail closed."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        try:
            pynvml.nvmlDeviceGetHandleByIndex(int(device_index))
            return True
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        return False


def wait_until_below_mb(threshold_mb: int, attempts: int = 3, sleep_s: float = 2.0,
                        device_index: int = 0) -> bool:
    """Bounded retry (sleep + re-probe) that machine-wide used VRAM has fallen
    below ``threshold_mb`` -- absorbs Windows teardown latency after a sidecar
    exits (PASS-PM A-S3 guard). Returns True as soon as it does, False if still
    above after ``attempts``. If NVML is unavailable, returns False (fail
    closed -- the caller cannot prove the device is clear)."""
    if not nvml_available(device_index):
        return False
    attempts = max(1, int(attempts))
    for i in range(attempts):
        if probe_used_mb(device_index) < int(threshold_mb):
            return True
        if i < attempts - 1:
            time.sleep(max(0.0, float(sleep_s)))
    return False


__all__ = [
    "LeaseContext", "LeaseTimeout", "LeaseError",
    "acquire", "release", "probe_used_mb", "nvml_available",
    "wait_until_below_mb", "is_held", "read_owner",
]
