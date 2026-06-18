"""Shared, dep-free subprocess-sidecar lifecycle helpers (A-Seam, promoted).

The model-agnostic video / image / 3D platforms isolate a dependency-contaminating
engine (one that pins its own torch / cu128 stack) in its OWN venv as a supervised
subprocess worker driven over line-delimited JSON (invariant V-12). The proven
shape for that worker lifecycle was first shipped for the Path-B AUDIO sidecars
(``nodes/_otr_audio_engines/_otr_sidecar.py``); this module promotes the SAME
helpers to the shared namespace so the cu128-isolated video sidecars and the
later C image + B 3D sidecars import ONE copy instead of each re-deriving it --
exactly as ``gpu_residency`` (AS-3) was promoted out of A.

Why a copy and not an import of the audio module: importing anything under
``nodes/_otr_audio_engines`` pulls that package's ``__init__`` (which hard-imports
torch at module scope) and would break the video cold-import invariant (V-12).
This module imports only the stdlib (``json`` / ``os`` / ``queue`` / ``threading``)
so importing it pulls in NOTHING heavy; the audio package keeps its own frozen
copy and is untouched.

* :func:`read_protocol_line` -- a bounded, Windows-safe read of one protocol line
  (``select`` does not work on Windows pipes, so a daemon reader thread is used).
  A worker that hangs during model load or a forward can no longer block the
  ComfyUI render thread forever; the caller kills + raises a NAMED error instead.
* :func:`close_worker` -- idempotent teardown that ALWAYS closes the stdin/stdout
  pipes AND the stderr file handle and reaps the process (kill + wait), so a
  failed start / mid-request crash never leaks a handle or a zombie. Safe to call
  twice (tolerates already-closed handles) and with ``proc=None``.

Import is side-effect-free. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import json
import os
import queue
import threading

__all__ = [
    "startup_timeout",
    "request_timeout",
    "remove_quietly",
    "read_protocol_line",
    "close_worker",
]


def _env_float(name, default):
    try:
        v = float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        v = float(default)
    # A non-positive timeout would make queue.get raise ValueError (not a
    # protocol error the callers handle); fall back to the default.
    return v if v > 0 else float(default)


def startup_timeout():
    """Readiness timeout (s). Generous + configurable: the first worker line may
    include a one-time model DOWNLOAD on a fresh install."""
    return _env_float("OTR_SIDECAR_STARTUP_TIMEOUT", 1800.0)


def request_timeout():
    """Per-request forward timeout (s), configurable."""
    return _env_float("OTR_SIDECAR_REQUEST_TIMEOUT", 600.0)


def remove_quietly(path):
    """Best-effort temp-file removal (used on every failure path)."""
    try:
        if path:
            os.remove(path)
    except OSError:
        pass


def read_protocol_line(proc, timeout, what):
    """Read ONE line from ``proc.stdout`` with a timeout (daemon reader thread).

    Returns the line (str). Raises ``TimeoutError`` if nothing arrives within
    ``timeout`` seconds, or ``EOFError`` if the worker closed stdout (died). On
    timeout the orphaned thread is harmless: the caller kills the process, so its
    pending ``readline`` returns EOF and the thread exits.
    """
    q: "queue.Queue" = queue.Queue(maxsize=1)

    def _reader():
        try:
            q.put(proc.stdout.readline())
        except BaseException as exc:  # noqa: BLE001 -- surfaced as EOF to the caller
            q.put(exc)

    threading.Thread(target=_reader, daemon=True).start()
    try:
        item = q.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError("%s timed out after %.0fs" % (what, timeout))
    if isinstance(item, BaseException):
        raise EOFError("%s read failed: %s" % (what, item))
    if item == "":
        raise EOFError("%s: worker closed stdout" % what)
    return item


def close_worker(proc, stderr_handle):
    """Idempotent teardown: best-effort graceful stop, then reap (kill + wait),
    close the stdin/stdout pipes, and close the stderr handle. Never raises and
    tolerates already-closed handles (double-close from unload after a request
    failure). Teardown must not mask a render result."""
    if proc is not None:
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"stop": True}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                pass
        for pipe in (getattr(proc, "stdin", None), getattr(proc, "stdout", None)):
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:  # noqa: BLE001 -- already-closed is fine
                    pass
    if stderr_handle is not None:
        try:
            stderr_handle.close()
        except Exception:  # noqa: BLE001 -- close() on a closed file raises ValueError on 3.11+
            pass
