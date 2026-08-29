"""Shared lifecycle helpers for the Path-B audio sidecars (chatterbox / dia).

Centralizes the things the polish roundtable flagged across both new adapters:

* :func:`read_protocol_line` -- a bounded, Windows-safe read of one protocol line
  (``select`` does not work on Windows pipes, so a daemon reader thread is used).
  A worker that hangs during model load or a forward can no longer block the
  ComfyUI render thread forever; the caller kills + raises a NAMED error instead.
* :func:`close_worker` -- idempotent teardown that ALWAYS closes the stdin/stdout
  pipes AND the stderr file handle and reaps the process (kill + wait), so a
  failed start / mid-request crash never leaks a handle or a zombie. Safe to call
  twice (tolerates already-closed handles) and with ``proc=None``.
* :func:`load_wav_as_audio` -- one shared load of a worker-written WAV into the
  main venv as an AUDIO dict (soundfile, never torchaudio.load), removing the
  temp file whether or not the read succeeds. torch/soundfile import lazily
  inside the call, so importing this module stays side-effect-free.

The proven IndexTTS2 adapter keeps its own inline lifecycle (byte-identical,
shipped); these helpers are for the NEW opt-in sidecars. Import is side-effect-free.
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import json
import os
import queue
import threading


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
    """Per-line forward timeout (s), configurable."""
    return _env_float("OTR_SIDECAR_REQUEST_TIMEOUT", 600.0)


def remove_quietly(path):
    """Best-effort temp-file removal (used on every failure path)."""
    try:
        if path:
            os.remove(path)
    except OSError:
        pass


def load_wav_as_audio(path, fallback_sample_rate):
    """Load a worker-written WAV into the main venv as an AUDIO dict. soundfile,
    NOT torchaudio.load -- the Blackwell venv routes load() through the
    uninstalled torchcodec backend. The temp file is removed even when the read
    fails (the worker's output dir must never accumulate)."""
    import soundfile as sf
    import torch
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # [T, C]
    finally:
        remove_quietly(path)
    wav = torch.from_numpy(data.T).contiguous()  # [C, T]
    return {"waveform": wav, "sample_rate": int(sr or fallback_sample_rate)}


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
    failure). Teardown must not mask a render result (I-7)."""
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
