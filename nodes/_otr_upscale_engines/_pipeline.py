"""Helpers for the upscale model pipeline: bounded pipe reads, ffmpeg subprocess
lifecycle, tensor fit-and-pad, engine output validation.

The centerpiece is `_run_model_pipeline` (defined in
`nodes/otr_silent_composite.py`, not here) which implements the
FFMPEG-owns-TIME + MODEL-owns-SPACE split:

* ffmpeg decoder runs `trim,setpts,fps,tpad` with `-frames:v n_frames` on the
  decode side so the Python loop receives exactly n_frames of conformed CFR
  source-resolution frames.
* Python loop is a dumb spatial map: read → per-frame model call → decrease-fit
  + pad to canvas → write to encoder stdin.
* ffmpeg encoder has its own `-frames:v n_frames` cap.
* `count_video_frames(seg_path) == n_frames` asserted before concat (Codex r3
  MF-1's non-negotiable guard).

Cold-import clean: this module is stdlib-only at import time; torch/numpy are
imported lazily inside the helpers that need them.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import typing

if typing.TYPE_CHECKING:
    import torch


# ---------------------------------------------------------------------------
# Bounded pipe reads
# ---------------------------------------------------------------------------
def _read_frames_exact(pipe, bytes_per_frame: int, n_frames: int) -> bytes:
    """Read exactly ``n_frames * bytes_per_frame`` bytes from ``pipe``.

    On EOF, returns whatever frame-aligned bytes it did read (may be less than
    ``n_frames * bytes_per_frame``, MUST be a multiple of ``bytes_per_frame``).
    Raises :class:`RuntimeError` if the tail is non-frame-aligned (pipe ended
    mid-frame — a signal of decoder failure).

    Caller distinguishes "EOF before n_frames" from "EOF at n_frames" by
    checking the return length against `n_frames * bytes_per_frame`.
    """
    if bytes_per_frame <= 0:
        raise ValueError(f"bytes_per_frame must be positive; got {bytes_per_frame}")
    if n_frames <= 0:
        return b""
    want = int(n_frames) * int(bytes_per_frame)
    buf = bytearray()
    while len(buf) < want:
        chunk = pipe.read(want - len(buf))
        if not chunk:
            break  # EOF
        buf.extend(chunk)
    if len(buf) % bytes_per_frame != 0:
        raise RuntimeError(
            f"decoder EOF at non-frame-aligned position: got {len(buf)} bytes, "
            f"bytes_per_frame={bytes_per_frame} (short read at frame boundary)")
    return bytes(buf)


# ---------------------------------------------------------------------------
# Subprocess lifecycle (BUG_BIBLE 09.02 pipe rules)
# ---------------------------------------------------------------------------
def _tempfile_stderr(prefix: str) -> "tuple":
    """Open a NamedTemporaryFile in write-binary mode for a subprocess's stderr.

    Returns ``(fileobj, path_str)``. Caller passes ``fileobj`` to
    ``subprocess.Popen(stderr=fileobj)`` and closes it AFTER the subprocess is
    fully drained. ``_tail_path(path_str, n)`` reads back the tail on failure
    (Antigravity r3 MF-2: reading via file OBJECT after the subprocess writes
    to it gets zero bytes because the offset is at EOF; reading via the PATH
    with a fresh handle is safe).
    """
    fd, path = tempfile.mkstemp(prefix=f"otr_upscale_{prefix}_", suffix=".stderr")
    fileobj = os.fdopen(fd, "wb")
    return (fileobj, path)


def _close_stderr(*fileobjs) -> None:
    """Close each fileobj; swallow already-closed."""
    for f in fileobjs:
        try:
            if not f.closed:
                f.flush()
                f.close()
        except (ValueError, OSError):
            pass


def _tail_path(path: str, n: int) -> str:
    """Read the last ``n`` bytes of ``path`` as text (UTF-8, errors replaced).

    Handles the Antigravity r3 MF-2 case: the subprocess wrote to a file
    handle whose offset is at EOF; the path-based re-open reads from the start.
    """
    try:
        with open(path, "rb") as f:
            data = f.read()
        tail = data[-int(n):] if int(n) > 0 else data
        return tail.decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


def _kill_and_wait_all(procs, timeout: float = 30.0) -> None:
    """For each subprocess in ``procs``: kill(), then wait(timeout).

    Called ONLY on the error path or on wait-timeout. Never call this on a
    successful (already-waited) process. `returncode is None` after wait is
    treated as a failure by the caller.
    """
    for p in procs:
        try:
            if p.poll() is None:
                p.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            p.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            pass


def _unlink_if_exists(path) -> None:
    """rm the file, swallow FileNotFoundError only."""
    try:
        os.unlink(str(path))
    except FileNotFoundError:
        pass
    except (OSError, TypeError):
        pass


# ---------------------------------------------------------------------------
# Video probe (thin wrapper over nodes/otr_silent_composite.py:probe_video)
# ---------------------------------------------------------------------------
def _rate_is_numeric(rate) -> bool:
    """Both halves of an ffprobe rate are NUMBERS, so a zero is a reported zero.

    This module's own tolerance, and it stays here: ``"0/0"`` is what ffprobe
    writes for a container that records no rate, and the only caller of
    :func:`_probe_video_dims` reads width and height. A field that is not a
    number at all is a different thing and still fails loud.
    """
    parts = str(rate).split("/")
    if len(parts) > 2:
        return False
    try:
        for part in parts:
            float(part)
    except (TypeError, ValueError):
        return False
    return True


def _probe_video_dims(ffmpeg: str, src: str):
    """Probe ``src`` and return ``(int(width), int(height), float(fps))``.

    Wraps ``nodes/otr_silent_composite.py:probe_video`` (which returns a
    string-valued dict) and converts to concrete types. Sonnet 5 MF-1:
    ``ffprobe``'s ``avg_frame_rate`` / ``r_frame_rate`` come back as rational
    STRINGS like ``"25/1"`` or ``"30000/1001"`` -- ``float()`` on those raises
    ``ValueError``. That N/D split used to be COPIED here from two other
    files; lean-mean order 8 moved the rule to
    ``nodes/_otr_shared/ffprobe.py:parse_rate`` so there is one of it.

    Raises :class:`RuntimeError` on missing keys OR unparseable rate.
    """
    # Late import to keep this module cold-import clean.
    from .._otr_shared import ffprobe as _ffprobe  # noqa: WPS433 - deliberate late import
    from ..otr_silent_composite import probe_video  # noqa: WPS433 - deliberate late import

    info = probe_video(str(src))
    if not info or "width" not in info or "height" not in info:
        raise RuntimeError(
            f"ffprobe returned incomplete dims for {src!r}: {info!r}")
    try:
        w = int(info["width"])
        h = int(info["height"])
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            f"ffprobe returned non-int dims for {src!r}: {info!r} :: {e}")
    # Rate MAY be avg_frame_rate or r_frame_rate; prefer avg_frame_rate if present.
    rate = info.get("avg_frame_rate") or info.get("r_frame_rate") or ""
    if not rate:
        raise RuntimeError(
            f"ffprobe returned no frame rate for {src!r}: {info!r}")
    fps = _ffprobe.parse_rate(rate)
    if fps is None:
        if not _rate_is_numeric(rate):
            raise RuntimeError(
                f"ffprobe rate {rate!r} for {src!r} is not a valid frame rate")
        fps = 0.0
    return (w, h, fps)


# ---------------------------------------------------------------------------
# Engine output validation
# ---------------------------------------------------------------------------
def _validate_engine_output(out, expected_scale: int, in_shape) -> None:
    """Assert BHWC rank-4, batch preserved, `H_out = H_in * expected_scale`,
    dtype float, finite values.

    Raises :class:`RuntimeError` on any mismatch so a malformed adapter output
    fails LOUD at the pipe layer instead of corrupting the encoded segment.
    """
    import torch

    if not isinstance(out, torch.Tensor):
        raise RuntimeError(
            f"engine.upscale_frames returned non-Tensor: {type(out).__name__}")
    if out.ndim != 4:
        raise RuntimeError(
            f"engine.upscale_frames returned rank {out.ndim} != 4 "
            f"(expected BHWC): shape={tuple(out.shape)}")
    in_b, in_h, in_w, in_c = tuple(in_shape)
    out_b, out_h, out_w, out_c = tuple(out.shape)
    if out_b != in_b:
        raise RuntimeError(
            f"batch dim changed: in={in_b} out={out_b}")
    if out_c != in_c:
        raise RuntimeError(
            f"channel dim changed: in={in_c} out={out_c}")
    if out_h != in_h * int(expected_scale) or out_w != in_w * int(expected_scale):
        raise RuntimeError(
            f"spatial scale wrong: in=({in_h},{in_w}) x{expected_scale} = "
            f"({in_h*expected_scale},{in_w*expected_scale}) but got "
            f"({out_h},{out_w})")
    if not out.dtype.is_floating_point:
        raise RuntimeError(
            f"engine output dtype {out.dtype} is not floating-point")
    if not bool(torch.isfinite(out).all()):
        raise RuntimeError("engine output contains non-finite values (nan/inf)")


# ---------------------------------------------------------------------------
# Fit-and-pad (decrease-fit both directions + black-pad to canvas)
# ---------------------------------------------------------------------------
def _fit_and_pad_bhwc(x, canvas_w: int, canvas_h: int):
    """Decrease-fit BOTH directions (upscale if under, downscale if over) then
    black-pad to (canvas_h, canvas_w).

    Mirrors ffmpeg's ``scale=W:H:force_original_aspect_ratio=decrease`` +
    ``pad=W:H:(ow-iw)/2:(oh-ih)/2:color=black``.

    Sonnet r3 SF-1: pad while BCHW using the standard 4-tuple
    ``(pad_l, pad_r, pad_t, pad_b)`` to avoid PyTorch's BHWC 6-tuple edge
    cases. Antigravity r4 MF-2: return ``.contiguous()`` so downstream
    ``.numpy()`` doesn't crash on stride mismatch.
    """
    import torch
    import torch.nn.functional as F

    if x.ndim != 4:
        raise ValueError(f"_fit_and_pad_bhwc expects BHWC rank-4; got {x.ndim}")
    _, h, w, _ = x.shape
    scale = min(float(canvas_w) / float(w), float(canvas_h) / float(h))
    new_w = max(1, int(round(float(w) * scale)))
    new_h = max(1, int(round(float(h) * scale)))
    # BHWC -> BCHW for interpolate + pad
    x = x.permute(0, 3, 1, 2).contiguous()
    x = F.interpolate(x, size=(new_h, new_w), mode="bicubic",
                      align_corners=False, antialias=True)
    pad_l = (canvas_w - new_w) // 2
    pad_r = canvas_w - new_w - pad_l
    pad_t = (canvas_h - new_h) // 2
    pad_b = canvas_h - new_h - pad_t
    x = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=0.0)
    # BCHW -> BHWC and force contiguity for downstream .numpy()
    x = x.permute(0, 2, 3, 1).contiguous()
    return x


__all__ = [
    "_read_frames_exact",
    "_tempfile_stderr",
    "_close_stderr",
    "_tail_path",
    "_kill_and_wait_all",
    "_unlink_if_exists",
    "_probe_video_dims",
    "_validate_engine_output",
    "_fit_and_pad_bhwc",
]
