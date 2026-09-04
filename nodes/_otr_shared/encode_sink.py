"""Raw-video ffmpeg sink helpers for render profiling.

This module is intentionally small and torch-free. Runtime renderers can keep
their existing encoders while profiling scripts use this shared sink to measure
pipe/write time separately from PIL frame drawing.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time
from pathlib import Path
from typing import Optional

from .ffmpeg import resolve_ffmpeg
from .scope_draw import cfr_flags

try:
    from . import proc as otr_proc
except ImportError:  # pragma: no cover -- loaded flat
    try:
        from _otr_shared import proc as otr_proc  # type: ignore  # nodes/ on sys.path
    except ImportError:
        import proc as otr_proc  # type: ignore  # _otr_shared/ on sys.path


def find_ffmpeg(ffmpeg: str = "ffmpeg") -> Optional[str]:
    """The pack's ONE ffmpeg answer (``.ffmpeg.resolve_ffmpeg``). This copy
    never read ``OTR_FFMPEG`` at all and treated its signature default as a
    choice; the owner does neither."""
    return resolve_ffmpeg(ffmpeg)


#: Cached for the process, PER BINARY. The probe costs about a second and the
#: answer cannot change while we run -- but it is an answer about ONE build,
#: so the key is the normalized binary path (kibitz r2, 2026-09-04): a caller
#: naming a different ffmpeg gets its own probe, not the first caller's.
_NVENC_PROBE: dict = {}
#: The probe takes up to twenty seconds; two first callers racing on the same
#: binary must run it once, not twice (the lock this replaced lived in
#: `video_engine`, in front of a cache that is gone -- it belongs HERE).
_NVENC_PROBE_LOCK = threading.Lock()


def has_nvenc(ffmpeg: str) -> bool:
    """Can h264_nvenc actually ENCODE here -- not merely exist in the build.

    THE OBVIOUS TEST IS WRONG AND COST A WHOLE EPISODE (2026-08-30). This used
    to answer with ``"h264_nvenc" in (ffmpeg -codecs)``, which reports whether
    ffmpeg was COMPILED with nvenc. On any container that ships a full ffmpeg
    without the NVIDIA encode library -- the normal case on rented GPUs --
    that is true while encoding is impossible:

        [h264_nvenc] Cannot load libnvidia-encode.so.1
        [vost#0:0/h264_nvenc] Error while opening encoder
        Conversion failed!

    Callers stream raw frames into ffmpeg's stdin, so a dead encoder surfaces
    as BrokenPipeError or a RuntimeError on the first write -- eighteen minutes
    into a render on the leg that found this, long after the script and audio
    were finished, naming nothing useful.

    **This is the SINGLE SOURCE OF TRUTH for nvenc, and every other site
    delegates to it.** There were two identical string tests, here and in
    ``video_engine``; fixing one left the other to fail the same render at a
    later node, which is exactly what happened. The node now delegates here.

    THAT CLAIM WAS WRONG FOR FOUR DAYS, AND IT READ AS COVERAGE (2026-09-03).
    This docstring said "the ONLY nvenc decision in the pack" while a THIRD
    string test still lived in ``scope_draw._has_nvenc`` -- and the four viz_*
    engines encode through that module, not through :class:`RawVideoSink`, so
    they never reached this probe. A rented 4090 that lists h264_nvenc and
    cannot open a session found it. ``scope_draw`` now delegates here too, and
    ``tests/test_nvenc_single_decision.py`` fails if a fourth copy appears.

    Probes at 256x256 deliberately: NVENC rejects tiny frames outright with
    "Frame Dimension less than the minimum supported value", so a 64x64 probe
    reports a HEALTHY card as unavailable and silently drops it to CPU.
    """
    if not ffmpeg:
        return False
    # A BARE name would key the cache on <cwd>/ffmpeg and probe whatever
    # PATH says; resolve it through the owner first (the pin, then PATH). An
    # explicit path is the caller's choice and is probed as given.
    ffmpeg = str(ffmpeg)
    if not os.path.dirname(ffmpeg):
        ffmpeg = resolve_ffmpeg(ffmpeg)
        if not ffmpeg:
            # Nothing to probe and nothing to remember: a name that does
            # not resolve must not key the cache on <cwd>/name (cursor r4).
            return False
    key = os.path.normcase(os.path.abspath(ffmpeg))
    with _NVENC_PROBE_LOCK:
        cached = _NVENC_PROBE.get(key)
        if cached is not None:
            return cached
        try:
            out = otr_proc.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "nullsrc=s=256x256:d=0.1",
                 "-c:v", "h264_nvenc", "-frames:v", "1",
                 "-f", "null", "-"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            verdict = out.returncode == 0
        except Exception:  # noqa: BLE001 -- a probe must never be fatal.
            verdict = False
        _NVENC_PROBE[key] = verdict
        return verdict


@dataclass
class RawVideoSinkStats:
    mode: str
    frames: int = 0
    bytes_written: int = 0
    pipe_seconds: float = 0.0
    encode_seconds: float = 0.0
    codec: str = ""
    used_nvenc: bool = False
    output_path: str = ""


class RawVideoSink:
    """Context-managed raw RGB24 frame sink.

    ``mode='none'`` measures draw-only code without spawning ffmpeg.
    ``mode='null'`` pipes frames through ffmpeg to the null muxer.
    ``mode='mp4'`` writes a silent yuv420p mp4.
    """

    def __init__(
        self,
        *,
        mode: str,
        width: int,
        height: int,
        fps: int,
        ffmpeg: str = "ffmpeg",
        output_path: Optional[Path] = None,
        prefer_nvenc: bool = True,
    ) -> None:
        self.mode = str(mode or "none").lower()
        if self.mode not in {"none", "null", "mp4"}:
            raise ValueError(f"RawVideoSink: unsupported mode {mode!r}")
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.ffmpeg = ffmpeg
        self.output_path = Path(output_path) if output_path else None
        self.prefer_nvenc = bool(prefer_nvenc)
        self.proc: Optional[otr_proc.Popen] = None
        self.stats = RawVideoSinkStats(mode=self.mode)

    def __enter__(self) -> "RawVideoSink":
        if self.mode == "none":
            return self
        fb = find_ffmpeg(self.ffmpeg)
        if not fb:
            raise RuntimeError("RawVideoSink: ffmpeg not found.")
        use_nvenc = self.prefer_nvenc and has_nvenc(fb)
        codec = "h264_nvenc" if use_nvenc else "libx264"
        cmd = [
            fb, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "rgb24", "-r", str(self.fps), "-i", "-",
            "-an",
        ]
        if self.mode == "null":
            cmd += ["-f", "null", "-"]
        else:
            if self.output_path is None:
                raise ValueError("RawVideoSink: output_path is required for mp4 mode.")
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd += ["-c:v", codec]
            if use_nvenc:
                cmd += ["-preset", "p5", "-rc", "vbr", "-b:v", "8M"]
            else:
                cmd += ["-preset", "medium", "-crf", "20"]
            cmd += [
                "-pix_fmt", "yuv420p", *cfr_flags(fb), "-r", str(self.fps),
                "-color_primaries", "bt709", "-color_trc", "bt709",
                "-colorspace", "bt709", "-movflags", "+faststart",
                str(self.output_path),
            ]
        self.proc = otr_proc.popen(
            cmd,
            stdin=otr_proc.PIPE,
            stdout=otr_proc.DEVNULL,
            stderr=otr_proc.PIPE,
        )
        self.stats.codec = codec
        self.stats.used_nvenc = use_nvenc
        self.stats.output_path = str(self.output_path or "")
        return self

    def write(self, frame) -> None:
        if hasattr(frame, "tobytes"):
            payload = frame.tobytes()
        elif isinstance(frame, (bytes, bytearray, memoryview)):
            payload = bytes(frame)
        else:
            raise TypeError("RawVideoSink.write expects a numpy-like frame or bytes.")
        self.stats.frames += 1
        self.stats.bytes_written += len(payload)
        if self.mode == "none":
            return
        if self.proc is None or self.proc.stdin is None:
            raise RuntimeError("RawVideoSink.write called before ffmpeg started.")
        t0 = time.perf_counter()
        self.proc.stdin.write(payload)
        self.stats.pipe_seconds += time.perf_counter() - t0

    def close(self) -> RawVideoSinkStats:
        if self.mode == "none":
            return self.stats
        if self.proc is None:
            return self.stats
        t0 = time.perf_counter()
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace") if self.proc.stderr else ""
        self.proc.wait()
        self.stats.encode_seconds += time.perf_counter() - t0
        if self.proc.returncode != 0:
            raise RuntimeError(f"RawVideoSink: ffmpeg failed: {err[-800:]}")
        return self.stats

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None and self.proc is not None:
            try:
                if self.proc.stdin is not None:
                    self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.kill()
            except OSError:
                pass
            return None
        self.close()
        return None

