"""Raw-video ffmpeg sink helpers for render profiling.

This module is intentionally small and torch-free. Runtime renderers can keep
their existing encoders while profiling scripts use this shared sink to measure
pipe/write time separately from PIL frame drawing.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from .scope_draw import cfr_flags


def find_ffmpeg(ffmpeg: str = "ffmpeg") -> Optional[str]:
    if ffmpeg and (shutil.which(ffmpeg) or os.path.isfile(ffmpeg)):
        return ffmpeg
    return shutil.which("ffmpeg")


def has_nvenc(ffmpeg: str) -> bool:
    try:
        out = subprocess.run(
            [ffmpeg, "-hide_banner", "-codecs"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "h264_nvenc" in (out.stdout or "")
    except Exception:  # noqa: BLE001 -- profiling helper: absence just means CPU encode.
        return False


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
        self.proc: Optional[subprocess.Popen] = None
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
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
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

