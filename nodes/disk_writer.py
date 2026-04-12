"""
DiskWriter — Single-threaded serialized queue for all MP4 writes.

Writes execute only when no MemoryBoundary is in progress.
This prevents disk I/O from competing with VRAM reclamation,
which can cause CUDA context drops on Blackwell.

Doctrine: disk writes never overlap MemoryBoundary windows.
"""

import logging
import os
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger("OTR")


# Shared event: cleared during boundary, set otherwise.
# MemoryBoundary clears this before starting; sets it when done.
boundary_idle = threading.Event()
boundary_idle.set()  # Start in idle (writes allowed)


@dataclass
class WriteJob:
    """A queued MP4 write job."""
    src_path: Optional[str]          # Path to raw frames file or existing MP4
    dest_path: str                   # Final output path
    duration_s: float                # Target duration
    fps: int = 25
    width: int = 768
    height: int = 512
    completed: bool = field(default=False, init=False)
    error: Optional[str] = field(default=None, init=False)


class DiskWriter:
    """Single-threaded queue for all MP4 writes.

    Usage:
        writer = DiskWriter()
        writer.enqueue(src_path, dest_path, duration_s)
        writer.enqueue(src_path2, dest_path2, duration_s2)
        completed_paths = writer.drain(timeout_s=300)
    """

    def __init__(self):
        self._queue: deque[WriteJob] = deque()
        self._lock = threading.Lock()

    def enqueue(self, src_path: str, dest_path: str, duration_s: float,
                fps: int = 25, width: int = 768, height: int = 512) -> None:
        """Add a write job to the queue."""
        job = WriteJob(
            src_path=src_path,
            dest_path=dest_path,
            duration_s=duration_s,
            fps=fps,
            width=width,
            height=height,
        )
        with self._lock:
            self._queue.append(job)
        log.info("[DiskWriter] Enqueued: %s (%.1fs)",
                 os.path.basename(dest_path), duration_s)

    def drain(self, timeout_s: float = 300) -> list:
        """Process all queued writes sequentially. Returns list of completed paths.

        Blocks until all jobs are done or timeout is reached.
        Each write waits for boundary_idle before starting.
        """
        completed = []
        deadline = _monotonic() + timeout_s

        while True:
            with self._lock:
                if not self._queue:
                    break
                job = self._queue.popleft()

            remaining = deadline - _monotonic()
            if remaining <= 0:
                log.warning("[DiskWriter] Timeout reached, %d jobs remaining",
                            len(self._queue) + 1)
                break

            # Wait for boundary to finish before writing
            if not boundary_idle.wait(timeout=min(remaining, 30)):
                log.warning("[DiskWriter] Boundary still active after 30s, proceeding anyway")

            try:
                self._execute_write(job)
                completed.append(job.dest_path)
            except Exception as e:
                job.error = str(e)
                log.error("[DiskWriter] Write failed for %s: %s",
                          os.path.basename(job.dest_path), e)

            job.completed = True

        log.info("[DiskWriter] Drained %d/%d jobs",
                 len(completed), len(completed) + len(self._queue))
        return completed

    @staticmethod
    def _execute_write(job: WriteJob) -> None:
        """Execute a single write job using FFmpeg."""
        os.makedirs(os.path.dirname(job.dest_path) or ".", exist_ok=True)

        if job.src_path and os.path.isfile(job.src_path):
            # Copy/re-encode existing file to target duration
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", job.src_path,
                "-t", f"{job.duration_s:.3f}",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-an",
                job.dest_path,
            ]
        else:
            raise FileNotFoundError(f"Source not found: {job.src_path}")

        result = subprocess.run(
            ffmpeg_cmd, capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed (rc={result.returncode}): {result.stderr[-300:]}"
            )


def _monotonic():
    """Monotonic clock for timeout calculations."""
    import time
    return time.monotonic()
