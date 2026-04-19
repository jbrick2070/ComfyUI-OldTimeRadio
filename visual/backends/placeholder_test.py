"""
otr_v2.visual.backends.placeholder_test  --  Day 1 canary
==========================================================

Minimum-viable backend used to verify the ``OTR_VISUAL_BACKEND``
dispatch plumbing end-to-end before FLUX / PuLID / LTX / Wan2.1 /
Florence-2 land in Days 2-7 of the sprint.

Contract:
    1. Read shotlist.json from the job dir.
    2. Write ``STATUS.json`` -> RUNNING.
    3. Emit a 2x2 solid-color PNG (``render.png``) per shot with a
       ``meta.json`` tagged ``backend=placeholder_test``.
    4. Write ``STATUS.json`` -> READY with shot count in detail.

No GPU, no ffmpeg, no third-party deps.  Safe to run while Bark TTS
holds the card.  Explicitly NOT wired into production workflows --
the real stack lives in ``flux_anchor.py``, ``pulid_portrait.py``,
``ltx_motion.py``, ``wan_i2v.py``, ``florence2_mask.py`` (each gets
its own file as it lands).
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

from ._base import (
    Backend,
    STATUS_READY,
    STATUS_RUNNING,
    STATUS_ERROR,
    write_status,
    atomic_write_json,
)


def _tiny_png(path: Path, r: int, g: int, b: int) -> None:
    """Write a 2x2 solid-color PNG with no external deps."""
    width, height = 2, 2

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = bytes([0] + [r, g, b] * width)
    raw = row * height
    idat = zlib.compress(raw, 1)

    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr)
    png += _chunk(b"IDAT", idat)
    png += _chunk(b"IEND", b"")
    path.write_bytes(png)


class PlaceholderTestBackend(Backend):
    """Day 1 canary backend.  Zero deps, deterministic, audio-safe."""

    name = "placeholder_test"

    def run(self, job_dir: Path) -> None:
        out_dir = self.out_dir_for(job_dir)
        try:
            shots = self.load_shotlist(job_dir)
        except (FileNotFoundError, ValueError) as exc:
            write_status(out_dir, STATUS_ERROR, f"{type(exc).__name__}: {exc}")
            return

        if not shots:
            write_status(out_dir, STATUS_ERROR, "shotlist has zero shots")
            return

        write_status(
            out_dir, STATUS_RUNNING,
            f"placeholder_test backend: {len(shots)} shots",
        )

        for i, shot in enumerate(shots):
            shot_id = shot.get("shot_id", f"shot_{i:03d}")
            shot_dir = out_dir / shot_id
            shot_dir.mkdir(parents=True, exist_ok=True)
            r = 20 + (i * 13) % 60
            g = 40 + (i * 19) % 60
            b = 60 + (i * 29) % 60
            _tiny_png(shot_dir / "render.png", r, g, b)
            atomic_write_json(shot_dir / "meta.json", {
                "shot_id": shot_id,
                "backend": self.name,
                "env_prompt": shot.get("env_prompt", ""),
                "camera": shot.get("camera", ""),
                "duration_sec": float(shot.get("duration_sec", 9)),
            })

        write_status(
            out_dir, STATUS_READY,
            f"placeholder_test READY: {len(shots)} shots rendered",
            backend=self.name,
        )
