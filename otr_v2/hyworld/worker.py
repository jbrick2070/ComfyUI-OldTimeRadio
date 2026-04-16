"""
worker.py  --  HyWorld sidecar worker (runs in hyworld2 conda env)
===================================================================
This script is spawned by OTR_HyworldBridge as a subprocess.
It reads the contract files from io/hyworld_in/<job_id>/,
runs HyWorld inference, and writes results to io/hyworld_out/<job_id>/.

IMPORTANT: This runs in a SEPARATE Python environment (torch 2.4, CUDA 12.4)
from the main ComfyUI process (torch 2.10, CUDA 13.0).  Do NOT import
any OTR node code or ComfyUI modules.

Usage (called by bridge.py, not by humans):
    python otr_v2/hyworld/worker.py <path_to_job_dir>

Status protocol:
    The worker writes STATUS.json to io/hyworld_out/<job_id>/ with:
        {"status": "RUNNING"|"READY"|"ERROR"|"OOM", "detail": "...", ...}
    The poll node reads this file to determine completion.

Stub implementation:  Until WorldMirror 2.0 is installed in the hyworld2
conda env, this worker reads the shotlist and creates placeholder assets
(solid-color PNG stills per shot) so the full pipeline can be tested
end-to-end without GPU inference.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import time
import traceback
from pathlib import Path


def _write_status(out_dir: Path, status: str, detail: str = "") -> None:
    """Write STATUS.json for the poll node."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "STATUS.json").write_text(json.dumps({
        "status": status,
        "detail": detail,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2), encoding="utf-8")


def _create_placeholder_png(path: Path, width: int = 1280, height: int = 720,
                             r: int = 30, g: int = 30, b: int = 40) -> None:
    """
    Write a minimal valid PNG (solid color, no external deps).
    Uses raw DEFLATE via zlib.  No Pillow required.
    """
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    # IDAT — uncompressed scanlines (filter byte 0 + RGB per pixel)
    row = bytes([0] + [r, g, b] * width)
    raw = row * height
    compressed = zlib.compress(raw, 1)

    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr_data)
    png += _chunk(b"IDAT", compressed)
    png += _chunk(b"IEND", b"")

    path.write_bytes(png)


def run_stub(job_dir: Path) -> None:
    """
    Stub worker: read shotlist, create placeholder stills per shot.
    Replace this function with real WorldMirror 2.0 inference when
    the hyworld2 env is ready.
    """
    # Derive paths
    job_id = job_dir.name
    otr_root = job_dir.parent.parent.parent  # io/hyworld_in/<job_id> -> repo root
    out_dir = otr_root / "io" / "hyworld_out" / job_id

    _write_status(out_dir, "RUNNING", "Stub worker generating placeholders")

    # Read shotlist
    shotlist_path = job_dir / "shotlist.json"
    if not shotlist_path.exists():
        _write_status(out_dir, "ERROR", "shotlist.json not found in job dir")
        return

    shotlist = json.loads(shotlist_path.read_text(encoding="utf-8"))
    shots = shotlist.get("shots", [])

    if not shots:
        _write_status(out_dir, "ERROR", "shotlist has zero shots")
        return

    # Create per-shot placeholder stills
    for i, shot in enumerate(shots):
        shot_id = shot.get("shot_id", f"shot_{i:03d}")
        shot_dir = out_dir / shot_id
        shot_dir.mkdir(parents=True, exist_ok=True)

        # Vary color slightly per shot for visual distinction
        r = 30 + (i * 17) % 60
        g = 30 + (i * 23) % 60
        b = 40 + (i * 31) % 60

        _create_placeholder_png(shot_dir / "render.png", r=r, g=g, b=b)

        # Write shot metadata
        (shot_dir / "meta.json").write_text(json.dumps({
            "shot_id": shot_id,
            "env_prompt": shot.get("env_prompt", ""),
            "camera": shot.get("camera", ""),
            "duration_sec": shot.get("duration_sec", 9),
            "backend": "stub_placeholder",
        }, indent=2), encoding="utf-8")

    _write_status(out_dir, "READY", f"Stub: {len(shots)} placeholder stills generated")


def run_worldmirror(job_dir: Path) -> None:
    """
    Real WorldMirror 2.0 inference.  Activated when the hyworld2 env
    has the model installed.

    TODO: Implement when conda env + weights are verified.
    Skeleton left here so the entry point is clear.
    """
    job_id = job_dir.name
    otr_root = job_dir.parent.parent.parent
    out_dir = otr_root / "io" / "hyworld_out" / job_id

    _write_status(out_dir, "RUNNING", "WorldMirror 2.0 inference starting")

    try:
        # Step 1: Check if WorldMirror is importable
        from hyworld2.worldrecon.pipeline import WorldMirrorPipeline  # type: ignore
    except ImportError:
        _write_status(out_dir, "ERROR", "WorldMirrorPipeline not installed in this env")
        return

    # Step 2: Load model (first run downloads weights)
    # Step 3: For each shot, run inference on panorama images
    # Step 4: Write gaussians.ply, depth maps, rendered frames to out_dir
    # Step 5: Write READY status

    _write_status(out_dir, "ERROR", "WorldMirror integration not yet implemented")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python worker.py <job_dir>", file=sys.stderr)
        sys.exit(1)

    job_dir = Path(sys.argv[1])
    if not job_dir.is_dir():
        print(f"Job directory not found: {job_dir}", file=sys.stderr)
        sys.exit(1)

    job_id = job_dir.name
    otr_root = job_dir.parent.parent.parent
    out_dir = otr_root / "io" / "hyworld_out" / job_id

    try:
        # Try real inference first; fall back to stub
        try:
            from hyworld2.worldrecon.pipeline import WorldMirrorPipeline  # type: ignore  # noqa: F401
            run_worldmirror(job_dir)
        except ImportError:
            run_stub(job_dir)
    except Exception:
        _write_status(out_dir, "ERROR", traceback.format_exc()[-500:])
        sys.exit(1)


if __name__ == "__main__":
    main()
