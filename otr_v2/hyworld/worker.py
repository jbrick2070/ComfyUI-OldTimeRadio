"""
worker.py  --  HyWorld sidecar worker (runs in hyworld2 conda env, or main venv fallback)
=========================================================================================
This script is spawned by OTR_HyworldBridge as a subprocess.
It reads the contract files from io/hyworld_in/<job_id>/,
runs HyWorld inference, and writes results to io/hyworld_out/<job_id>/.

IMPORTANT: This may run in a SEPARATE Python environment (torch 2.4, CUDA 12.4)
from the main ComfyUI process (torch 2.10, CUDA 13.0).  Do NOT import
any OTR node code or ComfyUI modules.  The worker also runs under the
main ComfyUI .venv as a fallback (stub modes only) when the hyworld2
conda env is unavailable.

Usage (called by bridge.py, not by humans):
    python otr_v2/hyworld/worker.py <path_to_job_dir>

Status protocol:
    The worker writes STATUS.json to io/hyworld_out/<job_id>/ with:
        {"status": "RUNNING"|"READY"|"ERROR"|"OOM", "detail": "...", ...}
    The poll node reads this file to determine completion.

Tiered execution path (selected at runtime, no model contention with audio):
    1. WorldMirror 2.0 inference  - if hyworld2 env + weights present
    2. Motion stub (real MP4)     - if ffmpeg on PATH (Ken Burns clips
                                    driven by shotlist camera adjective).
                                    Uses CPU only, safe to run while Bark
                                    TTS holds the GPU.
    3. Still stub (solid PNG)     - last-resort, no external deps.

The motion stub is the current default for testing the full Bridge ->
Poll -> Renderer path with an MP4 output.  Real generative video
(SVD / LTX-Video) is the next phase and requires GPU coordination
with the audio pipeline (must wait until BatchBark releases VRAM).
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
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


# ---------------------------------------------------------------------------
# ffmpeg discovery + Ken Burns motion clip generation
# ---------------------------------------------------------------------------

# Output dimensions for stub clips.  Matches renderer default; the renderer
# will rescale at concat time anyway, but rendering at the target size keeps
# zoompan math honest.
_CLIP_WIDTH = 1280
_CLIP_HEIGHT = 720
_CLIP_FPS = 24

# Camera adjective (from shotlist.py voice-traits map) -> ffmpeg zoompan recipe.
# Each entry is a function (duration_sec, width, height) -> filter_chain str.
# Formulas use `on` (current input frame) and `d` (total animation frames)
# because -loop 1 feeds zoompan one new frame per output frame.

def _motion_static(d: int, w: int, h: int) -> str:
    return f"zoompan=z=1.0:d={d}:s={w}x{h}:fps={_CLIP_FPS}"

def _motion_slow_push_in(d: int, w: int, h: int) -> str:
    # 1.00 -> ~1.30 over the duration, centered crop.
    return (
        f"zoompan=z='min(1.0+0.30*on/{d},1.30)':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={d}:s={w}x{h}:fps={_CLIP_FPS}"
    )

def _motion_fast_dolly(d: int, w: int, h: int) -> str:
    # Faster zoom, slight off-center for canted feel.
    return (
        f"zoompan=z='min(1.0+0.55*on/{d},1.55)':"
        f"x='iw*0.45-(iw/zoom/2)':y='ih*0.55-(ih/zoom/2)':"
        f"d={d}:s={w}x{h}:fps={_CLIP_FPS}"
    )

def _motion_clean_push(d: int, w: int, h: int) -> str:
    return (
        f"zoompan=z='min(1.0+0.40*on/{d},1.40)':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={d}:s={w}x{h}:fps={_CLIP_FPS}"
    )

def _motion_whip_pan(d: int, w: int, h: int) -> str:
    # Constant zoom 1.3, sweep horizontally across the available range.
    return (
        f"zoompan=z=1.30:"
        f"x='(iw-iw/zoom)*on/{d}':y='ih/2-(ih/zoom/2)':"
        f"d={d}:s={w}x{h}:fps={_CLIP_FPS}"
    )

def _motion_low_angle(d: int, w: int, h: int) -> str:
    # Constant zoom 1.25, drift y from bottom to top (looking up).
    return (
        f"zoompan=z=1.25:"
        f"x='iw/2-(iw/zoom/2)':y='(ih-ih/zoom)*(1-on/{d})':"
        f"d={d}:s={w}x{h}:fps={_CLIP_FPS}"
    )

def _motion_macro(d: int, w: int, h: int) -> str:
    # Very slow zoom 1.0 -> 1.15, centered.
    return (
        f"zoompan=z='min(1.0+0.15*on/{d},1.15)':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={d}:s={w}x{h}:fps={_CLIP_FPS}"
    )

def _motion_slow_drift(d: int, w: int, h: int) -> str:
    # Default: gentle zoom 1.0 -> 1.20 with a small horizontal drift.
    return (
        f"zoompan=z='min(1.0+0.20*on/{d},1.20)':"
        f"x='(iw-iw/zoom)*(0.40+0.20*on/{d})':y='ih/2-(ih/zoom/2)':"
        f"d={d}:s={w}x{h}:fps={_CLIP_FPS}"
    )

# First-substring-match wins (case-insensitive).  Matches the camera
# adjectives produced by shotlist._camera_from_traits().
_CAMERA_MOTION_MAP: list[tuple[str, callable]] = [
    ("locked off",      _motion_static),
    ("clean push",      _motion_clean_push),
    ("slow handheld",   _motion_slow_push_in),
    ("fast dolly",      _motion_fast_dolly),
    ("whip-pan",        _motion_whip_pan),
    ("low angle",       _motion_low_angle),
    ("macro detail",    _motion_macro),
    ("slow drift",      _motion_slow_drift),
]


def _camera_to_motion(camera: str, duration_sec: float, w: int, h: int) -> tuple[str, str]:
    """Resolve a shotlist camera adjective to an ffmpeg filter chain.

    Returns (motion_label, filter_chain_string).
    """
    # Total animation frames at our fixed output FPS.
    d = max(1, int(round(duration_sec * _CLIP_FPS)))
    cam_lower = (camera or "").lower()
    for needle, fn in _CAMERA_MOTION_MAP:
        if needle in cam_lower:
            return (needle, fn(d, w, h))
    return ("default_drift", _motion_slow_drift(d, w, h))


def _find_ffmpeg() -> str | None:
    """Locate ffmpeg.  Mirrors renderer._find_ffmpeg search order."""
    candidates = [
        "ffmpeg",
        r"C:\Users\jeffr\Documents\ComfyUI\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for c in candidates:
        resolved = shutil.which(c)
        if resolved:
            return resolved
    return None


def _make_motion_clip(
    ffmpeg: str,
    still_path: Path,
    out_path: Path,
    duration_sec: float,
    camera: str,
) -> tuple[bool, str]:
    """Render a Ken Burns MP4 from a still using the camera-derived motion.

    Returns (ok, motion_label).  On failure, motion_label is the ffmpeg
    stderr tail so the caller can record it.
    """
    motion_label, vfilter = _camera_to_motion(camera, duration_sec, _CLIP_WIDTH, _CLIP_HEIGHT)
    # Compute exact target frame count.  zoompan multiplies frames (d output
    # frames per input frame), so we must:
    #   - feed exactly 1 input frame  (-framerate 1 -loop 1 -t 1)
    #   - cap output at N frames      (-frames:v N)
    # Otherwise the duration explodes by ~d^2.  See BUG-LOCAL-014.
    target_frames = max(1, int(round(duration_sec * _CLIP_FPS)))
    cmd = [
        ffmpeg, "-y",
        "-loop", "1",
        "-framerate", "1",
        "-t", "1",
        "-i", str(still_path),
        "-vf", vfilter,
        "-frames:v", str(target_frames),
        "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-r", str(_CLIP_FPS),
        "-an",  # no audio track in the clip; renderer muxes episode WAV separately
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return (False, f"ffmpeg-spawn-error:{exc.__class__.__name__}")
    if result.returncode != 0 or not out_path.exists():
        return (False, f"ffmpeg-rc{result.returncode}:{(result.stderr or '')[-200:]}")
    return (True, motion_label)


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

    # ffmpeg is the gate between still-stub and motion-stub modes.  When
    # present we upgrade each shot from a static PNG to a Ken Burns MP4
    # driven by the shotlist camera adjective.  CPU-only -> safe to run
    # alongside Bark TTS on the GPU.
    ffmpeg = _find_ffmpeg()

    motion_ok = 0
    motion_failed = 0
    backend_label = "stub_motion_clip" if ffmpeg else "stub_placeholder_still"

    # Create per-shot assets (still always written; mp4 written when ffmpeg is up).
    for i, shot in enumerate(shots):
        shot_id = shot.get("shot_id", f"shot_{i:03d}")
        shot_dir = out_dir / shot_id
        shot_dir.mkdir(parents=True, exist_ok=True)

        # Vary color slightly per shot for visual distinction
        r = 30 + (i * 17) % 60
        g = 30 + (i * 23) % 60
        b = 40 + (i * 31) % 60

        still_path = shot_dir / "render.png"
        _create_placeholder_png(still_path, r=r, g=g, b=b)

        camera = shot.get("camera", "")
        duration = float(shot.get("duration_sec", 9))
        motion_label = "still_only"
        ffmpeg_detail = ""

        if ffmpeg is not None:
            mp4_path = shot_dir / "render.mp4"
            ok, label = _make_motion_clip(ffmpeg, still_path, mp4_path, duration, camera)
            if ok:
                motion_ok += 1
                motion_label = label
            else:
                motion_failed += 1
                ffmpeg_detail = label  # holds the error tail

        # Write shot metadata (now includes resolved motion + backend)
        meta = {
            "shot_id": shot_id,
            "env_prompt": shot.get("env_prompt", ""),
            "camera": camera,
            "duration_sec": duration,
            "backend": backend_label,
            "motion": motion_label,
        }
        if ffmpeg_detail:
            meta["ffmpeg_detail"] = ffmpeg_detail
        (shot_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if ffmpeg is None:
        detail = f"Still-only stub: {len(shots)} placeholder PNGs (ffmpeg not found)"
    elif motion_failed == 0:
        detail = f"Motion stub: {motion_ok} Ken Burns MP4 clips generated"
    else:
        detail = (
            f"Motion stub: {motion_ok} MP4 clips, {motion_failed} fell back to still"
        )
    _write_status(out_dir, "READY", detail)


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
