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

# Phase A: atomic writes + VRAM coordinator scaffold.  These imports use
# absolute-from-file resolution so the module works whether the worker
# is launched as ``python otr_v2/hyworld/worker.py <job_dir>`` (no
# package context) or imported as ``otr_v2.hyworld.worker``.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
try:
    from _atomic import atomic_write_json  # type: ignore
except ImportError:
    # Last-resort fallback: degrade to non-atomic write.  This keeps the
    # worker runnable even from a stripped-down env, at the cost of the
    # STATUS.json race.  Logged so we notice if it ever happens in prod.
    def atomic_write_json(path: Path, data, indent: int = 2) -> None:  # type: ignore
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(data, indent=indent, ensure_ascii=False),
            encoding="utf-8",
        )

try:
    from vram_coordinator import VRAMCoordinator  # type: ignore
except ImportError:
    # If the coordinator module isn't reachable for some reason (e.g.
    # the worker is launched from a stripped-down env without the rest
    # of otr_v2.hyworld), fall back to a no-op stand-in.  Today the
    # worker is CPU-only, so this is purely scaffolding for Phase B.
    class VRAMCoordinator:  # type: ignore
        def __init__(self, *a, **kw): pass
        def acquire(self, *a, **kw):
            from contextlib import contextmanager
            @contextmanager
            def _noop():
                yield self
            return _noop()
        def release(self, *a, **kw): return False
        def status(self): return None
        def is_held(self): return False


# Phase B v0: SD 1.5 anchor frame generation, gated behind an env var so
# the default behavior (solid-color placeholder PNG + Ken Burns) is
# unchanged until Jeffrey explicitly opts in.
#
#   set OTR_HYWORLD_ANCHOR=sd15   (PowerShell: $env:OTR_HYWORLD_ANCHOR="sd15")
#
# When unset / set to anything else, ``_anchor_gen_module`` is None and the
# worker falls through to the existing solid-color placeholder path.
_ANCHOR_BACKEND = os.environ.get("OTR_HYWORLD_ANCHOR", "").strip().lower()
_USE_ANCHOR_GEN = _ANCHOR_BACKEND in {"sd15", "sd1.5", "anchor"}
_anchor_gen_module = None  # type: ignore[assignment]
if _USE_ANCHOR_GEN:
    try:
        import anchor_gen as _anchor_gen_module  # type: ignore
    except ImportError:
        _anchor_gen_module = None
        _USE_ANCHOR_GEN = False


def _write_status(out_dir: Path, status: str, detail: str = "") -> None:
    """Write STATUS.json for the poll node.  Atomic to avoid mid-write races."""
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "STATUS.json", {
        "status": status,
        "detail": detail,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })


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

    # Phase B v0: if SD 1.5 anchor backend is enabled, render real anchor
    # PNGs into each shot dir BEFORE the per-shot loop runs.  The per-shot
    # loop will then detect a non-placeholder PNG already on disk and skip
    # the solid-color fallback, but otherwise the loop is unchanged --
    # ffmpeg still consumes ``render.png`` from the shot dir as input.
    anchor_results: dict = {}
    anchor_ok = 0
    anchor_failed = 0
    if _USE_ANCHOR_GEN and _anchor_gen_module is not None:
        _write_status(out_dir, "RUNNING", "Generating SD 1.5 anchors")
        try:
            anchor_results = _anchor_gen_module.generate_for_shotlist(
                shots, out_dir,
            )
            for ar in anchor_results.values():
                if ar.error:
                    anchor_failed += 1
                else:
                    anchor_ok += 1
        except Exception as exc:  # noqa: BLE001 -- log, fall back per-shot
            _write_status(
                out_dir, "RUNNING",
                f"Anchor backend crashed ({type(exc).__name__}); falling back to placeholders",
            )
            anchor_results = {}

    if anchor_ok > 0:
        backend_label = "sd15_anchor_motion_clip" if ffmpeg else "sd15_anchor_still"
    else:
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

        # If anchor_gen wrote a usable PNG into the shot dir, keep it.
        # Otherwise (anchor disabled, anchor failed, or anchor missing),
        # fall back to the solid-color placeholder.
        anchor_used = False
        ar = anchor_results.get(shot_id)
        if ar is not None and not ar.error and still_path.exists() and still_path.stat().st_size > 0:
            anchor_used = True
        else:
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

        # Write shot metadata (now includes resolved motion + backend
        # and the anchor cache key when SD 1.5 actually rendered the still).
        meta = {
            "shot_id": shot_id,
            "env_prompt": shot.get("env_prompt", ""),
            "camera": camera,
            "duration_sec": duration,
            "backend": backend_label,
            "motion": motion_label,
            "anchor_used": anchor_used,
        }
        if anchor_used and ar is not None:
            meta["anchor_cache_key"] = ar.cache_key
            meta["anchor_cache_hit"] = ar.cache_hit
            meta["anchor_seed"] = ar.seed
        if ffmpeg_detail:
            meta["ffmpeg_detail"] = ffmpeg_detail
        atomic_write_json(shot_dir / "meta.json", meta)

    # Compose the final READY detail string.  Surface anchor stats only
    # when the anchor backend was actually requested -- keeps the existing
    # placeholder-mode message identical to today.
    anchor_segment = ""
    if _USE_ANCHOR_GEN:
        anchor_segment = (
            f"; SD15 anchors: {anchor_ok} ok, {anchor_failed} failed"
        )
    if ffmpeg is None:
        detail = f"Still-only stub: {len(shots)} placeholder PNGs (ffmpeg not found){anchor_segment}"
    elif motion_failed == 0:
        detail = f"Motion stub: {motion_ok} Ken Burns MP4 clips generated{anchor_segment}"
    else:
        detail = (
            f"Motion stub: {motion_ok} MP4 clips, {motion_failed} fell back to still"
            f"{anchor_segment}"
        )
    _write_status(out_dir, "READY", detail)


def run_worldmirror(job_dir: Path) -> None:
    """
    Real WorldMirror 2.0 inference.  Activated when the hyworld2 env
    has the model installed.

    Wraps GPU work in a VRAMCoordinator.acquire() so it cannot collide
    with Bark TTS or another worker pass.  The coordinator is a Phase
    A scaffold; until Bark also acquires the same lock, this protects
    only against concurrent worker runs -- still useful and zero
    behavioural cost when there's no contention.

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

    # Step 2-5 happen inside the GPU gate.  Bark TTS does not yet
    # acquire this lock, so today the gate only blocks worker-vs-worker
    # collisions.  When Phase B (SDXL anchor) lands, Bark will be
    # taught to acquire this same lock during the audio render window.
    coord = VRAMCoordinator()
    try:
        with coord.acquire(owner="hyworld_worker", job_id=job_id, timeout=1800):
            # Step 2: Load model (first run downloads weights)
            # Step 3: For each shot, run inference on panorama images
            # Step 4: Write gaussians.ply, depth maps, rendered frames to out_dir
            # Step 5: Write READY status
            _write_status(out_dir, "ERROR", "WorldMirror integration not yet implemented")
    except TimeoutError as e:
        _write_status(out_dir, "ERROR", f"VRAM gate timeout: {e}")


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
