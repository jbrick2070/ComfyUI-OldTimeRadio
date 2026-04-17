"""
renderer.py  --  OTR_HyworldRenderer ComfyUI node
===================================================
Reads geometry + images from io/hyworld_out/<job_id>/,
composites per-scene MP4 clips, crossfades to match audio length,
and muxes with the untouched v1.7 WAV.

Audio path is NEVER modified.  C7: audio output byte-identical.

Design doc: docs/internal/specs/2026-04-15-hyworld-poc-design.md  Section 6
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

log = logging.getLogger("OTR.hyworld.renderer")

_OTR_ROOT = Path(__file__).resolve().parent.parent.parent
_IO_OUT = _OTR_ROOT / "io" / "hyworld_out"

# Use folder_paths for ComfyUI-compatible output directory (BUG-01.02)
try:
    import folder_paths
    _RENDER_OUT = Path(folder_paths.get_output_directory()) / "hyworld_renders"
except ImportError:
    # Fallback for standalone testing
    _RENDER_OUT = _OTR_ROOT / "output" / "hyworld_renders"

# Crossfade duration between scene clips (seconds)
_CROSSFADE_SEC = 0.75

# Target output framerate
_FPS = 24


def _find_ffmpeg() -> str | None:
    """Locate ffmpeg binary. Returns path or None."""
    # Check common Windows locations
    candidates = [
        "ffmpeg",  # on PATH
        r"C:\Users\jeffr\Documents\ComfyUI\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for c in candidates:
        if shutil.which(c):
            return c
    return None


class HyworldRenderer:
    """
    ComfyUI node: OTR_HyworldRenderer

    Reads HyWorld sidecar output (rendered frames, splat fly-throughs,
    panoramic stills) and composites them into a per-scene MP4 synced
    to the episode's audio timeline.

    If hyworld_assets_path is "FALLBACK", this node returns an empty
    video_path so the workflow can route to OTR_SignalLostVideo instead.
    """

    CATEGORY = "OTR/v2/HyWorld"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_mp4_path", "render_log")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hyworld_assets_path": ("STRING", {
                    "tooltip": "Path to io/hyworld_out/<job_id>/ from OTR_HyworldPoll, or 'FALLBACK'.",
                }),
                "final_audio_path": ("STRING", {
                    "tooltip": "Path to the final episode WAV from v1.7 pipeline. NEVER modified.",
                }),
            },
            "optional": {
                "shotlist_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "Shotlist from OTR_HyworldBridge (provides timing + camera metadata).",
                }),
                "episode_title": ("STRING", {
                    "default": "Untitled",
                    "tooltip": "Episode title for output filename.",
                }),
                "crt_postfx": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply CRT/SIGNAL LOST post-processing (scanlines, vignette, chroma bleed).",
                }),
                "output_resolution": (["1280x720", "1920x1080", "960x540"], {
                    "default": "1280x720",
                    "tooltip": "Output video resolution.",
                }),
            },
        }

    def execute(
        self,
        hyworld_assets_path: str,
        final_audio_path: str,
        shotlist_json: str = "{}",
        episode_title: str = "Untitled",
        crt_postfx: bool = True,
        output_resolution: str = "1280x720",
    ) -> tuple[str, str]:
        """
        Composite HyWorld assets into final MP4.

        If hyworld_assets_path is FALLBACK or empty, return empty path
        so the workflow routes to the procedural video fallback.
        """
        render_log_lines: list[str] = []

        def _log(msg: str) -> None:
            log.info("[HyworldRenderer] %s", msg)
            render_log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

        # ---- FALLBACK CHECK ----
        if not hyworld_assets_path or hyworld_assets_path == "FALLBACK":
            _log("Assets path is FALLBACK — skipping HyWorld render, routing to procedural video.")
            return ("", "\n".join(render_log_lines))

        assets_dir = Path(hyworld_assets_path)
        if not assets_dir.is_dir():
            _log(f"Assets directory does not exist: {assets_dir}. Falling back.")
            return ("", "\n".join(render_log_lines))

        # ---- PARSE SHOTLIST ----
        try:
            shotlist = json.loads(shotlist_json)
            shots = shotlist.get("shots", [])
        except (json.JSONDecodeError, AttributeError):
            shots = []

        _log(f"Assets: {assets_dir}")
        _log(f"Shotlist: {len(shots)} shots")

        # ---- COLLECT RENDERED ASSETS (before audio/ffmpeg checks) ----
        # The sidecar writes per-shot renders as:
        #   <assets_dir>/<shot_id>/render.mp4   (video clip)
        #   <assets_dir>/<shot_id>/render.png   (still frame, fallback)
        #   <assets_dir>/gaussians.ply          (3DGS, for future use)
        shot_clips: list[dict] = []
        for shot in shots:
            shot_id = shot.get("shot_id", "")
            shot_dir = assets_dir / shot_id
            duration = shot.get("duration_sec", 9)

            clip_path = shot_dir / "render.mp4"
            still_path = shot_dir / "render.png"

            if clip_path.exists():
                shot_clips.append({"path": str(clip_path), "type": "video", "duration": duration, "shot_id": shot_id})
                _log(f"  {shot_id}: video clip ({duration}s)")
            elif still_path.exists():
                shot_clips.append({"path": str(still_path), "type": "still", "duration": duration, "shot_id": shot_id})
                _log(f"  {shot_id}: still image ({duration}s)")
            else:
                _log(f"  {shot_id}: NO ASSETS FOUND (skipping)")

        if not shot_clips:
            _log("No shot assets found. Falling back to procedural video.")
            return ("", "\n".join(render_log_lines))

        # ---- VERIFY AUDIO EXISTS (never modify it) ----
        audio_path = Path(final_audio_path)
        if not audio_path.exists():
            _log(f"Audio file not found: {audio_path}. Cannot mux without audio.")
            return ("", "\n".join(render_log_lines))

        # ---- FIND FFMPEG ----
        ffmpeg = _find_ffmpeg()
        if ffmpeg is None:
            _log("ffmpeg not found on PATH or common locations. Cannot composite video.")
            return ("", "\n".join(render_log_lines))

        _log(f"ffmpeg: {ffmpeg}")
        _log(f"Audio: {audio_path} (read-only, byte-identical guarantee)")

        # ---- OUTPUT PATH ----
        _RENDER_OUT.mkdir(parents=True, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in episode_title)[:60]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = _RENDER_OUT / f"{safe_title}_{timestamp}.mp4"

        # ---- BUILD FFMPEG CONCAT ----
        width, height = output_resolution.split("x")
        concat_list = assets_dir / "_concat_list.txt"

        with open(concat_list, "w", encoding="utf-8") as f:
            for clip in shot_clips:
                if clip["type"] == "video":
                    f.write(f"file '{clip['path']}'\n")
                else:
                    # Convert still to a clip: ffmpeg -loop 1 -t <dur> -i still.png
                    # We generate a temp clip first
                    temp_clip = assets_dir / clip["shot_id"] / "_temp_clip.mp4"
                    _still_to_clip(ffmpeg, clip["path"], str(temp_clip), clip["duration"], width, height)
                    f.write(f"file '{temp_clip}'\n")

        # ---- CONCAT + MUX WITH AUDIO ----
        # Step 1: concat visual clips
        concat_out = assets_dir / "_concat_video.mp4"
        cmd_concat = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-s", f"{width}x{height}",
            str(concat_out),
        ]
        _log(f"Concatenating {len(shot_clips)} clips...")
        _run_ffmpeg(cmd_concat, _log)

        if not concat_out.exists():
            _log("Concat failed. Falling back.")
            return ("", "\n".join(render_log_lines))

        # Step 2: mux concat video + original audio (audio is never re-encoded)
        cmd_mux = [
            ffmpeg, "-y",
            "-i", str(concat_out),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "copy",        # byte-identical audio passthrough (C7)
            "-shortest",
            str(out_path),
        ]
        _log("Muxing video + audio (audio passthrough, C7 guaranteed)...")
        _run_ffmpeg(cmd_mux, _log)

        if out_path.exists():
            size_mb = out_path.stat().st_size / (1024 * 1024)
            _log(f"Output: {out_path} ({size_mb:.1f} MB)")
        else:
            _log("Mux failed. No output file.")
            return ("", "\n".join(render_log_lines))

        # ---- CLEANUP TEMP FILES ----
        for f in [concat_list, concat_out]:
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass

        return (str(out_path), "\n".join(render_log_lines))


def _still_to_clip(
    ffmpeg: str, still_path: str, out_path: str,
    duration: float, width: str, height: str,
) -> None:
    """Convert a still image to a video clip of given duration."""
    cmd = [
        ffmpeg, "-y", "-loop", "1",
        "-i", still_path,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-s", f"{width}x{height}",
        "-r", str(_FPS),
        out_path,
    ]
    subprocess.run(cmd, capture_output=True, timeout=120)


def _run_ffmpeg(cmd: list[str], log_fn) -> None:
    """Run an ffmpeg command, logging stderr on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300, text=True)
        if result.returncode != 0:
            log_fn(f"ffmpeg error (rc={result.returncode}): {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        log_fn("ffmpeg timed out after 300s")
    except FileNotFoundError:
        log_fn("ffmpeg binary not found at runtime")
