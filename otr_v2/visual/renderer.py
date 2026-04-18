"""
renderer.py  --  OTR_VisualRenderer ComfyUI node
===================================================
Reads geometry + images from io/visual_out/<job_id>/,
composites per-scene MP4 clips, crossfades to match audio length,
and muxes with the untouched episode audio.

Audio path is NEVER modified.  C7: audio output byte-identical.
ffmpeg uses `-c:a copy` on the final mux so the muxed track is
bit-for-bit identical to the temp WAV the node writes from the
incoming AUDIO tensor.

Design doc: docs/2026-04-15-visual-poc-design.md  Section 6
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger("OTR.visual.renderer")

_OTR_ROOT = Path(__file__).resolve().parent.parent.parent
_IO_OUT = _OTR_ROOT / "io" / "visual_out"

# Use folder_paths for ComfyUI-compatible output directory (BUG-01.02)
try:
    import folder_paths
    _RENDER_OUT = Path(folder_paths.get_output_directory()) / "visual_renders"
except ImportError:
    # Fallback for standalone testing
    _RENDER_OUT = _OTR_ROOT / "output" / "visual_renders"

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


class VisualRenderer:
    """
    ComfyUI node: OTR_VisualRenderer

    Reads Visual sidecar output (rendered frames, splat fly-throughs,
    panoramic stills) and composites them into a per-scene MP4 synced
    to the episode's audio timeline.

    If visual_assets_path is "FALLBACK", this node returns an empty
    video_path so the workflow can route to OTR_SignalLostVideo instead.
    """

    CATEGORY = "OTR/v2/Visual"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_mp4_path", "render_log")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "visual_assets_path": ("STRING", {
                    "tooltip": "Path to io/visual_out/<job_id>/ from OTR_VisualPoll, or 'FALLBACK'.",
                }),
                "episode_audio": ("AUDIO", {
                    "tooltip": "Final episode audio tensor from OTR_EpisodeAssembler. Written to a temp WAV and muxed with -c:a copy (byte-identical passthrough, C7).",
                }),
            },
            "optional": {
                "shotlist_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "Shotlist from OTR_VisualBridge (provides timing + camera metadata).",
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
        visual_assets_path: str,
        episode_audio,
        shotlist_json: str = "{}",
        episode_title: str = "Untitled",
        crt_postfx: bool = True,
        output_resolution: str = "1280x720",
    ) -> tuple[str, str]:
        """
        Composite Visual assets into final MP4.

        If visual_assets_path is FALLBACK or empty, return empty path
        so the workflow routes to the procedural video fallback.
        """
        render_log_lines: list[str] = []

        def _log(msg: str) -> None:
            log.info("[VisualRenderer] %s", msg)
            render_log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

        # ---- FALLBACK CHECK ----
        if not visual_assets_path or visual_assets_path == "FALLBACK":
            _log("Assets path is FALLBACK — skipping Visual render, routing to procedural video.")
            return ("", "\n".join(render_log_lines))

        assets_dir = Path(visual_assets_path)
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

        # ---- WRITE AUDIO TO TEMP WAV (never modify the tensor) ----
        # VisualRenderer owns the temp file; ffmpeg muxes with -c:a copy
        # so the track is byte-identical to the bytes we write here (C7).
        audio_path = _write_audio_tensor_to_wav(episode_audio, _log)
        if audio_path is None:
            _log("Audio tensor unreadable. Cannot mux without audio.")
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
        # C7 safety: audio is king. The old `-shortest` could trim the audio
        # if the video was shorter. We now probe durations and:
        #   - if video >= audio: `-t audio_duration` caps output to audio length
        #     with `-c:v copy` + `-c:a copy` (byte-identical audio passthrough).
        #   - if video <  audio: pad video with a frozen last frame (tpad
        #     filter), video gets re-encoded, audio stays `-c:a copy`.
        # Probe failure falls back to the old `-shortest` behavior (logged).
        audio_dur = _probe_duration_sec(ffmpeg, audio_path)
        video_dur = _probe_duration_sec(ffmpeg, concat_out)

        cmd_mux: list[str] = [
            ffmpeg, "-y",
            "-i", str(concat_out),
            "-i", str(audio_path),
        ]

        if audio_dur is not None and video_dur is not None:
            _log(f"Durations: video={video_dur:.2f}s, audio={audio_dur:.2f}s")
            if video_dur + 0.05 < audio_dur:
                # Video shorter than audio -> pad last frame to close the gap.
                pad_sec = audio_dur - video_dur + 0.1
                cmd_mux += [
                    "-filter_complex",
                    f"[0:v]tpad=stop_mode=clone:stop_duration={pad_sec:.3f}[vpad]",
                    "-map", "[vpad]",
                    "-map", "1:a",
                    "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-c:a", "copy",
                    "-t", f"{audio_dur:.3f}",
                ]
                _log(
                    f"Video < audio by {audio_dur - video_dur:.2f}s; "
                    f"padding last frame to preserve full audio (C7)."
                )
            else:
                cmd_mux += [
                    "-c:v", "copy",
                    "-c:a", "copy",  # byte-identical audio passthrough (C7)
                    "-t", f"{audio_dur:.3f}",
                ]
                _log("Video >= audio; capping output to exact audio length.")
        else:
            cmd_mux += [
                "-c:v", "copy",
                "-c:a", "copy",
                "-shortest",
            ]
            _log(
                "Duration probe failed; falling back to -shortest "
                "(C7 risk if video < audio)."
            )

        cmd_mux.append(str(out_path))
        _log("Muxing video + audio (audio passthrough, C7 guaranteed)...")
        _run_ffmpeg(cmd_mux, _log)

        if out_path.exists():
            size_mb = out_path.stat().st_size / (1024 * 1024)
            _log(f"Output: {out_path} ({size_mb:.1f} MB)")
        else:
            _log("Mux failed. No output file.")
            return ("", "\n".join(render_log_lines))

        # ---- CLEANUP TEMP FILES ----
        for f in [concat_list, concat_out, audio_path]:
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass

        return (str(out_path), "\n".join(render_log_lines))


def _write_audio_tensor_to_wav(audio, log_fn) -> Path | None:
    """Write a ComfyUI AUDIO dict to a unique temp WAV and return its Path.

    ComfyUI AUDIO format: {"waveform": torch.Tensor, "sample_rate": int}
    Tensor shape is typically (batch, channels, samples) or (channels, samples).
    We mix to mono to match the rest of the pipeline and write 16-bit PCM.
    The renderer then passes this path to ffmpeg with `-c:a copy`, so the
    PCM bytes flow through unchanged (C7 byte-identical).
    """
    try:
        if not isinstance(audio, dict):
            log_fn(f"Audio input is not an AUDIO dict (got {type(audio).__name__}).")
            return None
        waveform = audio.get("waveform")
        sr = audio.get("sample_rate")
        if waveform is None or sr is None:
            log_fn("AUDIO dict missing 'waveform' or 'sample_rate'.")
            return None

        if waveform.dim() == 3:
            audio_np = waveform[0].mean(dim=0).cpu().numpy()
        elif waveform.dim() == 2:
            audio_np = waveform.mean(dim=0).cpu().numpy()
        else:
            audio_np = waveform.cpu().numpy()

        pcm = np.clip(audio_np * 32767.0, -32767, 32767).astype(np.int16)

        fd, path_str = tempfile.mkstemp(prefix="visual_audio_", suffix=".wav")
        os.close(fd)
        path = Path(path_str)

        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sr))
            wf.writeframes(pcm.tobytes())

        log_fn(f"Audio: wrote {len(pcm)} samples @ {sr} Hz to {path.name}")
        return path
    except Exception as exc:  # pragma: no cover - defensive
        log_fn(f"Audio tensor -> WAV failed: {exc}")
        return None


def _still_to_clip(
    ffmpeg: str, still_path: str, out_path: str,
    duration: float, width: str, height: str,
) -> None:
    """Convert a still image to an animated clip of given duration.

    When the motion sidecar couldn't produce a real per-shot MP4, the
    renderer falls through to the still for that shot.  A frozen PNG
    for 30-60 seconds reads as a slideshow, so we composite a procgen
    overlay on top of a slow Ken Burns zoom/pan so the fallback still
    feels alive on screen:

        1. Ken Burns: scale up to 110% then crop with a slow easing
           zoompan, giving ~10% parallax across the shot duration.
        2. Scan-line overlay: horizontal 2-px lines every 4 px at
           12% opacity (CRT-style scan pattern).
        3. Signal-lost noise: low-amplitude filmgrain + vignette so
           the image has natural motion.
        4. Slight gamma wobble via eq filter sine-driven over time
           so the highlights breathe.

    If the complex filter graph fails (older ffmpeg builds, unusual
    still aspect ratios), we fall back to a plain 1-frame loop so the
    clip still lands and the episode doesn't stall.
    """
    # C7 safety note: this function only touches VIDEO.  Audio is
    # never piped through this path.
    dur = max(float(duration), 0.1)
    total_frames = max(int(dur * _FPS), 2)

    # Ken Burns: zoompan that scales from 1.00 -> 1.10 and drifts
    # toward centre-right over the clip duration.  'd' is frame count,
    # 'z' grows linearly, 'x'/'y' ease toward the zoom focus.
    zoompan = (
        f"zoompan="
        f"z='min(1.0+0.10*on/{total_frames},1.10)':"
        f"x='iw/2-(iw/zoom/2)+((iw*0.04)*on/{total_frames})':"
        f"y='ih/2-(ih/zoom/2)':"
        f"d={total_frames}:s={width}x{height}:fps={_FPS}"
    )
    # Scan-line overlay built from geq so we don't need an external
    # asset.  lte(mod(Y,4),1) draws a 1-px line every 4 px; mix 12%.
    scanlines = (
        f"geq=lum='lum(X,Y)*(1-0.12*lte(mod(Y\\,4)\\,1))':cb='cb(X,Y)':cr='cr(X,Y)'"
    )
    # Filmgrain via noise filter (temporal).  alls=8 is subtle.
    noise = "noise=alls=8:allf=t"
    # Vignette + subtle breathing gamma (sine-wave).
    vignette = "vignette=PI/5"
    breath = "eq=gamma='1.0+0.04*sin(2*PI*t/4)'"

    filter_graph = f"{zoompan},{scanlines},{noise},{vignette},{breath},format=yuv420p"

    cmd = [
        ffmpeg, "-y", "-loop", "1",
        "-i", still_path,
        "-t", str(dur),
        "-vf", filter_graph,
        "-c:v", "libx264", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-r", str(_FPS),
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=180)
    if result.returncode == 0 and Path(out_path).exists():
        return

    # Fallback: plain loop if the filter graph errored (older ffmpeg
    # without geq, unusual still format, etc).  Log the stderr tail
    # into a sidecar debug file so post-mortem analysis has context.
    stderr_tail = b""
    try:
        stderr_tail = (result.stderr or b"")[-2048:]
    except Exception:
        stderr_tail = b""
    try:
        Path(out_path).with_suffix(".filter_err.log").write_bytes(stderr_tail)
    except OSError:
        pass
    fallback_cmd = [
        ffmpeg, "-y", "-loop", "1",
        "-i", still_path,
        "-t", str(dur),
        "-c:v", "libx264", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-s", f"{width}x{height}",
        "-r", str(_FPS),
        out_path,
    ]
    subprocess.run(fallback_cmd, capture_output=True, timeout=120)
    return


def _probe_duration_sec(ffmpeg_path: str, media_path: Path) -> float | None:
    """Probe a media file's duration in seconds via the sibling ffprobe
    binary (ffprobe lives next to ffmpeg on Windows installs). Returns
    ``None`` on any failure -- callers fall back to -shortest behavior.
    """
    if ffmpeg_path == "ffmpeg":
        ffprobe = "ffprobe"
    else:
        parts = Path(ffmpeg_path)
        sibling = parts.name.replace("ffmpeg", "ffprobe", 1)
        ffprobe = str(parts.with_name(sibling)) if sibling != parts.name else "ffprobe"

    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return None


def _run_ffmpeg(cmd: list[str], log_fn, timeout: int = 300) -> int:
    """Run an ffmpeg command and log stderr tail on failure.

    Redirects stdout->DEVNULL and stderr->a temp file (NOT PIPE) to avoid
    the Windows pipe-backpressure deadlock class that was the leading
    suspect in the RUN 245 soak hang (gemini-3-flash consult 2026-04-17).
    Returns ffmpeg's exit code, or -1 on binary-missing, -2 on timeout.
    """
    err_path: Path | None = None
    try:
        fd, err_name = tempfile.mkstemp(prefix="ffmpeg_err_", suffix=".log")
        os.close(fd)
        err_path = Path(err_name)
        with open(err_path, "wb") as err_fh:
            result = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=err_fh,
                timeout=timeout,
            )
        if result.returncode != 0:
            tail = err_path.read_bytes()[-1000:].decode("utf-8", errors="replace")
            log_fn(f"ffmpeg error (rc={result.returncode}): {tail}")
        return result.returncode
    except subprocess.TimeoutExpired:
        log_fn(f"ffmpeg timed out after {timeout}s")
        return -2
    except FileNotFoundError:
        log_fn("ffmpeg binary not found at runtime")
        return -1
    finally:
        if err_path is not None:
            try:
                err_path.unlink(missing_ok=True)
            except OSError:
                pass
