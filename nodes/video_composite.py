"""
video_composite.py  --  OTR_VideoComposite ComfyUI node
=======================================================

In-graph ffmpeg-driven episode compositor. Replaces
``scripts/render_episode_concat.py`` for production runs; the CLI
script stays as an ad-hoc smoke tool.

Layered output (per Jeffrey's spec, locked 2026-04-27):

    1920x1080 canvas, 25 fps
    ┌──────────┬──────────────┬──────────┐
    │          │              │          │
    │ proc gen │   HuMo       │ proc gen │
    │ (left)   │   624x1080   │ (right)  │
    │          │              │          │
    └──────────┴──────────────┴──────────┘
     648 px      624 px         648 px

    + proc gen layer additive-blended on top at 50% opacity
      so CRT visualizer waveforms / spectrum bars glow over HuMo
      face during dialogue and fill the canvas during silence

The proc gen mp4 produced by OTR_SignalLostVideo carries audio-
reactive CRT visuals + the full episode audio. This node uses it
as both the base layer AND the audio source for the final mux. No
separate audio mux step is required.

Inputs:
    procgen_video_path  STRING  Path to the SignalLostVideo output
                                mp4 (full-episode audio-reactive
                                CRT base layer, 1920x1080 @ 24 fps,
                                AAC 48 kHz audio embedded).
    clips_dir           STRING  Directory containing per-line HuMo
                                clips named `<line_id>.mp4`. Output
                                of OTR_BatchHumoRender.
    ledger_json         STRING  Production Ledger -- either inline
                                JSON or a path to *_ledger.json.
                                Used to walk lines in beat order
                                and compute per-clip overlay
                                windows.
    blend_mode          CHOICE  Composite blend mode for proc gen
                                on top: addition / screen / lighten
                                / overlay / normal.
    blend_opacity       FLOAT   0.0 - 1.0, default 0.5.
    canvas_width        INT     Default 1920.
    canvas_height       INT     Default 1080.
    canvas_fps          INT     Default 25.
    humo_target_height  INT     HuMo lanczos-fit height. Default
                                1080. Width derives from HuMo's
                                native 480:832 aspect (= 624).

Outputs:
    final_mp4_path  STRING  Path to the rendered composite.
    report          STRING  Human-readable summary log.

Per-line timing source priority:
    1. ledger.lines[].dur_s (and start_s) when populated by
       SceneSequencer
    2. ffprobe each clip's actual duration (fallback: every clip we
       find on disk gets ffprobed for duration)
    3. clip_length (from BatchHumoRender) as last-resort uniform
       window
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("OTR.video_composite")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ffprobe_dur(path: Path, ffprobe: str = "ffprobe") -> float | None:
    """Return mp4 duration in seconds, or None on failure."""
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(path)],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())
    except Exception as exc:
        log.warning("[VideoComposite] ffprobe %s failed: %s", path, exc)
        return None


def _load_ledger(arg: str) -> dict:
    s = (arg or "").strip()
    if not s:
        raise RuntimeError("VideoComposite: ledger_json is empty")
    if s.startswith("{"):
        return json.loads(s)
    p = Path(s)
    if not p.exists():
        raise RuntimeError(f"VideoComposite: ledger path not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_clip_timeline(
    ledger: dict,
    clips_dir: Path,
    fallback_clip_length: float,
    ffprobe: str = "ffprobe",
) -> list[tuple[Path, float, float]]:
    """Build [(clip_path, start_s, dur_s), ...] for every ledger line
    that has a clip on disk. Cumulative start_s across the timeline
    -- ignores ledger.start_s except as a hint (we trust the actual
    rendered clip duration over ledger arithmetic)."""
    lines = ledger.get("lines") or []
    timeline: list[tuple[Path, float, float]] = []
    cumulative = 0.0
    for ln in lines:
        line_id = str(ln.get("line_id") or "")
        if not line_id:
            continue
        clip_path = clips_dir / f"{line_id}.mp4"
        if not clip_path.exists():
            log.info("[VideoComposite] line %s: clip not on disk, skipping", line_id)
            continue
        # Prefer ledger dur_s if positive, else ffprobe, else fallback
        dur = ln.get("dur_s")
        if dur is None or float(dur) <= 0:
            dur = _ffprobe_dur(clip_path, ffprobe)
        if dur is None or float(dur) <= 0:
            dur = float(fallback_clip_length)
        timeline.append((clip_path, cumulative, float(dur)))
        cumulative += float(dur)
    return timeline


def _build_filter_graph(
    procgen_video: Path,
    timeline: list[tuple[Path, float, float]],
    canvas_w: int,
    canvas_h: int,
    canvas_fps: int,
    humo_h: int,
    blend_mode: str,
    blend_opacity: float,
    episode_dur: float,
) -> tuple[str, list[Path]]:
    """Build the ffmpeg -filter_complex string + ordered input list.

    Inputs (0-indexed):
        [0]  procgen video (base layer + audio source)
        [1..N] HuMo clips, one per timeline entry

    Filter graph:
        [0:v] fps -> scale -> setsar -> [procgen]
        color=black canvas of episode_dur -> [base]
        [i:v] scale 624:humo_h -> setsar -> [hi]
        chain: [base] overlay each [hi] at x=offset_x,y=0 with
               enable='between(t,start,start+dur)'
        final: [base_with_humo] [procgen] blend=all_mode=BLEND:
               all_opacity=OPACITY -> [v]
    """
    humo_w = round(humo_h * (480 / 832) / 8) * 8  # snap to mult of 8 (h264 chroma safe)
    offset_x = (canvas_w - humo_w) // 2

    parts = []
    # [0] procgen → time-aligned + scaled
    parts.append(
        f"[0:v]fps={canvas_fps},scale={canvas_w}:{canvas_h},setsar=1[procgen]"
    )
    # Black base canvas
    parts.append(
        f"color=black:s={canvas_w}x{canvas_h}:r={canvas_fps}:d={episode_dur:.3f},setsar=1[base0]"
    )
    # Per-clip scale + overlay chain
    last_label = "base0"
    inputs_list: list[Path] = [procgen_video]
    for idx, (clip_path, start_s, dur_s) in enumerate(timeline):
        in_idx = idx + 1
        inputs_list.append(clip_path)
        parts.append(
            f"[{in_idx}:v]fps={canvas_fps},scale={humo_w}:{humo_h},setsar=1,"
            f"setpts=PTS-STARTPTS+{start_s:.3f}/TB[h{idx}]"
        )
        new_label = f"s{idx}"
        end_s = start_s + dur_s
        parts.append(
            f"[{last_label}][h{idx}]overlay=x={offset_x}:y=0:"
            f"enable='between(t,{start_s:.3f},{end_s:.3f})'[{new_label}]"
        )
        last_label = new_label
    # Final additive blend with proc gen on top
    parts.append(
        f"[{last_label}][procgen]blend=all_mode={blend_mode}:"
        f"all_opacity={blend_opacity:.3f}[v]"
    )
    return ";".join(parts), inputs_list


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class VideoComposite:
    """Compose proc gen base + N HuMo clips into a final 1920x1080 mp4."""

    CATEGORY = "OTR/v2/Visual"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_mp4_path", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "procgen_video_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": (
                        "Path to OTR_SignalLostVideo's mp4 output "
                        "(audio-reactive CRT base, full-episode audio)."
                    ),
                }),
                "clips_dir": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": (
                        "Directory of per-line HuMo clips "
                        "(<line_id>.mp4). Output of "
                        "OTR_BatchHumoRender."
                    ),
                }),
                "ledger_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Ledger JSON or path to *_ledger.json.",
                }),
            },
            "optional": {
                "blend_mode": (
                    ["addition", "screen", "lighten", "overlay", "normal"],
                    {"default": "addition"},
                ),
                "blend_opacity": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "canvas_width": ("INT", {"default": 1920, "min": 256, "max": 3840, "step": 8}),
                "canvas_height": ("INT", {"default": 1080, "min": 256, "max": 2160, "step": 8}),
                "canvas_fps": ("INT", {"default": 25, "min": 12, "max": 60}),
                "humo_target_height": ("INT", {
                    "default": 1080, "min": 480, "max": 2160, "step": 8,
                }),
                "fallback_clip_length": ("FLOAT", {
                    "default": 7.0, "min": 1.0, "max": 9.0, "step": 0.04,
                }),
                "ffmpeg": ("STRING", {
                    "default": "ffmpeg",
                    "multiline": False,
                    "tooltip": "ffmpeg binary path or name on PATH.",
                }),
            },
        }

    def execute(
        self,
        procgen_video_path: str,
        clips_dir: str,
        ledger_json: str,
        blend_mode: str = "addition",
        blend_opacity: float = 0.5,
        canvas_width: int = 1920,
        canvas_height: int = 1080,
        canvas_fps: int = 25,
        humo_target_height: int = 1080,
        fallback_clip_length: float = 7.0,
        ffmpeg: str = "ffmpeg",
    ):
        t_start = time.time()
        ffprobe = ffmpeg.replace("ffmpeg", "ffprobe") if ffmpeg.endswith("ffmpeg") else "ffprobe"

        # ---- Validate inputs ----
        procgen = Path(procgen_video_path.strip())
        if not procgen.exists():
            return ("", f"error: procgen_video_path not found: {procgen}")
        clips = Path(clips_dir.strip())
        if not clips.exists():
            return ("", f"error: clips_dir not found: {clips}")
        if not (shutil.which(ffmpeg) or Path(ffmpeg).exists()):
            return ("", f"error: ffmpeg not found at {ffmpeg!r}")

        ledger = _load_ledger(ledger_json)
        episode_id = ledger.get("episode_id", "episode")

        # ---- Resolve output_dir (canonical OTR tree) ----
        # output_dir = ComfyUI/output/otr/episodes/<episode_id>/
        comfy_output = Path(r"C:\Users\jeffr\Documents\ComfyUI\output")
        out_dir = comfy_output / "otr" / "episodes" / episode_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_mp4 = out_dir / f"{episode_id}.mp4"

        # ---- Build clip timeline ----
        timeline = _build_clip_timeline(ledger, clips, fallback_clip_length, ffprobe)
        if not timeline:
            return ("", f"error: no HuMo clips found in {clips}")
        episode_dur = _ffprobe_dur(procgen, ffprobe) or sum(d for _, _, d in timeline)

        # ---- Build filter graph ----
        filter_graph, inputs_list = _build_filter_graph(
            procgen_video=procgen,
            timeline=timeline,
            canvas_w=canvas_width,
            canvas_h=canvas_height,
            canvas_fps=canvas_fps,
            humo_h=humo_target_height,
            blend_mode=blend_mode,
            blend_opacity=blend_opacity,
            episode_dur=episode_dur,
        )

        # ---- Build ffmpeg command ----
        cmd: list[str] = [ffmpeg, "-y", "-loglevel", "warning"]
        for inp in inputs_list:
            cmd.extend(["-i", str(inp)])
        cmd.extend([
            "-filter_complex", filter_graph,
            "-map", "[v]",
            "-map", "0:a",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            str(out_mp4),
        ])
        log.info("[VideoComposite] ffmpeg: %s", " ".join(cmd[:6] + ["..."] + [cmd[-1]]))
        log.debug("[VideoComposite] full cmd: %s", cmd)

        report = [
            f"VideoComposite: episode={episode_id}",
            f"  procgen={procgen.name}",
            f"  clips_dir={clips}",
            f"  timeline: {len(timeline)} clip(s), episode_dur={episode_dur:.2f}s",
            f"  blend_mode={blend_mode} opacity={blend_opacity}",
            f"  canvas={canvas_width}x{canvas_height}@{canvas_fps}fps",
        ]
        for clip_path, s, d in timeline[:8]:
            report.append(f"    {clip_path.name}: t={s:.2f}-{(s+d):.2f}s")
        if len(timeline) > 8:
            report.append(f"    ... +{len(timeline) - 8} more")

        # ---- Run ffmpeg ----
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=900,
            )
            if r.returncode != 0:
                report.append(f"  FFMPEG FAIL rc={r.returncode}")
                report.append(f"  stderr: {r.stderr[-800:]}")
                return (str(out_mp4), "\n".join(report))
        except subprocess.TimeoutExpired:
            return (str(out_mp4), "error: ffmpeg timeout (>900s)")
        except Exception as exc:
            return (str(out_mp4), f"error: ffmpeg launch failed: {exc}")

        total_ms = int((time.time() - t_start) * 1000)
        report.append(f"  ffmpeg complete in {total_ms} ms")
        report.append(f"  out: {out_mp4}")
        return (str(out_mp4), "\n".join(report))


__all__ = ["VideoComposite"]
