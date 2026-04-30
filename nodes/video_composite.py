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
import sys as _sys
import time
from pathlib import Path
from typing import Any

# Path-helper bootstrap (see ``batch_humo_render.py`` for full
# rationale). ``_otr_paths`` lives as a sibling module; this prepend
# makes it importable both via the parent ``nodes`` package
# (production) and via direct file load (test fixtures).
_NODES_DIR = Path(__file__).resolve().parent
if str(_NODES_DIR) not in _sys.path:
    _sys.path.insert(0, str(_NODES_DIR))
from _otr_paths import (  # noqa: E402
    episodes_for_obs_dir,
    otr_audio_dir,
    otr_legacy_audio_dir,
)

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
    """Backwards-compat shim around _load_ledger_with_path. Returns
    only the parsed ledger dict for callers that don't need the
    source path. New code should prefer _load_ledger_with_path so
    the source path is available for write-back."""
    ledger, _ = _load_ledger_with_path(arg)
    return ledger


def _load_ledger_with_path(arg: str) -> tuple[dict, "Path | None"]:
    """Accept inline JSON, ledger.json path, .mp4 path (suffix-swap
    to ledger), or empty (auto-pick newest non-pending). Mirrors
    BatchHumoRender._load_ledger_with_path so wiring is consistent.

    Returns (ledger_dict, ledger_path_or_None). Path is None for
    inline JSON; BUG-LOCAL-089 final_video_path write-back skips
    in that case rather than synthesizing a new file.
    """
    s = (arg or "").strip()

    if not s:
        audio_dirs = [
            otr_audio_dir(),
            otr_legacy_audio_dir(),
        ]
        cands = []
        for d in audio_dirs:
            if d.exists():
                cands.extend(
                    p for p in d.glob("*_ledger.json")
                    if not p.name.startswith("pending_")
                )
        if not cands:
            raise RuntimeError("VideoComposite: ledger_json empty and auto-pick found no ledger")
        p = max(cands, key=lambda x: x.stat().st_mtime)
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f), p

    if s.startswith("{"):
        return json.loads(s), None

    p = Path(s)
    # .mp4 path -> swap suffix to _ledger.json (SignalLostVideo convention)
    if p.suffix.lower() == ".mp4":
        ledger_p = p.with_suffix("").parent / f"{p.stem}_ledger.json"
        if ledger_p.exists():
            with open(ledger_p, "r", encoding="utf-8") as f:
                return json.load(f), ledger_p
        raise RuntimeError(
            f"VideoComposite: derived ledger from .mp4 not found: {ledger_p}"
        )

    if not p.exists():
        raise RuntimeError(f"VideoComposite: ledger path not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f), p


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

    Filter graph (BUG-LOCAL-092 architecture):
        [0:v] fps -> scale -> setsar -> [procgen]
        [i:v] fps -> scale 624:humo_h -> setsar ->
              setpts (per-clip start_s) -> [hi]
        chain: [procgen] overlay each [hi] at x=offset_x,y=0 with
               enable='between(t,start,start+dur)'
        (optional) blend pass for `additive procgen-over-humo`
                    feel -- lower default opacity, screen mode by
                    default; opacity=0 disables the pass entirely.
        final: [...] -> [v]

    Why this order: procgen is the BASE (full opacity), HuMo
    overlays opaquely at the center pillarbox during each line's
    time window. The pillarbox sides ALWAYS show full procgen
    (the audio-reactive CRT visualizer Jeffrey wants), and the
    center swaps to HuMo only while a character is speaking.
    The previous order (black canvas + HuMo on top + blanket-
    blend procgen at addition@0.5) drowned everything in the
    procgen color cast (BUG-LOCAL-092 magenta/pink wash).
    """
    humo_w = round(humo_h * (480 / 832) / 8) * 8  # snap to mult of 8 (h264 chroma safe)
    offset_x = (canvas_w - humo_w) // 2

    parts = []
    # [0] procgen → time-aligned + scaled = BASE layer (sides + initial center)
    parts.append(
        f"[0:v]fps={canvas_fps},scale={canvas_w}:{canvas_h},setsar=1[procgen]"
    )
    # Per-clip scale + overlay chain. HuMo overlays directly on procgen.
    last_label = "procgen"
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

    # Optional subtle procgen-over-HuMo "audio-reactive sheen" pass.
    # opacity=0 disables it entirely. Default is now low (0.0 in
    # widgets) so the composite is clean by default; users who want
    # the additive sheen can dial up blend_opacity in the widget.
    if blend_opacity and blend_opacity > 0:
        parts.append(
            f"[{last_label}][procgen]blend=all_mode={blend_mode}:"
            f"all_opacity={blend_opacity:.3f}[v]"
        )
    else:
        # Just rename the last overlay output to [v]
        parts.append(f"[{last_label}]copy[v]")
    return ";".join(parts), inputs_list


# ---------------------------------------------------------------------------
# humo_concat mode helpers (audio_source = "humo_concat")
# ---------------------------------------------------------------------------

def _make_gap_segment(
    image: Path,
    audio_source: Path,
    start_s: float,
    end_s: float,
    canvas_w: int,
    canvas_h: int,
    canvas_fps: int,
    out_path: Path,
    ffmpeg: str,
) -> Path:
    """Render a static-image mp4 covering [start_s, end_s] of the
    timeline. Image is letterboxed to canvas dimensions; audio is
    sliced from `audio_source` (typically the procgen mp4 carrying
    the master_mix track).

    Used to fill gap windows between HuMo dialogue clips in the
    humo_concat mode so the timeline has full visual + audio
    coverage. Falls back to silence if audio_source missing.
    """
    dur = max(0.0, end_s - start_s)
    if dur <= 0.0:
        raise ValueError(f"_make_gap_segment: non-positive duration {dur}")
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-loop", "1", "-framerate", str(canvas_fps), "-i", str(image),
        "-ss", f"{start_s:.4f}", "-i", str(audio_source),
        "-vf", (
            f"scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=decrease,"
            f"pad={canvas_w}:{canvas_h}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"fps={canvas_fps}"
        ),
        "-t", f"{dur:.4f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-shortest",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_path


def _normalize_humo_segment(
    clip: Path,
    canvas_w: int,
    canvas_h: int,
    canvas_fps: int,
    humo_target_h: int,
    out_path: Path,
    ffmpeg: str,
) -> Path:
    """Pillarbox a HuMo clip into the canvas dimensions while
    PRESERVING its native audio (the source of mechanical lip-sync
    for the humo_concat mode). HuMo clips are 480x832 portrait;
    pillarbox to canvas_w x canvas_h with black side bars.
    """
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(clip),
        "-vf", (
            f"scale=-2:{humo_target_h}:force_original_aspect_ratio=decrease,"
            f"pad={canvas_w}:{canvas_h}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"fps={canvas_fps}"
        ),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast",
        # KEY: keep HuMo's native audio for perfect lip-sync.
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return out_path


def _resolve_bookend_image(
    ledger: dict,
    procgen: Path,
    fallback_dir: Path,
    ffmpeg: str,
) -> Path:
    """Locate the radio bookend image. Priority:
    1. ledger.radio_bookend_path (top-level field stamped by BatchFluxRender)
    2. ledger.meta.radio_bookend_path
    3. Fallback: extract first frame from procgen mp4
    """
    cand = ledger.get("radio_bookend_path") or ledger.get("meta", {}).get("radio_bookend_path")
    if cand:
        p = Path(cand)
        if p.exists():
            return p
        log.warning(
            "[VideoComposite] ledger.radio_bookend_path=%s does not exist; "
            "falling back to procgen frame",
            p,
        )
    fallback = fallback_dir / "bookend_fallback.png"
    if not fallback.exists():
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(procgen), "-vframes", "1",
            str(fallback),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log.info("[VideoComposite] bookend fallback: extracted %s", fallback)
    return fallback


def _render_humo_concat_mode(
    *,
    ledger: dict,
    clips_dir: Path,
    procgen: Path,
    out_mp4: Path,
    canvas_w: int,
    canvas_h: int,
    canvas_fps: int,
    humo_target_h: int,
    fallback_clip_length: float,
    ffmpeg: str,
    ffprobe: str,
) -> tuple[Path, list[str]]:
    """Build the final mp4 in humo_concat mode.

    Pipeline:
      1. Sort ledger.clips[] by start_s, build (clip_path, start_s, dur_s)
         tuples honouring REAL master-mix timestamps (not cumulative).
      2. Resolve radio bookend image (or procgen-frame fallback).
      3. For each pair of adjacent clips with a gap between them,
         create a gap segment: bookend image + master_mix audio slice.
      4. For each HuMo clip, normalize to canvas dims keeping native
         audio (mechanical lip-sync guarantee).
      5. ffmpeg concat-demuxer joins all segments in time order.

    Returns (final_mp4_path, report_lines).
    """
    report: list[str] = []
    t0 = time.time()

    # 1. Build timeline from ledger.clips[] using real master-mix start_s
    ledger_clips = ledger.get("clips") or []
    timeline: list[tuple[Path, float, float]] = []
    for entry in ledger_clips:
        line_id = str(entry.get("line_id") or "")
        if not line_id:
            continue
        cp = Path(entry.get("mp4_path") or "")
        if not cp.exists():
            cp = clips_dir / f"{line_id}.mp4"
        if not cp.exists():
            log.info(
                "[VideoComposite/humo_concat] skip %s: clip not on disk",
                line_id,
            )
            continue
        start_s = float(entry.get("start_s") or 0.0)
        dur_s = float(entry.get("dur_s") or 0.0)
        if dur_s <= 0:
            dur_s = _ffprobe_dur(cp, ffprobe) or float(fallback_clip_length)
        timeline.append((cp, start_s, dur_s))
    if not timeline:
        raise RuntimeError("humo_concat: no usable HuMo clips in ledger.clips[]")
    timeline.sort(key=lambda t: t[1])

    # 2. Episode duration (from procgen so we cover trailing music)
    episode_dur = _ffprobe_dur(procgen, ffprobe) or (timeline[-1][1] + timeline[-1][2])
    report.append(
        f"humo_concat: {len(timeline)} HuMo clip(s), episode duration "
        f"{episode_dur:.2f}s"
    )

    # 3. Resolve bookend image
    seg_dir = out_mp4.parent / "_humo_concat_segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    bookend = _resolve_bookend_image(ledger, procgen, seg_dir, ffmpeg)
    report.append(f"humo_concat: bookend image = {bookend.name}")

    # 4. Build segment list (alternating gap, clip, gap, ...)
    segments: list[Path] = []
    cursor = 0.0
    for i, (clip_path, start_s, dur_s) in enumerate(timeline):
        if start_s - cursor > 0.05:  # 50ms tolerance for float drift
            gap_path = seg_dir / f"gap_{i:03d}.mp4"
            try:
                _make_gap_segment(
                    bookend, procgen, cursor, start_s,
                    canvas_w, canvas_h, canvas_fps,
                    gap_path, ffmpeg,
                )
                segments.append(gap_path)
            except Exception as exc:
                log.warning(
                    "[VideoComposite/humo_concat] gap[%d, %.2f-%.2f] failed: %s",
                    i, cursor, start_s, exc,
                )
        clip_seg = seg_dir / f"clip_{i:03d}.mp4"
        try:
            _normalize_humo_segment(
                clip_path, canvas_w, canvas_h, canvas_fps,
                humo_target_h, clip_seg, ffmpeg,
            )
            segments.append(clip_seg)
        except Exception as exc:
            log.warning(
                "[VideoComposite/humo_concat] clip[%d] %s normalize failed: %s",
                i, clip_path.name, exc,
            )
            continue
        cursor = start_s + dur_s
    # Trailing gap
    if cursor < episode_dur - 0.05:
        try:
            tail_seg = seg_dir / "gap_999.mp4"
            _make_gap_segment(
                bookend, procgen, cursor, episode_dur,
                canvas_w, canvas_h, canvas_fps,
                tail_seg, ffmpeg,
            )
            segments.append(tail_seg)
        except Exception as exc:
            log.warning(
                "[VideoComposite/humo_concat] trailing gap %.2f-%.2f failed: %s",
                cursor, episode_dur, exc,
            )

    if not segments:
        raise RuntimeError("humo_concat: no segments produced")
    report.append(f"humo_concat: {len(segments)} segment(s) ready for concat")

    # 5. Concat-demuxer assembly
    list_file = seg_dir / "concat.txt"
    list_file.write_text(
        "\n".join(f"file '{s.as_posix()}'" for s in segments),
        encoding="utf-8",
    )
    cmd = [
        ffmpeg, "-y", "-loglevel", "warning",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    elapsed = time.time() - t0
    report.append(
        f"humo_concat: assembled {out_mp4.name} in {elapsed:.1f}s"
    )
    return out_mp4, report


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
                # BUG-LOCAL-092: defaults flipped.
                # Old defaults (addition @ 0.5) blanket-drowned the
                # composite in procgen color cast (Jeffrey's "horrendously
                # pink" feedback). New defaults: `lighten` mode, opacity
                # 0.0 (sheen pass disabled by default). User can dial
                # opacity up for the audio-reactive sheen flash; lighten
                # mode preserves HuMo skin tones since it only takes the
                # max(procgen, humo) per channel rather than additively
                # tinting.
                "blend_mode": (
                    ["lighten", "screen", "addition", "overlay", "normal"],
                    {"default": "lighten"},
                ),
                "blend_opacity": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
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
                "cleanup_clips_after_assembly": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "[DEFAULT OFF -- experimental] If ON, after the "
                        "final mp4 is written and verified, delete every "
                        "per-line HuMo clip mp4 listed in ledger.clips[]. "
                        "Saves disk space (33 clips x 25 MB = ~800 MB per "
                        "episode) once the per-clip files are no longer "
                        "needed for re-encoding or VideoComposite re-runs. "
                        "Each deleted clip's ledger entry gets "
                        "'cleaned_up: true' stamped so a postmortem can "
                        "tell deleted-on-purpose from missing-due-to-bug. "
                        "Recommended OFF until BUG-101/102/106 are fully "
                        "stable -- you may want to re-encode a single "
                        "clip without re-running the 5-hour HuMo render."
                    ),
                }),
                "audio_source": (
                    ["humo_concat", "master_mix"],
                    {"default": "humo_concat",
                     "tooltip": (
                         "humo_concat (DEFAULT, talking-radio): "
                         "concat-demuxer pipeline where each HuMo clip "
                         "keeps its NATIVE audio (mechanical lip-sync) "
                         "and gap windows render the radio bookend image "
                         "with master_mix audio slices (music + SFX + "
                         "ANNOUNCER continuous in the gaps). "
                         "Falls back to procgen frame if "
                         "ledger.radio_bookend_path is missing. On any "
                         "internal failure, falls through to master_mix "
                         "so the run never breaks.\n\n"
                         "master_mix: legacy 2-layer filter graph. Audio "
                         "= procgen.wav (master_mix). HuMo clips overlay "
                         "as VIDEO only; their native audio is discarded. "
                         "Risk: lip-sync drift if master_mix differs from "
                         "what HuMo was conditioned on. Use only for "
                         "comparison or if humo_concat fails to deliver."
                     )},
                ),
            },
        }

    def execute(
        self,
        procgen_video_path: str,
        clips_dir: str,
        ledger_json: str,
        blend_mode: str = "lighten",
        blend_opacity: float = 0.0,
        canvas_width: int = 1920,
        canvas_height: int = 1080,
        canvas_fps: int = 25,
        humo_target_height: int = 1080,
        fallback_clip_length: float = 7.0,
        ffmpeg: str = "ffmpeg",
        cleanup_clips_after_assembly: bool = False,
        audio_source: str = "humo_concat",
    ):
        # `cleanup_clips_after_assembly` widget is wired but the deletion
        # logic itself is deferred (would happen post-assembly when the
        # final mp4 is verified). Until then we just accept the kwarg
        # without crashing and log if it was requested -- ComfyUI passes
        # every INPUT_TYPES key as a kwarg, so the parameter MUST be in
        # the signature even if the side-effect isn't implemented yet.
        if cleanup_clips_after_assembly:
            log.info(
                "[VideoComposite] cleanup_clips_after_assembly=True requested "
                "but deletion logic is not yet implemented -- clips kept "
                "for now (this is a known TODO, not a regression)."
            )
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

        ledger, ledger_path = _load_ledger_with_path(ledger_json)
        episode_id = ledger.get("episode_id", "episode")

        # ---- Resolve output_dir (broadcast tree, OBS-safe) ----
        # output_dir = ComfyUI/output/episodes_for_obs/<episode_id>/
        # Sibling of output/otr/ (NOT a child) so OBS's directory_sorter
        # can point at output/episodes_for_obs/ and only see finished
        # episodes -- never the per-line HuMo clip pieces under
        # output/otr/videos/<ep_id>/.
        out_dir = episodes_for_obs_dir(episode_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_mp4 = out_dir / f"{episode_id}.mp4"

        # ---- Branch on audio_source ----
        # humo_concat mode: bypass the existing 2-layer filter graph
        # entirely. Build a concat-demuxer pipeline where each HuMo
        # clip keeps its NATIVE audio (mechanical lip-sync) and gap
        # windows are filled with the radio bookend image + master_mix
        # audio slices (talking-radio identity continuous).
        if audio_source == "humo_concat":
            log.info(
                "[VideoComposite] audio_source=humo_concat -- routing to "
                "concat-demuxer pipeline (perfect lip-sync, bookend gaps)"
            )
            try:
                final_mp4, concat_report = _render_humo_concat_mode(
                    ledger=ledger,
                    clips_dir=clips,
                    procgen=procgen,
                    out_mp4=out_mp4,
                    canvas_w=canvas_width,
                    canvas_h=canvas_height,
                    canvas_fps=canvas_fps,
                    humo_target_h=humo_target_height,
                    fallback_clip_length=fallback_clip_length,
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                )
            except Exception as exc:
                log.exception(
                    "[VideoComposite] humo_concat failed; falling back to "
                    "master_mix mode: %s", exc,
                )
                # Fall through to master_mix path on any failure.
            else:
                # Stamp final_video_path in ledger same as master_mix path.
                if ledger_path is not None:
                    try:
                        ledger["final_video_path"] = str(final_mp4)
                        with open(ledger_path, "w", encoding="utf-8") as _f:
                            json.dump(ledger, _f, indent=2)
                    except Exception as _stamp_exc:  # noqa: BLE001
                        log.warning(
                            "[VideoComposite] humo_concat ledger stamp failed: %s",
                            _stamp_exc,
                        )
                report_text = (
                    "VideoComposite (humo_concat mode)\n"
                    + "\n".join(f"  {ln}" for ln in concat_report)
                )
                return (str(final_mp4), report_text)

        # ---- Build clip timeline (master_mix mode) ----
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

        # ---- BUG-LOCAL-089: write final_video_path back to ledger ----
        # The broadcast-tree mp4 is the canonical "this episode is
        # done" artifact. Persist its path in the ledger so OBS
        # schedulers, post-mortem tools, and any downstream re-runs
        # can resolve it via ledger lookup. Skipped silently when the
        # ledger came from inline JSON (no on-disk source).
        if ledger_path is not None:
            try:
                ledger["final_video_path"] = str(out_mp4)
                ledger["total_episode_dur_s"] = (
                    float(episode_dur) if episode_dur else
                    ledger.get("total_episode_dur_s")
                )
                with open(ledger_path, "w", encoding="utf-8") as f:
                    json.dump(ledger, f, indent=2, ensure_ascii=False)
                log.info(
                    "[VideoComposite] ledger updated: final_video_path -> %s",
                    out_mp4.name,
                )
                report.append(
                    f"  ledger updated: final_video_path={out_mp4.name}"
                )
            except Exception as exc:
                log.warning(
                    "[VideoComposite] ledger final_video_path write-back failed: %s",
                    exc,
                )
                report.append(f"  ledger write-back FAILED: {exc}")
        else:
            log.info(
                "[VideoComposite] inline ledger (no path) -- "
                "skipping final_video_path write-back"
            )
        return (str(out_mp4), "\n".join(report))


__all__ = ["VideoComposite"]
