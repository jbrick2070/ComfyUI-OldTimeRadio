"""OTR_PostUpscaleProcgenBlend -- post-RTXUpscale procgen visual blend.

BUG-LOCAL-030 Phase B (2026-05-03 EVENING, Jeffrey final spec):

The per-clip-mux composite path (master_mix_per_clip_mux mode) renders
HuMo + LTX clips into a 1472x832 canvas with black pillarbox bars on
each side of HuMo character clips. RTXUpscale takes that 1472x832 to
1920x1080 for delivery.

Procgen (OTR_SignalLostVideo) renders SEPARATELY at native 1920x1080
(NOT at the prior 832x480 that got upscaled with everything else). This
node takes:

  - source_mp4_path:  1920x1080 RTXUpscale output (HuMo + LTX content)
  - procgen_mp4_path: 1920x1080 procgen render (CRT scanlines + audio-
                      reactive flicker + waveform visualizer)

and produces a final 1920x1080 mp4 with procgen blended on top via
ffmpeg -filter_complex blend, audio passing through with ``-c:a copy``
(zero audio re-encodes).

Why post-upscale instead of pre-composite:
  1. Procgen is SYNTHETIC (CRT scanlines, geometric patterns). Running
     it through RTX VSR (designed for natural / AI-rendered content)
     produces ringing artifacts and softens the sharp digital character.
     Native-rendering procgen at 1920x1080 keeps it crisp.
  2. The per-clip-mux composite is the C7 byte-identity-protected audio
     path. Adding a procgen blend INTO that composite means re-encoding
     the audio mux, which makes byte-identity harder to guarantee.
     Post-upscale blend is purely visual (-c:a copy) so audio stays
     untouched between SignalLostVideo and final delivery.
  3. Procgen visually FILLS the HuMo character pillarbox bars from
     BUG-030 Phase A simple-pillarbox composite, turning the visible
     black surround into the SIGNAL LOST CRT signature (audio-reactive
     scanlines + flicker over the otherwise-static black bars).

Bypass mode (``bypass=True`` widget): copies source -> output
verbatim with no procgen overlay. Useful for A/B comparison or when
the procgen visual is unwanted (e.g. clean uplift for an external editor).
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_BLEND_MODE = "lighten"
_BLEND_MODE_CHOICES = ["lighten", "screen", "addition", "overlay", "normal"]


def _build_blend_cmd(
    source_mp4: Path,
    procgen_mp4: Path,
    out_mp4: Path,
    blend_mode: str,
    blend_opacity: float,
    ffmpeg: str,
) -> list[str]:
    """Build the ffmpeg command for procgen-over-source blend.

    Filter chain:
      [0:v]                                       # source (RTXUpscale output)
      [1:v] scale + fps conform                   # procgen scaled if needed
      [src][pgn] blend=all_mode={mode}:all_opacity={opacity}[v]

    Audio: ``-map 0:a? -c:a copy`` -- pass source audio through
    untouched (zero re-encodes, C7-safe).

    Video: libx264 yuv420p crf 18 preset fast -- one re-encode pass
    only on the visual blend.
    """
    # Conform procgen to source dims + fps so blend works correctly.
    # If procgen is already 1920x1080 (BUG-030 Phase B default), the
    # scale is a no-op; if it's still at the legacy 832x480, this
    # upscales it (loses crispness, hence the recommendation to render
    # procgen at 1920x1080 native).
    filter_complex = (
        f"[1:v]scale=-2:ih:force_original_aspect_ratio=decrease,"
        f"crop=iw:ih,setpts=PTS-STARTPTS[pgn];"
        f"[0:v][pgn]blend=all_mode={blend_mode}:"
        f"all_opacity={blend_opacity:.3f}[v]"
    )
    return [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(source_mp4),
        "-i", str(procgen_mp4),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        "-shortest",
        str(out_mp4),
    ]


class PostUpscaleProcgenBlend:
    """BUG-LOCAL-030 Phase B: post-RTXUpscale procgen visual blend.

    Takes the RTXUpscale output (1920x1080 HuMo + LTX content with
    black HuMo pillarbox bars from Phase A simple-pillarbox composite)
    and overlays the 1920x1080 native procgen render on top via
    ffmpeg -filter_complex blend. Audio passes through with ``-c:a copy``
    so the C7 byte-identity guarantee from per-clip-mux holds end-to-end.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_mp4_path": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": (
                        "Path to RTXUpscale output mp4 "
                        "(1920x1080 HuMo + LTX composite)."
                    ),
                }),
                "procgen_mp4_path": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": (
                        "Path to OTR_SignalLostVideo procgen mp4 "
                        "(1920x1080 native per BUG-030 Phase B)."
                    ),
                }),
            },
            "optional": {
                "blend_mode": (_BLEND_MODE_CHOICES, {"default": _DEFAULT_BLEND_MODE}),
                "blend_opacity": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Procgen overlay opacity. 0.0 = procgen invisible "
                        "(equivalent to bypass). 1.0 = procgen fully "
                        "replaces source where the blend mode says so. "
                        "0.5 default is a moderate sheen."
                    ),
                }),
                "ffmpeg": ("STRING", {
                    "default": "ffmpeg",
                    "multiline": False,
                    "tooltip": "ffmpeg binary path or PATH-resolvable name.",
                }),
                "bypass": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Skip the blend entirely and copy source -> output "
                        "verbatim. Useful for A/B comparison vs the procgen "
                        "blend, or when procgen visual is unwanted (clean "
                        "uplift for an external editor)."
                    ),
                }),
                "out_suffix": ("STRING", {
                    "default": "_procgen_blended",
                    "multiline": False,
                    "tooltip": (
                        "Filename suffix for the blended output. Final "
                        "filename: ``<source_stem><out_suffix>.mp4`` placed "
                        "in the same dir as ``source_mp4_path``."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_mp4_path", "report")
    FUNCTION = "blend"
    OUTPUT_NODE = True
    CATEGORY = "OldTimeRadio/video"

    def blend(
        self,
        source_mp4_path: str,
        procgen_mp4_path: str,
        blend_mode: str = _DEFAULT_BLEND_MODE,
        blend_opacity: float = 0.5,
        ffmpeg: str = "ffmpeg",
        bypass: bool = False,
        out_suffix: str = "_procgen_blended",
    ):
        report_lines: list[str] = []
        src = Path(source_mp4_path).resolve() if source_mp4_path else None
        pgn = Path(procgen_mp4_path).resolve() if procgen_mp4_path else None

        if src is None or not src.is_file():
            msg = (
                f"PostUpscaleProcgenBlend: source mp4 missing or not a file: "
                f"{source_mp4_path!r}"
            )
            log.warning("[PostUpscaleProcgenBlend] %s", msg)
            return ("", msg)

        output_path = src.with_name(f"{src.stem}{out_suffix}{src.suffix}")

        if bypass:
            try:
                shutil.copy2(src, output_path)
                msg = (
                    f"PostUpscaleProcgenBlend: bypass=True, copied "
                    f"{src.name} -> {output_path.name}"
                )
                log.info("[PostUpscaleProcgenBlend] %s", msg)
                return (str(output_path), msg)
            except Exception as exc:  # noqa: BLE001
                msg = f"PostUpscaleProcgenBlend: bypass copy failed: {exc}"
                log.warning("[PostUpscaleProcgenBlend] %s", msg)
                return ("", msg)

        if pgn is None or not pgn.is_file():
            # Procgen missing: gracefully degrade by copying source to
            # output (same behavior as bypass) so the pipeline still
            # produces a deliverable.
            try:
                shutil.copy2(src, output_path)
                msg = (
                    f"PostUpscaleProcgenBlend: procgen mp4 missing "
                    f"({procgen_mp4_path!r}); skipped blend, copied "
                    f"source -> output ({src.name} -> {output_path.name})"
                )
                log.warning("[PostUpscaleProcgenBlend] %s", msg)
                return (str(output_path), msg)
            except Exception as exc:  # noqa: BLE001
                msg = (
                    f"PostUpscaleProcgenBlend: procgen missing AND copy "
                    f"fallback failed: {exc}"
                )
                log.warning("[PostUpscaleProcgenBlend] %s", msg)
                return ("", msg)

        cmd = _build_blend_cmd(
            source_mp4=src, procgen_mp4=pgn, out_mp4=output_path,
            blend_mode=blend_mode, blend_opacity=float(blend_opacity),
            ffmpeg=ffmpeg,
        )
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            msg = (
                f"PostUpscaleProcgenBlend: ffmpeg blend failed -- "
                f"{stderr[:200]}; falling back to source-copy so the "
                f"pipeline still produces a deliverable"
            )
            log.warning("[PostUpscaleProcgenBlend] %s", msg)
            try:
                shutil.copy2(src, output_path)
                return (str(output_path), msg + f"; copied source -> {output_path.name}")
            except Exception as copy_exc:  # noqa: BLE001
                return ("", msg + f"; copy fallback also failed: {copy_exc}")

        report_lines.append(
            f"PostUpscaleProcgenBlend: blended {src.name} + {pgn.name} -> "
            f"{output_path.name} (mode={blend_mode}, opacity={blend_opacity:.3f})"
        )
        log.info("[PostUpscaleProcgenBlend] %s", report_lines[-1])
        return (str(output_path), "\n".join(report_lines))


__all__ = ["PostUpscaleProcgenBlend"]
