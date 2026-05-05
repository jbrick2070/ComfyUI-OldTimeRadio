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

# BUG-LOCAL-096 (2026-05-04 EVENING): default bumped from "lighten"
# at 0.5 to "screen" at 1.0 to bring procgen colors at full intensity.
# BUG-LOCAL-099 (2026-05-04 LATE EVENING): "screen" produced a global
# magenta tint when procgen has uniform color regions -- the radio room
# walls, porthole, and TV all came out pink because screen adds color
# values everywhere. Switched to "lighten" at 1.0 -- pixel-wise
# max(upscale, procgen). Bright procgen elements (SIGNAL LOST text,
# scanlines, waveform) show through at full intensity; mid-tone
# procgen content (ambient color cast) defers to the upscale wherever
# the upscale is brighter. Keeps the brightness Jeffrey asked for in
# BUG-096 without the color-cast side effect.
_DEFAULT_BLEND_MODE = "lighten"
_DEFAULT_BLEND_OPACITY = 1.0
# BUG-LOCAL-103 (2026-05-04 LATE EVENING): pre-blend shadow crush.
# Pixel inspection of a procgen mp4 dark region (signal_lost_echo_in_stasis
# 0:22, 99% of frame is luminance < 32) showed the "black" background is
# NOT true #000000 -- mean RGB was (4.86, 4.54, 10.06) with a clear blue
# cast (B is 2x R/G), and only 0.014% of pixels are actually #000. Any
# brighter-than-source blend (lighten, screen, addition, dodge) lifts the
# source dark areas toward this blue tint, producing a magenta/pink wash
# in highlights -- the BUG-099 symptom Jeffrey was chasing. Fix: clamp
# any procgen RGB channel below threshold to 0 BEFORE blend, normalizing
# the "near black" noise floor to true black so blend modes have nothing
# to leak. Threshold 18 = covers the 5-10 noise floor with margin while
# preserving the 95-max green flecks of legit motion content. 0 disables.
_DEFAULT_SHADOW_CRUSH = 18
# BUG-LOCAL-102 (2026-05-04 LATE EVENING): expanded the dropdown to include
# the popular ffmpeg blend filter modes so the BUG-099 tuning workflow can
# A/B test all useful options without code edits. Pre-102 the dropdown only
# accepted 5 modes (lighten, screen, addition, overlay, normal); the test
# workflow Jeffrey wanted with 8 modes had 5 of them rejected as invalid
# dropdown values (red text on the canvas). All 16 modes below are valid
# ffmpeg blend=all_mode= values per ffmpeg filter docs. Sorted by usefulness
# for OTR's overlay-on-upscale aesthetic, brightest-first then darken-tier.
_BLEND_MODE_CHOICES = [
    "lighten",       # default; max(A, B) per pixel
    "screen",        # 1 - (1-A)(1-B); always brighter
    "addition",      # A + B (clamped); brightest possible
    "overlay",       # multiply darks + screen lights
    "hardlight",     # like overlay but B-driven
    "softlight",     # gentler overlay
    "dodge",         # extreme bright lift
    "vividlight",    # combined dodge/burn around 0.5
    "linearlight",   # linear-dodge / linear-burn around 0.5
    "pinlight",      # darken or lighten depending on B
    "multiply",      # A * B; darkens
    "darken",        # min(A, B) per pixel
    "burn",          # darken inverse of dodge
    "difference",    # |A - B|; high contrast
    "exclusion",     # softer difference
    "normal",        # just opacity over upscale (B over A)
]


def _stamp_ledger_final_video_path(
    blended_mp4: Path,
    source_mp4: Path,
    procgen_mp4: Path,
    blend_mode: str,
    blend_opacity: float,
) -> tuple[bool, str]:
    """Stamp the in-flight ledger with the post-blend final deliverable path.

    BUG-LOCAL-030 audit gap fix (2026-05-03 EVENING): without this stamp,
    ``ledger.final_video_path`` still pointed at the pre-blend ``_1080p.mp4``
    that RTXUpscale wrote, even though the actual final deliverable is
    the procgen-blended ``_1080p_procgen_blended.mp4``. Anyone reading
    the ledger to find "the final mp4" picked the wrong file.

    Updates two fields (avoiding the meta.paths block per the schema
    doc rule "Never write to meta.paths from outside _build_meta_paths"
    -- that block is owned by the ledger save path and rebuilt fresh
    on every save):
      - ``ledger["final_video_path"]`` -> blended mp4 absolute path
        (top-level; this is the canonical "final mp4" pointer that
        downstream tooling reads)
      - ``ledger["meta"]["post_upscale_blend"]`` -> forensics block
        with source/procgen/out paths + blend params (so a future
        debugger can see what got blended into what AND find the
        blended file directly via meta.post_upscale_blend.blended_mp4)

    Best-effort -- never raises. Failure to stamp is logged as a warning;
    the actual blended mp4 is still on disk and discoverable via the
    canonical filename pattern. Returns ``(stamped: bool, msg: str)``.

    Uses ``in_flight_ledger_path()`` (BUG-LOCAL-021 Phase G singleton
    discovery) to find the ledger file. If no singleton is active (e.g.
    headless test), the stamp is silently skipped.
    """
    try:
        # Late imports: keep this helper standalone (no side-effects at
        # module import) so the broader pipeline doesn't gain a new
        # import chain just because the blend node is registered.
        try:
            from . import _otr_ledger as _OTRL  # type: ignore
        except ImportError:  # pragma: no cover -- direct-script fallback
            import sys as _sys
            _NODES_DIR = Path(__file__).resolve().parent
            if str(_NODES_DIR) not in _sys.path:
                _sys.path.insert(0, str(_NODES_DIR))
            import _otr_ledger as _OTRL  # type: ignore

        ledger_p = _OTRL.in_flight_ledger_path()
        if ledger_p is None:
            return (False, "no in-flight ledger singleton; skipped stamp")

        led = _OTRL.load_ledger_safe(ledger_p)
        if led is None:
            return (False, f"could not load ledger {ledger_p.name}; skipped stamp")

        # Update top-level final_video_path.
        led["final_video_path"] = str(blended_mp4)

        # Forensics block: what got blended into what + with what params.
        # Stamped under meta.post_upscale_blend (NOT under meta.paths,
        # which is owned exclusively by _build_meta_paths per schema doc).
        meta = led.setdefault("meta", {})
        meta["post_upscale_blend"] = {
            "source_mp4": str(source_mp4),
            "procgen_mp4": str(procgen_mp4),
            "blended_mp4": str(blended_mp4),
            "blend_mode": blend_mode,
            "blend_opacity": float(blend_opacity),
        }

        ok = _OTRL.save_ledger_safe(ledger_p, led)
        if not ok:
            return (False, f"save_ledger_safe returned False for {ledger_p.name}")
        return (True, f"stamped ledger {ledger_p.name}: final_video_path -> {blended_mp4.name}")
    except Exception as exc:  # noqa: BLE001
        return (False, f"ledger stamp failed: {type(exc).__name__}: {exc}")


def _build_blend_cmd(
    source_mp4: Path,
    procgen_mp4: Path,
    out_mp4: Path,
    blend_mode: str,
    blend_opacity: float,
    ffmpeg: str,
    shadow_crush_threshold: int = _DEFAULT_SHADOW_CRUSH,
    green_only_overlay: bool = False,
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
    # BUG-LOCAL-031 FIX (2026-05-03 EVENING): add filter-level
    # ``shortest=1`` to the blend filter. This clamps the VIDEO output
    # to the shorter of the two video inputs (the source mp4 from
    # RTXUpscale, ~50s) instead of running to the longer procgen
    # input (~94-114s). Audio is mapped separately via ``-map 0:a?
    # -c:a copy`` so the audio stream is untouched -- C7 byte-identity
    # holds.
    #
    # NOTE: this is the FILTER-LEVEL ``shortest=1`` (inside the blend
    # filter), NOT the muxer-level ``-shortest`` flag. The muxer flag
    # would cut audio when the shortest STREAM ends, which IS the C7
    # risk that drove us to drop ``-shortest`` earlier. The filter
    # flag only affects what the video filter emits; the muxer copies
    # the full audio stream untouched.
    # BUG-LOCAL-103 shadow crush: if threshold > 0, insert a lutrgb step
    # AFTER the scale/crop and BEFORE the blend. The lutrgb expression
    # multiplies each channel by 0 when val < threshold, else passes val
    # through unchanged -- forcing near-black procgen pixels to true 0
    # so brighter-than-source blend modes don't lift source darks toward
    # the procgen noise floor. Commas inside the gte() expression are
    # escaped with backslashes per ffmpeg filter-arg quoting rules.
    crush = max(0, int(shadow_crush_threshold))
    if crush > 0:
        crush_step = (
            f",lutrgb="
            f"r=val*gte(val\\,{crush}):"
            f"g=val*gte(val\\,{crush}):"
            f"b=val*gte(val\\,{crush})"
        )
    else:
        crush_step = ""
    # BUG-LOCAL-104 green-only overlay: when enabled, zero out the procgen
    # R and B channels via colorchannelmixer BEFORE the blend. Only the
    # procgen G channel survives, which means brighter-than-source blends
    # (lighten/screen/addition) only ever lift the source G channel where
    # procgen has green wireframe pixels. Source R and B pass through
    # untouched so the source scene's color (warm room, magenta porthole,
    # whatever) is preserved verbatim. The visible result is a pure
    # phosphor-green CRT overlay sitting on top of any source content,
    # which is the v1.7 SIGNAL LOST CRT signature Jeffrey wants restored.
    if green_only_overlay:
        green_only_step = (
            ",colorchannelmixer="
            "rr=0:rg=0:rb=0:"
            "gr=0:gg=1:gb=0:"
            "br=0:bg=0:bb=0"
        )
    else:
        green_only_step = ""
    filter_complex = (
        f"[1:v]scale=-2:ih:force_original_aspect_ratio=decrease,"
        f"crop=iw:ih,setpts=PTS-STARTPTS{crush_step}{green_only_step}[pgn];"
        f"[0:v][pgn]blend=all_mode={blend_mode}:"
        f"all_opacity={blend_opacity:.3f}:shortest=1[v]"
    )
    # BUG-LOCAL-030 C7 hardening (2026-05-03 EVENING, post-round-robin
    # Gemini catch): NO ``-shortest`` flag here. ffmpeg ``-shortest`` on
    # an A/V mux stops writing the audio stream as soon as the SHORTEST
    # input ends -- which means if procgen is even 40ms shorter than
    # source (likely from 24fps procgen vs 25fps source frame
    # quantization on the same master_mix duration), the trailing audio
    # gets truncated and Rule C7 byte-identity is silently broken.
    #
    # Without ``-shortest``, ffmpeg framesync defaults to holding the
    # last procgen frame as the overlay continues; audio passes through
    # via ``-c:a copy`` from source until source EOF. Result: visual
    # may have a 40ms tail of held procgen frame (imperceptible);
    # audio reaches the full source duration; C7 holds.
    # BUG-LOCAL-030 long-form hardening (2026-05-03 EVENING, post
    # round-robin risk-#10 review): cap thread fanout + raise mux queue
    # so a long-form (>5 min) episode does not spike DRAM via
    # thread x framebuffer multiplication and does not error out with
    # "Too many packets buffered for output stream" on the blend pass.
    # ChatGPT + Gemini both flagged this; Gemini's exact recommendation.
    return [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(source_mp4),
        "-i", str(procgen_mp4),
        "-filter_complex", filter_complex,
        "-filter_complex_threads", "2",
        "-filter_threads", "2",
        "-threads", "4",
        "-max_muxing_queue_size", "1024",
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
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
                    "default": _DEFAULT_BLEND_OPACITY,
                    "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "Procgen overlay opacity. 0.0 = procgen invisible "
                        "(equivalent to bypass). 1.0 = procgen contributes "
                        "at full strength via the selected blend mode. "
                        "Default 1.0 + 'screen' mode (BUG-LOCAL-096) gives "
                        "the canonical bright-additive overlay -- procgen "
                        "colors at full intensity, upscale still visible "
                        "underneath. Drop to 0.5 for a moderate sheen "
                        "instead of full strength."
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
                "green_only_overlay": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "BUG-LOCAL-104: when True, zero the procgen R and "
                        "B channels (colorchannelmixer) BEFORE the blend so "
                        "only the procgen G channel ever contributes. Source "
                        "R/B pass through untouched (no color lift); source "
                        "G gets boosted only where the procgen has green "
                        "wireframe pixels. Result: pure phosphor-green v1.7 "
                        "CRT overlay sitting on top of whatever the source "
                        "scene already looks like. Use this when the source "
                        "(HuMo+LTX upscale) has its own scene color and you "
                        "want the green CRT to be visible on top regardless. "
                        "Pair with blend_mode='lighten' or 'addition'."
                    ),
                }),
                "shadow_crush_threshold": ("INT", {
                    "default": _DEFAULT_SHADOW_CRUSH,
                    "min": 0, "max": 50, "step": 1,
                    "tooltip": (
                        "BUG-LOCAL-103: pre-blend shadow crush. Any procgen "
                        "RGB channel value below this threshold is forced to "
                        "0 BEFORE the blend, normalizing near-black noise "
                        "(e.g. (5,5,10) blue-tinted procgen black) to true "
                        "#000000 so brighter blends (lighten/screen/addition) "
                        "don't lift source darks toward the procgen tint. "
                        "Default 18 covers a (5-10) noise floor with margin "
                        "while preserving real motion content (procgen "
                        "highlights up to ~95). 0 disables -- use that to "
                        "verify the un-crushed pink/cast symptom or to A/B "
                        "test a procgen render with hardened prompt that "
                        "already produces true black."
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
        blend_opacity: float = _DEFAULT_BLEND_OPACITY,
        ffmpeg: str = "ffmpeg",
        bypass: bool = False,
        out_suffix: str = "_procgen_blended",
        green_only_overlay: bool = False,
        shadow_crush_threshold: int = _DEFAULT_SHADOW_CRUSH,
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

        # BUG-LOCAL-030 long-form hardening (2026-05-03 EVENING, post
        # round-robin risk-#10 review): phase barrier handoff from
        # RTXUpscale. Reclaim DRAM/VRAM that PyTorch may still be
        # holding from upscaling, then check the canary before kicking
        # off filter_complex blend (which buffers frames from BOTH
        # input mp4s). DRAM canary degrades open -- if we can't verify
        # we still attempt the blend; only an actively low reading
        # produces a warning in the report.
        try:
            from . import _otr_memory as _OTRM  # type: ignore
            _OTRM.phase_gc("PostUpscaleProcgenBlend entry")
            _ok, _reason = _OTRM.dram_canary(label="PostUpscaleProcgenBlend entry")
            if not _ok:
                report_lines.append(
                    f"PostUpscaleProcgenBlend: DRAM canary WARNING -- {_reason}"
                )
        except Exception as _exc:  # noqa: BLE001
            log.warning(
                "[PostUpscaleProcgenBlend] memory hygiene helper "
                "unavailable (%s); proceeding", _exc,
            )

        cmd = _build_blend_cmd(
            source_mp4=src, procgen_mp4=pgn, out_mp4=output_path,
            blend_mode=blend_mode, blend_opacity=float(blend_opacity),
            ffmpeg=ffmpeg,
            shadow_crush_threshold=int(shadow_crush_threshold),
            green_only_overlay=bool(green_only_overlay),
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
            f"{output_path.name} (mode={blend_mode}, opacity={blend_opacity:.3f}, "
            f"shadow_crush={int(shadow_crush_threshold)}, "
            f"green_only={bool(green_only_overlay)})"
        )
        log.info("[PostUpscaleProcgenBlend] %s", report_lines[-1])

        # BUG-LOCAL-030 audit gap fix: stamp ledger.final_video_path with
        # the post-blend deliverable path so downstream tooling reading
        # the ledger picks the right "final mp4" -- not the pre-blend
        # _1080p.mp4 RTXUpscale wrote earlier. Best-effort; never raises.
        # Only invoked on successful real blend (NOT bypass / missing
        # procgen / ffmpeg failure paths -- those leave the previous
        # ledger.final_video_path intact, which is correct for those
        # fallback modes since the output IS just a copy of source).
        stamped, stamp_msg = _stamp_ledger_final_video_path(
            blended_mp4=output_path,
            source_mp4=src,
            procgen_mp4=pgn,
            blend_mode=blend_mode,
            blend_opacity=float(blend_opacity),
        )
        if stamped:
            log.info("[PostUpscaleProcgenBlend] %s", stamp_msg)
            report_lines.append(f"  ledger: {stamp_msg}")
        else:
            log.warning("[PostUpscaleProcgenBlend] ledger NOT stamped: %s", stamp_msg)
            report_lines.append(f"  ledger: WARN ledger not stamped ({stamp_msg})")

        return (str(output_path), "\n".join(report_lines))


__all__ = ["PostUpscaleProcgenBlend"]
