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
