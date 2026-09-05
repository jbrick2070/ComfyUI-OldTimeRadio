"""OTR_PostUpscaleProcgenBlend -- procgen visual blend over the composite.

Takes the OTR_SilentComposite's 1920x1080 silent output and blends the
1920x1080 procgen render (OTR_SignalLostVideo -- CRT scanlines + audio-
reactive flicker + waveform visualizer) on top via ffmpeg
-filter_complex blend. Audio passes through with ``-c:a copy`` (zero
audio re-encodes; C7 byte identity preserved end-to-end).

Inputs:
  - source_mp4_path:  1920x1080 composite output
  - procgen_mp4_path: 1920x1080 procgen render

Why the procgen blend runs POST-composite rather than pre-composite:
  1. Procgen is SYNTHETIC (CRT scanlines, geometric patterns). Running
     it through a model-based super-resolution pass alongside natural /
     AI-rendered clip content would introduce ringing artifacts.
     Native-rendering procgen at 1920x1080 keeps it crisp.
  2. The composite is the C7 byte-identity-protected audio path. Adding
     a procgen blend INTO that composite means re-encoding the audio mux,
     which makes byte-identity harder to guarantee. This node's blend
     stays purely visual (-c:a copy) so audio is untouched.
  3. Procgen visually FILLS pillarbox surround from the composite's
     mixed 4:3 / 16:9 sources, turning the visible black surround into
     the SIGNAL LOST CRT signature (audio-reactive scanlines + flicker
     over the otherwise-static black bars).

Bypass mode (``bypass=True`` widget): copies source -> output verbatim
with no procgen overlay. Useful for A/B comparison or when the procgen
visual is unwanted (e.g. clean uplift for an external editor).

**History note (2026-08-08):** the retired ``OTR_RTXUpscale`` node used
to sit between OTR_SilentComposite and this node, taking a smaller
composite canvas up to 1080p via an NVIDIA-only RTX VSR pass. That
stage was ripped as part of queue item 8; the composite chain now
delivers 1080p directly (via ``render.composite_w/h`` per profile), and
per-clip model enhancement lives inside SilentComposite itself
(``nodes/_otr_upscale_engines/`` -- device-selectable across vendors).
This node's inputs are unchanged.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

# Ensure sibling node modules (e.g. _otr_shared) resolve when this file is
# loaded FLAT by ComfyUI's custom-node loader -- the flat fallbacks of the
# two try/except imports below need it. Package-relative comes FIRST in
# each: the flat spelling resolves through this insert even inside the
# package and yields a SECOND module instance of the owner it names
# (kibitz runpod-found-fixes r3/r4, 2026-09-04).
_NODES_DIR = os.path.dirname(os.path.abspath(__file__))
if _NODES_DIR not in sys.path:
    sys.path.insert(0, _NODES_DIR)

try:
    from ._otr_shared import proc as otr_proc  # noqa: E402
except ImportError:  # pragma: no cover -- flat / standalone load
    from _otr_shared import proc as otr_proc  # type: ignore  # noqa: E402

try:
    from ._otr_shared.ffmpeg import resolve_ffmpeg  # noqa: E402
except ImportError:  # pragma: no cover -- flat (sys.path) load
    from _otr_shared.ffmpeg import resolve_ffmpeg  # type: ignore  # noqa: E402


def _ffmpeg_bin(ffmpeg: str = "ffmpeg") -> str:
    """The pack's ONE ffmpeg answer, kept as a string. This node's widget
    default is the bare literal, and until 2026-09-04 that literal went
    straight into every subprocess here (decode, blend, bars), so a box whose
    ffmpeg is reachable only through OTR_FFMPEG blended nothing -- and no
    resolver copy existed for the guard to catch."""
    # NEVER reflect the argument back (2026-09-04). This used to return
    # `str(ffmpeg).strip()` when resolution found nothing, which handed an
    # UNRESOLVED caller string straight to argv[0] -- so a rejected value
    # came back through the fallback and was spawned anyway. "" means "this
    # box has no ffmpeg", and blend() degrades on it by name.
    return resolve_ffmpeg(ffmpeg) or ""

try:
    from ._otr_shared import ffprobe as _ffp  # noqa: E402
except ImportError:  # pragma: no cover -- flat (sys.path) load
    from _otr_shared import ffprobe as _ffp  # type: ignore  # noqa: E402

# NOTE: the SDH caption builder import (build_ass_from_ledger) was removed here in
# the 2026-07-04 widget-audit Batch 3 -- captions migrated to node 86
# OTR_CaptionBurn. This node no longer builds or burns captions.

log = logging.getLogger(__name__)


#: Characters that are SYNTAX inside an ffmpeg filtergraph, so a filename
#: carrying one changes what the graph means rather than what it reads:
#: `,` ends a filter, `;` ends a chain, `[` `]` delimit pad labels, `:` and `=`
#: separate a filter's options, `'` and `\` are the escaping mechanism itself.
#: A SPACE is deliberately NOT here -- it is legal in a filename and harmless.
_FILTERGRAPH_SYNTAX = set(",;:=[]'\\")


def _reject_filtergraph_syntax(name: str) -> str:
    """``name`` unchanged, or ``ValueError`` if it would alter the graph.

    The pack's own episode stems cannot trip this -- every one goes through
    ``production_ledger._slugify``, which maps ``[^a-z0-9]+`` to ``_`` -- so
    this can never fire on a normal render. It exists because the stem is
    reachable from a workflow-supplied path, and `ass={name}` is interpolated
    into an UNQUOTED filtergraph.
    """
    bad = sorted(set(name) & _FILTERGRAPH_SYNTAX)
    if bad:
        raise ValueError(
            "caption filename %r contains ffmpeg filtergraph syntax (%s); "
            "refusing to build the graph. Rename the output so its stem is "
            "plain text." % (name, " ".join(repr(c) for c in bad)))
    return name


def _ass_filter_arg(ass_path: str) -> tuple[str, str]:
    """Return (filter_basename, cwd) for the ffmpeg ``ass=`` filter.

    Windows drive-letter colons cannot be reliably escaped inside an ffmpeg
    filtergraph (``ass=C\\:/...`` fails to parse). The robust cross-platform
    trick is to reference the subtitle file by BASENAME only and run ffmpeg
    with its working directory set to the file's folder -- no colon, no
    separators, nothing the filtergraph parser can choke on. Input/output mp4
    paths stay absolute (they are command args, not filtergraph tokens).
    """
    p = Path(ass_path)
    return (_reject_filtergraph_syntax(p.name), str(p.parent))


# SDH caption resolution (_resolve_captions_ass) was REMOVED here in the
# 2026-07-04 widget-audit Batch 3: caption ownership migrated to node 86
# OTR_CaptionBurn, which ports this same suffix-strip + sibling-audio ledger
# resolution into its own _resolve_ledger_path. This node no longer resolves or
# burns captions.

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
# BUG-LOCAL-106 (2026-05-04 LATE EVENING): production default flipped
# from "lighten" + green_only=False to "screen" + green_only=True after
# BUG-105 A/B confirmed `screen_GREEN_crush18` was the visibly correct
# combo on echo_in_stasis source. The `screen` formula
# (out = A + B - A*B) preserves source color in R and B (since procgen
# R and B are zeroed by green_only_overlay) and lifts source G exactly
# where the green CRT wireframe lives -- visible v1.7 SIGNAL LOST
# phosphor on top of any HuMo+LTX scene render. Pre-106 default
# (lighten) collapses to white over fully-saturated source pixels
# (max(255,0,255), (0,255,0)) = (255,255,255) -- looks like glare,
# not phosphor. screen avoids that math collapse.
_DEFAULT_BLEND_MODE = "screen"
_DEFAULT_BLEND_OPACITY = 1.0
_DEFAULT_GREEN_ONLY = True
# SDH caption constants removed 2026-07-04 (widget-audit Batch 3): caption
# ownership migrated to node 86 OTR_CaptionBurn (which defines its own).
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
    ``ledger.final_video_path`` still pointed at the pre-blend mp4 the
    upstream stage wrote, even though the actual final deliverable is
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


def _probe_dims(path: Path, ffmpeg: str = "ffmpeg") -> "Optional[tuple]":
    """(width, height) of the first video stream via ffprobe, or None.
    Best-effort: a probe failure returns None and the blend falls back to the
    legacy self-referential scale (the pre-2026-06-09 behavior).

    The blend's ``ffmpeg`` widget still gets first say -- the shared resolver
    tries its sibling before anything else -- but a box that pins OTR_FFPROBE
    is now heard too, which the hand-rolled ``ffprobe.exe`` swap never was."""
    try:
        out = _ffp.probe_raw(
            ["-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0",
             str(path)],
            ffmpeg=ffmpeg, timeout=30)
        if out.returncode != 0:
            return None
        w, h = (out.stdout or "").strip().split("x")[:2]
        return int(w), int(h)
    except Exception:  # noqa: BLE001
        return None


def _build_blend_cmd(
    source_mp4: Path,
    procgen_mp4: Path,
    out_mp4: Path,
    blend_mode: str,
    blend_opacity: float,
    ffmpeg: str,
    shadow_crush_threshold: int = _DEFAULT_SHADOW_CRUSH,
    green_only_overlay: bool = False,
    captions_ass_path: Optional[str] = None,
    source_dims: "Optional[tuple]" = None,
    scopes_mp4: "Optional[Path]" = None,
) -> list[str]:
    """Build the ffmpeg command for procgen-over-source blend.

    Filter chain:
      [0:v]                                       # source (composite output)
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
    # to the shorter of the two video inputs (the composited source
    # mp4, ~50s) instead of running to the longer procgen
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
    # BUG-LOCAL-104 green-only overlay + BUG-LOCAL-105 RGB planar pin:
    # when enabled, zero the procgen R and B channels via colorchannelmixer
    # BEFORE the blend so only the procgen G channel survives. THEN pin
    # both inputs to gbrp (planar RGB) before blend and back to yuv420p
    # after, so the per-channel `blend` math runs in RGB instead of YUV.
    # Without the format pin, ffmpeg auto-converts to yuv420p to match
    # libx264's pix_fmt and the blend filter ends up running 'lighten'
    # on Y/U/V planes -- which produces color-shifting garbage that
    # swallows the 1%-sparse green wireframe entirely. With the pin,
    # 'lighten/screen/addition' run as honest per-RGB-channel ops and
    # the green CRT overlays cleanly on top of any source color.
    #
    # Filter chain when green_only_overlay=True:
    #   [1:v] scale -> crop -> setpts -> lutrgb (crush) ->
    #         colorchannelmixer (zero R+B) -> format=gbrp [pgn]
    #   [0:v] format=gbrp [main]
    #   [main][pgn] blend=mode:opacity:shortest -> format=yuv420p [v]
    #
    # Filter chain when green_only_overlay=False (legacy path):
    #   [1:v] scale -> crop -> setpts -> lutrgb (crush) [pgn]
    #   [0:v][pgn] blend=mode:opacity:shortest [v]
    if green_only_overlay:
        green_only_step = (
            ",colorchannelmixer="
            "rr=0:rg=0:rb=0:"
            "gr=0:gg=1:gb=0:"
            "br=0:bg=0:bb=0"
            ",format=gbrp"
        )
        main_format_step = "[0:v]format=gbrp[main];"
        main_input_label = "[main]"
        post_blend_format = ",format=yuv420p"
    else:
        green_only_step = ""
        main_format_step = ""
        main_input_label = "[0:v]"
        post_blend_format = ""
    # SDH open-caption burn (P1): when an .ass path is provided, route the
    # blend output through an intermediate label and burn captions with the
    # libass ``ass`` filter at native 1080p. Pure video op -- audio is still
    # mapped via ``-c:a copy`` below, so C7 byte-identity is untouched.
    blend_label = "[vpre]" if captions_ass_path else "[v]"
    # conform helper (procgen scaled to the probed source dims; the scopes use
    # the same target). Defined here so the 3-input branch can reuse it.
    if source_dims:
        sw, sh = int(source_dims[0]), int(source_dims[1])
        _conform = f"scale={sw}:{sh}"
    else:
        _conform = "scale=-2:ih:force_original_aspect_ratio=decrease,crop=iw:ih"

    # -- §4D 3-INPUT branch: procgen FLOOR + scene-aware SCOPES ----------
    # [0:v]=source(upscaled) [1:v]=procgen floor [2:v]=scopes_only.mp4. A NEW
    # gbrp-throughout double-blend (NOT appended to the 2-input graph -- that
    # one converts to yuv420p before blend 1): screen the floor on, THEN
    # LIGHTEN (max) the scopes on -- lighten (not a 2nd screen) so the two
    # green layers do not compound brightness where they overlap. The scopes
    # are green-only by construction; the colorchannelmixer zero-R+B is a
    # belt-and-braces guard. Audio is still -map 0:a? -c:a copy (C7-safe).
    if scopes_mp4 is not None:
        scp_zero = (
            "colorchannelmixer="
            "rr=0:rg=0:rb=0:gr=0:gg=1:gb=0:br=0:bg=0:bb=0")
        # green_only_step ALREADY ends with ',format=gbrp' when the overlay is
        # ON, so only add the gbrp pin here when it is OFF. Appending a second
        # 'format=gbrp' would collapse into 'format=gbrpformat=gbrp' -> ffmpeg
        # rejects the bogus 'gbrpformat' option and the WHOLE §4D blend (scope
        # overlay + SDH caption burn) silently falls back to source-copy.
        # BUG-LOCAL-402.
        pgn_gbrp = "" if green_only_step else ",format=gbrp"
        fc = (
            f"[0:v]format=gbrp[main];"
            f"[1:v]{_conform},setpts=PTS-STARTPTS{crush_step}{green_only_step}{pgn_gbrp}[pgn];"
            f"[2:v]{_conform},setpts=PTS-STARTPTS,setsar=1,{scp_zero},"
            # BUG-LOCAL-410: the scopes track ends at the master-audio length,
            # but the composite/floor now run ~20s longer for the rolling-
            # credits post-roll. Without padding, the `shortest=1` lighten-blend
            # below would re-clamp the deliverable back to the (shorter) scopes
            # length and re-cut the credits scroll. Pad the scopes with BLACK
            # past its end (lighten(credits, black) == credits, so the tail is
            # untouched); `shortest=1` then clamps to the composite/floor length.
            f"format=gbrp,tpad=stop_mode=add:color=black:stop_duration=3600[scp];"
            f"[main][pgn]blend=all_mode={blend_mode}:"
            f"all_opacity={blend_opacity:.3f}:shortest=1[tmp];"
            f"[tmp][scp]blend=all_mode=lighten:shortest=1,"
            f"format=yuv420p{blend_label}"
        )
        if captions_ass_path:
            ass_name, _ass_cwd = _ass_filter_arg(captions_ass_path)
            fc += f";[vpre]ass={ass_name}[v]"
        return [
            ffmpeg, "-y", "-loglevel", "error",
            "-i", str(source_mp4),
            "-i", str(procgen_mp4),
            "-i", str(scopes_mp4),
            "-filter_complex", fc,
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

    # 2026-06-09 dim-conform fix (capstone soak catch): the legacy
    # ``scale=-2:ih`` referenced the PROCGEN'S OWN height -- a no-op that only
    # worked when the source arrived already at 1080p. The blend
    # filter hard-fails on a size mismatch (1472x832 source vs 1920x1080
    # procgen). Scale the procgen EXPLICITLY to the probed source dims (the
    # ~0.4% aspect difference is imperceptible for a full-frame CRT overlay).
    if source_dims:
        sw, sh = int(source_dims[0]), int(source_dims[1])
        conform = f"scale={sw}:{sh}"
    else:
        conform = "scale=-2:ih:force_original_aspect_ratio=decrease,crop=iw:ih"
    filter_complex = (
        f"[1:v]{conform},setpts=PTS-STARTPTS{crush_step}{green_only_step}[pgn];"
        f"{main_format_step}"
        f"{main_input_label}[pgn]blend=all_mode={blend_mode}:"
        f"all_opacity={blend_opacity:.3f}:shortest=1{post_blend_format}{blend_label}"
    )
    if captions_ass_path:
        ass_name, _ass_cwd = _ass_filter_arg(captions_ass_path)
        filter_complex += f";[vpre]ass={ass_name}[v]"
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


# --------------------------------------------------------------------------- #
# BUG 4 (2026-06-20 operator): ALWAYS-ON audio-reactive bottom bars overlay.
# A SEPARATE, unique overlay layer -- decoupled from OTR_SceneAwareScopes'
# scene-aware floor (which gets covered by full-frame clips and suppressed for
# portrait/un-probeable/credits frames). Default ON (`audio_bars='bottom'`); a
# manual `off` switch makes it byte-identical to today. The bars are PIL-rendered
# with the SAME `scope_draw.freq_bars_green` look, seated just ABOVE the lower-15%
# caption safe-area (captions stay ABOVE the bars), and lighten-blended onto the
# final at a partial opacity so they read as an accent, not a full-green wash.
# --------------------------------------------------------------------------- #

#: Bars lighten-blend opacity (operator: "ok if it's 60% not a full green").
_BARS_OPACITY = 0.6
#: Bars strip seats above the lower-15% caption safe-area (mirrors
#: otr_scene_aware_scopes._BARS_CAPTION_SAFE_FRAC so captions never collide).
_BARS_CAPTION_SAFE_FRAC = 0.15


def _bars_strip_geom(w: int, h: int) -> "tuple[int, int, int, int]":
    """(x, y, w, h) for the bottom bars strip -- a wide green frequency strip with
    a 5% side margin, ~10% tall, seated just above the caption safe-area. Mirrors
    otr_scene_aware_scopes._bars_geom so the look/placement is identical."""
    margin = int(w * 0.05)
    strip_h = max(8, int(h * 0.10))
    safe = int(h * _BARS_CAPTION_SAFE_FRAC)
    y = h - safe - strip_h
    return margin, y, w - 2 * margin, strip_h


def _probe_fps(path: Path, ffmpeg: str) -> float:
    """Source video fps via ffprobe (r_frame_rate). Defaults to 25.0 (the OTR
    canonical) on any failure so the bars layer always has a sane rate.

    The old binary spelling here was ``str(ffmpeg).replace("ffmpeg", "ffprobe")``
    -- a blind string swap that never checked the result exists, and that
    rewrote any directory named ``ffmpeg`` along the way. It also parsed the
    rational rate a THIRD time, in a third dialect. Both jobs are the shared
    boundary's now; the 25.0 fallback is still this node's own call."""
    try:
        out = _ffp.probe_raw(
            ["-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0",
             str(path)],
            ffmpeg=ffmpeg, timeout=30)
        if out.returncode != 0:
            return 25.0
        return _ffp.parse_rate((out.stdout or "").strip()) or 25.0
    except Exception:  # noqa: BLE001
        return 25.0


def _find_master_audio(source_mp4: Path) -> Path:
    """The master-mix audio for this episode. At the PostUpscaleProcgenBlend stage
    the video is SILENT (audio is muxed LATER by OTR_MasterAudioMux), so the bars
    must read the MASTER WAV from the episode's ``audio/`` dir (``*_master.wav``,
    newest), NOT the silent source mp4. Falls back to the source mp4 itself when no
    master wav is present (e.g. a source that already carries audio)."""
    try:
        adir = source_mp4.parent / "audio"
        cands = sorted(adir.glob("*_master.wav"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            return cands[0]
    except Exception:  # noqa: BLE001
        pass
    return source_mp4


def _load_master_audio_np(source_mp4: Path, ffmpeg: str):
    """Decode the episode MASTER audio to a mono float32 numpy array + sample rate
    via ffmpeg -> a temp pcm_s16le wav -> stdlib ``wave`` (no soundfile dependency).
    Returns ``(audio_np, sr)`` or ``(None, 0)``. The master audio is only READ --
    never altered -- so audio byte-identity is untouched."""
    import tempfile
    import wave
    import numpy as np
    fb = _ffmpeg_bin(ffmpeg)  # the owner's answer, even when called standalone
    audio_src = _find_master_audio(source_mp4)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        otr_proc.run(
            [fb, "-y", "-loglevel", "error", "-i", str(audio_src),
             "-vn", "-ac", "1", "-ar", "22050", "-acodec", "pcm_s16le", tmp_path],
            check=True, stdout=otr_proc.PIPE, stderr=otr_proc.PIPE)
        with wave.open(tmp_path, "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return (a if a.size else None), int(sr)
    except Exception as exc:  # noqa: BLE001
        log.warning("[PostUpscaleProcgenBlend] audio_bars: master-audio decode "
                    "failed (%s); bars layer skipped this run", exc)
        return None, 0
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _render_bars_only_mp4(source_mp4: Path, bars_out: Path, w: int, h: int,
                          fps: float, ffmpeg: str):
    """Render the SEPARATE bars-only layer: black frames with the green
    `freq_bars_green` strip painted in the bottom band, driven by the master
    audio envelope. Returns the path or None (LOUD) on any failure so the caller
    degrades to the normal captioned blend."""
    try:
        try:
            from _otr_shared.scope_draw import (  # type: ignore
                analyze_audio_np, encode_silent_mp4, freq_bars_green)
        except Exception:  # pragma: no cover -- packaged import
            from nodes._otr_shared.scope_draw import (  # type: ignore
                analyze_audio_np, encode_silent_mp4, freq_bars_green)
        import numpy as np
        from PIL import Image, ImageDraw
        a, sr = _load_master_audio_np(source_mp4, ffmpeg)
        if a is None or sr <= 0:
            return None
        total = int(np.ceil(len(a) / float(sr) * float(fps)))
        if total <= 0:
            return None
        _vol, freqs, _wav = analyze_audio_np(a, sr, total, int(round(fps)))
        mx, by, bw, bh = _bars_strip_geom(int(w), int(h))

        def _frames():
            for i in range(total):
                img = Image.new("RGB", (int(w), int(h)), (0, 0, 0))
                freq_bars_green(ImageDraw.Draw(img), freqs[i], mx, by, bw, bh)
                yield np.asarray(img, dtype=np.uint8)

        encode_silent_mp4(_frames(), total, str(bars_out),
                          int(w), int(h), int(round(fps)), ffmpeg)
        return bars_out if bars_out.is_file() else None
    except Exception as exc:  # noqa: BLE001
        log.warning("[PostUpscaleProcgenBlend] audio_bars: bars-layer render "
                    "failed (%s); falling back to the no-bars captioned blend", exc)
        return None


def _build_bars_caption_cmd(composite_mp4: Path, bars_mp4: Path, out_mp4: Path,
                            opacity: float, ffmpeg: str,
                            captions_ass_path: Optional[str],
                            source_dims: "Optional[tuple]") -> list[str]:
    """The SECOND, isolated pass: lighten the green-only bars layer onto the
    composite at ``opacity``, THEN burn captions (so captions stay ABOVE the
    bars). gbrp + colorchannelmixer(zero R+B) mirrors the proven green-only path;
    tpad black past the bars end so a shorter bars track never clamps the credits
    post-roll (BUG-410 pattern). Audio is ``-c:a copy`` (byte-identical)."""
    if source_dims:
        sw, sh = int(source_dims[0]), int(source_dims[1])
        conform = f"scale={sw}:{sh}"
    else:
        conform = "scale=-2:ih:force_original_aspect_ratio=decrease,crop=iw:ih"
    blend_label = "[vpre]" if captions_ass_path else "[v]"
    fc = (
        f"[0:v]format=gbrp[main];"
        f"[1:v]{conform},setpts=PTS-STARTPTS,setsar=1,"
        f"colorchannelmixer=rr=0:rg=0:rb=0:gr=0:gg=1:gb=0:br=0:bg=0:bb=0,"
        f"format=gbrp,tpad=stop_mode=add:color=black:stop_duration=3600[bars];"
        f"[main][bars]blend=all_mode=lighten:all_opacity={opacity:.3f}:shortest=1,"
        f"format=yuv420p{blend_label}"
    )
    if captions_ass_path:
        ass_name, _cwd = _ass_filter_arg(captions_ass_path)
        fc += f";[vpre]ass={ass_name}[v]"
    return [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(composite_mp4),
        "-i", str(bars_mp4),
        "-filter_complex", fc,
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
    """BUG-LOCAL-030 Phase B: final procgen visual blend.

    Takes the composite output (1920x1080 HuMo + LTX content with
    black HuMo pillarbox bars from Phase A simple-pillarbox composite)
    and overlays the 1920x1080 native procgen render on top via
    ffmpeg -filter_complex blend. Audio passes through with ``-c:a copy``
    so the C7 byte-identity guarantee from per-clip-mux holds end-to-end.

    The class name is historical: it read a standalone RTXUpscale stage
    until queue item 8 (2026-08-08) ripped that node. Its input is now
    SilentComposite's output directly. See the module docstring.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_mp4_path": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": (
                        "Path to the SilentComposite output mp4 "
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
                "blend_mode": (_BLEND_MODE_CHOICES, {
                    "default": _DEFAULT_BLEND_MODE,
                    "tooltip": "Compositing operator for the procgen overlay "
                               "onto the upscaled master (screen lightens, "
                               "overlay boosts contrast, normal replaces). "
                               "Works with blend_opacity; the shipped pair is "
                               "the qualified look.",
                }),
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
                    "tooltip": "DEPRECATED and IGNORED (2026-09-04). A workflow value cannot name the binary this pack runs -- it arrives over an unauthenticated /prompt request. Set the OTR_FFMPEG environment variable to pin a build.",
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
                # BUG-LOCAL-104 / BUG-LOCAL-105: this widget MUST stay LAST
                # in the optional dict. ComfyUI parses widgets_values
                # positionally; inserting a new BOOLEAN earlier in the dict
                # caused saved workflows to read their old position-N INT
                # value into this slot (bool(18) -> True), accidentally
                # enabling the green-only path while ALSO shifting the
                # crush threshold to its default. Always append new
                # widgets at the end (BUG-LOCAL-097 rule restated).
                "green_only_overlay": ("BOOLEAN", {
                    "default": _DEFAULT_GREEN_ONLY,
                    "tooltip": (
                        "BUG-LOCAL-104/105/106: when True, zero the procgen R "
                        "and B channels (colorchannelmixer) BEFORE the "
                        "blend so only the procgen G channel ever "
                        "contributes. Filter chain also pins both inputs "
                        "to gbrp planar RGB before blend and back to "
                        "yuv420p after, so the per-channel math runs in "
                        "RGB instead of YUV (where 'lighten' on Y/U/V "
                        "creates color-shifting garbage that swallows a "
                        "1%-sparse green wireframe). Source R/B pass "
                        "through untouched (no color lift); source G gets "
                        "boosted only where the procgen has green "
                        "wireframe pixels. Pair with blend_mode='screen' "
                        "or 'addition' -- 'lighten' will turn the "
                        "wireframe pure white over fully-saturated "
                        "magenta source pixels because max(255,0,255) vs "
                        "(0,255,0) collapses to (255,255,255)."
                    ),
                }),
                # SDH open captions REMOVED from node 93 (2026-07-04 widget-audit
                # Batch 3): caption ownership migrated to node 86 OTR_CaptionBurn,
                # which now sits AFTER this blend (chain 84 -> 93 -> 86 -> 95 -> 85).
                # The blend no longer burns captions -- see blend() where
                # captions_ass_path is pinned None. widgets_values is positional
                # (BUG-LOCAL-097): the two tail caption widgets were removed
                # together and the JSON widgets_values (13 -> 11) + the scopes
                # input index (11 -> 9) were updated in the SAME commit.
                # §4D scene-aware scopes (APPENDED LAST per BUG-LOCAL-097: a new
                # input only ever goes at the end -- widgets_values is
                # positional). Path to OTR_SceneAwareScopes' scopes_only.mp4.
                # When provided, the blend switches to the 3-input gbrp
                # double-blend (procgen screen, then scopes lighten). Empty ->
                # the unchanged single-procgen-blend path.
                "scopes_mp4_path": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": (
                        "Optional path to OTR_SceneAwareScopes' scopes_only.mp4. "
                        "Present -> procgen is screened on, then the scene-aware "
                        "scopes are lightened on top (3-input gbrp blend). "
                        "Empty -> the standard single-procgen blend (unchanged)."
                    ),
                }),
                # BUG 4 (2026-06-20 operator): the ALWAYS-ON audio-reactive bottom
                # bars overlay -- a SEPARATE layer, NOT the scene-aware floor.
                # APPENDED LAST per BUG-LOCAL-097 (widgets_values is positional --
                # only ever append; the JSON node 94->93 widgets_values gets one new
                # 'bottom' value at the end, same commit).
                "audio_bars": (["bottom", "off"], {
                    "default": "bottom",
                    "tooltip": (
                        "ALWAYS-ON bottom audio-reactive green bars overlay, "
                        "DEFAULT ON. 'bottom' paints a green frequency strip along "
                        "the bottom of EVERY frame (landscape AND portrait), driven "
                        "by the master audio, lighten-blended at 60%, seated above "
                        "the caption safe-area so captions stay ON TOP -- decoupled "
                        "from the scene-aware floor so it shows no matter what clip "
                        "is above/below. 'off' = byte-identical video to the "
                        "pre-overlay path. Audio is never touched (byte-identical)."
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
        shadow_crush_threshold: int = _DEFAULT_SHADOW_CRUSH,
        green_only_overlay: bool = _DEFAULT_GREEN_ONLY,
        scopes_mp4_path: str = "",
        audio_bars: str = "bottom",
    ):
        report_lines: list[str] = []
        # B1 (2026-09-04): the widget is UNTRUSTED /prompt input, not
        # operator intent. Discarded HERE, at the node boundary, so no
        # helper underneath can be handed it.
        try:
            from ._otr_shared.ffmpeg import widget_ffmpeg_is_ignored
        except ImportError:  # pragma: no cover -- flat (sys.path) load
            from _otr_shared.ffmpeg import widget_ffmpeg_is_ignored  # type: ignore
        ffmpeg = widget_ffmpeg_is_ignored(ffmpeg, "OTR_PostUpscaleProcgenBlend")
        ffmpeg = _ffmpeg_bin(ffmpeg)
        src = Path(source_mp4_path).resolve() if source_mp4_path else None
        pgn = Path(procgen_mp4_path).resolve() if procgen_mp4_path else None
        scp = Path(scopes_mp4_path).resolve() if scopes_mp4_path else None
        if scp is not None and not scp.is_file():
            log.warning("[PostUpscaleProcgenBlend] scopes_mp4 %r not a file; "
                        "ignoring (single-procgen blend)", scopes_mp4_path)
            scp = None

        if src is None or not src.is_file():
            msg = (
                f"PostUpscaleProcgenBlend: source mp4 missing or not a file: "
                f"{source_mp4_path!r}"
            )
            log.warning("[PostUpscaleProcgenBlend] %s", msg)
            return ("", msg)

        # LAYOUT CHANGE (operator directive 2026-06-09): otr/obs holds ONLY
        # the muxed FINAL deliverable (published by OTR_MasterAudioMux). The
        # blend output is now a SILENT intermediate consumed by the mux, so it
        # lands NEXT TO its source inside the per-episode folder
        # (otr/episodes/<ep>/) -- never in obs. (Pre-refactor the blend WAS
        # the terminal deliverable and wrote to otr_obs_dir() directly.)
        output_path = src.parent / f"{src.stem}{out_suffix}{src.suffix}"

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

        # A MISSING TOOL IS NOT A CRASH (2026-09-04). `_ffmpeg_bin` now
        # answers "" when this box has no ffmpeg, and every path below spawns
        # one. Without this, "" reaches otr_proc.run(["", ...]) and the process
        # owner raises ExecutableNotAllowed -- a RuntimeError, which _run_blend's
        # CalledProcessError handler does not catch, so the node dies with an
        # allowlist message instead of "ffmpeg not found".
        #
        # Placed AFTER the bypass and procgen-missing exits above on purpose:
        # both of those only copy, so they must keep working on a box with no
        # ffmpeg at all rather than start requiring one.
        if not ffmpeg:
            try:
                shutil.copy2(src, output_path)
                msg = (
                    f"PostUpscaleProcgenBlend: ffmpeg not found "
                    f"(OTR_FFMPEG / PATH); skipped blend, copied "
                    f"source -> output ({src.name} -> {output_path.name})"
                )
                log.warning("[PostUpscaleProcgenBlend] %s", msg)
                return (str(output_path), msg)
            except Exception as exc:  # noqa: BLE001
                msg = (
                    f"PostUpscaleProcgenBlend: ffmpeg not found AND copy "
                    f"fallback failed: {exc}"
                )
                log.warning("[PostUpscaleProcgenBlend] %s", msg)
                return ("", msg)

        # BUG-LOCAL-030 long-form hardening (2026-05-03 EVENING, post
        # round-robin risk-#10 review): phase barrier handoff from the
        # composite. Reclaim DRAM/VRAM that PyTorch may still be
        # holding from rendering, then check the canary before kicking
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

        # SDH open captions were MIGRATED OUT of this blend to node 86
        # OTR_CaptionBurn (2026-07-04 widget-audit Batch 3): CaptionBurn now runs
        # AFTER this node (chain 84 -> 93 -> 86 -> 95 -> 85) and is the single
        # caption owner. The blend therefore NEVER burns captions -- captions_ass_
        # path stays None so the shared ffmpeg cmd builders take their no-caption
        # path (bars two-pass and everything else byte-identical to the prior
        # no-caption render).
        captions_ass_path = None

        src_dims = _probe_dims(src, ffmpeg=ffmpeg)
        if src_dims:
            report_lines.append("conform: procgen scaled to source %dx%d"
                                % src_dims)
        if scp is not None:
            report_lines.append(f"scopes: 3-input blend (+{scp.name})")

        # BUG 4: the ALWAYS-ON audio bars are a SEPARATE second pass so the
        # (historically fragile) procgen/scopes blend is untouched. When ON the
        # main blend produces the composite, then a second pass lighten-blends the
        # bars layer on top. 'off' -> the legacy single-pass blend, byte-identical
        # to today. NOTE: captions_ass_path is pinned None above (caption ownership
        # migrated to node 86 OTR_CaptionBurn, upstream), so neither pass burns
        # captions -- the bars cmd builders take their no-caption path.
        want_bars = (str(audio_bars or "off").lower() == "bottom"
                     and pgn is not None)
        run_cwd = str(Path(captions_ass_path).parent) if captions_ass_path else None
        _main_captions = None if want_bars else captions_ass_path
        _main_out = (
            output_path.with_name(output_path.stem + "__nobars_tmp" + output_path.suffix)
            if want_bars else output_path)

        def _run_blend(out_mp4, caps):
            _cmd = _build_blend_cmd(
                source_mp4=src, procgen_mp4=pgn, out_mp4=out_mp4,
                blend_mode=blend_mode, blend_opacity=float(blend_opacity),
                ffmpeg=ffmpeg,
                shadow_crush_threshold=int(shadow_crush_threshold),
                green_only_overlay=bool(green_only_overlay),
                captions_ass_path=caps, source_dims=src_dims, scopes_mp4=scp)
            otr_proc.run(_cmd, check=True, stdout=otr_proc.PIPE,
                           stderr=otr_proc.PIPE,
                           cwd=(str(Path(caps).parent) if caps else None))

        # -- main blend (procgen [+ scopes]); captions deferred when bars are on --
        try:
            _run_blend(_main_out, _main_captions)
        except otr_proc.CalledProcessError as exc:
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

        # -- BUG 4 second pass: bars layer (lighten @60%) then captions on top --
        if want_bars:
            _bw, _bh = (src_dims if src_dims else (1920, 1080))
            _fps = _probe_fps(src, ffmpeg)
            bars_tmp = output_path.with_name(
                output_path.stem + "__bars_tmp" + output_path.suffix)
            bars_path = _render_bars_only_mp4(src, bars_tmp, _bw, _bh, _fps, ffmpeg)
            _bars_ok = False
            if bars_path is not None:
                try:
                    otr_proc.run(
                        _build_bars_caption_cmd(
                            _main_out, bars_path, output_path, _BARS_OPACITY,
                            ffmpeg, captions_ass_path, src_dims),
                        check=True, stdout=otr_proc.PIPE, stderr=otr_proc.PIPE,
                        cwd=run_cwd)
                    _bars_ok = True
                    report_lines.append(
                        "audio_bars: bottom overlay (lighten @%.2f)"
                        % _BARS_OPACITY)
                except otr_proc.CalledProcessError as exc:
                    _se = exc.stderr.decode("utf-8", "replace") if exc.stderr else ""
                    log.warning("[PostUpscaleProcgenBlend] audio_bars: overlay pass "
                                "failed (%s); re-running the normal blend",
                                _se[:200])
            if not _bars_ok:
                # LOUD degrade: ship the NORMAL blend so the deliverable still
                # exists (captions are burned upstream at node 86, not here).
                try:
                    _run_blend(output_path, captions_ass_path)
                    report_lines.append("audio_bars: SKIPPED (bars layer "
                                        "unavailable); shipped normal blend")
                except otr_proc.CalledProcessError:
                    try:
                        shutil.copy2(src, output_path)
                    except Exception:  # noqa: BLE001
                        pass
            for _t in (_main_out, bars_tmp):
                try:
                    if Path(_t) != output_path and Path(_t).is_file():
                        os.remove(_t)
                except OSError:
                    pass

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
        # composite written earlier. Best-effort; never raises.
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
