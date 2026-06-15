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
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Ensure sibling node modules (e.g. _otr_paths) resolve when this file is
# loaded by ComfyUI's custom-node loader. Mirrors the pattern used in
# rtx_upscale.py so otr_obs_dir() is reachable as the canonical write
# target for the broadcast-ready blended mp4.
_NODES_DIR = os.path.dirname(os.path.abspath(__file__))
if _NODES_DIR not in sys.path:
    sys.path.insert(0, _NODES_DIR)

from _otr_paths import otr_audio_dir, otr_obs_dir  # noqa: E402

try:
    from _otr_captions import build_ass_from_ledger  # noqa: E402
except Exception:  # pragma: no cover -- captions optional; never block a render
    build_ass_from_ledger = None  # type: ignore

log = logging.getLogger(__name__)


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


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
    return (p.name, str(p.parent))


def _resolve_captions_ass(source_mp4: Path, style: str = "") -> tuple[Optional[str], str]:
    """Build the SDH .ass for this episode. Returns (ass_path, message).

    Best-effort: any failure returns (None, reason) and the caller proceeds
    with an un-captioned blend -- captions never block the deliverable.
    ``style`` is the caller-resolved preset (widget value, env-overridden);
    empty falls back to sdh_standard. Margin comes from OTR_CAPTION_MARGIN_V
    (default 90). The episode id is the source stem; the ledger is resolved
    via otr_audio_dir(), falling back to the in-flight ledger singleton.
    """
    if build_ass_from_ledger is None:
        return (None, "caption builder unavailable (_otr_captions import failed)")
    # Episode id = the source stem MINUS the post-chain suffixes (capstone
    # production fix 2026-06-10): the blend's source is now the per-episode
    # "<slug>_silent.mp4" (or "..._procgen_blended.mp4"), so the raw stem
    # resolved "<slug>_silent_ledger.json" -- which never exists -- and
    # captions only worked when the SAME-process in-flight singleton happened
    # to point at this episode. A ComfyUI Desktop render misses entirely.
    episode_id = source_mp4.stem
    for _suffix in ("_procgen_blended", "_silent"):
        if episode_id.endswith(_suffix):
            episode_id = episode_id[: -len(_suffix)]
    ledger = otr_audio_dir(episode_id) / f"{episode_id}_ledger.json"
    if not ledger.is_file():
        # Layout-relative fallback: the source lives INSIDE the episode folder
        # (otr/episodes/<slug>/...), so its sibling audio/ dir carries the
        # ledger regardless of which output root the server pinned.
        sib = source_mp4.parent / "audio"
        cands = sorted(sib.glob("*_ledger.json")) if sib.is_dir() else []
        if cands:
            ledger = cands[0]
    if not ledger.is_file():
        try:
            from . import _otr_ledger as _OTRL  # type: ignore
        except ImportError:  # pragma: no cover
            import _otr_ledger as _OTRL  # type: ignore
        alt = _OTRL.in_flight_ledger_path()
        if alt is not None and Path(alt).is_file():
            ledger = Path(alt)
        else:
            return (None, f"ledger not found for {episode_id}")
    style = (style or "").strip() or "sdh_standard"
    try:
        margin_v = int(os.environ.get("OTR_CAPTION_MARGIN_V", "90") or 90)
    except ValueError:
        margin_v = 90
    out_path, report = build_ass_from_ledger(ledger, style=style, margin_v=margin_v)
    return (out_path, report)

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
# SDH open-caption widgets (P4). Clean master is the default -> captions OFF.
_CAPTION_STYLE_CHOICES = ["sdh_standard", "otr_crt"]
_DEFAULT_CAPTION_STYLE = "sdh_standard"
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


def _probe_dims(path: Path, ffmpeg: str = "ffmpeg") -> "Optional[tuple]":
    """(width, height) of the first video stream via ffprobe, or None.
    Best-effort: a probe failure returns None and the blend falls back to the
    legacy self-referential scale (the pre-2026-06-09 behavior)."""
    ffprobe = "ffprobe"
    if ffmpeg and ffmpeg != "ffmpeg":
        cand = str(Path(ffmpeg).with_name("ffprobe.exe"))
        if Path(cand).is_file():
            ffprobe = cand
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0",
             str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        w, h = out.split("x")[:2]
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
    # worked when the source was already 1080p (post-RTXUpscale). The blend
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
                # SDH open captions (P4). APPENDED AT THE END per BUG-LOCAL-097
                # (widgets_values is positional -- only ever append). These two
                # are now the tail; do not insert anything after them without
                # also extending the workflow JSON widgets_values + validator.
                "burn_captions": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Burn SDH open captions into the final 1080p output. "
                        "Default ON = accessible delivery. Turn OFF for a clean "
                        "master (festival/archival). When ON: "
                        "white dialogue on an opaque box, bold-white speaker "
                        "labels with a subtle pastel outline (color is never the "
                        "speaker cue -- color-blind safe), synced from the ledger "
                        "line timing. Audio stays -c:a copy (byte-identical). "
                        "Env OTR_BURN_CAPTIONS=1 also forces this on."
                    ),
                }),
                "caption_style": (_CAPTION_STYLE_CHOICES, {
                    "default": _DEFAULT_CAPTION_STYLE,
                    "tooltip": (
                        "Caption style preset. 'sdh_standard' (default): Arial, "
                        "white-on-black-box, accessibility master. 'otr_crt': "
                        "green-CRT themed variant. Env OTR_CAPTION_STYLE overrides."
                    ),
                }),
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
        burn_captions: bool = True,
        caption_style: str = _DEFAULT_CAPTION_STYLE,
        scopes_mp4_path: str = "",
    ):
        report_lines: list[str] = []
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

        # SDH open captions (P4): enabled by the burn_captions widget OR the
        # OTR_BURN_CAPTIONS env (env forces on for headless delivery). Clean
        # master is the default (OFF). Style precedence: OTR_CAPTION_STYLE env
        # overrides the caption_style widget. Best-effort -- a caption failure
        # NEVER blocks the blend.
        captions_ass_path = None
        if bool(burn_captions) or _env_truthy("OTR_BURN_CAPTIONS"):
            style = str(os.environ.get("OTR_CAPTION_STYLE", "") or caption_style).strip()
            captions_ass_path, cap_msg = _resolve_captions_ass(src, style=style)
            log.info("[PostUpscaleProcgenBlend] captions: %s", cap_msg)
            report_lines.append(f"captions: {cap_msg}")

        src_dims = _probe_dims(src, ffmpeg=ffmpeg)
        if src_dims:
            report_lines.append("conform: procgen scaled to source %dx%d"
                                % src_dims)
        cmd = _build_blend_cmd(
            source_mp4=src, procgen_mp4=pgn, out_mp4=output_path,
            blend_mode=blend_mode, blend_opacity=float(blend_opacity),
            ffmpeg=ffmpeg,
            shadow_crush_threshold=int(shadow_crush_threshold),
            green_only_overlay=bool(green_only_overlay),
            captions_ass_path=captions_ass_path,
            source_dims=src_dims,
            scopes_mp4=scp,
        )
        if scp is not None:
            report_lines.append(f"scopes: 3-input blend (+{scp.name})")
        # Run with cwd = the .ass folder so the libass filter resolves the
        # subtitle by basename (sidesteps Windows drive-colon filtergraph
        # escaping). All input/output paths in cmd are absolute.
        run_cwd = str(Path(captions_ass_path).parent) if captions_ass_path else None
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           cwd=run_cwd)
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
