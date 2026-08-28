"""OTR_SilentComposite -- normalize render output into ONE always-silent video (A-S3 / CW-4).

Produces the canonical, ALWAYS-SILENT composite the terminal ``OTR_MasterAudioMux``
then muxes the frozen master audio onto. For M1 (first watchable episode) the base
is the radio-floor video (``OTR_SignalLostVideo``); later windows composite real
engine clips here too. Whatever the source, the output is guaranteed:

* **silent** -- ``-an`` strips any audio (invariant V-1: only MasterAudioMux adds audio),
* **CFR** at the canonical fps (``fps`` filter + ``-vsync cfr``; no VFR drift),
* **yuv420p**, even (mod-2) dimensions, padded to the canonical canvas,
* **bt709 IDENTITY-tagged** -- untagged input is TAGGED bt709, NEVER matrix-converted
  (no silent BT.601->709 shift); the scale/pad filters do not touch the color matrix,
* duration preserved from the (audio-derived) base, so the mux duration assert passes.

Pure ffmpeg, cold-import clean (stdlib only) -- no torch, no CUDA residency.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import logging

try:  # ComfyUI loads these node modules flat as well as packaged
    from ._otr_shared import ffprobe as _ffp
except ImportError:  # pragma: no cover -- flat (sys.path) test import
    from _otr_shared import ffprobe as _ffp  # type: ignore

log = logging.getLogger("OTR")

# Queue item 8 (2026-08-08): upscale-engine namespace for per-clip model
# enhancement. Imported at MODULE TOP so the roster is populated by the time
# INPUT_TYPES first runs. Python's package-loading rule means
# `_otr_upscale_engines/__init__.py` runs first, registering every adapter,
# so the dropdown built from `_upscale_engine_names()` is complete on first
# call. Import guarded so a namespace-import failure never breaks
# SilentComposite (roster audit logs the details).
try:
    from ._otr_upscale_engines.registry import (
        all_engine_names as _upscale_engine_names,
        get_engine as _get_upscale_engine,
        assert_usable as _assert_upscale_usable,
    )
    from ._otr_upscale_engines._resolve import (
        resolve_device as _resolve_upscale_device,
    )
except Exception as _upscale_import_err:  # noqa: BLE001
    # Fail-open: if the package fails to import, fall back to a hardcoded
    # ["off"] roster so the composite still runs today's byte-identical path.
    log.warning(
        "[OTR_SilentComposite] _otr_upscale_engines import failed (%s); "
        "upscale_engine dropdown falls back to ['off']",
        _upscale_import_err)
    _upscale_engine_names = lambda: ["off"]  # noqa: E731

    def _assert_upscale_usable(name, role):
        if name != "off":
            raise RuntimeError(
                f"upscale engine {name!r} unavailable: namespace import failed")
        return name

    def _get_upscale_engine(name):
        class _NullOff:
            name = "off"

            def model_fingerprint_parts(self):
                # The real engines declare this; the stub must too, or
                # IS_CHANGED would raise on the exact degraded path this
                # fallback exists to survive and return nan on every
                # evaluation -- breaking caching for plain `off` renders.
                return ()
        return _NullOff()

    def _resolve_upscale_device(v):
        import torch
        return torch.device("cpu")


#: Failure identities already reported by IS_CHANGED's fingerprint step.
#: IS_CHANGED runs on EVERY prompt evaluation, so an unguarded log line would
#: flood; but staying silent is worse -- a mis-typed engine id in a profile
#: would become a permanent invisible cache miss on a node that takes minutes.
#: Keyed on (engine_id, exception class name): small, stable, and free of the
#: formatted message, so it cannot grow with varying paths. Bounded per Bug
#: Bible 06.04 (an unbounded module-level cache is itself a named defect).
#: No lock: two threads may both emit the line, and a duplicate log entry is a
#: far cheaper outcome than a lock on a per-prompt code path.
_FINGERPRINT_LOG_ONCE_CACHE: set = set()
_FINGERPRINT_LOG_ONCE_MAX = 64


def _log_fingerprint_failure_once(engine_id, exc) -> None:
    """Emit one warning per distinct (engine, error type); never raise."""
    key = (str(engine_id), type(exc).__name__)
    if key in _FINGERPRINT_LOG_ONCE_CACHE:
        return
    if len(_FINGERPRINT_LOG_ONCE_CACHE) >= _FINGERPRINT_LOG_ONCE_MAX:
        _FINGERPRINT_LOG_ONCE_CACHE.clear()
    _FINGERPRINT_LOG_ONCE_CACHE.add(key)
    # %r on the free-form values: an exception string can carry newlines, and
    # a raw one would forge log records (Bug Bible 12.83).
    # Wording is deliberately "while it persists", not "until it is fixed":
    # a checkpoint deleted between the resolver's is_file() and the
    # fingerprint's os.stat raises once and then settles into the stable
    # absent marker on the next evaluation, with nothing to fix.
    log.warning(
        "[OTR_SilentComposite] upscale model fingerprint unavailable for "
        "engine %r (%s: %r); IS_CHANGED falls open to nan, so this node "
        "re-executes every run while that persists.",
        str(engine_id), type(exc).__name__, str(exc))


def _ffmpeg_bin(ffmpeg: str) -> str:
    """The ffmpeg this box should run, or ``""`` when it has none.

    HONOURS ``OTR_FFMPEG`` BEFORE PATH (2026-08-28). It did not, and that was
    the mirror image of a bug the pack had already fixed once: the shared
    ``_otr_shared/ffprobe.py`` resolver exists because only `otr_credits_roll`
    honoured ``OTR_FFPROBE`` while every other caller trusted PATH. That
    consolidation was scoped to the PROBE; the ENCODER kept the same hole here,
    in `otr_caption_burn`, `otr_master_audio_mux` and `otr_silent_composite` --
    which are the caption burn, the terminal audio mux and the silent-video
    normalize, i.e. the LAST three stages of an episode.

    So on a box where ffmpeg is reachable only through ``OTR_FFMPEG`` -- the
    AMD/Mac/alternate-box case the variant workflows exist for -- every earlier
    stage would succeed (the video engines all honour the variable) and the
    episode would die at the end, having spent the whole render.

    The explicit widget argument still wins when it resolves: an operator who
    typed a path meant it. The env var is consulted only when the passed value
    does not resolve, and PATH remains the last resort.
    """
    if ffmpeg and (shutil.which(ffmpeg) or os.path.isfile(ffmpeg)):
        return ffmpeg
    cand = (os.environ.get("OTR_FFMPEG") or "").strip()
    if cand and (os.path.isfile(cand) or shutil.which(cand)):
        return cand
    return shutil.which("ffmpeg") or ""


def _ffprobe_bin() -> str:
    """The ffprobe this box should run, or ``""`` when it has none.

    THE POLICY IS THIS MODULE'S AND IT DOES NOT MOVE: every probe here answers
    ``-1`` / ``0.0`` on an absent tool and the composite carries on, because a
    finished episode is not thrown away over a missing diagnostic. Only the
    SEARCH is shared, which is how ``OTR_FFPROBE`` finally reaches the A/V-sync
    frame count.
    """
    return _ffp.resolve_ffprobe() or ""


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def count_audio_streams(path: str) -> int:
    fp = _ffprobe_bin()
    if not fp:
        return -1
    p = _run([fp, "-v", "error", "-select_streams", "a", "-show_entries",
              "stream=index", "-of", "csv=p=0", path])
    return len([ln for ln in (p.stdout or "").splitlines() if ln.strip()])


def probe_video(path: str) -> dict:
    """w,h,pix_fmt,avg_frame_rate of v:0 (for the CFR / color / mod-2 asserts)."""
    fp = _ffprobe_bin()
    if not fp:
        return {}
    p = _run([fp, "-v", "error", "-select_streams", "v:0", "-show_entries",
              "stream=width,height,pix_fmt,avg_frame_rate,r_frame_rate",
              "-of", "default=noprint_wrappers=1", path])
    out: dict = {}
    for line in (p.stdout or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _even(n: int) -> int:
    n = int(n)
    return n if n % 2 == 0 else n - 1


#: Composite sharpening amount (ffmpeg ``unsharp`` luma_amount). The LTX-AV
#: quality bakeoff (2026-06-27) found the native render is SOFT once the
#: composite upscales it to the 1472x832 canvas; a lanczos resample + a light
#: unsharp on the clip/still paths recovers ~+9% Laplacian sharpness at ZERO GPU
#: cost. Env OTR_COMPOSITE_UNSHARP_AMOUNT (default 0.4; the bakeoff also exposed
#: 0.8 as the heavier option -- an operator eyeball item for face halos). A bad
#: value falls back to the default (cosmetic knob -- not fail-loud).
_UNSHARP_AMOUNT_DEFAULT = 0.4


def _unsharp_amount() -> float:
    try:
        return float(os.environ.get("OTR_COMPOSITE_UNSHARP_AMOUNT",
                                    _UNSHARP_AMOUNT_DEFAULT))
    except (TypeError, ValueError):
        return _UNSHARP_AMOUNT_DEFAULT


#: THE DECLARED DELIVERY MODES. A lane DECLARES how its rows must be enlarged
#: and the composite obeys the declaration -- it never asks which engine made a
#: row. ``None`` is the legacy state and means "the historical real-clip or
#: floor/gap behaviour", which is why absence has to stay distinguishable from
#: every explicit value.
DELIVERY_SCALE_MODE_CLEAN = "lanczos_clean_full_frame"
DELIVERY_SCALE_MODES = frozenset({DELIVERY_SCALE_MODE_CLEAN})


def _has_model_eligible_clips(manifest):
    """PURE: does this manifest contain a row the upscale MODEL could act on?

    True only for a real on-disk video clip (``exists``, a non-empty path, not a
    directory handoff) whose ``delivery_scale_mode`` is ``None`` -- the exact
    legacy state. A row that DECLARES an explicit delivery mode has told us how
    it must be enlarged, and every declared mode today is model-ineligible.

    THIS IS WHAT LETS A PROMPT-OWNED EPISODE IGNORE A STALE ESRGAN SELECTION.
    Without it, an all-Ghost manifest would still resolve, assert and LOAD a
    model none of its rows can use -- and a load-time failure (missing weights,
    no spandrel, CUDA OOM) would kill a render that never needed the model at
    all. A MIXED manifest still loads it once, for the legacy rows that earn it.
    """
    for row in ((manifest or {}).get("clips") or []):
        if not isinstance(row, dict):
            continue
        if not row.get("exists"):
            continue
        if not str(row.get("path") or "").strip():
            continue
        if str(row.get("type") or "") == "directory":
            continue
        if row.get("delivery_scale_mode") is not None:
            continue
        return True
    return False


def _scale_filter(w, h, fps, *, sharpen=None, pad=True, in_label=None,
                  out_label=None, pre="", post="", mode=None):
    """The ONE shared composite scale chain (LTX-AV quality wire, 2026-06-27).

    Emits, IN ORDER: ``scale=w:h:force_original_aspect_ratio=decrease`` (with
    ``:flags=lanczos`` appended ONLY when ``sharpen``), then -- iff ``sharpen`` --
    ``unsharp=5:5:<amt>:5:5:0.0`` (``<amt>`` = ``OTR_COMPOSITE_UNSHARP_AMOUNT``,
    default 0.4), then -- iff ``pad`` -- ``pad=w:h:(ow-iw)/2:(oh-ih)/2:color=black``,
    then ``fps=fps``.

    ``sharpen=True``  -> lanczos + unsharp (the soft-native fix; clip + real
    still-plate + RGBA foreground paths).
    ``sharpen=False`` -> the legacy bilinear scale,pad,fps chain BYTE-IDENTICAL
    (the procgen floor + black gap-fill + the silent-canonical normalize -- not
    sharpened, since the floor/credits roll must not be touched).
    ``pad=False``     -> NO pad (the straight-RGBA foreground: ``pad ...
    color=black`` would paint opaque borders over the alpha edges -> destroy the
    matte). The overlay re-centers the fg, so the pad is unnecessary there.

    Returns a plain filter chain usable directly as ``-vf`` when ``in_label`` /
    ``out_label`` are ``None``; a labeled ``[in]...[out]`` segment for a
    ``-filter_complex`` graph when BOTH labels are given. ``pre`` is inserted
    immediately after the input label (e.g. ``format=rgba,`` for the fg, or the
    floor ``trim=...,setpts=...,`` prefix); ``post`` is appended before the output
    label (e.g. the floor ``,tpad=stop_mode=clone:...`` hold).

    ``mode`` (Ghost Signal, 2026-08-22) is a lane's DECLARED enlargement.
    ``mode=None`` is every historical caller and every emitted filter string is
    byte-identical to before. ``mode`` together with an explicitly conflicting
    ``sharpen`` FAILS LOUD rather than silently letting one win: they are two
    answers to the same question and a render should not have to guess.

    ``lanczos_clean_full_frame`` replaces only the SPATIAL portion -- it emits
    ``scale=<w>:<h>:flags=lanczos,fps=<fps>`` with no aspect pad, no crop and no
    unsharp. ``pre`` and ``post`` are preserved verbatim, so ``_seg_vf`` still
    contributes its ``trim,setpts`` prefix and its ``tpad`` safety suffix."""
    if mode is not None:
        if mode not in DELIVERY_SCALE_MODES:
            raise ValueError(
                "OTR_SilentComposite: unknown delivery_scale_mode %r (known: "
                "%s). An unrecognised explicit mode fails loud -- it must never "
                "fall through to the legacy chain, because that would silently "
                "deliver a lane something other than what it declared."
                % (mode, sorted(DELIVERY_SCALE_MODES)))
        if sharpen is not None:
            raise ValueError(
                "OTR_SilentComposite: delivery_scale_mode=%r was passed "
                "together with sharpen=%r. Those are two conflicting answers to "
                "the same question; pass one." % (mode, sharpen))
        chain = pre + ",".join([
            "scale=%d:%d:flags=lanczos" % (int(w), int(h)),
            "fps=%d" % int(fps),
        ]) + post
        if in_label is not None and out_label is not None:
            return "[%s]%s[%s]" % (in_label, chain, out_label)
        return chain
    if sharpen is None:
        raise ValueError(
            "OTR_SilentComposite: _scale_filter needs either sharpen= or "
            "mode=; neither was given.")
    scale = "scale=%d:%d:force_original_aspect_ratio=decrease" % (int(w), int(h))
    parts = [scale]
    if sharpen:
        parts[0] = scale + ":flags=lanczos"
        parts.append("unsharp=5:5:%g:5:5:0.0" % _unsharp_amount())
    if pad:
        parts.append("pad=%d:%d:(ow-iw)/2:(oh-ih)/2:color=black"
                     % (int(w), int(h)))
    parts.append("fps=%d" % int(fps))
    chain = pre + ",".join(parts) + post
    if in_label is not None and out_label is not None:
        return "[%s]%s[%s]" % (in_label, chain, out_label)
    return chain


def normalize_to_silent_canonical(in_path: str, out_path: str, *, w: int = 1472,
                                  h: int = 832, fps: int = 25, ffmpeg: str = "ffmpeg"):
    """Re-encode ``in_path`` into the canonical ALWAYS-SILENT clip; FAIL CLOSED.

    Returns ``(out_path, report)``; raises ``ValueError`` on a gate failure.
    """
    report: list = []
    fb = _ffmpeg_bin(ffmpeg)
    if not fb:
        raise ValueError(f"OTR_SilentComposite: ffmpeg not found ({ffmpeg!r})")
    if not os.path.isfile(in_path):
        raise ValueError(f"OTR_SilentComposite: input missing: {in_path!r}")
    w, h, fps = _even(w), _even(h), max(1, int(fps))
    # The normalize path is the procgen floor / single-base passthrough -> NOT
    # sharpened (sharpen=False keeps it byte-identical to the legacy chain).
    vf = _scale_filter(w, h, fps, sharpen=False)
    cmd = [
        fb, "-y", "-loglevel", "error",
        "-i", in_path,
        "-an",                                  # V-1: strip ALL audio
        "-vf", vf,
        "-vsync", "cfr",                        # constant frame rate (no VFR drift)
        "-pix_fmt", "yuv420p",
        # TAG bt709 (identity) -- do NOT matrix-convert an untagged source.
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        out_path,
    ]
    assert "-shortest" not in cmd, "V-2: -shortest must not appear in the composite"
    p = _run(cmd)
    if p.returncode != 0:
        raise ValueError(f"OTR_SilentComposite: ffmpeg failed :: {p.stderr.strip()[:300]}")

    # gate: the composite MUST be silent (V-1) + yuv420p + even dims.
    na = count_audio_streams(out_path)
    if na != 0:
        raise ValueError(f"OTR_SilentComposite: output has {na} audio stream(s); must be 0 (V-1)")
    info = probe_video(out_path)
    if info.get("pix_fmt") and info["pix_fmt"] != "yuv420p":
        raise ValueError(f"OTR_SilentComposite: pix_fmt {info['pix_fmt']} != yuv420p")
    try:
        ow, oh = int(info.get("width", w)), int(info.get("height", h))
        if ow % 2 or oh % 2:
            raise ValueError(f"OTR_SilentComposite: non-mod-2 output dims {ow}x{oh}")
    except ValueError:
        raise
    report.append(f"silent_canonical {info.get('width', w)}x{info.get('height', h)} "
                  f"yuv420p bt709 cfr@{fps} audio_streams=0 OK")
    return out_path, report


# --------------------------------------------------------------------------- #
# Per-beat assemble (Chunk C): a frame-accurate CFR timeline from a clip
# manifest. Frame counts ONLY (never seconds) -- the assembled frame count is
# asserted == the audio-derived budget (the pre-mux A/V sync guard). The frozen
# audio is never touched (every segment is encoded with -an).
# --------------------------------------------------------------------------- #
def count_video_frames(path: str) -> int:
    """Authoritative decoded frame count of v:0 (the assemble A/V-sync gate);
    -1 when ffprobe or the file is missing."""
    fp = _ffprobe_bin()
    if not fp or not os.path.isfile(path):
        return -1
    p = _run([fp, "-v", "error", "-select_streams", "v:0", "-count_frames",
              "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path])
    out = (p.stdout or "").strip().splitlines()
    if out and out[0].strip().isdigit():
        return int(out[0].strip())
    return -1


def _probe_duration(path):
    """Container duration in seconds (the master length the assembled video must
    match); 0.0 when unavailable."""
    fp = _ffprobe_bin()
    if not fp or not os.path.isfile(path):
        return 0.0
    p = _run([fp, "-v", "error", "-show_entries", "format=duration",
              "-of", "default=noprint_wrappers=1:nokey=1", path])
    out = (p.stdout or "").strip().splitlines()
    try:
        return float(out[0]) if out else 0.0
    except (ValueError, IndexError):
        return 0.0


def _probe_audio_duration(path):
    """FIRST AUDIO STREAM duration in seconds; 0.0 when absent/unavailable.

    The credits-tail cap (2026-06-10): the procgen base VIDEO runs ~20s past
    the master mix it carries (its own silent post-roll), so capping the
    assembled length at the base's AUDIO duration keeps the credits riding
    under the closing theme while honoring the terminal mux's v<=a gate."""
    fp = _ffprobe_bin()
    if not fp or not os.path.isfile(path):
        return 0.0
    p = _run([fp, "-v", "error", "-select_streams", "a:0",
              "-show_entries", "stream=duration",
              "-of", "default=noprint_wrappers=1:nokey=1", path])
    out = (p.stdout or "").strip().splitlines()
    try:
        return float(out[0]) if out else 0.0
    except (ValueError, IndexError):
        return 0.0


#: A real clip whose on-disk frames are below this fraction of its beat target is
#: a gross underrun -- the composite would hold the last frame for most of the beat
#: (the wan_ti2v 17/280 freeze). LOUD-warn so a future short-clip engine is caught,
#: not silently frozen. Env OTR_CLIP_UNDERRUN_FRAC (0 disables).
_CLIP_UNDERRUN_FRAC = 0.5


class ClipUnderrunsItsBeat(RuntimeError):
    """A real clip is SHORTER than the beat it must cover, at composite time.

    Terminal since 2026-08-02. This is the last place frames were reused to
    cover audio, and it survived the engine-layer mirror rip precisely because
    it lives in the assembler rather than in any adapter -- the mirror was
    deleted from ``wrapper_bridge``, the boomerang retired in ``eng_ltx_video``,
    and the composite went on stream-looping the same short clip.

    Its own docstring named the real fix and called itself the interim: "the
    real fix is phrase-chunking -- render the beat's correct duration so it
    never underruns -- tracked as a follow-up; this is the safe interim
    behavior." Coverage planning IS that follow-up and it is live, so the
    interim can go.
    """

    def __init__(self, shot_id, engine_id, real, target):
        self.shot_id, self.engine_id = shot_id, engine_id
        self.real, self.target = int(real), int(target)
        super().__init__(
            "beat %s (%s) rendered %d frame(s) but must cover %d -- a %d-frame "
            "shortfall. NO FILL: looping replays earlier frames against later "
            "audio and holding freezes the picture, and every second of audio "
            "gets ORIGINAL video. Fix the RENDER, not the timeline: let coverage "
            "planning split this beat, shorten the line, or raise the tier's cap."
            % (shot_id or "?", engine_id or "?", int(real), int(target),
               int(target) - int(real)))


def _should_loop_fill(row, target_n):
    """RETIRED 2026-08-02 -- always False. See :class:`ClipUnderrunsItsBeat`.

    Kept as a named no-op rather than deleted at its two call sites, so the
    retirement is visible where the decision used to be made and the underrun
    CHECK stays wired in one place.

    What it did: a real clip shorter than its beat was stream-looped to fill,
    except on ``audio_driven_face`` lanes, which held the last frame instead --
    because looping a lip-synced mouth desyncs it from its own audio, and a
    held frame at least fails honestly. Both branches covered audio with
    something other than original video, which is the rule this closes.

    Pure."""
    return False


def _warn_clip_underrun(row, target_n, *, will_loop=False):
    """RAISES on a real clip shorter than its beat. Was a LOUD warning.

    The warning existed under a no-loud-fail rule, and it was honest about what
    happened next: "the composite will HOLD the last frame for the rest of the
    beat. A motion engine should loop/ping-pong-extend to the target." Both of
    those remedies are now forbidden, which leaves nothing for a warning to warn
    ABOUT -- the shortfall has no legal outcome, so it is terminal.

    It also had a fractional threshold (``OTR_CLIP_UNDERRUN_FRAC``): only a clip
    below some percentage of its beat was worth mentioning. That made sense when
    the question was "is this bad enough to look at". It does not survive the
    rule, because a one-frame shortfall is still 40 ms of audio with no original
    video behind it. Any shortfall raises.

    A frame-DIRECTORY clip (the 3D alpha handoff) is exempt -- its frames are
    counted by its own dir encoder, not by this row's ``frame_count``.

    ``will_loop`` is vestigial and always False; the parameter stays so the two
    call sites keep their shape while ``_should_loop_fill`` is a named no-op.
    """
    real = int((row or {}).get("frame_count") or 0)
    tgt = int(target_n or 0)
    if real <= 0 or tgt <= 0 or real >= tgt:
        return
    path = str((row or {}).get("path") or "")
    if path and os.path.isdir(path):
        return
    raise ClipUnderrunsItsBeat(
        (row or {}).get("shot_id") or (row or {}).get("beat_id"),
        (row or {}).get("engine_id"), real, tgt)


def closing_window_authorizes_loop(manifest, cursor, target_total_frames):
    """Is the tail beginning at ``cursor`` PROVABLY the closing theme? PURE.

    THE ONLY THING IN THIS PIPELINE THAT MAY AUTHORIZE A LOOP. Operator ruling
    2026-08-06: *"there is no mirror or ping pong unless for credits"* -- and the
    one sanctioned reuse is the CLOSING-THEME BACKDROP, not the credits roll
    (``OTR_CreditsRoll`` freezes a frame and never loops).

    Before this, the tail block looped the last drama clip to fill ANY shortfall
    between the assembled body and the master length, whatever caused it. That
    is indistinguishable from the sanctioned case by construction: both are "the
    video ran out before the audio did". A named constant was rejected twice on
    exactly this point -- vocabulary without a manifest-backed window is not
    enforcement.

    The window comes from ``render_driver.closing_theme_frame_window``, derived
    from ledger rows carrying ``speaker_role == "music_close"`` AND
    ``start_s_space == "master_mix"``, and it is EMITTED ONLY when the manifest
    is positioned -- so its frames and this cursor are on one ruler.

    ``S <= cursor < target_total_frames <= E``: the tail must BEGIN at or after
    the theme starts, must have somewhere to go, and must END no later than the
    theme does. A tail that starts before the theme is drama time and gets
    floor/black; a tail running past the theme's end would loop over whatever
    follows it.

    **FALSE ON ANY DOUBT.** A missing window, a malformed one, an unreadable
    cursor -- every one of them returns False and the caller falls to
    floor/black. That asymmetry is deliberate: a black tail is a cosmetic
    disappointment, and an unauthorized loop is the defect this whole build
    exists to remove.
    """
    if not isinstance(manifest, dict):
        return False
    window = manifest.get("closing_theme_frame_window")
    if not isinstance(window, dict):
        return False
    start = window.get("start")
    end = window.get("end")
    # ``bool`` is an ``int``; a window of ``{"start": True}`` must not read as
    # frame 1. Same rejection ``acceptance.frame_count`` makes, same reason.
    if isinstance(start, bool) or isinstance(end, bool):
        return False
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    try:
        position = int(cursor)
        total = int(target_total_frames)
    except (TypeError, ValueError, OverflowError):
        return False
    return start <= position < total <= end


def plan_timeline_segments(manifest, *, floor_available=False, floor_frames=0,
                           target_total_frames=None, fps=None):
    """Pure: the frame-accurate per-beat segment plan from a clip manifest.

    POSITION mode (every beat carries ``start_s`` AND a ``target_total_frames``
    master length is given): place each beat at ``round(start_s*fps)`` and give
    the later beat ownership at its start. A visible slot ends at the earliest
    of its requested end, the next positioned start, or the master boundary.
    This preserves real gaps while removing duplicated video frames from audio
    crossfades. Floor/black fills head, inter-beat gaps, and tail so the assembled
    length == the master length (the pre-mux A/V-sync guard).

    SEQUENTIAL mode (no complete start_s set) retains the legacy behavior:
    concatenate full ``target_frame_count`` requests, then tail-fill when the
    target is longer. A real on-disk clip is used, else the floor (timeline-
    aligned slice in sequential mode) or black. Returns ``(segments,
    total_frames)``; each segment is ``{order, shot_id, source, path,
    src_start_frame, n_frames, engine_id}``. Frame counts only."""
    rows = [r for r in ((manifest or {}).get("clips") or [])
            if int((r or {}).get("target_frame_count") or 0) > 0]
    fps = int(fps or (manifest or {}).get("fps") or 25)
    ff = int(floor_frames or 0)
    gap_src = "floor" if floor_available else "black"
    segments = []
    cursor = 0

    def emit(source, path, n, src_start, shot_id=None, engine_id=None, loop=False,
             bg_still_path="", delivery_scale_mode=None):
        if int(n) <= 0:
            return
        segments.append({
            "order": len(segments), "shot_id": shot_id, "source": source,
            "path": path or "", "src_start_frame": int(src_start),
            "n_frames": int(n), "engine_id": engine_id, "loop": bool(loop),
            # THE LANE'S DECLARED ENLARGEMENT (2026-08-22), carried from the
            # manifest row to the encoder. None for floor/black rows, which have
            # no lane to declare anything, and None for every legacy clip.
            "delivery_scale_mode": delivery_scale_mode,
            # C1 (textured-hero 3D PoC): a per-clip generated background plate
            # for a directory-alpha clip (mesh_stage). Empty for every other
            # beat -> the legacy floor/black background is byte-identical.
            "bg_still_path": str(bg_still_path or "")})

    def _floor_aligned(start_cursor, n):
        """Timeline-aligned floor slice start (production restore 2026-06-10):
        gaps used to slice the procgen from frame 0 (the open repeated); the
        OLD episodes ran the procgen CONTINUOUSLY underneath, so head gaps show
        the radio open, inter-beat gaps the matching mid-roll, and the TAIL the
        rolling-credits post-roll. Clamped so n frames remain. (The procgen may
        run at a different fps than the composite -- ~4% drift at 24v25 -- an
        acceptable skew for a background/credits roll.)"""
        if gap_src != "floor":
            return 0
        return min(int(start_cursor), max(0, ff - int(n)))

    positioned = (target_total_frames is not None and rows
                  and all(r.get("start_s") is not None for r in rows))
    if positioned:
        # Stable manifest order breaks equal-start ties. The earlier row gets a
        # zero-width visible slot and the later row owns that frame boundary;
        # make the collapse loud below rather than silently serializing two
        # simultaneous audio rows into duplicate video time.
        ordered = sorted(
            enumerate(rows),
            key=lambda pair: (float(pair[1].get("start_s") or 0), pair[0]),
        )
        starts = [max(0, int(round(float(r.get("start_s") or 0) * fps)))
                  for _, r in ordered]
        timeline_end = max(0, int(target_total_frames))
        for index, (_, r) in enumerate(ordered):
            requested_n = int(r.get("target_frame_count") or 0)
            start_frame = min(starts[index], timeline_end)
            requested_end = start_frame + requested_n
            next_start = (min(starts[index + 1], timeline_end)
                          if index + 1 < len(starts) else timeline_end)
            slot_end = min(requested_end, next_start, timeline_end)
            n = max(0, slot_end - start_frame)
            if start_frame > cursor:                       # head / inter-beat gap
                gap_n = start_frame - cursor
                emit(gap_src, "", gap_n, _floor_aligned(cursor, gap_n))
                cursor = start_frame
            if n <= 0:
                log.warning(
                    "[OTR.composite] positioned beat %s has zero visible "
                    "frames at start=%d (next=%d requested=%d timeline=%d); "
                    "later/equal-start row owns the boundary",
                    r.get("shot_id") or r.get("beat_id"), start_frame,
                    next_start, requested_n, timeline_end)
                continue
            if r.get("exists") and r.get("path"):
                _fill = _should_loop_fill(r, n)
                _warn_clip_underrun(r, n, will_loop=_fill)
                emit("clip", r.get("path"), n, 0, r.get("shot_id"),
                     r.get("engine_id"), loop=_fill,
                     bg_still_path=r.get("bg_still_path"),
                     delivery_scale_mode=r.get("delivery_scale_mode"))
            else:
                emit(gap_src, "", n, _floor_aligned(cursor, n),
                     r.get("shot_id"), r.get("engine_id"))
            cursor = slot_end
    else:                                                  # SEQUENTIAL (legacy)
        for r in rows:
            n = int(r.get("target_frame_count") or 0)
            if r.get("exists") and r.get("path"):
                _fill = _should_loop_fill(r, n)
                _warn_clip_underrun(r, n, will_loop=_fill)
                emit("clip", r.get("path"), n, 0, r.get("shot_id"),
                     r.get("engine_id"), loop=_fill,
                     bg_still_path=r.get("bg_still_path"),
                     delivery_scale_mode=r.get("delivery_scale_mode"))
            elif floor_available:
                start = min(cursor, max(0, ff - n)) if ff else cursor
                emit("floor", "", n, start, r.get("shot_id"), r.get("engine_id"))
            else:
                emit("black", "", n, 0, r.get("shot_id"), r.get("engine_id"))
            cursor += n
    if target_total_frames is not None and int(target_total_frames) > cursor:
        # Tail to the master length (the closing-theme region, A/V-sync). Hold
        # the LAST drama clip on screen as the backdrop (operator 2026-06-17
        # "credits over the scene" look): a short clip loops, a long one plays
        # its head; fall back to the procgen END-slice / black when there is no
        # real clip. NOTE (credits enrichment 2026-07-03): this fills only to the
        # MASTER length; the credits POST-ROLL past the master (the old BUG-410
        # floor-extend) is GONE -- the unified credits roll is now a SILENT tail
        # appended LATE by OTR_CreditsRoll. It used to reproduce this
        # looped-last-clip backdrop by re-reading the clip manifest; since
        # 2026-07-29 (WIRE-W6) it freezes the final frame of the body video
        # THIS function assembles, so the manifest has exactly one reader.
        tail_n = int(target_total_frames) - cursor
        _clip_rows = [r for r in rows if r.get("exists") and r.get("path")]
        if positioned:
            _last_clip = (max(_clip_rows, key=lambda r: float(r.get("start_s") or 0))
                          if _clip_rows else None)
        else:
            _last_clip = _clip_rows[-1] if _clip_rows else None
        # THE LOOP IS NOW EARNED, NOT ASSUMED (no-mirror step 4, 2026-08-06).
        #
        # BLAST RADIUS, stated plainly: any episode whose tail is not PROVABLY
        # the closing-theme window now gets floor/black where it used to get a
        # looped last clip. That is a visible change to those episodes and it is
        # the point -- the old behaviour could not tell the operator's one
        # sanctioned reuse from a drama beat that simply ran short, because both
        # look like "the video ended before the audio did".
        if _last_clip is not None and closing_window_authorizes_loop(
                manifest, cursor, target_total_frames):
            # The closing tail REUSES the last real row, so it must reuse that
            # row's declared enlargement too or the shared projection is lossy.
            # A successful all-Ghost episode never reaches here -- its coverage
            # is exact and leaves no tail -- but a MIXED manifest can, and a
            # projection that is only correct for the lanes we happened to test
            # is not a projection.
            emit("clip", _last_clip.get("path"), tail_n, 0,
                 _last_clip.get("shot_id"), _last_clip.get("engine_id"), loop=True,
                 bg_still_path=_last_clip.get("bg_still_path"),
                 delivery_scale_mode=_last_clip.get("delivery_scale_mode"))
        else:
            if _last_clip is not None:
                # LOUD, because this is the case that used to loop and no longer
                # does. An operator watching a black tail needs to know it was a
                # refusal rather than a missing clip -- and needs the numbers, so
                # a genuinely mis-derived window can be diagnosed from the log
                # instead of by re-running the render.
                log.warning(
                    "[OTR.composite] TAIL NOT LOOPED: %d frame(s) from cursor "
                    "%d to %s are not provably inside the closing-theme window "
                    "%r -- filling with %s instead. Only the closing theme may "
                    "reuse a clip (operator ruling 2026-08-06).",
                    tail_n, cursor, target_total_frames,
                    (manifest or {}).get("closing_theme_frame_window"), gap_src)
            emit(gap_src, "", tail_n, _floor_aligned(cursor, tail_n))
        cursor = int(target_total_frames)
    return segments, cursor


def timeline_quality_report(manifest, segments):
    """S-A legibility floor (staged commit 1): a SEPARATE post-plan view that
    asserts each real-clip beat DELIVERS its planned visible span and flags any
    beat that would freeze. PURE -- never overwrites the raw manifest
    ``frame_count`` (which stays engine-produced); it only reads the planned
    segments.

    In positioned mode, a later beat owns its start boundary, so intentional
    crossfade overlap frames are trimmed from the earlier beat's visible slot.
    ``target_frame_count`` remains the full render request; the added
    ``planned_visible_frame_count`` and ``overlap_trimmed_frame_count`` make the
    distinction explicit. ``delivered_frames_ok`` is True when NO visible slot
    is held on an under-length clip. Sequential manifests retain their strict
    delivered == requested contract."""
    clips = {str(r.get("shot_id")): r
             for r in ((manifest or {}).get("clips") or [])
             if r.get("shot_id") and int((r or {}).get("target_frame_count") or 0) > 0}
    positioned = (bool(clips)
                  and int((manifest or {}).get("total_target_frames") or 0) > 0
                  and all(r.get("start_s") is not None for r in clips.values()))
    seg_by_shot = {}
    for s in (segments or []):
        if s.get("source") == "clip" and s.get("shot_id"):
            seg_by_shot.setdefault(str(s["shot_id"]), []).append(s)
    beats = []
    ok_all = True
    for sid, r in clips.items():
        tgt = int(r.get("target_frame_count") or 0)
        raw = int(r.get("frame_count") or 0)
        segs = seg_by_shot.get(sid, [])
        delivered = sum(int(s.get("n_frames") or 0) for s in segs)
        planned_visible = delivered
        looped = any(s.get("loop") for s in segs)
        if not segs:
            status = "no_clip_segment"
        elif raw > 0 and raw < planned_visible:
            status = "looped_fill" if looped else "held_last_frame"
        else:
            status = "ok"
        beats.append({
            "shot_id": sid, "beat_id": r.get("beat_id"),
            "engine_id": r.get("engine_id"), "target_frame_count": tgt,
            "requested_frame_count": tgt, "frame_count": raw,
            "rendered_frame_count": raw,
            "delivered_frame_count": delivered,
            "planned_visible_frame_count": planned_visible,
            "overlap_trimmed_frame_count": (
                max(0, tgt - planned_visible) if positioned else 0),
            "looped": bool(looped), "quality_status": status})
        if status == "held_last_frame":
            ok_all = False
        elif not positioned and segs and delivered != tgt:
            ok_all = False
    return {"delivered_frames_ok": ok_all, "beats": beats,
            "underran": [b for b in beats
                         if b["quality_status"] in ("looped_fill", "held_last_frame")]}


_FREEZE_RE = None


def parse_freezedetect(stderr):
    """Parse ffmpeg ``freezedetect`` stderr into a list of frozen spans
    ``[{start, end}]`` (seconds). PURE -- offline-testable. An open
    freeze_start with no freeze_end (frozen through EOF) yields end=None."""
    global _FREEZE_RE
    if _FREEZE_RE is None:
        import re
        _FREEZE_RE = re.compile(
            r"lavfi\.freezedetect\.freeze_(start|end)[:=]\s*([0-9.]+)")
    spans = []
    cur = None
    for tag, val in _FREEZE_RE.findall(stderr or ""):
        v = float(val)
        if tag == "start":
            cur = {"start": v, "end": None}
            spans.append(cur)
        elif tag == "end" and cur is not None:
            cur["end"] = v
            cur = None
    return spans


def freezedetect_silent(video_path, *, ffmpeg="ffmpeg", noise_db=-60, dur_s=2.0):
    """Run ffmpeg ``freezedetect`` over the SILENT video only (never the master)
    and return the parsed frozen spans. Used by the S-A live legibility proof;
    NOT wired into the default assemble path (it adds a full decode pass).
    FAIL-SOFT: returns ``[]`` if ffmpeg is unavailable."""
    fb = _ffmpeg_bin(ffmpeg)
    if not fb or not os.path.isfile(video_path):
        return []
    p = _run([fb, "-hide_banner", "-i", video_path, "-vf",
              "freezedetect=n=%ddB:d=%s" % (int(noise_db), str(dur_s)),
              "-map", "0:v:0", "-f", "null", os.devnull])
    return parse_freezedetect((p.stderr or "") + (p.stdout or ""))


def _seg_vf(w, h, fps, start_frame, sharpen=True, mode=None):
    """The per-segment ``-vf`` chain: an optional source trim, then the SHARED
    scale chain (lanczos+unsharp when ``sharpen``), then the tpad last-frame hold.
    PRESERVES the ``trim -> scale/unsharp/pad/fps -> tpad`` ordering. ``sharpen``
    is True for a real clip (the soft-native fix) and False for the procgen-floor
    slice (byte-identical to the legacy chain)."""
    trim = ("trim=start_frame=%d,setpts=PTS-STARTPTS," % int(start_frame)) \
        if int(start_frame) > 0 else ""
    # A DECLARED mode owns the spatial chain outright, so `sharpen` is not
    # forwarded alongside it -- passing both is the conflict `_scale_filter`
    # refuses by name. The trim prefix and the tpad suffix survive either way.
    if mode is not None:
        return _scale_filter(
            w, h, fps, mode=mode, pre=trim,
            post=",tpad=stop_mode=clone:stop_duration=3600")
    return _scale_filter(
        w, h, fps, sharpen=sharpen, pre=trim,
        post=",tpad=stop_mode=clone:stop_duration=3600")


def _color_args(out_path):
    # TAG bt709 identity (never matrix-convert) + canonical yuv420p H.264.
    return ["-vsync", "cfr", "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast", out_path]


def _encode_segment(fb, src, n_frames, seg_path, *, w, h, fps, start_frame=0,
                    loop=False, sharpen=True, engine=None,
                    delivery_scale_mode=None):
    """One canonical silent segment of EXACTLY ``n_frames`` from ``src`` (a clip,
    or the floor sliced at ``start_frame``): truncates a long source, holds the
    last frame (tpad clone) for a short one. FAIL CLOSED on ffmpeg error.

    ``loop=True`` stream-loops the input (-stream_loop -1) so a SHORT source
    REPEATS to fill ``n_frames`` -- the credits-tail backdrop keeps moving
    instead of freezing on the last frame (operator 2026-06-17). The tpad clone
    in _seg_vf stays as a safety but never triggers under an infinite loop.

    ``sharpen`` is True for a real engine clip (lanczos+unsharp soft-native fix)
    and False for the procgen-floor slice (NOT sharpened -- byte-identical).

    Queue item 8 (2026-08-08): ``engine`` is an UpscaleEngine instance loaded on
    a device (or None). When non-None AND ``engine.name != "off"`` AND
    ``sharpen=True``, dispatch to ``_run_model_pipeline`` (real-clip model path).
    Every other combination takes the byte-identical ffmpeg fast path -- ``off``
    is a sentinel that flows through the SAME lines as today, so directory/floor/
    black paths remain byte-identical whether a profile picks ``off`` or a real
    engine."""
    # A DECLARED DELIVERY MODE WINS, AND IT IS CHECKED FIRST -- before any
    # source-size probe, before any engine branch. A mixed manifest may well
    # have loaded a model for its legacy rows, and this row must not touch it:
    # the lane declared clean full-frame Lanczos and a model pass is neither
    # clean nor full-frame. Explicitly first, so no later branch can reach
    # `_run_model_pipeline` with a declared row.
    if delivery_scale_mode is not None:
        loop_args = ["-stream_loop", "-1"] if loop else []
        cmd = [fb, "-y", "-loglevel", "error"] + loop_args + ["-i", src, "-an",
               "-vf", _seg_vf(w, h, fps, start_frame,
                              mode=delivery_scale_mode),
               "-frames:v", str(int(n_frames))] + _color_args(seg_path)
        p = _run(cmd)
        if p.returncode != 0:
            raise ValueError(
                "OTR_SilentComposite: segment encode failed (%s) :: %s"
                % (os.path.basename(seg_path), p.stderr.strip()[:200]))
        return
    if engine is None or engine.name == "off" or not sharpen:
        # OBSERVABILITY (2026-08-09). A completed render used to say NOTHING
        # about whether the upscale stage engaged: neither this branch nor the
        # model branch logged, and the node's status string -- the only carrier
        # of "upscale=<engine>@<device>" -- never reaches /history. A live
        # 8-beat leg with upscale_engine='spandrel_esrgan' therefore finished
        # green while leaving zero evidence either way, which is not a proof.
        # Logged ONLY when an engine is actually selected, so the `off` default
        # emits nothing and its byte-identical path stays byte-identical.
        if engine is not None and engine.name != "off":
            log.info(
                "[OTR_SilentComposite] upscale FAST PATH for %s: engine=%s "
                "sharpen=%s (model runs only on sharpened real-clip segments)",
                os.path.basename(seg_path), engine.name, sharpen)
        loop_args = ["-stream_loop", "-1"] if loop else []
        cmd = [fb, "-y", "-loglevel", "error"] + loop_args + ["-i", src, "-an",
               "-vf", _seg_vf(w, h, fps, start_frame, sharpen=sharpen),
               "-frames:v", str(int(n_frames))] + _color_args(seg_path)
        p = _run(cmd)
        if p.returncode != 0:
            raise ValueError(
                "OTR_SilentComposite: segment encode failed (%s) :: %s"
                % (os.path.basename(seg_path), p.stderr.strip()[:200]))
        return

    # Model path: probe source dims first; if source already >= canvas at 1x,
    # the model would upsample to 2x then downsample back, which changes
    # pixels without advancing the goal (Codex r4 SF-3). Skip to the fast path.
    from ._otr_upscale_engines._pipeline import _probe_video_dims
    try:
        _src_w, _src_h, _ = _probe_video_dims(fb, src)
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            "OTR_SilentComposite: probe failed for %r: %s" % (src, e))
    if _src_w >= int(w) and _src_h >= int(h):
        log.info(
            "[OTR_SilentComposite] model skip (source %dx%d >= canvas %dx%d "
            "at 1x); fast path", _src_w, _src_h, int(w), int(h))
        loop_args = ["-stream_loop", "-1"] if loop else []
        cmd = [fb, "-y", "-loglevel", "error"] + loop_args + ["-i", src, "-an",
               "-vf", _seg_vf(w, h, fps, start_frame, sharpen=True),
               "-frames:v", str(int(n_frames))] + _color_args(seg_path)
        p = _run(cmd)
        if p.returncode != 0:
            raise ValueError(
                "OTR_SilentComposite: segment encode failed (%s) :: %s"
                % (os.path.basename(seg_path), p.stderr.strip()[:200]))
        return
    # The positive receipt. Names the engine, the device it was loaded on, and
    # the geometry it is actually changing -- so a finished render proves the
    # model ran rather than leaving it to be inferred from silence.
    log.info(
        "[OTR_SilentComposite] upscale MODEL PATH for %s: engine=%s device=%s "
        "src=%dx%d -> canvas=%dx%d (%d frames)",
        os.path.basename(seg_path), engine.name, getattr(engine, "device", None),
        _src_w, _src_h, int(w), int(h), int(n_frames))
    _run_model_pipeline(fb=fb, src=src, seg_path=seg_path,
                        n_frames=n_frames, w=w, h=h, fps=fps,
                        start_frame=start_frame, loop=loop, engine=engine,
                        src_w=_src_w, src_h=_src_h)


def _run_model_pipeline(*, fb, src, seg_path, n_frames, w, h, fps,
                        start_frame, loop, engine, src_w, src_h):
    """FFMPEG OWNS TIME, MODEL OWNS SPACE (Fable r3 amendment).

    Decoder runs the full _seg_vf chain MINUS scale/pad/sharpen:
      trim=start_frame, setpts=PTS-STARTPTS, fps=<target>,
      tpad=stop_mode=clone:stop_duration=3600
    with -frames:v n_frames on decode so the Python loop receives EXACTLY
    n_frames of conformed CFR source-resolution frames. Python loop is a dumb
    spatial map: read 1 frame -> model 2x (BHWC->BCHW inside the adapter) ->
    _fit_and_pad_bhwc decrease-fit + pad -> write to encoder stdin. Encoder
    has its own -frames:v n_frames. count_video_frames(seg_path) == n_frames
    is asserted BEFORE returning (Codex r3 MF-1 non-negotiable guard)."""
    import numpy as np
    import torch

    from ._otr_upscale_engines._pipeline import (
        _read_frames_exact, _tempfile_stderr, _close_stderr, _tail_path,
        _kill_and_wait_all, _unlink_if_exists,
        _validate_engine_output, _fit_and_pad_bhwc,
    )

    dec_vf_parts = []
    if int(start_frame) > 0:
        dec_vf_parts.append("trim=start_frame=%d" % int(start_frame))
        dec_vf_parts.append("setpts=PTS-STARTPTS")
    dec_vf_parts.append("fps=%d" % int(fps))
    dec_vf_parts.append("tpad=stop_mode=clone:stop_duration=3600")
    # Fable final gate MF-1: force EXPLICIT bt709 on the yuv->rgb24 decode.
    # Every clip source is bt709-tagged by the V-1 contract (matrix stamps in
    # eng_wan_ti2v / eng_ltx_av / eng_visualizer). Without this, ffmpeg 8's
    # negotiation still honors the tag correctly here -- but the encoder side
    # (below) is what needed the fix; adding this belt keeps the pair
    # symmetric and removes the version dependency for future ffmpeg builds.
    dec_vf_parts.append("scale=in_color_matrix=bt709")
    dec_vf_parts.append("format=rgb24")
    dec_vf = ",".join(dec_vf_parts)
    dec_args = [fb, "-y", "-loglevel", "error"]
    if loop:
        dec_args += ["-stream_loop", "-1"]
    dec_args += ["-i", src, "-vf", dec_vf,
                 "-frames:v", str(int(n_frames)),
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-an", "-"]

    # Fable final gate MF-1: force EXPLICIT bt709 tv on the rgb->yuv encode.
    # The encoder's rawvideo rgb24 stdin carries NO color tag, so ffmpeg's
    # auto-inserted rgb->yuv420p conversion falls to swscale's default matrix
    # (bt601 historically; version-dependent at best) while `_color_args`
    # TAGS the output bt709. Asymmetric round-trip = real color shift on
    # exactly the model-enhanced segments. This -vf forces bt709+tv range
    # BEFORE libx264, matching the tagging.
    enc_args = [fb, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", "%dx%d" % (int(w), int(h)),
                "-r", "%d" % int(fps),
                "-i", "-", "-an",
                "-frames:v", str(int(n_frames)),
                "-vf", "scale=out_color_matrix=bt709:out_range=tv"
                ] + _color_args(seg_path)

    dec_stderr_fobj, dec_stderr_path = _tempfile_stderr("dec")
    enc_stderr_fobj, enc_stderr_path = _tempfile_stderr("enc")
    dec = subprocess.Popen(dec_args, stdout=subprocess.PIPE,
                           stderr=dec_stderr_fobj, bufsize=0)
    try:
        enc = subprocess.Popen(enc_args, stdin=subprocess.PIPE,
                               stderr=enc_stderr_fobj, bufsize=0)
    except Exception:  # noqa: BLE001
        _kill_and_wait_all([dec], timeout=30)
        _close_stderr(dec_stderr_fobj, enc_stderr_fobj)
        _unlink_if_exists(dec_stderr_path)
        _unlink_if_exists(enc_stderr_path)
        raise

    # BATCH = 1: spandrel's ImageModelDescriptor.__call__ enforces (1,C,H,W).
    # Having a per-pipe BATCH > 1 just adds partial-batch-on-EOF complexity
    # for zero throughput gain (Antigravity r4 MF-1, Codex r4 MF-1).
    BATCH = 1
    src_frame_bytes = int(src_w) * int(src_h) * 3
    frames_written = 0
    error_raised = False
    # HELD-FRAME MEMOIZATION (operator, 2026-08-25: "if the upscaler were smart
    # that this is the same frame I'm not gonna waste tokens on it, just reuse
    # the last upscaled frame").
    #
    # WHY THIS IS BIT-EXACT AND NOT AN APPROXIMATION. The engine is a plain
    # deterministic conv net run under `torch.inference_mode()` with the
    # descriptor in `.eval()` (eng_spandrel_esrgan.py:311, :352) -- no dropout,
    # no sampling, no state carried between calls -- and `_fit_and_pad_bhwc` is
    # pure geometry. Identical input bytes therefore produce identical output
    # bytes, so reusing the previous result is memoization, not an
    # approximation: it cannot change one byte of the encoded segment, only
    # skip work that would recompute the same answer.
    #
    # WHY A ONE-SLOT CACHE IS THE RIGHT SIZE. On a `still_*` lane a beat is ONE
    # image held across the whole segment, so consecutive frames are identical
    # and a single previous-frame slot captures the entire win (hundreds of
    # model calls per segment collapse to one). On a genuinely moving lane
    # every frame differs, the slot never hits, and the only cost paid is the
    # comparison below.
    #
    # WHY A DIRECT BYTES COMPARE RATHER THAN A HASH: `bytes.__eq__` is memcmp,
    # it is faster than hashing 3.7 MB, and it has NO collision risk at all --
    # a hash collision here would silently emit the wrong picture, which is a
    # worse failure than the cost it would save.
    prev_in_bytes = None
    prev_out_bytes = None
    frames_reused = 0
    try:
        while frames_written < int(n_frames):
            want = min(BATCH, int(n_frames) - frames_written)
            buf = _read_frames_exact(dec.stdout, src_frame_bytes, want)
            if not buf:
                raise RuntimeError(
                    "decoder EOF before n_frames satisfied "
                    "(%d/%d) src=%r" % (frames_written, int(n_frames), src))
            got = len(buf) // src_frame_bytes
            # Guarded on got == 1: with BATCH > 1 a buffer holds several frames
            # and equality would only mean "this RUN of frames repeats", so the
            # cached output would be right by luck rather than by construction.
            if got == 1 and prev_in_bytes is not None and buf == prev_in_bytes:
                enc.stdin.write(prev_out_bytes)
                frames_written += got
                frames_reused += got
                continue
            arr = np.frombuffer(buf, dtype=np.uint8).copy()
            arr = arr.reshape((got, int(src_h), int(src_w), 3))
            # engine.device (Antigravity r3 MF-3) -- public property, never
            # reach into engine._descriptor.
            bhwc = torch.from_numpy(arr).to(engine.device).float() / 255.0
            out_bhwc = engine.upscale_frames(bhwc)
            _validate_engine_output(
                out_bhwc, expected_scale=int(engine.intrinsic_scale),
                in_shape=bhwc.shape)
            out_bhwc = _fit_and_pad_bhwc(out_bhwc, int(w), int(h))
            # .contiguous() before .numpy() -- non-contiguous tensors after a
            # permute would raise or corrupt strides (Antigravity r4 MF-2).
            out_u8 = (out_bhwc.clamp(0, 1) * 255.0).byte().contiguous().cpu().numpy()
            out_bytes = out_u8.tobytes()
            enc.stdin.write(out_bytes)
            if got == 1:
                prev_in_bytes = buf
                prev_out_bytes = out_bytes
            frames_written += got
        enc.stdin.close()
        # OBSERVABILITY, deliberately unconditional for the model path. This
        # file already carries the lesson that an upscale stage which logs
        # nothing leaves a green leg with zero evidence either way (see the
        # 2026-08-09 note on the fast path); a cache that silently did nothing
        # would be exactly that failure again, one layer down.
        if frames_reused:
            log.info(
                "[OTR_SilentComposite] upscale HELD-FRAME REUSE for %s: "
                "%d/%d frame(s) served from the previous result "
                "(%d model call(s) run)",
                os.path.basename(seg_path), frames_reused, int(n_frames),
                int(n_frames) - frames_reused)
        # Check return codes and read stderr WHILE handles are open (path-based
        # so the offset issue can't bite -- Antigravity r3 MF-2 + r4 MF-1).
        try:
            dec.wait(timeout=30)
            enc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            error_raised = True
            _kill_and_wait_all([dec, enc], timeout=30)
            raise RuntimeError(
                "ffmpeg wait timeout; dec_tail=%r enc_tail=%r"
                % (_tail_path(dec_stderr_path, 400),
                   _tail_path(enc_stderr_path, 400)))
        if dec.returncode not in (0, None):
            error_raised = True
            raise RuntimeError(
                "ffmpeg decode rc=%s :: %s"
                % (dec.returncode, _tail_path(dec_stderr_path, 400)))
        if enc.returncode not in (0, None):
            error_raised = True
            raise RuntimeError(
                "ffmpeg encode rc=%s :: %s"
                % (enc.returncode, _tail_path(enc_stderr_path, 400)))
        got_frames = count_video_frames(seg_path)
        if got_frames != int(n_frames):
            error_raised = True
            raise RuntimeError(
                "segment frame count mismatch: %d != %d (src=%r seg=%r)"
                % (got_frames, int(n_frames), src, seg_path))
    except Exception:
        error_raised = True
        _kill_and_wait_all([dec, enc], timeout=30)
        _unlink_if_exists(seg_path)
        raise
    finally:
        _close_stderr(dec_stderr_fobj, enc_stderr_fobj)
        if not error_raised:
            _unlink_if_exists(dec_stderr_path)
            _unlink_if_exists(enc_stderr_path)


def _encode_black(fb, n_frames, seg_path, *, w, h, fps):
    """A generated black segment of EXACTLY ``n_frames`` (the ultimate gap-fill
    so an episode with neither a clip nor a floor still assembles)."""
    cmd = [fb, "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", "color=c=black:s=%dx%d:r=%d" % (w, h, fps),
           "-frames:v", str(int(n_frames))] + _color_args(seg_path)
    p = _run(cmd)
    if p.returncode != 0:
        raise ValueError("OTR_SilentComposite: black segment failed (%s) :: %s"
                         % (os.path.basename(seg_path), p.stderr.strip()[:200]))


# --------------------------------------------------------------------------- #
# Directory-clip read path (3D plan 7.2 -- the character_3d alpha handoff).
# A clip row may point at a DIRECTORY of straight-alpha RGBA frames (PNG/EXR,
# sorted by name) instead of an mp4. The segment encoder lists the frames via
# the shared nodes/_otr_video_engines/directory_clip.py rule, plays them at
# the canonical fps via an ffconcat list (no %06d numbering assumption),
# OVERLAYS them centered on the background (the floor slice when available,
# else black -- the W7 provider-plate composite arrives with the 3D lane),
# and FLATTENS to opaque yuv420p. ffmpeg's overlay expects STRAIGHT
# (non-premultiplied) alpha -- exactly the validated contract. The webm/vp9 /
# mov/prores4444 alpha-VIDEO branches are CUT from v1 [H-RT, GPT].
# --------------------------------------------------------------------------- #
def _is_frame_dir(path) -> bool:
    """True when ``path`` is a readable directory-clip frame dir (tolerant)."""
    try:
        from ._otr_video_engines.directory_clip import frame_dir_summary
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_video_engines.directory_clip import frame_dir_summary  # type: ignore
    ok, _n, _b = frame_dir_summary(path)
    return ok


def _write_frames_concat(frames, listfile, fps):
    """An ffconcat v1 list playing each frame for exactly 1/fps seconds
    (sorted-by-name order is the caller's contract)."""
    with open(listfile, "w", encoding="utf-8") as f:
        f.write("ffconcat version 1.0\n")
        for fp in frames:
            f.write("file '%s'\n" % fp.replace("\\", "/").replace("'", "'\\''"))
            f.write("duration %.6f\n" % (1.0 / float(fps)))
        # concat-demuxer quirk: the LAST entry's duration is honored only if
        # the file is listed once more (else the final frame is dropped early)
        f.write("file '%s'\n" % frames[-1].replace("\\", "/").replace("'", "'\\''"))


def _encode_segment_from_dir(fb, frame_dir, n_frames, seg_path, *, w, h, fps,
                             bg_path="", bg_start_frame=0, bg_is_still=False):
    """One canonical silent segment of EXACTLY ``n_frames`` from a straight-
    alpha frame DIRECTORY: frames sorted by name -> overlay (centered) on the
    background -> flatten yuv420p. FAIL CLOSED on a bad directory or ffmpeg
    error. ``bg_path`` (the floor) is sliced at ``bg_start_frame``; absent ->
    black.

    ``bg_is_still=True`` (the textured-hero 3D PoC, C1): ``bg_path`` is a single
    STILL PLATE image, not a video -- it is looped (``-loop 1``) and held for the
    full ``n_frames`` (the generated background behind the turntable mesh). The
    slice ``bg_start_frame`` is irrelevant for a still and ignored."""
    try:
        from ._otr_video_engines.directory_clip import list_directory_frames
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_video_engines.directory_clip import list_directory_frames  # type: ignore
    frames = list_directory_frames(frame_dir)   # raises ValueError, named
    workdir = os.path.dirname(seg_path)
    listfile = os.path.join(
        workdir, os.path.basename(seg_path) + ".frames.ffconcat")
    _write_frames_concat(frames, listfile, fps)
    if bg_path and os.path.isfile(bg_path) and bg_is_still:
        # A single still plate: loop it to fill the beat (no trim/tpad needed --
        # the loop + the overlay eof_action=repeat hold it for every frame). A
        # REAL still plate IS sharpened (sharpen=True) -- it is composited content,
        # not the procgen floor.
        bg_in = ["-loop", "1", "-i", bg_path]
        bg_filter = _scale_filter(w, h, fps, sharpen=True,
                                  in_label="0:v", out_label="bg")
    elif bg_path and os.path.isfile(bg_path):
        # The procgen-floor video slice -> NOT sharpened (byte-identical to the
        # legacy chain); preserve the trim prefix + tpad hold.
        bg_in = ["-i", bg_path]
        trim = (("trim=start_frame=%d,setpts=PTS-STARTPTS,"
                 % int(bg_start_frame)) if int(bg_start_frame) > 0 else "")
        bg_filter = _scale_filter(
            w, h, fps, sharpen=False, in_label="0:v", out_label="bg",
            pre=trim, post=",tpad=stop_mode=clone:stop_duration=3600")
    else:
        bg_in = ["-f", "lavfi", "-i",
                 "color=c=black:s=%dx%d:r=%d" % (w, h, fps)]
        bg_filter = "[0:v]null[bg]"
    # FOREGROUND: straight-alpha RGBA -> sharpen=True but pad=False (a black pad
    # would paint opaque borders over the alpha edges -> destroy the matte). The
    # overlay re-centers the fg, so no pad is needed.
    fg_filter = _scale_filter(w, h, fps, sharpen=True, pad=False,
                              in_label="1:v", out_label="fg", pre="format=rgba,")
    # 3D image streams chunk 7: DEFAULT = straight-alpha SOURCE-OVER at FULL
    # opacity. Compositing in RGB (overlay format=rgb) makes an OPAQUE mesh pixel
    # (alpha==255) fully REPLACE the plate -- no premultiplied-edge ghost, no
    # double-exposure of the background plate through the mesh. The mesh already
    # renders straight-alpha, so its own alpha is the matte. The prior auto-format
    # look is preserved as the NAMED opt-in style OTR_MESH_COMPOSITE_STYLE=blend
    # (never the default).
    _style = os.environ.get("OTR_MESH_COMPOSITE_STYLE", "source_over").strip().lower()
    _ov_format = "auto" if _style == "blend" else "rgb"
    graph = (bg_filter + ";" + fg_filter +
             ";[bg][fg]overlay=(W-w)/2:(H-h)/2:eof_action=repeat:format=%s,"
             "format=yuv420p" % _ov_format)
    cmd = [fb, "-y", "-loglevel", "error"] + bg_in + [
        "-f", "concat", "-safe", "0", "-i", listfile, "-an",
        "-filter_complex", graph,
        "-frames:v", str(int(n_frames))] + _color_args(seg_path)
    p = _run(cmd)
    if p.returncode != 0:
        raise ValueError(
            "OTR_SilentComposite: directory-clip segment failed (%s) :: %s"
            % (os.path.basename(seg_path), p.stderr.strip()[:300]))


def assemble_silent_timeline(manifest, base_video_path, out_path, *, w=1472,
                             h=832, fps=25, ffmpeg="ffmpeg", engine=None):
    """Assemble a beat-ordered clip manifest into ONE always-silent canonical
    CFR video; FAIL CLOSED. Sequential beats retain their full requested spans;
    positioned beats are conformed to their non-overlapping visible slots. A
    missing clip is gap-filled from the floor (``base_video_path``) or black.
    The assembled frame count is asserted == the audio-derived timeline boundary
    (the pre-mux A/V sync guard). Returns ``(out_path, report)``."""
    report: list = []
    fb = _ffmpeg_bin(ffmpeg)
    if not fb:
        raise ValueError("OTR_SilentComposite: ffmpeg not found (%r)" % ffmpeg)
    w, h, fps = _even(w), _even(h), max(1, int(fps))
    floor_ok = bool(base_video_path) and os.path.isfile(base_video_path)
    floor_frames = count_video_frames(base_video_path) if floor_ok else 0
    # The post-audio ledger's accepted timeline boundary is primary. Positioned
    # manifests may overlap full render requests at audio crossfades, so their
    # render-work sum is deliberately NOT the output length. Legacy/sequential
    # manifests still carry their old requested-frame sum here.
    mft_total = int((manifest or {}).get("timeline_total_frames")
                    or (manifest or {}).get("total_target_frames") or 0)
    target_total = mft_total if mft_total > 0 else None
    _rows = [r for r in ((manifest or {}).get("clips") or [])
             if int((r or {}).get("target_frame_count") or 0) > 0]
    manifest_positioned = (bool(_rows)
                           and all(r.get("start_s") is not None for r in _rows))
    if floor_ok:
        # Cross-check/fallback against the actual MASTER MIX. A positioned
        # manifest may reconcile downward as well as upward: refusing to shrink
        # here was the terminal live failure when two 0.5 s crossfades were
        # counted twice. Sequential/legacy manifests preserve the historical
        # grow-only behavior. Ceil covers the final fractional audio frame; the
        # mux keeps its independent, narrow terminal tolerance.
        master_dur = 0.0
        try:
            import glob as _glob
            _sib = os.path.join(os.path.dirname(base_video_path),
                                "*_master.wav")
            for _cand in _glob.glob(_sib):
                master_dur = max(master_dur, _probe_audio_duration(_cand))
        except Exception:  # noqa: BLE001 -- best-effort; fallbacks below
            pass
        if master_dur <= 0:
            master_dur = _probe_audio_duration(base_video_path)
        if master_dur <= 0:
            master_dur = _probe_duration(base_video_path)
        if master_dur > 0:
            base_total = max(1, int(math.ceil(master_dur * fps)))
            reconcile = (base_total > 0 and (
                target_total is None
                or (manifest_positioned and base_total != target_total)
                or (not manifest_positioned and base_total > target_total)))
            if reconcile:
                if target_total is not None and manifest_positioned:
                    report.append(
                        "A/V-sync master reconcile: %d -> %d frames "
                        "(positioned ledger cross-check)"
                        % (target_total, base_total))
                elif target_total is not None:
                    report.append(
                        "A/V-sync tail to master: %d -> %d frames "
                        "(closing-theme backdrop)"
                        % (target_total, base_total))
                target_total = base_total
        # CREDITS FLOOR-EXTEND RIPPED (credits enrichment 2026-07-03, BUG-410
        # retired). This block USED to extend the composite PAST the master to
        # the procgen floor's FULL frame count so ~20s of green rolling credits
        # scrolled in silence after the closing theme (the second "credits
        # organ", Fable BUILD-BREAKER #2). Under the silent-tail model the
        # unified credits roll is appended LATE as a SILENT tail by
        # OTR_CreditsRoll (which since 2026-07-29 rides over the FROZEN FINAL
        # FRAME of the body this function assembles, and DECLARES its duration
        # to the credits-aware mux guard). The composite now ends at the MASTER
        # length -- no floor-extend.
    segments, total = plan_timeline_segments(
        manifest, floor_available=floor_ok, floor_frames=floor_frames,
        target_total_frames=target_total, fps=fps)
    if not segments or total <= 0:
        raise ValueError("OTR_SilentComposite: manifest has no renderable beats")
    # S-A legibility floor: a SEPARATE delivered-frames view (raw manifest
    # frame_count untouched). LOUD-report any underran beat + whether clip-fill
    # filled it; a held_last_frame beat (fill OFF) is a legibility failure.
    qa = timeline_quality_report(manifest, segments)
    if qa["underran"]:
        report.append(
            "clip-fill: %d beat(s) underran -> %s"
            % (len(qa["underran"]),
               ", ".join("%s %s(%d/%d)"
                         % (b["shot_id"], b["quality_status"],
                            b["rendered_frame_count"],
                            b["planned_visible_frame_count"])
                         for b in qa["underran"])))
    if not qa["delivered_frames_ok"]:
        _held = sum(1 for b in qa["beats"]
                    if b["quality_status"] == "held_last_frame")
        report.append(
            "LEGIBILITY (LOUD): %d beat(s) did NOT deliver planned visible frames "
            "(held_last_frame -> static murk); clip-fill is OFF "
            "(OTR_CLIP_FILL=0)" % _held)
    workdir = tempfile.mkdtemp(prefix="otr_assemble_")
    try:
        seg_paths = []
        for seg in segments:
            seg_path = os.path.join(workdir, "seg_%04d.mp4" % seg["order"])
            kind = seg["source"]
            if kind == "clip":
                # 3D plan 7.2: a clip row may be a straight-alpha frame
                # DIRECTORY (the character_3d handoff) instead of an mp4.
                if os.path.isdir(seg["path"]) and _is_frame_dir(seg["path"]):
                    kind = "dir_clip"
                elif not os.path.isfile(seg["path"]):
                    kind = "floor" if floor_ok else "black"   # vanished clip
            if kind == "dir_clip":
                # C1 (textured-hero 3D PoC): a per-clip generated still plate is
                # the background behind the turntable mesh when present; else the
                # legacy floor slice / black (byte-identical for every non-3D
                # directory clip).
                plate = seg.get("bg_still_path") or ""
                use_still = bool(plate) and os.path.isfile(plate)
                _encode_segment_from_dir(
                    fb, seg["path"], seg["n_frames"], seg_path,
                    w=w, h=h, fps=fps,
                    bg_path=(plate if use_still
                             else (base_video_path if floor_ok else "")),
                    bg_start_frame=seg["src_start_frame"],
                    bg_is_still=use_still)
            elif kind == "clip":
                # A REAL engine clip -> sharpen (lanczos+unsharp soft-native fix).
                # Queue item 8 (2026-08-08): engine THREADS here (Antigravity r4
                # MF-3 explicit-forward). This is the ONE branch that consumes
                # the upscale model; dir_clip / floor / black paths NEVER get
                # engine=engine, so their output stays byte-identical for every
                # profile whether upscale_stage.engine is "off" or non-off.
                _encode_segment(fb, seg["path"], seg["n_frames"], seg_path,
                                w=w, h=h, fps=fps, start_frame=0,
                                loop=bool(seg.get("loop")), sharpen=True,
                                engine=engine,
                                delivery_scale_mode=seg.get(
                                    "delivery_scale_mode"))
            elif kind == "floor":
                # The procgen floor slice -> NOT sharpened (byte-identical chain).
                _encode_segment(fb, base_video_path, seg["n_frames"], seg_path,
                                w=w, h=h, fps=fps, start_frame=seg["src_start_frame"],
                                sharpen=False, engine=None)
            else:
                _encode_black(fb, seg["n_frames"], seg_path, w=w, h=h, fps=fps)
            seg_paths.append(seg_path)
        listfile = os.path.join(workdir, "concat.txt")
        with open(listfile, "w", encoding="utf-8") as f:
            for sp in seg_paths:
                f.write("file '%s'\n" % sp.replace("'", "'\\''"))
        cmd = [fb, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
               "-i", listfile, "-an", "-c", "copy", out_path]
        assert "-shortest" not in cmd
        p = _run(cmd)
        if p.returncode != 0:
            raise ValueError("OTR_SilentComposite: concat failed :: %s"
                             % p.stderr.strip()[:300])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    na = count_audio_streams(out_path)
    if na != 0:
        raise ValueError("OTR_SilentComposite: assembled has %d audio stream(s); "
                         "must be 0 (V-1)" % na)
    got = count_video_frames(out_path)
    if got != total:
        raise ValueError("OTR_SilentComposite: assembled %d frames != audio-derived "
                         "budget %d frames (A/V sync guard)" % (got, total))
    info = probe_video(out_path)
    report.append("assembled %d beats -> %d frames @%dfps %sx%s silent; "
                  "budget %d OK" % (len(segments), got, fps,
                                    info.get("width", w), info.get("height", h), total))
    # S-A post-assemble QA artifact (best-effort): self-documents per-beat
    # delivered frames + clip-fill status next to the silent video.
    try:
        with open(out_path + ".qa.json", "w", encoding="utf-8") as _qf:
            json.dump(qa, _qf, ensure_ascii=True, indent=1)
    except Exception:  # noqa: BLE001 -- QA artifact is best-effort
        pass
    return out_path, report


class OTRSilentComposite:
    """Registered as ``OTR_SilentComposite``. Render output -> ONE always-silent
    canonical video (bt709/yuv420p/CFR, audio stripped). Mux happens downstream."""

    CATEGORY = "OldTimeRadio/v2/video"
    FUNCTION = "composite"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("silent_video_path", "report")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_video_path": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": (
                        "Base video to normalize (M1: the OTR_SignalLostVideo "
                        "radio-floor mp4). Any audio is STRIPPED here (V-1)."
                    ),
                }),
            },
            "optional": {
                "canvas_w": ("INT", {"default": 1472, "min": 16, "max": 7680}),
                "canvas_h": ("INT", {"default": 832, "min": 16, "max": 4320}),
                "fps": ("INT", {"default": 25, "min": 1, "max": 120}),
                "ffmpeg": ("STRING", {"default": "ffmpeg"}),
                "output_path": ("STRING", {
                    "default": "",
                    "tooltip": "Silent composite path. Empty -> <output>/otr/episodes/<stem>_silent.mp4.",
                }),
                "gate_in": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "Optional ordering signal (opaque STRING).",
                }),
                "clip_manifest_json": ("STRING", {
                    "default": "{}", "multiline": True, "forceInput": True,
                    "tooltip": (
                        "OTR_VideoRenderBatch(mode=episode) clip manifest. When "
                        "it carries per-beat clips the composite ASSEMBLES a "
                        "frame-accurate CFR timeline (each beat conformed to its "
                        "audio-derived frame count; gaps filled from "
                        "base_video_path). Empty -> the single-base floor path."
                    ),
                }),
                # Queue item 8 (2026-08-08): post-render upscale/enhance stage.
                # APPEND at end of widgets_values per BUG-LOCAL-097.
                # The dropdown auto-populates from the upscale registry's live
                # roster; "off" is the byte-identical default.
                "upscale_engine": (list(_upscale_engine_names()) or ["off"], {
                    "default": "off",
                    "tooltip": (
                        "Post-render upscale/enhance stage. Runs INSIDE the "
                        "composite per-clip (sharpen=True branch only); floor / "
                        "directory / black paths are unchanged. off = today's "
                        "lanczos+unsharp, byte-identical."
                    ),
                }),
                "upscale_device": ("STRING", {
                    "default": "cpu", "multiline": False,
                    "tooltip": (
                        "cpu | cuda | cuda:N. Composite itself runs on CPU "
                        "(ffmpeg); this widget selects the device the model "
                        "loads on. Rejected invalid tokens fail loud."
                    ),
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def IS_CHANGED(cls, base_video_path="", canvas_w=1472, canvas_h=832, fps=25,
                    ffmpeg="ffmpeg", output_path="", gate_in="",
                    clip_manifest_json="{}",
                    upscale_engine="off", upscale_device="cpu", **kw):
        """Fingerprint ALL external inputs regardless of engine. ComfyUI relies
        on ``nan != nan`` to force re-execution, so returning ``float("nan")``
        when a stat fails is the correct fail-open per Bug Bible 06.02/06.07.
        Codex r4 MF-6: fingerprint includes clip manifest paths, background
        stills, sibling master WAV, OTR_COMPOSITE_UNSHARP_AMOUNT env var, and
        the engine identity."""
        try:
            parts: list = []
            # 1. Base video (source of the assemble/normalize path).
            if base_video_path and os.path.isfile(base_video_path):
                st = os.stat(base_video_path)
                parts.append(("base", base_video_path, st.st_mtime_ns, st.st_size))
            # 2. Clips from manifest.
            try:
                manifest = json.loads(clip_manifest_json or "{}")
            except Exception:  # noqa: BLE001
                return float("nan")
            for row in ((manifest or {}).get("clips") or []):
                for k in ("path", "bg_still_path"):
                    p = row.get(k)
                    if not p:
                        continue
                    if os.path.isfile(p):
                        st = os.stat(p)
                        parts.append((k, p, st.st_mtime_ns, st.st_size))
                    elif os.path.isdir(p):
                        # Directory-clip: sorted (name, mtime, size) tuples
                        # (in-place frame replacement invisible to dir stat --
                        # Codex r4 MF-6).
                        try:
                            frames = sorted(os.listdir(p))
                        except OSError:
                            return float("nan")
                        dir_parts: list = []
                        for fname in frames:
                            fp = os.path.join(p, fname)
                            try:
                                fst = os.stat(fp)
                                dir_parts.append((fname, fst.st_mtime_ns, fst.st_size))
                            except OSError:
                                return float("nan")
                        parts.append(("dir", p, tuple(dir_parts)))
            # 3. Sibling master WAV (composite reads it for timeline length).
            if base_video_path:
                base_dir = os.path.dirname(base_video_path)
                if os.path.isdir(base_dir):
                    try:
                        for candidate in sorted(os.listdir(base_dir)):
                            if candidate.endswith("_master.wav"):
                                fp = os.path.join(base_dir, candidate)
                                try:
                                    st = os.stat(fp)
                                    parts.append(("master_wav", fp,
                                                  st.st_mtime_ns, st.st_size))
                                except OSError:
                                    pass
                    except OSError:
                        pass
            # 3b. THE ORDERED DELIVERY-MODE VECTOR (2026-08-22). Changing only
            # a row's declared enlargement changes the output pixels and nothing
            # else in this fingerprint would notice, so the vector goes in on
            # its own. The cadence COUNTS deliberately stay out: they vary per
            # beat and are already implied by the frame counts.
            # SCOPED TO A MANIFEST THAT HAS CLIPS, and the scope is the fix
            # for a real over-reach. `_has_model_eligible_clips({})` is False
            # for an EMPTY manifest too -- but an empty manifest is the
            # single-base normalize path, whose historical contract includes
            # BUG-06.07's fail-open: a raising fingerprint must return a bare
            # nan. Skipping the fingerprint there would have quietly replaced
            # that nan with a stable key. The Ghost case is specifically "there
            # ARE clips and every one of them declares its own enlargement".
            _rows = [r for r in ((manifest or {}).get("clips") or [])
                     if isinstance(r, dict)]
            _model_eligible = (not _rows) or _has_model_eligible_clips(manifest)
            parts.append(("delivery_modes", tuple(
                str(r.get("delivery_scale_mode") or "")
                for r in ((manifest or {}).get("clips") or [])
                if isinstance(r, dict))))
            # 4. Environment values that affect output pixels (Codex r4 MF-6).
            #
            # The unsharp amount is EXCLUDED when no row is model-eligible: the
            # clean full-frame chain emits no unsharp at all, so folding the env
            # var in there would invalidate a cached composite for a knob that
            # cannot reach a single pixel of it.
            parts.append(("env",
                          (os.environ.get("OTR_COMPOSITE_UNSHARP_AMOUNT", "")
                           if _model_eligible else ""),
                          os.environ.get("OTR_MESH_COMPOSITE_STYLE", "")))
            # 5. Engine identity + whatever model state that engine declares.
            #
            # The engine is ASKED rather than special-cased. This block used to
            # test `upscale_engine == "spandrel_esrgan"` and stat a hardcoded
            # "RealESRGAN_x2plus.pth", which meant engine #2 would register,
            # appear in the dropdown, run, and contribute NO model bytes to the
            # cache key -- and it resolved the file differently from the loader
            # besides. Both go away by asking the owner of the fact.
            parts.append(("engine", str(upscale_engine), str(upscale_device)))
            # ...but its MODEL BYTES only when the model can actually run. An
            # inactive stale engine must not become a dependency of a composite
            # it cannot touch -- otherwise a Ghost-only episode fails its
            # fingerprint on missing weights it never needed.
            if not _model_eligible:
                return repr(parts)
            try:
                engine = _get_upscale_engine(upscale_engine)
                parts.extend(engine.model_fingerprint_parts())
            except Exception as exc:  # noqa: BLE001
                # An unknown engine id (KeyError from the registry) or a real
                # stat failure lands here. nan is the correct fail-open, but a
                # SILENT nan on a minutes-long node is how a typo hides for
                # weeks -- so say it once, naming the engine.
                _log_fingerprint_failure_once(upscale_engine, exc)
                return float("nan")
            return repr(parts)
        except Exception:  # noqa: BLE001
            return float("nan")

    def _default_out(self, base_video_path: str) -> str:
        try:
            import folder_paths  # type: ignore
            root = folder_paths.get_output_directory()
        except Exception:  # noqa: BLE001
            root = "."
        # OUTPUT HYGIENE (operator directive 2026-06-09): every per-episode
        # asset lives INSIDE that episode's own folder under otr/episodes/<ep>/
        # -- never loose at the episodes root. The base video stem IS the
        # episode slug.
        stem = os.path.splitext(os.path.basename(base_video_path or "episode"))[0]
        out_dir = os.path.join(root, "otr", "episodes", stem)
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"{stem}_silent.mp4")

    def composite(self, base_video_path, canvas_w=1472, canvas_h=832, fps=25,
                  ffmpeg="ffmpeg", output_path="", gate_in="",
                  clip_manifest_json="{}",
                  upscale_engine="off", upscale_device="cpu"):
        out = output_path.strip() or self._default_out(base_video_path)
        manifest = {}
        try:
            m = json.loads(clip_manifest_json or "{}")
            if isinstance(m, dict):
                manifest = m
        except (ValueError, TypeError):
            manifest = {}
        assemble = bool(manifest.get("clips"))
        # the assemble timeline follows the audio-derived budget fps when present
        eff_fps = int(manifest.get("fps") or fps) if assemble else int(fps)

        # Queue item 8 (2026-08-08): resolve the upscale engine + load it once
        # BEFORE the segment loop, unload it in an outer finally. Load is
        # GATED on BOTH (engine.name != "off") AND (assemble is True):
        # * Off engine: HONEST-SWITCH LAW default -- never touch the model
        #   pipeline; a stale device value must NOT break the byte-identical
        #   path (Sonnet 5 pre-implementation MF-2).
        # * Single-base normalize path (assemble=False): normalize_to_silent_
        #   canonical never consumes the engine, so loading it there is pure
        #   waste AND -- worse -- a load-time failure (missing model file, no
        #   spandrel, CUDA OOM) would break a render that never needed the
        #   model at all (Sonnet 5 QA-on-diff MF-2).
        # * A manifest with NO model-eligible row (2026-08-22): every clip
        #   DECLARED its own enlargement, so a selected upscaler cannot act on
        #   anything here. Resolving/asserting/loading it anyway would let a
        #   STALE selection -- a profile that once picked ESRGAN -- fail a
        #   render that has no use for the model. Fall back to the existing
        #   "off" sentinel WITHOUT resolving the stale choice, which is the same
        #   byte-identical path an explicitly-off profile already takes. A MIXED
        #   manifest still loads it once, for the legacy rows that earn it.
        _model_eligible = _has_model_eligible_clips(manifest)
        if assemble and not _model_eligible:
            log.info(
                "[OTR_SilentComposite] upscale SKIPPED: no model-eligible clip "
                "in this manifest (every real row declares its own "
                "delivery_scale_mode), so the selected engine %r is not "
                "resolved or loaded.", upscale_engine)
            engine = _get_upscale_engine("off")
        else:
            _assert_upscale_usable(upscale_engine, "upscale_stage")
            engine = _get_upscale_engine(upscale_engine)
        _engine_active = (engine.name != "off") and assemble
        if _engine_active:
            device = _resolve_upscale_device(upscale_device)
            engine.load(device)
            # Load succeeds silently in the adapter, so without this a reader
            # cannot tell "loaded fine" from "never engaged". Report the
            # RESOLVED checkpoint too: on the headless topology folder_paths
            # maps upscale_models at a dir that holds no .pth, so the file is
            # reached only via the repo-relative fallback -- exactly the
            # divergence 088dabc8 fixed, and worth seeing per run.
            _resolved = None
            try:
                _resolver = getattr(engine, "_resolve_model", None)
                if callable(_resolver):
                    _resolved = _resolver()[1]
            except Exception:  # noqa: BLE001 -- diagnostics must never fail a render
                _resolved = None
            log.info(
                "[OTR_SilentComposite] upscale engine LOADED: %s on %s "
                "(checkpoint=%s)", engine.name, device, _resolved or "n/a")
        try:
            try:
                if assemble:
                    silent, report = assemble_silent_timeline(
                        manifest, base_video_path, out, w=int(canvas_w),
                        h=int(canvas_h), fps=eff_fps, ffmpeg=ffmpeg,
                        engine=engine,
                    )
                else:
                    # normalize_to_silent_canonical is the single-base path;
                    # it does NOT consume the engine (Codex r4 MF-7). The
                    # load was gated above so no unload is needed here.
                    silent, report = normalize_to_silent_canonical(
                        base_video_path, out, w=int(canvas_w), h=int(canvas_h),
                        fps=int(fps), ffmpeg=ffmpeg,
                    )
            except ValueError as exc:
                log.error("[OTR_SilentComposite] %s", exc)
                return ("", f"error: {exc}")
        finally:
            if _engine_active:
                engine.unload()
        for line in report:
            log.info("[OTR_SilentComposite] %s", line)
        mode = "assemble" if assemble else "single-base"
        engine_tag = ("" if engine.name == "off"
                      else " upscale=%s@%s" % (engine.name, upscale_device))
        return (silent, "OTR_SilentComposite OK (%s%s) -> %s\n%s"
                % (mode, engine_tag, silent, "\n".join(report)))


__all__ = ["OTRSilentComposite", "normalize_to_silent_canonical",
           "count_audio_streams", "probe_video", "count_video_frames",
           "plan_timeline_segments", "assemble_silent_timeline"]
