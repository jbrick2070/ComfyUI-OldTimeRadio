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
import os
import shutil
import subprocess
import tempfile
import logging

log = logging.getLogger("OTR")


def _ffmpeg_bin(ffmpeg: str) -> str:
    return ffmpeg if (shutil.which(ffmpeg) or os.path.isfile(ffmpeg)) else ""


def _ffprobe_bin() -> str:
    return shutil.which("ffprobe") or ""


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


def _scale_filter(w, h, fps, *, sharpen, pad=True, in_label=None,
                  out_label=None, pre="", post=""):
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
    label (e.g. the floor ``,tpad=stop_mode=clone:...`` hold)."""
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


def _should_loop_fill(row, target_n):
    """S-A clip-fill: a real clip that UNDERRUNS its beat target loops to fill
    (the composite's own recommendation) instead of holding the last frame (the
    HuMo 177/434 murk). Decision reads the RAW engine-output ``frame_count``
    (``build_clip_manifest`` keeps it raw) vs the beat ``target_frame_count``;
    ANY shortfall fills. A frame-DIRECTORY clip (the 3D alpha handoff) is exempt
    -- it has its own dir encoder. Env ``OTR_CLIP_FILL=0`` restores the legacy
    held-last-frame behavior. Pure (a single os.path.isdir probe)."""
    if os.environ.get("OTR_CLIP_FILL", "1") == "0":
        return False
    real = int((row or {}).get("frame_count") or 0)
    tgt = int(target_n or 0)
    if real <= 0 or tgt <= 0 or real >= tgt:
        return False
    p = str((row or {}).get("path") or "")
    if p and os.path.isdir(p):
        return False
    return True


def _warn_clip_underrun(row, target_n, *, will_loop=False):
    """LOUD-warn (never raise -- no-loud-fail rule) when a real clip row carries
    far fewer on-disk frames than its beat target, so the composite WOULD hold
    the last frame for most of the beat. A loop-fill row is exempt -- once
    clip-fill is on (``will_loop``) the clip REPEATS to fill, so the held-frame
    murk can no longer happen and the LOUD warning is silenced. Pure except for
    the warning; clip-fill Piece 5."""
    try:
        frac = float(os.environ.get("OTR_CLIP_UNDERRUN_FRAC",
                                    _CLIP_UNDERRUN_FRAC))
    except (TypeError, ValueError):
        frac = _CLIP_UNDERRUN_FRAC
    if frac <= 0 or will_loop or (row or {}).get("loop"):
        return
    real = int((row or {}).get("frame_count") or 0)
    tgt = int(target_n or 0)
    if real > 0 and tgt > 0 and real < frac * tgt:
        log.warning(
            "[OTR.composite] CLIP UNDERRUN (LOUD): beat %s engine %r rendered "
            "%d frame(s) for a %d-frame target (%.0f%%) -- the composite will "
            "HOLD the last frame for the rest of the beat. A motion engine should "
            "loop/ping-pong-extend to the target (clip-fill); investigate %r.",
            (row or {}).get("shot_id") or (row or {}).get("beat_id"),
            (row or {}).get("engine_id"), real, tgt,
            100.0 * real / tgt, (row or {}).get("engine_id"))


def plan_timeline_segments(manifest, *, floor_available=False, floor_frames=0,
                           target_total_frames=None, fps=None):
    """Pure: the frame-accurate per-beat segment plan from a clip manifest.

    POSITION mode (every beat carries ``start_s`` AND a ``target_total_frames``
    master length is given): place each beat at ``round(start_s*fps)`` and
    floor/black gap-fill the head (the +intro shift), the inter-beat silences,
    and the tail (the closing theme) so the assembled length == the master length
    (the pre-mux A/V-sync guard). SEQUENTIAL mode (no start_s): concat the beats,
    then tail-fill to ``target_total_frames`` when given. Each beat is EXACTLY
    ``target_frame_count`` frames; a real on-disk clip is used, else the floor
    (timeline-aligned slice in sequential mode) or black. Returns ``(segments,
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
             bg_still_path=""):
        if int(n) <= 0:
            return
        segments.append({
            "order": len(segments), "shot_id": shot_id, "source": source,
            "path": path or "", "src_start_frame": int(src_start),
            "n_frames": int(n), "engine_id": engine_id, "loop": bool(loop),
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
        for r in sorted(rows, key=lambda x: float(x.get("start_s") or 0)):
            n = int(r.get("target_frame_count") or 0)
            start_frame = int(round(float(r.get("start_s") or 0) * fps))
            if start_frame > cursor:                       # head / inter-beat gap
                gap_n = start_frame - cursor
                emit(gap_src, "", gap_n, _floor_aligned(cursor, gap_n))
                cursor = start_frame
            if r.get("exists") and r.get("path"):
                _fill = _should_loop_fill(r, n)
                _warn_clip_underrun(r, n, will_loop=_fill)
                emit("clip", r.get("path"), n, 0, r.get("shot_id"),
                     r.get("engine_id"), loop=_fill,
                     bg_still_path=r.get("bg_still_path"))
            else:
                emit(gap_src, "", n, _floor_aligned(cursor, n),
                     r.get("shot_id"), r.get("engine_id"))
            cursor += n
    else:                                                  # SEQUENTIAL (legacy)
        for r in rows:
            n = int(r.get("target_frame_count") or 0)
            if r.get("exists") and r.get("path"):
                _fill = _should_loop_fill(r, n)
                _warn_clip_underrun(r, n, will_loop=_fill)
                emit("clip", r.get("path"), n, 0, r.get("shot_id"),
                     r.get("engine_id"), loop=_fill,
                     bg_still_path=r.get("bg_still_path"))
            elif floor_available:
                start = min(cursor, max(0, ff - n)) if ff else cursor
                emit("floor", "", n, start, r.get("shot_id"), r.get("engine_id"))
            else:
                emit("black", "", n, 0, r.get("shot_id"), r.get("engine_id"))
            cursor += n
    if target_total_frames is not None and int(target_total_frames) > cursor:
        # Tail to the master length + the credits post-roll. The §4D floor layer
        # lighten-blends the GREEN rolling credits on top; THIS segment is the
        # BACKDROP behind them. BUG-410 look-QA follow-on: hold the LAST drama
        # clip on screen (the 6/5 "credits over the scene" look) instead of the
        # dark CRT telemetry card -- a short clip holds its last frame (tpad
        # clone in _encode_segment), a long one plays its head. Fall back to the
        # procgen END-slice / black when there is no real clip.
        tail_n = int(target_total_frames) - cursor
        _clip_rows = [r for r in rows if r.get("exists") and r.get("path")]
        if positioned:
            _last_clip = (max(_clip_rows, key=lambda r: float(r.get("start_s") or 0))
                          if _clip_rows else None)
        else:
            _last_clip = _clip_rows[-1] if _clip_rows else None
        if _last_clip is not None:
            # LOOP the tail clip (operator 2026-06-17: "loop the ending video so
            # it's not static") -- the short last drama clip REPEATS to fill the
            # credits tail instead of tpad-cloning its final frame (the frozen
            # image). The green rolling credits still ride on top.
            emit("clip", _last_clip.get("path"), tail_n, 0,
                 _last_clip.get("shot_id"), _last_clip.get("engine_id"), loop=True,
                 bg_still_path=_last_clip.get("bg_still_path"))
        else:
            emit(gap_src, "", tail_n, _floor_aligned(cursor, tail_n))
        cursor = int(target_total_frames)
    return segments, cursor


def timeline_quality_report(manifest, segments):
    """S-A legibility floor (staged commit 1): a SEPARATE post-plan view that
    asserts each real-clip beat DELIVERS its full target span and flags any beat
    that would freeze. PURE -- never overwrites the raw manifest ``frame_count``
    (which stays engine-produced); it only reads the planned segments.

    Per clip beat: ``frame_count`` (raw engine output), ``target_frame_count``,
    ``delivered_frame_count`` (sum of its clip segments), ``looped`` (clip-fill
    engaged), and ``quality_status`` one of: ``ok`` (clip already >= target),
    ``looped_fill`` (underran, now loop-filled to target), ``held_last_frame``
    (underran with fill OFF -> the murk), ``no_clip_segment`` (fell to the
    still/floor spine -- not a clip-fill failure). ``delivered_frames_ok`` is
    True when NO beat is held_last_frame and every clip beat delivered ==
    target."""
    clips = {str(r.get("shot_id")): r
             for r in ((manifest or {}).get("clips") or [])
             if r.get("shot_id") and int((r or {}).get("target_frame_count") or 0) > 0}
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
        looped = any(s.get("loop") for s in segs)
        if not segs:
            status = "no_clip_segment"
        elif raw > 0 and raw < tgt:
            status = "looped_fill" if looped else "held_last_frame"
        else:
            status = "ok"
        beats.append({
            "shot_id": sid, "beat_id": r.get("beat_id"),
            "engine_id": r.get("engine_id"), "target_frame_count": tgt,
            "frame_count": raw, "delivered_frame_count": delivered,
            "looped": bool(looped), "quality_status": status})
        if status == "held_last_frame":
            ok_all = False
        elif segs and delivered != tgt:
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


def _seg_vf(w, h, fps, start_frame, sharpen=True):
    """The per-segment ``-vf`` chain: an optional source trim, then the SHARED
    scale chain (lanczos+unsharp when ``sharpen``), then the tpad last-frame hold.
    PRESERVES the ``trim -> scale/unsharp/pad/fps -> tpad`` ordering. ``sharpen``
    is True for a real clip (the soft-native fix) and False for the procgen-floor
    slice (byte-identical to the legacy chain)."""
    trim = ("trim=start_frame=%d,setpts=PTS-STARTPTS," % int(start_frame)) \
        if int(start_frame) > 0 else ""
    return _scale_filter(
        w, h, fps, sharpen=sharpen, pre=trim,
        post=",tpad=stop_mode=clone:stop_duration=3600")


def _color_args(out_path):
    # TAG bt709 identity (never matrix-convert) + canonical yuv420p H.264.
    return ["-vsync", "cfr", "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast", out_path]


def _encode_segment(fb, src, n_frames, seg_path, *, w, h, fps, start_frame=0,
                    loop=False, sharpen=True):
    """One canonical silent segment of EXACTLY ``n_frames`` from ``src`` (a clip,
    or the floor sliced at ``start_frame``): truncates a long source, holds the
    last frame (tpad clone) for a short one. FAIL CLOSED on ffmpeg error.

    ``loop=True`` stream-loops the input (-stream_loop -1) so a SHORT source
    REPEATS to fill ``n_frames`` -- the credits-tail backdrop keeps moving
    instead of freezing on the last frame (operator 2026-06-17). The tpad clone
    in _seg_vf stays as a safety but never triggers under an infinite loop.

    ``sharpen`` is True for a real engine clip (lanczos+unsharp soft-native fix)
    and False for the procgen-floor slice (NOT sharpened -- byte-identical)."""
    loop_args = ["-stream_loop", "-1"] if loop else []
    cmd = [fb, "-y", "-loglevel", "error"] + loop_args + ["-i", src, "-an",
           "-vf", _seg_vf(w, h, fps, start_frame, sharpen=sharpen),
           "-frames:v", str(int(n_frames))] + _color_args(seg_path)
    p = _run(cmd)
    if p.returncode != 0:
        raise ValueError("OTR_SilentComposite: segment encode failed (%s) :: %s"
                         % (os.path.basename(seg_path), p.stderr.strip()[:200]))


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
                             h=832, fps=25, ffmpeg="ffmpeg"):
    """Assemble a beat-ordered clip manifest into ONE always-silent canonical
    CFR video; FAIL CLOSED. Each beat is conformed to EXACTLY its
    ``target_frame_count`` (truncate if long, hold last frame if short); a
    missing clip is gap-filled from the floor (``base_video_path``) or black. The
    assembled frame count is asserted == the audio-derived budget total (the
    pre-mux A/V sync guard). Returns ``(out_path, report)``."""
    report: list = []
    fb = _ffmpeg_bin(ffmpeg)
    if not fb:
        raise ValueError("OTR_SilentComposite: ffmpeg not found (%r)" % ffmpeg)
    w, h, fps = _even(w), _even(h), max(1, int(fps))
    floor_ok = bool(base_video_path) and os.path.isfile(base_video_path)
    floor_frames = count_video_frames(base_video_path) if floor_ok else 0
    # Derive target_total from the manifest's audio-derived frame budget first
    # (the sum of target_frame_count across all shots, set by the render driver
    # from the ledger's per-beat durations). This is the correct master length
    # regardless of what the base video's container duration reports.
    # Fall back to base video duration only when the manifest carries no budget
    # (legacy / non-assemble path).
    mft_total = int((manifest or {}).get("total_target_frames") or 0)
    target_total = mft_total if mft_total > 0 else None
    if floor_ok:
        # PRODUCTION RESTORE (2026-06-10): the procgen base runs the FULL
        # episode length -- opening theme pad + drama + the rolling-credits
        # post-roll under the closing theme. The beats-only budget CUT the
        # video at the last drama beat, so the credits roll vanished from the
        # rewired chain. Extend the assembled length to the base duration so
        # the tail segment (sliced from the procgen END by the planner)
        # restores the credits. The mux gate stays safe: the base was rendered
        # to the master mix length, so v_dur <= a_dur + tol still holds.
        # The cap is the MASTER MIX duration -- NOT the base's video length
        # (the procgen runs ~20s past the master with its own silent
        # post-roll) and NOT the base's embedded audio either (the encoder
        # silence-pads it to the video length; both live catches 2026-06-10:
        # 123.5s video vs 103.6s master = the terminal mux REFUSED). The real
        # master WAV lives in the base's sibling audio dir (the per-episode
        # layout) -- probe the LONGEST *_master.wav there (a stub copy can
        # coexist after the rename); fall back to the base audio stream, then
        # the container. -1 frame headroom absorbs rounding.
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
            base_total = max(0, int(round(master_dur * fps)) - 1)
            if base_total > 0 and (target_total is None
                                   or base_total > target_total):
                if target_total is not None:
                    report.append(
                        "credits post-roll restored: tail %d -> %d frames "
                        "(procgen end-slice under the closing theme)"
                        % (target_total, base_total))
                target_total = base_total
        # BUG-LOCAL-410: the master-mix cap above only carries the credits that
        # fit UNDER the closing theme; the procgen floor renders ~20s MORE of
        # SCROLLING credits past the master (the rolling-credits post-roll).
        # Extend the assembled length to the floor's FULL video frame count so
        # the scroll survives instead of being cut to the static title card.
        # The mux now permits this intentional SILENT credits tail
        # (OTR_MasterAudioMux: v <= a + OTR_MAX_CREDITS_TAIL_S); the audio stays
        # byte-identical (the credits roll in silence after the theme ends).
        if floor_frames > 0 and (target_total is None
                                 or int(floor_frames) > int(target_total)):
            report.append(
                "BUG-410 credits scroll restored: tail %s -> %d frames "
                "(full procgen post-roll)" % (target_total, int(floor_frames)))
            target_total = int(floor_frames)
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
                            b["frame_count"], b["target_frame_count"])
                         for b in qa["underran"])))
    if not qa["delivered_frames_ok"]:
        _held = sum(1 for b in qa["beats"]
                    if b["quality_status"] == "held_last_frame")
        report.append(
            "LEGIBILITY (LOUD): %d beat(s) did NOT deliver full target frames "
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
                _encode_segment(fb, seg["path"], seg["n_frames"], seg_path,
                                w=w, h=h, fps=fps, start_frame=0,
                                loop=bool(seg.get("loop")), sharpen=True)
            elif kind == "floor":
                # The procgen floor slice -> NOT sharpened (byte-identical chain).
                _encode_segment(fb, base_video_path, seg["n_frames"], seg_path,
                                w=w, h=h, fps=fps, start_frame=seg["src_start_frame"],
                                sharpen=False)
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
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

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
                  ffmpeg="ffmpeg", output_path="", gate_in="", clip_manifest_json="{}"):
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
        try:
            if assemble:
                silent, report = assemble_silent_timeline(
                    manifest, base_video_path, out, w=int(canvas_w),
                    h=int(canvas_h), fps=eff_fps, ffmpeg=ffmpeg,
                )
            else:
                silent, report = normalize_to_silent_canonical(
                    base_video_path, out, w=int(canvas_w), h=int(canvas_h),
                    fps=int(fps), ffmpeg=ffmpeg,
                )
        except ValueError as exc:
            log.error("[OTR_SilentComposite] %s", exc)
            return ("", f"error: {exc}")
        for line in report:
            log.info("[OTR_SilentComposite] %s", line)
        mode = "assemble" if assemble else "single-base"
        return (silent, "OTR_SilentComposite OK (%s) -> %s\n%s"
                % (mode, silent, "\n".join(report)))


__all__ = ["OTRSilentComposite", "normalize_to_silent_canonical",
           "count_audio_streams", "probe_video", "count_video_frames",
           "plan_timeline_segments", "assemble_silent_timeline"]
