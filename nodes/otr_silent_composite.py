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
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"fps={fps}"
    )
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

    def emit(source, path, n, src_start, shot_id=None, engine_id=None):
        if int(n) <= 0:
            return
        segments.append({
            "order": len(segments), "shot_id": shot_id, "source": source,
            "path": path or "", "src_start_frame": int(src_start),
            "n_frames": int(n), "engine_id": engine_id})

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
                emit("clip", r.get("path"), n, 0, r.get("shot_id"), r.get("engine_id"))
            else:
                emit(gap_src, "", n, _floor_aligned(cursor, n),
                     r.get("shot_id"), r.get("engine_id"))
            cursor += n
    else:                                                  # SEQUENTIAL (legacy)
        for r in rows:
            n = int(r.get("target_frame_count") or 0)
            if r.get("exists") and r.get("path"):
                emit("clip", r.get("path"), n, 0, r.get("shot_id"), r.get("engine_id"))
            elif floor_available:
                start = min(cursor, max(0, ff - n)) if ff else cursor
                emit("floor", "", n, start, r.get("shot_id"), r.get("engine_id"))
            else:
                emit("black", "", n, 0, r.get("shot_id"), r.get("engine_id"))
            cursor += n
    if target_total_frames is not None and int(target_total_frames) > cursor:
        # Tail to the master length. With a floor available this slice is the
        # END of the procgen -- the ROLLING-CREDITS post-roll riding under the
        # closing theme (production restore 2026-06-10).
        tail_n = int(target_total_frames) - cursor
        emit(gap_src, "", tail_n, _floor_aligned(cursor, tail_n))
        cursor = int(target_total_frames)
    return segments, cursor


def _seg_vf(w, h, fps, start_frame):
    trim = ("trim=start_frame=%d,setpts=PTS-STARTPTS," % int(start_frame)) \
        if int(start_frame) > 0 else ""
    return (
        "%sscale=%d:%d:force_original_aspect_ratio=decrease,"
        "pad=%d:%d:(ow-iw)/2:(oh-ih)/2:color=black,fps=%d,"
        "tpad=stop_mode=clone:stop_duration=3600" % (trim, w, h, w, h, fps)
    )


def _color_args(out_path):
    # TAG bt709 identity (never matrix-convert) + canonical yuv420p H.264.
    return ["-vsync", "cfr", "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast", out_path]


def _encode_segment(fb, src, n_frames, seg_path, *, w, h, fps, start_frame=0):
    """One canonical silent segment of EXACTLY ``n_frames`` from ``src`` (a clip,
    or the floor sliced at ``start_frame``): truncates a long source, holds the
    last frame (tpad clone) for a short one. FAIL CLOSED on ffmpeg error."""
    cmd = [fb, "-y", "-loglevel", "error", "-i", src, "-an",
           "-vf", _seg_vf(w, h, fps, start_frame),
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
    segments, total = plan_timeline_segments(
        manifest, floor_available=floor_ok, floor_frames=floor_frames,
        target_total_frames=target_total, fps=fps)
    if not segments or total <= 0:
        raise ValueError("OTR_SilentComposite: manifest has no renderable beats")
    workdir = tempfile.mkdtemp(prefix="otr_assemble_")
    try:
        seg_paths = []
        for seg in segments:
            seg_path = os.path.join(workdir, "seg_%04d.mp4" % seg["order"])
            kind = seg["source"]
            if kind == "clip" and not os.path.isfile(seg["path"]):
                kind = "floor" if floor_ok else "black"   # vanished clip -> gap-fill
            if kind == "clip":
                _encode_segment(fb, seg["path"], seg["n_frames"], seg_path,
                                w=w, h=h, fps=fps, start_frame=0)
            elif kind == "floor":
                _encode_segment(fb, base_video_path, seg["n_frames"], seg_path,
                                w=w, h=h, fps=fps, start_frame=seg["src_start_frame"])
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
