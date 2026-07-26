"""Shared PURE helpers for the two in-process Wan motion engines.

``wan_i2v`` (14B I2V; ``WanImageToVideo`` graph) and ``wan_ti2v`` (5B TI2V;
``Wan22ImageToVideoLatent`` graph) share the SAME image/aspect/dims/clip-contract
mechanics but have DIFFERENT loaders, node candidates and graphs. Per GO_FORWARD
section 4A ("share only pure dims/aspect/materialize/canonicalize helpers; keep
loaders + node candidates + graph SEPARATE") this module factors ONLY the pure
helpers, via:

* the module-level M7 silent-clip-contract functions (``_parse_fps`` /
  ``ffprobe_clip_fields`` / ``validate_silent_clip_contract``) -- engine-agnostic
  ffprobe proof of the emitted mp4's color/stream contract; and
* :class:`WanInitImageMixin` -- the pure init-image staging (N9 no-stretch / S7
  per-shot-seed name / S10 Pillow-required), aspect resolution, canvas dims, the
  deterministic build-render-request (V-7), the offline aux-loader file resolver
  (M6), and the self-declared clip dict.

Each engine keeps its OWN ``_ckpt_path`` / ``_loader_names`` / ``_loader_mode`` /
``_node_candidates`` / ``_build_graph`` / ``assert_usable`` / ``render_clip``.

Cold-import clean (V-12): module scope imports only the stdlib + motion_common
(itself dep-free). Pillow / ffprobe / wrapper_bridge are imported LAZILY inside the
methods, never here. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import os

from . import motion_common as _MC

_WAN_DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, distorted, watermark, text, static")


# --------------------------------------------------------------------------- #
# M7 silent-clip contract proof (GO_FORWARD 4A) -- ffprobe the emitted mp4 and
# PROVE the color/stream contract before the mux trusts the self-declared dict.
# Module-level + engine-agnostic; both Wan engines (and any future probe) reuse
# the SAME proof. Re-exported from eng_wan_i2v for back-compatible test imports.
# --------------------------------------------------------------------------- #
def _parse_fps(rate):
    """An ffprobe ``num/den`` frame-rate string -> rounded int fps (0 if
    unparseable / zero denominator)."""
    try:
        num, den = str(rate).split("/")
        den = float(den)
        return int(round(float(num) / den)) if den else 0
    except (ValueError, ZeroDivisionError, AttributeError):
        return 0


def ffprobe_clip_fields(path, *, ffprobe="ffprobe"):
    """Probe a clip's stream + color contract (read-only). Returns
    ``{codec_types, video_codec, pix_fmt, color_space, color_primaries,
    color_transfer, fps, width, height}``. Raises a NAMED GraphExecutionError
    on an ffprobe failure (a missing ffprobe is a broken install, same class as
    the encoder's missing-ffmpeg).

    ``width``/``height`` joined the query for multi-clip coverage chunk 6
    (2026-07-26). They cost nothing -- they come out of the SAME stream read --
    and the assembler needs them: segments of one beat that differ in canvas
    concatenate into a clip that changes size partway through, which no
    downstream consumer expects and the mux will not tell you about. They are
    absent from an audio-only stream, so they read as 0 rather than raising.

    This helper deliberately does NOT count frames: it runs on EVERY emitted
    clip, and counting means decoding. Use :func:`ffprobe_counted_frames` at
    the assembly boundary, where the cost is paid once and the truth matters."""
    import json as _json
    import subprocess as _sp

    from . import wrapper_bridge as _wb
    try:
        proc = _sp.run(
            [ffprobe, "-v", "error", "-show_entries",
             "stream=codec_type,codec_name,pix_fmt,color_primaries,"
             "color_transfer,color_space,avg_frame_rate,r_frame_rate,"
             "width,height",
             "-of", "json", path],
            stdout=_sp.PIPE, stderr=_sp.PIPE)
    except FileNotFoundError as exc:
        raise _wb.GraphExecutionError("ffprobe not found: %s" % exc)
    if proc.returncode != 0:
        raise _wb.GraphExecutionError(
            "ffprobe failed for %r: %s"
            % (path, proc.stderr.decode("utf-8", "replace")[:300]))
    data = _json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    streams = data.get("streams") or []
    vids = [s for s in streams if s.get("codec_type") == "video"]
    v = vids[0] if vids else {}
    rate = v.get("avg_frame_rate") or v.get("r_frame_rate")
    return {
        "codec_types": [s.get("codec_type") for s in streams],
        "video_codec": v.get("codec_name"),
        "pix_fmt": v.get("pix_fmt"),
        "color_space": v.get("color_space"),
        "color_primaries": v.get("color_primaries"),
        "color_transfer": v.get("color_transfer"),
        "fps": _parse_fps(rate),
        "width": int(v.get("width") or 0),
        "height": int(v.get("height") or 0),
    }


def ffprobe_counted_frames(path, *, ffprobe="ffprobe"):
    """The DECODED frame count of ``path`` -- the only count that cannot lie.

    Multi-clip coverage chunk 6 (2026-07-26). A container's ``nb_frames``
    header is metadata: it is what the muxer WROTE, not what a decoder will
    read back, and a concatenated clip can carry a header that disagrees with
    its own picture data. Once a beat is assembled from several rendered
    segments, "did the assembly produce exactly the frames the plan promised"
    is the whole question, and a header cannot answer it. ``-count_frames``
    decodes and counts.

    THIS IS EXPENSIVE BY DESIGN -- it decodes the file. Call it at the
    assembly/verification boundary, once per assembled beat. Never per clip in
    the render loop; :func:`ffprobe_clip_fields` is the cheap per-clip probe.

    Raises a NAMED GraphExecutionError when ffprobe is missing, fails, or
    reports something that is not a frame count. It never guesses: an
    unreadable count is a failed verification, not a zero.
    """
    import json as _json
    import subprocess as _sp

    from . import wrapper_bridge as _wb
    try:
        proc = _sp.run(
            [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
             "-show_entries", "stream=nb_read_frames", "-of", "json", path],
            stdout=_sp.PIPE, stderr=_sp.PIPE)
    except FileNotFoundError as exc:
        raise _wb.GraphExecutionError("ffprobe not found: %s" % exc)
    if proc.returncode != 0:
        raise _wb.GraphExecutionError(
            "ffprobe -count_frames failed for %r: %s"
            % (path, proc.stderr.decode("utf-8", "replace")[:300]))
    data = _json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise _wb.GraphExecutionError(
            "ffprobe -count_frames found NO video stream in %r -- there is "
            "nothing to count, so the assembled beat cannot be verified" % path)
    raw = streams[0].get("nb_read_frames")
    try:
        counted = int(raw)
    except (TypeError, ValueError):
        raise _wb.GraphExecutionError(
            "ffprobe -count_frames reported nb_read_frames=%r for %r, which is "
            "not a count. NO GUESS -- an unreadable count is a failed "
            "verification, not a zero." % (raw, path))
    if counted < 0:
        raise _wb.GraphExecutionError(
            "ffprobe -count_frames reported %d frames for %r" % (counted, path))
    return counted


def assemble_beat_segments(segments, out_path, *, expect_frames, expect_fps,
                           ffmpeg="ffmpeg", ffprobe="ffprobe"):
    """Assemble rendered SEGMENTS into ONE beat clip -- TRANSACTIONALLY.

    Multi-clip coverage chunk 6d (2026-07-26). ``segments`` is a sequence of
    ``(path, drop_head, keep_frames)`` in play order. Returns ``out_path``.

    TRANSACTIONAL means the caller never receives a half-right beat. Three
    things are proven about the assembled file BEFORE it is handed back, and
    any of them failing removes the output and raises:

    * every segment shares ONE canvas -- segments that differ in size
      concatenate into a clip that changes dimensions partway through, which
      no downstream consumer expects and the mux will not mention;
    * the assembled clip DECODES to exactly ``expect_frames`` frames, counted
      with ``-count_frames`` rather than read off a container header the muxer
      wrote;
    * it still honours the silent-clip contract (bt709 / yuv420p / one video
      stream / no audio), because an assembled beat is a CanonicalClip like
      any other and the mux is entitled to assume that.

    The partitioner refuses a plan whose segments cannot sum to the beat
    exactly. This is the other end of that promise: it proves the RENDER kept
    it. Without the count, "the plan was exact" and "the video is the right
    length" are two different claims and only one of them was ever checked.
    """
    import os as _os

    from . import wrapper_bridge as _wb
    rows = [(str(p), int(d), int(k)) for p, d, k in segments]
    if not rows:
        raise _wb.GraphExecutionError(
            "beat assembly was handed NO segments")

    # SHAPE, not just SIZE (2026-07-26 QA panel). Differing fps desynchronises
    # the concatenated timestamps and differing pixel formats make the filter
    # graph convert mid-beat; both are as invisible in a header as a size
    # change and as visible on screen.
    shapes = {}
    for path, _drop, _keep in rows:
        fields = ffprobe_clip_fields(path, ffprobe=ffprobe)
        key = (fields["width"], fields["height"], fields["fps"],
               fields["pix_fmt"])
        shapes.setdefault(key, []).append(path)
    if len(shapes) > 1:
        raise _wb.GraphExecutionError(
            "beat assembly refuses segments of MIXED shape %r -- the assembled "
            "clip would change canvas, frame rate or pixel format partway "
            "through the beat. NO FALLBACK (no silent rescale or resample)."
            % ({"%dx%d@%sfps %s" % k: v for k, v in shapes.items()},))

    try:
        # The concat itself is INSIDE the transaction (2026-07-26 QA panel).
        # It was outside, so an ffmpeg failure left whatever it had partially
        # written sitting at ``out_path`` -- which is exactly the "a beat that
        # failed must not survive on disk" promise this block makes, broken by
        # the one failure most likely to produce a partial file.
        _wb.run_ffmpeg(_wb.ffmpeg_concat_segments_cmd(rows, out_path,
                                                      ffmpeg=ffmpeg))
        if not _os.path.exists(out_path) or _os.path.getsize(out_path) <= 0:
            raise _wb.GraphExecutionError(
                "beat assembly produced no usable clip at %r (ffmpeg exited 0 "
                "but wrote %s)" % (out_path,
                                   "no file" if not _os.path.exists(out_path)
                                   else "0 bytes"))
        counted = ffprobe_counted_frames(out_path, ffprobe=ffprobe)
        if counted != int(expect_frames):
            raise _wb.GraphExecutionError(
                "beat assembly produced %d frame(s) but the coverage plan "
                "promised %d. The segments rendered and the beat still came "
                "out the wrong length, so the clip would drift against its own "
                "audio. NO FALLBACK -- a header would have said this was fine."
                % (counted, int(expect_frames)))
        validate_silent_clip_contract(
            ffprobe_clip_fields(out_path, ffprobe=ffprobe), expect_fps)
    except Exception:
        # TRANSACTIONAL: a beat that failed verification must not survive on
        # disk, or the next pass finds a plausible file and trusts it.
        try:
            _os.unlink(out_path)
        except OSError:
            pass
        raise
    return out_path


def validate_silent_clip_contract(fields, expected_fps):
    """PURE: assert a probed clip honours the OTR silent-clip contract -- EXACTLY
    one video stream, NO audio (V-1), h264 / yuv420p / bt709 colorspace, and the
    engine fps. The primaries/transfer tags are asserted only when ffprobe
    surfaces them (libx264 + yuv420p reliably reports colorspace, not always the
    primaries/transfer); ``unknown`` counts as unset. Raises GraphExecutionError
    NAMED on any mismatch."""
    from . import wrapper_bridge as _wb
    types = list(fields.get("codec_types") or [])
    if "audio" in types:
        raise _wb.GraphExecutionError(
            "silent-clip contract: clip carries an AUDIO stream (V-1: only the "
            "mux adds audio); streams=%r" % types)
    if types.count("video") != 1:
        raise _wb.GraphExecutionError(
            "silent-clip contract: expected EXACTLY one video stream, got %r"
            % types)
    checks = (("video_codec", "h264"), ("pix_fmt", "yuv420p"),
              ("color_space", "bt709"))
    for key, want in checks:
        got = fields.get(key)
        if got != want:
            raise _wb.GraphExecutionError(
                "silent-clip contract: %s=%r, expected %r" % (key, got, want))
    for key in ("color_primaries", "color_transfer"):
        got = fields.get(key)
        if got and got not in ("bt709", "unknown"):
            raise _wb.GraphExecutionError(
                "silent-clip contract: %s=%r, expected bt709 or unset" % (key, got))
    fps = int(fields.get("fps") or 0)
    if fps != int(expected_fps):
        raise _wb.GraphExecutionError(
            "silent-clip contract: fps=%r, expected %r" % (fps, int(expected_fps)))


# --------------------------------------------------------------------------- #
# Pure init-image / aspect / dims / clip helpers shared by both Wan engines.
# No loaders, no node candidates, no graph -- those stay per-engine (4A CUT).
# --------------------------------------------------------------------------- #
class WanInitImageMixin:
    """Mixin of the PURE Wan helpers both engines share. CPU-testable; no heavy
    import. Subclasses provide ``name`` / ``family`` / ``target_fps`` and the
    loader+graph methods."""

    # ---- offline aux-loader file resolution (M6) ----
    def _resolve_model_file(self, categories, name, env_dir):
        """Full path of a model file: an explicit dir override (``env_dir``)
        wins, then ComfyUI ``folder_paths`` (honours extra_model_paths.yaml),
        then the standard ``models/<category>/<name>`` layout. Returns ``None``
        when absent everywhere. The offline invariant means NO runtime fetch --
        a missing file is fail-closed, never silently downloaded."""
        base = os.environ.get(env_dir)
        if base:
            cand = os.path.join(base, name)
            return cand if os.path.exists(cand) else None
        return self._resolve_model_file_by_token(categories, name)

    def _resolve_model_file_by_token(self, categories, name):
        """Where the LOADER would find ``name`` -- ``folder_paths`` then the
        standard ``models/<category>/`` layout, IGNORING any ``*_DIR`` override.

        Split out of :meth:`_resolve_model_file` (which still calls it, so no
        existing caller's behaviour changes) because the two questions are not
        the same question. A ``*_DIR`` override wins there on file EXISTENCE
        ALONE and never consults ``folder_paths`` -- but the graphs hand their
        loader nodes a BARE TOKEN, and ComfyUI resolves that through
        ``folder_paths``. So a directory override can satisfy a preflight while
        being invisible to the loader that actually loads the weights.

        An adapter that wants to CHECK its override against reality asks this;
        an adapter that just wants the historical answer keeps asking
        ``_resolve_model_file``."""
        for category in categories:
            try:
                import folder_paths            # ComfyUI runtime only
                hit = folder_paths.get_full_path(category, name)
                if hit:
                    return hit
            except Exception:                   # noqa: BLE001 -- no ComfyUI (tests)
                pass
            cand = os.path.join(self._comfy_root(), "models", category, name)
            if os.path.exists(cand):
                return cand
        return None

    def _missing_loaders(self):
        """(label, basename) for every required aux loader file absent on disk
        (the UNET is checked separately by each engine's ``_installed``)."""
        return [(label, name)
                for (label, cats, name, env) in self._aux_loader_files()
                if self._resolve_model_file(cats, name, env) is None]

    # ---- init image ----
    def _init_image_ref(self, request):
        """The init image path from ``asset_refs{init_image}`` (or "")."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        assets = get("asset_refs") or {}
        if isinstance(assets, dict):
            return assets.get("init_image") or ""
        return ""

    def _aspect_plan(self, request):
        """The pad / crop / fit transform mapping the init image into the canvas
        with ONE uniform scale (never a stretch, pre-mortem N9). Returns ``None``
        when the canvas or init dims are absent (the GPU smoke probes the real
        init dims), but still validates the policy token fail-closed."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        canvas = get("canvas") or {}
        c_get = canvas.get if isinstance(canvas, dict) else (
            lambda k, d=None: getattr(canvas, k, d))
        dst_w = int(c_get("w", 0) or 0)
        dst_h = int(c_get("h", 0) or 0)
        policy = (c_get("aspect_policy", _MC.DEFAULT_ASPECT_POLICY)
                  or _MC.DEFAULT_ASPECT_POLICY)
        src_w = int(get("init_w", 0) or 0)
        src_h = int(get("init_h", 0) or 0)
        if min(dst_w, dst_h, src_w, src_h) <= 0:
            _MC.assert_aspect_policy(policy)     # validate the token even unsized
            return None
        return _MC.resolve_aspect_transform(src_w, src_h, dst_w, dst_h, policy)

    def _aspect_policy(self, request):
        """The canvas aspect policy (default ``pad``); fail-closed on an unknown
        token (an implicit stretch is forbidden, N9)."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        canvas = get("canvas") or {}
        c_get = canvas.get if isinstance(canvas, dict) else (
            lambda k, d=None: getattr(canvas, k, d))
        policy = (c_get("aspect_policy", _MC.DEFAULT_ASPECT_POLICY)
                  or _MC.DEFAULT_ASPECT_POLICY)
        _MC.assert_aspect_policy(policy)
        return policy

    def _dims(self, request):
        """(width, height) from the request canvas (landscape default 832x480)."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        canvas = get("canvas") or {}
        c_get = canvas.get if isinstance(canvas, dict) else (
            lambda k, d=None: getattr(canvas, k, d))
        return int(c_get("w", 0) or 0) or 832, int(c_get("h", 0) or 0) or 480

    def _staged_init_name(self, request, width, height):
        """S7: the staged init-image basename, unique per (shot, seed, dims) yet
        DETERMINISTIC for a given (shot, seed) so the render-twice determinism
        contract (V-7) holds. Shared prefix ``otr_wan_init_`` -- shot+seed+dims
        keep the two engines' staged inits from clobbering each other."""
        import re
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        shot = str(get("shot_id") or get("request_id") or "wan")
        seeds = get("seed_bundle") or {}
        seed = (seeds.get("request_seed") if isinstance(seeds, dict)
                else getattr(seeds, "request_seed", 0)) or 0
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", shot)[:48]
        return "otr_wan_init_%s_s%d_%dx%d.png" % (
            safe, int(seed), int(width), int(height))

    def _materialize_init_image(self, request, src_path, width, height):
        """Pad / crop the init image into the (width, height) canvas with ONE
        uniform scale per the aspect policy (N9: never a silent stretch), then
        stage the DERIVED width x height image under a per-shot/seed name (S7) and
        return its basename. Uses the TRUE on-disk dims. Pillow is REQUIRED (S10):
        a missing Pillow OR an unreadable source fails LOUD. Fail-closed NAMED on a
        missing source."""
        from . import wrapper_bridge as _wb
        if not src_path or not os.path.exists(src_path):
            raise _wb.GraphExecutionError(
                "%s init image missing: %r" % (self.name, src_path))
        policy = self._aspect_policy(request)
        try:
            from PIL import Image
        except ImportError as exc:               # S10: Pillow is mandatory
            raise _wb.GraphExecutionError(
                "%s requires Pillow to materialize the init image into the canvas "
                "without a silent stretch (N9); install Pillow (%s)"
                % (self.name, exc))
        try:
            img = Image.open(src_path).convert("RGB")
        except Exception as exc:                 # noqa: BLE001 -- unreadable source
            raise _wb.GraphExecutionError(
                "%s init image %r is unreadable (S10: no silent raw-stage "
                "fallback): %s" % (self.name, src_path, exc))
        sw, sh = img.size
        plan = _MC.resolve_aspect_transform(sw, sh, int(width), int(height), policy)
        resized = img.resize((plan["scaled_w"], plan["scaled_h"]), Image.LANCZOS)
        if policy == "crop":
            cx, cy = plan["crop_x"], plan["crop_y"]
            canvas = resized.crop((cx, cy, cx + int(width), cy + int(height)))
        else:                                    # pad | fit -> black letter/pillarbox
            canvas = Image.new("RGB", (int(width), int(height)), (0, 0, 0))
            canvas.paste(resized, (plan["pad_x"], plan["pad_y"]))
        dst_dir = _wb.comfy_input_dir()
        os.makedirs(dst_dir, exist_ok=True)
        name = self._staged_init_name(request, width, height)
        canvas.save(os.path.join(dst_dir, name))
        return name

    def _build_render_request(self, request):
        """Pure: the normalized inference request the Wan wrapper consumes.
        Deterministic (seed + aspect plan flow straight through) -- the
        render-twice determinism contract (V-7)."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        timing = get("timing") or {}
        t_get = timing.get if isinstance(timing, dict) else (
            lambda k, d=None: getattr(timing, k, d))
        seeds = get("seed_bundle") or {}
        s_get = seeds.get if isinstance(seeds, dict) else (
            lambda k, d=None: getattr(seeds, k, d))
        return {
            "init_image": self._init_image_ref(request),
            "fps": int(self.target_fps),
            "target_frame_count": int(t_get("target_frame_count", 0) or 0),
            "seed": int(s_get("request_seed", 0) or 0),
            "aspect_plan": self._aspect_plan(request),
        }

    def _clip_from_raw(self, raw, request):
        """The self-declared clip dict canonicalize() returns. ``engine_id`` /
        ``family`` track the concrete engine via ``self``; the M7 ffprobe proof in
        render_clip has already PROVEN these fields on the real mp4."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        raw = raw or {}
        return {
            "clip_id": get("shot_id") or get("request_id") or ("%s_clip" % self.name),
            "type": "video", "path": raw.get("out_path", ""),
            "container": "mp4", "codec": "h264", "pixel_format": "yuv420p",
            "fps": int(self.target_fps),
            "frame_count": int(raw.get("frame_count", 0) or 0),
            "has_audio": False,            # V-1: only OTR_MasterAudioMux emits audio
            "color_primaries": "bt709", "transfer": "bt709", "matrix": "bt709",
            "engine_id": self.name, "family": self.family,
            # PASS-PM / NEWBUG-1 (2026-07-20): the REAL render-window NVML peak the
            # engine threaded into raw (None off the GPU box / when not measured).
            # The render batch reads clip.get("vram_peak_mb") for the manifest receipt.
            "vram_peak_mb": raw.get("vram_peak_mb"),
        }

    @staticmethod
    def _comfy_root():
        """ComfyUI install root (.../custom_nodes/OTR/nodes/_otr_video_engines ->
        up four)."""
        here = os.path.abspath(__file__)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
        return os.path.dirname(os.path.dirname(repo_root))


__all__ = [
    "_parse_fps", "ffprobe_clip_fields", "ffprobe_counted_frames",
    "assemble_beat_segments", "validate_silent_clip_contract",
    "WanInitImageMixin", "_WAN_DEFAULT_NEGATIVE",
]
