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
    color_transfer, fps}``. Raises a NAMED GraphExecutionError on an ffprobe
    failure (a missing ffprobe is a broken install, same class as the encoder's
    missing-ffmpeg)."""
    import json as _json
    import subprocess as _sp

    from . import wrapper_bridge as _wb
    try:
        proc = _sp.run(
            [ffprobe, "-v", "error", "-show_entries",
             "stream=codec_type,codec_name,pix_fmt,color_primaries,"
             "color_transfer,color_space,avg_frame_rate,r_frame_rate",
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
    }


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
        }

    @staticmethod
    def _comfy_root():
        """ComfyUI install root (…/custom_nodes/OTR/nodes/_otr_video_engines ->
        up four)."""
        here = os.path.abspath(__file__)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
        return os.path.dirname(os.path.dirname(repo_root))


__all__ = [
    "_parse_fps", "ffprobe_clip_fields", "validate_silent_clip_contract",
    "WanInitImageMixin", "_WAN_DEFAULT_NEGATIVE",
]
