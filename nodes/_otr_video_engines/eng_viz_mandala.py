r"""``viz_mxc_mandala`` -- the Cosmic Radio Mandala, a pycairo audio-reactive
tuning-eye mandala (operator 2026-06-30, kibitz-hardened r1-r4).

A SEPARATE selectable engine alongside ``viz_mxc_cpu`` (the zero-dep numpy/PIL
rainbow scope stays as the zero-dep alternate; this is the pycairo UPGRADE the
operator asked for after finding the PIL rainbow underwhelming). Pure CPU vector
graphics (pycairo -- cross-platform, no GPU/shaders) painting a centered tuning
eye + radio-dial concentric rings/spokes + an outer solid spectrum band, in a
muted iridescent palette on a deep radio-bronze field. The CRT scanlines +
vignette + film grain are applied as a POST-PROCESS over the rendered RGB frame
(:func:`nodes._otr_shared.scope_draw.apply_crt_post_rgb`) -- native-cairo CRT
glue was cut for v1 (PIL roundtrip reuses the SAME machinery as
``paint_frame``/``paint_rainbow_frame``).

AUDIO-OPTIONAL (``required_inputs=()``): reacts to audio where present and
idles a slow mandala on silence -- ALSO the no-image floor for
retired_role_a/background (fits every role by capability, C2). ``accepts_still=
False`` -> mints NO still, so it never triggers an image model on a non-audio
slot. has_audio is always False (only OTR_MasterAudioMux adds audio,
test_audio_byte_identical invariant). NO FALLBACKS (fallback_engine=None);
assert_usable fails LOUD (probes BOTH pycairo AND ffmpeg, separate messages).
Cold-import clean (V-12: cairo / soundfile / PIL / scope_draw imported lazily
inside load/assert_usable/render_clip -- cairo is NEVER at module scope,
matching pycairo NOT being added to the main requirements, so a box without
system libcairo never breaks any OTHER engine's install). UTF-8, no BOM,
ASCII source.

Ported from the operator-approved prototype
(docs/2026-06-30-viz-rainbow/mandala_proto.py); see
docs/2026-06-30-viz-rainbow/MANDALA_ENGINE_PLAN.md for the full grounded spec
(kibitz r1-r4 CONVERGED). Config (env): ``OTR_FFMPEG`` ffmpeg path.
"""
from __future__ import annotations

import logging
import os

from . import motion_common as _MC
from .registry import EngineUnusable, EngineUsabilityReason, register

_LOG = logging.getLogger("OTR.video.viz_mxc_mandala")


@register
class VizMxcMandalaEngine:
    """The pycairo Cosmic Radio Mandala engine (engine_id ``viz_mxc_mandala``)."""

    name = "viz_mxc_mandala"
    family = "abstract"
    #: UI-sort metadata only (C2: capability is the eligibility rule). Listed in
    #: every role so the dropdown offers it everywhere; required_inputs=() makes it
    #: fit every role BY CAPABILITY.
    roles = ("announcer_visual", "music_visual", "character_video")
    default_roles = ()                  # opt-in selectable only; never an auto-default
    commercial_clean = True             # own code + pycairo (LGPL/MPL) + ffmpeg encode
    requires_flag = None                # registry IS the menu; no flag gate
    #: AUDIO-OPTIONAL -- () fits every role by capability (mirrors viz_mxc_cpu);
    #: render_clip idles a slow mandala when no audio_ref is present.
    required_inputs = ()
    #: Opt OUT of the still coverage gate: audio-reactive, mints NO image, so a
    #: non-audio slot never triggers an image model (the operator's z_image complaint).
    accepts_still = False
    render_aspect = "wide"              # 16:9; no portrait geometry branch
    declared_isolation = _MC.ISOLATION_IN_PROCESS
    target_fps = 25                     # HARD-LOCK (matches the overlay + mux)
    engine_version = "1"
    fallback_engine = None              # NO FALLBACKS: a failed beat fails LOUD

    def __init__(self):
        self._loaded = False

    # ---- residency (none: CPU pycairo + ffmpeg only) ----
    def load(self):
        try:
            import cairo  # noqa: F401
        except ImportError as exc:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "viz_mxc_mandala needs pycairo (pip install pycairo)",
                kind="video") from exc
        from .._otr_shared import scope_draw as _sd
        if not _sd.find_ffmpeg(os.environ.get("OTR_FFMPEG", "ffmpeg")):
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "viz_mxc_mandala needs ffmpeg on PATH (or set OTR_FFMPEG)",
                kind="video")
        self._loaded = True

    def unload(self):
        self._loaded = False

    def prepare(self, host_caps, profile, session_ctx):
        return {"engine_id": self.name}

    def teardown(self, prepared):
        return None

    # ---- usability (fail LOUD; no NVML / weights / node gate) ----
    def assert_usable(self, host_caps, profile, request_template=None):
        try:
            import cairo  # noqa: F401
        except ImportError as exc:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "viz_mxc_mandala needs pycairo (pip install pycairo)",
                kind="video") from exc
        from .._otr_shared import scope_draw as _sd
        if not _sd.find_ffmpeg(os.environ.get("OTR_FFMPEG", "ffmpeg")):
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "viz_mxc_mandala needs ffmpeg on PATH (or set OTR_FFMPEG)",
                kind="video")
        # audio_ref is NOT gated (audio-optional): a silent beat idles the mandala.
        return self.name

    # ---- pure helpers (CPU-testable; identical shape to viz_mxc_cpu) ----
    @staticmethod
    def _ref_path(ref):
        if not ref:
            return ""
        if isinstance(ref, str):
            return ref
        if isinstance(ref, dict):
            return ref.get("path") or ""
        return getattr(ref, "path", "") or ""

    def _canvas_dims(self, request):
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        canvas = get("canvas") or {}
        c_get = canvas.get if isinstance(canvas, dict) else (
            lambda k, d=None: getattr(canvas, k, d))
        w = int(c_get("w", 0) or 0) or 1472
        h = int(c_get("h", 0) or 0) or 832
        return w, h

    def _build_render_request(self, request):
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        timing = get("timing") or {}
        t_get = timing.get if isinstance(timing, dict) else (
            lambda k, d=None: getattr(timing, k, d))
        seeds = get("seed_bundle") or {}
        s_get = seeds.get if isinstance(seeds, dict) else (
            lambda k, d=None: getattr(seeds, k, d))
        return {
            "audio_path": self._ref_path(get("audio_ref")),
            "target_frame_count": int(t_get("target_frame_count", 0) or 0),
            "seed": int(s_get("request_seed", 0) or 0),
        }

    def _clip_from_raw(self, raw, request):
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        raw = raw or {}
        return {
            "clip_id": get("shot_id") or get("request_id") or "viz_mxc_mandala_clip",
            "type": "video", "path": raw.get("out_path", ""),
            "container": "mp4", "codec": "h264", "pixel_format": "yuv420p",
            "fps": int(self.target_fps),
            "frame_count": int(raw.get("frame_count", 0) or 0),
            "has_audio": False,
            "color_primaries": "bt709", "transfer": "bt709", "matrix": "bt709",
            "engine_id": self.name, "family": self.family,
            "qc": {"mode": raw.get("mode", "idle"),
                   "audio_used": bool(raw.get("audio_used", False))},
        }

    # ---- render ----
    def render_clip(self, request, prepared=None):
        """Decode the per-beat audio (if any) -> analyse -> paint full-16:9
        MANDALA frames on ONE reused cairo surface/context -> CRT-post each
        frame -> encode ONE silent mp4. Audio-optional: no audio_ref -> idle
        mandala from silence (also the scene/background no-image floor)."""
        import cairo
        import numpy as np

        from ._tmp import otr_engine_tmp_mp4
        from .._otr_shared import scope_draw as _sd

        plan = self._build_render_request(request)
        fps = int(self.target_fps)
        total = int(plan["target_frame_count"])
        if total <= 0:
            total = fps
        w, h = self._canvas_dims(request)

        import soundfile as sf
        audio_path = plan["audio_path"]
        audio_used = bool(audio_path and os.path.exists(audio_path))
        if audio_used:
            audio_np, sr = sf.read(audio_path, dtype="float32", always_2d=False)
            audio_np = np.asarray(audio_np, dtype=np.float32)
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
        else:
            sr = 24000
            audio_np = np.zeros(int(sr * total / max(1, fps)) + sr, dtype=np.float32)
            if not os.environ.get("OTR_TEST_MODE"):
                _LOG.info("[OTR video] viz_mxc_mandala: beat has no audio_ref -> "
                          "idle mandala from silence (%d frames)", total)

        volume, freqs, waves = _sd.analyze_audio_np(audio_np, int(sr), total, fps)
        signal, _trig, _loss = _sd.dual_ema(volume)
        vol_arr = np.asarray(volume, dtype=np.float32)
        onsets = np.zeros_like(vol_arr)
        if len(vol_arr) > 1:
            onsets[1:] = np.where((vol_arr[1:] - vol_arr[:-1]) > 0.06, 1.0, 0.0)
        scanlines = _sd.build_scanlines(w, h)
        vignette = _sd.build_vignette(w, h)
        rng_key = "viz_mxc_mandala|%d" % int(plan["seed"])

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surface)

        def _frames():
            for fi in range(total):
                _sd.paint_mandala(ctx, w, h, fi, total, fps, float(volume[fi]),
                                  freqs[fi], float(signal[fi]), float(onsets[fi]))
                rgb = _sd.mandala_surface_to_rgb(surface, w, h)
                rgb = _sd.apply_crt_post_rgb(rgb, scanlines, vignette, fi, rng_key,
                                             vol=float(volume[fi]))
                yield np.asarray(rgb, dtype=np.uint8)

        out_path = otr_engine_tmp_mp4("otr_viz_mandala_")
        _sd.encode_silent_mp4(_frames(), total, out_path, w, h, fps,
                              os.environ.get("OTR_FFMPEG", "ffmpeg"))
        if not os.environ.get("OTR_TEST_MODE"):
            _LOG.info("[OTR video] viz_mxc_mandala %dx%d x%d frames (audio=%s) -> %s",
                      w, h, total, audio_used, out_path)
        return {"out_path": out_path, "frame_count": total,
                "mode": "reactive" if audio_used else "idle", "audio_used": audio_used}

    def canonicalize(self, raw, request, profile):
        return self._clip_from_raw(raw, request)


__all__ = ["VizMxcMandalaEngine"]
