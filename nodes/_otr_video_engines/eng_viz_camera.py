r"""``viz_camera`` -- the OTR-native Golden Flicker camera visualizer.

This is the operator-approved "camera, not skull" Golden Flicker look brought
INTO this repo as first-party OTR code. It is a selectable abstract visualizer
beside ``viz_green``, ``viz_mxc_cpu``, and ``viz_mxc_mandala``: warm gold noir,
twin spinning reels, projector beam, mandala rings, spectrum spokes, and the same
CRT scanlines/vignette/grain glue as the other OTR CPU visualizers.

Pure numpy + PIL + ffmpeg, no GPU, no runtime dependency on the separate
``golden-flicker`` repo. AUDIO-OPTIONAL (``required_inputs=()``): it reacts to
audio when present and idles on silence, so it fits announcer/music/character
slots by capability. ``accepts_still=False`` -> it mints NO still and never
triggers an image model. The clip is silent; OTR_MasterAudioMux adds audio later.
"""
from __future__ import annotations

import logging
import os

from . import motion_common as _MC
from .registry import EngineUnusable, EngineUsabilityReason, register

_LOG = logging.getLogger("OTR.video.viz_camera")


@register
class VizCameraEngine:
    """The ffmpeg-only procedural Golden Flicker camera engine."""

    name = "viz_camera"
    family = "abstract"
    roles = ("announcer_visual", "music_visual", "character_video")
    default_roles = ()
    commercial_clean = True
    requires_flag = None
    required_inputs = ()
    accepts_still = False
    render_aspect = "wide"
    declared_isolation = _MC.ISOLATION_IN_PROCESS
    target_fps = 25
    engine_version = "1"
    fallback_engine = None

    def __init__(self):
        self._loaded = False

    def load(self):
        from .._otr_shared import scope_draw as _sd
        if not _sd.find_ffmpeg(os.environ.get("OTR_FFMPEG", "ffmpeg")):
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "viz_camera needs ffmpeg on PATH (or set OTR_FFMPEG)", kind="video")
        self._loaded = True

    def unload(self):
        self._loaded = False

    def prepare(self, host_caps, profile, session_ctx):
        return {"engine_id": self.name}

    def teardown(self, prepared):
        return None

    def assert_usable(self, host_caps, profile, request_template=None):
        from .._otr_shared import scope_draw as _sd
        if not _sd.find_ffmpeg(os.environ.get("OTR_FFMPEG", "ffmpeg")):
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "viz_camera needs ffmpeg on PATH (or set OTR_FFMPEG)", kind="video")
        return self.name

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
            "clip_id": get("shot_id") or get("request_id") or "viz_camera_clip",
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

    def render_clip(self, request, prepared=None):
        """Decode optional beat audio, paint camera visualizer frames, encode mp4."""
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
                _LOG.info("[OTR video] viz_camera: beat has no audio_ref -> idle "
                          "camera visualizer from silence (%d frames)", total)

        volume, freqs, waves = _sd.analyze_audio_np(audio_np, int(sr), total, fps)
        signal, _trig, loss = _sd.dual_ema(volume)
        scanlines = _sd.build_scanlines(w, h)
        vignette = _sd.build_vignette(w, h)
        rng_key = "viz_camera|%d" % int(plan["seed"])

        def _frames():
            for fi in range(total):
                img = _sd.paint_golden_camera_frame(
                    w, h, fi, total, fps, float(volume[fi]), freqs[fi], waves[fi],
                    float(signal[fi]), float(loss[fi]), scanlines, vignette,
                    rng_key=rng_key)
                yield np.asarray(img, dtype=np.uint8)

        out_path = otr_engine_tmp_mp4("otr_viz_camera_")
        _sd.encode_silent_mp4(_frames(), total, out_path, w, h, fps,
                              os.environ.get("OTR_FFMPEG", "ffmpeg"))
        if not os.environ.get("OTR_TEST_MODE"):
            _LOG.info("[OTR video] viz_camera %dx%d x%d frames (audio=%s) -> %s",
                      w, h, total, audio_used, out_path)
        return {"out_path": out_path, "frame_count": total,
                "mode": "reactive" if audio_used else "idle", "audio_used": audio_used}

    def canonicalize(self, raw, request, profile):
        return self._clip_from_raw(raw, request)


__all__ = ["VizCameraEngine"]
