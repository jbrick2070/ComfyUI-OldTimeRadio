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
from .frame_contract import CONTINUITY_NONE, FrameContract

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
    #: NO ``render_canvas`` DECLARATION, deliberately (lane 12, 2026-08-11 --
    #: lesson L19, and the premise was re-checked here rather than inherited).
    #: ``render_clip`` paints and encodes at exactly the ``w, h`` the request
    #: carries: `paint_golden_camera_frame`, the scanline and vignette tables
    #: and the encoder are all built from those two numbers, and there is no
    #: latent grid, trained input size or canvas-dependent constant anywhere in
    #: the path. So the 1472x832 an episode hands this lane is the default of
    #: ``OTR_VIDEO_LANDSCAPE_CANVAS`` -- an operator lever -- and NOT a fact
    #: about the engine. Because ``declared_render_canvas`` is applied LAST and
    #: overrules every earlier channel, declaring here would silently make this
    #: the one visualizer that ignores that lever. The profile canvas channel is
    #: declared INERT instead, in
    #: ``test_lane_preflight_matrix.PROFILE_CANVAS_DOCUMENTED_DEAD``.
    #: S1 (2026-07-25) per-model still plan (spec
    #: ``docs/2026-07-25-still-plans-locked-build-spec.md`` section 3, Shape
    #: C -- "nothing"). ``viz_camera`` mints NO still. The empty tuple is
    #: the EXPLICIT "needs no images" declaration; a missing ``still_plan``
    #: would be treated as UNKNOWN and fail closed by the S1 audit.
    still_plan = ()
    declared_isolation = _MC.ISOLATION_IN_PROCESS
    target_fps = 25
    #: THE FRAME LADDER (chunk 7a, 2026-07-26). UNBOUNDED -- a procedural
    #: visualizer synthesises frames indefinitely at ``target_fps``, so there
    #: is no ceiling and no split. CONTINUITY none: nothing here consumes a
    #: supplied first frame, which is exactly why these beats may jump cut
    #: without owing the image phase a still at all.
    #:
    #: ``continuity=`` IS PASSED (lane 12, 2026-08-11 -- lesson L3). The
    #: sentence above had claimed "CONTINUITY none" since this engine was
    #: written while the keyword was never passed, so the value was a dataclass
    #: DEFAULT -- the right answer nobody had decided. It is true here for a
    #: reason this lane can state: ``render_clip`` paints every frame from the
    #: beat's own audio analysis and a per-beat rng key, and reads no
    #: predecessor frame, so no terminal state exists for a successor segment
    #: to inherit. Declared per lane because each visualizer owns its own
    #: contract -- there is no shared base here (lane 10's `_CheapFamilyBase`
    #: fix reached the still shelf, not this family).
    frame_contract = FrameContract(
        min_frames=1,
        max_frames=0,
        quantum=1,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_NONE,
    )
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
            # SURFACE 11 (2026-08-06): a procedural lane paints every frame it
            # delivers, so nothing here is extended. UNBOUNDED, so no
            # ``native_frame_count`` -- see ``cheap_families._floor_clip``.
            "extension_mode": "none",
            "qc": {"mode": raw.get("mode", "idle"),
                   "audio_used": bool(raw.get("audio_used", False))},
        }

    def render_clip(self, request, prepared=None):
        """Decode optional beat audio, paint camera visualizer frames, encode mp4."""
        import numpy as np

        from ._tmp import otr_engine_tmp_mp4
        from .wan_shared import (ffprobe_clip_fields,
                                 validate_silent_clip_contract)
        from .wrapper_bridge import proven_frame_count
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
        # M7 (2026-07-28): THE SECOND ENCODER. These four viz_* engines write
        # through scope_draw.encode_silent_mp4, not encode_frames_to_silent_mp4,
        # so neither half of the clip proof ever reached them and the roster
        # gate could not see them either -- it grepped two literal call
        # spellings and this is a third. frame_count is the integer timing
        # authority, and it was the pre-computed loop bound, self-declared.
        validate_silent_clip_contract(ffprobe_clip_fields(out_path), fps)
        proven = proven_frame_count(out_path, total)
        if not os.environ.get("OTR_TEST_MODE"):
            _LOG.info("[OTR video] viz_camera %dx%d x%d frames (audio=%s) -> %s",
                      w, h, proven, audio_used, out_path)
        return {"out_path": out_path, "frame_count": proven,
                "mode": "reactive" if audio_used else "idle", "audio_used": audio_used}

    def canonicalize(self, raw, request, profile):
        return self._clip_from_raw(raw, request)


__all__ = ["VizCameraEngine"]
