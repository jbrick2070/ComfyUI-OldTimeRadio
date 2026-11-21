r"""``viz_mxc_mandala`` -- the Cosmic Radio Mandala, an audio-reactive
tuning-eye mandala (operator 2026-06-30, kibitz-hardened r1-r4).

A SEPARATE selectable engine alongside ``viz_mxc_cpu`` (the rainbow scope),
painting a centered tuning eye + radio-dial concentric rings/spokes + an outer
solid spectrum band, in a muted iridescent palette on a deep radio-bronze
field. The CRT scanlines + vignette + film grain are applied as a POST-PROCESS
over the rendered RGB frame (:func:`nodes._otr_shared.scope_draw.apply_crt_post_rgb`),
the same machinery as ``paint_frame``/``paint_rainbow_frame``.

PILLOW, NOT CAIRO (2026-11-20). It was first painted with pycairo, which
publishes Windows wheels only, so the engine could not run on a stock Mac or
Linux box and a Mac episode died ten minutes in at its first beat. The operator
asked for a universal port: :func:`scope_draw.paint_mandala_rgb` paints the
same geometry with Pillow at 2x and reduces it, so the lane needs nothing the
pack does not already install, on any OS.

AUDIO-OPTIONAL (``required_inputs=()``): reacts to audio where present and
idles a slow mandala on silence -- ALSO the no-image floor for
retired_role_a/background (fits every role by capability, C2). ``accepts_still=
False`` -> mints NO still, so it never triggers an image model on a non-audio
slot. has_audio is always False (only OTR_MasterAudioMux adds audio,
test_audio_byte_identical invariant). NO FALLBACKS (fallback_engine=None);
assert_usable fails LOUD when ffmpeg is missing. Cold-import clean (V-12:
soundfile / PIL / scope_draw imported lazily inside load/assert_usable/
render_clip). UTF-8, no BOM, ASCII source.

Ported from the operator-approved prototype; the full grounded spec
converged over a kibitz r1-r4 arc. Config (env): ``OTR_FFMPEG`` ffmpeg path.
"""
from __future__ import annotations

import logging
import os

from . import motion_common as _MC
from .frame_contract import CONTINUITY_NONE, FrameContract
from .registry import EngineUnusable, EngineUsabilityReason, register

try:
    from .._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

_LOG = logging.getLogger("OTR.video.viz_mxc_mandala")


@register
class VizMxcMandalaEngine:
    """The Cosmic Radio Mandala engine (engine_id ``viz_mxc_mandala``)."""

    name = "viz_mxc_mandala"
    family = "abstract"
    #: UI-sort metadata only (C2: capability is the eligibility rule). Listed in
    #: every role so the dropdown offers it everywhere; required_inputs=() makes it
    #: fit every role BY CAPABILITY.
    roles = ("announcer_visual", "music_visual", "character_video")
    default_roles = ()                  # opt-in selectable only; never an auto-default
    commercial_clean = True             # own code + Pillow + ffmpeg encode
    requires_flag = None                # registry IS the menu; no flag gate
    #: AUDIO-OPTIONAL -- () fits every role by capability (mirrors viz_mxc_cpu);
    #: render_clip idles a slow mandala when no audio_ref is present.
    required_inputs = ()
    #: Opt OUT of the still coverage gate: audio-reactive, mints NO image, so a
    #: non-audio slot never triggers an image model (the operator's z_image complaint).
    accepts_still = False
    render_aspect = "wide"              # 16:9; no portrait geometry branch
    #: NO ``render_canvas`` DECLARATION, deliberately (lane 14, 2026-08-11 --
    #: lesson L19). This is the visualizer most likely to have needed one, since
    #: it paints through a graphics library rather than bare numpy -- so the
    #: premise was re-checked here with more suspicion than on lanes 11-13.
    #: It holds: ``paint_mandala_rgb(w, h, ...)`` lays the mandala out from the
    #: request's own dimensions, the scanline/vignette tables take them too, and
    #: the encoder is handed them last. Nothing imposes a canvas of its own.
    #:
    #: So the 1472x832 an episode hands this lane is ``OTR_VIDEO_LANDSCAPE_CANVAS``'s
    #: default -- an operator lever -- and declaring would overrule it for this
    #: lane alone, because ``declared_render_canvas`` is applied LAST. The
    #: profile canvas channel is declared INERT instead, in
    #: ``test_lane_preflight_matrix.PROFILE_CANVAS_DOCUMENTED_DEAD``.
    #: S1 (2026-07-25) per-model still plan (spec section 3, Shape C --
    #: "nothing"). Empty tuple = EXPLICIT "needs no images"; a missing
    #: ``still_plan`` would be UNKNOWN and fail closed by the S1 audit.
    still_plan = ()
    declared_isolation = _MC.ISOLATION_IN_PROCESS
    target_fps = 25                     # HARD-LOCK (matches the overlay + mux)
    #: THE FRAME LADDER (chunk 7a, 2026-07-26). UNBOUNDED -- a procedural
    #: visualizer synthesises frames indefinitely at ``target_fps``, so there
    #: is no ceiling and no split. CONTINUITY none: nothing here consumes a
    #: supplied first frame, which is exactly why these beats may jump cut
    #: without owing the image phase a still at all.
    #:
    #: ``continuity=`` IS PASSED (lane 14, 2026-08-11 -- lesson L3), closing the
    #: last of the four visualizers. The sentence above had claimed "CONTINUITY
    #: none" since this engine was written while the keyword was never passed,
    #: so the value was a dataclass DEFAULT. True here for a reason worth
    #: stating precisely: ``paint_mandala_rgb`` repaints the full field every
    #: frame from that frame's own audio analysis (only the static background
    #: wash is cached), and nothing reads a predecessor frame's PIXELS. So
    #: there is no terminal state a successor segment could inherit.
    frame_contract = FrameContract(
        min_frames=1,
        max_frames=0,
        quantum=1,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_NONE,
    )
    engine_version = "1"
    fallback_engine = None              # NO FALLBACKS: a failed beat fails LOUD

    def __init__(self):
        self._loaded = False

    # ---- residency (none: CPU Pillow + ffmpeg only) ----
    def load(self):
        from .._otr_shared import scope_draw as _sd
        if not _sd.find_ffmpeg(None):
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
        from .._otr_shared import scope_draw as _sd
        if not _sd.find_ffmpeg(None):
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "viz_mxc_mandala needs ffmpeg on PATH (or set OTR_FFMPEG)",
                kind="video")
        # audio_ref is NOT gated (audio-optional): a silent beat idles the mandala.
        return self.name

    # ---- pure helpers (CPU-testable) ----
    def _canvas_dims(self, request):
        return _MC.viz_canvas_dims(request)

    def _build_render_request(self, request):
        """Pure: the normalized inputs render_clip consumes."""
        return _MC.viz_render_request(request)

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
            # SURFACE 11 (2026-08-06): a procedural lane paints every frame it
            # delivers, so nothing here is extended. UNBOUNDED, so no
            # ``native_frame_count`` -- see ``cheap_families._floor_clip``.
            "extension_mode": "none",
            "qc": {"mode": raw.get("mode", "idle"),
                   "audio_used": bool(raw.get("audio_used", False))},
        }

    # ---- render ----
    def render_clip(self, request, prepared=None):
        """Decode the per-beat audio (if any) -> analyse -> paint full-16:9
        MANDALA frames -> CRT-post each frame -> encode ONE silent mp4.
        Audio-optional: no audio_ref -> idle mandala from silence (also the
        scene/background no-image floor)."""
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
            if not otr_env.get("OTR_TEST_MODE"):
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

        def _frames():
            for fi in range(total):
                rgb = _sd.paint_mandala_rgb(w, h, fi, total, fps, float(volume[fi]),
                                            freqs[fi], float(signal[fi]),
                                            float(onsets[fi]))
                rgb = _sd.apply_crt_post_rgb(rgb, scanlines, vignette, fi, rng_key,
                                             vol=float(volume[fi]))
                yield np.asarray(rgb, dtype=np.uint8)

        out_path = otr_engine_tmp_mp4("otr_viz_mandala_")
        _sd.encode_silent_mp4(_frames(), total, out_path, w, h, fps,
                              None)
        # M7 (2026-07-28): THE SECOND ENCODER. These four viz_* engines write
        # through scope_draw.encode_silent_mp4, not encode_frames_to_silent_mp4,
        # so neither half of the clip proof ever reached them and the roster
        # gate could not see them either -- it grepped two literal call
        # spellings and this is a third. frame_count is the integer timing
        # authority, and it was the pre-computed loop bound, self-declared.
        fields = ffprobe_clip_fields(out_path)
        validate_silent_clip_contract(fields, fps)
        proven = proven_frame_count(out_path, total, preprobed_fields=fields)
        if not otr_env.get("OTR_TEST_MODE"):
            _LOG.info("[OTR video] viz_mxc_mandala %dx%d x%d frames (audio=%s) -> %s",
                      w, h, proven, audio_used, out_path)
        return {"out_path": out_path, "frame_count": proven,
                "mode": "reactive" if audio_used else "idle", "audio_used": audio_used}

    def canonicalize(self, raw, request, profile):
        return self._clip_from_raw(raw, request)


__all__ = ["VizMxcMandalaEngine"]
