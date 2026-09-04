r"""``viz_mxc_cpu`` -- the OTR multi-colored ("mxc") audio-reactive visualizer.

The creative RAINBOW replacement for the retired ``abstract`` floor (operator
2026-06-30, kibitz-hardened). Pure numpy + PIL + ffmpeg -- runs on ANY box
(AMD / Mac / Intel), no GPU, no shaders (the operator constraint). A vintage
radio-dial / SIGNAL-SPECTRUM sweep in a MUTED rainbow on a dark noir field with
the same CRT scanlines + vignette + film grain as ``visualizer`` (the OTR
mystique), painted via :func:`nodes._otr_shared.scope_draw.paint_rainbow_frame`.

AUDIO-OPTIONAL (``required_inputs=()``): reacts to audio where present
(announcer/music/character) and idles a procedural rainbow on silence -- so it is
ALSO the no-image floor for retired_role_a/background (fits every role by capability,
C2). ``accepts_still=False`` -> it mints NO still, so it never triggers an image
model on a non-audio slot. has_audio is always False (only OTR_MasterAudioMux adds
audio, test_audio_byte_identical invariant). NO FALLBACKS (fallback_engine=None);
assert_usable fails LOUD. Cold-import clean (V-12: soundfile / PIL / scope_draw
imported lazily inside render_clip). UTF-8, no BOM, ASCII source.

The GPU-shader tier (``viz_mxc_gpu``) is DEFERRED (a later opt-in spike); this CPU
tier never depends on it. Config (env): ``OTR_FFMPEG`` ffmpeg path.
"""
from __future__ import annotations

import logging
import os

from . import motion_common as _MC
from .registry import EngineUnusable, EngineUsabilityReason, register
from .frame_contract import CONTINUITY_NONE, FrameContract

_LOG = logging.getLogger("OTR.video.viz_mxc")


@register
class VizMxcCpuEngine:
    """The ffmpeg-only procedural rainbow visualizer (engine_id ``viz_mxc_cpu``)."""

    name = "viz_mxc_cpu"
    family = "abstract"
    #: UI-sort metadata only (C2: capability is the eligibility rule). Listed in
    #: every role so the dropdown offers it everywhere; required_inputs=() makes it
    #: fit every role BY CAPABILITY.
    roles = ("announcer_visual", "music_visual", "character_video")
    default_roles = ()                  # never an auto-default; explicit pick only
    commercial_clean = True             # own code + ffmpeg encode only
    requires_flag = None                # registry IS the menu; no flag gate
    #: AUDIO-OPTIONAL -- () fits every role by capability (the abstract replacement);
    #: render_clip idles a procedural rainbow when no audio_ref is present.
    required_inputs = ()
    #: Opt OUT of the still coverage gate: audio-reactive, mints NO image, so a
    #: non-audio slot never triggers an image model (the operator's z_image complaint).
    accepts_still = False
    render_aspect = "wide"              # 16:9; no portrait geometry branch
    #: NO ``render_canvas`` DECLARATION, deliberately (lane 13, 2026-08-11 --
    #: lesson L19, premise re-checked on THIS engine's own render path rather
    #: than inherited from lanes 11-12). ``render_clip`` derives every geometric
    #: fact from the request's ``w, h``: ``paint_rainbow_frame`` gets them
    #: directly and lays the dial out through ``ring_geom(w, h)``, the scanline
    #: table, the vignette and the small font are built from them, and the
    #: encoder is handed the same pair. No latent grid, no trained input size,
    #: no canvas-dependent constant anywhere.
    #:
    #: So the 1472x832 an episode hands this lane is the default of
    #: ``OTR_VIDEO_LANDSCAPE_CANVAS`` -- an operator lever -- and not a fact
    #: about the engine. ``declared_render_canvas`` is applied LAST and overrules
    #: every earlier channel, so declaring here would silently make this the one
    #: visualizer that ignores that lever. Especially wrong on THIS lane, whose
    #: whole point is that it "runs on ANY box (AMD / Mac / Intel)" -- pinning a
    #: canvas is the opposite of portable. The profile canvas channel is
    #: declared INERT instead, in
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
    #: ``continuity=`` IS PASSED (lane 13, 2026-08-11 -- lesson L3). The sentence
    #: above had claimed "CONTINUITY none" since this engine was written while
    #: the keyword was never passed, so the value was a dataclass DEFAULT -- the
    #: right answer nobody had decided. True here for a stateable reason:
    #: ``render_clip`` paints every frame from the beat's own audio analysis and
    #: a per-beat rng key, and reads no predecessor frame, so no terminal state
    #: exists for a successor segment to inherit. Declared per lane because each
    #: visualizer owns its own contract (lane 10's shared-base fix reached the
    #: still shelf, not this family).
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

    # ---- residency (none: CPU + ffmpeg only) ----
    def load(self):
        from .._otr_shared import scope_draw as _sd
        if not _sd.find_ffmpeg(None):
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "viz_mxc_cpu needs ffmpeg on PATH (or set OTR_FFMPEG)", kind="video")
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
                "viz_mxc_cpu needs ffmpeg on PATH (or set OTR_FFMPEG)", kind="video")
        # audio_ref is NOT gated (audio-optional): a silent beat idles the rainbow.
        return self.name

    # ---- pure helpers (CPU-testable) ----
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
            "clip_id": get("shot_id") or get("request_id") or "viz_mxc_cpu_clip",
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
        """Decode the per-beat audio (if any) -> analyse -> paint full-16:9 RAINBOW
        frames -> encode ONE silent mp4. Audio-optional: no audio_ref -> idle rainbow
        from silence (also the scene/background no-image floor)."""
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
                _LOG.info("[OTR video] viz_mxc_cpu: beat has no audio_ref -> idle "
                          "rainbow from silence (%d frames)", total)

        volume, freqs, waves = _sd.analyze_audio_np(audio_np, int(sr), total, fps)
        signal, _trig, loss = _sd.dual_ema(volume)
        scanlines = _sd.build_scanlines(w, h)
        vignette = _sd.build_vignette(w, h)
        font = _sd._small_font(h)
        rng_key = "viz_mxc_cpu|%d" % int(plan["seed"])

        def _frames():
            for fi in range(total):
                img = _sd.paint_rainbow_frame(
                    w, h, fi, total, fps, float(volume[fi]), freqs[fi], waves[fi],
                    float(signal[fi]), float(loss[fi]), scanlines, vignette,
                    rng_key=rng_key, font_small=font)
                yield np.asarray(img, dtype=np.uint8)

        out_path = otr_engine_tmp_mp4("otr_viz_mxc_")
        _sd.encode_silent_mp4(_frames(), total, out_path, w, h, fps,
                              None)
        # M7 (2026-07-28): THE SECOND ENCODER. These four viz_* engines write
        # through scope_draw.encode_silent_mp4, not encode_frames_to_silent_mp4,
        # so neither half of the clip proof ever reached them and the roster
        # gate could not see them either -- it grepped two literal call
        # spellings and this is a third. frame_count is the integer timing
        # authority, and it was the pre-computed loop bound, self-declared.
        validate_silent_clip_contract(ffprobe_clip_fields(out_path), fps)
        proven = proven_frame_count(out_path, total)
        if not os.environ.get("OTR_TEST_MODE"):
            _LOG.info("[OTR video] viz_mxc_cpu %dx%d x%d frames (audio=%s) -> %s",
                      w, h, proven, audio_used, out_path)
        return {"out_path": out_path, "frame_count": proven,
                "mode": "reactive" if audio_used else "idle", "audio_used": audio_used}

    def canonicalize(self, raw, request, profile):
        return self._clip_from_raw(raw, request)


__all__ = ["VizMxcCpuEngine"]
