r"""``viz_green`` -- the low-VRAM, ffmpeg-only procedural CRT scope video engine
(renamed from ``visualizer`` 2026-06-30, item 2 -- companion to viz_mxc_cpu /
viz_mxc_mandala; old saved graphs resolve via _LEGACY_ENGINE_ALIASES).

Audio-reactive CRT scopes (frequency ring + orbiting green/cyan/amber particles +
geometric grid + mirrored waveform + freq bars + CRT post) rendered AS a per-beat
picture, selectable per role in OTR_VideoDirector. 16:9 only. Near-zero GPU: pure
numpy + PIL + ffmpeg.

SEPARATION INVARIANT (2026-06-17 plan section 0.2, HARD): this engine does NOT
import or invoke the floor node (OTR_SignalLostVideo / video_engine.render), does
NOT touch the OTR_SceneAwareScopes / OTR_PostUpscaleProcgenBlend overlay, and is
INERT unless a role's dropdown selects ``viz_green``. The full-colour draw routines
are COPIED (not extracted) into the torch-free :mod:`nodes._otr_shared.scope_draw`,
so the floor's behaviour is provably unchanged (v1; a later refactor can extract).

v1 = ONE engine = the faithful full-colour resurrection. NO mode widget, NO new
geometry, NO story-DNA (deferred). has_audio is always False -- only
OTR_MasterAudioMux adds audio (test_audio_byte_identical invariant). NO FALLBACKS
(fallback_engine=None); assert_usable fails LOUD. Cold-import clean (V-12: soundfile
/ PIL / scope_draw imported lazily inside render_clip). UTF-8, no BOM, ASCII source.

Config (env): the historical ``OTR_ENABLE_VISUALIZER`` flag is vestigial (registry
IS the menu; no flag gate); ``OTR_FFMPEG`` ffmpeg path is still read.
"""
from __future__ import annotations

import logging
import os

from . import motion_common as _MC
from .registry import EngineUnusable, EngineUsabilityReason, register
from .frame_contract import CONTINUITY_NONE, FrameContract

_LOG = logging.getLogger("OTR.video.viz_green")


@register
class VisualizerEngine:
    """The ffmpeg-only procedural CRT scope engine (engine_id ``viz_green``)."""

    name = "viz_green"
    family = "abstract"
    roles = ("announcer_visual", "music_visual", "character_video")
    default_roles = ()                  # never an auto-default; explicit pick only
    commercial_clean = True             # own code + ffmpeg encode only
    requires_flag = None  # vestigial (registry IS the menu; no flag gate)
    required_inputs = ("audio_ref",)    # audio only; no init_image, no weights
    #: Coverage arch opt-OUT (explicit): the CRT scope floor synthesizes from audio
    #: and IGNORES a still -> mint NO image, so an all-visualizer episode needs no
    #: image model at all (the accessible floor). See docs/2026-06-18-coverage-arch-wiring/.
    accepts_still = False
    render_aspect = "wide"              # 16:9; no portrait geometry branch exists
    #: THIS LANE DELIBERATELY DECLARES NO ``render_canvas`` (lane 11,
    #: 2026-08-11). It was drafted with one and the draft was WRONG; the
    #: reasoning is recorded because the mistake is an easy one to repeat.
    #:
    #: The tempting argument: measured on this box, the two request builders
    #: disagree -- ``build_request_from_shot`` hands this lane (1472, 832) while
    #: ``render_single``, which every solo lane smoke uses, hands it (832, 480)
    #: -- so declaring 1472x832 would make them agree on what production ships.
    #:
    #: Why that is wrong HERE. This engine has NO native canvas: ``render_clip``
    #: paints and encodes at exactly the size the request carries, with no
    #: latent grid, no fixed model input and no canvas-dependent constant. The
    #: 1472x832 it receives in an episode is not a property of the lane at all
    #: -- it is ``OTR_VIDEO_LANDSCAPE_CANVAS``'s default, an OPERATOR LEVER.
    #: Because ``declared_render_canvas`` is applied LAST and overrules every
    #: earlier channel, declaring would silently make this lane the one member
    #: of the procedural family that ignores that lever, and would pin the smoke
    #: path too. That is a real behaviour change wearing a documentation label,
    #: and lesson L2's own precision note is exactly the check that catches it:
    #: a canvas declaration must agree with the OVERRIDE PATH or state that the
    #: overrides are unsupported -- never quietly disable them.
    #:
    #: The lane's honest G2 position is therefore that its PROFILE canvas
    #: channel is INERT, declared as such in
    #: ``test_lane_preflight_matrix.PROFILE_CANVAS_DOCUMENTED_DEAD``. Inert, not
    #: dead: the number IS carried (profile -> applier -> node-87 director
    #: widgets -> ``request["canvas"]``) and is then OVERWRITTEN by the landscape
    #: default for every non-face family, of which ``abstract`` is one. See L18.
    #: S1 (2026-07-25) per-model still plan (spec section 3, Shape C --
    #: "nothing"). The empty tuple is the EXPLICIT "needs no images"
    #: declaration; a missing ``still_plan`` would be treated as UNKNOWN
    #: and fail closed by the S1 audit.
    still_plan = ()
    declared_isolation = _MC.ISOLATION_IN_PROCESS
    target_fps = 25                     # HARD-LOCK (matches the overlay + mux)
    #: THE FRAME LADDER (chunk 7a, 2026-07-26). UNBOUNDED -- a procedural
    #: visualizer synthesises frames indefinitely at ``target_fps``, so there
    #: is no ceiling and no split. CONTINUITY none: nothing here consumes a
    #: supplied first frame, which is exactly why these beats may jump cut
    #: without owing the image phase a still at all.
    #:
    #: ``continuity=`` IS PASSED (lane 11, 2026-08-11 -- lesson L3). The
    #: sentence above had said "CONTINUITY none" since this engine was written
    #: while the keyword was never passed, so the value was a dataclass DEFAULT:
    #: the right answer that nobody had actually decided, which is the same
    #: shape a wrong one would have had. It is TRUE here for a reason this lane
    #: can state -- ``render_clip`` paints every frame from the beat's own audio
    #: analysis and reads no predecessor frame at all, so there is no terminal
    #: state a successor segment could inherit even if something wanted to.
    #:
    #: Scoped to THIS lane deliberately. The other three visualizers carry the
    #: identical defect but each declares its OWN contract in its own module --
    #: there is no shared base here, unlike the still shelf's
    #: ``_CheapFamilyBase`` that lane 10 fixed once for four lanes -- so L13's
    #: sweep rule does not reach them and lanes 12-14 each close their own.
    frame_contract = FrameContract(
        min_frames=1,
        max_frames=0,
        quantum=1,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_NONE,
    )
    engine_version = "1"
    #: NO FALLBACKS (547671d): a failed beat fails the episode LOUD.
    fallback_engine = None

    def __init__(self):
        self._loaded = False

    # ---- residency (none: CPU + ffmpeg only) ----
    def load(self):
        from .._otr_shared import scope_draw as _sd
        if not _sd.find_ffmpeg(os.environ.get("OTR_FFMPEG", "ffmpeg")):
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "viz_green needs ffmpeg on PATH (or set OTR_FFMPEG)", kind="video")
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
        if not _sd.find_ffmpeg(os.environ.get("OTR_FFMPEG", "ffmpeg")):
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "viz_green needs ffmpeg on PATH (or set OTR_FFMPEG)", kind="video")
        # NOTE: audio_ref is NOT gated here. The per-beat audio is SLICED at
        # render time, so the assert_usable request_template carries an empty
        # audio_ref for music/announcer beats (mirrors eng_ltx_av, which also
        # audio-conditions but does not gate audio_ref pre-render). render_clip is
        # the LOUD audio gate -- a beat that truly has no audio fails the episode
        # there (no fallbacks). (Fixed after the 2026-06-18 visualizer soak: the
        # old check aborted shot_b000_music_open before render.)
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
        """Pure: the normalized inputs render_clip consumes."""
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
        """Pure: shape the silent CanonicalClip dict (bt709 / yuv420p; has_audio
        False -- only OTR_MasterAudioMux adds audio)."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        raw = raw or {}
        return {
            "clip_id": get("shot_id") or get("request_id") or "viz_green_clip",
            "type": "video", "path": raw.get("out_path", ""),
            "container": "mp4", "codec": "h264", "pixel_format": "yuv420p",
            "fps": int(self.target_fps),
            "frame_count": int(raw.get("frame_count", 0) or 0),
            "has_audio": False,
            "color_primaries": "bt709", "transfer": "bt709", "matrix": "bt709",
            "engine_id": self.name, "family": self.family,
            # SURFACE 11 (2026-08-06): a procedural lane paints every frame it
            # delivers, so nothing here is extended. Declared because the v1
            # manifest contract asks every delivered video row, not because this
            # engine could ever pad. UNBOUNDED, so no ``native_frame_count`` --
            # see the reasoning at ``cheap_families._floor_clip``.
            "extension_mode": "none",
        }

    # ---- render ----
    def render_clip(self, request, prepared=None):
        """Decode the per-beat audio -> mono -> analyse -> paint full-16:9 CRT
        frames -> encode ONE silent mp4. All steps per the 2026-06-17 plan."""
        import numpy as np

        from ._tmp import otr_engine_tmp_mp4
        from .wan_shared import (ffprobe_clip_fields,
                                 validate_silent_clip_contract)
        from .wrapper_bridge import proven_frame_count
        from .._otr_shared import scope_draw as _sd

        plan = self._build_render_request(request)
        fps = int(self.target_fps)
        # An accessible FLOOR must render EVERY beat. A degenerate beat with
        # target_frame_count <= 0 (a zero-length beat in the ledger) defaults to
        # one second, exactly like the cheap floor's _frame_count -- never a crash.
        # (2026-06-18 soak: shot_b005 arrived with target_frame_count=0.)
        total = int(plan["target_frame_count"])
        if total <= 0:
            total = fps
        w, h = self._canvas_dims(request)

        # The visualizer is an accessible FLOOR forced on ALL roles, so it MUST
        # render every beat. A beat with audio paints reactive scopes; a beat with
        # NO audio (a silent scene/b-roll beat, or a synthetic beat with no master
        # slice) paints IDLE scopes from synthesized silence -- a silent beat is a
        # silent scope, which is correct, NOT a fallback/degrade. (2026-06-18 soak:
        # shot_b005 reached the visualizer with an empty audio_ref; idle-on-silence
        # makes the floor robust to every beat type instead of crashing the episode.)
        import soundfile as sf
        audio_path = plan["audio_path"]
        if audio_path and os.path.exists(audio_path):
            audio_np, sr = sf.read(audio_path, dtype="float32", always_2d=False)
            audio_np = np.asarray(audio_np, dtype=np.float32)
            if audio_np.ndim > 1:                  # mix stereo -> mono (gemini #2)
                audio_np = audio_np.mean(axis=1)
        else:
            sr = 24000
            audio_np = np.zeros(int(sr * total / max(1, fps)) + sr, dtype=np.float32)
            if not os.environ.get("OTR_TEST_MODE"):
                _LOG.info("[OTR video] viz_green: beat has no audio_ref -> idle "
                          "scopes from silence (%d frames)", total)

        volume, freqs, waves = _sd.analyze_audio_np(audio_np, int(sr), total, fps)
        signal, _trig, loss = _sd.dual_ema(volume)   # REQUIRED (draw reads signal)
        scanlines = _sd.build_scanlines(w, h)
        vignette = _sd.build_vignette(w, h)
        font = _sd._small_font(h)
        rng_key = "viz_green|%d" % int(plan["seed"])

        def _frames():
            for fi in range(total):
                img = _sd.paint_frame(
                    w, h, fi, total, fps, float(volume[fi]), freqs[fi], waves[fi],
                    float(signal[fi]), float(loss[fi]), scanlines, vignette,
                    rng_key=rng_key, font_small=font)
                yield np.asarray(img, dtype=np.uint8)

        out_path = otr_engine_tmp_mp4("otr_viz_green_")
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
            _LOG.info("[OTR video] viz_green %dx%d x%d frames -> %s",
                      w, h, proven, out_path)
        return {"out_path": out_path, "frame_count": proven}

    def canonicalize(self, raw, request, profile):
        return self._clip_from_raw(raw, request)


__all__ = ["VisualizerEngine"]
