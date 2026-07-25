"""Cheap radio-floor video families -- the no-heavy-engine adapters (A-S3 / CW-4).

The first concrete video engines registered in the platform: cheap, CPU/ffmpeg
families that need NO heavy model, so a watchable episode (M1) renders before any
GPU engine exists. Each registers exactly like a future heavy engine (HuMo / LTX
/ Wan) -- model-agnostic, selected per role; no model is "primary".

Families (schemas.FAMILIES): ``static_motion`` (still_motion -- a still with a slow
pan), ``static_image_gen`` (still_pan -- a provided still with a pan / still_flat --
the same still held flat). Each produces an ALWAYS-SILENT ``CanonicalClip``
(``has_audio`` is always False -- audio is added ONLY by ``OTR_MasterAudioMux``,
V-1). (The ``abstract`` procedural floor + the ``station_card`` card were RETIRED
2026-06-30, C0; the ``abstract`` family name lives on via ``eng_visualizer``.)

Cold-import clean (V-12): module scope imports only the dep-free registry + the
role vocabulary. ffmpeg / PIL / numpy / torch are imported LAZILY inside
``render_clip`` (the render slice wires up with the interactive episode smoke;
the platform's selection / role-filter / usability logic is fully CPU-tested
without rendering).
"""
from __future__ import annotations

import logging

from .registry import register
from .._otr_shared.still_plan_helpers import StillPlanRow

log = logging.getLogger("OTR.video.cheap_families")


#: S1 (2026-07-25) per-model still plan for the cheap still-based families
#: (still_motion / still_pan / still_flat / still_word). All Shape A --
#: scene spine per spec section 3. FILE-LOCAL, fully declared. Note the
#: spec's note in section 3: for the still_* engines, requiredness is
#: DECLARED here, never derived from ``required_inputs`` (which would be a
#: back door around the plan). Nothing reads the plan at S1 -- S2 wires it in.
_CHEAP_FAMILY_STILL_PLAN = (
    StillPlanRow(kind="scene_open", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     "Wide establishing shot; the scene an audience is "
                     "entering."),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_beat", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     "Wide continuity framing for the beat, matching the "
                     "scene_open geometry."),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_character", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     "Wide framing that keeps the named character legible in "
                     "the scene."),
                 style_tail_policy="full"),
    StillPlanRow(kind="portrait", cardinality="per_subject",
                 target_class="portrait", aspect="inherit_engine",
                 required="never",
                 framing_geometry="",
                 style_tail_policy="full"),
)


class _CheapFamilyBase:
    """Shared shell for a cheap, no-heavy-engine video family. Render lifecycle
    is the ffmpeg/CPU slice (lazy); the registry only reads the core metadata."""

    name = "cheap"
    roles: tuple = ()
    default_roles: tuple = ()
    #: Still-aspect identity (2026-06-17): every cheap floor / still family fills
    #: the 16:9 canvas (BUG-407: PORTRAIT is HuMo-ONLY). Subclasses inherit this;
    #: still_parallax / mesh_stage (which extend this base) are wide too.
    render_aspect = "wide"
    commercial_clean = True
    requires_flag = None            # cheap families are always available (no opt-in)
    invocable = True
    invocability_reason = ""
    family = "abstract"
    required_inputs: tuple = ()
    engine_version = "1"
    #: True when this family animates a provided still (asset_refs.init_image /
    #: still) with a pan; False families always synthesize a procedural
    #: floor. Either way render_clip ALWAYS produces a valid silent clip.
    uses_still = False
    #: When True a MISSING/absent base still is a LOUD failure (NO dark lavfi
    #: floor fallback) instead of the synthesized slate. still_word sets this:
    #: its whole contract is "hold the minted word/title still" -- a silent
    #: black floor would swallow a mint failure exactly where it matters
    #: (NO FALLBACKS, operator directive 2026-07-02). Default False keeps every
    #: other cheap family's always-renders floor behavior byte-identical.
    _require_still = False

    def load(self) -> None:  # cheap families hold no resident weights
        return None

    def unload(self) -> None:
        return None

    def assert_usable(self, host_caps, profile, request_template=None):
        """Cheap families are usable wherever ffmpeg exists; the real ffmpeg
        check runs in render_clip. Returns the validated name (no heavy dep)."""
        return self.name

    def prepare(self, host_caps, profile, session_ctx):  # pragma: no cover
        return {"engine_id": self.name}

    # ---- canvas / input resolution (pure; CPU-testable) ----
    @staticmethod
    def _get(request):
        return request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))

    def _canvas_dims(self, request):
        """(width, height, fps) from the request canvas, with platform defaults
        (832x480 @ 25) when a field is absent. The cheap floor never stretches:
        a still is scaled to cover with one uniform scale (bridge), a synth source
        fills the canvas exactly."""
        get = self._get(request)
        canvas = get("canvas") or {}
        c_get = canvas.get if isinstance(canvas, dict) else (
            lambda k, d=None: getattr(canvas, k, d))
        w = int(c_get("w", 0) or 0) or 832
        h = int(c_get("h", 0) or 0) or 480
        fps = int(c_get("fps", 0) or 0) or 25
        return w, h, fps

    def _frame_count(self, request, fps):
        """The clip length in frames (timing.target_frame_count), defaulting to one
        second (``fps`` frames) when unspecified so the floor always renders."""
        get = self._get(request)
        timing = get("timing") or {}
        t_get = timing.get if isinstance(timing, dict) else (
            lambda k, d=None: getattr(timing, k, d))
        n = int(t_get("target_frame_count", 0) or 0)
        return n if n > 0 else int(fps)

    def _still_path(self, request):
        """A provided still/portrait path from asset_refs (or "" when none)."""
        get = self._get(request)
        assets = get("asset_refs") or {}
        if isinstance(assets, dict):
            for key in ("still", "init_image", "image"):
                v = assets.get(key)
                if isinstance(v, str) and v:
                    return v
                if isinstance(v, dict) and v.get("path"):
                    return v["path"]
        return ""

    def _lavfi_source(self, w, h, fps):
        """The libavfilter source for this family's synthesized floor (a dark
        radio-slate field by default; a family may override for its own look)."""
        return "color=c=0x0A0E14:s=%dx%d:r=%d" % (int(w), int(h), int(fps))

    def render_clip(self, request, prepared=None):
        """Render ONE always-silent CanonicalClip via ffmpeg (the CPU radio floor).

        A provided still (asset_refs) is animated with a gentle pan when
        ``uses_still``; otherwise the family's libavfilter source is synthesized.
        Output is the platform's silent bt709 / yuv420p contract (V-1: only
        OTR_MasterAudioMux ever adds audio). This family's canonicalize() is
        identity, so the canonical clip is returned here directly. With valid
        inputs it ALWAYS succeeds -- the fallback-chain terminus the A-S6 chain
        humo -> humo_1.7B -> still_motion converges on."""
        from . import wrapper_bridge as _wb       # lazy import: cold-import clean
        import os
        from ._tmp import otr_engine_tmp_mp4
        w, h, fps = self._canvas_dims(request)
        n = self._frame_count(request, fps)
        out_path = otr_engine_tmp_mp4("otr_floor_%s_" % self.name)
        still = self._still_path(request) if self.uses_still else ""
        if still and os.path.exists(still):
            # ``_still_motion`` chooses the pan (cover+crop, default) vs the
            # FLAT hold (fit+pad, no crop -> a face is never cut) -- the still_flat
            # 'just a still, no motion' contract.
            if getattr(self, "_still_motion", True):
                cmd = _wb.ffmpeg_still_motion_cmd(still, out_path, w, h, fps, n)
            else:
                cmd = _wb.ffmpeg_still_static_cmd(still, out_path, w, h, fps, n)
        elif self._require_still:
            # NO FALLBACKS (operator 2026-07-02): a still-REQUIRED family
            # (still_word) refuses the dark lavfi floor -- a missing base still
            # is a mint failure and must be LOUD, never a silently-black clip.
            raise RuntimeError(
                "%s requires a base still but none was provided/exists "
                "(asset_refs still/init_image=%r) -- refusing the dark floor "
                "(NO FALLBACKS). The image phase must mint this beat's still "
                "before the video render." % (self.name, still))
        else:
            cmd = _wb.ffmpeg_lavfi_floor_cmd(
                out_path, w, h, fps, n, source=self._lavfi_source(w, h, fps))
        _wb.run_ffmpeg(cmd)
        return self._floor_clip(request, out_path, fps, n)

    def _floor_clip(self, request, out_path, fps, n):
        """Pure: shape the rendered floor mp4 into the silent CanonicalClip dict
        (bt709 / yuv420p; frame_count is the integer timing authority)."""
        get = self._get(request)
        return {
            "clip_id": get("shot_id") or get("request_id") or ("%s_clip" % self.name),
            "type": "video", "path": out_path,
            "container": "mp4", "codec": "h264", "pixel_format": "yuv420p",
            "fps": int(fps), "frame_count": int(n),
            "has_audio": False,            # V-1: only OTR_MasterAudioMux emits audio
            "color_primaries": "bt709", "transfer": "bt709", "matrix": "bt709",
            "engine_id": self.name, "family": self.family,
        }

    def canonicalize(self, raw, request, profile):
        """Identity: render_clip already emits the canonical silent clip dict."""
        return raw

    def teardown(self, prepared) -> None:  # pragma: no cover
        return None


# 2026-06-30 (C0, operator directive): ``AbstractFamily`` (engine_id "abstract")
# and ``StationCardFamily`` (engine_id "station_card") were RETIRED -- abstract was
# redundant with the real ``visualizer`` audio-reactive scope (and the future
# ``visualizer_rainbow`` fills the fun audio-reactive slot), and station_card was the
# broken black card. Both are UNREGISTERED + their CAPABILITIES rows removed. The
# family NAME "abstract" survives (visualizer is family="abstract"; the cheap-base
# default + schemas.FAMILIES keep it). NO FALLBACKS (2026-07-02, Sprint A): the
# UNIVERSAL_FLOOR/chain machinery was RIPPED -- there is no floor terminus role.


@register
class StillMotionFamily(_CheapFamilyBase):
    name = "still_motion"
    family = "static_motion"
    #: S1 per-model still plan (Shape A -- see module constant above).
    still_plan = _CHEAP_FAMILY_STILL_PLAN
    roles = ("announcer_visual", "music_visual", "character_video")
    # rip-sfx-broll (2026-07-01): its only default role (retired_role_a) was
    # removed. NO FALLBACKS (2026-07-02, Sprint A): still_motion lost its
    # UNIVERSAL_FLOOR role with the chain rip -- it stays a REGISTERED
    # SELECTABLE engine (capability: text_prompt), but nothing degrades to it
    # and no role auto-defaults to it.
    default_roles = ()
    required_inputs = ("text_prompt",)
    uses_still = True               # pan over a provided still when present
    accepts_still = True            # C1: mint the selected still (coverage gate) so
    #                                 a still_motion beat shows the chosen image, not
    #                                 the dark floor (D2 BLACK fix)


# 2026-06-18: the cheap ``visualizer`` floor stub was SUPERSEDED by the real
# procedural CRT scope engine (nodes/_otr_video_engines/eng_visualizer.py,
# engine_id "visualizer"). The stub here was a minimal ffmpeg-floor family; the new
# engine is the faithful full-colour resurrection (audio analysis + the ring /
# particles / grid / waveform / bars / CRT-post look) and OWNS the "visualizer"
# name now. Removed to avoid a duplicate registration (the scope-visualizer plan
# wrongly assumed the name was unregistered).


@register
class StillPanFamily(_CheapFamilyBase):
    """A provided still given a slow pan/zoom (cover+crop) for the beat --
    the 'still, but moving' option. CPU/ffmpeg-only, no weights, no VRAM, always
    renders -> commercial-clean. The still is minted by the SEPARATELY-chosen image
    engine; this family only animates it (so it is independent of the image engine).
    ``_still_motion`` defaults True (the pan); contrast ``still_flat`` (flat hold)."""
    name = "still_pan"
    family = "static_image_gen"
    #: S1 per-model still plan (Shape A).
    still_plan = _CHEAP_FAMILY_STILL_PLAN
    # A plain still needs only text_prompt (supplied by EVERY role), so it is the
    # fast, universal "just a still" pick -- eligible in all roles. BUG-LOCAL-401:
    # music_visual was missing, so music_video_model='still_pan'
    # failed OTR_VideoDirector role validation even though a still is valid there.
    roles = ("announcer_visual", "music_visual", "character_video")
    default_roles = ()              # selectable peer; not the in-stack default
    required_inputs = ("text_prompt",)
    commercial_clean = True         # own ffmpeg + the chosen still; no model license
    uses_still = True               # animate a provided still (pan) when present
    accepts_still = True            # C1: mint the selected still (coverage gate) so an
    #                                 opener/beat that picks still_pan shows the chosen
    #                                 image instead of the dark floor (D2 BLACK fix)


@register
class StillFlatFamily(_CheapFamilyBase):
    """A DEAD-FLAT still: the selected image held STATIC (no pan/zoom, fit+pad so a
    face is NEVER cropped) for the beat -- the 'I want stills, not video' option
    (operator 2026-06-18). Eligible in every role; CPU/ffmpeg-only, no weights, no
    VRAM, always renders -> commercial-clean + validated. ``accepts_still`` so the
    image dispatcher mints the role's selected still for it (the central coverage
    gate); ``_still_motion=False`` selects the flat hold over the pan."""
    name = "still_flat"
    family = "static_image_gen"
    #: S1 per-model still plan (Shape A).
    still_plan = _CHEAP_FAMILY_STILL_PLAN
    roles = ("announcer_visual", "music_visual", "character_video")
    default_roles = ()              # selectable peer; not an auto-default
    required_inputs = ("text_prompt",)
    commercial_clean = True         # own ffmpeg + the chosen still; no model license
    uses_still = True               # display the provided still...
    _still_motion = False           # ...STATIC (flat hold, fit+pad, no crop)
    accepts_still = True            # mint the selected still for it (coverage gate)


@register
class StillWordFamily(_CheapFamilyBase):
    """still_word: a ``still_flat`` SIBLING whose base still is minted from a
    WORD/TITLE-driven prompt instead of the cinematic scene composer -- the
    'the words ARE the picture' option (Sprint B, 2026-07-03, operator).

    MODEL-AGNOSTIC by construction: the IMAGE engine that mints the base still
    is chosen INDEPENDENTLY in the image dropdown; this VIDEO engine only holds
    the minted still dead-flat (fit+pad, no crop -- identical render mechanics
    to ``still_flat``). The ONLY delta vs still_flat is the PROMPT, composed
    upstream by ``otr_meta_brief_image_prompt.compose_still_word_prompt``
    (character/announcer beats -> the beat's spoken line as a readable word
    card; music beats -> an abstract picture of the episode title, no words).

    NO FALLBACKS: ``_require_still`` makes a missing base still fail LOUD in
    render_clip -- never the dark floor -- because a black floor is exactly how
    a word/title mint failure would hide. Selectable per role; never a default
    (``default_roles=()``)."""
    name = "still_word"
    family = "static_image_gen"
    #: S1 per-model still plan (Shape A).
    still_plan = _CHEAP_FAMILY_STILL_PLAN
    roles = ("announcer_visual", "music_visual", "character_video")
    default_roles = ()              # selectable peer; never an auto-default
    required_inputs = ("text_prompt",)
    commercial_clean = True         # own ffmpeg + the chosen still; no model license
    uses_still = True               # display the provided (word/title) still...
    _still_motion = False           # ...STATIC (flat hold, fit+pad, no crop)
    accepts_still = True            # the image dispatcher mints its worded/title still
    _require_still = True           # NO dark floor: a missing still fails LOUD


__all__ = [
    "StillMotionFamily", "StillPanFamily", "StillFlatFamily", "StillWordFamily",
]
