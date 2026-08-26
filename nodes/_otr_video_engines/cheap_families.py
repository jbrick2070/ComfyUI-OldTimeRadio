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
import os

from .registry import EngineUnusable, EngineUsabilityReason, register
from .._otr_shared.still_plan_helpers import StillPlanRow
from .frame_contract import CONTINUITY_NONE, FrameContract
from .wan_shared import ffprobe_clip_fields, validate_silent_clip_contract

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
                     "full-frame macro, centered subject"),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_beat", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     ("cinematic three-quarter framing, the subject shown "
                      "whole with clear space around it inside frame, "
                      "balanced composition")),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_character", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     ("cinematic medium shot, the character framed within a "
                      "wide 16:9 environment, full head and shoulders with "
                      "clear headroom inside frame, face unobstructed, "
                      "balanced landscape composition")),
                 style_tail_policy="full"),
    StillPlanRow(kind="portrait", cardinality="per_subject",
                 target_class="portrait", aspect="inherit_engine",
                 required="never",
                 framing_geometry=("in-character cinematic medium shot, head "
                                   "and shoulders, face clearly visible, "
                                   "subject centred with natural headroom "
                                   "above the head (never crop the top of the "
                                   "head)"),
                 style_tail_policy="full"),
)

#: still_word's OWN plan -- derived from the shared one, never a mutation of it.
#:
#: WHY A DERIVED COPY AND NOT AN EDIT (2026-08-26, and this is a ship-blocker
#: caught in review). `_CHEAP_FAMILY_STILL_PLAN` is the SAME tuple object on
#: still_motion, still_pan, still_flat AND still_word. Setting
#: ``identity="none"`` on its ``scene_character`` row in place would strip the
#: face anchor from the three CINEMATIC cheap lanes as well -- and those are
#: exactly the lanes carrying the 71 unanchored faces measured 2026-08-26, with
#: still_flat being what `workflows/otr_canonical.json` renders production with.
#: The fix would have shipped green and left production broken.
#:
#: still_word is the one member that genuinely owes no identity: its
#: ``scene_character`` row inherits face framing from the shared plan, but the
#: lane actually mints TYPOGRAPHY from the spoken line
#: (`otr_meta_brief_image_prompt.compose_still_word_prompt`), so the inherited
#: face framing was always a misdeclaration. Declaring ``identity="none"`` here
#: states the truth that row never told.
_STILL_WORD_STILL_PLAN = tuple(
    row._replace(identity="none") if row.kind == "scene_character" else row
    for row in _CHEAP_FAMILY_STILL_PLAN
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
    #: THE FRAME LADDER (chunk 7a, 2026-07-26). UNBOUNDED, and that is a
    #: real declaration rather than a shrug: ``_frame_count(request, fps)``
    #: derives the length from the beat's own duration, so any length at any
    #: rate is legal and there is nothing to split. ``max_frames=0`` means no
    #: ceiling, and an engine with no ceiling never needs a second clip.
    #: CONTINUITY none -- a cheap family paints its OWN still on every frame
    #: and has no mechanism that would consume a predecessor's terminal frame.
    #: Inherited by still_motion / still_pan / still_flat / still_word AND by
    #: mesh_stage, whose camera walk is equally unbounded.
    #:
    #: ``continuity=`` IS PASSED, and that is the point (lane 10, 2026-08-11,
    #: lesson L3 + L13). The comment above had REASONED about continuity since
    #: the shelf was written while the keyword was never passed, so every lane
    #: sharing this base inherited ``CONTINUITY_NONE`` as a dataclass DEFAULT --
    #: the right value arrived at by nobody deciding it, which is exactly the
    #: shape a wrong value would have had. Six lanes read it (the four still
    #: families, ``mesh_stage``, ``still_parallax``), so it is fixed at the
    #: shared mechanism rather than per lane.
    #:
    #: NONE is also the honest answer for ``mesh_stage`` specifically, and its
    #: reason is different from the still shelf's: ``build_blender_cmd`` takes
    #: ``start_angle`` / ``arc_degrees``, so a chained successor segment would
    #: need the predecessor's TERMINAL ORBIT ANGLE threaded forward to continue
    #: the turntable, and nothing threads it. Until something does, a chain
    #: would snap the camera back to the arc's start on every segment boundary.
    frame_contract = FrameContract(
        min_frames=1,
        max_frames=0,
        quantum=1,
        allow_tail_trim=True,
        continuity=CONTINUITY_NONE,
    )
    engine_version = "1"
    #: True when this family animates a provided still (asset_refs.init_image /
    #: still) with a pan; False families always synthesize a procedural
    #: floor. A ``uses_still`` family that is ALSO ``_require_still`` (all four
    #: still families, as of lanes 15-17) REFUSES instead of flooring, so the
    #: old "either way render_clip ALWAYS produces a valid silent clip" is no
    #: longer true and was removed.
    uses_still = False
    #: When True a MISSING/absent base still is a LOUD failure (NO dark lavfi
    #: floor fallback) instead of the synthesized slate. still_word sets this:
    #: its whole contract is "hold the minted word/title still" -- a silent
    #: black floor would swallow a mint failure exactly where it matters
    #: (NO FALLBACKS, operator directive 2026-07-02).
    #:
    #: ``still_word`` was FIRST and is no longer alone: lanes 15, 16 and 17 gave
    #: ``still_motion``, ``still_pan`` and ``still_flat`` the same refusal, one
    #: lane at a time and each on its own evidence, because Sprint B's reasoning
    #: generalised -- a silent black floor swallows a mint failure wherever it
    #: happens, not only on a word card.
    #:
    #: THE DEFAULT STAYS False, deliberately: it is the right default for a
    #: ``uses_still = False`` family that synthesises its own picture, so a new
    #: cheap family OPTS IN to refusing rather than inheriting it unnoticed.
    #: Who has ruled is pinned in
    #: ``tests/test_video_cheap_render.py::test_which_still_families_REFUSE_a_missing_still_is_pinned_per_lane``.
    _require_still = False

    def load(self) -> None:  # cheap families hold no resident weights
        return None

    def unload(self) -> None:
        return None

    def assert_usable(self, host_caps, profile, request_template=None):
        """Fail CLOSED on a missing ffmpeg, HERE, at preflight.

        S8b-12(a), lane 15, 2026-08-11. This used to `return self.name`
        unconditionally, under a comment saying "the real ffmpeg check runs in
        render_clip" -- which is true and is the whole problem: render_clip runs
        mid-beat, after the writer, the TTS, the master freeze and the stills
        have all been paid for, so a box without ffmpeg discovered it at the
        most expensive possible moment. Every viz_* lane already gates ffmpeg at
        BOTH boundaries; these four gated it at neither, which is the same shape
        of hole lane 8 closed on ltx_8gb's node classes and lane 10 on
        mesh_stage's.

        SHARED-BASE FIX (lesson L13): this lands once and covers all four still
        families, because they share this method. That is deliberate -- a
        defect in a shared mechanism is a defect in every adapter sharing it,
        and fixing it per lane would leave three lanes with the hole and a
        ledger claiming the class was dealt with.

        The probe is `scope_draw.find_ffmpeg`, the same one the visualizers use,
        so there is ONE answer to "is ffmpeg here" across the whole CPU shelf.
        `OTR_FFMPEG` is honoured identically.
        """
        from .._otr_shared import scope_draw as _sd
        if not _sd.find_ffmpeg(os.environ.get("OTR_FFMPEG", "ffmpeg")):
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "%s needs ffmpeg on PATH (or set OTR_FFMPEG) -- refused at "
                "PREFLIGHT so a missing encoder costs nothing, instead of "
                "surfacing mid-render after the writer, TTS and stills are "
                "already paid for" % self.name, kind="video")
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
        identity, so the canonical clip is returned here directly.

        "With valid inputs it ALWAYS succeeds -- the fallback-chain terminus the
        A-S6 chain humo -> humo_1.7B -> still_motion converges on" USED TO BE
        THE NEXT SENTENCE, and it has been false since 2026-07-02: that chain
        was ripped, nothing degrades here, and as of lane 15 ``still_motion``
        REFUSES a missing still rather than painting a black beat. A docstring
        describing a mechanism that is gone is the defect lesson L6 is about."""
        from . import wrapper_bridge as _wb       # lazy import: cold-import clean
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
            # refuses the dark lavfi floor -- a missing base still is a mint
            # failure and must be LOUD, never a silently-black clip.
            # ALL FOUR still families take this path now (still_word from the
            # start; still_motion lane 15, still_pan 16, still_flat 17).
            raise RuntimeError(
                "%s requires a base still but none was provided/exists "
                "(asset_refs still/init_image=%r) -- refusing the dark floor "
                "(NO FALLBACKS). The image phase must mint this beat's still "
                "before the video render." % (self.name, still))
        else:
            # THE SYNTHESISED FLOOR -- UNREACHABLE FROM ANY REGISTERED ENGINE
            # TODAY, AND DELIBERATELY KEPT (lane 17, 2026-08-11).
            #
            # It is reached only by a family with `uses_still = False` (which
            # empties `still` above) or one that has not taken the refusal. All
            # four registered still families now do both the opposite, so today
            # nothing reaches this line -- `test_the_synthesised_floor_is_
            # UNREACHABLE_from_every_registered_engine` asserts exactly that,
            # so the claim cannot rot into a guess.
            #
            # Lane 16's ledger entry said this branch should be DELETED as dead
            # code the moment the last family ruled. That was revised on
            # reaching it: `uses_still = False` is a DOCUMENTED capability of
            # this shelf (see the attribute's own comment -- "False families
            # always synthesize a procedural floor"), so this is a control with
            # no occupant, not dead code. Lane 4's precedent governs: a control
            # whose last occupant leaves gets REWRITTEN with the reason, never
            # deleted, because the invariant outlives every occupant. Deleting
            # it would also strand `_lavfi_source` and `ffmpeg_lavfi_floor_cmd`.
            cmd = _wb.ffmpeg_lavfi_floor_cmd(
                out_path, w, h, fps, n, source=self._lavfi_source(w, h, fps))
        _wb.run_ffmpeg(cmd)
        # M7 (added 2026-07-27, post-push QA fan-out). These four families
        # (still_motion / still_pan / still_flat / still_word) never touch
        # encode_frames_to_silent_mp4 -- they build the mp4 from an ffmpeg arg
        # list instead -- so the sweep that added the proof to every ENCODER
        # walked straight past them, and _floor_clip goes on to hand-write
        # container / codec / pixel_format / color_* as literals. still_motion
        # is also the terminus of the documented humo -> humo_1.7B ->
        # still_motion degrade chain, so it is the LAST place a self-declared
        # clip should be trusted. Safe to prove: all four command builders end
        # in the same _bt709_encode_args tail as the encoder, so a floor clip
        # already satisfies this contract by construction.
        validate_silent_clip_contract(ffprobe_clip_fields(out_path), fps)
        # M7 COUNT half (2026-07-28). The colour/stream half landed above on
        # 2026-07-27; the count stayed self-declared -- ``n`` is what was ASKED
        # for, and _floor_clip stamped it as CanonicalClip.frame_count, the
        # integer timing authority the composite positions the beat by.
        # All three builders use ffmpeg's frame-exact ``-frames:v`` rather than
        # a duration, so a miscount is less likely here than from a Python
        # generator -- but "less likely" is what the first encoder's count was
        # too, right up until the sweep looked. This reads the muxer's own
        # count off the SAME stream read ffprobe_clip_fields just performed, so
        # it costs no decode.
        proven = _wb.proven_frame_count(out_path, n)
        return self._floor_clip(request, out_path, fps, proven)

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
            # SURFACE 11 (no-mirror enforcement, 2026-08-06): the v1 manifest
            # contract asks EVERY delivered video row how its frames got there,
            # and this one builder answers for the whole cheap-family shelf --
            # the still-motion pan, the flat static hold, the lavfi radio floor
            # and ``still_parallax``, which all reach the wire through here.
            # Without it the v1 contract would indict the honest floor lanes.
            #
            # ``"none"`` is exact: each of these clips is synthesized to its
            # target length in ONE ffmpeg pass (zoompan / loop+frames:v / lavfi),
            # so there is no shorter render that anything extended afterwards.
            #
            # NO ``native_frame_count``, DELIBERATELY. These families are
            # UNBOUNDED -- ``frame_contract.can_split`` is False for them, so a
            # beat can never overflow one render -- and under the 7.3 ruling an
            # unbounded lane owes no native-count evidence. Claiming that a flat
            # hold's frames were all "natively rendered" would be true of the
            # ffmpeg pass and misleading about the picture, and an over-claim
            # here is exactly what a padded clip would want to say.
            "extension_mode": "none",
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
    #: S8b-12(b), lane 15, 2026-08-11 -- THE BLACK-BEAT DEFECT, CLOSED HERE.
    #: A missing base still used to emit the dark lavfi floor: a silent, black,
    #: structurally VALID clip that the composite then positioned like any other
    #: beat. The spec calls it "the historical black-beat defect, still
    #: reachable", and NO FALLBACKS (operator 2026-07-02) says a failure must be
    #: LOUD rather than watchable-and-wrong.
    #:
    #: Safe to flip, and that was VERIFIED rather than assumed, because the
    #: argument against it is stale. This family was once the terminus of the
    #: ``humo -> humo_1.7B -> still_motion`` degrade chain -- but that chain was
    #: RIPPED on 2026-07-02: ``UNIVERSAL_FLOOR`` / ``FLOOR_NAMES`` /
    #: ``make_fallback_of`` are gone (only comments recording the rip remain),
    #: ``default_roles`` is empty, and nothing degrades here automatically. So
    #: this engine renders only because an operator SELECTED it, and
    #: ``accepts_still`` means the image dispatcher mints its still. A missing
    #: still therefore means MINTING FAILED -- exactly the case that must not
    #: ship as a black beat.
    #:
    #: SCOPED TO THIS LANE deliberately: the base default stays False, so
    #: ``still_pan`` (lane 16) and ``still_flat`` (lane 17) are byte-identical
    #: until their own packets decide. ``check_ltx_open_health`` is untouched --
    #: it detects a degraded OPEN after the fact and is about engine SELECTION,
    #: not about a still that failed to mint.
    _require_still = True


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
    the 'still, but moving' option. CPU/ffmpeg-only, no weights, no VRAM ->
    commercial-clean. The still is minted by the SEPARATELY-chosen image
    engine; this family only animates it (so it is independent of the image engine).
    ``_still_motion`` defaults True (the pan); contrast ``still_flat`` (flat hold).

    "always renders" was in that first sentence and is NO LONGER TRUE (lane 16,
    2026-08-11): a MISSING still is now a LOUD refusal rather than a dark floor.
    It still always renders when it HAS its still, which is every beat where the
    image phase did its job."""
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
    #: S8b-12(b), lane 16, 2026-08-11 -- the black-beat defect, closed here too.
    #: Lane 15 deliberately did NOT make this call for this lane; it is made now,
    #: on this lane's own evidence, and the evidence is the same:
    #: ``default_roles`` is empty and nothing routes here automatically (the
    #: ``UNIVERSAL_FLOOR`` / ``FLOOR_NAMES`` / ``make_fallback_of`` machinery was
    #: ripped 2026-07-02 and only comments recording the rip remain), so this
    #: engine renders because an operator SELECTED it -- and ``accepts_still``
    #: means the image dispatcher mints its still. A missing still therefore
    #: means MINTING FAILED, and the dark lavfi floor turned that into a silent
    #: black beat the composite positioned like any other (L21).
    #:
    #: ``still_flat`` took the same refusal in lane 17, so ALL FOUR still
    #: families now refuse. The base default stays False regardless: it is the
    #: right default for a ``uses_still = False`` family that synthesises its
    #: own picture (a documented shelf capability with no occupant today).
    _require_still = True
    uses_still = True               # animate a provided still (pan) when present
    accepts_still = True            # C1: mint the selected still (coverage gate) so an
    #                                 opener/beat that picks still_pan shows the chosen
    #                                 image instead of the dark floor (D2 BLACK fix)


@register
class StillFlatFamily(_CheapFamilyBase):
    """A DEAD-FLAT still: the selected image held STATIC (no pan/zoom, fit+pad so a
    face is NEVER cropped) for the beat -- the 'I want stills, not video' option
    (operator 2026-06-18). Eligible in every role; CPU/ffmpeg-only, no weights, no
    VRAM -> commercial-clean + validated. ("always renders" was here and is no
    longer true: lane 17 gave this family `_require_still`, so a MISSING still
    is a loud refusal rather than a black hold.) ``accepts_still`` so the
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
    #: S8b-12(b), lane 17, 2026-08-11 -- the LAST still family to rule, and the
    #: one where the case is strongest: this engine's entire contract is "hold
    #: the chosen image". A missing still is not a degraded version of that, it
    #: is the absence of the thing, and the dark lavfi floor turned it into a
    #: silent black beat the composite positioned like any other.
    #:
    #: Same evidence as lanes 15-16, re-run on this lane: ``default_roles`` is
    #: empty, nothing routes here automatically (the chain was ripped
    #: 2026-07-02; the only "floor" references left are a campaign-harness
    #: DETECTOR and comments), and ``accepts_still`` means the dispatcher mints
    #: this lane's still -- so absent means MINTING FAILED.
    #:
    #: With this, all four still families refuse. The base default stays False
    #: anyway: it is the correct default for a ``uses_still = False`` family
    #: that synthesises its own picture, which is a documented capability of
    #: this shelf with no occupant today (see the render_clip note).
    _require_still = True


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
    #: S1 per-model still plan (Shape A) -- the WORD-CARD fork, not the shared
    #: tuple. See _STILL_WORD_STILL_PLAN for why still_word alone differs.
    still_plan = _STILL_WORD_STILL_PLAN
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
