"""HuMo audio-driven-face motion adapter (A-S6 / CW-7) -- in-process, default-OFF.

HuMo is OTR's heaviest engine: an audio-conditioned image-to-video model that
animates a reference PORTRAIT (init_image) in sync with a speech AUDIO reference
into a talking-character clip. It is the ``audio_driven_face`` family -- the
talking-head path for the announcer + character roles. Like ltx_video / wan_i2v
it runs IN-PROCESS in the main ComfyUI cu130 venv (it loads
MODEL+CLIP+VAE+AUDIO_ENCODER internally via ``comfy.model_management``) and is the
SINGLE resident heavy engine while it holds the AS-3 lease. Native output is
480x832 @ 25 fps; a portrait init is fit to the canvas with ONE uniform scale,
never stretched, and the compositor pillarboxes the portrait clip (pre-mortem N9).

Registered DEFAULT-OFF / dark (empty ``default_roles`` + gated behind
``OTR_ENABLE_HUMO``) so it shows in the static per-role dropdown (V-6) but is
never a default and fails CLOSED until the operator enables it AND the HuMo
wrapper + checkpoints are installed and verified on the GPU box (the A-S6 smoke).
No model is "primary" -- HuMo is one peer adapter among the motion engines.

NO FALLBACKS (operator directive 2026-07-02: NO fallbacks / NO auto-defaults
anywhere): a render-time failure raises a named RenderError LOUD -- there is
no degrade to the 1.7B tier and no still floor (``fallback_engine = None`` on
every tier). The audio that drives HuMo is the FROZEN
master; HuMo emits an ALWAYS-SILENT clip (``has_audio`` False) -- only
``OTR_MasterAudioMux`` ever adds audio (invariant V-1).

Cold-import clean (V-12): module scope imports only stdlib + the dep-free shared
helpers + the registry. torch / the HuMo wrapper are imported LAZILY in ``load``
/ ``render_clip`` (the GPU-smoke render slice), never here. UTF-8, no BOM,
ASCII-only source.

Config (env): ``OTR_ENABLE_HUMO`` opt-in flag; ``OTR_HUMO_CKPT`` the primary
checkpoint path the load probe checks (verify-at-build; the full multi-handle
MODEL+CLIP+VAE+AUDIO_ENCODER load is confirmed on the GPU box).
"""
from __future__ import annotations

import logging
import os

from . import motion_common as _MC
from .._otr_shared.still_plan_helpers import StillPlanRow
from .frame_contract import CONTINUITY_SOFT_REFERENCE, FrameContract
from .registry import EngineUnusable, EngineUsabilityReason, register
from .wan_shared import ffprobe_clip_fields, validate_silent_clip_contract

#: Module logger, matching the other video engines' naming.
#:
#: ADDED 2026-08-02, and it was overdue: ``_LOG`` was already referenced in the
#: beat-teardown handler below but never defined anywhere in this module. That
#: handler only runs when ``reclaim_idle_models`` raises during a MULTI-CLIP
#: teardown, so the missing name sat harmless until multi-clip went live -- at
#: which point a failed reclaim would have raised ``NameError`` from inside the
#: ``except`` block and thrown away the real exception it was trying to report.
#: A broken error path is invisible precisely until the moment it is needed.
_LOG = logging.getLogger("OTR.video.humo")

_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
_COMFY_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))

#: PROMPT-STYLE OVERLAY -- STORED, NOT WIRED (item C, 2026-08-17). Schema, caps
#: and the adoption gate: 2026-08-17-per-engine-prompt-style-guide-RESEARCH.md
#: in the docs dir -- deliberately named WITHOUT a path prefix, because
#: ``tools/engine_matrix.py`` scrapes engine sources for cap-evidence citations
#: and a phrasing doc is not frame evidence. The directive is the only half that
#: may ever reach a model or a prompt; 240 chars, hard, pinned by
#: ``tests/test_prompt_style_directives.py``.
PROMPT_STYLE_DIRECTIVE = (
    "Keep it short: long prompts crowd the speech tokens. Name the speaker and "
    "the framing, then stop. Keep the mouth visible and the head toward camera. "
    "Add no movement the beat does not state."
)

#: Humans only -- never injected, never sent to a model.
PROMPT_STYLE_NOTES = """\
CONFIG AS SHIPPED: HuMo 14B (plus a 1.7B tier) for audio-driven talking
portraits, native portrait 480x832 @ 25fps. The 14B and the 1.7B PORTRAIT tiers
run cfg 1.0, so the negative is INERT there; the 1.7B LANDSCAPE tier runs cfg 2.5,
where it is LIVE. Lip-sync fidelity is the priority on every tier.

"KEEP IT SHORT" IS FIRST BECAUSE IT IS THE ONLY RULE HERE WITH A MEASUREMENT
BEHIND IT, and it is the strongest brevity case in the stack. The P4 probe
measured articulation collapsing 4.15 -> 1.18 from a prompt-register change, and
Fable's reading of that result is that long prompts drown the speech tokens. On a
lane whose entire output is a mouth matching supplied audio, that is not a quality
preference -- it is the deliverable degrading. "Then stop" is deliberately blunt
for the same reason: the failure mode is a writer who keeps going because nothing
told it to halt.

"ADD NO MOVEMENT THE BEAT DOES NOT STATE" IS THE SCAFFOLD'S ``Motion: NONE``
SEMANTICS, NOT A RETURN TO DAMPING. Read the distinction carefully, because the
2026-08-17 kinetic amendment ripped "SUBTLE motion" out of the clause writer on
the LTX lane and that directive must not be quietly undone here. The difference:
the LTX defect was an instruction forbidding motion the BEAT DID ask for. This is
a prohibition on inventing motion the beat did NOT ask for -- the research
scaffold's own wording, and it names the reason the operator left the two
surviving "subtle" strings -- ``render_driver._IA2V_TALKING_CLAUSE_CHARACTER`` and
``_CHAR_FACE_FALLBACK_PROMPT`` -- deliberately unchanged rather than
find-and-replacing them: on a close-up lip-sync beat, kinetic motion can walk the
subject out of their own frame. Fidelity to the beat, in both directions.
(GO_FORWARD_PLAN cites those as ``render_driver.py:1354`` and ``:1483``; both are
STALE. Cite the SYMBOL, never the address -- every line number in these notes
would have rotted the moment this diff inserted lines above it.)

THE MIXED-cfg TIER SPLIT IS WHY THE DIRECTIVE NEVER MENTIONS THE NEGATIVE. One
engine here spans cfg 1.0 (inert) and cfg 2.5 (live) depending on tier, so any
directive clause about negative phrasing would be true on one tier and false on
another within the same 240 chars. The ownership answer settles it anyway -- video
negatives are frozen recipe -- but this engine is the clearest case that the
overlay cannot carry per-tier conditioning facts and should not try.

THIS ENGINE ALSO CARRIES AN UNGATED ENV NEGATIVE (D-TER): it reads
``OTR_HUMO_NEGATIVE`` unconditionally on every production render, where
``eng_ltx_8gb`` keeps its equivalent behind ``_prequalification_active()``. Cited
by env-var name rather than line, because GO_FORWARD's ``:748`` shifted by exactly
the 56 lines this diff inserted above it. Open static-audit finding, needs the
operator, untouched by this overlay.

PROVENANCE: authored by the driver from this engine's shipped configuration plus
the five directive rules in the RESEARCH doc. NOT a measured finding and not a
research-panel output -- the three pasted research rounds specify the consumer
scaffold and leave the per-engine overlay as a placeholder. Treat this string as
a hypothesis until the probe A/B runs at a fixed seed.
"""

# A-ship in-process forward (the native graph PROVEN in scripts/render_humo_batch.py
# build_humo_prompt, verified on this RTX 5080 at 480x832 fp8). The graph TOPOLOGY +
# widget values are legacy-proven; the exact checkpoint / encoder FILENAMES are
# env-overridable (VERIFY-ON-GPU) so the operator confirms them against the
# installed models without editing code. Native portrait 480x832 @ 25 fps; the Wan
# 2.1 VAE 4n+1 length rule is enforced via wrapper_bridge.quantize_frames_4n1.
_HUMO_MIN_FRAMES = 33          # below this has hung this hardware (legacy floor)
_HUMO_MAX_FRAMES = 177         # last empirically verified ceiling at 480x832 fp8
# THE 14B CAP, UNIFIED ACROSS ORIENTATIONS (2026-08-02).
#
# WHAT IT USED TO SAY, and why it is gone: "the 14B fp8 tier rides ~15.9 GB at
# 832x480 ... 49 (4n+1) is the bakeoff-proven safe length
# (docs/2026-06-27-humo-bakeoff)". Two things were wrong with that.
#
#   1. THE CITED RECEIPT IS NOT IN THIS REPO. `docs/2026-06-27-humo-bakeoff`
#      has never existed here -- only the scripts that would have produced it.
#      A number nobody can check gated the heaviest engine, and it asserted
#      15.9 GB against a 14.5 GB working ceiling, which is a breach on its face.
#   2. IT APPLIED TO ONE ORIENTATION ONLY. The same checkpoint ran uncapped to
#      177 in PORTRAIT and was pinned to 49 in LANDSCAPE -- at an identical
#      399,360 pixels. External research (2026-08-02, grade A) settles the
#      architecture: HuMo/Wan use square patch `[1,2,2]`, a GLOBAL attention
#      window `[-1,-1]`, and RoPE reshaped to `f*h*w`, so 480x832 and 832x480
#      both yield a 1,560-token DiT grid per latent time. Swapping height and
#      width changes positional VALUES, not allocation SIZES. There is no
#      published counterexample and no mechanism for a 3.6x frame-cap ratio.
#
# So the orientation split was never physics. Both 14B routes now share ONE cap.
#
# WHY 97, and not 177 or 49. 97 is the length HuMo was TRAINED at, and its
# authors warn that longer generation may degrade -- a quality bound with a real
# citation, which is more than either previous number had. It also moves each
# route in its safe direction: portrait comes DOWN from an unqualified 177, and
# landscape comes UP from a cap whose evidence is missing.
#
# WHAT IS STILL OWED. 97 is a QUALITY-supported target, not a measured memory
# ceiling -- nobody has published a peak for this checkpoint on a 16 GB card at
# either orientation. This engine now carries a `VramPeakProbe` (it was the only
# heavy lane without one), so the paired ladder 49/65/81/97 in BOTH orientations
# can finally be measured on this box. The prediction is that the two
# orientations match at every rung; if they do not, the defect is in tiling or
# kernel selection, not in the model. Until that ladder runs, this number is a
# reasoned bound, and it says so.
#
# env override: OTR_HUMO_14B_SAFE_FRAMES.
_HUMO_14B_SAFE_RENDER_FRAMES = 97
# (HuMo render dims live in _otr_shared.aspect now -- portrait 480x832 / wide
# 832x480 -- selected by each engine's render_aspect, so the constants moved out.)
# An ASCII negative (CLAUDE.md: ASCII-only source). HuMo's best negative is the
# ByteDance Chinese default; set OTR_HUMO_NEGATIVE to it on the box to match the
# legacy template exactly.
_HUMO_DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, jpeg artifacts, distorted, deformed, "
    "extra fingers, bad hands, bad face, static, watermark, text")

# Production keystone tier baked in as the DEFAULT (the FAST pre-refactor config
# that scored 37.5/45): the 14B Kijai fp8 UNET + the lightx2v 480p distill LoRA
# at 6 steps / cfg 1.0 / ModelSamplingSD3 shift 8. The LoRA, steps, cfg and shift
# defaults below already match this tier; only the UNET name differs from the
# legacy 1.7B placeholder, so this is the single value that pins the tier. The
# weight lives in diffusion_models and is resolved via ComfyUI folder_paths, so
# it is found through extra_model_paths.yaml without a hardcoded box path. The
# 1.7B tier (no-LoRA, ~20 steps) is far slower; override OTR_HUMO_UNET_NAME /
# OTR_HUMO_CKPT + OTR_HUMO_LORA_NAME=none + OTR_HUMO_STEPS only to fall back to it.
_HUMO_DEFAULT_UNET = "Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors"


#: S1 (2026-07-25) per-model still plan for the PORTRAIT HuMo engines --
#: ``humo`` and ``humo_1.7B``, both shipping ``render_aspect="portrait"``
#: (spec section 3, Shape A -- scene spine, audio-driven-face variant with
#: portrait REQUIRED). FILE-LOCAL, fully declared.
#:
#: S1b SPLIT (2026-07-25): S1 had all FOUR HuMo engines sharing ONE plan
#: object across TWO shipped aspects. That was survivable only while the
#: portrait row's ``framing_geometry`` was a paraphrase. It is not survivable
#: now: the producer picks ``PORTRAIT_GEOMETRY`` for a portrait renderer and
#: ``WIDE_PORTRAIT_GEOMETRY`` for a wide one (``_style_anchor_for_aspect`` --
#: a three-quarter body shot decapitates the subject in a short landscape
#: frame, the 2026-06-17 operator catch), so one static string cannot serve
#: both. The wide siblings get their own plan below. This also honours the
#: spec's own "no inheritance or shared defaults" rule, and
#: ``tests/test_still_plan_layer2_parity.py`` now enforces it so the sharing
#: cannot regrow.
#:
#: The ``aspect="inherit_engine"`` FIELD is unchanged -- it still drives the
#: DIMENSIONS through ``resolve_row_aspect``. Only the authored layer-2 TEXT
#: is pinned per file. S2's HuMo cutover is the FOUR hosts-off bookend
#: role-cells (portrait humo / humo_1.7B x announcer / music, redirected by
#: ``_enforce_radio_is_host`` to the WIDE ``ltx_audio_in`` that renders them);
#: with ``OTR_ENABLE_HUMO_HOSTS=1`` a portrait HuMo keeps its portrait still.
_HUMO_STILL_PLAN = (
    StillPlanRow(kind="scene_open", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     "full-frame macro, centered subject"),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_beat", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     ("cinematic three-quarter framing, people shown with "
                      "full heads and clear headroom inside frame, faces "
                      "unobstructed, balanced composition")),
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
                 required="always",
                 framing_geometry=(
                     ("in-character cinematic three-quarter portrait, full "
                      "head and face clearly visible with natural headroom "
                      "above the head (never crop the top of the head)")),
                 style_tail_policy="full"),
)

#: S1b (2026-07-25) per-model still plan for the LANDSCAPE HuMo siblings --
#: ``humo_1.7B_169`` and ``humo_14B_169``, both shipping
#: ``render_aspect="wide"``. Structurally IDENTICAL to
#: :data:`_HUMO_STILL_PLAN`; it exists as a SEPARATE plan object because the
#: portrait row's layer-2 text differs by shipped aspect (see the split note
#: above). Fully declared -- no inheritance from, and no shared rows with,
#: the portrait plan, per spec section 6.
#:
#: The ComfyUI dropdown labels this split to the operator as "(portrait)" vs
#: "(16:9)" and that is a VISIBLE PRODUCT CONTRACT: these two siblings were
#: ALREADY wide before this build, nothing about them flips, and a blanket
#: "HuMo -> wide" would be a regression rather than a fix.
_HUMO_169_STILL_PLAN = (
    StillPlanRow(kind="scene_open", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     "full-frame macro, centered subject"),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_beat", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     ("cinematic three-quarter framing, people shown with "
                      "full heads and clear headroom inside frame, faces "
                      "unobstructed, balanced composition")),
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
                 required="always",
                 framing_geometry=(
                     ("in-character cinematic medium shot, head and "
                      "shoulders, face clearly visible, subject centred with "
                      "natural headroom above the head (never crop the top of "
                      "the head)")),
                 style_tail_policy="full"),
)


@register
class HuMoEngine(_MC.MotionEngineBase):
    """The humo audio-driven-face adapter (in-process, default-OFF / dark)."""

    name = "humo"
    family = "audio_driven_face"
    #: THE CANVAS, DECLARED (lane 4, 2026-08-11) -- the LAST HuMo tier to get
    #: one, which is why it held the "declares NOTHING" differential control
    #: until now. Portrait 480x832 is where the 14B was measured on both boot
    #: lanes (14.98 GiB default / 13.22 GiB diet warm, HUMO_BAKEOFF +
    #: ENVELOPE_LADDERS, indexed in docs/evidence/video_evidence_manifest.json).
    #: Without it the request fell through to the 1472x832 landscape default on
    #: the tier whose whole job is the pillarbox talking head. Both axes
    #: /32-legal (15 x 26). A contradicting OTR_HUMO_WIDTH/HEIGHT is a named
    #: refusal; see :meth:`HuMoEngine._native_dims`.
    render_canvas = (480, 832)
    #: BOOT CONTRACTS. Both, for the same reason as every other HuMo tier: it
    #: has shipped under `default`, so requiring the diet would retire a
    #: shipping lane. What `default` COSTS is stated rather than hidden --
    #: 14.98 GiB, over the 14.5 GiB gate.
    compatible_boot_contracts = ("default", "humo_diet")
    #: S1 per-model still plan (see ``_HUMO_STILL_PLAN`` above -- audio-
    #: driven-face with portrait REQUIRED).
    still_plan = _HUMO_STILL_PLAN
    # RADIO IS THE HOST (reversal of Route-A 2026-06-28, operator 2026-06-30):
    # HuMo's finetuned weights only animate a FACE. Route-A had added
    # announcer_visual/music_visual here + a radio-face-still workaround
    # (render_driver._radio_bookend_image, since REMOVED) so the operator could
    # pick HuMo for the music/announcer bookends; the eyeballed render showed a
    # generic human host, not a radio -- confirming the ORIGINAL 2026-05-01
    # BUG-LOCAL-129 finding (nodes/_otr_speaker_role.py) still holds. HuMo now
    # serves ONLY character_video (real dialogue lip-sync). This tuple is
    # UI-SORT/self-description metadata only (role_compat.engine_fits_role is
    # capability-only and does not consult it -- see registry.VideoEngineRegistry
    # docstring); the actual hard gate is render_driver._enforce_radio_is_host,
    # which wires the previously-dormant _otr_speaker_role.is_never_humo_role
    # into real dispatch and redirects any announcer_visual/music_visual +
    # audio_driven_face pick to ltx_audio_in (the LTX-2.3 audio-in lane already
    # DEFAULT for those two roles -- see eng_ltx_av.py default_roles).
    roles = ("character_video",)
    default_roles = ()
    required_inputs = ("audio_ref", "init_image")
    #: THE FRAME LADDER (chunk 7a, 2026-07-26). ``_HUMO_MIN_FRAMES`` ..
    #: ``_HUMO_MAX_FRAMES`` -- 33 is the legacy floor (below it has hung this
    #: hardware) and 177 is the last empirically verified ceiling at 480x832
    #: fp8. 4n+1 stride. Inherited by all four HuMo variants.
    #: CONTINUITY: soft_reference. The graph wires LoadImage ->
    #: WanHuMoImageToVideo ``ref_image``, NOT ``start_image``: per the Bug
    #: Bible the reference is an identity hint, not a first-frame lock. These
    #: beats take an honest jump cut rather than a pretended chain.
    #: THE CONTRACT MUST MATCH WHAT ``render_clip`` WILL ACTUALLY RENDER. This
    #: is the inheritance trap ``HuMo14BLandscapeEngine`` documents below and it
    #: applied here too: ``render_clip`` does
    #: ``render_max = cap if cap is not None else _HUMO_MAX_FRAMES``, so once the
    #: 14B portrait route carries a cap, advertising 177 would promise 1.8x the
    #: frames it can produce and every beat above the cap would raise.
    frame_contract = FrameContract(
        min_frames=_HUMO_MIN_FRAMES,
        max_frames=_HUMO_14B_SAFE_RENDER_FRAMES,
        quantum=4,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_SOFT_REFERENCE,
    )
    commercial_clean = False            # license is profile data; verify-at-build
    requires_flag = None  # vestigial (registry IS the menu; no flag gate)
    engine_version = "1"
    declared_isolation = _MC.ISOLATION_IN_PROCESS
    target_fps = 25
    #: Per-beat render-frame cap (Route-A). ``None`` = uncapped (use
    #: ``_HUMO_MAX_FRAMES``, the legacy behaviour for humo / humo_1.7B*). A heavy
    #: tier that rides thin VRAM headroom (the 14B) overrides this with a safe
    #: integer; render_clip then renders at the cap and EXACT-FITS the decoded
    #: frames to the audio-derived target (trim/mirror-extend) so frame_count
    #: still matches.
    #: THIS CLASS IS THE 14B PORTRAIT ROUTE -- it loads the same
    #: ``Wan2_1-HuMo-14B_fp8`` checkpoint as ``humo_14B_169``, at the same
    #: 399,360 pixels. Until 2026-08-02 it was ``None`` (uncapped to 177) while
    #: its landscape twin was pinned to 49, and nothing but orientation
    #: separated them. Both now share the 14B cap.
    #:
    #: The 1.7B subclasses declare their OWN ladder and their own ``None``
    #: (see ``HuMo17BEngine``): they load a different, far lighter checkpoint,
    #: so inheriting a 14B memory bound would be a bound about the wrong model.
    safe_render_frames = _HUMO_14B_SAFE_RENDER_FRAMES
    #: Render aspect == this engine's IDENTITY (the operator picks the look with
    #: one dropdown choice): 'portrait' 480x832 (the classic pillarbox) here on
    #: the base; the humo_1.7B_169 variant overrides to 'wide' 832x480. The
    #: character still + composite canvas follow this (see _otr_shared.aspect).
    render_aspect = "portrait"
    fallback_engine = None               # NO FALLBACKS (2026-07-02): fail LOUD

    # ---- config resolution (env override -> box default) ----
    @staticmethod
    def _resolve_unet(env_explicit, env_name, default_name):
        """ONE UNET resolver for every HuMo tier (lane 2, 2026-08-11).

        Order: an explicit full-path env wins; then ComfyUI ``folder_paths``
        (which honours ``extra_model_paths.yaml`` and is the resolution the
        LOADER node will actually do); then the standard
        ``<comfy_root>/models/diffusion_models``; then the CONFIGURED models
        root. Returns the joined comfy-root path when nothing exists anywhere,
        so the refusal message still names a concrete place it looked.

        THE LAST PROBE IS THE LANE-1 LESSON, INHERITED (L1). Both HuMo
        ``_ckpt_path`` implementations stopped at the comfy-root join, which on
        a box that keeps its weights in a configured root elsewhere answers a
        confident wrong NO off the ComfyUI runtime -- the identical shape that
        shipped wan_i2v dead. Two copies of the same chain was also two places
        to fix it, so there is now one.
        """
        if env_explicit and os.environ.get(env_explicit):
            return os.environ[env_explicit]
        name = os.environ.get(env_name) or default_name
        try:
            import folder_paths  # type: ignore
            hit = folder_paths.get_full_path("diffusion_models", name)
            if hit:
                return hit
        except Exception:  # noqa: BLE001 - headless/CPU: fall through to joins
            pass
        from .wan_shared import configured_models_root
        joined = os.path.join(_COMFY_ROOT, "models", "diffusion_models", name)
        if os.path.exists(joined):
            return joined
        configured = os.path.join(
            configured_models_root(), "diffusion_models", name)
        return configured if os.path.exists(configured) else joined

    def _ckpt_path(self):
        return self._resolve_unet(
            "OTR_HUMO_CKPT", "OTR_HUMO_UNET_NAME", _HUMO_DEFAULT_UNET)

    def _installed(self):
        """True iff the primary checkpoint exists on disk (no import -- cheap,
        headless-safe). The full MODEL+CLIP+VAE+AUDIO_ENCODER multi-handle load
        (+ the low/high/gguf tier pick) is the GPU smoke."""
        return os.path.exists(self._ckpt_path())

    # ---- sampler tier (overridable per HuMo tier; the 1.7B downgrade isolates
    # ---- its own steps/cfg so the 14B OTR_HUMO_STEPS/CFG never bleed into it) --
    def _steps(self):
        """KSampler steps for this tier. 14B + lightx2v distill = 6 (fast)."""
        return int(os.environ.get("OTR_HUMO_STEPS", "6"))

    def _cfg(self):
        """KSampler cfg for this tier. 14B + lightx2v distill = 1.0."""
        return float(os.environ.get("OTR_HUMO_CFG", "1.0"))

    # ---- usability (fail-closed BEFORE any forward; no heavy import) ----
    def assert_usable(self, host_caps, profile, request_template=None):
        """Fail closed before any forward on checkpoint presence
        (verify-at-build). Imports nothing heavy -- runs at
        lock/validate time on the CPU box. HuMo loads in-process; its
        SageAttention tolerance is a GPU-smoke verify item, NOT a hard CPU gate
        (unlike ltx_video's BUG-070 int8-PV abort)."""
        if not self._installed():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "humo checkpoint not found at %s; install the HuMo wrapper + "
                "ckpt and verify on the GPU box (set OTR_HUMO_CKPT)"
                % self._ckpt_path(), kind="video")
        self._assert_boot_contract(profile)
        return self.name

    def _assert_boot_contract(self, profile):
        """The profile's boot contract must be one this tier is proven on, and
        the RUNNING SERVER must actually be in that state (S8, lane 2).

        Two checks, deliberately separate. The first is pure and always runs:
        does this engine claim the contract the profile selected? The second
        probes ``comfy.cli_args`` -- what this process was really started with
        -- and is a NO-OP off a ComfyUI server, because the CPU suite cannot
        know and pretending otherwise would make the branch unreachable while
        the tests stayed green.

        Why the server probe and not the profile text: a check that reads the
        same config the launcher was supposed to honour cannot distinguish
        "applied" from "written down", so a dead channel would produce a
        falsely PASSING check -- which is precisely how `launch.extra_args`
        went years describing a clamp it never applied.
        """
        from .._otr_shared import boot_contracts as _bc
        problems = _bc.check_engine_against_profile(self, profile or {})
        if problems:
            raise EngineUnusable(
                self.name, self.family,
                EngineUsabilityReason.INCOMPATIBLE_PROFILE,
                "; ".join(problems), kind="video")
        _bc.assert_running_server(_bc.contract_for_profile(profile or {}))

    #: Terminal node of the graph (its IMAGE output is encoded to the clip).
    _TERMINAL = "vaedecode"

    # ---- beat session (WIRE-W4a, 2026-07-29) ----
    #: The ComfyUI ``folder_paths`` categories each loader token resolves
    #: through -- the same lookup the loader NODE will do, which is the only
    #: resolution the identity may describe (a ``*_DIR``-style override that
    #: the loader cannot see must not be able to satisfy it).
    _LOADER_CATEGORIES = {
        "unet": ("diffusion_models", "unet"),
        "lora": ("loras",),
        "clip": ("text_encoders", "clip"),
        "vae": ("vae",),
        "whisper": ("audio_encoders",),
    }

    def _humo_file_receipt(self, path):
        """A BOUNDED, stable receipt for a model file: (basename, size,
        mtime_ns). Deliberately NOT a content hash -- ``session_identity()`` is
        re-read before EVERY segment and hashing a 14B UNET per segment would
        cost more than the render it protects. Size + mtime catches a swapped
        or rebuilt weight, which is the drift being guarded against.

        Its own four lines ON PURPOSE. ``wan_shared`` says the same thing for
        the WAN lanes and ``eng_ltx_8gb`` for its own, and that file's docstring
        records why: reaching across lanes for a four-line stat is the coupling
        this build keeps paying for. The MECHANISM is a convention; the DATA is
        per adapter.

        ``None`` when the file cannot be stat'ed -- the CALLER decides whether
        that is fatal, because the caller knows which NAMED refusal to raise."""
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (os.path.basename(path), int(st.st_size), int(st.st_mtime_ns))

    def _humo_token_path(self, categories, name):
        """Where the LOADER would find ``name``: ``folder_paths`` first (so
        ``extra_model_paths.yaml`` is honoured), then the standard
        ``models/<category>/`` layout. ``None`` when absent everywhere -- the
        offline invariant means a missing file is fail-closed, never fetched."""
        for category in categories:
            try:
                import folder_paths                  # ComfyUI runtime only
                hit = folder_paths.get_full_path(category, name)
                if hit:
                    return hit
            except Exception:            # noqa: BLE001 -- no ComfyUI (tests)
                pass
            cand = os.path.join(_COMFY_ROOT, "models", category, name)
            if os.path.exists(cand):
                return cand
        return None

    def _humo_session_receipts(self):
        """``(label, token, receipt)`` for every file this beat's handles are.

        Built from the adapter's OWN ``_loader_names()`` so a tier that points
        at a different UNET, LoRA or encoder gets it in the identity for free
        -- the three HuMo subclasses override exactly that method and nothing
        here needs to know they exist."""
        names = self._loader_names()
        out = [("unet", names["unet"],
                self._humo_file_receipt(self._ckpt_path() or ""))]
        for label in ("lora", "clip", "vae", "whisper"):
            token = names.get(label) or ""
            if label == "lora" and self._lora_is_skipped(token):
                out.append((label, "", None))       # a tier that runs LoRA-free
                continue
            path = self._humo_token_path(
                self._LOADER_CATEGORIES[label], token) if token else None
            out.append((label, token,
                        self._humo_file_receipt(path) if path else None))
        return tuple(out)

    @staticmethod
    def _lora_is_skipped(lora_name):
        """THE ONE reading of "this tier runs LoRA-free".

        The lightx2v distill LoRA is a 14B-shaped adapter and is INCOMPATIBLE
        with the 1.7B tiers, which switch it off by name. Both the graph and
        the session have to agree about that: a session that hoisted a ``lora``
        node the graph never defines would wire a handle nothing reads, and one
        that skipped a node the graph DOES define would let it reload every
        segment. This used to be spelled out inside ``_build_graph`` alone."""
        return (not lora_name) or str(lora_name).strip().lower() in (
            "none", "skip", "off")

    def _session_node_ids(self):
        """The graph ids ``prepare`` hoists: every pure file-to-handle LOADER.

        WIDER THAN THE WAN LANES, and the difference is a real property of this
        family rather than a preference. WAN renders with ``free_after_use=True``
        precisely so umt5 and the diffusion UNET are never co-resident, so
        hoisting its CLIP would delete that mitigation. HuMo renders FULLY
        RESIDENT by contract (BUG-265: forcing inter-node eviction fragmented
        the allocator into an OOM), so every loader is resident for the whole
        render anyway -- hoisting them changes how many times they are READ,
        not how much is held.

        It is also what makes skipping the between-segment reclaim safe. See
        :meth:`render_clip`: without the hoist, segment 2's graph would build
        NEW umt5 and whisper handles beside the ones segment 1 left resident,
        and dropping the reclaim would accumulate rather than reuse.

        ``modelsampling`` is deliberately NOT here -- it is a cheap clone+patch
        of an already-resident MODEL and is correctly per-render. Neither are
        ``loadaudio``/``audioenc``: those are the one thing that genuinely
        DIFFERS per segment on a lip-synced lane."""
        ids = ["unet", "clip", "vae", "audioenc_loader"]
        if not self._lora_is_skipped(self._loader_names().get("lora")):
            ids.insert(1, "lora")
        return tuple(ids)

    def session_identity(self):
        """What this adapter's HANDLES are. ``BeatSession`` refuses a
        multi-segment beat without it -- which is why all three HuMo lanes came
        back NO_RENDER on the 2026-07-28 campaign.

        It carries the engine name (which is what picks the tier's steps, cfg
        and native dims), every loader file's token AND receipt, and whether
        the LoRA is MERGED -- because it is merged into the hoisted patcher, so
        a beat that turned it off midway would be reusing a model the later
        segments' graph no longer describes.

        It EXCLUDES per-segment state: prompt, seed, frame count, audio slice.
        It also excludes steps/cfg, which are KSampler arguments rather than
        anything baked into a handle.

        Not cached -- the whole job is to notice a weight that MOVED, so the
        receipts are re-stat'ed on every ask."""
        return (self.name,
                "lora_merged=%s" % (
                    not self._lora_is_skipped(self._loader_names().get("lora")),),
                repr(self._humo_session_receipts()))

    def prepare(self, host_caps, profile, session_ctx):
        """Load every heavy handle ONCE PER BEAT instead of once per segment.

        The ``eng_wan_i2v`` sequence, followed deliberately: ``super().prepare()``
        takes the cross-process GPU lease and resolves classes, a LOADER-ONLY
        mini-graph runs with ``on_result`` registering each patcher as it lands
        (if a later node raised, ``run_graph`` never returns a results dict and
        an unregistered patcher is VRAM nothing will ever detach), and ANY
        failure tears down what was taken before re-raising.

        ``session_ctx`` is KEPT, because :meth:`render_clip` has no other way to
        know whether it is one segment of a beat -- and on this lane that
        decides whether the post-decode VRAM reclaim runs now or at teardown.

        NO ``free_after_use``: this graph is executed exactly the way
        ``render_clip`` executes its own, because BUG-265 established that
        forcing inter-node eviction on this stack fragments the allocator into
        an OOM."""
        from . import wrapper_bridge as _wb
        # session_ctx comes back IN `prepared` from the base now (2026-08-01) --
        # the local re-add that used to live here is the base's job.
        prepared = super().prepare(host_caps, profile, session_ctx)
        try:
            classes = getattr(self, "_classes", None) \
                or _wb.resolve_graph_classes(self._node_candidates())
            bucket = prepared.setdefault("patchers", self._patchers)
            seen = {id(p) for p in bucket}

            def _register(nid, out):
                obj = out[0] if out else None
                if (obj is not None and id(obj) not in seen
                        and callable(getattr(obj, "detach", None))):
                    bucket.append(obj)
                    seen.add(id(obj))

            wanted = set(self._session_node_ids())
            # BUILT FROM THE REAL GRAPH, never hand-copied. A second literal
            # spelling of the loader inputs is a second thing to keep in step
            # with `_build_graph`, and the two drifting is exactly how a
            # session hoists a handle the render graph would have built
            # differently.
            full = self._build_graph("_session_.png", "_session_.wav",
                                     {"seed": 0}, _HUMO_MIN_FRAMES,
                                     *self._native_dims())
            loaders = {nid: spec for nid, spec in full.items() if nid in wanted}
            missing = wanted - set(loaders)
            if missing:
                raise _wb.GraphExecutionError(
                    "%s wanted to hoist %s but its render graph defines no such "
                    "node(s) -- the session and the graph disagree about what "
                    "this lane loads. NO FALLBACK."
                    % (self.name, sorted(missing)))
            results = _wb.run_graph(loaders, classes, on_result=_register)
            absent = [nid for nid in wanted if not results.get(nid)]
            if absent:
                raise _wb.GraphExecutionError(
                    "%s hoisted loader(s) %s returned no output; the beat "
                    "session has nothing to reuse and every segment would "
                    "silently reload" % (self.name, sorted(absent)))
            prepared["external_results"] = {
                nid: results[nid] for nid in wanted}
        except BaseException:
            try:
                self.teardown(prepared)
            except BaseException:          # noqa: BLE001 -- never mask the cause
                pass
            raise
        return prepared

    def teardown(self, prepared):
        """Drop the beat-scoped handles, RECLAIM once, then delegate.

        ``prepared["external_results"]`` holds strong references to every
        hoisted handle; the base teardown detaches the tracked patchers,
        unloads, releases the lease and then waits for machine-wide VRAM to
        settle, and every one of those steps would run with this dict still
        pinning the tensors it is reclaiming.

        THE RECLAIM MOVES HERE FOR A MULTI-SEGMENT BEAT (see
        :meth:`render_clip`). It exists to stop CROSS-BEAT accumulation -- "the
        resident stack drops back down before the next soak beat starts" -- and
        between two segments of ONE beat there is nothing to drop back for,
        because the next segment needs the same stack. Running it once here
        keeps the cross-beat promise exactly.

        Never raises out of teardown (it is a finally-path)."""
        from . import wrapper_bridge as _wb
        if isinstance(prepared, dict):
            prepared.pop("external_results", None)
            if (prepared.get("session_ctx") or {}).get("multi_clip"):
                try:
                    _wb.reclaim_idle_models(reason="humo beat teardown")
                except Exception:          # noqa: BLE001 -- best-effort
                    _LOG.warning("[OTR video] %s beat-teardown reclaim failed",
                                 self.name, exc_info=True)
        return super().teardown(prepared)

    # ---- in-process graph spec (proven topology; filenames VERIFY-ON-GPU) ----
    def _node_candidates(self):
        """Ordered ComfyUI node-class candidates per graph node (all core /
        comfy_extras classes used by the proven legacy HuMo graph)."""
        return {
            "unet": ("UNETLoader",),
            "lora": ("LoraLoaderModelOnly",),
            "modelsampling": ("ModelSamplingSD3",),
            "clip": ("CLIPLoader",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "vae": ("VAELoader",),
            "loadaudio": ("LoadAudio",),
            "audioenc_loader": ("AudioEncoderLoader",),
            "audioenc": ("AudioEncoderEncode",),
            "loadimage": ("LoadImage",),
            "humo": ("WanHuMoImageToVideo",),
            "ksampler": ("KSampler",),
            "vaedecode": ("VAEDecode",),
        }

    def _loader_names(self):
        """Model / encoder FILENAMES the loader nodes consume. Defaults are the
        legacy-proven names; each is env-overridable so the operator points at the
        installed files without editing code (VERIFY-ON-GPU)."""
        return {
            "unet": os.environ.get("OTR_HUMO_UNET_NAME")
            or os.path.basename(self._ckpt_path()),
            "lora": os.environ.get(
                "OTR_HUMO_LORA_NAME",
                "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"),
            "clip": os.environ.get(
                "OTR_HUMO_CLIP_NAME", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
            "vae": os.environ.get("OTR_HUMO_VAE_NAME", "wan_2.1_vae.safetensors"),
            "whisper": os.environ.get(
                "OTR_HUMO_AUDIO_ENCODER_NAME", "whisper_large_v3_fp16.safetensors"),
        }

    def _native_dims(self):
        """HuMo's render size from this engine's ``render_aspect``: 'portrait'
        480x832 (the classic talking-head pillarbox, humo_1.7B) or 'wide' 832x480
        (the 16:9 character beat, humo_1.7B_169). The aspect is the engine's
        identity, so the operator picks the look with ONE dropdown choice.

        ON A TIER THAT DECLARES A CANVAS, THE ENV OVERRIDES CANNOT WIN
        (lane 2, 2026-08-11). ``render_canvas`` is applied LAST in the request
        chain and overrules ledger and env, so a tier that declares one and then
        renders at ``OTR_HUMO_WIDTH``/``HEIGHT`` would reopen the exact
        request-versus-render disagreement the declaration exists to close --
        and it would reopen it INVISIBLY, because the request would keep saying
        the declared size. Two channels naming one canvas are not allowed to
        disagree, so a contradicting override is a NAMED refusal rather than a
        silent winner. An override that AGREES is fine and passes through; a
        tier that declares nothing keeps the old behaviour untouched.
        """
        from .._otr_shared.aspect import humo_dims_for_aspect
        _dw, _dh = humo_dims_for_aspect(self.render_aspect)
        w = int(os.environ.get("OTR_HUMO_WIDTH", _dw))
        h = int(os.environ.get("OTR_HUMO_HEIGHT", _dh))
        declared = getattr(self, "render_canvas", None)
        if declared and (w, h) != (int(declared[0]), int(declared[1])):
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                "%s declares render_canvas %dx%d, but OTR_HUMO_WIDTH/HEIGHT "
                "ask for %dx%d. The declaration is applied last and overrules "
                "the request, so honouring the override here would render one "
                "size while every receipt reported the other. Unset the "
                "overrides, or set them to the declared canvas."
                % (self.name, int(declared[0]), int(declared[1]), w, h),
                kind="video")
        return w, h

    def _build_graph(self, image_name, audio_name, plan, length, width, height,
                     external_results=None):
        """The declarative HuMo graph (wrapper_bridge.run_graph format). Mirrors
        the proven build_humo_prompt wiring node-for-node:
        UNETLoader->LoRA->ModelSamplingSD3->KSampler (model); CLIPLoader->pos/neg;
        VAELoader; LoadAudio->AudioEncoderEncode(+AudioEncoderLoader);
        LoadImage->ref_image; WanHuMoImageToVideo->KSampler->VAEDecode.

        ``external_results`` names the ids the CALLER already produced and owns
        for the whole beat (WIRE-W4a: :meth:`prepare` loads every loader once).
        Those node DEFINITIONS are omitted while every wire that reads them
        stays -- the executor resolves them from the caller's handles instead.
        Omission is a contract, not an optimisation: ``run_graph`` REFUSES a
        graph that also defines an id the caller supplied, because then two
        parties claim to produce one output.

        Conditional on purpose. A caller that prepared nothing gets the loader
        nodes exactly as before, byte for byte -- which is the single-clip path
        and every sibling test that hand-builds ``prepared={"patchers": []}``."""
        from . import wrapper_bridge as _wb
        names = self._loader_names()
        steps = self._steps()
        cfg = self._cfg()
        positive = plan.get("text_prompt") or "a person speaking, subtle facial motion"
        negative = os.environ.get("OTR_HUMO_NEGATIVE", _HUMO_DEFAULT_NEGATIVE)
        W = _wb.Wire
        # The lightx2v distill LoRA is a 14B-shaped adapter: it is INCOMPATIBLE
        # with the 1.7B tier (shape-mismatch -> not merged + wasted VRAM). Make
        # it optional so the 1.7B tier runs LoRA-free (set OTR_HUMO_LORA_NAME to
        # none/skip and raise OTR_HUMO_STEPS, since the distill shortcut is gone).
        lora_name = names["lora"]
        skip_lora = self._lora_is_skipped(lora_name)
        graph = {
            "unet": {"class": "unet",
                     "inputs": {"unet_name": names["unet"],
                                "weight_dtype": "default"}},
            "clip": {"class": "clip",
                     "inputs": {"clip_name": names["clip"], "type": "wan",
                                "device": "default"}},
            "pos": {"class": "pos",
                    "inputs": {"text": positive, "clip": W("clip", 0)}},
            "neg": {"class": "neg",
                    "inputs": {"text": negative, "clip": W("clip", 0)}},
            "vae": {"class": "vae", "inputs": {"vae_name": names["vae"]}},
            "loadaudio": {"class": "loadaudio", "inputs": {"audio": audio_name}},
            "audioenc_loader": {"class": "audioenc_loader",
                                "inputs": {"audio_encoder_name": names["whisper"]}},
            "audioenc": {"class": "audioenc",
                         "inputs": {"audio_encoder": W("audioenc_loader", 0),
                                    "audio": W("loadaudio", 0)}},
            "loadimage": {"class": "loadimage", "inputs": {"image": image_name}},
            "humo": {"class": "humo",
                     "inputs": {"width": int(width), "height": int(height),
                                "length": int(length), "batch_size": 1,
                                "positive": W("pos", 0), "negative": W("neg", 0),
                                "vae": W("vae", 0),
                                "audio_encoder_output": W("audioenc", 0),
                                "ref_image": W("loadimage", 0)}},
        }
        model_src = "unet"
        if not skip_lora:
            graph["lora"] = {"class": "lora",
                             "inputs": {"lora_name": lora_name,
                                        "strength_model": 1.0,
                                        "model": W("unet", 0)}}
            model_src = "lora"
        graph["modelsampling"] = {
            "class": "modelsampling",
            "inputs": {"shift": 8.0, "model": W(model_src, 0)}}
        graph["ksampler"] = {
            "class": "ksampler",
            "inputs": {"seed": int(plan.get("seed", 0)), "steps": steps,
                       "cfg": cfg, "sampler_name": "uni_pc",
                       "scheduler": "simple", "denoise": 1.0,
                       "model": W("modelsampling", 0),
                       "positive": W("humo", 0), "negative": W("humo", 1),
                       "latent_image": W("humo", 2)}}
        graph["vaedecode"] = {
            "class": "vaedecode",
            "inputs": {"samples": W("ksampler", 0), "vae": W("vae", 0)}}
        for nid in set(external_results or ()):
            graph.pop(nid, None)
        return graph

    def _retain_model_patchers(self, results, prepared):
        """Best-effort V-4: keep the MODEL ModelPatchers the graph produced so
        teardown can detach(unpatch_all=True) them (the lease + VRAM settle-wait in
        MotionEngineBase.teardown is the real residency guard)."""
        bucket = prepared.setdefault("patchers", self._patchers) \
            if isinstance(prepared, dict) else self._patchers
        seen = {id(p) for p in bucket}
        for nid in ("unet", "lora", "modelsampling"):
            out = results.get(nid)
            if not out:
                continue
            obj = out[0]
            if id(obj) not in seen and callable(getattr(obj, "detach", None)):
                bucket.append(obj)
                seen.add(id(obj))

    # ---- residency (resolve the installed wrapper nodes; weights load on call) -
    def load(self):
        """Fail CLOSED until installed, then RESOLVE the installed ComfyUI wrapper
        node classes (fail-closed NAMED if absent). The heavy weight load happens
        when the loader nodes execute inside render_clip (ComfyUI's own model
        management), so load() stays cheap and the AS-3 lease brackets the real
        residency. The live VRAM<=14.5 GB peak + render-twice pixels are the A-S6
        GPU smoke (operator)."""
        if not self._installed():
            raise RuntimeError(
                "humo not installed: checkpoint missing at %s -- install the HuMo "
                "wrapper + ckpt, set OTR_ENABLE_HUMO=1, and run the A-S6 GPU "
                "smoke" % self._ckpt_path())
        from . import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    def render_clip(self, request, prepared):
        """Drive ONE audio-driven-face clip via the in-process HuMo wrapper graph.

        Stages the portrait + speech into ComfyUI's input dir (the proven legacy
        LoadImage / LoadAudio pattern), executes the proven node graph in
        dependency order, encodes the decoded IMAGE batch to a SILENT bt709 clip
        (V-1), retains the MODEL patchers for V-4 teardown, and asserts the
        mid-render NVML ceiling. Returns the raw ``{out_path, frame_count}`` that
        canonicalize() normalises. Fail-closed NAMED if a wrapper node is missing
        or an input is absent."""
        from . import wrapper_bridge as _wb
        from ._tmp import otr_engine_tmp_mp4
        plan = self._build_render_request(request)            # pure, CPU-tested
        if not plan["audio_path"] or not plan["init_image"]:
            raise _wb.GraphExecutionError(
                "humo requires both audio_ref and init_image (got audio=%r "
                "init_image=%r)" % (plan["audio_path"], plan["init_image"]))
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates())
        audio_name = _wb.stage_into_comfy_input(plan["audio_path"])
        image_name = _wb.stage_into_comfy_input(plan["init_image"])
        target_fc = int(plan["target_frame_count"] or 0)
        # Route-A: a capped tier (the 14B) renders at its VRAM-safe cap, then
        # exact-fits the decoded frames back to the audio target below. Uncapped
        # tiers (humo / humo_1.7B*) keep the legacy 177-frame ceiling unchanged.
        cap = self.safe_render_frames
        if cap is not None:
            cap = int(os.environ.get("OTR_HUMO_14B_SAFE_FRAMES", cap))
        render_max = cap if cap is not None else _HUMO_MAX_FRAMES
        length = _wb.quantize_frames_4n1(
            target_fc or self.target_fps,
            min_frames=_HUMO_MIN_FRAMES, max_frames=render_max)
        width, height = self._native_dims()
        # The beat-scoped handles prepare() hoisted, if any. Their node
        # definitions drop out of the graph and run_graph resolves the wires
        # from these instead -- and it adds them to `keep`, so nothing can evict
        # a handle the NEXT segment still needs.
        session = (prepared or {}) if isinstance(prepared, dict) else {}
        external = session.get("external_results") or {}
        multi_clip = bool((session.get("session_ctx") or {}).get("multi_clip"))
        graph = self._build_graph(image_name, audio_name, plan, length, width,
                                  height, external_results=external)
        # Render FULLY RESIDENT -- the proven BUG-265 low_vram_default path: the
        # HuMo-1.7B stack (3.3 GB + umt5/whisper) stays resident with zero offload
        # on a 16 GB card. (No free_after_use: forcing inter-node model eviction
        # only fragmented the allocator into an OOM -- it is NOT what the working
        # OTR_BatchHumoRender does.)
        # PEAK TELEMETRY (2026-08-02). HuMo was the ONLY heavy lane with no
        # ``VramPeakProbe``, which is why its frame ceilings have never been
        # checkable: the 49-frame landscape cap asserts ~15.9 GB and cites
        # `docs/2026-06-27-humo-bakeoff`, a file that is not in this repo, and
        # nothing this engine ever ran could confirm or refute it. Every other
        # heavy engine reports its render-window peak, and `ltx_audio_in`'s 79
        # logged samples are what finally settled ITS frame question -- from
        # data already on disk. Telemetry only, no enforcement (the admission
        # boundary in ``render_driver`` owns refusal), and the log line matches
        # the other engines' wording so one grep harvests them all.
        probe = _MC.VramPeakProbe(interval_s=0.1).start()
        try:
            results = _wb.run_graph(graph, classes, external_results=external)
        finally:
            render_peak = probe.stop()
            if render_peak:
                _LOG.info("[%s] render-window VRAM peak: %d MB  "
                          "(length=%d canvas=%dx%d)",
                          self.name, int(render_peak), int(length),
                          int(width), int(height))
        images = results[self._TERMINAL][0]                   # VAEDecode IMAGE batch
        self._retain_model_patchers(results, prepared)
        frames = _wb.images_to_uint8(images)
        # Route-A exact-fit: a capped tier may have rendered FEWER frames than the
        # audio target (or, on a tiny beat quantized up to the floor, more) -- trim
        # so the encoded clip's frame_count EQUALS target_fc (else the manifest
        # timing drifts).
        #
        # NO MIRROR ON A LIP-SYNCED LANE (operator directive 2026-07-25: "lip
        # sync needs continuity, no render backwards, that doesn't work").
        # HuMo is ``audio_driven_face`` -- its whole output is a mouth driven by
        # this beat's speech. ``fit_frames_to_target`` used to mirror-extend a
        # SHORT capped render up to the audio target, and the back half of that
        # mirror cycle is the render in REVERSE, so a capped 14B beat longer
        # than the cap shipped a face that speaks forwards and then unspeaks
        # backwards, in time with forward audio. Trimming is still fine (it
        # never reverses time); a SHORT render is now TERMINAL and LOUD, with
        # the remedy in the message. This can fail a beat that previously
        # "succeeded" -- those episodes had backwards lip sync.
        # THE GUARD IS UNCONDITIONAL (S8b item 3, lane 3, 2026-08-11).
        #
        # It used to read ``if cap is not None and target_fc > 0``, which made
        # the HONESTY check conditional on a VRAM knob -- two unrelated
        # questions sharing one branch. An uncapped tier (`humo_1.7B`,
        # `humo_1.7B_169`, `humo`) declares ``safe_render_frames = None``, so it
        # skipped the exact fit entirely: a beat asking for more than the
        # 177-frame ceiling rendered 177 and returned them stamped
        # ``extension_mode: "none"`` with ``native_frame_count == frame_count``
        # -- indistinguishable from an honest clip on any path that reaches
        # ``render_shot`` without a stamped coverage plan. The video simply ran
        # out before the audio, and nothing said so.
        #
        # ``fit_frames_to_target`` TRIMS a long render (always safe, never
        # reverses time) and RAISES on a short one. Both answers are right for
        # every tier; only the message needs to know whether a cap was involved.
        #
        # This can fail a beat that previously "succeeded" -- the same trade the
        # 14B took on 2026-07-25. Those episodes shipped a face that stopped
        # moving while the line kept going.
        if target_fc > 0:
            try:
                frames = _wb.fit_frames_to_target(
                    frames, target_fc)
            except _wb.MirrorExtensionForbidden as exc:
                if cap is not None:
                    limit_note = (
                        "This tier's VRAM-safe cap is %d frame(s) and the beat "
                        "needs %d. Raise OTR_HUMO_14B_SAFE_FRAMES if the card "
                        "can afford it, route this beat to an uncapped HuMo "
                        "tier, or give the beat a shorter line."
                        % (int(cap), int(target_fc)))
                else:
                    limit_note = (
                        "This tier is uncapped, so it rendered its %d-frame "
                        "ceiling and the beat needs %d -- the beat is longer "
                        "than any single HuMo render. Split it into segments "
                        "through the coverage planner (JUMP, one fresh still "
                        "each) or give the beat a shorter line."
                        % (int(_HUMO_MAX_FRAMES), int(target_fc)))
                raise _wb.GraphExecutionError(
                    "%s: %s. %s NO FALLBACK -- fix the routing, not the "
                    "frames. Mirroring it would play the mouth backwards."
                    % (self.name, exc, limit_note)) from exc
        out_path = otr_engine_tmp_mp4("otr_humo_")
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path, self.target_fps)
        # Restore the proven HuMo VRAM discipline the refactor dropped: the frames
        # are on disk, so evict the umt5 CLIP + whisper encoders (BUG-291 detach
        # reclaim, the same the legacy free_otr_pipeline_residue used as
        # inter-phase cleanup) so the resident stack drops back down before the
        # next soak beat starts (no cross-beat accumulation). LOUD; no
        # unload_all_models (V-4/V-5).
        #
        # NOT BETWEEN THE SEGMENTS OF ONE BEAT (WIRE-W4a). The sentence above
        # is the whole justification -- "before the next soak beat starts" --
        # and between two segments of the SAME beat there is nothing to drop
        # back for, because the next segment needs the same stack. Reclaiming
        # here would detach(unpatch_all=True) the very handles ``prepare``
        # hoisted, so the "one load per beat" contract would survive as a
        # Python reference while the weights bounced to CPU and back every
        # segment. ``teardown`` runs it once, which keeps the cross-beat
        # promise exactly as it was.
        if not multi_clip:
            _wb.reclaim_idle_models(reason="humo post-decode")
        # M7 (added 2026-07-27): PROVE the silent-clip color/stream contract on
        # the emitted mp4 before canonicalize self-declares it -- ffprobe is the
        # source of truth, the hardcoded dict is not. Four siblings (wan_i2v,
        # wan_ti2v, ltx_8gb, ltx_video) have carried this line; this adapter and
        # eng_ltx_av did not, so their clips reached the ledger entirely
        # self-declared while CanonicalClip.frame_count is documented as "the
        # integer timing authority". The two derivations agree today only
        # because every producer pipes exact bytes -- a property of the current
        # producers, not a contract anything enforced.
        validate_silent_clip_contract(ffprobe_clip_fields(path), self.target_fps)
        # THE HONESTY RECEIPTS (2026-08-06). ONE stamp here serves all four
        # registered subclasses -- ``humo``, ``humo_1.7B``, ``humo_1.7B_169``
        # and ``humo_14B_169`` -- because every one of them inherits this render
        # path and this return. Four surfaces, one line, and no way for a tier to
        # be added later that silently answers neither question.
        #
        # ``"none"`` IS PROVEN HERE, not assumed. The only thing between the
        # decode and this return is ``fit_frames_to_target``, which TRIMS a long
        # render and RAISES ``MirrorExtensionForbidden`` on a short one (see the
        # no-mirror note above it). There is no branch on this lane that can add
        # a frame the model did not draw -- mirroring an audio-driven mouth would
        # play it backwards, which is why the operator's 2026-07-25 directive
        # removed it. So every delivered frame is a rendered frame, in order.
        #
        # EMITTED scope: ``n`` is what the encoded clip carries, so the gap
        # ``frame_count - native_frame_count`` is the manufactured count and is
        # zero here by construction.
        return {"out_path": path, "frame_count": n,
                "native_frame_count": n,
                "extension_mode": "none",
                # THE MANIFEST ROW, FILLED (S8b item 6, lane 2, 2026-08-11).
                # ``render_peak`` has been MEASURED and LOGGED a few lines up
                # since 2026-08-06 and then dropped on the floor, so every HuMo
                # clip reached the ledger with vram_peak_mb / recipe / quant /
                # render_canvas null -- and `render_driver` fell back to an
                # INSTANTANEOUS VRAM read, which is a sample at an arbitrary
                # moment wearing the name of a peak. S2's envelope work is built
                # on these numbers, so a plausible wrong one is worse here than
                # a missing one. The WAN lanes closed the same gap at
                # eng_wan_ti2v.py; this mirrors it, and ONE stamp serves all
                # four registered HuMo tiers because they share this return.
                "vram_peak_mb": int(render_peak) if render_peak else None,
                "recipe": self._recipe_receipt(),
                **self._clip_telemetry(width, height)}

    def _recipe_receipt(self):
        """The receipt string threaded onto the manifest row.

        Carries the TIER (the four HuMo engines are different weights and
        different ceilings) and whether the distill LoRA was in the graph,
        because those two facts are what a reader needs to know whether two
        clips are comparable. Versioned in the string, like the WAN lanes: bump
        the version when the graph changes, never edit a shipped one.

        RECEIPTS READ ``_lora_is_skipped``, NOT RAW TRUTHINESS (retro review
        r1, 2026-08-11). This line was ``if self._loader_names().get("lora")``,
        and the 1.7B tiers set that token to the STRING ``"none"`` -- which is
        truthy. So both 1.7B engines rendered LoRA-free (the graph reads
        ``_lora_is_skipped`` and skips the loader correctly) while stamping
        ``humo_1p7B_v1_lora``, and ``use_lora=True`` alongside it. The graph and
        the receipt disagreed about the same fact for six lanes, under a green
        preflight row, and it reached a PUBLISHED artifact:
        ``otr_credits_roll.py:238-239`` prints "lora" from ``use_lora``.
        """
        lora = self._loader_names().get("lora")
        return "%s_v1%s" % (
            self.name.replace(".", "p"),
            "" if (not lora or self._lora_is_skipped(lora)) else "_lora")

    def _clip_telemetry(self, width, height):
        """``quant`` / ``use_lora`` / ``render_canvas`` for the manifest row.

        ``render_canvas`` is exactly the field that would have answered the
        14B request-vs-render question immediately instead of costing a lane
        audit to find. ``use_lora`` stays a BOOL -- the rollup groups on it and
        credits truthiness-tests it, so a filename here would make the rollup
        incoherent across engines."""
        _lora = self._loader_names().get("lora")
        return {
            "quant": self._quant_label(),
            # SAME defect as _recipe_receipt above, same fix: the 1.7B tiers'
            # "none" token is truthy, so this published use_lora=True for a
            # render that never loaded one -- and credits print "lora" from it.
            "use_lora": bool(_lora) and not self._lora_is_skipped(_lora),
            "render_canvas": "%dx%d" % (int(width), int(height)),
        }

    def _quant_label(self):
        """The quant token of the UNET actually loaded, read off the RESOLVED
        basename rather than assumed, so a swapped weight cannot leave a
        receipt describing the file it replaced. ``None`` when the name carries
        no recognisable token."""
        name = str(self._loader_names().get("unet") or "").lower()
        for token in ("fp8_e5m2", "fp8_e4m3fn", "fp8", "fp16", "bf16",
                      "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s",
                      "q3_k_m"):
            if token in name:
                return token.upper() if token.startswith("q") else token
        return None

    def canonicalize(self, raw, request, profile):
        """Normalize a rendered clip into the ALWAYS-SILENT bt709 / yuv420p
        CanonicalClip contract (frame_count is the integer timing authority)."""
        return self._clip_from_raw(raw, request)

    # ---- pure helpers (CPU-testable; no wrapper, no heavy import) ----
    @staticmethod
    def _ref_path(ref):
        """Pull a filesystem path out of an audio_ref / init_image that may be a
        bare string OR a mapping carrying a ``path`` key (the schema AudioRef
        shape). Returns "" when nothing path-like is present."""
        if not ref:
            return ""
        if isinstance(ref, str):
            return ref
        if isinstance(ref, dict):
            return ref.get("path") or ""
        return getattr(ref, "path", "") or ""

    def _init_image_ref(self, request):
        """The portrait init image path from ``asset_refs{init_image}`` (or "")."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        assets = get("asset_refs") or {}
        if isinstance(assets, dict):
            return assets.get("init_image") or ""
        return ""

    def _aspect_plan(self, request):
        """The pad / crop / fit transform mapping the portrait init into the
        canvas with ONE uniform scale (never a stretch, pre-mortem N9). Returns
        ``None`` when the canvas or init dims are absent (the GPU smoke probes the
        real init dims), but still validates the policy token fail-closed."""
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

    def _build_render_request(self, request):
        """Pure: the normalized inference request the HuMo wrapper consumes, from
        a VideoRequest-shaped object OR a plain dict. Deterministic (seed + audio
        + init + aspect flow straight through) -- the render-twice determinism
        contract (V-7). The audio_ref is the FROZEN master that DRIVES the face;
        the output clip stays silent (V-1)."""
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
            "audio_path": self._ref_path(get("audio_ref")),
            "text_prompt": get("text_prompt") or "",
            "fps": int(self.target_fps),
            "target_frame_count": int(t_get("target_frame_count", 0) or 0),
            "seed": int(s_get("request_seed", 0) or 0),
            "aspect_plan": self._aspect_plan(request),
        }

    def _clip_from_raw(self, raw, request):
        """Pure: shape a worker / wrapper result into the silent CanonicalClip
        dict (bt709 / yuv420p; frame_count is the integer timing authority)."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        raw = raw or {}
        return {
            "clip_id": get("shot_id") or get("request_id") or "humo_clip",
            "type": "video", "path": raw.get("out_path", ""),
            "container": "mp4", "codec": "h264", "pixel_format": "yuv420p",
            "fps": int(self.target_fps),
            "frame_count": int(raw.get("frame_count", 0) or 0),
            "has_audio": False,            # V-1: only OTR_MasterAudioMux emits audio
            "color_primaries": "bt709", "transfer": "bt709", "matrix": "bt709",
            "engine_id": self.name, "family": self.family,
            # The second producer seam, shared by all four HuMo tiers. Stamping
            # the raw return without this line would leave the receipt on the
            # floor of ``canonicalize``.
            "native_frame_count": raw.get("native_frame_count"),
            "extension_mode": raw.get("extension_mode"),
            # S8b item 6 (lane 2): the identity/telemetry fields the driver and
            # the render-batch rollup have carried for months with nothing
            # producing them. Passed through rather than recomputed, so the
            # receipt describes THIS render and not a fresh read taken later.
            "vram_peak_mb": raw.get("vram_peak_mb"),
            "recipe": raw.get("recipe"),
            "quant": raw.get("quant"),
            "use_lora": raw.get("use_lora"),
            "render_canvas": raw.get("render_canvas"),
        }


#: 1.7B fallback-tier defaults (the lighter, no-LoRA talking face). Env-isolated
#: from the 14B knobs via the OTR_HUMO_17B_* namespace so a 14B OTR_HUMO_STEPS=6
#: / OTR_HUMO_CFG=1.0 never bleeds into this slower tier.
_HUMO_17B_UNET = "humo_1.7B_fp16.safetensors"


@register
class HuMo17BEngine(HuMoEngine):
    """The 1.7B HuMo downgrade tier. The OOM/VRAM hard auto-downgrade from the
    14B keystone lands HERE before the still floor, so a heavy episode that
    can't fit the 14B keeps a REAL audio-driven talking face. Config is isolated
    from the 14B env: its own UNET, NO LoRA (the lightx2v distill is 14B-shaped
    and shape-mismatches the 1.7B tier), and its own steps/cfg via OTR_HUMO_17B_*
    (defaults: 20 steps -- the distill shortcut is gone so it needs more steps --
    and cfg 1.0, dropped from 5.0 which produced a blue colour cast, fixed
    2026-06-17). Shares roles / required_inputs / requires_flag / the in-process
    graph topology with the 14B base; only the tier config differs.

    "Degrades on to the zero-VRAM still floor (humo -> humo_1.7B ->
    still_motion)" USED TO END THIS DOCSTRING, four lines above
    ``fallback_engine = None``. It has been false since NO FALLBACKS (operator
    2026-07-02) ripped ``UNIVERSAL_FLOOR`` / ``FLOOR_NAMES`` /
    ``make_fallback_of``: this tier degrades onto NOTHING and a failed beat
    fails the episode LOUD. Corrected in lane 15 (2026-08-11), which had to
    grep that chain's machinery to prove it was gone before ``still_motion``
    could stop painting a black beat on a missing still -- and found this
    sentence still asserting the opposite one file over. A docstring
    contradicting the attribute below it is the defect lesson L6 is about."""

    name = "humo_1.7B"
    engine_version = "1"
    fallback_engine = None               # NO FALLBACKS (2026-07-02): fail LOUD
    #: THE CANVAS, DECLARED (lane 3, 2026-08-11). Portrait 480x832 is where the
    #: diet leg was measured -- 12.84 GiB warm at 480x832x129 (HUMO_DIET,
    #: indexed in docs/evidence/video_evidence_manifest.json) -- and without a
    #: declaration the request fell through to the 1472x832 landscape default
    #: on a lane whose entire identity is the pillarbox talking head. Both axes
    #: /32-legal (15 x 26). A contradicting OTR_HUMO_WIDTH/HEIGHT is now a
    #: named refusal; see :meth:`HuMoEngine._native_dims`.
    render_canvas = (480, 832)
    #: BOOT CONTRACTS (S8, lane 3). This tier declares BOTH, deliberately: it
    #: is the LONG-BEAT lane and the auto-downgrade target, so requiring the
    #: diet would regress a shipping lane and remove the floor a heavy episode
    #: falls to. The diet is how a PROFILE casts it, not a demand it may make.
    compatible_boot_contracts = ("default", "humo_diet")
    #: ITS OWN LADDER AND ITS OWN CAP, because it is its own MODEL (2026-08-02).
    #: The 14B base now carries a 14B-sized cap, and this tier must not inherit
    #: a memory bound measured on a checkpoint roughly eight times its size --
    #: this is the downgrade target a heavy episode falls to precisely because
    #: the 14B did not fit. Uncapped to the shared ``_HUMO_MAX_FRAMES`` ceiling,
    #: which is where these two tiers have always run.
    safe_render_frames = None
    frame_contract = FrameContract(
        min_frames=_HUMO_MIN_FRAMES,
        max_frames=_HUMO_MAX_FRAMES,
        quantum=4,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_SOFT_REFERENCE,
    )
    #: S1 per-model still plan (see ``_HUMO_STILL_PLAN`` -- shared HuMo shape,
    #: portrait REQUIRED for the audio-driven-face lane).
    still_plan = _HUMO_STILL_PLAN

    def _ckpt_path(self):
        # Same resolver as every other tier (lane 2) -- this was a second copy
        # of the chain, so it was a second place the L1 defect had to be fixed.
        # The 1.7B lane's own packet is lane 3; it is NOT marked green by this,
        # it just stops carrying a bug lane 1 already paid for.
        return self._resolve_unet(
            "OTR_HUMO_17B_CKPT", "OTR_HUMO_17B_UNET_NAME", _HUMO_17B_UNET)

    def _loader_names(self):
        names = super()._loader_names()
        names["unet"] = os.path.basename(self._ckpt_path())
        # The 14B-shaped lightx2v distill LoRA is incompatible with 1.7B: run
        # LoRA-free (more steps below make up for the lost distill shortcut).
        names["lora"] = os.environ.get("OTR_HUMO_17B_LORA_NAME", "none")
        return names

    def _steps(self):
        return int(os.environ.get("OTR_HUMO_17B_STEPS", "20"))

    def _cfg(self):
        # cfg 1.0 (was 5.0): the LoRA-free 1.7B tier at cfg 5.0 produced a strong
        # BLUE colour cast -- red crushed, blue pushed (B-R measured +44.6 vs the
        # source still's +25 on 2026-06-17) -- while cfg 1.0 is colour-balanced
        # (B-R +2.8, red recovered) on the identical beat. HuMo's talking-face
        # motion is driven by the AUDIO conditioning, not cfg, so the lip motion
        # survives the drop. Tune via OTR_HUMO_17B_CFG.
        return float(os.environ.get("OTR_HUMO_17B_CFG", "1.0"))


@register
class HuMo17BLandscapeEngine(HuMo17BEngine):
    """1.7B HuMo in 16:9 LANDSCAPE (832x480) -- the modern character-beat look,
    selectable in the role dropdowns ALONGSIDE the portrait humo_1.7B (the
    operator wants both: portrait for old-time radio, wide for the new standard).
    Same checkpoint / roles / required inputs / in-process graph as humo_1.7B;
    only the render aspect and the cfg default differ. The 16:9 cfg sweet spot
    measured 2026-06-16 (the cfg sweep) is 2.5, so this tier defaults there while
    portrait humo_1.7B now uses cfg 1.0 (de-blued 2026-06-17, was 5.0). A 16:9
    character still (832x480) feeds it --
    meta_brief mints the still to match the selected engine's aspect via the video
    policy, so ONE dropdown pick aligns dims, cfg, still and composite canvas.
    Not in the auto-fallback chain (it is a deliberate operator pick): a failure
    degrades like humo_1.7B (-> still floor), never a silent aspect swap."""

    name = "humo_1.7B_169"
    render_aspect = "wide"
    #: THE CANVAS, DECLARED (lane 3, 2026-08-11). The landscape twin of the
    #: portrait 1.7B: same checkpoint, same VRAM class, 832x480 instead of
    #: 480x832. It has NO receipt of its own at any rung -- the manifest says
    #: so -- so this declaration fixes the request-vs-render channel and does
    #: NOT claim an envelope. Both axes /32-legal (26 x 15).
    render_canvas = (832, 480)
    #: Same reasoning as its portrait twin: BOTH contracts, because this tier
    #: has shipped under `default`.
    compatible_boot_contracts = ("default", "humo_diet")
    #: S1b per-model still plan (see ``_HUMO_169_STILL_PLAN`` -- the LANDSCAPE
    #: HuMo plan). ``render_aspect="wide"`` drives ``resolve_row_aspect`` to
    #: size portraits WIDE, and the plan's portrait row carries the WIDE
    #: layer-2 geometry to match. This sibling was ALREADY wide before the
    #: still-plans build; nothing about it flips.
    still_plan = _HUMO_169_STILL_PLAN

    def _cfg(self):
        # 16:9 sweet spot (cfg sweep 2026-06-16); override via OTR_HUMO_17B_169_CFG.
        return float(os.environ.get("OTR_HUMO_17B_169_CFG", "2.5"))


@register
class HuMo14BLandscapeEngine(HuMoEngine):
    """14B HuMo in 16:9 LANDSCAPE (832x480) -- the 2026-06-09 keystone quality
    (the fast 14B Kijai + lightx2v 6-step distill) in the modern WIDE character-beat
    aspect. The operator wants the 14B look in 16:9, NON-blue: the 1.7B downgrade
    tier carries a color cast (red-crushed / blue-pushed, measured 2026-06-17),
    but the 14B shares the same wan_2.1_vae and its latents MATCH it, so the 14B
    output is colour-correct. Same 14B checkpoint / LoRA / steps / cfg / roles /
    in-process graph as the humo base; ONLY the render aspect differs
    (portrait 480x832 -> wide 832x480). Selectable in all three role dropdowns
    alongside the portrait 14B and the 1.7B tiers. A 16:9 character still (832x480)
    feeds it -- meta_brief mints the still to the selected engine's aspect via the
    video policy, so one dropdown pick aligns dims, still and composite canvas.
    Degrades like the 14B base (-> humo_1.7B -> still); a deliberate
    operator pick, never a silent aspect swap."""

    name = "humo_14B_169"
    render_aspect = "wide"
    #: THE CANVAS, DECLARED (lane 2, 2026-08-11). This tier's request was being
    #: rewritten to the 1472x832 landscape default while the graph rendered
    #: 832x480 -- a 3.07x pixel disagreement, harmless ONLY because
    #: ``_aspect_plan``'s output is never read by ``_build_graph``. It becomes a
    #: real error the moment admission, still sizing or composite scaling
    #: trusts ``request.canvas``, and S2's admission work is exactly that.
    #:
    #: 832x480 is not a preference, it is where the hero cast was MEASURED:
    #: 13.06 GiB warm / 13.17 cold at 832x480x97 under the ``humo_diet`` boot
    #: (ENVELOPE_LADDERS job A, indexed in
    #: docs/evidence/video_evidence_manifest.json), 1.44 GiB under the gate,
    #: against 14.98 GiB unclamped. Declaring it makes the request match the
    #: render the number was taken from. Both axes /32-legal (26 x 15).
    #: See :meth:`HuMoEngine._native_dims` for why OTR_HUMO_WIDTH/HEIGHT now
    #: refuse rather than silently win on a declaring tier.
    render_canvas = (832, 480)
    #: BOOT CONTRACTS THIS TIER IS PROVEN ON (S8, lane 2). ``default`` stays in
    #: the list because this tier HAS shipped under it -- only a lane that never
    #: did may REQUIRE a contract, and dropping default here would retire a
    #: shipping lane by side effect. What ``default`` costs is stated rather
    #: than hidden: 14.98 GiB, which is OVER the 14.5 GiB gate. The hero cast is
    #: therefore expressed where casting belongs, in the PROFILE
    #: (`otr_w45_humo_14b_169.json` selects `humo_diet`), not by an engine
    #: refusing a boot it can technically run.
    compatible_boot_contracts = ("default", "humo_diet")
    #: S1b per-model still plan (see ``_HUMO_169_STILL_PLAN`` -- the LANDSCAPE
    #: HuMo plan). ``render_aspect="wide"`` drives portraits WIDE via
    #: ``resolve_row_aspect``, and the plan's portrait row carries the WIDE
    #: layer-2 geometry to match. Already wide before this build.
    still_plan = _HUMO_169_STILL_PLAN
    #: Route-A VRAM frame cap (only this 14B tier; base humo / humo_1.7B* stay
    #: uncapped). render_clip renders at the cap then exact-fits to the audio
    #: target. env override: OTR_HUMO_14B_SAFE_FRAMES (after a higher-frame probe).
    safe_render_frames = _HUMO_14B_SAFE_RENDER_FRAMES

    #: ITS OWN LADDER, because its ceiling is its own (chunk 7a QA panel,
    #: 2026-07-26). This tier inherited HuMoEngine's ``max_frames=177`` for
    #: exactly as long as it took a panel to read ``render_clip``, which does
    #: ``render_max = cap if cap is not None else _HUMO_MAX_FRAMES`` -- and
    #: ``cap`` is ``safe_render_frames`` = **97** HERE (it was 49 when this
    #: comment was written and became 97 on 2026-08-02; corrected lane 2,
    #: 2026-08-11, because a stale number in the comment that explains a
    #: number is worse than no comment) and None on all three
    #: siblings. So the inherited contract advertised 1.8x the frames this
    #: engine will actually render, and any beat over 97 frames raises
    #: ``MirrorExtensionForbidden`` today (mirroring an audio-synced lane would
    #: play the mouth backwards, which is correctly forbidden rather than
    #: worked around).
    #:
    #: This is the inheritance trap the roster audit was written to look for and
    #: did not catch: the audit checks each contract against ITSELF, never
    #: against the subclass's runtime behaviour. A subclass that changes what it
    #: renders must change what it declares, and 49 is the number.
    #:
    #: 97 = 4*24+1, so it is on the shared 4n+1 ladder and the quantum is
    #: unchanged. min stays 33: below that has hung this hardware. 97 frames at
    #: 25 fps is 3.88 s per render, which is the hero cast's real CONTENT
    #: constraint -- longer hero beats chain as JUMP segments (soft_reference),
    #: one fresh still each. That is a stills-budget fact, not a VRAM one.
    frame_contract = FrameContract(
        min_frames=_HUMO_MIN_FRAMES,
        max_frames=_HUMO_14B_SAFE_RENDER_FRAMES,
        quantum=4,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_SOFT_REFERENCE,
    )

    def _cfg(self):
        # Inherit the 14B distill cfg (1.0); a 16:9 tuning hook is available via
        # OTR_HUMO_14B_169_CFG without disturbing the portrait 14B's OTR_HUMO_CFG.
        return float(os.environ.get("OTR_HUMO_14B_169_CFG", str(super()._cfg())))


__all__ = [
    "HuMoEngine", "HuMo14BLandscapeEngine",
    "HuMo17BEngine", "HuMo17BLandscapeEngine",
]
