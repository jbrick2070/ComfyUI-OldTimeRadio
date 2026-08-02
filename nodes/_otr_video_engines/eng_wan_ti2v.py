"""Wan 2.2 TI2V-5B image->video motion adapter (8GB tier) -- in-process, default-OFF.

The 5B TI2V sibling of ``wan_i2v``. It animates a still into motion on the 5B
Wan 2.2 model, which the distribution targets at the 8GB tier (GGUF Q5_K_M ~3.6 GB
UNET; Apache-2.0, commercial-clean). It runs on CORE ComfyUI Wan nodes -- NOT the
KJ wrapper (same BUG-070 / numpy-pin reasons as wan_i2v).

The 5B graph DIFFERS from the 14B I2V graph: the 5B uses ``Wan22ImageToVideoLatent``
(vae/width/height/length/batch_size + optional start_image -> LATENT) rather than
``WanImageToVideo`` (which bundles positive/negative/latent), so the KSampler takes
the CLIPTextEncode positive/negative DIRECTLY and the latent from
``Wan22ImageToVideoLatent``. The 5B core node class + the loader presence were
captured from a LIVE ``/object_info`` on the 5080 (2026-06-14, VERIFY-AT-BUILD,
GO_FORWARD 4A) before this engine was coded.

The 5B REQUIRES the Wan2.2 VAE (``wan2.2_vae.safetensors``) -- its latent
compression differs from the 2.1 VAE, so feeding the 2.1 VAE corrupts the decode.
``assert_usable`` fails CLOSED (M8) if the resolved VAE basename is empty or is the
2.1 VAE. The umt5 CLIP is shared with wan_i2v.

Registered DEFAULT-OFF / dark (empty ``default_roles`` + gated behind
``OTR_ENABLE_WAN_TI2V``); fails CLOSED until the operator enables it AND the GGUF +
the Wan2.2 VAE are on disk. The pure dims/aspect/materialize/canonicalize helpers +
the M7 silent-clip contract proof are SHARED with wan_i2v via :mod:`wan_shared`;
the loaders, node candidates and the 5B graph below are engine-specific. Import-time
is cold-import clean (V-12). UTF-8, no BOM, ASCII-only.

Config (env). READ THE SPLIT -- since the WAN recipe freeze (2026-07-27) the
render knobs do NOT bind on a production leg:

* STILL LIVE, every leg -- ASSET LOCATION and the render-length CEILING, never
  the recipe: ``OTR_ENABLE_WAN_TI2V`` (vestigial opt-in flag);
  ``OTR_WAN_TI2V_CKPT`` the GGUF (or safetensors) UNET path the load probe
  checks; ``OTR_WAN_TI2V_UNET_NAME`` / ``OTR_WAN_TI2V_CLIP_NAME`` /
  ``OTR_WAN_TI2V_VAE_NAME`` loader basenames; ``OTR_WAN_TI2V_LOADER`` /
  ``OTR_WAN_TI2V_CLIP_LOADER`` gguf|safetensors (else INFERRED from the
  resolved basename -- they stay live WITH the names they follow, because
  freezing the loader class while its filename still moved would give one fact
  two owners); ``OTR_WAN_TI2V_CLIP_DIR`` / ``OTR_WAN_TI2V_VAE_DIR`` /
  ``OTR_WAN_TI2V_UNET_DIR`` dir overrides; and ``OTR_WAN_TI2V_MAX_FRAMES``,
  which is a CEILING rather than a recipe value and is the shipped 8GB tier's
  ``launch.env`` channel (``config/profiles/otr_8gb_wan.json``).
* FROZEN IN CODE as ``WAN_TI2V_RECIPE``, ignored-with-a-warning:
  ``OTR_WAN_TI2V_STEPS`` / ``_CFG`` / ``_SHIFT`` / ``_SAMPLER`` / ``_SCHEDULER``
  / ``_NEGATIVE`` / ``_TILED_VAE`` / ``_VAE_TILE`` / ``_VAE_OVERLAP`` /
  ``_VAE_TEMPORAL`` / ``_VAE_TEMPORAL_OVERLAP``. Their defaults live in that
  dict, not here, so this list cannot drift from the values.
* THE CONSENT ACT: set ``OTR_WAN_TI2V_PREQUALIFICATION=1`` and the frozen knobs
  bind again, range-checked and fail-closed, for a MEASUREMENT run -- whose
  clips stamp a ``+prequalification`` recipe receipt so a sweep artifact is
  never mistaken for a production one.

Why the freeze: a production episode is submitted to an ALREADY-BOOTED server,
so a knob exported at launch cannot bind the work (``PBUG-20260723-02``). Code
binds on every leg; an environment cannot.
"""
from __future__ import annotations

import logging
import os

from . import motion_common as _MC
from . import recipe_departures as _RD
from . import wan_recipe as _WR
from . import wan_shared as _WS
from .._otr_shared.role_compat import ROLES
from .._otr_shared.still_plan_helpers import StillPlanRow
from .frame_contract import CONTINUITY_STRICT_FIRST_FRAME, FrameContract
from .registry import EngineUnusable, EngineUsabilityReason, register
from .wan_shared import (
    _WAN_DEFAULT_NEGATIVE, ffprobe_clip_fields, validate_silent_clip_contract)

_LOG = logging.getLogger("OTR.video.wan_ti2v")

_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
_COMFY_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))

#: The 2.1 VAE basename the 5B must NOT use (M8): its latent compression differs.
_WAN21_VAE_BASENAME = "wan_2.1_vae.safetensors"
#: The APPROVED Wan2.2 VAE basenames (floor fail-closed whitelist, 2026-06-18
#: roundtable): any other present-but-wrong VAE corrupts the 5B decode, so the
#: guard accepts only these rather than merely rejecting empty / the 2.1 VAE.
_WAN22_VAE_ALLOWED = frozenset({"wan2.2_vae.safetensors"})
#: Default 8GB-tier GGUF (Q5_K_M; Apache-2.0). Resolved via folder_paths /
#: OTR_WAN_TI2V_CKPT at runtime; this basename is the fallback the loader reads.
_TI2V_DEFAULT_UNET = "Wan2.2-TI2V-5B-Q5_K_M.gguf"
#: Floor default umt5 CLIP = the GGUF encoder (2026-06-18 roundtable). The official
#: fp8 (umt5_xxl_fp8_e4m3fn_scaled.safetensors) throws Float8_e4m3fn on Mac MPS
#: (ComfyUI #9255); GGUF dequantizes to fp16 -> the cross-platform 8GB path. Override
#: with OTR_WAN_TI2V_CLIP_NAME (+ OTR_WAN_TI2V_CLIP_LOADER) for an fp16 safetensors.
_TI2V_DEFAULT_CLIP = "umt5-xxl-encoder-Q5_K_M.gguf"
# Frame budgeting (2026-06-18 CLIP-FILL roundtable, supersedes the static 8GB
# floor): the old code hard-CLAMPED every clip to 17 frames (0.68s @ 25fps) so a
# ~280-frame beat froze after 0.68s (the composite held the last frame). The fix
# PREDICTS the VRAM-affordable length per beat via motion_common.
# compute_real_frame_budget (a zero-cost mem_get_info read + a cost model -- never
# react-to-OOM) bounded by the beat's audio-derived target, then the render
# ping-pong-extends that short render up to the full target so the beat is FILLED
# with motion. _TI2V_MIN_FRAMES is the motion floor (always >= this many 4n+1
# frames of real motion); _TI2V_MAX_FRAMES is the absolute hard cap;
# OTR_WAN_TI2V_MAX_FRAMES (server boot env) or the tier profile's
# video.max_render_frames (ledger-stamped, the production channel) lets a
# tiny/8GB card pin a lower hard cap.
_TI2V_MIN_FRAMES = 17
_TI2V_DEFAULT_FRAMES = 17
_TI2V_MAX_FRAMES = 177
#: Reference render canvas used when _floor_length is called without explicit dims
#: (matches OTR_VIDEO_LANDSCAPE_CANVAS / the cost-model telemetry reference).
_TI2V_COST_REF_W = 1472
_TI2V_COST_REF_H = 832

#: The defined recipe-receipt string threaded into the manifest.
#:
#: THE VERSION LIVES IN THE STRING, not in a separate constant. This value rides
#: ``_clip_from_raw`` -> the manifest row -> the render-batch receipt ->
#: ``stamp_durable(meta.render_engines)``, so it is what a PUBLISHED EPISODE's
#: DURABLE LEDGER can be asked "which recipe rendered you". Before this chunk a
#: WAN clip stamped ``recipe: None`` -- there was not even a wrong receipt to
#: catch a drift with. A bare version constant that never reached this string
#: would leave the recipe invisible to every consumer that matters.
RECIPE_WAN_TI2V = "wan22_ti2v_5b_i2v_single_pass_v1"

#: THE CONSENT ACT that re-opens this adapter's recipe knobs. PER ADAPTER, not
#: shared with ``wan_i2v``: one switch for both tiers would stamp a
#: ``+prequalification`` receipt on a clip that had rendered with the frozen
#: recipe, and a receipt that lies in the safer direction still lies.
PREQUALIFICATION_ENV = "OTR_WAN_TI2V_PREQUALIFICATION"

#: THE FROZEN wan_ti2v RECIPE, v1 (LANE 1, 2026-07-27).
#:
#: WHAT v1 IS, STATED HONESTLY: these are TODAY'S SHIPPED DEFAULTS, lifted
#: verbatim from the env fallbacks they replace -- NOT a measured selection.
#: No WAN sweep has run. Freezing them is behaviour-preserving on any box that
#: set nothing, and it is reversible: a prequalification run measures and
#: produces v2 (bump the version INSIDE ``RECIPE_WAN_TI2V`` and add a new dict;
#: never edit a versioned dict in place, or receipts already on disk stop being
#: interpretable).
#:
#: ``max_frames`` IS DELIBERATELY NOT HERE. It is a render-length CEILING, and
#: unlike the ltx tier's it is a LIVE SHIPPED CHANNEL: ``otr_8gb_wan.json`` sets
#: both ``launch.env.OTR_WAN_TI2V_MAX_FRAMES`` and ``video.max_render_frames``.
#: Folding it into the recipe would silently retire the 8GB tier's launch
#: contract (``PBUG-20260723-02``, the bug this freeze descends from).
WAN_TI2V_RECIPE_V1 = {
    "steps": 30,
    "cfg": 5.0,
    #: ModelSamplingSD3 sigma shift; 5.0 is the 5B value (the 14B uses 8.0).
    "shift": 5.0,
    "sampler": "euler",
    "scheduler": "simple",
    #: The negative conditioning text. A RENDER INPUT, so it belongs here for
    #: the same reason the samplers do: read from ``os.environ`` it made two
    #: boxes produce visibly different clips that both stamped the same receipt.
    #: This adapter has no per-shot negative channel -- ``_build_graph`` reads
    #: only this -- so before the freeze the server's boot was its SOLE author.
    "negative": _WAN_DEFAULT_NEGATIVE,
    #: Tiled decode is ON for this tier: the video-VAE decode is a top VRAM-peak
    #: driver at 8GB. (The ltx tier measured the same direction on 2026-07-27 --
    #: tiled holds the peak FLAT across clip length where untiled climbs with it.
    #: That is ltx's measurement, not this adapter's, so it is context for a
    #: future WAN sweep and not a claim about these numbers.)
    "tiled_vae": True,
    #: The tiled-decode geometry, frozen with the flag that reaches it: the day
    #: a measured v2 changes tiling is the day four boot-environment values
    #: would otherwise start deciding what a published clip looks like.
    "vae_tile": 256,
    "vae_overlap": 64,
    "vae_temporal": 16,
    "vae_temporal_overlap": 8,
}

#: THE ONE NAME EVERY CONSUMER READS. Bumping a recipe is repointing this and
#: the version inside ``RECIPE_WAN_TI2V`` -- never editing a versioned dict.
WAN_TI2V_RECIPE = WAN_TI2V_RECIPE_V1

#: The env var each frozen field was read from, kept so the demotion can NAME
#: what it is ignoring. Presence is all this map is used for outside the consent
#: act -- see ``wan_recipe.ignored_override_keys``.
_RECIPE_ENV_KEYS = {
    "steps": "OTR_WAN_TI2V_STEPS",
    "cfg": "OTR_WAN_TI2V_CFG",
    "shift": "OTR_WAN_TI2V_SHIFT",
    "sampler": "OTR_WAN_TI2V_SAMPLER",
    "scheduler": "OTR_WAN_TI2V_SCHEDULER",
    "negative": "OTR_WAN_TI2V_NEGATIVE",
    "tiled_vae": "OTR_WAN_TI2V_TILED_VAE",
    "vae_tile": "OTR_WAN_TI2V_VAE_TILE",
    "vae_overlap": "OTR_WAN_TI2V_VAE_OVERLAP",
    "vae_temporal": "OTR_WAN_TI2V_VAE_TEMPORAL",
    "vae_temporal_overlap": "OTR_WAN_TI2V_VAE_TEMPORAL_OVERLAP",
}


#: S1 (2026-07-25) per-model still plan for wan_ti2v (spec section 3,
#: Shape A -- scene spine). FILE-LOCAL, fully declared (no cross-module
#: sharing per the spec's "no inheritance or shared defaults" rule).
#: scene_open + scene_beat + scene_character all WIDE + required=always;
#: portrait per subject at the engine's aspect (inherit_engine), not
#: required. Nothing reads the plan at S1 -- S2 wires it in.
_WAN_TI2V_STILL_PLAN = (
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
                 required="never",
                 framing_geometry=("in-character cinematic medium shot, head "
                                   "and shoulders, face clearly visible, "
                                   "subject centred with natural headroom "
                                   "above the head (never crop the top of the "
                                   "head)"),
                 style_tail_policy="full"),
)


@register
class WanTi2vEngine(_WS.WanInitImageMixin, _MC.MotionEngineBase):
    """The wan_ti2v 5B image->video adapter (8GB tier; in-process, sidecar_optional)."""

    name = "wan_ti2v"
    family = "image_to_video"
    #: Still-aspect identity (2026-06-17): Wan TI2V renders 16:9, so the director
    #: mints a WIDE init still (non-HuMo, non-mesh-portrait).
    render_aspect = "wide"
    #: S1 per-model still plan (see ``_WAN_TI2V_STILL_PLAN`` above).
    still_plan = _WAN_TI2V_STILL_PLAN
    # FLEXIBLE (operator 2026-06-18): eligible for EVERY role -- role_compat is the
    # real gate (it admits wan_ti2v only where the role supplies the required
    # init_image: announcer/music/character/retired_role_a do; the pure background_
    # abstract floor does not, so it's correctly excluded there). Opening `roles`
    # lets the operator pick wan_ti2v for the announcer slot (it animates that beat's
    # selected still); the required_inputs check still prevents a truly broken pick.
    roles = ROLES
    default_roles = ()
    required_inputs = ("init_image",)
    # ---- THE RECIPE SEAM (kibitz r2 MF2 / r4, 2026-08-01) ----------------- #
    #: These five name the module-level constants this adapter's recipe accessors
    #: used to read DIRECTLY. They are CLASS-LEVEL on purpose: a subclass (see
    #: eng_fastwan_8gb) shares this adapter's whole 5B substrate -- prepare, the
    #: beat hoist, teardown, the graph, tiled decode -- but owns a DIFFERENT
    #: recipe, and a module global cannot be overridden by declaring a class
    #: attribute. Before this seam, `_tiled_vae`, `_negative_prompt`,
    #: `_tile_geometry`, `_resolve_render_config` and `_recipe_departures` all
    #: read the module names, so a subclass declaring its own recipe would have
    #: SILENTLY rendered with wan_ti2v's -- and stamped its own receipt on it.
    #:
    #: BEHAVIOUR-PRESERVING for wan_ti2v: each points at the same object the
    #: methods read before, so the resolved config, the receipt, the graph and
    #: every refusal message are unchanged. That is pinned by test.
    recipe_id = RECIPE_WAN_TI2V
    recipe_data = WAN_TI2V_RECIPE
    recipe_env_keys = _RECIPE_ENV_KEYS
    prequalification_env = PREQUALIFICATION_ENV
    #: The render-length CEILING's env channel. Per-adapter for the same reason
    #: the consent act is: one key for two tiers would let the WAN tier's ceiling
    #: silently cap a different engine that never opted into it.
    max_frames_env = "OTR_WAN_TI2V_MAX_FRAMES"
    #: The graph node id whose slot 0 carries LATENT into the VAE decode. Named
    #: rather than hardcoded in ``_vaedecode_inputs`` so a subclass can swap the
    #: sampler node without duplicating the frozen tile geometry: fastwan_8gb
    #: replaces KSampler with a SamplerCustom chain, and a decode still wired to
    #: ``"ksampler"`` would point at a node that is not in the graph.
    _samples_node = "ksampler"
    #: The graph node id whose slot 0 carries MODEL into ModelSamplingSD3. The
    #: bare UNET here; fastwan_8gb points it at its LoRA loader so the patched
    #: model -- not the base -- is what gets sampled. Wiring the base by mistake
    #: renders a 3-step clip through an un-distilled model: no error, just a
    #: ruined render wearing a FastWan receipt.
    _model_source_node = "unet"
    #: THE FRAME LADDER (chunk 7a, 2026-07-26). ``_TI2V_MIN_FRAMES`` /
    #: ``_TI2V_MAX_FRAMES`` are this adapter's own named constants; the 4-frame
    #: quantum is Wan's 4n+1 latent stride (17 = 4*4+1, 177 = 4*44+1).
    #: CONTINUITY: the graph wires LoadImage -> Wan22ImageToVideoLatent
    #: ``start_image``, a hard first-frame lock.
    frame_contract = FrameContract(
        min_frames=_TI2V_MIN_FRAMES,
        max_frames=_TI2V_MAX_FRAMES,
        quantum=4,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_STRICT_FIRST_FRAME,
    )
    commercial_clean = True             # GGUF + VAE are Apache-2.0 (see MODEL_MANIFEST)
    requires_flag = None  # vestigial (registry IS the menu; no flag gate)
    engine_version = "1"
    declared_isolation = _MC.ISOLATION_SIDECAR_OPTIONAL
    target_fps = 25
    #: Floor sampler/scheduler whitelist (2026-06-18 roundtable): only core, rock-
    #: solid CROSS-PLATFORM choices -- uni_pc/sa_solver/MoEKSampler are NVIDIA-leaning
    #: / custom / NaN-prone on MPS. assert_usable fails closed on anything else; the
    #: default below is IN the whitelist (so an unset env passes -- no self-reject).
    _PORTABLE_SAMPLERS = frozenset({"euler"})
    _PORTABLE_SCHEDULERS = frozenset({"simple", "beta", "normal"})
    _DEFAULT_SAMPLER = "euler"
    _DEFAULT_SCHEDULER = "simple"

    # ---- config resolution (env override -> box default) ----
    def _ckpt_path(self):
        return os.environ.get("OTR_WAN_TI2V_CKPT") or os.path.join(
            _COMFY_ROOT, "models", "diffusion_models", _TI2V_DEFAULT_UNET)

    def _installed(self):
        """The 5B UNET is present if the explicit path exists OR folder_paths
        resolves its basename (the GGUF lives in C:\\ComfyUI-Models via
        extra_model_paths, not under the comfy root's models/)."""
        if os.path.exists(self._ckpt_path()):
            return True
        return self._resolve_model_file(
            ("diffusion_models", "unet"), self._loader_names()["unet"],
            "OTR_WAN_TI2V_UNET_DIR") is not None

    def resolve_isolation(self):
        """``in_process`` by default; escalates to ``sidecar_required`` when
        SageAttention is resident (BUG-070). Pure."""
        return _MC.resolve_isolation(
            self.declared_isolation, _MC.sageattention_patched())

    # ---- usability (fail-closed BEFORE any forward; no heavy import) ----
    def _aux_loader_files(self):
        """(label, folder_paths categories, basename, dir-override env) for the
        REQUIRED aux graph loaders beyond the UNET ckpt: the umt5 CLIP and the
        Wan2.2 VAE (M6)."""
        names = self._loader_names()
        return (
            ("CLIP/umt5", ("text_encoders", "clip"), names["clip"],
             "OTR_WAN_TI2V_CLIP_DIR"),
            ("VAE", ("vae",), names["vae"], "OTR_WAN_TI2V_VAE_DIR"),
        )

    def assert_usable(self, host_caps, profile, request_template=None):
        """Fail closed before any forward: the UNET ckpt, the M8 Wan2.2-VAE
        guard, then ALL aux graph loaders (umt5 CLIP + the 2.2 VAE) present on
        disk."""
        # Fail CLOSED on a bad/non-portable render knob (sampler whitelist + env
        # range-checks) BEFORE any forward -- never a raw crash mid-render.
        self._resolve_render_config()
        if not self._installed():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "%s UNET not found at %s; fetch the Wan2.2 TI2V-5B GGUF and "
                "set OTR_WAN_TI2V_CKPT (or drop it in models/diffusion_models)"
                % (self.name, self._ckpt_path()), kind="video")
        # M8: the 5B REQUIRES the Wan2.2 VAE -- an empty / 2.1 / any other
        # wrong-but-present VAE silently corrupts the decode. Fail CLOSED unless the
        # resolved basename is in the approved Wan2.2 whitelist (2026-06-18 roundtable
        # tightened this from "not empty / not 2.1" to an explicit allow-list).
        vae_base = os.path.basename(self._loader_names()["vae"] or "").lower()
        if vae_base not in _WAN22_VAE_ALLOWED:
            _why = ("the 2.1 VAE (latent compression differs)"
                    if vae_base == _WAN21_VAE_BASENAME
                    else "empty" if not vae_base else "not an approved Wan2.2 VAE")
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "%s requires the Wan2.2 VAE; resolved basename %r is %s -- the "
                "5B decode needs one of %s (M8). Set OTR_WAN_TI2V_VAE_NAME"
                "=wan2.2_vae.safetensors"
                % (self.name, vae_base, _why, sorted(_WAN22_VAE_ALLOWED)), kind="video")
        missing = self._missing_loaders()
        if missing:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "%s required loader file(s) absent: %s -- the 5B graph needs "
                "the umt5 CLIP and the Wan2.2 VAE on disk too, not just the UNET "
                "(offline invariant, no runtime fetch); fix the *_NAME envs or "
                "point OTR_WAN_TI2V_CLIP_DIR / OTR_WAN_TI2V_VAE_DIR at the "
                "installed basenames"
                % (self.name, ", ".join("%s=%r" % (lbl, nm) for lbl, nm in missing)),
                kind="video")
        return self.name

    #: Terminal node of the graph (its IMAGE output is encoded to the clip).
    _TERMINAL = "vaedecode"

    # ---- beat session (WIRE-W3b, 2026-07-29) ----
    #: The node ids :meth:`prepare` hoists out of the segment graph. ONE entry,
    #: for the same arithmetic reason as ``eng_wan_i2v``: ``render_clip`` runs
    #: with ``free_after_use=True`` so the umt5 text encode frees before the 5B
    #: UNET and the 2.2 video VAE decode. Hoisting the CLIP would pin it
    #: resident for the whole beat and undo the one mitigation this tier has.
    _SESSION_NODES = ("unet",)

    def session_identity(self):
        """What this adapter's HANDLES are -- engine, recipe, loader modes,
        weights. ``BeatSession`` refuses a multi-segment beat without it.

        BOTH loader modes are carried, and unlike ``eng_wan_i2v`` this adapter
        really has two: the UNET mode picks ``UNETLoader`` vs
        ``UnetLoaderGGUF``, and the CLIP mode picks ``CLIPLoader`` vs
        ``CLIPLoaderGGUF`` -- whose input schemas DIFFER (the GGUF one takes no
        ``device``). A beat that switched either class mid-way would be reusing
        a handle its own graph no longer knows how to have produced.

        The weight receipts include the CLIP and the VAE even though neither is
        hoisted (r4/A5): this adapter distinguishes INCOMPATIBLE Wan2.2 VAE
        generations, so a VAE swapped between segments is exactly the drift
        this exists to catch.

        Per-segment state -- prompt, seed, frame count, canvas, init image --
        is deliberately EXCLUDED. Not cached: the whole job is to notice a
        weight that MOVED, so the receipts are re-stat'ed on every ask."""
        return (self.name,
                self._recipe_receipt(),
                "unet_loader=%s" % (self._loader_mode(),),
                "clip_loader=%s" % (self._clip_loader_mode(),),
                repr(self._wan_session_receipts()))

    def prepare(self, host_caps, profile, session_ctx):
        """Load the 5B UNET ONCE PER BEAT instead of once per segment.

        Mirrors ``eng_wan_i2v.prepare`` step for step (resolve the frozen
        config BEFORE the lease is taken, ``super().prepare()``, a loader-only
        mini-graph registering the patcher through ``on_result`` as it lands,
        tear down and re-raise on ANY failure). Two things are this adapter's
        own:

        1. ``session_ctx`` IS KEPT. It is the only channel that tells
           :meth:`render_clip` whether its output will be concatenated with
           other segments into one beat, and that decides whether the
           ping-pong extension is legal (see :meth:`render_clip`). The base
           ``prepare`` drops the argument, and nothing else on the request says
           it -- a coverage-planned segment's request is shaped exactly like a
           single-clip beat's.

        2. THE COST OF THE HOIST IS MEASURED, not assumed. ``_floor_length``
           predicts an affordable render length from live free VRAM minus a
           cost-model ``overhead`` documented as "the resident model + fixed
           buffers". Hoisting the UNET moves ~6 GB from *free* into *resident*
           BEFORE that read happens, so the overhead would be charged twice and
           the predictor would refuse renders it used to allow -- loudly, via
           ``MotionBudgetError``, on a lane that works today. Measuring the
           delta across the loader graph and handing it back to the budget
           restores exactly the memory state the pre-hoist code measured, with
           a number read off this box rather than a constant somebody tuned.
           ``None`` (no torch/CUDA) stays ``None`` and the budget stays
           geometry-only, as before.
        """
        from . import wrapper_bridge as _wb
        self._resolve_render_config()           # range-checked, fail-closed
        # session_ctx comes back IN `prepared` from the base now (2026-08-01).
        # This adapter used to re-add it locally; so did eng_humo, and the four
        # adapters that did NOT is how ltx_video's boomerang regression shipped.
        prepared = super().prepare(host_caps, profile, session_ctx)
        try:
            classes = getattr(self, "_classes", None) \
                or _wb.resolve_graph_classes(self._node_candidates())
            names = self._loader_names()
            bucket = prepared.setdefault("patchers", self._patchers)
            seen = {id(p) for p in bucket}

            def _register(nid, out):
                """Own the handle the moment it exists, not after the graph
                returns: if a later node raised, ``run_graph`` never returns a
                results dict and an unregistered patcher is VRAM nothing will
                ever detach. Slot 0 is the MODEL patcher ``_detach_patchers``
                knows how to unwind; the duck-typed ``detach`` check keeps that
                honest rather than a positional assumption."""
                model = out[0] if out else None
                if (model is not None and id(model) not in seen
                        and callable(getattr(model, "detach", None))):
                    bucket.append(model)
                    seen.add(id(model))

            free_before = _MC.free_vram_mb()
            results = _wb.run_graph(
                self._hoist_graph(names), classes, on_result=_register)
            free_after = _MC.free_vram_mb()
            out = results.get("unet") or ()
            if not out:
                raise _wb.GraphExecutionError(
                    "%s hoisted UNET %r returned no output; the beat "
                    "session has nothing to reuse and every segment would "
                    "silently reload" % (self.name, names["unet"]))
            prepared["external_results"] = results
            prepared["hoisted_vram_mb"] = self._hoist_cost_mb(
                free_before, free_after)
        except BaseException:
            try:
                self.teardown(prepared)
            except BaseException:          # noqa: BLE001 -- never mask the cause
                pass
            raise
        return prepared

    def _unet_inputs(self, names):
        """The UNET loader's inputs. ONE implementation: ``prepare`` hoists with
        it and ``_build_graph`` would otherwise build the same dict a second time
        and drift from it. GGUF's loader takes no ``weight_dtype``."""
        if self._loader_mode() == "gguf":
            return {"unet_name": names["unet"]}
        return {"unet_name": names["unet"], "weight_dtype": "default"}

    def _hoist_graph(self, names):
        """The loader-only mini-graph :meth:`prepare` runs once per beat.

        A SEAM, not a constant: every node returned here is hoisted, registered
        through ``on_result`` and torn down by ``_detach_patchers``, so an engine
        whose model needs more than the bare UNET -- fastwan_8gb routes it through
        a LoRA loader -- extends the graph HERE rather than reimplementing the
        beat-session lifecycle. Whatever this returns must also be popped from
        ``_build_graph``'s graph (``external_results`` does that automatically)."""
        return {"unet": {"class": "unet", "inputs": self._unet_inputs(names)}}

    @staticmethod
    def _hoist_cost_mb(free_before, free_after):
        """MB the hoisted handles took out of free VRAM, or ``None``.

        ``None`` whenever either read is unavailable (no torch/CUDA) so the
        caller keeps its geometry-only path. NEGATIVE deltas read as 0.0: a
        concurrent process freeing memory during the load would otherwise hand
        the budget MORE room than the box has, which is the one direction this
        correction must never move."""
        if free_before is None or free_after is None:
            return None
        return max(0.0, float(free_before) - float(free_after))

    def teardown(self, prepared):
        """Drop the beat-scoped handles BEFORE delegating.

        ``prepared["external_results"]`` holds a strong reference to the
        hoisted UNET. The base teardown detaches the patchers, unloads,
        releases the lease and then WAITS for machine-wide VRAM to settle --
        every one of those steps would run with this dict still pinning the
        very tensors it is trying to reclaim, so the stability wait would be
        watching its own referent. Clearing first costs nothing.

        Never raises out of teardown (it is a finally-path)."""
        if isinstance(prepared, dict):
            prepared.pop("external_results", None)
        return super().teardown(prepared)

    # ---- in-process graph spec (CORE Wan 2.2 TI2V-5B; verified vs /object_info) -
    def _loader_mode(self):
        """``gguf`` (``UnetLoaderGGUF``) or ``safetensors`` (``UNETLoader``).
        Explicit via ``OTR_WAN_TI2V_LOADER``; else inferred from the unet
        extension (the 8GB-tier default is the GGUF)."""
        mode = (os.environ.get("OTR_WAN_TI2V_LOADER") or "").strip().lower()
        if mode in ("gguf", "safetensors"):
            return mode
        return ("gguf" if self._loader_names()["unet"].lower().endswith(".gguf")
                else "safetensors")

    def _clip_loader_mode(self):
        """``gguf`` (``CLIPLoaderGGUF``) or ``safetensors`` (``CLIPLoader``) for the
        umt5 encoder. Explicit via ``OTR_WAN_TI2V_CLIP_LOADER``; else inferred from
        the clip extension (the floor default is the GGUF umt5 -- fp8 safetensors
        throws Float8_e4m3fn on Mac MPS, ComfyUI #9255, 2026-06-18 roundtable)."""
        mode = (os.environ.get("OTR_WAN_TI2V_CLIP_LOADER") or "").strip().lower()
        if mode in ("gguf", "safetensors"):
            return mode
        return ("gguf" if self._loader_names()["clip"].lower().endswith(".gguf")
                else "safetensors")

    def _tiled_vae(self):
        """Whether to decode through ``VAEDecodeTiled`` (FROZEN ON: the video-VAE
        decode is a top VRAM-peak driver at 8GB).

        Under the consent act ``OTR_WAN_TI2V_TILED_VAE`` binds again and an
        unrecognised value fails CLOSED rather than collapsing to False -- a
        sweep must not be able to mistype the knob it is varying, decode the
        other way, and stamp a receipt saying it had measured this one."""
        if not _WR.prequalification_active(self.prequalification_env):
            return bool(self.recipe_data["tiled_vae"])
        # The prequalification DEFAULT is the FROZEN value, not a literal: a
        # sweep that opens the knobs but re-exports only some of them must
        # measure the recipe it is validating, not a third configuration.
        return _WR.config_flag(self, self.recipe_env_keys["tiled_vae"],
                               self.recipe_data["tiled_vae"])

    def _negative_prompt(self):
        """The FROZEN negative conditioning.

        This adapter has no per-shot negative channel -- ``_build_graph`` reads
        this and nothing else -- so before the freeze the server's boot
        environment was the SOLE author of the negative conditioning, and two
        boxes rendered visibly different clips from the same episode while both
        stamped the same (empty) receipt."""
        frozen = str(self.recipe_data["negative"])
        if not _WR.prequalification_active(self.prequalification_env):
            return frozen
        return _WR.config_text(self, self.recipe_env_keys["negative"], frozen,
                               strip=False)

    def _node_candidates(self):
        """Ordered ComfyUI node-class candidates per graph node (CORE Wan 2.2
        TI2V-5B + ComfyUI-GGUF, schema-verified vs the live /object_info 2026-06-18).
        The 5B latent node is ``Wan22ImageToVideoLatent`` (NOT ``WanImageToVideo``).
        The CLIP loader + the VAE-decode node switch per the floor knobs above."""
        unet_cls = (("UnetLoaderGGUF",) if self._loader_mode() == "gguf"
                    else ("UNETLoader",))
        clip_cls = (("CLIPLoaderGGUF",) if self._clip_loader_mode() == "gguf"
                    else ("CLIPLoader",))
        vaedecode_cls = (("VAEDecodeTiled",) if self._tiled_vae()
                         else ("VAEDecode",))
        return {
            "unet": unet_cls,
            "modelsampling": ("ModelSamplingSD3",),
            "clip": clip_cls,
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "vae": ("VAELoader",),
            "loadimage": ("LoadImage",),
            "latent": ("Wan22ImageToVideoLatent",),
            "ksampler": ("KSampler",),
            "vaedecode": vaedecode_cls,
        }

    def _loader_names(self):
        """Model FILENAMES the loader nodes consume (env-overridable). The VAE
        defaults to the Wan2.2 VAE (the 5B requirement, M8)."""
        return {
            "unet": os.environ.get("OTR_WAN_TI2V_UNET_NAME")
            or os.path.basename(self._ckpt_path()),
            "clip": os.environ.get(
                "OTR_WAN_TI2V_CLIP_NAME", _TI2V_DEFAULT_CLIP),
            "vae": os.environ.get("OTR_WAN_TI2V_VAE_NAME", "wan2.2_vae.safetensors"),
        }

    #: Why the portable-sampler whitelist exists, appended to its refusal so the
    #: message still says WHY when the check itself lives in ``wan_recipe``.
    _SAMPLER_HINT = (" -- wan_ti2v is the 8GB/Mac/AMD floor; "
                     "uni_pc/sa_solver/MoEKSampler are not portable. Use a "
                     "heavier engine for those.")

    def _resolve_render_config(self):
        """The FROZEN render knobs (shared by assert_usable and _build_graph).

        PRODUCTION: the recipe binds whatever the server booted with, and
        anything the environment is trying to say is NAMED and ignored -- named
        because a knob that silently does nothing is a complaint this build has
        already collected, and never PARSED because a stale malformed value must
        not be able to kill a leg it cannot influence.

        PREQUALIFICATION: the knobs are open, every value is range-checked and
        fails CLOSED with a named MALFORMED_CONFIG, and each honoured override
        is announced so a sweep's log says what it actually measured."""
        if not _WR.prequalification_active(self.prequalification_env):
            _WR.warn_ignored(_LOG, self.name, self.recipe_id,
                             _WR.ignored_override_keys(self.recipe_env_keys),
                             self.prequalification_env)
            # SPELLED OUT rather than a dict slice so BOTH legs return the SAME
            # KEY SET: the recipe also carries the tiled-decode fields, whose
            # owners are _tiled_vae() / _vaedecode_inputs(), and a return shape
            # that varied by mode would hand the next reader a KeyError that
            # only reproduces under the consent act.
            return {
                "steps": self.recipe_data["steps"],
                "cfg": self.recipe_data["cfg"],
                "shift": self.recipe_data["shift"],
                "sampler": self.recipe_data["sampler"],
                "scheduler": self.recipe_data["scheduler"],
            }
        _WR.warn_honoured(_LOG, self.name, self.recipe_id,
                          _WR.ignored_override_keys(self.recipe_env_keys),
                          self.prequalification_env)
        _num = _WR.config_number
        return {
            "steps": _num(self, self.recipe_env_keys["steps"],
                          self.recipe_data["steps"], 1, 100, int),
            "cfg": _num(self, self.recipe_env_keys["cfg"],
                        self.recipe_data["cfg"], 0.0, 30.0, float),
            "shift": _num(self, self.recipe_env_keys["shift"],
                          self.recipe_data["shift"], 0.1, 20.0, float),
            "sampler": _WR.config_text(
                self, self.recipe_env_keys["sampler"], self.recipe_data["sampler"],
                allowed=self._PORTABLE_SAMPLERS, hint=self._SAMPLER_HINT),
            "scheduler": _WR.config_text(
                self, self.recipe_env_keys["scheduler"],
                self.recipe_data["scheduler"],
                allowed=self._PORTABLE_SCHEDULERS),
        }

    def _tile_geometry(self, key):
        """One tiled-decode geometry value: frozen, or range-checked under the
        consent act.

        THE one implementation -- ``_vaedecode_inputs`` builds the graph from it
        and ``_recipe_departures`` reports it. It used to swallow a bad value
        and substitute the default, which made these four the only knobs on this
        adapter that failed OPEN: a sweep could mistype the value it was
        measuring, render at something else, and stamp a receipt saying it had
        measured it."""
        dflt = int(self.recipe_data[key])
        if not _WR.prequalification_active(self.prequalification_env):
            return dflt
        lo, hi = _WR.VAE_TILE_BOUNDS[key]
        return _WR.config_number(self, self.recipe_env_keys[key], dflt, lo, hi, int)

    def _recipe_departures(self, knobs=None):
        """Which frozen knobs THIS CELL actually changed. ``{}`` on production.

        RESOLVED values, never env presence: re-exporting a knob at the value it
        already has changes nothing, and a receipt claiming a departure there
        would describe a cell that does not exist.

        NOTHING IS RESOLVED ON A PRODUCTION LEG. The early return is structural,
        not an optimisation -- it keeps the "never parse what cannot bind"
        guarantee from depending on every accessor staying correct forever, and
        keeps a production leg from emitting the demotion notice twice.

        The tile geometry is reported ONLY when tiled decode ran: a knob the
        render never reached is not a departure that describes the clip."""
        if not _WR.prequalification_active(self.prequalification_env):
            return {}
        knobs = self._resolve_render_config() if knobs is None else knobs
        resolved = {
            "steps": knobs["steps"], "cfg": knobs["cfg"],
            "shift": knobs["shift"], "sampler": knobs["sampler"],
            "scheduler": knobs["scheduler"],
            "negative": self._negative_prompt(),
            "tiled_vae": self._tiled_vae(),
        }
        if resolved["tiled_vae"]:
            for key in _WR.VAE_TILE_BOUNDS:
                resolved[key] = self._tile_geometry(key)
        return _RD.departures(self.recipe_data, resolved)

    def _recipe_receipt(self, knobs=None):
        """THE stamp. The single stamp site goes through here and nowhere else
        -- a second site reaching for the bare constant would put an unmarked,
        or an unnamed, sweep clip into the durable ledger."""
        return _WR.recipe_receipt(self.recipe_id, self.prequalification_env,
                                  self._recipe_departures(knobs))

    def _floor_length(self, target_frame_count, width=None, height=None,
                      hoisted_vram_mb=None):
        """The VRAM-PREDICTED render length (4n+1) for this beat (clip-fill fix).

        REPLACES the old hard 17-frame "8GB floor" that froze every wan clip to
        0.68s: ``motion_common.compute_real_frame_budget`` reads live free VRAM
        (zero-cost ``mem_get_info``) + a cost model to PREDICT how many of the
        beat's audio-derived ``target_frame_count`` frames fit under the ceiling --
        never react-to-OOM. The render then ping-pong-extends this (possibly short)
        render up to the full target so the beat is FILLED with motion. The motion
        floor (17) always wins; an absolute hard cap for a tiny/8GB card comes
        from this adapter's own ``max_frames_env`` key (``OTR_WAN_TI2V_MAX_FRAMES``
        here; a subclass declares its own) or, on a production leg, the tier's
        ledger-stamped ``max_render_frames`` (default = the engine max, not 17).

        ``hoisted_vram_mb`` (WIRE-W3b) is what :meth:`prepare` MEASURED the
        beat-scoped handles cost, added back to the live free reading. The cost
        model's ``overhead`` term is documented as "the resident model + fixed
        buffers"; once the UNET is hoisted, free VRAM has already been charged
        for those weights, so without this the same GB is subtracted twice and
        a render that fits refuses with ``MotionBudgetError``. ``None`` (no
        session, or no torch/CUDA) leaves the reading exactly as it was."""
        from . import wrapper_bridge as _wb
        target = int(target_frame_count or _TI2V_DEFAULT_FRAMES)
        # Hard-cap resolution (2026-07-24 WAN 8GB launch contract). BEFORE this,
        # the ONLY channel was the env pin -- which a production episode leg can
        # never see, because it is submitted to an already-booted server and
        # `launch.env` applies to the server process. An 8GB leg therefore
        # inherited the 177-frame engine max, asked for a whole 7s beat, and died
        # in the cost model (2026-07-23 wan_8gb__lumina_image__media_archive).
        # Order: explicit env pin (operator override) -> the tier's
        # profile-carried ceiling (ledger-stamped, the normal path) -> engine max.
        _raw_env = (os.environ.get(self.max_frames_env) or "").strip()
        try:
            hard_cap = int(_raw_env) if _raw_env else 0
        except (TypeError, ValueError):
            hard_cap = 0
        if hard_cap <= 0:
            hard_cap = self.profile_max_render_frames()
        if hard_cap <= 0:
            hard_cap = _TI2V_MAX_FRAMES
        hard_cap = max(_TI2V_MIN_FRAMES, min(hard_cap, _TI2V_MAX_FRAMES))
        target = max(1, min(target, hard_cap))
        if width is None or height is None:
            width, height = _TI2V_COST_REF_W, _TI2V_COST_REF_H
        free_mb = _MC.free_vram_mb()
        if free_mb is not None and hoisted_vram_mb is not None:
            free_mb = float(free_mb) + float(hoisted_vram_mb)
        budget = _MC.compute_real_frame_budget(
            free_mb, target, int(width), int(height), self.name)
        return _wb.quantize_frames_4n1(
            budget, min_frames=_TI2V_MIN_FRAMES, max_frames=hard_cap)

    def _planned_length(self, target_frame_count, width=None, height=None,
                        hoisted_vram_mb=None):
        """The render length for a COVERAGE-PLANNED segment: its own, whole.

        A planned segment's length came off ``coverage_plan.partition_beat``,
        which only ever emits lengths this adapter's own ``frame_contract``
        declares legal -- so there is nothing to snap and nothing to floor. The
        VRAM predictor is deliberately NOT consulted: it exists to decide how
        much of an open-ended beat to render, and this length is not a
        preference, it is the arithmetic the rest of the beat was planned
        around. A segment that renders anything else leaves the beat short or
        long by exactly that much.

        THE TWO WAYS THIS CAN REFUSE, both knowable from pure numbers before
        anything is staged:

        1. The length is not on this adapter's ladder. That means the stamped
           plan and the declared contract disagree, which is a build error, not
           a render-time condition.
        2. The tier pinned a render ceiling BELOW the planned length. WAN is
           deliberately excluded from ``frame_contract.PLANNING_CAP_ENGINES``
           -- its ceiling is a RENDER cap that the single-clip path fills
           around with the ping-pong, and narrowing the planner by it would
           turn every WAN beat into a pile of 17-frame renders and rewrite the
           low-VRAM tier ``PBUG-20260723-02`` fixed. The consequence is that a
           tier ceiling and a multi-clip plan CAN contradict each other, and
           when they do the honest answer is to say so by name rather than
           render short and mirror the difference.
        """
        from . import wrapper_bridge as _wb
        target = int(target_frame_count or 0)
        contract = self.frame_contract
        if not contract.is_legal_length(target):
            raise _wb.GraphExecutionError(
                "%s was handed a coverage-planned segment of %d frame(s), "
                "which is not on its declared ladder (min %d, max %d, quantum "
                "%d). The stamped plan and the adapter's frame_contract "
                "disagree. NO FALLBACK -- snapping to a nearby rung would make "
                "the assembled beat a different length than the plan says."
                % (self.name, target, int(contract.min_frames), int(contract.max_frames),
                   int(contract.quantum)))
        ceiling = self.profile_max_render_frames()
        _raw_env = (os.environ.get(self.max_frames_env) or "").strip()
        try:
            env_cap = int(_raw_env) if _raw_env else 0
        except (TypeError, ValueError):
            env_cap = 0
        if env_cap > 0:
            ceiling = env_cap
        if 0 < ceiling < target:
            raise _wb.GraphExecutionError(
                "%s was handed a coverage-planned segment of %d frame(s) "
                "but this tier pins its render ceiling at %d. WAN's ceiling is "
                "a RENDER cap, not a planning cap (frame_contract."
                "PLANNING_CAP_ENGINES), so the planner never saw it and the "
                "two now contradict each other. NO FALLBACK -- rendering %d "
                "and mirroring the rest would put padded frames inside a beat "
                "claiming real multi-clip coverage. Raise max_render_frames "
                "for this tier or route the beat to a single-clip engine."
                % (self.name, target, ceiling, ceiling))
        # AND IT MUST FIT. The planned path used to skip the VRAM predictor
        # entirely -- survivable while planned segments were rare, not now that
        # this engine is coverage-planned and the mirror that hid an
        # unaffordable render is gone. Same accounting as the single-clip
        # branch: live free + hoisted, so resident weights are not charged
        # twice. Skipped only when the caller has no canvas to price.
        if width and height:
            free_mb = _MC.free_vram_mb()
            if free_mb is not None and hoisted_vram_mb is not None:
                free_mb = float(free_mb) + float(hoisted_vram_mb)
            _MC.assert_frame_affordable(free_mb, target, int(width),
                                        int(height), self.name)
        return target

    def _build_graph(self, request, image_name, plan, length, width, height,
                     external_results=None):
        """The declarative Wan TI2V-5B graph (wrapper_bridge.run_graph format).
        ``Wan22ImageToVideoLatent`` builds the latent from the VAE + init image; the
        KSampler takes the CLIPTextEncode positive/negative DIRECTLY (unlike the
        14B's WanImageToVideo, which bundles them). ModelSamplingSD3 applies the 5B
        sigma shift (default 5.0).

        ``external_results`` names the ids the CALLER already produced and owns
        for the whole beat (WIRE-W3b: :meth:`prepare` loads the UNET once).
        Those node DEFINITIONS are omitted while every wire that reads them
        stays -- the executor resolves them from the caller's handles instead.
        Omitting rather than leaving them is not an optimisation: ``run_graph``
        REFUSES a graph that also defines an id the caller supplied, because
        then two parties claim to produce one output.

        Conditional on purpose. A caller that prepared nothing -- every sibling
        test that hand-builds ``prepared={"patchers": []}`` -- gets the loader
        nodes exactly as before, byte for byte."""
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        names = self._loader_names()
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        cfg_knobs = self._resolve_render_config()        # range-checked, fail-closed
        steps = cfg_knobs["steps"]
        cfg = cfg_knobs["cfg"]
        shift = cfg_knobs["shift"]
        sampler = cfg_knobs["sampler"]
        scheduler = cfg_knobs["scheduler"]
        positive = get("text_prompt") or "subtle natural motion"
        negative = self._negative_prompt()               # frozen, fail-closed
        unet_inputs = self._unet_inputs(names)
        # CLIPLoaderGGUF takes clip_name + type ONLY (no `device` arg, verified vs
        # /object_info 2026-06-18); the core CLIPLoader also takes device.
        if self._clip_loader_mode() == "gguf":
            clip_inputs = {"clip_name": names["clip"], "type": "wan"}
        else:
            clip_inputs = {"clip_name": names["clip"], "type": "wan",
                           "device": "default"}
        graph = {
            "unet": {"class": "unet", "inputs": unet_inputs},
            "modelsampling": {"class": "modelsampling",
                              "inputs": {"model": W(self._model_source_node, 0),
                                         "shift": shift}},
            "clip": {"class": "clip", "inputs": clip_inputs},
            "pos": {"class": "pos",
                    "inputs": {"text": positive, "clip": W("clip", 0)}},
            "neg": {"class": "neg",
                    "inputs": {"text": negative, "clip": W("clip", 0)}},
            "vae": {"class": "vae", "inputs": {"vae_name": names["vae"]}},
            "loadimage": {"class": "loadimage", "inputs": {"image": image_name}},
            "latent": {"class": "latent",
                       "inputs": {"width": int(width), "height": int(height),
                                  "length": int(length), "batch_size": 1,
                                  "vae": W("vae", 0),
                                  "start_image": W("loadimage", 0)}},
            "ksampler": {"class": "ksampler",
                         "inputs": {"seed": int(plan.get("seed", 0)), "steps": steps,
                                    "cfg": cfg, "sampler_name": sampler,
                                    "scheduler": scheduler, "denoise": 1.0,
                                    "model": W("modelsampling", 0),
                                    "positive": W("pos", 0),
                                    "negative": W("neg", 0),
                                    "latent_image": W("latent", 0)}},
            "vaedecode": {"class": "vaedecode",
                          "inputs": self._vaedecode_inputs(W)},
        }
        for nid in set(external_results or ()):
            graph.pop(nid, None)
        return graph

    def _vaedecode_inputs(self, W):
        """VAEDecode inputs; the tiled path adds the schema-verified tile/temporal
        knobs (FROZEN; temporal_size = frames decoded at a time = the big
        video-VAE peak lever)."""
        base = {"samples": W(self._samples_node, 0), "vae": W("vae", 0)}
        if not self._tiled_vae():
            return base
        # ``_tile_geometry`` is the ONE implementation (LANE 2). It used to be a
        # closure here, and ``_recipe_departures`` would have needed a second
        # copy to report what a sweep cell actually decoded with.
        _i = self._tile_geometry
        base.update({
            "tile_size": _i("vae_tile"),
            "overlap": _i("vae_overlap"),
            "temporal_size": _i("vae_temporal"),
            "temporal_overlap": _i("vae_temporal_overlap"),
        })
        return base

    # ---- residency ----
    def load(self):
        """Fail CLOSED until installed, then resolve the installed CORE Wan node
        classes. Weights load when the loader nodes execute in render_clip."""
        if not self._installed():
            raise RuntimeError(
                "%s not installed: UNET missing at %s -- fetch the Wan2.2 "
                "TI2V-5B GGUF, set OTR_ENABLE_WAN_TI2V=1"
                % (self.name, self._ckpt_path()))
        from . import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    def render_clip(self, request, prepared):
        """Drive ONE image->video clip via the in-process Wan TI2V-5B graph: stage
        the init image, execute the graph, encode the decoded IMAGE batch to a
        SILENT bt709 clip (V-1), retain the MODEL patcher for V-4 teardown, and
        assert the mid-render NVML ceiling. M7 ffprobe-proves the silent-clip
        contract before the mux trusts it. Fail-closed NAMED on a missing node /
        init image.

        THE PING-PONG IS NARROWED, NOT RIPPED (WIRE-W3b, 2026-07-29). On a
        SINGLE-CLIP beat this adapter still renders short on purpose and fills
        the beat with a mirror-extended clip -- that is the shipped 8 GB tier
        contract (``PBUG-20260723-02``) and it is untouched. On a
        COVERAGE-PLANNED segment it is forbidden: ``coverage_plan``'s own
        docstring says a planned segment delivers the frames it was planned
        for, and a mirrored tail on a lane declaring ``strict_first_frame``
        would hand the NEXT segment a mirrored frame to chain from. Worse, the
        pad wears the right frame count, so a render that did not happen passes
        ``render_driver``'s ``got != segment.render_frames`` gate wearing the
        number of one that did.

        ``prepared["session_ctx"]["multi_clip"]`` is the discriminator, and it
        is the ONLY honest one available here: ``BeatSession`` sets it from the
        stamped plan's segment count, whereas a segment's REQUEST is shaped
        exactly like a single-clip beat's. A caller that prepared nothing reads
        as single-clip, which is the conservative direction."""
        from . import wrapper_bridge as _wb
        from ._tmp import otr_engine_tmp_mp4
        plan = self._build_render_request(request)            # pure, CPU-tested
        if not plan["init_image"]:
            raise _wb.GraphExecutionError(
                "%s requires init_image (got %r)" % (self.name, plan["init_image"]))
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates())
        width, height = self._dims(request)
        session = (prepared or {}) if isinstance(prepared, dict) else {}
        multi_clip = bool((session.get("session_ctx") or {}).get("multi_clip"))
        target_frames = int(plan.get("target_frame_count") or 0)
        # THE LENGTH IS DECIDED BEFORE ANYTHING IS STAGED (the eng_ltx_8gb B4
        # ordering, adopted here): a request this adapter cannot render is
        # knowable from pure numbers, so the refusal must not arrive after an
        # init image has been written -- and never after the GPU work.
        if multi_clip:
            length = self._planned_length(
                target_frames, width, height,
                session.get("hoisted_vram_mb"))
        else:
            # CLIP-FILL: PREDICT the VRAM-affordable render length for THIS
            # canvas (never react-to-OOM); the short render is ping-pong-
            # extended to the beat target below so the composite never
            # freeze-fills the beat.
            length = self._floor_length(target_frames, width, height,
                                        session.get("hoisted_vram_mb"))
        image_name = self._materialize_init_image(
            request, plan["init_image"], width, height)
        # The beat-scoped handles prepare() hoisted, if any. Their node
        # definitions drop out of the graph and run_graph resolves the wires
        # from these instead -- and it adds them to `keep`, so the
        # free_after_use pass below cannot evict a handle the NEXT segment
        # still needs.
        external = session.get("external_results") or {}
        graph = self._build_graph(request, image_name, plan, length, width, height,
                                  external_results=external)
        # free_after_use: the umt5 text-encode frees before the 5B UNET + the 2.2
        # VAE decode; "unet" kept for V-4 patcher teardown, "vae" for the decode,
        # the terminal for the IMAGE read-out. The NVML peak probe spans the whole
        # render window so the render-phase peak gates the ceiling.
        probe = _MC.VramPeakProbe(interval_s=0.1).start()
        try:
            results = _wb.run_graph(graph, classes, free_after_use=True,
                                    keep={"unet", "vae", self._TERMINAL},
                                    external_results=external)
            images = results[self._TERMINAL][0]               # VAEDecode IMAGE batch
        finally:
            render_peak = probe.stop()
        bucket = prepared.setdefault("patchers", self._patchers) \
            if isinstance(prepared, dict) else self._patchers
        model = results.get("unet", (None,))[0]
        if model is not None and callable(getattr(model, "detach", None)) \
                and id(model) not in {id(p) for p in bucket}:
            bucket.append(model)
        frames = _wb.images_to_uint8(images)
        n_native = len(frames)
        # THE PIPELINE INVARIANT (WIRE-W3b, the eng_ltx_8gb B4 invariant #1
        # adopted here). The graph was asked for exactly ``length`` frames, so
        # anything else is an under-delivery -- and until now the ping-pong
        # below absorbed it for ANY reason, not just the deliberate VRAM cap,
        # so a decode that silently returned short was indistinguishable from
        # one that did what it was told.
        if n_native != length:
            raise _wb.GraphExecutionError(
                "%s asked its graph for %d frame(s) and decoded %d. NO "
                "FALLBACK -- padding the difference is how a render that did "
                "not happen gets counted as one." % (self.name, length, n_native))
        extension_mode = "none"
        if target_frames > n_native:
            if multi_clip:
                # Unreachable by construction (``_planned_length`` returns the
                # planned length or refuses, and the invariant above proves the
                # decode matched it), and it stays because "unreachable" is a
                # claim about today's callers. If it ever fires, the beat is
                # about to be assembled from part-real, part-mirrored segments.
                raise _wb.GraphExecutionError(
                    "%s segment rendered %d real frame(s) of a planned "
                    "%d. NO FALLBACK -- ping-pong padding a COVERAGE-PLANNED "
                    "segment puts mirrored frames inside a beat that claims "
                    "real multi-clip coverage, and hands the next chained "
                    "segment a mirrored frame to continue from."
                    % (self.name, n_native, target_frames))
            # NO MIRRORS (operator ruling 2026-08-02). Every second of audio
            # gets ORIGINAL video, so a short render is a refusal, never a
            # ping-pong. Coverage planning below routes the beat to several
            # affordable NATIVE segments; once it does, this is unreachable.
            raise _wb.GraphExecutionError(
                "%s rendered %d real frame(s) of a %d-frame beat and REFUSES to "
                "ping-pong the difference. Every second of audio gets ORIGINAL "
                "video -- a mirrored tail is re-used video wearing a new "
                "timestamp. The render was VRAM-bounded to %d frames at %dx%d; "
                "route this beat through coverage planning so it is covered by "
                "several affordable NATIVE segments, or free VRAM so the whole "
                "beat fits one render. NO FALLBACK."
                % (self.name, n_native, target_frames, n_native, width, height))
        # NAMED FOR THE ENGINE THAT RENDERED IT (2026-08-01). This was a literal
        # "otr_wan_ti2v_", so the first live fastwan_8gb render wrote
        # otr_wan_ti2v_<hash>.mp4 -- a scratch file whose name attributes the work
        # to the wrong engine, which is exactly the thread you pull when a render
        # looks wrong and the filename sends you to the other adapter.
        out_path = otr_engine_tmp_mp4("otr_%s_" % self.name)
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path, self.target_fps)
        # M7: PROVE the silent-clip color/stream contract on the emitted mp4.
        validate_silent_clip_contract(ffprobe_clip_fields(path), self.target_fps)
        if not os.environ.get("OTR_TEST_MODE"):
            post_mb = _MC.vram_used_mb() or 0
            _LOG.info("[OTR video] wan_ti2v VRAM render-phase peak %s MB / post %s "
                      "MB", render_peak, post_mb)
        # NEWBUG-1 (2026-07-20): thread the MEASURED render-window peak into the raw
        # so _clip_from_raw stamps it -> the manifest vram_peak_mb receipt is populated
        # for the wan lane (was ALWAYS None: the peak was logged, never returned).
        # Required for the wan_8gb VRAM-acceptance telemetry.
        #
        # `recipe` is the same shape of receipt for the render CONFIG: it rides
        # _clip_from_raw -> the manifest row -> stamp_durable(meta.render_engines),
        # so a published episode can be asked which recipe rendered it. It was
        # None on every WAN clip until the freeze landed.
        #
        # `native_frame_count` / `extension_mode` (WIRE-W3b, 2026-07-29): how
        # many of the emitted frames this engine actually RENDERED, and how the
        # rest got there. Without them a ping-ponged clip and a real one are
        # indistinguishable downstream -- both carry the right frame_count --
        # so the W5 acceptance grader has no way to reject a mirror-filled clip
        # on a lane claiming real multi-clip coverage.
        return {"out_path": path, "frame_count": n, "vram_peak_mb": render_peak,
                "recipe": self._recipe_receipt(),
                "native_frame_count": n_native,
                "extension_mode": extension_mode,
                **self._clip_telemetry(width, height)}

    def _clip_telemetry(self, width, height):
        """``quant`` / ``use_lora`` / ``render_canvas`` for the manifest row.

        THE GAP THIS CLOSES. `render_driver` has carried these three onto the
        manifest since S-B/E5 and `otr_video_render_batch` rolls them up as
        IDENTITY fields -- but no WAN adapter ever PRODUCED them, so every clip
        shipped `null` for all three. The first live `fastwan_8gb` episode
        stamped its recipe correctly and still reported `quant: null`, which is
        how the hole was found: the plumbing was complete except for the source.

        Why it matters: `render_canvas` is exactly the field that would have
        answered the 2026-07-23 live-vs-bench VRAM question immediately instead
        of costing a separate investigation.

        ``use_lora`` is a BOOL and stays one -- `eng_ltx_av` sets it True/False,
        `_ROLLUP_IDENTITY_FIELDS` groups on it and `otr_credits_roll`
        truthiness-tests it, so a filename here would make the rollup incoherent
        ACROSS engines. The artifact's identity lives in the recipe receipt and
        the model manifest, which is where a reader can act on it."""
        return {
            "quant": self._quant_label(),
            "use_lora": bool(self._loader_names().get("lora")),
            "render_canvas": "%dx%d" % (int(width), int(height)),
        }

    def _quant_label(self):
        """The quant token of the UNET actually loaded, e.g. ``'Q5_K_M'``.

        Read off the resolved basename rather than assumed, so a swapped GGUF
        cannot leave a receipt describing the file it replaced. ``None`` when the
        loader is not a GGUF (a safetensors UNET has no quant) or the basename
        carries no recognizable token -- an honest absence, never a guess."""
        if self._loader_mode() != "gguf":
            return None
        base = str(self._loader_names().get("unet") or "")
        stem = base.rsplit(".", 1)[0]
        for token in reversed(stem.split("-")):
            t = token.strip()
            if t.upper().startswith("Q") and any(c.isdigit() for c in t):
                return t
        return None

    def canonicalize(self, raw, request, profile):
        return self._clip_from_raw(raw, request)


__all__ = ["WanTi2vEngine", "RECIPE_WAN_TI2V", "WAN_TI2V_RECIPE",
           "PREQUALIFICATION_ENV"]
