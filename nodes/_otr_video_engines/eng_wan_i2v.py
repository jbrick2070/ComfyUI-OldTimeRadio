"""Wan 2.2 image->video motion adapter (A-S5 / CW-6) -- in-process, default-OFF.

Wan i2v animates a still init image into motion. It runs on CORE ComfyUI Wan
nodes (``UNETLoader`` / ``CLIPLoader`` type=wan / ``WanImageToVideo`` / ``KSampler``
/ ``VAEDecode`` + a ``ModelSamplingSD3`` sigma shift) -- NOT the KJ Wan wrapper.
The KJ wrapper's pin audit demands numpy<2 / transformers<=4.51.3, but this box is
numpy 2.4 / transformers 5.5, and KJNodes is what drags in SageAttention; core
Comfy supports the A14B natively, so the engine deliberately stays off that dep
path (GO_FORWARD section 1A, 2026-06-12).

Like ltx_video it runs IN-PROCESS in the main cu130 venv by default. Isolation is
``sidecar_optional``: ``resolve_isolation`` escalates to a cu128 sidecar ONLY when
SageAttention is ACTIVE (``comfy.model_management.sage_attention_enabled()``, the
real switch -- mere module residency from Comfy's import-time probe does not
count, BUG-070). The headless WAN lane boots without ``--use-sage-attention`` so
the in-process path is the norm. Registered DARK -- empty ``default_roles``, so
nothing selects it unless a profile's ``role_overrides`` does; it fails CLOSED
until the checkpoint resolves on disk. ``OTR_ENABLE_WAN_I2V`` is VESTIGIAL and
gates nothing: the registry IS the menu, and no code reads that variable
(corrected lane 1, 2026-08-11 -- the old wording here said the lane was "gated
behind" it, which sent a reader looking for a switch that does not exist while
the real reason the lane would not start was a wrong checkpoint default).

init_image aspect: the init is MATERIALIZED into the canvas (padded / cropped per
``aspect_policy`` with ONE uniform scale, default ``pad``) and the derived image
is staged, so ``WanImageToVideo``'s internal resize is a no-op and a portrait init
NEVER silently stretches into a landscape canvas (pre-mortem N9). Import-time is
cold-import clean (V-12). UTF-8, no BOM, ASCII-only.

The pure dims/aspect/materialize/canonicalize helpers + the M7 silent-clip
contract proof are SHARED with wan_ti2v via :mod:`wan_shared` (GO_FORWARD 4A:
"share only pure helpers; keep loaders + node candidates + graph SEPARATE"); the
loaders, node candidates and the 14B I2V graph below stay engine-specific.

Config (env). READ THE SPLIT -- since the WAN recipe freeze (2026-07-27) the
render knobs do NOT bind on a production leg:

* STILL LIVE, every leg -- ASSET LOCATION only: ``OTR_ENABLE_WAN_I2V``
  (vestigial, read by nothing); ``OTR_WAN_I2V_CKPT`` an explicit full path that
  wins over the default; ``OTR_WAN_I2V_UNET_NAME`` / ``OTR_WAN_I2V_CLIP_NAME`` /
  ``OTR_WAN_I2V_VAE_NAME`` loader basenames; ``OTR_WAN_I2V_LOADER``
  safetensors|gguf (else INFERRED from the resolved basename -- it stays live
  WITH the name it follows); ``OTR_WAN_I2V_UNET_DIR`` / ``OTR_WAN_I2V_CLIP_DIR``
  / ``OTR_WAN_I2V_VAE_DIR`` dir overrides.

  NONE of them is REQUIRED to start the lane (lane 1, 2026-08-11). The UNET
  default names the installed artifact and :meth:`WanI2VEngine._installed`
  falls back to ``folder_paths``, so a box that installed the weight anywhere
  ComfyUI knows about resolves with nothing set. That is the difference between
  fixing this box and fixing the lane.
* FROZEN IN CODE as ``WAN_I2V_RECIPE``, ignored-with-a-warning: ``OTR_WAN_STEPS``
  / ``OTR_WAN_CFG`` / ``OTR_WAN_SHIFT`` / ``OTR_WAN_SAMPLER`` /
  ``OTR_WAN_SCHEDULER`` / ``OTR_WAN_NEGATIVE``. These six were read INLINE in
  ``_build_graph`` with bare ``int()`` / ``float()`` -- no range check, no named
  refusal, so a mistyped value died as a raw ValueError mid-render.
* THE CONSENT ACT: set ``OTR_WAN_I2V_PREQUALIFICATION=1`` and they bind again,
  range-checked and fail-closed, for a MEASUREMENT run -- whose clips stamp a
  ``+prequalification`` recipe receipt.

The six frozen names are UN-NAMESPACED (``OTR_WAN_*``, not ``OTR_WAN_I2V_*``)
and are deliberately left that way here. Renaming an operator-facing knob is an
operator's call, not a coder's, and the freeze already removes the power that
made the missing namespace dangerous: they cannot bind a production leg at all.
Flagged for the operator rather than changed silently.

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
from .._otr_shared.still_plan_helpers import StillPlanRow
from .frame_contract import CONTINUITY_STRICT_FIRST_FRAME, FrameContract
from .registry import EngineUnusable, EngineUsabilityReason, register
# Re-export the shared M7 clip-contract helpers so render_clip below and the
# existing test imports (``from ...eng_wan_i2v import validate_silent_clip_contract``)
# keep resolving after the helpers moved to wan_shared.
from .wan_shared import (  # noqa: F401
    _WAN_DEFAULT_NEGATIVE, _parse_fps, ffprobe_clip_fields,
    validate_silent_clip_contract)

_LOG = logging.getLogger("OTR.video.wan_i2v")

_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
_COMFY_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))

# A-ship in-process forward. Shared mechanics (resolution, declarative executor,
# silent bt709 encode, VRAM guard) are proven in wrapper_bridge; the GRAPH below is
# the CORE-NODE Wan 2.2 image->video topology, VERIFIED against the installed
# /object_info on the 5080 (2026-06-12, GO_FORWARD 1A): UNETLoader (or UnetLoaderGGUF)
# -> ModelSamplingSD3(shift) -> KSampler.model; CLIPLoader(type=wan) -> CLIPTextEncode
# x2; VAELoader + LoadImage -> WanImageToVideo[positive,negative,latent] -> KSampler
# (uni_pc/simple) -> VAEDecode. The bare /prompt smoke (scripts/otr_wan_smoke.py)
# rendered this graph end-to-end (832x480, 81f, real motion, no warp). Filenames /
# knobs are env-overridable; the init_image is materialized padded/cropped (never
# stretched, N9). NOTE: the low-noise 14B fp8 is a SINGLE expert -- the two-expert
# HIGH/LOW MoE handoff (Path B) is a future option, not wired here.
_WAN_MIN_FRAMES = 33
_WAN_MAX_FRAMES = 177

#: THE INSTALLED UNET BASENAME (lane 1, 2026-08-11). The default used to be
#: ``checkpoints/wan2.2-i2v.safetensors`` -- a placeholder name in a placeholder
#: category that exists on no box this project has ever run on, so
#: ``assert_usable`` raised ``MISSING_MODEL`` before the first forward and the
#: lane shipped DEAD. The artifact the operator actually installed is the
#: low-noise 14B fp8-scaled single expert, and it lives under
#: ``diffusion_models`` like every other Wan weight here (the sibling
#: ``eng_wan_ti2v`` has always defaulted into that category).
#:
#: Naming it here rather than only in a launcher env is what makes the lane
#: resolve with NOTHING set, which is the point: an env pin fixes one box, a
#: correct default plus the ``folder_paths`` fallback in :meth:`_installed`
#: fixes every box that installed the weight anywhere ComfyUI knows about.
_I2V_DEFAULT_UNET = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"

#: The defined recipe-receipt string threaded into the manifest.
#:
#: THE VERSION LIVES IN THE STRING, not in a separate constant, so it reaches
#: ``_clip_from_raw`` -> the manifest row -> ``stamp_durable(meta.render_engines)``
#: and a published episode can be asked which recipe rendered it. This adapter
#: stamped ``recipe: None`` on every clip before the freeze.
RECIPE_WAN_I2V = "wan22_14b_i2v_single_pass_v1"

#: THE CONSENT ACT for THIS adapter. Separate from wan_ti2v's on purpose: one
#: switch for both tiers would stamp ``+prequalification`` on a clip that had
#: rendered with the frozen recipe, and a receipt that lies in the safer
#: direction still lies.
PREQUALIFICATION_ENV = "OTR_WAN_I2V_PREQUALIFICATION"

#: THE FROZEN wan_i2v RECIPE, v1 (LANE 1, 2026-07-27).
#:
#: TODAY'S SHIPPED DEFAULTS, lifted verbatim from the inline ``os.environ``
#: fallbacks they replace -- NOT a measured selection; no WAN sweep has run.
#: Behaviour-preserving on any box that set nothing, and reversible: a
#: prequalification run measures and produces v2 (bump the version INSIDE
#: ``RECIPE_WAN_I2V`` and add a new dict; never edit a versioned dict in place,
#: or receipts already on disk stop being interpretable).
#:
#: ``sampler`` is ``uni_pc``, NOT the portable-floor ``euler`` that wan_ti2v
#: freezes. That difference is deliberate and pre-existing: wan_ti2v is the
#: 8GB/Mac/AMD floor and carries a portability whitelist; the 14B I2V lane never
#: had one, and inventing one here would refuse a configuration that ships and
#: works today. The freeze preserves behaviour; it does not add policy.
WAN_I2V_RECIPE_V1 = {
    "steps": 20,
    "cfg": 3.5,
    #: ModelSamplingSD3 sigma shift; 8.0 is the 14B value (the 5B uses 5.0).
    "shift": 8.0,
    "sampler": "uni_pc",
    "scheduler": "simple",
    #: The negative conditioning text. A RENDER INPUT: this adapter has no
    #: per-shot negative channel, so before the freeze the server's boot
    #: environment was the SOLE author of it.
    "negative": _WAN_DEFAULT_NEGATIVE,
}

#: THE ONE NAME EVERY CONSUMER READS.
WAN_I2V_RECIPE = WAN_I2V_RECIPE_V1

#: The env var each frozen field was read from, kept so the demotion can NAME
#: what it is ignoring. Presence is all this map is used for outside the consent
#: act. The names are un-namespaced for the reason given in the module docstring.
_RECIPE_ENV_KEYS = {
    "steps": "OTR_WAN_STEPS",
    "cfg": "OTR_WAN_CFG",
    "shift": "OTR_WAN_SHIFT",
    "sampler": "OTR_WAN_SAMPLER",
    "scheduler": "OTR_WAN_SCHEDULER",
    "negative": "OTR_WAN_NEGATIVE",
}


#: S1 (2026-07-25) per-model still plan for wan_i2v (spec
#: ``docs/2026-07-25-still-plans-locked-build-spec.md`` section 3, Shape A --
#: scene spine). Module-scoped so the class body carries only the reference,
#: not the 40-line authoring block; this is FILE-LOCAL (no cross-module
#: sharing per the spec's "no inheritance or shared defaults" rule -- the plan
#: is FULLY DECLARED here and the class assigns the whole tuple, never a
#: partial override). scene_open + scene_beat + scene_character all WIDE +
#: required=always; portrait per subject at the engine's own aspect
#: (inherit_engine), not required (character portraits are consumed only when
#: a beat names a cast row). Nothing reads the plan at S1 -- S2 wires it in.
_WAN_I2V_STILL_PLAN = (
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
class WanI2VEngine(_WS.WanInitImageMixin, _MC.MotionEngineBase):
    """The wan_i2v image->video adapter (in-process default; sidecar_optional)."""

    name = "wan_i2v"
    family = "image_to_video"
    #: Still-aspect identity (2026-06-17): Wan I2V renders 16:9, so the director
    #: mints a WIDE init still (non-HuMo, non-mesh-portrait).
    render_aspect = "wide"
    #: S1 per-model still plan (see ``_WAN_I2V_STILL_PLAN`` above).
    still_plan = _WAN_I2V_STILL_PLAN
    # Animate a still into motion -- the roles that can supply an init image
    # (music visual, character). (retired_role_a/retired_role_b removed
    # 2026-07-01, rip-sfx-broll.)
    roles = ("music_visual", "character_video")
    default_roles = ()
    required_inputs = ("init_image",)
    commercial_clean = False            # license is profile data; verify-at-build
    requires_flag = None  # vestigial (registry IS the menu; no flag gate)
    engine_version = "1"
    declared_isolation = _MC.ISOLATION_SIDECAR_OPTIONAL
    target_fps = 25
    #: THE FRAME LADDER (chunk 7a, 2026-07-26). ``_WAN_MIN_FRAMES`` /
    #: ``_WAN_MAX_FRAMES`` are the adapter's own named constants; the 4-frame
    #: quantum is Wan's 4n+1 latent stride (33 = 4*8+1, 177 = 4*44+1).
    #: CONTINUITY: the graph wires LoadImage -> WanImageToVideo ``start_image``
    #: (see ``_build_graph``), which IS a hard first-frame lock, so a successor
    #: segment may begin exactly on its predecessor's terminal frame.
    frame_contract = FrameContract(
        min_frames=_WAN_MIN_FRAMES,
        max_frames=_WAN_MAX_FRAMES,
        quantum=4,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_STRICT_FIRST_FRAME,
    )

    #: THE CANVAS, DECLARED (lane 1, 2026-08-11). The profile has always said
    #: 832x480, but with no declaration here ``build_request_from_shot`` fell
    #: through to the 1472x832 landscape default -- 3.07x the pixels the tier
    #: was configured for, and the likely cause of the 15.3 GB over-ceiling
    #: measurement of 2026-08-08. ``declared_render_canvas`` is applied LAST and
    #: overrules ledger and env, so declaring it is what actually closes the
    #: channel; the profile keeps one job, a drift guard.
    #:
    #: THE RE-MEASUREMENT THIS COMMENT USED TO ASK FOR IS DONE, and the ruling
    #: is KEEP. At the declared canvas the 14B fp8 measures 13.93 GiB warm /
    #: 14.05 cold at 832x480x33 (vram-recipe-lab exoneration receipts, indexed
    #: in docs/evidence/video_evidence_manifest.json), which fits the 14.5 GiB
    #: gate. It does NOT displace wan_ti2v as the default WAN lane: a 0.10 GiB
    #: cold margin is too thin to make a lane the one everything falls back to.
    #: Honest bound: only the f33 rung has warm evidence, so the f177 ceiling
    #: this contract declares is model-legal and NOT machine-qualified.
    #: Both axes /32-legal (26 x 15).
    render_canvas = (832, 480)

    # ---- config resolution (env override -> box default) ----
    def _ckpt_path(self):
        return os.environ.get("OTR_WAN_I2V_CKPT") or os.path.join(
            _COMFY_ROOT, "models", "diffusion_models", _I2V_DEFAULT_UNET)

    def _installed(self):
        """The 14B UNET is present if the explicit path exists OR
        ``folder_paths`` resolves its basename.

        THE LANE-1 FIX (lesson L1). This was a bare ``os.path.exists`` on
        :meth:`_ckpt_path`, which knows exactly one location; ``folder_paths``
        knows every configured model root, including the
        ``extra_model_paths.yaml`` entries that put this project's weights in
        ``C:\\ComfyUI-Models`` rather than under the comfy root's ``models/``.
        So a correctly installed weight read as missing and the lane refused
        before its first forward. Mirrors ``eng_wan_ti2v._installed`` exactly --
        the sibling has had this fallback all along, which is the whole reason
        one WAN lane worked and the other did not.
        """
        if os.path.exists(self._ckpt_path()):
            return True
        return self._resolve_model_file(
            ("diffusion_models", "unet", "checkpoints"),
            self._loader_names()["unet"], "OTR_WAN_I2V_UNET_DIR") is not None

    def resolve_isolation(self):
        """Runtime isolation tier: ``in_process`` by default, escalating to
        ``sidecar_required`` when SageAttention is resident (BUG-070). Pure read
        of the declared tier + ``sys.modules``."""
        return _MC.resolve_isolation(
            self.declared_isolation, _MC.sageattention_patched())

    # ---- usability (fail-closed BEFORE any forward; no heavy import) ----
    def _aux_loader_files(self):
        """(label, folder_paths categories, basename, dir-override env) for the
        REQUIRED aux graph loaders beyond the UNET ckpt: the umt5 CLIP and the
        VAE. M6 (GO_FORWARD 4A): a forward dies mid-graph without these, so
        assert_usable proves all three loaders present, not just the ckpt."""
        names = self._loader_names()
        return (
            ("CLIP/umt5", ("text_encoders", "clip"), names["clip"],
             "OTR_WAN_I2V_CLIP_DIR"),
            ("VAE", ("vae",), names["vae"], "OTR_WAN_I2V_VAE_DIR"),
        )

    def assert_usable(self, host_caps, profile, request_template=None):
        """Fail closed before any forward: ALL three graph loaders
        (UNET + umt5 CLIP + VAE) present on disk. Wan TOLERATES Sage by
        escaping to a sidecar (``resolve_isolation``), so it does NOT hard-fail
        on Sage the way ltx_video does."""
        # Fail CLOSED on a bad render knob BEFORE any forward -- under the
        # consent act only, where the knobs can actually bind. On a production
        # leg this is where the demotion notice is emitted.
        self._resolve_render_config()
        if not self._installed():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "wan_i2v checkpoint %r did not resolve: not at %s, and "
                "folder_paths found it in none of diffusion_models / unet / "
                "checkpoints. Install the Wan 2.2 I2V UNET under a configured "
                "models root, or point OTR_WAN_I2V_UNET_DIR at its directory, "
                "or set OTR_WAN_I2V_CKPT to its full path (the core UNETLoader "
                "reads the basename)"
                % (self._loader_names()["unet"], self._ckpt_path()),
                kind="video")
        missing = self._missing_loaders()
        if missing:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "wan_i2v required loader file(s) absent: %s -- the core Wan graph "
                "needs the umt5 CLIP and the VAE on disk too, not just the UNET "
                "ckpt (offline invariant, no runtime fetch); fix the *_NAME envs "
                "or point OTR_WAN_I2V_CLIP_DIR / OTR_WAN_I2V_VAE_DIR at the "
                "installed basenames"
                % ", ".join("%s=%r" % (lbl, nm) for lbl, nm in missing),
                kind="video")
        return self.name

    #: Terminal node of the graph (its IMAGE output is encoded to the clip).
    _TERMINAL = "vaedecode"

    #: The ONE node id hoisted for the whole beat. See :meth:`prepare`.
    _SESSION_NODES = ("unet",)

    def session_identity(self):
        """What this adapter's HANDLES are -- engine, recipe, loader modes,
        weights. ``BeatSession`` refuses a multi-segment beat without it.

        WHY THIS ALONE WOULD HAVE BEEN A TRAP, recorded because it was very
        nearly built that way: declaring an identity SILENCES
        ``SessionIdentityUnavailable`` and then the segment graph still
        executes ``UNETLoader`` every segment. The refusal would be gone and
        the reload would not. So this landed in the SAME chunk as
        :meth:`prepare`'s hoist, never before it.

        WHAT IT CARRIES AND WHY:

        * the recipe receipt -- the frozen sampling knobs are baked into the
          patcher ``ModelSamplingSD3`` clones, and the version rides inside the
          receipt string, so a recipe bump moves the identity for free.
        * BOTH loader MODES. The UNET's mode picks the loader CLASS
          (``UNETLoader`` vs ``UnetLoaderGGUF``), and a beat that switched
          class mid-way would be reusing a handle its own graph no longer knows
          how to have produced.
        * every loader file's token AND receipt -- including the CLIP and the
          VAE, which are NOT hoisted. r4/A5: both adapters load a VAE
          independently and TI2V distinguishes INCOMPATIBLE VAE generations, so
          a VAE swapped between segments is exactly the drift this exists to
          catch even though the handle itself is per-segment.

        It EXCLUDES per-segment state -- prompt, seed, frame count, canvas,
        init image -- which change every segment by design.

        Not cached: the whole job is to notice a weight that MOVED, so the
        receipts are re-stat'ed on every ask."""
        return (self.name,
                self._recipe_receipt(),
                "unet_loader=%s" % (self._loader_mode(),),
                repr(self._wan_session_receipts()))

    def prepare(self, host_caps, profile, session_ctx):
        """Load the UNET ONCE PER BEAT instead of once per segment.

        THE UNET ONLY, and the reason is arithmetic rather than taste. The
        umt5 CLIP and the VAE are deliberately LEFT in the segment graph
        because ``render_clip`` runs with ``free_after_use=True`` precisely so
        that umt5-fp8 and the 14B fp8 UNET are never co-resident through the
        sampler -- the bare smoke without it peaked at 14,499 MB on a 16 GiB
        card. Hoisting the CLIP would pin ~9 GB resident for the whole beat and
        nullify the one mitigation standing between this lane and an OOM. The
        text encode happens per segment either way, so hoisting it buys wall
        clock and pays for it in the only currency this tier is short of.

        ``ModelSamplingSD3`` stays per segment too: it is a cheap clone+patch
        of an already-resident MODEL and is correctly per-render.

        Order matters, and each step is here because of a specific defect
        (the ``eng_ltx_8gb`` B1b sequence, followed deliberately):

        1. Resolve the frozen config BEFORE ``super().prepare()`` so a bad
           render knob is refused before the cross-process GPU lease is taken.
           A raise after the lease strands it for the life of the server.
        2. ``super().prepare()`` takes the lease and resolves classes.
        3. Execute a LOADER-ONLY mini-graph, registering the patcher through
           ``on_result`` as it lands -- if a later node raised, ``run_graph``
           never returns a results dict and an unregistered patcher is VRAM
           nothing will ever detach.
        4. On ANY failure tear down what we took and re-raise, with the
           teardown itself wrapped so a raising ``unload`` cannot replace the
           real error with its own.
        """
        from . import wrapper_bridge as _wb
        self._resolve_render_config()           # range-checked, fail-closed
        prepared = super().prepare(host_caps, profile, session_ctx)
        try:
            classes = getattr(self, "_classes", None) \
                or _wb.resolve_graph_classes(self._node_candidates())
            names = self._loader_names()
            bucket = prepared.setdefault("patchers", self._patchers)
            seen = {id(p) for p in bucket}

            def _register(nid, out):
                """Own the handle the moment it exists, not after the graph
                returns. Slot 0 is the MODEL patcher ``_detach_patchers``
                knows how to unwind; the duck-typed ``detach`` check is what
                keeps that honest rather than a positional assumption."""
                model = out[0] if out else None
                if (model is not None and id(model) not in seen
                        and callable(getattr(model, "detach", None))):
                    bucket.append(model)
                    seen.add(id(model))

            if self._loader_mode() == "gguf":
                unet_inputs = {"unet_name": names["unet"]}
            else:
                unet_inputs = {"unet_name": names["unet"],
                               "weight_dtype": "default"}
            results = _wb.run_graph(
                {"unet": {"class": "unet", "inputs": unet_inputs}},
                classes, on_result=_register)
            out = results.get("unet") or ()
            if not out:
                raise _wb.GraphExecutionError(
                    "wan_i2v hoisted UNET %r returned no output; the beat "
                    "session has nothing to reuse and every segment would "
                    "silently reload" % (names["unet"],))
            prepared["external_results"] = results
        except BaseException:
            try:
                self.teardown(prepared)
            except BaseException:          # noqa: BLE001 -- never mask the cause
                pass
            raise
        return prepared

    def teardown(self, prepared):
        """Drop the beat-scoped handles BEFORE delegating.

        ``prepared["external_results"]`` holds a strong reference to the
        hoisted UNET. The base teardown detaches the patchers, unloads,
        releases the lease and then WAITS for machine-wide VRAM to settle --
        and every one of those steps would run with this dict still pinning the
        very tensors it is trying to reclaim, so the stability wait would be
        watching its own referent. Clearing first costs nothing.

        Never raises out of teardown (it is a finally-path)."""
        if isinstance(prepared, dict):
            prepared.pop("external_results", None)
        return super().teardown(prepared)

    # ---- in-process graph spec (CORE Wan i2v; verified vs /object_info) ----
    def _loader_mode(self):
        """The UNET loader mode: ``safetensors`` (core ``UNETLoader``) or ``gguf``
        (``UnetLoaderGGUF``). Explicit via ``OTR_WAN_I2V_LOADER``; else inferred
        from the unet filename extension. Fail-closed: the resolver raises NAMED if
        the chosen loader class is not installed."""
        mode = (os.environ.get("OTR_WAN_I2V_LOADER") or "").strip().lower()
        if mode in ("gguf", "safetensors"):
            return mode
        return ("gguf" if self._loader_names()["unet"].lower().endswith(".gguf")
                else "safetensors")

    def _node_candidates(self):
        """Ordered ComfyUI node-class candidates per graph node (CORE Wan 2.2 i2v,
        verified against the installed /object_info 2026-06-12). The unet loader
        switches on ``_loader_mode`` so both the fp8 safetensors and a GGUF quant
        resolve fail-closed."""
        unet_cls = (("UnetLoaderGGUF",) if self._loader_mode() == "gguf"
                    else ("UNETLoader",))
        return {
            "unet": unet_cls,
            "modelsampling": ("ModelSamplingSD3",),
            "clip": ("CLIPLoader",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "vae": ("VAELoader",),
            "loadimage": ("LoadImage",),
            "wan": ("WanImageToVideo",),
            "ksampler": ("KSampler",),
            "vaedecode": ("VAEDecode",),
        }

    def _resolve_render_config(self):
        """The FROZEN render knobs (shared by assert_usable and _build_graph).

        PRODUCTION: the recipe binds whatever the server booted with, and
        anything the environment is trying to say is NAMED and ignored -- never
        PARSED, so a stale malformed value cannot kill a leg it cannot
        influence.

        PREQUALIFICATION: the knobs are open, every value is range-checked and
        fails CLOSED with a named MALFORMED_CONFIG. Before the freeze these six
        were read INLINE in ``_build_graph`` with bare ``int()`` / ``float()``,
        so a mistyped value surfaced as a raw ValueError from the middle of a
        graph build with nothing naming the knob that caused it."""
        if not _WR.prequalification_active(PREQUALIFICATION_ENV):
            _WR.warn_ignored(_LOG, self.name, RECIPE_WAN_I2V,
                             _WR.ignored_override_keys(_RECIPE_ENV_KEYS),
                             PREQUALIFICATION_ENV)
            # SPELLED OUT, not a dict slice, so both legs return the SAME KEY
            # SET and no reader can hit a KeyError that reproduces in one mode.
            return {
                "steps": WAN_I2V_RECIPE["steps"],
                "cfg": WAN_I2V_RECIPE["cfg"],
                "shift": WAN_I2V_RECIPE["shift"],
                "sampler": WAN_I2V_RECIPE["sampler"],
                "scheduler": WAN_I2V_RECIPE["scheduler"],
                "negative": WAN_I2V_RECIPE["negative"],
            }
        _WR.warn_honoured(_LOG, self.name, RECIPE_WAN_I2V,
                          _WR.ignored_override_keys(_RECIPE_ENV_KEYS),
                          PREQUALIFICATION_ENV)
        _num = _WR.config_number
        return {
            "steps": _num(self, _RECIPE_ENV_KEYS["steps"],
                          WAN_I2V_RECIPE["steps"], 1, 100, int),
            "cfg": _num(self, _RECIPE_ENV_KEYS["cfg"],
                        WAN_I2V_RECIPE["cfg"], 0.0, 30.0, float),
            "shift": _num(self, _RECIPE_ENV_KEYS["shift"],
                          WAN_I2V_RECIPE["shift"], 0.1, 20.0, float),
            # No portability whitelist on this lane -- see the recipe's comment.
            # An exported-empty value means unset, as everywhere else.
            "sampler": _WR.config_text(self, _RECIPE_ENV_KEYS["sampler"],
                                       WAN_I2V_RECIPE["sampler"]),
            "scheduler": _WR.config_text(self, _RECIPE_ENV_KEYS["scheduler"],
                                         WAN_I2V_RECIPE["scheduler"]),
            "negative": _WR.config_text(self, _RECIPE_ENV_KEYS["negative"],
                                        WAN_I2V_RECIPE["negative"],
                                        strip=False),
        }

    def _recipe_departures(self, knobs=None):
        """Which frozen knobs THIS CELL actually changed. ``{}`` on production.

        Every field of this adapter's recipe comes from one resolver, so the
        departure walk is the whole resolved dict against the whole frozen one
        -- there is no in-effect gate to apply, because there is no knob here
        that a render can fail to reach.

        NOTHING IS RESOLVED ON A PRODUCTION LEG: the early return keeps the
        "never parse what cannot bind" guarantee structural rather than
        dependent on the resolver staying correct, and keeps the demotion
        notice from being emitted twice per leg."""
        if not _WR.prequalification_active(PREQUALIFICATION_ENV):
            return {}
        knobs = self._resolve_render_config() if knobs is None else knobs
        return _RD.departures(WAN_I2V_RECIPE, knobs)

    def _recipe_receipt(self, knobs=None):
        """THE stamp -- the single stamp site goes through here and nowhere
        else."""
        return _WR.recipe_receipt(RECIPE_WAN_I2V, PREQUALIFICATION_ENV,
                                  self._recipe_departures(knobs))

    def _loader_names(self):
        """Model FILENAMES the loader nodes consume (env-overridable; defaults are
        placeholders the operator confirms against the installed Wan models)."""
        return {
            "unet": os.environ.get("OTR_WAN_I2V_UNET_NAME")
            or os.path.basename(self._ckpt_path()),
            "clip": os.environ.get(
                "OTR_WAN_I2V_CLIP_NAME", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
            "vae": os.environ.get("OTR_WAN_I2V_VAE_NAME", "wan_2.1_vae.safetensors"),
        }

    def _build_graph(self, request, image_name, plan, length, width, height,
                     external_results=None):
        """The declarative Wan i2v graph (wrapper_bridge.run_graph format). CORE
        Wan 2.2 topology, verified against the installed /object_info. The unet
        loader emits ``weight_dtype`` only on the safetensors path; the GGUF path
        (UnetLoaderGGUF) takes ``unet_name`` alone. ModelSamplingSD3 applies the
        sigma shift (~8.0 for the 14B) the bare placeholder omitted.

        ``external_results`` names the ids the CALLER already produced and owns
        for the whole beat (WIRE-W3: ``prepare()`` loads the UNET once). Those
        node DEFINITIONS are omitted while every wire that reads them stays --
        the executor resolves them from the caller's handles instead. Omitting
        rather than leaving them is not an optimisation: ``run_graph`` REFUSES a
        graph that also defines an id the caller supplied, because then two
        parties claim to produce one output.

        Conditional on purpose. A caller that prepared nothing -- the
        single-clip path, which is what production runs today, and every
        sibling test that hand-builds ``prepared={"patchers": []}`` -- gets the
        loader nodes exactly as before, byte for byte."""
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        names = self._loader_names()
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        knobs = self._resolve_render_config()         # frozen, range-checked
        steps = knobs["steps"]
        cfg = knobs["cfg"]
        shift = knobs["shift"]
        sampler = knobs["sampler"]
        scheduler = knobs["scheduler"]
        positive = get("text_prompt") or "subtle natural motion"
        negative = knobs["negative"]
        if self._loader_mode() == "gguf":
            unet_inputs = {"unet_name": names["unet"]}
        else:
            unet_inputs = {"unet_name": names["unet"], "weight_dtype": "default"}
        graph = {
            "unet": {"class": "unet", "inputs": unet_inputs},
            "modelsampling": {"class": "modelsampling",
                              "inputs": {"model": W("unet", 0), "shift": shift}},
            "clip": {"class": "clip",
                     "inputs": {"clip_name": names["clip"], "type": "wan",
                                "device": "default"}},
            "pos": {"class": "pos",
                    "inputs": {"text": positive, "clip": W("clip", 0)}},
            "neg": {"class": "neg",
                    "inputs": {"text": negative, "clip": W("clip", 0)}},
            "vae": {"class": "vae", "inputs": {"vae_name": names["vae"]}},
            "loadimage": {"class": "loadimage", "inputs": {"image": image_name}},
            "wan": {"class": "wan",
                    "inputs": {"width": int(width), "height": int(height),
                               "length": int(length), "batch_size": 1,
                               "positive": W("pos", 0), "negative": W("neg", 0),
                               "vae": W("vae", 0), "start_image": W("loadimage", 0)}},
            "ksampler": {"class": "ksampler",
                         "inputs": {"seed": int(plan.get("seed", 0)), "steps": steps,
                                    "cfg": cfg, "sampler_name": sampler,
                                    "scheduler": scheduler, "denoise": 1.0,
                                    "model": W("modelsampling", 0),
                                    "positive": W("wan", 0),
                                    "negative": W("wan", 1),
                                    "latent_image": W("wan", 2)}},
            "vaedecode": {"class": "vaedecode",
                          "inputs": {"samples": W("ksampler", 0), "vae": W("vae", 0)}},
        }
        for nid in set(external_results or ()):
            graph.pop(nid, None)
        return graph

    # ---- residency (resolve the installed wrapper nodes; weights load on call) -
    def load(self):
        """Fail CLOSED until installed, then RESOLVE the installed CORE Wan node
        classes (fail-closed NAMED if absent). Weight load happens when the loader
        nodes execute inside render_clip; the live render-phase VRAM<=14.5 GB peak
        is asserted there via the NVML probe."""
        if not self._installed():
            raise RuntimeError(
                "wan_i2v not installed: checkpoint missing at %s -- fetch the Wan "
                "2.2 ckpt, set OTR_ENABLE_WAN_I2V=1 (core Comfy Wan nodes)"
                % self._ckpt_path())
        from . import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    def render_clip(self, request, prepared):
        """Drive ONE image->video clip via the in-process Wan wrapper graph: stage
        the init image into ComfyUI's input dir, execute the graph, encode the
        decoded IMAGE batch to a SILENT bt709 clip (V-1), retain the MODEL patcher
        for V-4 teardown, and assert the mid-render NVML ceiling. Returns the raw
        ``{out_path, frame_count}`` canonicalize() normalises. Fail-closed NAMED if
        a wrapper node or the init image is missing."""
        from . import wrapper_bridge as _wb
        from ._tmp import otr_engine_tmp_mp4
        plan = self._build_render_request(request)            # pure, CPU-tested
        if not plan["init_image"]:
            raise _wb.GraphExecutionError(
                "wan_i2v requires init_image (got %r)" % plan["init_image"])
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates())
        width, height = self._dims(request)
        # Materialize the init into the canvas (pad/crop, ONE uniform scale, N9) and
        # stage THAT, so WanImageToVideo's internal resize is a no-op and the
        # declared aspect_policy (default pad) is actually honoured.
        image_name = self._materialize_init_image(
            request, plan["init_image"], width, height)
        length = _wb.quantize_frames_4n1(
            plan["target_frame_count"] or self.target_fps,
            min_frames=_WAN_MIN_FRAMES, max_frames=_WAN_MAX_FRAMES)
        # The beat-scoped handles prepare() hoisted, if any. Their node
        # definitions drop out of the graph and run_graph resolves the wires
        # from these instead -- and it adds them to `keep`, so the
        # free_after_use pass below cannot evict a handle the NEXT segment
        # still needs. A caller that prepared nothing gets {} and the
        # single-clip path is byte-identical to before.
        external = (prepared.get("external_results")
                    if isinstance(prepared, dict) else None) or {}
        graph = self._build_graph(request, image_name, plan, length, width, height,
                                  external_results=external)
        # free_after_use (2026-06-09, the LTX capstone catch applied here
        # preemptively): umt5-fp8 + the 14B fp8 Wan UNET must not stay
        # co-resident through the sampler on a 16 GB card. "unet" is kept for
        # the V-4 patcher teardown, "vae" for the decode, the terminal for the
        # IMAGE read-out. The NVML peak probe spans the whole render window (first
        # heavy load through VAEDecode) so the render-phase peak (not a post-encode
        # instantaneous read) is what gates the 14.5 GB ceiling. (CS-4: if this
        # peak ever busts, split text-encode into a pre-pass that frees umt5 before
        # the 14B loads -- the bare /prompt smoke without free_after_use peaked at
        # the very edge, 14499 MB; the engine's free_after_use is the mitigation.)
        probe = _MC.VramPeakProbe(interval_s=1.0).start()
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
        out_path = otr_engine_tmp_mp4("otr_wan_")
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path, self.target_fps)
        # M7: PROVE the silent-clip color/stream contract on the emitted mp4
        # before canonicalize self-declares it -- ffprobe is the source of truth,
        # the hardcoded dict is not. Fail-LOUD NAMED on any drift.
        validate_silent_clip_contract(ffprobe_clip_fields(path), self.target_fps)
        if not os.environ.get("OTR_TEST_MODE"):
            post_mb = _MC.vram_used_mb() or 0
            _LOG.info("[OTR video] wan_i2v VRAM render-phase peak %s MB / post %s "
                      "MB", render_peak, post_mb)
        # `vram_peak_mb`: NEWBUG-1's fix (2026-07-20) landed for wan_ti2v and
        # was never applied here, so this adapter MEASURED the render-window
        # peak, logged it, and then discarded it -- `_clip_from_raw` read
        # raw.get("vram_peak_mb") and got None on every wan_i2v clip forever.
        # render_shot then silently fell back to an INSTANTANEOUS post-render
        # read, which can under-report the true peak, and that weaker number
        # rolls up into the episode figure and the credits card. Found by the
        # pre-push QA fan-out; the field has a real consumer and no other owner.
        #
        # `recipe`: the same shape of receipt for the render CONFIG, riding
        # into stamp_durable(meta.render_engines) so a published episode can be
        # asked which recipe rendered it.
        #
        # `native_frame_count` / `extension_mode` (WIRE-W3b, 2026-07-29): this
        # adapter has NEVER extended -- it renders the 4n+1 rung it asked for
        # and emits it. The fields are stamped anyway, because the W5 grader
        # reads them off the manifest and an ABSENT field is indistinguishable
        # from an unanswered one: a family where one adapter declares "none"
        # and its sibling declares nothing gives the grader no lane to trust.
        return {"out_path": path, "frame_count": n,
                "vram_peak_mb": render_peak,
                "recipe": self._recipe_receipt(),
                "native_frame_count": len(frames),
                "extension_mode": "none"}

    def canonicalize(self, raw, request, profile):
        return self._clip_from_raw(raw, request)


__all__ = ["WanI2VEngine", "RECIPE_WAN_I2V", "WAN_I2V_RECIPE",
           "PREQUALIFICATION_ENV"]
