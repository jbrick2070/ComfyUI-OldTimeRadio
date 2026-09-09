"""``animatediff15_lightning_video`` -- EXPERIMENTAL. The Ghost graph on a
DISTILLED module.

AN EXPERIMENTAL LANE, and the operator named it one (2026-09-09: *"you may as
well wire it in, if it fails it fails -- label it as an exp lane"*). It is not
a production path and must not become one by accident: it has never rendered
through the OTR adapter path, its ``device_backends`` row still says ``cuda``
only, its ``default_roles`` is empty so it can only ever be chosen per beat
from the director dropdown, and it carries no qualified cost row. Judge it by
eye, keep it or drop it, but do not build anything on top of it yet.

THE SPEED LANE, AND IT EXISTS BECAUSE OF A MEASUREMENT. AnimateDiff renders
CORRECTLY on Apple Silicon -- proven 2026-09-08 with clips faithful to the
``recur_frac`` style and judged by eye -- and is simply unusable there: the M4
measured **122-134 s/it**, and the golden recipe is 20 steps at cfg 8.0, which
is **40 UNet passes per beat** because ComfyUI batches cond+uncond whenever
cfg > 1. Roughly three quarters of an hour for one beat. That, not any defect,
is why no AnimateDiff clip has ever reached ``otr/obs/`` from a Mac.

This lane is 8 steps at cfg 1.0: **8 UNet passes**. Same weights family, same
seven node classes, same canvas, same cadence, same context window, no new code
path -- the recipe seam landed first (``8f882788``) and the parent already reads
all six cells through ``self``, so this file adds a class and nothing else.

ADDITIVE, exactly as the v3 peers were: the two lanes an operator can actually
select -- ``animatediff15_v3_haunted_video`` and its still-in lab peer -- keep
every cell they had, and a comparison against them is one-variable.

A NOTE ON THE PARENT, because it is not what it looks like.
``GhostSignalEngine`` is the class that rendered the published episode, but
``animatediff15_video`` IS NOT REGISTERED: the operator retired every
non-haunted AnimateDiff on 2026-08-23 (*"delete any animatediff that are not
haunted"*) and the id is tombstoned in ``RETIRED_ENGINE_IDS``. The class
survives as the unregistered base its siblings inherit, which is exactly what
this lane wants -- the golden RECIPE without the v3 adapter, reached by
subclassing a base rather than stripping a peer.

That retirement is also the honest tension in this lane and it is not resolved
here: this is a CLEAN AnimateDiff, and clean AnimateDiffs are the thing that
instruction deleted. It is built clean because the upstream receipt is clean and
a first render that also adds an unproven adapter would be testing two things at
once. If the operator wants the haunted look at Lightning speed, that is one
more subclass setting ``lora_name``/``lora_min_bytes``/``lora_strength`` -- the
adapter patches the SD1.5 UNet's attention and the motion module is a separate
insertion, so they are not obviously in conflict -- but the v3 adapter was
trained beside the v3 module, nothing upstream composes it with a distilled one,
and that is a second lane behind a second receipt, not a default here.

PROVENANCE -- A PROVEN GRAPH, NOT A GUESS
-----------------------------------------
``ByteDance/AnimateDiff-Lightning`` (63.2M downloads), and the cells below are
read off its own ``comfyui/animatediff_lightning_workflow.json`` rather than off
its README: ``ADE_AnimateDiffLoaderGen1`` widgets are the module name and
``sqrt_linear (AnimateDiff)``; the KSampler widgets are steps / cfg 1.0 /
``euler`` / ``sgm_uniform`` / denoise 1.0. The repo's own Note in that file
carries the one hard rule -- *"Make sure loading the correct Animatediff-
Lightning checkpoint corresponding to the inference steps"* -- which is why
``LIGHTNING_STEPS`` and the ``8step`` in the filename are pinned together by a
test rather than left as two numbers that happen to agree today.

NO DOMAIN ADAPTER, AND THAT IS FROM THE SOURCE. The official graph is
checkpoint -> loader -> KSampler with no LoRA node anywhere. The v3 adapter is
v3-PAIRED; composing it with a non-v3 distilled module has no upstream receipt.
So this lane subclasses ``GhostSignalEngine`` -- the BASE, where
``lora_name`` is ``None`` -- and not the haunted v3 peer. With no name there is
no loader node, no artifact to require and no third patcher to release.

THE COLLISION, STATED RATHER THAN BURIED
----------------------------------------
cfg 1.0 does not merely weaken the negative prompt; it deletes the pass.
``comfy/samplers.py:610`` in the pinned ComfyUI::

    if math.isclose(cond_scale, 1.0) and model_options.get(
            "disable_cfg1_optimization", False) == False:
        uncond_ = None

That is where the second half of the speedup comes from, and it is the precise
reason this codebase ALREADY refused a distilled lane once:
``eng_ghost_signal_official.py`` records *"AnimateLCM is deliberately absent and
stays absent: its CFG 1-2 regime kills the live negative, and the Ghost
lettering defence needs real unconditional conditioning"*, and
``eng_ghost_signal.py`` says the same of the golden cfg. SD1.5 volunteers
lettering into anything resembling a sign, a poster or a radio dial -- which is
most of this show -- and the golden lane fights that with negative tokens on
every beat.

The decision here is to ship the OFFICIAL cfg anyway, because 1.0 is the value
the module was distilled at and the only one carrying an upstream receipt: a
first render that also deviates on cfg would be testing two things at once. The
lettering risk is a LOOK defect an operator sees in one beat, and its fix is one
cell. Three things keep that honest rather than sloppy:

* ``sampler_inputs_for`` stamps ``negative_effective`` FALSE at cfg 1.0. A
  receipt that lists a negative prompt which conditioned nothing is a receipt
  that lies, and the shot-cache identity carries the flag too, so a cfg sweep
  cannot collide with a cfg-1.0 render in the cache.
* ``OTR_LIGHTNING_CFG`` is the sweep knob, following ``lora_strength`` and
  ``OTR_STILLIN_LAB_DENOISE``. Trying cfg 2.0 is an env var, never an edit.
* The lane is additive, so nothing that works today can regress into it.

WHAT IS UNPROVEN AND IS NOT CLAIMED. Upstream renders 16 frames with NO context
options. This lane inherits Ghost's 125-latent beat through the non-circular
Standard Static 16/4 sliding window, which is the same architecture but not the
same shape upstream measured. If the distilled module degrades across window
boundaries, that shows up as drift at the seams, and the honest response is a
receipt and a shorter beat -- not a quiet cap.
"""
from __future__ import annotations

import math

from .eng_ghost_signal import GhostSignalEngine
from .registry import EngineUnusable, EngineUsabilityReason, register

try:
    from .._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

#: The 8-step ComfyUI build. The step count is IN the filename and in
#: ``LIGHTNING_STEPS`` below, and upstream's own Note makes the pairing a rule
#: rather than a coincidence -- so a test pins them to each other. Swapping to
#: the 4-step artifact means changing BOTH.
MM_LIGHTNING_NAME = "animatediff_lightning_8step_comfyui.safetensors"

#: 908,929,664 bytes on disk. THE FLOOR TRAVELS WITH THE MODULE and it has to:
#: this artifact is legitimately SMALLER than ``v3_sd15_mm.ckpt`` (1,673,262,583)
#: and less than half of ``mm-p_0.5``, so the inherited 1.7 GB floor would have
#: refused a byte-perfect file as "truncated". That is the same defect that
#: killed the v3 lane's first live leg, and the reason the recipe seam was cut
#: before this file was written. 860 MB is under its own artifact and within 15%
#: of it, so a truncated fetch is still NAMED (G1.3).
MM_LIGHTNING_MIN_BYTES = 860_000_000

#: AND THE FLOOR CANNOT DO THE OTHER HALF OF THE JOB, which is why the filename
#: is pinned to ``LIGHTNING_STEPS`` by a test instead. Read off the upstream
#: repo listing: the 1-step, 2-step, 4-step and 8-step ComfyUI checkpoints are
#: ALL EXACTLY 908,929,664 bytes. No size check can tell them apart, and loading
#: the 4-step artifact under an 8-step receipt raises nothing anywhere -- it
#: just samples a schedule the weights were not distilled for. The byte floor
#: catches a truncated fetch; only the name catches the wrong variant.

#: The distilled cells. Read off upstream's own workflow JSON, not its README.
LIGHTNING_STEPS = 8
LIGHTNING_CFG = 1.0
LIGHTNING_SAMPLER_NAME = "euler"
LIGHTNING_SCHEDULER = "sgm_uniform"
LIGHTNING_DENOISE = 1.0

#: ``BetaSchedules.SQRT_LINEAR`` in the INSTALLED pack
#: (``ComfyUI-AnimateDiff-Evolved/animatediff/utils_model.py:149``). The
#: parenthetical is part of the dropdown value; ``"sqrt_linear"`` alone is a
#: DIFFERENT option (``RAW_SQRT_LINEAR``, line 163) and would silently select
#: another schedule.
LIGHTNING_BETA_SCHEDULE = "sqrt_linear (AnimateDiff)"

#: THE EXTERNAL DECODER. SD1.5's baked VAE is soft; this is the standard
#: AnimateDiff decoder upgrade and it was A/B'd on this machine on 2026-09-09 --
#: identical prompt, seed 42, cfg 1.0, 8 steps, 512x288, ONLY the decoder
#: changed. Cleaner glass, better foliage separation, less milky haze, and the
#: operator judged it directly. Decode cost is unchanged (the 28.8 s in that run
#: was a warm checkpoint, not a faster VAE).
#:
#: LICENCE NOTE, AND IT IS THE GOOD DIRECTION: MIT. That is MORE permissive than
#: anything else this lane loads -- the SD1.5 checkpoint and the Lightning
#: module are both CreativeML Open RAIL-M -- so adopting it cannot weaken the
#: lane's licence position. ``commercial_clean`` is unchanged for the reasons
#: that already applied to the other two artifacts.
VAE_FT_MSE_NAME = "vae-ft-mse-840000-ema-pruned.safetensors"

#: 334,641,190 bytes on disk, verified against the Hub API rather than taken
#: from a summary (sha256 735e4c3a...f3c75, revision 629b3ad3...). The floor
#: sits under its own artifact and within the family's 15% margin, so a
#: truncated fetch is still NAMED.
VAE_FT_MSE_MIN_BYTES = 320_000_000

#: The cfg sweep knob. Distilled at 1.0; the operator may want the live negative
#: back at the cost of doubling the UNet passes, and that must not be an edit.
LIGHTNING_CFG_ENV = "OTR_LIGHTNING_CFG"

#: The cfg above which the unconditional pass is actually evaluated. Not a
#: preference -- it is ``comfy/samplers.py``'s own ``math.isclose(cond_scale,
#: 1.0)`` test, restated here so the receipt's ``negative_effective`` flag is
#: computed from the SAME rule the sampler applies.
def negative_is_live(cfg) -> bool:
    """True when ComfyUI will actually evaluate the unconditional pass.

    Mirrors ``comfy/samplers.py::sampling_function``: at ``isclose(cfg, 1.0)``
    the uncond conditioning is replaced with ``None`` and never runs, so every
    negative token on the beat is inert. The receipt reads this, so it can never
    claim a defence the sampler did not apply.
    """
    try:
        value = float(cfg)
    except (TypeError, ValueError):
        return False
    return not math.isclose(value, 1.0)


@register
class GhostSignalLightningEngine(GhostSignalEngine):
    """``animatediff15_lightning_video`` -- Ghost on AnimateDiff-Lightning 8-step."""

    name = "animatediff15_lightning_video"

    motion_module_name = MM_LIGHTNING_NAME
    motion_min_bytes = MM_LIGHTNING_MIN_BYTES
    #: REPOINTED, never edited in place, because clips already exist on disk
    #: under the pre-decoder id and a receipt that changes meaning retroactively
    #: makes every older receipt uninterpretable.
    recipe_receipt_id = "animatediff_sd15_lightning8_ftmse_static16_512x288_v1"

    #: All six recipe cells, through the seam. Declaring fewer would sample on
    #: the golden 20-step / cfg-8.0 recipe while stamping a Lightning receipt --
    #: wrong pixels under a confident label, which is the exact failure
    #: ``docs/ADDING_IMAGE_AND_VIDEO_LANES.md`` names.
    steps = LIGHTNING_STEPS
    sampler_name = LIGHTNING_SAMPLER_NAME
    scheduler = LIGHTNING_SCHEDULER
    denoise = LIGHTNING_DENOISE
    beta_schedule = LIGHTNING_BETA_SCHEDULE

    #: THE EXTERNAL DECODER, declared here and nowhere else so the two lanes
    #: with published episodes keep decoding exactly as they always have.
    vae_name = VAE_FT_MSE_NAME
    vae_min_bytes = VAE_FT_MSE_MIN_BYTES

    #: THE FRAME MATH, AND THIS LANE IS WHERE IT LANDS FIRST. Every other video
    #: lane resolves a beat into lengths its model actually accepts; this family
    #: never did, because its constraint is a context window rather than a VRAM
    #: ceiling and the window scheduler quietly absorbs an illegal count by
    #: backing its final window up. At the 125-source beat this lane renders,
    #: that leaves the last window overlapping the previous by 15 of 16 frames
    #: instead of 4.
    #:
    #: Rounding up to 136 costs NOTHING in sampler time -- the model is invoked
    #: once per window and both counts are eleven windows -- so the only price is
    #: decoding 11 surplus frames the lane already discards and already reports
    #: in ``model_frame_count``. See ``ghost_legal_source_count``.
    #:
    #: ON THIS LANE AND NOT ITS SIBLINGS, deliberately: they have published
    #: episodes and changing their latent count changes their pictures. This one
    #: has never rendered, so it can start correct instead of being corrected.
    align_source_to_context_window = True

    #: ADAPTIVE HOLD (PBUG-20260909-01). This lane's first real episode rendered
    #: five beats and then asked for 160 latents on the sixth, which REBOOTED the
    #: machine -- unified memory, so an OOM is not a process kill. With this on,
    #: that beat runs at hold 3 instead: 112 latents, the same 320 delivered
    #: frames, the same audio sync, no jump cuts, and 8.3 unique sources per
    #: second instead of 12.5. Beats that already fit keep hold 2 untouched.
    #:
    #: On this lane and not its siblings for the usual reason: they have
    #: published episodes and a cadence change changes their pictures.
    adaptive_hold_for_memory = True

    #: DECLARED EXPLICITLY (G3.6) rather than inherited. This lane consumes no
    #: still of any kind, and restating it here is what keeps the portrait-free
    #: role set correct if a future parent ever changes its mind.
    accepts_still = False
    still_plan: tuple = ()

    #: LICENCE TRUTH, AND IT IS BETTER THAN THE GOLDEN LANE'S. Lightning is
    #: published under CreativeML Open RAIL-M -- the SAME licence as the SD1.5
    #: checkpoint, and a real grant, unlike ``mm-p_0.5``'s host which publishes
    #: none. That removes the BLOCKER standing between Ghost Signal and a
    #: community template pack. It does not by itself make the lane commercially
    #: clean: RAIL-M carries use restrictions and this build has not had them
    #: reviewed, and claiming otherwise is exactly the overclaim the admission
    #: rules exist to stop. Same reasoning, same value, as the v3 lane.
    commercial_clean = False

    # ---- the cfg dial ---------------------------------------------------- #
    @property
    def cfg(self) -> float:
        """The distilled default, or a deliberate sweep.

        A PROPERTY, and not the ``hold_factor`` ``__init__`` pattern, for two
        reasons that both come from ``engine_registry_base.py:148``: ``register``
        instantiates the class AT IMPORT and keeps ONE shared instance, so (a)
        an ``__init__`` that raises would take the whole director dropdown down
        for every machine, and (b) a value frozen at construction would freeze
        whatever the environment held when ComfyUI imported -- useless for a
        knob whose entire purpose is to change between renders in one process.
        The class-level ``<property object>`` footgun the parent warns about on
        ``hold_factor`` does not bite here: nothing in the repo reads ``.cfg``
        off a class, only off an instance, and a test pins that.

        Unlike ``lora_strength`` this one REFUSES a malformed value instead of
        falling back to the default: cfg decides whether
        the negative prompt runs at all, and a silently-defaulted cfg would
        print a receipt whose ``negative_effective`` flag describes a render
        nobody asked for.
        """
        raw = otr_env.get(LIGHTNING_CFG_ENV)
        if raw is None or not str(raw).strip():
            return float(LIGHTNING_CFG)
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError):
            value = float("nan")
        if not math.isfinite(value) or value <= 0.0:
            raise EngineUnusable(
                self.name, self.family,
                EngineUsabilityReason.MALFORMED_CONFIG,
                "%s: %s=%r is not a finite positive number. cfg decides whether "
                "the unconditional pass runs at all, so it is never defaulted "
                "silently -- a receipt must name the guidance that made the "
                "picture." % (self.name, LIGHTNING_CFG_ENV, raw),
                kind="video")
        return value

    def assert_usable(self, host_caps, profile, request_template=None):
        """The parent's gates, plus resolving the dial so a malformed sweep is
        named HERE rather than 8 steps into a beat."""
        super().assert_usable(host_caps, profile, request_template)
        _ = self.cfg
        return self.name

    # ---- receipts --------------------------------------------------------- #
    def sampler_inputs_for(self, request):
        """The parent's cells plus the one fact the parent cannot know.

        ``negative_effective`` is computed from the resolved cfg through the
        same ``isclose(cfg, 1.0)`` rule the sampler applies, so the receipt can
        never claim the lettering defence ran when ComfyUI skipped it.
        """
        out = super().sampler_inputs_for(request)
        out["negative_effective"] = negative_is_live(out.get("cfg"))
        return out

    def shot_cache_identity(self, request):
        """cfg joins the identity, so a cfg-2.0 sweep cannot be served a
        cfg-1.0 clip out of the shot cache.

        LOSSLESS, NOT ``%.4f``. The first version of this rounded, and rounding
        is wrong here in a way that is specific rather than theoretical: cfg
        1.00004 formats to the same ``cfg=1.0000`` as cfg 1.0, yet
        ``math.isclose(1.00004, 1.0)`` is FALSE -- so those two renders differ in
        whether ComfyUI evaluates the unconditional pass at all, and one could
        be served the other's clip. ``float.hex()`` round-trips every double
        exactly, so no two distinct cfgs can collide.

        ``negative_effective`` rides along explicitly rather than being implied
        by the cfg token: the receipt asserts it, so the cache key that decides
        whether a clip may be reused must assert it too.
        """
        cfg = float(self.cfg)
        plan = self._build_render_request(request)
        return super().shot_cache_identity(request) + (
            "cfg=%s" % cfg.hex(),
            "negative_effective=%s" % negative_is_live(cfg),
            # THE CADENCE THAT WILL RUN. The parent's identity carries
            # `unique_source_count`, which usually differs when the hold does --
            # but "usually" is not "always", and a clip rendered at hold 3 must
            # never be served to a beat that resolved to hold 2.
            "hold=%d" % int(plan.get("hold", self.hold_factor)),
        )


__all__ = ["GhostSignalLightningEngine",
           "VAE_FT_MSE_NAME", "VAE_FT_MSE_MIN_BYTES",
           "MM_LIGHTNING_NAME", "MM_LIGHTNING_MIN_BYTES",
           "LIGHTNING_STEPS", "LIGHTNING_CFG", "LIGHTNING_SAMPLER_NAME",
           "LIGHTNING_SCHEDULER", "LIGHTNING_DENOISE",
           "LIGHTNING_BETA_SCHEDULE", "LIGHTNING_CFG_ENV",
           "negative_is_live"]
