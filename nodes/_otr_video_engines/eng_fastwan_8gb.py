"""``fastwan_8gb`` -- FastWan 2.2 TI2V-5B as a 3-step DMD distillation.

ADDITIVE, NOT A REPLACEMENT (operator ruling 2026-08-01). ``wan_ti2v`` (public
``wan_8gb``) stays exactly as it is. This is a THROUGHPUT tier, and the label
should be read that way: it is the same base weights at the same canvas producing
the same amount of motion, ~2.7x sooner. It is NOT a quality upgrade and NOT a
longer-clip upgrade -- the four-arm bench measured arm A and arm C at an identical
peak delta (6563.1 / 6531.1 / 6563.1 MiB at 17 / 49 / 81 frames), because step
count does not change peak activation.

WHY A SUBCLASS. FastWan differs from the incumbent by a rank-128 LoRA, 3 steps
instead of 30, cfg 1.0 instead of 5.0, and a restart transition instead of euler.
Everything else -- the beat-scoped hoist, ``on_result`` patcher registration, the
``hoisted_vram_mb`` cost correction, teardown, session identity, the frame ladder,
tiled decode, init-image staging, aspect policy -- is the same 5B substrate.
Re-parenting to ``WanInitImageMixin`` would have duplicated roughly 800 lines of
lifecycle; subclassing touches the incumbent's RECIPE zero times, so the
2026-07-27 freeze holds.

THE SEAM THIS RIDES. The parent's recipe accessors used to read module-level
constants, which a class attribute cannot override -- a subclass declaring its own
recipe would have SILENTLY rendered with ``wan_ti2v``'s and stamped a FastWan
receipt on the result. That was fixed first (``recipe_id`` / ``recipe_data`` /
``recipe_env_keys`` / ``prequalification_env`` / ``max_frames_env`` /
``_model_source_node`` / ``_samples_node`` / ``_hoist_graph``), and this adapter is
the first consumer of it.
"""

import os

from . import eng_wan_ti2v as _WT
from . import motion_common as _MC
from . import wan_recipe as _WR
from . import wrapper_bridge as _wb
from .registry import register

#: The recipe-receipt string threaded into the manifest. The version lives IN the
#: string (see the incumbent's note): bumping a recipe means repointing
#: ``recipe_data`` and bumping this, never editing a versioned dict in place, or
#: receipts already on disk stop being interpretable.
RECIPE_FASTWAN_8GB = "fastwan22_ti2v_5b_dmd3_i2v_v1"

#: PROMPT-STYLE OVERLAY -- STORED, NOT WIRED (item C, 2026-08-17). Schema, caps
#: and the adoption gate: 2026-08-17-per-engine-prompt-style-guide-RESEARCH.md
#: in the docs dir -- deliberately named WITHOUT a path prefix, because
#: ``tools/engine_matrix.py`` scrapes engine sources for cap-evidence citations
#: and a phrasing doc is not frame evidence. The directive is the only half that
#: may ever reach a model or a prompt; 240 chars, hard, pinned by
#: ``tests/test_prompt_style_directives.py``.
PROMPT_STYLE_DIRECTIVE = (
    "There is no unconditional branch, so exclusions have no effect: state every "
    "requirement positively. Name the subject, then one action and its speed. Do "
    "not restate the set; the still fixes it. Keep it short."
)

#: Humans only -- never injected, never sent to a model.
PROMPT_STYLE_NOTES = """\
CONFIG AS SHIPPED: FastWan on the 8 GB tier -- a DMD-distilled LoRA over the same
Q5_K_M Wan 2.2 TI2V-5B GGUF base ``wan_ti2v`` uses, run as ONE forward pass per
step with NO unconditional branch, i.e. cfg 1.0. The negative is INERT.

WHY THE NEGATIVE FACT LEADS, and why it is worded as a mechanism rather than a
number. On the other inert lanes the negative is skipped because cfg is 1.0; here
the distilled recipe runs ONE forward pass per step with no unconditional branch
at all, so on the shipped path there is nothing for a negative to condition.
Naming the mechanism keeps a future reader from "fixing" the directive when they
notice the sibling ``wan_ti2v`` runs cfg 5.0 with a live negative over the same
base weights.

BE PRECISE ABOUT "CANNOT BE UNDONE" -- an earlier draft of this note claimed no
env knob or prequalification override could re-enable a negative here, and a
Sonnet QA pass proved that false. There IS such a path: ``prequalification_active``
plus the ``cfg`` entry in ``_FASTWAN_RECIPE_ENV_KEYS`` lets the INHERITED
``WanTi2vEngine._resolve_render_config`` move cfg off 1.0, and the moment cfg
leaves 1.0 the unconditional branch is back and the frozen negative starts
counting. That is a deliberate CONSENT act for measurement rather than a
production render -- and it is exactly why the consent env is this adapter's own
(``PREQUALIFICATION_ENV_FASTWAN``) instead of shared with ``wan_ti2v``: one switch
for two tiers would open both.

What genuinely does NOT exist is a negative-TEXT channel. ``_resolve_render_config``
reads steps, cfg, shift, sampler, scheduler, sigmas and lora_strength, and no
"negative" key -- so nothing can change WHAT the negative says, only whether it
counts. The directive is written for the shipped path.

"KEEP IT SHORT" IS THE 8 GB TIER TALKING. This adapter exists to fit a tighter
budget than its sibling, and it is the same distilled-model reasoning as the LTX
lanes: few steps, so early conditioning dominates and late correction is not
available. Short is not a stylistic preference here, it is what the sampler can
actually act on.

THE BASE WEIGHTS ARE BIT-IDENTICAL TO ``wan_ti2v``'S, AND THE DIRECTIVES STILL
DIFFER -- deliberately. Same weights, opposite guidance regime (cfg 1.0 with no
unconditional branch versus cfg 5.0 with a live negative), so the phrasing advice
diverges even though the checkpoint does not. This is the case that shows why the
overlay is keyed per ENGINE rather than per model file.

EXTERNAL RESEARCH (2026-08-17) -- **NOT RUN SEPARATELY FOR THIS LANE, and saying
so rather than implying coverage.** The other nine lanes got their own web lookup;
this one did not, because FastWan is a DMD-distilled LoRA over the same Wan 2.2
TI2V-5B base as ``wan_ti2v`` and there is no separate published prompting guidance
for the distillation. What carries over, and what does not:
  * CARRIES OVER: the subject-first MECHANISM recorded on
    ``eng_wan_ti2v.PROMPT_STYLE_NOTES`` -- Wan 2.2 captions were subject-first in
    training and the weights favour early tokens. Same base weights, so the same
    mechanism applies, and the directive's "Name the subject, then..." rests on it.
  * DOES NOT CARRY OVER: everything about the guidance regime. Published Wan
    guidance assumes cfg 5-7 with a live negative and treats negative prompting as
    essential for artifact control. This lane runs one forward pass per step with no
    unconditional branch, so that entire half of the upstream advice is inert here.
    Importing it would be the single most likely way to get this engine's directive
    wrong.
  * OWED, if anyone wants this lane fully covered: a lookup on DMD / distilled
    step-count regimes specifically, not on Wan 2.2 in general.

PROVENANCE: authored by the driver from this engine's shipped configuration plus
the five directive rules in the RESEARCH doc, with only the inherited Wan
subject-first finding folded in. NOT a measured finding on our lane, and NOT a
researched lane in its own right. Treat this string as a hypothesis until the probe
A/B runs at a fixed seed.
"""

#: THE CONSENT ACT, this adapter's own. Never shared with ``wan_ti2v``: one switch
#: for two tiers would open both and stamp ``+prequalification`` on a clip that had
#: rendered frozen, and a receipt that lies in the safer direction still lies.
PREQUALIFICATION_ENV_FASTWAN = "OTR_FASTWAN_8GB_PREQUALIFICATION"

#: The LoRA that IS the distillation. Kijai's rank-128 bf16 extraction of FastWan
#: 2.2 over the SAME Q5_K_M GGUF base arm A uses -- base weights bit-identical to
#: the incumbent's, which is why the two measure the same peak.
FASTWAN_LORA_NAME = "Wan2_2_5B_FastWanFullAttn_lora_rank_128_bf16.safetensors"

#: THE SCHEDULE, AS A STRING, because that is the wire format. ``ManualSigmas``
#: takes a COMMA-SEPARATED STRING, not a tuple -- verified against the working
#: bench graph that qualified this recipe (node ``9b``; the bench itself was
#: retired 2026-08-23, the finding it produced stands).
#: A tuple here would not serialize into the graph.
#:
#: These are FastVideo's ``denoising_step_list`` divided by 1000. ``steps`` is
#: ``len(sigmas) - 1`` and that identity is asserted below, so the two can never
#: drift into a graph that evaluates the model a different number of times than
#: the receipt claims.
FASTWAN_SIGMAS = "1.0, 0.757, 0.522, 0.0"

#: THE FROZEN fastwan_8gb RECIPE, v1. Every value is pinned from the FastVideo
#: CODE PATH (``DmdDenoisingStage``) and the qualified bench cell -- never a model
#: card, never community usage advice.
FASTWAN_8GB_RECIPE_V1 = {
    #: 3 model evaluations. This IS the distillation's claim.
    "steps": 3,
    #: 1.0 = ONE forward pass per step (no uncond branch). Half of why it is fast;
    #: raising it would both slow it and leave the distilled regime.
    "cfg": 1.0,
    #: ModelSamplingSD3 sigma shift; 5.0 is the 5B value, same as the incumbent.
    "shift": 5.0,
    #: Not a stock sampler name -- the transition is the registered
    #: OTR_DMDRestartSamplerSelect node. Named here so the receipt says what ran.
    "sampler": "dmd_restart",
    #: The schedule is literal, not generated by a scheduler.
    "scheduler": "manual_sigmas",
    "sigmas": FASTWAN_SIGMAS,
    "lora_strength": 1.0,
    #: Frozen negative, inherited verbatim from the incumbent's v1 default.
    #: Bound to the VERSIONED dict, not the mutable `WAN_TI2V_RECIPE` alias: a
    #: future WAN bump that touched `negative` would otherwise silently rewrite
    #: THIS frozen recipe while FastWan's own receipt still claimed v1 (r3
    #: wiring panel, codex). Lane 6 bumped that alias to v2 and did not move
    #: `negative`, so this pin is behaviour-preserving today.
    "negative": _WT.WAN_TI2V_RECIPE_V1["negative"],
    #: Tiled decode ON: the video-VAE decode is a top VRAM-peak driver at 8GB, and
    #: it is why peak stays FLAT across clip length on this tier.
    "tiled_vae": True,
    "vae_tile": 256,
    "vae_overlap": 64,
    "vae_temporal": 16,
    "vae_temporal_overlap": 8,
}

#: THE ONE NAME EVERY CONSUMER READS.
FASTWAN_8GB_RECIPE = FASTWAN_8GB_RECIPE_V1

# The recipe cannot ship claiming a step count its own schedule does not produce.
assert FASTWAN_8GB_RECIPE["steps"] == len(
    [s for s in FASTWAN_SIGMAS.split(",") if s.strip()]) - 1, (
    "fastwan_8gb: steps must equal len(sigmas) - 1")

#: Each frozen field's env name, kept so the demotion can NAME what it ignores.
_FASTWAN_RECIPE_ENV_KEYS = {
    "steps": "OTR_FASTWAN_8GB_STEPS",
    "cfg": "OTR_FASTWAN_8GB_CFG",
    "shift": "OTR_FASTWAN_8GB_SHIFT",
    "sampler": "OTR_FASTWAN_8GB_SAMPLER",
    "scheduler": "OTR_FASTWAN_8GB_SCHEDULER",
    "sigmas": "OTR_FASTWAN_8GB_SIGMAS",
    "lora_strength": "OTR_FASTWAN_8GB_LORA_STRENGTH",
    "negative": "OTR_FASTWAN_8GB_NEGATIVE",
    "tiled_vae": "OTR_FASTWAN_8GB_TILED_VAE",
    "vae_tile": "OTR_FASTWAN_8GB_VAE_TILE",
    "vae_overlap": "OTR_FASTWAN_8GB_VAE_OVERLAP",
    "vae_temporal": "OTR_FASTWAN_8GB_VAE_TEMPORAL",
    "vae_temporal_overlap": "OTR_FASTWAN_8GB_VAE_TEMPORAL_OVERLAP",
}


@register
class FastWan8gbEngine(_WT.WanTi2vEngine):
    """FastWan 2.2 TI2V-5B, 3-step DMD restart (8GB tier; throughput)."""

    name = "fastwan_8gb"
    engine_version = "1"

    #: FALSE, and NOT inherited (kibitz r4). The incumbent declares True because
    #: its GGUF and VAE are Apache-2.0 at the source. FastWan's base weights are
    #: the SAME file, and FastVideo itself declares apache-2.0 -- but the LoRA
    #: actually loaded is a Kijai extraction from a repo with NO repo-level
    #: licence file (docs/2026-07-31-arm-c-fastwan-BUILD-SPEC.md s6A). "Both
    #: upstreams say apache-2.0" is not the same claim as "this artifact is
    #: notice-compliant", and a commercial_clean flag is read as the second.
    #: Under-claim until the notice chain is resolved; flipping this is a
    #: one-line change once it is.
    commercial_clean = False

    # ---- the recipe seam ------------------------------------------------- #
    recipe_id = RECIPE_FASTWAN_8GB
    recipe_data = FASTWAN_8GB_RECIPE
    recipe_env_keys = _FASTWAN_RECIPE_ENV_KEYS
    prequalification_env = PREQUALIFICATION_ENV_FASTWAN
    #: Its OWN ceiling channel. Inheriting the incumbent's would let the WAN
    #: tier's ceiling silently cap an engine that never opted into it.
    max_frames_env = "OTR_FASTWAN_8GB_MAX_FRAMES"

    #: THE CANVAS, DECLARED (kibitz r2/r3). ``render_driver.declared_render_canvas``
    #: applies this LAST in ``build_request_from_shot``, so it cannot be displaced
    #: by the shared landscape default. The reason to declare is BOOT-INDEPENDENCE,
    #: not VRAM: the incumbent reaches the same 832x480 only through
    #: ``launch.env.OTR_VIDEO_LANDSCAPE_CANVAS``, which binds only if the server was
    #: booted with that profile -- the PBUG-20260723-02 dead-channel class. Both
    #: axes are /32-legal (26 x 15).
    render_canvas = (832, 480)

    #: The graph swaps KSampler for a SamplerCustom chain, so the model source and
    #: the latent source both move. Named on the parent precisely so this is two
    #: lines rather than two duplicated methods.
    _model_source_node = "lora"
    _samples_node = "sampler"
    #: Hoist the LoRA-patched model too -- see :meth:`_hoist_graph`.
    _SESSION_NODES = ("unet", "lora")

    #: The transition is a registered node, not a stock sampler name, so the
    #: incumbent's cross-platform whitelist does not describe this engine.
    _PORTABLE_SAMPLERS = frozenset({"dmd_restart"})
    _PORTABLE_SCHEDULERS = frozenset({"manual_sigmas"})
    _SAMPLER_HINT = (" -- fastwan_8gb's transition IS the recipe "
                     "(OTR_DMDRestartSamplerSelect); a stock sampler name here "
                     "would render a 3-step clip through the wrong transition.")

    # ---- assets ---------------------------------------------------------- #
    def _lora_name(self):
        """The LoRA FILENAME the loader node consumes (env-overridable)."""
        return os.environ.get("OTR_FASTWAN_8GB_LORA_NAME", FASTWAN_LORA_NAME)

    def _loader_names(self):
        """The incumbent's three files plus the LoRA that IS the distillation."""
        names = super()._loader_names()
        names["lora"] = self._lora_name()
        return names

    def _aux_loader_files(self):
        """Adds the LoRA so ``assert_usable`` FAILS CLOSED when it is absent.

        Without this row a missing LoRA would sail through preflight and render a
        3-step clip through the UN-DISTILLED base model -- no error, just ruined
        output wearing a FastWan receipt. It also lands the LoRA in
        ``session_identity`` for free: ``_wan_session_receipts`` is built from
        ``_loader_names()`` + this, so a swapped or resized LoRA opens a new beat
        session without anything here being edited."""
        return super()._aux_loader_files() + (
            ("LoRA/FastWan", ("loras",), self._lora_name(),
             "OTR_FASTWAN_8GB_LORA_DIR"),
        )

    # ---- graph ----------------------------------------------------------- #
    def _node_candidates(self):
        """The incumbent's graph with the LoRA loader added and KSampler replaced.

        ``OTR_DMDRestartSamplerSelect`` is OURS (``dmd_sampler.py``, registered in
        the pack's root mappings); the other three are stock core nodes."""
        cands = super()._node_candidates()
        cands.pop("ksampler", None)          # replaced by the chain below
        cands["lora"] = ("LoraLoaderModelOnly",)
        cands["sigmas"] = ("ManualSigmas",)
        cands["dmdsampler"] = ("OTR_DMDRestartSamplerSelect",)
        cands["sampler"] = ("SamplerCustom",)
        return cands

    def _hoist_graph(self, names):
        """Hoist the base UNET **and** the LoRA-patched model, once per beat.

        BOTH are patchers and both must be tracked: ``_detach_patchers`` walks
        ``prepared["patchers"]`` and detaches what it finds there, so an untracked
        patcher is VRAM nothing will ever reclaim. The parent's ``on_result``
        registration fires per node as each lands, which is what makes this safe
        when a later node raises -- ``run_graph`` never returns a results dict in
        that case, and a handle registered only afterwards would leak."""
        graph = super()._hoist_graph(names)
        graph["lora"] = {
            "class": "lora",
            "inputs": {"model": _wb.Wire("unet", 0),
                       "lora_name": names["lora"],
                       "strength_model": float(
                           self._resolve_render_config()["lora_strength"])},
        }
        return graph

    def _build_graph(self, request, image_name, plan, length, width, height,
                     external_results=None):
        """The incumbent's graph with the KSampler node swapped for the chain
        ``OTR_DMDRestartSamplerSelect`` + ``ManualSigmas`` + ``SamplerCustom``.

        THE SEED TRAP, pinned by test: ``KSampler`` takes ``seed``;
        ``SamplerCustom`` takes ``noise_seed``. Copying the incumbent's input dict
        would land ``seed`` on a node that does not read it, and every clip would
        render on SamplerCustom's own default -- no error, no log line, just a beat
        whose segments do not vary the way the plan says. The repo already learned
        this once: ``run_video_arm_bakeoff.py`` carries a hand-written dual-key
        reader for exactly this split."""
        graph = super()._build_graph(request, image_name, plan, length,
                                     width, height,
                                     external_results=external_results)
        cfg_knobs = self._resolve_render_config()
        base = graph.pop("ksampler")         # KeyError here = the parent moved
        graph["sigmas"] = {"class": "sigmas",
                           "inputs": {"sigmas": str(cfg_knobs["sigmas"])}}
        graph["dmdsampler"] = {"class": "dmdsampler", "inputs": {}}
        graph["sampler"] = {
            "class": "sampler",
            "inputs": {
                "model": base["inputs"]["model"],       # ModelSamplingSD3 output
                "positive": base["inputs"]["positive"],
                "negative": base["inputs"]["negative"],
                "latent_image": base["inputs"]["latent_image"],
                "add_noise": True,
                "noise_seed": int(plan.get("seed", 0)),   # NOT "seed"
                "cfg": float(cfg_knobs["cfg"]),
                "sampler": _wb.Wire("dmdsampler", 0),
                "sigmas": _wb.Wire("sigmas", 0),
            },
        }
        for nid in set(external_results or ()):
            graph.pop(nid, None)
        return graph

    # ---- recipe ---------------------------------------------------------- #
    def _resolve_render_config(self):
        """The incumbent's five knobs plus this engine's two.

        ``sigmas`` and ``lora_strength`` have no incumbent equivalent, so they are
        resolved here through the same consent act rather than read raw -- a sweep
        that opens the knobs must measure the recipe it is validating."""
        cfg = super()._resolve_render_config()
        frozen_sigmas = str(self.recipe_data["sigmas"])
        frozen_strength = float(self.recipe_data["lora_strength"])
        if not _WR.prequalification_active(self.prequalification_env):
            cfg["sigmas"] = frozen_sigmas
            cfg["lora_strength"] = frozen_strength
            return cfg
        cfg["sigmas"] = _WR.config_text(
            self, self.recipe_env_keys["sigmas"], frozen_sigmas)
        cfg["lora_strength"] = _WR.config_number(
            self, self.recipe_env_keys["lora_strength"], frozen_strength,
            0.0, 2.0, float)
        return cfg


# The 4n+1 ladder and the VRAM cost row. The floor is REQUIRED: an engine with no
# row gets _DEFAULT_MOTION_FLOOR = 1, which would let a length below the 5B VAE's
# 4-frame quantum reach the model. The cost row is byte-identical to the default
# today and buys only explicitness -- it is NOT a refit, and the standing ruling
# against refitting from the bench holds (the bench drives stock nodes, so the
# admission estimator never runs there).
_MC.FRAME_MOTION_FLOOR["fastwan_8gb"] = 17
_MC.FRAME_COST_MODEL["fastwan_8gb"] = (7000.0, 185.0)

__all__ = [
    "FastWan8gbEngine", "RECIPE_FASTWAN_8GB", "PREQUALIFICATION_ENV_FASTWAN",
    "FASTWAN_8GB_RECIPE", "FASTWAN_SIGMAS", "FASTWAN_LORA_NAME",
]
