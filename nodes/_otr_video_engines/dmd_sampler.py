"""The DMD / Self-Forcing restart transition, as a registered ComfyUI SAMPLER.

THE ONE OWNER (kibitz r2 F2 / r4). This transition used to live in the diagnostic
bench helper (``scripts/bench_helper/otr_bakeoff_helper``), which is vendored under
the CLAUDE.md s0A bench carve-out and installed into ``custom_nodes`` by the bench
runner. A PRODUCTION engine may not depend on a diagnostic package, and two copies
of a sampling transition is exactly how a silent divergence ships -- so the
implementation MOVED here and the bench helper imports it. Its own class and its
``NODE_CLASS_MAPPINGS`` entry are deleted; a duplicate registration under the same
node key is ambiguous, and a helper-only registration is unavailable in production.

WHY A REGISTERED NODE rather than an in-process ``KSAMPLER`` object: the bench
submits an API-format JSON graph to real ComfyUI, which resolves by REGISTERED CLASS
NAME. ``wrapper_bridge._resolve_value`` would happily pass an in-process object
through as a literal, so both are mechanically possible -- but one registered node
consumed both ways means the pinned transition has exactly ONE implementation and
the recipe contract is enforceable on both paths.

NOT AN ODE MARCH. At every non-terminal step the reference implementations predict
x0 and then RE-NOISE that x0 to the NEXT timestep with FRESH noise and zero
carry-over of the previous latent (FastVideo ``DmdDenoisingStage.forward``).
``sample_euler`` is deterministic and ``sample_euler_ancestral_RF`` retains a
fraction of the previous latent, so neither expresses this.

Heavy imports (``comfy``, torch, tqdm) are LAZY -- this module is imported at pack
load time and ``test_cold_import_no_heavy_libs`` fails on a module-scope torch.
"""

import logging
import uuid

_LOG = logging.getLogger("OTR.dmd_sampler")

#: Node key. Unchanged from the bench helper's, deliberately: the pinned bench
#: graphs (``scripts/bench_graphs/arm_c_fastwan_lora_gguf.json``) name this class
#: and are digest-pinned in the campaign manifest, so renaming it would invalidate
#: the very cells that qualified the recipe.
DMD_NODE_KEY = "OTR_DMDRestartSamplerSelect"


class DmdModelSamplingError(RuntimeError):
    """The loaded model exposes no usable ``model_sampling``.

    NO FALLBACK (kibitz r2 F2b). The bench copy used to log
    ``noise_scaling=FALLBACK-flow`` and then PROCEED with an inline CONST
    approximation -- loud, but it still rendered, and the bench would grade that
    render as a PASS for a recipe whose transition was APPROXIMATED rather than
    taken from the loaded model. A receipt that says ``dmd3`` must mean the model's
    own noise scaling ran, so this refuses before step 1 instead."""


def _dmd_restart_sampler(model, x, sigmas, extra_args=None, callback=None,
                         disable=None, noise_sampler=None, **kwargs):
    """The DMD / Self-Forcing multi-step transition, as a ComfyUI SAMPLER.

    NOT an ODE march. At every non-terminal step the reference implementations
    predict x0 and then RE-NOISE x0 to the NEXT timestep with FRESH noise, with
    zero carry-over of the previous latent:

        FastVideo DmdDenoisingStage.forward / quanhaol
        pipeline/wan22_fewstep_inference.py / guandeh17 Self-Forcing
        pipeline/causal_inference.py:
            pred = generator(x, t)
            x = scheduler.add_noise(pred, torch.randn_like(pred), t_next)

    No stock ComfyUI sampler does this. ``sample_euler`` is deterministic
    (``x = x + d*dt``; its noise injection is behind ``s_churn > 0``, default 0)
    and ``sample_euler_ancestral_RF`` retains ``sigma_down/sigma_i`` of the
    previous x, reaching ``x = denoised`` only at the terminal sigma. Feeding the
    right sigma COORDINATES to the wrong TRANSITION renders something plausible
    and reports a VRAM number for a recipe nobody trained.

    The re-noise uses the model's OWN ``model_sampling.noise_scaling`` so the
    parameterization can never drift from the loaded model. For a rectified-flow
    CONST sampling that is ``sigma*noise + (1-sigma)*x0``, which is what a flow
    scheduler's ``add_noise`` computes -- but it is READ FROM THE MODEL, never
    reimplemented here, and its absence is a refusal (see
    :class:`DmdModelSamplingError`) rather than an inline approximation.
    """
    from tqdm.auto import trange

    from comfy.k_diffusion.sampling import default_noise_sampler

    extra_args = {} if extra_args is None else extra_args
    if noise_sampler is None:
        noise_sampler = default_noise_sampler(x, seed=extra_args.get("seed"))

    # REFUSE BEFORE STEP 1, not mid-trajectory: the transition is defined by the
    # model's own noise scaling, and with only three steps a single approximated
    # re-noise is a third of the whole trajectory.
    try:
        model_sampling = model.inner_model.model_patcher.get_model_object(
            "model_sampling")
    except Exception as exc:  # noqa: BLE001 -- re-raised NAMED below
        raise DmdModelSamplingError(
            "OTR_DMDRestartSamplerSelect could not resolve 'model_sampling' from "
            "the loaded model (%r). The DMD restart transition re-noises x0 to the "
            "next timestep THROUGH the model's own noise_scaling -- there is no "
            "faithful substitute. NO FALLBACK: an approximated transition wearing a "
            "dmd3 receipt is a false receipt. Check the model loaded and its "
            "patcher chain." % (exc,)) from exc
    if model_sampling is None or not callable(
            getattr(model_sampling, "noise_scaling", None)):
        raise DmdModelSamplingError(
            "OTR_DMDRestartSamplerSelect resolved model_sampling=%r, which exposes "
            "no callable noise_scaling(). The DMD restart transition cannot run "
            "without it. NO FALLBACK -- see this class's docstring."
            % (model_sampling,))

    marker = uuid.uuid4().hex[:12]
    steps = [float(s) for s in sigmas.tolist()]
    msg = ("[OTR_DMDRestart] marker=%s transition=restart(predict_x0->renoise_"
           "fresh) sigmas=%s timesteps=%s noise_scaling=%s"
           % (marker,
              ",".join("%.6g" % s for s in steps),
              ",".join("%.6g" % (s * 1000.0) for s in steps),
              type(model_sampling).__name__))
    print(msg, flush=True)
    _LOG.warning(msg)

    s_in = x.new_ones([x.shape[0]])
    n = len(sigmas) - 1
    # trange, exactly as the stock k_diffusion samplers do, for two reasons that
    # are not cosmetic: the harness parses ``s/it`` out of this bar (a plain loop
    # silently degrades ``s_per_it`` to the wall/steps FALLBACK, which is not
    # comparable against a log-parsed figure from another arm), and
    # otr_render_watchdog.ps1 treats absent progress as a heartbeat stall.
    for i in trange(n, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i],
                      "sigma_hat": sigmas[i], "denoised": denoised})
        s_next = sigmas[i + 1]
        if float(s_next) <= 0.0:
            # terminal step: the x0 prediction IS the result, never re-noised
            x = denoised
            continue
        noise = noise_sampler(sigmas[i], s_next)
        x = model_sampling.noise_scaling(s_next, noise, denoised)
    return x


class OTR_DMDRestartSamplerSelect:
    """Emit the DMD/Self-Forcing restart transition as a SAMPLER.

    Drop-in for ``KSamplerSelect`` on the ``SamplerCustom.sampler`` input. Pair it
    with ``ManualSigmas`` carrying the reference denoising_step_list divided by
    1000 -- note ``ManualSigmas`` takes a COMMA-SEPARATED STRING
    (``"1.0, 0.757, 0.522, 0.0"``), not a tuple. The node takes no widgets on
    purpose: the schedule lives in the sigmas and the guidance lives in
    ``SamplerCustom.cfg``, so there is exactly one owner for each and nothing here
    to drift."""

    CATEGORY = "OTR/video"
    FUNCTION = "get_sampler"
    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("SAMPLER",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def get_sampler(self):
        import comfy.samplers
        return (comfy.samplers.KSAMPLER(_dmd_restart_sampler),)


NODE_CLASS_MAPPINGS = {DMD_NODE_KEY: OTR_DMDRestartSamplerSelect}
NODE_DISPLAY_NAME_MAPPINGS = {
    DMD_NODE_KEY: "OTR DMD Restart Sampler (predict-x0 + renoise)",
}

__all__ = [
    "DMD_NODE_KEY", "DmdModelSamplingError", "OTR_DMDRestartSamplerSelect",
    "_dmd_restart_sampler",
    "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
]
