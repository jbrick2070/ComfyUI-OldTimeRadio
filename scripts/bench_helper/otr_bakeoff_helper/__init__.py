"""otr_bakeoff_helper -- a SIBLING ComfyUI custom-node package for the HuMo
STANDALONE VRAM bakeoff (roundtables/2026-06-27-humo-optim/final.md).

DIAGNOSTIC ONLY. This package is a sibling of ComfyUI-OldTimeRadio (both live
under ``...\\ComfyUI\\custom_nodes\\``); it is auto-loaded by ComfyUI, edits
NOTHING in the OTR pack, and is removable. It registers ONE node:

  OTR_BakeoffReclaim -- a LATENT passthrough that, when it executes, evicts ONLY
  the umt5 text-encoder (CLIP) + the whisper audio-encoder model weights off the
  GPU, while KEEPING the diffusion UNET / its LoRA+ModelSamplingSD3 patches and
  the VAE resident. Spliced on the HuMo latent edge (WanHuMoImageToVideo output
  slot 2 -> KSampler.latent_image) so topological order forces the encoder
  eviction AFTER conditioning + ref-image encode and BEFORE the heavy 14B
  sampler -- the "two-stage" fit lever, measured without touching production.

Why NOT wrapper_bridge.reclaim_idle_models: that detaches EVERY currently-loaded
model (incl. the 14B UNET); here we must keep the sampler's UNET + VAE resident
and drop only the ~5 GB encoder block. So this node classifies each resident
model and detaches ONLY text/audio encoders -- never a diffusion UNET or a VAE.

IS_CHANGED is always-dirty so the executor never cache-skips the node (it MUST
run every prompt, including the same-session sentinel leg). Each call prints a
unique marker so the headless runner can assert the eviction actually ran.

UTF-8, no BOM. ASCII-only source. SFW.
"""
from __future__ import annotations

import logging
import uuid

_LOG = logging.getLogger("OTR.bakeoff.reclaim")

# Keyword buckets for classifying a resident ComfyUI model by its inner module /
# class name. Checked in THIS order (encoders first) so a text encoder whose
# patcher is a generic ModelPatcher is never mis-bucketed as the UNET.
_AUDIO_ENC_KEYS = ("whisper", "audioencoder", "audio_encoder", "audio_enc")
_TEXT_ENC_KEYS = ("umt5", "t5xxl", "t5", "clip", "textencoder", "text_encoder",
                  "cond_stage", "sd1_clip", "sdxl_clip", "te_model", "gemma",
                  "llama", "byt5", "pile_t5")
_VAE_KEYS = ("vae", "autoencod", "first_stage")
_UNET_KEYS = ("basemodel", "model_base", "diffusion", "unet", "wan21",
              "wanmodel", "humo", "ltxv", "flux", "sd3", "wan_model")


def _model_haystack(lm):
    """Lower-cased name blob for a current_loaded_models entry: its patcher class,
    the INNER model's class + module, and the wrapper class. Pure; no comfy import."""
    patcher = getattr(lm, "model", lm)
    inner = getattr(patcher, "model", None)
    parts = [type(lm).__name__, type(patcher).__name__]
    if inner is not None:
        parts.append(type(inner).__name__)
        parts.append(getattr(type(inner), "__module__", "") or "")
    return " ".join(p for p in parts if p).lower()


def classify_loaded_model(lm):
    """Bucket a resident model: 'audio_encoder' | 'text_encoder' | 'vae' | 'unet'
    | 'other'. Encoders are matched FIRST (the inner class name carries umt5 / t5 /
    whisper) so the generic ModelPatcher of a CLIP is never read as the UNET. Pure;
    CPU-testable with stand-in objects."""
    hay = _model_haystack(lm)
    if any(k in hay for k in _AUDIO_ENC_KEYS):
        return "audio_encoder"
    if any(k in hay for k in _TEXT_ENC_KEYS):
        return "text_encoder"
    if any(k in hay for k in _VAE_KEYS):
        return "vae"
    if any(k in hay for k in _UNET_KEYS):
        return "unet"
    return "other"


def _detach_patcher(lm):
    """Move ONE resident model's weights to the offload (CPU) device, the BUG-291
    detach the OTR engines use (wrapper_bridge.reclaim_idle_models / _otr_vram_levers
    step 3a). Returns True if a detach/move happened. Best-effort; never raises."""
    patcher = getattr(lm, "model", lm)
    det = getattr(patcher, "detach", None)
    if callable(det):
        try:
            det(unpatch_all=True)
            return True
        except TypeError:
            try:
                det()
                return True
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass
    inner = getattr(patcher, "model", None)          # fallback: weights -> CPU
    if inner is not None and hasattr(inner, "to"):
        try:
            inner.to("cpu")
            return True
        except Exception:  # noqa: BLE001
            pass
    return False


def evict_encoders_only(loaded, detach_fn=_detach_patcher,
                        classify_fn=classify_loaded_model):
    """Detach ONLY the text/audio-encoder models in ``loaded`` (a
    current_loaded_models list), KEEPING every UNET / VAE / other resident.

    Returns a report dict: counts per bucket, the evicted/kept buckets, and a
    fail-closed ``sampler_survived`` flag (False iff a UNET or VAE was detached --
    which never happens by construction; the flag is the asserted invariant). Pure
    + dependency-injected so it is CPU-testable with stand-in objects."""
    evicted_kinds, kept_kinds = [], []
    evicted = 0
    for lm in loaded:
        kind = classify_fn(lm)
        if kind in ("text_encoder", "audio_encoder"):
            if detach_fn(lm):
                evicted += 1
            evicted_kinds.append(kind)
        else:
            kept_kinds.append(kind)
    sampler_survived = not any(k in ("unet", "vae") for k in evicted_kinds)
    return {
        "total": len(loaded),
        "evicted": evicted,
        "evicted_kinds": evicted_kinds,
        "kept_kinds": kept_kinds,
        "n_unet_kept": kept_kinds.count("unet"),
        "n_vae_kept": kept_kinds.count("vae"),
        "n_text_evicted": evicted_kinds.count("text_encoder"),
        "n_audio_evicted": evicted_kinds.count("audio_encoder"),
        "sampler_survived": sampler_survived,
    }


def _current_loaded_models():
    """ComfyUI's resident-model list (lazy import; None off the ComfyUI box)."""
    try:
        import comfy.model_management as _mm  # type: ignore
    except Exception:  # noqa: BLE001 -- not inside ComfyUI -> nothing resident
        return None
    try:
        return list(getattr(_mm, "current_loaded_models", None) or [])
    except Exception:  # noqa: BLE001
        return []


def _soft_free():
    """GC + ComfyUI soft cache empty so a VRAM probe reflects the encoder drop."""
    try:
        import gc
        gc.collect()
        import comfy.model_management as _mm  # type: ignore
        _mm.soft_empty_cache()
    except Exception:  # noqa: BLE001
        pass


class OTR_BakeoffReclaim:
    """LATENT passthrough that evicts the umt5 CLIP + whisper encoders mid-graph.

    Splice on the HuMo latent edge: WanHuMoImageToVideo (slot 2) -> this node ->
    KSampler.latent_image. Topo order then runs it AFTER the ref-image encode and
    BEFORE the sampler, so the ~5 GB encoder block is off the GPU during the heavy
    14B forward. Keeps the UNET + VAE resident (asserts it). Always-dirty."""

    CATEGORY = "OTR/bakeoff"
    FUNCTION = "reclaim"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"samples": ("LATENT",)},
            "optional": {"reason": ("STRING",
                                    {"default": "humo bakeoff pre-sampler"})},
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Always-dirty: the executor must NEVER cache-skip this node (it has to run
        # every prompt, including the same-session sentinel leg). A fresh uuid per
        # call is unique regardless of clock resolution (time.time() can repeat).
        return uuid.uuid4().hex

    def reclaim(self, samples, reason="humo bakeoff pre-sampler"):
        marker = uuid.uuid4().hex[:12]
        loaded = _current_loaded_models()
        if loaded is None:
            # Off the ComfyUI box (no comfy / no torch): pure passthrough.
            msg = ("[OTR_BakeoffReclaim] marker=%s NO-COMFY passthrough (%s)"
                   % (marker, reason))
            print(msg, flush=True)
            _LOG.warning(msg)
            return (samples,)
        report = evict_encoders_only(loaded)
        _soft_free()
        if not report["sampler_survived"]:
            raise RuntimeError(
                "[OTR_BakeoffReclaim] marker=%s ABORT -- a UNET/VAE was detached "
                "(evicted_kinds=%s); the sampler model did NOT survive"
                % (marker, report["evicted_kinds"]))
        msg = ("[OTR_BakeoffReclaim] ENCODER-EVICT marker=%s reason=%r resident=%d "
               "evicted=%d (text=%d audio=%d) kept unet=%d vae=%d other=%d "
               "sampler_survived=True"
               % (marker, reason, report["total"], report["evicted"],
                  report["n_text_evicted"], report["n_audio_evicted"],
                  report["n_unet_kept"], report["n_vae_kept"],
                  report["kept_kinds"].count("other")))
        print(msg, flush=True)
        _LOG.warning(msg)
        return (samples,)


class OTR_BakeoffVramReset:
    """LATENT passthrough that resets the CUDA peak-memory counter RIGHT BEFORE the
    sampler, so OTR_BakeoffVramProbe reads the TRUE sampler+decode peak
    (max_memory_allocated SINCE this reset), not a cumulative figure. Splice on the
    latent edge AFTER WanHuMoImageToVideo / OTR_BakeoffReclaim and BEFORE KSampler.
    Lazy torch import (cold-import clean); always-dirty IS_CHANGED."""

    CATEGORY = "OTR/bakeoff"
    FUNCTION = "reset"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",)}}

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return uuid.uuid4().hex

    def reset(self, samples):
        marker = uuid.uuid4().hex[:12]
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                msg = ("[OTR_BakeoffVramReset] marker=%s reset_peak_memory_stats OK"
                       % marker)
            else:
                msg = "[OTR_BakeoffVramReset] marker=%s NO-CUDA passthrough" % marker
        except Exception as e:  # noqa: BLE001
            msg = "[OTR_BakeoffVramReset] marker=%s no-torch (%r)" % (marker, e)
        print(msg, flush=True)
        _LOG.warning(msg)
        return (samples,)


class OTR_BakeoffVramProbe:
    """IMAGE passthrough that logs the TRUE CUDA peak since the last reset
    (max_memory_allocated) + the peak reserved pool (max_memory_reserved), in MB.
    Splice on the image edge AFTER VAEDecode and BEFORE SaveImage. The gap between
    allocated (true demand) and reserved/NVML (cache) is the whole question for HuMo
    fit. Lazy torch import; always-dirty IS_CHANGED."""

    CATEGORY = "OTR/bakeoff"
    FUNCTION = "probe"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return uuid.uuid4().hex

    def probe(self, images):
        marker = uuid.uuid4().hex[:12]
        try:
            import torch
            if torch.cuda.is_available():
                mb = 1024.0 * 1024.0
                alloc = torch.cuda.max_memory_allocated() / mb
                resv = torch.cuda.max_memory_reserved() / mb
                msg = ("[OTR_BakeoffVramProbe] marker=%s max_allocated_mb=%.1f "
                       "max_reserved_mb=%.1f" % (marker, alloc, resv))
            else:
                msg = "[OTR_BakeoffVramProbe] marker=%s NO-CUDA passthrough" % marker
        except Exception as e:  # noqa: BLE001
            msg = "[OTR_BakeoffVramProbe] marker=%s no-torch (%r)" % (marker, e)
        print(msg, flush=True)
        _LOG.warning(msg)
        return (images,)


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
    previous x, reaching ``x = denoised`` only at the terminal sigma. Feeding
    the right sigma COORDINATES to the wrong TRANSITION renders something
    plausible and reports a VRAM number for a recipe nobody trained -- exactly
    the failure this bench forbids.

    The re-noise uses the model's OWN ``model_sampling.noise_scaling`` so the
    parameterization can never drift from the loaded model. For a
    rectified-flow CONST sampling that is ``sigma*noise + (1-sigma)*x0``, which
    is what a flow scheduler's ``add_noise`` computes.
    """
    from tqdm.auto import trange

    from comfy.k_diffusion.sampling import default_noise_sampler

    extra_args = {} if extra_args is None else extra_args
    if noise_sampler is None:
        noise_sampler = default_noise_sampler(x, seed=extra_args.get("seed"))

    model_sampling = None
    try:
        model_sampling = model.inner_model.model_patcher.get_model_object(
            "model_sampling")
    except Exception:  # noqa: BLE001
        model_sampling = None

    marker = uuid.uuid4().hex[:12]
    steps = [float(s) for s in sigmas.tolist()]
    msg = ("[OTR_DMDRestart] marker=%s transition=restart(predict_x0->renoise_"
           "fresh) sigmas=%s timesteps=%s noise_scaling=%s"
           % (marker,
              ",".join("%.6g" % s for s in steps),
              ",".join("%.6g" % (s * 1000.0) for s in steps),
              type(model_sampling).__name__ if model_sampling is not None
              else "FALLBACK-flow"))
    print(msg, flush=True)
    _LOG.warning(msg)

    s_in = x.new_ones([x.shape[0]])
    n = len(sigmas) - 1
    # trange, exactly as the stock k_diffusion samplers do, for two reasons that
    # are not cosmetic: the harness parses ``s/it`` out of this bar (a plain
    # loop silently degrades ``s_per_it`` to the wall/steps FALLBACK, which is
    # not comparable against a log-parsed figure from another arm), and
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
        if model_sampling is not None:
            x = model_sampling.noise_scaling(s_next, noise, denoised)
        else:
            # flow fallback, identical to CONST.noise_scaling
            sn = s_next.view(s_next.shape[:1] + (1,) * (x.ndim - 1)) \
                if s_next.nelement() > 1 else s_next.view(())
            x = sn * noise + (1.0 - sn) * denoised
    return x


class OTR_DMDRestartSamplerSelect:
    """Emit the DMD/Self-Forcing restart transition as a SAMPLER.

    Drop-in for ``KSamplerSelect`` on the ``SamplerCustom.sampler`` input. Pair
    it with ``ManualSigmas`` carrying the reference denoising_step_list divided
    by 1000. The node takes no widgets on purpose: the schedule lives in the
    sigmas and the guidance lives in ``SamplerCustom.cfg``, so there is exactly
    one owner for each and nothing here to drift."""

    CATEGORY = "OTR/bakeoff"
    FUNCTION = "get_sampler"
    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("SAMPLER",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def get_sampler(self):
        import comfy.samplers
        return (comfy.samplers.KSAMPLER(_dmd_restart_sampler),)


NODE_CLASS_MAPPINGS = {
    "OTR_BakeoffReclaim": OTR_BakeoffReclaim,
    "OTR_BakeoffVramReset": OTR_BakeoffVramReset,
    "OTR_BakeoffVramProbe": OTR_BakeoffVramProbe,
    "OTR_DMDRestartSamplerSelect": OTR_DMDRestartSamplerSelect,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OTR_BakeoffReclaim": "OTR Bakeoff Reclaim (encoder evict)",
    "OTR_BakeoffVramReset": "OTR Bakeoff VRAM Reset (peak stats)",
    "OTR_BakeoffVramProbe": "OTR Bakeoff VRAM Probe (true peak)",
    "OTR_DMDRestartSamplerSelect": "OTR DMD Restart Sampler (predict-x0 + renoise)",
}

__all__ = [
    "OTR_BakeoffReclaim", "OTR_BakeoffVramReset", "OTR_BakeoffVramProbe",
    "OTR_DMDRestartSamplerSelect",
    "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
    "classify_loaded_model", "evict_encoders_only",
]
