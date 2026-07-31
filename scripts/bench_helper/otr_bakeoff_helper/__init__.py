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


NODE_CLASS_MAPPINGS = {
    "OTR_BakeoffReclaim": OTR_BakeoffReclaim,
    "OTR_BakeoffVramReset": OTR_BakeoffVramReset,
    "OTR_BakeoffVramProbe": OTR_BakeoffVramProbe,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OTR_BakeoffReclaim": "OTR Bakeoff Reclaim (encoder evict)",
    "OTR_BakeoffVramReset": "OTR Bakeoff VRAM Reset (peak stats)",
    "OTR_BakeoffVramProbe": "OTR Bakeoff VRAM Probe (true peak)",
}

__all__ = [
    "OTR_BakeoffReclaim", "OTR_BakeoffVramReset", "OTR_BakeoffVramProbe",
    "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
    "classify_loaded_model", "evict_encoders_only",
]
