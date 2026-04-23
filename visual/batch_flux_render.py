"""
batch_flux_render.py  --  OTR_BatchFluxRender ComfyUI node
===========================================================
Render N FLUX images in lockstep from the polished script JSON,
in a single graph run, sharing one MODEL/CLIP/VAE load.

2026-04-23 additions:
- fast_batch widget (default True): stack N prompts into one
  batched CONDITIONING and fire ONE KSampler call for the whole
  batch. Saves N-1 model-prep cycles, amortizes attention.
- comfy.model_management.load_models_gpu([model]) pre-pin at the
  start of execute() so per-KSampler load_models_gpu calls are
  cheap no-ops (reduces "Model Flux prepared" log chatter).
- freeze_seed still applies in serial mode only; fast_batch shares
  base seed across shots (ComfyUI's sampler varies noise per
  batch index internally).
- fast_batch falls through to serial on any stacking/shape error.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

log = logging.getLogger("OTR.visual.batch_flux_render")


_DEFAULT_FALLBACK = (
    "cinematic 35mm film still, dimly lit starship bridge, red alert "
    "lighting, holographic console glow, tense crew silhouettes, "
    "volumetric haze, shallow depth of field"
)

_DEFAULT_STYLE_SUFFIX = (
    "cinematic, 35mm film, anamorphic lens, volumetric lighting, "
    "heavy vignette, muted color grade, sharp focus"
)


def _lazy_nodes():
    refs: dict[str, Any] = {}
    import nodes  # type: ignore
    refs["CLIPTextEncode"] = getattr(nodes, "CLIPTextEncode", None)
    refs["KSampler"] = getattr(nodes, "KSampler", None)
    refs["VAEDecode"] = getattr(nodes, "VAEDecode", None)
    refs["EmptyLatentImage"] = getattr(nodes, "EmptyLatentImage", None)
    empty_sd3 = None
    flux_guidance = None
    try:
        from comfy_extras.nodes_sd3 import EmptySD3LatentImage  # type: ignore
        empty_sd3 = EmptySD3LatentImage
    except Exception:
        pass
    try:
        from comfy_extras.nodes_flux import FluxGuidance  # type: ignore
        flux_guidance = FluxGuidance
    except Exception:
        try:
            from comfy_extras.nodes_model_advanced import FluxGuidance  # type: ignore
            flux_guidance = FluxGuidance
        except Exception:
            pass
    refs["EmptySD3LatentImage"] = empty_sd3
    refs["FluxGuidance"] = flux_guidance
    return refs


def _parse_env_prompts(script_json, batch_limit, fallback, style_suffix):
    if not script_json or not script_json.strip():
        log.info("[BatchFluxRender] empty script_json; using fallback x1")
        return [f"{fallback}, {style_suffix}".rstrip(", ")]
    try:
        payload = json.loads(script_json)
    except json.JSONDecodeError as exc:
        log.warning("[BatchFluxRender] script_json JSONDecodeError (%s)", exc)
        return [f"{fallback}, {style_suffix}".rstrip(", ")]
    if isinstance(payload, dict):
        tokens = payload.get("tokens") or payload.get("script") or []
    elif isinstance(payload, list):
        tokens = payload
    else:
        log.warning("[BatchFluxRender] unexpected script_json root %s", type(payload).__name__)
        return [f"{fallback}, {style_suffix}".rstrip(", ")]
    env_tokens = [t for t in tokens if isinstance(t, dict) and t.get("type") == "environment"]
    if not env_tokens:
        log.info("[BatchFluxRender] no environment tokens (total=%d); using fallback x1", len(tokens))
        return [f"{fallback}, {style_suffix}".rstrip(", ")]
    limit = max(1, min(batch_limit, len(env_tokens)))
    selected = env_tokens[:limit]
    prompts = []
    for token in selected:
        desc = (token.get("description") or "").strip()
        if not desc:
            desc = fallback
        parts = [desc]
        if style_suffix and style_suffix.strip():
            parts.append(style_suffix.strip())
        prompts.append(", ".join(parts))
    log.info("[BatchFluxRender] queued %d env prompt(s) from %d available", len(prompts), len(env_tokens))
    return prompts


def _stack_conditioning(cond_list):
    import torch  # type: ignore
    if not cond_list:
        return None
    try:
        main_tensors = [c[0][0] for c in cond_list]
        ref_shape = main_tensors[0].shape
        if not all(t.shape == ref_shape for t in main_tensors):
            log.warning("[BatchFluxRender] main cond tensor shape mismatch; cannot fast-batch")
            return None
        stacked_main = torch.cat(main_tensors, dim=0)
        pooled_list = []
        for c in cond_list:
            meta = c[0][1]
            if not isinstance(meta, dict) or "pooled_output" not in meta:
                log.warning("[BatchFluxRender] cond missing pooled_output; cannot fast-batch")
                return None
            pooled_list.append(meta["pooled_output"])
        stacked_pooled = torch.cat(pooled_list, dim=0)
        merged_meta = dict(cond_list[0][0][1])
        merged_meta["pooled_output"] = stacked_pooled
        return [[stacked_main, merged_meta]]
    except Exception as exc:
        log.warning("[BatchFluxRender] _stack_conditioning failed: %s", exc)
        return None


def _expand_conditioning(cond, batch_size):
    try:
        main = cond[0][0]
        if main.shape[0] != 1:
            return None
        main_N = main.expand(batch_size, *main.shape[1:]).contiguous()
        meta = dict(cond[0][1])
        pooled = meta.get("pooled_output")
        if pooled is None or pooled.shape[0] != 1:
            return None
        pooled_N = pooled.expand(batch_size, *pooled.shape[1:]).contiguous()
        meta["pooled_output"] = pooled_N
        return [[main_N, meta]]
    except Exception as exc:
        log.warning("[BatchFluxRender] _expand_conditioning failed: %s", exc)
        return None


def _try_fast_batch(*, prompts, model, clip, vae, text_enc, guidance_node,
                    empty_latent_cls, sampler, decoder, negative, seed, steps,
                    cfg, sampler_name, scheduler, width, height, guidance,
                    report_lines):
    t0 = time.time()
    N = len(prompts)
    try:
        log.info("[BatchFluxRender] fast_batch: encoding %d prompt(s) before single KSampler call", N)
        raw_conds = []
        for i, pt in enumerate(prompts):
            preview = pt[:120] + ("..." if len(pt) > 120 else "")
            log.info("[BatchFluxRender] shot %d/%d seed=%d: %s", i + 1, N, seed + i, preview)
            raw_conds.append(text_enc.encode(clip, pt)[0])
    except Exception as exc:
        log.warning("[BatchFluxRender] fast_batch encode failed: %s", exc)
        return None
    batched_pos = _stack_conditioning(raw_conds)
    if batched_pos is None:
        return None
    if guidance_node is not None:
        try:
            batched_pos = guidance_node.append(batched_pos, guidance)[0]
        except Exception as exc:
            log.warning("[BatchFluxRender] fast_batch FluxGuidance failed: %s", exc)
            return None
    batched_neg = _expand_conditioning(negative, N)
    if batched_neg is None:
        log.warning("[BatchFluxRender] could not expand negative to batch=%d", N)
        return None
    try:
        latent = empty_latent_cls.generate(width, height, N)[0]
    except Exception as exc:
        log.warning("[BatchFluxRender] EmptyLatent batch=%d failed: %s", N, exc)
        return None
    try:
        log.info("[BatchFluxRender] fast_batch: one KSampler call for %d shot(s) @ steps=%d cfg=%.2f sampler=%s/%s",
                 N, steps, cfg, sampler_name, scheduler)
        samples = sampler.sample(model, seed, steps, cfg, sampler_name, scheduler,
                                 batched_pos, batched_neg, latent, 1.0)[0]
    except Exception as exc:
        log.exception("[BatchFluxRender] fast_batch KSampler failed: %s", exc)
        report_lines.append(f"  fast_batch KSampler error: {exc}")
        return None
    try:
        image_batch = decoder.decode(vae, samples)[0]
    except Exception as exc:
        log.exception("[BatchFluxRender] fast_batch VAEDecode failed: %s", exc)
        report_lines.append(f"  fast_batch VAEDecode error: {exc}")
        return None
    total_ms = int((time.time() - t0) * 1000)
    avg = total_ms // max(N, 1)
    for i in range(N):
        report_lines.append(f"  shot {i + 1}: batched ({avg} ms/shot avg)")
        log.info("[BatchFluxRender] shot %d done in %d ms (batched avg)", i + 1, avg)
    return image_batch, total_ms


class BatchFluxRender:
    CATEGORY = "OTR/v2/Visual"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers  # type: ignore
            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
        except Exception:
            samplers = ["euler"]
            schedulers = ["simple"]
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "script_json": ("STRING", {"multiline": True, "default": ""}),
                "batch_limit": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (samplers, {"default": "euler"}),
                "scheduler": (schedulers, {"default": "simple"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 8}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
            },
            "optional": {
                "fallback_prompt": ("STRING", {"multiline": True, "default": _DEFAULT_FALLBACK}),
                "style_suffix": ("STRING", {"multiline": False, "default": _DEFAULT_STYLE_SUFFIX}),
                "freeze_seed": ("BOOLEAN", {"default": False}),
                "fast_batch": ("BOOLEAN", {"default": True}),
            },
        }

    def execute(self, model, clip, vae, script_json, batch_limit, seed, steps,
                cfg, sampler_name, scheduler, width, height, guidance,
                fallback_prompt=_DEFAULT_FALLBACK, style_suffix=_DEFAULT_STYLE_SUFFIX,
                freeze_seed=False, fast_batch=True):
        t_start = time.time()
        prompts = _parse_env_prompts(script_json, batch_limit, fallback_prompt, style_suffix)

        # Pin MODEL on GPU so per-KSampler load_models_gpu calls are cheap.
        try:
            import comfy.model_management as mm  # type: ignore
            try:
                mm.load_models_gpu([model], force_full_load=True)
            except TypeError:
                mm.load_models_gpu([model])
            log.info("[BatchFluxRender] pinned MODEL via load_models_gpu")
        except Exception as exc:
            log.debug("[BatchFluxRender] pre-pin skipped: %s", exc)

        refs = _lazy_nodes()
        CLIPTextEncode = refs["CLIPTextEncode"]
        KSampler = refs["KSampler"]
        VAEDecode = refs["VAEDecode"]
        EmptySD3 = refs["EmptySD3LatentImage"]
        EmptyBasic = refs["EmptyLatentImage"]
        FluxGuidance = refs["FluxGuidance"]

        if CLIPTextEncode is None or KSampler is None or VAEDecode is None:
            raise RuntimeError("BatchFluxRender: ComfyUI nodes module missing CLIPTextEncode / KSampler / VAEDecode")
        if EmptySD3 is None and EmptyBasic is None:
            raise RuntimeError("BatchFluxRender: neither EmptySD3LatentImage nor EmptyLatentImage available")

        text_enc = CLIPTextEncode()
        sampler = KSampler()
        decoder = VAEDecode()
        empty_latent_cls = EmptySD3() if EmptySD3 else EmptyBasic()
        guidance_node = FluxGuidance() if FluxGuidance else None

        try:
            negative = text_enc.encode(clip, "")[0]
        except Exception as exc:
            raise RuntimeError(f"BatchFluxRender: negative CLIPTextEncode failed: {exc}")

        report_lines = [
            f"BatchFluxRender: {len(prompts)} shot(s) @ {width}x{height}, "
            f"steps={steps}, cfg={cfg}, guidance={guidance}, "
            f"sampler={sampler_name}/{scheduler}, "
            f"mode={'fast_batch' if fast_batch else 'serial'}",
        ]

        # FAST BATCH PATH
        if fast_batch and len(prompts) > 1:
            batched_result = _try_fast_batch(
                prompts=prompts, model=model, clip=clip, vae=vae,
                text_enc=text_enc, guidance_node=guidance_node,
                empty_latent_cls=empty_latent_cls, sampler=sampler,
                decoder=decoder, negative=negative, seed=seed, steps=steps,
                cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                width=width, height=height, guidance=guidance,
                report_lines=report_lines,
            )
            if batched_result is not None:
                image_batch, total_ms = batched_result
                report_lines.append(
                    f"Total: {len(prompts)} image(s) in {total_ms} ms via fast_batch (one KSampler call)"
                )
                log.info("[BatchFluxRender] batch complete: %d image(s) in %d ms (fast_batch)",
                         len(prompts), total_ms)
                return (image_batch, "\n".join(report_lines))
            log.info("[BatchFluxRender] fast_batch fell through, running serial loop")
            report_lines.append("  fast_batch failed -- fell through to serial loop")

        # SERIAL LOOP PATH
        images = []
        for shot_index, prompt_text in enumerate(prompts):
            shot_t0 = time.time()
            shot_seed = seed if freeze_seed else (seed + shot_index)
            preview = prompt_text[:120] + ("..." if len(prompt_text) > 120 else "")
            log.info("[BatchFluxRender] shot %d/%d seed=%d: %s",
                     shot_index + 1, len(prompts), shot_seed, preview)
            try:
                positive = text_enc.encode(clip, prompt_text)[0]
                if guidance_node is not None:
                    positive = guidance_node.append(positive, guidance)[0]
                latent = empty_latent_cls.generate(width, height, 1)[0]
                samples = sampler.sample(model, shot_seed, steps, cfg, sampler_name,
                                         scheduler, positive, negative, latent, 1.0)[0]
                img = decoder.decode(vae, samples)[0]
                images.append(img)
                shot_ms = int((time.time() - shot_t0) * 1000)
                log.info("[BatchFluxRender] shot %d done in %d ms", shot_index + 1, shot_ms)
                report_lines.append(f"  shot {shot_index + 1}: {shot_ms} ms  {preview}")
            except Exception as exc:
                log.exception("[BatchFluxRender] shot %d failed: %s", shot_index + 1, exc)
                report_lines.append(f"  shot {shot_index + 1}: FAILED ({exc})")

        if not images:
            raise RuntimeError("BatchFluxRender: all shots failed")
        import torch  # type: ignore
        image_batch = torch.cat(images, dim=0)
        total_ms = int((time.time() - t_start) * 1000)
        report_lines.append(
            f"Total: {len(images)} image(s) in {total_ms} ms (avg {total_ms // max(len(images), 1)} ms/shot)"
        )
        log.info("[BatchFluxRender] batch complete: %d image(s) in %d ms", len(images), total_ms)
        return (image_batch, "\n".join(report_lines))


__all__ = ["BatchFluxRender"]
