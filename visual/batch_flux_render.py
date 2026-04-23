"""
batch_flux_render.py  --  OTR_BatchFluxRender ComfyUI node
===========================================================
Render N FLUX images in lockstep from the polished script JSON,
in a single graph run, sharing one MODEL/CLIP/VAE load.

Motivation (Jeffrey, 2026-04-23):
    OTR_VisualExtractFluxPrompt picks ONE environment token at
    a time. Queuing the graph N times works but reloads nothing
    and is fiddly. For a 4-5 shot episode preview, we want all
    environments rendered in a single Run button press, in the
    order they appear in the script.

What it does:
    1. Parse the cleaned_script_json from OTR_VisualPromptCoercion.
    2. Pull the first `batch_limit` environment tokens, each with
       its polished description + style suffix.
    3. Loop per prompt, reusing the resident MODEL/CLIP/VAE:
         CLIPTextEncode(clip, prompt)  -> positive conditioning
         CLIPTextEncode(clip, "")      -> negative conditioning
         FluxGuidance(positive, 3.5)   -> FLUX-style guidance
         EmptySD3LatentImage(w, h, 1)  -> 16-channel latent
         KSampler(model, seed+i, ...)  -> sampled latent
         VAEDecode(vae, latent)        -> image
    4. Concatenate all images along the batch dim into one IMAGE
       tensor. SaveImage will write them as N separate files with
       ComfyUI's usual indexing.

Design notes (why this shape):
    - Delegates to ComfyUI's OWN node classes (nodes.CLIPTextEncode,
      nodes.KSampler, nodes.VAEDecode, comfy_extras.nodes_sd3.
      EmptySD3LatentImage, comfy_extras.nodes_flux.FluxGuidance)
      so loader/sampler behavior stays in lockstep with upstream.
      No re-implementation, no divergence. This follows the memory
      feedback "prefer ComfyUI-native loader over reinventing".
    - Seed auto-increments per shot (seed + shot_index) so each
      environment gets a different noise pattern. Flip
      `freeze_seed=True` to reuse the base seed for all shots
      (useful for A/B prompt comparisons).
    - ZERO torch/diffusers/model imports at module scope. Node-scan
      stays cheap; heavy work only inside execute().
    - Never raises: parsing errors fall back to a single render
      with the `fallback_prompt` so the graph still completes.

TEST workflow wiring (replaces the single-shot chain)::

    OTR_VisualPromptCoercion.cleaned_script_json --> script_json
    OTR_CheckpointLoaderGated.MODEL               --> model
    OTR_CheckpointLoaderGated.CLIP                --> clip
    OTR_CheckpointLoaderGated.VAE                 --> vae
    OTR_BatchFluxRender.images                    --> SaveImage.images

Replaces the CLIPTextEncode + FluxGuidance + EmptySD3LatentImage
+ KSampler + VAEDecode subgraph with one node that owns the loop.
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
    """Import ComfyUI's stock node classes lazily.

    Returns a dict of node-class references. If a class isn't
    available in this ComfyUI version, the dict entry is None and
    the caller falls back gracefully.
    """
    refs: dict[str, Any] = {}
    import nodes  # type: ignore
    refs["CLIPTextEncode"] = getattr(nodes, "CLIPTextEncode", None)
    refs["KSampler"] = getattr(nodes, "KSampler", None)
    refs["VAEDecode"] = getattr(nodes, "VAEDecode", None)
    refs["EmptyLatentImage"] = getattr(nodes, "EmptyLatentImage", None)

    # comfy_extras ships with SD3 / FLUX helpers, packaged slightly
    # differently across ComfyUI revisions. Try the common paths.
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
        # Older ComfyUI may expose it in nodes_advanced or a plain module
        try:
            from comfy_extras.nodes_model_advanced import FluxGuidance  # type: ignore
            flux_guidance = FluxGuidance
        except Exception:
            pass

    refs["EmptySD3LatentImage"] = empty_sd3
    refs["FluxGuidance"] = flux_guidance
    return refs


def _parse_env_prompts(
    script_json: str,
    batch_limit: int,
    fallback: str,
    style_suffix: str,
) -> list[str]:
    """Pull up to batch_limit environment token descriptions.

    Returns a list of polished, style-suffixed prompt strings. Always
    returns at least one entry (the fallback) so the render loop
    always fires at least once.
    """
    if not script_json or not script_json.strip():
        log.info("[BatchFluxRender] empty script_json; using fallback x1")
        return [f"{fallback}, {style_suffix}".rstrip(", ")]

    try:
        payload = json.loads(script_json)
    except json.JSONDecodeError as exc:
        log.warning(
            "[BatchFluxRender] script_json JSONDecodeError (%s); "
            "using fallback x1",
            exc,
        )
        return [f"{fallback}, {style_suffix}".rstrip(", ")]

    if isinstance(payload, dict):
        tokens = payload.get("tokens") or payload.get("script") or []
    elif isinstance(payload, list):
        tokens = payload
    else:
        log.warning(
            "[BatchFluxRender] unexpected script_json root %s; "
            "using fallback x1",
            type(payload).__name__,
        )
        return [f"{fallback}, {style_suffix}".rstrip(", ")]

    env_tokens = [
        t for t in tokens
        if isinstance(t, dict) and t.get("type") == "environment"
    ]
    if not env_tokens:
        log.info(
            "[BatchFluxRender] no environment tokens (total=%d); "
            "using fallback x1",
            len(tokens),
        )
        return [f"{fallback}, {style_suffix}".rstrip(", ")]

    limit = max(1, min(batch_limit, len(env_tokens)))
    selected = env_tokens[:limit]

    prompts: list[str] = []
    for i, token in enumerate(selected):
        desc = (token.get("description") or "").strip()
        if not desc:
            desc = fallback
        parts = [desc]
        if style_suffix and style_suffix.strip():
            parts.append(style_suffix.strip())
        prompts.append(", ".join(parts))

    log.info(
        "[BatchFluxRender] queued %d env prompt(s) from %d available",
        len(prompts),
        len(env_tokens),
    )
    return prompts


class BatchFluxRender:
    """Render N FLUX images in lockstep from the script JSON."""

    CATEGORY = "OTR/v2/Visual"
    FUNCTION = "execute"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "report")

    @classmethod
    def INPUT_TYPES(cls):
        # KSampler's sampler_name/scheduler lists are long and
        # depend on the runtime. Pull them from nodes.KSampler so
        # the dropdown always matches the stock node's options.
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
                "script_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Wire from OTR_VisualPromptCoercion."
                        "cleaned_script_json. The first batch_limit "
                        "environment tokens will be rendered."
                    ),
                }),
                "batch_limit": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 16,
                    "step": 1,
                    "tooltip": (
                        "Max images to render this run. Each shot "
                        "reuses the resident MODEL/CLIP/VAE so the "
                        "marginal cost is just sampling + decode."
                    ),
                }),
                "seed": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": (
                        "Base seed. Per-shot seed is seed+shot_index "
                        "unless freeze_seed is on."
                    ),
                }),
                "steps": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "cfg": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": (
                        "FLUX typically uses cfg=1.0 (guidance is "
                        "baked in via FluxGuidance)."
                    ),
                }),
                "sampler_name": (samplers, {"default": "euler"}),
                "scheduler": (schedulers, {"default": "simple"}),
                "width": ("INT", {
                    "default": 1024, "min": 256, "max": 2048, "step": 8,
                }),
                "height": ("INT", {
                    "default": 1024, "min": 256, "max": 2048, "step": 8,
                }),
                "guidance": ("FLOAT", {
                    "default": 3.5,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": (
                        "FluxGuidance value. 3.5 is the FLUX-dev "
                        "reference default; 2.5 is more varied, "
                        "4.5+ is more prompt-adherent."
                    ),
                }),
            },
            "optional": {
                "fallback_prompt": ("STRING", {
                    "multiline": True,
                    "default": _DEFAULT_FALLBACK,
                }),
                "style_suffix": ("STRING", {
                    "multiline": False,
                    "default": _DEFAULT_STYLE_SUFFIX,
                }),
                "freeze_seed": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Reuse the base seed for every shot so only "
                        "the prompt changes. Useful for A/B prompt "
                        "comparisons at a fixed composition."
                    ),
                }),
            },
        }

    def execute(
        self,
        model,
        clip,
        vae,
        script_json: str,
        batch_limit: int,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        width: int,
        height: int,
        guidance: float,
        fallback_prompt: str = _DEFAULT_FALLBACK,
        style_suffix: str = _DEFAULT_STYLE_SUFFIX,
        freeze_seed: bool = False,
    ):
        t_start = time.time()
        prompts = _parse_env_prompts(
            script_json, batch_limit, fallback_prompt, style_suffix
        )

        refs = _lazy_nodes()
        CLIPTextEncode = refs["CLIPTextEncode"]
        KSampler = refs["KSampler"]
        VAEDecode = refs["VAEDecode"]
        EmptySD3 = refs["EmptySD3LatentImage"]
        EmptyBasic = refs["EmptyLatentImage"]
        FluxGuidance = refs["FluxGuidance"]

        if CLIPTextEncode is None or KSampler is None or VAEDecode is None:
            raise RuntimeError(
                "BatchFluxRender: ComfyUI nodes module missing "
                "CLIPTextEncode / KSampler / VAEDecode. Version "
                "mismatch?"
            )
        if EmptySD3 is None and EmptyBasic is None:
            raise RuntimeError(
                "BatchFluxRender: neither EmptySD3LatentImage nor "
                "EmptyLatentImage is available."
            )

        text_enc = CLIPTextEncode()
        sampler = KSampler()
        decoder = VAEDecode()
        empty_latent_cls = EmptySD3() if EmptySD3 else EmptyBasic()
        guidance_node = FluxGuidance() if FluxGuidance else None

        # Negative conditioning is empty for FLUX. Encode once and reuse.
        try:
            negative = text_enc.encode(clip, "")[0]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"BatchFluxRender: negative CLIPTextEncode failed: {exc}"
            )

        images = []
        per_shot_ms: list[int] = []
        report_lines: list[str] = [
            f"BatchFluxRender: {len(prompts)} shot(s) @ {width}x{height}, "
            f"steps={steps}, cfg={cfg}, guidance={guidance}, "
            f"sampler={sampler_name}/{scheduler}",
        ]

        for shot_index, prompt_text in enumerate(prompts):
            shot_t0 = time.time()
            shot_seed = seed if freeze_seed else (seed + shot_index)
            preview = prompt_text[:120] + ("..." if len(prompt_text) > 120 else "")
            log.info(
                "[BatchFluxRender] shot %d/%d seed=%d: %s",
                shot_index + 1,
                len(prompts),
                shot_seed,
                preview,
            )

            try:
                positive = text_enc.encode(clip, prompt_text)[0]
                if guidance_node is not None:
                    # FluxGuidance.append(conditioning, guidance)
                    positive = guidance_node.append(positive, guidance)[0]

                latent = empty_latent_cls.generate(width, height, 1)[0]

                samples = sampler.sample(
                    model,
                    shot_seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive,
                    negative,
                    latent,
                    1.0,  # denoise
                )[0]

                img = decoder.decode(vae, samples)[0]
                images.append(img)

                shot_ms = int((time.time() - shot_t0) * 1000)
                per_shot_ms.append(shot_ms)
                log.info(
                    "[BatchFluxRender] shot %d done in %d ms",
                    shot_index + 1,
                    shot_ms,
                )
                report_lines.append(
                    f"  shot {shot_index + 1}: {shot_ms} ms  {preview}"
                )
            except Exception as exc:  # noqa: BLE001
                log.exception(
                    "[BatchFluxRender] shot %d failed: %s",
                    shot_index + 1,
                    exc,
                )
                report_lines.append(
                    f"  shot {shot_index + 1}: FAILED ({exc})"
                )
                # Continue rather than abort -- partial batch is
                # still useful for debugging.

        if not images:
            raise RuntimeError(
                "BatchFluxRender: all shots failed; check the log above."
            )

        # Concatenate along batch dimension.
        import torch  # type: ignore
        image_batch = torch.cat(images, dim=0)

        total_ms = int((time.time() - t_start) * 1000)
        report_lines.append(
            f"Total: {len(images)} image(s) in {total_ms} ms "
            f"(avg {total_ms // max(len(images), 1)} ms/shot)"
        )
        report = "\n".join(report_lines)
        log.info(
            "[BatchFluxRender] batch complete: %d image(s) in %d ms",
            len(images),
            total_ms,
        )

        return (image_batch, report)


__all__ = ["BatchFluxRender"]
