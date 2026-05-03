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

# Step 1 (ROADMAP P0, 2026-04-30 lock) -- radio still FLUX prompt.
# Used as the I2V reference image for HuMo lip-sync renders during
# ANY non-dialogue line (announcer, music_open, music_close,
# music_inter, sfx).  Per Jeffrey 2026-04-30: every audio second
# of every episode is a HuMo clip; people speaking get people
# lip-syncing, sfx + music + announcer all get the radio
# lip-syncing -- the radio is the performer, not a static prop.
#
# Prompt is built DYNAMICALLY per episode from story context
# (ledger.meta.gen_params.style + style) so the
# radio's aesthetic matches the episode's tone.  See
# `_build_dynamic_radio_prompt()` below.  This constant is the
# safety fallback used only when ledger context is unavailable.
_RADIO_FALLBACK_PROMPT = (
    "sci-fi retrofuturistic radio broadcast unit, glowing CRT "
    "frequency display, copper vacuum tubes haloed in plasma, "
    "brushed steel chassis with art-deco engraving, dim amber "
    "and cyan rim lighting, dust-mote atmosphere, 35mm film "
    "grain, broadcast-distressed cinematic aesthetic, sharp "
    "focus, centered composition, 1080p"
)


# SIGNAL LOST universal suffix.  Appended to every radio prompt
# regardless of the user's tonal direction so all radios share
# the broadcast-distress identity.
_RADIO_PROMPT_SUFFIX = (
    "35mm film grain, broadcast-distressed cinematic aesthetic, "
    "sharp focus, centered composition, 1080p"
)


def _build_dynamic_radio_prompt(led):
    """Build a radio still FLUX prompt from story context.

    BUG-LOCAL-024 (Phase H, 2026-05-03): the radio is the visual
    anchor for the announcer, music cues, and SFX — every audio
    second that isn't a HuMo lip-sync clip falls back to this
    image. Jeffrey: "we always need flux to look at the story
    and render that radio for the music, announcer and sfx."

    Resolution order, each field tried until one yields a non-empty
    descriptor (logged at INFO so the runtime tail shows which
    branch fired):
      1. ``ledger.meta.gen_params_initial.style``     (primary; the
         widget the user picked, e.g. "noir mystery")
      2. ``ledger.meta.gen_params.style``             (back-compat
         for ledgers that used the spine-ledger schema)
      3. ``ledger.meta.gen_params_initial.style_custom``  (free-text
         override the user typed when style="custom")
      4. First scene's environment hint from
         ``ledger.scenes[0].env`` or ``ledger.scenes[0].description``
         — gives the radio a setting-aware tone even when style
         is empty
      5. ``ledger.episode_id`` (slug) — last resort before the
         hardcoded fallback, so the radio at least reflects the
         episode title vibe instead of a generic sci-fi prop
      6. ``_RADIO_FALLBACK_PROMPT`` — true last resort

    A scene-context hint (first scene's env/description, truncated)
    is APPENDED to the resolved descriptor whenever it's available
    AND distinct from the resolved descriptor itself, so the radio
    composition picks up specific episode atmosphere on top of the
    general style. Bounded length: scene-context hint capped at 60
    chars, total prompt body capped before the universal suffix.

    Hostile-input safe: any wrong type lands on the safe default.

    Examples:
      style="noir mystery", first_scene_env="rain-slicked alley"
        -> "noir mystery radio broadcast unit, set in rain-slicked
            alley, 35mm film grain, ..."
      style empty, first_scene_env="cramped cargo bay vibrating"
        -> "cramped cargo bay vibrating radio broadcast unit, ..."
      everything empty
        -> _RADIO_FALLBACK_PROMPT
    """
    if not led or not isinstance(led, dict):
        log.info("[BatchFluxRender] radio prompt: led missing -> fallback")
        return _RADIO_FALLBACK_PROMPT
    meta = led.get("meta") if isinstance(led, dict) else None
    if not isinstance(meta, dict):
        meta = {}
    gp = meta.get("gen_params_initial")
    if not isinstance(gp, dict):
        gp = meta.get("gen_params")
    if not isinstance(gp, dict):
        gp = {}

    def _safe_str(x):
        return x.strip() if isinstance(x, str) else ""

    # Tier 1-3: style from widget
    descriptor = _safe_str(gp.get("style"))
    branch = "gen_params_initial.style"
    if not descriptor:
        descriptor = _safe_str(gp.get("style_custom"))
        if descriptor:
            branch = "gen_params_initial.style_custom"

    # Tier 4: first scene env / description
    first_scene_env = ""
    scenes = led.get("scenes") if isinstance(led, dict) else None
    if isinstance(scenes, list) and scenes:
        first = scenes[0] if isinstance(scenes[0], dict) else {}
        first_scene_env = (
            _safe_str(first.get("env"))
            or _safe_str(first.get("description"))
        )
    if not descriptor and first_scene_env:
        descriptor = first_scene_env
        branch = "first_scene_env"

    # Tier 5: episode_id slug (last resort before hardcoded fallback)
    if not descriptor:
        ep_id = _safe_str(led.get("episode_id"))
        if ep_id and not ep_id.startswith("pending_"):
            # Strip the "signal_lost_" prefix and trailing timestamp
            # for a more natural descriptor
            slug = ep_id
            if slug.startswith("signal_lost_"):
                slug = slug[len("signal_lost_"):]
            # Strip trailing _<8digits>_<6digits> timestamp if present
            import re as _re
            slug = _re.sub(r"_\d{8}_\d{6}$", "", slug)
            slug = slug.replace("_", " ").strip()
            if slug:
                descriptor = slug
                branch = "episode_id_slug"

    # Tier 6: hardcoded fallback
    if not descriptor:
        log.info("[BatchFluxRender] radio prompt: all tiers empty -> "
                 "hardcoded fallback")
        return _RADIO_FALLBACK_PROMPT

    # Bounded scene-context hint, only when distinct from descriptor
    scene_hint = ""
    if (first_scene_env
            and first_scene_env != descriptor
            and branch != "first_scene_env"):
        scene_hint = first_scene_env[:60].strip().rstrip(",")

    # Cap descriptor to keep total prompt body reasonable. FLUX prompts
    # past ~200 tokens lose composition focus; 80 chars is a sane bound.
    descriptor_capped = descriptor[:80].strip().rstrip(",")

    if scene_hint:
        body = f"{descriptor_capped} radio broadcast unit, set in {scene_hint}"
        log.info(
            "[BatchFluxRender] radio prompt: branch=%s + scene_hint=%r "
            "-> %s ...",
            branch, scene_hint[:40], body[:60],
        )
    else:
        body = f"{descriptor_capped} radio broadcast unit"
        log.info(
            "[BatchFluxRender] radio prompt: branch=%s -> %s ...",
            branch, body[:60],
        )

    return f"{body}, {_RADIO_PROMPT_SUFFIX}"


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
                "radio_bookend_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Radio still FLUX prompt. Default (empty) "
                        "builds dynamically from ledger.meta.gen_params "
                        "(style + style) so each "
                        "episode's radio matches its tone. Set this "
                        "field to a non-empty string to override the "
                        "dynamic builder with your verbatim prompt. "
                        "Set to literal 'DISABLED' (case-insensitive) "
                        "to skip rendering entirely. Output saved to "
                        "output/otr/stills/radio_bookend_<ep_id>.png "
                        "and stamped into ledger.radio_bookend_path "
                        "+ ledger.meta.radio_bookend_path."
                    ),
                }),
                "radio_bookend_seed": ("INT", {
                    "default": 4242,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "tooltip": (
                        "Deterministic seed for the radio bookend so "
                        "the same image gets used across episodes "
                        "(unless explicitly changed)."
                    ),
                }),
            },
        }

    def execute(self, model, clip, vae, script_json, batch_limit, seed, steps,
                cfg, sampler_name, scheduler, width, height, guidance,
                fallback_prompt=_DEFAULT_FALLBACK, style_suffix=_DEFAULT_STYLE_SUFFIX,
                freeze_seed=False, fast_batch=True,
                radio_bookend_prompt="",
                radio_bookend_seed=4242):
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
                # BUG-LOCAL-127 fix (2026-05-01): the original code
                # returned here without ever rendering the radio
                # bookend, so fast_batch (the default) silently
                # skipped the entire radio still pipeline. Symptom
                # surfaced as "wanted radio still but it's missing"
                # downstream in BatchHumoRender. The bookend pass is
                # now invoked before BOTH this fast_batch return and
                # the serial-loop return below.
                self._render_radio_bookend_step(
                    radio_bookend_prompt=radio_bookend_prompt,
                    radio_bookend_seed=radio_bookend_seed,
                    model=model, clip=clip, vae=vae,
                    text_enc=text_enc, guidance_node=guidance_node,
                    empty_latent_cls=empty_latent_cls, sampler=sampler,
                    decoder=decoder, negative=negative,
                    steps=steps, cfg=cfg,
                    sampler_name=sampler_name, scheduler=scheduler,
                    width=width, height=height, guidance=guidance,
                    report_lines=report_lines,
                )
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

        # Step 1 (ROADMAP P0, 2026-04-30 lock) -- render the radio
        # still via the helper so BOTH this serial-loop path and the
        # fast_batch path above invoke it.  See BUG-LOCAL-127.
        self._render_radio_bookend_step(
            radio_bookend_prompt=radio_bookend_prompt,
            radio_bookend_seed=radio_bookend_seed,
            model=model, clip=clip, vae=vae,
            text_enc=text_enc, guidance_node=guidance_node,
            empty_latent_cls=empty_latent_cls, sampler=sampler,
            decoder=decoder, negative=negative,
            steps=steps, cfg=cfg,
            sampler_name=sampler_name, scheduler=scheduler,
            width=width, height=height, guidance=guidance,
            report_lines=report_lines,
        )

        return (image_batch, "\n".join(report_lines))

    def _render_radio_bookend_step(
        self,
        radio_bookend_prompt,
        radio_bookend_seed,
        model, clip, vae,
        text_enc, guidance_node,
        empty_latent_cls, sampler,
        decoder, negative,
        steps, cfg,
        sampler_name, scheduler,
        width, height, guidance,
        report_lines,
    ):
        """Render + stamp the radio still.  Called from BOTH the
        fast_batch return path and the serial-loop return path
        (BUG-LOCAL-127 fix 2026-05-01).

        Three modes (widget value):
          - ``"DISABLED"`` (case-insensitive): skip rendering.
          - non-empty otherwise: use as verbatim override.
          - empty (default): build dynamically from ledger context
            (style + style).

        The radio still is consumed by BatchHumoRender as the I2V
        reference for all non-dialogue HuMo clips (announcer,
        music_*, sfx).  Rendering ALWAYS attempts under the new
        default so wall-to-wall HuMo coverage has its non-dialogue
        reference image ready.

        Never raises -- a render failure is logged with full
        traceback (log.exception) and noted in report_lines, but the
        outer execute() always returns the main image batch.
        """
        widget_str = (radio_bookend_prompt or "").strip()
        if widget_str.upper() == "DISABLED":
            log.info(
                "[BatchFluxRender] radio bookend DISABLED via widget "
                "sentinel (no render attempted)"
            )
            report_lines.append("  radio_bookend: DISABLED (widget)")
            return
        mode = "OVERRIDE" if widget_str else "DYNAMIC"
        log.info(
            "[BatchFluxRender] radio bookend %s mode (seed=%d)",
            mode, int(radio_bookend_seed),
        )
        # 2026-05-01 EVENING (Jeffrey, post round-robin pixel-policy
        # synthesis): render the radio still at FLUX-native ~1MP, then
        # Lanczos-downscale to 832x480 before save.
        #
        # Why 1248x720:
        #   - FLUX is trained on a ~1MP pixel manifold. Rendering below
        #     ~1MP (e.g. 832x480 = 0.4MP) degrades composition + anatomy
        #     visibly per Gemini round-robin 2026-05-01 (both ladders
        #     converged on this).
        #   - 1248x720 is EXACT 1.7333:1 (= 832/480), so no aspect
        #     distortion and no crop math when downscaling. Both round-
        #     robin runs proposed 1344x768 (1.75:1) which requires a
        #     center-crop to 1332x768 to preserve aspect; 1248x720
        #     is the cleaner alternative the consults didn't surface.
        #   - 1248/8 = 156 and 720/8 = 90, both mod-8 valid for FLUX.
        #   - 1248/832 = 1.5 exactly and 720/480 = 1.5 exactly --
        #     clean integer-ratio downscale, no fractional resampling.
        #   - 0.898 MP is a hair below FLUX's 1MP sweet spot but well
        #     within its trained manifold (FLUX handles 0.85-1.15 MP
        #     fluently per community benchmarks).
        # Cast stills DON'T get this override -- they keep the user's
        # widget dims (typically 1024x1024) because HuMo I2V works
        # best with square face crops.
        _RADIO_BOOKEND_W = 1248
        _RADIO_BOOKEND_H = 720
        try:
            self._render_and_save_radio_bookend(
                # widget_str is "" in DYNAMIC mode -- callee detects
                # the empty string and builds dynamically.
                prompt_text=widget_str,
                model=model, clip=clip, vae=vae,
                text_enc=text_enc, guidance_node=guidance_node,
                empty_latent_cls=empty_latent_cls, sampler=sampler,
                decoder=decoder, negative=negative,
                seed=int(radio_bookend_seed), steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                width=_RADIO_BOOKEND_W, height=_RADIO_BOOKEND_H,
                guidance=guidance,
                report_lines=report_lines,
            )
        except Exception as exc:
            # BUG-LOCAL-121 diagnostic hardening (2026-05-01,
            # round-robin Q3 Symptom 2):
            # log.exception emits the full stack trace at ERROR level
            # so the FLUX-side root cause is visible in the log
            # without re-running.
            log.exception(
                "[BatchFluxRender] radio bookend render failed: %s",
                exc,
            )
            report_lines.append(
                f"  radio_bookend: FAILED ({exc})"
            )

    @staticmethod
    def _render_and_save_radio_bookend(
        prompt_text, model, clip, vae, text_enc, guidance_node,
        empty_latent_cls, sampler, decoder, negative,
        seed, steps, cfg, sampler_name, scheduler,
        width, height, guidance, report_lines,
    ):
        """Step 1: render the radio still (one extra FLUX call) and
        save it to ``output/otr/stills/radio_bookend_<ep_id>.png``.

        ``prompt_text``:
          - Empty string  -> build dynamically from ledger context
                             (style + style); falls
                             back to ``_RADIO_FALLBACK_PROMPT`` if
                             ledger is unavailable.
          - Non-empty     -> use verbatim as override.

        Stamps both ``ledger.radio_bookend_path`` (top-level) AND
        ``ledger.meta.radio_bookend_path`` for belt-and-suspenders.
        Downstream (BatchHumoRender) reads the radio still as the
        I2V reference image for any non-dialogue HuMo clip
        (announcer, music_*, sfx).  Per Jeffrey 2026-04-30: every
        audio second is a HuMo clip; people speaking get people
        lip-syncing, everything else gets the radio lip-syncing.
        """
        import numpy as np  # type: ignore
        from pathlib import Path
        from PIL import Image  # type: ignore

        # Lazily import path + ledger helpers from nodes/.  Done up
        # front (BEFORE FLUX render) so dynamic-prompt mode can pull
        # genre/style from the ledger before the render kicks off.
        try:
            import sys as _sys
            _NODES_DIR = Path(__file__).resolve().parents[1] / "nodes"
            if str(_NODES_DIR) not in _sys.path:
                _sys.path.insert(0, str(_NODES_DIR))
            import _otr_paths as _OTRP  # type: ignore
            import _otr_ledger as _OTRL  # type: ignore
            import production_ledger as _PROD_LEDGER  # type: ignore
        except Exception as exc:
            log.warning(
                "[BatchFluxRender] radio still: helper import failed (%s)",
                exc,
            )
            return

        # BUG-LOCAL-021 (Phase G, 2026-05-03): use the in-flight Ledger
        # singleton to identify the current episode, NOT the global
        # mtime walker. Prior to this fix, `find_most_recent_ledger`
        # picked whichever `*_ledger.json` had the newest mtime under
        # `otr/episodes/` -- that could be a leftover from a prior
        # episode, causing the radio bookend to be stamped to the
        # WRONG ledger (proven in soak run 2026-05-02 where a May 2
        # run stamped to an April 26 episode_id).
        #
        # Same bug shape as BUG-LOCAL-014 (spacesaver wrong-episode
        # wipe). Phase A fixed it for rtx_upscale; this site was
        # missed. The singleton's `path` property advances correctly
        # through Ledger.rename_episode (Phase B), so it tracks the
        # in-flight episode by construction. ComfyUI sequential queue
        # + LLMScriptWriter's IS_CHANGED=time.time() prevent the
        # singleton from ever going stale across queued runs.
        try:
            _led_singleton = _PROD_LEDGER.get_ledger()
            ledger_p = Path(_led_singleton.path)
            if not ledger_p.exists():
                # Fall back to mtime walker as last resort -- shouldn't
                # happen in normal pipeline order but defends against
                # standalone test invocations of this node.
                log.warning(
                    "[BatchFluxRender] radio still: singleton path %s does "
                    "not exist on disk; falling back to mtime walker",
                    ledger_p,
                )
                ledger_p = _OTRL.find_most_recent_ledger(
                    [_OTRP.otr_episodes_root(), _OTRP.otr_legacy_audio_dir()]
                )
        except Exception as _exc:
            log.warning(
                "[BatchFluxRender] radio still: singleton lookup failed (%s); "
                "falling back to mtime walker", _exc,
            )
            ledger_p = _OTRL.find_most_recent_ledger(
                [_OTRP.otr_episodes_root(), _OTRP.otr_legacy_audio_dir()]
            )
        led = None
        episode_id = "episode"
        if ledger_p is not None:
            led = _OTRL.load_ledger_safe(ledger_p)
            if led is not None:
                episode_id = (led.get("episode_id") or "episode").strip()
        # BUG-LOCAL-121 diagnostic hardening (2026-05-01,
        # round-robin Q3 Symptom 2 hypothesis (c)):
        # Stamp the episode_id we're about to write into the bookend
        # filename so a downstream "wanted radio still but it's
        # missing" warning in BatchHumoRender can be compared
        # directly. If the two episode_ids differ, the ledger was
        # renamed/swapped between FLUX and HuMo phases.
        log.info(
            "[BatchFluxRender] radio bookend stage: ledger=%s "
            "episode_id=%s (will save as radio_bookend_%s.png)",
            ledger_p.name if ledger_p else "<none>",
            episode_id,
            episode_id,
        )

        # Resolve the prompt: empty widget -> dynamic build; non-empty
        # widget -> verbatim override.
        widget_prompt = (prompt_text or "").strip()
        if widget_prompt:
            resolved_prompt = widget_prompt
            prompt_source = "override"
        else:
            resolved_prompt = _build_dynamic_radio_prompt(led)
            # Diagnose which branch the dynamic builder took.  Post
            # 2026-04-30 consolidation: free-text style is
            # the single tonal knob; either we have it (dynamic) or
            # we don't (fallback).  No genre-vs-style split.
            if led is None:
                prompt_source = "fallback (no ledger)"
            else:
                meta = led.get("meta") or {}
                gp = meta.get("gen_params_initial") or meta.get("gen_params") or {}
                if not isinstance(gp, dict):
                    gp = {}
                raw_style = gp.get("style")
                style = raw_style.strip() if isinstance(raw_style, str) else ""
                if style:
                    prompt_source = f"dynamic (style={style!r})"
                else:
                    prompt_source = "fallback (no style)"

        log.info(
            "[BatchFluxRender] radio still prompt source=%s, len=%d, "
            "first 80 chars: %s",
            prompt_source, len(resolved_prompt), resolved_prompt[:80],
        )

        # Now run the FLUX render with the resolved prompt.
        t0 = time.time()
        positive = text_enc.encode(clip, resolved_prompt)[0]
        if guidance_node is not None:
            positive = guidance_node.append(positive, guidance)[0]
        latent = empty_latent_cls.generate(width, height, 1)[0]
        samples = sampler.sample(
            model, seed, steps, cfg, sampler_name, scheduler,
            positive, negative, latent, 1.0,
        )[0]
        img = decoder.decode(vae, samples)[0]
        # img shape: [B, H, W, C] in 0..1 float

        stills_dir = _OTRP.otr_stills_dir()
        stills_dir.mkdir(parents=True, exist_ok=True)
        out_path = stills_dir / f"radio_bookend_{episode_id}.png"

        # 2026-05-01 EVENING: render at FLUX-native 1248x720 (above),
        # then Lanczos-downscale to the canonical 832x480 consumer dims
        # before saving. Result on disk is the same 832x480 PNG every
        # downstream node already expects (LTX I2V ref, VideoComposite
        # static-radio fill via BUG-129a) -- no consumer code needs to
        # change. The downscale gives us FLUX's 1MP-class detail compressed
        # cleanly into 832x480, instead of asking FLUX to render at
        # sub-megapixel where it composes weakly. Lanczos is the
        # canonical anti-aliased downsample for high-frequency detail
        # like dial markings + tube edges -- bilinear (PIL.BILINEAR)
        # would soften the radio's mechanical detail.
        _CONSUMER_W = 832
        _CONSUMER_H = 480

        # Save image. img is shape [1, H, W, C] tensor; convert to PIL,
        # downscale Lanczos, then save.
        try:
            arr = img[0].detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img[0])
            arr = np.clip(arr * 255.0, 0, 255).astype("uint8")
            pil_img = Image.fromarray(arr)
            # Only downscale if we rendered larger than consumer dims.
            # Defensive: a future override could pass _RADIO_BOOKEND_W ==
            # _CONSUMER_W in which case PIL.resize would still work but
            # be a no-op.
            if pil_img.width > _CONSUMER_W or pil_img.height > _CONSUMER_H:
                pil_img = pil_img.resize(
                    (_CONSUMER_W, _CONSUMER_H), Image.Resampling.LANCZOS,
                )
                log.info(
                    "[BatchFluxRender] radio still Lanczos-downscaled "
                    "%dx%d -> %dx%d (FLUX-native render -> consumer dims)",
                    arr.shape[1], arr.shape[0], _CONSUMER_W, _CONSUMER_H,
                )
            pil_img.save(out_path)
        except Exception as exc:
            log.warning(
                "[BatchFluxRender] radio still save failed: %s", exc,
            )
            return

        elapsed_ms = int((time.time() - t0) * 1000)
        log.info(
            "[BatchFluxRender] radio still rendered + saved: %s (%d ms)",
            out_path, elapsed_ms,
        )
        report_lines.append(
            f"  radio_bookend: {elapsed_ms} ms -> {out_path.name} "
            f"({prompt_source})"
        )

        # Stamp ledger paths for BatchHumoRender + VideoComposite to
        # find.  Belt-and-suspenders: stamp under both top-level
        # `radio_bookend_path` AND `meta.radio_bookend_path` since
        # video_composite.py reads either location.
        if ledger_p is not None and led is not None:
            try:
                led["radio_bookend_path"] = str(out_path)
                meta = led.setdefault("meta", {})
                meta["radio_bookend_path"] = str(out_path)
                meta["radio_bookend_prompt_source"] = prompt_source
                _OTRL.save_ledger_safe(ledger_p, led)
                log.info(
                    "[BatchFluxRender] radio still ledger stamp OK: "
                    "ledger=%s, path=%s",
                    ledger_p.name, out_path,
                )
            except Exception as exc:
                log.warning(
                    "[BatchFluxRender] radio still ledger stamp failed: %s",
                    exc,
                )
        else:
            log.warning(
                "[BatchFluxRender] radio still ledger stamp SKIPPED: "
                "no ledger found in audio dirs (file rendered but "
                "downstream nodes will not be able to locate it)"
            )


__all__ = ["BatchFluxRender"]
