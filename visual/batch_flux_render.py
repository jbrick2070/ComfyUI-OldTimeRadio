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


# ---------------------------------------------------------------------------
# BUG-LOCAL-231 alt-c/e/f telemetry helpers (Jeffrey 2026-05-19 09:30
# directive). 3-smoke clean-state battery falsified alt-a: fire-time
# state was essentially identical across 3 cold-launch iters that spanned
# 4x in sampler step 1 s/it. The variance is downstream of fire. These
# helpers add three new signals at sampler entry / during sampler:
#   - cudnn / cublas determinism flags (alt-e signal: if benchmark=True
#     the cudnn autotuner picks different kernels per cold launch ->
#     run-to-run variance is expected).
#   - nvidia-smi snapshot at sampler entry (alt-f signal: GPU clock /
#     power state / thermal -- a slow iter at lower clock than a fast
#     iter implies hardware throttling).
#   - background poller that captures LHM + nvsmi every 5 sec during
#     sampler.sample (alt-c signal: D3D Shared spillover spike during
#     the sampler in slow iters but not fast iters would confirm
#     offloader thrash).
# All best-effort with broad except; never raises into the sampler path.
# Pure logging -- no behavior change.
# ---------------------------------------------------------------------------

def _log_flux_sampler_precheck() -> None:
    """One-line snapshot of cudnn/cublas determinism + nvsmi clocks at
    sampler entry. Disambiguates alt-e (kernel non-determinism) and
    alt-f (thermal throttling) per BUG-LOCAL-231 2026-05-19 09:30
    decision tree. Best-effort; never raises.
    """
    try:
        import torch  # local import; harmless if absent
        cudnn_bench = getattr(torch.backends.cudnn, "benchmark", "?")
        cudnn_det = getattr(torch.backends.cudnn, "deterministic", "?")
        cudnn_tf32 = getattr(torch.backends.cudnn, "allow_tf32", "?")
        matmul_tf32 = getattr(torch.backends.cuda.matmul, "allow_tf32", "?")
        cuda_init = bool(torch.cuda.is_initialized())
        log.info(
            "[OTR-FLUX-SAMPLER-PRECHECK] cudnn.benchmark=%s "
            "cudnn.deterministic=%s cudnn.allow_tf32=%s "
            "matmul.allow_tf32=%s cuda.is_initialized=%s",
            cudnn_bench, cudnn_det, cudnn_tf32, matmul_tf32, cuda_init,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("[OTR-FLUX-SAMPLER-PRECHECK] flags log failed: %s", exc)

    try:
        import subprocess
        nvsmi = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=clocks.gr,clocks.mem,power.draw,power.state,"
             "temperature.gpu,utilization.gpu,utilization.memory",
             "--format=csv,noheader"],
            text=True, timeout=3.0,
        ).strip()
        log.info("[OTR-FLUX-NVSMI] at sampler entry: %s", nvsmi)
    except Exception as exc:  # noqa: BLE001
        log.warning("[OTR-FLUX-NVSMI] snapshot failed: %s", exc)


class _FluxSamplerPoller:
    """Background thread polling LHM + nvsmi every interval_s seconds
    during sampler.sample(). Captures alt-c (D3D Shared spillover
    during sampler) signal. Logs one line per poll. Daemon thread; no
    cleanup needed if it lingers a tick after stop().

    Usage:
        p = _FluxSamplerPoller(interval_s=5.0)
        p.start()
        try:
            samples = sampler.sample(...)
        finally:
            p.stop()
    """

    def __init__(self, interval_s: float = 5.0):
        import threading
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread = None
        self._tick = 0

    def start(self):
        import threading
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="OTR-FluxSamplerPoller",
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        # Do NOT join; if the thread is blocked on HTTP urlopen we
        # would hang the sampler return. Daemon thread dies with the
        # process if it lingers a tick.

    def _run(self):
        import json as _json
        import subprocess
        import urllib.request
        while not self._stop_event.wait(self.interval_s):
            self._tick += 1
            lhm_used = float("nan")
            d3d_shared = float("nan")
            try:
                with urllib.request.urlopen(
                    "http://localhost:8085/data.json", timeout=2.0,
                ) as r:
                    data = _json.loads(r.read().decode("utf-8"))
                stack = [data]
                while stack:
                    n = stack.pop()
                    if not isinstance(n, dict):
                        continue
                    txt = n.get("Text", "")
                    val = n.get("Value", "")
                    if (
                        txt == "GPU Memory Used"
                        and isinstance(val, str)
                        and val.endswith(" MB")
                    ):
                        try:
                            lhm_used = float(val[:-3])
                        except ValueError:
                            pass
                    elif (
                        txt == "D3D Shared Memory Used"
                        and isinstance(val, str)
                        and val.endswith(" MB")
                    ):
                        try:
                            d3d_shared = float(val[:-3])
                        except ValueError:
                            pass
                    for c in (n.get("Children") or []):
                        stack.append(c)
            except Exception:  # noqa: BLE001
                pass

            nvsmi = ""
            try:
                nvsmi = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=clocks.gr,power.draw,temperature.gpu,"
                     "utilization.gpu",
                     "--format=csv,noheader"],
                    text=True, timeout=2.0,
                ).strip()
            except Exception:  # noqa: BLE001
                pass

            log.info(
                "[OTR-FLUX-SAMPLER-POLL] tick=%d lhm_used=%.0f MB "
                "d3d_shared=%.0f MB nvsmi=%s",
                self._tick, lhm_used, d3d_shared, nvsmi,
            )


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

    # Sprint C C5c (2026-05-15): pre-fetch meta.story_brief + status.
    # The brief replaces the weak scenes[0].env Tier 4 fallback when
    # widget style / style_custom are empty. status is logged for
    # E-07 observability regardless of whether the brief actually
    # gets used (operator can see whether reflection succeeded).
    from ..nodes._otr_story_brief_helpers import (
        get_story_brief_full,
        get_story_brief_status,
    )
    _brief = get_story_brief_full(meta)
    _brief_status = get_story_brief_status(meta)
    log.info(
        "[BatchFluxRender] radio prompt: story_brief_status=%s "
        "(brief_chars=%d)", _brief_status, len(_brief),
    )

    # Tier 1-3: style from widget
    descriptor = _safe_str(gp.get("style"))
    branch = "gen_params_initial.style"
    if not descriptor:
        descriptor = _safe_str(gp.get("style_custom"))
        if descriptor:
            branch = "gen_params_initial.style_custom"

    # Sprint C C5c (2026-05-15): the legacy Tier 4 first-scene-env
    # tier is replaced by meta.story_brief. The brief is a richer
    # post-script descriptor; scenes[0].env was a thin fallback that
    # often surfaced "INT. ROOM -- NIGHT" boilerplate at the bookend.
    # When the brief is absent / failed, the chain falls through to
    # Tier 5 (episode_id slug) directly.
    first_scene_env = ""  # retained for the scene-context hint below
    scenes = led.get("scenes") if isinstance(led, dict) else None
    if isinstance(scenes, list) and scenes:
        first = scenes[0] if isinstance(scenes[0], dict) else {}
        first_scene_env = (
            _safe_str(first.get("env"))
            or _safe_str(first.get("description"))
        )
    if not descriptor and _brief:
        descriptor = _brief
        branch = "meta.story_brief"

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
    # Sprint C C5c (2026-05-15): wires meta.story_brief into the env-
    # prompt body. The brief is inserted between the env description
    # and the style_suffix tail per refinement section 6. Status is
    # logged so a failed reflection surfaces in the render log per
    # E-07 ("story_brief_status=...").
    from ..nodes._otr_story_brief_helpers import (
        get_story_brief_full,
        get_story_brief_status,
    )

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
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    elif isinstance(payload, list):
        tokens = payload
        meta = {}
    else:
        log.warning("[BatchFluxRender] unexpected script_json root %s", type(payload).__name__)
        return [f"{fallback}, {style_suffix}".rstrip(", ")]
    env_tokens = [t for t in tokens if isinstance(t, dict) and t.get("type") == "environment"]
    if not env_tokens:
        log.info("[BatchFluxRender] no environment tokens (total=%d); using fallback x1", len(tokens))
        return [f"{fallback}, {style_suffix}".rstrip(", ")]
    limit = max(1, min(batch_limit, len(env_tokens)))
    selected = env_tokens[:limit]
    brief = get_story_brief_full(meta)
    brief_status = get_story_brief_status(meta)
    prompts = []
    for token in selected:
        desc = (token.get("description") or "").strip()
        if not desc:
            desc = fallback
        parts = [desc]
        if brief:
            parts.append(brief)
        if style_suffix and style_suffix.strip():
            parts.append(style_suffix.strip())
        prompts.append(", ".join(parts))
    log.info(
        "[BatchFluxRender] queued %d env prompt(s) from %d available "
        "(story_brief_status=%s; brief_chars=%d)",
        len(prompts), len(env_tokens), brief_status, len(brief),
    )
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


def _try_fast_batch(*, positive_conds, prompts, model, vae,
                    empty_latent_cls, sampler, decoder, negative, seed, steps,
                    cfg, sampler_name, scheduler, width, height,
                    report_lines):
    """BUG-LOCAL-231 fix (2026-05-18): accept PRE-ENCODED
    ``positive_conds`` (already including guidance) instead of raw
    ``prompts + text_enc + clip + guidance_node + guidance``. Pre-fix
    the encoding happened here, AFTER ``execute()`` had already
    pinned the MODEL, which forced ComfyUI's model_management to
    partial-unload the pinned MODEL to fit T5XXL CLIP. Moving the
    encode into ``execute()`` before the pin lets CLIP get evicted
    before MODEL is pinned with full headroom. ``prompts`` is kept
    purely for logging (shot preview lines).
    """
    t0 = time.time()
    N = len(positive_conds)
    if N == 0:
        return None
    try:
        for i, pt in enumerate(prompts[:N]):
            preview = pt[:120] + ("..." if len(pt) > 120 else "")
            log.info("[BatchFluxRender] shot %d/%d seed=%d: %s",
                     i + 1, N, seed + i, preview)
        log.info(
            "[BatchFluxRender] fast_batch: using %d pre-encoded "
            "positive(s) for single KSampler call", N,
        )
    except Exception as exc:
        log.warning("[BatchFluxRender] fast_batch logging failed: %s", exc)
    batched_pos = _stack_conditioning(positive_conds)
    if batched_pos is None:
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
                "skip_env_stills": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "BUG-LOCAL-078 follow-up (2026-05-03 EVENING). "
                        "When True (default), SKIP the per-shot env-still "
                        "FLUX pass entirely. Returns a 1x16x16 placeholder "
                        "IMAGE so downstream OTR_SaveToEpisodeWorkspace "
                        "doesn't crash. The radio bookend is still "
                        "rendered in this node (separate code path) "
                        "regardless of this flag. Env stills are dead "
                        "code in the active pipeline after the BUG-078 "
                        "portrait fix: HuMo's tier 1 lookup hits the "
                        "new portraits/<char_id>_portrait.png so the "
                        "tier 4 env-still fallback never fires, and "
                        "VideoComposite's Phase A simple-pillarbox "
                        "branch hardcodes background_png=None so its "
                        "layered branch (which would consume env "
                        "stills) is dormant. Saves ~2-3 minutes of "
                        "FLUX time per episode. Set False to re-enable "
                        "the env-still pass if a future Phase C "
                        "layered backdrop mode revives the consumer."
                    ),
                }),
            },
        }

    def execute(self, model, clip, vae, script_json, batch_limit, seed, steps,
                cfg, sampler_name, scheduler, width, height, guidance,
                fallback_prompt=_DEFAULT_FALLBACK, style_suffix=_DEFAULT_STYLE_SUFFIX,
                freeze_seed=False, fast_batch=True,
                radio_bookend_prompt="",
                radio_bookend_seed=4242,
                skip_env_stills=True):
        t_start = time.time()
        prompts = _parse_env_prompts(script_json, batch_limit, fallback_prompt, style_suffix)

        # Sprint E E9 / M2: check meta.freeze_unload_ok. The cascade
        # stamps False if its finally-block unload_llm raised; that
        # means Mistral-Nemo may still be holding ~8 GB on the GPU.
        # FLUX FP8 needs ~12 GB free, so attempt one defensive unload
        # before pinning the FLUX MODEL to dodge OOM. Logged at WARN
        # so the soak tail surfaces the recovery attempt.
        try:
            import json as _json
            _meta_flag = None
            if script_json and script_json.strip().startswith("{"):
                _meta = (_json.loads(script_json).get("meta") or {})
                _meta_flag = _meta.get("freeze_unload_ok")
            if _meta_flag is False:
                log.warning(
                    "[BatchFluxRender] meta.freeze_unload_ok=False -- "
                    "cascade teardown reported unload_llm failure; "
                    "attempting defensive unload before FLUX MODEL pin"
                )
                try:
                    from nodes._otr_model_loader import unload_llm as _du
                    _du()
                except Exception as _exc:
                    log.warning(
                        "[BatchFluxRender] defensive unload_llm raised %r; "
                        "proceeding to FLUX load anyway (may OOM)", _exc,
                    )
        except Exception as _exc:
            # Reading meta is best-effort. A malformed script_json
            # surface (legacy list form) should not block FLUX render.
            log.debug("[BatchFluxRender] freeze_unload_ok read failed: %r", _exc)

        # BUG-LOCAL-231 fix (2026-05-18): the old "pin MODEL on GPU"
        # block that used to live here pinned BEFORE any CLIPTextEncode
        # ran. ComfyUI's model_management then had to partial-unload
        # the pinned MODEL to fit T5XXL CLIP (per-shot ~8.5 GB freed +
        # ~11 GB MODEL re-load before sampler, at 154 s/step). The new
        # order is:
        #   1. Encode negative
        #   2. Resolve bookend prompt + encode bookend positive
        #      (unless DISABLED widget sentinel)
        #   3. If NOT skip_env_stills: pre-encode the N per-shot
        #      positive conditionings
        #   4. mm.free_memory(...) -- explicit eviction so CLIP gets
        #      freed before MODEL pin (the "natural offload" assumption
        #      isn't reliable; pin alone may not free CLIP, see Jeffrey
        #      2026-05-18 push-back)
        #   5. mm.load_models_gpu([model]) -- pin MODEL with full
        #      headroom (force_full_load dropped per Jeffrey 2026-05-18:
        #      ComfyUI ignores it during T5XXL load anyway, and normal
        #      offload behavior is preferable if VRAM tightens elsewhere)
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

        # ---- BUG-LOCAL-231 fix step 2: pre-encode bookend positive ----
        # Resolve the bookend prompt + encode now, BEFORE the pin, so
        # CLIP gets evicted naturally and the explicit free_memory()
        # below releases its VRAM. DISABLED widget yields a None
        # `bookend_positive` and the bookend render is skipped later.
        widget_str = (radio_bookend_prompt or "").strip()
        bookend_positive = None
        bookend_resolved = None
        bookend_prompt_source = None
        bookend_ledger_p = None
        bookend_episode_id = None
        bookend_led = None
        if widget_str.upper() == "DISABLED":
            log.info(
                "[BatchFluxRender] radio bookend DISABLED via widget "
                "sentinel (no render attempted)"
            )
        else:
            (bookend_resolved, bookend_prompt_source, bookend_ledger_p,
             bookend_episode_id, bookend_led) = (
                self._resolve_radio_bookend_prompt(widget_str)
            )
            if bookend_resolved is not None:
                try:
                    bookend_positive = text_enc.encode(clip, bookend_resolved)[0]
                    if guidance_node is not None:
                        bookend_positive = (
                            guidance_node.append(bookend_positive, guidance)[0]
                        )
                except Exception as exc:
                    log.warning(
                        "[BatchFluxRender] bookend pre-encode failed (%s); "
                        "bookend render will be skipped", exc,
                    )
                    bookend_positive = None

        # ---- BUG-LOCAL-231 fix step 3: pre-encode per-shot positives ----
        # When skip_env_stills=True, the N prompts are NOT rendered, so
        # we don't need to pre-encode them. Otherwise pre-encode them
        # all upfront so CLIP can be evicted before the MODEL pin.
        positive_conds = []
        if not skip_env_stills and prompts:
            try:
                for i, pt in enumerate(prompts):
                    pc = text_enc.encode(clip, pt)[0]
                    if guidance_node is not None:
                        pc = guidance_node.append(pc, guidance)[0]
                    positive_conds.append(pc)
                log.info(
                    "[BatchFluxRender] pre-encoded %d per-shot positive(s)",
                    len(positive_conds),
                )
            except Exception as exc:
                log.warning(
                    "[BatchFluxRender] per-shot pre-encode failed at i=%d "
                    "(%s); falling back to per-shot encode inside serial loop",
                    len(positive_conds), exc,
                )
                positive_conds = []  # fall back to per-shot encoding

        # ---- BUG-LOCAL-231 fix step 4: nuclear eviction + pin ----
        # Original fix used mm.free_memory(11.5 GB, [model.load_device]).
        # 2026-05-18 post-fix smoke (HEAD 36bcfc0) showed mm.free_memory
        # was NOT sufficient: sampler step 1 still at 139 s/it (vs target
        # 10-15 s/step), LHM D3D Shared 974 MB during sampler (vs <200 MB
        # target). Jeffrey's pre-authorized escalation: swap to Option B
        # mm.unload_all_models() per the acceptance gate rule "if LHM
        # shows D3D Shared > 200 MB during sampler the eviction didn't
        # work and the fix needs to escalate to option B
        # (unload_all_models)."
        #
        # unload_all_models() unconditionally evicts every model from
        # the model_management registry -- CLIP, T5XXL, FLUX MODEL
        # (which we re-load in the next line), residual audio caches,
        # everything. Then load_models_gpu pins MODEL freshly, alone,
        # with full VRAM headroom for the sampler. The 0.5-1 s of
        # re-load tax is amortized across all 20 sampler steps.
        try:
            import comfy.model_management as mm  # type: ignore
            import gc as _gc
            import torch as _torch  # type: ignore
            try:
                mm.unload_all_models()
                # BUG-07.03 invariant (Bug Bible regression): every
                # unload_all_models() call must be paired with
                # gc.collect() + torch.cuda.empty_cache(). Dereferencing
                # the model registry does NOT free VRAM; PyTorch's
                # caching allocator holds reserved blocks until
                # empty_cache() returns them. Without this pairing,
                # unload_all_models() is cosmetic only and the next
                # load_models_gpu() will see the SAME memory pressure.
                _gc.collect()
                _torch.cuda.empty_cache()
                log.info(
                    "[BatchFluxRender] unload_all_models() + "
                    "gc.collect() + empty_cache() complete (nuclear "
                    "eviction before MODEL pin, BUG-LOCAL-231 Option "
                    "B escalation)"
                )
            except Exception as exc:
                log.warning(
                    "[BatchFluxRender] mm.unload_all_models() raised %r; "
                    "proceeding to load_models_gpu without explicit "
                    "eviction (sampler may thrash)", exc,
                )
            mm.load_models_gpu([model])
            log.info("[BatchFluxRender] pinned MODEL via load_models_gpu")
        except Exception as exc:
            log.debug("[BatchFluxRender] pre-pin skipped: %s", exc)

        report_lines = [
            f"BatchFluxRender: {len(prompts)} shot(s) @ {width}x{height}, "
            f"steps={steps}, cfg={cfg}, guidance={guidance}, "
            f"sampler={sampler_name}/{scheduler}, "
            f"mode={'fast_batch' if fast_batch else 'serial'}",
        ]

        # BUG-LOCAL-078 follow-up (2026-05-03 EVENING): if skip_env_stills
        # is True (default), bypass the per-shot env-still FLUX pass
        # entirely. The radio bookend is still rendered below (separate
        # code path); HuMo's per-cast portraits come from the new
        # OTR_BatchFluxPortraitRender node. Env stills are dead code in
        # the active pipeline after BUG-078 wires up tier 1 portraits.
        # Saves ~2-3 minutes of FLUX time per episode. Returns a tiny
        # placeholder IMAGE so OTR_SaveToEpisodeWorkspace doesn't crash.
        if skip_env_stills:
            log.info(
                "[BatchFluxRender] skip_env_stills=True -- bypassing "
                "per-shot env-still FLUX pass; rendering radio bookend "
                "only (BUG-LOCAL-078 follow-up)"
            )
            report_lines.append(
                "skip_env_stills=True; bypassed env-still pass "
                "(saved ~2-3 min FLUX time)"
            )
            if bookend_positive is None:
                # DISABLED widget OR pre-encode failed -- skip the
                # bookend render entirely (matches pre-fix behavior of
                # _render_radio_bookend_step's DISABLED short-circuit).
                report_lines.append("  radio_bookend: DISABLED (widget)")
            else:
                try:
                    _RADIO_BOOKEND_W = 1248
                    _RADIO_BOOKEND_H = 720
                    self._render_and_save_radio_bookend(
                        positive=bookend_positive,
                        resolved_prompt=bookend_resolved,
                        prompt_source=bookend_prompt_source,
                        ledger_p=bookend_ledger_p,
                        episode_id=bookend_episode_id,
                        led=bookend_led,
                        model=model, vae=vae,
                        empty_latent_cls=empty_latent_cls, sampler=sampler,
                        decoder=decoder, negative=negative,
                        seed=int(radio_bookend_seed), steps=steps, cfg=cfg,
                        sampler_name=sampler_name, scheduler=scheduler,
                        width=_RADIO_BOOKEND_W, height=_RADIO_BOOKEND_H,
                        report_lines=report_lines,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.exception(
                        "[BatchFluxRender] radio bookend render failed in "
                        "skip_env_stills mode: %s", exc,
                    )
                    report_lines.append(
                        f"  radio bookend FAILED in skip mode: {exc}"
                    )
            import torch as _torch  # type: ignore
            placeholder = _torch.zeros((1, 16, 16, 3), dtype=_torch.float32)
            elapsed_ms = int((time.time() - t_start) * 1000)
            report_lines.append(
                f"BatchFluxRender done in {elapsed_ms} ms (skip mode)"
            )
            return (placeholder, "\n".join(report_lines))

        # FAST BATCH PATH
        # BUG-LOCAL-231 fix: only run fast_batch if pre-encoding
        # succeeded; otherwise fall through to the serial loop
        # (which has its own per-shot encode fallback).
        if (fast_batch and len(prompts) > 1
                and len(positive_conds) == len(prompts)):
            batched_result = _try_fast_batch(
                positive_conds=positive_conds, prompts=prompts,
                model=model, vae=vae,
                empty_latent_cls=empty_latent_cls, sampler=sampler,
                decoder=decoder, negative=negative, seed=seed, steps=steps,
                cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                width=width, height=height,
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
                    bookend_positive=bookend_positive,
                    bookend_resolved=bookend_resolved,
                    bookend_prompt_source=bookend_prompt_source,
                    bookend_ledger_p=bookend_ledger_p,
                    bookend_episode_id=bookend_episode_id,
                    bookend_led=bookend_led,
                    radio_bookend_seed=radio_bookend_seed,
                    model=model, vae=vae,
                    empty_latent_cls=empty_latent_cls, sampler=sampler,
                    decoder=decoder, negative=negative,
                    steps=steps, cfg=cfg,
                    sampler_name=sampler_name, scheduler=scheduler,
                    report_lines=report_lines,
                )
                return (image_batch, "\n".join(report_lines))
            log.info("[BatchFluxRender] fast_batch fell through, running serial loop")
            report_lines.append("  fast_batch failed -- fell through to serial loop")

        # SERIAL LOOP PATH
        # BUG-LOCAL-231 fix: use pre-encoded positive_conds[i] when
        # available (the new path). Fall back to per-shot encode if
        # pre-encoding failed earlier (positive_conds is empty).
        images = []
        for shot_index, prompt_text in enumerate(prompts):
            shot_t0 = time.time()
            shot_seed = seed if freeze_seed else (seed + shot_index)
            preview = prompt_text[:120] + ("..." if len(prompt_text) > 120 else "")
            log.info("[BatchFluxRender] shot %d/%d seed=%d: %s",
                     shot_index + 1, len(prompts), shot_seed, preview)
            try:
                if shot_index < len(positive_conds):
                    positive = positive_conds[shot_index]
                else:
                    # Fallback path (pre-encode failed) -- per-shot
                    # encode happens AFTER the MODEL pin; expect a
                    # CLIP/MODEL swap cycle here. Logged at WARN so
                    # the soak tail surfaces the degraded path.
                    log.warning(
                        "[BatchFluxRender] shot %d: pre-encode missing, "
                        "encoding inside loop (sampler may thrash)",
                        shot_index + 1,
                    )
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
        bookend_positive,
        bookend_resolved,
        bookend_prompt_source,
        bookend_ledger_p,
        bookend_episode_id,
        bookend_led,
        radio_bookend_seed,
        model, vae,
        empty_latent_cls, sampler,
        decoder, negative,
        steps, cfg,
        sampler_name, scheduler,
        report_lines,
    ):
        """Render + stamp the radio still.  Called from BOTH the
        fast_batch return path and the serial-loop return path
        (BUG-LOCAL-127 fix 2026-05-01).

        BUG-LOCAL-231 fix (2026-05-18): accepts the pre-encoded
        ``bookend_positive`` + already-resolved data instead of the
        old ``radio_bookend_prompt`` widget value + ``text_enc + clip``
        + ``guidance_node + guidance``. Resolution + encode happen in
        ``execute()`` BEFORE the MODEL pin so CLIP gets evicted
        cleanly. ``bookend_positive=None`` means DISABLED widget OR
        pre-encode failed; we skip the render in that case.

        Never raises -- a render failure is logged with full
        traceback (log.exception) and noted in report_lines, but the
        outer execute() always returns the main image batch.
        """
        if bookend_positive is None:
            log.info(
                "[BatchFluxRender] radio bookend DISABLED via widget "
                "sentinel OR pre-encode failed (no render attempted)"
            )
            report_lines.append("  radio_bookend: DISABLED (widget)")
            return
        log.info(
            "[BatchFluxRender] radio bookend render (seed=%d, source=%s)",
            int(radio_bookend_seed), bookend_prompt_source,
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
                positive=bookend_positive,
                resolved_prompt=bookend_resolved,
                prompt_source=bookend_prompt_source,
                ledger_p=bookend_ledger_p,
                episode_id=bookend_episode_id,
                led=bookend_led,
                model=model, vae=vae,
                empty_latent_cls=empty_latent_cls, sampler=sampler,
                decoder=decoder, negative=negative,
                seed=int(radio_bookend_seed), steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                width=_RADIO_BOOKEND_W, height=_RADIO_BOOKEND_H,
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
    def _resolve_radio_bookend_prompt(prompt_text):
        """BUG-LOCAL-231 fix (2026-05-18): split out the bookend
        prompt-resolution path so execute() can resolve + pre-encode
        BEFORE pinning the MODEL. Pre-fix this work happened inside
        _render_and_save_radio_bookend AFTER the pin was held, which
        forced ComfyUI's model_management to partial-unload the
        pinned MODEL to fit T5XXL CLIP (~8.5 GB freed + ~11 GB re-load
        per shot at 154 s/step). Resolving + encoding before the pin
        lets CLIP load, encode, and yield naturally; the explicit
        free_memory + load_models_gpu in execute() then pins MODEL
        with full headroom.

        Returns:
          (resolved_prompt, prompt_source, ledger_p, episode_id, led)
          or (None, None, None, None, None) on helper-import failure
          (caller should skip the bookend render).
        """
        from pathlib import Path

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
            return (None, None, None, None, None)

        # BUG-LOCAL-021 (Phase G, 2026-05-03) singleton lookup.
        try:
            _led_singleton = _PROD_LEDGER.get_ledger()
            ledger_p = Path(_led_singleton.path)
            if not ledger_p.exists():
                log.warning(
                    "[BatchFluxRender] radio still: singleton path %s does "
                    "not exist on disk; falling back to mtime walker",
                    ledger_p,
                )
                ledger_p = _OTRL.find_most_recent_ledger(
                    [_OTRP.otr_episodes_root()]
                )
        except Exception as _exc:
            log.warning(
                "[BatchFluxRender] radio still: singleton lookup failed (%s); "
                "falling back to mtime walker", _exc,
            )
            ledger_p = _OTRL.find_most_recent_ledger(
                [_OTRP.otr_episodes_root()]
            )
        led = None
        episode_id = "episode"
        if ledger_p is not None:
            led = _OTRL.load_ledger_safe(ledger_p)
            if led is not None:
                episode_id = (led.get("episode_id") or "episode").strip()
        log.info(
            "[BatchFluxRender] radio bookend stage: ledger=%s "
            "episode_id=%s (will save as radio_bookend_%s.png)",
            ledger_p.name if ledger_p else "<none>",
            episode_id,
            episode_id,
        )

        # Resolve prompt: empty widget -> dynamic; non-empty -> verbatim.
        widget_prompt = (prompt_text or "").strip()
        if widget_prompt:
            resolved_prompt = widget_prompt
            prompt_source = "override"
        else:
            resolved_prompt = _build_dynamic_radio_prompt(led)
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
        return (resolved_prompt, prompt_source, ledger_p, episode_id, led)

    @staticmethod
    def _render_and_save_radio_bookend(
        positive, resolved_prompt, prompt_source, ledger_p, episode_id, led,
        model, vae,
        empty_latent_cls, sampler, decoder, negative,
        seed, steps, cfg, sampler_name, scheduler,
        width, height, report_lines,
    ):
        """Step 1: render the radio still (one extra FLUX call) and
        save it to ``output/otr/stills/radio_bookend_<ep_id>.png``.

        BUG-LOCAL-231 fix (2026-05-18): prompt resolution + CLIP
        encode moved into ``_resolve_radio_bookend_prompt`` and
        ``execute()`` respectively; this function now receives a
        pre-encoded ``positive`` conditioning + already-resolved
        ledger data and runs sampler + decoder + save only. Pre-fix
        the resolution + encode happened here AFTER ``execute()``
        had already pinned the MODEL via ``load_models_gpu``, which
        forced ComfyUI's ``model_management`` to partial-unload the
        pinned MODEL to fit T5XXL CLIP (per-shot ~8.5 GB free +
        ~11 GB re-load cycle at 154 s/step). Resolving + encoding
        in ``execute()`` before the explicit free_memory + pin
        breaks that cycle.

        Stamps both ``ledger.radio_bookend_path`` (top-level) AND
        ``ledger.meta.radio_bookend_path`` for belt-and-suspenders.
        Downstream (BatchHumoRender) reads the radio still as the
        I2V reference image for any non-dialogue HuMo clip
        (announcer, music_*, sfx).
        """
        import numpy as np  # type: ignore
        from pathlib import Path
        from PIL import Image  # type: ignore

        # _otr_paths is needed for the save-path lookup below;
        # _otr_ledger is needed for the stamp at the end. Both are
        # imported on the sys.path that _resolve_radio_bookend_prompt
        # already extended, so this is a cheap re-import.
        try:
            import sys as _sys
            _NODES_DIR = Path(__file__).resolve().parents[1] / "nodes"
            if str(_NODES_DIR) not in _sys.path:
                _sys.path.insert(0, str(_NODES_DIR))
            import _otr_paths as _OTRP  # type: ignore
            import _otr_ledger as _OTRL  # type: ignore
        except Exception as exc:
            log.warning(
                "[BatchFluxRender] radio still: helper import failed "
                "in save path (%s)", exc,
            )
            return

        # Pre-resolved data + pre-encoded positive are passed in by
        # the caller (execute() in skip_env_stills mode, or
        # _render_radio_bookend_step in fast_batch / serial mode).
        # We just sample + decode + save here.
        t0 = time.time()
        latent = empty_latent_cls.generate(width, height, 1)[0]

        # BUG-LOCAL-231 alt-c/e/f telemetry (Jeffrey 2026-05-19 09:30):
        # snapshot cudnn / cublas / nvsmi state at sampler entry and
        # start a background poller that logs LHM + nvsmi every 5s
        # during sampler.sample. The 3-smoke clean-state battery
        # (08:41-09:26 today) falsified alt-a -- fire-time tuples were
        # identical across 3 cold launches but sampler step 1 spanned
        # 4x (42.38 / >90 / 158.86 s/it). The variance is downstream
        # of fire; these signals catch it where it manifests.
        _log_flux_sampler_precheck()
        _poller = _FluxSamplerPoller(interval_s=5.0)
        _poller.start()
        try:
            samples = sampler.sample(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, latent, 1.0,
            )[0]
        finally:
            _poller.stop()
        img = decoder.decode(vae, samples)[0]
        # img shape: [B, H, W, C] in 0..1 float

        # BUG-LOCAL-028 fix (2026-05-03): pass episode_id so the radio
        # bookend lands in the per-episode workspace at
        # ``output/otr/episodes/<ep>/stills/radio_bookend_<ep>.png``
        # instead of the legacy ``output/otr/_legacy_stills/`` flat dir.
        # Without the arg, ``otr_stills_dir()`` falls back to
        # ``_legacy_stills/`` per ``nodes/_otr_paths.py:208-218``, which
        # orphans the radio still from the per-episode workspace and
        # leaves VideoComposite (which reads from the per-episode
        # ``stills/`` subdir) with no scenery layer. Same Phase G blast
        # radius as BUG-LOCAL-021 — that fix updated the LEDGER discovery
        # to use the singleton; this fix updates the SAVE PATH to use the
        # episode_id we already resolved (line 768/772 above).
        stills_dir = _OTRP.otr_stills_dir(episode_id)
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
