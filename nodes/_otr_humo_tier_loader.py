"""nodes/_otr_humo_tier_loader.py -- OTR_HuMoTierLoader.

Single-node HuMo model-tier loader. Picks one of three tiers and loads
the full HuMo stack -- diffusion model + umt5 text encoder + wan VAE +
Whisper audio encoder -- ready to feed OTR_BatchHumoRender.

Decision context (BUG-LOCAL-265, round-robin 2026-05-24, Option C):
OTR ships open-source to varied consumer GPUs. HuMo-17B/14B fp8
(~16.5 GB) thrashes on a 16 GB card inside the full pipeline -- it
offloads ~10 GB to system RAM and renders at 140-279 s/it (one clip
took 3h43m in one soak). HuMo-1.7B fp16 (3.3 GB) is fully resident with
zero offload and the operator rated its 20-step output "acceptable".
So OTR ships HuMo-1.7B as the default and keeps the big model opt-in:

  low_vram_default  -- HuMo-1.7B fp16, 20 steps, cfg 5.0, no distill
                       LoRA (the lightx2v distill LoRA is 14B-only).
                       The shipped default; broad hardware compat,
                       no thrash in or out of the pipeline.
  high_quality      -- HuMo-17B/14B fp8, 6 steps, cfg 1.0, lightx2v
                       14B distill LoRA. Best quality; needs a 16 GB+
                       card AND large system RAM for the ~10 GB offload.
  experimental_gguf -- HuMo-17B GGUF, 6 steps, cfg 1.0, distill LoRA.
                       Advanced only -- the GGUF per-step dequant tax
                       made it SLOWER than fp8 in the bracket test and
                       its quality was not judged. Not a default.

Auto-downgrade (the hard rule): when a high tier is selected but free
VRAM -- measured AFTER the Lever-1 pipeline-residue free -- is below
``vram_safety_threshold_gb``, the node either downgrades to
low_vram_default (``auto_downgrade`` ON) or stops with a clear error
(``auto_downgrade`` OFF). Either way the user never unknowingly hits
the 3h43m in-pipeline thrash path.

Design:
  - Mirrors OTR_DeferredCheckpointLoader: a thin OTR wrapper that
    delegates real loading to ComfyUI's own loader node classes
    (resolved from ``nodes.NODE_CLASS_MAPPINGS``, invoked via each
    class's ``FUNCTION`` attribute -- so no method name is hardcoded).
    The MODEL/CLIP/VAE objects it produces are ComfyUI ModelPatcher
    objects owned by ``comfy.model_management`` -- so ComfyUI's model
    manager can evict them, unlike OTR's out-of-band LLM / Bark caches.
    (The actual residue free + CUDA cache flush is delegated to
    ``_otr_vram_levers.free_otr_pipeline_residue`` -- this file does no
    VRAM teardown itself.)
  - Tiering lives HERE, upstream, not inside OTR_BatchHumoRender.
    BatchHumoRender keeps its clean "take a pre-loaded MODEL/CLIP/VAE/
    AUDIO_ENCODER" surface. This node also emits the tier's ``steps``
    and ``cfg`` so a single tier switch reconfigures the whole render.
  - Prime Directive 6: no ``model_id`` widget. The HuMo model files
    are picked via per-tier model-file dropdowns; ``NON_LLM_MODEL_
    WIDGET_OK`` marks them as model-file pickers, not LLM model ids
    (same opt-in OTR_DeferredCheckpointLoader uses for ``ckpt_name``).
  - The tier-resolution + per-tier config logic is split into pure
    static helpers (``_resolve_tier``, ``_tier_config``) so it is
    unit-testable with no ComfyUI / torch / model-file dependency.
"""
from __future__ import annotations

import logging
import math
import sys as _sys
from pathlib import Path

# Sibling-module bootstrap (see _otr_vram_levers / batch_humo_render for
# the rationale): makes _otr_vram_levers reachable as a top-level module
# when this file is loaded standalone by the test harness.
_NODES_DIR = Path(__file__).resolve().parent
if str(_NODES_DIR) not in _sys.path:
    _sys.path.insert(0, str(_NODES_DIR))

# Lever-1 residue-freer. Relative import in production ComfyUI; flat
# import when this file is loaded standalone by the test harness.
# _otr_vram_levers does only lazy (in-function) torch/comfy imports, so
# importing it at module-load time is safe headless.
try:
    from ._otr_vram_levers import free_otr_pipeline_residue  # type: ignore
except ImportError:  # pragma: no cover - exercised only standalone
    from _otr_vram_levers import free_otr_pipeline_residue  # type: ignore

log = logging.getLogger("OTR.nodes._otr_humo_tier_loader")


# ---------------------------------------------------------------------------
# Canonical file defaults (the bracket-test stack -- BUG-LOCAL-265)
# ---------------------------------------------------------------------------

_DEFAULT_HUMO_1P7B = "humo_1.7B_fp16.safetensors"
_DEFAULT_HUMO_HIGHQ = "humo_17B_fp8_e4m3fn.safetensors"
_DEFAULT_HUMO_GGUF = "Wan2_1-HuMo-17B_Q5_K_M.gguf"
_DEFAULT_CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
_DEFAULT_VAE = "wan_2.1_vae.safetensors"
_DEFAULT_AUDIO_ENCODER = "whisper_large_v3_fp16.safetensors"
_DEFAULT_DISTILL_LORA = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"

# ModelSamplingSD3 shift -- 8 for the Wan 2.1 HuMo stack on every tier
# (matches workflows/humo_smoke_1p7b.json node 48 and the 14B/GGUF chain).
_MODEL_SAMPLING_SHIFT = 8.0

# Per-tier render config. Pure data -- no I/O. _tier_config returns a
# copy so callers cannot mutate the table.
_TIER_TABLE = {
    "low_vram_default": {
        "use_gguf": False,
        "use_lora": False,
        "steps": 20,
        "cfg": 5.0,
        "label": "HuMo-1.7B fp16 (low-VRAM default)",
    },
    "high_quality": {
        "use_gguf": False,
        "use_lora": True,
        "steps": 6,
        "cfg": 1.0,
        "label": "HuMo-17B/14B fp8 + lightx2v 14B distill LoRA (high quality)",
    },
    "experimental_gguf": {
        "use_gguf": True,
        "use_lora": True,
        "steps": 6,
        "cfg": 1.0,
        "label": "HuMo-17B GGUF + lightx2v 14B distill LoRA (experimental)",
    },
}

_TIER_CHOICES = ["low_vram_default", "high_quality", "experimental_gguf"]


def _file_list(folder_key: str, default_name: str) -> list:
    """Model-file dropdown options for ``folder_key``.

    Returns the live ComfyUI folder listing; falls back to a one-element
    ``[default_name]`` list when ComfyUI's ``folder_paths`` is absent or
    the folder is empty (headless test runs -- the conftest stub returns
    an empty list). ``default_name`` is prepended when missing so the
    canonical file is always offerable. Never raises.
    """
    try:
        import folder_paths  # type: ignore
        names = list(folder_paths.get_filename_list(folder_key) or [])
    except Exception:  # noqa: BLE001
        names = []
    if not names:
        return [default_name]
    if default_name not in names:
        names = [default_name] + names
    return names


class HuMoTierLoader:
    """Load the HuMo stack for one of three VRAM/quality tiers."""

    CATEGORY = "OTR/loaders"
    FUNCTION = "load"
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "AUDIO_ENCODER", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "audio_encoder", "steps", "cfg", "report")
    OUTPUT_NODE = False
    # Opt-in marker (same as OTR_DeferredCheckpointLoader.ckpt_name):
    # the model-file picker widgets below are model-FILE pickers, not
    # LLM model-id widgets, so the two-slot LLM rule does not apply.
    NON_LLM_MODEL_WIDGET_OK = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tier": (_TIER_CHOICES, {
                    "default": "low_vram_default",
                    "tooltip": (
                        "HuMo model tier (BUG-LOCAL-265 Option C).\n"
                        "  low_vram_default -- HuMo-1.7B fp16, 20 steps, "
                        "no distill LoRA. Fully resident on a 16 GB card, "
                        "zero offload, no thrash. The shipped default.\n"
                        "  high_quality -- HuMo-17B/14B fp8, 6 steps, "
                        "lightx2v distill LoRA. Best quality; needs a "
                        "16 GB+ card AND large system RAM.\n"
                        "  experimental_gguf -- HuMo-17B GGUF. Advanced "
                        "only: GGUF dequant tax made it slower than fp8 "
                        "in the bracket test; quality unjudged."
                    ),
                }),
                "humo_1p7b_model": (
                    _file_list("diffusion_models", _DEFAULT_HUMO_1P7B),
                    {"default": _DEFAULT_HUMO_1P7B,
                     "tooltip": "HuMo-1.7B fp16 diffusion model "
                                "(low_vram_default tier)."},
                ),
                "humo_highq_model": (
                    _file_list("diffusion_models", _DEFAULT_HUMO_HIGHQ),
                    {"default": _DEFAULT_HUMO_HIGHQ,
                     "tooltip": "HuMo-17B/14B fp8 diffusion model "
                                "(high_quality tier)."},
                ),
                "humo_gguf_model": (
                    _file_list("diffusion_models", _DEFAULT_HUMO_GGUF),
                    {"default": _DEFAULT_HUMO_GGUF,
                     "tooltip": "HuMo-17B GGUF diffusion model "
                                "(experimental_gguf tier). Requires the "
                                "ComfyUI-GGUF custom node."},
                ),
                "clip_model": (
                    _file_list("text_encoders", _DEFAULT_CLIP),
                    {"default": _DEFAULT_CLIP,
                     "tooltip": "umt5_xxl text encoder (CLIPLoader, "
                                "type=wan). Same file for every tier."},
                ),
                "vae_model": (
                    _file_list("vae", _DEFAULT_VAE),
                    {"default": _DEFAULT_VAE,
                     "tooltip": "wan 2.1 VAE. Same file for every tier."},
                ),
                "audio_encoder_model": (
                    _file_list("audio_encoders", _DEFAULT_AUDIO_ENCODER),
                    {"default": _DEFAULT_AUDIO_ENCODER,
                     "tooltip": "Whisper Large v3 audio encoder. Same "
                                "file for every tier."},
                ),
                "distill_lora": (
                    _file_list("loras", _DEFAULT_DISTILL_LORA),
                    {"default": _DEFAULT_DISTILL_LORA,
                     "tooltip": "lightx2v 14B distill LoRA. Applied on the "
                                "high_quality + experimental_gguf tiers "
                                "only -- it is 14B-only and is NOT applied "
                                "on the 1.7B low_vram_default tier."},
                ),
                "auto_downgrade": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "When ON and a high tier lacks VRAM headroom "
                        "(see vram_safety_threshold_gb), fall back to "
                        "low_vram_default (HuMo-1.7B) with a warning. "
                        "When OFF, raise a clear error instead. Either "
                        "way the user never silently hits the 3h43m "
                        "in-pipeline thrash path (BUG-LOCAL-265)."
                    ),
                }),
                "vram_safety_threshold_gb": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.0,
                    "max": 48.0,
                    "step": 0.5,
                    "tooltip": (
                        "Minimum free VRAM (GB), measured AFTER the "
                        "Lever-1 pipeline-residue free, required to load "
                        "a high tier. Below this the auto_downgrade rule "
                        "fires. Default 10.0 GB: HuMo-17B keeps ~10 GB "
                        "resident; less headroom means heavy offload to "
                        "system RAM and the thrash path. Lower it to "
                        "force the high tier on a tighter card."
                    ),
                }),
            },
            "optional": {
                "unload_gate": ("STRING", {
                    "forceInput": True,
                    "tooltip": (
                        "Optional ordering gate. Wire OTR_UnloadAll's "
                        "unload_done output here so ComfyUI's executor "
                        "topo-sorts this loader AFTER the FLUX -> "
                        "UnloadAll branch -- the auto_downgrade VRAM "
                        "check then reads a post-cleanup card state. "
                        "Value is ignored; only the dependency edge "
                        "matters."
                    ),
                }),
            },
        }

    # -- pure helpers (no ComfyUI / torch / model-file dependency) --------

    @staticmethod
    def _tier_config(resolved_tier: str) -> dict:
        """Per-tier render config for ``resolved_tier``. Pure.

        Returns a fresh dict ``{use_gguf, use_lora, steps, cfg, label}``.
        Raises ``ValueError`` on an unknown tier.
        """
        spec = _TIER_TABLE.get(resolved_tier)
        if spec is None:
            raise ValueError(
                f"OTR_HuMoTierLoader: unknown tier {resolved_tier!r}; "
                f"expected one of {_TIER_CHOICES}."
            )
        return dict(spec)

    @staticmethod
    def _resolve_tier(
        tier: str,
        auto_downgrade: bool,
        threshold_gb: float,
        free_vram_gb,
    ) -> tuple:
        """Resolve the effective tier, applying the auto-downgrade rule.

        Pure -- ``free_vram_gb`` is passed in (the caller measures it),
        so this is fully unit-testable. Returns ``(resolved_tier,
        note)``. Raises ``RuntimeError`` when a high tier lacks VRAM
        headroom AND ``auto_downgrade`` is False.
        """
        if tier == "low_vram_default":
            return "low_vram_default", (
                "tier 'low_vram_default' (HuMo-1.7B) -- fits any 16 GB "
                "card, no VRAM check required."
            )
        if tier not in _TIER_CHOICES:
            raise ValueError(
                f"OTR_HuMoTierLoader: unknown tier {tier!r}; expected one "
                f"of {_TIER_CHOICES}."
            )

        # High tier (high_quality / experimental_gguf): gate on VRAM.
        try:
            free_val = float(free_vram_gb)
        except (TypeError, ValueError):
            free_val = float("nan")
        if not math.isfinite(free_val):
            return tier, (
                f"tier '{tier}': VRAM probe unavailable (headless or no "
                f"CUDA) -- cannot verify headroom; loading as requested."
            )
        if free_val >= float(threshold_gb):
            return tier, (
                f"tier '{tier}' cleared: {free_val:.1f} GB free VRAM >= "
                f"{float(threshold_gb):.1f} GB safety threshold."
            )

        # Insufficient headroom.
        thrash = (
            "HuMo-17B offloads ~10 GB to system RAM and renders at "
            "140-279 s/it -- one clip took 3h43m (BUG-LOCAL-265)"
        )
        if auto_downgrade:
            return "low_vram_default", (
                f"AUTO-DOWNGRADE: tier '{tier}' needs >= "
                f"{float(threshold_gb):.1f} GB free VRAM but only "
                f"{free_val:.1f} GB is free after the pipeline-residue "
                f"clean. Fell back to low_vram_default (HuMo-1.7B) to "
                f"avoid the in-pipeline thrash path -- {thrash}. Free "
                f"VRAM upstream or lower vram_safety_threshold_gb to "
                f"force the high tier."
            )
        raise RuntimeError(
            f"OTR_HuMoTierLoader: tier '{tier}' needs >= "
            f"{float(threshold_gb):.1f} GB free VRAM but only "
            f"{free_val:.1f} GB is free after the pipeline-residue clean. "
            f"Loading it now would thrash -- {thrash}. Enable "
            f"auto_downgrade to fall back to HuMo-1.7B, switch tier to "
            f"low_vram_default, or free VRAM upstream."
        )

    # -- ComfyUI loader-node invocation -----------------------------------

    @staticmethod
    def _invoke(node_type: str, **kwargs):
        """Resolve a ComfyUI node class and call it via its FUNCTION.

        Reads ``FUNCTION`` off the class at runtime, so no method name
        is hardcoded. Raises a clear ``RuntimeError`` when the node is
        not registered (e.g. ComfyUI-GGUF not installed).
        """
        try:
            import nodes as _comfy_nodes  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"OTR_HuMoTierLoader: cannot import ComfyUI 'nodes' "
                f"module to resolve {node_type!r}: {exc}"
            )
        mapping = getattr(_comfy_nodes, "NODE_CLASS_MAPPINGS", {}) or {}
        cls = mapping.get(node_type)
        if cls is None:
            extra = (
                "Install the ComfyUI-GGUF custom node (city96) to use "
                "the experimental_gguf tier."
                if node_type == "UnetLoaderGGUF"
                else "It is a core ComfyUI node required to load the "
                     "HuMo stack."
            )
            raise RuntimeError(
                f"OTR_HuMoTierLoader: ComfyUI node {node_type!r} is not "
                f"registered in NODE_CLASS_MAPPINGS. {extra}"
            )
        inst = cls()
        fn_name = getattr(cls, "FUNCTION", None) or "execute"
        fn = getattr(inst, fn_name, None)
        if fn is None:
            raise RuntimeError(
                f"OTR_HuMoTierLoader: node {node_type!r} declares "
                f"FUNCTION={fn_name!r} but has no such method."
            )
        out = fn(**kwargs)
        if not isinstance(out, (tuple, list)) or not out:
            raise RuntimeError(
                f"OTR_HuMoTierLoader: node {node_type!r} returned {out!r}; "
                f"expected a non-empty tuple."
            )
        return out

    def _load_diffusion_model(self, file_name: str, use_gguf: bool):
        if use_gguf:
            return self._invoke("UnetLoaderGGUF", unet_name=file_name)[0]
        return self._invoke(
            "UNETLoader", unet_name=file_name, weight_dtype="default",
        )[0]

    def _apply_distill_lora(self, model, lora_name: str):
        return self._invoke(
            "LoraLoaderModelOnly", model=model, lora_name=lora_name,
            strength_model=1.0,
        )[0]

    def _apply_model_sampling(self, model):
        return self._invoke(
            "ModelSamplingSD3", model=model, shift=_MODEL_SAMPLING_SHIFT,
        )[0]

    def _load_clip(self, clip_name: str):
        return self._invoke("CLIPLoader", clip_name=clip_name, type="wan")[0]

    def _load_vae(self, vae_name: str):
        return self._invoke("VAELoader", vae_name=vae_name)[0]

    def _load_audio_encoder(self, audio_encoder_name: str):
        return self._invoke(
            "AudioEncoderLoader", audio_encoder_name=audio_encoder_name,
        )[0]

    # -- execute ----------------------------------------------------------

    def load(
        self,
        tier: str,
        humo_1p7b_model: str,
        humo_highq_model: str,
        humo_gguf_model: str,
        clip_model: str,
        vae_model: str,
        audio_encoder_model: str,
        distill_lora: str,
        auto_downgrade: bool,
        vram_safety_threshold_gb: float,
        unload_gate=None,
    ):
        # unload_gate is a pure ordering edge -- never consumed.
        del unload_gate

        # 1. Lever 1: free the OTR pipeline residue (writer LLM, Bark,
        #    FLUX) so the free-VRAM reading the auto-downgrade rule keys
        #    off is the real post-cleanup card state.
        free_report = free_otr_pipeline_residue(
            reason="OTR_HuMoTierLoader pre-load",
        )
        free_gb = free_report.get("free_gb_after")

        # 2. Resolve the effective tier (auto-downgrade rule).
        resolved_tier, note = self._resolve_tier(
            tier, bool(auto_downgrade),
            float(vram_safety_threshold_gb), free_gb,
        )
        if resolved_tier != tier:
            log.warning("[HuMoTierLoader] %s", note)
        else:
            log.info("[HuMoTierLoader] %s", note)

        spec = self._tier_config(resolved_tier)
        model_file = {
            "low_vram_default": humo_1p7b_model,
            "high_quality": humo_highq_model,
            "experimental_gguf": humo_gguf_model,
        }[resolved_tier]

        log.info(
            "[HuMoTierLoader] loading %s | model=%s steps=%d cfg=%.1f "
            "lora=%s",
            spec["label"], model_file, spec["steps"], spec["cfg"],
            distill_lora if spec["use_lora"] else "(none)",
        )

        # 3. Load the diffusion model, apply the distill LoRA on the high
        #    tiers only, then ModelSamplingSD3 (shift 8) on every tier.
        model = self._load_diffusion_model(model_file, spec["use_gguf"])
        if spec["use_lora"]:
            model = self._apply_distill_lora(model, distill_lora)
        model = self._apply_model_sampling(model)

        # 4. Text encoder / VAE / audio encoder -- identical every tier.
        clip = self._load_clip(clip_model)
        vae = self._load_vae(vae_model)
        audio_encoder = self._load_audio_encoder(audio_encoder_model)

        # 5. Diagnostic report.
        report_lines = [
            "OTR_HuMoTierLoader",
            f"  requested tier : {tier}",
            f"  resolved tier  : {resolved_tier} -- {spec['label']}",
            f"  note           : {note}",
            f"  diffusion model: {model_file}",
            f"  distill LoRA   : "
            + (distill_lora if spec["use_lora"] else "(not applied)"),
            f"  steps / cfg    : {spec['steps']} / {spec['cfg']:.1f}",
            f"  model sampling : ModelSamplingSD3 shift={_MODEL_SAMPLING_SHIFT}",
            f"  free VRAM      : "
            + (
                f"{free_gb:.1f} GB after pipeline-residue clean"
                if isinstance(free_gb, float) and math.isfinite(free_gb)
                else "unavailable (headless / no CUDA)"
            ),
            f"  residue freed  : {', '.join(free_report.get('steps_run') or []) or '(none)'}",
        ]
        if free_report.get("steps_failed"):
            report_lines.append(
                f"  residue WARN   : {', '.join(free_report['steps_failed'])}"
            )
        report = "\n".join(report_lines)
        log.info("[HuMoTierLoader] ready:\n%s", report)

        return (
            model, clip, vae, audio_encoder,
            int(spec["steps"]), float(spec["cfg"]), report,
        )


__all__ = ["HuMoTierLoader"]
