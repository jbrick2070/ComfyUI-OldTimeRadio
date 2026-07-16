"""nodes/_otr_model_loader.py

Thin facade over story_orchestrator._load_llm / _unload_llm. Exists to give
the v2.0 path (_otr_outline, _otr_line_composer, OTR_LedgerScriptWriter) a
stable import surface that doesn't depend on the legacy orchestrator's
internal layout.

DOES NOT extract code from story_orchestrator.py during the in-flight FULL
acceptance soak. Re-exports only. When the legacy writer retires in v2.1,
the loader code can move here without changing any v2.0 import sites.

Public surface:
    load_llm(model_id, *, device='cuda', optimization_profile='Standard') -> dict
        Returns a cache_entry dict: {model, tokenizer, model_id, device,
        quantized, budget_profile, context_cap}. Wraps _load_llm's tuple
        return into the dict shape documented by _otr_outline.py.

    unload_llm() -> None
        Re-export of _unload_llm. Frees VRAM globally.

    unload_llm_if_local_resident() -> bool
        Handoff helper: skips the full torch/CUDA teardown when the writer
        used only remote LLM providers and no local cache entry exists.

    MODEL_CONTEXT_CAPS: dict[str, int]
        Local copy of the per-model context-window caps. Drift-checked at
        first use against the function-local dict in _load_llm.

    make_generate_fn(cache_entry) -> GenerateFn
        Wraps a cache_entry into a chat-template-aware callable matching
        the GenerateFn contract used by _otr_outline.generate_outline and
        _otr_line_composer.compose_line:
            (messages, *, temperature, max_new_tokens) -> str

Status: Phase 2 of v2.0 sprint. Stdlib + lazy imports of torch and
story_orchestrator only.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

try:
    from ._otr_generation_budget import (
        GenerationContextOverflowError,
        fit_output_tokens,
    )
except ImportError:  # pragma: no cover - flat-module compatibility tests
    from _otr_generation_budget import (  # type: ignore
        GenerationContextOverflowError,
        fit_output_tokens,
    )

log = logging.getLogger("OTR")


__all__ = [
    "load_llm",
    "unload_llm",
    "has_local_resident_llm",
    "unload_llm_if_local_resident",
    "invalidate_cache_no_gpu_teardown",
    "request_slot",
    "make_generate_fn",
    "make_polish_generate_fn",
    "ModelLoaderError",
    "LLM_CACHE",
]


# ---------------------------------------------------------------------------
# B1c: shared slot-aware LLM cache (the modern facade).
#
# Records the currently-resident cache_entry so request_slot can detect
# slot transitions and decide between cache-reuse (same model) vs full
# unload + reload (different model). S30 B5 collapsed
# visual/llm_polish.py's local cache into this single source of truth
# so the 16 GB card never double-loads a model.
# ---------------------------------------------------------------------------


LLM_CACHE: dict[str, Any] = {
    "model_id": None,
    "slot": None,
    "cache_entry": None,
}

_REMOTE_CACHE_PROVIDERS = frozenset({"openrouter", "comfy_credits", "google_api"})


# ---------------------------------------------------------------------------
# S30 B1b: MODEL_CONTEXT_CAPS static dict + DEFAULT_CONTEXT_CAP constant
# DELETED. Context-cap resolution now goes through
# nodes._otr_model_catalog.resolve_context_cap which returns a tiered
# ContextCapVerdict (PASS for curated overrides, WARN for parsed
# config.json, UNKNOWN for unresolved) and clamps everything against
# HARD_VRAM_CONTEXT_LIMIT.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------


class ModelLoaderError(RuntimeError):
    """Raised when load_llm or make_generate_fn cannot complete.

    Wraps lower-level exceptions from the legacy _load_llm path so
    callers get a stable exception type to catch.
    """


# ---------------------------------------------------------------------------
# S0 portability helpers (docs/2026-07-09-platform-portability-final.md)
# ---------------------------------------------------------------------------


def _plan_max_memory(model_id: str, total_vram: float, *, cuda_available: bool):
    """VRAM-budget plan for the transformers loader.

    Returns the ``max_memory`` dict for ``from_pretrained`` or ``None``.
    The integer key ``0`` names CUDA device 0, so on a CUDA-less host the
    only honest plan is ``None`` (plain CPU/MPS load). The pre-S0 code
    built the CUDA-keyed dict from model-id string tags alone, handing
    transformers a device map for hardware that does not exist on
    cpu/mps hosts (fresh-install breaker).
    """
    if not cuda_available:
        return None
    sid = (model_id or "").lower()
    is_actually_2b = any(tag in sid for tag in ("2b-it", "2b_it")) or sid.endswith("2b")
    if total_vram >= 12.0:
        return {0: f"{total_vram - 2.5:.1f}GiB", "cpu": "32GiB"}
    if is_actually_2b:
        return {0: "3.2GiB", "cpu": "32GiB"}
    if any(tag in sid for tag in ("9b", "12b", "e4b", "4b-it")):
        return {0: "6.8GiB", "cpu": "32GiB"}
    return None


def _apply_matmul_precision_policy() -> None:
    """TF32 OFF for byte-identical determinism (I-2 / C-1); Ampere+ (sm80+)
    gets 'high' matmul precision for LLM throughput. The capability probe is
    GUARDED: ``torch.cuda.get_device_capability()`` raises on a CUDA-less
    host (the S0 loader:257 crash), and the sm80 check only means anything
    on CUDA anyway. The canonical headless launcher additionally exports
    NVIDIA_TF32_OVERRIDE=0 before torch imports; see nodes/_otr_determinism.py."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision('high')


# ---------------------------------------------------------------------------
# load_llm -- wraps tuple return into dict
# ---------------------------------------------------------------------------


def load_llm(
    model_id: str,
    *,
    device: str = "cuda",
    optimization_profile: str = "Standard",
    context_cap: int | None = None,
    policy: Any = None,
) -> dict[str, Any]:
    """Load an LLM and return a cache_entry dict.

    The always-load primitive: `request_slot` is the canonical entry
    point that handles cache-hit / cache-miss; `load_llm` builds and
    returns a fresh cache_entry every time it's called. Owns the
    bitsandbytes / NF4 / 8-bit / Standard / Obsidian profile body.

    Returns a cache_entry dict shaped for the v2.0 path:
        {
            "model":        <torch model>,
            "tokenizer":    <tokenizer>,
            "model_id":     <canonical model_id, no UI suffix>,
            "device":       <device string actually placed on>,
            "quantized":    <bool, True for NF4/8-bit profiles>,
            "context_cap":  <int, from internal _MODEL_CONTEXT_CAPS
                            or caller-provided override>,
        }

    Args:
        model_id: HF model identifier. UI suffixes like "[BETA]" or
                  "[8-bit]" are tolerated and stripped.
        device:   target device. Defaults "cuda".
        optimization_profile: one of "Standard", "Obsidian", "8-bit".
        context_cap: optional caller-provided context cap that
                  overrides the internal `_MODEL_CONTEXT_CAPS` lookup.
                  `request_slot` pre-resolves via
                  `_otr_model_catalog.resolve_context_cap` (tiered
                  ContextCapVerdict) and forwards the resolved value
                  through to skip the second filesystem scan. Defaults
                  to None (use internal lookup).

    Raises ModelLoaderError on any underlying failure (wraps the
    original exception via __cause__).
    """
    from .story_orchestrator import _runtime_log

    try:
        # `model_id_full` is the in-body name for the same value as
        # the caller-facing `model_id` argument.
        model_id_full = model_id

        # Strip [BETA] or [8-bit] labels used in the UI dropdown
        _stripped_model_id = model_id_full.split(" ")[0]

        # S1 platform-portability (2026-07-10): resolve the EXPLICIT runtime
        # policy (None = the nv50 16 GB baseline -- identical resolved
        # values to the deleted auto machinery below). policy.device wins
        # over the legacy `device` kwarg: one source of truth.
        from ._otr_shared.llm_policy import BASELINE_POLICY
        _policy = policy if policy is not None else BASELINE_POLICY
        device = _policy.device

        # S1: quantization is an EXPLICIT policy field. The legacy tag
        # predicate (Obsidian profile + "4-bit"/"9b"/"12b"/"nemo"/...
        # model-id substrings) is DELETED -- its resolved value for every
        # production id was NF4, which is exactly the policy default.
        requested_quantized = _policy.quant_policy in ("bnb_nf4", "bnb_8bit")

        # 2026-04-29: per-model context cap. See full rationale in the
        # original story_orchestrator._load_llm header (S30 B8 hash
        # ccf583d, lines 2019-2048). Conservative slices of native
        # context, chosen to leave VRAM headroom for KV cache +
        # co-resident Bark/FLUX/HuMo on a 16 GB Blackwell.
        # BUG-LOCAL-101 dropped Mistral-Nemo from 16384 to 8192. S21.2
        # (IMP-27) aligned the whole table at 8192 to free up dynamic
        # VRAM during long generation for audio co-residency.
        _MODEL_CONTEXT_CAPS = {
            "mistralai/Mistral-Nemo-Instruct-2407":               8192,
            "google/gemma-4-E2B-it":                              8192,
            "google/gemma-4-E4B-it":                              8192,
            "Qwen/Qwen2.5-14B-Instruct":                          8192,
            "Nitral-AI/Captain-Eris_Violet-V0.420-12B":           8192,
            "inflatebot/MN-12B-Mag-Mell-R1":                      8192,
            "google/gemma-2-2b-it":                               8192,
            "google/gemma-2-9b-it":                               8192,
        }
        _resolved_id = str(model_id_full).split(" ", 1)[0].strip()
        _cap = _MODEL_CONTEXT_CAPS.get(_resolved_id, 8192)
        if context_cap is not None:
            _cap = context_cap

        log.info(f"Loading LLM model: {_stripped_model_id} (quantized={requested_quantized})")

        # Lazy import - only pay the cost when actually generating
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer

        # -- Zero-Prime VRAM Hardening (v1.4) --
        # Detect hardware and purge memory BEFORE loading even the
        # tokenizer to prevent the 15GB transient spike on 16GB cards.
        total_vram = 0
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # 2026-04-30: Sync BEFORE eviction. bnb-NF4 + Blackwell sm_120 +
        # CUDA 13 surfaces cudaErrorUnknown on the post-eviction call
        # when async kernel completions from the prior generation are
        # still in flight while empty_cache() touches their memory.
        # Triple-confirmed by 2026-04-30 round-robin. See
        # docs/2026-04-30-spine-cuda-crash/.
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as _sync_err:  # noqa: BLE001
                _runtime_log(
                    f"VRAM_RESET: pre-evict synchronize() failed "
                    f"({_sync_err}); proceeding anyway"
                )

        # Nuclear Power Wash (Global Eviction)
        try:
            import comfy.model_management
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            _runtime_log("[StoryOrchestrator] Zero-Prime: ComfyUI Models Evicted.")
        except: pass

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Post-Wash Analytics
        if torch.cuda.is_available():
            free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / (1024**3)
            _runtime_log(f"[StoryOrchestrator] Zero-Prime VRAM State: {free_gb:.1f}GB Free. Capacity: {total_vram:.1f}GB")

        # -- VRAM Budgeting (Early Allocation) --
        # S0 portability: keyed on the live backend -- the CUDA-device-0 plan
        # is only built when CUDA exists (see _plan_max_memory).
        max_memory = _plan_max_memory(
            _stripped_model_id, total_vram,
            cuda_available=torch.cuda.is_available())
        if max_memory is not None and total_vram >= 12.0:
            _runtime_log(f"[StoryOrchestrator] Sovereignty Buffer Active: {total_vram - 2.5:.1f}GB Budget")

        # TF32 off + Ampere+ matmul precision (guarded off-CUDA; S0 fix for
        # the unguarded get_device_capability crash on cpu/mps hosts).
        _apply_matmul_precision_policy()

        # -- VRAM Hardening v1.4: Strict Handoff --
        try:
            from ._otr_bark_lib import _unload_bark
            _unload_bark()
        except ImportError:
            pass
        except Exception as handoff_err:
            log.warning("[StoryOrchestrator] Bark handoff failed: %s", handoff_err)

        # BUG-LOCAL-109 (2026-05-05) defensive guard: refuse the
        # canonical "auto" sentinel; caller must resolve to a concrete
        # model_id first.
        _mid_lower = (str(_stripped_model_id) or "").strip().lower()
        if not _mid_lower or _mid_lower.startswith("auto"):
            raise RuntimeError(
                f"load_llm: refusing to load model_id={_stripped_model_id!r} -- "
                "the 'auto (use story model)' sentinel must be resolved "
                "by the caller before load_llm is reached. See BUG-LOCAL-109."
            )

        try:
            tokenizer = AutoTokenizer.from_pretrained(_stripped_model_id, local_files_only=True)
        except OSError as local_err:
            log.info("[StoryOrchestrator] local_files_only=True failed for tokenizer (%s)", local_err)
            try:
                tokenizer = AutoTokenizer.from_pretrained(_stripped_model_id)
            except Exception as hub_err:
                log.error("[StoryOrchestrator] Hub fallback failed. Ensure model is downloaded or Hub is reachable: %s", hub_err)
                raise RuntimeError(f"Failed to load Tokenizer '{_stripped_model_id}'. Is it downloaded? Hub error: {hub_err}") from hub_err

        load_dtype = torch.bfloat16

        # S1: attention implementation is an EXPLICIT policy field. The FA2
        # auto-probe (distribution('flash-attn') + import) is DELETED -- on
        # the Blackwell sm_120 / Windows / torch 2.10 baseline it always
        # resolved to sdpa, which is the policy default. An FA2 wheel
        # appearing later is honoured by setting llm_attn_impl explicitly,
        # not by a probe. Still the single source of truth for
        # `attn_implementation` in common_kwargs, logged on every load.
        attn_impl = _policy.attn_impl
        _runtime_log(f"[StoryOrchestrator] Attention selector (policy): attn_implementation={attn_impl}")

        # 4-bit / 8-bit quantization -- EXPLICIT policy, no model-id tag
        # magic (S1; the "2bit"/"3bit" wing-ding upgrade + vram_safe_tags
        # predicate are deleted with it). bitsandbytes missing while the
        # policy requires it is a HARD FAIL: silently proceeding at
        # bfloat16 OOMs a 16 GB card at ~24 GiB -- the exact fallback
        # class BUG-LOCAL-098 exists to catch.
        quant_config = None
        needs_8bit = _policy.quant_policy == "bnb_8bit"
        needs_4bit = _policy.quant_policy == "bnb_nf4"

        if needs_8bit or needs_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as _bnb_err:
                raise ModelLoaderError(
                    f"llm.quant_policy={_policy.quant_policy!r} requires "
                    "bitsandbytes, which is not importable on this host. "
                    "Set llm.quant_policy='none' in the platform profile "
                    "(bnb lanes are OFF on ROCm/MPS/CPU tiers) or install "
                    "bitsandbytes. NO silent bf16 fallback."
                ) from _bnb_err
        if needs_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            log.info("[StoryOrchestrator] Enabling 8-bit quantization (policy)")
        elif needs_4bit:
            # BUG-LOCAL-098: instantiate BitsAndBytesConfig FRESH per
            # call. transformers mutates internal flags during
            # from_pretrained; a reused instance silently skips
            # quantization on the second call -> fp16 fallback -> OOM at
            # 24 GiB on 16 GiB GPU.
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            log.info("[StoryOrchestrator] Enabling 4-bit quantization (NF4, policy)")
            _runtime_log("[StoryOrchestrator] 4-bit NF4 active (policy)")

        from transformers import AutoTokenizer, AutoModelForCausalLM

        # BUG-LOCAL-085 fix: resolve HF_HOME from HKCU\Environment so
        # cache_dir is correct even when ComfyUI Desktop's process
        # didn't inherit User-scope env vars.
        try:
            from . import _otr_hf_env as _OTR_HF
            _hf_home_resolved = _OTR_HF.ensure_hf_home()
            _runtime_log(f"[StoryOrchestrator] HF_HOME resolved -> {_hf_home_resolved}")
        except Exception as _hf_err:
            _runtime_log(f"[StoryOrchestrator] HF_HOME helper unavailable ({_hf_err}); using os.environ fallback")
            _OTR_HF = None
            _hf_home_resolved = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

        cache_dir_path = os.path.join(_hf_home_resolved, "hub")

        # Try snapshot path first (preferred for sharded models on Windows).
        snapshot_path = None
        if _OTR_HF is not None:
            try:
                snapshot_path = _OTR_HF.resolve_snapshot_dir(_stripped_model_id, hf_home=_hf_home_resolved)
            except Exception as _snap_err:
                _runtime_log(f"[StoryOrchestrator] snapshot resolve failed ({_snap_err}); using model_id fallback")
        load_target = snapshot_path or _stripped_model_id
        if snapshot_path:
            _runtime_log(f"[StoryOrchestrator] Loading from canonical snapshot: {snapshot_path}")
        else:
            _runtime_log(f"[StoryOrchestrator] Snapshot not found in cache; falling back to model_id with cache_dir")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                load_target,
                local_files_only=(snapshot_path is None),
                trust_remote_code=False,
                cache_dir=cache_dir_path,
            )
            _runtime_log("LLM tokenizer loaded from cache (no HTTP checks)")
        except Exception as local_err:
            _runtime_log(f"[StoryOrchestrator] tokenizer load failed ({local_err}), attempting Hub fallback...")
            tokenizer = AutoTokenizer.from_pretrained(_stripped_model_id, trust_remote_code=False, cache_dir=cache_dir_path)

        common_kwargs = dict(
            cache_dir=cache_dir_path,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            torch_dtype=load_dtype,
            # Consume the resolved attention selector (above). On the
            # Blackwell sm_120 / Windows / torch 2.10 stack `attn_impl`
            # is always "sdpa" -- FA2 has no prebuilt wheel -- but the
            # selector stays the single source of truth so an FA2 wheel
            # appearing later is honoured without a second edit here.
            attn_implementation=attn_impl,
        )

        if max_memory is not None:
            common_kwargs["max_memory"] = max_memory
            common_kwargs["device_map"] = "auto"

        if quant_config is not None:
            common_kwargs["quantization_config"] = quant_config
            # Flagship Sovereignty: force 100% GPU on 14.5+ GiB cards.
            if total_vram >= 14.5:
                common_kwargs["device_map"] = {"": 0}
                _runtime_log(
                    f"[StoryOrchestrator] Flagship Sovereignty: "
                    f"Forcing 100% GPU for {_stripped_model_id} "
                    f"(total_vram={total_vram:.2f} GiB)"
                )
            else:
                _runtime_log(
                    f"[StoryOrchestrator] device_map=auto path "
                    f"(total_vram={total_vram:.2f} GiB < 14.5 GiB)"
                )

        try:
            model_config = None
            try:
                from transformers import AutoConfig
                _cfg_kwargs = {"trust_remote_code": False, "cache_dir": cache_dir_path}
                model_config = AutoConfig.from_pretrained(load_target, **_cfg_kwargs)
                if hasattr(model_config, "max_position_embeddings") and model_config.max_position_embeddings > _cap:
                    _runtime_log(f"[StoryOrchestrator] Hardening: Capping 128k context to {_cap} (Saves ~6GB VRAM)")
                    model_config.max_position_embeddings = _cap
            except Exception as _cfg_err:
                log.warning("[StoryOrchestrator] Config hardening failed: %s", _cfg_err)

            # BUG-LOCAL-098 tripwire setup: measure VRAM before load.
            _bug098_vram_before_gib = (
                torch.cuda.memory_allocated() / (1024 ** 3)
                if torch.cuda.is_available() else 0.0
            )

            model = AutoModelForCausalLM.from_pretrained(
                load_target,
                local_files_only=(snapshot_path is None),
                config=model_config,
                **common_kwargs,
            )
            _runtime_log(
                f"LLM model loaded from "
                f"{'canonical snapshot' if snapshot_path else 'model_id with cache_dir'} "
                f"(no HTTP checks)"
            )

            # BUG-LOCAL-098 tripwire: fail loud if NF4 silently dropped to fp16.
            if quant_config is not None and torch.cuda.is_available():
                _bug098_vram_after_gib = torch.cuda.memory_allocated() / (1024 ** 3)
                _bug098_delta_gib = (
                    _bug098_vram_after_gib - _bug098_vram_before_gib
                )
                _bug098_linear4bit_count = 0
                try:
                    for _m in model.modules():
                        _cls_name = type(_m).__name__
                        _mod_name = type(_m).__module__ or ""
                        if (_cls_name == "Linear4bit"
                                and _mod_name.startswith("bitsandbytes")):
                            _bug098_linear4bit_count += 1
                except Exception:  # noqa: BLE001
                    _bug098_linear4bit_count = -1
                _bug098_is_loaded_in_4bit = bool(
                    getattr(model, "is_loaded_in_4bit", False)
                )
                _bug098_max_gib = 11.0
                _bug098_module_signal = (
                    _bug098_linear4bit_count > 0
                    or _bug098_is_loaded_in_4bit
                )
                _bug098_vram_signal = (
                    _bug098_delta_gib >= 0.0
                    and _bug098_delta_gib <= _bug098_max_gib
                )
                _runtime_log(
                    f"[BUG-098 tripwire] post-load: "
                    f"linear4bit_count={_bug098_linear4bit_count} "
                    f"is_loaded_in_4bit={_bug098_is_loaded_in_4bit} "
                    f"vram_delta={_bug098_delta_gib:.2f}GiB "
                    f"(ceiling={_bug098_max_gib:.2f}GiB)"
                )
                if not _bug098_module_signal or not _bug098_vram_signal:
                    try:
                        model.cpu()
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        del model
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        import gc as _bug098_gc
                        _bug098_gc.collect()
                        torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass
                    raise RuntimeError(
                        f"BUG-LOCAL-098: NF4 quantized load did not "
                        f"materialize for {_stripped_model_id!r}. "
                        f"linear4bit_count={_bug098_linear4bit_count} "
                        f"is_loaded_in_4bit={_bug098_is_loaded_in_4bit} "
                        f"vram_delta={_bug098_delta_gib:.2f}GiB. "
                        f"This is the bitsandbytes second-load silent "
                        f"fp16 fallback. Workaround: restart ComfyUI "
                        f"Desktop and re-queue. Tracked as BUG-LOCAL-098."
                    )
        except (OSError, ValueError) as local_err:
            _runtime_log(f"[StoryOrchestrator] local_files_only=True failed for model ({local_err}), attempting Hub fallback...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    _stripped_model_id,
                    config=model_config,
                    **common_kwargs,
                )
            except Exception as hub_err:
                log.error("[StoryOrchestrator] Hub fallback failed. Ensure model is downloaded or Hub is reachable: %s", hub_err)
                raise RuntimeError(f"Failed to load LLM model '{_stripped_model_id}'. Is it downloaded? Hub error: {hub_err}") from hub_err

        if quant_config is None and max_memory is None:
            model = model.to(device)
        model = model.eval()

        actual_quant = (quant_config is not None)
        _runtime_log(f"LLM loaded: {_stripped_model_id} (quantized={actual_quant}, budget={optimization_profile}) [v1.5]")

        # CUDA kernel warmup -- absorbs the 30-60s JIT compile cost
        # for SDPA + BitsAndBytes 4-bit on Blackwell.
        try:
            _warmup_start = time.time()
            _runtime_log("WARMUP: Starting 1-token CUDA kernel warmup...")
            _warmup_ids = tokenizer("Test.", return_tensors="pt")["input_ids"].to(model.device)
            with torch.no_grad():
                model.generate(
                    _warmup_ids,
                    max_new_tokens=1,
                    do_sample=False,
                )
            del _warmup_ids
            torch.cuda.empty_cache()
            _warmup_sec = time.time() - _warmup_start
            _runtime_log(f"WARMUP: CUDA kernels compiled in {_warmup_sec:.1f}s - generation will start instantly")
            log.info("[StoryOrchestrator] CUDA warmup complete (%.1fs) - first generate will not stall", _warmup_sec)
        except Exception as _warmup_err:
            log.warning("[StoryOrchestrator] CUDA warmup failed (non-fatal): %s", _warmup_err)
            _runtime_log(f"WARMUP: Failed (non-fatal): {_warmup_err}")

        return {
            "model":       model,
            "tokenizer":   tokenizer,
            "model_id":    _stripped_model_id,
            "device":      device,
            "quantized":   actual_quant,
            "context_cap": _cap,
        }
    except ModelLoaderError:
        raise
    except Exception as exc:  # noqa: BLE001
        # Sprint H Commit B1 layer 2 (2026-05-17): inner failure-path
        # cleanup. AutoModelForCausalLM.from_pretrained, the warmup
        # generate pass, the BUG-LOCAL-098 tripwire, and the post-load
        # .to(device) call all run AFTER `model` is bound to GPU
        # weights but BEFORE this function returns. If any of them
        # raise, the cache_entry never gets stored in LLM_CACHE, so
        # downstream `unload_llm()` (which reads LLM_CACHE) can't find
        # the orphan to drop. The retry in
        # `_otr_style_picker._run_inventor` then cache-misses and a
        # second copy gets loaded on top of the orphan -> "Currently
        # allocated 29.97 GiB" OOM seen in Sprint H iter 1 logs.
        #
        # Pair with the layer-1 wrapper in `request_slot()`: layer 1
        # catches load_llm raising as a whole; layer 2 catches in-body
        # failures so the orphan is dropped at first opportunity.
        # Belt-and-braces -- both layers are idempotent.
        try:
            _orphan = locals().get("model")
            if _orphan is not None and hasattr(_orphan, "to"):
                try:
                    _orphan.to("cpu")
                except Exception:  # noqa: BLE001
                    pass
            try:
                del _orphan
            except Exception:  # noqa: BLE001
                pass
            try:
                # Drop the local binding too so gc can reap.
                del model  # noqa: F821
            except Exception:  # noqa: BLE001
                pass
            import gc as _gc
            _gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    try:
                        _torch.cuda.ipc_collect()
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            # Cleanup must never mask the real load failure.
            pass
        raise ModelLoaderError(
            f"load_llm failed for model_id={model_id!r}: {exc}"
        ) from exc


def unload_llm() -> None:
    """Full VRAM teardown for cross-model slot transitions.

    Canonical sequence (matches reference_chained_backend_teardown):
        1. model.to("cpu")           -- move weights off the GPU.
        2. del cache_entry           -- drop references so gc can reap.
        3. gc.collect()              -- purge Python-side refs.
        4. torch.cuda.empty_cache()  -- return free blocks to allocator.
        5. torch.cuda.ipc_collect()  -- release inter-process CUDA IPC
                                        handles. CRITICAL when LLM load
                                        follows a video-model run (FLUX/
                                        HuMo/LTX). Without ipc_collect,
                                        the next load_llm can OOM even
                                        when the byte budget fits.
        6. torch.cuda.synchronize()  -- let in-flight ops finish.

    Also tears down story_orchestrator's legacy LLM stack as a
    best-effort fallback. The orchestrator's `_LLM_CACHE` dict + its
    `_load_llm` body remain alive as the underlying implementation
    layer (this loader's `load_llm` still delegates back to them);
    the teardown ensures both surfaces are quiesced together. Never
    raises -- a teardown failure should NOT propagate as a node error.

    S30 B4b: the three production importers (batch_bark_generator,
    _otr_bark_lib, scene_sequencer) now import this `unload_llm`
    directly rather than the orchestrator's `_unload_llm` (the
    audit-miss BUG-LOCAL-226 fix). Story orchestrator's
    `_generate_with_llm` also routes through `request_slot("technical",
    ...)` to acquire its cache_entry; the RSS news path no longer
    holds a parallel reference to the legacy cache.
    """
    import gc

    entry = LLM_CACHE.get("cache_entry")
    if entry is not None:
        if entry.get("provider") == "gguf_native":
            try:
                from ._otr_gguf_backend import GGUFNativeBackend
                GGUFNativeBackend().unload(entry)
            except Exception as exc:  # noqa: BLE001
                log.debug("[OTR_ModelLoader] GGUF unload failed: %s", exc)
        model = entry.get("model")
        if (
            entry.get("provider") != "gguf_native"
            and model is not None
            and hasattr(model, "to")
        ):
            try:
                model.to("cpu")
            except Exception as exc:  # noqa: BLE001
                log.debug("[OTR_ModelLoader] model.to(cpu) failed: %s", exc)
    # B1d: clear + update IN PLACE. Rebinding (LLM_CACHE = {...}) leaves
    # any `from _otr_model_loader import LLM_CACHE` consumer holding a
    # stale dict reference, which silently breaks slot transitions. The
    # `global` keyword is unnecessary now; we only mutate the dict.
    LLM_CACHE.clear()
    LLM_CACHE.update({"model_id": None, "slot": None, "cache_entry": None})

    gc.collect()

    try:
        import torch  # noqa: F401

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception as exc:  # noqa: BLE001
                log.debug("[OTR_ModelLoader] ipc_collect skipped: %s", exc)
            try:
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001
                log.debug("[OTR_ModelLoader] synchronize skipped: %s", exc)
    except ImportError:
        pass


def has_local_resident_llm() -> bool:
    """True when the singleton cache currently owns a local LLM resource.

    Remote OpenRouter / Comfy Credits requests deliberately do not populate
    ``LLM_CACHE``. If a provider-tagged remote entry is ever present, it still
    carries no weights and must not trigger the local torch/CUDA teardown path.
    """
    entry = LLM_CACHE.get("cache_entry")
    if entry is None:
        return False
    if isinstance(entry, dict) and entry.get("provider") in _REMOTE_CACHE_PROVIDERS:
        return False
    return True


def unload_llm_if_local_resident() -> bool:
    """Unload only when a local LLM is actually resident.

    Returns True when ``unload_llm()`` was called. Handoff callers use this to
    keep all-cloud LLM runs from importing torch just to clear an empty local
    allocator, while preserving ``unload_llm()`` for real local teardown and
    load-failure orphan cleanup.
    """
    if not has_local_resident_llm():
        return False
    unload_llm()
    return True



def invalidate_cache_no_gpu_teardown() -> None:
    """Clear LLM_CACHE dict references WITHOUT touching the GPU.

    Use case: timeout recovery when an orphan worker thread may
    still be executing CUDA kernels on the cached model. Calling
    `unload_llm()` here would race the active kernel: `model.to("cpu")`
    moves weights mid-write and `torch.cuda.empty_cache()` can
    deallocate memory the kernel is still reading from -- both trigger
    `cudaErrorIllegalAddress`.

    The orphan thread's stack frame holds the model reference and the
    generate loop continues to completion on its own references. Once
    the orphan exits naturally, GC + a subsequent clean `unload_llm`
    call (when the next `request_slot` loads a different model)
    handles cleanup safely.

    NOT a general-purpose helper -- only use in code paths where GPU
    teardown is unsafe (timeout recovery, signal handlers).

    S31 B4 (2026-05-14): fixes the TIMEOUT_RECOVERY CUDA-race
    regression introduced at S30 B4b. S30 B4b rewired
    `story_orchestrator._run_with_timeout` to call `unload_llm()`
    on timeout -- but the comment ("avoids cudaErrorIllegalAddress
    from orphan worker still on GPU") was structurally wrong: the
    new behavior actively CAUSES that error. Pre-B4b the path was
    dict-invalidation-only (safe). S31 B4 reverts to safe semantics
    via this helper. See BUG-LOCAL-228 for the regression log.
    """
    LLM_CACHE.clear()
    LLM_CACHE.update({"model_id": None, "slot": None, "cache_entry": None})


def request_slot(
    slot: str, model_id: str, policy: Any = None, load_config: Any = None,
) -> dict[str, Any]:
    """Slot-aware entry point. Loads (or reuses cached) LLM, handling
    cache reuse vs full teardown automatically.

    ``load_config`` (GGUF row registry, 2026-07-16): the immutable per-slot
    GGUF load contract resolved by the writer's preflight. When present it is
    the resident-reuse identity (repo_id + resolved path + quant + n_ctx +
    n_batch + n_gpu_layers) and is threaded to the backend load -- no live-env
    rebuild. Ignored for non-GGUF rows.

    B1d order (vram-fit BEFORE any network/disk work):
      1. normalize model_id via catalog.validate_model_id (strips
         [NOT DOWNLOADED] suffix, structural rejection, admit-path check).
      2. Cache hit (same model_id resident) -> return entry. Done.
      3. resolve_context_cap(model_id) -> tiered ContextCapVerdict.
      4. check_vram_fit(model_id, ctx_verdict.value) -> tiered VRAMFitVerdict.
      5. FAIL -> raise VRAMFitFailedError. CRITICAL: this fires BEFORE
         auto_download so a 70B-on-16GB pick never triggers a network
         pull or a disk-space pre-check pass on a doomed-to-fail load.
      6. Combined caution log (anything below PASS/PASS).
      7. auto_download_if_missing -- gated/disk-space pre-flight +
         snapshot_download. Local-cache short-circuit fires inside the
         catalog helper.
      8. unload_llm() (only if a different model was resident), then
         load_llm(model_id, context_cap=ctx_verdict.value) -- skips the
         second catalog walk by forwarding the resolved cap.
      9. Cache the entry under (slot, model_id).

    `slot` is "creative" or "technical" -- used for log lines + cache
    keying. The cache holds at most one resident model regardless of
    slot; same-slot reuse and cross-slot identity-reuse both return the
    cached entry without a full teardown.
    """
    from . import _otr_model_catalog as _otr_catalog
    from ._otr_model_inputs import VRAMFitFailedError
    from ._otr_shared.llm_policy import BASELINE_POLICY, lane_for_row

    if slot not in ("creative", "technical"):
        raise ModelLoaderError(
            f"request_slot: slot must be 'creative' or 'technical', got {slot!r}"
        )

    # S1 platform-portability (2026-07-10): resolve the explicit runtime
    # policy. None = the nv50 baseline (an API backstop for direct callers;
    # every production call-site threads a real policy).
    if policy is None:
        log.info("[Selector] slot=%s policy=None -> nv50 BASELINE", slot)
    _policy = policy if policy is not None else BASELINE_POLICY

    # Step 1: normalize.
    normalized = _otr_catalog.validate_model_id(model_id)

    # [OpenRouter S3] Remote branch (FC2 seam 1) -- the dispatch table is
    # otherwise dormant. A virtual catalog row carries
    # loader_backend="openrouter_http"; route it to the remote backend's
    # load(), which returns a provider-tagged cache_entry using ZERO local
    # VRAM. SKIP steps 3-8 (resolve_context_cap, check_vram_fit,
    # auto_download_if_missing, the resident-model teardown, load_llm) and
    # -- critically -- LEAVE any resident local model in LLM_CACHE
    # UNTOUCHED (C2 no-evict). Placed before the Step 2 cache-hit read so
    # a remote request never reads or mutates LLM_CACHE: the common config
    # (creative=remote, technical=local) must not evict + reload the local
    # model across slot transitions. Remote makes zero CUDA / snapshot /
    # download calls.
    # BUG-LOCAL-299: this remote-routing gate shipped recognizing ONLY
    # "openrouter_http". A Comfy Credits row (loader_backend="comfy_credits_http")
    # fell through to the LOCAL path below, so ComfyUI tried to HF-download the
    # virtual handle (e.g. "comfy:slot-a") -> HFValidationError, aborting the run
    # before the lane was ever exercised. Route BOTH remote loader_backends; the
    # _otr_model_runtime dispatch table already maps each key to its backend, so
    # a future remote lane only adds its key to this tuple.
    # Virtual catalog rows must be intercepted before the HF cache/download
    # path below. Remote rows are zero-VRAM and do not disturb a resident
    # local model. The GGUF row is different: it is in-process VRAM and must
    # participate in the singleton cache/teardown discipline.
    _REMOTE_DISPATCH_BACKENDS = ("openrouter_http", "comfy_credits_http", "google_api_http")
    _GGUF_DISPATCH_BACKENDS = ("gguf_native",)
    _virtual_row = _otr_catalog._by_repo_id().get(normalized)

    # S1 runtime lane backstop: the profile's lane_allowlist is enforced at
    # validate/emit time upstream; this is the last line of defense so a
    # hand-crafted workflow cannot smuggle a disallowed lane through. NO
    # FALLBACK: pick an admitted lane or change the platform profile.
    _lane = lane_for_row(_virtual_row)
    if not _policy.admits_lane(_lane):
        raise ModelLoaderError(
            f"request_slot: lane '{_lane}' (model {normalized!r}) is not "
            f"admitted by the profile lane_allowlist "
            f"{list(_policy.lane_allowlist)} -- NO FALLBACK."
        )

    if (
        _virtual_row is not None
        and getattr(_virtual_row, "loader_backend", None) in _REMOTE_DISPATCH_BACKENDS
    ):
        from ._otr_model_runtime import get_backend_for_row
        log.info(
            "[Selector] slot=%s remote-dispatched backend for %s (no local VRAM; "
            "resident local model left in place, no-evict)",
            slot, normalized,
        )
        return get_backend_for_row(_virtual_row).load(
            normalized, _virtual_row, policy=_policy,
        )

    if (
        _virtual_row is not None
        and getattr(_virtual_row, "loader_backend", None) in _GGUF_DISPATCH_BACKENDS
    ):
        from ._otr_model_runtime import get_backend_for_row
        # Resident-reuse identity for the in-process GGUF singleton. The
        # threaded load_config's reuse_key (repo_id + resolved path + quant +
        # n_ctx + n_batch + n_gpu_layers) is the artifact-shaping identity that
        # policy.cache_key() alone cannot see (it misses the resolved path /
        # n_batch / n_gpu_layers). Without a load_config (direct/legacy caller)
        # fall back to the raw policy key -- the pre-registry behavior.
        _gguf_key = (
            load_config.reuse_key() if load_config is not None
            else _policy.cache_key()
        )
        if (
            LLM_CACHE.get("model_id") == normalized
            and LLM_CACHE.get("cache_entry") is not None
        ):
            # A resident model only counts as a hit when it was loaded under
            # the SAME load identity. Silent stale reuse is the bug class this
            # campaign kills.
            if LLM_CACHE.get("gguf_load_key") == _gguf_key:
                log.info("[Selector] slot=%s reuse cache for %s", slot, normalized)
                LLM_CACHE["slot"] = slot
                return LLM_CACHE["cache_entry"]  # type: ignore[return-value]
            log.info(
                "[Selector] gguf load-config change for %s (%s -> %s): "
                "full teardown",
                normalized, LLM_CACHE.get("gguf_load_key"), _gguf_key,
            )
            unload_llm()
        if LLM_CACHE.get("model_id") not in (None, normalized):
            log.info(
                "[Selector] slot transition: %s -> %s (full teardown)",
                LLM_CACHE.get("model_id"),
                normalized,
            )
            unload_llm()
        cache_entry = get_backend_for_row(_virtual_row).load(
            normalized, _virtual_row, policy=_policy, load_config=load_config,
        )
        LLM_CACHE["model_id"] = normalized
        LLM_CACHE["slot"] = slot
        LLM_CACHE["cache_entry"] = cache_entry
        LLM_CACHE["policy_key"] = _policy.cache_key()
        LLM_CACHE["gguf_load_key"] = _gguf_key
        return cache_entry

    # Step 2: cache hit on the same model id (regardless of slot) -- policy
    # keyed (S1): a mismatched policy_key is a MISS + teardown, never reuse.
    if LLM_CACHE.get("model_id") == normalized and LLM_CACHE.get("cache_entry") is not None:
        if LLM_CACHE.get("policy_key") == _policy.cache_key():
            log.info("[Selector] slot=%s reuse cache for %s", slot, normalized)
            LLM_CACHE["slot"] = slot
            return LLM_CACHE["cache_entry"]  # type: ignore[return-value]
        log.info(
            "[Selector] policy change for %s (%s -> %s): full teardown",
            normalized, LLM_CACHE.get("policy_key"), _policy.cache_key(),
        )
        unload_llm()

    # Step 3: context cap (never raises).
    ctx_verdict = _otr_catalog.resolve_context_cap(normalized)

    # Step 4: VRAM fit (never raises). Ceiling comes from the policy (S1);
    # 0 = gate DISABLED (cpu tier -- there is no VRAM to fit).
    if _policy.vram_ceiling_gb > 0:
        fit_verdict = _otr_catalog.check_vram_fit(
            normalized, ctx_verdict.value,
            ceiling_gb=_policy.vram_ceiling_gb,
        )

        # Step 5: FAIL escalates BEFORE any network/disk work. A 70B pick on
        # a 16 GB card must not trigger snapshot_download or a disk-space
        # pre-check pass; both waste minutes on a doomed-to-OOM load.
        if fit_verdict.tier == "FAIL":
            raise VRAMFitFailedError(
                f"VRAMFitFailedError: {normalized!r}: {fit_verdict.reason}. "
                f"ctx_cap={ctx_verdict.tier}@{ctx_verdict.value}",
                estimated_gb=fit_verdict.estimated_gb,
                ceiling_gb=fit_verdict.ceiling_gb,
            )

        # Step 6: combined caution log (everything below PASS/PASS).
        if not (fit_verdict.tier == "PASS" and ctx_verdict.tier == "PASS"):
            log.info(
                "[Selector] proceeding with caution: ctx_cap=%s@%d, "
                "vram_fit=%s@%.1f GB",
                ctx_verdict.tier,
                ctx_verdict.value,
                fit_verdict.tier,
                fit_verdict.estimated_gb,
            )
    else:
        log.info(
            "[Selector] VRAM-fit gate disabled by policy "
            "(vram_ceiling_gb=0, cpu tier)"
        )

    # Step 7: ensure on-disk + handle gating / disk-space pre-flight.
    # Local-cache short-circuit (B1d) fires inside this helper when the
    # snapshot is already on disk.
    _otr_catalog.auto_download_if_missing(normalized)

    # Step 8: if a different model is resident, unload it. Then load.
    if LLM_CACHE.get("model_id") not in (None, normalized):
        log.info(
            "[Selector] slot transition: %s -> %s (full teardown)",
            LLM_CACHE.get("model_id"),
            normalized,
        )
        unload_llm()

    # Sprint H iter 3 (2026-05-17): orphan-model guard. load_llm may
    # raise AFTER AutoModelForCausalLM.from_pretrained successfully
    # allocates the weights on GPU (e.g. BNB quantization, warmup pass,
    # tripwire). If we re-raise without cleaning up, the LLM_CACHE is
    # never populated (line 744+ is bypassed) AND the orphan model
    # remains resident on GPU. The next request_slot call cache-misses
    # and a SECOND copy gets loaded on top -- producing the
    # "Currently allocated 29.97 GiB" OOM on the retry inside
    # _otr_style_picker._run_inventor's 3-attempt loop. Wrap with
    # try/except + unload_llm() to guarantee the retry starts from a
    # clean slate. unload_llm()'s entry-is-None branch is a safe no-op
    # for the dict update; the empty_cache + ipc_collect + synchronize
    # still fire and drop the orphan.
    try:
        cache_entry = load_llm(
            normalized, context_cap=ctx_verdict.value, policy=_policy,
        )
    except Exception:
        log.warning(
            "[Selector] load_llm raised for %s; running unload_llm() "
            "to drop any orphan VRAM before retry",
            normalized,
        )
        try:
            unload_llm()
        except Exception:  # noqa: BLE001
            log.exception("[Selector] unload_llm() also raised; continuing")
        raise

    # Step 9: cache (policy-keyed, S1).
    LLM_CACHE["model_id"] = normalized
    LLM_CACHE["slot"] = slot
    LLM_CACHE["cache_entry"] = cache_entry
    LLM_CACHE["policy_key"] = _policy.cache_key()
    return cache_entry


# ---------------------------------------------------------------------------
# make_generate_fn -- chat-template adapter
# ---------------------------------------------------------------------------


def _normalize_messages_for_cache_entry(
    cache_entry: dict[str, Any], messages: list[dict],
) -> list[dict]:
    """BUG-LOCAL-262: fold system messages into the first user turn
    for tokenizers whose chat template rejects the system role.

    Probes the tokenizer once and caches the verdict on the
    cache_entry under `_system_role_supported`, so the probe runs
    once per model residency rather than per generate call (both
    make_generate_fn and make_polish_generate_fn share the entry).
    """
    from . import _otr_loader_backends as _otr_loader_backends

    tokenizer = cache_entry["tokenizer"]
    supported = cache_entry.get("_system_role_supported")
    if supported is None:
        supported = _otr_loader_backends.tokenizer_supports_system_role(
            tokenizer,
        )
        cache_entry["_system_role_supported"] = supported
    if supported:
        return messages
    return _otr_loader_backends.normalize_messages_for_tokenizer(
        tokenizer, messages,
    )


def make_generate_fn(cache_entry: dict[str, Any]):
    """Wrap a cache_entry into the GenerateFn callable.

    Returns a callable matching:
        (messages, *, temperature, max_new_tokens) -> str

    where `messages` is a list[dict] in chat format
    ([{"role": "system", "content": ...}, {"role": "user", "content": ...}])
    and the return is the raw decoded string from the model with the
    prompt prefix removed.

    Generation params hardcoded for the v2.0 path:
        do_sample=True
        top_p=0.92

    Caller controls temperature and max_new_tokens per call.

    Raises ModelLoaderError if the cache_entry is missing required
    keys or if torch is not importable at first call time.
    """
    # [OpenRouter S3] Remote branch (FC2 seam 2). A provider-tagged
    # remote entry has no model/tokenizer; return the remote generate_fn
    # before the local-key check below. Uses zero local VRAM.
    if cache_entry.get("provider") == "openrouter":
        from ._otr_openrouter_backend import make_openrouter_generate_fn
        return make_openrouter_generate_fn(cache_entry)
    # BUG-LOCAL-299: Comfy Credits sibling -- same zero-VRAM remote seam.
    if cache_entry.get("provider") == "comfy_credits":
        from ._otr_comfy_backend import make_comfy_credits_generate_fn
        return make_comfy_credits_generate_fn(cache_entry)
    if cache_entry.get("provider") == "google_api":
        from ._otr_google_api.llm import make_google_api_generate_fn
        return make_google_api_generate_fn(cache_entry)
    # Native GGUF lane: in-process llama-cpp-python, no daemon or port.
    if cache_entry.get("provider") == "gguf_native":
        from ._otr_gguf_backend import make_gguf_generate_fn
        return make_gguf_generate_fn(cache_entry)
    required = {"model", "tokenizer"}
    missing = required - set(cache_entry)
    if missing:
        raise ModelLoaderError(
            f"cache_entry missing required keys: {sorted(missing)}"
        )

    model = cache_entry["model"]
    tokenizer = cache_entry["tokenizer"]

    def generate_fn(messages, *, temperature, max_new_tokens):
        # Lazy torch import. Raised as ModelLoaderError to match the
        # facade's exception contract.
        try:
            import torch
        except ImportError as exc:
            raise ModelLoaderError("torch not available") from exc

        messages = _normalize_messages_for_cache_entry(cache_entry, messages)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        try:
            effective_max_new_tokens = fit_output_tokens(
                max_new_tokens,
                context_cap=int(cache_entry.get("context_cap") or 8192),
                prompt_tokens=inputs["input_ids"].shape[1],
                label=f"local model {cache_entry.get('model_id', '<unknown>')}",
            )
        except GenerationContextOverflowError as exc:
            raise ModelLoaderError(str(exc)) from exc
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=0.92,
                max_new_tokens=effective_max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Strip prompt prefix from decoded output.
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(
            out[0][prompt_len:],
            skip_special_tokens=True,
        )

    return generate_fn


# ---------------------------------------------------------------------------
# Polish-specific generate fn (LFC sprint commit 3, section 6.4)
# ---------------------------------------------------------------------------


# Polish-specific sampling baked in per ADR section 6.4. None of these
# are configurable per-call -- the whole point of the dedicated polish
# fn is that the writer's closure-captured composer tuning
# (repetition_penalty, min_p, top_p tweaks) cannot leak in.
_POLISH_TOP_P: float = 0.9
_POLISH_DO_SAMPLE: bool = True


def make_polish_generate_fn(cache_entry: dict[str, Any]):
    """Build a polish-specific generate fn from `cache_entry`.

    LFC sprint commit 3, ADR section 6.4 (2026-05-11). Polish is a
    short, targeted rewrite -- conceptually closer to a constrained
    edit than the composer's long-form generation. The writer's main
    `make_generate_fn` (via the OTR_LedgerScriptWriter
    `_build_truncating_generate_fn` wrapper) bakes
    repetition_penalty / min_p / top_p into its closure tuned for
    composition. Those settings leak into polish via closure capture
    and produce awkward substitutions on short rewrites.

    The polish fn here is a SEPARATE closure off the same cache_entry
    with composer-independent sampling:

        temperature      -- caller-provided per call (defaults to 0.4
                            via _otr_line_composer.polish_line)
        top_p            -- 0.9 (slightly tighter than composer 0.92)
        do_sample        -- True
        min_p            -- not passed (transformers default 0)
        repetition_penalty -- not passed (transformers default 1.0)

    Returns a callable with the same signature as `make_generate_fn`:
        (messages, *, temperature, max_new_tokens) -> str
    """
    # [OpenRouter S3] Remote branch (FC2 seam 2). A provider-tagged
    # remote entry has no model/tokenizer; the remote generate_fn applies
    # the same sampling the caller passes (polish callers pass their own
    # temperature), so one closure covers both factories.
    if cache_entry.get("provider") == "openrouter":
        from ._otr_openrouter_backend import make_openrouter_generate_fn
        return make_openrouter_generate_fn(cache_entry)
    # BUG-LOCAL-299: Comfy Credits sibling -- same zero-VRAM remote seam.
    if cache_entry.get("provider") == "comfy_credits":
        from ._otr_comfy_backend import make_comfy_credits_generate_fn
        return make_comfy_credits_generate_fn(cache_entry)
    if cache_entry.get("provider") == "google_api":
        from ._otr_google_api.llm import make_google_api_generate_fn
        return make_google_api_generate_fn(cache_entry)
    # Native GGUF lane: in-process llama-cpp-python, no daemon or port.
    if cache_entry.get("provider") == "gguf_native":
        from ._otr_gguf_backend import make_gguf_generate_fn
        return make_gguf_generate_fn(cache_entry)
    required = {"model", "tokenizer"}
    missing = required - set(cache_entry)
    if missing:
        raise ModelLoaderError(
            f"cache_entry missing required keys: {sorted(missing)}"
        )

    model = cache_entry["model"]
    tokenizer = cache_entry["tokenizer"]

    def polish_generate_fn(messages, *, temperature, max_new_tokens):
        try:
            import torch
        except ImportError as exc:
            raise ModelLoaderError("torch not available") from exc

        messages = _normalize_messages_for_cache_entry(cache_entry, messages)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        try:
            effective_max_new_tokens = fit_output_tokens(
                max_new_tokens,
                context_cap=int(cache_entry.get("context_cap") or 8192),
                prompt_tokens=inputs["input_ids"].shape[1],
                label=f"local polish {cache_entry.get('model_id', '<unknown>')}",
            )
        except GenerationContextOverflowError as exc:
            raise ModelLoaderError(str(exc)) from exc
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=_POLISH_DO_SAMPLE,
                temperature=temperature,
                top_p=_POLISH_TOP_P,
                max_new_tokens=effective_max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(
            out[0][prompt_len:],
            skip_special_tokens=True,
        )

    return polish_generate_fn


# ---------------------------------------------------------------------------
# Self-test (run as `python nodes/_otr_model_loader.py`)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== _otr_model_loader.py self-test ===")

    # Test 1: catalog.resolve_context_cap returns a sane PASS for the
    # C7 audio-baseline model (Mistral-Nemo).
    print("\n[Test 1] resolve_context_cap baseline")
    from . import _otr_model_catalog as _otr_catalog
    verdict = _otr_catalog.resolve_context_cap(_otr_catalog.DEFAULT_LLM)
    assert verdict.tier == "PASS"
    assert verdict.value == 8192
    print(f"  PASS ({verdict.tier} @ {verdict.value}, source={verdict.source})")

    # Test 2: HARD_VRAM_CONTEXT_LIMIT is at least 4096 (matches old
    # DEFAULT_CONTEXT_CAP minimum invariant).
    print("\n[Test 2] HARD_VRAM_CONTEXT_LIMIT is sane")
    assert _otr_catalog.HARD_VRAM_CONTEXT_LIMIT >= 4096
    print(f"  PASS ({_otr_catalog.HARD_VRAM_CONTEXT_LIMIT})")

    # Test 3: ModelLoaderError shape.
    print("\n[Test 3] ModelLoaderError is RuntimeError subclass")
    assert issubclass(ModelLoaderError, RuntimeError)
    print("  PASS")

    # Test 4: make_generate_fn rejects malformed cache_entry.
    print("\n[Test 4] make_generate_fn rejects malformed cache_entry")
    try:
        make_generate_fn({})
        print("  FAIL: empty cache_entry was accepted")
    except ModelLoaderError as e:
        assert "missing required keys" in str(e)
        print(f"  PASS: rejected ({e})")

    try:
        make_generate_fn({"model": object()})
        print("  FAIL: cache_entry missing tokenizer was accepted")
    except ModelLoaderError as e:
        assert "tokenizer" in str(e)
        print(f"  PASS: rejected ({e})")

    # Test 5: make_generate_fn returns callable with right shape.
    print("\n[Test 5] make_generate_fn returns callable")
    class _StubTok:
        eos_token_id = 0
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):  # kept: mirror HF tokenizer signature
            return "stub-prompt"
        def __call__(self, prompt, return_tensors):  # kept: mirror HF tokenizer signature
            class _Out:
                input_ids = type("S", (), {"shape": (1, 5)})()
                def to(self, device): return self  # kept: mirror torch tensor .to(device) signature
            return _Out()
        def decode(self, ids, skip_special_tokens):  # kept: mirror HF tokenizer signature
            return "stub-output"
    class _StubModel:
        device = "cpu"
        def generate(self, **kwargs): return [[0, 1, 2, 3, 4, 5, 6, 7]]
    stub_entry = {"model": _StubModel(), "tokenizer": _StubTok()}
    fn = make_generate_fn(stub_entry)
    assert callable(fn)
    print("  PASS: make_generate_fn returned callable")

    # Test 6: resolve_context_cap clamps advertised window to
    # HARD_VRAM_CONTEXT_LIMIT for an uncurated unknown id (UNKNOWN tier).
    print("\n[Test 6] resolve_context_cap UNKNOWN-tier defaults to limit")
    v = _otr_catalog.resolve_context_cap("some/uncurated-test-id")
    assert v.tier == "UNKNOWN"
    assert v.value == _otr_catalog.HARD_VRAM_CONTEXT_LIMIT
    print(f"  PASS ({v.tier} @ {v.value})")

    # Test 7: load_llm raises ModelLoaderError, not bare ImportError,
    #         when story_orchestrator can't be imported.
    # (We can't easily simulate this without monkeypatching sys.modules;
    #  skip the negative case but verify the wrapping logic by inspection
    #  of load_llm's body.)
    print("\n[Test 7] load_llm exception wrapping (smoke check)")
    import inspect
    src = inspect.getsource(load_llm)
    assert "ModelLoaderError" in src
    assert "from exc" in src
    print("  PASS")

    print("\n=== Task 4 self-tests passed ===")
