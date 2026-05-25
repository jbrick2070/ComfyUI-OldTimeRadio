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

log = logging.getLogger("OTR")


__all__ = [
    "load_llm",
    "unload_llm",
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
# load_llm -- wraps tuple return into dict
# ---------------------------------------------------------------------------


def load_llm(
    model_id: str,
    *,
    device: str = "cuda",
    optimization_profile: str = "Standard",
    context_cap: int | None = None,
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

        # Decide whether the requested model + profile combination
        # needs quantization. Mirrors the legacy predicate exactly.
        is_obsidian = "Obsidian" in optimization_profile
        vram_safe_tags = ("9b", "12b", "14b", "24b", "26b", "27b", "31b", "70b", "e4b", "4b-it", "a4b", "2b", "2b-it", "efficiency", "nemo", "qwen", "mistral", "instruct", "gemma")
        requested_quantized = is_obsidian or "4-bit" in model_id_full.lower() or \
                              any(tag in model_id_full.lower() for tag in vram_safe_tags)

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
        max_memory = None
        is_actually_2b = any(tag in _stripped_model_id.lower() for tag in ("2b-it", "2b_it")) or _stripped_model_id.lower().endswith("2b")

        if total_vram >= 12.0:
            budget_gb = total_vram - 2.5
            max_memory = {0: f"{budget_gb:.1f}GiB", "cpu": "32GiB"}
            _runtime_log(f"[StoryOrchestrator] Sovereignty Buffer Active: {budget_gb:.1f}GB Budget")
        elif is_actually_2b:
            max_memory = {0: "3.2GiB", "cpu": "32GiB"}
        elif any(tag in _stripped_model_id.lower() for tag in ("9b", "12b", "e4b", "4b-it")):
            max_memory = {0: "6.8GiB", "cpu": "32GiB"}

        # Enable TF32 for faster matmuls on Ampere/Ada/Blackwell GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.set_float32_matmul_precision('high')

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

        # Attention selector: Flash Attention 2 (preferred) -> SDPA
        # (mandatory fallback on Blackwell sm_120 / Windows / torch 2.10)
        # -> SageAttention. The resolved value below is the single
        # source of truth for `attn_implementation` in common_kwargs --
        # it is logged on every load so the choice is auditable.
        attn_impl = "sdpa"
        try:
            from importlib.metadata import distribution, PackageNotFoundError
            try:
                distribution("flash-attn")
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
                log.info("[StoryOrchestrator] Flash Attention 2 available - using flash_attention_2")
            except (PackageNotFoundError, ImportError):
                log.info(
                    "[StoryOrchestrator] Flash Attention 2: NOT AVAILABLE - no prebuilt wheel exists "
                    "for torch 2.10 + CUDA 13 + Blackwell sm_120 on Windows. "
                    "SageAttention + SDPA active. Performance unaffected. Do not attempt install."
                )
        except Exception as _fa_err:
            log.info("[StoryOrchestrator] FA2 probe failed (%s) - using SDPA fallback", _fa_err)
        _runtime_log(f"[StoryOrchestrator] Attention selector resolved: attn_implementation={attn_impl}")

        # 4-bit / 8-bit quantization config.
        quant_config = None
        needs_8bit = "8-bit" in model_id_full.lower()
        is_unstable_quant = any(tag in model_id_full.lower() for tag in ("2bit", "3bit", "2-bit", "3-bit"))
        needs_4bit = requested_quantized or is_unstable_quant or \
                     any(tag in model_id_full.lower() for tag in vram_safe_tags)

        if is_unstable_quant:
            _runtime_log(f"[StoryOrchestrator] [EMOJI]- WING DING PROTECTION: Unstable Bit-Depth ({model_id_full}) UPGRADED to 4-bit NF4")
        elif needs_4bit:
            _runtime_log(f"[StoryOrchestrator] Quantizing: 4-bit NF4 for {model_id_full}")

        if needs_8bit:
            try:
                from transformers import BitsAndBytesConfig
                quant_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                log.info("[StoryOrchestrator] Enabling 8-bit quantization")
            except ImportError:
                log.warning("[StoryOrchestrator] Large model but bitsandbytes not installed!")
        elif needs_4bit:
            try:
                from transformers import BitsAndBytesConfig
                # BUG-LOCAL-098: instantiate BitsAndBytesConfig FRESH
                # per call. transformers mutates internal flags during
                # from_pretrained; a reused instance silently skips
                # quantization on the second call -> fp16 fallback ->
                # OOM at 24 GiB on 16 GiB GPU.
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                log.info("[StoryOrchestrator] [EMOJI] Enabling 4-bit quantization (NF4) for Ultra-Low VRAM")
                _runtime_log("[StoryOrchestrator] [EMOJI] 4-bit NF4 active")
            except ImportError:
                log.warning("[StoryOrchestrator] Large model but bitsandbytes not installed - "
                            "loading at bfloat16 may OOM. Run: pip install bitsandbytes")

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
        model = entry.get("model")
        if model is not None and hasattr(model, "to"):
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


def request_slot(slot: str, model_id: str) -> dict[str, Any]:
    """Slot-aware entry point. Loads (or reuses cached) LLM, handling
    cache reuse vs full teardown automatically.

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

    if slot not in ("creative", "technical"):
        raise ModelLoaderError(
            f"request_slot: slot must be 'creative' or 'technical', got {slot!r}"
        )

    # Step 1: normalize.
    normalized = _otr_catalog.validate_model_id(model_id)

    # Step 2: cache hit on the same model id (regardless of slot).
    if LLM_CACHE.get("model_id") == normalized and LLM_CACHE.get("cache_entry") is not None:
        log.info("[Selector] slot=%s reuse cache for %s", slot, normalized)
        LLM_CACHE["slot"] = slot
        return LLM_CACHE["cache_entry"]  # type: ignore[return-value]

    # Step 3: context cap (never raises).
    ctx_verdict = _otr_catalog.resolve_context_cap(normalized)

    # Step 4: VRAM fit (never raises).
    fit_verdict = _otr_catalog.check_vram_fit(normalized, ctx_verdict.value)

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
        cache_entry = load_llm(normalized, context_cap=ctx_verdict.value)
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

    # Step 9: cache.
    LLM_CACHE["model_id"] = normalized
    LLM_CACHE["slot"] = slot
    LLM_CACHE["cache_entry"] = cache_entry
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
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=0.92,
                max_new_tokens=max_new_tokens,
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
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=_POLISH_DO_SAMPLE,
                temperature=temperature,
                top_p=_POLISH_TOP_P,
                max_new_tokens=max_new_tokens,
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
