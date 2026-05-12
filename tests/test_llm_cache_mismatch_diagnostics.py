"""Regression test for BUG-LOCAL-065: LLM cache-mismatch reload thrash.

Bug: between LLM phases (title-gen, OpenClose, draft, critique, revision,
arc-enhancer) inside a single OTR_LedgerScriptWriter run, the cache mismatch
check at story_orchestrator.py:_load_llm fired between every phase. Each
mismatch unloaded the model and re-cold-loaded it (~15-17 s wall-clock per
phase) even though all inputs (model_id, optimization_profile, context_cap)
were identical.

Root cause investigation (live log 2026-04-26 14:57:05 + 15:00:16):
- runtime log printed "LLM cache mismatch (Context Cap: 6144) - reloading
  to enforce budget" -- but the message did not say WHICH field drifted.
- prime suspect: the parameter-device check at line 957-961 returned
  current_model_device == "cpu" because bitsandbytes 4-bit quantized models
  have non-deterministic parameter iteration order; the first parameter can
  report 'cpu' (quantization metadata buffers, embed-token offload pages)
  even though the model is correctly resident on cuda for inference.

Fix:
1. Scan up to 8 parameters and accept the cache as valid if ANY of them
   are on cuda. Only flag eviction when ALL inspected params are off-cuda.
2. Replace the inline boolean cache check with explicit per-field delta
   collection so the runtime log shows exactly which fields drifted.

This test exercises the diagnostic logic without requiring GPU/transformers:
it constructs a fake _LLM_CACHE dict and asserts that the field-by-field
delta computation correctly identifies (or does not identify) drift for
the realistic intra-run scenarios.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _stub_heavy_imports():
    """Story orchestrator imports torch/transformers/comfy at module load.
    Replace them with light stubs so this test can run without the GPU
    stack installed.
    """
    for name in ("torch", "torch.cuda", "transformers", "comfy",
                 "comfy.model_management"):
        sys.modules.setdefault(name, types.ModuleType(name))
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        if not hasattr(torch, "cuda"):
            torch.cuda = types.SimpleNamespace(
                is_available=lambda: False,
                empty_cache=lambda: None,
                memory_allocated=lambda: 0,
                memory_reserved=lambda: 0,
                ipc_collect=lambda: None,
                get_device_properties=lambda i: types.SimpleNamespace(
                    total_memory=16 * 1024**3),
            )


def _compute_cache_deltas(cache, *, device, requested_quantized, model_id,
                          optimization_profile, _cap, current_model_device,
                          any_cuda_param):
    """Mirror of the diagnostic logic embedded in _load_llm.

    Reproduced here as a pure function so the regression test can exercise
    every drift scenario without touching the heavy import path.
    """
    cache_deltas = []
    if cache.get("model") is None:
        return cache_deltas
    if str(cache["device"]) != str(device):
        cache_deltas.append(("device", cache["device"], device))
    if cache["quantized"] != requested_quantized:
        cache_deltas.append(("quantized", cache["quantized"],
                             requested_quantized))
    if cache["model_id"] != model_id:
        cache_deltas.append(("model_id", cache["model_id"], model_id))
    if cache.get("budget_profile") != optimization_profile:
        cache_deltas.append(("budget_profile",
                             cache.get("budget_profile"),
                             optimization_profile))
    if cache.get("VERSION") != "v1.5":
        cache_deltas.append(("VERSION", cache.get("VERSION"), "v1.5"))
    if cache.get("context_cap") != _cap:
        cache_deltas.append(("context_cap",
                             cache.get("context_cap"), _cap))
    if ("cuda" in str(device)
            and "cpu" in current_model_device
            and not any_cuda_param):
        cache_deltas.append(("model_evicted_to_cpu",
                             current_model_device, str(device)))
    return cache_deltas


def _warm_cache():
    """Realistic mid-run cache state for Mistral-Nemo + Pro Ultra Quality."""
    return {
        "model": object(),       # placeholder -- presence == loaded
        "tokenizer": object(),
        "device": "cuda",
        "quantized": True,
        "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "budget_profile": "Pro (Ultra Quality)",
        "VERSION": "v1.5",
        "context_cap": 6144,
    }


# ---- the realistic-run scenarios that were thrashing in BUG-LOCAL-065 ----


class TestNoMismatchAcrossLLMPhases:
    """All these scenarios call _load_llm with identical args -- there
    should be ZERO drift, so the cache should be reused."""

    def test_title_gen_to_openclose_no_drift(self):
        cache = _warm_cache()
        deltas = _compute_cache_deltas(
            cache,
            device="cuda",
            requested_quantized=True,
            model_id="mistralai/Mistral-Nemo-Instruct-2407",
            optimization_profile="Pro (Ultra Quality)",
            _cap=6144,
            current_model_device="cuda:0",
            any_cuda_param=True,
        )
        assert deltas == [], f"unexpected drift: {deltas}"

    def test_openclose_to_draft_no_drift(self):
        cache = _warm_cache()
        deltas = _compute_cache_deltas(
            cache,
            device="cuda",
            requested_quantized=True,
            model_id="mistralai/Mistral-Nemo-Instruct-2407",
            optimization_profile="Pro (Ultra Quality)",
            _cap=6144,
            current_model_device="cuda:0",
            any_cuda_param=True,
        )
        assert deltas == []

    def test_draft_to_critique_to_revision_no_drift(self):
        cache = _warm_cache()
        for _phase in ("critique", "revision", "arc_enhancer"):
            deltas = _compute_cache_deltas(
                cache,
                device="cuda",
                requested_quantized=True,
                model_id="mistralai/Mistral-Nemo-Instruct-2407",
                optimization_profile="Pro (Ultra Quality)",
                _cap=6144,
                current_model_device="cuda:0",
                any_cuda_param=True,
            )
            assert deltas == [], f"phase={_phase} drift={deltas}"

    def test_mistral_nemo_warm_run_no_drift(self):
        cache = _warm_cache()
        cache["model_id"] = "mistralai/Mistral-Nemo-Instruct-2407"
        deltas = _compute_cache_deltas(
            cache,
            device="cuda",
            requested_quantized=True,
            model_id="mistralai/Mistral-Nemo-Instruct-2407",
            optimization_profile="Pro (Ultra Quality)",
            _cap=6144,
            current_model_device="cuda:0",
            any_cuda_param=True,
        )
        assert deltas == []


class TestQuantizedFirstParamCpuFalsePositive:
    """The pre-fix behavior: bnb 4-bit puts a CPU-resident metadata
    buffer first in parameters() ordering, so first param reports 'cpu'.
    The fix is to scan up to 8 params and accept ANY cuda placement."""

    def test_first_param_cpu_but_others_cuda_does_not_evict(self):
        cache = _warm_cache()
        deltas = _compute_cache_deltas(
            cache,
            device="cuda",
            requested_quantized=True,
            model_id="mistralai/Mistral-Nemo-Instruct-2407",
            optimization_profile="Pro (Ultra Quality)",
            _cap=6144,
            current_model_device="cpu",  # first param reports cpu
            any_cuda_param=True,          # but later params are cuda
        )
        assert deltas == [], f"false-positive eviction: {deltas}"

    def test_all_params_cpu_does_evict(self):
        cache = _warm_cache()
        deltas = _compute_cache_deltas(
            cache,
            device="cuda",
            requested_quantized=True,
            model_id="mistralai/Mistral-Nemo-Instruct-2407",
            optimization_profile="Pro (Ultra Quality)",
            _cap=6144,
            current_model_device="cpu",
            any_cuda_param=False,  # genuinely evicted
        )
        keys = [k for k, _, _ in deltas]
        assert "model_evicted_to_cpu" in keys


class TestRealMismatchesStillDrift:
    """The fix must NOT mask legitimate cache-invalidation triggers."""

    def test_model_id_change_drifts(self):
        cache = _warm_cache()
        deltas = _compute_cache_deltas(
            cache,
            device="cuda",
            requested_quantized=True,
            # warm cache holds Mistral-Nemo (the production default);
            # request a different model to confirm legitimate model-id
            # mismatches still register as drift after the BUG-066 fix.
            model_id="google/gemma-2-9b-it",  # changed
            optimization_profile="Pro (Ultra Quality)",
            _cap=6144,
            current_model_device="cuda:0",
            any_cuda_param=True,
        )
        keys = [k for k, _, _ in deltas]
        assert "model_id" in keys

    def test_profile_change_drifts(self):
        cache = _warm_cache()
        deltas = _compute_cache_deltas(
            cache,
            device="cuda",
            requested_quantized=True,
            model_id="mistralai/Mistral-Nemo-Instruct-2407",
            optimization_profile="Standard",  # changed
            _cap=6144,
            current_model_device="cuda:0",
            any_cuda_param=True,
        )
        keys = [k for k, _, _ in deltas]
        assert "budget_profile" in keys

    def test_context_cap_change_drifts(self):
        cache = _warm_cache()
        cache["context_cap"] = 8192
        deltas = _compute_cache_deltas(
            cache,
            device="cuda",
            requested_quantized=True,
            model_id="mistralai/Mistral-Nemo-Instruct-2407",
            optimization_profile="Pro (Ultra Quality)",
            _cap=6144,
            current_model_device="cuda:0",
            any_cuda_param=True,
        )
        keys = [k for k, _, _ in deltas]
        assert "context_cap" in keys

    def test_version_bump_drifts(self):
        cache = _warm_cache()
        cache["VERSION"] = "v1.4"  # legacy cache survived a code update
        deltas = _compute_cache_deltas(
            cache,
            device="cuda",
            requested_quantized=True,
            model_id="mistralai/Mistral-Nemo-Instruct-2407",
            optimization_profile="Pro (Ultra Quality)",
            _cap=6144,
            current_model_device="cuda:0",
            any_cuda_param=True,
        )
        keys = [k for k, _, _ in deltas]
        assert "VERSION" in keys

    def test_quantized_flag_change_drifts(self):
        cache = _warm_cache()
        deltas = _compute_cache_deltas(
            cache,
            device="cuda",
            requested_quantized=False,  # user picked Obsidian + non-quant
            model_id="mistralai/Mistral-Nemo-Instruct-2407",
            optimization_profile="Pro (Ultra Quality)",
            _cap=6144,
            current_model_device="cuda:0",
            any_cuda_param=True,
        )
        keys = [k for k, _, _ in deltas]
        assert "quantized" in keys


class TestEmptyCacheIsNotADrift:
    """If the cache is fresh (model is None), there are no deltas to report
    -- the load path simply runs."""

    def test_fresh_cache_returns_empty(self):
        empty_cache = {"model": None}
        deltas = _compute_cache_deltas(
            empty_cache,
            device="cuda",
            requested_quantized=True,
            model_id="mistralai/Mistral-Nemo-Instruct-2407",
            optimization_profile="Pro (Ultra Quality)",
            _cap=6144,
            current_model_device="cpu",
            any_cuda_param=False,
        )
        assert deltas == []
