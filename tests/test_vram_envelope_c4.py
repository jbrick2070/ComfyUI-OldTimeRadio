"""Sprint C C4: VRAM envelope lock-in (regression-guard tests only).

Per SPRINT.md §A.7 + §A.8 + §A.9: the four envelope items below all
shipped in earlier sprints (S21.1 ceiling + context cap, S30 B1b
Gemma-4-E4B override, S31 B4 non-blocking _run_with_timeout). C4 ships
no production code change -- it pins the constants so future drift
trips a loud test failure instead of a silent regression.

Tests come in two shapes:

1. **Module-attribute assertions.** Import the catalog/orchestrator
   modules and assert the constants resolve to the expected values.
   Fast, no GPU needed.

2. **AST walks.** Walk the orchestrator's `_run_with_timeout` source
   to confirm the non-blocking shutdown + cache-invalidation pattern.
   Catches accidental reverts to the BUG-LOCAL-228 anti-pattern (a
   blocking sync-barrier).

The audio C7 byte-identical canary runs via the existing
`tests/test_audio_byte_identical.py` and is not duplicated here.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_ORCHESTRATOR_PATH = _REPO_ROOT / "nodes" / "story_orchestrator.py"


# ---------------------------------------------------------------------------
# 1. DEFAULT_LLM (L-1) -- audio C7 baseline pin
# ---------------------------------------------------------------------------


def test_default_llm_is_mistral_nemo():
    """L-1: DEFAULT_LLM must remain `mistralai/Mistral-Nemo-Instruct-2407`.

    This is the audio C7 byte-identical baseline. Any change here
    requires a deliberate baseline-reset sprint (out of Sprint C
    scope per §8 deferred items).
    """
    catalog = importlib.import_module("nodes._otr_model_catalog")
    assert catalog.DEFAULT_LLM == "mistralai/Mistral-Nemo-Instruct-2407", (
        f"DEFAULT_LLM drift: expected "
        f"'mistralai/Mistral-Nemo-Instruct-2407', got "
        f"{catalog.DEFAULT_LLM!r}. Audio C7 baseline pinned to "
        "Mistral-Nemo -- any change requires a deliberate baseline-reset "
        "sprint, not a casual edit."
    )


# ---------------------------------------------------------------------------
# 2-3. DEFAULT_VRAM_CEILING_GB + HARD_VRAM_CONTEXT_LIMIT
# ---------------------------------------------------------------------------


def test_default_vram_ceiling_14_5():
    """S21.1: VRAM ceiling is 14.5 GB on the 16 GB envelope."""
    catalog = importlib.import_module("nodes._otr_model_catalog")
    assert catalog.DEFAULT_VRAM_CEILING_GB == 14.5, (
        f"DEFAULT_VRAM_CEILING_GB drift: expected 14.5, got "
        f"{catalog.DEFAULT_VRAM_CEILING_GB}. The 14.5 GB ceiling on a "
        "16 GB envelope leaves 1.5 GB of headroom for transient "
        "activations and CUDA fragmentation -- the value was sized "
        "through S21.1 OOM forensics."
    )


def test_hard_context_limit_8192():
    """S21.1 + S30 B1b: hard context cap of 8192 tokens."""
    catalog = importlib.import_module("nodes._otr_model_catalog")
    # Default value when OTR_HARD_VRAM_CONTEXT_LIMIT env var is unset
    # (resolved at module import time by _hard_vram_context_limit()).
    # Operators on larger hardware can raise via env var; the default
    # locks 8192 so VRAM budgets are predictable across models.
    assert catalog.HARD_VRAM_CONTEXT_LIMIT == 8192, (
        f"HARD_VRAM_CONTEXT_LIMIT drift: expected 8192, got "
        f"{catalog.HARD_VRAM_CONTEXT_LIMIT}. The 8192 cap matches "
        "Mistral-Nemo and Gemma-4 native windows AND leaves headroom "
        "for KV-cache growth without paying the 16K context tax."
    )


# ---------------------------------------------------------------------------
# 4. Gemma-4-E4B-it context override
# ---------------------------------------------------------------------------


def test_gemma_4_e4b_context_override_8192():
    """S30 B1b: Gemma-4-E4B-it has an explicit 8192 context override."""
    catalog = importlib.import_module("nodes._otr_model_catalog")
    overrides = catalog.CURATED_CONTEXT_OVERRIDES
    assert "google/gemma-4-E4B-it" in overrides, (
        "google/gemma-4-E4B-it missing from CURATED_CONTEXT_OVERRIDES; "
        "without the explicit override the loader would fall back to "
        "the model's native 32K window and blow the VRAM ceiling."
    )
    assert overrides["google/gemma-4-E4B-it"] == 8192, (
        f"google/gemma-4-E4B-it override drift: expected 8192, got "
        f"{overrides['google/gemma-4-E4B-it']}"
    )


# ---------------------------------------------------------------------------
# 5-6. _run_with_timeout non-blocking + cache invalidation (BUG-LOCAL-228)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def orchestrator_ast() -> ast.AST:
    return ast.parse(_ORCHESTRATOR_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def run_with_timeout_node(orchestrator_ast) -> ast.FunctionDef:
    for node in ast.walk(orchestrator_ast):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_run_with_timeout"
        ):
            return node
    raise AssertionError(
        "Could not locate `_run_with_timeout` function in "
        "nodes/story_orchestrator.py"
    )


def test_run_with_timeout_non_blocking(run_with_timeout_node):
    """S31 B4 / BUG-LOCAL-228: `executor.shutdown` is called with
    `wait=False` so the orphan worker drains naturally instead of
    blocking the calling thread on an uninterruptible CUDA kernel.
    """
    saw_non_blocking_shutdown = False
    for node in ast.walk(run_with_timeout_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_shutdown = (
            isinstance(func, ast.Attribute) and func.attr == "shutdown"
        )
        if not is_shutdown:
            continue
        for kw in node.keywords:
            if kw.arg == "wait" and isinstance(kw.value, ast.Constant):
                if kw.value.value is False:
                    saw_non_blocking_shutdown = True
    assert saw_non_blocking_shutdown, (
        "`_run_with_timeout` no longer calls `executor.shutdown(wait=False)`. "
        "The non-blocking pattern is the BUG-LOCAL-228 contract -- "
        "reverting to a blocking shutdown will freeze ComfyUI's queue "
        "when an LLM timeout fires."
    )


def test_run_with_timeout_invalidates_cache(run_with_timeout_node):
    """S31 B4 / BUG-LOCAL-228: the timeout branch calls
    `invalidate_cache_no_gpu_teardown` so the next request_slot loads
    a fresh model without touching the GPU during the orphan-drain
    window.
    """
    saw_invalidate = False
    for node in ast.walk(run_with_timeout_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "invalidate_cache_no_gpu_teardown"
        ):
            saw_invalidate = True
        elif (
            isinstance(func, ast.Name)
            and func.id == "invalidate_cache_no_gpu_teardown"
        ):
            saw_invalidate = True
    assert saw_invalidate, (
        "`_run_with_timeout` no longer references "
        "`invalidate_cache_no_gpu_teardown` -- the cache-dict "
        "invalidation pattern is part of the BUG-LOCAL-228 contract. "
        "Without it the next request_slot would reuse a cache entry "
        "whose tensors are still being mutated by the orphan worker, "
        "producing nondeterministic behavior or a CUDA error."
    )
