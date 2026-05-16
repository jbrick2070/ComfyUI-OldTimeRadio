"""Sprint D D1c -- runtime-gated talkie loader smoke.

Two assertions, both runtime-gated under OTR_REGRESSION_RUNTIME=1:

  test_talkie_load_smoke_runtime_gated
      Runs the talkie GPTQ int4 adapter through one load + warmup
      pass and asserts the resulting cache_entry shape matches the
      legacy load_llm contract (dict carrying model + tokenizer +
      device + model_id keys).

  test_talkie_unload_returns_vram_clean_runtime_gated
      Loads talkie, runs warmup, unloads, asserts torch.cuda
      allocated VRAM drops to ~zero post-unload (within tolerance
      for caching allocator slack).

Both tests require:
  * HF_TOKEN visible in shell env (talkie is a gated repo).
  * OTR_REGRESSION_RUNTIME=1.
  * Talkie GPTQ int4 weights downloadable (~7.5 GB one-time pull).
  * CUDA device available (cuda.is_available() True).

Per Sprint D D1c exit condition both tests stage as skipped under
the default pytest-only acceptance discipline. Sprint A or operator-
discretion runs them with the runtime gate set.

The GPTQ adapter's load/generate/unload currently raise
NotImplementedError at D1b -- D1c reserves these test fixtures and
flips them live in D-future once the AutoGPTQForCausalLM runtime
path actually lands. The runtime gate provides a clean staging
slot now without commit-loop friction.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OTR_REGRESSION_RUNTIME = os.environ.get("OTR_REGRESSION_RUNTIME") == "1"

TALKIE_REPO_ID = "talkie-lm/talkie-1930-13b-it"


@pytest.mark.skipif(
    not OTR_REGRESSION_RUNTIME,
    reason="runtime gate: set OTR_REGRESSION_RUNTIME=1 to exercise",
)
def test_talkie_load_smoke_runtime_gated() -> None:
    """Load talkie GPTQ int4, run 1-token warmup, assert cache_entry
    shape matches legacy load_llm contract.
    """
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; runtime smoke needs GPU")

    from nodes import _otr_model_catalog as catalog  # noqa: PLC0415
    from nodes import _otr_model_runtime as runtime  # noqa: PLC0415

    rows = {m.repo_id: m for m in catalog.CURATED_LLM_MODELS}
    talkie = rows[TALKIE_REPO_ID]
    adapter = runtime.get_backend_for_row(talkie)

    cache_entry = adapter.load(TALKIE_REPO_ID, talkie)

    # Contract: cache_entry is a dict carrying at minimum these keys
    # (matching the legacy load_llm shape in nodes/_otr_model_loader.py).
    for required_key in ("model", "tokenizer", "model_id", "device"):
        assert required_key in cache_entry, (
            f"cache_entry missing required key {required_key!r}; "
            f"present keys: {sorted(cache_entry)}"
        )
    assert cache_entry["model_id"] == TALKIE_REPO_ID, (
        f"cache_entry.model_id = {cache_entry['model_id']!r}; "
        f"expected {TALKIE_REPO_ID!r}"
    )

    # 1-token warmup pass per CLAUDE.md prime directive 2.
    messages = [{"role": "user", "content": "Test."}]
    out = adapter.generate(cache_entry, messages, max_new_tokens=1)
    assert isinstance(out, str), (
        f"adapter.generate returned {type(out).__name__}; "
        f"expected str"
    )

    adapter.unload(cache_entry)


@pytest.mark.skipif(
    not OTR_REGRESSION_RUNTIME,
    reason="runtime gate: set OTR_REGRESSION_RUNTIME=1 to exercise",
)
def test_talkie_unload_returns_vram_clean_runtime_gated() -> None:
    """After load + warmup + unload, torch.cuda allocated VRAM drops
    to within a small tolerance of pre-load level.

    The caching allocator holds some memory in its pool post-unload
    for fast re-allocation; the assertion uses a generous tolerance
    rather than strict zero.
    """
    import gc  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; runtime smoke needs GPU")

    from nodes import _otr_model_catalog as catalog  # noqa: PLC0415
    from nodes import _otr_model_runtime as runtime  # noqa: PLC0415

    rows = {m.repo_id: m for m in catalog.CURATED_LLM_MODELS}
    talkie = rows[TALKIE_REPO_ID]
    adapter = runtime.get_backend_for_row(talkie)

    torch.cuda.empty_cache()
    pre_load_gb = torch.cuda.memory_allocated() / 1e9

    cache_entry = adapter.load(TALKIE_REPO_ID, talkie)
    adapter.generate(
        cache_entry, [{"role": "user", "content": "Test."}],
        max_new_tokens=1,
    )
    post_load_gb = torch.cuda.memory_allocated() / 1e9
    assert post_load_gb > pre_load_gb, (
        f"VRAM allocated post-load ({post_load_gb:.2f} GB) did not "
        f"exceed pre-load ({pre_load_gb:.2f} GB); load may have failed"
    )

    adapter.unload(cache_entry)
    del cache_entry
    gc.collect()
    torch.cuda.empty_cache()

    post_unload_gb = torch.cuda.memory_allocated() / 1e9
    drift_gb = post_unload_gb - pre_load_gb
    # Generous tolerance for caching allocator slack.
    assert drift_gb < 1.0, (
        f"VRAM drift after unload: {drift_gb:.2f} GB (pre-load "
        f"{pre_load_gb:.2f} GB, post-unload {post_unload_gb:.2f} GB). "
        f"Allocator should release within 1.0 GB tolerance; larger "
        f"drift indicates a leak in the GPTQ adapter unload path."
    )
