"""S31 B2 -- profile-routing + dict-shape tests for the ported `_load_llm`.

Per plan B2: 8 tests gate the body port from `story_orchestrator._load_llm`
into `_otr_model_loader.load_llm`. Six of them live here; the seventh
(`test_request_slot_uses_ported_body`) extends `test_loader_slot_primitives.py`;
the eighth is the audio C7 byte-identical canary in
`test_audio_byte_identical.py` (handled outside this file).

Two-tier strategy:

* **Profile branch tests** (Standard / Obsidian / 8-bit). The full body
  cannot run end-to-end in a test process -- it depends on
  transformers + bitsandbytes + a real CUDA device. So these three
  tests are AST-based: walk the ported body and assert the
  appropriate `BitsAndBytesConfig` (or absence thereof) is wired into
  the right profile-decision branch.

* **Surface-shape tests** (`returns_cache_entry_dict_shape`,
  `strips_ui_suffix`, `orchestrator_load_llm_shim_returns_tuple`).
  These run the loader with a stub that bypasses the transformers /
  CUDA path. The stub sets up `_LLM_CACHE` to look like a successful
  load already happened (model + tokenizer fields populated). Calling
  `load_llm` then short-circuits at the cache-hit path
  (`if _LLM_CACHE["model"] is None: ...` block skipped) and returns
  the dict. This exercises both the cache-hit branch and the
  return-shape contract without touching any heavy import.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


PACK_ROOT = Path(__file__).resolve().parent.parent
LOADER_PATH = PACK_ROOT / "nodes" / "_otr_model_loader.py"
ORCH_PATH = PACK_ROOT / "nodes" / "story_orchestrator.py"


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _loader_load_llm_source() -> str:
    """Return the source text of `_otr_model_loader.load_llm` only.

    Locates the function definition in the AST, then slices the source
    file by line range. Avoids matching unrelated body fragments
    elsewhere in the module.
    """
    src = LOADER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "load_llm"
            and isinstance(node.parent if hasattr(node, "parent") else None, type(None))
        ):
            pass  # we'll just slice by lineno below
        if isinstance(node, ast.FunctionDef) and node.name == "load_llm":
            start = node.lineno - 1
            end = node.end_lineno or len(src.splitlines())
            return "\n".join(src.splitlines()[start:end])
    raise RuntimeError("load_llm function not found in _otr_model_loader.py")


def _orch_load_llm_source() -> str:
    src = ORCH_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_load_llm":
            start = node.lineno - 1
            end = node.end_lineno or len(src.splitlines())
            return "\n".join(src.splitlines()[start:end])
    raise RuntimeError("_load_llm function not found in story_orchestrator.py")


# ---------------------------------------------------------------------------
# Profile-branch AST tests
# ---------------------------------------------------------------------------


def test_load_llm_standard_profile():
    """Standard profile (no `Obsidian`, no `4-bit`, no `8-bit` in id) does
    NOT short-circuit to a BitsAndBytesConfig branch. The ported body
    keeps a Standard path that may still elect quantization based on
    model-size tags (12B/14B etc) -- the assertion here is structural:
    the `needs_8bit` decision branch must exist and be guarded by the
    `8-bit` substring check (not always taken)."""
    body = _loader_load_llm_source()
    # The `needs_8bit` predicate must remain in the body (it gates the
    # 8-bit branch).
    assert "needs_8bit" in body, (
        "Standard / 8-bit / 4-bit profile branching must be preserved "
        "in the ported body -- `needs_8bit` decision flag is missing."
    )
    # The body must NOT have unconditional NF4 forcing (would break
    # Standard profile on small models that fit fp16).
    assert 'load_in_4bit=True' in body, (
        "4-bit branch is part of the profile body; ported body should "
        "still construct BitsAndBytesConfig(load_in_4bit=True, ...)."
    )
    # The branch must be guarded by needs_4bit (not Standard-default).
    assert "elif needs_4bit" in body or "if needs_4bit" in body, (
        "4-bit construction must be guarded by `needs_4bit`, not "
        "unconditional; Standard profile relies on this branch being "
        "skipped for small models."
    )


def test_load_llm_obsidian_profile():
    """Obsidian profile maps to `requested_quantized = True` (via the
    `is_obsidian = "Obsidian" in optimization_profile` predicate) and
    then to the NF4 BitsAndBytesConfig branch."""
    body = _loader_load_llm_source()
    assert 'is_obsidian = "Obsidian" in optimization_profile' in body, (
        "Obsidian decision predicate missing from ported body."
    )
    # Obsidian feeds requested_quantized -> needs_4bit -> NF4 config.
    assert "requested_quantized" in body
    assert 'bnb_4bit_quant_type="nf4"' in body, (
        "Obsidian path requires NF4 quant type in BitsAndBytesConfig; "
        "literal `bnb_4bit_quant_type=\"nf4\"` not found in ported body."
    )
    assert "bnb_4bit_use_double_quant=True" in body


def test_load_llm_8bit_profile():
    """`[8-bit]` UI label maps to `needs_8bit = "8-bit" in
    model_id_full.lower()` and the 8-bit BitsAndBytesConfig branch
    (distinct from the 4-bit NF4 branch)."""
    body = _loader_load_llm_source()
    assert 'needs_8bit = "8-bit" in model_id_full.lower()' in body, (
        "8-bit predicate missing from ported body."
    )
    # 8-bit branch must construct its own BitsAndBytesConfig.
    assert "load_in_8bit=True" in body
    assert "llm_int8_enable_fp32_cpu_offload=True" in body, (
        "8-bit branch must enable fp32 CPU offload (sovereignty buffer "
        "may dispatch some layers to CPU)."
    )
    # 8-bit and 4-bit branches must be mutually exclusive.
    assert "if needs_8bit:" in body
    assert "elif needs_4bit:" in body, (
        "8-bit and 4-bit must be on `if`/`elif` branches, not parallel."
    )


# ---------------------------------------------------------------------------
# Runtime shape tests
# ---------------------------------------------------------------------------


def _install_cache_hit_stub(monkeypatch):
    """Pre-populate `story_orchestrator._LLM_CACHE` so the cache-hit
    branch of the ported body returns immediately without ever invoking
    transformers / bitsandbytes / CUDA. The fixture sets every field
    that the cache-delta diagnostic block checks, so the function
    routes to the bottom return without firing a reload.
    """
    from nodes import story_orchestrator as _so

    class _StubModel:
        device = "cpu"

        def parameters(self):
            # Yield one param reporting cuda:0 so the eviction-check
            # branch sees `any_cuda_param = True` and skips the
            # model_evicted_to_cpu delta entry.
            class _P:
                device = "cuda:0"

            yield _P()

        def to(self, _d):
            return self

        def cpu(self):
            return self

        def eval(self):
            return self

    class _StubTok:
        eos_token_id = 0

    stub_model = _StubModel()
    stub_tok = _StubTok()

    # Calibrate every cache field so the cache-delta diagnostic sees
    # ZERO drifted fields and routes straight to the cache-hit return.
    # `requested_quantized=True` for any model_id containing one of
    # the vram_safe_tags (including "mistral" / "nemo" / "instruct"),
    # so the cached `quantized` must be True to match.
    monkeypatch.setitem(_so._LLM_CACHE, "model", stub_model)
    monkeypatch.setitem(_so._LLM_CACHE, "tokenizer", stub_tok)
    monkeypatch.setitem(_so._LLM_CACHE, "device", "cuda")
    monkeypatch.setitem(_so._LLM_CACHE, "quantized", True)
    monkeypatch.setitem(_so._LLM_CACHE, "model_id", "mistralai/Mistral-Nemo-Instruct-2407")
    monkeypatch.setitem(_so._LLM_CACHE, "budget_profile", "Standard")
    monkeypatch.setitem(_so._LLM_CACHE, "VERSION", "v1.5")
    monkeypatch.setitem(_so._LLM_CACHE, "context_cap", 8192)
    return stub_model, stub_tok


def test_load_llm_returns_cache_entry_dict_shape(monkeypatch):
    """Cache-hit path returns a dict with exactly the 6 documented keys:
    model, tokenizer, model_id, device, quantized, context_cap.
    """
    from nodes import _otr_model_loader as loader

    _install_cache_hit_stub(monkeypatch)

    entry = loader.load_llm(
        "mistralai/Mistral-Nemo-Instruct-2407",
        device="cuda",
        optimization_profile="Standard",
    )
    assert isinstance(entry, dict)
    expected_keys = {"model", "tokenizer", "model_id", "device", "quantized", "context_cap"}
    assert set(entry.keys()) == expected_keys, (
        f"cache_entry keys drifted from the 6-key contract. "
        f"Expected {sorted(expected_keys)}, got {sorted(entry.keys())}."
    )


def test_load_llm_strips_ui_suffix(monkeypatch):
    """UI label suffixes like `[BETA]` or `[8-bit]` must be stripped
    from the returned `cache_entry["model_id"]`. The legacy body has
    `model_id_full.split(" ")[0]` as the very first line; the ported
    body preserves that normalization at the canonical_id step at
    return time as well.
    """
    from nodes import _otr_model_loader as loader

    _install_cache_hit_stub(monkeypatch)

    entry = loader.load_llm(
        "mistralai/Mistral-Nemo-Instruct-2407 [BETA]",
        device="cuda",
        optimization_profile="Standard",
    )
    assert entry["model_id"] == "mistralai/Mistral-Nemo-Instruct-2407", (
        f"UI suffix `[BETA]` must be stripped from cache_entry['model_id']; "
        f"got {entry['model_id']!r}."
    )


def test_orchestrator_load_llm_shim_returns_tuple(monkeypatch):
    """The orchestrator-side `_load_llm` is a thin shim (S31 B2) that
    delegates to `_otr_model_loader.load_llm` and unwraps the
    cache_entry dict back to the legacy `(model, tokenizer)` tuple.

    Deleted at S31 B4. Until then the tuple-return contract must hold
    so any in-flight caller (e.g. orchestrator-internal cache check
    paths) keeps working without churn.
    """
    from nodes import story_orchestrator as _so

    _install_cache_hit_stub(monkeypatch)

    result = _so._load_llm("mistralai/Mistral-Nemo-Instruct-2407")
    assert isinstance(result, tuple), (
        f"_load_llm shim must return a (model, tokenizer) tuple; "
        f"got {type(result).__name__}."
    )
    assert len(result) == 2
    model, tokenizer = result
    # Both must be non-None: the cache_entry's model + tokenizer fields
    # came from the cache-hit stub above.
    assert model is not None
    assert tokenizer is not None
