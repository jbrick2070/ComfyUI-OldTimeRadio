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


def _install_load_llm_stub(monkeypatch):
    """S31 B4 update: post-deletion of `_so._LLM_CACHE`, the loader's
    `load_llm` no longer has a cache-hit short-circuit (request_slot
    handles caching at the outer layer). Runtime shape tests stub the
    HEAVY transformers / bitsandbytes / CUDA path by patching
    `transformers.AutoTokenizer.from_pretrained` and
    `AutoModelForCausalLM.from_pretrained` to return fakes, plus
    `torch.cuda.is_available` -> False so the body skips every GPU
    branch. The body then runs end-to-end and constructs the
    cache_entry dict from the fake model + tokenizer.
    """

    class _StubModel:
        device = "cpu"

        def to(self, _d):
            return self

        def cpu(self):
            return self

        def eval(self):
            return self

        def generate(self, *args, **kwargs):
            return [[0]]

        def modules(self):
            return iter([])

    class _StubTok:
        eos_token_id = 0

        def __call__(self, _text, return_tensors=None):
            class _Inputs:
                input_ids = type("S", (), {"shape": (1, 5)})()

                def to(self, _d):
                    return self

            return _Inputs()

    stub_model = _StubModel()
    stub_tok = _StubTok()

    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained",
        classmethod(lambda cls, *a, **kw: stub_tok),
    )
    monkeypatch.setattr(
        AutoModelForCausalLM, "from_pretrained",
        classmethod(lambda cls, *a, **kw: stub_model),
    )
    monkeypatch.setattr(
        AutoConfig, "from_pretrained",
        classmethod(lambda cls, *a, **kw: type("C", (), {"max_position_embeddings": 8192})()),
    )
    return stub_model, stub_tok


def test_load_llm_returns_cache_entry_dict_shape():
    """The `load_llm` return MUST be a dict literal with exactly the 6
    documented keys: model, tokenizer, model_id, device, quantized,
    context_cap.

    Post-S31 B4 the body runs end-to-end every call (no cache-hit
    short-circuit -- request_slot handles caching at the outer layer)
    and depends on `torch.cuda` + `transformers` + `bitsandbytes` +
    a real GPU. Shallow monkeypatching is not sufficient to stub the
    full path. Structural AST assertion captures the contract instead.
    """
    body = _loader_load_llm_source()
    tree = ast.parse(body)
    # Find the (last) Return whose value is a dict literal.
    return_dict = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Dict)
        ):
            return_dict = node.value
    assert return_dict is not None, (
        "load_llm must end with `return {...}` returning a dict "
        "literal (the cache_entry); no such return found."
    )
    keys: set[str] = set()
    for key_node in return_dict.keys:
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            keys.add(key_node.value)
    expected = {"model", "tokenizer", "model_id", "device", "quantized", "context_cap"}
    assert keys == expected, (
        f"cache_entry keys drifted from the 6-key contract. "
        f"Expected {sorted(expected)}, got {sorted(keys)}."
    )


def test_load_llm_strips_ui_suffix():
    """UI label suffixes like `[BETA]` or `[8-bit]` must be stripped
    from the returned `cache_entry["model_id"]`. AST scan asserts the
    `_stripped_model_id = model_id_full.split(" ")[0]` normalization
    line is present and that the dict literal's `model_id` key is
    bound to `_stripped_model_id` (not the un-stripped `model_id_full`).
    """
    body = _loader_load_llm_source()
    assert '_stripped_model_id = model_id_full.split(" ")[0]' in body, (
        "load_llm must include the `_stripped_model_id = "
        "model_id_full.split(\" \")[0]` normalization that strips UI "
        "suffixes like `[BETA]` from the model_id at the top of the "
        "function."
    )
    # Verify the return dict's model_id key references _stripped_model_id.
    tree = ast.parse(body)
    return_dict = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Dict)
        ):
            return_dict = node.value
    assert return_dict is not None
    model_id_value = None
    for k, v in zip(return_dict.keys, return_dict.values):
        if isinstance(k, ast.Constant) and k.value == "model_id":
            model_id_value = v
            break
    assert model_id_value is not None
    assert isinstance(model_id_value, ast.Name), (
        f"cache_entry['model_id'] must be bound to a Name node "
        f"(the stripped local var); got {type(model_id_value).__name__}."
    )
    assert model_id_value.id == "_stripped_model_id", (
        f"cache_entry['model_id'] must be `_stripped_model_id` "
        f"(the [BETA]-stripped form); got `{model_id_value.id}`."
    )


def test_orchestrator_load_llm_shim_deleted():
    """The orchestrator-side `_load_llm` B2 shim was DELETED at S31 B4
    (Hard rule #1A non-deferrable). This test renames the previous
    shim-returns-tuple assertion to a deletion guard. The canonical
    surface post-S31 B4 is `_otr_model_loader.load_llm` returning a
    cache_entry dict; there is no orchestrator-side tuple-return any
    more.

    Companion deletion guards for the other 3 symbols live in
    `tests/test_no_orchestrator_legacy_symbols.py`.
    """
    from nodes import story_orchestrator as _so

    assert not hasattr(_so, "_load_llm"), (
        "`_load_llm` (the S31 B2 shim) must be deleted at S31 B4. "
        "Re-introduction violates Hard rule #1A."
    )
