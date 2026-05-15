"""S30 B1c tests -- unload_llm + request_slot + check_vram_fit.

Hard-mock fixture replaces every heavy path (snapshot_download, transformers
loaders, torch CUDA primitives, story_orchestrator._load_llm) so the test
suite never triggers a real download, GPU load, or CPU-side weight move.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from nodes import _otr_model_catalog as catalog
from nodes import _otr_model_loader as loader
from nodes._otr_model_inputs import VRAMFitFailedError


# ---------------------------------------------------------------------------
# Hard-mock fixture -- autouse to be safe (every test gets the stubs)
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "stub-prompt"

    def __call__(self, prompt, return_tensors):
        class _O:
            input_ids = type("S", (), {"shape": (1, 5)})()

            def to(self, device):
                return self

        return _O()

    def decode(self, ids, skip_special_tokens):
        return "stub-output"


class _FakeModel:
    device = "cpu"

    def __init__(self, model_id: str = ""):
        self.model_id = model_id
        self._cpu_moves = 0

    def to(self, device):
        self._cpu_moves += 1
        return self

    def generate(self, **kwargs):
        return [[0, 1, 2, 3, 4, 5, 6, 7]]


def _fake_story_orchestrator_load_llm(*args, **kwargs):
    model_id = kwargs.get("model_id_full") or (args[0] if args else "")
    return _FakeModel(model_id), _FakeTokenizer()


@pytest.fixture(autouse=True)
def _hard_mock_loader_paths(monkeypatch, tmp_path):
    """Patch every heavy primitive before any test runs.

    Per the plan: missing any of these patches risks pytest triggering
    a real 24 GB Mistral-Nemo download or a real from_pretrained GPU load.
    """
    # Disk-space check returns plenty.
    monkeypatch.setattr(catalog, "_free_disk_bytes_for", lambda _p: 500 * 1024**3)
    # auto_download_if_missing test seam: no-op snapshot_download.
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    monkeypatch.setenv("HF_TOKEN", "fake-token-for-tests")
    # No real HF API; for curated entries the catalog skips the call anyway.

    # story_orchestrator._load_llm seam: patch the attribute directly
    # on whatever module is in sys.modules, so the fake survives across
    # other tests that imported the real module before mine. Still
    # useful as a back-stop -- post-S31 B2 the body lives in
    # `_otr_model_loader.load_llm` (see seam below) and the orchestrator's
    # `_load_llm` is a ~10-line shim that delegates to the loader.
    import nodes.story_orchestrator as _real_so

    monkeypatch.setattr(_real_so, "_load_llm", _fake_story_orchestrator_load_llm)
    monkeypatch.setattr(_real_so, "_unload_llm", lambda: None)
    if hasattr(_real_so, "_LLM_CACHE"):
        # Reset orchestrator cache between tests so unload_llm's
        # legacy-teardown branch is a no-op (we don't want to actually
        # touch any model objects left behind by earlier suites).
        monkeypatch.setitem(
            _real_so._LLM_CACHE, "model", None  # type: ignore[union-attr]
        )
        monkeypatch.setitem(
            _real_so._LLM_CACHE, "tokenizer", None  # type: ignore[union-attr]
        )
        monkeypatch.setitem(
            _real_so._LLM_CACHE, "model_id", None  # type: ignore[union-attr]
        )

    # S31 B2: ported-body seam. `_otr_model_loader.load_llm` is now the
    # canonical home of the bitsandbytes / NF4 / 8-bit profile body
    # (PURE COPY out of `story_orchestrator._load_llm`). The legacy
    # `_so._load_llm` patch above is preserved as a back-stop but the
    # primary seam tests should drive through is the loader-side one.
    # Returning a cache_entry dict directly skips the entire transformers
    # / bitsandbytes path and never touches GPU.
    def _fake_loader_load_llm(model_id, **kwargs):
        canonical_id = str(model_id).split(" ")[0].strip()
        return {
            "model":       _FakeModel(canonical_id),
            "tokenizer":   _FakeTokenizer(),
            "model_id":    canonical_id,
            "device":      kwargs.get("device", "cpu"),
            "quantized":   False,
            "context_cap": kwargs.get("context_cap") or 8192,
        }

    monkeypatch.setattr(loader, "load_llm", _fake_loader_load_llm)

    # CUDA primitives: pretend cuda unavailable so the teardown skips
    # real torch.cuda.* calls and never touches a real GPU.
    try:
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    except ImportError:
        pass

    # Reset cache between tests so cache-hit assertions are deterministic.
    loader.LLM_CACHE.update({"model_id": None, "slot": None, "cache_entry": None})

    # auto_download_if_missing seam: monkeypatch to a no-op since we
    # don't actually want to import huggingface_hub for unit tests.
    monkeypatch.setattr(
        catalog,
        "auto_download_if_missing",
        lambda repo_id, **kw: f"/fake/snapshot/{repo_id}",
    )

    yield


# ---------------------------------------------------------------------------
# check_vram_fit -- verdict shape + tier behavior
# ---------------------------------------------------------------------------


def test_check_vram_fit_curated_pass_returns_pass_tier():
    v = catalog.check_vram_fit(catalog.DEFAULT_LLM, 8192)
    assert v.tier == "PASS"
    assert v.soak_tested is True
    assert v.estimated_gb > 0


def test_check_vram_fit_oversized_returns_fail():
    """B1d: TEST_OVERSIZED_LLM (uncurated) trips FAIL via the
    SPECIAL_VRAM_ESTIMATES_GB table -- no curated-splice mocking
    required. The fix closes the prior gap where uncurated oversized
    ids fell through to UNKNOWN instead of FAIL.
    """
    v = catalog.check_vram_fit(catalog.TEST_OVERSIZED_LLM, 8192)
    assert v.tier == "FAIL"
    # SPECIAL pins TEST_OVERSIZED_LLM at 42 GB resident; FAIL ratio
    # kicks in at 1.5x 14.5 GB = 21.75 GB.
    assert v.estimated_gb >= 30
    assert "pick a smaller model" in v.reason.lower()


def test_check_vram_fit_special_table_overrides_curated_lookup():
    """If a repo_id is in SPECIAL_VRAM_ESTIMATES_GB AND also (somehow)
    curated, the SPECIAL value wins. Guards against future splicing
    that could mask a known-oversize entry."""
    import nodes._otr_model_catalog as c

    fake_curated = catalog.CuratedModel(
        repo_id=catalog.TEST_OVERSIZED_LLM,
        requires_auth=True,
        loader_backend="transformers_safetensors",
        vram_fit_tier="FAIL",
        approx_safetensors_gb=140.0,
        notes="test-only oversize",
    )
    original_curated = c.CURATED_LLM_MODELS
    c.CURATED_LLM_MODELS = original_curated + (fake_curated,)
    try:
        v = catalog.check_vram_fit(catalog.TEST_OVERSIZED_LLM, 8192)
        assert v.tier == "FAIL"
        # SPECIAL value (42) wins, not curated/2 (70).
        assert v.estimated_gb == pytest.approx(42.0, abs=0.1)
    finally:
        c.CURATED_LLM_MODELS = original_curated


def test_check_vram_fit_warn_for_curated_warn_tier():
    """Qwen2.5-14B is curated WARN-tier."""
    v = catalog.check_vram_fit("Qwen/Qwen2.5-14B-Instruct", 8192)
    assert v.tier == "WARN"
    assert v.soak_tested is False


def test_check_vram_fit_unknown_for_arbitrary_uncurated():
    v = catalog.check_vram_fit("totally/uncurated-model", 8192)
    assert v.tier == "UNKNOWN"
    assert v.soak_tested is False


def test_check_vram_fit_custom_ceiling():
    """Operator can override the ceiling for testing or bigger rigs."""
    v = catalog.check_vram_fit(catalog.DEFAULT_LLM, 8192, ceiling_gb=64.0)
    assert v.tier == "PASS"
    assert v.ceiling_gb == 64.0


# ---------------------------------------------------------------------------
# unload_llm -- teardown order
# ---------------------------------------------------------------------------


def test_unload_llm_preserves_cache_identity():
    """B1d: id(LLM_CACHE) must NOT change across unload_llm(). The
    previous code rebound the module-level name (LLM_CACHE = {...}),
    which leaves any `from _otr_model_loader import LLM_CACHE` consumer
    holding a stale dict reference -- silently breaking slot transitions
    on the next request_slot call.
    """
    original_id = id(loader.LLM_CACHE)
    loader.LLM_CACHE["model_id"] = catalog.DEFAULT_LLM
    loader.LLM_CACHE["slot"] = "creative"
    loader.LLM_CACHE["cache_entry"] = {
        "model": _FakeModel(),
        "tokenizer": _FakeTokenizer(),
    }
    loader.unload_llm()
    assert id(loader.LLM_CACHE) == original_id
    assert loader.LLM_CACHE["model_id"] is None
    assert loader.LLM_CACHE["slot"] is None
    assert loader.LLM_CACHE["cache_entry"] is None


def test_unload_llm_clears_llm_cache():
    # Prime cache.
    loader.LLM_CACHE["model_id"] = catalog.DEFAULT_LLM
    loader.LLM_CACHE["slot"] = "creative"
    loader.LLM_CACHE["cache_entry"] = {"model": _FakeModel(), "tokenizer": _FakeTokenizer()}
    loader.unload_llm()
    assert loader.LLM_CACHE["model_id"] is None
    assert loader.LLM_CACHE["slot"] is None
    assert loader.LLM_CACHE["cache_entry"] is None


def test_unload_llm_idempotent_on_empty_cache():
    """Calling on an empty cache must not raise."""
    loader.LLM_CACHE.update({"model_id": None, "slot": None, "cache_entry": None})
    loader.unload_llm()  # must not raise
    assert loader.LLM_CACHE["model_id"] is None


def test_unload_llm_moves_model_to_cpu():
    fake_model = _FakeModel()
    loader.LLM_CACHE["model_id"] = catalog.DEFAULT_LLM
    loader.LLM_CACHE["cache_entry"] = {"model": fake_model, "tokenizer": _FakeTokenizer()}
    loader.unload_llm()
    assert fake_model._cpu_moves == 1


# ---------------------------------------------------------------------------
# request_slot -- 8-step sequence
# ---------------------------------------------------------------------------


def test_request_slot_rejects_unknown_slot_name():
    with pytest.raises(loader.ModelLoaderError):
        loader.request_slot("not-a-slot", catalog.DEFAULT_LLM)


def test_request_slot_creative_loads_default_llm():
    entry = loader.request_slot("creative", catalog.DEFAULT_LLM)
    assert entry["model_id"] == catalog.DEFAULT_LLM
    assert loader.LLM_CACHE["slot"] == "creative"
    assert loader.LLM_CACHE["model_id"] == catalog.DEFAULT_LLM


def test_request_slot_uses_ported_body():
    """S31 B2 architecture: `request_slot` drives the end-to-end load
    through `_otr_model_loader.load_llm` (the canonical home of the
    ported ~613-LOC bitsandbytes body). It must NOT route through
    `story_orchestrator._load_llm` -- the orchestrator's `_load_llm`
    is a B2-only shim that delegates BACK to the loader and is
    deleted at S31 B4.

    Structural assertion: AST-walk `request_slot`'s body looking for
    a `Call` to `load_llm` (the loader's). Reject any `Call` to
    `_load_llm` (orchestrator-side legacy name)."""
    import ast
    from pathlib import Path

    loader_src = Path(loader.__file__).read_text(encoding="utf-8")
    tree = ast.parse(loader_src)
    request_slot_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "request_slot":
            request_slot_fn = node
            break
    assert request_slot_fn is not None, "request_slot not found in loader"

    call_to_load_llm = False
    forbidden_legacy_calls: list[str] = []
    for sub in ast.walk(request_slot_fn):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        # `load_llm(...)` (bare Name) -- loader's local resolution.
        if isinstance(fn, ast.Name) and fn.id == "load_llm":
            call_to_load_llm = True
        # `<x>.load_llm(...)` -- module-qualified loader call.
        elif isinstance(fn, ast.Attribute) and fn.attr == "load_llm":
            call_to_load_llm = True
        # `<x>._load_llm(...)` -- forbidden legacy orchestrator call.
        elif isinstance(fn, ast.Attribute) and fn.attr == "_load_llm":
            forbidden_legacy_calls.append(f"line {sub.lineno}: .{fn.attr}")
        # Bare `_load_llm(...)` after import -- also forbidden.
        elif isinstance(fn, ast.Name) and fn.id == "_load_llm":
            forbidden_legacy_calls.append(f"line {sub.lineno}: {fn.id}")
    assert call_to_load_llm, (
        "request_slot must call `load_llm(...)` (the ported body in "
        "this same module). No such call found in the AST."
    )
    assert not forbidden_legacy_calls, (
        "request_slot must NOT call the legacy orchestrator "
        "`_load_llm` -- post-S31 B2 the body lives here in the loader, "
        "and the orchestrator surface is a one-commit-deep shim. "
        "Offenders:\n  " + "\n  ".join(forbidden_legacy_calls)
    )


def test_request_slot_same_model_returns_cached_entry(monkeypatch):
    """Slot 1 == Slot 2 must reuse one model cache."""
    load_calls = []
    original_load = loader.load_llm
    monkeypatch.setattr(
        loader,
        "load_llm",
        lambda mid, **kw: (load_calls.append(mid) or original_load(mid, **kw)),
    )

    entry1 = loader.request_slot("creative", catalog.DEFAULT_LLM)
    entry2 = loader.request_slot("technical", catalog.DEFAULT_LLM)

    assert entry1 is entry2  # exact same cached object
    assert len(load_calls) == 1  # second call hit the cache, no reload


def test_request_slot_different_model_triggers_full_teardown(monkeypatch):
    """Slot 1 != Slot 2 must call unload_llm exactly once between loads."""
    unload_calls: list[int] = []
    original_unload = loader.unload_llm

    def counting_unload():
        unload_calls.append(1)
        original_unload()

    monkeypatch.setattr(loader, "unload_llm", counting_unload)

    loader.request_slot("creative", catalog.DEFAULT_LLM)
    assert len(unload_calls) == 0  # first load has no prior resident
    loader.request_slot("technical", catalog.TEST_TECHNICAL_LLM)
    assert len(unload_calls) == 1  # transition unloaded once
    assert loader.LLM_CACHE["model_id"] == catalog.TEST_TECHNICAL_LLM


def test_request_slot_oversized_fails_before_download(monkeypatch):
    """B1d: VRAMFitFailedError must fire BEFORE auto_download_if_missing
    so a 70B-on-16GB pick never triggers a network pull or disk-space
    pre-check pass on a doomed-to-OOM load. Asserts both the exception
    type and the order-of-operations (download never called).
    """
    download_calls: list[str] = []

    def counting_auto_download(repo_id, **kw):
        download_calls.append(repo_id)
        return f"/fake/snapshot/{repo_id}"

    monkeypatch.setattr(
        catalog, "auto_download_if_missing", counting_auto_download
    )

    with pytest.raises(VRAMFitFailedError) as exc:
        loader.request_slot("creative", catalog.TEST_OVERSIZED_LLM)

    # FAIL fired before any download / disk-space pre-check.
    assert download_calls == []
    # SPECIAL pins TEST_OVERSIZED_LLM at 42 GB resident.
    assert exc.value.estimated_gb >= 30
    assert "pick a smaller model" in str(exc.value).lower()
