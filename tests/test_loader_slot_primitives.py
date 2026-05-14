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
    # other tests that imported the real module before mine.
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
    # Inject a synthetic 70B-class curated entry via monkey-patched
    # _by_repo_id? Simpler: assert on the real-world TEST_OVERSIZED_LLM
    # which is uncurated -> UNKNOWN. To get FAIL we need a curated
    # entry over the limit. Use a manual call site instead.
    fake_curated = catalog.CuratedModel(
        repo_id="meta-llama/Llama-3.1-70B-Instruct",
        requires_auth=True,
        loader_backend="transformers_safetensors",
        vram_fit_tier="FAIL",
        approx_safetensors_gb=140.0,
        notes="test-only oversize",
    )
    # Splice into the by-repo-id lookup for this test only.
    import nodes._otr_model_catalog as c

    original_curated = c.CURATED_LLM_MODELS
    c.CURATED_LLM_MODELS = original_curated + (fake_curated,)
    try:
        v = catalog.check_vram_fit("meta-llama/Llama-3.1-70B-Instruct", 8192)
        assert v.tier == "FAIL"
        # _estimate_resident_gb divides BF16-download-size by 2 to
        # match the OTR loader's 8-bit/NF4 default. 140 GB -> 70 GB
        # resident; FAIL ratio kicks in at 1.5x ceiling (21.75 GB).
        assert v.estimated_gb >= 50
        assert "pick a smaller model" in v.reason.lower()
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


def test_request_slot_propagates_vram_fit_fail():
    """A clearly-oversize model raises VRAMFitFailedError before any load."""
    fake_curated = catalog.CuratedModel(
        repo_id="meta-llama/Llama-3.1-70B-Instruct",
        requires_auth=True,
        loader_backend="transformers_safetensors",
        vram_fit_tier="FAIL",
        approx_safetensors_gb=140.0,
        notes="test-only oversize",
    )
    import nodes._otr_model_catalog as c

    original_curated = c.CURATED_LLM_MODELS
    c.CURATED_LLM_MODELS = original_curated + (fake_curated,)
    try:
        with pytest.raises(VRAMFitFailedError) as exc:
            loader.request_slot("creative", "meta-llama/Llama-3.1-70B-Instruct")
        assert exc.value.estimated_gb >= 50  # 140 BF16 -> ~70 resident
        assert "pick a smaller model" in str(exc.value).lower()
    finally:
        c.CURATED_LLM_MODELS = original_curated
