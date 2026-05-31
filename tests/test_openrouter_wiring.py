"""S3 -- the remote branch is wired into the LIVE loader path.

Proves FC2: request_slot routes an openrouter_http row to the remote
backend, skipping every local step (context cap / vram fit / download /
teardown / load_llm) and leaving the resident local model in LLM_CACHE
UNTOUCHED (C2 no-evict); and the three generate-fn factories return a
working remote callable for a provider-tagged entry. No network, no
torch -- the only HTTP seam is patched.
"""
from __future__ import annotations

import pytest

from nodes import _otr_model_loader as loader
from nodes import _otr_model_catalog as cat
from nodes import _otr_openrouter_backend as orb


SLOT_A = "openrouter:slot-a"


@pytest.fixture(autouse=True)
def _reset_budget(monkeypatch):
    orb.reset_run_budget()
    monkeypatch.setattr(orb.time, "sleep", lambda *_a, **_k: None)
    yield
    orb.reset_run_budget()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OTR_ENABLE_OPENROUTER", "1")
    monkeypatch.setenv("OPENROUTER_MODEL_A", "anthropic/claude-3.5-sonnet")
    monkeypatch.setenv("OPENROUTER_MODEL_B", "openai/gpt-4o")


@pytest.fixture
def cache_snapshot():
    """Snapshot + restore the module-level LLM_CACHE so a test that
    seeds a fake resident model can't leak into the next test."""
    saved = dict(loader.LLM_CACHE)
    yield
    loader.LLM_CACHE.clear()
    loader.LLM_CACHE.update(saved)


def _ok_post(content="remote reply"):
    return {"status_code": 200, "json": {"choices": [{"message": {"content": content}}]}, "text": ""}


# ---------------------------------------------------------------------------
# request_slot remote branch: zero local work
# ---------------------------------------------------------------------------


def test_request_slot_remote_skips_all_local_steps(enabled, monkeypatch, cache_snapshot):
    """The remote branch must call NONE of resolve_context_cap /
    check_vram_fit / auto_download_if_missing / load_llm / unload_llm."""
    called: list[str] = []

    def trip(name):
        def _f(*a, **k):
            called.append(name)
            raise AssertionError(f"{name} must not run on the remote path")
        return _f

    monkeypatch.setattr(cat, "resolve_context_cap", trip("resolve_context_cap"))
    monkeypatch.setattr(cat, "check_vram_fit", trip("check_vram_fit"))
    monkeypatch.setattr(cat, "auto_download_if_missing", trip("auto_download_if_missing"))
    monkeypatch.setattr(loader, "load_llm", trip("load_llm"))
    monkeypatch.setattr(loader, "unload_llm", trip("unload_llm"))

    entry = loader.request_slot("creative", SLOT_A)
    assert called == []
    assert entry["provider"] == "openrouter"
    assert entry["slug"] == "anthropic/claude-3.5-sonnet"
    assert "model" not in entry and "tokenizer" not in entry


def test_request_slot_remote_does_not_evict_resident_local(enabled, monkeypatch, cache_snapshot):
    """C2 no-evict: a resident local model survives a remote request --
    LLM_CACHE is byte-identical before/after (same object identity)."""
    sentinel_entry = {"model": object(), "tokenizer": object(), "context_cap": 8192}
    loader.LLM_CACHE.clear()
    loader.LLM_CACHE.update(
        {"model_id": "mistralai/Mistral-Nemo-Instruct-2407",
         "slot": "technical", "cache_entry": sentinel_entry}
    )
    # Tripwire unload_llm: eviction would call it.
    monkeypatch.setattr(
        loader, "unload_llm",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not evict")),
    )

    loader.request_slot("creative", SLOT_A)

    assert loader.LLM_CACHE["model_id"] == "mistralai/Mistral-Nemo-Instruct-2407"
    assert loader.LLM_CACHE["cache_entry"] is sentinel_entry
    assert loader.LLM_CACHE["slot"] == "technical"


def test_request_slot_remote_raises_clean_when_disabled(monkeypatch):
    """Disabled: the row is not curated, so validate_model_id raises a
    clean UnknownModelError (no remote path reachable, C3)."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OTR_ENABLE_OPENROUTER", raising=False)
    from nodes._otr_model_inputs import UnknownModelError
    with pytest.raises(UnknownModelError):
        loader.request_slot("creative", SLOT_A)


def test_request_slot_local_path_unaffected(monkeypatch, cache_snapshot):
    """A non-remote id still flows through the local steps (regression
    guard: the remote branch must not intercept local ids)."""
    seen: list[str] = []
    monkeypatch.setattr(cat, "validate_model_id", lambda mid, **k: "mistralai/Mistral-Nemo-Instruct-2407")
    monkeypatch.setattr(cat, "resolve_context_cap", lambda mid, **k: seen.append("ctx") or _verdict())
    monkeypatch.setattr(cat, "check_vram_fit", lambda mid, v, **k: seen.append("vram") or _fit())
    monkeypatch.setattr(cat, "auto_download_if_missing", lambda mid, **k: seen.append("dl"))
    monkeypatch.setattr(loader, "load_llm", lambda mid, **k: seen.append("load") or {"model": 1, "tokenizer": 1})
    loader.LLM_CACHE.clear()
    loader.LLM_CACHE.update({"model_id": None, "slot": None, "cache_entry": None})

    loader.request_slot("creative", "mistralai/Mistral-Nemo-Instruct-2407")
    assert seen == ["ctx", "vram", "dl", "load"]


def _verdict():
    from nodes._otr_model_catalog import ContextCapVerdict
    return ContextCapVerdict(tier="PASS", value=8192, source="test")


def _fit():
    from nodes._otr_model_catalog import VRAMFitVerdict
    return VRAMFitVerdict(
        tier="PASS", estimated_gb=10.0, ceiling_gb=14.5,
        reason="test", soak_tested=True,
    )


# ---------------------------------------------------------------------------
# generate-fn factories return a working remote callable
# ---------------------------------------------------------------------------


def _remote_entry(enabled_backend):
    return enabled_backend.load(SLOT_A, _row())


def _row():
    import types
    return types.SimpleNamespace(repo_id=SLOT_A, loader_backend="openrouter_http", context_window=8192)


def test_make_generate_fn_returns_remote_callable(enabled, monkeypatch):
    monkeypatch.setattr(orb, "_post_chat_completion", lambda **k: _ok_post("via make_generate_fn"))
    entry = orb.OpenRouterBackend().load(SLOT_A, _row())
    fn = loader.make_generate_fn(entry)
    out = fn([{"role": "user", "content": "x"}], temperature=0.6, max_new_tokens=64)
    assert out == "via make_generate_fn"


def test_make_polish_generate_fn_returns_remote_callable(enabled, monkeypatch):
    monkeypatch.setattr(orb, "_post_chat_completion", lambda **k: _ok_post("via polish"))
    entry = orb.OpenRouterBackend().load(SLOT_A, _row())
    fn = loader.make_polish_generate_fn(entry)
    out = fn([{"role": "user", "content": "x"}], temperature=0.4, max_new_tokens=64)
    assert out == "via polish"


def test_build_truncating_generate_fn_returns_remote_callable(enabled, monkeypatch):
    from nodes.OTR_LedgerScriptWriter import _build_truncating_generate_fn
    monkeypatch.setattr(orb, "_post_chat_completion", lambda **k: _ok_post("via truncating"))
    entry = orb.OpenRouterBackend().load(SLOT_A, _row())
    fn = _build_truncating_generate_fn(entry, top_p=0.92)
    out = fn([{"role": "user", "content": "x"}], temperature=0.7, max_new_tokens=64, stop=None)
    assert out == "via truncating"


def test_make_generate_fn_local_still_requires_keys():
    """A local entry missing model/tokenizer still errors (the remote
    branch must not swallow the local contract)."""
    with pytest.raises(loader.ModelLoaderError):
        loader.make_generate_fn({"provider": "local"})
