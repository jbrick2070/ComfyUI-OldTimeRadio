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

    # S31 B4: the 4 legacy symbols `_load_llm` / `_unload_llm` /
    # `_LLM_CACHE` / `_generate_with_llm` are DELETED from
    # `story_orchestrator`. The legacy `_so._load_llm` patch and the
    # `_so._LLM_CACHE` reset block that used to live here are gone.
    # The canonical seam is the loader-side `load_llm` stub installed
    # below.

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
    """check_vram_fit reports WARN for a curated WARN-tier row.

    2026-08-25: this used to read the live Qwen2.5-14B row, which was the
    last WARN-tier entry in the catalog until it was pruned. The WARN
    MECHANISM is still worth covering, so the row is now synthetic --
    otherwise removing the last WARN model would have silently deleted the
    only test of what WARN does, which is the vacuity class this repo has
    already been bitten by twice.
    """
    import nodes._otr_model_catalog as c

    fake_warn = catalog.CuratedModel(
        repo_id="test-only/warn-tier-row",
        requires_auth=False,
        loader_backend="transformers_safetensors",
        vram_fit_tier="WARN",
        approx_safetensors_gb=28.0,
        notes="test-only WARN row",
    )
    original_curated = c.CURATED_LLM_MODELS
    c.CURATED_LLM_MODELS = original_curated + (fake_warn,)
    try:
        v = catalog.check_vram_fit("test-only/warn-tier-row", 8192)
        assert v.tier == "WARN"
        assert v.soak_tested is False
    finally:
        c.CURATED_LLM_MODELS = original_curated


def test_every_curated_local_row_is_pass_tier():
    """Operator ruling 2026-08-25: only easy-to-load LLMs ship.

    "If it doesn't fit nicely or requires Ollama rip it from the dropdown
    and blast radius." A WARN row is by definition one that is NOT
    soak-tested to fit -- Qwen2.5-14B's own note conceded it needed
    "quantization or offload to fit 16 GB" and was for "users with bigger
    rigs". Shipping one in the dropdown promises a load that may not
    happen on the 16 GB target card.

    This is the gate that keeps the ruling from decaying: a new WARN (or
    UNKNOWN/FAIL) row fails HERE, by name, instead of failing in front of
    an operator mid-render.
    """
    offenders = [
        f"{row.repo_id} (tier={row.vram_fit_tier})"
        for row in catalog.CURATED_LLM_MODELS
        if row.vram_fit_tier != "PASS"
    ]
    assert not offenders, (
        "curated rows must be PASS-tier (operator 2026-08-25, "
        "'I only want easy to load LLMs'); offenders: "
        + ", ".join(offenders)
    )


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


def test_unload_if_local_resident_skips_empty_and_remote(monkeypatch):
    calls = []
    monkeypatch.setattr(loader, "unload_llm", lambda: calls.append(1))

    loader.LLM_CACHE.update({"model_id": None, "slot": None, "cache_entry": None})
    assert loader.unload_llm_if_local_resident() is False
    assert calls == []

    for provider in ("openrouter", "comfy_credits"):
        loader.LLM_CACHE.update({
            "model_id": f"{provider}:slot-a",
            "slot": "creative",
            "cache_entry": {"provider": provider},
        })
        assert loader.unload_llm_if_local_resident() is False
    assert calls == []


def test_unload_if_local_resident_calls_canonical_unload(monkeypatch):
    calls = []
    monkeypatch.setattr(loader, "unload_llm", lambda: calls.append(1))
    loader.LLM_CACHE.update({
        "model_id": catalog.DEFAULT_LLM,
        "slot": "creative",
        "cache_entry": {"model": _FakeModel(), "tokenizer": _FakeTokenizer()},
    })
    assert loader.unload_llm_if_local_resident() is True
    assert calls == [1]


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


# ---------------------------------------------------------------------------
# S31 B4: unload_llm simplified + invalidate_cache_no_gpu_teardown added
# ---------------------------------------------------------------------------


def test_unload_llm_no_orchestrator_fallback_block():
    """S31 B4 simplification: `_otr_model_loader.unload_llm` MUST NOT
    import or reference `story_orchestrator._LLM_CACHE` (which was
    deleted at B4). The legacy-fallback teardown block that touched
    `_so._LLM_CACHE` is removed."""
    import ast
    from pathlib import Path

    loader_src = Path(loader.__file__).read_text(encoding="utf-8")
    tree = ast.parse(loader_src)
    unload_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "unload_llm":
            unload_fn = node
            break
    assert unload_fn is not None, "unload_llm not found in loader"

    offenders: list[str] = []
    for sub in ast.walk(unload_fn):
        # Import of story_orchestrator inside unload_llm? Forbidden.
        if isinstance(sub, ast.ImportFrom):
            mod = sub.module or ""
            if mod.endswith("story_orchestrator"):
                offenders.append(
                    f"line {sub.lineno}: from {mod} import ..."
                )
        if isinstance(sub, ast.Import):
            # `for imp in sub.names`: ast.alias is the AST node type
            # for each imported name. Local variable named `imp` (not
            # `alias`) to keep the forbidden-pattern sweep's
            # `\balias\b` marker clean (S28 _RENAME_ALIASES lock).
            for imp in sub.names:
                if imp.name.endswith("story_orchestrator"):
                    offenders.append(
                        f"line {sub.lineno}: import {imp.name}"
                    )
        # Reference to `_LLM_CACHE` (the legacy orchestrator dict)?
        # The modern dict is `LLM_CACHE` without underscore prefix.
        if isinstance(sub, (ast.Attribute, ast.Name)):
            name = (
                sub.attr if isinstance(sub, ast.Attribute) else sub.id
            )
            if name == "_LLM_CACHE":
                offenders.append(
                    f"line {sub.lineno}: reference to _LLM_CACHE"
                )
    assert not offenders, (
        "S31 B4: `_otr_model_loader.unload_llm` must NOT import "
        "story_orchestrator and must NOT reference `_LLM_CACHE` "
        "(the legacy orchestrator dict is deleted). Offenders:\n  "
        + "\n  ".join(offenders)
    )


def test_invalidate_cache_no_gpu_teardown_clears_dict():
    """S31 B4: new lifecycle helper clears LLM_CACHE references
    in-place (id stable, the 3 canonical keys all set back to None)."""
    original_id = id(loader.LLM_CACHE)
    # Pre-populate.
    loader.LLM_CACHE["model_id"] = "some/model"
    loader.LLM_CACHE["slot"] = "creative"
    loader.LLM_CACHE["cache_entry"] = {"model": object(), "tokenizer": object()}

    loader.invalidate_cache_no_gpu_teardown()

    assert id(loader.LLM_CACHE) == original_id, (
        "B1d invariant: invalidate_cache_no_gpu_teardown must clear "
        "in-place (id stable). Rebinding to a fresh dict breaks any "
        "`from _otr_model_loader import LLM_CACHE` consumer."
    )
    assert loader.LLM_CACHE == {
        "model_id": None,
        "slot": None,
        "cache_entry": None,
    }, (
        "post-invalidate dict shape drifted; expected exactly 3 keys "
        "(model_id / slot / cache_entry) all set to None."
    )


def test_invalidate_cache_no_gpu_teardown_no_gpu_calls():
    """S31 B4: the WHOLE POINT of this helper is that it must NOT
    touch the GPU. AST-scan its body for forbidden GPU calls (the
    timeout-recovery path uses this helper specifically to avoid the
    `cudaErrorIllegalAddress` race when an orphan worker thread is
    still executing CUDA kernels on the cached model)."""
    import ast
    from pathlib import Path

    loader_src = Path(loader.__file__).read_text(encoding="utf-8")
    tree = ast.parse(loader_src)
    helper_fn = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "invalidate_cache_no_gpu_teardown"
        ):
            helper_fn = node
            break
    assert helper_fn is not None, (
        "invalidate_cache_no_gpu_teardown not found in loader; "
        "S31 B4 add-helper step did not land."
    )

    forbidden_calls: list[str] = []
    forbidden_patterns = {
        "to": "model.to(cpu)",                  # weight teardown
        "cpu": "model.cpu()",                   # alias for to(cpu)
        "empty_cache": "torch.cuda.empty_cache()",
        "synchronize": "torch.cuda.synchronize()",
        "ipc_collect": "torch.cuda.ipc_collect()",
        "collect": "gc.collect()",
    }
    for sub in ast.walk(helper_fn):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        attr = None
        if isinstance(fn, ast.Attribute):
            attr = fn.attr
        elif isinstance(fn, ast.Name):
            attr = fn.id
        if attr in forbidden_patterns:
            forbidden_calls.append(
                f"line {sub.lineno}: {forbidden_patterns[attr]}"
            )
    assert not forbidden_calls, (
        "S31 B4 contract: `invalidate_cache_no_gpu_teardown` must "
        "NOT touch the GPU. Forbidden call sites found:\n  "
        + "\n  ".join(forbidden_calls)
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
    """Slot 1 != Slot 2 must call its self-teardown exactly once between
    loads. request_slot's own self-triggered transitions route through
    _self_unload (an ownership-checked claim, PBUG-20260825-04 r3) rather
    than the public, unconditional unload_llm() -- monkeypatch the real
    call target."""
    unload_calls: list[int] = []
    original_self_unload = loader._self_unload

    def counting_self_unload(my_epoch, *, slot):
        unload_calls.append(1)
        return original_self_unload(my_epoch, slot=slot)

    monkeypatch.setattr(loader, "_self_unload", counting_self_unload)

    loader.request_slot("creative", catalog.DEFAULT_LLM)
    assert len(unload_calls) == 0  # first load has no prior resident
    loader.request_slot("technical", catalog.TEST_TECHNICAL_LLM)
    assert len(unload_calls) == 1  # transition unloaded once
    assert loader.LLM_CACHE["model_id"] == catalog.TEST_TECHNICAL_LLM


def test_load_failure_cleanup_does_not_tear_down_a_foreign_live_entry(monkeypatch):
    """r4 kibitz finding (Cursor): request_slot's load-failure except block
    used to call the raw, UNCONDITIONAL unload_llm() to drop orphan VRAM
    before a retry (Sprint H iter 3). Because it is unconditional, a
    slow/abandoned call whose OWN load later fails could tear down --
    model.to("cpu"), epoch bump -- a completely DIFFERENT, legitimate
    caller's model that got published in the meantime, purely because this
    call's own (unrelated) load happened to fail. Reproduced here
    single-threaded: while the failing call's load_llm() is "running", a
    third party invalidates and publishes its own fresh entry; the failing
    call's except-block cleanup must leave that foreign entry untouched."""
    # Establish a resident model so the failing call's own Step 8 has
    # something to (successfully) self-unload before its load_llm() runs.
    loader.request_slot("technical", catalog.TEST_TECHNICAL_LLM)

    foreign_entry = {"marker": "foreign-live-model"}

    def _failing_load_llm(model_id, **kwargs):
        # Simulates a THIRD PARTY publishing its own fresh, unrelated
        # entry WHILE this call's load is in flight -- e.g. a genuinely
        # concurrent request_slot call on another thread, or (as in the
        # live PBUG-20260825-04 incident) a prior orphan's own timeout
        # handler invalidating and a later prompt's request_slot
        # succeeding before this call's failing load unwinds.
        loader.invalidate_cache_no_gpu_teardown()
        loader._publish_cache_entry_if_current(
            loader._current_cache_epoch(), {
                "model_id": "test/foreign-model",
                "slot": "technical",
                "cache_entry": foreign_entry,
                "policy_key": "foreign-policy",
            },
        )
        raise RuntimeError("simulated load failure")

    monkeypatch.setattr(loader, "load_llm", _failing_load_llm)

    with pytest.raises(RuntimeError, match="simulated load failure"):
        loader.request_slot("creative", catalog.DEFAULT_LLM)

    assert loader.LLM_CACHE.get("cache_entry") is foreign_entry, (
        "a different, legitimate caller's freshly-published model must "
        "survive an unrelated, later-failing call's cleanup -- tearing it "
        "down here is exactly the r4 regression"
    )
    assert loader.LLM_CACHE.get("model_id") == "test/foreign-model"


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
