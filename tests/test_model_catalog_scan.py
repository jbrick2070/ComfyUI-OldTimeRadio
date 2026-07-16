"""S30 B1a tests -- offline catalog dataclass + scan + dropdown + validator.

Pure-Python tests against the new nodes/_otr_model_catalog.py + nodes/_otr_model_inputs.py
module surfaces. No HF API calls; no real GPU; no real downloads.

Coverage:
    * CURATED_LLM_MODELS shape (CuratedModel dataclass, derived sets).
    * scan_local_llm_cache walks a fixture hub root.
    * build_dropdown_choices applies the [NOT DOWNLOADED] suffix.
    * validate_model_id strips suffix, structurally rejects unsafe ids,
      admits on each of curated / locally-scanned / arbitrary-org-name
      paths, raises UnknownModelError with actionable recovery hint on miss.

B1a2 will add tests/test_model_catalog_download.py for the network surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes import _otr_model_catalog as catalog
from nodes._otr_model_inputs import UnknownModelError, MissingModelInputError, require_model


@pytest.fixture(autouse=True)
def offline_catalog_env(monkeypatch):
    """Keep the offline catalog tests independent of operator remote-LLM env."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OTR_ENABLE_COMFY_CREDITS", raising=False)


# ---------------------------------------------------------------------------
# Fixture HF cache
# ---------------------------------------------------------------------------


def _make_snapshot(
    hub_root: Path,
    org: str,
    name: str,
    *,
    advertised_context: int | None = 8192,
    architectures: tuple[str, ...] | None = ("LlamaForCausalLM",),
    weights: bool = True,
) -> Path:
    """Create a fake HF cache snapshot directory structure under `hub_root`.

    Layout matches the real HuggingFace hub cache:
        hub_root/models--<org>--<name>/snapshots/<commit_sha>/config.json
    Returns the snapshot path.

    A config.json is written when `advertised_context` or `architectures`
    is non-None. Pass both as None to model a diffusers pipeline repo,
    which ships a model_index.json and carries no root config.json.

    A non-empty ``model.safetensors`` weight blob is written when
    ``weights`` is True (default), so the snapshot counts as materialized
    (on_disk). Pass ``weights=False`` to model a config-only / metadata
    pull that is present on the HF layout but not loadable.
    """
    repo_dir = hub_root / f"models--{org}--{name}"
    snapshot = repo_dir / "snapshots" / "fakecommit0001"
    snapshot.mkdir(parents=True, exist_ok=True)
    if advertised_context is not None or architectures is not None:
        cfg_data: dict = {}
        if advertised_context is not None:
            cfg_data["max_position_embeddings"] = advertised_context
        if architectures is not None:
            cfg_data["architectures"] = list(architectures)
        cfg = snapshot / "config.json"
        cfg.write_text(json.dumps(cfg_data), encoding="utf-8")
    if weights:
        (snapshot / "model.safetensors").write_bytes(b"\0" * 64)
    return snapshot


@pytest.fixture
def empty_hub_root(tmp_path: Path) -> Path:
    """A clean hub root with no models installed."""
    root = tmp_path / "hub"
    root.mkdir()
    return root


@pytest.fixture
def hub_root_with_mistral_nemo(tmp_path: Path) -> Path:
    root = tmp_path / "hub"
    root.mkdir()
    _make_snapshot(root, "mistralai", "Mistral-Nemo-Instruct-2407", advertised_context=131072)
    return root


@pytest.fixture
def hub_root_with_uncurated(tmp_path: Path) -> Path:
    """User has Llama-3-8B locally (not in OTR's curated set)."""
    root = tmp_path / "hub"
    root.mkdir()
    _make_snapshot(root, "meta-llama", "Llama-3-8B-Instruct", advertised_context=8192)
    return root


# ---------------------------------------------------------------------------
# Constants + dataclass shape
# ---------------------------------------------------------------------------


def test_curated_set_is_a_tuple_of_curated_model():
    assert isinstance(catalog.CURATED_LLM_MODELS, tuple)
    assert len(catalog.CURATED_LLM_MODELS) >= 4
    for entry in catalog.CURATED_LLM_MODELS:
        assert isinstance(entry, catalog.CuratedModel)
        assert "/" in entry.repo_id
        assert entry.vram_fit_tier in ("PASS", "WARN", "UNKNOWN", "FAIL")
        # transformers_gptq_int4 stays in the accepted set even though
        # no curated row uses it at present -- the GPTQ backend is
        # parked for a future period model (the talkie row that used
        # it was removed 2026-05-22).
        assert entry.loader_backend in (
            "transformers_safetensors",
            "transformers_multimodal_text_only",
            "transformers_gptq_int4",
        )
        if entry.provider == "local":
            assert entry.approx_safetensors_gb > 0
        else:
            assert entry.approx_safetensors_gb == 0


def test_default_llm_constant_is_in_curated_set():
    repo_ids = {m.repo_id for m in catalog.CURATED_LLM_MODELS}
    assert catalog.DEFAULT_LLM in repo_ids


def test_test_technical_llm_constant_is_in_curated_set():
    repo_ids = {m.repo_id for m in catalog.CURATED_LLM_MODELS}
    assert catalog.TEST_TECHNICAL_LLM in repo_ids


def test_test_oversized_llm_constant_NOT_in_curated_set():
    """TEST_OVERSIZED_LLM is the 70B target used only by VRAM-fit tests."""
    repo_ids = {m.repo_id for m in catalog.CURATED_LLM_MODELS}
    assert catalog.TEST_OVERSIZED_LLM not in repo_ids


def test_gated_curated_models_derived_from_requires_auth():
    expected = {m.repo_id for m in catalog.CURATED_LLM_MODELS if m.requires_auth}
    assert set(catalog.GATED_CURATED_MODELS) == expected
    assert catalog.DEFAULT_LLM in catalog.GATED_CURATED_MODELS  # Mistral is gated


def test_default_llm_is_pass_tier_for_c7_baseline():
    """The C7 audio-baseline model MUST be PASS-tier."""
    for m in catalog.CURATED_LLM_MODELS:
        if m.repo_id == catalog.DEFAULT_LLM:
            assert m.vram_fit_tier == "PASS"
            return
    pytest.fail("DEFAULT_LLM not found in curated set")


# ---------------------------------------------------------------------------
# Local cache scan
# ---------------------------------------------------------------------------


def test_scan_empty_hub_returns_empty_list(empty_hub_root):
    assert catalog.scan_local_llm_cache(hub_root=empty_hub_root) == []


def test_scan_with_mistral_nemo_returns_one_result(hub_root_with_mistral_nemo):
    results = catalog.scan_local_llm_cache(hub_root=hub_root_with_mistral_nemo)
    assert len(results) == 1
    r = results[0]
    assert r.repo_id == catalog.DEFAULT_LLM
    assert r.on_disk is True
    assert r.snapshot_path is not None
    assert r.advertised_context == 131072


def test_scan_skips_non_models_dirs(tmp_path):
    root = tmp_path / "hub"
    root.mkdir()
    (root / "datasets--foo--bar").mkdir()
    (root / "models--mistralai--Mistral-Nemo-Instruct-2407" / "snapshots" / "x").mkdir(parents=True)
    results = catalog.scan_local_llm_cache(hub_root=root)
    assert len(results) == 1
    assert results[0].repo_id == catalog.DEFAULT_LLM


# ---------------------------------------------------------------------------
# Dropdown builder
# ---------------------------------------------------------------------------


def test_dropdown_empty_cache_marks_all_curated_not_downloaded(empty_hub_root):
    entries = catalog.build_dropdown_choices(hub_root=empty_hub_root)
    active_ids = set(catalog._by_repo_id())
    assert {e.repo_id for e in entries} == active_ids
    by_id = catalog._by_repo_id()
    for e in entries:
        provider = by_id[e.repo_id].provider
        if provider == "gguf_native":
            assert isinstance(e.on_disk, bool)
            assert e.label == e.repo_id + catalog.LOCAL_GGUF_SUFFIX
            assert catalog.NOT_DOWNLOADED_SUFFIX not in e.label
        elif provider != "local":
            assert e.on_disk is True
            assert catalog.NOT_DOWNLOADED_SUFFIX not in e.label
        else:
            assert e.on_disk is False
            # Off-disk local row: [NOT DOWNLOADED] ONLY. The [LOCAL HF]
            # badge marks a PRESENT row; emitting both was the
            # contradictory "[LOCAL HF] [NOT DOWNLOADED]" double suffix.
            assert catalog.LOCAL_HF_SUFFIX not in e.label
            assert e.label == e.repo_id + catalog.NOT_DOWNLOADED_SUFFIX


def test_dropdown_with_mistral_nemo_marks_only_that_on_disk(hub_root_with_mistral_nemo):
    entries = catalog.build_dropdown_choices(hub_root=hub_root_with_mistral_nemo)
    by_id = catalog._by_repo_id()
    for e in entries:
        if e.repo_id == catalog.DEFAULT_LLM:
            assert e.on_disk is True
            assert e.label == catalog.DEFAULT_LLM
            assert catalog.NOT_DOWNLOADED_SUFFIX not in e.label
        elif by_id.get(e.repo_id) and by_id[e.repo_id].provider == "gguf_native":
            assert e.label == e.repo_id + catalog.LOCAL_GGUF_SUFFIX
            assert catalog.NOT_DOWNLOADED_SUFFIX not in e.label
        elif by_id.get(e.repo_id) and by_id[e.repo_id].provider != "local":
            assert e.on_disk is True
            assert catalog.NOT_DOWNLOADED_SUFFIX not in e.label
        else:
            assert e.on_disk is False
            # Off-disk local row: [NOT DOWNLOADED] ONLY. The [LOCAL HF]
            # badge marks a PRESENT row; emitting both was the
            # contradictory "[LOCAL HF] [NOT DOWNLOADED]" double suffix.
            assert catalog.LOCAL_HF_SUFFIX not in e.label
            assert e.label == e.repo_id + catalog.NOT_DOWNLOADED_SUFFIX


def test_dropdown_appends_uncurated_locally_scanned_at_end(hub_root_with_uncurated):
    entries = catalog.build_dropdown_choices(hub_root=hub_root_with_uncurated)
    uncurated_entry = next(
        (e for e in entries if e.repo_id == "meta-llama/Llama-3-8B-Instruct"), None
    )
    assert uncurated_entry is not None
    assert uncurated_entry.on_disk is True
    assert uncurated_entry.curated is False
    assert uncurated_entry.label == "meta-llama/Llama-3-8B-Instruct"


# ---------------------------------------------------------------------------
# Dropdown builder -- non-LLM cache filter (BUG-LOCAL-257)
# ---------------------------------------------------------------------------


@pytest.fixture
def hub_root_mixed_model_types(tmp_path: Path) -> Path:
    """The real-world HF_HOME layout: a non-curated writer LLM sharing
    the hub cache with diffusion pipelines and a vision model. OTR's
    HF_HOME consolidates every model type into one hub/, so the cache
    walk sees FLUX / LTX-Video / Depth-Anything next to the LLMs."""
    root = tmp_path / "hub"
    root.mkdir()
    # Non-curated causal LM -- belongs in the writer dropdown.
    _make_snapshot(
        root, "meta-llama", "Llama-3-8B-Instruct",
        architectures=("LlamaForCausalLM",),
    )
    # Diffusion pipelines -- model_index.json, no root config.json.
    flux = _make_snapshot(
        root, "black-forest-labs", "FLUX.1-dev",
        advertised_context=None, architectures=None,
    )
    (flux / "model_index.json").write_text("{}", encoding="utf-8")
    ltx = _make_snapshot(
        root, "diffusers", "LTX-Video-0.9.0",
        advertised_context=None, architectures=None,
    )
    (ltx / "model_index.json").write_text("{}", encoding="utf-8")
    # Vision model -- root config.json present, architecture not *ForCausalLM.
    _make_snapshot(
        root, "depth-anything", "Depth-Anything-V2-Large-hf",
        advertised_context=None,
        architectures=("DepthAnythingForDepthEstimation",),
    )
    return root


def test_dropdown_excludes_diffusion_pipelines(hub_root_mixed_model_types):
    """FLUX and LTX-Video (diffusers repos: model_index.json, no root
    config.json) must not appear in the writer model dropdown."""
    entries = catalog.build_dropdown_choices(hub_root=hub_root_mixed_model_types)
    repo_ids = {e.repo_id for e in entries}
    assert "black-forest-labs/FLUX.1-dev" not in repo_ids
    assert "diffusers/LTX-Video-0.9.0" not in repo_ids


def test_dropdown_excludes_vision_model(hub_root_mixed_model_types):
    """A transformers vision model (config.json present, architecture
    not *ForCausalLM) must not appear in the writer model dropdown."""
    entries = catalog.build_dropdown_choices(hub_root=hub_root_mixed_model_types)
    repo_ids = {e.repo_id for e in entries}
    assert "depth-anything/Depth-Anything-V2-Large-hf" not in repo_ids


def test_dropdown_keeps_uncurated_causal_lm_amid_mixed_cache(hub_root_mixed_model_types):
    """A non-curated causal LM sharing the cache with diffusion / vision
    checkpoints still appears -- the filter narrows by model type, not
    by curation."""
    entries = catalog.build_dropdown_choices(hub_root=hub_root_mixed_model_types)
    entry = next(
        (e for e in entries if e.repo_id == "meta-llama/Llama-3-8B-Instruct"), None
    )
    assert entry is not None
    assert entry.curated is False
    assert entry.on_disk is True


def test_dropdown_curated_set_exempt_from_cache_filter(hub_root_mixed_model_types):
    """The curated rows are the explicit writer set -- always present,
    never subject to the causal-LM cache filter."""
    entries = catalog.build_dropdown_choices(hub_root=hub_root_mixed_model_types)
    repo_ids = {e.repo_id for e in entries}
    curated_ids = {m.repo_id for m in catalog.CURATED_LLM_MODELS}
    assert curated_ids <= repo_ids


def test_snapshot_is_causal_lm_true_for_causal_config(tmp_path):
    snap = _make_snapshot(
        tmp_path, "x", "writer-lm", architectures=("MistralForCausalLM",)
    )
    assert catalog._snapshot_is_causal_lm(str(snap)) is True


def test_snapshot_is_causal_lm_false_for_missing_config(tmp_path):
    """Diffusers pipeline case: no root config.json."""
    snap = _make_snapshot(
        tmp_path, "x", "diffusion-repo",
        advertised_context=None, architectures=None,
    )
    assert catalog._snapshot_is_causal_lm(str(snap)) is False


def test_snapshot_is_causal_lm_false_for_non_causal_architecture(tmp_path):
    """Vision model case: config.json present, architecture not causal."""
    snap = _make_snapshot(
        tmp_path, "x", "vision-model",
        advertised_context=None, architectures=("CLIPModel",),
    )
    assert catalog._snapshot_is_causal_lm(str(snap)) is False


def test_snapshot_is_causal_lm_false_for_none_path():
    assert catalog._snapshot_is_causal_lm(None) is False


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_validator_strips_not_downloaded_suffix(empty_hub_root, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    labelled = catalog.DEFAULT_LLM + catalog.NOT_DOWNLOADED_SUFFIX
    out = catalog.validate_model_id(labelled, hub_root=empty_hub_root)
    assert out == catalog.DEFAULT_LLM


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (
            catalog.TEST_TECHNICAL_LLM + catalog.LOCAL_HF_SUFFIX,
            catalog.TEST_TECHNICAL_LLM,
        ),
        (
            catalog.TEST_TECHNICAL_LLM
            + catalog.LOCAL_HF_SUFFIX
            + catalog.NOT_DOWNLOADED_SUFFIX,
            catalog.TEST_TECHNICAL_LLM,
        ),
        (
            "unsloth/gemma-4-12b-it-GGUF" + catalog.LOCAL_GGUF_SUFFIX,
            "unsloth/gemma-4-12b-it-GGUF",
        ),
    ],
)
def test_validator_strips_local_display_suffixes(empty_hub_root, monkeypatch, label, expected):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    out = catalog.validate_model_id(label, hub_root=empty_hub_root)
    assert out == expected


def test_validator_admits_curated(empty_hub_root):
    out = catalog.validate_model_id(catalog.DEFAULT_LLM, hub_root=empty_hub_root)
    assert out == catalog.DEFAULT_LLM


def test_validator_admits_locally_scanned_uncurated(hub_root_with_uncurated, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "0")
    out = catalog.validate_model_id(
        "meta-llama/Llama-3-8B-Instruct", hub_root=hub_root_with_uncurated
    )
    assert out == "meta-llama/Llama-3-8B-Instruct"


def test_validator_admits_arbitrary_org_name_when_auto_download_enabled(empty_hub_root, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    out = catalog.validate_model_id("some-user/some-model", hub_root=empty_hub_root)
    assert out == "some-user/some-model"


def test_validator_rejects_arbitrary_when_auto_download_disabled(empty_hub_root, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "0")
    monkeypatch.delenv("OTR_MODEL_CATALOG_ALLOW_REMOTE", raising=False)
    with pytest.raises(UnknownModelError) as exc:
        catalog.validate_model_id("some-user/some-model", hub_root=empty_hub_root)
    assert "some-user/some-model" in str(exc.value)
    assert "huggingface-cli" in str(exc.value)


def test_validator_rejects_path_traversal(empty_hub_root, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    with pytest.raises(UnknownModelError) as exc:
        catalog.validate_model_id("../etc/passwd", hub_root=empty_hub_root)
    assert "traversal" in str(exc.value).lower() or ".." in str(exc.value)


def test_validator_rejects_absolute_path(empty_hub_root, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    with pytest.raises(UnknownModelError) as exc:
        catalog.validate_model_id("/etc/passwd", hub_root=empty_hub_root)
    assert "absolute" in str(exc.value).lower() or "/" in str(exc.value)


def test_validator_rejects_windows_drive_letter(empty_hub_root, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    with pytest.raises(UnknownModelError):
        catalog.validate_model_id("C:/Windows/foo", hub_root=empty_hub_root)


def test_validator_rejects_backslash(empty_hub_root, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    with pytest.raises(UnknownModelError):
        catalog.validate_model_id(r"some\path\here", hub_root=empty_hub_root)


def test_validator_rejects_gguf_extension(empty_hub_root, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    with pytest.raises(UnknownModelError) as exc:
        catalog.validate_model_id("foo/bar.gguf", hub_root=empty_hub_root)
    assert "gguf" in str(exc.value).lower() or "transformers" in str(exc.value)


def test_validator_recovery_hint_lists_top_installed(hub_root_with_mistral_nemo, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "0")
    monkeypatch.delenv("OTR_MODEL_CATALOG_ALLOW_REMOTE", raising=False)
    with pytest.raises(UnknownModelError) as exc:
        catalog.validate_model_id("totally/unknown-id-xyz", hub_root=hub_root_with_mistral_nemo)
    # Top installed alternative (Mistral-Nemo) should be in the message.
    assert catalog.DEFAULT_LLM in str(exc.value)


# ---------------------------------------------------------------------------
# require_model helper
# ---------------------------------------------------------------------------


def test_require_model_passes_non_empty_string():
    assert require_model(catalog.DEFAULT_LLM, slot="creative") == catalog.DEFAULT_LLM


def test_require_model_raises_on_empty():
    with pytest.raises(MissingModelInputError) as exc:
        require_model("", slot="technical")
    assert "technical" in str(exc.value)


def test_require_model_raises_on_none():
    with pytest.raises(MissingModelInputError):
        require_model(None, slot="creative")


def test_require_model_raises_on_whitespace_only():
    with pytest.raises(MissingModelInputError):
        require_model("   ", slot="creative")


# ---------------------------------------------------------------------------
# B1b: resolve_context_cap (dynamic context-cap with HARD_VRAM clamp)
# ---------------------------------------------------------------------------


def test_hard_vram_context_limit_default_is_8192(monkeypatch):
    """Test seam: we re-read the env var each call. Module-level constant
    is fixed at import time so we re-import via the helper."""
    monkeypatch.delenv("OTR_HARD_VRAM_CONTEXT_LIMIT", raising=False)
    # The module-level constant captured the value at import; re-test
    # via the resolver function on a model that triggers the limit.
    assert catalog._hard_vram_context_limit() == 8192


def test_hard_vram_context_limit_env_override(monkeypatch):
    monkeypatch.setenv("OTR_HARD_VRAM_CONTEXT_LIMIT", "16384")
    assert catalog._hard_vram_context_limit() == 16384


def test_hard_vram_context_limit_garbage_env_falls_back(monkeypatch):
    monkeypatch.setenv("OTR_HARD_VRAM_CONTEXT_LIMIT", "not-a-number")
    assert catalog._hard_vram_context_limit() == 8192


def test_resolve_context_cap_pass_for_curated_override(empty_hub_root):
    """C7 audio-baseline guard: Mistral-Nemo must return PASS @ 8192."""
    v = catalog.resolve_context_cap(catalog.DEFAULT_LLM, hub_root=empty_hub_root)
    assert v.tier == "PASS"
    assert v.value == 8192
    assert "curated-override" in v.source


def test_resolve_context_cap_warn_for_uncurated_with_config(hub_root_with_uncurated):
    """Uncurated model with parseable config.json returns WARN."""
    v = catalog.resolve_context_cap(
        "meta-llama/Llama-3-8B-Instruct", hub_root=hub_root_with_uncurated
    )
    assert v.tier == "WARN"
    # advertised 8192 is below the limit; value matches.
    assert v.value == 8192
    assert "config.json" in v.source


def test_resolve_context_cap_clamps_warn_to_hard_limit(tmp_path):
    """Uncurated model advertising 128k must be clamped to HARD_VRAM
    limit (8192 on the 16 GB target)."""
    root = tmp_path / "hub"
    root.mkdir()
    _make_snapshot(root, "huge", "Context-128k-Model", advertised_context=131072)
    v = catalog.resolve_context_cap("huge/Context-128k-Model", hub_root=root)
    assert v.tier == "WARN"
    assert v.value == 8192
    assert "131072" in v.source


def test_resolve_context_cap_unknown_for_unresolved_model(empty_hub_root):
    """Neither curated nor locally-scanned -> UNKNOWN @ limit."""
    v = catalog.resolve_context_cap(
        "totally/uncurated-no-snapshot", hub_root=empty_hub_root
    )
    assert v.tier == "UNKNOWN"
    assert v.value == catalog.HARD_VRAM_CONTEXT_LIMIT


def test_resolve_context_cap_never_raises_on_arbitrary_input(empty_hub_root):
    """The contract is: return a verdict, never raise. B1c's request_slot
    makes the combined fit/cap escalation decision; resolve_context_cap
    is policy-free."""
    for weird in ["", " ", "foo", "no slash here", "/leading-slash"]:
        v = catalog.resolve_context_cap(weird, hub_root=empty_hub_root)
        assert v.tier in ("PASS", "WARN", "UNKNOWN")
        assert v.value >= 512


# ---------------------------------------------------------------------------
# Regression: HF cache-root resolution honors HF_HUB_CACHE, weight-blob
# completeness gates on_disk, and the dropdown state suffix is EXCLUSIVE
# (no contradictory "[LOCAL HF] [NOT DOWNLOADED]" double suffix).
#
# Root cause (2026-07-16): _hf_hub_root() read only HF_HOME + the legacy
# HUGGINGFACE_HUB_CACHE, so on a box that sets the modern HF_HUB_CACHE the
# scanner fell through to the stale ~/.cache default and reported real,
# on-disk Gemma weights as NOT DOWNLOADED. Separately, e412e84b appended
# [NOT DOWNLOADED] on top of the [LOCAL HF] badge for a Google Gemma row
# whose scan missed, emitting the contradictory double suffix.
# ---------------------------------------------------------------------------


def test_hf_hub_root_honors_hf_hub_cache(tmp_path, monkeypatch):
    """The modern HF_HUB_CACHE var (what huggingface_hub itself honors)
    must resolve as the cache root. Regression guard for the mislabel bug."""
    cache = tmp_path / "models_cache"
    cache.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    assert catalog._hf_hub_root() == cache


def test_hf_hub_root_precedence_matches_huggingface_hub(tmp_path, monkeypatch):
    """HF_HUB_CACHE wins over HF_HOME/hub, mirroring huggingface_hub so the
    scanner and the model loader can never disagree about the cache root."""
    cache = tmp_path / "wins"
    cache.mkdir()
    home = tmp_path / "home"
    (home / "hub").mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HOME", str(home))
    assert catalog._hf_hub_root() == cache


def test_hf_hub_root_falls_back_to_hf_home_hub(tmp_path, monkeypatch):
    """With neither HF_HUB_CACHE nor HUGGINGFACE_HUB_CACHE set, HF_HOME/hub
    is used (OTR's shared-root convention)."""
    home = tmp_path / "home"
    (home / "hub").mkdir(parents=True)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(home))
    assert catalog._hf_hub_root() == home / "hub"


def test_hf_hub_root_legacy_huggingface_hub_cache_still_read(tmp_path, monkeypatch):
    """The legacy HUGGINGFACE_HUB_CACHE alias remains honored."""
    cache = tmp_path / "legacy"
    cache.mkdir()
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(cache))
    monkeypatch.delenv("HF_HOME", raising=False)
    assert catalog._hf_hub_root() == cache


def test_snapshot_has_weights_true_with_materialized_blob(tmp_path):
    snap = _make_snapshot(tmp_path, "x", "writer-lm", weights=True)
    assert catalog._snapshot_has_weights(snap) is True


def test_snapshot_has_weights_false_for_config_only(tmp_path):
    snap = _make_snapshot(tmp_path, "x", "writer-lm", weights=False)
    assert catalog._snapshot_has_weights(snap) is False


def test_scan_config_only_snapshot_reports_not_on_disk(tmp_path):
    """A snapshot dir carrying config.json but no weight blob is on the HF
    layout yet NOT usable -> on_disk False (distinguish config-present from
    fully-materialized-and-usable)."""
    root = tmp_path / "hub"
    root.mkdir()
    _make_snapshot(root, "google", "gemma-2-2b-it", weights=False)
    results = {r.repo_id: r for r in catalog.scan_local_llm_cache(hub_root=root)}
    assert "google/gemma-2-2b-it" in results
    assert results["google/gemma-2-2b-it"].on_disk is False


def test_scan_materialized_snapshot_reports_on_disk(tmp_path):
    root = tmp_path / "hub"
    root.mkdir()
    _make_snapshot(root, "google", "gemma-2-2b-it", weights=True)
    results = {r.repo_id: r for r in catalog.scan_local_llm_cache(hub_root=root)}
    assert results["google/gemma-2-2b-it"].on_disk is True


# Every curated LOCAL row, asserted one-by-one against a known disk state.
# `expected_badge` is the sole state suffix the label may carry.
_LOCAL_ROW_CASES = [
    ("google/gemma-4-E2B-it", True, catalog.LOCAL_HF_SUFFIX),
    ("google/gemma-4-E2B-it", False, catalog.NOT_DOWNLOADED_SUFFIX),
    ("google/gemma-4-E4B-it", True, catalog.LOCAL_HF_SUFFIX),
    ("google/gemma-4-E4B-it", False, catalog.NOT_DOWNLOADED_SUFFIX),
    ("google/gemma-2-2b-it", True, catalog.LOCAL_HF_SUFFIX),
    ("google/gemma-2-2b-it", False, catalog.NOT_DOWNLOADED_SUFFIX),
    ("mistralai/Mistral-Nemo-Instruct-2407", True, ""),  # non-gemma: bare id
    ("mistralai/Mistral-Nemo-Instruct-2407", False, catalog.NOT_DOWNLOADED_SUFFIX),
    ("Qwen/Qwen2.5-14B-Instruct", True, ""),
    ("Qwen/Qwen2.5-14B-Instruct", False, catalog.NOT_DOWNLOADED_SUFFIX),
]


@pytest.mark.parametrize(("repo_id", "present", "expected_badge"), _LOCAL_ROW_CASES)
def test_curated_local_row_label_matches_disk(tmp_path, repo_id, present, expected_badge):
    """Per-item contract: a curated local row's label reflects exactly its
    disk state, with EXACTLY ONE state suffix -- never both, never the
    contradictory [LOCAL HF] [NOT DOWNLOADED]."""
    root = tmp_path / "hub"
    root.mkdir()
    org, name = repo_id.split("/", 1)
    _make_snapshot(root, org, name, weights=present)

    entry = next(
        e for e in catalog.build_dropdown_choices(hub_root=root)
        if e.repo_id == repo_id
    )
    assert entry.on_disk is present
    assert entry.label == repo_id + expected_badge
    # Exclusive suffix invariant: the two state badges are mutually exclusive.
    assert not (
        catalog.LOCAL_HF_SUFFIX in entry.label
        and catalog.NOT_DOWNLOADED_SUFFIX in entry.label
    )
    # The stored label round-trips back to the canonical repo id.
    assert catalog._strip_label_suffix(entry.label) == repo_id


@pytest.fixture
def hub_root_operator_gemma_mix(tmp_path: Path) -> Path:
    """Mirror the operator's real cache: E4B materialized (weights present),
    E2B + gemma-2-2b config-only (metadata pull, no weight blob)."""
    root = tmp_path / "opmix_hub"  # distinct name -> safe to co-request empty_hub_root
    root.mkdir()
    _make_snapshot(root, "google", "gemma-4-E4B-it", weights=True)
    _make_snapshot(root, "google", "gemma-4-E2B-it", weights=False)
    _make_snapshot(root, "google", "gemma-2-2b-it", weights=False)
    return root


def test_no_dropdown_label_carries_double_state_suffix(
    hub_root_operator_gemma_mix, empty_hub_root
):
    """No label may carry BOTH [LOCAL HF] and [NOT DOWNLOADED] in any cache
    state -- the exact contradiction the operator reported."""
    for hub in (hub_root_operator_gemma_mix, empty_hub_root):
        for e in catalog.build_dropdown_choices(hub_root=hub):
            assert not (
                catalog.LOCAL_HF_SUFFIX in e.label
                and catalog.NOT_DOWNLOADED_SUFFIX in e.label
            ), f"double state suffix in {e.label!r}"


def test_operator_gemma_mix_exact_labels(hub_root_operator_gemma_mix):
    """The reported scenario, resolved: materialized E4B -> [LOCAL HF];
    config-only E2B + gemma-2-2b -> [NOT DOWNLOADED] (single suffix each)."""
    by_repo = {
        e.repo_id: e
        for e in catalog.build_dropdown_choices(hub_root=hub_root_operator_gemma_mix)
    }
    assert by_repo["google/gemma-4-E4B-it"].label == (
        "google/gemma-4-E4B-it" + catalog.LOCAL_HF_SUFFIX
    )
    assert by_repo["google/gemma-4-E2B-it"].label == (
        "google/gemma-4-E2B-it" + catalog.NOT_DOWNLOADED_SUFFIX
    )
    assert by_repo["google/gemma-2-2b-it"].label == (
        "google/gemma-2-2b-it" + catalog.NOT_DOWNLOADED_SUFFIX
    )
