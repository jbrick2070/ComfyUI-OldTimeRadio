"""Offline HF cache contracts for the Gemma4Unified writer lane."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from nodes import _otr_hf_env as hf_env
from nodes import _otr_model_catalog as catalog
from nodes import _otr_model_loader as loader


MODEL_ID = "google/gemma-4-12b-it"


def _reset_hf_cache() -> None:
    hf_env._CACHE.update({"hf_home": None, "resolved": False})


@pytest.fixture(autouse=True)
def _isolate_hf_cache_state():
    """Keep the module-level HF resolver cache from leaking across tests."""
    previous = dict(hf_env._CACHE)
    previous_env = {
        name: os.environ.get(name)
        for name in ("HF_HOME", "HF_HUB_CACHE")
    }
    _reset_hf_cache()
    try:
        yield
    finally:
        hf_env._CACHE.clear()
        hf_env._CACHE.update(previous)
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _snapshot(home: Path, revision: str) -> Path:
    path = (
        home / "hub" / "models--google--gemma-4-12b-it"
        / "snapshots" / revision
    )
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


def test_ensure_hf_home_exports_the_hub_subdirectory(monkeypatch, tmp_path):
    _reset_hf_cache()
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    assert hf_env.ensure_hf_home() == str(tmp_path)
    assert os.environ["HF_HUB_CACHE"] == str(tmp_path / "hub")


def test_snapshot_resolver_prefers_weights_and_composes_newer_metadata(
    tmp_path,
):
    weighted = _snapshot(tmp_path, "weighted")
    metadata = _snapshot(tmp_path, "metadata")
    (weighted / "model.safetensors").write_bytes(b"weights")
    (metadata / "chat_template.jinja").write_text(
        "{{ messages }}", encoding="utf-8",
    )
    os.utime(weighted, (1, 1))
    os.utime(metadata, (2, 2))

    assert hf_env.resolve_snapshot_dir(MODEL_ID, str(tmp_path)) == str(weighted)
    assert hf_env.resolve_snapshot_file(
        MODEL_ID, "chat_template.jinja", str(tmp_path),
    ) == str(metadata / "chat_template.jinja")


def test_snapshot_resolver_rejects_metadata_only_cache(tmp_path):
    _snapshot(tmp_path, "metadata")
    assert hf_env.resolve_snapshot_dir(MODEL_ID, str(tmp_path)) is None


def test_snapshot_file_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError, match="basename"):
        hf_env.resolve_snapshot_file(MODEL_ID, "../config.json", str(tmp_path))


def test_gemma_version_guard_is_early_and_actionable():
    with pytest.raises(loader.ModelLoaderError, match=r"transformers>=5\.10\.4"):
        loader._require_transformers_model_support(MODEL_ID, "5.5.0")
    loader._require_transformers_model_support(MODEL_ID, "5.10.4")
    loader._require_transformers_model_support("example/other", "4.40.0")


def test_request_slot_uses_complete_canonical_cache_without_download(
    monkeypatch, tmp_path,
):
    _reset_hf_cache()
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    snap = _snapshot(tmp_path, "weighted")
    (snap / "model.safetensors").write_bytes(b"weights")

    real_auto_download = catalog.auto_download_if_missing
    seen: dict[str, Path] = {}

    def guarded_auto_download(repo_id, *, hub_root):
        seen["hub_root"] = hub_root

        def network_forbidden(**_kwargs):
            raise AssertionError("snapshot_download must not run for a local cache")

        return real_auto_download(
            repo_id,
            hub_root=hub_root,
            _snapshot_download=network_forbidden,
        )

    monkeypatch.setattr(catalog, "auto_download_if_missing", guarded_auto_download)
    monkeypatch.setattr(
        loader, "_require_transformers_model_support", lambda *_args: None,
    )
    monkeypatch.setattr(
        loader,
        "load_llm",
        lambda model_id, **_kwargs: {
            "model": object(), "tokenizer": object(), "model_id": model_id,
        },
    )
    loader.LLM_CACHE.update({
        "model_id": None, "slot": None, "cache_entry": None,
    })
    try:
        entry = loader.request_slot("creative", MODEL_ID)
    finally:
        loader.LLM_CACHE.update({
            "model_id": None, "slot": None, "cache_entry": None,
        })

    assert entry["model_id"] == MODEL_ID
    assert seen["hub_root"] == tmp_path / "hub"
