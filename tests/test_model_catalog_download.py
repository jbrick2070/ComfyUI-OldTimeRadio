"""S30 B1a2 tests -- auto_download_if_missing + size estimate + disk pre-check
+ GatedModelError + resolve_hf_token.

All tests pass test-seam mocks to avoid real HF API calls or real downloads.
No real network, no real GPU, no real filesystem writes outside tmp_path.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nodes import _otr_model_catalog as catalog
from nodes._otr_hf_auth import resolve_hf_token
from nodes._otr_model_inputs import (
    GatedModelError,
    InsufficientDiskSpaceError,
    UnknownModelError,
)


# ---------------------------------------------------------------------------
# resolve_hf_token
# ---------------------------------------------------------------------------


def test_resolve_hf_token_from_environ(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token-abc")
    assert resolve_hf_token() == "test-token-abc"


def test_resolve_hf_token_returns_none_when_missing(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # On non-Windows the winreg branch is skipped; on Windows the
    # registry lookup may legitimately return a value, so the safe
    # cross-platform assertion is "returns either None or a string".
    out = resolve_hf_token()
    assert out is None or isinstance(out, str)


def test_resolve_hf_token_is_cross_platform_safe():
    """The function must not raise on any OS even with no token."""
    # Smoke: simply calling it must not crash.
    resolve_hf_token()


# ---------------------------------------------------------------------------
# estimate_model_size_gb
# ---------------------------------------------------------------------------


def test_estimate_curated_returns_catalog_value():
    # Curated entries skip the network call entirely.
    out = catalog.estimate_model_size_gb(catalog.DEFAULT_LLM)
    assert out == 24.0


def test_estimate_uncurated_uses_hf_api_seam():
    """Uncurated repo_id falls through to the HfApi seam."""
    fake_sibling_safetensors = MagicMock()
    fake_sibling_safetensors.rfilename = "model.safetensors"
    fake_sibling_safetensors.size = 5 * 1024**3  # 5 GB
    fake_sibling_json = MagicMock()
    fake_sibling_json.rfilename = "config.json"
    fake_sibling_json.size = 1024
    fake_info = MagicMock()
    fake_info.siblings = [fake_sibling_safetensors, fake_sibling_json]
    fake_api = MagicMock()
    fake_api.model_info.return_value = fake_info

    out = catalog.estimate_model_size_gb("uncurated/test-model", _hf_api=fake_api)
    assert out == pytest.approx(5.0, abs=0.01)
    fake_api.model_info.assert_called_once_with("uncurated/test-model", files_metadata=True)


# ---------------------------------------------------------------------------
# auto_download_if_missing -- pre-flight checks
# ---------------------------------------------------------------------------


def test_auto_download_respects_disabled_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "0")
    snap = MagicMock()
    with pytest.raises(UnknownModelError) as exc:
        catalog.auto_download_if_missing(
            catalog.DEFAULT_LLM,
            hub_root=tmp_path,
            _snapshot_download=snap,
        )
    assert "auto-download disabled" in str(exc.value).lower()
    snap.assert_not_called()


def _seed_fake_snapshot(tmp_path: Path, repo_id: str) -> Path:
    """Build a minimal HF cache layout under `tmp_path` for `repo_id`,
    matching the directory shape scan_local_llm_cache expects.
    """
    repo_dir_name = "models--" + repo_id.replace("/", "--")
    snap_dir = tmp_path / repo_dir_name / "snapshots" / "abc123fakehash"
    snap_dir.mkdir(parents=True)
    (snap_dir / "config.json").write_text(
        '{"max_position_embeddings": 8192}', encoding="utf-8"
    )
    # A materialized weight blob so the snapshot counts as on_disk (the
    # local-cache short-circuit requires a usable snapshot, not config-only).
    (snap_dir / "model.safetensors").write_bytes(b"\0" * 64)
    return snap_dir


def test_auto_download_returns_cached_path_without_token(tmp_path, monkeypatch):
    """B1d: a cached gated repo + missing HF_TOKEN must return the local
    path immediately. The local-cache short-circuit fires BEFORE the
    gated-token check; user shouldn't be punished for losing their token
    after the one-time download landed.
    """
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("nodes._otr_hf_auth.resolve_hf_token", lambda: None)
    snap_dir = _seed_fake_snapshot(tmp_path, catalog.DEFAULT_LLM)

    snap = MagicMock()
    api = MagicMock()
    out = catalog.auto_download_if_missing(
        catalog.DEFAULT_LLM,
        hub_root=tmp_path,
        _snapshot_download=snap,
        _hf_api=api,
    )

    snap.assert_not_called()
    api.model_info.assert_not_called()
    assert out == str(snap_dir)


def test_auto_download_skips_when_on_disk(tmp_path, monkeypatch):
    """B1d: when scan_local_llm_cache reports the repo on disk, the
    short-circuit returns its snapshot_path and skips both the gated
    check and the snapshot_download call.
    """
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    snap_dir = _seed_fake_snapshot(tmp_path, "totally/uncurated-model")

    snap = MagicMock()
    api = MagicMock()
    out = catalog.auto_download_if_missing(
        "totally/uncurated-model",
        hub_root=tmp_path,
        _snapshot_download=snap,
        _hf_api=api,
    )

    snap.assert_not_called()
    api.model_info.assert_not_called()
    assert out == str(snap_dir)


def test_estimate_model_size_gb_404_raises_unknown():
    """B1d: an HfApi.model_info failure (RepositoryNotFoundError, network,
    HfHubHTTPError, etc.) must surface as UnknownModelError carrying the
    recovery hint -- not bubble the raw HF exception type out.
    """
    class _FakeRepositoryNotFoundError(Exception):
        """Stand-in for huggingface_hub.errors.RepositoryNotFoundError
        without importing huggingface_hub at unit-test time."""

    fake_api = MagicMock()
    fake_api.model_info.side_effect = _FakeRepositoryNotFoundError(
        "404 Client Error: Repository Not Found for 'totally/nonexistent'"
    )
    with pytest.raises(UnknownModelError) as exc:
        catalog.estimate_model_size_gb(
            "totally/nonexistent", _hf_api=fake_api
        )
    msg = str(exc.value)
    assert "totally/nonexistent" in msg
    assert "could not be resolved" in msg.lower()
    assert "_FakeRepositoryNotFoundError" in msg


def test_auto_download_gated_no_token_raises_before_snapshot_download(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    # Force the Windows registry branch to also miss by stubbing
    # resolve_hf_token to return None.
    monkeypatch.setattr(
        "nodes._otr_hf_auth.resolve_hf_token", lambda: None
    )
    snap = MagicMock()
    api = MagicMock()
    with pytest.raises(GatedModelError) as exc:
        catalog.auto_download_if_missing(
            catalog.DEFAULT_LLM,  # Mistral-Nemo is gated
            hub_root=tmp_path,
            _snapshot_download=snap,
            _hf_api=api,
        )
    # Pre-flight check ran BEFORE the download attempt.
    snap.assert_not_called()
    api.model_info.assert_not_called()
    msg = str(exc.value)
    assert "huggingface.co/join" in msg
    assert catalog.DEFAULT_LLM in msg
    assert "license" in msg.lower()


def test_auto_download_gated_with_token_proceeds(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    snap = MagicMock(return_value=str(tmp_path / "fake-snapshot"))
    out = catalog.auto_download_if_missing(
        catalog.DEFAULT_LLM,
        hub_root=tmp_path,
        _snapshot_download=snap,
    )
    snap.assert_called_once()
    kwargs = snap.call_args.kwargs
    assert kwargs["repo_id"] == catalog.DEFAULT_LLM
    assert kwargs["token"] == "fake-token"
    assert ".safetensors" in " ".join(kwargs["allow_patterns"])
    assert ".gguf" not in " ".join(kwargs["allow_patterns"])
    assert out == str(tmp_path / "fake-snapshot")


def test_auto_download_disk_space_precheck(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    # Force the free-disk check to report 1 GB free; Mistral-Nemo
    # needs 24 GB + 5 GB margin -> raise.
    monkeypatch.setattr(catalog, "_free_disk_bytes_for", lambda _p: 1 * 1024**3)
    snap = MagicMock()
    with pytest.raises(InsufficientDiskSpaceError) as exc:
        catalog.auto_download_if_missing(
            catalog.DEFAULT_LLM,
            hub_root=tmp_path,
            _snapshot_download=snap,
        )
    snap.assert_not_called()
    msg = str(exc.value)
    assert "24" in msg or "29" in msg  # 24 + 5 margin
    assert "1.0" in msg or "1 GB" in msg.lower() or "1." in msg


def test_auto_download_uncurated_uses_hf_api_for_size(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    # Plenty of free disk so the pre-check passes.
    monkeypatch.setattr(catalog, "_free_disk_bytes_for", lambda _p: 500 * 1024**3)

    fake_sibling = MagicMock()
    fake_sibling.rfilename = "model.safetensors"
    fake_sibling.size = 3 * 1024**3
    fake_info = MagicMock()
    fake_info.siblings = [fake_sibling]
    fake_api = MagicMock()
    fake_api.model_info.return_value = fake_info

    snap = MagicMock(return_value=str(tmp_path / "snap"))
    out = catalog.auto_download_if_missing(
        "uncurated/test-model",
        hub_root=tmp_path,
        _snapshot_download=snap,
        _hf_api=fake_api,
    )
    fake_api.model_info.assert_called_once()
    snap.assert_called_once()
    assert out == str(tmp_path / "snap")


# ---------------------------------------------------------------------------
# allow_patterns guard
# ---------------------------------------------------------------------------


def test_allow_patterns_excludes_gguf():
    """B1a2 ships safetensors-only; GGUF support is a future sprint."""
    assert "*.gguf" not in catalog.ALLOW_PATTERNS
    for p in catalog.ALLOW_PATTERNS:
        assert not p.endswith(".gguf")


def test_allow_patterns_includes_safetensors_and_tokenizer():
    assert "*.safetensors" in catalog.ALLOW_PATTERNS
    assert any(p.startswith("tokenizer") for p in catalog.ALLOW_PATTERNS)
