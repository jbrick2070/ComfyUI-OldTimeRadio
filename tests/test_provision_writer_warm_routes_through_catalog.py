"""The provisioner's writer warmer downloads through the catalog (cursor QA on
8f8ccebb): it used to call snapshot_download(cache_dir=...) itself, so a
provisioned box got its writer in the hub cache -- the layout the LLM folder
change retired -- and the runtime never created the folder.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("OTR_TEST_MODE", "1")


def _provision():
    spec = importlib.util.spec_from_file_location(
        "otr_provision_warm_test", REPO / "scripts" / "otr_provision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_warmer_calls_the_catalog_not_snapshot_download(tmp_path, monkeypatch):
    provision = _provision()
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    seen = {}

    class _Catalog:
        class _Row:
            repo_id = "Qwen/Qwen3.5-4B"
            provider = "local"
            loader_backend = next(iter(provision.WRITER_TRANSFORMERS_BACKENDS))
        CURATED_LLM_MODELS = (_Row(),)
        ALLOW_PATTERNS = ("*.json", "*.safetensors")

        @staticmethod
        def auto_download_if_missing(repo_id, *, hub_root=None, _snapshot_download=None, **kw):
            seen.update(repo_id=repo_id, hub_root=hub_root, seam=_snapshot_download)
            return str(tmp_path / "LLM" / "Qwen--Qwen3.5-4B")

    monkeypatch.setattr(provision, "_load_writer_catalog", lambda: _Catalog)
    lines = []
    monkeypatch.setattr(provision, "say", lambda *a: lines.append(a))

    def never(**kwargs):
        raise AssertionError("the warmer must not call snapshot_download directly")

    provision.warm_profile_writer_models(
        {"llm": {"creative_model": "Qwen/Qwen3.5-4B"}}, _snapshot_download=never)
    assert seen["repo_id"] == "Qwen/Qwen3.5-4B"
    assert seen["hub_root"] == Path(tmp_path / "hf" / "hub")
    assert seen["seam"] is never                       # the seam is passed through, not called here
    assert any(a[0] == "OK" for a in lines)


def test_the_real_standalone_catalog_can_run_the_download(tmp_path, monkeypatch):
    """THE ONE THE MOCK ABOVE CANNOT PROVE (Sonnet QA on 02758478). The
    provisioner loads the catalog BY PATH under a non-package name; a bare
    relative import inside auto_download_if_missing raised ImportError on
    entry there, and the warmer swallowed it as FAILED. So: load the catalog
    exactly as the provisioner does and run the function for real, with the
    download seam faked."""
    provision = _provision()
    catalog = provision._load_writer_catalog()          # the real module, standalone
    assert not hasattr(catalog, "__package__") or not catalog.__package__, \
        "this test must exercise the standalone (non-package) load"
    llm_root = tmp_path / "LLM"
    llm_root.mkdir()
    monkeypatch.setenv("OTR_LLM_DIR", str(llm_root))
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    monkeypatch.setattr(catalog, "estimate_model_size_gb", lambda *a, **k: 1.0)
    monkeypatch.setattr(catalog, "_free_disk_bytes_for", lambda p: 10 ** 12)
    from tests.test_llm_folder import FOLDER, REPO_ID, _plain
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        _plain(llm_root, receipt=False)                 # a complete folder lands
        return kwargs["local_dir"]

    out = catalog.auto_download_if_missing(
        REPO_ID, hub_root=tmp_path / "hub", _snapshot_download=fake_download)
    (kwargs,) = calls
    assert kwargs["local_dir"] == str(llm_root / FOLDER)
    assert out == str(llm_root / FOLDER)


def test_the_standalone_catalog_refuses_cleanly_when_the_hub_is_unreachable(tmp_path, monkeypatch):
    """The refusal branches import too. On a provisioned box with no Hub
    access, estimate_model_size_gb's except branch must raise
    UnknownModelError, not ImportError (which the warmer would swallow)."""
    provision = _provision()
    catalog = provision._load_writer_catalog()
    monkeypatch.setenv("OTR_LLM_DIR", str(tmp_path / "LLM"))
    (tmp_path / "LLM").mkdir()
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")

    class _NoHub:
        def model_info(self, *a, **k):
            raise ConnectionError("no route to the Hub")

    # An UNCURATED id: a curated row takes its size from the catalog and never
    # asks the Hub, so only an unknown repo reaches the branch under test.
    from tests.test_llm_folder import REPO_ID
    with pytest.raises(Exception) as err:
        catalog.auto_download_if_missing(
            "someorg/not-a-curated-model", hub_root=tmp_path / "hub", _hf_api=_NoHub(),
            _snapshot_download=lambda **k: (_ for _ in ()).throw(AssertionError("must not download")))
    assert type(err.value).__name__ == "UnknownModelError", repr(err.value)
    assert not isinstance(err.value, ImportError)

    # and the plain "auto-download off" refusal, the other branch a box hits
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "0")
    with pytest.raises(Exception) as err2:
        catalog.auto_download_if_missing(REPO_ID, hub_root=tmp_path / "hub")
    assert type(err2.value).__name__ == "UnknownModelError", repr(err2.value)


def test_the_warmer_source_has_no_direct_snapshot_download_call():
    src = (REPO / "scripts" / "otr_provision.py").read_text(encoding="utf-8")
    start = src.index("def warm_profile_writer_models(")
    body = src[start:src.index("\ndef ", start + 1)]
    assert "auto_download_if_missing(" in body
    assert "_snapshot_download(\n" not in body and "_snapshot_download(repo_id" not in body
