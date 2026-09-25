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


def test_the_warmer_source_has_no_direct_snapshot_download_call():
    src = (REPO / "scripts" / "otr_provision.py").read_text(encoding="utf-8")
    start = src.index("def warm_profile_writer_models(")
    body = src[start:src.index("\ndef ", start + 1)]
    assert "auto_download_if_missing(" in body
    assert "_snapshot_download(\n" not in body and "_snapshot_download(repo_id" not in body
