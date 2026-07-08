"""Public-domain source-bank skeleton tests."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

from nodes import _otr_public_domain_sources as pd
from nodes import _otr_source_payload as osp

REPO = Path(__file__).resolve().parents[1]
SAMPLE_MANIFEST = (
    REPO / "config" / "source_banks" / "public_domain_story" / "manifest.sample.json"
)
MODULE_PATH = REPO / "nodes" / "_otr_public_domain_sources.py"


def _manifest():
    return pd.load_public_domain_manifest(SAMPLE_MANIFEST)


def test_sample_manifest_validates():
    manifest = _manifest()
    assert manifest["schema_version"] == "v1"
    source = manifest["sources"][0]
    assert source["license_status"] == "public_domain_us"
    assert source["recommended_word_budget"] == 300
    assert source["units"][0]["unit_id"] == "arrival"


def test_manifest_unknown_keys_fail_loud():
    manifest = _manifest()
    manifest["sources"][0]["unexpected"] = "nope"
    with pytest.raises(pd.PublicDomainManifestError, match="unknown"):
        pd.validate_public_domain_manifest(manifest)


def test_manifest_requires_rights_metadata():
    manifest = _manifest()
    del manifest["sources"][0]["license_url"]
    with pytest.raises(pd.PublicDomainManifestError, match="license_url"):
        pd.validate_public_domain_manifest(manifest)


def test_manifest_rejects_absolute_or_parent_text_paths():
    manifest = _manifest()
    manifest["sources"][0]["units"][0]["text_path"] = "../escape.txt"
    with pytest.raises(pd.PublicDomainManifestError, match="relative"):
        pd.validate_public_domain_manifest(manifest)


def test_manifest_rejects_duplicate_source_or_unit_ids():
    manifest = _manifest()
    manifest["sources"].append(dict(manifest["sources"][0]))
    with pytest.raises(pd.PublicDomainManifestError, match="duplicate source_id"):
        pd.validate_public_domain_manifest(manifest)

    manifest = _manifest()
    manifest["sources"][0]["units"].append(dict(manifest["sources"][0]["units"][0]))
    with pytest.raises(pd.PublicDomainManifestError, match="duplicate unit_id"):
        pd.validate_public_domain_manifest(manifest)


def test_resolve_source_ref():
    resolved = pd.resolve_manifest_unit(_manifest(), "gutenberg-time-machine-sample:arrival")
    assert resolved.source_ref == "gutenberg-time-machine-sample:arrival"
    assert resolved.source["title"] == "The Time Machine"
    assert resolved.unit["label"] == "The impossible arrival"


def test_resolve_source_ref_fails_loud():
    with pytest.raises(pd.PublicDomainSourceRefError, match="source_id:unit_id"):
        pd.resolve_manifest_unit(_manifest(), "arrival")
    with pytest.raises(pd.PublicDomainSourceRefError, match="unknown"):
        pd.resolve_manifest_unit(_manifest(), "gutenberg-time-machine-sample:nope")


def test_text_canonicalizer_strips_gutenberg_boilerplate_and_limits():
    raw = """
    *** START OF THE PROJECT GUTENBERG EBOOK THE TIME MACHINE ***
    First line.\n\nSecond&nbsp;line.
    *** END OF THE PROJECT GUTENBERG EBOOK THE TIME MACHINE ***
    License tail.
    """
    assert pd.canonicalize_public_domain_text(raw, max_chars=80) == "First line. Second line."


def test_payload_from_manifest_unit_has_exact_legacy_keys():
    resolved = pd.resolve_manifest_unit(_manifest(), "gutenberg-time-machine-sample:arrival")
    payload = pd.payload_from_manifest_unit(
        resolved,
        text="The machine stood in the room. The witnesses doubted him.",
    )
    assert set(payload) == osp.SOURCE_PAYLOAD_KEYS
    assert payload["headline"] == "The Time Machine - The impossible arrival"
    assert payload["source"] == "Project Gutenberg"
    assert "H. G. Wells" in payload["seed_text"]


def test_sidecars_are_separate_from_payload():
    resolved = pd.resolve_manifest_unit(_manifest(), "gutenberg-time-machine-sample:arrival")
    rights = pd.source_rights_from_unit(resolved)
    meta = pd.source_meta_from_unit(resolved)
    assert rights["license_status"] == "public_domain_us"
    assert meta["source_ref"] == resolved.source_ref
    payload = pd.payload_from_manifest_unit(resolved, text="A compact excerpt.")
    assert "source_rights" not in payload
    assert "source_meta" not in payload


def test_fetch_public_domain_source_uses_default_ref_and_sidecars():
    from nodes import _otr_story_routing as routing

    bank = routing.get_bank("public_domain_story")
    result = pd.fetch_public_domain_source(bank=bank)

    payload, meta, rights = osp.normalize_fetch_result(
        result, origin="public_domain_source")
    assert set(payload) == osp.SOURCE_PAYLOAD_KEYS
    assert payload["headline"] == "The Time Machine - The impossible arrival"
    assert payload["summary"].startswith("A shaken traveler")
    assert "START OF THE PROJECT GUTENBERG" not in payload["full_text"]
    assert payload["source"] == "Project Gutenberg"
    assert payload["date"] == "1895"
    assert payload["link"] == "https://www.gutenberg.org/ebooks/35"
    assert "The Time Machine by H. G. Wells" in payload["seed_text"]
    assert "Unit: The impossible arrival" in payload["seed_text"]
    assert meta["source_ref"] == "gutenberg-time-machine-sample:arrival"
    assert meta["recommended_word_budget"] == 300
    assert rights["license_status"] == "public_domain_us"


def test_fetch_public_domain_source_honors_explicit_ref():
    from nodes import _otr_story_routing as routing

    bank = routing.get_bank("public_domain_story")
    result = pd.fetch_public_domain_source(
        bank=bank,
        source_ref="gutenberg-time-machine-sample:arrival",
    )
    assert result.source_meta["source_ref"] == "gutenberg-time-machine-sample:arrival"


def test_fetch_public_domain_source_missing_defaults_fail_loud():
    from types import SimpleNamespace

    bank = SimpleNamespace(source_bank_id="public_domain_story", defaults={})
    with pytest.raises(pd.PublicDomainManifestError, match="manifest_path"):
        pd.fetch_public_domain_source(bank=bank)

    bank = SimpleNamespace(
        source_bank_id="public_domain_story",
        defaults={"manifest_path": str(SAMPLE_MANIFEST)},
    )
    with pytest.raises(pd.PublicDomainSourceRefError, match="source_ref"):
        pd.fetch_public_domain_source(bank=bank)


def test_cache_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_SOURCE_BANK_CACHE_DIR", str(tmp_path / "cache"))
    assert pd.source_bank_cache_root() == tmp_path / "cache"


def test_atomic_write_json_replaces_existing_file(tmp_path):
    target = tmp_path / "cache" / "manifest.json"
    pd.atomic_write_json(target, {"version": 1})
    pd.atomic_write_json(target, {"version": 2, "items": ["a"]})
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "items": ["a"],
        "version": 2,
    }
    assert not list(target.parent.glob("*.tmp"))


def test_module_import_is_lazy(monkeypatch):
    def _boom(self, *args, **kwargs):
        raise AssertionError(f"import-time file read attempted: {self}")

    monkeypatch.setattr(Path, "read_text", _boom)
    try:
        mod = importlib.reload(pd)
        assert mod is pd
    finally:
        monkeypatch.undo()
        importlib.reload(pd)


def test_module_has_no_network_or_heavy_imports():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    banned = {"requests", "urllib", "torch", "transformers", "feedparser"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names = {(node.module or "").split(".")[0]}
        else:
            continue
        assert not (names & banned)
