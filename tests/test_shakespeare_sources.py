"""Shakespeare source-bank fixture tests."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

from nodes import _otr_shakespeare_sources as shx
from nodes import _otr_source_payload as osp

REPO = Path(__file__).resolve().parents[1]
SAMPLE_MANIFEST = (
    REPO / "config" / "source_banks" / "shakespeare" / "curated_scenes.sample.json"
)
MODULE_PATH = REPO / "nodes" / "_otr_shakespeare_sources.py"


@pytest.fixture(autouse=True)
def _production_story_routing_root(monkeypatch):
    """Keep these tests independent of synthetic registry tests run earlier."""
    from nodes import _otr_story_routing as routing

    monkeypatch.setattr(
        routing,
        "_STORY_PACKS_ROOT",
        REPO / "nodes" / "story_packs",
    )
    routing._clear_caches()
    yield
    routing._clear_caches()


def _manifest():
    return shx.load_shakespeare_manifest(SAMPLE_MANIFEST)


def test_sample_manifest_validates():
    manifest = _manifest()
    assert manifest["schema_version"] == "v1"
    scene = manifest["scenes"][0]
    assert scene["source_ref"] == "folger-macbeth:act1-scene3-witches"
    assert scene["play_title"] == "Macbeth"
    assert scene["commercial_use_allowed"] is False
    assert scene["license_label"] == "CC BY-NC 3.0"
    assert scene["recommended_word_budget"] == 300


def test_manifest_unknown_keys_and_rights_fail_loud():
    manifest = _manifest()
    manifest["scenes"][0]["unexpected"] = "nope"
    with pytest.raises(shx.ShakespeareManifestError, match="unknown"):
        shx.validate_shakespeare_manifest(manifest)

    manifest = _manifest()
    del manifest["scenes"][0]["license_url"]
    with pytest.raises(shx.ShakespeareManifestError, match="license_url"):
        shx.validate_shakespeare_manifest(manifest)

    manifest = _manifest()
    manifest["scenes"][0]["commercial_use_allowed"] = "no"
    with pytest.raises(shx.ShakespeareManifestError, match="commercial_use_allowed"):
        shx.validate_shakespeare_manifest(manifest)


def test_manifest_rejects_absolute_or_parent_text_paths():
    manifest = _manifest()
    manifest["scenes"][0]["text_path"] = "../escape.txt"
    with pytest.raises(shx.ShakespeareManifestError, match="relative"):
        shx.validate_shakespeare_manifest(manifest)


def test_manifest_rejects_duplicate_source_refs():
    manifest = _manifest()
    manifest["scenes"].append(dict(manifest["scenes"][0]))
    with pytest.raises(shx.ShakespeareManifestError, match="duplicate source_ref"):
        shx.validate_shakespeare_manifest(manifest)


def test_resolve_shakespeare_scene():
    resolved = shx.resolve_shakespeare_scene(
        _manifest(),
        "folger-macbeth:act1-scene3-witches",
    )
    assert resolved.source_ref == "folger-macbeth:act1-scene3-witches"
    assert resolved.scene["play_code"] == "Mac"
    assert resolved.scene["act"] == 1
    assert resolved.scene["scene"] == 3


def test_resolve_shakespeare_scene_fails_loud():
    with pytest.raises(shx.ShakespeareSourceRefError, match="source_ref"):
        shx.resolve_shakespeare_scene(_manifest(), "")
    with pytest.raises(shx.ShakespeareSourceRefError, match="unknown"):
        shx.resolve_shakespeare_scene(_manifest(), "Mac:nope")


def test_payload_from_scene_has_exact_legacy_keys():
    resolved = shx.resolve_shakespeare_scene(
        _manifest(),
        "folger-macbeth:act1-scene3-witches",
    )
    payload = shx.payload_from_scene(
        resolved,
        text="FIRST WITCH: All hail, Macbeth!\nBANQUO: Why do you start?",
    )
    assert set(payload) == osp.SOURCE_PAYLOAD_KEYS
    assert payload["headline"] == "Macbeth, Act 1, Scene 3"
    assert payload["source"] == "Folger Shakespeare"
    assert "CC BY-NC 3.0" in payload["date"]
    assert "Speakers: FIRST WITCH, BANQUO" in payload["seed_text"]


def test_sidecars_are_separate_and_noncommercial():
    resolved = shx.resolve_shakespeare_scene(
        _manifest(),
        "folger-macbeth:act1-scene3-witches",
    )
    rights = shx.source_rights_from_scene(resolved)
    meta = shx.source_meta_from_scene(resolved)
    assert rights["commercial_use_allowed"] is False
    assert rights["license_label"] == "CC BY-NC 3.0"
    assert meta["source_ref"] == resolved.source_ref
    payload = shx.payload_from_scene(resolved, text="FIRST WITCH: All hail.")
    assert "source_rights" not in payload
    assert "source_meta" not in payload


def test_fetch_shakespeare_scene_uses_default_ref_and_sidecars():
    from nodes import _otr_story_routing as routing

    bank = routing.get_bank("shakespeare")
    result = shx.fetch_shakespeare_scene(bank=bank)
    payload, meta, rights = osp.normalize_fetch_result(
        result, origin="shakespeare_folger")
    assert set(payload) == osp.SOURCE_PAYLOAD_KEYS
    assert payload["headline"] == "Macbeth, Act 1, Scene 3"
    assert "Three witches greet Macbeth" in payload["summary"]
    assert "FIRST WITCH" in payload["full_text"]
    assert meta["source_ref"] == "folger-macbeth:act1-scene3-witches"
    assert meta["play_title"] == "Macbeth"
    assert rights["commercial_use_allowed"] is False
    assert rights["license_url"] == "https://www.folger.edu/copyright-policy/"


def test_fetch_shakespeare_scene_honors_explicit_ref():
    from nodes import _otr_story_routing as routing

    bank = routing.get_bank("shakespeare")
    result = shx.fetch_shakespeare_scene(
        bank=bank,
        source_ref="folger-macbeth:act1-scene3-witches",
    )
    assert result.source_meta["source_ref"] == "folger-macbeth:act1-scene3-witches"


def test_fetch_shakespeare_scene_missing_defaults_fail_loud():
    from types import SimpleNamespace

    bank = SimpleNamespace(source_bank_id="shakespeare", defaults={})
    with pytest.raises(shx.ShakespeareManifestError, match="manifest_path"):
        shx.fetch_shakespeare_scene(bank=bank)

    bank = SimpleNamespace(
        source_bank_id="shakespeare",
        defaults={"manifest_path": str(SAMPLE_MANIFEST)},
    )
    with pytest.raises(shx.ShakespeareSourceRefError, match="source_ref"):
        shx.fetch_shakespeare_scene(bank=bank)


def test_parse_folger_scene_xml_snippet():
    parsed = shx.parse_folger_scene(
        """
        <TEI xmlns="http://www.tei-c.org/ns/1.0">
          <text><body><div>
            <stage>Thunder.</stage>
            <sp><speaker>First Witch</speaker><l>All hail, Macbeth!</l></sp>
            <sp><speaker>Banquo</speaker><l>Why do you start?</l></sp>
          </div></body></text>
        </TEI>
        """,
        play_code="Mac",
        act=1,
        scene=3,
    )
    assert parsed.play_code == "Mac"
    assert parsed.speakers == ("FIRST WITCH", "BANQUO")
    assert parsed.stage_directions == ("Thunder.",)
    assert "All hail, Macbeth" in parsed.text


def test_module_import_is_lazy(monkeypatch):
    def _boom(self, *args, **kwargs):
        raise AssertionError(f"import-time file read attempted: {self}")

    monkeypatch.setattr(Path, "read_text", _boom)
    try:
        mod = importlib.reload(shx)
        assert mod is shx
    finally:
        monkeypatch.undo()
        importlib.reload(shx)


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
