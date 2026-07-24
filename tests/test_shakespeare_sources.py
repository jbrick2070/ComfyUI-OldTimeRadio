"""Shakespeare source-bank fixture tests."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

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


def _write_two_scene_manifest(tmp_path: Path) -> tuple[Path, str, str]:
    manifest = _manifest()
    first = dict(manifest["scenes"][0])
    second = dict(first)
    first["text_path"] = "fixtures/first.txt"
    second.update({
        "source_ref": "folger-temp:act2-scene1-second",
        "play_code": "Tmp",
        "play_title": "Tempest",
        "act": 2,
        "scene": 1,
        "scene_label": "Act 2, Scene 1 - a second selected scene",
        "synopsis": "A second fixture scene proves blank source_ref can draw from the deck.",
        "text_path": "fixtures/second.txt",
    })
    manifest["scenes"] = [first, second]
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "first.txt").write_text(
        "FIRST WITCH: A first fixture line.", encoding="utf-8")
    (fixtures / "second.txt").write_text(
        "PROSPERO: A second fixture line.", encoding="utf-8")
    manifest_path = tmp_path / "curated_scenes.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, first["source_ref"], second["source_ref"]


def test_sample_manifest_validates():
    manifest = _manifest()
    assert manifest["schema_version"] == "v1"
    assert len(manifest["scenes"]) >= 10
    scene = manifest["scenes"][0]
    assert scene["source_ref"] == "folger-macbeth:act1-scene3-witches"
    assert scene["play_title"] == "Macbeth"
    assert scene["commercial_use_allowed"] is False
    assert scene["license_label"] == "CC BY-NC 3.0"
    assert scene["recommended_word_budget"] == 300


def test_scene_deck_has_source_links_and_nonempty_assets():
    manifest = _manifest()
    play_titles = {scene["play_title"] for scene in manifest["scenes"]}
    assert len(play_titles) >= 8
    for scene in manifest["scenes"]:
        assert scene["source_url"].startswith(
            "https://www.folger.edu/explore/shakespeares-works/")
        assert scene["synopsis"].strip()
        assert scene["scene_label"].strip()
        text_path = SAMPLE_MANIFEST.parent / scene["text_path"]
        assert text_path.is_file()
        assert text_path.read_text(encoding="utf-8").strip()


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


def test_select_shakespeare_scene_ref_uses_manifest_deck(tmp_path):
    manifest_path, _first_ref, second_ref = _write_two_scene_manifest(tmp_path)
    manifest = shx.load_shakespeare_manifest(manifest_path)

    class PickSecond:
        def choice(self, items):
            return items[1]

    assert shx.select_shakespeare_scene_ref(manifest, rng=PickSecond()) == second_ref


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


def test_fetch_shakespeare_scene_blank_ref_selects_from_manifest_and_sidecars(monkeypatch):
    from nodes import _otr_story_routing as routing

    bank = routing.get_bank("shakespeare")
    monkeypatch.setattr(
        shx,
        "select_shakespeare_scene_ref",
        lambda manifest: "folger-macbeth:act1-scene3-witches",
    )
    result = shx.fetch_shakespeare_scene(bank=bank)
    payload, meta, rights = osp.normalize_fetch_result(
        result, origin="shakespeare_folger")
    assert set(payload) == osp.SOURCE_PAYLOAD_KEYS
    assert payload["headline"] == "Macbeth, Act 1, Scene 3"
    assert "Three watchers greet Macbeth" in payload["summary"]
    assert "FIRST WITCH" in payload["full_text"]
    assert meta["source_ref"] == "folger-macbeth:act1-scene3-witches"
    assert meta["play_title"] == "Macbeth"
    assert rights["commercial_use_allowed"] is False
    assert rights["license_url"] == "https://www.folger.edu/copyright-policy/"


def test_fetch_shakespeare_scene_blank_ref_randomizes_across_manifest(
    tmp_path, monkeypatch,
):
    manifest_path, _first_ref, second_ref = _write_two_scene_manifest(tmp_path)
    bank = SimpleNamespace(
        source_bank_id="shakespeare",
        defaults={"manifest_path": str(manifest_path)},
    )
    monkeypatch.setattr(
        shx,
        "select_shakespeare_scene_ref",
        lambda manifest: second_ref,
    )

    result = shx.fetch_shakespeare_scene(bank=bank)

    assert result.source_meta["source_ref"] == second_ref
    assert result.source_meta["play_title"] == "Tempest"
    assert "second fixture line" in result.payload["full_text"]


def test_fetch_shakespeare_scene_honors_explicit_ref(monkeypatch):
    from nodes import _otr_story_routing as routing

    def _should_not_randomize(_manifest):
        raise AssertionError("explicit source_ref should bypass random selection")

    monkeypatch.setattr(shx, "select_shakespeare_scene_ref", _should_not_randomize)

    bank = routing.get_bank("shakespeare")
    result = shx.fetch_shakespeare_scene(
        bank=bank,
        source_ref="folger-macbeth:act1-scene3-witches",
    )
    assert result.source_meta["source_ref"] == "folger-macbeth:act1-scene3-witches"


def test_fetch_shakespeare_scene_missing_defaults_fail_loud():
    bank = SimpleNamespace(source_bank_id="shakespeare", defaults={})
    with pytest.raises(shx.ShakespeareManifestError, match="manifest_path"):
        shx.fetch_shakespeare_scene(bank=bank)

    bank = SimpleNamespace(
        source_bank_id="shakespeare",
        defaults={
            "manifest_path": str(SAMPLE_MANIFEST),
            "selection_mode": "fixed",
        },
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
