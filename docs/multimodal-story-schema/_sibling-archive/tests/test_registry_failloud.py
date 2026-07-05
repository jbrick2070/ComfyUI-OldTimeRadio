"""Registry fail-loud + no-fallback contract."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from upstream_story_lab.registry import Registry, RegistryError, UnknownIdError

ROOT = Path(__file__).resolve().parents[1]


def _copy_fixtures(tmp_path: Path) -> Path:
    root = tmp_path / "lab"
    shutil.copytree(ROOT / "fixtures", root / "fixtures")
    return root


def test_registry_loads_real_fixtures(registry) -> None:
    assert set(registry.banks) == {
        "science_news", "media_archive", "public_domain_story", "custom_source_bank",
    }
    assert len(registry.packs) == 12
    assert set(registry.styles) == {
        "anime", "archival_documentary", "cartoon", "paper_origami", "sci_fi_radio",
    }
    assert set(registry.pipelines) == {"legacy_many_pass", "simple_4_prompt_experimental"}


def test_unknown_ids_raise(registry) -> None:
    with pytest.raises(UnknownIdError):
        registry.bank("mystery_bank")
    with pytest.raises(UnknownIdError):
        registry.pack("media_archive", "space_opera", "legacy_many_pass")
    with pytest.raises(UnknownIdError):
        registry.style("vaporwave")
    with pytest.raises(UnknownIdError):
        registry.pipeline("nine_pass_hyperloop")


def test_missing_fixture_dir_fails_loud(tmp_path) -> None:
    root = _copy_fixtures(tmp_path)
    shutil.rmtree(root / "fixtures" / "visual_styles")
    with pytest.raises(RegistryError, match="visual style folder missing"):
        Registry(root)


def test_missing_style_file_is_not_silently_served(tmp_path) -> None:
    """The v1 silent Python fallback is dead: deleting a style JSON that a
    bank defaults to must fail loudly at load."""

    root = _copy_fixtures(tmp_path)
    (root / "fixtures" / "visual_styles" / "archival_documentary.json").unlink()
    with pytest.raises(RegistryError, match="default_visual_style"):
        Registry(root)


def test_duplicate_pack_key_fails(tmp_path) -> None:
    root = _copy_fixtures(tmp_path)
    src = root / "fixtures" / "story_packs" / "media_archive" / "gentle_thriller.json"
    dup = src.with_name("gentle_thriller_copy.json")
    dup.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(RegistryError, match="duplicate story pack key"):
        Registry(root)


def test_undeclared_template_variable_fails_at_load(tmp_path) -> None:
    root = _copy_fixtures(tmp_path)
    path = root / "fixtures" / "story_packs" / "media_archive" / "gentle_thriller.json"
    pack = json.loads(path.read_text(encoding="utf-8"))
    pack["prompt_stages"]["line_grounding"] = "Ground this in {sourec_label}."
    path.write_text(json.dumps(pack), encoding="utf-8")
    with pytest.raises(RegistryError, match="undeclared variables"):
        Registry(root)


def test_motion_prompt_dead_role_rejected(tmp_path) -> None:
    root = _copy_fixtures(tmp_path)
    path = root / "fixtures" / "visual_styles" / "archival_documentary.json"
    style = json.loads(path.read_text(encoding="utf-8"))
    style["motion_prompts"]["scene_broll"] = "slow move through an archive"
    path.write_text(json.dumps(style), encoding="utf-8")
    with pytest.raises(Exception, match="scene_broll"):
        Registry(root)


def test_custom_source_bank_not_runnable(registry) -> None:
    with pytest.raises(RegistryError, match="not\\s+runnable"):
        registry.source_packet("custom_source_bank")


def test_pd_manifests_validate_and_paths_are_safe(registry) -> None:
    manifests = registry.public_domain_manifests()
    assert len(manifests) == 3
    for manifest, _path in manifests:
        assert manifest.rights_status == "public_domain"


def test_pd_unsafe_path_rejected(tmp_path) -> None:
    root = _copy_fixtures(tmp_path)
    path = root / "fixtures" / "public_domain_sources" / "book_chapter_sample" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["text_files"] = ["../escape.txt"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RegistryError, match="unsafe path"):
        Registry(root).public_domain_manifests()
