"""The boot note that explains AnimateDiff-Evolved's expected red line."""
from __future__ import annotations

from nodes._otr_boot_notes import ade_motion_module_note


def _tree(tmp_path, ade_name="comfyui-animatediff-evolved"):
    custom = tmp_path / "custom_nodes"
    (custom / ade_name / "models").mkdir(parents=True)
    models = tmp_path / "models"
    (models / "animatediff_models").mkdir(parents=True)
    return custom, models


def test_ade_with_no_motion_module_gets_the_note(tmp_path):
    custom, models = _tree(tmp_path)
    note = ade_motion_module_note([str(custom)], str(models))
    assert note and "No motion models found" in note and "expected" in note


def test_no_note_once_a_motion_module_exists_in_either_folder(tmp_path):
    custom, models = _tree(tmp_path)
    (models / "animatediff_models" / "v3_sd15_mm.ckpt").write_bytes(b"x")
    assert ade_motion_module_note([str(custom)], str(models)) is None
    custom2, models2 = _tree(tmp_path / "b", ade_name="ComfyUI-AnimateDiff-Evolved")
    (custom2 / "ComfyUI-AnimateDiff-Evolved" / "models" / "m.safetensors").write_bytes(b"x")
    assert ade_motion_module_note([str(custom2)], str(models2)) is None


def test_no_note_without_the_pack_or_when_it_is_disabled(tmp_path):
    custom = tmp_path / "custom_nodes"
    custom.mkdir()
    assert ade_motion_module_note([str(custom)], str(tmp_path / "models")) is None
    (custom / "comfyui-animatediff-evolved.disabled").mkdir()
    assert ade_motion_module_note([str(custom)], str(tmp_path / "models")) is None


def test_the_note_is_wired_into_the_pack_load():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "__init__.py").read_text(
        encoding="utf-8")
    assert "ade_motion_module_note as _otr_ade_note" in src
    assert '_otr_fp.get_folder_paths("custom_nodes")' in src
