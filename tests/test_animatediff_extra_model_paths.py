"""Windows extra_model_paths addendum names animatediff_models."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_windows_addendum_maps_animatediff_models():
    text = (REPO / "config" / "otr_windows_extra_model_paths.yaml").read_text(
        encoding="utf-8")
    assert "animatediff_models:" in text
    assert "C:/ComfyUI-Models" in text
    mac = (REPO / "config" / "otr_mac_extra_model_paths.yaml").read_text(
        encoding="utf-8")
    assert "animatediff_models" in mac
