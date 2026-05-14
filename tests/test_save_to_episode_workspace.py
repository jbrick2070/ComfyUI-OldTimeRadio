"""BUG-LOCAL-028 regression: FLUX env stills + radio bookend must land
in the per-episode workspace, not the legacy flat dir.

Three properties pinned by these tests:

1. With an active in-flight Ledger singleton, ``OTR_SaveToEpisodeWorkspace``
   resolves to ``output/otr/episodes/<episode_id>/stills/`` (or
   .../portraits/), NOT ``_legacy_stills/`` / ``_legacy_portraits/``.

2. With NO singleton (headless / standalone test), it falls back to the
   legacy flat dir without raising.

3. The auto-counter resets per-episode (starts at 1 in a fresh dir),
   does NOT share the global counter that stock ``SaveImage`` had.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np  # type: ignore
import pytest

# Match repo import pattern.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OTR_TEST_MODE", "1")


def _fake_image_tensor(h: int = 128, w: int = 128) -> np.ndarray:
    """High-entropy RGB image as a numpy array in 0..1 float, ComfyUI tensor shape.

    Sized to clear the SaveToEpisodeWorkspace ``_MIN_VALID_PNG_BYTES`` gate
    (4096 bytes) that catches empty FLUX outputs. Random RGB data compresses
    poorly under PNG deflate -- a 128x128 fixture lands at ~30-40 KB, well
    over the gate. Was 8x8 before the gate shipped (2026-05-08 morning run);
    that produced a 268-byte PNG that tripped the empty-FLUX detector and
    raised BAD_IMAGE_SAVE in every save_to_episode_workspace test.
    """
    return np.random.RandomState(42).random((h, w, 3)).astype(np.float32)


@pytest.fixture
def save_node():
    """Import lazily so the conftest CUDA mask + path setup are in effect."""
    from nodes.otr_save_to_episode_workspace import SaveToEpisodeWorkspace

    return SaveToEpisodeWorkspace()


def test_input_types_shape(save_node):
    spec = save_node.INPUT_TYPES()
    assert "required" in spec
    assert "images" in spec["required"]
    assert "role_kind" in spec["required"]
    assert "filename_pattern" in spec["required"]
    role_choices = spec["required"]["role_kind"][0]
    assert "stills" in role_choices
    assert "portraits" in role_choices


def test_node_class_is_output_node(save_node):
    """Save sinks must declare OUTPUT_NODE so ComfyUI runs them every queue."""
    assert getattr(save_node, "OUTPUT_NODE", False) is True
    assert save_node.RETURN_TYPES == ()
    assert save_node.FUNCTION == "save"


def test_save_to_per_episode_dir_when_singleton_active(save_node, tmp_path):
    """With an active singleton, image lands at
    ``<comfy_out>/otr/episodes/<ep>/stills/<pattern>_NNNNN_.png``."""
    ep_id = "signal_lost_test_episode_20260503_010101"
    fake_comfy_out = tmp_path / "ComfyUI_output"

    # Build a fake ledger on disk + a singleton path that points at it.
    ep_dir = fake_comfy_out / "otr" / "episodes" / ep_id
    audio_dir = ep_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = audio_dir / f"{ep_id}_ledger.json"
    ledger_path.write_text(
        '{"episode_id": "' + ep_id + '", "schema_version": "l3-test"}',
        encoding="utf-8",
    )

    with mock.patch(
        "nodes._otr_paths.comfy_output_dir",
        return_value=fake_comfy_out,
    ), mock.patch(
        "nodes._otr_ledger.in_flight_ledger_path",
        return_value=ledger_path,
    ):
        result = save_node.save(
            images=[_fake_image_tensor()],
            role_kind="stills",
            filename_pattern="full_env",
        )

    expected_dir = ep_dir / "stills"
    saved = list(expected_dir.glob("full_env_*.png"))
    assert len(saved) == 1, (
        f"expected 1 PNG under per-episode stills/, got {saved}"
    )
    # Counter should start at 1 for a fresh per-episode dir.
    assert saved[0].name == "full_env_00001_.png"
    # UI dict must exist so ComfyUI shows the saved-image preview.
    assert "ui" in result
    assert "images" in result["ui"]
    assert len(result["ui"]["images"]) == 1


def test_portraits_role_routes_to_portraits_dir(save_node, tmp_path):
    """role_kind="portraits" -> per-episode portraits/ subdir."""
    ep_id = "signal_lost_portraits_test_20260503_010102"
    fake_comfy_out = tmp_path / "ComfyUI_output"

    ep_dir = fake_comfy_out / "otr" / "episodes" / ep_id
    audio_dir = ep_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = audio_dir / f"{ep_id}_ledger.json"
    ledger_path.write_text(
        '{"episode_id": "' + ep_id + '", "schema_version": "l3-test"}',
        encoding="utf-8",
    )

    with mock.patch(
        "nodes._otr_paths.comfy_output_dir",
        return_value=fake_comfy_out,
    ), mock.patch(
        "nodes._otr_ledger.in_flight_ledger_path",
        return_value=ledger_path,
    ):
        save_node.save(
            images=[_fake_image_tensor()],
            role_kind="portraits",
            filename_pattern="char_portrait",
        )

    expected_dir = ep_dir / "portraits"
    saved = list(expected_dir.glob("char_portrait_*.png"))
    assert len(saved) == 1
    # Confirm we did NOT also write to stills/ (would mean role_kind ignored).
    stills_dir = ep_dir / "stills"
    assert not stills_dir.exists() or not list(stills_dir.glob("*.png"))


def test_falls_back_to_legacy_dir_when_no_singleton(save_node, tmp_path):
    """No singleton -> legacy flat dir, no exception, image still saved."""
    fake_comfy_out = tmp_path / "ComfyUI_output"

    with mock.patch(
        "nodes._otr_paths.comfy_output_dir",
        return_value=fake_comfy_out,
    ), mock.patch(
        "nodes._otr_ledger.in_flight_ledger_path",
        return_value=None,
    ):
        result = save_node.save(
            images=[_fake_image_tensor()],
            role_kind="stills",
            filename_pattern="full_env",
        )

    legacy_dir = fake_comfy_out / "otr" / "_legacy_stills"
    saved = list(legacy_dir.glob("full_env_*.png"))
    assert len(saved) == 1, (
        f"fallback path -- expected 1 PNG in _legacy_stills/, got {saved}"
    )
    assert "ui" in result


def test_per_episode_counter_starts_at_1(save_node, tmp_path):
    """Per-episode counter is independent of any global flat-dir counter.

    Stock SaveImage shared a global counter -- that's why
    ``output/otr/stills/full_env_00213.png`` accumulated across episodes.
    The new node's counter starts at 1 in each fresh per-episode stills/
    dir.
    """
    ep_id = "signal_lost_counter_test_20260503_010103"
    fake_comfy_out = tmp_path / "ComfyUI_output"
    ep_dir = fake_comfy_out / "otr" / "episodes" / ep_id
    audio_dir = ep_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = audio_dir / f"{ep_id}_ledger.json"
    ledger_path.write_text(
        '{"episode_id": "' + ep_id + '"}',
        encoding="utf-8",
    )

    with mock.patch(
        "nodes._otr_paths.comfy_output_dir",
        return_value=fake_comfy_out,
    ), mock.patch(
        "nodes._otr_ledger.in_flight_ledger_path",
        return_value=ledger_path,
    ):
        # First save -> counter 1
        save_node.save(
            images=[_fake_image_tensor()],
            role_kind="stills",
            filename_pattern="full_env",
        )
        # Second save -> counter 2
        save_node.save(
            images=[_fake_image_tensor()],
            role_kind="stills",
            filename_pattern="full_env",
        )

    expected_dir = ep_dir / "stills"
    saved = sorted(p.name for p in expected_dir.glob("full_env_*.png"))
    assert saved == ["full_env_00001_.png", "full_env_00002_.png"], (
        f"counter must start at 1 + increment -- got {saved}"
    )


def test_save_never_raises_on_mkdir_failure(save_node, tmp_path):
    """Save must NEVER raise -- a broken save can't crash the workflow."""
    fake_comfy_out = tmp_path / "ComfyUI_output"

    with mock.patch(
        "nodes._otr_paths.comfy_output_dir",
        return_value=fake_comfy_out,
    ), mock.patch(
        "nodes._otr_ledger.in_flight_ledger_path",
        return_value=None,
    ), mock.patch(
        "pathlib.Path.mkdir",
        side_effect=OSError("disk full"),
    ):
        # Must return without raising; UI may be empty.
        result = save_node.save(
            images=[_fake_image_tensor()],
            role_kind="stills",
            filename_pattern="full_env",
        )
    # Either {} or {"ui": {"images": []}} is acceptable -- the contract is
    # "no exception escapes."
    assert isinstance(result, dict)


def test_node_registered_in_init():
    """OTR_SaveToEpisodeWorkspace must be in NODE_CLASS_MAPPINGS."""
    import importlib

    # Import the package via its actual module path.
    sys.path.insert(0, str(_REPO_ROOT.parent))
    try:
        import importlib
        _pkg = importlib.import_module(_REPO_ROOT.name)
        assert "OTR_SaveToEpisodeWorkspace" in _pkg.NODE_CLASS_MAPPINGS, (
            "node not registered in __init__.py"
        )
    finally:
        sys.path.pop(0)
