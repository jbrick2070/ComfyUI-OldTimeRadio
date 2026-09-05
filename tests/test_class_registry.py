"""Tests for the v2 audio/casting class registry (piece 6)."""
import pathlib
import re

from nodes._otr_class_registry import NEW_AUDIO_NODES, NEW_NODE_SPECS, V2_AUDIO_CATEGORY, new_node_modules_table

_PLACEHOLDER_MARKERS = ("[EMOJI]", "[TODO]", "[PLACEHOLDER]", "[FIXME]")
_EXPECTED_KEYS = {
    "OTR_BatchCharacterVoices",
    "OTR_AnnouncerVoice",
    "OTR_StableAudioTheme",
    "OTR_CastLock",
}


def test_expected_keys_present():
    assert set(NEW_NODE_SPECS) == _EXPECTED_KEYS


def test_keys_have_otr_prefix_and_are_unique():
    keys = [s.key for s in NEW_AUDIO_NODES]
    assert len(keys) == len(set(keys)), "duplicate keys"
    for k in keys:
        assert k.startswith("OTR_"), k


def test_display_names_lead_with_space_and_no_placeholder():
    for s in NEW_AUDIO_NODES:
        assert s.display_name.startswith(" "), s.key
        assert s.display_name.strip(), s.key
        for marker in _PLACEHOLDER_MARKERS:
            assert marker not in s.display_name, (s.key, marker)


def test_module_paths_and_class_names_well_formed():
    for s in NEW_AUDIO_NODES:
        assert s.module_path.startswith(".nodes."), s.key
        assert s.class_name.isidentifier(), s.key
        assert not s.class_name.startswith("OTR_"), (
            f"{s.key}: audio node class names are bare (BatchBarkGenerator-style)"
        )


def test_modules_table_shape_matches_init_convention():
    table = new_node_modules_table()
    assert set(table) == _EXPECTED_KEYS
    for key, value in table.items():
        assert isinstance(value, tuple) and len(value) == 3, key
        module_path, class_name, display_name = value
        assert module_path == NEW_NODE_SPECS[key].module_path
        assert class_name == NEW_NODE_SPECS[key].class_name
        assert display_name == NEW_NODE_SPECS[key].display_name


def test_strict_types_cli_includes_dynamic_registry_keys():
    """The link-validator CLI resolves NODE_CLASS_MAPPINGS keys via a pure AST
    parse (no ComfyUI / torch import). Nodes 80-83 are merged dynamically from
    this registry (not a dict literal in __init__.py), so the CLI must union
    them or ``--strict-types`` false-flags them as missing (2026-07-04
    widget-audit standalone fix)."""
    from tools.validate_workflow_links import (
        load_node_class_mappings,
        load_registry_keys,
    )

    assert load_registry_keys() == _EXPECTED_KEYS
    assert _EXPECTED_KEYS <= load_node_class_mappings()


def test_new_keys_do_not_collide_with_existing_literal_registrations():
    init_path = pathlib.Path(__file__).resolve().parent.parent / "__init__.py"
    text = init_path.read_text(encoding="utf-8")
    existing_literal_keys = set(re.findall(r'"(OTR_\w+)"\s*:', text))
    assert _EXPECTED_KEYS.isdisjoint(existing_literal_keys), (
        "new node keys collide with existing literal _NODE_MODULES keys: "
        f"{_EXPECTED_KEYS & existing_literal_keys}"
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
