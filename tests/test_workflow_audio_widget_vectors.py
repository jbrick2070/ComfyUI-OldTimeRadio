"""C6 (S24, 2026-05-13) -- AudioGen + MusicGen widget-vector pins.

Reflects each class's INPUT_TYPES() to compute the expected widget
shape, then asserts the production workflow JSON's widgets_values
arrays match in length AND that allow_silence_fallback is pinned
False (the strict-failure default after C2 + C3).

Catches the class of bug the C3 widget realignment hit: stale
widgets from a deleted required input shifting every subsequent
slot. If a future cleanbreak deletes another required input without
shrinking the JSON widget vector, this test fires before the
runtime mis-mapping ships.
"""
from __future__ import annotations

import json
import pathlib

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CANONICAL_WORKFLOW = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"


def _load_workflow() -> dict:
    return json.loads(CANONICAL_WORKFLOW.read_text(encoding="utf-8"))


def _find_node(wf: dict, node_type: str) -> dict | None:
    for n in wf.get("nodes") or []:
        if n.get("type") == node_type:
            return n
    return None


# -------------------------------------------------------------------
# AudioGen
# -------------------------------------------------------------------


def test_audiogen_widget_vector_length_matches_input_types():
    """The AudioGen node's widgets_values length must equal the
    declared widget slot count (required + optional in INPUT_TYPES
    order). Pre-C3 the vector carried a stale '{}' from the deleted
    production_plan_json input; this test catches that drift."""
    from nodes.batch_audiogen_generator import BatchAudioGenGenerator
    it = BatchAudioGenGenerator.INPUT_TYPES()
    expected_slots = (
        list((it.get("required") or {}).keys())
        + list((it.get("optional") or {}).keys())
    )
    wf = _load_workflow()
    node = _find_node(wf, "OTR_BatchAudioGenGenerator")
    assert node is not None, (
        "C6: OTR_BatchAudioGenGenerator missing from workflow JSON."
    )
    wv = node.get("widgets_values") or []
    assert len(wv) == len(expected_slots), (
        f"C6: AudioGen widgets_values length drift. Expected "
        f"{len(expected_slots)} (per INPUT_TYPES "
        f"{expected_slots!r}); got {len(wv)} values: {wv!r}. "
        f"A required input was likely deleted without shrinking "
        f"the JSON widget vector."
    )


def test_audiogen_allow_silence_fallback_pinned_false():
    """The allow_silence_fallback widget must be pinned to False so
    production never silently substitutes silence on a transformers
    ImportError (S17.2 strict-default contract)."""
    from nodes.batch_audiogen_generator import BatchAudioGenGenerator
    it = BatchAudioGenGenerator.INPUT_TYPES()
    expected_slots = (
        list((it.get("required") or {}).keys())
        + list((it.get("optional") or {}).keys())
    )
    idx = expected_slots.index("allow_silence_fallback")
    wf = _load_workflow()
    node = _find_node(wf, "OTR_BatchAudioGenGenerator")
    wv = node.get("widgets_values") or []
    assert wv[idx] is False, (
        f"C6: AudioGen allow_silence_fallback at widget index {idx} "
        f"must be False; got {wv[idx]!r}. Pinning True would "
        f"silently swallow transformers ImportError as silence in "
        f"production -- Directive 1 breach."
    )


# -------------------------------------------------------------------
# MusicGen
# -------------------------------------------------------------------


def test_musicgen_widget_vector_length_matches_input_types():
    """The MusicGen node's widgets_values length must equal the
    declared widget slot count."""
    from nodes.musicgen_theme import MusicGenTheme
    it = MusicGenTheme.INPUT_TYPES()
    expected_slots = (
        list((it.get("required") or {}).keys())
        + list((it.get("optional") or {}).keys())
    )
    wf = _load_workflow()
    node = _find_node(wf, "OTR_MusicGenTheme")
    assert node is not None, (
        "C6: OTR_MusicGenTheme missing from workflow JSON."
    )
    wv = node.get("widgets_values") or []
    assert len(wv) == len(expected_slots), (
        f"C6: MusicGen widgets_values length drift. Expected "
        f"{len(expected_slots)} (per INPUT_TYPES "
        f"{expected_slots!r}); got {len(wv)} values: {wv!r}."
    )


def test_musicgen_allow_silence_fallback_pinned_false():
    """Same strict-default pin for MusicGen (C3 / S24)."""
    from nodes.musicgen_theme import MusicGenTheme
    it = MusicGenTheme.INPUT_TYPES()
    expected_slots = (
        list((it.get("required") or {}).keys())
        + list((it.get("optional") or {}).keys())
    )
    idx = expected_slots.index("allow_silence_fallback")
    wf = _load_workflow()
    node = _find_node(wf, "OTR_MusicGenTheme")
    wv = node.get("widgets_values") or []
    assert wv[idx] is False, (
        f"C6: MusicGen allow_silence_fallback at widget index {idx} "
        f"must be False; got {wv[idx]!r}."
    )


# -------------------------------------------------------------------
# No stale '{}' / '[]' residue at unexpected positions
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "node_type, klass_module, klass_name",
    [
        ("OTR_BatchAudioGenGenerator",
         "nodes.batch_audiogen_generator", "BatchAudioGenGenerator"),
        ("OTR_MusicGenTheme",
         "nodes.musicgen_theme", "MusicGenTheme"),
    ],
)
def test_no_stale_dict_residue_in_widget_vector(
    node_type, klass_module, klass_name
):
    """Each widget value must be of the type the corresponding
    INPUT_TYPES declares -- catches the C3 drift where a deleted
    required input left a stale '{}' at position 1 shifting every
    subsequent slot.

    This is a shape-check, not a value-check: STRING widgets accept
    any string; INT widgets accept any int; BOOLEAN widgets accept
    bool. A stale '{}' in a position the schema declared INT would
    fire here.
    """
    import importlib
    klass = getattr(importlib.import_module(klass_module), klass_name)
    it = klass.INPUT_TYPES()
    decl = []
    for bucket in ("required", "optional"):
        for name, spec in (it.get(bucket) or {}).items():
            decl.append((name, spec))
    wf = _load_workflow()
    node = _find_node(wf, node_type)
    wv = node.get("widgets_values") or []
    for i, (name, spec) in enumerate(decl):
        if i >= len(wv):
            break
        type_decl = spec[0]
        val = wv[i]
        if type_decl == "STRING":
            assert isinstance(val, str), (
                f"{node_type} widget[{i}] ({name}) is type {type_decl!r} "
                f"but value is {val!r} ({type(val).__name__}). "
                f"Stale-residue drift suspected."
            )
        elif type_decl == "INT":
            assert isinstance(val, int) and not isinstance(val, bool), (
                f"{node_type} widget[{i}] ({name}) is type {type_decl!r} "
                f"but value is {val!r} ({type(val).__name__})."
            )
        elif type_decl == "FLOAT":
            assert isinstance(val, (int, float)) and not isinstance(val, bool), (
                f"{node_type} widget[{i}] ({name}) is type {type_decl!r} "
                f"but value is {val!r} ({type(val).__name__})."
            )
        elif type_decl == "BOOLEAN":
            assert isinstance(val, bool), (
                f"{node_type} widget[{i}] ({name}) is type {type_decl!r} "
                f"but value is {val!r} ({type(val).__name__}). "
                f"This is exactly the C3 drift class."
            )
        elif isinstance(type_decl, list):
            # Enum / dropdown -- value must be a string in the list,
            # but we accept any string (ComfyUI tolerates this).
            assert isinstance(val, str), (
                f"{node_type} widget[{i}] ({name}) is enum "
                f"({type_decl!r}) but value is {val!r}."
            )
