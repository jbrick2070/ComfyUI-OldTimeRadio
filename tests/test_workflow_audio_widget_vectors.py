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
    """Node-shaped view of the legacy audio engines' FROZEN widget vectors,
    read from config/legacy_invocation_manifest.json.

    Wave 2b replaced the legacy per-engine audio-generator INSTANCES in
    full.json with the v2 audio lane; the lane delegates to those engines using
    the manifest's frozen widget vectors. So the drift these tests guard -- the
    production widget vector vs the class INPUT_TYPES -- now lives in the
    manifest, not the workflow JSON. Each manifest entry is surfaced as a
    ``{type, widgets_values}`` node so the per-node assertions below are
    unchanged (and now pin the exact vector the v2 lane delegates with)."""
    manifest = json.loads(
        (REPO_ROOT / "config" / "legacy_invocation_manifest.json")
        .read_text(encoding="utf-8")
    )
    nodes = [
        {"type": ntype,
         "widgets_values": [w.get("default") for w in entry.get("widgets") or []]}
        for ntype, entry in (manifest.get("nodes") or {}).items()
    ]
    return {"nodes": nodes}


def _find_node(wf: dict, node_type: str) -> dict | None:
    for n in wf.get("nodes") or []:
        if n.get("type") == node_type:
            return n
    return None


_WIDGET_TYPES = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN"})
_COMPANION_INT_WIDGETS = ("seed", "noise_seed")


def _serialized_slot_names(input_types: dict) -> list[str]:
    """Ordered names of the widget slots that actually occupy widgets_values,
    matching ComfyUI's serialization (and scripts/otr_api._serialized_slot_names
    + nodes/_otr_workflow_validator): socket (non-widget) inputs excluded,
    forceInput widgets dropped (they become socket-only, no slot), and an INT
    seed/noise_seed widget gains a hidden control_after_generate companion slot.

    Pre-S14.3 these tests used a naive ``required + optional`` key count, which
    overcounts a forceInput widget (e.g. node 14 OTR_MusicGenTheme's
    ``script_json``) -- the BUG-281 drift. This helper is the correct count.
    """
    names: list[str] = []
    for bucket in ("required", "optional"):
        for name, spec in (input_types.get(bucket) or {}).items():
            t = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
            is_widget = isinstance(t, (list, tuple)) or (
                isinstance(t, str) and t.upper() in _WIDGET_TYPES
            )
            opts = (
                spec[1]
                if isinstance(spec, (list, tuple)) and len(spec) > 1
                and isinstance(spec[1], dict)
                else {}
            )
            if not is_widget or opts.get("forceInput"):
                continue
            names.append(name)
            if isinstance(t, str) and t.upper() == "INT" and name in _COMPANION_INT_WIDGETS:
                names.append(f"{name}__control_after_generate")
    return names


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
    expected_slots = _serialized_slot_names(it)
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
    expected_slots = _serialized_slot_names(it)
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
    expected_slots = _serialized_slot_names(it)
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
    expected_slots = _serialized_slot_names(it)
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
            t = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
            is_widget = isinstance(t, (list, tuple)) or (
                isinstance(t, str) and t.upper() in _WIDGET_TYPES
            )
            opts = (
                spec[1]
                if isinstance(spec, (list, tuple)) and len(spec) > 1
                and isinstance(spec[1], dict)
                else {}
            )
            if not is_widget or opts.get("forceInput"):
                continue  # socket-only / forceInput inputs occupy no slot
            decl.append((name, spec))
            if isinstance(t, str) and t.upper() == "INT" and name in _COMPANION_INT_WIDGETS:
                decl.append((f"{name}__control_after_generate", ("STRING", {})))
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


# -------------------------------------------------------------------
# S25 / MG-12 (BUG-LOCAL-210 production-drift regression pin)
# -------------------------------------------------------------------


def test_pre_cleanbreak_widget_vector_with_stale_empty_dict_fails():
    """S25/MG-12: ensure the BUG-LOCAL-210 production-drift surface
    is pinned with the exact pre-cleanbreak vector, not just the
    synthetic shape fixtures elsewhere in this file.

    History: before C3 the AudioGen widget vector in the canonical
    workflow JSON was a 7-entry list -- ['[]', '{}', '', model_id,
    3.0, 3.0, False]. The {} at position 1 was the leftover default
    value of the deleted REQUIRED `production_plan_json` input.
    ComfyUI silently positionally mapped the remaining 5 values to
    INPUT_TYPES's 5 declared widget slots, shifting model_id ->
    episode_seed, guidance_scale -> model_id, etc.

    The fix (C3, commit f7a5ca0) realigned the vector to 6 entries
    by dropping the {}. This test re-asserts the *shape* check
    against the production drift's exact length-mismatch failure
    mode, complementing the synthetic-vector tests above.
    """
    from nodes.batch_audiogen_generator import BatchAudioGenGenerator
    it = BatchAudioGenGenerator.INPUT_TYPES()
    expected_slots = _serialized_slot_names(it)

    # The exact pre-C3 vector that shipped BUG-LOCAL-210. Length 7.
    pre_cleanbreak_vector = [
        "[]",
        "{}",
        "",
        "facebook/audiogen-medium",
        3.0,
        3.0,
        False,
    ]
    assert len(pre_cleanbreak_vector) != len(expected_slots), (
        "S25/MG-12 regression: the pre-cleanbreak vector length now "
        "matches the current INPUT_TYPES slot count, which means "
        "either INPUT_TYPES grew an entry without the workflow JSON "
        "being updated, or this test fixture is stale. Either way, "
        "audit before clearing."
    )

    # And the production workflow JSON must NOT carry the stale vector.
    wf = _load_workflow()
    node = _find_node(wf, "OTR_BatchAudioGenGenerator")
    assert node is not None
    wv = node.get("widgets_values") or []
    assert wv != pre_cleanbreak_vector, (
        "S25/MG-12 regression: the production workflow JSON's "
        "OTR_BatchAudioGenGenerator widget vector is identical to "
        "the pre-cleanbreak BUG-LOCAL-210 fixture. The C3 fix was "
        "reverted in the JSON."
    )
    # Length must match INPUT_TYPES exactly (the C6 length test
    # above covers this in the general case; this test is the
    # historical-fixture anchor specifically for BUG-LOCAL-210).
    assert len(wv) == len(expected_slots), (
        f"S25/MG-12: widget vector length drift -- got {len(wv)}, "
        f"expected {len(expected_slots)}. BUG-LOCAL-210 drift class."
    )
