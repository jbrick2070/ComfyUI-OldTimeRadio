"""S14.1 -- Workflow contract validator tests (commit A: test-only).

Pins the six checks in ``_workflow_validation.validate_workflow_contract``
plus the canonical CI test that the production workflow validates
clean. Per the S10-S15 plan's S14.1 spec.

Auto-invoke on workflow load (S14.2 / commit B) does NOT land in
this commit -- the plan's Q-D10 vote chose a one-week false-positive
observation window before unconditional auto-validation goes live.
"""
from __future__ import annotations

import copy
import json
import pathlib

import pytest

from nodes._workflow_validation import (
    DELETED_NODE_TYPES,
    FORBIDDEN_INPUT_SOCKETS,
    G5_RESERVED_LINK_IDS,
    WorkflowDeletedNodeError,
    WorkflowInputSocketError,
    WorkflowReservedLinkIDError,
    WorkflowUnknownNodeTypeError,
    WorkflowValidationError,
    WorkflowWidgetDriftError,
    validate_workflow_contract,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CANONICAL_WORKFLOW = REPO_ROOT / "workflows" / "otr_canonical.json"


def _load_canonical_workflow() -> dict:
    return json.loads(CANONICAL_WORKFLOW.read_text(encoding="utf-8"))


def _otr_node_class_mappings() -> dict:
    """Build a NODE_CLASS_MAPPINGS dict by parsing
    ``__init__.py``'s ``_NODE_MODULES`` registration table and
    importing each referenced class by qualified module name.

    Tests run with the repo dir on sys.path so ``nodes`` and
    ``visual`` are importable as top-level packages -- meaning a
    leading-dot relative path from ``__init__.py`` like
    ``.nodes.scene_sequencer`` resolves to ``nodes.scene_sequencer``
    via importlib.

    Modules that fail to import (missing optional deps, etc.) are
    skipped silently -- the validator's check #1 (unknown OTR
    type) raises only if the type is missing from this mapping;
    for test purposes we want every legitimate type registered.
    """
    import ast
    import importlib

    init_src = (REPO_ROOT / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(init_src)

    nm_dict_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "_NODE_MODULES":
                    nm_dict_node = node.value
                    break
        if nm_dict_node is not None:
            break
    if not isinstance(nm_dict_node, ast.Dict):
        return {}

    merged: dict = {}
    for k_node, v_node in zip(nm_dict_node.keys, nm_dict_node.values):
        if not isinstance(k_node, ast.Constant) or not isinstance(k_node.value, str):
            continue
        if not isinstance(v_node, ast.Tuple) or len(v_node.elts) < 2:
            continue
        type_name = k_node.value
        mod_path_node = v_node.elts[0]
        cls_name_node = v_node.elts[1]
        if not isinstance(mod_path_node, ast.Constant) or not isinstance(cls_name_node, ast.Constant):
            continue
        mod_path = mod_path_node.value      # ".nodes.scene_sequencer"
        cls_name = cls_name_node.value      # "SceneSequencer"
        if not isinstance(mod_path, str) or not mod_path.startswith("."):
            continue
        qualified = mod_path[1:]            # "nodes.scene_sequencer"
        try:
            mod = importlib.import_module(qualified)
        except Exception:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is not None:
            merged[type_name] = cls
    return merged


# ---------------------------------------------------------------------------
# Canonical workflow CI test (the headline gate)
# ---------------------------------------------------------------------------


def test_canonical_workflow_passes_contract_validator():
    """The shipping production workflow MUST validate clean. If this
    fails, CI fails, and the workflow needs a fix before merge."""
    wf = _load_canonical_workflow()
    mappings = _otr_node_class_mappings()
    # Should NOT raise.
    validate_workflow_contract(wf, mappings)


# ---------------------------------------------------------------------------
# Per-exception-class adversarial tests (one focused failure per check)
# ---------------------------------------------------------------------------


def test_unknown_otr_node_type_raises():
    """Adding a node with an OTR_-prefixed type that isn't registered
    raises WorkflowUnknownNodeTypeError when ``strict_unknown_types``
    is opt-in. The validator's default mode (``strict_unknown_types=False``)
    is permissive so test environments with partial registry loading
    can still validate the rest of the contract."""
    wf = _load_canonical_workflow()
    wf["nodes"].append({
        "id": 9999,
        "type": "OTR_NotRegisteredEverWherever",
        "inputs": [],
        "outputs": [],
        "widgets_values": [],
    })
    with pytest.raises(WorkflowUnknownNodeTypeError):
        validate_workflow_contract(
            wf, _otr_node_class_mappings(),
            strict_unknown_types=True,
        )


def test_unknown_otr_node_type_passes_in_default_mode():
    """Default mode (strict_unknown_types=False) skips the unknown-
    type check so missing OTR_* types in the registry don't blow
    up the rest of the validator."""
    wf = _load_canonical_workflow()
    wf["nodes"].append({
        "id": 9988,
        "type": "OTR_NotRegisteredAtAll",
        "inputs": [],
        "outputs": [],
        "widgets_values": [],
    })
    # Should NOT raise.
    validate_workflow_contract(wf, _otr_node_class_mappings())


def test_deleted_node_type_raises():
    """Adding a node with a type in DELETED_NODE_TYPES raises
    WorkflowDeletedNodeError -- regardless of whether the type
    happens to be in NODE_CLASS_MAPPINGS or not."""
    deleted = next(iter(DELETED_NODE_TYPES))
    wf = _load_canonical_workflow()
    wf["nodes"].append({
        "id": 9998,
        "type": deleted,
        "inputs": [],
        "outputs": [],
        "widgets_values": [],
    })
    with pytest.raises(WorkflowDeletedNodeError) as excinfo:
        validate_workflow_contract(wf, _otr_node_class_mappings())
    assert deleted in str(excinfo.value)


def test_forbidden_input_socket_raises():
    """A node carrying a FORBIDDEN_INPUT_SOCKETS name in its inputs[]
    list raises WorkflowInputSocketError -- catches stale Director-
    derived wirings."""
    forbidden = next(iter(FORBIDDEN_INPUT_SOCKETS))
    wf = _load_canonical_workflow()
    # Find a node we own (OTR_-prefixed) and inject a forbidden input.
    target = next(n for n in wf["nodes"] if n["type"].startswith("OTR_"))
    target = copy.deepcopy(target)
    target["id"] = 9997
    target["inputs"] = list(target.get("inputs") or []) + [
        {"name": forbidden, "type": "STRING", "link": None},
    ]
    wf["nodes"].append(target)
    with pytest.raises(WorkflowInputSocketError) as excinfo:
        validate_workflow_contract(wf, _otr_node_class_mappings())
    assert forbidden in str(excinfo.value)


def test_reserved_link_id_raises():
    """Adding a link with an ID in G5_RESERVED_LINK_IDS raises
    WorkflowReservedLinkIDError."""
    reserved = next(iter(G5_RESERVED_LINK_IDS))
    wf = _load_canonical_workflow()
    # Pick two existing nodes for the link's src/dst.
    n0 = wf["nodes"][0]["id"]
    n1 = wf["nodes"][1]["id"]
    wf["links"] = list(wf["links"]) + [
        [reserved, n0, 0, n1, 0, "STRING"],
    ]
    # last_link_id must be updated so the max-equality check doesn't
    # fire first and mask the reserved-id check.
    wf["last_link_id"] = max(L[0] for L in wf["links"])
    with pytest.raises(WorkflowReservedLinkIDError):
        validate_workflow_contract(wf, _otr_node_class_mappings())


def test_duplicate_link_id_raises():
    """A workflow with two links sharing the same ID raises
    WorkflowValidationError (not a typed subclass -- structural
    integrity error)."""
    wf = _load_canonical_workflow()
    # Take an existing link, append a duplicate of it.
    existing = wf["links"][0]
    wf["links"] = list(wf["links"]) + [list(existing)]
    with pytest.raises(WorkflowValidationError) as excinfo:
        validate_workflow_contract(wf, _otr_node_class_mappings())
    assert "Duplicate link" in str(excinfo.value)


def test_last_link_id_mismatch_raises():
    """Tampering with last_link_id without updating links (or vice
    versa) raises WorkflowValidationError."""
    wf = _load_canonical_workflow()
    wf["last_link_id"] = max(L[0] for L in wf["links"]) + 100
    with pytest.raises(WorkflowValidationError) as excinfo:
        validate_workflow_contract(wf, _otr_node_class_mappings())
    assert "last_link_id" in str(excinfo.value)


def test_orphan_link_raises():
    """A link whose src or dst node ID isn't present in nodes[]
    raises WorkflowValidationError."""
    wf = _load_canonical_workflow()
    # Pick an existing link, change its dst to a node ID that
    # definitely isn't in the workflow.
    new_link_id = max(L[0] for L in wf["links"]) + 1
    wf["links"] = list(wf["links"]) + [
        [new_link_id, wf["nodes"][0]["id"], 0, 999999, 0, "STRING"],
    ]
    wf["last_link_id"] = new_link_id
    with pytest.raises(WorkflowValidationError) as excinfo:
        validate_workflow_contract(wf, _otr_node_class_mappings())
    assert "Orphan" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Exception-hierarchy guards (Q-D9 vote pin)
# ---------------------------------------------------------------------------


def test_all_typed_exceptions_subclass_workflow_validation_error():
    """Q-D9 vote: WorkflowValidationError root + 5 typed children.
    Catching the root MUST catch every typed child."""
    children = (
        WorkflowReservedLinkIDError,
        WorkflowInputSocketError,
        WorkflowWidgetDriftError,
        WorkflowDeletedNodeError,
        WorkflowUnknownNodeTypeError,
    )
    for child in children:
        assert issubclass(child, WorkflowValidationError), (
            f"{child.__name__} must subclass WorkflowValidationError "
            f"per Q-D9 vote."
        )


def test_workflow_validation_error_is_value_error_subclass():
    """Q-D9 vote: WorkflowValidationError subclasses ValueError so
    existing exception handlers that catch ValueError still see
    these errors (no new-root surprise behavior)."""
    assert issubclass(WorkflowValidationError, ValueError)


# ---------------------------------------------------------------------------
# Reserved-set drift guards
# ---------------------------------------------------------------------------


def test_g5_reserved_link_ids_set_pinned():
    """The reserved set is intentionally small (just the FreezeCascade
    fanout IDs). Drift guard: if a future change adds or removes IDs
    from this set, the test fires and forces the author to update
    this pin in lockstep with the cascade's link assignments."""
    assert G5_RESERVED_LINK_IDS == frozenset({111, 112})


def test_deleted_node_types_includes_director():
    """The deleted Director class is the most prominent retirement;
    pin its presence so a future DELETED_NODE_TYPES rewrite doesn't
    accidentally drop the most important entry."""
    assert "OTR_LLMDirector" in DELETED_NODE_TYPES


def test_forbidden_input_sockets_includes_director_json():
    """director_json was the alternate-name socket the deleted
    LLMDirector emitted. Ensure the forbidden set still flags it."""
    assert "director_json" in FORBIDDEN_INPUT_SOCKETS
