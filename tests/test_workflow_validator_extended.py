"""S16.2 + S16.3 + S16.5: extended validator surface tests.

S16.2 (IMP-18): check #5 must scan widget.name, node title, and
properties[Node name for S&R] in addition to inp.name.

S16.3 (IMP-14): check #3 must be positional + empty-string aware --
"widgets_values is non-empty" was the old heuristic; that allowed a
node with one filled slot and N blank slots to pass for all N+1
required inputs.

S16.5 (IMP-22, IMP-23): malformed-link threshold is 6 elements
(ComfyUI tuple shape), not 5; duplicate-ID accumulator must dedup so
an ID appearing 3 times reports once.
"""
from __future__ import annotations

import pytest

from nodes._workflow_validation import (
    FORBIDDEN_INPUT_SOCKETS,
    WorkflowInputSocketError,
    WorkflowValidationError,
    WorkflowWidgetDriftError,
    validate_workflow_contract,
)


# -------------------------------------------------------------------
# S16.2: extended forbidden-surface scan
# -------------------------------------------------------------------


def _empty_mappings() -> dict:
    """Mappings dict empty enough that the OTR-type checks skip."""
    return {}


def test_check5_catches_widget_name():
    """A nested widget.name in FORBIDDEN_INPUT_SOCKETS must trip check 5."""
    wf = {
        "nodes": [
            {
                "id": 1,
                "type": "ThirdPartyNode",
                "inputs": [
                    {
                        "name": "script_json",
                        "type": "STRING",
                        "link": 21,
                        "widget": {"name": "production_plan_json"},
                    }
                ],
            }
        ],
        "links": [],
    }
    with pytest.raises(WorkflowInputSocketError) as exc:
        validate_workflow_contract(wf, _empty_mappings())
    assert "widget" in str(exc.value)
    assert "production_plan_json" in str(exc.value)


def test_check5_catches_title():
    """A node.title set to a FORBIDDEN name (exact match) trips check 5.

    Substring-in-title hits (e.g. "Video Plan (wired from Director)")
    are caught by tests/test_workflow_director_freedom.py via regex
    scan; that's a different + broader surface guard. This test
    pins the exact-match contract that the validator promises.
    """
    wf = {
        "nodes": [
            {
                "id": 1,
                "type": "ThirdPartyNode",
                "title": "production_plan_json",
                "inputs": [],
            }
        ],
        "links": [],
    }
    with pytest.raises(WorkflowInputSocketError) as exc:
        validate_workflow_contract(wf, _empty_mappings())
    assert "title" in str(exc.value)


def test_check5_catches_s_and_r():
    """properties[Node name for S&R] is the fourth scanned surface."""
    wf = {
        "nodes": [
            {
                "id": 1,
                "type": "ThirdPartyNode",
                "inputs": [],
                "properties": {"Node name for S&R": "voice_map_json"},
            }
        ],
        "links": [],
    }
    with pytest.raises(WorkflowInputSocketError) as exc:
        validate_workflow_contract(wf, _empty_mappings())
    assert "s_and_r" in str(exc.value)


def test_check5_still_catches_plain_socket_name():
    """Original socket-name detection must still fire (regression)."""
    wf = {
        "nodes": [
            {
                "id": 1,
                "type": "ThirdPartyNode",
                "inputs": [{"name": "production_plan_json", "type": "STRING"}],
            }
        ],
        "links": [],
    }
    with pytest.raises(WorkflowInputSocketError) as exc:
        validate_workflow_contract(wf, _empty_mappings())
    assert "socket" in str(exc.value)


# -------------------------------------------------------------------
# S16.3: positional widget-drift + empty-string detection
# -------------------------------------------------------------------


class _FakeNodeWithTwoRequired:
    """Stub OTR node class with two required string inputs.

    Used so check #3 has something to introspect. Returns
    INPUT_TYPES() declaring both keys in deterministic order.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "alpha_in": ("STRING", {}),
                "beta_in": ("STRING", {}),
            }
        }


def _two_required_mappings() -> dict:
    return {"OTR_TwoRequired": _FakeNodeWithTwoRequired}


def test_check3_empty_string_widget_passes():
    """Empty-string widget value is OK -- INPUT_TYPES.required can
    legitimately declare ``default: ""`` (e.g. episode_title auto-
    derive). S16.3 plan deviation: bare "" is not contract drift.

    Pin this behavior so a future "tighten check 3" change has to
    acknowledge this constraint explicitly.
    """
    wf = {
        "nodes": [
            {"id": 50, "type": "SourceNode", "inputs": []},
            {
                "id": 1,
                "type": "OTR_TwoRequired",
                "inputs": [
                    {"name": "alpha_in", "type": "STRING", "link": 99},
                ],
                # alpha_in is wired; beta_in is unwired; widget slot
                # is "" -- legitimate ComfyUI default-empty case.
                "widgets_values": [""],
            },
        ],
        "links": [[99, 50, 0, 1, 0, "STRING"]],
        "last_link_id": 99,
    }
    # Should NOT raise.
    validate_workflow_contract(wf, _two_required_mappings())


def test_check3_filled_widget_passes():
    """Same shape but widget slot has a real value -- must pass."""
    wf = {
        "nodes": [
            {"id": 50, "type": "SourceNode", "inputs": []},
            {
                "id": 1,
                "type": "OTR_TwoRequired",
                "inputs": [
                    {"name": "alpha_in", "type": "STRING", "link": 99},
                ],
                "widgets_values": ["real_value"],
            },
        ],
        "links": [[99, 50, 0, 1, 0, "STRING"]],
        "last_link_id": 99,
    }
    validate_workflow_contract(wf, _two_required_mappings())


def test_check3_widget_values_exhausted():
    """More unwired required inputs than widget slots must raise."""
    wf = {
        "nodes": [
            {
                "id": 1,
                "type": "OTR_TwoRequired",
                "inputs": [],
                "widgets_values": ["only_one_slot"],
            }
        ],
        "links": [],
    }
    with pytest.raises(WorkflowWidgetDriftError) as exc:
        validate_workflow_contract(wf, _two_required_mappings())
    assert "exhausted" in str(exc.value)


def test_check3_none_widget_value_fails():
    """A None widget value counts as unfilled. None is Comfy's
    "dropped socket" marker -- widget-storage drift, not a legitimate
    default value."""
    wf = {
        "nodes": [
            {"id": 50, "type": "SourceNode", "inputs": []},
            {
                "id": 1,
                "type": "OTR_TwoRequired",
                "inputs": [
                    {"name": "alpha_in", "type": "STRING", "link": 99},
                ],
                "widgets_values": [None],
            },
        ],
        "links": [[99, 50, 0, 1, 0, "STRING"]],
        "last_link_id": 99,
    }
    with pytest.raises(WorkflowWidgetDriftError) as exc:
        validate_workflow_contract(wf, _two_required_mappings())
    assert "beta_in" in str(exc.value)
    assert "explicit None" in str(exc.value)


# -------------------------------------------------------------------
# S16.5: link-tuple threshold + duplicate-ID dedup
# -------------------------------------------------------------------


def test_link_tuple_requires_six_elements():
    """A 5-element link entry must raise (ComfyUI tuple is 6 elements)."""
    wf = {
        "nodes": [{"id": 1, "type": "X", "inputs": []}],
        "links": [[1, 1, 0, 1, 0]],  # 5-element, missing the type tag
        "last_link_id": 1,
    }
    with pytest.raises(WorkflowValidationError) as exc:
        validate_workflow_contract(wf, _empty_mappings())
    assert ">=6" in str(exc.value)


def test_duplicate_link_id_reported_once_per_id():
    """ID 42 appearing 3 times must report as [42], not [42, 42]."""
    wf = {
        "nodes": [{"id": 1, "type": "X", "inputs": []}],
        "links": [
            [42, 1, 0, 1, 0, "STRING"],
            [42, 1, 0, 1, 0, "STRING"],
            [42, 1, 0, 1, 0, "STRING"],
        ],
        "last_link_id": 42,
    }
    with pytest.raises(WorkflowValidationError) as exc:
        validate_workflow_contract(wf, _empty_mappings())
    msg = str(exc.value)
    # Should be [42], not [42, 42] or [42, 42, 42]
    assert "[42]" in msg, f"expected [42] dedup, got: {msg}"


def test_g5_reserved_link_id_still_caught():
    """Existing G5-reserved link test must still pass (regression)."""
    from nodes._workflow_validation import WorkflowReservedLinkIDError
    wf = {
        "nodes": [{"id": 1, "type": "X", "inputs": []}],
        "links": [[111, 1, 0, 1, 0, "STRING"]],
        "last_link_id": 111,
    }
    with pytest.raises(WorkflowReservedLinkIDError):
        validate_workflow_contract(wf, _empty_mappings())


# -------------------------------------------------------------------
# Module-level sanity: FORBIDDEN_INPUT_SOCKETS still has the 5 names
# -------------------------------------------------------------------


def test_forbidden_input_sockets_membership():
    """Sanity pin: the 5 retired socket names are still in the set."""
    expected = {
        "production_plan_json",
        "director_json",
        "voice_map_json",
        "sfx_plan_json",
        "music_plan_json",
    }
    assert expected <= FORBIDDEN_INPUT_SOCKETS
