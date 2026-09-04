"""An OTR node whose INPUT_TYPES() raises must FAIL the widget-drift gate.

WHY THIS FILE EXISTS (kibitz r1, 2026-09-04, unanimous across three lanes).
`widget_vector_drift` backs the S14.3 HARD GATE that halts a run at queue time
when a node's saved `widgets_values` no longer matches its INPUT_TYPES slot
count -- the BUG-210 / 253 / 281 silent-widget-shift class. Its per-node loop
used to read:

    try:
        expected = _expected_slot_count(cls.INPUT_TYPES() or {})
    except Exception:
        continue

so any node whose `INPUT_TYPES()` raised was silently REMOVED from the check,
and the gate could report `widget_vector_drift=0` having never looked at it.
`tests/test_workflow_graph_integrity_guards.py` notes the raise branch had no
coverage -- this is that coverage.

The driver's first instinct was "fail only if the node is ours, a third-party
import quirk should not refuse a render". That was a misread: third-party types
are already skipped at `cls is None` before the try, so every class reaching the
hatch is an OTR class. There is no foreign-node cost, and a pack node that
cannot describe its own inputs is precisely what the gate is for.
"""
import pytest

from nodes import _otr_workflow_validator as V


class _Healthy:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"a": ("INT", {}), "b": ("STRING", {})}}


class _Broken:
    @classmethod
    def INPUT_TYPES(cls):
        raise RuntimeError("schema exploded")


def _graph(*nodes):
    return {"nodes": [{"id": i + 1, "type": t, "widgets_values": wv}
                      for i, (t, wv) in enumerate(nodes)]}


def test_a_node_whose_input_types_raises_is_a_finding_not_an_exemption():
    ncm = {"OTR_Broken": _Broken}
    findings = V.widget_vector_drift(_graph(("OTR_Broken", [1, 2])), ncm)
    assert len(findings) == 1, findings
    f = findings[0]
    assert "OTR_Broken" in f
    assert "INPUT_TYPES() raised" in f
    assert "RuntimeError" in f and "schema exploded" in f
    assert "treated as drift" in f


def test_the_finding_does_not_hide_a_real_drift_on_a_sibling():
    """One broken node must not swallow the loop: a genuine mismatch on the
    next node is still reported alongside it."""
    ncm = {"OTR_Broken": _Broken, "OTR_Healthy": _Healthy}
    graph = _graph(("OTR_Broken", [1]), ("OTR_Healthy", [1, 2, 3]))  # 3 != 2
    findings = V.widget_vector_drift(graph, ncm)
    assert len(findings) == 2, findings
    assert any("OTR_Broken" in f for f in findings)
    assert any("OTR_Healthy" in f and "widgets_values=3 != expected 2" in f
               for f in findings)


def test_third_party_nodes_are_still_skipped_not_reported():
    """The `cls is None` skip above the hatch is the ONLY place a non-OTR type
    leaves the loop, and it must stay that way -- the fix must not start
    reporting drift on nodes the pack does not own."""
    ncm = {"OTR_Healthy": _Healthy}
    graph = _graph(("SomeoneElsesNode", [9, 9, 9, 9]), ("OTR_Healthy", [1, 2]))
    assert V.widget_vector_drift(graph, ncm) == []


def test_a_healthy_graph_still_reports_nothing():
    ncm = {"OTR_Healthy": _Healthy}
    assert V.widget_vector_drift(_graph(("OTR_Healthy", [1, 2])), ncm) == []


def test_the_function_still_never_raises():
    """Its docstring promises 'Pure; never raises' and the hard gate relies on
    getting a LIST back. A raising INPUT_TYPES must surface as a finding, never
    as an exception out of this function."""
    ncm = {"OTR_Broken": _Broken}
    out = V.widget_vector_drift(_graph(("OTR_Broken", [1])), ncm)
    assert isinstance(out, list)


# --------------------------------------------------------------------------- #
# The PUBLIC path (kibitz r2, Codex must-fix 7): a helper that returns text
# does not prove the execution gate halts. Drive the node's own validate().
# --------------------------------------------------------------------------- #
def _mapping_the_validator_will_scan(monkeypatch):
    """The SAME lookup ``WorkflowValidator.validate()`` performs outside the
    ComfyUI process: ``from .. import NODE_CLASS_MAPPINGS`` cannot resolve for
    a top-level ``nodes`` package, so it scans ``sys.modules`` for the first
    OTR_-keyed mapping. Replicated here so the class we break is the class the
    validator will actually consult, whichever package object this session
    loaded first; an empty session gets a stub mapping registered."""
    import sys
    import types
    for mod in list(sys.modules.values()):
        cand = getattr(mod, "NODE_CLASS_MAPPINGS", None)
        if isinstance(cand, dict) and any(
                isinstance(k, str) and k.startswith("OTR_") for k in cand):
            return cand
    stub = types.ModuleType("_otr_drift_gate_stub")
    stub.NODE_CLASS_MAPPINGS = {"OTR_DriftGateStub": _Healthy}
    monkeypatch.setitem(sys.modules, "_otr_drift_gate_stub", stub)
    return stub.NODE_CLASS_MAPPINGS


def _one_node_graph(tmp_path, ntype):
    import json
    p = tmp_path / "one_node.json"
    p.write_text(json.dumps({
        "last_node_id": 1, "last_link_id": 0, "links": [],
        "nodes": [{"id": 1, "type": ntype, "inputs": [], "outputs": [],
                   "widgets_values": []}],
    }), encoding="utf-8")
    return str(p)


def test_the_public_validate_path_halts_on_a_raising_schema(tmp_path,
                                                            monkeypatch):
    """Through the node's own ``validate()`` the run HALTS. Which check fires
    is worth knowing: ``validate_workflow_contract``
    (``_workflow_validation.py:233``) raises on a raising INPUT_TYPES() for
    OTR_-prefixed types BEFORE the widget-vector gate runs, so the public path
    was already fail-closed. The hatch this file guards mattered for every
    STANDALONE caller of ``widget_vector_drift`` (the canonical-graph
    integrity test) and for any mapped class without the OTR_ prefix. Either
    way the contract is the same: no silent pass."""
    from nodes._otr_workflow_validator import WorkflowValidator
    ncm = _mapping_the_validator_will_scan(monkeypatch)
    monkeypatch.setitem(ncm, "OTR_Broken", _Broken)
    path = _one_node_graph(tmp_path, "OTR_Broken")
    with pytest.raises(Exception, match=r"INPUT_TYPES\(\) raised"):
        WorkflowValidator().validate(
            workflow_json_path=path, validate_anyway=True,
            strict_unknown_types=False)


def test_validate_anyway_false_is_the_only_bypass(tmp_path, monkeypatch):
    """The documented diagnostic bypass, and nothing else, skips the halt."""
    from nodes._otr_workflow_validator import WorkflowValidator
    ncm = _mapping_the_validator_will_scan(monkeypatch)
    monkeypatch.setitem(ncm, "OTR_Broken", _Broken)
    path = _one_node_graph(tmp_path, "OTR_Broken")
    (msg,) = WorkflowValidator().validate(
        workflow_json_path=path, validate_anyway=False,
        strict_unknown_types=False)
    assert "validate_anyway=False" in msg and "skipped" in msg
