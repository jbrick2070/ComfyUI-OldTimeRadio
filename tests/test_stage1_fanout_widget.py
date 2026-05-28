"""Sprint 4.5 (2026-05-28) -- use_stage1_fanout widget + plumbing.

Pinned invariants:
  * OTR_LedgerScriptWriter.INPUT_TYPES exposes a
    use_stage1_fanout BOOLEAN widget defaulting to False.
  * _resolve_inputs accepts the kwarg and threads onto the
    resolved dict.
  * The signature carries the kwarg through run() with default
    False (back-compat: no fan-out unless explicitly opted in).

The actual diagnostic pass requires the LLM + shadow pass
infrastructure and lives in the operator's 5-episode soak; the
helper itself is covered by tests/test_stage1_fanout_and_select.py.
This file pins the WIRING surface so the soak gate can rely on
the widget being present, named, and threaded.
"""
from __future__ import annotations

import inspect

from nodes.OTR_LedgerScriptWriter import (
    OTR_LedgerScriptWriter,
    _resolve_inputs,
)


def test_input_types_carry_use_stage1_fanout_widget():
    schema = OTR_LedgerScriptWriter.INPUT_TYPES()
    opt = schema.get("optional", {}) or {}
    assert "use_stage1_fanout" in opt
    widget_def = opt["use_stage1_fanout"]
    assert widget_def[0] == "BOOLEAN"
    # Default MUST be False -- the diagnostic pass triples the
    # Stage 1 cost and should only fire on explicit opt-in.
    assert widget_def[1]["default"] is False


def test_resolve_inputs_default_threads_false():
    resolved = _resolve_inputs(
        custom_premise="placeholder premise for unit test only",
        style_custom="solo-test",
    )
    assert resolved.get("use_stage1_fanout") is False


def test_resolve_inputs_threads_true_when_opted_in():
    resolved = _resolve_inputs(
        custom_premise="placeholder premise for unit test only",
        style_custom="solo-test",
        use_stage1_fanout=True,
    )
    assert resolved.get("use_stage1_fanout") is True


def test_run_signature_carries_kwarg_default_false():
    sig = inspect.signature(OTR_LedgerScriptWriter.run)
    assert "use_stage1_fanout" in sig.parameters
    assert sig.parameters["use_stage1_fanout"].default is False
