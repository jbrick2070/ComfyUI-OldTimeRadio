"""Sprint 2.2 (2026-05-28) -- writer halt on news-brief exhaustion.

Pinned invariants:
  * OTR_LedgerScriptWriter.INPUT_TYPES exposes the
    `news_briefs_required` widget defaulting to True.
  * _resolve_inputs propagates the flag onto the resolved dict.
  * Default behaviour (no operator override) is HALT on news
    exhaustion, per Jeffrey 2026-05-27.

The actual halt path requires loading the LLM + running
build_news_briefs, which is out of scope for unit tests; the
behavioural check lives in the operator's 5-episode soak.
"""
from __future__ import annotations

from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter, _resolve_inputs


def test_input_types_carry_news_briefs_required_widget():
    schema = OTR_LedgerScriptWriter.INPUT_TYPES()
    opt = schema.get("optional", {}) or {}
    assert "news_briefs_required" in opt
    widget_def = opt["news_briefs_required"]
    assert widget_def[0] == "BOOLEAN"
    # Default MUST be True per Jeffrey 2026-05-27 directive.
    assert widget_def[1]["default"] is True


def test_resolve_inputs_default_threads_true():
    """When the caller does not pass the kwarg, the default fires
    and the resolved dict carries True (production behaviour)."""
    resolved = _resolve_inputs(
        custom_premise="placeholder premise for unit test only",
        style_custom="solo-test",
    )
    assert resolved.get("news_briefs_required") is True


def test_resolve_inputs_threads_false_when_overridden():
    """Operator opt-out (graceful-degrade back-compat)."""
    resolved = _resolve_inputs(
        custom_premise="placeholder premise for unit test only",
        style_custom="solo-test",
        news_briefs_required=False,
    )
    assert resolved.get("news_briefs_required") is False
