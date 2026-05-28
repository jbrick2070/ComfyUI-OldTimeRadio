"""tests/test_bug_local_287_critic_failure_ships.py

BUG-LOCAL-287 (2026-05-27). When the Stage 7 shadow critic CALL
fails (Gemma emits empty output, adapter cannot build, etc.) the
shadow stamps a marker dict on `meta['stage7_shadow_critic']`.
decide_escalation_scope used to see `verdict='' failing_axes=[]`
on this marker and fall through to LINE scope (the legacy reroll
path), which dispatched the Sprint 5B story_critic-driven reroll
loop, which exhausted its 2-cycle budget chasing the same flat
lines, which stamped needs_full_rerun, which halted Bark.

Live evidence: pending_20260527_170250 (Gemma technical slot,
9-beat superconductivity episode) -- exact failure shape above.

The fix routes critic-failure markers to NONE (ship) instead of
LINE. The episode passed Stage 3 line validators and the doctor's
diagnosis; without a Stage 7 signal there's no grounds to keep
regenerating.
"""
from __future__ import annotations

import pytest

from nodes._otr_reroll_escalation import (
    EscalationScope,
    _critic_call_failed,
    decide_escalation_scope,
)


# ---------------------------------------------------------------------------
# _critic_call_failed pin
# ---------------------------------------------------------------------------


class TestCriticCallFailedDetector:
    """The helper that distinguishes real verdicts from failure markers."""

    def test_none_is_not_failure(self):
        # None is handled separately by decide_escalation_scope
        # (returns LINE fallback). The helper itself returns False.
        assert _critic_call_failed(None) is False

    def test_empty_dict_is_not_failure(self):
        # The cascade coerces a missing Stage 7 marker to {} via
        # `meta.get("stage7_shadow_critic") or {}`. Empty dict
        # means Stage 7 was never run (operator did not opt into
        # shadow diagnostics) -- NOT that it ran and failed. The
        # LINE-fallback path handles "no signal" already.
        assert _critic_call_failed({}) is False

    def test_setup_failure_marker_is_failure(self):
        marker = {
            "ok": False,
            "skipped": False,
            "shadow_setup_failed": True,
            "error_type": "ValueError",
            "error_message": "could not build ledger view",
        }
        assert _critic_call_failed(marker) is True

    def test_runtime_failure_marker_is_failure(self):
        # CriticCallFailedError shape -- Gemma empty output.
        marker = {
            "ok": False,
            "skipped": False,
            "shadow_setup_failed": True,
            "error_type": "CriticCallFailedError",
            "error_message": (
                "Whole-episode critic exhausted retry budget after 2 "
                "attempt(s); last error: JSONDecodeError: Expecting "
                "value: line 1 column 1 (char 0)"
            ),
        }
        assert _critic_call_failed(marker) is True

    def test_skipped_marker_is_failure(self):
        marker = {
            "ok": True,
            "skipped": True,
            "reason": "adapter could not build a Stage1Plan view",
        }
        assert _critic_call_failed(marker) is True

    def test_real_ship_verdict_is_not_failure(self):
        # Real CriticResult.to_dict() shape with verdict='ship'.
        result = {
            "verdict": "ship",
            "mean_score": 3.80,
            "failing_axes": [],
            "regeneration_hint": "",
        }
        assert _critic_call_failed(result) is False

    def test_real_discard_verdict_is_not_failure(self):
        result = {
            "verdict": "discard",
            "mean_score": 2.40,
            "failing_axes": ["emotional_arc"],
            "regeneration_hint": "add a second character",
        }
        assert _critic_call_failed(result) is False

    def test_empty_verdict_dict_is_failure(self):
        # Edge: a CriticResult-shaped dict that lost its verdict.
        result = {"verdict": "", "mean_score": 0.0, "failing_axes": []}
        assert _critic_call_failed(result) is True

    def test_dataclass_with_verdict_is_not_failure(self):
        # CriticResult dataclass-like object.
        class _Stub:
            def __init__(self):
                self.verdict = "discard"
                self.failing_axes = ["pacing"]
        assert _critic_call_failed(_Stub()) is False

    def test_dataclass_without_verdict_is_failure(self):
        class _Stub:
            verdict = ""
        assert _critic_call_failed(_Stub()) is True


# ---------------------------------------------------------------------------
# decide_escalation_scope integration pin
# ---------------------------------------------------------------------------


class TestEscalationOnCriticFailure:
    """The escalation decision when Stage 7 is dark."""

    def test_critic_failure_marker_routes_to_none(self):
        """The exact marker shape Stage7Shadow stamps on
        CriticCallFailedError -- routes to NONE (ship)."""
        marker = {
            "ok": False,
            "skipped": False,
            "shadow_setup_failed": True,
            "error_type": "CriticCallFailedError",
            "error_message": (
                "last error: JSONDecodeError: Expecting value: "
                "line 1 column 1 (char 0)"
            ),
        }
        decision = decide_escalation_scope(
            marker,
            story_critic_targets=[
                # Even with 6 story_critic targets pending, the
                # critic-failed route IGNORES them. Without an
                # authoritative Stage 7 signal we have no grounds
                # to escalate.
                {"beat_id": "b002"},
                {"beat_id": "b003"},
                {"beat_id": "b004"},
                {"beat_id": "b006"},
                {"beat_id": "b007"},
                {"beat_id": "b008"},
            ],
        )
        assert decision.scope == EscalationScope.NONE
        assert decision.target_beat_ids == []
        assert "BUG-LOCAL-287" in decision.reason
        assert "ship" in decision.reason.lower()

    def test_setup_failure_marker_routes_to_none(self):
        """The adapter-failed marker also routes to NONE."""
        marker = {
            "ok": True,
            "skipped": True,
            "reason": "adapter could not build a Stage1Plan view",
        }
        decision = decide_escalation_scope(marker)
        assert decision.scope == EscalationScope.NONE

    def test_real_ship_still_routes_to_none(self):
        """The BUG-LOCAL-281 ship gate path is unchanged."""
        result = {
            "verdict": "ship",
            "mean_score": 3.80,
            "failing_axes": [],
        }
        decision = decide_escalation_scope(result)
        assert decision.scope == EscalationScope.NONE
        # The reason differs from the BUG-287 route -- one says
        # 'verdict=ship', the other says 'critic FAILED'.
        assert "verdict=ship" in decision.reason

    def test_real_discard_with_structural_still_routes_to_episode(self):
        """The Wave 1 Agent C structural-failure path is unchanged."""
        result = {
            "verdict": "discard",
            "mean_score": 2.40,
            "failing_axes": ["emotional_arc", "continuity"],
        }
        decision = decide_escalation_scope(result)
        assert decision.scope == EscalationScope.EPISODE

    def test_none_critic_result_still_routes_to_line(self):
        """The pre-BUG-287 None fallback is preserved (covers the
        rare case where the shadow critic ran but stamped literally
        None on meta -- still safer than NONE)."""
        decision = decide_escalation_scope(
            None,
            story_critic_targets=[{"beat_id": "b002"}],
        )
        assert decision.scope == EscalationScope.LINE
        assert decision.target_beat_ids == ["b002"]

    def test_empty_dict_critic_result_still_routes_to_line(self):
        """Stage 7 not enabled (cascade coerces missing key to {}) ->
        line fallback per the pre-fix behavior. Only an explicit
        failure marker triggers the new ship route."""
        decision = decide_escalation_scope(
            {},
            story_critic_targets=[{"beat_id": "b002"}],
        )
        assert decision.scope == EscalationScope.LINE
        assert decision.target_beat_ids == ["b002"]
