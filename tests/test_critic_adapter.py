"""T2 critic-axes adapter + keep-best + escalation gate (2026-06-23).

Pure / CPU. The adapter bridges the real StoryCriticReport fields
(arc_verdict / reroll_targets / continuity_issues -- NOT failing_axes /
regeneration_hint) into the escalation router's vocabulary, gated behind
OTR_ENABLE_CRITIC_ESCALATION (default OFF => byte-identical). UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_story_select as SEL  # noqa: E402


@dataclass
class _FakeTarget:
    hint: str = ""


@dataclass
class _FakeReport:
    arc_verdict: str = "strong"
    reroll_targets: List[Any] = field(default_factory=list)
    continuity_issues: List[Any] = field(default_factory=list)


class TestAdapter:
    def test_strong_no_failure(self):
        axes, hint = SEL.critic_report_to_refine_signals(_FakeReport("strong"))
        assert axes == [] and hint == ""

    def test_flat_maps_to_emotional_arc(self):
        axes, hint = SEL.critic_report_to_refine_signals(_FakeReport("flat"))
        assert axes == ["emotional_arc"]
        assert "arc" in hint.lower()

    def test_mid_collapse_maps_to_resolution_and_arc(self):
        axes, _ = SEL.critic_report_to_refine_signals(_FakeReport("mid_collapse"))
        assert axes == ["resolution", "emotional_arc"]

    def test_continuity_issues_add_continuity_axis(self):
        rep = _FakeReport("uneven", continuity_issues=[{"line_id": "b1", "issue": "x"}])
        axes, _ = SEL.critic_report_to_refine_signals(rep)
        assert "continuity" in axes and "emotional_arc" in axes

    def test_reroll_hints_become_regeneration_hint(self):
        rep = _FakeReport(
            "flat",
            reroll_targets=[_FakeTarget("raise the personal stake"),
                            _FakeTarget("land the choice on-stage"),
                            _FakeTarget("raise the personal stake")],  # dup
        )
        _, hint = SEL.critic_report_to_refine_signals(rep)
        assert "raise the personal stake" in hint
        assert "land the choice on-stage" in hint
        assert hint.count("raise the personal stake") == 1   # de-duped

    def test_dict_shaped_targets(self):
        rep = _FakeReport("flat", reroll_targets=[{"hint": "fix the open"}])
        _, hint = SEL.critic_report_to_refine_signals(rep)
        assert "fix the open" in hint

    def test_axes_are_structural(self):
        # Every derived axis must live in the escalation router's STRUCTURAL_AXES
        # so decide_escalation_scope routes to EPISODE.
        from nodes._otr_reroll_escalation import STRUCTURAL_AXES
        for verdict in ("uneven", "flat", "mid_collapse"):
            rep = _FakeReport(verdict, continuity_issues=[{"x": 1}])
            axes, _ = SEL.critic_report_to_refine_signals(rep)
            assert set(axes) <= STRUCTURAL_AXES

    def test_malformed_report_degrades(self):
        class _Bad:
            @property
            def arc_verdict(self):
                raise RuntimeError("boom")
        assert SEL.critic_report_to_refine_signals(_Bad()) == ([], "")

    def test_hint_bounded(self):
        rep = _FakeReport("flat", reroll_targets=[_FakeTarget("x " * 500)])
        _, hint = SEL.critic_report_to_refine_signals(rep)
        assert len(hint) <= 400


class TestEscalationGate:
    def test_default_off(self):
        assert SEL.critic_escalation_enabled(env={}) is False

    def test_on_variants(self):
        for v in ("1", "true", "YES", "on"):
            assert SEL.critic_escalation_enabled(env={"OTR_ENABLE_CRITIC_ESCALATION": v}) is True

    def test_off_variants(self):
        for v in ("0", "false", "no", "off", ""):
            assert SEL.critic_escalation_enabled(env={"OTR_ENABLE_CRITIC_ESCALATION": v}) is False


class TestBuildEscalationSignal:
    def test_off_is_empty_and_no_meta(self):
        meta = {}
        sig = SEL.build_escalation_signal(_FakeReport("flat"), meta, env={})
        assert sig == {}
        assert "story_quality" not in meta          # byte-identical: no key added

    def test_on_strong_arc_ships(self):
        meta = {}
        sig = SEL.build_escalation_signal(
            _FakeReport("strong"), meta,
            env={"OTR_ENABLE_CRITIC_ESCALATION": "1"},
        )
        assert sig["verdict"] == "ship"
        assert sig["failing_axes"] == []
        assert meta["story_quality"]["critic_failing_axes"] == []

    def test_on_weak_arc_discards_with_structural_axes(self):
        meta = {}
        sig = SEL.build_escalation_signal(
            _FakeReport("mid_collapse"), meta,
            env={"OTR_ENABLE_CRITIC_ESCALATION": "1"},
        )
        assert sig["verdict"] == "discard"
        assert "resolution" in sig["failing_axes"]
        assert meta["story_quality"]["critic_regeneration_hint"]

    def test_on_routes_to_episode(self):
        # End-to-end: a weak arc signal must route decide_escalation_scope to
        # EPISODE (whole-episode regenerate).
        from nodes._otr_reroll_escalation import decide_escalation_scope, EscalationScope
        sig = SEL.build_escalation_signal(
            _FakeReport("flat"), {}, env={"OTR_ENABLE_CRITIC_ESCALATION": "1"},
        )
        decision = decide_escalation_scope(sig)
        assert decision.scope is EscalationScope.EPISODE


class TestKeepBestMonotonicity:
    def test_non_monotonic_picks_peak(self):
        # The live failure mode: pass1 peaks at 72, then drifts to 65. Keep-best
        # MUST ship the peak (index 0), never the last.
        assert SEL.keep_best_index([72, 65]) == 0

    def test_improving_picks_last(self):
        assert SEL.keep_best_index([60, 70, 80]) == 2

    def test_tie_picks_earliest(self):
        assert SEL.keep_best_index([80, 80]) == 0
        assert SEL.keep_best_index([50, 80, 80]) == 1

    def test_empty_is_zero(self):
        assert SEL.keep_best_index([]) == 0

    def test_single(self):
        assert SEL.keep_best_index([42]) == 0
