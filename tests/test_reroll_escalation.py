"""Sprint 10B Wave 1 Agent C regression -- critic-driven reroll
escalation.

Covers nodes/_otr_reroll_escalation.decide_escalation_scope:

  * verdict='ship'                         -> NONE
  * structural failing_axes                -> EPISODE
  * local-only failing_axes + targets      -> BEAT
  * local-only failing_axes + NO targets   -> LINE (fallthrough)
  * mixed failing_axes                     -> LINE (fallthrough)
  * unknown axis names                     -> LINE (fallthrough)
  * critic_result=None                     -> LINE (fallback)
  * dict input shape                       -> same as dataclass input

Plus source-level pins on the cascade wiring:
  * the new escalation module is imported in _otr_freeze_cascade.py
  * the cascade dispatches on escalation.scope
  * meta.reroll_escalation is stamped for soak audit
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass, field   # noqa: E402
from typing import List   # noqa: E402

from nodes._otr_reroll_escalation import (   # noqa: E402
    EscalationDecision,
    EscalationScope,
    LOCAL_AXES,
    STRUCTURAL_AXES,
    decide_escalation_scope,
)


# ---------------------------------------------------------------------------
# Fixtures -- minimal CriticResult-like shapes
# ---------------------------------------------------------------------------


@dataclass
class _FakeCriticResult:
    verdict: str = "ship"
    failing_axes: List[str] = field(default_factory=list)
    mean_score: float = 4.0
    regeneration_hint: str = ""


def _targets(*beat_ids: str) -> List[dict]:
    return [{"beat_id": b} for b in beat_ids]


# ---------------------------------------------------------------------------
# Axis constants
# ---------------------------------------------------------------------------


class TestAxisConstants:
    def test_structural_axes_documented(self):
        """The structural axes are exactly the four arc-level rubric
        axes from the design doc."""
        assert STRUCTURAL_AXES == frozenset({
            "premise_clarity",
            "continuity",
            "resolution",
            "emotional_arc",
        })

    def test_local_axes_documented(self):
        """The local axes are exactly the three line-level rubric
        axes."""
        assert LOCAL_AXES == frozenset({
            "character_distinctiveness",
            "dialogue_naturalness",
            "pacing",
        })

    def test_axes_disjoint(self):
        """Structural and local axes must NOT overlap. An axis being
        both would make the decision ambiguous."""
        assert not (STRUCTURAL_AXES & LOCAL_AXES)


# ---------------------------------------------------------------------------
# Ship verdict -> NONE
# ---------------------------------------------------------------------------


class TestVerdictShipReturnsNone:
    def test_ship_with_no_failing_axes(self):
        cr = _FakeCriticResult(verdict="ship", failing_axes=[])
        d = decide_escalation_scope(cr)
        assert d.scope is EscalationScope.NONE
        assert d.target_beat_ids == []
        assert "ship" in d.reason.lower()

    def test_ship_with_failing_axes_still_none(self):
        """Verdict trumps failing_axes. If the critic returned ship,
        we ship even if some axes were below threshold."""
        cr = _FakeCriticResult(
            verdict="ship",
            failing_axes=["premise_clarity"],
        )
        d = decide_escalation_scope(cr)
        assert d.scope is EscalationScope.NONE

    def test_ship_case_insensitive(self):
        cr = _FakeCriticResult(verdict="SHIP", failing_axes=[])
        d = decide_escalation_scope(cr)
        assert d.scope is EscalationScope.NONE

    def test_ship_passes_regeneration_hint(self):
        """Even on a ship verdict, if a hint was emitted it round-trips
        for diagnostic stamping."""
        cr = _FakeCriticResult(
            verdict="ship",
            regeneration_hint="trim the middle act",
        )
        d = decide_escalation_scope(cr)
        assert d.regeneration_hint == "trim the middle act"


# ---------------------------------------------------------------------------
# Structural failure -> EPISODE
# ---------------------------------------------------------------------------


class TestStructuralFailureEscalatesEpisode:
    def test_premise_clarity_failure(self):
        cr = _FakeCriticResult(
            verdict="discard",
            failing_axes=["premise_clarity"],
        )
        d = decide_escalation_scope(cr)
        assert d.scope is EscalationScope.EPISODE
        assert d.target_beat_ids == [], (
            "EPISODE scope must not carry per-beat targets"
        )
        assert "structural" in d.reason.lower()

    def test_continuity_failure(self):
        cr = _FakeCriticResult(
            verdict="discard",
            failing_axes=["continuity"],
        )
        d = decide_escalation_scope(cr)
        assert d.scope is EscalationScope.EPISODE

    def test_resolution_failure(self):
        cr = _FakeCriticResult(
            verdict="discard",
            failing_axes=["resolution"],
        )
        d = decide_escalation_scope(cr)
        assert d.scope is EscalationScope.EPISODE

    def test_emotional_arc_failure(self):
        cr = _FakeCriticResult(
            verdict="discard",
            failing_axes=["emotional_arc"],
        )
        d = decide_escalation_scope(cr)
        assert d.scope is EscalationScope.EPISODE

    def test_mixed_structural_plus_local_still_episode(self):
        """ANY structural hit triggers episode scope. Local axes
        being present doesn't downgrade to BEAT/LINE -- the structural
        problem dominates."""
        cr = _FakeCriticResult(
            verdict="discard",
            failing_axes=["premise_clarity", "dialogue_naturalness"],
        )
        d = decide_escalation_scope(cr, story_critic_targets=_targets("b002"))
        assert d.scope is EscalationScope.EPISODE

    def test_episode_passes_regeneration_hint(self):
        cr = _FakeCriticResult(
            verdict="discard",
            failing_axes=["resolution"],
            regeneration_hint="ending must close the trial arc",
        )
        d = decide_escalation_scope(cr)
        assert d.regeneration_hint == "ending must close the trial arc"


# ---------------------------------------------------------------------------
# Local-only failure + targets -> BEAT
# ---------------------------------------------------------------------------


class TestLocalFailureWithTargetsEscalatesBeat:
    def test_pacing_alone_with_targets_beat_scope(self):
        cr = _FakeCriticResult(
            verdict="discard",
            failing_axes=["pacing"],
        )
        d = decide_escalation_scope(
            cr, story_critic_targets=_targets("b003", "b005"),
        )
        assert d.scope is EscalationScope.BEAT
        assert d.target_beat_ids == ["b003", "b005"]
        assert "local" in d.reason.lower()

    def test_two_local_axes_with_targets_beat_scope(self):
        cr = _FakeCriticResult(
            verdict="discard",
            failing_axes=["dialogue_naturalness", "pacing"],
        )
        d = decide_escalation_scope(
            cr, story_critic_targets=_targets("b002"),
        )
        assert d.scope is EscalationScope.BEAT
        assert d.target_beat_ids == ["b002"]

    def test_local_failure_without_targets_falls_back_to_line(self):
        """Local axes signal where the problem is at the line level,
        but BEAT scope needs an explicit list of beats to target. If
        the legacy story critic didn't name any, fall back to LINE."""
        cr = _FakeCriticResult(
            verdict="discard",
            failing_axes=["dialogue_naturalness"],
        )
        d = decide_escalation_scope(cr, story_critic_targets=[])
        assert d.scope is EscalationScope.LINE


# ---------------------------------------------------------------------------
# Mixed / unmapped / None -> LINE (fallback)
# ---------------------------------------------------------------------------


class TestFallbackToLine:
    def test_none_critic_result_falls_back(self):
        d = decide_escalation_scope(None)
        assert d.scope is EscalationScope.LINE
        assert "none" in d.reason.lower() or "fall back" in d.reason.lower()

    def test_unknown_axis_falls_back_to_line(self):
        """A failing_axis the rubric doesn't classify (neither
        structural nor local) means the decision is undefined.
        Default to LINE (the conservative fallback)."""
        cr = _FakeCriticResult(
            verdict="discard",
            failing_axes=["some_invented_axis"],
        )
        d = decide_escalation_scope(cr, story_critic_targets=_targets("b002"))
        assert d.scope is EscalationScope.LINE

    def test_empty_failing_axes_with_discard_verdict_falls_back(self):
        """Verdict=discard but no failing_axes is a weird shape; the
        decision is undefined. Fall back to LINE."""
        cr = _FakeCriticResult(verdict="discard", failing_axes=[])
        d = decide_escalation_scope(cr)
        assert d.scope is EscalationScope.LINE


# ---------------------------------------------------------------------------
# Dict input -- meta-stamped CriticResult shape
# ---------------------------------------------------------------------------


class TestDictInput:
    """The cascade stamps Stage 7's CriticResult on
    meta['stage7_shadow_critic'] as a dict (via CriticResult.to_dict).
    decide_escalation_scope must accept the dict shape too."""

    def test_dict_ship_returns_none(self):
        cr_dict = {
            "verdict": "ship",
            "failing_axes": [],
            "mean_score": 4.2,
        }
        d = decide_escalation_scope(cr_dict)
        assert d.scope is EscalationScope.NONE

    def test_dict_structural_returns_episode(self):
        cr_dict = {
            "verdict": "discard",
            "failing_axes": ["premise_clarity", "continuity"],
            "mean_score": 2.1,
            "regeneration_hint": "rework the open and the close",
        }
        d = decide_escalation_scope(cr_dict)
        assert d.scope is EscalationScope.EPISODE
        assert d.regeneration_hint == "rework the open and the close"


# ---------------------------------------------------------------------------
# EscalationDecision serialization
# ---------------------------------------------------------------------------


class TestEscalationDecisionToDict:
    def test_to_dict_keys(self):
        d = EscalationDecision(
            scope=EscalationScope.EPISODE,
            reason="test",
            target_beat_ids=[],
            regeneration_hint="hint",
        )
        out = d.to_dict()
        assert set(out.keys()) == {
            "scope", "reason", "target_beat_ids", "regeneration_hint",
        }
        assert out["scope"] == "episode"   # enum value, not member name

    def test_scope_round_trips_as_string(self):
        """The scope serializes as the enum's string value so meta
        dicts are JSON-friendly. Soak greps grep on the value."""
        for member, expected in (
            (EscalationScope.NONE, "none"),
            (EscalationScope.EPISODE, "episode"),
            (EscalationScope.BEAT, "beat"),
            (EscalationScope.LINE, "line"),
        ):
            d = EscalationDecision(scope=member, reason="x")
            assert d.to_dict()["scope"] == expected


# ---------------------------------------------------------------------------
# Cascade wiring -- source-level pins
# ---------------------------------------------------------------------------


class TestCascadeWiring:
    """Source-level pins on _otr_freeze_cascade.py confirming the
    escalation module is imported and dispatches on scope. Source
    checks rather than integration tests because the cascade
    function has many dependencies (LLM slot scheduler, ledger,
    etc.) and integration setup is heavy."""

    def _src(self) -> str:
        from pathlib import Path
        p = (
            Path(__file__).resolve().parent.parent
            / "nodes" / "_otr_freeze_cascade.py"
        )
        return p.read_text(encoding="utf-8")

    def test_escalation_module_imported(self):
        src = self._src()
        assert "_otr_reroll_escalation" in src, (
            "Cascade must import the escalation module to dispatch "
            "on scope."
        )

    def test_decide_escalation_scope_called(self):
        src = self._src()
        assert "decide_escalation_scope" in src

    def test_escalation_marker_present(self):
        """The Wave 1 Agent C marker comment must live in source so
        future readers can find the dispatch point."""
        src = self._src()
        assert "Sprint 10B Wave 1 Agent C" in src

    def test_meta_stamp_reroll_escalation(self):
        """The cascade must stamp meta['reroll_escalation'] from
        EscalationDecision.to_dict() so soak diagnostics see the
        decision after the run."""
        src = self._src()
        assert '"reroll_escalation"' in src or "'reroll_escalation'" in src

    def test_episode_scope_stamps_needs_full_rerun(self):
        """When scope is EPISODE, the cascade must construct a
        RerollDisposition with verdict='needs_full_rerun' (skipping
        the legacy reroll cycles entirely)."""
        src = self._src()
        # The episode branch builds a RerollDisposition right there.
        assert "EscalationScope.EPISODE" in src
        # And the verdict literal lands.
        assert 'verdict="needs_full_rerun"' in src
