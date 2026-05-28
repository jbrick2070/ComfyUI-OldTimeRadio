"""Sprint 10B Wave 1 Agent C -- critic-driven reroll escalation.

The legacy Sprint 5C reroll loop targets INDIVIDUAL LINES that the
legacy story critic flagged as flat. On a structural failure
(premise_clarity, continuity, resolution, emotional_arc) the right
unit is NOT the line -- it's the whole episode. The line is fine in
isolation; what's broken is the arc the line sits inside. Rerolling
the line individually can't fix that.

This module decides what SCOPE to escalate to based on the Stage 7
whole-episode critic's verdict + failing_axes:

  * verdict='ship'                              -> EscalationScope.NONE
  * failing_axes intersects STRUCTURAL_AXES     -> EscalationScope.EPISODE
  * failing_axes subset of LOCAL_AXES AND
    story_critic_targets names beats            -> EscalationScope.BEAT
  * otherwise (mixed / unmapped)                -> EscalationScope.LINE

EscalationScope.NONE        -- ship the episode as composed. The
                               BUG-LOCAL-281 gate from Wave 0.
EscalationScope.EPISODE     -- stamp meta.freeze_verdict='needs_full_rerun'
                               + meta.regeneration_hint immediately.
                               Skip the Sprint 5C line reroll (it
                               cannot fix structural problems; the
                               cycles would exhaust pointlessly,
                               which is what BUG-LOCAL-279 / 280 /
                               281 surfaced over 2026-05-26..27).
EscalationScope.BEAT        -- target the beats the story critic
                               named for recomposition. (Currently
                               routes to the existing line-reroll
                               with the same target set; future
                               Wave 2 work can split beat-from-line.)
EscalationScope.LINE        -- the legacy Sprint 5C line reroll path
                               unchanged.

Wire from `_otr_freeze_cascade.py`. Gated by the writer widget
`enable_critic_escalation`. Default OFF preserves the existing
ship-gate / legacy-line-reroll behavior; ON activates the typed
escalation decision.

This module is PURE. No LLM, no I/O.

# LLM slot: technical
# Reason: the decision is deterministic logic over already-stamped
# critic results; no LLM call introduced here. The slot tag keeps
# the slot sweep happy on this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


# ---------------------------------------------------------------------------
# Axis classification
# ---------------------------------------------------------------------------


# Structural axes describe failures that cannot be repaired at the
# line level. They are about the whole-episode arc and require a
# whole-episode regenerate.
STRUCTURAL_AXES: frozenset[str] = frozenset({
    "premise_clarity",
    "continuity",
    "resolution",
    "emotional_arc",
})


# Local axes describe failures that COULD be repaired at the beat
# or line level. They are about individual dialogue craft, not the
# arc shape.
LOCAL_AXES: frozenset[str] = frozenset({
    "character_distinctiveness",
    "dialogue_naturalness",
    "pacing",
})


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class EscalationScope(str, Enum):
    """Where the reroll should fire. String enum so it round-trips
    through JSON / meta dicts cleanly."""

    NONE = "none"           # ship as composed
    EPISODE = "episode"     # whole-episode regenerate (needs_full_rerun)
    BEAT = "beat"           # recompose specific beats
    LINE = "line"           # legacy Sprint 5C line reroll


@dataclass(frozen=True)
class EscalationDecision:
    """The output of decide_escalation_scope.

    scope:                where to escalate
    reason:               human-readable explanation; stamped on meta
                          for soak audit
    target_beat_ids:      beats the escalation should focus on; empty
                          for NONE / EPISODE scopes. Populated from
                          story_critic_targets when the scope is BEAT
                          or LINE.
    regeneration_hint:    pass-through from CriticResult.regeneration_hint
                          when present. Empty string when absent.
    """

    scope: EscalationScope
    reason: str
    target_beat_ids: List[str] = field(default_factory=list)
    regeneration_hint: str = ""

    def to_dict(self) -> dict:
        """JSON-friendly view for stamping on meta.reroll_escalation."""
        return {
            "scope": self.scope.value,
            "reason": self.reason,
            "target_beat_ids": list(self.target_beat_ids),
            "regeneration_hint": self.regeneration_hint,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_verdict(critic_result: Any) -> str:
    """Read .verdict off either a CriticResult dataclass or a dict
    stamped to meta. Empty string when absent."""
    if critic_result is None:
        return ""
    verdict = getattr(critic_result, "verdict", None)
    if verdict is None and isinstance(critic_result, dict):
        verdict = critic_result.get("verdict")
    return str(verdict or "").strip().lower()


def _coerce_failing_axes(critic_result: Any) -> set[str]:
    """Read .failing_axes as a set of lowercased axis keys."""
    if critic_result is None:
        return set()
    axes = getattr(critic_result, "failing_axes", None)
    if axes is None and isinstance(critic_result, dict):
        axes = critic_result.get("failing_axes")
    if not axes:
        return set()
    try:
        return {str(a).strip().lower() for a in axes if a}
    except TypeError:
        return set()


def _coerce_regeneration_hint(critic_result: Any) -> str:
    """Read .regeneration_hint off either a CriticResult dataclass
    or a dict. Empty string when absent."""
    if critic_result is None:
        return ""
    hint = getattr(critic_result, "regeneration_hint", None)
    if hint is None and isinstance(critic_result, dict):
        hint = critic_result.get("regeneration_hint")
    return str(hint or "").strip()


def _critic_call_failed(critic_result: Any) -> bool:
    """BUG-LOCAL-287 (2026-05-27): detect a critic-failure marker.

    The Stage 7 shadow critic's broad-except handler in
    `_otr_freeze_cascade.py` stamps one of two shapes on
    `meta['stage7_shadow_critic']` when the call itself failed
    (NOT when the critic returned a real verdict):

      * Setup failure (adapter could not build the ledger view):
            {'ok': True/False, 'skipped': True, 'reason': '...'}

      * Runtime failure (CriticCallFailedError -- typically
        empty-output JSONDecodeError on the technical model;
        Gemma 4B is a frequent offender):
            {'ok': False, 'skipped': False,
             'shadow_setup_failed': True,
             'error_type': '...', 'error_message': '...'}

    A real `CriticResult.to_dict()` carries a 'verdict' key with a
    non-empty string. Distinguishing these two shapes lets the
    escalation router treat 'critic dark' as 'ship' (no signal to
    escalate on) rather than the default 'fall back to LINE reroll'
    that historically exhausted on Sprint 5B story_critic targets.

    Returns True iff `critic_result` looks like a failure marker.
    Dataclass `CriticResult` instances always have a 'verdict' set
    and never satisfy this predicate.
    """
    if critic_result is None:
        return False
    # Dataclass path -- check the verdict attr. A real CriticResult
    # has a non-empty verdict ('ship' or 'discard').
    if not isinstance(critic_result, dict):
        verdict = getattr(critic_result, "verdict", "")
        return not (verdict and str(verdict).strip())
    # Dict path. The cascade coerces missing Stage 7 to {} via
    # `meta.get("stage7_shadow_critic") or {}` -- an EMPTY dict
    # means Stage 7 was never run (operator did not opt into
    # shadow diagnostics), NOT that it ran and failed. Treat that
    # as "no signal known" -- the existing legacy-LINE fallback
    # path applies, same as the critic_result-is-None case in
    # decide_escalation_scope. Only mark as failed when the dict
    # carries one of the explicit failure-marker keys.
    if not critic_result:
        return False
    # Explicit failure-marker keys the shadow stamps on a runtime
    # or setup failure.
    if critic_result.get("ok") is False:
        return True
    if critic_result.get("shadow_setup_failed") is True:
        return True
    if critic_result.get("skipped") is True:
        return True
    if "error_type" in critic_result or "error_message" in critic_result:
        return True
    # A real CriticResult.to_dict() always carries a non-empty
    # 'verdict' key. A dict that lacks one but doesn't trip any
    # marker above is a shape we have not seen -- conservatively
    # treat as failed so the cascade ships rather than burns
    # reroll cycles on stale targets.
    verdict = str(critic_result.get("verdict") or "").strip()
    if not verdict:
        return True
    return False


def _extract_target_beat_ids(
    story_critic_targets: Optional[List[dict]],
) -> List[str]:
    """Pull beat_ids out of the legacy story critic's reroll_targets
    list. Each target is a dict with at minimum 'beat_id' or
    'line_id' (the writer sometimes calls them line_ids, but
    semantically they're the same per-beat indices)."""
    if not story_critic_targets:
        return []
    out: List[str] = []
    seen: set[str] = set()
    for tgt in story_critic_targets:
        if not isinstance(tgt, dict):
            continue
        bid = tgt.get("beat_id") or tgt.get("line_id") or ""
        bid = str(bid).strip()
        if bid and bid not in seen:
            seen.add(bid)
            out.append(bid)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def decide_escalation_scope(
    critic_result: Any,
    *,
    story_critic_targets: Optional[List[dict]] = None,
) -> EscalationDecision:
    """Decide where to escalate based on the Stage 7 whole-episode
    critic's verdict.

    Args:
        critic_result:        a CriticResult dataclass OR a dict
                              previously stamped on meta. The
                              function reads .verdict / .failing_axes
                              / .regeneration_hint via getattr or
                              dict.get -- both shapes work.
        story_critic_targets: the legacy story critic's
                              reroll_targets list (each a dict with
                              'beat_id'). Used to bind beat-scope
                              targets when the failing_axes are
                              all local.

    Returns:
        EscalationDecision -- typed, JSON-friendly via .to_dict().

    PD1: never raises. Bad inputs degrade to LINE scope with a
    'fallback' reason -- the legacy reroll path is the safe default.
    """
    if critic_result is None:
        return EscalationDecision(
            scope=EscalationScope.LINE,
            reason=(
                "critic_result is None -- fall back to legacy line "
                "reroll"
            ),
            target_beat_ids=_extract_target_beat_ids(
                story_critic_targets,
            ),
        )

    # BUG-LOCAL-287 (2026-05-27): when the Stage 7 critic CALL failed
    # (the shadow stamped a `{'ok': False, ...}` or
    # `{'shadow_setup_failed': True, ...}` marker on meta because
    # CriticCallFailedError fired or the adapter could not build),
    # we have NO signal about the episode's quality. Falling back to
    # the legacy LINE reroll is wrong in this case: the legacy reroll
    # is driven by the Sprint 5B story_critic's targets (a DIFFERENT
    # critic from Stage 7) and on long episodes it consistently
    # exhausts its 2-cycle budget chasing the same "flat lines"
    # over and over, which stamps needs_full_rerun and halts Bark.
    # Live evidence: pending_20260527_170250 (Gemma technical slot,
    # 9-beat superconductivity episode) -- Stage 7 critic emitted
    # empty output twice, shadow stamped marker, Agent C routed to
    # LINE, legacy reroll exhausted 2 cycles on the same 6 targets,
    # episode failed to ship.
    # The correct behavior when Stage 7 is dark: SHIP. The episode
    # passed Stage 3 line validators, the doctor's diagnosis ran,
    # and the multiturn composer's best-of-N already picked the
    # cleanest line per beat. Without an authoritative whole-episode
    # signal we don't have grounds to keep regenerating.
    if _critic_call_failed(critic_result):
        return EscalationDecision(
            scope=EscalationScope.NONE,
            reason=(
                "Stage 7 shadow critic FAILED to produce a verdict "
                "(empty output / shadow setup failure) -- no signal "
                "to escalate on; ship the composed episode rather "
                "than burn legacy-reroll cycles on the Sprint 5B "
                "story_critic's separate target list. See "
                "BUG-LOCAL-287."
            ),
            target_beat_ids=[],
            regeneration_hint="",
        )

    verdict = _coerce_verdict(critic_result)
    failing_axes = _coerce_failing_axes(critic_result)
    hint = _coerce_regeneration_hint(critic_result)
    target_beat_ids = _extract_target_beat_ids(story_critic_targets)

    # Ship verdict -- no reroll. Matches the BUG-LOCAL-281 Wave 0
    # ship gate.
    if verdict == "ship":
        return EscalationDecision(
            scope=EscalationScope.NONE,
            reason=(
                "Stage 7 critic verdict=ship -- episode passes the "
                "whole-episode rubric; no reroll needed"
            ),
            target_beat_ids=[],
            regeneration_hint=hint,
        )

    # Non-ship verdict (typically 'discard'). Decide scope.
    structural_hits = failing_axes & STRUCTURAL_AXES
    local_hits = failing_axes & LOCAL_AXES

    if structural_hits:
        return EscalationDecision(
            scope=EscalationScope.EPISODE,
            reason=(
                f"Stage 7 critic verdict={verdict!r} with structural "
                f"failing_axes={sorted(structural_hits)!r} -- the "
                f"problem is the arc, not the lines. Whole-episode "
                f"regenerate."
            ),
            target_beat_ids=[],   # episode scope -- no per-beat targets
            regeneration_hint=hint,
        )

    if local_hits and not (failing_axes - LOCAL_AXES) and target_beat_ids:
        # All failing axes are local AND we have specific beats to
        # target. Recompose those beats.
        return EscalationDecision(
            scope=EscalationScope.BEAT,
            reason=(
                f"Stage 7 critic verdict={verdict!r} with local-only "
                f"failing_axes={sorted(local_hits)!r} and "
                f"{len(target_beat_ids)} story-critic beat target(s) "
                f"-- recompose the named beats."
            ),
            target_beat_ids=target_beat_ids,
            regeneration_hint=hint,
        )

    # Fall through: any other shape (no failing_axes; mixed but no
    # beat targets; unmapped axis names; etc.). The legacy line
    # reroll handles these.
    return EscalationDecision(
        scope=EscalationScope.LINE,
        reason=(
            f"Stage 7 critic verdict={verdict!r} failing_axes="
            f"{sorted(failing_axes)!r}; no structural hit AND no "
            f"local+targets binding -- fall back to legacy line reroll"
        ),
        target_beat_ids=target_beat_ids,
        regeneration_hint=hint,
    )
