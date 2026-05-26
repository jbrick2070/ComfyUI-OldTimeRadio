"""Sprint 10A step 6-B -- Stage 2 multi-turn roleplay call site.

Per docs/story-generator-final-plan.md Stage 2: generate N candidate
lines for each beat via the multi-turn chain (built by
_otr_stage2_prompt.build_stage2_prompt_chain), score each candidate
through Stage 3 validators (validate_line), ship the highest scorer.

This module owns:
  * compose_line(plan, beat, generate_fn, n=4, ...) -- the per-beat
    best-of-N entry point.
  * score_candidate(plan, beat, line_text, ...) -- the scoring
    function each candidate runs through.
  * Stage2LineRecord -- per-beat diagnostic record (candidates +
    scores + chosen) stamped on the ledger.

Two modes per the plan's LineComposer 'mode switch':
  * roleplay_multiturn (A2, default): the full 4-turn chain.
  * roleplay_single_turn (A1): one prompt with everything, no
    priming chain. Kept for the A/B lab; same scorer.

The legacy structured/composer path is NOT replaced by this commit.
6-B lands the new path; integration (writer routes through here when
a feature widget is on) is later.

This module is dormant until the operator flips the new path on.

# LLM slot: creative
# Reason: Stage 2 dialogue rendering is creative-axis work per
# project rule 6 (creative_writing_model slot).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional

from ._otr_stage1_plan import Stage1Beat, Stage1Plan
from ._otr_stage2_prompt import (
    Stage2PromptChain,
    assemble_full_chat,
    build_stage2_prompt_chain,
)
from ._otr_stage3_validators import ValidationResult, validate_line


log = logging.getLogger("OTR")


# ---------------------------------------------------------------------------
# Constants -- best-of-N defaults
# ---------------------------------------------------------------------------

# Per the final plan: 'Start N=4. Generate N candidates, score each
# via Stage 3 validators, ship the highest scorer. Increase N if
# quality justifies.'
DEFAULT_BEST_OF_N: int = 4

# Sampling temperature for Turn 4. The plan calls for creative-axis
# generation; the writer's current creative slot ships at ~0.85
# baseline. We use the same starting point so an A1/A2 lab
# comparison stays apples-to-apples.
DEFAULT_TURN4_TEMPERATURE: float = 0.85

# Token budget for Turn 4. Generous -- the validator catches over-
# length; we'd rather give the model room than truncate mid-thought.
DEFAULT_TURN4_MAX_NEW_TOKENS: int = 200


# ---------------------------------------------------------------------------
# Mode literal
# ---------------------------------------------------------------------------

STAGE2_MODE = Literal["roleplay_multiturn", "roleplay_single_turn"]
"""Per the plan's LineComposer mode switch:
  * roleplay_multiturn (A2): the full 4-turn priming chain.
  * roleplay_single_turn (A1): one prompt with all the context
    crammed in. No priming chain; used as the A/B lab fallback.
"""


# ---------------------------------------------------------------------------
# Diagnostic records
# ---------------------------------------------------------------------------


@dataclass
class Stage2CandidateRecord:
    """One candidate line + its score breakdown."""

    index: int                       # 0..N-1
    text: str                        # the generated line
    score: float                     # higher = better
    elapsed_seconds: float
    validation: ValidationResult     # full per-line audit
    # Convenience flags (computed from validation):
    error_count: int = 0
    warn_count: int = 0


@dataclass
class Stage2LineRecord:
    """Per-beat best-of-N result. Stamped on meta.stage2_lines per
    episode so soak diagnostics can grep candidate quality across
    runs.
    """

    beat_id: str
    speaker: str
    mode: str                            # one of STAGE2_MODE
    candidates: List[Stage2CandidateRecord] = field(default_factory=list)
    chosen_index: int = -1               # winner; -1 if none viable
    chosen_text: str = ""
    fallback_reason: Optional[str] = None  # set when chosen_text is a
                                           # fallback (no clean candidate)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


# Per-finding penalty weights. Errors hit hardest -- a line with an
# error-level Stage 3 finding is worse than one with several warns.
_PENALTY_ERROR: float = 10.0
_PENALTY_WARN: float = 1.0

# Bonus for landing close to the beat's length_target_words. The
# closer to target, the higher the bonus. Caps at 5.0 for an exact
# match.
_LENGTH_BONUS_MAX: float = 5.0


def score_candidate(
    plan: Stage1Plan,
    beat: Stage1Beat,
    line_text: str,
    *,
    banned_phrases: Optional[List[str]] = None,
    min_on_beat_ratio: float = 0.10,
) -> tuple[float, ValidationResult]:
    """Score one candidate line against the Stage 3 validators +
    a length-closeness bonus.

    Returns (score, validation_result).

    Score formula:
        base = 100.0
        - _PENALTY_ERROR * #errors
        - _PENALTY_WARN * #warns
        + length_bonus(actual, target)

    where length_bonus = _LENGTH_BONUS_MAX * (1 - |actual-target|/target),
    floored at 0.
    """
    validation = validate_line(
        plan, beat, line_text,
        banned_phrases=banned_phrases,
        min_on_beat_ratio=min_on_beat_ratio,
    )
    errors = len(validation.errors)
    warns = len(validation.warns)
    actual = len([t for t in (line_text or "").split() if any(c.isalnum() for c in t)])
    target = max(1, beat.length_target_words)
    closeness = 1.0 - min(1.0, abs(actual - target) / target)
    length_bonus = _LENGTH_BONUS_MAX * max(0.0, closeness)
    score = (
        100.0
        - _PENALTY_ERROR * errors
        - _PENALTY_WARN * warns
        + length_bonus
    )
    return score, validation


# ---------------------------------------------------------------------------
# Single-turn (A1) prompt builder
# ---------------------------------------------------------------------------


def build_single_turn_prompt(
    *,
    plan: Stage1Plan,
    beat: Stage1Beat,
    speaker_name: str,
    previous_speaker: Optional[str] = None,
    previous_line: Optional[str] = None,
) -> List[dict]:
    """Build the A1 'one prompt with everything' chat for an A/B
    comparison against the A2 multi-turn chain. Same context, same
    Turn 4 spec, but no priming acknowledgments.
    """
    speaker = next((c for c in plan.cast if c.name == speaker_name), None)
    if speaker is None:
        # Reserved speaker (ANNOUNCER, MUSIC) -- single-turn doesn't
        # support these; the caller should route announcer beats to
        # the existing dedicated pass.
        raise ValueError(
            f"build_single_turn_prompt: speaker {speaker_name!r} not in "
            f"plan.cast (reserved speakers are not supported by the "
            f"Stage 2 single-turn mode)"
        )
    facts_block = "\n".join(f"  - {f}" for f in plan.running_facts) or "  (none)"
    arc_block = (
        f"Setup: {plan.arc.setup}\n"
        f"Complication: {plan.arc.complication}\n"
        f"Resolution: {plan.arc.resolution}"
    )
    prev_clause = ""
    if previous_speaker and previous_line:
        prev_clause = (
            f'Previous line ({previous_speaker}): "{previous_line}"\n\n'
        )
    user = (
        f"You are {speaker.name}. {speaker.persona}\n\n"
        f"Pronouns: {speaker.pronouns}. Role in arc: {speaker.arc_role}.\n\n"
        f"Premise: {plan.premise}\n\n"
        f"Arc:\n{arc_block}\n\n"
        f"Established facts the dialogue must respect:\n{facts_block}\n\n"
        f"{prev_clause}"
        f"Your beat: {beat.intent}\n"
        f"Length target: about {beat.length_target_words} words.\n"
        f"Emotional register: {beat.emotional_register}\n\n"
        f"Speak your line. Just the line -- no stage directions, "
        f"no narration, no self-naming."
    )
    return [{"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# Best-of-N compose entry point
# ---------------------------------------------------------------------------


def compose_line(
    *,
    plan: Stage1Plan,
    beat: Stage1Beat,
    generate_fn: Callable[..., str],
    mode: STAGE2_MODE = "roleplay_multiturn",
    n: int = DEFAULT_BEST_OF_N,
    temperature: float = DEFAULT_TURN4_TEMPERATURE,
    max_new_tokens: int = DEFAULT_TURN4_MAX_NEW_TOKENS,
    previous_speaker: Optional[str] = None,
    previous_line: Optional[str] = None,
    banned_phrases: Optional[List[str]] = None,
    min_on_beat_ratio: float = 0.10,
) -> Stage2LineRecord:
    """Generate N candidate lines for this beat; score each via
    Stage 3 validators; return the best-scoring as the shipped line.

    Args:
        plan: Stage 1 plan (cast + arc + running_facts + beats).
        beat: the Stage1Beat to render.
        generate_fn: a creative-slot generate fn with signature
            (messages, *, temperature, max_new_tokens) -> str.
            ConstrainedGenerateFn from _otr_constrained_generate is
            ALSO compatible (it just ignores the constraint on
            free-form dialogue text).
        mode: 'roleplay_multiturn' (A2, default) or
            'roleplay_single_turn' (A1).
        n: number of candidates per beat. Default 4.
        temperature / max_new_tokens: forwarded to generate_fn.
        previous_speaker / previous_line: Turn-4 context.

    Returns:
        Stage2LineRecord with candidates, scores, and the winner.

    Pure function -- no I/O, no ledger touches. Caller decides what
    to do with the record + line.

    # LLM slot: creative
    # Reason: Turn 4 dialogue rendering is creative-axis work per
    # project rule 6.
    """
    record = Stage2LineRecord(
        beat_id=beat.beat_id,
        speaker=beat.speaker,
        mode=mode,
    )

    # Resolve speaker (must be a cast member for Stage 2 -- announcer
    # beats route to the existing announcer pass).
    speaker = next((c for c in plan.cast if c.name == beat.speaker), None)
    if speaker is None:
        record.fallback_reason = (
            f"speaker {beat.speaker!r} is not in plan.cast (likely a "
            f"reserved speaker like ANNOUNCER/MUSIC); Stage 2 does "
            f"not generate for reserved speakers."
        )
        return record

    # Build the prompt chain ONCE per beat.
    if mode == "roleplay_multiturn":
        chain = build_stage2_prompt_chain(
            speaker=speaker,
            beat=beat,
            arc=plan.arc,
            premise=plan.premise,
            running_facts=plan.running_facts,
            previous_speaker=previous_speaker,
            previous_line=previous_line,
        )
        chat_messages = assemble_full_chat(chain)
    elif mode == "roleplay_single_turn":
        chat_messages = build_single_turn_prompt(
            plan=plan,
            beat=beat,
            speaker_name=beat.speaker,
            previous_speaker=previous_speaker,
            previous_line=previous_line,
        )
    else:
        raise ValueError(
            f"compose_line: mode must be one of "
            f"['roleplay_multiturn', 'roleplay_single_turn']; "
            f"got {mode!r}"
        )

    # Run N candidates.
    for i in range(n):
        t0 = time.monotonic()
        try:
            # LLM slot: creative
            # Reason: Turn 4 dialogue rendering is the creative-axis
            # call per project rule 6.
            raw = generate_fn(
                chat_messages,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            log.warning(
                "[Stage2Compose] beat=%s candidate %d/%d generate "
                "raised %s: %s",
                beat.beat_id, i + 1, n,
                type(exc).__name__, str(exc)[:200],
            )
            record.candidates.append(Stage2CandidateRecord(
                index=i,
                text="",
                score=float("-inf"),
                elapsed_seconds=elapsed,
                validation=ValidationResult(),
                error_count=0,
                warn_count=0,
            ))
            continue
        elapsed = time.monotonic() - t0
        text = (raw or "").strip()
        score, validation = score_candidate(
            plan, beat, text,
            banned_phrases=banned_phrases,
            min_on_beat_ratio=min_on_beat_ratio,
        )
        record.candidates.append(Stage2CandidateRecord(
            index=i,
            text=text,
            score=score,
            elapsed_seconds=elapsed,
            validation=validation,
            error_count=len(validation.errors),
            warn_count=len(validation.warns),
        ))

    # Pick the winner: highest score. Ties broken by lower index
    # (earlier candidate wins -- matches Python's max() default
    # behavior on equal keys with stable ordering).
    viable = [c for c in record.candidates if c.text]
    if not viable:
        record.fallback_reason = (
            f"all {n} candidates returned empty / generate raised"
        )
        record.chosen_index = -1
        record.chosen_text = ""
        return record

    winner = max(viable, key=lambda c: c.score)
    record.chosen_index = winner.index
    record.chosen_text = winner.text
    log.info(
        "[Stage2Compose] beat=%s mode=%s n=%d winner=#%d score=%.1f "
        "(errors=%d warns=%d)",
        beat.beat_id, mode, n, winner.index, winner.score,
        winner.error_count, winner.warn_count,
    )
    return record
