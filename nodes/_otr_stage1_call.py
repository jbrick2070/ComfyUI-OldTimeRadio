"""Sprint 10A step 3-B -- Stage 1 plan call site.

Per docs/story-generator-final-plan.md Stage 1: one LLM call with
grammar-constrained generation produces the entire episode plan
(premise + arc + cast + beats + running_facts). This module is the
call site.

Inputs:
  * generate_fn -- a ConstrainedGenerateFn already bound to Stage1Plan.
    Built by _otr_constrained_generate.make_constrained_generate_fn
    with schema_model=Stage1Plan.
  * news_seed -- the seed text (RSS body OR custom premise).
  * num_characters -- target cast size (1..6, matches widget).
  * target_words -- total episode word budget.
  * episode_title_hint -- optional working title (writer widget; can
    be empty).

Output: a parsed + semantically-validated Stage1Plan instance.

Failure modes:
  * Stage1PlanGenerationError -- raised on parse / validation / call
    failures after the retry budget is exhausted. Carries the last
    raw LLM output + the underlying exception for forensic logging.

Diagnostic logging:
  * Every attempt logs at INFO with `[Stage1Plan] attempt N/N status=...`
    so the operator soak gate (>=19/20 first-attempt valid plans) can
    be measured by grep on a log tail.
  * Attempt 1 success is the success case the operator gate tracks.

This module is dormant until 3-C wires it into the writer behind a
feature flag.

# LLM slot: technical
# Reason: structured-JSON pass under grammar constraint -- routes to
# the technical model slot per project rule 6.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from pydantic import ValidationError

from . import _otr_lmfe_compat  # noqa: F401  (compat shim, transitive lmfe import)
from ._otr_stage1_plan import Stage1Plan, parse_and_validate_plan


log = logging.getLogger("OTR")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Retry budget for Stage 1. Lower than _otr_structured_call's three
# rungs because constrained decoding makes parse failures impossible
# at the token layer -- the only remaining failure modes are
# semantic (cast/beat cross-field) and decode-cutoff (max_new_tokens
# exhausted before the JSON closed). Two attempts is enough: the
# first to fail fast, the second to retry at lower temperature so
# the sampler picks the most likely tokens.
_STAGE1_MAX_ATTEMPTS: int = 2

# Token budget. A complete Stage1Plan in JSON is roughly:
#   premise + arc.3x + cast.6x.6fields + beats.40x.6fields +
#   running_facts.20x + key + brace overhead
# That tops out around ~3500 tokens. 4096 gives margin; Mistral-Nemo
# has plenty of headroom relative to its 8192-token context cap.
_STAGE1_MAX_NEW_TOKENS: int = 4096

# Base temperature. Lower than the writer's creative-pass default
# (~0.7..0.9) because Stage 1 is structural planning, not creative
# rendering. The model should make grounded structural decisions
# rather than reach for novelty. Constrained decoding will reject
# invalid tokens anyway, but lower temperature also means fewer
# regenerations on the structural retry.
_STAGE1_BASE_TEMPERATURE: float = 0.55

# Retry temperature. Lower than base per the 2B principle from
# _otr_structured_call: raising entropy on a JSON retry encourages
# more drift, not less. A constrained-decoding retry pursues the
# most-likely valid continuation.
_STAGE1_RETRY_TEMPERATURE: float = 0.35


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class Stage1PlanGenerationError(RuntimeError):
    """Raised when Stage 1 plan generation exhausts the retry budget.

    The pipeline-level handler turns this into a discard-and-rerun on
    the WHOLE episode (per the final plan's discard-and-rerun design
    -- surgical reroll is deleted). Operator may see this in the soak
    log if the constrained-decoding gate (>=19/20) is missed.
    """

    def __init__(
        self,
        *,
        attempts: int,
        last_raw_output: str,
        last_error: Exception,
    ) -> None:
        self.attempts = attempts
        self.last_raw_output = last_raw_output
        self.last_error = last_error
        super().__init__(
            f"Stage1Plan generation exhausted retry budget after "
            f"{attempts} attempt(s); last error: "
            f"{type(last_error).__name__}: {last_error}"
        )


# ---------------------------------------------------------------------------
# Diagnostic record (stamped on the writer ledger for soak measurement)
# ---------------------------------------------------------------------------


@dataclass
class Stage1AttemptRecord:
    """One Stage 1 call attempt. Stamped onto meta.stage1_attempts.

    Operator soak gate (>=19/20 first-attempt valid) is measured by
    counting episodes where attempts[0].status == 'valid_first_attempt'.
    """

    attempt: int          # 1-indexed
    status: str           # 'valid_first_attempt', 'valid_after_retry', 'parse_failed', 'semantic_failed', 'decode_cutoff'
    elapsed_seconds: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


_STAGE1_SYSTEM_PROMPT = """You are a structural planner for an old-time-radio drama. Your job is to produce a complete episode plan as a single JSON object.

The plan binds together: a one-sentence premise, a three-act arc, a cast list with explicit voice + pronoun assignments, a beat-by-beat outline, and a short list of running facts the dialogue must honor.

You do NOT write any dialogue at this stage. You produce the SKELETON the dialogue stage will later fill in.

Hard rules:
  - length_target_words MUST be an integer between 5 and 200 inclusive. Never emit 0, negative numbers, or values above 200. A typical beat is 15 to 45 words.
  - Every cast member gets a unique Bark voice_id from the v2/en_speaker_0 through v2/en_speaker_9 pool. The Bark v2 pool is binary: voice_id 0, 1, 3, 5, 6, 8 are male-coded; voice_id 2, 4, 7, 9 are female-coded. Pick a voice_id whose coded gender matches the cast member's gender.
  - Cast gender must match conventional name expectations. ROBINSON, COLE, NELSON, OSCAR, REGINALD, VANCE, HAYES are conventionally male first names; ANYA, MARGOT, NORA, MIRA, ALICE, EDNA are conventionally female. Do NOT cast a male-coded first name as gender=female or a female-coded first name as gender=male.
  - Cast gender + pronouns must match (male -> he/him, female -> she/her, nonbinary -> they/them).
  - beat_id format is bNNN, three-digit padded, monotonically increasing.
  - Every beat's speaker is either a cast member name (exact match) OR the literal 'ANNOUNCER' (bookend lines) OR the literal 'MUSIC' (music_inter cue beats).
  - callback_to references must point to EARLIER beats only.
  - The plan must be SFW: no profanity, no graphic violence, no sexual content.
  - Output JSON ONLY -- no prose preamble, no explanation, no postamble."""


def build_stage1_user_prompt(
    *,
    news_seed: str,
    num_characters: int,
    target_words: int,
    episode_title_hint: str = "",
) -> str:
    """Build the user-turn prompt for Stage 1.

    The system prompt above establishes role + hard rules. This user
    prompt carries the per-episode parameters and the news seed.
    """
    # Estimate beat count from target_words. The plan's
    # length_target_words per beat is variable, but a sensible
    # planning heuristic is ~20-30 words per beat for character lines
    # plus a couple of announcer bookends. Use 25 wpb average plus 2
    # bookends; clamp to the schema's [3, 40] range.
    estimated_beats = max(3, min(40, 2 + (target_words // 25)))

    title_clause = (
        f"Working title hint: {episode_title_hint!r}. You may keep, "
        f"replace, or ignore it as fits the story."
        if episode_title_hint.strip()
        else "No working title provided -- choose your own arc focus."
    )

    return f"""News seed:
\"\"\"
{news_seed}
\"\"\"

Episode parameters:
  - Speaking characters (cast size, excludes ANNOUNCER): {num_characters}
  - Target total dialogue word count: {target_words}
  - Estimated beat count (target -- you may adjust within the schema's 3..40 range): {estimated_beats}
  - {title_clause}

Produce the complete Stage1Plan JSON object now. JSON only."""


# ---------------------------------------------------------------------------
# Generation entry point
# ---------------------------------------------------------------------------


def generate_stage1_plan(
    generate_fn,
    *,
    news_seed: str,
    num_characters: int,
    target_words: int,
    episode_title_hint: str = "",
    max_attempts: int = _STAGE1_MAX_ATTEMPTS,
) -> tuple[Stage1Plan, list[Stage1AttemptRecord]]:
    """Run Stage 1 plan generation.

    Args:
        generate_fn: a ConstrainedGenerateFn (from
            make_constrained_generate_fn) bound to schema_model=
            Stage1Plan. Signature: (messages, *, temperature,
            max_new_tokens) -> str.
        news_seed: RSS body or custom premise text.
        num_characters: cast size (1..6).
        target_words: total episode word budget.
        episode_title_hint: optional working title widget value.
        max_attempts: retry budget. Default 2.

    Returns:
        (plan, attempts) -- the validated Stage1Plan and the per-
        attempt record list. Attempt 1 in the list is the first call;
        if it succeeded, len(attempts) == 1 and attempts[0].status ==
        'valid_first_attempt' -- this is the soak gate signal.

    Raises:
        Stage1PlanGenerationError on retry-budget exhaustion. Pipeline
        handler turns this into a whole-episode discard-and-rerun.
    """
    import time

    if num_characters < 1 or num_characters > 6:
        raise ValueError(
            f"num_characters must be in 1..6 (schema cast bound); got {num_characters}"
        )
    if not news_seed.strip():
        raise ValueError("news_seed must be non-empty")

    user_prompt = build_stage1_user_prompt(
        news_seed=news_seed,
        num_characters=num_characters,
        target_words=target_words,
        episode_title_hint=episode_title_hint,
    )
    messages = [
        {"role": "system", "content": _STAGE1_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    attempts: list[Stage1AttemptRecord] = []
    last_raw = ""
    last_exc: Exception = RuntimeError("no attempt ran")

    for attempt_idx in range(1, max_attempts + 1):
        temperature = (
            _STAGE1_BASE_TEMPERATURE
            if attempt_idx == 1
            else _STAGE1_RETRY_TEMPERATURE
        )
        log.info(
            "[Stage1Plan] attempt %d/%d (temperature=%.2f) start",
            attempt_idx, max_attempts, temperature,
        )
        t0 = time.monotonic()
        try:
            # LLM slot: technical
            # Reason: Stage 1 is a structured-JSON pass under grammar
            # constraint -- routes to the technical model slot per
            # project rule 6.
            raw = generate_fn(
                messages,
                temperature=temperature,
                max_new_tokens=_STAGE1_MAX_NEW_TOKENS,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            attempts.append(Stage1AttemptRecord(
                attempt=attempt_idx,
                status="generate_failed",
                elapsed_seconds=elapsed,
                error_type=type(exc).__name__,
                error_message=str(exc)[:300],
            ))
            log.warning(
                "[Stage1Plan] attempt %d generate() raised %s: %s",
                attempt_idx, type(exc).__name__, str(exc)[:200],
            )
            last_exc = exc
            continue

        last_raw = raw

        # Decode-cutoff detection: under constrained decoding the JSON
        # is guaranteed valid as far as it goes, but max_new_tokens
        # may have stopped it before the close. json.loads will
        # raise; bubble that out as a distinct status.
        try:
            plan_dict = json.loads(raw)
        except json.JSONDecodeError as exc:
            elapsed = time.monotonic() - t0
            attempts.append(Stage1AttemptRecord(
                attempt=attempt_idx,
                status="decode_cutoff",
                elapsed_seconds=elapsed,
                error_type=type(exc).__name__,
                error_message=str(exc)[:300],
            ))
            log.warning(
                "[Stage1Plan] attempt %d JSON parse failed (likely "
                "max_new_tokens cutoff): %s",
                attempt_idx, str(exc)[:200],
            )
            last_exc = exc
            continue

        # Pydantic + semantic validators (belt-and-braces -- the
        # constrained sampler covers schema-level shape, but
        # cross-field semantics live in validate_plan_semantics).
        try:
            plan = parse_and_validate_plan(plan_dict)
        except (ValidationError, ValueError) as exc:
            elapsed = time.monotonic() - t0
            status = (
                "parse_failed"
                if isinstance(exc, ValidationError)
                else "semantic_failed"
            )
            attempts.append(Stage1AttemptRecord(
                attempt=attempt_idx,
                status=status,
                elapsed_seconds=elapsed,
                error_type=type(exc).__name__,
                error_message=str(exc)[:300],
            ))
            log.warning(
                "[Stage1Plan] attempt %d %s: %s",
                attempt_idx, status, str(exc)[:200],
            )
            last_exc = exc
            continue

        # Success
        elapsed = time.monotonic() - t0
        status = "valid_first_attempt" if attempt_idx == 1 else "valid_after_retry"
        attempts.append(Stage1AttemptRecord(
            attempt=attempt_idx,
            status=status,
            elapsed_seconds=elapsed,
        ))
        log.info(
            "[Stage1Plan] attempt %d status=%s in %.2fs (cast=%d, beats=%d, running_facts=%d)",
            attempt_idx, status, elapsed,
            len(plan.cast), len(plan.beats), len(plan.running_facts),
        )
        return plan, attempts

    # All attempts failed
    raise Stage1PlanGenerationError(
        attempts=len(attempts),
        last_raw_output=last_raw,
        last_error=last_exc,
    )
