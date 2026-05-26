r"""Sprint 10A step 7 -- Whole-episode critic call site.

Per docs/story-generator-final-plan.md step 7: 'Implement the whole-
episode critic against the rubric from step 1. Binary verdict only --
no surgical reroll path exists.'

The critic call:
  1. Loads the rubric via _otr_critic_rubric.load_rubric()
  2. Builds a system + user prompt embedding all 10 axes + their
     anchors + the threshold rule.
  3. Calls the technical-slot LLM under grammar constraint against
     the WholeEpisodeCriticReport schema (via the existing
     _otr_constrained_generate.make_constrained_generate_fn from
     step 3-B).
  4. Parses + validates the response; computes verdict locally from
     the LOCAL ship/discard threshold rule (NOT the LLM's claimed
     verdict -- we trust the LLM's scores but compute the binary
     outcome deterministically).
  5. Returns a CriticResult with the report + a regeneration hint
     for Stage 1 if discard.

This module is dormant until step 8 / writer wiring uses it.

# LLM slot: technical
# Reason: critic is a structured-JSON pass per project rule 6; runs
# on the technical_model slot.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError, conint

from . import _otr_lmfe_compat  # noqa: F401  (compat shim, may be needed by constrained backend)
from ._otr_critic_rubric import Rubric, RubricAxis, ShipThreshold, load_rubric
from ._otr_stage1_plan import Stage1Plan


log = logging.getLogger("OTR")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CRITIC_BASE_TEMPERATURE: float = 0.30
_CRITIC_RETRY_TEMPERATURE: float = 0.15
_CRITIC_MAX_NEW_TOKENS: int = 2048
_CRITIC_MAX_ATTEMPTS: int = 2


# ---------------------------------------------------------------------------
# Verdict literal
# ---------------------------------------------------------------------------


CRITIC_VERDICT = Literal["ship", "discard"]


# ---------------------------------------------------------------------------
# Pydantic schema -- the structured output the LLM produces
# ---------------------------------------------------------------------------


class AxisScore(BaseModel):
    """One axis: integer score 1-5 + a one-sentence justification."""

    score: conint(ge=1, le=5) = Field(  # type: ignore[valid-type]
        ...,
        description="Integer score from 1 (worst) to 5 (best) per the anchor descriptions.",
    )
    justification: str = Field(
        ...,
        min_length=5,
        max_length=400,
        description="One sentence explaining the score.",
    )


class WholeEpisodeCriticReport(BaseModel):
    """The complete critic verdict. Shape matches the audit-trail
    section of the rubric markdown.

    NOTE: the LLM emits an axis_scores object with the 10 axis keys
    that the rubric defines. We could pin each key as a separate
    field, but that couples the schema to the rubric file
    structurally. Instead we accept axis_scores as a dict keyed by
    axis_json_key, and the consumer validates the key set matches
    the rubric's 10 keys post-parse. That keeps the rubric as the
    single source of truth.
    """

    verdict: CRITIC_VERDICT = Field(
        ...,
        description="The LLM's binary verdict. The final verdict used by the pipeline is computed locally from the threshold rule, NOT this field, but we accept it for forensic comparison.",
    )
    axis_scores: dict[str, AxisScore] = Field(
        ...,
        description="Map from axis json_key to AxisScore. Must include exactly the 10 axes defined in the rubric.",
    )
    mean_score: float = Field(
        ...,
        ge=0.0,
        le=5.0,
        description="LLM's computed mean. We re-compute locally; this is for forensic comparison.",
    )
    failing_axes: List[str] = Field(
        default_factory=list,
        description="LLM's list of axes that failed the threshold. Re-computed locally.",
    )
    regeneration_hint: str = Field(
        ...,
        min_length=5,
        max_length=600,
        description="One paragraph telling Stage 1 what to do differently if the verdict is discard.",
    )


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class CriticCallFailedError(RuntimeError):
    """Raised when the critic call exhausts its retry budget. Pipeline
    handler typically falls back to a SOFT verdict (treat as ship-
    with-warns) rather than discard, since a critic-call failure
    isn't itself evidence of a bad episode.
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
            f"Whole-episode critic exhausted retry budget after "
            f"{attempts} attempt(s); last error: "
            f"{type(last_error).__name__}: {last_error}"
        )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class CriticAttemptRecord:
    """One critic call attempt."""

    attempt: int
    status: str            # 'valid_first_attempt' | 'valid_after_retry' | 'parse_failed' | 'rubric_key_mismatch' | 'generate_failed'
    elapsed_seconds: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class CriticResult:
    """Aggregate result for one whole-episode critic call.

    The pipeline-level handler reads `verdict` (the FINAL binary
    outcome computed locally from the threshold rule) and routes
    accordingly: ship -> proceed to render; discard -> regenerate
    from Stage 1 with `regeneration_hint` in the prompt context.
    """

    verdict: str              # 'ship' or 'discard'
    report: Optional[WholeEpisodeCriticReport]  # None if all attempts failed
    attempts: List[CriticAttemptRecord] = field(default_factory=list)
    mean_score: float = 0.0
    failing_axes: List[str] = field(default_factory=list)
    regeneration_hint: str = ""

    def to_dict(self) -> dict:
        """Serialize for meta.whole_episode_critic ledger stamp."""
        return {
            "verdict": self.verdict,
            "mean_score": self.mean_score,
            "failing_axes": list(self.failing_axes),
            "regeneration_hint": self.regeneration_hint,
            "report": (
                self.report.model_dump() if self.report is not None else None
            ),
            "attempts": [
                {
                    "attempt": a.attempt,
                    "status": a.status,
                    "elapsed_seconds": a.elapsed_seconds,
                    "error_type": a.error_type,
                    "error_message": a.error_message,
                }
                for a in self.attempts
            ],
        }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _format_rubric_for_prompt(rubric: Rubric) -> str:
    """Embed the rubric's axes + anchors + threshold into a prompt-
    friendly block. The LLM sees the SAME rubric the operator does."""
    lines: List[str] = []
    lines.append("RUBRIC -- score each axis from 1 (worst) to 5 (best):")
    lines.append("")
    for axis in rubric.axes:
        lines.append(f"{axis.ordinal}. {axis.name}  (json_key: {axis.json_key})")
        lines.append(f"   Definition: {axis.definition}")
        for anchor in axis.anchors:
            lines.append(f"   - {anchor}")
        lines.append("")
    th = rubric.threshold
    lines.append("THRESHOLD (the pipeline computes the final ship/discard verdict locally from this rule):")
    lines.append(
        f"  * 'ship' iff NO axis scores below {th.min_axis_score} "
        f"AND mean of all axis scores >= {th.min_mean_score} "
        f"AND axis '{th.sfw_axis_key}' >= {th.sfw_axis_min}."
    )
    lines.append("  * Otherwise 'discard'.")
    return "\n".join(lines)


def _format_rendered_ledger_for_prompt(
    rendered_lines: List[dict], max_lines: int = 60,
) -> str:
    """Render the ledger's voiced lines as a compact transcript for
    the critic prompt. Truncate at max_lines to keep the prompt
    inside the context window for any reasonable episode length."""
    out: List[str] = []
    count = 0
    for line in rendered_lines:
        if not isinstance(line, dict):
            continue
        bid = line.get("beat_id", "????")
        speaker = line.get("speaker") or line.get("char_name") or "???"
        text = (line.get("text") or "").strip()
        if not text:
            continue
        out.append(f"  [{bid}] {speaker}: {text}")
        count += 1
        if count >= max_lines:
            out.append(f"  ... ({len(rendered_lines) - count} more lines elided)")
            break
    return "\n".join(out) if out else "  (no voiced lines)"


_CRITIC_SYSTEM_PROMPT = """You are a story editor evaluating a finished old-time-radio drama episode against an explicit rubric. Score each of the 10 axes integer 1-5 per the anchor descriptions; write a one-sentence justification per axis. Output JSON ONLY -- no prose preamble."""


def build_critic_prompt(
    *,
    plan: Stage1Plan,
    rendered_lines: List[dict],
    rubric: Rubric,
) -> List[dict]:
    """Build the chat messages list for the critic call."""
    rubric_block = _format_rubric_for_prompt(rubric)
    transcript_block = _format_rendered_ledger_for_prompt(rendered_lines)
    cast_block = "\n".join(
        f"  - {c.name} ({c.gender}, {c.pronouns}): {c.persona}"
        for c in plan.cast
    )
    arc_block = (
        f"  Setup: {plan.arc.setup}\n"
        f"  Complication: {plan.arc.complication}\n"
        f"  Resolution: {plan.arc.resolution}"
    )
    facts_block = (
        "\n".join(f"  - {f}" for f in plan.running_facts)
        if plan.running_facts
        else "  (none)"
    )
    user = (
        f"Episode plan:\n\n"
        f"Premise: {plan.premise}\n\n"
        f"Arc:\n{arc_block}\n\n"
        f"Cast:\n{cast_block}\n\n"
        f"Running facts:\n{facts_block}\n\n"
        f"Rendered transcript:\n{transcript_block}\n\n"
        f"{rubric_block}\n\n"
        f"Output a JSON object with this exact shape:\n"
        f"{{\n"
        f'  "verdict": "ship" | "discard",\n'
        f'  "axis_scores": {{\n'
        + ",\n".join(
            f'    "{axis.json_key}": {{"score": <1-5>, '
            f'"justification": "<one sentence>"}}'
            for axis in rubric.axes
        )
        + "\n  },\n"
        f'  "mean_score": <float 0-5>,\n'
        f'  "failing_axes": [<json_key strings>],\n'
        f'  "regeneration_hint": "<one paragraph for Stage 1 if discard>"\n'
        f"}}\n\n"
        f"JSON only. No prose."
    )
    return [
        {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
        {"role": "user",   "content": user},
    ]


# ---------------------------------------------------------------------------
# Verdict computation (local, deterministic)
# ---------------------------------------------------------------------------


def compute_verdict_locally(
    report: WholeEpisodeCriticReport,
    rubric: Rubric,
) -> tuple[str, float, List[str]]:
    """Compute the binary ship/discard verdict locally from the
    report's axis scores using the rubric's threshold rule.

    We DO NOT trust the LLM's own claimed verdict field -- if the
    LLM hand-waves a 'ship' on scores that don't actually meet
    threshold, we override.

    Returns (verdict, mean_score, failing_axes_list).
    """
    th = rubric.threshold
    scores = [a.score for a in report.axis_scores.values()]
    if not scores:
        return ("discard", 0.0, ["empty_axis_scores"])
    mean = sum(scores) / len(scores)
    failing: List[str] = []
    for key, axis_score in report.axis_scores.items():
        if axis_score.score < th.min_axis_score:
            failing.append(key)
    # SFW special case
    sfw_axis = report.axis_scores.get(th.sfw_axis_key)
    if sfw_axis is not None and sfw_axis.score < th.sfw_axis_min:
        if th.sfw_axis_key not in failing:
            failing.append(th.sfw_axis_key)
    if mean < th.min_mean_score:
        failing.append("__mean_below_threshold__")
    verdict = "discard" if failing else "ship"
    return (verdict, mean, failing)


# ---------------------------------------------------------------------------
# Call entry point
# ---------------------------------------------------------------------------


def _validate_axis_keys(
    report: WholeEpisodeCriticReport, rubric: Rubric,
) -> Optional[str]:
    """Return None if axis_scores keys match the rubric exactly; an
    error message string if not."""
    expected = {a.json_key for a in rubric.axes}
    got = set(report.axis_scores.keys())
    if got != expected:
        missing = expected - got
        extra = got - expected
        bits = []
        if missing:
            bits.append(f"missing={sorted(missing)}")
        if extra:
            bits.append(f"extra={sorted(extra)}")
        return f"axis_scores key mismatch: {'; '.join(bits)}"
    return None


def run_whole_episode_critic(
    *,
    plan: Stage1Plan,
    rendered_lines: List[dict],
    generate_fn: Callable[..., str],
    rubric: Optional[Rubric] = None,
    max_attempts: int = _CRITIC_MAX_ATTEMPTS,
) -> CriticResult:
    """Run the whole-episode critic.

    Args:
        plan: the Stage 1 plan the episode was built from.
        rendered_lines: list of L3 ledger 'lines' rows (each with
            'beat_id' / 'text' / 'speaker' fields).
        generate_fn: a creative or technical generate fn signature
            (messages, *, temperature, max_new_tokens) -> str.
            For grammar-constrained sampling pass a fn built by
            _otr_constrained_generate.make_constrained_generate_fn
            (Stage1Plan? no -- bind the critic schema here, not
            Stage1Plan).
        rubric: optional pre-loaded rubric. Default loads the
            project canonical via _otr_critic_rubric.load_rubric().
        max_attempts: retry budget. Default 2.

    Returns:
        CriticResult with the final verdict computed LOCALLY from
        the threshold rule.

    Raises:
        CriticCallFailedError on retry-budget exhaustion. Pipeline-
        level handler must catch and decide whether to fall back
        soft-ship or hard-discard.

    # LLM slot: technical
    # Reason: structured-JSON pass under grammar constraint per
    # project rule 6.
    """
    if rubric is None:
        rubric = load_rubric()

    messages = build_critic_prompt(
        plan=plan,
        rendered_lines=rendered_lines,
        rubric=rubric,
    )

    attempts: List[CriticAttemptRecord] = []
    last_raw = ""
    last_exc: Exception = RuntimeError("no attempt ran")

    for attempt_idx in range(1, max_attempts + 1):
        temperature = (
            _CRITIC_BASE_TEMPERATURE
            if attempt_idx == 1
            else _CRITIC_RETRY_TEMPERATURE
        )
        t0 = time.monotonic()
        log.info(
            "[WholeEpisodeCritic] attempt %d/%d (temperature=%.2f)",
            attempt_idx, max_attempts, temperature,
        )
        try:
            # LLM slot: technical
            # Reason: structured-JSON pass under grammar constraint
            # per project rule 6.
            raw = generate_fn(
                messages,
                temperature=temperature,
                max_new_tokens=_CRITIC_MAX_NEW_TOKENS,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            attempts.append(CriticAttemptRecord(
                attempt=attempt_idx,
                status="generate_failed",
                elapsed_seconds=elapsed,
                error_type=type(exc).__name__,
                error_message=str(exc)[:300],
            ))
            log.warning(
                "[WholeEpisodeCritic] attempt %d generate raised %s",
                attempt_idx, type(exc).__name__,
            )
            last_exc = exc
            continue

        last_raw = raw

        # Parse JSON.
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            elapsed = time.monotonic() - t0
            attempts.append(CriticAttemptRecord(
                attempt=attempt_idx,
                status="parse_failed",
                elapsed_seconds=elapsed,
                error_type=type(exc).__name__,
                error_message=str(exc)[:300],
            ))
            log.warning(
                "[WholeEpisodeCritic] attempt %d JSON parse failed",
                attempt_idx,
            )
            last_exc = exc
            continue

        # Pydantic validate.
        try:
            report = WholeEpisodeCriticReport(**payload)
        except ValidationError as exc:
            elapsed = time.monotonic() - t0
            attempts.append(CriticAttemptRecord(
                attempt=attempt_idx,
                status="parse_failed",
                elapsed_seconds=elapsed,
                error_type=type(exc).__name__,
                error_message=str(exc)[:300],
            ))
            last_exc = exc
            continue

        # Axis-key match against rubric.
        key_err = _validate_axis_keys(report, rubric)
        if key_err is not None:
            elapsed = time.monotonic() - t0
            attempts.append(CriticAttemptRecord(
                attempt=attempt_idx,
                status="rubric_key_mismatch",
                elapsed_seconds=elapsed,
                error_type="AxisKeyMismatch",
                error_message=key_err[:300],
            ))
            last_exc = ValueError(key_err)
            continue

        # Success: compute the binary verdict locally.
        verdict, mean, failing = compute_verdict_locally(report, rubric)
        elapsed = time.monotonic() - t0
        status = (
            "valid_first_attempt"
            if attempt_idx == 1
            else "valid_after_retry"
        )
        attempts.append(CriticAttemptRecord(
            attempt=attempt_idx,
            status=status,
            elapsed_seconds=elapsed,
        ))
        log.info(
            "[WholeEpisodeCritic] verdict=%s mean=%.2f failing_axes=%s",
            verdict, mean, failing,
        )
        return CriticResult(
            verdict=verdict,
            report=report,
            attempts=attempts,
            mean_score=mean,
            failing_axes=failing,
            regeneration_hint=report.regeneration_hint,
        )

    raise CriticCallFailedError(
        attempts=len(attempts),
        last_raw_output=last_raw,
        last_error=last_exc,
    )
