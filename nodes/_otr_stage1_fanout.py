"""nodes/_otr_stage1_fanout.py -- Sprint 4.3 (2026-05-28).

Stage 1 fan-out helper. Generates N candidate Stage1Plans (default
N=3) each guided by a distinct diversity-knob system-prompt prefix,
and returns the list for OTR_BeatSelector / select_winning_beat_sheet
to score.

Why fan out at the BEAT SHEET, not at line revision (round-robin
§5b): take more shots at a good story, do not repair a bad one.
Small local models score taste unreliably; they CAN write three
structurally different beat sheets when nudged with distinct system
prompts, and the Sprint 2 structural validators + Sprint 4 selector
can pick the strongest one mechanically.

The three diversity knobs name three different sources of dramatic
pressure (per the round-robin's worked example):

  MORAL_DILEMMA          Two characters face equally bad choices;
                         the costly choice is between values, not
                         between actions and consequences.
  BUREAUCRATIC_ABSURD    Ordinary process IS the antagonist; the
                         characters fight the rules, not each
                         other.
  INTIMATE_PERSONAL_COST The cost lands on a body or relationship
                         in the room; the stakes are not abstract.

Each knob is a system-prompt PREFIX that the caller appends to the
existing Stage 1 system prompt. None of the knobs override the
existing schema constraints (beat ids, cast roster, etc.) -- they
nudge the LLM toward a particular kind of conflict architecture.

This module is PURE -- no LLM call (the caller passes generate_fn),
no I/O. The fan-out runs sequentially by default (VRAM safety on
the single-GPU machine); a parallel mode can land later when the
loader supports concurrent slot use.

UTF-8 no BOM.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

log = logging.getLogger("OTR")


__all__ = [
    "DiversityKnob",
    "DIVERSITY_KNOB_PROMPTS",
    "ALL_DIVERSITY_KNOBS",
    "FanOutResult",
    "fan_out_stage1_plans",
]


class DiversityKnob:
    """Enum of canonical diversity-knob labels."""

    MORAL_DILEMMA = "moral_dilemma"
    BUREAUCRATIC_ABSURD = "bureaucratic_absurd"
    INTIMATE_PERSONAL_COST = "intimate_personal_cost"


ALL_DIVERSITY_KNOBS: Tuple[str, str, str] = (
    DiversityKnob.MORAL_DILEMMA,
    DiversityKnob.BUREAUCRATIC_ABSURD,
    DiversityKnob.INTIMATE_PERSONAL_COST,
)


# System-prompt PREFIX per knob. The caller appends the existing
# Stage 1 system prompt below the prefix so the LLM still sees all
# the usual schema rules. Each prefix is one short paragraph -- the
# point is to nudge, not to overconstrain.
DIVERSITY_KNOB_PROMPTS: dict[str, str] = {
    DiversityKnob.MORAL_DILEMMA: (
        "Frame this episode as a MORAL DILEMMA. The two principal "
        "characters face equally bad choices. The costly choice is "
        "between values they both hold, not between an action and "
        "its consequences. Neither side is the villain; the conflict "
        "is what each is willing to give up to honor the value they "
        "hold most. The ending must turn on which value wins.\n"
    ),
    DiversityKnob.BUREAUCRATIC_ABSURD: (
        "Frame this episode as a BUREAUCRATIC ABSURDITY. Ordinary "
        "process IS the antagonist -- a form, a policy, a deadline, "
        "a chain of approvals that makes the right thing impossible "
        "on principle. The characters fight the rules, not each "
        "other. Comedy is allowed (often required); the costly "
        "choice is whether to break the process or to be broken by "
        "it. The ending must turn on the process bending or "
        "winning.\n"
    ),
    DiversityKnob.INTIMATE_PERSONAL_COST: (
        "Frame this episode as INTIMATE PERSONAL COST. The cost "
        "lands on a body or a relationship in the room -- not on a "
        "city, not on humanity, not on the audit firm. Stakes are "
        "concrete: one person's job, one person's health, one "
        "relationship between two people. The costly choice is "
        "whose body, whose relationship, whose loss. The ending "
        "must turn on what is permanently changed in that body or "
        "that relationship.\n"
    ),
}


@dataclass
class FanOutResult:
    """One candidate's outcome from a fan-out run."""

    knob: str
    plan: Any                       # Stage1Plan on success, None on failure
    raw_output: Optional[str] = None
    error: Optional[str] = None     # human-readable failure on plan == None

    @property
    def ok(self) -> bool:
        return self.plan is not None


def fan_out_stage1_plans(
    *,
    generate_fn: Callable[..., str],
    parse_fn: Callable[[str], Any],
    base_system_prompt: str,
    user_prompt: str,
    knobs: Optional[List[str]] = None,
    base_temperature: float = 0.8,
    max_new_tokens: int = 4096,
) -> List[FanOutResult]:
    """Generate one Stage1Plan per diversity knob.

    Sequential -- the loader holds one model slot resident at a
    time on the 16 GB Laptop, so running three concurrent Stage 1
    calls would not fit. Total cost is N times the single-call
    cost; on Mistral-Nemo NF4 that's roughly 3 * 60-90 sec for the
    three-knob default at the prompts above.

    Args:
        generate_fn: LLM call (messages, *, temperature,
            max_new_tokens) -> str. Caller binds this to the
            creative slot via _otr_constrained_generate.make_*.
        parse_fn:    raw_output -> Stage1Plan or raises on
            invalid. Caller binds this to
            `_otr_stage1_plan.parse_and_validate_plan`.
        base_system_prompt: the existing Stage 1 system prompt
            (the LLM gets the diversity knob's prefix + this).
        user_prompt: the existing Stage 1 user prompt.
        knobs: subset of ALL_DIVERSITY_KNOBS to run; None ->
            all three. Pass a single-element list for a degenerate
            "fan-out of one" useful for unit tests.
        base_temperature: temperature for every call. Slightly
            higher than the default Stage 1 temperature so the
            knob-prefixed prompts have room to diverge.
        max_new_tokens: per-call cap.

    Returns:
        List[FanOutResult] in knob order. Failed candidates are
        FanOutResult(knob, plan=None, error=<message>); the caller
        passes only `.plan` of successful results to the selector.

    # LLM slot: creative
    # Reason: outline generation is a creative pass per project
    # rule 6. The knob prefix nudges narrative architecture; the
    # underlying parse stays structural.
    """
    knobs_to_run = list(knobs) if knobs is not None else list(ALL_DIVERSITY_KNOBS)
    results: List[FanOutResult] = []
    for knob in knobs_to_run:
        prefix = DIVERSITY_KNOB_PROMPTS.get(knob, "")
        system = (prefix + base_system_prompt).strip()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        log.info(
            "[Stage1FanOut] knob=%s temperature=%.2f max_new_tokens=%d",
            knob, base_temperature, max_new_tokens,
        )
        try:
            # LLM slot: creative
            # Reason: outline generation is a creative pass per
            # project rule 6; the knob prefix nudges narrative
            # architecture while the underlying parse stays
            # structural.
            raw = generate_fn(
                messages,
                temperature=base_temperature,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[Stage1FanOut] knob=%s generate_fn raised %s: %s",
                knob, type(exc).__name__, str(exc)[:200],
            )
            results.append(FanOutResult(
                knob=knob, plan=None,
                error=f"generate_fn: {type(exc).__name__}: {exc}",
            ))
            continue

        try:
            plan = parse_fn(raw)
        except Exception as exc:  # noqa: BLE001 -- pydantic + semantic
            log.warning(
                "[Stage1FanOut] knob=%s parse_fn raised %s: %s",
                knob, type(exc).__name__, str(exc)[:200],
            )
            results.append(FanOutResult(
                knob=knob, plan=None, raw_output=raw,
                error=f"parse_fn: {type(exc).__name__}: {exc}",
            ))
            continue

        results.append(FanOutResult(
            knob=knob, plan=plan, raw_output=raw,
        ))
        log.info(
            "[Stage1FanOut] knob=%s OK (beats=%d cast=%d)",
            knob,
            len(getattr(plan, "beats", []) or []),
            len(getattr(plan, "cast", []) or []),
        )

    return results
