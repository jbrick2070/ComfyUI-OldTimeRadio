r"""Sprint 10B Wave 1 Agent E -- Editor agent (dormant).

Per docs/2026-05-26-good-story-writer-architecture__02_design.md
Section 4 Wave 1 Agent E: the Editor reads a DirectorBrief (Section
3.1) + a Writer draft, and emits a typed EditorVerdict (Section 3.2)
that scores the draft against the existing critic rubric and lists
the failing axes by their json_key.

This module is the in-loop sibling of nodes/_otr_whole_episode_critic.py
(the post-pipeline critic). Both consume the same rubric; the Editor
runs INSIDE the writers' room loop and gives the Writer a tight-loop
revision signal, while the whole-episode critic remains a Stage-7
end-of-pipeline gate.

Wave 1 lands this module + node + tests; the node is registered but
dormant -- Wave 2 Agent F (Story Room loop) wires the Editor into
the writer's revision cycles.

Acceptance (per Section 4): the Editor must correctly fail the
hand-constructed "Spray of Hope" draft from
docs/2026-05-26-good-story-writer-architecture__01_problem-statement.md
Section 1.1 on at least `premise_clarity`, `resolution`, and
`emotional_arc`.

# LLM slot: technical
# Reason: structured constrained-decode verdict per project rule 6.
# Editor turns route to the writer's `technical_model` slot per the
# Section 2 routing rule.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError

from . import _otr_lmfe_compat  # noqa: F401 -- compat shim, may be needed by constrained backend
from ._otr_critic_rubric import Rubric, load_rubric


log = logging.getLogger("OTR")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EDITOR_BASE_TEMPERATURE: float = 0.30
_EDITOR_RETRY_TEMPERATURE: float = 0.15
_EDITOR_MAX_NEW_TOKENS: int = 1024
_EDITOR_MAX_ATTEMPTS: int = 2


# ---------------------------------------------------------------------------
# DirectorBrief contract surface
# ---------------------------------------------------------------------------
#
# Wave 1 Agent D owns the canonical DirectorBrief dataclass at
# nodes/_otr_director_brief.py per Section 3.1. Wave 1 agents run in
# parallel, so at this commit's HEAD the Director module may not yet
# be importable. Define a structural Protocol locally so the Editor's
# typing contract pins ONLY the field shape Section 3.1 freezes; Wave
# 2 Agent F reconciles by importing the real dataclass when wiring
# the writers' room loop.
#
# Section 3.1 fields (frozen for Sprint 10B):
#   news_premise: str
#   dramatic_question: str
#   opposed_desires: str
#   arc_shape: str
#   audience_feel: str
#   forbidden_drift: List[str]
#   raw_brief: str
# ---------------------------------------------------------------------------


@runtime_checkable
class DirectorBriefLike(Protocol):
    """Structural type for the Director's brief. Matches Section 3.1."""

    news_premise: str
    dramatic_question: str
    opposed_desires: str
    arc_shape: str
    audience_feel: str
    forbidden_drift: List[str]
    raw_brief: str


def _coerce_director_brief(brief: Any) -> Dict[str, Any]:
    """Normalize a DirectorBriefLike OR a dict into a plain dict the
    prompt builder can format. Accepts:

      * an object exposing the Section 3.1 attributes (Agent D's
        dataclass, or any structural match).
      * a dict already shaped per Section 3.1.
      * a JSON string carrying the dict (the node entry deserializes
        the serialized brief before calling run_editor).

    Missing fields default to safe empties so the Editor does not
    crash on a partially-populated brief; the prompt instructs the
    LLM to score on what is present.
    """
    if isinstance(brief, str):
        try:
            brief = json.loads(brief)
        except json.JSONDecodeError as exc:
            raise EditorCallFailedError(
                attempts=0,
                last_raw_output=brief[:200],
                last_error=exc,
            )
    if isinstance(brief, dict):
        d = brief
    else:
        # Best-effort attribute pull.
        d = {
            "news_premise": getattr(brief, "news_premise", ""),
            "dramatic_question": getattr(brief, "dramatic_question", ""),
            "opposed_desires": getattr(brief, "opposed_desires", ""),
            "arc_shape": getattr(brief, "arc_shape", ""),
            "audience_feel": getattr(brief, "audience_feel", ""),
            "forbidden_drift": list(getattr(brief, "forbidden_drift", []) or []),
            "raw_brief": getattr(brief, "raw_brief", ""),
        }
    return {
        "news_premise": str(d.get("news_premise", "") or ""),
        "dramatic_question": str(d.get("dramatic_question", "") or ""),
        "opposed_desires": str(d.get("opposed_desires", "") or ""),
        "arc_shape": str(d.get("arc_shape", "") or ""),
        "audience_feel": str(d.get("audience_feel", "") or ""),
        "forbidden_drift": [
            str(x) for x in (d.get("forbidden_drift", []) or [])
        ],
        "raw_brief": str(d.get("raw_brief", "") or ""),
    }


# ---------------------------------------------------------------------------
# EditorVerdict dataclass (Section 3.2)
# ---------------------------------------------------------------------------


@dataclass
class EditorVerdict:
    """The Editor's verdict on a Writer's draft.

    Per Section 3.2:
      pass_decision: True = ship; False = revise.
      failing_axes:  rubric axis json_keys that did NOT pass.
      per_axis_notes: axis_key -> 1-3 sentence note.
      overall_note:  free-form summary.
      cycle:         0-indexed Editor pass number.
    """

    pass_decision: bool
    failing_axes: List[str] = field(default_factory=list)
    per_axis_notes: Dict[str, str] = field(default_factory=dict)
    overall_note: str = ""
    cycle: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass_decision": bool(self.pass_decision),
            "failing_axes": list(self.failing_axes),
            "per_axis_notes": dict(self.per_axis_notes),
            "overall_note": str(self.overall_note),
            "cycle": int(self.cycle),
        }


# ---------------------------------------------------------------------------
# Pydantic schema (mirrors EditorVerdict for constrained decode)
# ---------------------------------------------------------------------------


class EditorVerdictSchema(BaseModel):
    """Constrained-decode binding for the Editor call.

    The Pydantic model mirrors the EditorVerdict dataclass shape one
    for one. lm-format-enforcer keeps the sampler inside this schema's
    JSON subset, and the consumer json.loads + pydantic-validates the
    raw output (belt and braces).
    """

    pass_decision: bool = Field(
        ...,
        description="True = the draft passes the rubric; False = revise.",
    )
    failing_axes: List[str] = Field(
        default_factory=list,
        description=(
            "Rubric axis json_keys that did NOT pass. Empty when "
            "pass_decision is True."
        ),
    )
    per_axis_notes: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map from axis json_key to a 1-3 sentence note explaining "
            "the score. Every failing axis MUST appear here; passing "
            "axes MAY appear if the Editor wants to flag a near miss."
        ),
    )
    overall_note: str = Field(
        default="",
        max_length=1200,
        description="One paragraph summarising the verdict for the Writer.",
    )
    cycle: int = Field(
        default=0,
        ge=0,
        le=10,
        description="0-indexed Editor pass number for this episode.",
    )


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class EditorCallFailedError(RuntimeError):
    """Raised when the Editor call exhausts its retry budget.

    Pipeline handler (Wave 2 Agent F) decides whether to soft-pass the
    Writer's last draft or hard-fail the room; this module's job is to
    surface the failure loud rather than silently smooth it.
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
            f"Editor pass exhausted retry budget after {attempts} "
            f"attempt(s); last error: "
            f"{type(last_error).__name__}: {last_error}"
        )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


_EDITOR_SYSTEM_PROMPT = (
    "You are the EDITOR in a small writers' room for an old-time "
    "radio drama. Read the DIRECTOR's brief and the WRITER's current "
    "draft, then return a typed verdict scoring the draft against "
    "the rubric axes listed in the user message.\n\n"
    "Your job is to give the Writer per-axis notes that are concrete "
    "enough to act on in one revision turn -- name the line, the "
    "missing premise reference, the unearned beat. Do not rewrite the "
    "draft yourself; the Writer revises in the next turn.\n\n"
    "Output JSON ONLY -- no prose preamble. The pass_decision field "
    "is the load-bearing signal: True ships, False routes to another "
    "Writer turn with your failing_axes + per_axis_notes attached."
)


def _format_rubric_axes_block(rubric: Rubric) -> str:
    """Render the rubric axes for the Editor prompt.

    The Editor is anchored to axes by their json_key (Section 4 Agent
    E). Each axis line names the json_key the Editor MUST use in
    failing_axes / per_axis_notes, the human-readable name, and the
    one-paragraph 'What it checks' definition.
    """
    lines: List[str] = ["RUBRIC AXES (use these json_keys verbatim):"]
    for axis in rubric.axes:
        lines.append(
            f"  - {axis.json_key}  ({axis.name})\n"
            f"      {axis.definition}"
        )
    return "\n".join(lines)


def _format_director_brief_block(brief: Dict[str, Any]) -> str:
    """Render the DirectorBrief fields into a prompt-friendly block."""
    forbidden = brief.get("forbidden_drift") or []
    forbidden_str = (
        "\n".join(f"    - {item}" for item in forbidden)
        if forbidden
        else "    (none)"
    )
    lines = [
        "DIRECTOR'S BRIEF:",
        f"  News premise: {brief.get('news_premise', '')}",
        f"  Dramatic question: {brief.get('dramatic_question', '')}",
        f"  Opposed desires: {brief.get('opposed_desires', '')}",
        f"  Arc shape: {brief.get('arc_shape', '')}",
        f"  Audience feel: {brief.get('audience_feel', '')}",
        "  Forbidden drift:",
        forbidden_str,
    ]
    raw = brief.get("raw_brief", "")
    if raw:
        lines.append("  Raw brief (verbatim):")
        for ln in raw.splitlines() or [""]:
            lines.append(f"    {ln}")
    return "\n".join(lines)


def build_editor_prompt(
    director_brief: Any,
    writer_draft: str,
    rubric: Optional[Rubric] = None,
    cycle: int = 0,
) -> List[dict]:
    """Build the system + user message chain for the Editor call.

    Args:
        director_brief: a DirectorBriefLike object, a dict, or a JSON
            string carrying the dict.
        writer_draft: the current Writer draft as one prose string.
        rubric: optional pre-loaded rubric. Defaults to load_rubric().
        cycle: 0-indexed Editor pass number for this episode. The
            Editor stamps this back on the EditorVerdict so the loop
            can audit per-cycle drift.

    Returns:
        A list of two messages (system, user) ready for generate_fn.
    """
    if rubric is None:
        rubric = load_rubric()

    brief_dict = _coerce_director_brief(director_brief)
    rubric_block = _format_rubric_axes_block(rubric)
    brief_block = _format_director_brief_block(brief_dict)
    draft = (writer_draft or "").strip() or "(empty draft)"

    axis_keys = [a.json_key for a in rubric.axes]
    json_shape = (
        "{\n"
        '  "pass_decision": <true|false>,\n'
        '  "failing_axes": [<axis json_keys that did NOT pass>],\n'
        '  "per_axis_notes": {\n'
        + ",\n".join(
            f'    "{k}": "<1-3 sentence note on this axis>"' for k in axis_keys
        )
        + "\n  },\n"
        '  "overall_note": "<one paragraph summary for the Writer>",\n'
        f'  "cycle": {int(cycle)}\n'
        "}"
    )

    user = (
        f"This is editor cycle {int(cycle)} for this episode.\n\n"
        f"{brief_block}\n\n"
        f"WRITER'S CURRENT DRAFT:\n"
        f"--- begin draft ---\n"
        f"{draft}\n"
        f"--- end draft ---\n\n"
        f"{rubric_block}\n\n"
        "EDITOR INSTRUCTIONS:\n"
        "  1. Decide pass_decision (True = ship this draft as-is, "
        "False = the Writer should revise).\n"
        "  2. List every failing rubric axis by its json_key in "
        "failing_axes. Use the json_keys above verbatim.\n"
        "  3. For each axis in failing_axes, supply a 1-3 sentence "
        "note in per_axis_notes that names what is missing or "
        "broken concretely enough for the Writer to act in one "
        "revision pass. You MAY add notes for passing axes too if "
        "you want to flag a near miss.\n"
        "  4. overall_note: one short paragraph that captures the "
        "verdict in the Writer's voice (no rewrite -- just direction).\n"
        f"  5. Stamp cycle = {int(cycle)} verbatim.\n\n"
        "Output a single JSON object matching this exact shape:\n"
        f"{json_shape}\n\n"
        "JSON only. No prose preamble. Failing_axes MUST be a subset "
        "of the rubric json_keys above."
    )
    return [
        {"role": "system", "content": _EDITOR_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Editor entry point
# ---------------------------------------------------------------------------


def _filter_failing_axes(
    failing_axes: List[str], rubric: Rubric,
) -> List[str]:
    """Drop any axis key the LLM emitted that is not in the rubric.

    Constrained decode keeps the JSON shape valid but does not pin the
    string values inside `failing_axes` to a fixed enum. We belt-and-
    braces drop unknown keys so a downstream consumer can iterate the
    list against the rubric without a KeyError.
    """
    known = {a.json_key for a in rubric.axes}
    return [k for k in failing_axes if k in known]


def run_editor(
    director_brief: Any,
    writer_draft: str,
    *,
    generate_fn: Callable[..., str],
    rubric: Optional[Rubric] = None,
    cycle: int = 0,
    max_attempts: int = _EDITOR_MAX_ATTEMPTS,
) -> EditorVerdict:
    """Run one Editor pass against the current Writer draft.

    Args:
        director_brief: DirectorBriefLike OR dict OR serialized JSON.
        writer_draft: the current draft prose.
        generate_fn: a constrained-decode closure bound to
            EditorVerdictSchema via
            _otr_constrained_generate.make_constrained_generate_fn.
            Signature: (messages, *, temperature, max_new_tokens) -> str.
        rubric: pre-loaded rubric. Defaults to load_rubric().
        cycle: 0-indexed Editor pass number for this episode.
        max_attempts: retry budget. Default 2.

    Returns:
        An EditorVerdict with pass_decision + failing_axes +
        per_axis_notes populated from the LLM output.

    Raises:
        EditorCallFailedError on retry-budget exhaustion. Wave 2's
        Story Room loop decides whether to soft-pass the last draft
        or hard-fail the room.

    # LLM slot: technical
    # Reason: structured constrained-decode pass per project rule 6.
    """
    if rubric is None:
        rubric = load_rubric()

    messages = build_editor_prompt(
        director_brief=director_brief,
        writer_draft=writer_draft,
        rubric=rubric,
        cycle=cycle,
    )

    last_raw = ""
    last_exc: Exception = RuntimeError("no attempt ran")
    attempts_made = 0

    for attempt_idx in range(1, max_attempts + 1):
        attempts_made = attempt_idx
        temperature = (
            _EDITOR_BASE_TEMPERATURE
            if attempt_idx == 1
            else _EDITOR_RETRY_TEMPERATURE
        )
        t0 = time.monotonic()
        log.info(
            "[EditorPass] cycle=%d attempt %d/%d (temperature=%.2f)",
            cycle, attempt_idx, max_attempts, temperature,
        )
        try:
            # LLM slot: technical
            # Reason: structured constrained-decode verdict per
            # project rule 6.
            raw = generate_fn(
                messages,
                temperature=temperature,
                max_new_tokens=_EDITOR_MAX_NEW_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001 -- surface any backend failure
            log.warning(
                "[EditorPass] cycle=%d attempt %d generate raised %s",
                cycle, attempt_idx, type(exc).__name__,
            )
            last_exc = exc
            continue

        last_raw = raw

        # Parse JSON.
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.warning(
                "[EditorPass] cycle=%d attempt %d JSON parse failed",
                cycle, attempt_idx,
            )
            last_exc = exc
            continue

        # Pydantic validate against the schema.
        try:
            parsed = EditorVerdictSchema(**payload)
        except ValidationError as exc:
            log.warning(
                "[EditorPass] cycle=%d attempt %d schema validation "
                "failed: %s",
                cycle, attempt_idx, str(exc)[:200],
            )
            last_exc = exc
            continue

        # Drop unknown axis keys, then build the dataclass.
        failing = _filter_failing_axes(parsed.failing_axes, rubric)
        per_axis = {
            k: v for k, v in parsed.per_axis_notes.items()
            if k in {a.json_key for a in rubric.axes}
        }
        verdict = EditorVerdict(
            pass_decision=bool(parsed.pass_decision),
            failing_axes=failing,
            per_axis_notes=per_axis,
            overall_note=parsed.overall_note or "",
            cycle=int(parsed.cycle if parsed.cycle is not None else cycle),
        )
        elapsed = time.monotonic() - t0
        log.info(
            "[EditorPass] cycle=%d attempt %d pass_decision=%s "
            "failing_axes=%s elapsed=%.2fs",
            cycle, attempt_idx, verdict.pass_decision,
            verdict.failing_axes, elapsed,
        )
        return verdict

    raise EditorCallFailedError(
        attempts=attempts_made,
        last_raw_output=last_raw,
        last_error=last_exc,
    )


__all__ = [
    "DirectorBriefLike",
    "EditorVerdict",
    "EditorVerdictSchema",
    "EditorCallFailedError",
    "build_editor_prompt",
    "run_editor",
]
