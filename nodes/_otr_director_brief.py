"""nodes/_otr_director_brief.py -- Sprint 10B Wave 1 Agent D (2026-05-27).

The Director agent. Reads a news seed + locked cast names + (optionally)
the whole-episode critic rubric, and emits a typed `DirectorBrief`: a
short prose direction the Writer (Wave 2 Agent F) can act on.

Design references:
- `docs/2026-05-26-good-story-writer-architecture__02_design.md`
  Section 4 Wave 1 Agent D ("Director agent"): owns this module +
  `nodes/OTR_DirectorBrief.py` + `tests/test_otr_director_brief.py`.
- Section 3.1 freezes the `DirectorBrief` dataclass shape that Wave 1
  Agent E (Editor) and Wave 2 Agent F (Story Room loop) consume.
- Section 2.5's worked example for "Spray of Hope" is the load-bearing
  illustration of what a good Director brief looks like: a named news
  premise, a one-sentence dramatic question, opposed desires, an arc
  shape, an audience feel, and a typed `forbidden_drift` list of
  failure shapes to avoid.

This module is PURE: no I/O, no GPU, no ComfyUI imports, no writer
imports. The only LLM call site is `run_director`, tagged
`# LLM slot: technical` since this is a structured constrained-decode
pass (per project rule 6: structured / JSON passes route to the
technical slot).

Per Prime Directive 6 (PD6) the module exposes no `model_id` widget;
the consuming node (`nodes/OTR_DirectorBrief.py`) receives the
technical-slot model id over a STRING input socket wired from the
writer's `technical_model` broadcast output.

Wave 1 lands this module + node + tests DORMANT. Wave 2 Agent F wires
the Director's output into the Story Room loop. Today's pipeline is
unchanged.

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
4-space indentation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, Field, ValidationError


log = logging.getLogger("OTR")


__all__ = [
    "DirectorBrief",
    "DirectorBriefSchema",
    "DirectorCallFailedError",
    "build_director_prompt",
    "run_director",
]


# ---------------------------------------------------------------------------
# Constants -- LLM ladder settings for the Director pass
# ---------------------------------------------------------------------------

# The Director pass is structured-JSON extraction, not creative prose;
# a low base temperature keeps the model honest about the schema while
# leaving a small amount of headroom for varied phrasing across the six
# prose fields. The structural-retry temperature is STRICTLY LOWER per
# the 2B principle (a JSON-schema re-roll lowers entropy, never raises
# it).
_DIRECTOR_BASE_TEMPERATURE: float = 0.35
_DIRECTOR_STRUCTURAL_RETRY_TEMPERATURE: float = 0.15

# Token budget. Six short prose fields + a 0-5 entry list of forbidden
# drift phrases + a raw_brief paragraph fit comfortably under 700
# tokens; headroom keeps a model from being truncated mid-object.
_DIRECTOR_MAX_NEW_TOKENS: int = 768

# Per-field hard caps for the prose. Each field is "one to three short
# sentences" -- a few hundred chars per field is the natural envelope.
# The lower bound rejects an empty / placeholder response from the LLM
# without driving the constrained decoder into an infinite retry on
# fields that legitimately want to be terse.
_PROSE_FIELD_MIN_CHARS: int = 20
_PROSE_FIELD_MAX_CHARS: int = 400

# raw_brief is the full paragraph the model writes for context-hungry
# downstream consumers (Story Room transcript audit). One short
# paragraph; bounded.
_RAW_BRIEF_MIN_CHARS: int = 40
_RAW_BRIEF_MAX_CHARS: int = 1200

# forbidden_drift is a small typed list of short phrases naming the
# failure shapes to avoid (e.g. "code-red theatrics"). 0..5 entries;
# each entry capped so the model cannot smuggle a paragraph in here.
_FORBIDDEN_DRIFT_MAX_ENTRIES: int = 5
_FORBIDDEN_DRIFT_PHRASE_MAX_CHARS: int = 80


# ---------------------------------------------------------------------------
# DirectorBrief dataclass -- the frozen contract (design Section 3.1)
# ---------------------------------------------------------------------------


@dataclass
class DirectorBrief:
    """The Director's brief for one episode.

    Free-form prose at the field level: the LLM writes each field as
    one-to-three sentences. The structure exists so downstream agents
    (Writer, Editor) can pull specific fields without re-parsing.

    Field shapes (per design Section 3.1):
      news_premise       what the episode IS about; 1-2 sentences.
      dramatic_question  what's at stake; 1 sentence.
      opposed_desires    whose desire opposes whose; 1-2 sentences.
      arc_shape          what changes between open and close; 1-2 sentences.
      audience_feel      what the audience should feel; 1 sentence.
      forbidden_drift    things to NOT drift toward; 0-5 short phrases.
      raw_brief          full prose brief; for context-hungry agents.

    The dataclass is the cross-module wire format (Wave 1 Agent E reads
    it, Wave 2 Agent F consumes it). The Pydantic mirror below
    (`DirectorBriefSchema`) is the constrained-decode shape only.
    """

    news_premise: str
    dramatic_question: str
    opposed_desires: str
    arc_shape: str
    audience_feel: str
    forbidden_drift: List[str] = field(default_factory=list)
    raw_brief: str = ""

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialization.

        The Director node emits this dict on its STRING output socket;
        downstream agents reconstruct a `DirectorBrief` from it via
        `DirectorBrief(**payload)`.
        """
        return asdict(self)


# ---------------------------------------------------------------------------
# Pydantic schema -- the constrained-decode shape mirroring DirectorBrief
# ---------------------------------------------------------------------------


class DirectorBriefSchema(BaseModel):
    """Constrained-decode schema for the Director's structured output.

    Mirrors the `DirectorBrief` dataclass field-for-field. The Pydantic
    model exists so the existing `make_constrained_generate_fn` pipeline
    (sibling of Stage 1 plan, continuity ledger, whole-episode critic)
    can bind a JsonSchemaParser to it and constrain the sampler to a
    schema-valid token sequence.

    Length bounds are tight enough that an empty or runaway response
    fails the schema gate immediately; loose enough that one-sentence
    fields and three-sentence fields both fit.
    """

    news_premise: str = Field(
        ...,
        min_length=_PROSE_FIELD_MIN_CHARS,
        max_length=_PROSE_FIELD_MAX_CHARS,
        description=(
            "What the episode IS about. Name the news premise in one or "
            "two sentences. A listener should be able to identify the "
            "underlying news story from this field alone."
        ),
    )
    dramatic_question: str = Field(
        ...,
        min_length=_PROSE_FIELD_MIN_CHARS,
        max_length=_PROSE_FIELD_MAX_CHARS,
        description=(
            "The single dramatic question of the episode. One sentence. "
            "Frame what is at stake for the protagonist."
        ),
    )
    opposed_desires: str = Field(
        ...,
        min_length=_PROSE_FIELD_MIN_CHARS,
        max_length=_PROSE_FIELD_MAX_CHARS,
        description=(
            "Whose desire opposes whose. One or two sentences naming the "
            "two characters and the specific incompatibility between "
            "what each wants."
        ),
    )
    arc_shape: str = Field(
        ...,
        min_length=_PROSE_FIELD_MIN_CHARS,
        max_length=_PROSE_FIELD_MAX_CHARS,
        description=(
            "What changes between the episode's open and its close. One "
            "or two sentences describing the arc: setup, turn, landing."
        ),
    )
    audience_feel: str = Field(
        ...,
        min_length=_PROSE_FIELD_MIN_CHARS,
        max_length=_PROSE_FIELD_MAX_CHARS,
        description=(
            "What the audience should feel at the end. One sentence "
            "naming the emotional target (e.g. 'quiet weight', not "
            "'action')."
        ),
    )
    forbidden_drift: List[str] = Field(
        default_factory=list,
        max_length=_FORBIDDEN_DRIFT_MAX_ENTRIES,
        description=(
            "Zero to five short phrases naming failure shapes the Writer "
            "must avoid (e.g. 'code-red theatrics', 'generic medical "
            "crisis'). Each phrase under 80 characters."
        ),
    )
    raw_brief: str = Field(
        default="",
        max_length=_RAW_BRIEF_MAX_CHARS,
        description=(
            "Full prose brief, one short paragraph, for context-hungry "
            "downstream agents (e.g. Story Room transcript audit). May "
            "restate the structured fields in natural prose."
        ),
    )


# ---------------------------------------------------------------------------
# Failure mode -- exhausted retry ladder
# ---------------------------------------------------------------------------


class DirectorCallFailedError(RuntimeError):
    """Raised when `run_director` exhausts its retry budget.

    Mirrors the StructuredCallFailedError contract: carries the helper
    name, attempts made, and the last error observed. The Director call
    fails LOUD; the consuming node decides whether to retry the workflow
    or stamp the failure on `meta.director` for forensic inspection.
    """

    def __init__(
        self,
        *,
        attempts: int,
        last_error: Optional[BaseException],
    ) -> None:
        self.attempts = attempts
        self.last_error = last_error
        last_text = (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else "no error captured"
        )
        super().__init__(
            f"[OTR_DirectorBrief] 'run_director' failed after "
            f"{attempts} attempt(s); last error -> {last_text}"
        )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


_DIRECTOR_SYSTEM_PROMPT = """\
You are a director briefing a writers' room for a short audio drama.
The writers will use your brief to draft one episode of a single-take
two-hander. Your job is to convert a real news seed plus a locked cast
into a concrete dramatic direction the writers can act on.

You are NOT a form-filler. Write each field as natural prose, the way
a director would speak to writers across a table. But return your
direction as ONE JSON object with the field shape below. No Markdown,
no preamble, no commentary outside the JSON.

{
  "news_premise":      "what the episode IS about, 1-2 sentences",
  "dramatic_question": "what is at stake, 1 sentence",
  "opposed_desires":   "whose desire opposes whose, 1-2 sentences",
  "arc_shape":         "what changes between open and close, 1-2 sentences",
  "audience_feel":     "what the audience should feel, 1 sentence",
  "forbidden_drift":   ["0-5 short phrases naming failure shapes to avoid"],
  "raw_brief":         "one short paragraph restating the direction in prose"
}

RULES:
- The news premise must be NAMED concretely. A listener who never read
  the news seed must be able to identify the underlying story from the
  episode's dialogue alone. Pin the specific detail (the spray, the
  exoplanet, the bubble-net technique), not a generic frame.
- Opposed desires must be specific, not abstract. Name BOTH characters
  from the cast list and say exactly what each wants.
- forbidden_drift is the typed list of failure shapes the writers must
  avoid -- the generic dramatic patterns this premise will tempt them
  toward. If you do not have specific drift to call out, return an
  empty list, not a placeholder.
- audience_feel is one emotional target, not a list. "quiet weight" is
  good; "exciting and thrilling and moving" is not.
- Use no profanity. Keep the brief safe for work.
"""


def _format_cast_list(cast_names: List[str]) -> str:
    """Render the cast list for the user prompt.

    The Director consumes cast NAMES, not full Stage1CastMember rows;
    upstream cast-lock is what produces the names. An empty / missing
    cast list is allowed: the Director can still name a premise, but
    the opposed_desires field will be flagged downstream when no two
    characters can be named.
    """
    rows = [str(n).strip() for n in (cast_names or []) if str(n).strip()]
    if not rows:
        return "(no cast names supplied)"
    return "\n".join(f"- {name}" for name in rows)


def _format_rubric_axes(rubric: Any) -> str:
    """Render the critic-rubric axis names as a hint block.

    The Director is told which axes the Editor will later score the
    episode against, so the brief sets up an episode that can pass the
    critic. The block is informational, not a contract; the Director
    does not score, only directs.

    `rubric` may be a `_otr_critic_rubric.Rubric` (the production
    shape) OR None (when callers want to skip the hint). Duck-typed so
    a test fixture can pass a stub object with an `axes` attribute.
    """
    if rubric is None:
        return ""
    axes = getattr(rubric, "axes", None) or []
    if not axes:
        return ""
    names: list[str] = []
    for axis in axes:
        # Prefer json_key (slug); fall back to name (mixed case) so a
        # stub fixture with only `name` still renders.
        slug = getattr(axis, "json_key", "") or getattr(axis, "name", "")
        if slug:
            names.append(str(slug))
    if not names:
        return ""
    return (
        "Critic rubric axes (the Editor will later score the episode "
        "against these; design your brief so the writers can hit them):\n"
        + "\n".join(f"- {name}" for name in names)
    )


def build_director_prompt(
    news_seed: str,
    cast_names: List[str],
    rubric: Any = None,
) -> List[dict]:
    """Assemble the [system, user] message list for the Director call.

    Args:
        news_seed: the news premise the episode is about. Plain prose
            (one or two sentences is typical -- the upstream news
            interpreter normalizes raw RSS bodies before this point).
        cast_names: the locked cast for this episode, as a list of
            character names. The Director uses these names verbatim
            in the opposed_desires field.
        rubric: optional `_otr_critic_rubric.Rubric` instance. When
            present, the rubric's axis json_keys are appended to the
            user prompt so the Director's brief sets up an episode the
            critic can pass. When None, the rubric hint is omitted.

    Returns:
        A two-message list: a system prompt framing the LLM as a
        director, and a user prompt carrying the news seed, cast, and
        optional rubric hint.

    The function is PURE -- no LLM call, no I/O. Tests bind the output
    to a mock `generate_fn` to drive `run_director` without a live
    model.
    """
    seed = (news_seed or "").strip()
    cast_block = _format_cast_list(cast_names)
    rubric_block = _format_rubric_axes(rubric)

    parts: list[str] = [
        "News seed (the real story this episode is about):",
        seed if seed else "(no news seed supplied)",
        "",
        "Cast (use these names verbatim in opposed_desires):",
        cast_block,
    ]
    if rubric_block:
        parts.append("")
        parts.append(rubric_block)
    parts.append("")
    parts.append("Return the director's brief as ONE JSON object now.")

    user_message = "\n".join(parts)
    return [
        {"role": "system", "content": _DIRECTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


# ---------------------------------------------------------------------------
# run_director -- the single LLM call (fails LOUD on exhausted retries)
# ---------------------------------------------------------------------------


def _attempt_parse_and_validate(raw: str) -> DirectorBriefSchema:
    """Parse `raw` as JSON and validate against `DirectorBriefSchema`.

    Returns the validated Pydantic instance on success. Raises
    `json.JSONDecodeError` on unparseable output or
    `pydantic.ValidationError` on schema mismatch. The Director's
    retry loop catches both at the same boundary.
    """
    payload = json.loads(raw or "")
    return DirectorBriefSchema.model_validate(payload)


def _schema_to_brief(schema: DirectorBriefSchema) -> DirectorBrief:
    """Convert a validated `DirectorBriefSchema` to a `DirectorBrief`.

    Strips whitespace on prose fields and clamps the `forbidden_drift`
    entries to the documented per-phrase length so the wire payload
    stays bounded even if the Pydantic length cap is later relaxed.
    """
    drift_clean: List[str] = []
    for phrase in (schema.forbidden_drift or []):
        s = str(phrase or "").strip()
        if not s:
            continue
        if len(s) > _FORBIDDEN_DRIFT_PHRASE_MAX_CHARS:
            s = s[:_FORBIDDEN_DRIFT_PHRASE_MAX_CHARS].rstrip()
        drift_clean.append(s)
    return DirectorBrief(
        news_premise=schema.news_premise.strip(),
        dramatic_question=schema.dramatic_question.strip(),
        opposed_desires=schema.opposed_desires.strip(),
        arc_shape=schema.arc_shape.strip(),
        audience_feel=schema.audience_feel.strip(),
        forbidden_drift=drift_clean,
        raw_brief=(schema.raw_brief or "").strip(),
    )


def run_director(
    news_seed: str,
    cast_names: List[str],
    *,
    generate_fn: Callable[..., str],
    rubric: Any = None,
    max_attempts: int = 2,
) -> DirectorBrief:
    """Run the Director pass and return a populated `DirectorBrief`.

    Args:
        news_seed: the news premise the episode is about.
        cast_names: the locked cast names for this episode.
        generate_fn: a ConstrainedGenerateFn bound to
            `DirectorBriefSchema` (built by
            `_otr_constrained_generate.make_constrained_generate_fn`).
            Signature: `(messages, *, temperature, max_new_tokens) -> str`.
        rubric: optional `_otr_critic_rubric.Rubric` -- the rubric the
            Editor will later score the episode against. When supplied,
            the axis json_keys appear in the prompt as a design hint.
        max_attempts: retry budget. Default 2 -- one base attempt plus
            one structural-retry at lower temperature.

    Returns:
        A populated `DirectorBrief` on the first attempt that yields a
        schema-valid object.

    Raises:
        `DirectorCallFailedError` when the retry budget exhausts. The
        consuming node decides whether to surface this as a hard
        workflow failure or stamp it on `meta.director`. (Wave 2 Agent
        F's Story Room loop will be the production consumer; today's
        DORMANT node simply propagates the error.)
    """
    messages = build_director_prompt(news_seed, cast_names, rubric=rubric)

    last_error: Optional[BaseException] = None
    attempts = 0

    for attempt_idx in range(1, max_attempts + 1):
        attempts = attempt_idx
        temperature = (
            _DIRECTOR_BASE_TEMPERATURE
            if attempt_idx == 1
            else _DIRECTOR_STRUCTURAL_RETRY_TEMPERATURE
        )
        log.info(
            "[OTR_DirectorBrief] attempt %d/%d (temperature=%.2f)",
            attempt_idx, max_attempts, temperature,
        )
        try:
            # LLM slot: technical
            # Reason: Director brief is a structured constrained-decode
            # pass (JSON object validated against DirectorBriefSchema)
            # per project rule 6; routes to the writer's technical slot.
            raw = generate_fn(
                messages,
                temperature=temperature,
                max_new_tokens=_DIRECTOR_MAX_NEW_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001 -- slot fn (LLM call) varies
            log.warning(
                "[OTR_DirectorBrief] attempt %d generate_fn raised %s: %s",
                attempt_idx, type(exc).__name__, str(exc)[:200],
            )
            last_error = exc
            continue

        try:
            schema = _attempt_parse_and_validate(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            log.warning(
                "[OTR_DirectorBrief] attempt %d parse/validate failed: %s",
                attempt_idx, str(exc)[:200],
            )
            last_error = exc
            continue

        brief = _schema_to_brief(schema)
        log.info(
            "[OTR_DirectorBrief] success at attempt %d "
            "(news_premise %d chars, %d forbidden_drift entries)",
            attempt_idx, len(brief.news_premise), len(brief.forbidden_drift),
        )
        return brief

    raise DirectorCallFailedError(attempts=attempts, last_error=last_error)
