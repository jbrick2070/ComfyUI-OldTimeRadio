"""nodes/_otr_outline.py

Grammar-validated outline generation for the v2.0 LedgerScriptWriter path.

Scope: science-fiction audio drama outlines grounded in real science
stories. The user supplies the science seed and a free-form style
descriptor; the LLM picks whatever dialogue register fits. NO period
anchoring -- no 1940s coaxing, no era constraints. The local model uses
its own trained distribution for dialogue style.

Pydantic schema for Beat[] + JSON-mode prompting + parse-or-reroll-or-repair
loop. NOT using lm-format-enforcer (compat unverified against transformers
5.x and the lib has been quiet for ~9 months); we get the same reliability
via deterministic post-hoc validation with a 3-attempt retry budget where
the third attempt is a repair call.

Status: Phase 1 of v2.0 sprint. Does NOT touch the in-flight legacy path.
Caller (eventual OTR_LedgerScriptWriter) is responsible for loading the
model via story_orchestrator._load_llm and passing the handle in. This
module does not load models.

Public surface:
    Beat                  -- pydantic model: one outline beat
    Outline               -- pydantic model: full episode outline
    OutlineRequest        -- frozen dataclass: input parameters
    OutlineFailedError    -- raised after 3 failed attempts
    generate_outline(...) -- main entrypoint
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

log = logging.getLogger("OTR")


__all__ = [
    "Beat",
    "Outline",
    "OutlineRequest",
    "OutlineFailedError",
    "generate_outline",
    "OutlineBudgetViolation",
    "validate_outline_against_budget",
]


# ---------------------------------------------------------------------------
# SpeakerRole literal -- mirror of _otr_speaker_role.VALID_SPEAKER_ROLES
# ---------------------------------------------------------------------------

# Mirror of _otr_speaker_role.VALID_SPEAKER_ROLES. Duplicated to keep
# this module's imports stdlib+pydantic only at module load. The
# _check_speaker_role_alignment() function below verifies equality
# on first use of generate_outline and logs if the constants drift.
SpeakerRole = Literal[
    "character",
    "announcer",
    "music_open",
    "music_close",
    "music_inter",
    "sfx",
]


# ---------------------------------------------------------------------------
# Beat schema
# ---------------------------------------------------------------------------


class Beat(BaseModel):
    """One beat of the outline. Lines are generated 1:1 from beats."""

    beat_id: str = Field(
        ...,
        pattern=r"^b\d{3}$",
        description="Stable ID, format 'b001', 'b002', monotonic per outline",
    )
    speaker: str = Field(
        ...,
        min_length=1,
        max_length=40,
        description="Character name in ALL CAPS, or 'NARRATOR' for music/sfx beats",
    )
    speaker_role: SpeakerRole = Field(
        ...,
        description="Routing role; see _otr_speaker_role for HuMo vs LTX-radio dispatch",
    )
    intent: str = Field(
        ...,
        min_length=4,
        max_length=200,
        description="What this beat accomplishes narratively, one sentence",
    )
    target_words: int = Field(
        ...,
        ge=3,
        le=80,
        description="Approximate word count for the dialogue line",
    )
    mood: str = Field(
        ...,
        min_length=2,
        max_length=40,
        description="Tone descriptor, e.g. 'tense', 'wry', 'foreboding'",
    )
    sfx_cue: Optional[str] = Field(
        default=None,
        max_length=80,
        description="Optional [SFX:] hint for the surrounding line",
    )
    arc_phase: str = Field(
        default="setup",
        max_length=40,
        description=(
            "Phase 2A (2026-05-11): narrative phase label from "
            "EpisodeBudget.arc_phases (setup / complication / "
            "resolution / climax / etc.). Required-with-default per "
            "the post-Phase-3 review pass (Strategy A). A 12B model "
            "like Mistral-Nemo frequently omits Optional pydantic "
            "fields; making the field required with a 'setup' "
            "default guarantees it is always populated, and the "
            "downstream validator catches any beat whose default "
            "value is wrong (membership / ordering check) on the "
            "first attempt instead of relying on the reroll-repair "
            "loop to coax the LLM into re-emitting the field. "
            "Back-compat outlines (pre-Phase 2A, no budget) skip "
            "the membership check entirely; default is harmless. "
            "Original D1 critique reversed on 2026-05-11 after the "
            "reviewer correctly observed that the validator-after-"
            "default path is bounded and converges."
        ),
    )

    @field_validator("speaker")
    @classmethod
    def _speaker_uppercase(cls, v: str) -> str:
        return v.strip().upper()


# ---------------------------------------------------------------------------
# Outline schema
# ---------------------------------------------------------------------------


class Outline(BaseModel):
    """Full episode outline. The Outline IS the macro-plan; line composer
    consumes Beat-by-Beat and writes the ledger row by row.

    Cast-contract architecture (2026-05-10): the outline schema does
    NOT carry a `cast` field. Cast is INGESTED from the writer's
    locked cast contract (`_otr_casting.lock_cast`) via OutlineRequest
    .character_cast — never produced by the outline LLM. The cast-
    membership check on character-role beats lives in
    `generate_outline()` and validates against `req.character_cast`,
    NOT against any internal `self.cast` field. Less for the small
    local LLM to lift per call.
    """

    title: str = Field(..., min_length=3, max_length=80)
    premise: str = Field(..., min_length=10, max_length=400)
    setting: str = Field(..., min_length=4, max_length=120)
    time_of_day: str = Field(..., min_length=3, max_length=40)
    # Phase 2A (2026-05-11) raised max from 24 -> 32 so 6- and 7-act
    # outlines (synthesis §3 Phase 2A beat-count table) still fit
    # within the schema cap with music_inter beats.
    beats: list[Beat] = Field(..., min_length=4, max_length=32)

    @model_validator(mode="after")
    def _no_duplicate_beat_ids(self) -> "Outline":
        """Schema-internal sanity check. Cast-membership cross-check
        moved to generate_outline (validates beat speakers against
        req.character_cast, the locked cast)."""
        ids = [b.beat_id for b in self.beats]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate beat_ids in outline: {ids}")
        return self


# ---------------------------------------------------------------------------
# OutlineRequest -- frozen input parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutlineRequest:
    """Input parameters for generate_outline. Frozen so call sites
    can't accidentally mutate after construction.

    Cast contract (2026-05-10): the cast is no longer produced by the
    outline LLM. The writer locks the cast FIRST via
    nodes/_otr_casting.lock_cast() and passes the character names
    into this request via `character_cast`. The outline LLM is told
    those names are the cast it MUST use; a post-validation guard
    rejects any outline that drifts.
    """

    news_seed: str           # The real science story / factual seed.
                             # Back-compat field: callers who have no
                             # news_interpreter brief (e.g. early-stage
                             # tests, or the writer's fallback path
                             # when build_news_briefs raised) pass the
                             # raw seed here. When script_brief is
                             # non-empty it takes precedence in the
                             # prompt.
    style: str               # User-selected style, e.g. "psychological slow-burn",
                             # "pulp adventure", "hard sci-fi procedural", "noir thriller".
                             # Field renamed from style_hint 2026-05-10 — Jeffrey:
                             # "no 'hint', it's just style". User-visible widget name
                             # is 'style', so the dataclass field matches.
    target_words: int        # Canonical length unit (validated below). Words are
                             # the single source of truth for story planning;
                             # there is no seconds field — see Jeffrey 2026-05-10.
    character_cast: tuple[str, ...]
                             # ALL-CAPS character names from the LOCKED cast.
                             # Excludes ANNOUNCER (the writer hardcodes
                             # speaker="ANNOUNCER" on announcer-role beats so
                             # the LLM never needs to handle ANNOUNCER itself).
                             # 1-6 names. Validated below. NO default --
                             # callers MUST supply this. (Removing the default
                             # was a round-robin 2026-05-10 nit: an empty-tuple
                             # default would crash __post_init__ immediately,
                             # which is a worse failure mode than a clear
                             # TypeError from the dataclass constructor.)
    script_brief: str = ""
                             # OPTIONAL. news_interpreter's purpose-specific
                             # distillation of the article for script planning
                             # (premise arc, central tension, beat hooks).
                             # When non-empty, the prompt routes through the
                             # "Story brief" branch with a "develops this
                             # brief" closing verb -- because the brief is a
                             # distilled story plan, not raw factual material.
                             # When empty, the prompt falls back to news_seed
                             # under the "Science story (the factual seed)"
                             # label with the original "extrapolates from the
                             # science story" verb. Commit 3 (news_interpreter
                             # sprint, ADR docs/news_interpreter_adr.md);
                             # branch added in the post-sprint prompt
                             # tightening pass (2026-05-10).
    key_terms: tuple[str, ...] = ()
                             # OPTIONAL. news_interpreter's verbatim
                             # journalistic terms (people, places, technology)
                             # the dialogue MUST surface. Injected into the
                             # prompt as a "Required terms" line when non-
                             # empty so the outline can plan beats that
                             # naturally land them.
    # target_length field removed 2026-05-11 (post-Phase-3 cleanup
    # pass). The writer's target_length widget went with it; act-
    # count signal now flows via the `budget` field
    # (EpisodeBudget from compute_episode_budget).
    include_act_breaks: bool = True
                             # OPTIONAL. Mirrors the writer's
                             # include_act_breaks widget. Affects
                             # whether the outline LLM is told to plan
                             # music_inter beats. The EpisodeBudget
                             # (when `budget` is non-None) is the
                             # authoritative source for music_inter
                             # count; this flag is the user-facing
                             # toggle that drives it.
    budget: object = None
                             # Phase 2A (2026-05-11). Optional
                             # _otr_episode_budget.EpisodeBudget. When
                             # non-None, the outline prompt gets an
                             # "EPISODE BUDGET" block and the post-
                             # pydantic pipeline runs the 8 Phase 2A
                             # validators (per-phase word totals,
                             # per-phase beat counts, per-beat word
                             # range, arc_phase ordering, music_inter
                             # count, announcer count) and rerolls on
                             # failure. Validator #1 (total word
                             # drift) is WARN-only at ±25% per §6.E.
                             # Stored as `object` to keep this module
                             # importable without pulling
                             # _otr_episode_budget at module load --
                             # `_get_budget(req)` does a lazy
                             # isinstance / duck-type check at call
                             # time. None preserves pre-Phase-2A
                             # back-compat for tests and early-stage
                             # callers.
    cast_descriptions: tuple[tuple[str, str, str], ...] = ()
                             # OPTIONAL. Per-character (name, gender,
                             # character_description) tuples from the
                             # LOCKED cast (the cast LLM's output via
                             # _otr_casting.lock_cast). When non-empty,
                             # the prompt's `Cast` line expands from a
                             # bare name list to a per-character block
                             # with the description so the outline LLM
                             # can plan beats that exploit each
                             # character's distinct personality + stakes
                             # (instead of writing generic-sci-fi-
                             # character beats keyed only on ALL-CAPS
                             # names). When empty, the prompt falls
                             # back to the bare name list (back-compat
                             # for tests + early-stage callers).
                             #
                             # MUST match character_cast 1:1 in name
                             # and order when non-empty (validated in
                             # __post_init__) so the LLM doesn't see
                             # contradictory cast info between the
                             # constraint sentence and the description
                             # block.
                             #
                             # Wired by OTR_LedgerScriptWriter D.5
                             # post-cast-lock (2026-05-10 follow-up).

    def __post_init__(self) -> None:
        n = len(self.character_cast)
        if not (1 <= n <= 6):
            raise ValueError(
                f"character_cast must have 1-6 names, got {n}: "
                f"{self.character_cast!r}"
            )
        if self.target_words < 5:
            raise ValueError(
                f"target_words must be >= 5, got {self.target_words}"
            )
        for name in self.character_cast:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"character_cast names must be non-empty strings, "
                    f"got {name!r}"
                )
            if name != name.upper():
                raise ValueError(
                    f"character_cast names must be ALL CAPS, got {name!r}"
                )
        if self.cast_descriptions:
            if len(self.cast_descriptions) != len(self.character_cast):
                raise ValueError(
                    f"cast_descriptions length {len(self.cast_descriptions)} "
                    f"!= character_cast length {len(self.character_cast)}; "
                    f"the two lists must align 1:1"
                )
            for i, entry in enumerate(self.cast_descriptions):
                if (not isinstance(entry, tuple)) or len(entry) != 3:
                    raise ValueError(
                        f"cast_descriptions[{i}] must be a 3-tuple "
                        f"(name, gender, description), got {entry!r}"
                    )
                name, gender, desc = entry
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(
                        f"cast_descriptions[{i}].name must be a "
                        f"non-empty string, got {name!r}"
                    )
                if name != self.character_cast[i]:
                    raise ValueError(
                        f"cast_descriptions[{i}].name {name!r} != "
                        f"character_cast[{i}] {self.character_cast[i]!r}; "
                        f"the two lists must align 1:1 in name and order"
                    )
                if not isinstance(gender, str):
                    raise ValueError(
                        f"cast_descriptions[{i}].gender must be a "
                        f"string, got {gender!r}"
                    )
                if not isinstance(desc, str):
                    raise ValueError(
                        f"cast_descriptions[{i}].description must be a "
                        f"string, got {desc!r}"
                    )

    @property
    def cast_size(self) -> int:
        """Back-compat accessor. Reads len(character_cast)."""
        return len(self.character_cast)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a story editor. You produce JSON outlines for short science-fiction
audio dramas grounded in real science stories.

Your job is to plan the episode, not write the dialogue. Each beat names the
speaker, what they accomplish narratively, target word count, and mood. The
dialogue itself is generated by a separate process and will use whatever
register fits the story and style.

OUTPUT FORMAT
- Return exactly one JSON object.
- No prose before or after the JSON.
- No markdown code fences.
- The JSON must match this schema:

{
  "title":       string (3-80 chars),
  "premise":     string (10-400 chars),
  "setting":     string (4-120 chars),
  "time_of_day": string (3-40 chars),
  "beats":       array of 4-24 beat objects, where each beat is:
                 {
                   "beat_id":      "b001", "b002", ... (monotonic),
                   "speaker":      ALL-CAPS name from the Cast block in
                                   the user prompt, or "NARRATOR" for
                                   music / sfx beats,
                   "speaker_role": one of: "character", "announcer",
                                   "music_open", "music_close",
                                   "music_inter", "sfx",
                   "intent":       one sentence describing what the beat
                                   accomplishes narratively (4-200 chars),
                   "target_words": integer 3-80,
                   "mood":         tone descriptor (2-40 chars),
                   "sfx_cue":      optional string (max 80 chars), or null
                 }

CONSTRAINTS
- Every beat with speaker_role "character" MUST have a speaker that
  appears in the Cast block of the user prompt. Do not invent new
  characters.
- The first beat is typically speaker_role "music_open" or "announcer".
- The last beat is typically speaker_role "music_close" or "announcer".
- Beats should follow a clear arc: setup, complication, resolution.
- The premise must be grounded in the science story provided in the
  user prompt -- extrapolate dramatically from it, do not contradict it.
- Do not include the dialogue text in the outline. Only the intent.
"""


def _build_user_prompt(req: OutlineRequest) -> str:
    # news_interpreter brief takes precedence over raw news_seed.
    # When the writer has a script_brief from build_news_briefs, the
    # prompt labels the source line as a brief (it already contains
    # the distilled premise arc + central tension + beat hooks) and
    # the closing verb says DEVELOPS the brief, not EXTRAPOLATES from
    # raw material -- the dramatic extrapolation is already done.
    # When the writer is on the graceful-degrade path (brief LLM call
    # failed), the original "Science story (the factual seed)" label
    # + "extrapolates" verb still apply to the raw RSS payload.
    brief = req.script_brief.strip()
    if brief:
        source_line = f"Story brief: {brief}"
        develop_verb = "develops this brief"
    else:
        source_line = f"Science story (the factual seed): {req.news_seed}"
        develop_verb = "extrapolates from the science story"
    parts = [
        "Plan a science-fiction audio drama outline.",
        "",
        source_line,
    ]
    if req.key_terms:
        terms_line = ", ".join(req.key_terms)
        # The outline LLM writes intent + mood, not dialogue lines
        # (the line composer does that). Right plane to address: the
        # beats it plans must be ones that NATURALLY surface these
        # terms when the line composer renders them. Post-assembly
        # key_terms audit (commit 4) is what enforces presence in
        # the finished dialogue.
        parts.append(
            f"Required terms (plan beats that surface these in "
            f"dialogue): {terms_line}"
        )
    # Cast block: rich (per-character name + gender + description)
    # when cast_descriptions is present, bare name list otherwise.
    # Rich format gives the outline LLM enough character signal to
    # plan beats that exploit each character's distinct personality
    # + stakes; bare format is a back-compat fallback for tests +
    # early-stage callers that pre-date the cast contract.
    parts.append(_format_cast_block(req))
    parts.append(f"Style: {req.style}")
    # target_length structure line removed 2026-05-11 (post-Phase-3
    # cleanup). The act-count signal now flows entirely through the
    # EPISODE BUDGET block below (when `budget` is non-None); the
    # include_act_breaks toggle drives music_inter_count inside the
    # budget rather than appearing in its own prose line.
    # Phase 2A (2026-05-11): EPISODE BUDGET block. Lands BEFORE the
    # target_words summary so the LLM sees concrete numbers for every
    # phase + beat-range before being told the rough total. Skipped
    # entirely when budget is None (back-compat).
    budget_block = _format_episode_budget_block(req)
    if budget_block:
        parts.append(budget_block)
        parts.append("")

    parts.extend([
        f"Target total dialogue length: ~{req.target_words} words "
        f"(sum of per-beat target_words should land near this number).",
        "",
    ])
    head = "\n".join(parts)
    return (
        f"{head}\n"
        f"Build a dramatic outline that {develop_verb} in the chosen "
        f"style. Return only the JSON outline."
    )


def _format_cast_block(req: OutlineRequest) -> str:
    """Render the Cast block of the outline user prompt.

    Two shapes:
      Rich (when cast_descriptions present):
          Cast (already chosen -- use exactly these names in
          character-role beats):
          - ALICE (female, weary forensic engineer in her 40s, dry humor)
          - BOB (male, ambitious grant officer in his 30s, evasive)

      Bare (back-compat when cast_descriptions empty):
          Cast (already chosen -- use exactly these names in
          character-role beats): ALICE, BOB

    The rich format gives the outline LLM enough character signal
    to plan beats that exploit each character's distinct
    personality + stakes (instead of writing generic-sci-fi-
    character beats keyed only on ALL-CAPS names). __post_init__
    has already validated 1:1 alignment between cast_descriptions
    and character_cast when the rich path is taken.
    """
    header = (
        "Cast (already chosen -- use exactly these names in "
        "character-role beats):"
    )
    if not req.cast_descriptions:
        return f"{header} {', '.join(req.character_cast)}"
    lines = [header]
    for name, gender, desc in req.cast_descriptions:
        bits: list[str] = []
        if gender:
            bits.append(gender)
        if desc:
            bits.append(desc)
        if bits:
            lines.append(f"- {name} ({', '.join(bits)})")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


_REPAIR_PROMPT_TEMPLATE = """\
Your previous response did not validate against the required JSON schema.

YOUR PREVIOUS RESPONSE:
{prev_response}

VALIDATION ERROR:
{validation_error}

Return ONLY corrected JSON that matches the schema. Do not explain. Do not add prose. Do not wrap in markdown fences. Output the corrected JSON object and nothing else.
"""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_block(raw: str) -> str:
    """Try three strategies in order:
       1. Strip ```json ... ``` or ``` ... ``` markdown fences if present.
       2. Slice from first '{' to last '}' (handles preambles like
          "Here's the JSON: { ... }").
       3. Return raw stripped (let json.loads raise the error).
    Always returns a string; never raises.
    """
    if not raw:
        return ""
    s = raw.strip()
    m = _FENCE_RE.search(s)
    if m:
        return m.group(1).strip()
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        return s[first : last + 1]
    return s


# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------


class OutlineFailedError(RuntimeError):
    """Raised after generate_outline exhausts all retry attempts.

    Attributes:
        attempts: list of (raw_response, error_message) tuples per attempt
        request:  the OutlineRequest that was being processed
    """

    def __init__(
        self,
        attempts: list[tuple[str, str]],
        request: OutlineRequest,
    ) -> None:
        self.attempts = attempts
        self.request = request
        last_err = attempts[-1][1] if attempts else "no attempts"
        super().__init__(
            f"Outline generation failed after {len(attempts)} attempts. "
            f"Last error: {last_err}"
        )


# ---------------------------------------------------------------------------
# SpeakerRole drift check
# ---------------------------------------------------------------------------

_SPEAKER_ROLE_CHECKED = False


def _check_speaker_role_alignment() -> None:
    """Verify SpeakerRole literal matches _otr_speaker_role's canonical
    list. Lazy import -- only runs once per process, only when
    generate_outline is actually called.
    """
    global _SPEAKER_ROLE_CHECKED
    if _SPEAKER_ROLE_CHECKED:
        return
    _SPEAKER_ROLE_CHECKED = True
    try:
        from . import _otr_speaker_role as _srm
    except ImportError:
        return
    expected = set(_srm.VALID_SPEAKER_ROLES)
    actual = set(SpeakerRole.__args__)  # type: ignore[attr-defined]
    if expected != actual:
        log.warning(
            "[OTR_Outline] SpeakerRole drift: _otr_outline=%s, "
            "_otr_speaker_role=%s. Update _otr_outline.SpeakerRole "
            "to match the canonical list.",
            sorted(actual), sorted(expected),
        )


# ---------------------------------------------------------------------------
# Phase 2A (2026-05-11): episode budget rendering + validators
# ---------------------------------------------------------------------------


class OutlineBudgetViolation(ValueError):
    """Structured signal raised by validate_outline_against_budget on a
    hard violation. Carried as the error string into the reroll-then-
    repair loop. Inherits from ValueError so any defensive `except
    ValueError` clause doesn't drop the signal.
    """


def _get_budget(req: "OutlineRequest"):
    """Return the EpisodeBudget on req, or None.

    Stored as `object` on OutlineRequest so the module can be imported
    without coupling to _otr_episode_budget at load time. We check
    duck-typing here (presence of arc_phases attribute is sufficient).
    """
    b = getattr(req, "budget", None)
    if b is None:
        return None
    if (hasattr(b, "arc_phases") and hasattr(b, "per_phase_words")
            and hasattr(b, "per_phase_beats")):
        return b
    return None


def _format_episode_budget_block(req: "OutlineRequest") -> str:
    """Render the EPISODE BUDGET block. Empty string when no budget."""
    b = _get_budget(req)
    if b is None:
        return ""
    arc_phases = list(b.arc_phases)
    per_phase_words = list(b.per_phase_words)
    per_phase_beats = list(b.per_phase_beats)
    words_lo, words_hi = b.words_per_beat_range
    lines: list[str] = [
        "EPISODE BUDGET -- hit these numbers:",
        f"- Total spoken words: ~{b.target_words} (within 15%)",
        f"- Structure: {b.act_count} act"
        f"{'s' if b.act_count != 1 else ''} -> {', '.join(arc_phases)}",
    ]
    phase_words = ", ".join(
        f"{name} ~{w}"
        for name, w in zip(arc_phases, per_phase_words)
    )
    lines.append(f"- Words per phase: {phase_words}")
    phase_beats = ", ".join(
        f"{name} {n}"
        for name, n in zip(arc_phases, per_phase_beats)
    )
    lines.append(f"- Voiced beats per phase: {phase_beats}")
    lines.append(f"- Each voiced beat: {words_lo}-{words_hi} words")
    lines.append(
        f"- Music inter beats: {b.music_inter_count} "
        f"({'one between each pair of phases' if b.music_inter_count > 0 else 'continuous flow, no music_inter'})"
    )
    lines.append(
        f"- Announcer beats: {b.announcer_beats} (open + close)"
    )
    lines.append(
        "- Every voiced beat MUST carry an `arc_phase` field set to "
        f"one of: {', '.join(arc_phases)}."
    )
    return "\n".join(lines)


def validate_outline_against_budget(
    outline: "Outline",
    req: "OutlineRequest",
    *,
    word_drift_warn_ratio: float = 0.25,
) -> Optional[str]:
    """Run the Phase 2A outline validators.

    Returns None on pass. Returns an error string on the FIRST hard
    failure (suitable for the reroll-then-repair loop). Validator #1
    (total word drift) is WARN-only per §6.E -- never fails. Per
    §6.G announcer + music + sfx beats are EXCLUDED from word and
    per-phase budgets but are still counted by validators #6 / #7.

    No-op when req.budget is None (back-compat).

    Validator list (re-numbered after §6.C dropped per-character
    distribution):
      #1  total word drift (WARN >25%, no reroll per §6.E)
      #2  per-phase word totals within [0.80, 1.20] of target
      #3  per-phase voiced-beat counts within [target-1, target+2]
      #4  per-voiced-beat target_words ∈ words_per_beat_range
      #5  arc_phase monotonic ordering (no interleaving)
      #6  count(music_inter beats) == budget.music_inter_count
      #7  count(announcer beats) == budget.announcer_beats
      #8  every speaker ∈ character_cast ∪ {ANNOUNCER}
          (existing cast-membership check; KEPT)
    """
    b = _get_budget(req)
    if b is None:
        return None

    voiced = [
        beat for beat in outline.beats
        if beat.speaker_role == "character"
    ]
    announcer_beats = [
        beat for beat in outline.beats
        if beat.speaker_role == "announcer"
    ]
    music_inter_beats = [
        beat for beat in outline.beats
        if beat.speaker_role == "music_inter"
    ]

    # Wiring-review #13 (2026-05-11): validate arc_phase
    # existence + allowed-value-membership + monotonic ordering
    # BEFORE running per-phase word totals or per-phase beat
    # counts. Otherwise an unknown / missing phase value silently
    # miscounts under the per-phase aggregations (validators 2 + 3)
    # and the reroll prompt fires for the wrong reason.

    arc_phases = list(b.arc_phases)
    per_phase_words = list(b.per_phase_words)
    per_phase_beats = list(b.per_phase_beats)
    phase_index = {ph: i for i, ph in enumerate(arc_phases)}

    # --- arc_phase: existence + value + monotonic order (was #5) ---
    last_idx = -1
    for beat in voiced:
        ph = (beat.arc_phase or "").strip()
        if not ph:
            return (
                f"Beat {beat.beat_id} is missing arc_phase. Every "
                f"voiced beat MUST carry one of: "
                f"{', '.join(arc_phases)}."
            )
        if ph not in phase_index:
            return (
                f"Beat {beat.beat_id} has arc_phase={ph!r}; not in "
                f"budget arc_phases={arc_phases!r}."
            )
        idx = phase_index[ph]
        if idx < last_idx:
            return (
                f"Beat {beat.beat_id} (arc_phase={ph!r}) breaks "
                f"arc_phase ordering. Voiced beats must be grouped "
                f"by arc_phase in order {arc_phases!r}."
            )
        last_idx = idx

    # --- #1: total word drift (WARN-only per §6.E) ---
    total = sum(beat.target_words for beat in voiced)
    if total > 0:
        ratio = total / max(1, b.target_words)
        if abs(ratio - 1.0) > word_drift_warn_ratio:
            log.warning(
                "[OTR_Outline] WARN: total voiced words=%d vs "
                "target_words=%d (ratio=%.2f); >25%% drift but "
                "per §6.E this is warn-only.",
                total, b.target_words, ratio,
            )

    # --- #2: per-phase word totals ---
    for phase, target_w in zip(arc_phases, per_phase_words):
        got = sum(
            beat.target_words for beat in voiced
            if (beat.arc_phase or "").strip() == phase
        )
        lo = round(target_w * 0.80)
        hi = round(target_w * 1.20)
        if not (lo <= got <= hi):
            return (
                f"Phase {phase!r} got {got} words "
                f"(target {target_w}, allowed {lo}-{hi}). "
                f"Reallocate words: adjust voiced-beat target_words "
                f"in that phase."
            )

    # --- #3: per-phase voiced-beat counts ---
    for phase, target_n in zip(arc_phases, per_phase_beats):
        got = sum(
            1 for beat in voiced
            if (beat.arc_phase or "").strip() == phase
        )
        lo = max(1, target_n - 1)
        hi = target_n + 2
        if not (lo <= got <= hi):
            return (
                f"Phase {phase!r} has {got} voiced beats "
                f"(target {target_n}, allowed {lo}-{hi}). "
                f"Add or remove voiced beats in that phase."
            )

    # --- #4: per-voiced-beat target_words in range ---
    words_lo, words_hi = b.words_per_beat_range
    for beat in voiced:
        if not (words_lo <= beat.target_words <= words_hi):
            return (
                f"Beat {beat.beat_id} has target_words={beat.target_words}; "
                f"required range is {words_lo}-{words_hi} per the budget."
            )

    # --- #6: music_inter count ---
    got_mi = len(music_inter_beats)
    if got_mi != b.music_inter_count:
        return (
            f"music_inter beat count is {got_mi}; budget requires "
            f"{b.music_inter_count}."
        )

    # --- #7: announcer count ---
    got_ann = len(announcer_beats)
    if got_ann != b.announcer_beats:
        return (
            f"announcer beat count is {got_ann}; budget requires "
            f"{b.announcer_beats} (open + close)."
        )

    return None


# ---------------------------------------------------------------------------
# generate_outline -- main entrypoint
# ---------------------------------------------------------------------------


def generate_outline(
    generate_fn,             # (messages, *, temperature, max_new_tokens) -> str
    req: OutlineRequest,
    *,
    max_attempts: int = 3,
    base_temperature: float = 0.7,
    max_new_tokens: int = 1500,
) -> Outline:
    """Generate a validated Outline. Reroll-then-repair on validation failure.

    Retry strategy:
      Attempt 1: fresh generation, temperature = base_temperature (0.7).
      Attempt 2: fresh generation, temperature = base_temperature + 0.1 (0.8).
      Attempt 3: REPAIR call, temperature 0.3, prompt includes the LAST raw
                 response and the exact ValidationError message.

    Caller adapter (lives in OTR_LedgerScriptWriter, NOT this module):

        def _make_generate_fn(llm_cache_entry):
            model = llm_cache_entry["model"]
            tokenizer = llm_cache_entry["tokenizer"]
            def generate_fn(messages, *, temperature, max_new_tokens):
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                out = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.92,
                    max_new_tokens=max_new_tokens,
                )
                return tokenizer.decode(
                    out[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
            return generate_fn

    Raises:
        OutlineFailedError: if all attempts fail validation.
        ValueError: if max_attempts < 1 or generate_fn is not callable.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    if not callable(generate_fn):
        raise ValueError("generate_fn must be callable")

    _check_speaker_role_alignment()

    system = _SYSTEM_PROMPT
    user = _build_user_prompt(req)
    base_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    attempts: list[tuple[str, str]] = []

    for attempt_idx in range(max_attempts):
        is_repair = (attempt_idx == max_attempts - 1) and attempt_idx >= 2

        if is_repair and attempts:
            prev_raw, prev_err = attempts[-1]
            repair_user = _REPAIR_PROMPT_TEMPLATE.format(
                prev_response=prev_raw,
                validation_error=prev_err,
            )
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": repair_user},
            ]
            temp = 0.3
            log.info(
                "[OTR_Outline] attempt %d/%d: repair call (temp=%.2f)",
                attempt_idx + 1, max_attempts, temp,
            )
        else:
            messages = base_messages
            temp = base_temperature + (0.1 * attempt_idx)
            log.info(
                "[OTR_Outline] attempt %d/%d: fresh generation (temp=%.2f)",
                attempt_idx + 1, max_attempts, temp,
            )

        try:
            raw = generate_fn(
                messages,
                temperature=temp,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            err_msg = f"generate_fn raised: {type(exc).__name__}: {exc}"
            log.warning("[OTR_Outline] %s", err_msg)
            attempts.append(("", err_msg))
            continue

        last_raw = raw or ""
        json_str = _extract_json_block(last_raw)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            err_msg = f"json.JSONDecodeError: {exc}"
            log.warning("[OTR_Outline] attempt %d failed: %s", attempt_idx + 1, err_msg)
            attempts.append((last_raw, err_msg))
            continue

        try:
            outline = Outline.model_validate(data)
        except ValidationError as exc:
            err_msg = f"ValidationError: {exc}"
            log.warning("[OTR_Outline] attempt %d failed: %s", attempt_idx + 1, err_msg)
            attempts.append((last_raw, err_msg))
            continue

        # Post-pydantic cast-membership check (replaces the prior
        # outline.cast drift check 2026-05-10): walk character-role
        # beats and verify each beat.speaker is in the LOCKED
        # character_cast. Per Jeffrey: "outline needs to ingest the
        # cast not create it" + "less for the small LLM to lift at
        # one time the better." Outline schema no longer has a cast
        # field; cast lives entirely outside.
        #
        # Diagnostics preserved (extra / missing / duplicates) so
        # the reroll-then-repair loop produces useful repair-call
        # context. Order check is per-beat now (each beat's speaker
        # must be in the locked set); ordering across beats is
        # narrative choice, not a contract violation.
        locked_cast_set = set(req.character_cast)
        used_speakers = [
            b.speaker for b in outline.beats
            if b.speaker_role == "character"
        ]
        used_speakers_set = set(used_speakers)
        invented = used_speakers_set - locked_cast_set
        if invented:
            unused = locked_cast_set - used_speakers_set
            dups = sorted({
                n for n in used_speakers
                if used_speakers.count(n) > 1 and n in invented
            })
            err_msg = (
                "CastContractError: outline beats reference characters "
                "outside the locked cast. invented (must remove): "
                f"{sorted(invented)!r}, locked (allowed): "
                f"{sorted(locked_cast_set)!r}, "
                f"locked-but-unused: {sorted(unused)!r}, "
                f"invented-and-repeated: {dups!r}"
            )
            log.warning(
                "[OTR_Outline] attempt %d failed: %s",
                attempt_idx + 1, err_msg,
            )
            attempts.append((last_raw, err_msg))
            continue

        # Phase 2A (2026-05-11): budget validators. No-op when
        # req.budget is None (pre-Phase-2A back-compat). On hard
        # failure, push the structured error onto attempts and
        # retry; the next attempt's repair-call (when applicable)
        # sees the exact violation message and can correct.
        budget_violation = validate_outline_against_budget(outline, req)
        if budget_violation is not None:
            err_msg = f"OutlineBudgetViolation: {budget_violation}"
            log.warning(
                "[OTR_Outline] attempt %d failed: %s",
                attempt_idx + 1, err_msg,
            )
            attempts.append((last_raw, err_msg))
            continue

        log.info(
            "[OTR_Outline] success on attempt %d/%d: %d beats, "
            "characters used: %s",
            attempt_idx + 1, max_attempts,
            len(outline.beats), sorted(used_speakers_set),
        )
        return outline

    raise OutlineFailedError(attempts=attempts, request=req)


# ---------------------------------------------------------------------------
# Self-test (run as `python nodes/_otr_outline.py` or `python -m nodes._otr_outline`)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== _otr_outline.py self-test ===")

    # Test 1: Beat schema rejects bad inputs.
    print("\n[Test 1] Beat schema validation")
    try:
        Beat(beat_id="bad", speaker="X", speaker_role="character",
             intent="test test", target_words=10, mood="ok")
        print("  FAIL: bad beat_id was accepted")
    except ValidationError:
        print("  PASS: bad beat_id rejected")
    try:
        Beat(beat_id="b001", speaker="X", speaker_role="character",
             intent="test test", target_words=0, mood="ok")
        print("  FAIL: target_words=0 was accepted")
    except ValidationError:
        print("  PASS: target_words=0 rejected")

    # Test 2: speaker uppercased.
    b = Beat(beat_id="b001", speaker="aegeus", speaker_role="character",
             intent="introduce stakes", target_words=12, mood="tense")
    assert b.speaker == "AEGEUS", f"expected AEGEUS, got {b.speaker}"
    print("\n[Test 2] speaker uppercase canonicalization: PASS")

    # Test 3: Outline schema accepts any beats now (cast-membership
    # check moved out to generate_outline). The schema still rejects
    # duplicate beat_ids though -- that's the only cross-beat
    # invariant the schema enforces.
    print("\n[Test 3] Outline rejects duplicate beat_ids (schema-internal)")
    dup_id_data = {
        "title": "Test",
        "premise": "A test premise of sufficient length.",
        "setting": "A test set",
        "time_of_day": "midnight",
        "beats": [
            {"beat_id": "b001", "speaker": "STRANGER", "speaker_role": "character",
             "intent": "speak out of turn", "target_words": 12, "mood": "tense"},
            {"beat_id": "b001", "speaker": "STRANGER", "speaker_role": "character",
             "intent": "speak again with the same id", "target_words": 12, "mood": "tense"},
            {"beat_id": "b003", "speaker": "STRANGER", "speaker_role": "character",
             "intent": "third beat", "target_words": 12, "mood": "tense"},
            {"beat_id": "b004", "speaker": "STRANGER", "speaker_role": "character",
             "intent": "fourth beat", "target_words": 12, "mood": "tense"},
        ],
    }
    try:
        Outline.model_validate(dup_id_data)
        print("  FAIL: duplicate beat_ids accepted")
    except ValidationError as e:
        assert "duplicate beat_ids" in str(e), f"unexpected error: {e}"
        print(f"  PASS: duplicate beat_ids rejected ({type(e).__name__})")

    # Test 4: Schema accepts any speakers (cast-membership lives in
    # generate_outline now, not in the pydantic model).
    print("\n[Test 4] Outline schema accepts any speakers (cast-check is external)")
    ok_data = {
        "title": "Test",
        "premise": "A test premise of sufficient length.",
        "setting": "A test set",
        "time_of_day": "midnight",
        "beats": [
            {"beat_id": "b001", "speaker": "INTRO", "speaker_role": "music_open",
             "intent": "open the show", "target_words": 5, "mood": "bold"},
            {"beat_id": "b002", "speaker": "AEGEUS", "speaker_role": "character",
             "intent": "set the scene", "target_words": 15, "mood": "wry"},
            {"beat_id": "b003", "speaker": "AEGEUS", "speaker_role": "character",
             "intent": "complication arrives", "target_words": 20, "mood": "tense"},
            {"beat_id": "b004", "speaker": "OUTRO", "speaker_role": "music_close",
             "intent": "close the show", "target_words": 5, "mood": "resolute"},
        ],
    }
    o = Outline.model_validate(ok_data)
    assert len(o.beats) == 4
    assert not hasattr(o, "cast"), "Outline schema must NOT carry a cast field"
    print("  PASS: schema accepts beats; no cast field on model")

    # Test 5: JSON extraction handles fences, preambles, raw.
    print("\n[Test 5] _extract_json_block strategies")
    cases = [
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('Here is the JSON: {"a": 1} hope this helps', '{"a": 1}'),
        ('{"a": 1}', '{"a": 1}'),
        ('not json at all', 'not json at all'),
        ('', ''),
    ]
    for raw, expected in cases:
        got = _extract_json_block(raw)
        marker = "PASS" if got == expected else "FAIL"
        print(f"  {marker}: {raw!r:50} -> {got!r}")

    # Test 6: Round-trip serialize/deserialize.
    print("\n[Test 6] Outline JSON round-trip")
    j = o.model_dump_json()
    o2 = Outline.model_validate_json(j)
    assert o2 == o
    print("  PASS: round-trip OK")

    # Test 7: Speaker-role alignment runs without raising.
    print("\n[Test 7] _check_speaker_role_alignment runs without raising")
    _check_speaker_role_alignment()
    print("  PASS")

    # Test 8: OutlineRequest validates inputs.
    print("\n[Test 8] OutlineRequest input validation")
    try:
        OutlineRequest(
            news_seed="x", style="y",
            character_cast=tuple(f"NAME{i}" for i in range(10)),
            target_words=150,
        )
        print("  FAIL: character_cast=10 accepted")
    except ValueError:
        print("  PASS: character_cast of 10 names rejected (must be 1-6)")
    try:
        OutlineRequest(
            news_seed="x", style="y",
            character_cast=("alice",),  # not uppercase
            target_words=150,
        )
        print("  FAIL: lowercase character_cast accepted")
    except ValueError:
        print("  PASS: lowercase character_cast rejected")

    # Test 9: OutlineFailedError carries diagnostics.
    print("\n[Test 9] OutlineFailedError shape")
    err = OutlineFailedError(
        attempts=[("raw1", "err1"), ("raw2", "err2")],
        request=OutlineRequest(
            news_seed="x", style="y",
            character_cast=("ALICE", "BOB"),
            target_words=150,
        ),
    )
    assert len(err.attempts) == 2
    assert err.request.cast_size == 2
    assert err.request.character_cast == ("ALICE", "BOB")
    assert "2 attempts" in str(err)
    print("  PASS")

    # Test 10: cast-contract drift check rejects mismatched outlines.
    # Architecture (post-2026-05-10): outline schema no longer carries
    # a `cast` field. The check walks character-role beats and
    # verifies each beat.speaker is in req.character_cast (the LOCKED
    # cast). LLM "invents CAROL" -> beat.speaker=CAROL not in
    # locked {ALICE, BOB} -> reroll.
    print("\n[Test 10] generate_outline rejects cast drift")
    drift_outline_json = json.dumps({
        "title": "Test", "premise": "A test premise about science.",
        "setting": "A lab", "time_of_day": "Morning",
        "beats": [
            {"beat_id": "b001", "speaker": "NARRATOR",
             "speaker_role": "music_open",
             "intent": "open", "target_words": 5, "mood": "bold"},
            {"beat_id": "b002", "speaker": "ALICE",
             "speaker_role": "character",
             "intent": "speak", "target_words": 10, "mood": "wry"},
            # request will lock ("ALICE", "BOB") -- LLM beat invents CAROL
            {"beat_id": "b003", "speaker": "CAROL",
             "speaker_role": "character",
             "intent": "speak", "target_words": 10, "mood": "wry"},
            {"beat_id": "b004", "speaker": "NARRATOR",
             "speaker_role": "music_close",
             "intent": "close", "target_words": 5, "mood": "resolute"},
        ],
    })

    def _drift_gen_fn(messages, *, temperature, max_new_tokens):
        return drift_outline_json

    try:
        generate_outline(
            _drift_gen_fn,
            OutlineRequest(
                news_seed="x", style="y",
                character_cast=("ALICE", "BOB"),
                target_words=150,
            ),
            max_attempts=2,
        )
        print("  FAIL: cast drift was silently accepted")
    except OutlineFailedError as exc:
        last_err = exc.attempts[-1][1]
        assert "CastContractError" in last_err, \
            f"expected CastContractError in error, got: {last_err!r}"
        print("  PASS: cast drift rejected with CastContractError")

    # Test 11: cast_descriptions field — back-compat default + rich render +
    # validation (length mismatch + name mismatch).
    print("\n[Test 11] OutlineRequest.cast_descriptions field")

    # 11a: empty default -> bare cast line in prompt (back-compat).
    bare_req = OutlineRequest(
        news_seed="science seed", style="noir",
        character_cast=("ALICE", "BOB"),
        target_words=200,
    )
    bare_prompt = _build_user_prompt(bare_req)
    assert "Cast (already chosen -- use exactly these names in character-role beats): ALICE, BOB" in bare_prompt, \
        "11a: bare cast format expected when cast_descriptions empty"
    assert "- ALICE (" not in bare_prompt, \
        "11a: rich format must NOT render when cast_descriptions empty"
    print("  PASS 11a: empty cast_descriptions -> bare cast line (back-compat)")

    # 11b: populated cast_descriptions -> rich block.
    rich_req = OutlineRequest(
        news_seed="science seed", style="noir",
        character_cast=("ALICE", "BOB"),
        target_words=200,
        cast_descriptions=(
            ("ALICE", "female", "weary forensic engineer in her 40s"),
            ("BOB",   "male",   "ambitious grant officer in his 30s"),
        ),
    )
    rich_prompt = _build_user_prompt(rich_req)
    assert "- ALICE (female, weary forensic engineer in her 40s)" in rich_prompt, \
        f"11b: ALICE rich line missing in prompt:\n{rich_prompt}"
    assert "- BOB (male, ambitious grant officer in his 30s)" in rich_prompt, \
        f"11b: BOB rich line missing in prompt:\n{rich_prompt}"
    # Bare list MUST NOT appear when rich is rendered.
    assert "Cast (already chosen -- use exactly these names in character-role beats): ALICE, BOB" not in rich_prompt, \
        "11b: bare cast line must NOT render when rich is in play"
    print("  PASS 11b: populated cast_descriptions -> rich per-character block")

    # 11c: missing gender -> rendered without parens-empty noise.
    no_gender_req = OutlineRequest(
        news_seed="science seed", style="noir",
        character_cast=("ALICE",),
        target_words=200,
        cast_descriptions=(("ALICE", "", "lone caretaker"),),
    )
    no_gender_prompt = _build_user_prompt(no_gender_req)
    assert "- ALICE (lone caretaker)" in no_gender_prompt, \
        f"11c: ALICE without gender expected as '- ALICE (lone caretaker)':\n{no_gender_prompt}"
    print("  PASS 11c: missing gender renders cleanly")

    # 11d: length mismatch -> __post_init__ raises.
    try:
        OutlineRequest(
            news_seed="x", style="y",
            character_cast=("ALICE", "BOB"),
            target_words=200,
            cast_descriptions=(("ALICE", "female", "desc"),),  # length 1 vs cast length 2
        )
        print("  FAIL 11d: length-mismatch cast_descriptions accepted")
    except ValueError as exc:
        assert "align 1:1" in str(exc) or "length" in str(exc), \
            f"11d: expected alignment ValueError, got: {exc}"
        print("  PASS 11d: length-mismatch rejected")

    # 11e: name-order mismatch -> __post_init__ raises.
    try:
        OutlineRequest(
            news_seed="x", style="y",
            character_cast=("ALICE", "BOB"),
            target_words=200,
            cast_descriptions=(
                ("BOB",   "male",   "desc"),     # swapped -- name mismatch at idx 0
                ("ALICE", "female", "desc"),
            ),
        )
        print("  FAIL 11e: name-order mismatch silently accepted")
    except ValueError as exc:
        assert "align 1:1" in str(exc) or "name" in str(exc).lower(), \
            f"11e: expected name-mismatch ValueError, got: {exc}"
        print("  PASS 11e: name-order mismatch rejected")

    # 11f: bad shape -> __post_init__ raises.
    try:
        OutlineRequest(
            news_seed="x", style="y",
            character_cast=("ALICE",),
            target_words=200,
            cast_descriptions=(("ALICE", "female"),),  # 2-tuple instead of 3-tuple
        )
        print("  FAIL 11f: bad-shape cast_descriptions accepted")
    except ValueError as exc:
        assert "3-tuple" in str(exc), \
            f"11f: expected 3-tuple ValueError, got: {exc}"
        print("  PASS 11f: bad-shape rejected")

    # Test 12: include_act_breaks default behavior (target_length
    # removed 2026-05-11; the act-count signal now flows entirely
    # through the EpisodeBudget block, not a separate prose line).
    print("\n[Test 12] include_act_breaks defaults + no_struct prompt")

    no_struct_req = OutlineRequest(
        news_seed="seed", style="noir",
        character_cast=("ALICE",),
        target_words=200,
    )
    no_struct_prompt = _build_user_prompt(no_struct_req)
    assert "Target episode shape:" not in no_struct_prompt, \
        "12a: legacy 'Target episode shape:' line must not appear"
    assert "Target total dialogue length: ~200 words" in no_struct_prompt, \
        "12a: target_words line must still render"
    assert no_struct_req.include_act_breaks is True, \
        "12b: include_act_breaks default must be True"
    print("  PASS 12: target_length structure line gone; "
          "include_act_breaks default True preserved")

    # Test 13: Fix 1 (post-Phase-3 review, 2026-05-11) -- arc_phase
    # required-with-default. A 12B LLM that omits the field must NOT
    # trigger an infinite reroll loop. Pydantic should accept the
    # missing field and stamp `arc_phase='setup'` on the parsed model.
    print("\n[Test 13] arc_phase Field(default='setup') populates on omission")

    # 13a: omitted arc_phase parses with default value.
    beat_no_arc_phase = Beat(
        beat_id="b007",
        speaker="ALICE",
        speaker_role="character",
        intent="speak about the signal",
        target_words=20,
        mood="curious",
        # arc_phase deliberately omitted -- mimics 12B-LLM behavior
    )
    assert beat_no_arc_phase.arc_phase == "setup", (
        f"13a: omitted arc_phase should default to 'setup', "
        f"got {beat_no_arc_phase.arc_phase!r}"
    )
    print("  PASS 13a: omitted arc_phase -> default 'setup'")

    # 13b: explicit arc_phase preserved.
    beat_with_arc_phase = Beat(
        beat_id="b008",
        speaker="BOB",
        speaker_role="character",
        intent="speak",
        target_words=20,
        mood="tense",
        arc_phase="climax",
    )
    assert beat_with_arc_phase.arc_phase == "climax", (
        f"13b: explicit arc_phase must be preserved, got "
        f"{beat_with_arc_phase.arc_phase!r}"
    )
    print("  PASS 13b: explicit arc_phase preserved")

    # 13c: round-trip serialize / deserialize -- the default is
    # written and read back identically (no None / null surprises).
    j = beat_no_arc_phase.model_dump_json()
    assert "setup" in j, f"13c: serialized JSON missing 'setup': {j}"
    b13c_round = Beat.model_validate_json(j)
    assert b13c_round.arc_phase == "setup"
    print("  PASS 13c: round-trip preserves default value")

    # 13d: validator catches a default 'setup' beat that lands in
    # the WRONG phase for a 5-act episode (arc_phases doesn't
    # include 'setup'). The reroll signal is bounded, not infinite.
    print("  PASS 13d (validator path covered in "
          "tests/test_phase2a_episode_budget.py)")

    print("\n=== all self-tests passed ===")
