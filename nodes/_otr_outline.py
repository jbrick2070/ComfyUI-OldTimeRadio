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

from ._otr_episode_budget import EpisodeBudget
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
    "stamp_dialogue_slot_ids",  # Sprint 1 keystone (2026-05-28)
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
        description="Character name in ALL CAPS, or 'NARRATOR' for music beats",
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
    dialogue_slot_id: Optional[str] = Field(
        default=None,
        pattern=r"^d\d{3}$",
        description=(
            "Sprint 1 keystone (2026-05-28). Sequence id for voiced "
            "beats only (d001, d002, ...), stamped in voiced-beat "
            "declaration order by `stamp_dialogue_slot_ids` after "
            "outline assembly. None on non-voiced beats (music_open, "
            "music_close, music_inter). Mirrors onto every "
            "ledger line via production_ledger.init_lines_from_outline "
            "so OTR_StoryRoomCommit can join StoryRoom dialogue rows "
            "to ledger lines by slot id rather than by raw beat_id. "
            "Voiced determination here: speaker_role in {character, "
            "announcer} -- announcer bookends are voiced (Kokoro "
            "renders them) and therefore get a slot id."
        ),
    )

    @field_validator("speaker")
    @classmethod
    def _speaker_uppercase(cls, v: str) -> str:
        return v.strip().upper()


# ---------------------------------------------------------------------------
# Arc references (story-spine Stream A) + Outline schema
# ---------------------------------------------------------------------------


class TurnRef(BaseModel):
    """Story-spine Stream A: pointer to the beat where the situation
    irreversibly changes. ``beat_index`` is a 0-based index into
    ``Outline.beats``. Stamped deterministically by the Path-C combiner
    (``_derive_arc_refs``), never emitted by an LLM -- the index cannot
    exist until Python has assembled the beat list."""

    beat_index: int = Field(..., ge=0, description="0-based index into Outline.beats")
    what_changes: str = Field(..., min_length=3, max_length=200)


class ButtonRef(BaseModel):
    """Story-spine Stream A: pointer to the closing beat that lands the
    payoff. ``beat_index`` is a 0-based index into ``Outline.beats``.
    Stamped by the combiner, same rationale as ``TurnRef``."""

    beat_index: int = Field(..., ge=0, description="0-based index into Outline.beats")
    payoff: str = Field(..., min_length=3, max_length=200)


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

    # Story-spine Stream A (2026-05-31): arc gate. central_tension rides
    # the Stage-1 _MacroShape LLM call; turning_point + button are
    # stamped by the Path-C combiner (_derive_arc_refs) because their
    # beat_index cannot exist until the beat list is assembled. All
    # three are Optional-with-default so legacy Outline(...) construction
    # and serialized round-trips stay valid; the production combiner
    # always populates them (acceptance: every shipped outline carries a
    # turn + a payoff). The _arc_refs_coherent validator below only
    # checks coherence WHEN present, so it never fail-closes a back-compat
    # outline.
    central_tension: str = Field(default="", max_length=300)
    turning_point: Optional[TurnRef] = Field(default=None)
    button: Optional[ButtonRef] = Field(default=None)

    @model_validator(mode="after")
    def _no_duplicate_beat_ids(self) -> "Outline":
        """Schema-internal sanity check. Cast-membership cross-check
        moved to generate_outline (validates beat speakers against
        req.character_cast, the locked cast)."""
        ids = [b.beat_id for b in self.beats]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate beat_ids in outline: {ids}")
        return self

    @model_validator(mode="after")
    def _arc_refs_coherent(self) -> "Outline":
        """Story-spine Stream A structural guard. Only fires when the arc
        refs are present (the combiner always sets them; legacy outlines
        leave them None and pass). Indices must be in range, the turn
        must precede the button, and the button must land in the back
        half of the episode. The combiner stamps valid refs by
        construction, so a raise here means a regression -- caught by the
        outline regression, not a recoverable runtime state."""
        n = len(self.beats)
        tp = self.turning_point
        bt = self.button
        if tp is not None and not (0 <= tp.beat_index < n):
            raise ValueError(
                f"turning_point.beat_index {tp.beat_index} out of range [0,{n})"
            )
        if bt is not None and not (0 <= bt.beat_index < n):
            raise ValueError(
                f"button.beat_index {bt.beat_index} out of range [0,{n})"
            )
        if tp is not None and bt is not None:
            if not (tp.beat_index < bt.beat_index):
                raise ValueError(
                    f"turning_point ({tp.beat_index}) must precede button "
                    f"({bt.beat_index})"
                )
            if bt.beat_index < n // 2:
                raise ValueError(
                    f"button.beat_index {bt.beat_index} must land in the back "
                    f"half of {n} beats (at/near the final beat)"
                )
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
    style_grammar: str = ""
                             # OPTIONAL (KILL 2 style-grammar, 2026-06-24). The
                             # rendered grammar block for the episode's selected
                             # StoryContract (label / sound_world / story_engine /
                             # ending_mode), injected at the MACRO prompt when
                             # non-empty. Empty (default; lever off) => byte-
                             # identical prompt. This is the ONLY place sound_world
                             # enters a prompt -- it shapes structure/mood, never a
                             # dialogue line.
    story_engine: str = ""
                             # OPTIONAL (KILL 2, 2026-06-24). The StoryContract's
                             # conflict-shape one-liner, threaded into the phase +
                             # beat prompts so a divergent premise is planned with a
                             # non-"console standoff" engine. Empty (default) =>
                             # byte-identical prompt.
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
                             # REQUIRED. _otr_episode_budget.EpisodeBudget
                             # built by OTR_LedgerScriptWriter via
                             # compute_episode_budget. The outline prompt
                             # always renders an "EPISODE BUDGET" block
                             # and the post-pydantic pipeline always runs
                             # the 8 Phase 2A validators. Validator #1
                             # (total word drift) is WARN-only at ±25%
                             # per §6.E.
                             #
                             # Stored as `object` to keep this module
                             # importable without pulling
                             # _otr_episode_budget at module load --
                             # `_get_budget(req)` does a lazy
                             # isinstance / duck-type check at call
                             # time.
                             #
                             # S28 cleanbreak: the `None` default is
                             # retained only because dataclass field
                             # ordering forbids a non-defaulted field
                             # after defaulted ones. __post_init__
                             # raises ValueError when budget is missing
                             # — the v2.0 contract requires a populated
                             # budget. Pre-S28 the None branch was a
                             # back-compat fallback for tests and
                             # early-stage callers; both producer sites
                             # (OTR_LedgerScriptWriter) populate it
                             # unconditionally, so the fallback was
                             # leak-prone and is now extinct.
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
    diversity_hint: str = ""
                             # OPTIONAL (best-of-N selector, 2026-06-23). A
                             # short STRUCTURAL-variation instruction the
                             # best-of-N selector sets per candidate (i>=1) to
                             # steer the outline toward a different dramatic
                             # approach (e.g. "open on the personal stake, not
                             # the institutional threat"). Rendered by
                             # _build_user_prompt ONLY when non-empty; empty
                             # (the default, and candidate 0 / every
                             # non-selector call) => byte-identical prompt to
                             # the pre-selector pipeline. A prompt overlay
                             # only -- NOT in-place beat surgery.
    prior_critique: str = ""
                             # OPTIONAL (refine loop v1, 2026-06-23). A short,
                             # normalized STRUCTURAL weakness from the prior
                             # refine pass's grader (sanitized via
                             # _otr_story_select.critique_to_hint), rendered into
                             # the MACRO + BEAT Path C prompts to steer a
                             # REVISION of the prior story. SEPARATE from
                             # diversity_hint (that stays v0 best-of-N steering).
                             # Empty (default) => byte-identical prompt.
    prior_macro: str = ""
                             # OPTIONAL (refine loop v1). A digest of the PRIOR
                             # winner's macro shape (Title/Premise/Setting + raw
                             # beat intents) so the MACRO prompt REVISES the
                             # existing spine instead of starting from scratch.
                             # Empty (default) => byte-identical prompt.

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
        # budget is REQUIRED for the v2.0 contract. OTR_LedgerScript
        # Writer always builds and passes a real EpisodeBudget via
        # compute_episode_budget. A missing budget here is a producer
        # leak — surface it loudly at construction time rather than
        # silently skipping budget rendering + validators downstream.
        # Check runs LAST so character_cast / cast_descriptions
        # validation errors still fire with their original messages
        # (otherwise the budget error would mask the upstream defect).
        if not isinstance(self.budget, EpisodeBudget):
            raise ValueError(
                "OutlineRequest.budget is required (v2.0 contract) "
                "and must be an EpisodeBudget. Build via "
                "_otr_episode_budget.compute_episode_budget with "
                "(target_words, act_count, include_act_breaks, "
                "num_characters) and pass it on construction; got "
                f"{type(self.budget).__name__}={self.budget!r}."
            )

    @property
    def cast_size(self) -> int:
        """Back-compat accessor. Reads len(character_cast)."""
        return len(self.character_cast)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a story editor for short science-fiction audio dramas grounded in real science. Return one JSON object only -- no prose, no fences.

Schema:
{
  "title":       str 3-80,
  "premise":     str 10-400,
  "setting":     str 4-120,
  "time_of_day": str 3-40,
  "beats":       array 4-32 of Beat objects:
                 {
                   "beat_id":      "b001", "b002", ... monotonic,
                   "speaker":      ALL-CAPS name from Cast, or "NARRATOR" for music beats,
                   "speaker_role": one of: character, announcer, music_open, music_close, music_inter,
                   "intent":       one sentence (4-200 chars), narrative purpose only,
                   "target_words": int 3-80,
                   "mood":         str 2-40
                 }
}

Rules:
- Every "character" beat speaker MUST come from the Cast block. Never invent names.
- First beat: music_open or announcer. Last beat: music_close or announcer.
- Beats trace setup, complication, resolution.
- Premise extrapolates dramatically from the science story without contradicting it.
- intent describes narrative purpose only; do not write dialogue text.
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
    # phase + beat-range before being told the rough total.
    # S28 cleanbreak: the `if budget_block:` guard was extinct — the
    # producer (OTR_LedgerScriptWriter) always supplies a budget so
    # _format_episode_budget_block always returns a non-empty string.
    parts.append(_format_episode_budget_block(req))
    parts.append("")

    parts.extend([
        f"Target total dialogue length: ~{req.target_words} words "
        f"(sum of per-beat target_words should land near this number).",
        "",
    ])
    # Best-of-N selector (2026-06-23): an optional structural-variation
    # overlay. The selector sets req.diversity_hint per candidate (i>=1) to
    # push each outline toward a different dramatic approach; candidate 0 and
    # every non-selector call leave it "" so the prompt is byte-identical to
    # the pre-selector pipeline. Rendered ONLY when non-empty.
    diversity_hint = req.diversity_hint.strip()
    if diversity_hint:
        parts.extend([
            f"Structural variation (take a different dramatic approach from "
            f"the other candidates -- vary which stake opens the story, who "
            f"drives the turn, and where the pressure lands): {diversity_hint}",
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


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------
# The naive first-'{'-to-last-'}' extractor was removed in the
# BUG-LOCAL-261 consolidation; outline JSON is now parsed via the shared
# _otr_json.parse_first_json_object. Package import in production; flat
# import when loaded standalone / under test.
try:
    from . import _otr_json
except ImportError:  # pragma: no cover - standalone / test load
    import _otr_json  # type: ignore

# Sprint 2A/2D: the shared structured-JSON retry ladder. generate_outline's
# three stages (macro / phase / beat) route through it. Package import in
# production; flat import when loaded standalone / under test.
try:
    from ._otr_structured_call import (
        structured_call,
        StructuredCallFailedError,
    )
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_structured_call import (  # type: ignore
        structured_call,
        StructuredCallFailedError,
    )

# Sprint 2C: typed repair-prompt factories. All three outline stages
# pass a dispatching factory; the phase stage additionally supplies a
# deterministic cast-membership repair callback (see
# _phase_cast_phantom_repair in generate_outline). Package import in
# production; flat import when loaded standalone / under test.
try:
    from ._otr_repair_prompts import make_dispatching_repair_factory
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_repair_prompts import make_dispatching_repair_factory  # type: ignore


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
    §6.G announcer + music beats are EXCLUDED from word and
    per-phase budgets but are still counted by validators #6 / #7.

    S28 cleanbreak: budget is now required at OutlineRequest
    construction time, so _get_budget always returns a populated
    EpisodeBudget here. The pre-S28 `if b is None: return None`
    branch is extinct.

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
    # S28 cleanbreak: dropped `if b is None: return None` no-op
    # fallback. Producer contract guarantees b is non-None here.

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
# Path C tree-style outline -- per-stage schemas + prompts + helpers
# ---------------------------------------------------------------------------
#
# Sprint H 3.7 retest #11 (2026-05-18) confirmed the outline failure
# family is model-agnostic (Mistral over-produces, Gemma under-then-
# over-corrects). Root cause: the legacy single mega-call asked the
# LLM to satisfy N independent sub-problems (logline + per-phase
# allocation + N beats + N moods + N target_words + N intents) in
# one 1500-token structured pass under pydantic strict validation.
#
# Path C breaks the single call into a tree of small calls following
# Jeffrey 2026-05-18 lean-prompt rule:
#
#   Stage 1 (1 call):              macro shape -- title, premise,
#                                  setting, time_of_day. Under 250
#                                  tokens output, 4 string fields.
#   Stage 2 (act_count calls):     per-phase beat skeleton --
#                                  speaker (from cast) per beat. Under
#                                  200 tokens output, schema:
#                                  list of {speaker} entries.
#   Stage 3 (num_voiced_beats):    per-beat fleshout -- intent + mood
#                                  for one beat. Under 150 tokens
#                                  output, 2 fields. target_words is
#                                  Python-owned (Sprint 3B): it left
#                                  the LLM schema; the combiner stamps
#                                  the per-phase allocation.
#   Python combiner:               Stamp beat_id (b001..), arc_phase
#                                  from budget, speaker_role
#                                  (character / announcer /
#                                  music_inter), and inject the open /
#                                  close announcer + music_inter
#                                  beats. Final Outline validates
#                                  against the existing pydantic +
#                                  budget validators.
#
# Total LLM calls per outline: 1 + act_count + num_voiced_beats.
# Smoke (target_words=300, act_count=3, include_act_breaks=False):
# 1 + 3 + 14 = 18 calls, each <= 250 tokens output.
#
# Each call has its own 3-attempt retry. Failures localize to one
# stage / one phase / one beat instead of poisoning the whole
# outline. The legacy single-call generate_outline + _SYSTEM_PROMPT +
# _build_user_prompt remain exported for back-compat (test imports,
# creative_prompt_router byte-identity check) but are no longer the
# main path.

# Stage 1 schema -- macro shape.
class _MacroShape(BaseModel):
    title: str = Field(..., min_length=3, max_length=80)
    premise: str = Field(..., min_length=10, max_length=400)
    setting: str = Field(..., min_length=4, max_length=120)
    time_of_day: str = Field(..., min_length=3, max_length=40)
    # Story-spine Stream A: the dramatic question. Required-with-default
    # (like Beat.arc_phase) -- a small local model frequently omits an
    # Optional field, so default="" guarantees the macro parses even when
    # omitted; the combiner then falls back to the premise. Capable
    # models emit it from the _MACRO_SYSTEM_PROMPT schema below.
    central_tension: str = Field(default="", max_length=300)


# Stage 2 schema -- per-phase speaker assignments only. Each beat is
# just a speaker name. beat_id, arc_phase, and speaker_role are
# stamped by Python in the combiner.
class _PhaseBeatSeed(BaseModel):
    speaker: str = Field(..., min_length=1, max_length=40)

    @field_validator("speaker")
    @classmethod
    def _speaker_uppercase(cls, v: str) -> str:
        return v.strip().upper()


class _PhaseSkeleton(BaseModel):
    beats: list[_PhaseBeatSeed] = Field(..., min_length=1, max_length=10)


# Stage 3 schema -- per-beat fleshout. intent + mood only.
#
# Sprint 3B (2026-05-25): `target_words` was dropped from this schema.
# Python is the sole authority for per-beat word counts -- the Stage 3
# inline block in generate_outline rebuilds every _BeatFleshout with
# `_allocate_phase_target_words`' allocation and discards whatever the
# LLM emitted. The field was therefore pure dead weight on the
# structured-output token budget: the model spent tokens deciding a
# number that was always overwritten before assembly. Removing it is a
# behaviour-neutral cleanup (the LLM-audit reclassified it from "fix"
# to "token-budget cleanup"). The Beat schema still carries
# target_words; the combiner stamps Python's allocation onto it.
#
# Extra-field tolerance: this model uses pydantic's default
# `extra="ignore"`, so a transitional model that still emits a
# `target_words` key parses cleanly -- the stray key is dropped, not
# rejected.
class _BeatFleshout(BaseModel):
    intent: str = Field(..., min_length=4, max_length=200)
    mood: str = Field(..., min_length=2, max_length=40)


_MACRO_SYSTEM_PROMPT = """\
You plan short science-fiction audio dramas. Return one JSON object only -- no prose, no fences.

Schema:
{
  "title":           3-80 chars,
  "premise":         10-400 chars; one sentence that extrapolates dramatically from the story,
  "setting":         4-120 chars; concrete place,
  "time_of_day":     3-40 chars; e.g. "midnight", "pre-dawn", "after first contact",
  "central_tension": 10-300 chars; the single dramatic question the episode answers, one sentence.
}
"""

_PHASE_SYSTEM_PROMPT = """\
You plan one phase of a science-fiction audio drama. Return one JSON object only -- no prose, no fences.

Schema:
{
  "beats": array of 1-10 objects, each:
    { "speaker": one ALL-CAPS name from the Cast block }
}

Rules:
- Use ONLY the exact ALL-CAPS names from the Cast block. Never invent a name or alter its spelling.
- Speaker variation across beats is optional, not required. Vary speakers only when it serves the scene; repeating the same speaker on consecutive beats is fine.
- The number of beats you return MUST equal the requested count.
"""

_BEAT_SYSTEM_PROMPT = """\
You flesh out one beat of a science-fiction audio drama. Return one JSON object only -- no prose, no fences.

Schema:
{
  "intent": 4-200 chars; one sentence on what this beat accomplishes narratively. NOT dialogue text.
  "mood":   2-40 chars; one tone descriptor.
}
"""


def _build_macro_user_prompt(req: OutlineRequest) -> str:
    """Stage 1 user prompt -- ask for title + premise + setting + time_of_day.

    Lean prompt (target <250 tokens). No beat / phase information --
    that lands in the Stage 2 / 3 prompts.
    """
    brief = req.script_brief.strip()
    if brief:
        source_line = f"Story brief: {brief}"
        verb = "develop this brief"
    else:
        source_line = f"Science story: {req.news_seed}"
        verb = "extrapolate dramatically from this story"
    parts = [
        "Plan the macro shape of a short audio drama.",
        "",
        source_line,
        f"Style: {req.style}",
    ]
    # KILL 2 (2026-06-24): inject the selected StoryContract's full grammar block
    # (label / sound_world / story_engine / ending_mode) so the body is composed in
    # that radio style. First + only caller of render_style_grammar (its prior
    # zero-callers state was the KILL-2 bug). Empty (lever off / unknown slug) =>
    # byte-identical to the pre-grammar prompt.
    _sg = req.style_grammar.strip()
    if _sg:
        parts.append(_sg)
    # Best-of-N / refine steering (2026-06-23): render req.diversity_hint in the
    # REAL Path C macro prompt. The legacy _build_user_prompt (where the hint was
    # first wired) is back-compat/test-only and never runs in production, so the
    # hint was DEAD until now -- best-of-N candidates varied only by RNG seed.
    # Empty hint => byte-identical to the pre-steer prompt.
    _dh = req.diversity_hint.strip()
    if _dh:
        parts.append(
            f"Structural variation (take a different dramatic approach -- vary "
            f"the premise angle, the central stake, and who drives the turn): "
            f"{_dh}"
        )
    # Refine loop (v1, 2026-06-23): REVISE the prior story's spine rather than
    # start from scratch. Empty prior_macro => byte-identical.
    _pm = req.prior_macro.strip()
    if _pm:
        _pc = req.prior_critique.strip()
        if _pc:
            parts.append(
                "Current premise/arc to REVISE (keep what works; change the "
                "premise/arc only if the weakness is structural):\n"
                f"{_pm}\nBiggest weakness to fix: {_pc}"
            )
        else:
            parts.append(
                "Current premise/arc to REVISE (improve the structure while "
                f"preserving the prior spine):\n{_pm}"
            )
    parts.extend([
        "",
        f"Task: {verb}. Return only the JSON object.",
    ])
    return "\n".join(parts)


def _build_phase_user_prompt(
    req: OutlineRequest,
    macro: _MacroShape,
    phase_name: str,
    phase_beat_count: int,
    arc_phases: tuple[str, ...],
    phase_index: int,
    *,
    story_engine: str = "",
) -> str:
    """Stage 2 user prompt -- ask for speaker assignment per beat in one phase.

    Lean prompt (target <200 tokens). Includes only what the LLM needs:
    title + premise + cast + which phase we're on + how many beats.
    """
    parts = [
        f"Title: {macro.title}",
        f"Premise: {macro.premise}",
        f"Setting: {macro.setting}",
    ]
    # KILL 2 (2026-06-24): the StoryContract conflict-shape engine on the phase
    # prompt (sound_world is deliberately NOT rendered here -- it stays at the
    # macro prompt only). Empty (lever off) => byte-identical phase prompt.
    _se = (story_engine or "").strip()
    if _se:
        parts.append(f"Story engine: {_se}")
    parts.extend([
        "",
        _format_cast_block(req),
        "",
        f"Arc phases in order: {', '.join(arc_phases)}",
        f"This phase: {phase_name} (phase {phase_index + 1} of {len(arc_phases)})",
        f"Beats to plan in this phase: {phase_beat_count}",
        "",
        f"Task: assign a speaker to each of the {phase_beat_count} beats "
        f"in the {phase_name!r} phase. Return only the JSON object "
        f"with a `beats` array containing exactly {phase_beat_count} "
        f"entries, each with a `speaker` field.",
    ])
    return "\n".join(parts)


def _phase_summary(phase_name: str) -> str:
    """One-line directional summary for an arc phase.

    Sprint 3B: reuses the project's existing ARC_PHASE_GUIDANCE table
    (the same one-liner the line composer's per-beat prompt consumes)
    so the beat-fleshout prompt and the composer agree on what each
    phase is FOR. Falls back to the bare phase name when the table has
    no entry (back-compat / unusual phase labels). No LLM call -- the
    table is a static dict.
    """
    try:  # lazy import: keep module load stdlib + pydantic only
        from ._otr_episode_budget import ARC_PHASE_GUIDANCE
    except ImportError:  # pragma: no cover - standalone / test load
        from _otr_episode_budget import ARC_PHASE_GUIDANCE  # type: ignore
    return ARC_PHASE_GUIDANCE.get(phase_name, phase_name)


def _build_beat_user_prompt(
    req: OutlineRequest,
    macro: _MacroShape,
    phase_name: str,
    beat_speaker: str,
    beat_position: tuple[int, int],
    *,
    previous_beat_intent: Optional[str] = None,
    next_beat_speaker: Optional[str] = None,
    phase_summary: Optional[str] = None,
    story_engine: str = "",
) -> str:
    """Stage 3 user prompt -- ask for intent + mood for one beat.

    Sprint 3B (2026-05-25): the beat is no longer fully isolated. The
    prompt now carries a 1-beat adjacency window so the LLM can write
    an intent that connects to its neighbours instead of a generic
    standalone beat:

      * `previous_beat_intent` -- the narrative intent of the beat
        immediately before this one (the real, already-generated
        intent: Stage 3 fleshes beats sequentially, so the previous
        beat's intent always exists by the time this one is built).
        Omitted entirely for the first voiced beat of the outline.
      * `next_beat_speaker` -- who speaks the beat immediately after
        this one. Stage 3 has not fleshed that beat yet, so its
        *intent* does not exist; the speaker (known from the phase
        skeleton) is the available forward signal -- enough for the
        LLM to land this beat as a hand-off to that speaker. Omitted
        entirely for the last voiced beat of the outline.
      * `phase_summary` -- a one-line statement of what the current
        arc phase is for (from ARC_PHASE_GUIDANCE).

    Each adjacency line is emitted ONLY when its value is present;
    a missing neighbour produces no line at all (never an empty or
    "None" placeholder). Adjacency window is 1 -- immediate
    neighbours only. target_words is intentionally NOT requested:
    Python owns the per-beat word allocation (see
    `_allocate_phase_target_words` and the Stage 3 block in
    generate_outline). Cross-beat coherence beyond the 1-beat window
    is still handled by the combiner + budget validators downstream.
    """
    beat_idx, beat_total = beat_position
    parts = [
        f"Title: {macro.title}",
        f"Premise: {macro.premise}",
        f"Setting: {macro.setting}",
        "",
        f"Phase: {phase_name}",
    ]
    # KILL 2 (2026-06-24): the conflict-shape engine on the per-beat prompt too
    # (sound_world stays at the macro prompt only). Empty => byte-identical.
    _se = (story_engine or "").strip()
    if _se:
        parts.append(f"Story engine: {_se}")
    if phase_summary:
        parts.append(f"Phase focus: {phase_summary}")
    parts.append(f"Beat {beat_idx + 1} of {beat_total} in this phase")
    parts.append(f"Speaker: {beat_speaker}")
    # Adjacency window (1): only emit a line when the neighbour exists.
    if previous_beat_intent:
        parts.append(f"Previous beat intent: {previous_beat_intent}")
    if next_beat_speaker:
        parts.append(f"Next beat is spoken by: {next_beat_speaker}")
    # Best-of-N / refine steering (2026-06-23): render req.diversity_hint in the
    # REAL Path C beat prompt (legacy _build_user_prompt is test-only). Empty =>
    # byte-identical.
    _dh = req.diversity_hint.strip()
    if _dh:
        parts.append(f"Structural variation: {_dh}")
    # Refine loop (v1): steer this beat to address the prior pass's weakness.
    # Empty prior_critique => byte-identical.
    _pc = req.prior_critique.strip()
    if _pc:
        parts.append(f"Address this weakness: {_pc}")
    parts.extend([
        "",
        "Task: write the intent (one sentence, NOT dialogue) and a "
        "mood descriptor for this beat. The intent MUST be an ACTION "
        "UNDER PRESSURE -- the speaker DOES something with stakes "
        "(reveal, refuse, demand, bargain, accuse, conceal, choose, "
        "threaten, confess), not merely discusses, reflects, or "
        "describes. RAISE THE STAKE: this beat's pressure must be higher "
        "than the previous beat's -- escalate, never tread water. It "
        "should follow on from the previous beat and set up the next "
        "where those are given. "
        # D2 (2026-06-22, story-quality lift): antagonist-stance consistency.
        # The weak-end failure (b-Chandra's Echo) was the antagonist reversing
        # his stance toward the protagonist with no turn beat. JSON-free,
        # no cross-run state -- a best-effort generation nudge.
        "KEEP STANCE CONSISTENT: each character's stance toward the "
        "protagonist and the central conflict must stay true to the want "
        "they have shown so far. A reversal -- an adversary relenting, an "
        "ally turning on them -- is allowed ONLY as a deliberate turn this "
        "beat earns and shows, never an unmotivated flip from the previous "
        "beat. Return only the JSON object.",
    ])
    return "\n".join(parts)


# C0 (story-quality R2): an action-under-pressure beat intent leads with (or
# contains) a stakes verb. Used as a measurement signal by the story-quality
# scan; NOT a hard outline-failing gate (a strict structured_call post_validator
# would flake outlines on weak models -- the prompt constraint is the lever).
_ACTION_PRESSURE_RE = re.compile(
    r"\b(reveal|refus|deny|denies|denied|demand|bargain|accus|conceal|hid|"
    r"hides|choos|chose|chooses|threaten|confess|betray|expos|defy|defies|"
    r"defied|insist|warn|confront|sacrific|risk|gambl|force)\w*",
    re.IGNORECASE,
)


def intent_is_action_under_pressure(intent) -> bool:
    """True when a beat intent reads as an action under pressure (carries a
    stakes verb). Pure; never raises."""
    try:
        return bool(_ACTION_PRESSURE_RE.search(str(intent or "")))
    except Exception:  # noqa: BLE001
        return False


def _allocate_phase_target_words(
    phase_total_words: int,
    n_beats: int,
    words_per_beat_range: tuple[int, int],
) -> list[int]:
    """Default per-beat target_words allocation for a single phase.

    Used as the seed for Stage 3 prompts (each beat is told a
    `[lo, hi]` window centered on its allocation) AND as the
    fallback when Stage 3 produces a value out-of-range. The Stage 3
    pydantic schema enforces the absolute Beat bounds (3..80); this
    function enforces the budget's local window.

    Greedy fill so the sum lands on phase_total_words and every entry
    stays in [words_per_beat_range].
    """
    lo, hi = words_per_beat_range
    if n_beats <= 0:
        return []
    base = phase_total_words // n_beats
    base = max(lo, min(hi, base))
    arr = [base] * n_beats
    delta = phase_total_words - sum(arr)
    idx = 0
    safety = n_beats * 6  # bounded; can't loop forever
    while delta != 0 and safety > 0:
        i = idx % n_beats
        if delta > 0 and arr[i] < hi:
            arr[i] += 1
            delta -= 1
        elif delta < 0 and arr[i] > lo:
            arr[i] -= 1
            delta += 1
        idx += 1
        safety -= 1
    return arr


# ---------------------------------------------------------------------------
# HOTFIX 2026-05-23 (BUG-LOCAL-259) -- outline cast-drift crash
#
# The Stage 2 speaker-assignment call could raise OutlineFailedError
# uncaught when the creative LLM assigned a beat speaker outside the
# locked cast, vaporizing a ~112 s ComfyUI run. The constant + helpers
# below back the locked HOTFIX plan (ROADMAP.md "## HOTFIX -- Outline
# cast-drift crash"): a deterministic, no-LLM phase skeleton (steps 1
# and 2), a low falling Stage 2 temperature schedule (step 3), and
# minimal speaker normalization (step 5).
# ---------------------------------------------------------------------------

# Step 3 (Sprint 2A/2D): Stage 2 (speaker routing) base + structural-
# retry temperatures for the structured_call ladder. Stage 2 assigns
# speakers from a fixed locked cast -- structured routing, not creative
# prose -- so both run low and the structural retry LOWERS temperature
# (the 2B principle). The ladder's typed-repair Attempt 3 then runs at
# its own static low temperature. Replaces the legacy RISING schedule
# (base + 0.1*attempt_idx -> 0.70 / 0.80) that raised temperature on
# exactly the constraint-adherence retries that needed it lowered.
_STAGE2_BASE_TEMPERATURE: float = 0.35
_STAGE2_STRUCTURAL_RETRY_TEMPERATURE: float = 0.25

# Step 5: characters an LLM strays around a speaker name -- a trailing
# colon ("LEMMY:"), surrounding quotes / brackets, stray dashes or
# whitespace. Stripped from BOTH ends only, so internal punctuation
# (e.g. "DR. LEMMY") is preserved.
_SPEAKER_EDGE_PUNCTUATION = " \t\r\n\"'`.,;:!?*_-()[]{}<>"


def _normalize_speaker(name: str) -> str:
    """Normalize an LLM-emitted speaker name for the cast-membership
    check: uppercase, strip whitespace, and drop stray surrounding
    punctuation (' LEMMY ', 'LEMMY:', '"LEMMY"' -> 'LEMMY').

    HOTFIX step 5 (2026-05-23). Deliberately NOT fuzzy matching --
    there is no edit-distance snap here. Broad fuzzy matching can
    silently assign the wrong actor in a multi-character cast; only an
    exact match AFTER this normalization is accepted by _phase_check.
    """
    return name.strip(_SPEAKER_EDGE_PUNCTUATION).upper()


def _deterministic_phase_skeleton(
    phase_beat_count: int,
    locked_cast: tuple[str, ...],
) -> _PhaseSkeleton:
    """Build a Stage 2 phase skeleton WITHOUT an LLM call: assign
    speakers by cycling the sorted locked cast across the beat
    positions.

    Backs two HOTFIX steps:
      * Step 1 -- singleton-cast bypass: a 1-character cast cycles to
        the sole name on every beat, so the Stage 2 LLM call (which
        has no decision to make) is skipped entirely.
      * Step 2 -- deterministic no-crash fallback: when the Stage 2
        retry budget is exhausted for a multi-character cast, this
        replaces `raise OutlineFailedError` so a recoverable
        cast-membership miss never vaporizes the run.

    Fully deterministic -- sorted cast cycled by beat position, no RNG
    -- so it is safe under the writer's seed / repro contract. Every
    emitted speaker is a member of locked_cast by construction, so the
    post-combine cast-leak check and the budget validators pass.
    """
    cast = sorted(locked_cast)
    # Defensive lower bound: a phase budget of 0 voiced beats should
    # never occur (compute_episode_budget allocates >= 1), but max(1,
    # ...) keeps this builder non-crashing against _PhaseSkeleton's
    # min_length=1 if it ever did.
    n = max(1, phase_beat_count)
    beats = [
        _PhaseBeatSeed(speaker=cast[i % len(cast)])
        for i in range(n)
    ]
    return _PhaseSkeleton(beats=beats)


def _structured_attempt_entry(
    label: str, exc: BaseException,
) -> tuple[str, str]:
    """Collapse a failed structured_call into one OutlineFailedError
    attempts entry.

    A stage's structured_call fails one of two ways: the retry ladder
    is exhausted (StructuredCallFailedError), or the slot fn itself
    raises and the exception propagates uncaught (structured_call does
    not catch slot-fn failures). Both map here to a single readable
    (raw, error) tuple for OutlineFailedError.attempts.
    """
    if isinstance(exc, StructuredCallFailedError):
        return (
            "",
            f"[{label}] structured_call exhausted after {exc.attempts} "
            f"attempt(s); last error: {exc.last_error}",
        )
    return (
        "",
        f"[{label}] slot fn raised: {type(exc).__name__}: {exc}",
    )


def _climactic_phase(arc_phases: tuple[str, ...]) -> Optional[str]:
    """Story-spine Stream A: pick the phase that holds the turn. Prefer an
    explicit 'climax' phase; else the penultimate phase; else the last.
    Pure; never raises."""
    if not arc_phases:
        return None
    lowered = [p.lower() for p in arc_phases]
    if "climax" in lowered:
        return arc_phases[lowered.index("climax")]
    if len(arc_phases) >= 2:
        return arc_phases[-2]
    return arc_phases[-1]


def _derive_arc_refs(
    beats: list[Beat], macro: "_MacroShape", arc_phases: tuple[str, ...],
) -> tuple[str, Optional[TurnRef], Optional[ButtonRef]]:
    """Story-spine Stream A: deterministically locate the turning-point
    and button beats and build the arc refs (Path C -- the beat_index
    cannot exist until the combiner has assembled the beat list).

    button = the last CHARACTER beat (the dramatic payoff). turn = the
    first character beat of the climactic phase, falling back to ~two-
    thirds through the character beats, always kept strictly < button.
    central_tension comes from the macro, falling back to the premise.

    Pure; never raises; always returns coherent refs (turn < button, both
    valid indices) for any outline with >= 2 character beats. For a
    degenerate outline (< 2 character beats) returns (central, None, None)
    so the Optional arc fields simply stay unset and the validator skips."""
    char_idxs = [i for i, b in enumerate(beats) if b.speaker_role == "character"]
    central = (getattr(macro, "central_tension", "") or "").strip()
    if not central:
        central = (macro.premise or "").strip()[:300]
    if len(char_idxs) < 2:
        return central, None, None
    button_idx = char_idxs[-1]
    turn_idx: Optional[int] = None
    climax_phase = _climactic_phase(arc_phases)
    if climax_phase is not None:
        for i in char_idxs:
            if beats[i].arc_phase == climax_phase and i < button_idx:
                turn_idx = i
                break
    if turn_idx is None:
        cand = char_idxs[min(len(char_idxs) - 1, (len(char_idxs) * 2) // 3)]
        turn_idx = cand if cand < button_idx else char_idxs[-2]
    if turn_idx >= button_idx:
        turn_idx = char_idxs[-2]
    return (
        central,
        TurnRef(beat_index=turn_idx, what_changes=beats[turn_idx].intent),
        ButtonRef(beat_index=button_idx, payoff=beats[button_idx].intent),
    )


def _assemble_outline(
    macro: _MacroShape,
    phase_skeletons: list[_PhaseSkeleton],
    beat_details: list[list[_BeatFleshout]],
    beat_allocations: list[list[int]],
    req: OutlineRequest,
    budget,
) -> Outline:
    """Combine stage outputs into the final Outline pydantic object.

    Python is authoritative for: beat_id sequence, speaker_role
    assignment, arc_phase tagging, announcer + music_inter beat
    insertion, AND per-voiced-beat target_words. The LLM contributes:
    macro shape (Stage 1), speaker selection per voiced beat (Stage 2),
    intent + mood per voiced beat (Stage 3).

    Sprint 3B (2026-05-25): target_words no longer rides on the
    _BeatFleshout LLM schema. `beat_allocations` carries Python's
    per-phase per-beat word allocation (from
    `_allocate_phase_target_words`) and aligns 1:1 with `beat_details`
    -- one inner list per phase, one int per voiced beat. The combiner
    stamps each allocation onto the corresponding Beat.target_words.
    """
    arc_phases = tuple(budget.arc_phases)
    per_phase_beats_target = tuple(budget.per_phase_beats)
    bid_counter = 1

    def _next_bid() -> str:
        nonlocal bid_counter
        out = f"b{bid_counter:03d}"
        bid_counter += 1
        return out

    beats: list[Beat] = []

    # Announcer open. arc_phase is the first arc_phase to satisfy the
    # validator's "every voiced beat has a known arc_phase" rule; for
    # announcer beats validator only checks count (#7), but pydantic
    # requires arc_phase to be a non-empty string with the right
    # ordering, so we pin it to the first phase.
    beats.append(Beat(
        beat_id=_next_bid(),
        speaker="ANNOUNCER",
        speaker_role="announcer",
        intent="Open the episode and orient the listener.",
        target_words=15,
        mood="welcoming",
        arc_phase=arc_phases[0],
    ))

    # Per-phase voiced beats.
    for phase_idx, (phase_name, skeleton, fleshouts, allocations) in enumerate(
        zip(arc_phases, phase_skeletons, beat_details, beat_allocations)
    ):
        for beat_seed, detail, allocation in zip(
            skeleton.beats, fleshouts, allocations
        ):
            beats.append(Beat(
                beat_id=_next_bid(),
                speaker=beat_seed.speaker,
                speaker_role="character",
                intent=detail.intent,
                target_words=allocation,
                mood=detail.mood,
                arc_phase=phase_name,
            ))

        # Insert music_inter beat between phases when budget asks.
        # music_inter_count is (act_count - 1) when include_act_breaks
        # is True, else 0. By construction we insert exactly that
        # many beats: one after every phase EXCEPT the last.
        if (
            budget.music_inter_count > 0
            and phase_idx < len(arc_phases) - 1
        ):
            beats.append(Beat(
                beat_id=_next_bid(),
                speaker="NARRATOR",
                speaker_role="music_inter",
                # S1 (2026-06-22): neutral, non-narrated intent. This row is
                # a music render contract, never spoken/captioned; the text
                # is suppressed at materialization (production_ledger
                # init_lines_from_outline). The old "Musical interlude
                # bridging <phase>..." string leaked into the transcript.
                intent="Bridge to the next phase with music only.",
                target_words=5,
                mood="transitional",
                arc_phase=phase_name,
            ))

    # Announcer close.
    # Story-grammar build (2026-06-24, C5): when OTR_ENABLE_STYLE_GRAMMAR is on,
    # steer the announcer close OFF the news-OUTCOME (the "console standoff"
    # sameness has the announcer narrate who won / what blew up, stealing the
    # climax from the characters' final beat). The close is NEVER removed -- it
    # stays an announcer beat (budget validator #7 counts announcer beats) and
    # keeps target_words/mood/arc_phase intact; only the intent prose changes.
    # Use the shared config reader so the announcer-close gate honors the SAME
    # default as the writer's style-grammar block (DEFAULT ON as of 2026-06-24;
    # OTR_ENABLE_STYLE_GRAMMAR=0 is the kill-switch => the exact pre-grammar
    # string => byte-identical). _otr_config is a stdlib-only leaf, no cycle.
    try:
        from . import _otr_config as _CFG
    except ImportError:  # pragma: no cover - standalone / test load
        import _otr_config as _CFG  # type: ignore
    if _CFG.style_grammar_enabled():
        # Kept <= Beat.intent's 200-char cap.
        _close_intent = (
            "Sign off on a single concrete final image or sound. Do NOT state, "
            "summarize, or resolve the outcome -- not who won or what was "
            "decided. The resolution belongs to the final beat."
        )
    else:
        # S2 (story-quality R2): close on a concrete final image, not a thesis.
        _close_intent = (
            "Close on a concrete final image showing what changed (use "
            "the central object if set); no moral, thesis, or "
            "news-summary tag."
        )
    beats.append(Beat(
        beat_id=_next_bid(),
        speaker="ANNOUNCER",
        speaker_role="announcer",
        intent=_close_intent,
        target_words=15,
        mood="reflective",
        arc_phase=arc_phases[-1],
    ))

    # Story-spine Stream A: stamp the arc refs from the assembled beats
    # (Path C -- beat_index cannot exist until now). Deterministic; the
    # combiner is the only production path, so every shipped outline
    # carries a turn + a payoff by construction.
    central_tension, turning_point, button = _derive_arc_refs(
        beats, macro, arc_phases,
    )
    outline = Outline(
        title=macro.title,
        premise=macro.premise,
        setting=macro.setting,
        time_of_day=macro.time_of_day,
        beats=beats,
        central_tension=central_tension,
        turning_point=turning_point,
        button=button,
    )
    stamp_dialogue_slot_ids(outline)
    return outline


# ---------------------------------------------------------------------------
# Sprint 1 keystone (2026-05-28) -- dialogue_slot_id stamping
# ---------------------------------------------------------------------------


def stamp_dialogue_slot_ids(outline: "Outline") -> "Outline":
    """Stamp d001..dNNN on voiced beats in declaration order.

    Voiced determination on the Path A `Beat` schema: speaker_role in
    {"character", "announcer"}. Announcer bookends are voiced (Kokoro
    renders them) and therefore get a slot id. Non-voiced beats
    (music_open / music_close / music_inter) keep
    `dialogue_slot_id = None`.

    Mutates the outline's beats in place and returns the same Outline
    for chaining. Safe to call more than once -- the second call
    re-stamps from d001, so two stamping passes on the same outline
    converge on identical ids.

    The stamping invariant is what lets `OTR_StoryRoomCommit` join
    extracted dialogue rows to ledger lines by slot id rather than by
    raw beat_id. The ledger inherits the slot id via
    `production_ledger.init_lines_from_outline`.
    """
    counter = 1
    for beat in outline.beats:
        if beat.speaker_role in ("character", "announcer"):
            beat.dialogue_slot_id = f"d{counter:03d}"
            counter += 1
        else:
            beat.dialogue_slot_id = None
    return outline


# ---------------------------------------------------------------------------
# generate_outline -- main entrypoint (Path C tree-style as of 2026-05-18)
# ---------------------------------------------------------------------------


def generate_outline(
    generate_fn,             # (messages, *, temperature, max_new_tokens) -> str
    req: OutlineRequest,
    *,
    max_attempts: int = 3,
    base_temperature: float = 0.7,
    max_new_tokens: int = 1500,   # legacy parameter; ignored under Path C
    creative_repo_id: str | None = None,  # Sprint D D2b: routes via resolver
) -> Outline:
    """Generate a validated Outline via a tree of small LLM calls.

    Path C (Sprint H 3.7 retest #11 follow-up, 2026-05-18):
      Stage 1 (1 call):              macro shape (title, premise,
                                     setting, time_of_day).
      Stage 2 (act_count calls):     per-phase speaker assignments.
      Stage 3 (num_voiced_beats):    per-beat intent + mood (Sprint 3B:
                                     target_words is Python-owned, not
                                     in the LLM schema).
      Python combiner:               beat_id, arc_phase, speaker_role,
                                     target_words stamping; announcer +
                                     music_inter beat insertion.

    Total LLM calls per outline: 1 + act_count + num_voiced_beats.
    Each call is independently retried (max_attempts each). Failures
    localize to one stage / phase / beat instead of poisoning the
    whole outline.

    Public surface UNCHANGED from the legacy single-mega-call
    implementation -- same arguments, same return type, same
    exceptions. The legacy `max_new_tokens` parameter is accepted but
    ignored (per-call max_new_tokens are stage-local).

    Raises:
        OutlineFailedError: if any stage exhausts its retry budget.
        ValueError: if max_attempts < 1 or generate_fn is not callable.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    if not callable(generate_fn):
        raise ValueError("generate_fn must be callable")

    _check_speaker_role_alignment()

    budget = _get_budget(req)
    # S28 cleanbreak: budget is required at OutlineRequest construction
    # (OutlineRequest.__post_init__ raises ValueError on a missing or
    # wrong-typed budget), so _get_budget always returns a populated
    # EpisodeBudget here. The pre-S28 missing-budget guard is extinct --
    # a bypassed dataclass is a producer defect and surfaces as an
    # AttributeError on the next line. Mirrors the same S28 cleanup in
    # validate_outline_against_budget.
    arc_phases = tuple(budget.arc_phases)
    per_phase_words = tuple(budget.per_phase_words)
    per_phase_beats = tuple(budget.per_phase_beats)
    words_per_beat_range = tuple(budget.words_per_beat_range)
    locked_cast_set = set(req.character_cast)

    # Sprint D D2b: creative-phase prompt routes via the resolver.
    # The new per-stage prompts replace the legacy _SYSTEM_PROMPT
    # for default config. The resolver still owns the period-prompt
    # branch for any period (otr_1940s_v1) row; under such a row the
    # system prompt is OTR_PERIOD_SYSTEM_PROMPT and is layered onto
    # every stage as a register / vocabulary nudge (NOT a schema
    # override). No curated row uses that branch at present.
    if creative_repo_id is None:
        period_system_overlay = None
    else:
        try:
            from ._otr_creative_prompt_router import (
                resolve_creative_system_prompt,
            )
            resolved = resolve_creative_system_prompt(
                creative_repo_id, phase="outline",
            )
            # If the resolver returned the legacy _SYSTEM_PROMPT
            # verbatim (object identity), no overlay -- modern
            # profile. Otherwise the period row is active and we
            # surface its preamble at the start of every stage
            # system prompt.
            if resolved is _SYSTEM_PROMPT:
                period_system_overlay = None
            else:
                period_system_overlay = resolved
        except Exception:  # noqa: BLE001
            period_system_overlay = None

    def _make_system(stage_system: str) -> str:
        if period_system_overlay is None:
            return stage_system
        return period_system_overlay + "\n\n" + stage_system

    all_attempts: list[tuple[str, str]] = []

    # ----------------------------- Stage 1 ---------------------------------
    # LLM slot: creative -- macro outline shapes the episode narrative
    macro_user = _build_macro_user_prompt(req)
    try:
        macro = structured_call(
            prompt=[
                {"role": "system",
                 "content": _make_system(_MACRO_SYSTEM_PROMPT)},
                {"role": "user", "content": macro_user},
            ],
            schema=_MacroShape,
            slot_fn=generate_fn,
            base_temperature=base_temperature,
            structural_retry_temperature=base_temperature / 2.0,
            repair_prompt_factory=make_dispatching_repair_factory(),
            max_new_tokens=250,
            max_attempts=max_attempts,
            helper_name="OTR_Outline.macro",
        )
    except Exception as exc:  # noqa: BLE001 -- ladder or slot-fn failure
        all_attempts.append(_structured_attempt_entry("macro", exc))
        raise OutlineFailedError(attempts=all_attempts, request=req) from exc

    # ----------------------------- Stage 2 ---------------------------------
    # HOTFIX 2026-05-23 (BUG-LOCAL-259): the Stage 2 speaker call no
    # longer crashes the run on a cast-membership miss. Step 1 skips
    # the LLM call entirely for a singleton cast; step 2 replaces the
    # exhausted-retry `raise OutlineFailedError` with a deterministic
    # round-robin skeleton; step 3 routes the call through a low,
    # falling temperature schedule; step 5 normalizes emitted speakers
    # before the cast-membership check.
    phase_skeletons: list[_PhaseSkeleton] = []
    singleton_cast = len(req.character_cast) == 1

    for phase_idx, (phase_name, phase_beat_count) in enumerate(
        zip(arc_phases, per_phase_beats)
    ):
        # Step 1: singleton-cast bypass. With one locked character
        # there is no speaker decision to make -- the sole name is the
        # only legal value for every beat. Skip the Stage 2 LLM call
        # and build the skeleton deterministically. This removes the
        # observed crash (num_characters=1, cast ['LEMMY'], the LLM
        # invented 'LEMMEY' / 'CAPTAIN' and raised OutlineFailedError).
        if singleton_cast:
            skeleton = _deterministic_phase_skeleton(
                phase_beat_count, req.character_cast,
            )
            log.info(
                "[OTR_Outline.phase[%s]] singleton-cast bypass: "
                "skipped Stage 2 LLM call, assigned %r to all %d beats",
                phase_name, req.character_cast[0], len(skeleton.beats),
            )
            phase_skeletons.append(skeleton)
            continue

        phase_user = _build_phase_user_prompt(
            req, macro, phase_name, phase_beat_count,
            arc_phases, phase_idx,
            story_engine=req.story_engine,
        )

        def _phase_check(parsed: _PhaseSkeleton) -> str | None:
            # Cast membership AND beat count match.
            if len(parsed.beats) != phase_beat_count:
                return (
                    f"phase {phase_name!r} beat count mismatch: "
                    f"got {len(parsed.beats)}, expected "
                    f"{phase_beat_count}"
                )
            # Step 5: normalize each emitted speaker before the
            # cast-membership check -- uppercase, strip whitespace,
            # drop stray surrounding punctuation ('ALICE:' -> 'ALICE').
            # If a beat normalizes cleanly into the locked cast, mutate
            # it to the canonical spelling so the assembled outline
            # carries the clean name. NOT fuzzy matching: a genuine
            # hallucination still fails and triggers the retry.
            for b in parsed.beats:
                if b.speaker in locked_cast_set:
                    continue
                normalized = _normalize_speaker(b.speaker)
                if normalized in locked_cast_set:
                    b.speaker = normalized
                    continue
                return (
                    f"phase {phase_name!r} beat speaker "
                    f"{b.speaker!r} is not in locked cast "
                    f"{sorted(locked_cast_set)!r}"
                )
            return None

        def _phase_cast_phantom_repair(
            failed_output: str, error: BaseException,
        ) -> Optional[_PhaseSkeleton]:
            """Sprint 2C cast_membership deterministic repair.

            When the phase skeleton was rejected for a speaker outside
            the locked cast, try to remap every phantom speaker to a
            cast member with the project's one Levenshtein matcher
            (auto_remap_phantom, threshold 3). If EVERY phantom resolves
            unambiguously, return the corrected _PhaseSkeleton --
            structured_call accepts it and makes no LLM repair call.
            Return None when the output cannot be parsed or any phantom
            is ambiguous; the dispatcher then builds an LLM cast-
            membership repair turn instead.
            """
            try:  # lazy import: keep _otr_outline off the reviewer import graph
                from ._otr_ledger_reviewer import auto_remap_phantom
            except ImportError:  # pragma: no cover - standalone / test load
                from _otr_ledger_reviewer import (  # type: ignore
                    auto_remap_phantom,
                )
            try:
                data = _otr_json.parse_first_json_object(failed_output or "")
                skeleton = _PhaseSkeleton.model_validate(data)
            except Exception:  # noqa: BLE001 -- unparseable: no deterministic fix
                return None
            remaps: list[tuple[int, str]] = []
            for idx, beat in enumerate(skeleton.beats):
                if beat.speaker in locked_cast_set:
                    continue
                canonical = auto_remap_phantom(
                    beat.speaker, locked_cast_set, threshold=3,
                )
                if canonical is None:
                    return None  # ambiguous phantom -- fall through to LLM
                remaps.append((idx, canonical))
            for idx, canonical in remaps:
                skeleton.beats[idx].speaker = canonical
            return skeleton

        # LLM slot: creative -- per-phase speaker routing
        try:
            skeleton = structured_call(
                prompt=[
                    {"role": "system",
                     "content": _make_system(_PHASE_SYSTEM_PROMPT)},
                    {"role": "user", "content": phase_user},
                ],
                schema=_PhaseSkeleton,
                slot_fn=generate_fn,
                base_temperature=_STAGE2_BASE_TEMPERATURE,
                structural_retry_temperature=(
                    _STAGE2_STRUCTURAL_RETRY_TEMPERATURE
                ),
                repair_prompt_factory=make_dispatching_repair_factory(
                    deterministic_repair=_phase_cast_phantom_repair,
                ),
                post_validator=_phase_check,
                max_new_tokens=200,
                max_attempts=max_attempts,
                helper_name=f"OTR_Outline.phase[{phase_name}]",
            )
        except Exception as exc:  # noqa: BLE001 -- ladder or slot-fn failure
            # Step 2: deterministic no-crash fallback. A Stage 2
            # exhaustion is almost always a cast-membership miss the
            # creative LLM would not stop making. A cast-membership
            # failure is ALWAYS deterministically recoverable (the
            # legal speaker set is known), so it must never vaporize
            # the run. Build the phase skeleton by round-robining the
            # locked cast instead of raising OutlineFailedError. The
            # failure stays recorded in all_attempts for diagnostics.
            all_attempts.append(
                _structured_attempt_entry(f"phase[{phase_name}]", exc)
            )
            skeleton = _deterministic_phase_skeleton(
                phase_beat_count, req.character_cast,
            )
            log.warning(
                "[OTR_Outline.phase[%s]] Stage 2 retries exhausted; "
                "fell back to deterministic round-robin speaker "
                "assignment across %d beats. The outline will "
                "complete with a plainer speaker pattern for this "
                "phase.",
                phase_name, len(skeleton.beats),
            )
        phase_skeletons.append(skeleton)

    # ----------------------------- Stage 3 ---------------------------------
    # Per-beat fleshout. The LLM contributes intent + mood per voiced
    # beat. Python owns target_words entirely: _allocate_phase_target_
    # words computes a per-phase allocation that satisfies the per-beat
    # range AND the per-phase sum AND the per-episode total by
    # construction -- a constraint LLMs (Mistral and Gemma alike,
    # retests #8-#11) cannot satisfy at once.
    #
    # Sprint 3B (2026-05-25): target_words left the _BeatFleshout LLM
    # schema (the model was spending structured-output tokens on a
    # number that was always discarded). The per-phase `allocations`
    # list now flows straight to the combiner via `beat_allocations`;
    # _BeatFleshout carries only the LLM's intent + mood. Sprint 3B
    # also gives each beat call a 1-beat adjacency window: the previous
    # beat's already-generated intent, the next beat's speaker, and a
    # one-line phase summary.
    beat_details: list[list[_BeatFleshout]] = []
    beat_allocations: list[list[int]] = []
    for phase_idx, (phase_name, phase_skel, phase_total_words) in enumerate(
        zip(arc_phases, phase_skeletons, per_phase_words)
    ):
        n_beats = len(phase_skel.beats)
        allocations = _allocate_phase_target_words(
            phase_total_words, n_beats, words_per_beat_range,
        )
        phase_summary = _phase_summary(phase_name)
        phase_details: list[_BeatFleshout] = []
        for beat_idx, beat_seed in enumerate(phase_skel.beats):
            # Adjacency window 1. The previous beat's intent is the
            # real, already-generated intent (Stage 3 is sequential,
            # so it always exists by the time this beat is built); for
            # the very first voiced beat of the outline there is no
            # previous beat and the line is omitted. The next beat's
            # intent does not exist yet -- Stage 3 has not reached it
            # -- so the forward signal is its speaker, read from the
            # phase skeleton; omitted for the outline's last voiced
            # beat.
            previous_beat_intent: Optional[str] = None
            if phase_details:
                previous_beat_intent = phase_details[-1].intent
            elif beat_details and beat_details[-1]:
                previous_beat_intent = beat_details[-1][-1].intent
            next_beat_speaker: Optional[str] = None
            if beat_idx + 1 < n_beats:
                next_beat_speaker = phase_skel.beats[beat_idx + 1].speaker
            else:
                for later_skel in phase_skeletons[phase_idx + 1:]:
                    if later_skel.beats:
                        next_beat_speaker = later_skel.beats[0].speaker
                        break
            beat_user = _build_beat_user_prompt(
                req, macro, phase_name, beat_seed.speaker,
                (beat_idx, n_beats),
                previous_beat_intent=previous_beat_intent,
                next_beat_speaker=next_beat_speaker,
                phase_summary=phase_summary,
                story_engine=req.story_engine,
            )
            # LLM slot: creative -- per-beat intent/mood (narrative pass)
            try:
                detail = structured_call(
                    prompt=[
                        {"role": "system",
                         "content": _make_system(_BEAT_SYSTEM_PROMPT)},
                        {"role": "user", "content": beat_user},
                    ],
                    schema=_BeatFleshout,
                    slot_fn=generate_fn,
                    base_temperature=base_temperature,
                    structural_retry_temperature=base_temperature / 2.0,
                    repair_prompt_factory=make_dispatching_repair_factory(),
                    max_new_tokens=150,
                    max_attempts=max_attempts,
                    helper_name=(
                        f"OTR_Outline.beat[{phase_name}.{beat_idx + 1}]"
                    ),
                )
            except Exception as exc:  # noqa: BLE001 -- ladder or slot-fn failure
                all_attempts.append(_structured_attempt_entry(
                    f"beat[{phase_name}.{beat_idx + 1}]", exc,
                ))
                raise OutlineFailedError(
                    attempts=all_attempts, request=req,
                ) from exc

            phase_details.append(detail)
        beat_details.append(phase_details)
        beat_allocations.append(allocations)

    # ----------------------------- Combine ---------------------------------
    outline = _assemble_outline(
        macro, phase_skeletons, beat_details, beat_allocations,
        req, budget,
    )

    # Cross-call invariants. Path C's per-stage retries already
    # enforce cast membership at the source (Stage 2 extra_check)
    # so the assembled outline should pass downstream validators
    # cleanly. This is a final belt-and-braces sweep matching the
    # legacy validator chain.
    used_speakers = [
        b.speaker for b in outline.beats
        if b.speaker_role == "character"
    ]
    invented = set(used_speakers) - locked_cast_set
    if invented:
        all_attempts.append(
            (
                "",
                "CastContractError (post-combine): invented speakers "
                f"{sorted(invented)!r} leaked past Stage 2 filter. "
                f"Locked cast: {sorted(locked_cast_set)!r}.",
            )
        )
        raise OutlineFailedError(attempts=all_attempts, request=req)

    budget_violation = validate_outline_against_budget(outline, req)
    if budget_violation is not None:
        all_attempts.append(
            ("", f"OutlineBudgetViolation: {budget_violation}")
        )
        raise OutlineFailedError(attempts=all_attempts, request=req)

    log.info(
        "[OTR_Outline] success: %d beats (%d voiced, %d announcer, "
        "%d music_inter); calls used: 1 macro + %d phase + %d beat "
        "= %d total",
        len(outline.beats), len(used_speakers),
        sum(1 for b in outline.beats if b.speaker_role == "announcer"),
        sum(1 for b in outline.beats if b.speaker_role == "music_inter"),
        len(phase_skeletons),
        sum(len(pd) for pd in beat_details),
        1 + len(phase_skeletons) + sum(len(pd) for pd in beat_details),
    )
    return outline


# ---------------------------------------------------------------------------
# Self-test (run as `python nodes/_otr_outline.py` or `python -m nodes._otr_outline`)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("=== _otr_outline.py self-test ===")

    # S28 cleanbreak: budget is now required at OutlineRequest
    # construction time. Shared EpisodeBudget for the harness tests
    # that aren't specifically exercising the budget-missing reject
    # path. Mirrors compute_episode_budget defaults. Try both import
    # styles so the harness still runs under either
    # `python nodes/_otr_outline.py` or `python -m nodes._otr_outline`.
    try:
        from ._otr_episode_budget import compute_episode_budget as _ceb
    except ImportError:
        from _otr_episode_budget import compute_episode_budget as _ceb  # type: ignore[no-redef]
    _HARNESS_BUDGET_200 = _ceb(200, 3, True, 2)
    _HARNESS_BUDGET_150 = _ceb(150, 3, True, 2)
    _HARNESS_BUDGET_200_1CHAR = _ceb(200, 3, True, 1)
    # 350 words at the matching default act count (3) is the proven-
    # satisfiable budget shape -- a complete outline validates against
    # it. Test 10 runs generate_outline to completion, so it needs a
    # consistent budget (unlike _HARNESS_BUDGET_150, where 150 words
    # forced into 3 acts is internally unsatisfiable -- fine for the
    # Tests 9/11/12 that never assemble a full outline).
    _HARNESS_BUDGET_350 = _ceb(350, 3, True, 2)

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

    # Test 5: JSON extraction handles fences, preambles, trailing data.
    print("\n[Test 5] _otr_json.extract_first_json_block strategies")
    cases = [
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('Here is the JSON: {"a": 1} hope this helps', '{"a": 1}'),
        ('{"a": 1}', '{"a": 1}'),
        ('{"a": 1}{"b": 2}', '{"a": 1}'),
        ('not json at all', ''),
        ('', ''),
    ]
    for raw, expected in cases:
        got = _otr_json.extract_first_json_block(raw)
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
            budget=_HARNESS_BUDGET_150,
        ),
    )
    assert len(err.attempts) == 2
    assert err.request.cast_size == 2
    assert err.request.character_cast == ("ALICE", "BOB")
    assert "2 attempts" in str(err)
    print("  PASS")

    # Test 10: generate_outline RECOVERS from Stage 2 cast drift
    # (HOTFIX 2026-05-23, BUG-LOCAL-259). Before the hotfix an
    # off-cast speaker from the Stage 2 LLM raised OutlineFailedError
    # and vaporized a ~112 s run. Now the exhausted-retry path builds
    # a deterministic round-robin skeleton instead, so the outline
    # completes with every speaker inside the locked cast. Full
    # behavioural coverage: tests/test_outline_cast_drift_hotfix.py.
    print("\n[Test 10] generate_outline recovers from Stage 2 cast drift")

    _T10_MACRO_JSON = json.dumps({
        "title": "Drift Recovery Test",
        "premise": "A premise about a faint science signal and its cost.",
        "setting": "A quiet observatory lab",
        "time_of_day": "midnight",
    })
    _T10_BEAT_JSON = json.dumps({
        "intent": "advance the scene toward the next turn",
        "target_words": 18,
        "mood": "tense",
    })

    def _drift_stage_gen(messages, *, temperature, max_new_tokens):
        system = messages[0]["content"]
        if "You plan one phase" in system:
            user = messages[1]["content"]
            m = re.search(r"Beats to plan in this phase: (\d+)", user)
            n = int(m.group(1)) if m else 1
            # CAROL is NOT in the locked ("ALICE", "BOB") cast -- every
            # Stage 2 attempt drifts and exhausts the retry budget.
            return json.dumps(
                {"beats": [{"speaker": "CAROL"} for _ in range(n)]}
            )
        if "You flesh out one beat" in system:
            return _T10_BEAT_JSON
        return _T10_MACRO_JSON

    _t10_outline = generate_outline(
        _drift_stage_gen,
        OutlineRequest(
            news_seed="x", style="y",
            character_cast=("ALICE", "BOB"),
            target_words=350,
            budget=_HARNESS_BUDGET_350,
        ),
        max_attempts=2,
    )
    _t10_speakers = {
        b.speaker for b in _t10_outline.beats
        if b.speaker_role == "character"
    }
    assert _t10_speakers, "Test 10: outline has no character beats"
    assert _t10_speakers <= {"ALICE", "BOB"}, (
        f"Test 10: drift recovery leaked off-cast speakers: "
        f"{_t10_speakers!r}"
    )
    print("  PASS: Stage 2 cast drift recovered via deterministic "
          "round-robin; all speakers in locked cast")

    # Test 11: cast_descriptions field — rich render + validation
    # (length mismatch + name mismatch). S28 cleanbreak: dropped 11a
    # (the bare-format / empty-default back-compat assertion). The
    # cast_descriptions=() back-compat is still tolerated by the
    # __post_init__ (cast_descriptions defaults to ()), but the
    # bare-format prompt rendering is no longer the asserted
    # default-shape contract — producers (OTR_LedgerScriptWriter
    # D.5 post-cast-lock) always populate cast_descriptions.
    print("\n[Test 11] OutlineRequest.cast_descriptions field")

    # 11b: populated cast_descriptions -> rich block.
    rich_req = OutlineRequest(
        news_seed="science seed", style="noir",
        character_cast=("ALICE", "BOB"),
        target_words=200,
        budget=_HARNESS_BUDGET_200,
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
        budget=_HARNESS_BUDGET_200_1CHAR,
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
        budget=_HARNESS_BUDGET_200_1CHAR,
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
