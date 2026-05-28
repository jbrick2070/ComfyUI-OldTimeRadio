r"""Sprint 10A Stage 1 -- Structural plan schema.

Per docs/story-generator-final-plan.md Stage 1: one LLM call produces
the entire episode plan as a structured object. The schema lives here;
the call site lands in commit 3-B; wiring to the writer lands in 3-C.

A plan is the upstream input to:
  * Stage 1 cast audit (step 4) -- deterministic name -> gender ->
    pronouns -> voice_id validation.
  * Stage 2 multi-turn roleplay (step 6) -- Turn 4 prompt builds from
    the plan's beat + cast rows + running_facts.
  * Stage 3 validators (step 5) -- pronoun-consistency uses the cast
    member's pronouns; continuity uses running_facts; on-beat uses
    the beat's intent.
  * Step 7 whole-episode critic -- the rubric checks against the
    rendered ledger, which was built from this plan.

The schema is intentionally strict: every field has bounded length /
regex / Literal constraints so lm-format-enforcer's JsonSchemaParser
can constrain the LLM at the token-sampling layer. Stage 1 cannot
emit a beat with `beat_id="b1"` or a cast member with `gender="cat"`;
the sampler refuses the token.

Bounds chosen to match existing OTR domain:
  * cast min/max = 1 / 6 (matches num_characters widget min/max)
  * voice_id regex = ^v2/en_speaker_[0-9]$ (matches the Bark v2 pool
    used throughout _otr_casting.py + _otr_bark_lib.py)
  * beat_id regex = ^b\d{3}$ (matches existing OTR ledger convention)
  * beats min/max = 3 / 40 (covers tiny-smoke 30-word episodes
    through 15-min full episodes per the target_words tooltip in
    OTR_LedgerScriptWriter.INPUT_TYPES)
  * length_target_words 5..200 (per-beat word target band)
  * running_facts max = 20 (state spine, kept bounded so Turn 3 prompt
    in Stage 2 stays inside the context window even at full episode
    length)
"""

from __future__ import annotations

import re
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Constants -- single source of truth for downstream consumers
# ---------------------------------------------------------------------------

VOICE_ID_PATTERN: str = r"^v2/en_speaker_[0-9]$"
"""Bark v2 voice preset regex. Cast audit (step 4) cross-references
this against the casting voice pool to confirm the LLM picked a real
preset, not a hallucinated 'v2/en_speaker_42' or similar."""

BEAT_ID_PATTERN: str = r"^b\d{3}$"
"""Beat-id regex matching the existing OTR ledger convention
(b001..b999). Cross-validated against the beat order at parse time."""

GENDER_LITERAL = Literal["male", "female", "nonbinary"]
"""Gender enumeration. Stage 1 cast audit (step 4) uses this as the
expected value of the deterministic name -> gender lookup."""

PRONOUNS_LITERAL = Literal["he/him", "she/her", "they/them"]
"""Pronouns enumeration. Stage 3 pronoun-consistency validator
(step 5) reads from here to know what to match for each speaker."""

CAST_MIN: int = 1
CAST_MAX: int = 6
BEATS_MIN: int = 3
BEATS_MAX: int = 40
RUNNING_FACTS_MAX: int = 20

# Beat-id pre-compiled regex for the cross-validators below
_BEAT_ID_RE = re.compile(BEAT_ID_PATTERN)


# ---------------------------------------------------------------------------
# Inner models
# ---------------------------------------------------------------------------


class Stage1Arc(BaseModel):
    """The three-act arc spine. Stage 2 Turn 2 prompt embeds this verbatim."""

    setup: str = Field(
        ...,
        min_length=10,
        max_length=400,
        description="What's established at episode start. One to two sentences.",
    )
    complication: str = Field(
        ...,
        min_length=10,
        max_length=400,
        description="What goes wrong / what pressure rises in act 2. One to two sentences.",
    )
    resolution: str = Field(
        ...,
        min_length=10,
        max_length=400,
        description="How the episode closes. One to two sentences.",
    )


class Stage1CastMember(BaseModel):
    """One cast member. Stage 1 produces one row per speaking character;
    the ANNOUNCER is NOT in this list (Kokoro handles it on a separate
    bus per the existing audio path).
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=40,
        description="Character name as it appears in dialogue attribution. ALL CAPS by convention.",
    )
    gender: GENDER_LITERAL = Field(
        ...,
        description="Cast audit (step 4) cross-references this against the deterministic name -> gender lookup; mismatch triggers repair or plan regeneration.",
    )
    pronouns: PRONOUNS_LITERAL = Field(
        ...,
        description="Stage 3 pronoun-consistency validator (step 5) reads from here to know what to match in every line referring to the speaker.",
    )
    voice_id: str = Field(
        ...,
        pattern=VOICE_ID_PATTERN,
        description="Bark v2 voice preset. Must match the v2/en_speaker_[0-9] pool; cast audit confirms it is in the casting voice roster.",
    )
    persona: str = Field(
        ...,
        min_length=20,
        max_length=400,
        description="Two to three sentences of character voice + concerns + speech texture. Stage 2 Turn 1 system prompt embeds this verbatim.",
    )
    arc_role: str = Field(
        ...,
        min_length=3,
        max_length=80,
        description="One short phrase locating the character in the arc (e.g. 'reluctant insider', 'pressure source', 'truth-teller').",
    )


class Stage1Beat(BaseModel):
    """One beat. Stage 2 generates one dialogue line per beat (or, for
    music_inter beats, no dialogue but the slot exists in the ledger).

    `callback_to` enables Stage 1 to wire continuity callbacks explicitly:
    'in beat b008, character X references the prop established in b002'.
    Stage 3 continuity validator (step 5) checks the referenced beat
    exists; the writer uses the callback to thread the running_facts
    consistently across the rendered dialogue.
    """

    beat_id: str = Field(
        ...,
        pattern=BEAT_ID_PATTERN,
        description="bNNN. Three-digit padded. Must be unique within the plan and monotonically increasing across the beat list.",
    )
    speaker: str = Field(
        ...,
        min_length=1,
        max_length=40,
        description="Cast member name (must match a Stage1CastMember.name) OR the literal 'ANNOUNCER' for bookend beats OR 'MUSIC' for music_inter beats. Cross-validated post-parse.",
    )
    intent: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="One-sentence description of what this beat accomplishes. Stage 2 Turn 4 reads from here; Stage 3 on-beat validator (step 5) checks the rendered line loosely matches.",
    )
    length_target_words: int = Field(
        ...,
        # BUG-LOCAL-282 (2026-05-27): floor relaxed from 5 to 0 so a
        # music_inter beat (speaker='MUSIC') can carry
        # length_target_words=0 -- music slots produce no dialogue and
        # legitimately want a zero target. The voiced-beats floor is
        # re-asserted by the `_voiced_beats_have_min_words` model
        # validator below so character/announcer beats still require
        # length_target_words >= 5 (Stage 3 length validator pin).
        ge=0,
        le=200,
        description="Target word count for the rendered dialogue line. Stage 3 length validator passes lines within [target * 0.5, target * 1.7]. Music beats (speaker='MUSIC') may be 0.",
    )
    emotional_register: str = Field(
        ...,
        min_length=3,
        max_length=80,
        description="One to three words describing the emotional temperature for this beat (e.g. 'wary curiosity', 'tight panic', 'measured relief').",
    )
    callback_to: Optional[str] = Field(
        default=None,
        description="Optional beat_id (bNNN) of an earlier beat this one calls back to. Used for continuity threading; Stage 3 continuity validator confirms the referenced beat exists in this plan.",
    )

    @field_validator("callback_to", mode="before")
    @classmethod
    def _validate_callback_format(cls, v: Any) -> Optional[str]:
        # BUG-LOCAL-292 (2026-05-27): coerce LLM-emitted no-callback
        # synonyms to None. Live evidence (pending_20260527_212428
        # Story Room Extract): Mistral-Nemo emitted the literal
        # string 'none' for callback_to on 17 beats; the strict
        # validator rejected, Extract retry budget burned. The
        # callback_to field is OPTIONAL by design -- treat any
        # textual no-callback indicator as None rather than failing
        # the whole schema gate.
        if v is None:
            return None
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                return None
            # Case-insensitive synonyms for "no callback".
            if stripped.lower() in {"none", "null", "n/a", "na",
                                    "no", "nil", "nothing", "-"}:
                return None
            if not _BEAT_ID_RE.match(stripped):
                raise ValueError(
                    f"callback_to must be a bNNN beat-id (e.g. "
                    f"'b002') or null/empty; got {v!r}"
                )
            return stripped
        raise ValueError(
            f"callback_to must be a string or null; got {type(v).__name__}"
        )

    @model_validator(mode="after")
    def _voiced_beats_have_min_words(self) -> "Stage1Beat":
        """Re-assert the length floor for voiced beats only.

        BUG-LOCAL-282 (2026-05-27). The unconditional `ge=5` on
        `length_target_words` was rejecting the LLM's perfectly valid
        `length_target_words=0` on music_inter beats (speaker='MUSIC')
        and the Stage 1 shadow pass kept exhausting its retry budget.
        The fix splits the rule:
          * speaker == 'MUSIC'  -> 0 is allowed (no dialogue planned).
          * everything else     -> floor of 5 stays (Stage 3 length
                                    validator pin).
        ANNOUNCER bookends still require >= 5 since announcer lines
        DO go through the announcer-pass renderer.
        """
        if self.speaker == "MUSIC":
            return self
        if self.length_target_words < 5:
            raise ValueError(
                f"length_target_words must be >= 5 for voiced beats "
                f"(speaker={self.speaker!r}); got "
                f"{self.length_target_words}. Use speaker='MUSIC' for "
                f"non-voiced music_inter beats with target 0."
            )
        return self


# ---------------------------------------------------------------------------
# Top-level plan
# ---------------------------------------------------------------------------


class Stage1Plan(BaseModel):
    """The complete Stage 1 structural plan.

    This is the upstream input every downstream stage reads. The schema
    is enforced both by:
      * lm-format-enforcer at sampling time (constrained decoding;
        the LLM cannot emit invalid tokens that would break the schema)
      * pydantic at parse time (belt-and-braces -- catches anything the
        sampler somehow lets slip, e.g. semantic constraints like the
        cross-validators below).
    """

    premise: str = Field(
        ...,
        min_length=10,
        max_length=300,
        description="One sentence describing the episode's central situation. Step 7 critic checks Premise clarity axis against this + the rendered ledger.",
    )
    arc: Stage1Arc = Field(
        ...,
        description="Three-act arc spine.",
    )
    cast: List[Stage1CastMember] = Field(
        ...,
        min_length=CAST_MIN,
        max_length=CAST_MAX,
        description=f"Speaking characters. Length {CAST_MIN}..{CAST_MAX}. ANNOUNCER is NOT in this list.",
    )
    beats: List[Stage1Beat] = Field(
        ...,
        min_length=BEATS_MIN,
        max_length=BEATS_MAX,
        description=f"Beat sequence. Length {BEATS_MIN}..{BEATS_MAX}. beat_ids must be unique and monotonically increasing.",
    )
    running_facts: List[str] = Field(
        default_factory=list,
        max_length=RUNNING_FACTS_MAX,
        description=f"Established facts the dialogue must respect. Each item is a short factual statement. Stage 2 Turn 3 prompt embeds these verbatim; Stage 3 continuity validator (step 5) checks no rendered line contradicts any of them. Max {RUNNING_FACTS_MAX} entries.",
    )

    @field_validator("running_facts")
    @classmethod
    def _validate_facts_nonempty_strings(cls, v: List[str]) -> List[str]:
        for i, fact in enumerate(v):
            if not isinstance(fact, str):
                raise ValueError(
                    f"running_facts[{i}] must be a string; got {type(fact).__name__}"
                )
            if not fact.strip():
                raise ValueError(
                    f"running_facts[{i}] must be a non-empty string"
                )
            if len(fact) > 400:
                raise ValueError(
                    f"running_facts[{i}] must be <= 400 chars; got {len(fact)}"
                )
        return v


# ---------------------------------------------------------------------------
# Cross-validators (post-parse, semantic constraints not expressible in JSON schema)
# ---------------------------------------------------------------------------


# Reserved speakers that are NOT in cast[]. ANNOUNCER is rendered by Kokoro
# on a separate audio bus; MUSIC is a music_inter cue slot.
RESERVED_SPEAKERS = frozenset({"ANNOUNCER", "MUSIC"})


def validate_plan_semantics(plan: Stage1Plan) -> None:
    """Run cross-field semantic validators on a parsed plan.

    Raises ValueError on the first failure. Step 4 cast audit calls
    this after the constrained decode + pydantic parse to confirm the
    plan is internally consistent.

    Checks:
      1. Cast names are unique.
      2. Cast voice_ids are unique.
      3. Beat ids are unique AND monotonically increasing.
      4. Every beat's speaker is either a cast member name OR a
         reserved speaker (ANNOUNCER, MUSIC).
      5. Every callback_to references a beat_id that exists in the
         plan AND comes earlier in the beat list.
    """

    # 1. Cast name uniqueness
    cast_names = [c.name for c in plan.cast]
    if len(cast_names) != len(set(cast_names)):
        dupes = sorted({n for n in cast_names if cast_names.count(n) > 1})
        raise ValueError(
            f"Stage1Plan: cast names must be unique; duplicates: {dupes}"
        )

    # 2. Cast voice_id uniqueness
    voice_ids = [c.voice_id for c in plan.cast]
    if len(voice_ids) != len(set(voice_ids)):
        dupes = sorted({v for v in voice_ids if voice_ids.count(v) > 1})
        raise ValueError(
            f"Stage1Plan: cast voice_ids must be unique; duplicates: {dupes}"
        )

    # 3. Beat id uniqueness + monotonic order
    beat_ids = [b.beat_id for b in plan.beats]
    if len(beat_ids) != len(set(beat_ids)):
        dupes = sorted({b for b in beat_ids if beat_ids.count(b) > 1})
        raise ValueError(
            f"Stage1Plan: beat_ids must be unique; duplicates: {dupes}"
        )
    parsed = [int(b[1:]) for b in beat_ids]
    if parsed != sorted(parsed):
        raise ValueError(
            f"Stage1Plan: beat_ids must be monotonically increasing; "
            f"got order {beat_ids}"
        )

    # 4. Beat speaker is cast member or reserved
    valid_speakers = set(cast_names) | RESERVED_SPEAKERS
    for b in plan.beats:
        if b.speaker not in valid_speakers:
            raise ValueError(
                f"Stage1Plan: beat {b.beat_id} speaker {b.speaker!r} is "
                f"neither a cast member ({sorted(cast_names)}) nor a "
                f"reserved speaker ({sorted(RESERVED_SPEAKERS)})"
            )

    # 5. callback_to targets exist + come earlier
    seen_so_far: set[str] = set()
    for b in plan.beats:
        if b.callback_to is not None:
            if b.callback_to not in beat_ids:
                raise ValueError(
                    f"Stage1Plan: beat {b.beat_id} callback_to "
                    f"{b.callback_to!r} references a beat not in the plan"
                )
            if b.callback_to not in seen_so_far:
                raise ValueError(
                    f"Stage1Plan: beat {b.beat_id} callback_to "
                    f"{b.callback_to!r} must reference an earlier beat; "
                    f"got a forward / self reference"
                )
        seen_so_far.add(b.beat_id)


def parse_and_validate_plan(plan_dict: dict) -> Stage1Plan:
    """Parse a dict into a Stage1Plan, then run semantic cross-validators.

    Single entry point for downstream consumers. Use this instead of
    `Stage1Plan(**plan_dict)` directly so the semantic checks always
    run.

    Raises:
        pydantic.ValidationError -- on schema-level failure
        ValueError -- on semantic cross-field failure
    """
    plan = Stage1Plan(**plan_dict)
    validate_plan_semantics(plan)
    return plan


# ---------------------------------------------------------------------------
# JSON schema export (for the lm-format-enforcer parser, commit 3-B)
# ---------------------------------------------------------------------------


def get_plan_json_schema() -> dict:
    """Return the JSON schema for Stage1Plan.

    Commit 3-B will feed this to lmformatenforcer.JsonSchemaParser so
    the LLM is constrained at the sampling layer. Exposing it via this
    function (rather than calling Stage1Plan.model_json_schema()
    directly at the call site) gives one obvious patch point if the
    schema needs runtime modification (e.g. tightening bounds during
    soak measurement).
    """
    return Stage1Plan.model_json_schema()
