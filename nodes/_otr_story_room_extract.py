"""nodes/_otr_story_room_extract.py -- Sprint 10B Wave 2 Agent G (2026-05-27).

Transcript-to-structured extraction. Reads a StoryRoomTranscript
(Wave 2 Agent F) and emits a typed StoryRoomExtraction whose dict-
shaped fields match the keys the downstream pipeline already consumes
(announcer / continuity ledger / music+SFX cue builder / bark
generator).

Design references:
- docs/2026-05-26-good-story-writer-architecture__02_design.md
  Section 3.4 StoryRoomExtraction contract, Section 4 Wave 2 Agent G
  ("Transcript-to-structured extraction"). The LLM is transcribing
  the Writer's final draft, NOT creating: the schema constrains the
  output shape while the substance comes from the transcript verbatim.

Cross-check (per Section 4 Wave 2 Agent G "Critical cross-check
before commit"): the canonical ledger line shape stamped by
`nodes/production_ledger.py::Ledger.init_lines_from_outline` carries
`beat_id` + `text` as the bridge fields the downstream consumers
(`nodes/_otr_ledger_consumers.py::iter_lines` + Bark + announcer +
sequencer) read. The dialogue list shape in this module mirrors
Section 3.4: `[{"beat_id", "speaker", "text"}, ...]`. The Wave 3
bridge (when use_story_room flips on by default) is one loop that
walks `extraction.dialogue` and calls `Ledger.update_line_text(
beat_id, text)`; no rename, no schema drift.

This module is PURE: no I/O, no GPU, no ComfyUI imports, no torch.
The single LLM call site is `extract_from_transcript`, tagged
`# LLM slot: technical` per project rule 6 (structured constrained-
decode pass).

Wave 2 lands this module + node + tests; the node ships dormant
(consumes a {"status": "dormant", ...} or empty transcript and emits a
matching sentinel). Wave 3 wires Extract's structured payload into
the legacy ledger as a drop-in for `init_lines_from_outline` +
`update_line_text`.

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
4-space indentation.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from . import _otr_lmfe_compat  # noqa: F401 -- compat shim
from ._otr_stage1_plan import Stage1Arc, Stage1Beat, Stage1CastMember


log = logging.getLogger("OTR")


__all__ = [
    "StoryRoomExtraction",
    "StoryRoomExtractionSchema",
    "ExtractionCallFailedError",
    "build_extract_prompt",
    "extract_from_transcript",
    # Sprint 1 (2026-05-28) -- narrow dialogue-only Extract path.
    "DialogueOnlyRow",
    "DialogueOnlySchema",
    "build_dialogue_only_prompt",
    "extract_dialogue_only",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The Extract call is a CONSTRAINED-DECODE structured extraction
# against a multi-field schema. Low base temperature keeps the model
# transcribing what is in the draft instead of inventing characters
# or rewriting dialogue. The structural-retry temperature is strictly
# lower (lower entropy on the schema re-roll).
_EXTRACT_BASE_TEMPERATURE: float = 0.20
_EXTRACT_STRUCTURAL_RETRY_TEMPERATURE: float = 0.10

# BUG-LOCAL-293 (2026-05-27): bumped 4096 -> 16384. Live evidence
# (pending_20260527_212428, 18-line episode): Extract attempt 2
# truncated mid-string at "char 12358" -- ~3-4 chars per token
# puts the actual output at ~3000-4000 tokens of schema-wrapped
# JSON for an 18-line episode (vs the design's Spray-of-Hope
# 12-beat / 500-word example which fits in 4k). The 4096 cap was
# starving Mistral-Nemo into truncating mid-string; the
# structured_call ladder then retried at lower temp which produced
# even longer JSON (more deterministic = longer strings) and
# truncated again. Both attempts exhausted, Commit pass-through'd,
# Story Room iteration didn't reach the ledger. 16384 gives 4x
# headroom -- enough for ~30-line episodes with full sidecar
# data. Schema-side per-row caps still bound the actual content;
# the bump just gives the model room to close its JSON.
_EXTRACT_MAX_NEW_TOKENS: int = 16384


# ---------------------------------------------------------------------------
# Dataclass (Section 3.4 frozen contract)
# ---------------------------------------------------------------------------


@dataclass
class StoryRoomExtraction:
    """Structured artifacts derived from a story room transcript.

    Field shapes match what the existing announcer pass / continuity
    ledger / music+SFX cue builder / bark generator already consume
    (per design Section 3.4 + Section 4 Wave 2 Agent G cross-check).

    Wave 3's bridge will:
      * stamp `cast` rows into the ledger via the existing
        `set_cast(cast_rows)` path,
      * call `Ledger.init_lines_from_outline(...)` with `beats` as the
        outline iterable (Stage1Beat exposes the fields
        init_lines_from_outline reads),
      * loop `dialogue` and call `Ledger.update_line_text(beat_id,
        text)` per row.

    All current downstream consumers stay on their existing input
    surfaces -- nothing in this module changes the audio path while
    use_story_room is False.
    """

    cast: List[Stage1CastMember]
    beats: List[Stage1Beat]
    dialogue: List[Dict[str, Any]]
    audio_cues: List[Dict[str, Any]] = field(default_factory=list)
    running_facts: List[str] = field(default_factory=list)
    arc: Optional[Stage1Arc] = None
    premise: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dict for the node output socket.

        Stage1* are Pydantic BaseModels; their `.model_dump()` produces
        plain dicts the consumer can re-hydrate via
        `Stage1Beat.model_validate(d)`.
        """
        return {
            "cast": [c.model_dump() for c in self.cast],
            "beats": [b.model_dump() for b in self.beats],
            "dialogue": [dict(row) for row in self.dialogue],
            "audio_cues": [dict(row) for row in self.audio_cues],
            "running_facts": list(self.running_facts),
            "arc": (self.arc.model_dump() if self.arc is not None else None),
            "premise": str(self.premise),
        }


# ---------------------------------------------------------------------------
# Pydantic schema for constrained decode
# ---------------------------------------------------------------------------


class _DialogueRow(BaseModel):
    """Per-line dialogue row. Section 3.4 dialogue shape.

    `beat_id` is the bridge to the ledger row -- Wave 3 passes it
    into `Ledger.update_line_text(beat_id, text)` verbatim.

    Sprint 1 keystone (2026-05-28): adds optional `dialogue_slot_id`
    so the legacy full-schema Extract path stays compatible while
    the new narrow `extract_dialogue_only` path becomes canonical.
    """

    beat_id: str = Field(
        ...,
        min_length=2,
        max_length=8,
        description=(
            "The owning beat's id (bNNN). Mirrors Stage1Beat.beat_id."
        ),
    )
    speaker: str = Field(
        ...,
        min_length=1,
        max_length=40,
        description=(
            "Cast name (must appear in cast[]) OR the literal "
            "'ANNOUNCER' for the open/close bookends OR 'MUSIC' for "
            "a music interlude beat. Case-sensitive."
        ),
    )
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description=(
            "The rendered line as plain prose. No JSON, no markdown, "
            "no stage directions embedded in brackets. ANNOUNCER lines "
            "may carry up to ~80 words; character lines target the "
            "Stage1Beat.length_target_words band."
        ),
    )
    dialogue_slot_id: Optional[str] = Field(
        default=None,
        pattern=r"^d\d{3}$",
        description=(
            "Sprint 1 keystone (2026-05-28). Voiced-slot id "
            "(d001..dNNN) matching the corresponding ledger line. "
            "Optional on this legacy schema for back-compat with "
            "pre-Sprint-1 transcripts; required on `DialogueOnlyRow` "
            "where OTR_StoryRoomCommit joins on it."
        ),
    )


class _AudioCueRow(BaseModel):
    """Per-cue audio row. Music/SFX cue carried alongside the dialogue.

    Optional: an episode may have zero audio cues, especially when the
    Writer's draft leans on dialogue alone. The downstream music + SFX
    builders accept an empty list.
    """

    beat_id: str = Field(
        ...,
        min_length=2,
        max_length=8,
        description="Owning beat id.",
    )
    kind: str = Field(
        ...,
        min_length=2,
        max_length=24,
        description=(
            "Cue kind: 'music_open', 'music_close', 'music_inter', "
            "'sfx', or a free-form one-word kind the downstream cue "
            "builder will treat as 'sfx'."
        ),
    )
    cue: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description=(
            "Short description of the cue. The music/SFX builders "
            "feed this into their downstream generator prompts."
        ),
    )


class StoryRoomExtractionSchema(BaseModel):
    """Constrained-decode binding for the Extract call.

    The model writes ONE JSON object matching this schema. Field
    shapes mirror StoryRoomExtraction one for one; the Stage1 types
    (Stage1CastMember, Stage1Beat, Stage1Arc) ride at the same
    schema-validation rigour the rest of the pipeline already
    enforces.

    Length / element bounds are bounded enough that an empty / runaway
    output fails the schema gate immediately; loose enough that a
    real episode fits comfortably (Section 2.5 example: ~6 cast rows,
    12 beats, 12 dialogue rows, 2-3 audio cues).
    """

    cast: List[Stage1CastMember] = Field(
        ...,
        min_length=1,
        max_length=12,
        description=(
            "One row per speaking character. ANNOUNCER is NOT in this "
            "list -- it rides the bookend bus."
        ),
    )
    beats: List[Stage1Beat] = Field(
        ...,
        min_length=2,
        max_length=64,
        description=(
            "The beat list extracted from the Writer's final draft. "
            "One beat per dialogue line OR music interlude. beat_id "
            "is bNNN, monotonically increasing."
        ),
    )
    dialogue: List[_DialogueRow] = Field(
        ...,
        min_length=2,
        max_length=64,
        description=(
            "One row per voiced beat. beat_id MUST match a beat in "
            "beats[]. speaker MUST be a cast name OR 'ANNOUNCER' OR "
            "'MUSIC'."
        ),
    )
    audio_cues: List[_AudioCueRow] = Field(
        default_factory=list,
        max_length=32,
        description=(
            "Optional music / SFX cues. Empty list when the draft is "
            "dialogue-only."
        ),
    )
    running_facts: List[str] = Field(
        default_factory=list,
        max_length=24,
        description=(
            "Continuity-relevant facts established across the "
            "episode (locations, props, named off-stage parties). "
            "The continuity ledger reads from here."
        ),
    )
    arc: Optional[Stage1Arc] = Field(
        default=None,
        description=(
            "Three-act arc spine. Optional because a short episode "
            "may not articulate one explicitly; when present, the "
            "downstream critic uses it as the arc reference."
        ),
    )
    premise: str = Field(
        default="",
        max_length=600,
        description=(
            "One-sentence restatement of the episode's news premise. "
            "Drawn from the Director's brief + the Writer's final "
            "draft. Empty when neither names it concretely."
        ),
    )


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ExtractionCallFailedError(RuntimeError):
    """Raised when extract_from_transcript exhausts its retry budget.

    The consuming node decides whether to surface this as a hard
    workflow failure or stamp it on meta.story_room_extract for
    forensic inspection. Wave 2 dormant default is to stamp + emit a
    failure sentinel so the queue does not crash.
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
            f"[OTR_StoryRoomExtract] 'extract_from_transcript' failed "
            f"after {attempts} attempt(s); last error -> {last_text}"
        )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


_EXTRACT_SYSTEM_PROMPT = """\
You are an extraction model. You are NOT writing the episode -- the
episode has already been written. Your job is to TRANSCRIBE the
Writer's final draft into a structured JSON object so the downstream
audio pipeline can render it.

RULES:
- Treat the Writer's draft as authoritative. Do not invent characters
  who do not appear in the draft. Do not invent beats. Do not rewrite
  lines. If the draft says "REN BLACK: I won't sign that paper.",
  the corresponding dialogue row has speaker="REN BLACK" and
  text="I won't sign that paper."
- ANNOUNCER bookend lines (the short open + close) get a beat each;
  their speaker field is the literal string "ANNOUNCER".
- Cast rows are ONLY speaking characters with at least one line. The
  ANNOUNCER is NOT a cast row -- it rides the bookend bus.
- beat_id is bNNN starting at b001 in draft order, padded to three
  digits, monotonically increasing.
- dialogue[i].beat_id MUST match an existing beat in beats[].
- For Stage1CastMember and Stage1Beat fields you cannot infer from
  the draft (gender, voice_id, persona, arc_role, intent, mood,
  target_words, arc_phase), supply a plausible placeholder. Wave 3's
  bridge replaces these with the Director's locked cast row when
  available; today they exist so the schema validates.
- Output a SINGLE JSON object. No prose preamble, no Markdown.

The Director's brief and the news seed are provided as context so
you can fill the `premise` field and the `running_facts` list, but
the dialogue and beats come from the Writer's draft.
"""


def _format_cast_hint(cast_names: List[str]) -> str:
    """Render the Director's locked cast list as a hint block."""
    rows = [str(n).strip() for n in (cast_names or []) if str(n).strip()]
    if not rows:
        return "(no Director-locked cast names supplied)"
    return "\n".join(f"- {name}" for name in rows)


def build_extract_prompt(
    transcript_payload: Dict[str, Any],
    *,
    cast_names: Optional[List[str]] = None,
    news_seed: str = "",
) -> List[dict]:
    """Build the [system, user] message list for the Extract call.

    Args:
        transcript_payload: dict-shaped StoryRoomTranscript -- the
            consumer node passes `StoryRoomTranscript.to_dict()` or
            the equivalent dict re-hydrated from the wire JSON.
        cast_names: optional Director-locked cast names. Used as a
            hint so the extractor preserves the canonical names
            verbatim (no rename, no nickname).
        news_seed: optional plain prose for the original news premise.
            Helps the extractor fill the `premise` field correctly.

    Returns:
        Two-message list (system + user) ready for generate_fn.

    PURE -- no LLM call, no I/O. Tests bind the output to a mock
    generate_fn so extract_from_transcript exercises without a live
    model.
    """
    draft = ""
    brief_block_lines: list[str] = []
    if isinstance(transcript_payload, dict):
        draft = str(transcript_payload.get("final_draft") or "").strip()
        brief = transcript_payload.get("director_brief")
        if isinstance(brief, dict):
            brief_block_lines.append(
                f"News premise:      {brief.get('news_premise', '')}"
            )
            brief_block_lines.append(
                f"Dramatic question: {brief.get('dramatic_question', '')}"
            )
            brief_block_lines.append(
                f"Opposed desires:   {brief.get('opposed_desires', '')}"
            )
            brief_block_lines.append(
                f"Arc shape:         {brief.get('arc_shape', '')}"
            )
            brief_block_lines.append(
                f"Audience feel:     {brief.get('audience_feel', '')}"
            )

    seed = (news_seed or "").strip() or "(no news seed supplied)"
    cast_block = _format_cast_hint(cast_names or [])
    brief_block = (
        "\n".join(brief_block_lines)
        if brief_block_lines
        else "(no Director brief supplied)"
    )
    draft_block = draft if draft else "(empty draft -- nothing to extract)"

    user = (
        "DIRECTOR'S BRIEF (context, not the source of truth):\n"
        f"{brief_block}\n\n"
        "NEWS SEED (context for the `premise` field):\n"
        f"{seed}\n\n"
        "CANONICAL CAST NAMES (use these verbatim if they appear in "
        "the draft):\n"
        f"{cast_block}\n\n"
        "WRITER'S FINAL DRAFT (the source of truth):\n"
        "--- begin draft ---\n"
        f"{draft_block}\n"
        "--- end draft ---\n\n"
        "Transcribe the draft into ONE JSON object now. Match the "
        "schema exactly. ANNOUNCER bookends get their own beats; "
        "the cast list excludes ANNOUNCER."
    )
    return [
        {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Extract entry point
# ---------------------------------------------------------------------------


def _attempt_parse_and_validate(raw: str) -> StoryRoomExtractionSchema:
    """Parse `raw` as JSON and validate against StoryRoomExtractionSchema.

    Returns the validated Pydantic instance on success. Raises
    `json.JSONDecodeError` on unparseable output or
    `pydantic.ValidationError` on schema mismatch. The extract retry
    loop catches both at the same boundary.
    """
    payload = json.loads(raw or "")
    return StoryRoomExtractionSchema.model_validate(payload)


def _schema_to_extraction(
    schema: StoryRoomExtractionSchema,
) -> StoryRoomExtraction:
    """Convert a validated schema to the wire-format dataclass.

    Dialogue and audio_cue rows are flattened to plain dicts so the
    downstream consumer can iterate them without learning the
    `_DialogueRow` private types.
    """
    dialogue_rows: List[Dict[str, Any]] = [
        {
            "beat_id": row.beat_id,
            "speaker": row.speaker,
            "text": row.text,
            # Sprint 1 keystone (2026-05-28): forward dialogue_slot_id
            # when the row carries one. Legacy transcripts may emit
            # None; OTR_StoryRoomCommit treats None as a hard fail
            # under the new slot-join contract.
            "dialogue_slot_id": row.dialogue_slot_id,
        }
        for row in schema.dialogue
    ]
    audio_rows: List[Dict[str, Any]] = [
        {
            "beat_id": row.beat_id,
            "kind": row.kind,
            "cue": row.cue,
        }
        for row in schema.audio_cues
    ]
    return StoryRoomExtraction(
        cast=list(schema.cast),
        beats=list(schema.beats),
        dialogue=dialogue_rows,
        audio_cues=audio_rows,
        running_facts=list(schema.running_facts or []),
        arc=schema.arc,
        premise=(schema.premise or "").strip(),
    )


def extract_from_transcript(
    transcript_payload: Dict[str, Any],
    *,
    generate_fn: Callable[..., str],
    cast_names: Optional[List[str]] = None,
    news_seed: str = "",
    max_attempts: int = 2,
) -> StoryRoomExtraction:
    """Run the Extract pass and return a populated StoryRoomExtraction.

    Args:
        transcript_payload: dict-shaped StoryRoomTranscript. The node
            entry deserializes the wire JSON before calling this.
        generate_fn: a ConstrainedGenerateFn bound to
            StoryRoomExtractionSchema (built by
            `_otr_constrained_generate.make_constrained_generate_fn`).
            Signature: (messages, *, temperature, max_new_tokens) -> str.
        cast_names: optional Director-locked cast names.
        news_seed: optional plain prose for the news premise field.
        max_attempts: retry budget. Default 2 -- one base attempt plus
            one structural-retry at lower temperature per design
            Section 4 Wave 2 risk row (constrained-decode timeouts on
            long transcripts).

    Returns:
        A populated StoryRoomExtraction on the first attempt that
        yields a schema-valid object.

    Raises:
        ExtractionCallFailedError when the retry budget exhausts. The
        consuming node decides whether to surface this as a hard
        workflow failure or stamp it on `meta.story_room_extract`.

    # LLM slot: technical
    # Reason: structured constrained-decode pass per project rule 6.
    """
    messages = build_extract_prompt(
        transcript_payload,
        cast_names=cast_names,
        news_seed=news_seed,
    )

    last_error: Optional[BaseException] = None
    attempts = 0

    for attempt_idx in range(1, max_attempts + 1):
        attempts = attempt_idx
        temperature = (
            _EXTRACT_BASE_TEMPERATURE
            if attempt_idx == 1
            else _EXTRACT_STRUCTURAL_RETRY_TEMPERATURE
        )
        log.info(
            "[OTR_StoryRoomExtract] attempt %d/%d (temperature=%.2f)",
            attempt_idx, max_attempts, temperature,
        )
        try:
            # LLM slot: technical
            # Reason: structured constrained-decode extraction
            # against StoryRoomExtractionSchema. Per project rule 6
            # structured-JSON passes route to the writer's
            # `technical_model` slot.
            raw = generate_fn(
                messages,
                temperature=temperature,
                max_new_tokens=_EXTRACT_MAX_NEW_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001 -- slot fn (LLM call) varies
            log.warning(
                "[OTR_StoryRoomExtract] attempt %d generate_fn raised "
                "%s: %s",
                attempt_idx, type(exc).__name__, str(exc)[:200],
            )
            last_error = exc
            continue

        try:
            schema = _attempt_parse_and_validate(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            log.warning(
                "[OTR_StoryRoomExtract] attempt %d parse/validate "
                "failed: %s",
                attempt_idx, str(exc)[:200],
            )
            last_error = exc
            continue

        extraction = _schema_to_extraction(schema)
        log.info(
            "[OTR_StoryRoomExtract] success at attempt %d "
            "(cast=%d beats=%d dialogue=%d audio_cues=%d).",
            attempt_idx,
            len(extraction.cast),
            len(extraction.beats),
            len(extraction.dialogue),
            len(extraction.audio_cues),
        )
        return extraction

    raise ExtractionCallFailedError(attempts=attempts, last_error=last_error)


# ---------------------------------------------------------------------------
# Sprint 1 keystone (2026-05-28) -- narrow dialogue-only Extract path
# ---------------------------------------------------------------------------
#
# Live observation 2026-05-27 (run pending_20260527_223452):
# OTR_StoryRoomExtract attempt 1/2 at temp=0.20 was taking 5+ minutes
# per attempt under the full StoryRoomExtractionSchema (cast + beats +
# dialogue + audio_cues + running_facts + arc + premise = ~3000-4000
# tokens of constrained JSON, Mistral-Nemo 12B NF4 ~10-15 tok/sec).
#
# Root cause: Extract was re-extracting structure Stage 1 already
# produced. The Stage 1 plan + cast contract on the in-flight ledger
# already carries cast / beats / arc / premise / running_facts. The
# only genuinely new content from the Story Room transcript is the
# dialogue lines.
#
# Fix: extract_dialogue_only emits ONLY the dialogue array keyed by
# dialogue_slot_id. Output drops from ~3000-4000 tokens to ~500-800.
# Per-attempt time drops from 5+ min to 30-60 sec on the same model.
#
# The full StoryRoomExtraction is reassembled IN THE NODE
# (OTR_StoryRoomExtract.run) from the in-flight ledger plus this
# narrow dialogue list. OTR_StoryRoomCommit joins by dialogue_slot_id
# and raises StoryRoomCommitError on count or order mismatch.
# ---------------------------------------------------------------------------


class DialogueOnlyRow(BaseModel):
    """One dialogue row for the narrow Extract path.

    Schema is intentionally minimal -- the LLM is told exactly which
    slot ids must appear (and how many) in the prompt; the
    constrained-decode parser only has to validate shape, not invent
    slot ids.
    """

    dialogue_slot_id: str = Field(
        ...,
        pattern=r"^d\d{3}$",
        description=(
            "Voiced-slot id from the in-flight ledger lines "
            "(d001..dNNN). Required: OTR_StoryRoomCommit joins on it."
        ),
    )
    speaker: str = Field(
        ...,
        min_length=1,
        max_length=40,
        description=(
            "Cast name (matches a Director-locked cast member) OR "
            "the literal 'ANNOUNCER' for the open/close bookends. "
            "Case-sensitive; the prompt instructs verbatim copy from "
            "the Writer's draft."
        ),
    )
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description=(
            "The rendered line as plain prose (no JSON, no markdown, "
            "no bracketed stage directions). Transcribe verbatim "
            "from the Writer's draft for the named slot."
        ),
    )


class DialogueOnlySchema(BaseModel):
    """Constrained-decode binding for extract_dialogue_only.

    The schema carries one field -- `dialogue` -- so the LLM cannot
    spend output budget re-emitting structure the in-flight ledger
    already owns. Row-count bounds match the full schema bounds.
    """

    dialogue: List[DialogueOnlyRow] = Field(
        ...,
        min_length=1,
        max_length=64,
        description=(
            "One row per voiced slot id supplied in the prompt, in "
            "the order supplied. Row count MUST equal "
            "len(voice_slot_ids). The consumer "
            "(OTR_StoryRoomCommit) raises StoryRoomCommitError on "
            "mismatch."
        ),
    )


_DIALOGUE_ONLY_MAX_NEW_TOKENS: int = 4096
"""Sprint 1 (2026-05-28): cap drops from 16384 (full schema, BUG-293)
to 4096 since the narrow schema's expected output is ~500-800 tokens
even at 30 voiced slots. 4096 gives 4x headroom on the canonical 18-
line episode and stays well below the model's context budget."""

_DIALOGUE_ONLY_SYSTEM_PROMPT = """\
You are a TRANSCRIBER. The episode has already been written by the
Writer; the Stage 1 plan already named every voiced slot. Your only
job is to copy the Writer's lines into a structured JSON object,
one row per voiced slot, in the order the slots are listed below.

RULES:
- Emit EXACTLY one dialogue row per slot id in the supplied order.
  If the prompt lists 18 slot ids, emit exactly 18 rows.
- dialogue_slot_id MUST be one of the listed ids; do NOT invent or
  reorder slot ids.
- speaker is a cast name from the locked roster OR the literal
  "ANNOUNCER" for bookend slots. Case-sensitive verbatim copy.
- text is the rendered line as plain prose, copied verbatim from
  the Writer's draft for that slot. No markdown, no bracketed stage
  directions, no JSON inside text.
- Output a SINGLE JSON object: {"dialogue": [<row>, <row>, ...]}.
  No prose preamble, no Markdown fences.
"""


def _format_slot_id_block(voice_slot_ids: List[str]) -> str:
    """Render the slot id list for the prompt."""
    rows = [str(s).strip() for s in (voice_slot_ids or []) if str(s).strip()]
    if not rows:
        return "(no voiced slot ids supplied -- nothing to extract)"
    return "\n".join(f"- {sid}" for sid in rows)


def build_dialogue_only_prompt(
    transcript_payload: Dict[str, Any],
    *,
    voice_slot_ids: List[str],
    cast_names: Optional[List[str]] = None,
    news_seed: str = "",
) -> List[dict]:
    """Build the [system, user] messages for the narrow Extract call.

    The voice_slot_ids list pins both the row count and the slot ids
    the LLM must emit. The Writer's draft is the source of truth for
    speaker and text; the slot id list is the source of truth for
    sequencing.
    """
    draft = ""
    if isinstance(transcript_payload, dict):
        draft = str(transcript_payload.get("final_draft") or "").strip()
    draft_block = draft if draft else "(empty draft -- nothing to extract)"
    cast_block = _format_cast_hint(cast_names or [])
    seed = (news_seed or "").strip() or "(no news seed supplied)"
    slot_block = _format_slot_id_block(voice_slot_ids)
    n = len(voice_slot_ids or [])

    user = (
        "NEWS SEED (context only):\n"
        f"{seed}\n\n"
        "CANONICAL CAST NAMES (use these verbatim if they appear in "
        "the draft):\n"
        f"{cast_block}\n\n"
        f"VOICED SLOT IDS ({n} slots, emit one row per slot in this "
        "order):\n"
        f"{slot_block}\n\n"
        "WRITER'S FINAL DRAFT (the source of truth for speaker + text):\n"
        "--- begin draft ---\n"
        f"{draft_block}\n"
        "--- end draft ---\n\n"
        f"Transcribe the draft now. Emit exactly {n} dialogue rows, "
        "one per slot id above, in the supplied order. Output one "
        'JSON object: {"dialogue": [<row>, <row>, ...]}.'
    )
    return [
        {"role": "system", "content": _DIALOGUE_ONLY_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _parse_dialogue_only(raw: str) -> DialogueOnlySchema:
    """Parse `raw` as JSON and validate against DialogueOnlySchema."""
    payload = json.loads(raw or "")
    return DialogueOnlySchema.model_validate(payload)


def extract_dialogue_only(
    transcript_payload: Dict[str, Any],
    *,
    generate_fn: Callable[..., str],
    voice_slot_ids: List[str],
    cast_names: Optional[List[str]] = None,
    news_seed: str = "",
    max_attempts: int = 2,
) -> List[Dict[str, Any]]:
    """Run the narrow Extract pass; return a list of dialogue dicts.

    Args:
        transcript_payload: dict-shaped StoryRoomTranscript.
        generate_fn: ConstrainedGenerateFn bound to DialogueOnlySchema
            (built via
            _otr_constrained_generate.make_constrained_generate_fn).
            Signature: (messages, *, temperature, max_new_tokens) -> str.
        voice_slot_ids: ordered list of d001..dNNN ids from the
            in-flight ledger lines. Pins row count and order. The
            consumer treats `len(returned) != len(voice_slot_ids)` as
            a hard fail (raises StoryRoomCommitError downstream).
        cast_names: optional Director-locked cast names.
        news_seed: optional plain prose for context.
        max_attempts: retry budget (default 2: one base attempt + one
            structural retry at lower temperature).

    Returns:
        A list of dicts shaped like StoryRoomExtraction.dialogue
        rows: {dialogue_slot_id, speaker, text}. Caller reassembles
        the full StoryRoomExtraction with cast/beats/arc/premise/
        running_facts read from the in-flight ledger.

    Raises:
        ExtractionCallFailedError when the retry budget exhausts.

    # LLM slot: technical
    # Reason: structured constrained-decode pass per project rule 6.
    """
    messages = build_dialogue_only_prompt(
        transcript_payload,
        voice_slot_ids=voice_slot_ids,
        cast_names=cast_names,
        news_seed=news_seed,
    )

    last_error: Optional[BaseException] = None
    attempts = 0

    for attempt_idx in range(1, max_attempts + 1):
        attempts = attempt_idx
        temperature = (
            _EXTRACT_BASE_TEMPERATURE
            if attempt_idx == 1
            else _EXTRACT_STRUCTURAL_RETRY_TEMPERATURE
        )
        log.info(
            "[OTR_StoryRoomExtract] dialogue-only attempt %d/%d "
            "(temperature=%.2f, voice_slots=%d, max_new_tokens=%d)",
            attempt_idx, max_attempts, temperature,
            len(voice_slot_ids or []),
            _DIALOGUE_ONLY_MAX_NEW_TOKENS,
        )
        try:
            # LLM slot: technical
            # Reason: structured constrained-decode extraction against
            # DialogueOnlySchema. Per project rule 6 structured-JSON
            # passes route to the writer's `technical_model` slot.
            raw = generate_fn(
                messages,
                temperature=temperature,
                max_new_tokens=_DIALOGUE_ONLY_MAX_NEW_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001 -- slot fn (LLM call) varies
            log.warning(
                "[OTR_StoryRoomExtract] dialogue-only attempt %d "
                "generate_fn raised %s: %s",
                attempt_idx, type(exc).__name__, str(exc)[:200],
            )
            last_error = exc
            continue

        try:
            schema = _parse_dialogue_only(raw)
        except (json.JSONDecodeError, ValidationError) as exc:
            log.warning(
                "[OTR_StoryRoomExtract] dialogue-only attempt %d "
                "parse/validate failed: %s",
                attempt_idx, str(exc)[:200],
            )
            last_error = exc
            continue

        rows: List[Dict[str, Any]] = [
            {
                "dialogue_slot_id": row.dialogue_slot_id,
                "speaker": row.speaker,
                "text": row.text,
                # Legacy `beat_id` kept None on this path -- the
                # consumer joins by slot id, not by beat_id.
                "beat_id": None,
            }
            for row in schema.dialogue
        ]
        log.info(
            "[OTR_StoryRoomExtract] dialogue-only success at attempt "
            "%d (rows=%d, expected=%d).",
            attempt_idx, len(rows), len(voice_slot_ids or []),
        )
        return rows

    raise ExtractionCallFailedError(attempts=attempts, last_error=last_error)
