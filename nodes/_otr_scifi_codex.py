"""Sci-Fi Codex v4 additive source-bank runner.

The lane owns its schemas, prompt seams, provenance graph, and ledger assembly.
It never fetches a source, loads a model, or edits LLM-authored dialogue.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Callable, Literal, Mapping, MutableMapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


log = logging.getLogger("OTR")

try:
    from ._otr_canon import EpisodeCanon
    from ._otr_json import parse_first_json_object
    from ._otr_rss_source_contract import meets_v4_rss_source_floor
    from ._otr_source_payload import validate_source_payload
    from ._otr_scifi_p0_contract import (
        MAX_CLAIM_CHARS,
        MAX_ENTITY_NAME_CHARS,
        MAX_ENTITY_ROWS,
        MAX_FACT_ROWS,
        MAX_NUMERIC_TOKEN_CHARS,
        MAX_NUMBER_ROWS,
        MAX_QUOTE_CHARS,
        MAX_SPANS_PER_EVIDENCE_ROW,
        MAX_TONE_CHARS,
        compact_p0_repair_context,
        p0_contract_instruction,
        p0_contract_receipt,
        p0_output_token_budget,
    )
    from ._otr_scifi_source_repair import repair_literal_source_metadata
    from ._otr_structured_call import (
        PostValidationError,
        REPAIR_TEMPERATURE,
        invoke_structured_slot,
        schema_shape_instruction,
        structured_call,
    )
    from . import _otr_ledger_freeze
    from .production_ledger import stamp_word_counts
except ImportError:  # pragma: no cover
    from _otr_canon import EpisodeCanon  # type: ignore
    from _otr_json import parse_first_json_object  # type: ignore
    from _otr_rss_source_contract import meets_v4_rss_source_floor  # type: ignore
    from _otr_source_payload import validate_source_payload  # type: ignore
    from _otr_scifi_p0_contract import (  # type: ignore
        MAX_CLAIM_CHARS,
        MAX_ENTITY_NAME_CHARS,
        MAX_ENTITY_ROWS,
        MAX_FACT_ROWS,
        MAX_NUMERIC_TOKEN_CHARS,
        MAX_NUMBER_ROWS,
        MAX_QUOTE_CHARS,
        MAX_SPANS_PER_EVIDENCE_ROW,
        MAX_TONE_CHARS,
        compact_p0_repair_context,
        p0_contract_instruction,
        p0_contract_receipt,
        p0_output_token_budget,
    )
    from _otr_scifi_source_repair import repair_literal_source_metadata  # type: ignore
    from _otr_structured_call import (  # type: ignore
        PostValidationError,
        REPAIR_TEMPERATURE,
        invoke_structured_slot,
        schema_shape_instruction,
        structured_call,
    )
    import _otr_ledger_freeze  # type: ignore
    from production_ledger import stamp_word_counts  # type: ignore


class ScifiCodexError(RuntimeError):
    """Base class for fail-loud Codex lane errors."""


class CodexPayloadShapeError(ScifiCodexError): pass
class CodexPayloadRouteError(ScifiCodexError): pass
class CodexPayloadThinError(ScifiCodexError): pass
class CodexPayloadOversizeError(ScifiCodexError): pass
class CodexTargetRangeError(ScifiCodexError): pass
class CodexPackContractError(ScifiCodexError): pass
class CodexPassError(ScifiCodexError): pass
class CodexSpokenTextError(ScifiCodexError): pass
class CodexGraphError(ScifiCodexError): pass
class CodexFactTraceExhaustedError(ScifiCodexError): pass
class CodexPreTailAuditError(ScifiCodexError): pass
class CodexTailFinalizerMissingError(ScifiCodexError): pass
class CodexLedgerSaveError(ScifiCodexError): pass
class CodexSavedLedgerAuditError(ScifiCodexError): pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def _schema_instruction(result_type: type[BaseModel]) -> str:
    return schema_shape_instruction(result_type)


class SourceSpanV4(_Strict):
    field: Literal["headline", "summary", "full_text", "seed_text"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1, max_length=MAX_QUOTE_CHARS)

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("source span end must be greater than start")
        return self


class SourcePayload7(_Strict):
    headline: str
    summary: str
    full_text: str
    source: str
    date: str
    link: str
    seed_text: str


class PayloadEnvelopeV4(_Strict):
    schema_version: Literal["scifi_codex.payload_envelope.v4"]
    payload: SourcePayload7
    source_mode: Literal["rss", "operator_pinned"]
    source_digest: str


class WordSteerV4(_Strict):
    requested_words: int = Field(ge=30, le=900)


class FactV4(_Strict):
    fact_id: str = Field(pattern=r"^F0[1-6]$")
    claim: str = Field(min_length=1, max_length=MAX_CLAIM_CHARS)
    source_spans: list[SourceSpanV4] = Field(
        min_length=1, max_length=MAX_SPANS_PER_EVIDENCE_ROW,
    )
    numeric_tokens: list[Annotated[str, Field(
        min_length=1, max_length=MAX_NUMERIC_TOKEN_CHARS,
    )]] = Field(default_factory=list, max_length=4)


class EntityV4(_Strict):
    entity_id: str = Field(pattern=r"^E0[1-4]$")
    name: str = Field(min_length=1, max_length=MAX_ENTITY_NAME_CHARS)
    source_spans: list[SourceSpanV4] = Field(
        min_length=1, max_length=MAX_SPANS_PER_EVIDENCE_ROW,
    )


class NumberV4(_Strict):
    number_id: str = Field(pattern=r"^N0[1-4]$")
    verbatim: str = Field(min_length=1, max_length=MAX_NUMERIC_TOKEN_CHARS)
    fact_id: str
    source_span: SourceSpanV4


class FactIndexV4(_Strict):
    facts: list[FactV4] = Field(min_length=1, max_length=MAX_FACT_ROWS)
    entities: list[EntityV4] = Field(max_length=MAX_ENTITY_ROWS)
    numbers: list[NumberV4] = Field(max_length=MAX_NUMBER_ROWS)
    tone: str = Field(min_length=1, max_length=MAX_TONE_CHARS)
    payload_sha256: str


class DramaticQuestionV4(_Strict):
    question: str = Field(min_length=1, max_length=160)
    consequence: str = Field(min_length=1, max_length=160)
    ending_direction: str = Field(min_length=1, max_length=120)


class CastPlanRowV4(_Strict):
    char_id: Literal["announcer", "c01", "c02", "c03"]
    name: str = Field(min_length=1, max_length=40)
    character_description: str = Field(min_length=1, max_length=160)
    gender: str = Field(min_length=1, max_length=24)
    role_in_conflict: str = Field(min_length=1, max_length=120)
    voice_slot: Literal["announcer", "c01", "c02", "c03"]


class CastPlanV4(_Strict):
    cast: list[CastPlanRowV4] = Field(min_length=2, max_length=4)


_CAST_NAME_RE = re.compile(r"(?:(?:Dr|Prof)\. )?[A-Z][a-z]+(?: [A-Z][a-z]+)*")


def _is_canonical_character_name(name: str) -> bool:
    """Accept ordinary title-cased full names without rewriting cast work."""
    return bool(_CAST_NAME_RE.fullmatch(name))


def _validate_cast_plan(cast: CastPlanV4) -> str | None:
    """Lock the fixed roster before downstream score/script generation."""
    by_id = {row.char_id: row for row in cast.cast}
    if len(by_id) != len(cast.cast):
        return "cast plan has duplicate char_id values"
    announcer = by_id.get("announcer")
    if announcer is None:
        return "cast plan must contain the fixed announcer row"
    if announcer.name != "ANNOUNCER":
        return "announcer row must use the exact fixed name ANNOUNCER"
    for row in cast.cast:
        if row.char_id != "announcer" and not _is_canonical_character_name(row.name):
            return f"cast name {row.name!r} is not a canonical Title-Case name"
    return None


def repair_cast_plan_metadata(failed_output: str) -> CastPlanV4 | None:
    """Normalize the fixed announcer identity without changing character work."""
    try:
        cast = CastPlanV4.model_validate(parse_first_json_object(failed_output))
    except Exception:
        return None
    rows = []
    changed = False
    for row in cast.cast:
        if row.char_id == "announcer" and row.name != "ANNOUNCER":
            rows.append(row.model_copy(update={"name": "ANNOUNCER"}))
            changed = True
        else:
            rows.append(row)
    return cast.model_copy(update={"cast": rows}) if changed else None


_RADIO_SCORE_CONTEXT_CAP_TOKENS = 8192
# Measured 2026-07-12 with the actual local Gemma E4B chat template: the
# max-width compact draft serializes to 1,418 tokens. Reserve that surface plus
# max(128, ceil(15%)) and a 16-token framing margin: 1,647 tokens.
_RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS = 1647
_RADIO_SCORE_MAX_SCENES = 3
_RADIO_SCORE_MAX_SHOTS_PER_SCENE = 2
_RADIO_SCORE_MAX_BEATS_PER_SCENE = 4
_RADIO_SCORE_MAX_BEATS = (
    _RADIO_SCORE_MAX_SCENES * _RADIO_SCORE_MAX_BEATS_PER_SCENE
)
_RADIO_SCORE_MAX_LINES_PER_BEAT = 2
_RADIO_SCORE_MAX_MUSIC_CUES = 3
_RADIO_SCORE_MAX_FACT_IDS_PER_BEAT = 2


class AdvisoryBeatV4(_Strict):
    beat_id: str = Field(pattern=r"^b\d{3}$")
    advisory_word_center: int = Field(ge=0, le=900)


class AdvisoryWordPlanV4(_Strict):
    advisory_total_center: int = Field(ge=1, le=900)
    per_beat: list[AdvisoryBeatV4] = Field(
        min_length=1, max_length=_RADIO_SCORE_MAX_BEATS,
    )


class BeatPlanV4(_Strict):
    beat_id: str = Field(pattern=r"^b\d{3}$")
    scene_id: str = Field(pattern=r"^scene_\d{3}$")
    shot_id: str = Field(pattern=r"^shot_\d{3}$")
    speaker: str = Field(min_length=1, max_length=40)
    char_id: Literal[
        "announcer", "c01", "c02", "c03",
        "music_open", "music_inter", "music_close",
    ]
    speaker_role: Literal[
        "character", "announcer", "music_open", "music_close", "music_inter",
    ]
    line_ids: list[Annotated[str, Field(pattern=r"^l\d{3}$")]] = Field(
        min_length=1, max_length=_RADIO_SCORE_MAX_LINES_PER_BEAT,
    )
    order: int = Field(ge=1, le=_RADIO_SCORE_MAX_BEATS)
    intent: str = Field(min_length=1, max_length=64)
    arc_phase: str = Field(min_length=1, max_length=28)
    fact_ids: list[Annotated[str, Field(pattern=r"^F0[1-6]$")]] = Field(
        default_factory=list, max_length=_RADIO_SCORE_MAX_FACT_IDS_PER_BEAT,
    )
    advisory_voiced_word_center: int = Field(ge=0, le=900)


class ShotPlanV4(_Strict):
    shot_id: str = Field(pattern=r"^shot_\d{3}$")
    scene_id: str = Field(pattern=r"^scene_\d{3}$")
    description: str = Field(min_length=1, max_length=72)
    visual_prompt: str = Field(min_length=1, max_length=120)


class ScenePlanV4(_Strict):
    scene_id: str = Field(pattern=r"^scene_\d{3}$")
    env: str = Field(min_length=1, max_length=56)
    description: str = Field(min_length=1, max_length=72)
    shots: list[ShotPlanV4] = Field(
        min_length=1, max_length=_RADIO_SCORE_MAX_SHOTS_PER_SCENE,
    )
    beats: list[BeatPlanV4] = Field(
        min_length=1, max_length=_RADIO_SCORE_MAX_BEATS_PER_SCENE,
    )


class MusicCueV4(_Strict):
    cue_id: Literal["music_open", "music_inter", "music_close"]
    placement: Literal["open", "inter", "close"]
    description: str = Field(min_length=1, max_length=80)
    generation_prompt: str = Field(min_length=1, max_length=120)
    anchor_line_id: str = Field(pattern=r"^l\d{3}$")
    anchor_beat_id: str = Field(pattern=r"^b\d{3}$")


class RadioScoreV4(_Strict):
    title: str = Field(min_length=1, max_length=64)
    premise: str = Field(min_length=1, max_length=144)
    setting: str = Field(min_length=1, max_length=80)
    advisory_word_plan: AdvisoryWordPlanV4
    scenes: list[ScenePlanV4] = Field(
        min_length=1, max_length=_RADIO_SCORE_MAX_SCENES,
    )
    music_cues: list[MusicCueV4] = Field(
        min_length=1, max_length=_RADIO_SCORE_MAX_MUSIC_CUES,
    )


class RadioScoreDraftBeatV4(_Strict):
    """The P3 author's scene-local decisions, without derived score metadata."""

    shot_index: int = Field(ge=0, le=_RADIO_SCORE_MAX_SHOTS_PER_SCENE - 1)
    char_id: Literal["announcer", "c01", "c02", "c03"]
    line_count: int = Field(ge=1, le=_RADIO_SCORE_MAX_LINES_PER_BEAT)
    intent: str = Field(min_length=1, max_length=64)
    arc_phase: str = Field(min_length=1, max_length=28)
    fact_ids: list[Annotated[str, Field(pattern=r"^F0[1-6]$")]] = Field(
        default_factory=list,
        max_length=_RADIO_SCORE_MAX_FACT_IDS_PER_BEAT,
    )


class RadioScoreDraftShotV4(_Strict):
    description: str = Field(min_length=1, max_length=72)
    visual_prompt: str = Field(min_length=1, max_length=120)


class RadioScoreDraftSceneV4(_Strict):
    env: str = Field(min_length=1, max_length=56)
    description: str = Field(min_length=1, max_length=72)
    shots: list[RadioScoreDraftShotV4] = Field(
        min_length=1, max_length=_RADIO_SCORE_MAX_SHOTS_PER_SCENE,
    )
    beats: list[RadioScoreDraftBeatV4] = Field(
        min_length=1, max_length=_RADIO_SCORE_MAX_BEATS_PER_SCENE,
    )


class RadioScoreDraftMusicCueV4(_Strict):
    cue_id: Literal["music_open", "music_inter", "music_close"]
    description: str = Field(min_length=1, max_length=80)
    generation_prompt: str = Field(min_length=1, max_length=120)
    anchor_beat_index: int = Field(ge=0, le=_RADIO_SCORE_MAX_BEATS - 1)
    anchor_line_index: int = Field(ge=0, le=_RADIO_SCORE_MAX_LINES_PER_BEAT - 1)


class RadioScoreDraftV4(_Strict):
    """Compact P3 transport. Python derives the final score mechanics."""

    title: str = Field(min_length=1, max_length=64)
    premise: str = Field(min_length=1, max_length=144)
    setting: str = Field(min_length=1, max_length=80)
    scenes: list[RadioScoreDraftSceneV4] = Field(
        min_length=1, max_length=_RADIO_SCORE_MAX_SCENES,
    )
    music_cues: list[RadioScoreDraftMusicCueV4] = Field(
        min_length=1, max_length=_RADIO_SCORE_MAX_MUSIC_CUES,
    )


_P3_TEXT_PATCH_MAX_TARGETS = 12
_P3_TEXT_PATCH_MAX_SOURCE_CHARS = 256
_P3_TEXT_PATCH_MAX_OUTPUT_TOKENS = 1024
_P3_TEXT_PATCH_PATH_MAX_CHARS = 80
_P3_TEXT_PATCH_SEAM = "codex_radio_score_text_patch"
_P3_TEXT_PATCH_KIND = "p3_authored_text_patch"


class _RadioScoreDraftTextPatchRowV4(_Strict):
    """One author-owned text replacement addressed by an opaque path token."""

    path: str = Field(min_length=1, max_length=_P3_TEXT_PATCH_PATH_MAX_CHARS)
    replacement_text: str = Field(min_length=1, max_length=144)


class _RadioScoreDraftTextPatchV4(_Strict):
    """Small P3-only patch root; never a replacement draft transport."""

    replacements: list[_RadioScoreDraftTextPatchRowV4] = Field(
        min_length=1, max_length=_P3_TEXT_PATCH_MAX_TARGETS,
    )


@dataclass(frozen=True)
class _P3TextPatchTarget:
    """One loc derived from the strict failed draft, never from model output."""

    loc: tuple[str | int, ...]
    path: str
    max_chars: int
    current_text: str


_DRAFT_ERROR_CODES = frozenset({
    "invalid_advisory", "beat_count", "shot_index", "unused_shot",
    "cast_id", "cast_coverage", "fact_id", "cue_id", "cue_anchor",
    "score_schema", "graph",
})


class RadioScoreDraftCompileError(ScifiCodexError):
    """Bounded, retry-safe reason why a complete draft cannot compile."""

    def __init__(self, *, code: str, path: str, detail: str) -> None:
        if code not in _DRAFT_ERROR_CODES:
            raise ValueError(f"unknown RadioScoreDraftCompileError code {code!r}")
        self.code = code
        self.path = " ".join(str(path).split())[:120] or "root"
        self.detail = " ".join(str(detail).split())[:240] or "invalid draft"
        super().__init__(f"draft.{self.code} at {self.path}: {self.detail}")


def _radio_score_draft_surface_receipt() -> dict[str, int | str | bool]:
    """Return the finite model-visible P3 draft surface and reservation."""
    return {
        "schema": "RadioScoreDraftV4",
        "context_cap_tokens": _RADIO_SCORE_CONTEXT_CAP_TOKENS,
        "max_new_tokens": _RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS,
        "input_token_reservation": (
            _RADIO_SCORE_CONTEXT_CAP_TOKENS
            - _RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS
        ),
        "full_result_json_schema_in_prompt": False,
        "max_scenes": _RADIO_SCORE_MAX_SCENES,
        "max_shots_per_scene": _RADIO_SCORE_MAX_SHOTS_PER_SCENE,
        "max_beats_per_scene": _RADIO_SCORE_MAX_BEATS_PER_SCENE,
        "max_total_shots": (
            _RADIO_SCORE_MAX_SCENES * _RADIO_SCORE_MAX_SHOTS_PER_SCENE
        ),
        "max_total_beats": _RADIO_SCORE_MAX_BEATS,
        "max_lines_per_beat": _RADIO_SCORE_MAX_LINES_PER_BEAT,
        "max_line_count_per_beat": _RADIO_SCORE_MAX_LINES_PER_BEAT,
        "max_music_cues": _RADIO_SCORE_MAX_MUSIC_CUES,
        "max_fact_ids_per_beat": _RADIO_SCORE_MAX_FACT_IDS_PER_BEAT,
        "max_title_chars": 64,
        "max_premise_chars": 144,
        "max_setting_chars": 80,
        "max_scene_env_chars": 56,
        "max_scene_description_chars": 72,
        "max_shot_description_chars": 72,
        "max_visual_prompt_chars": 120,
        "max_beat_intent_chars": 64,
        "max_arc_phase_chars": 28,
        "max_cue_description_chars": 80,
        "max_cue_generation_prompt_chars": 120,
    }


_RADIO_SCORE_DRAFT_SURFACE_INSTRUCTION = (
    "\nRadioScoreDraftV4 compact contract: return one JSON object only, rooted "
    "at exactly title, premise, setting, scenes, music_cues. title <=64; premise "
    "<=144; setting <=80. scenes has 1..3 items. Each scene has exactly env, "
    "description, shots, beats: env <=56; description <=72; shots has 1..2 items "
    "each with exactly description <=72 and visual_prompt <=120; beats has 1..4 "
    "items each with exactly shot_index, char_id, line_count, intent, arc_phase, "
    "fact_ids. shot_index is zero-based within this scene; char_id must be one "
    "accepted spoken cast ID; line_count is 1 or 2; intent <=64; arc_phase <=28; "
    "arc_phase is a narrative JSON string such as arrival, pressure, turn, or "
    "decision, never a number, word count, advisory center, or percentage. fact_ids "
    "is an ordered unique list of at most two allowed fact IDs. music_cues has 1..3 "
    "unique items, each exactly cue_id, description, generation_prompt, "
    "anchor_beat_index, anchor_line_index: cue_id MUST be exactly one of "
    "music_open, music_inter, music_close, never a descriptive music name; put any "
    "creative cue name in description. description <=80; generation_prompt <=120; "
    "anchor_beat_index is zero-based in flattened scene/beat order; "
    "anchor_line_index is zero-based within that beat. cue_id determines broad "
    "placement; the indices choose its exact anchor. Do not emit advisory_word_plan, "
    "any scene/shot/beat/line ID, order, parent, speaker, speaker_role, canonical "
    "cue anchor, spoken line text, wrapper, pass_id, artifact_inputs, or "
    "result_json_schema."
)


class ScriptLineV4(_Strict):
    line_id: str
    beat_id: str
    shot_id: str
    char_id: Literal[
        "announcer", "c01", "c02", "c03",
        "music_open", "music_inter", "music_close",
    ]
    speaker_role: Literal[
        "character", "announcer", "music_open", "music_close", "music_inter",
    ]
    text: str
    skip: bool = False
    tts_skip_reason: str | None = None
    traits: str = ""
    boundary: Literal["shot_start", "beat_start", "continue"]
    arc_phase: str
    compose_flags: list[str] = []
    beat_intent: str
    # The lane never builds a StoryRoom outline, so its accepted lines carry
    # the ledger's explicit ``None`` slot value.  A slot is metadata, not
    # dialogue, and making it nullable keeps the strict artifact aligned with
    # production_ledger.set_lines without fabricating an authored identifier.
    dialogue_slot_id: str | None
    fact_ids: list[str] = []


class ScriptArtifactV4(_Strict):
    schema_version: Literal["scifi_codex.script_artifact.v4"]
    title: str
    scenes: list[dict[str, Any]] = Field(min_length=1)
    lines: list[ScriptLineV4] = Field(min_length=1)
    music_cues: list[MusicCueV4] = Field(min_length=1)


_SCRIPT_ARTIFACT_FIELDS = frozenset(ScriptArtifactV4.model_fields)
_SCRIPT_LINE_FIELDS = frozenset(ScriptLineV4.model_fields)
_MUSIC_CUE_FIELDS = frozenset(MusicCueV4.model_fields)
_SCRIPT_LINE_REPAIRABLE_FIELDS = frozenset({"shot_id", "boundary"})
_SCRIPT_LINE_AUTHORED_FIELDS = frozenset({
    "line_id", "beat_id", "char_id", "speaker_role", "text",
    "arc_phase", "beat_intent", "dialogue_slot_id",
})
_SCRIPT_SCENE_FORBIDDEN_KEYS = frozenset({"speaker", "shots", "beats"})
_SCRIPT_ARTIFACT_ROOT_INSTRUCTION = (
    "\nSCRIPT ARTIFACT ROOT CONTRACT: Do not return a score, a scene, a beat, "
    "or a patch. Never echo the request envelope: pass_id, artifact_inputs, and "
    "result_json_schema are INPUT keys and must not appear anywhere in your "
    "output. Begin your response at the artifact root, with "
    '{"schema_version": "scifi_codex.script_artifact.v4", ...}. '
    "Return one root ScriptArtifactV4 object with exactly these root "
    "keys: schema_version, title, scenes, lines, music_cues. The input "
    "accepted_line_graph is a closed executable manifest: emit exactly one root-level "
    "lines item for every listed line_id, in the listed order, and no other lines. "
    "music_cues is separate cue metadata and never authorizes an additional music "
    "line. Output scenes are lightweight scene records only "
    "(scene_id, env, description); never nest shots or beats inside them."
)


class _PromptMustFitMessages(list[dict[str, str]]):
    """Tell the local slot wrapper to fail before it slices this prompt."""

    _otr_prompt_must_fit = True


class _P3TextPatchMessages(list[dict[str, str]]):
    """Keep the authored-text repair complete and output-bounded."""

    _otr_prompt_must_fit = True
    _otr_strict_remote_output_budget = True


def _has_forbidden_script_scene_keys(scenes: object) -> bool:
    """Reject score-shaped scene echoes without deleting story material."""
    if not isinstance(scenes, list):
        return True
    return any(
        not isinstance(scene, dict)
        or bool(_SCRIPT_SCENE_FORBIDDEN_KEYS & set(scene))
        for scene in scenes
    )


def _accepted_script_line_metadata(
    score: RadioScoreV4,
) -> dict[str, tuple[str, str, str]] | None:
    """Build the executable score order, or fail closed on graph ambiguity.

    This deliberately follows ``_assemble_ledger``: scene list order, then
    each scene's beat list order, then each beat's ordered ``line_ids``.  The
    model-facing ``BeatPlanV4.order`` field is not consumed by assembly, so it
    must not silently redefine production boundary semantics here.
    """
    scene_ids: set[str] = set()
    shot_ids: set[str] = set()
    beat_ids: set[str] = set()
    lines: dict[str, tuple[str, str, str]] = {}
    previous_shot: str | None = None
    previous_beat: str | None = None

    for scene in score.scenes:
        if not scene.scene_id or scene.scene_id in scene_ids:
            return None
        scene_ids.add(scene.scene_id)
        scene_shot_ids: set[str] = set()
        for shot in scene.shots:
            if (
                not shot.shot_id
                or shot.shot_id in shot_ids
                or shot.shot_id in scene_shot_ids
                or shot.scene_id != scene.scene_id
            ):
                return None
            shot_ids.add(shot.shot_id)
            scene_shot_ids.add(shot.shot_id)

        used_scene_shot_ids: set[str] = set()
        for beat in scene.beats:
            if (
                not beat.beat_id
                or beat.beat_id in beat_ids
                or beat.scene_id != scene.scene_id
                or beat.shot_id not in scene_shot_ids
            ):
                return None
            beat_ids.add(beat.beat_id)
            used_scene_shot_ids.add(beat.shot_id)
            for line_id in beat.line_ids:
                if not line_id or line_id in lines:
                    return None
                boundary = (
                    "shot_start"
                    if previous_shot != beat.shot_id
                    else "beat_start"
                    if previous_beat != beat.beat_id
                    else "continue"
                )
                lines[line_id] = (beat.beat_id, beat.shot_id, boundary)
                previous_shot = beat.shot_id
                previous_beat = beat.beat_id

        # A score with an orphaned shot cannot be used as an authoritative
        # mapping source; guessing which beat owns it would be a content edit.
        if used_scene_shot_ids != scene_shot_ids:
            return None

    return lines or None


def _validate_radio_score_graph(
    score: RadioScoreV4, advisory: AdvisoryWordPlanV4,
) -> str | None:
    """Accept only a score that closes the pre-P5 executable graph.

    P3 is the last stage allowed to decide the score's beats and line manifest.
    P5 may write line text but cannot safely invent a new line or infer where it
    belongs.  Locking the advisory plan here makes the score a complete source
    of truth before the metadata-only ScriptArtifactV4 repair ever runs.
    """
    if score.advisory_word_plan.model_dump(mode="json") != advisory.model_dump(mode="json"):
        return "score advisory_word_plan does not exactly match the locked plan"

    expected_beat_ids: list[str] = []
    for row in advisory.per_beat:
        expected_beat_ids.append(row.beat_id)
    if len(set(expected_beat_ids)) != len(expected_beat_ids):
        return "locked advisory plan has duplicate beat IDs"

    observed_beats = [
        beat
        for scene in score.scenes
        for beat in scene.beats
    ]
    if [beat.beat_id for beat in observed_beats] != expected_beat_ids:
        return "score beat IDs do not exactly match the locked advisory order"
    if [beat.order for beat in observed_beats] != list(range(1, len(observed_beats) + 1)):
        return "score beat order must be consecutive and match assembly order"

    line_metadata = _accepted_script_line_metadata(score)
    if line_metadata is None:
        return "score has missing or ambiguous executable line mappings"
    line_ids = list(line_metadata)
    canonical_line_ids = [f"l{i:03d}" for i in range(1, len(line_ids) + 1)]
    if line_ids != canonical_line_ids:
        return "score line IDs must be contiguous canonical IDs in assembly order"

    seen_cue_ids: set[str] = set()
    for cue in score.music_cues:
        if cue.cue_id in seen_cue_ids:
            return f"score has duplicate music cue {cue.cue_id!r}"
        seen_cue_ids.add(cue.cue_id)
        anchor = line_metadata.get(cue.anchor_line_id)
        if anchor is None:
            return f"music cue {cue.cue_id!r} anchors an unknown score line"
        if cue.anchor_beat_id != anchor[0]:
            return f"music cue {cue.cue_id!r} anchors the wrong score beat"
    return None


_DRAFT_SPOKEN_CHAR_IDS = frozenset({"announcer", "c01", "c02", "c03"})
_DRAFT_CUE_PLACEMENTS = {
    "music_open": "open",
    "music_inter": "inter",
    "music_close": "close",
}


def compile_radio_score_draft(
    draft: RadioScoreDraftV4,
    advisory: AdvisoryWordPlanV4,
    cast: CastPlanV4,
    fact_index: FactIndexV4,
) -> RadioScoreV4:
    """Compile model-owned score decisions into the strict executable score.

    The model owns story surface, scene-local shot choice, cast choice, line
    count, fact placement, and cue choice. The compiler owns only values that
    have one authoritative derivation from the accepted advisory/cast/facts:
    canonical IDs, parents, order, speaker metadata, word centers, cue
    placement, and canonical cue anchors.
    """
    advisory_rows = list(advisory.per_beat)
    advisory_ids = [row.beat_id for row in advisory_rows]
    if not advisory_rows or len(set(advisory_ids)) != len(advisory_ids):
        raise RadioScoreDraftCompileError(
            code="invalid_advisory", path="advisory.per_beat",
            detail="accepted advisory must contain unique beat IDs",
        )

    flat_draft_beat_count = sum(len(scene.beats) for scene in draft.scenes)
    if flat_draft_beat_count != len(advisory_rows):
        raise RadioScoreDraftCompileError(
            code="beat_count", path="scenes[*].beats",
            detail="flattened draft beat count must equal accepted advisory count",
        )

    cast_by_id = {row.char_id: row for row in cast.cast}
    if len(cast_by_id) != len(cast.cast) or "announcer" not in cast_by_id:
        raise RadioScoreDraftCompileError(
            code="cast_id", path="cast.cast",
            detail="accepted cast must contain unique IDs including announcer",
        )
    if any(char_id not in _DRAFT_SPOKEN_CHAR_IDS for char_id in cast_by_id):
        raise RadioScoreDraftCompileError(
            code="cast_id", path="cast.cast",
            detail="accepted cast contains an unsupported spoken ID",
        )

    accepted_fact_ids = [fact.fact_id for fact in fact_index.facts]
    if len(set(accepted_fact_ids)) != len(accepted_fact_ids):
        raise RadioScoreDraftCompileError(
            code="fact_id", path="fact_index.facts",
            detail="accepted fact index must contain unique fact IDs",
        )
    accepted_fact_id_set = set(accepted_fact_ids)

    compiled_scenes: list[ScenePlanV4] = []
    line_manifest: list[tuple[str, tuple[str, ...]]] = []
    used_cast_ids: set[str] = set()
    global_shot_number = 1
    global_beat_number = 0
    global_line_number = 1

    for scene_index, draft_scene in enumerate(draft.scenes):
        scene_path = f"scenes[{scene_index}]"
        scene_id = f"scene_{scene_index + 1:03d}"
        compiled_shots: list[ShotPlanV4] = []
        compiled_shot_ids: list[str] = []
        for draft_shot in draft_scene.shots:
            shot_id = f"shot_{global_shot_number:03d}"
            global_shot_number += 1
            compiled_shot_ids.append(shot_id)
            compiled_shots.append(ShotPlanV4(
                shot_id=shot_id,
                scene_id=scene_id,
                description=draft_shot.description,
                visual_prompt=draft_shot.visual_prompt,
            ))

        used_shot_indices: set[int] = set()
        compiled_beats: list[BeatPlanV4] = []
        for beat_index, draft_beat in enumerate(draft_scene.beats):
            beat_path = f"{scene_path}.beats[{beat_index}]"
            if not 0 <= draft_beat.shot_index < len(compiled_shot_ids):
                raise RadioScoreDraftCompileError(
                    code="shot_index", path=f"{beat_path}.shot_index",
                    detail="scene-local shot_index does not name a declared shot",
                )
            if draft_beat.char_id not in cast_by_id:
                raise RadioScoreDraftCompileError(
                    code="cast_id", path=f"{beat_path}.char_id",
                    detail="draft beat chooses a cast ID absent from accepted cast",
                )
            if len(set(draft_beat.fact_ids)) != len(draft_beat.fact_ids):
                raise RadioScoreDraftCompileError(
                    code="fact_id", path=f"{beat_path}.fact_ids",
                    detail="draft beat fact IDs must be unique",
                )
            for fact_id in draft_beat.fact_ids:
                if fact_id not in accepted_fact_id_set:
                    raise RadioScoreDraftCompileError(
                        code="fact_id", path=f"{beat_path}.fact_ids",
                        detail="draft beat references a fact absent from accepted P0",
                    )

            advisory_row = advisory_rows[global_beat_number]
            line_ids = tuple(
                f"l{line_number:03d}"
                for line_number in range(
                    global_line_number,
                    global_line_number + draft_beat.line_count,
                )
            )
            global_line_number += draft_beat.line_count
            cast_row = cast_by_id[draft_beat.char_id]
            speaker_role: Literal["character", "announcer"] = (
                "announcer" if draft_beat.char_id == "announcer" else "character"
            )
            compiled_beats.append(BeatPlanV4(
                beat_id=advisory_row.beat_id,
                scene_id=scene_id,
                shot_id=compiled_shot_ids[draft_beat.shot_index],
                speaker=cast_row.name,
                char_id=draft_beat.char_id,
                speaker_role=speaker_role,
                line_ids=list(line_ids),
                order=global_beat_number + 1,
                intent=draft_beat.intent,
                arc_phase=draft_beat.arc_phase,
                fact_ids=list(draft_beat.fact_ids),
                advisory_voiced_word_center=advisory_row.advisory_word_center,
            ))
            used_shot_indices.add(draft_beat.shot_index)
            used_cast_ids.add(draft_beat.char_id)
            line_manifest.append((advisory_row.beat_id, line_ids))
            global_beat_number += 1

        expected_shot_indices = set(range(len(compiled_shot_ids)))
        if used_shot_indices != expected_shot_indices:
            raise RadioScoreDraftCompileError(
                code="unused_shot", path=f"{scene_path}.shots",
                detail="every declared shot must own at least one beat",
            )
        compiled_scenes.append(ScenePlanV4(
            scene_id=scene_id,
            env=draft_scene.env,
            description=draft_scene.description,
            shots=compiled_shots,
            beats=compiled_beats,
        ))

    if used_cast_ids != set(cast_by_id):
        raise RadioScoreDraftCompileError(
            code="cast_coverage", path="scenes[*].beats[*].char_id",
            detail="every accepted cast ID must own at least one spoken beat",
        )

    compiled_cues: list[MusicCueV4] = []
    seen_cue_ids: set[str] = set()
    for cue_index, draft_cue in enumerate(draft.music_cues):
        cue_path = f"music_cues[{cue_index}]"
        if draft_cue.cue_id in seen_cue_ids:
            raise RadioScoreDraftCompileError(
                code="cue_id", path=f"{cue_path}.cue_id",
                detail="draft music cue IDs must be unique",
            )
        seen_cue_ids.add(draft_cue.cue_id)
        if not 0 <= draft_cue.anchor_beat_index < len(line_manifest):
            raise RadioScoreDraftCompileError(
                code="cue_anchor", path=f"{cue_path}.anchor_beat_index",
                detail="cue anchor beat index is outside the flattened draft beats",
            )
        anchor_beat_id, anchor_line_ids = line_manifest[draft_cue.anchor_beat_index]
        if not 0 <= draft_cue.anchor_line_index < len(anchor_line_ids):
            raise RadioScoreDraftCompileError(
                code="cue_anchor", path=f"{cue_path}.anchor_line_index",
                detail="cue anchor line index is outside its selected beat",
            )
        compiled_cues.append(MusicCueV4(
            cue_id=draft_cue.cue_id,
            placement=_DRAFT_CUE_PLACEMENTS[draft_cue.cue_id],
            description=draft_cue.description,
            generation_prompt=draft_cue.generation_prompt,
            anchor_line_id=anchor_line_ids[draft_cue.anchor_line_index],
            anchor_beat_id=anchor_beat_id,
        ))

    try:
        score = RadioScoreV4(
            title=draft.title,
            premise=draft.premise,
            setting=draft.setting,
            advisory_word_plan=advisory.model_copy(deep=True),
            scenes=compiled_scenes,
            music_cues=compiled_cues,
        )
    except ValidationError as exc:
        raise RadioScoreDraftCompileError(
            code="score_schema", path="compiled_score",
            detail=str(exc),
        ) from exc
    graph_error = _validate_radio_score_graph(score, advisory)
    if graph_error is not None:
        raise RadioScoreDraftCompileError(
            code="graph", path="compiled_score", detail=graph_error,
        )
    return score


def _radio_score_draft_structure_signature(
    draft: RadioScoreDraftV4,
) -> tuple[Any, ...]:
    """Return only rewrite-locked mechanical draft decisions."""
    return (
        tuple(
            (
                len(scene.shots),
                tuple(
                    (beat.shot_index, beat.char_id, beat.line_count)
                    for beat in scene.beats
                ),
            )
            for scene in draft.scenes
        ),
        tuple(
            (cue.cue_id, cue.anchor_beat_index, cue.anchor_line_index)
            for cue in draft.music_cues
        ),
    )


def project_radio_score_to_draft(score: RadioScoreV4) -> RadioScoreDraftV4:
    """Represent a compiled score in the exact compact P3-rewrite transport."""
    draft_scenes: list[dict[str, Any]] = []
    line_positions: dict[str, tuple[int, int, str]] = {}
    flat_beat_index = 0
    for scene_index, scene in enumerate(score.scenes):
        shot_index_by_id = {shot.shot_id: index for index, shot in enumerate(scene.shots)}
        if len(shot_index_by_id) != len(scene.shots):
            raise RadioScoreDraftCompileError(
                code="score_schema", path=f"scenes[{scene_index}].shots",
                detail="compiled score has duplicate shot IDs",
            )
        draft_beats: list[dict[str, Any]] = []
        for beat_index, beat in enumerate(scene.beats):
            beat_path = f"scenes[{scene_index}].beats[{beat_index}]"
            if beat.shot_id not in shot_index_by_id:
                raise RadioScoreDraftCompileError(
                    code="score_schema", path=f"{beat_path}.shot_id",
                    detail="compiled beat does not belong to a scene shot",
                )
            if beat.char_id not in _DRAFT_SPOKEN_CHAR_IDS:
                raise RadioScoreDraftCompileError(
                    code="score_schema", path=f"{beat_path}.char_id",
                    detail="compiled score uses a non-spoken beat that draft cannot represent",
                )
            if not 1 <= len(beat.line_ids) <= _RADIO_SCORE_MAX_LINES_PER_BEAT:
                raise RadioScoreDraftCompileError(
                    code="score_schema", path=f"{beat_path}.line_ids",
                    detail="compiled score has an unsupported line count",
                )
            for line_index, line_id in enumerate(beat.line_ids):
                if line_id in line_positions:
                    raise RadioScoreDraftCompileError(
                        code="score_schema", path=f"{beat_path}.line_ids",
                        detail="compiled score repeats a line ID",
                )
                line_positions[line_id] = (flat_beat_index, line_index, beat.beat_id)
            draft_beats.append({
                "shot_index": shot_index_by_id[beat.shot_id],
                "char_id": beat.char_id,
                "line_count": len(beat.line_ids),
                "intent": beat.intent,
                "arc_phase": beat.arc_phase,
                "fact_ids": list(beat.fact_ids),
            })
            flat_beat_index += 1
        draft_scenes.append({
            "env": scene.env,
            "description": scene.description,
            "shots": [
                {
                    "description": shot.description,
                    "visual_prompt": shot.visual_prompt,
                }
                for shot in scene.shots
            ],
            "beats": draft_beats,
        })

    draft_cues: list[dict[str, Any]] = []
    seen_cue_ids: set[str] = set()
    for cue_index, cue in enumerate(score.music_cues):
        cue_path = f"music_cues[{cue_index}]"
        if cue.cue_id in seen_cue_ids:
            raise RadioScoreDraftCompileError(
                code="cue_id", path=f"{cue_path}.cue_id",
                detail="compiled score has duplicate music cue IDs",
            )
        seen_cue_ids.add(cue.cue_id)
        anchor = line_positions.get(cue.anchor_line_id)
        if anchor is None or cue.anchor_beat_id != anchor[2]:
            raise RadioScoreDraftCompileError(
                code="cue_anchor", path=f"{cue_path}.anchor_line_id",
                detail="compiled cue does not point to its owning score line",
            )
        if cue.placement != _DRAFT_CUE_PLACEMENTS[cue.cue_id]:
            raise RadioScoreDraftCompileError(
                code="cue_id", path=f"{cue_path}.placement",
                detail="compiled cue placement does not match its cue ID",
            )
        draft_cues.append({
            "cue_id": cue.cue_id,
            "description": cue.description,
            "generation_prompt": cue.generation_prompt,
            "anchor_beat_index": anchor[0],
            "anchor_line_index": anchor[1],
        })
    try:
        return RadioScoreDraftV4(
            title=score.title,
            premise=score.premise,
            setting=score.setting,
            scenes=draft_scenes,
            music_cues=draft_cues,
        )
    except ValidationError as exc:
        raise RadioScoreDraftCompileError(
            code="score_schema", path="projected_draft", detail=str(exc),
        ) from exc


def _compact_p0_fact_context(fact_index: FactIndexV4) -> dict[str, Any]:
    """Keep P3's fact grounding without P0 span/provenance bulk."""
    return {
        "facts": [
            {"fact_id": fact.fact_id, "claim": fact.claim}
            for fact in fact_index.facts
        ],
        "tone": fact_index.tone,
    }


def repair_script_artifact_metadata(
    failed_output: str, score: RadioScoreV4,
) -> ScriptArtifactV4 | None:
    """Normalize only deterministic ScriptArtifactV4 metadata, or return None.

    The repair never creates, removes, or rewrites story text.  It uses the
    already accepted score graph as the sole authority for line shot IDs and
    boundary transitions, removes strict-model extras, and fails closed when a
    graph or raw-line mapping is incomplete or ambiguous.
    """
    def refuse(reason: str) -> None:
        log.info("[scifi_codex:ScriptArtifactV4] deterministic repair declined: %s", reason)

    expected = _accepted_script_line_metadata(score)
    if expected is None:
        refuse("accepted score graph is missing or ambiguous")
        return None
    try:
        data = parse_first_json_object(failed_output)
    except Exception as exc:
        refuse(f"failed output has no complete top-level object: {exc}")
        return None
    if not isinstance(data, dict):
        refuse("top-level artifact is not an object")
        return None
    raw_lines = data.get("lines")
    raw_music_cues = data.get("music_cues")
    raw_scenes = data.get("scenes")
    if (
        not isinstance(raw_lines, list)
        or not isinstance(raw_music_cues, list)
        or _has_forbidden_script_scene_keys(raw_scenes)
    ):
        refuse("artifact has missing graph arrays or score-shaped scene fields")
        return None

    changed = data.get("schema_version") != "scifi_codex.script_artifact.v4"
    repaired_lines: list[dict[str, Any]] = []
    observed_line_ids: set[str] = set()
    for raw_line in raw_lines:
        if not isinstance(raw_line, dict):
            refuse("script lines contain a non-object")
            return None
        line_id = raw_line.get("line_id")
        if (
            not isinstance(line_id, str)
            or not line_id
            or line_id in observed_line_ids
            or line_id not in expected
        ):
            refuse(f"script line mapping is missing, duplicate, or unknown: {line_id!r}")
            return None
        observed_line_ids.add(line_id)
        # The authored/script-meaning fields must already exist byte-for-byte.
        # The remaining absent fields are Pydantic's declared neutral metadata
        # defaults (skip/traits/compose flags/fact IDs); accepting those exact
        # defaults does not invent dialogue, premise, beat, or intent content.
        if not _SCRIPT_LINE_AUTHORED_FIELDS <= set(raw_line):
            refuse(f"line {line_id} is missing an authored field")
            return None
        repaired_line = {
            key: value for key, value in raw_line.items()
            if key in _SCRIPT_LINE_FIELDS
        }
        if len(repaired_line) != len(raw_line):
            changed = True
        beat_id, shot_id, boundary = expected[line_id]
        if raw_line.get("beat_id") != beat_id:
            refuse(f"line {line_id} does not retain its accepted beat")
            return None
        if raw_line.get("shot_id") != shot_id:
            changed = True
        if raw_line.get("boundary") != boundary:
            changed = True
        repaired_line["shot_id"] = shot_id
        repaired_line["boundary"] = boundary
        repaired_lines.append(repaired_line)
    if observed_line_ids != set(expected):
        refuse("script line IDs do not exactly cover the accepted score graph")
        return None

    repaired_music_cues: list[dict[str, Any]] = []
    for raw_cue in raw_music_cues:
        if not isinstance(raw_cue, dict):
            refuse("music cues contain a non-object")
            return None
        repaired_cue = {
            key: value for key, value in raw_cue.items()
            if key in _MUSIC_CUE_FIELDS
        }
        if len(repaired_cue) != len(raw_cue):
            changed = True
        repaired_music_cues.append(repaired_cue)

    repaired = {
        key: value for key, value in data.items()
        if key in _SCRIPT_ARTIFACT_FIELDS
    }
    if len(repaired) != len(data):
        changed = True
    repaired["schema_version"] = "scifi_codex.script_artifact.v4"
    repaired["lines"] = repaired_lines
    repaired["music_cues"] = repaired_music_cues
    if not changed:
        refuse("artifact has no deterministic metadata defect")
        return None
    try:
        return ScriptArtifactV4.model_validate(repaired)
    except Exception as exc:
        refuse(f"metadata-only result does not satisfy ScriptArtifactV4: {exc}")
        return None


def _validate_script_graph(script: ScriptArtifactV4, score: RadioScoreV4) -> None:
    """Require the accepted artifact to retain the score's exact metadata graph."""
    expected = _accepted_script_line_metadata(score)
    if expected is None:
        raise CodexGraphError("accepted score has missing or ambiguous script-line mappings")
    if _has_forbidden_script_scene_keys(script.scenes):
        raise CodexGraphError("script scenes contain forbidden score or legacy fields")
    observed: dict[str, ScriptLineV4] = {}
    for line in script.lines:
        if not line.line_id or line.line_id in observed:
            raise CodexGraphError("script artifact has a missing or duplicate line_id")
        observed[line.line_id] = line
    if set(observed) != set(expected):
        raise CodexGraphError("script line IDs do not exactly match the accepted score graph")
    for line_id, line in observed.items():
        beat_id, shot_id, boundary = expected[line_id]
        if line.beat_id != beat_id:
            raise CodexGraphError(f"line {line_id} does not resolve to its accepted beat")
        if line.shot_id != shot_id:
            raise CodexGraphError(f"line {line_id} does not resolve to its accepted shot")
        if line.boundary != boundary:
            raise CodexGraphError(f"line {line_id} has an invalid accepted-order boundary")


class StructureReviewV4(_Strict):
    verdict: Literal["pass", "rewrite"]
    issues: list[Annotated[str, Field(min_length=1, max_length=120)]] = Field(
        default_factory=list, max_length=6,
    )
    rationale: str = Field(default="", max_length=240)


class ListenerIssueV4(_Strict):
    """One diagnosis from the listening room.

    `issues` used to be typed `list[dict[str, str]]` -- a shapeless container. The seam
    names six diagnostic lenses (blurred causality, interchangeable voices, lecture,
    unused sound, stalled pacing, overclaiming coda) but never said what ONE issue looks
    like, so a model had to guess between a list of issues and a dict grouping issues
    under those six lenses. Both readings are reasonable; we were winning a coin flip.
    Gemma-4 called it the other way (`{"blurred_causality": [...]}`) and P6 died.

    So the SHAPE is pinned here and the VOCABULARY is left to the model: `category` is
    free text, not an enum, because a listener who coins a better word for the flaw than
    our six is doing its job, and a schema should never reject it for that.
    """

    category: str
    line_id: str | None = None
    direction: str


class ListenerReviewV4(_Strict):
    strengths: list[str] = []
    issues: list[ListenerIssueV4] = []
    require_full_retake: bool = True


# The listener's own words for the issue text.  Ordered by how directly the key names
# "what the writer should do"; the first key present wins.  Python never writes the
# direction -- it only finds which key the model put its sentence under.
_LISTENER_DIRECTION_KEYS = (
    "direction", "detail", "fix", "note", "issue", "problem",
    "suggestion", "description", "comment", "text",
)
_LISTENER_CATEGORY_KEYS = ("category", "type", "kind", "lens", "label")
_LISTENER_LINE_KEYS = ("line_id", "line", "line_ref", "id")


def _listener_issue_from(item: object, category: str) -> ListenerIssueV4 | None:
    """Map one loose issue into the typed shape, using only the model's own text."""
    if isinstance(item, str):
        return ListenerIssueV4(category=category, direction=item) if item.strip() else None
    if not isinstance(item, Mapping):
        return None

    direction = next(
        (
            str(item[k]).strip()
            for k in _LISTENER_DIRECTION_KEYS
            if isinstance(item.get(k), str) and str(item[k]).strip()
        ),
        None,
    )
    if direction is None:
        # A single unlabelled sentence is unambiguous; anything else would be a guess.
        strings = [v.strip() for v in item.values() if isinstance(v, str) and v.strip()]
        if len(strings) != 1:
            return None
        direction = strings[0]

    label = next(
        (
            str(item[k]).strip()
            for k in _LISTENER_CATEGORY_KEYS
            if isinstance(item.get(k), str) and str(item[k]).strip()
        ),
        category,
    )
    line_id = next(
        (
            str(item[k]).strip()
            for k in _LISTENER_LINE_KEYS
            if isinstance(item.get(k), str) and str(item[k]).strip()
        ),
        None,
    )
    return ListenerIssueV4(category=label or "unlabelled", line_id=line_id, direction=direction)


def repair_listener_review_shape(failed_output: str) -> ListenerReviewV4 | None:
    """Flatten a grouped listener review into the flat typed shape.

    A model told to look for six kinds of flaw may reasonably return
    `{"blurred_causality": [...], "stalled_pacing": [...]}` instead of a flat list --
    that is a container-shape choice, not a story defect, so Python is allowed to
    normalize it.  It re-homes the model's sentences; it never writes one.  Fails
    closed (returns None) on any issue it cannot map without inventing text, so the
    typed repair still gets its turn rather than a diagnosis being silently dropped.
    """
    try:
        raw = parse_first_json_object(failed_output)
    except Exception:
        return None
    if not isinstance(raw, Mapping):
        return None

    issues_in = raw.get("issues")
    flattened: list[ListenerIssueV4] = []
    if isinstance(issues_in, Mapping):
        for category, group in issues_in.items():
            items = group if isinstance(group, list) else [group]
            for item in items:
                mapped = _listener_issue_from(item, str(category))
                if mapped is None:
                    return None
                flattened.append(mapped)
    elif isinstance(issues_in, list):
        for item in issues_in:
            mapped = _listener_issue_from(item, "unlabelled")
            if mapped is None:
                return None
            flattened.append(mapped)
    elif issues_in is not None:
        return None

    strengths = [s.strip() for s in raw.get("strengths") or [] if isinstance(s, str) and s.strip()]
    retake = raw.get("require_full_retake")
    return ListenerReviewV4(
        strengths=strengths,
        issues=flattened,
        require_full_retake=retake if isinstance(retake, bool) else True,
    )


class FinalIssueV1(_Strict):
    issue_id: str
    severity: Literal["critical", "advisory"]
    line_id: str | None = None
    detail: str


class LineCheckV1(_Strict):
    line_id: str
    passed: bool
    fact_ids: list[str] = []


class FactCheckV1(_Strict):
    fact_id: str
    source_spans: list[str]
    audible: bool


class FinalAuditV4(_Strict):
    schema_version: Literal["scifi_codex.final_audit.v4"]
    script_digest: str
    verdict: Literal["pass", "rewrite"]
    issues: list[FinalIssueV1] = []
    line_checks: list[LineCheckV1] = []
    fact_checks: list[FactCheckV1] = []
    # `observed_word_counts: dict[str, int]` was REQUIRED here, and the audit seam told
    # the auditor to check the "exact word count" and pass only if every check is true.
    # That made an LLM -- which cannot count words -- the enforcer of a word-count gate,
    # and a disagreement it was bound to have could demand a full P9 rewrite. Word count
    # is advisory and never a gate; Python already measures the real one objectively in
    # the tail's word_receipt. Nothing ever read this field. Ripped, seam and all.


GenerateFn = Callable[..., str]


def _p3_text_patch_transport(slot_fn: GenerateFn) -> str | None:
    """Return one explicitly proven authored-text patch transport."""
    declared = getattr(slot_fn, "_otr_p3_text_patch_transport", None)
    if declared in ("exact_local", "full_message_remote"):
        return str(declared)
    return None


def _is_p3_patch_index(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _p3_text_patch_cap(loc: tuple[str | int, ...]) -> int | None:
    """Return the exact author-owned cap for one trusted Pydantic loc tuple."""
    if not all(isinstance(item, str) or _is_p3_patch_index(item) for item in loc):
        return None
    if len(loc) == 1 and isinstance(loc[0], str):
        return {"title": 64, "premise": 144, "setting": 80}.get(loc[0])
    if (
        len(loc) == 3
        and loc[0] == "scenes"
        and _is_p3_patch_index(loc[1])
        and isinstance(loc[2], str)
    ):
        return {"env": 56, "description": 72}.get(loc[2])
    if (
        len(loc) == 5
        and loc[0] == "scenes"
        and _is_p3_patch_index(loc[1])
        and loc[2] == "shots"
        and _is_p3_patch_index(loc[3])
        and isinstance(loc[4], str)
    ):
        return {"description": 72, "visual_prompt": 120}.get(loc[4])
    if (
        len(loc) == 5
        and loc[0] == "scenes"
        and _is_p3_patch_index(loc[1])
        and loc[2] == "beats"
        and _is_p3_patch_index(loc[3])
        and isinstance(loc[4], str)
    ):
        return {"intent": 64, "arc_phase": 28}.get(loc[4])
    if (
        len(loc) == 3
        and loc[0] == "music_cues"
        and _is_p3_patch_index(loc[1])
        and isinstance(loc[2], str)
    ):
        return {"description": 80, "generation_prompt": 120}.get(loc[2])
    return None


def _p3_text_patch_value_at(raw: object, loc: tuple[str | int, ...]) -> object | None:
    """Read only a trusted loc from a decoded root; never infer a path."""
    node: object = raw
    for item in loc:
        if isinstance(item, str):
            if not isinstance(node, Mapping) or item not in node:
                return None
            node = node[item]
        elif _is_p3_patch_index(item):
            if not isinstance(node, list) or item >= len(node):
                return None
            node = node[item]
        else:
            return None
    return node


def _p3_text_patch_set_at(
    raw: dict[str, Any], loc: tuple[str | int, ...], replacement: str,
) -> None:
    """Set a leaf through a previously derived trusted loc tuple only."""
    if not loc:
        raise ValueError("empty P3 text-patch location")
    node: object = raw
    for item in loc[:-1]:
        if isinstance(item, str):
            if not isinstance(node, dict) or item not in node:
                raise ValueError("unreachable P3 text-patch object path")
            node = node[item]
        elif _is_p3_patch_index(item):
            if not isinstance(node, list) or item >= len(node):
                raise ValueError("unreachable P3 text-patch list path")
            node = node[item]
        else:
            raise ValueError("invalid P3 text-patch path segment")
    leaf = loc[-1]
    if isinstance(leaf, str):
        if not isinstance(node, dict) or leaf not in node:
            raise ValueError("unreachable P3 text-patch leaf")
        node[leaf] = replacement
        return
    if _is_p3_patch_index(leaf):
        if not isinstance(node, list) or leaf >= len(node):
            raise ValueError("unreachable P3 text-patch list leaf")
        node[leaf] = replacement
        return
    raise ValueError("invalid P3 text-patch leaf segment")


def _derive_p3_text_patch_targets(
    raw: object, error: ValidationError,
) -> tuple[_P3TextPatchTarget, ...] | None:
    """Accept only a small, fully proven set of authored over-cap leaves."""
    if not isinstance(raw, dict):
        return None
    errors = error.errors()
    if not errors:
        return None
    seen_locs: set[tuple[str | int, ...]] = set()
    targets: list[_P3TextPatchTarget] = []
    for row in errors:
        if row.get("type") != "string_too_long":
            return None
        raw_loc = row.get("loc")
        if not isinstance(raw_loc, tuple):
            return None
        loc = tuple(raw_loc)
        if not all(
            isinstance(item, str) or _is_p3_patch_index(item)
            for item in loc
        ):
            return None
        if loc in seen_locs:
            return None
        expected_cap = _p3_text_patch_cap(loc)
        observed_cap = (row.get("ctx") or {}).get("max_length")
        if (
            expected_cap is None
            or not isinstance(observed_cap, int)
            or isinstance(observed_cap, bool)
            or observed_cap != expected_cap
        ):
            return None
        current_text = _p3_text_patch_value_at(raw, loc)
        if (
            not isinstance(current_text, str)
            or len(current_text) <= expected_cap
            or len(current_text) > _P3_TEXT_PATCH_MAX_SOURCE_CHARS
        ):
            return None
        path = ".".join(str(item) for item in loc)
        if not path or len(path) > _P3_TEXT_PATCH_PATH_MAX_CHARS:
            return None
        seen_locs.add(loc)
        targets.append(_P3TextPatchTarget(
            loc=loc,
            path=path,
            max_chars=expected_cap,
            current_text=current_text,
        ))
    if not 1 <= len(targets) <= _P3_TEXT_PATCH_MAX_TARGETS:
        return None
    return tuple(targets)


def _p3_text_patch_preflight(
    raw: dict[str, Any],
    targets: tuple[_P3TextPatchTarget, ...],
    post_validator: Callable[[BaseModel], str | None],
) -> bool:
    """Prove that only the selected prose leaves block the complete contract."""
    probe = copy.deepcopy(raw)
    try:
        for target in targets:
            _p3_text_patch_set_at(probe, target.loc, "x")
        candidate = RadioScoreDraftV4.model_validate(probe)
        return post_validator(candidate) is None
    except Exception:
        # A hidden schema/compiler/signature/graph defect is broader than this
        # patch. Preserve the existing full typed repair rather than guessing.
        return False


def _p3_text_patch_messages(
    pack: Any, targets: tuple[_P3TextPatchTarget, ...],
) -> list[dict[str, str]]:
    """Build the small, non-copyable local authoring request."""
    stages = getattr(pack, "prompt_stages", {}) or {}
    seam = str(stages.get(_P3_TEXT_PATCH_SEAM) or "").strip()
    if not seam:
        raise CodexPackContractError(
            f"missing Codex seam {_P3_TEXT_PATCH_SEAM!r}"
        )
    target_rows = [
        {
            "path": target.path,
            "max_chars": target.max_chars,
            "original_text": target.current_text,
        }
        for target in targets
    ]
    return [
        {
            "role": "system",
            "content": seam + "\nReturn one JSON object only, rooted at "
            "replacements. The target list is input evidence, never an output "
            "template. Do not return a RadioScoreDraftV4, a wrapper, a request "
            "field, or an explanation.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {"targets": target_rows},
                sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            ),
        },
    ]


def _p3_text_patch_contract_error(code: str) -> PostValidationError:
    """Return a bounded failure that cannot serialize rejected patch prose."""
    return PostValidationError(f"draft.text_patch at root: {code}")


def _merge_p3_text_patch(
    raw: dict[str, Any],
    targets: tuple[_P3TextPatchTarget, ...],
    patch: _RadioScoreDraftTextPatchV4,
) -> dict[str, Any]:
    """Apply exact one-for-one model prose only through trusted target locs."""
    by_path = {target.path: target for target in targets}
    if len(patch.replacements) != len(by_path):
        raise _p3_text_patch_contract_error("replacement_count")
    replacements: dict[str, str] = {}
    for row in patch.replacements:
        target = by_path.get(row.path)
        if target is None:
            raise _p3_text_patch_contract_error("unknown_path")
        if row.path in replacements:
            raise _p3_text_patch_contract_error("duplicate_path")
        if not row.replacement_text.strip():
            raise _p3_text_patch_contract_error("blank_replacement")
        if len(row.replacement_text) > target.max_chars:
            raise _p3_text_patch_contract_error("replacement_over_cap")
        replacements[row.path] = row.replacement_text
    if set(replacements) != set(by_path):
        raise _p3_text_patch_contract_error("missing_path")
    merged = copy.deepcopy(raw)
    try:
        for target in targets:
            _p3_text_patch_set_at(
                merged, target.loc, replacements[target.path],
            )
    except Exception as exc:
        raise _p3_text_patch_contract_error("trusted_path_unreachable") from exc
    return merged


def _run_p3_text_patch(
    *,
    slot_fn: GenerateFn,
    pack: Any,
    raw_draft: dict[str, Any],
    targets: tuple[_P3TextPatchTarget, ...],
    post_validator: Callable[[BaseModel], str | None],
    calls: list[dict[str, Any]],
    mark_attempt_complete: Callable[[int, str, BaseException | None], None],
    patch_transport: str,
) -> RadioScoreDraftV4:
    """Make P3's sole author-owned text patch call with a complete receipt."""
    patch_attempt_index = len(calls) + 1
    receipt: dict[str, Any] = {
        "temperature": REPAIR_TEMPERATURE,
        "max_new_tokens": _P3_TEXT_PATCH_MAX_OUTPUT_TOKENS,
        "raw_chars": None,
        "raw_sha256": None,
        "resolved_artifact_unwrapped": False,
        "parse_status": "pending",
        "schema_status": "pending",
        "draft_status": "pending",
        "compiler_status": "pending",
        "graph_status": "pending",
        "repair_kind": _P3_TEXT_PATCH_KIND,
        "patch_transport": patch_transport,
        "patch_status": "pending",
        "patch_targets": [
            {"path": target.path, "max_chars": target.max_chars}
            for target in targets
        ],
    }
    calls.append(receipt)
    if len(calls) != patch_attempt_index:
        raise CodexPackContractError("P3 text-patch receipt index drift")

    try:
        raw_patch = str(invoke_structured_slot(
            slot_fn,
            _P3TextPatchMessages(_p3_text_patch_messages(pack, targets)),
            temperature=REPAIR_TEMPERATURE,
            max_new_tokens=_P3_TEXT_PATCH_MAX_OUTPUT_TOKENS,
        ))
    except Exception:
        receipt.update({
            "parse_status": "terminal_error",
            "schema_status": "not_run",
            "draft_status": "terminal_error",
            "compiler_status": "not_run",
            "graph_status": "not_run",
            "patch_status": "terminal_error",
            "patch_error_code": "provider",
        })
        raise
    receipt.update({
        "raw_chars": len(raw_patch),
        "raw_sha256": hashlib.sha256(raw_patch.encode("utf-8")).hexdigest(),
    })

    try:
        patch_data = parse_first_json_object(raw_patch)
    except Exception as exc:
        receipt.update({
            "parse_status": "not_decoded",
            "schema_status": "not_run",
            "draft_status": "not_decoded",
            "compiler_status": "not_run",
            "graph_status": "not_run",
            "patch_status": "not_decoded",
            "patch_error_code": "json",
        })
        raise _p3_text_patch_contract_error("patch_json") from exc
    try:
        patch = _RadioScoreDraftTextPatchV4.model_validate(patch_data)
    except ValidationError as exc:
        receipt.update({
            "parse_status": "decoded",
            "schema_status": "rejected",
            "draft_status": "schema_rejected",
            "compiler_status": "not_run",
            "graph_status": "not_run",
            "patch_status": "schema_rejected",
            "patch_error_code": "patch_root",
        })
        raise _p3_text_patch_contract_error("patch_root") from exc
    try:
        merged_raw = _merge_p3_text_patch(raw_draft, targets, patch)
        merged_draft = RadioScoreDraftV4.model_validate(merged_raw)
    except PostValidationError:
        receipt.update({
            "parse_status": "decoded",
            "schema_status": "accepted",
            "draft_status": "patch_contract_rejected",
            "compiler_status": "not_run",
            "graph_status": "not_run",
            "patch_status": "contract_rejected",
            "patch_error_code": "coverage",
        })
        raise
    except ValidationError as exc:
        receipt.update({
            "parse_status": "decoded",
            "schema_status": "rejected",
            "draft_status": "schema_rejected",
            "compiler_status": "not_run",
            "graph_status": "not_run",
            "patch_status": "schema_rejected",
            "patch_error_code": "merged_draft",
        })
        raise _p3_text_patch_contract_error("merged_draft") from exc
    validation_error = post_validator(merged_draft)
    if validation_error is not None:
        detail = str(validation_error)
        draft_code = ""
        if detail.startswith("draft."):
            draft_code = detail.split(" ", 1)[0].removeprefix("draft.")
        receipt.update({
            "parse_status": "decoded",
            "schema_status": "accepted",
            "draft_status": "compiler_rejected",
            "compiler_status": draft_code or "post_validation_rejected",
            "graph_status": "rejected" if draft_code == "graph" else "not_run",
            "patch_status": "post_validation_rejected",
            "patch_error_code": "merged_contract",
        })
        raise PostValidationError(validation_error)

    mark_attempt_complete(patch_attempt_index, raw_patch, None)
    receipt["patch_status"] = "accepted"
    return merged_draft
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’-]*")
_DECORATION_RE = re.compile(r"[\r\n\t\[\]()\x60*]|^\s*\x60\x60\x60|^\s*[-*]\s+")
_ALL_CAPS_RE = re.compile(r"\b[A-Z]{2,}\b")
_LABEL_RE = re.compile(r"^\s*(?:ANNOUNCER|[A-Z][A-Za-z]+)\s*:")
_QUOTED_RE = re.compile(r"^\s*[\"'].*[\"']\s*$")


def _words(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _digest(payload: Mapping[str, str]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_payload_envelope(
    payload: Mapping[str, Any], resolved: Mapping[str, Any]
) -> tuple[PayloadEnvelopeV4, WordSteerV4]:
    try:
        clean = validate_source_payload(dict(payload), "scifi_codex")
    except Exception as exc:
        raise CodexPayloadShapeError(str(exc)) from exc
    seed_source = str(resolved.get("seed_source") or "")
    if seed_source == "custom_premise":
        mode = "operator_pinned"
        if len(clean["seed_text"].split()) < 8 or len(set(re.findall(r"[A-Za-z0-9]+", clean["seed_text"].lower()))) < 4:
            raise CodexPayloadThinError("operator-pinned payload is below the 8/4 thinness floor")
    elif seed_source == "rss_fetch":
        mode = "rss"
        if not meets_v4_rss_source_floor(clean["full_text"]):
            raise CodexPayloadThinError(
                "RSS payload is below the 400/80/12 v4 source floor"
            )
    else:
        raise CodexPayloadRouteError(
            "scifi_codex accepts only seed_source='rss_fetch' or 'custom_premise'"
        )
    if len(json.dumps(clean, ensure_ascii=False).encode("utf-8")) > 48_000:
        raise CodexPayloadOversizeError("source payload exceeds the 48,000-byte cap")
    try:
        steer = WordSteerV4(requested_words=resolved.get("target_words"))
    except Exception as exc:
        raise CodexTargetRangeError("target_words must be an integer from 30 through 900") from exc
    env = PayloadEnvelopeV4(
        schema_version="scifi_codex.payload_envelope.v4",
        payload=SourcePayload7.model_validate(clean),
        source_mode=mode,
        source_digest=_digest(clean),
    )
    return env, steer


def _p0_evidence_projection(
    envelope: PayloadEnvelopeV4,
) -> tuple[dict[str, str], frozenset[str]]:
    """Return the lossless, de-aliased source view P0 may cite.

    A0 remains the full canonical seven-key payload: it is hashed and remains
    the sole coordinate system for source-span validation.  P0 only needs
    fields that ``SourceSpanV4`` permits, though.  In particular, the RSS
    adapter commonly carries its fallback body as both ``summary`` and
    ``full_text``, then derives ``seed_text`` from headline plus summary.
    Repeating those aliases can push the model prompt past its context window
    and make the low-level prompt guard discard the schema contract.

    This projection never slices or rewrites text.  It builds a lossless
    *coordinate cover*: retain a source string unless it is already an exact
    substring of a retained one.  ``seed_text`` is considered first because a
    derived headline+summary can legally contain a quote that crosses the two
    original fields and therefore cannot be rehomed to either field alone.
    The returned allowlist prevents a model repair from reintroducing an
    omitted alias as a source-span field; the deterministic literal-span
    repair rehomes exact quotes into this cover when needed.
    """
    payload = envelope.payload.model_dump(mode="json")
    # A pinned premise is canonically supplied in seed_text; RSS seed_text is
    # often a headline+summary superstring.  In both cases it is the best
    # first field for preserving valid literal-span coordinates.
    field_order = ["seed_text", "full_text", "headline", "summary"]

    projected: dict[str, str] = {}
    retained_values: list[str] = []
    for field in field_order:
        value = payload[field]
        if value and not any(value in retained for retained in retained_values):
            projected[field] = value
            retained_values.append(value)
    if not projected:
        raise CodexPayloadShapeError("P0 has no legal source evidence fields")
    return projected, frozenset(projected)


def _p0_artifact_inputs(envelope: PayloadEnvelopeV4) -> dict[str, Any]:
    """Build P0's compact model view without changing A0 provenance."""
    evidence, allowed_fields = _p0_evidence_projection(envelope)
    return {
        "payload": {
            "schema_version": envelope.schema_version,
            "payload": evidence,
            "source_mode": envelope.source_mode,
            "source_digest": envelope.source_digest,
        },
        "allowed_source_fields": sorted(allowed_fields),
    }


def _span_ok(span: SourceSpanV4, payload: Mapping[str, str]) -> bool:
    source = payload.get(span.field)
    return (
        isinstance(source, str)
        and 0 <= span.start < span.end <= len(source)
        and span.quote == source[span.start:span.end]
    )


def _span_mismatch(span: SourceSpanV4, payload: Mapping[str, str]) -> str:
    expected = payload.get(span.field, "")[span.start:span.end]
    return (
        f"{span.field}[{span.start}:{span.end}] expected exact slice "
        f"{expected[:300]!r}; returned quote {span.quote[:300]!r}"
    )


def _validate_fact_index(
    index: FactIndexV4,
    payload: Mapping[str, str],
    *,
    allowed_source_fields: frozenset[str] | None = None,
    expected_payload_sha256: str | None = None,
) -> str | None:
    def span_error(span: SourceSpanV4, owner: str) -> str | None:
        if allowed_source_fields is not None and span.field not in allowed_source_fields:
            return f"{owner} cites source field {span.field!r} outside the supplied P0 evidence"
        if not _span_ok(span, payload):
            return f"{owner} has a non-literal source span: {_span_mismatch(span, payload)}"
        return None

    if (
        expected_payload_sha256 is not None
        and index.payload_sha256 != expected_payload_sha256
    ):
        return "fact index payload_sha256 does not match the accepted A0 digest"
    fact_ids = {f.fact_id for f in index.facts}
    for fact in index.facts:
        if not fact.source_spans:
            return f"fact {fact.fact_id} must contain at least one source span"
        for span in fact.source_spans:
            error = span_error(span, f"fact {fact.fact_id}")
            if error is not None:
                return error
    for number in index.numbers:
        if number.fact_id not in fact_ids:
            return f"number {number.number_id} does not resolve to a literal fact/span"
        error = span_error(number.source_span, f"number {number.number_id}")
        if error is not None:
            return error
    for entity in index.entities:
        if not entity.source_spans:
            return f"entity {entity.entity_id} must contain at least one source span"
        for span in entity.source_spans:
            error = span_error(span, f"entity {entity.entity_id}")
            if error is not None:
                return error
    return None


def _spoken_error(text: str, name: str = "") -> str | None:
    value = text or ""
    if not value.strip():
        return "spoken text is empty"
    if _DECORATION_RE.search(value) or _LABEL_RE.match(value) or _QUOTED_RE.match(value):
        return "spoken text contains stage direction, markup, or a role label"
    if _ALL_CAPS_RE.search(value):
        return "spoken text contains an all-caps lexical word"
    if name and re.match(r"^\s*" + re.escape(name.split()[0]) + r"\s*[,!:]", value, re.I):
        return "spoken text begins with a self-vocative"
    if any(not token.strip(".,!?;:'’-") for token in value.split()):
        return "spoken text contains a non-lexical token"
    return None


def validate_spoken_text_and_roster(
    script: ScriptArtifactV4, cast: CastPlanV4, score: RadioScoreV4
) -> None:
    locked = {row.char_id: row.name for row in cast.cast}
    if locked.get("announcer") != "ANNOUNCER":
        raise CodexSpokenTextError("announcer must be named ANNOUNCER")
    for row in cast.cast:
        if row.char_id != "announcer" and not _is_canonical_character_name(row.name):
            raise CodexSpokenTextError(f"cast name {row.name!r} is not a canonical Title-Case name")
    for line in script.lines:
        if line.char_id.startswith("music_"):
            if not line.skip or line.text or line.tts_skip_reason != "music_cue":
                raise CodexSpokenTextError(f"music line {line.line_id} has an invalid skip contract")
            continue
        if line.char_id not in locked:
            raise CodexSpokenTextError(f"line {line.line_id} uses an unlocked cast id")
        if line.speaker_role not in ("character", "announcer"):
            raise CodexSpokenTextError(f"line {line.line_id} has an illegal spoken role")
        err = _spoken_error(line.text, locked[line.char_id])
        if err:
            raise CodexSpokenTextError(f"{line.line_id}: {err}")
    voiced = {line.char_id for line in script.lines if not line.skip}
    if any(cid not in voiced for cid in locked):
        raise CodexGraphError("every locked cast row must own a voiced line")


def make_advisory_word_blueprint(requested_words: int, locked_beats: Sequence[str]) -> AdvisoryWordPlanV4:
    if not isinstance(requested_words, int) or isinstance(requested_words, bool) or not 30 <= requested_words <= 900:
        raise CodexTargetRangeError("requested_words must be 30..900")
    ids = list(locked_beats)
    if not ids:
        raise CodexGraphError("cannot allocate an empty beat plan")
    weights = [1 + (i % 3) for i in range(len(ids))]
    total = sum(weights)
    raw = [requested_words * w / total for w in weights]
    centers = [int(x) for x in raw]
    for i in sorted(range(len(ids)), key=lambda i: raw[i] - centers[i], reverse=True)[:requested_words - sum(centers)]:
        centers[i] += 1
    return AdvisoryWordPlanV4(
        advisory_total_center=requested_words,
        per_beat=[
            AdvisoryBeatV4(beat_id=beat_id, advisory_word_center=n)
            for beat_id, n in zip(ids, centers)
        ],
    )


def _radio_score_draft_output_token_budget(
    requested_words: int, beat_count: int,
) -> int:
    """Reserve P3 output from the measured finite draft surface.

    The compiler owns all canonical score mechanics. This reservation therefore
    covers only the compact draft's bounded authored decisions; each concrete
    base/restart/repair envelope still proves its own input fit before calling
    the local model.
    """
    if (
        not isinstance(requested_words, int)
        or isinstance(requested_words, bool)
        or not 30 <= requested_words <= 900
    ):
        raise CodexTargetRangeError("requested_words must be 30..900")
    if (
        not isinstance(beat_count, int)
        or isinstance(beat_count, bool)
        or not 1 <= beat_count <= _RADIO_SCORE_MAX_BEATS
    ):
        raise CodexTargetRangeError(
            f"beat_count must be an int in 1..{_RADIO_SCORE_MAX_BEATS}"
        )
    return _RADIO_SCORE_DRAFT_MAX_OUTPUT_TOKENS


def _script_output_token_budget(
    requested_words: int, accepted_line_count: int,
) -> int:
    """Reserve whole-script JSON output from BOTH drivers of its serialized size.

    A ScriptArtifactV4 is not sized by its dialogue alone: it serializes the
    strict per-line metadata graph (ids, boundary, arc phase, beat intent, flags)
    for every accepted line.  A 30-word script with 13 lines pays nearly all of
    that same metadata cost, so a budget derived from the word steer ALONE
    under-reserves exactly when the accepted graph is wide -- the P7 live cap
    (generated_tokens == max_new_tokens == 2800 -> truncated JSON -> "no
    decodable top-level JSON object").  Scale on the line count too.
    """
    if (
        not isinstance(requested_words, int)
        or isinstance(requested_words, bool)
        or not 30 <= requested_words <= 900
    ):
        raise CodexTargetRangeError("requested_words must be 30..900")
    if (
        not isinstance(accepted_line_count, int)
        or isinstance(accepted_line_count, bool)
        or accepted_line_count < 1
    ):
        raise CodexTargetRangeError("accepted_line_count must be a positive int")
    # ~4.5 tokens per requested word of dialogue, ~130 tokens of strict metadata
    # per accepted line, plus the artifact envelope (title, scenes, music cues).
    # The 2,800 floor is the observed complete-artifact floor for the canonical
    # 30-word graph; the 5,400 ceiling keeps the reservation inside the context
    # cap alongside the pass input.
    return min(
        5400,
        max(
            2800,
            int(requested_words * 4.5) + 130 * int(accepted_line_count) + 600,
        ),
    )


def _script_artifact_context(score: RadioScoreV4) -> dict[str, Any]:
    """Project the accepted score into a compact ScriptArtifactV4 context.

    Passing the full nested score next to an unconstrained ``scenes`` output
    field led the local writer to echo score scenes (including shots and
    beats) instead of emitting the root ``lines`` array. The projection
    preserves every script constraint while giving whole-script passes a flat,
    authoritative line graph.
    """
    scenes: list[dict[str, str]] = []
    line_graph: list[dict[str, Any]] = []
    for scene in score.scenes:
        scenes.append({
            "scene_id": scene.scene_id,
            "env": scene.env,
            "description": scene.description,
        })
        for beat in scene.beats:
            for line_id in beat.line_ids:
                line_graph.append({
                    "line_id": line_id,
                    "beat_id": beat.beat_id,
                    "shot_id": beat.shot_id,
                    "char_id": beat.char_id,
                    "speaker_role": beat.speaker_role,
                    "arc_phase": beat.arc_phase,
                    "beat_intent": beat.intent,
                    "fact_ids": list(beat.fact_ids),
                })
    return {
        "story_context": {
            "title": score.title,
            "premise": score.premise,
            "setting": score.setting,
            "scenes": scenes,
        },
        "accepted_line_graph": line_graph,
        "accepted_line_ids": [row["line_id"] for row in line_graph],
        "accepted_line_count": len(line_graph),
        "music_cues": [cue.model_dump(mode="json") for cue in score.music_cues],
    }


def _script_artifact_inputs(
    score: RadioScoreV4, fact_index: FactIndexV4, word_steer: WordSteerV4,
) -> dict[str, Any]:
    """Add P5-only fact and word-steer inputs to the shared script context."""
    return {
        **_script_artifact_context(score),
        "fact_index": {
            "facts": [
                {"fact_id": fact.fact_id, "claim": fact.claim}
                for fact in fact_index.facts
            ],
            "tone": fact_index.tone,
        },
        "initial_draft_word_steer": word_steer.model_dump(mode="json"),
    }


def invoke_codex_structured(
    *, pass_id: str, slot: Literal["creative", "technical"], slot_fn: GenerateFn,
    pack: Any, seam_refs: tuple[str, ...], artifact_inputs: Mapping[str, Any],
    result_type: type[BaseModel], post_validator: Callable[[BaseModel], str | None],
    base_temperature: float, structural_retry_temperature: float,
    max_new_tokens: int, call_journal: MutableMapping[str, Any],
    repair_score: RadioScoreV4 | None = None,
    prompt_must_fit: bool = False,
    clamp_overlong_strings: bool = True,
    include_result_json_schema: bool = True,
) -> BaseModel:
    if not seam_refs:
        raise CodexPackContractError(f"{pass_id} has no prompt seam")
    seams = []
    for seam in seam_refs:
        text = str((getattr(pack, "prompt_stages", {}) or {}).get(seam) or "")
        if not text:
            raise CodexPackContractError(f"missing Codex seam {seam!r}")
        seams.append(text)
    script_artifact_pass = pass_id in {"P5", "P7", "P9"}
    draft_score_pass = (
        pass_id in {"P3", "P3_rewrite"}
        and result_type is RadioScoreDraftV4
    )
    if pass_id in {"P3", "P3_rewrite"} and not draft_score_pass:
        raise CodexPackContractError(
            f"{pass_id} must use RadioScoreDraftV4 transport"
        )
    body = {"pass_id": pass_id, "artifact_inputs": artifact_inputs}
    schema_instruction = ""
    if include_result_json_schema:
        body["result_json_schema"] = result_type.model_json_schema()
        schema_instruction = _schema_instruction(result_type)
    if pass_id == "P0":
        schema_instruction += p0_contract_instruction(has_numeric_tokens=True)
    if draft_score_pass:
        schema_instruction += _RADIO_SCORE_DRAFT_SURFACE_INSTRUCTION
        if pass_id == "P3_rewrite":
            schema_instruction += (
                " Preserve the previous_draft structural decisions exactly: "
                "scene count, shots per scene, each beat's shot_index/char_id/"
                "line_count, and every cue_id/local anchor. Improve only "
                "creative prose or allowed fact placement in response to review."
            )
    if script_artifact_pass:
        schema_instruction += _SCRIPT_ARTIFACT_ROOT_INSTRUCTION
    messages = [{"role": "system", "content": "\n".join(seams) + schema_instruction}, {"role": "user", "content": json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)}]
    calls: list[dict[str, Any]] = []

    def mark_attempt_complete(
        attempt_index: int,
        _validated_raw: str,
        error: BaseException | None,
    ) -> None:
        """Store bounded status beside the original-wire response receipt."""
        if not 1 <= attempt_index <= len(calls):
            return
        receipt = calls[attempt_index - 1]
        if receipt.get("repair_kind") == _P3_TEXT_PATCH_KIND:
            # The direct P3 patch owns its real wire receipt.  The shared
            # structured ladder has no patch raw output when the factory
            # raises, so letting its generic callback classify that empty
            # string would falsely rewrite a JSON/patch-schema failure as a
            # decoded accepted draft.  On direct success this callback is
            # invoked explicitly by `_run_p3_text_patch` after the
            # merged artifact has cleared the complete validator.
            if error is None:
                receipt.update({
                    "parse_status": "decoded",
                    "schema_status": "accepted",
                    "draft_status": "accepted",
                    "compiler_status": "accepted",
                    "graph_status": "accepted",
                })
            return
        if error is None:
            receipt.update({
                "parse_status": "decoded",
                "schema_status": "accepted",
                "draft_status": "accepted" if draft_score_pass else "not_applicable",
                "compiler_status": "accepted" if draft_score_pass else "not_applicable",
                "graph_status": "accepted" if draft_score_pass else "not_applicable",
            })
            return
        if isinstance(error, json.JSONDecodeError):
            receipt.update({
                "parse_status": "not_decoded",
                "schema_status": "not_run",
                "draft_status": "not_decoded" if draft_score_pass else "not_applicable",
                "compiler_status": "not_run" if draft_score_pass else "not_applicable",
                "graph_status": "not_run" if draft_score_pass else "not_applicable",
            })
            return
        if isinstance(error, ValidationError):
            receipt.update({
                "parse_status": "decoded",
                "schema_status": "rejected",
                "draft_status": "schema_rejected" if draft_score_pass else "not_applicable",
                "compiler_status": "not_run" if draft_score_pass else "not_applicable",
                "graph_status": "not_run" if draft_score_pass else "not_applicable",
            })
            return
        if isinstance(error, PostValidationError):
            text = str(error)
            draft_code = ""
            if text.startswith("draft."):
                draft_code = text.split(" ", 1)[0].removeprefix("draft.")
            receipt.update({
                "parse_status": "decoded",
                "schema_status": "accepted",
                "draft_status": "compiler_rejected" if draft_score_pass else "not_applicable",
                "compiler_status": draft_code or "post_validation_rejected",
                "graph_status": (
                    "rejected" if draft_code == "graph"
                    else ("not_run" if draft_score_pass else "not_applicable")
                ),
            })
            return
        receipt.update({
            "parse_status": "terminal_error",
            "schema_status": "not_run",
            "draft_status": "terminal_error" if draft_score_pass else "not_applicable",
            "compiler_status": "not_run" if draft_score_pass else "not_applicable",
            "graph_status": "not_run" if draft_score_pass else "not_applicable",
        })

    def capture(messages_in, **kwargs):
        call_messages = (
            _PromptMustFitMessages(messages_in)
            if prompt_must_fit and isinstance(messages_in, list)
            else messages_in
        )
        raw = slot_fn(call_messages, **kwargs)
        original_raw = str(raw)
        resolved_artifact_unwrapped = False
        # Live P3 proof 2026-07-12: the local creative model obeyed the typed
        # repair semantically but wrapped the complete repaired object in the
        # single-key envelope {"resolved_artifact": {...}}. That envelope is
        # not RadioScoreV4 and must never reach Pydantic as if it were. Unwrap
        # only this exact, unambiguous repair transport shape; mixed roots,
        # non-object values, and every other extra key remain fail-loud.
        try:
            parsed_raw = parse_first_json_object(original_raw)
        except Exception:
            parsed_raw = None
        if (
            isinstance(parsed_raw, dict)
            and set(parsed_raw) == {"resolved_artifact"}
            and isinstance(parsed_raw["resolved_artifact"], dict)
        ):
            resolved_artifact_unwrapped = True
            raw = json.dumps(
                parsed_raw["resolved_artifact"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            log.warning(
                "[scifi_codex:%s] normalized exact resolved_artifact repair envelope",
                pass_id,
            )
        calls.append({
            "temperature": kwargs.get("temperature"),
            "max_new_tokens": kwargs.get("max_new_tokens"),
            "raw_chars": len(original_raw),
            "raw_sha256": hashlib.sha256(original_raw.encode("utf-8")).hexdigest(),
            "resolved_artifact_unwrapped": resolved_artifact_unwrapped,
            "parse_status": "decoded" if isinstance(parsed_raw, dict) else "pending",
            "schema_status": "pending",
            "draft_status": "pending" if draft_score_pass else "not_applicable",
            "compiler_status": "pending" if draft_score_pass else "not_applicable",
            "graph_status": "pending" if draft_score_pass else "not_applicable",
        })
        return raw

    # `structured_call` sees this wrapper rather than the original slot. Keep
    # OpenRouter's structured-output markers visible so base/full-repair P3
    # calls retain the same json_object transport as an unwrapped slot.
    capture._otr_openrouter = getattr(slot_fn, "_otr_openrouter", False)  # type: ignore[attr-defined]
    capture._otr_response_format = getattr(slot_fn, "_otr_response_format", None)  # type: ignore[attr-defined]

    def typed_repair_factory(*, original_prompt, failed_output, error):
        detail = " ".join(str(error).split())[:500] or "structured output rejected"
        if script_artifact_pass:
            log.info(
                "[scifi_codex:%s] attempting deterministic ScriptArtifactV4 metadata repair",
                pass_id,
            )
            deterministic = (
                repair_script_artifact_metadata(failed_output, repair_score)
                if repair_score is not None
                else None
            )
            # A deterministic metadata repair must still satisfy every content
            # validator before it can replace the model's typed-repair call.
            # Otherwise leave the original response for the creative repair.
            if deterministic is not None:
                deterministic_error = post_validator(deterministic)
                if deterministic_error is None:
                    return deterministic
                expected_graph = (
                    _accepted_script_line_metadata(repair_score)
                    if repair_score is not None
                    else None
                )
                l001 = next(
                    (line for line in deterministic.lines if line.line_id == "l001"),
                    None,
                )
                log.info(
                    "[scifi_codex:%s] deterministic ScriptArtifactV4 repair "
                    "rejected: %s; l001=%s expected=%s",
                    pass_id, deterministic_error,
                    (
                        (l001.beat_id, l001.shot_id, l001.boundary)
                        if l001 is not None else None
                    ),
                    expected_graph.get("l001") if expected_graph is not None else None,
                )
            repair_rules = (
                "This is a typed repair of the same ScriptArtifactV4, not a new "
                "creative response. Return one JSON object only. Its schema_version "
                "MUST be the exact literal scifi_codex.script_artifact.v4. Preserve "
                "every existing title, scene, beat, character intent, and line text. "
                "Remove every forbidden extra key. For each script line, use the "
                "accepted score graph's owning shot_id and set boundary to shot_start "
                "when the accepted previous line is in another shot, beat_start when "
                "it is in the same shot but another beat, or continue otherwise. "
                "The accepted_line_graph is closed: return every and only its line IDs; "
                "music_cues never create a line row."
            )
        elif draft_score_pass:
            # A live P3 compact draft can be structurally complete with only a
            # few authored strings over their strict caps. Let the author make
            # one bounded shortening decision over an explicitly proven
            # transport, but only after a probe proves that no hidden
            # compiler/signature/graph defect exists.
            patch_transport = _p3_text_patch_transport(slot_fn)
            if patch_transport is not None and isinstance(error, ValidationError):
                try:
                    failed_draft_for_patch = parse_first_json_object(failed_output)
                except Exception:
                    failed_draft_for_patch = None
                if isinstance(failed_draft_for_patch, dict):
                    patch_targets = _derive_p3_text_patch_targets(
                        failed_draft_for_patch, error,
                    )
                    if (
                        patch_targets is not None
                        and _p3_text_patch_preflight(
                            failed_draft_for_patch,
                            patch_targets,
                            post_validator,
                        )
                    ):
                        return _run_p3_text_patch(
                            slot_fn=slot_fn,
                            pack=pack,
                            raw_draft=failed_draft_for_patch,
                            targets=patch_targets,
                            post_validator=post_validator,
                            calls=calls,
                            mark_attempt_complete=mark_attempt_complete,
                            patch_transport=patch_transport,
                        )
            trusted_context = json.dumps(
                body["artifact_inputs"],
                sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            )
            if isinstance(error, json.JSONDecodeError):
                # A base syntax failure already consumed attempt one and the
                # shared ladder's lower-temperature structural retry consumed
                # attempt two. Its raw response is incomplete by definition;
                # never carry that partial story surface into call three.
                if len(calls) < 2:
                    raise CodexPackContractError(
                        "draft clean restart requires two completed decode attempts"
                    )
                restart_rules = (
                    "Start a fresh RadioScoreDraftV4 from the trusted references. "
                    "The prior responses were incomplete JSON and are intentionally "
                    "unavailable. Return one complete draft root only; do not return "
                    "a wrapper, a request field, or an explanation."
                )
                return [
                    {
                        "role": "system",
                        "content": "\n".join(seams) + schema_instruction
                        + "\n" + restart_rules,
                    },
                    {
                        "role": "user",
                        "content": "\n".join((
                            "TRUSTED REFERENCES ONLY -- not an output shape.",
                            "<draft_context>",
                            trusted_context,
                            "</draft_context>",
                        )),
                    },
                ]
            try:
                parsed_failed_draft = parse_first_json_object(failed_output)
            except Exception as exc:
                raise CodexPackContractError(
                    "draft semantic repair requires one complete parsed object"
                ) from exc
            if not isinstance(parsed_failed_draft, dict):
                raise CodexPackContractError(
                    "draft semantic repair requires an object root"
                )
            minified_failed_draft = json.dumps(
                parsed_failed_draft,
                sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            )
            repair_rules = (
                "This is a typed repair of the same RadioScoreDraftV4, not a new "
                "unconstrained outline. Preserve valid creative decisions wherever "
                "possible; repair only the bounded rejection. Return one complete draft "
                "root only. Do not return a wrapper, request field, canonical IDs, "
                "advisory metadata, speaker metadata, or spoken text."
            )
            return [
                {
                    "role": "system",
                    "content": "\n".join(seams) + schema_instruction
                    + "\n" + repair_rules,
                },
                {
                    "role": "user",
                    "content": "\n".join((
                        "INPUT REFERENCES ONLY -- they are not an output shape.",
                        "<failed_radio_score_draft>",
                        minified_failed_draft,
                        "</failed_radio_score_draft>",
                        "<rejection>",
                        detail,
                        "</rejection>",
                        "<trusted_draft_context>",
                        trusted_context,
                        "</trusted_draft_context>",
                    )),
                },
            ]
        elif pass_id == "P2":
            deterministic = repair_cast_plan_metadata(failed_output)
            if deterministic is not None and post_validator(deterministic) is None:
                return deterministic
            repair_rules = (
                "This is a typed repair of the same CastPlanV4. Return one JSON "
                "object only. Preserve every character description, gender, role, "
                "and voice slot. The row whose char_id is announcer MUST have the "
                "exact fixed name ANNOUNCER. Every non-announcer name must be one "
                "canonical Title-Case name, and every char_id must occur exactly once."
            )
        elif pass_id == "P0":
            repair_rules = (
                "This is a typed repair of the same FactIndexV4, not a new creative "
                "response. Return one complete JSON object only, rooted exactly at facts, "
                "entities, numbers, tone, and payload_sha256. IDs are fixed lexical tokens: "
                "facts use F01 through F06, entities E01 through E04, and numbers N01 "
                "through N04. Never emit bare F0, F1, E0, or N0. Every fact and entity "
                "has exactly one literal source span; calculate quote exactly as "
                "payload[field][start:end] from the supplied source evidence. Do not "
                "paraphrase, infer, or retain a mismatched span. tone is one nonempty scalar "
                "source-derived string, never an array or object. Preserve valid claims and "
                "remove only unsupported facts. The tagged input references are not an output "
                "template: never return a wrapper, request field, or tag name."
            )
            deterministic = repair_literal_source_metadata(
                failed_output,
                FactIndexV4,
                body["artifact_inputs"]["payload"]["payload"],
                zero_padded_ids=True,
                max_quote_chars=MAX_QUOTE_CHARS,
            )
            if deterministic is not None:
                # A0's digest is deterministic request metadata.  It is safe
                # to restore it only after the literal-span repair has kept a
                # concrete artifact; claims and source text remain untouched.
                deterministic = deterministic.model_copy(update={
                    "payload_sha256": body["artifact_inputs"]["payload"]["source_digest"],
                })
                deterministic_error = post_validator(deterministic)
                if deterministic_error is None:
                    return deterministic
                log.info(
                    "[scifi_codex:P0] deterministic literal-span repair "
                    "declined: %s", deterministic_error,
                )
            try:
                parsed = FactIndexV4.model_validate(parse_first_json_object(failed_output))
            except Exception:
                parsed = None
            if parsed is not None:
                expected_digest = body["artifact_inputs"]["payload"]["source_digest"]
                if parsed.payload_sha256 != expected_digest:
                    deterministic = parsed.model_copy(update={"payload_sha256": expected_digest})
                    deterministic_error = post_validator(deterministic)
                    if deterministic_error is None:
                        return deterministic
                    log.info(
                        "[scifi_codex:P0] deterministic digest repair "
                        "declined: %s", deterministic_error,
                    )
        elif pass_id == "P6":
            deterministic = repair_listener_review_shape(failed_output)
            if deterministic is not None and post_validator(deterministic) is None:
                log.info(
                    "[scifi_codex:P6] deterministic listener-issue shape repair accepted",
                )
                return deterministic
            repair_rules = (
                "This is a typed repair of the same ListenerReviewV4, not a new review. "
                "Preserve every strength and every diagnosis you already wrote, word for "
                "word. issues MUST be a flat JSON array, never an object grouping issues "
                "under category names. Each element is one object with category (your own "
                "short label for the flaw), line_id (the exact line ID, or null when the "
                "issue is about the whole play), and direction (what a writer should do). "
                "Return one JSON object only."
            )
        else:
            repair_rules = (
                "This is a typed repair of the same artifact. Preserve the existing premise, "
                "scene descriptions, beats, and content; repair only the fields named by the "
                "validation error. Every required nested graph field must be present. Copy "
                "parent scene_id into each shot and beat, copy a valid shot_id into each beat, "
                "copy each beat's speaker from the cast row matching its char_id, and provide "
                "every required visual_prompt without dropping existing content. For every "
                "script line, set boundary to shot_start for the first line in a shot, "
                "beat_start for the first line in a beat, or continue otherwise."
            )
        if pass_id == "P0":
            p0_payload = body["artifact_inputs"]["payload"]
            return [
                {
                    "role": "system",
                    "content": "\n".join(seams) + schema_instruction + "\n" + repair_rules,
                },
                {
                    "role": "user",
                    "content": compact_p0_repair_context(
                        failed_artifact=failed_output,
                        rejection=detail,
                        source_evidence=p0_payload["payload"],
                        source_digest=p0_payload["source_digest"],
                        allowed_source_fields=body["artifact_inputs"]["allowed_source_fields"],
                    ),
                },
            ]
        compact_request = {
            key: value for key, value in body.items()
            if key != "result_json_schema"
        }
        return [
            {"role": "system", "content": "\n".join(seams) + schema_instruction + "\n" + repair_rules},
            {"role": "user", "content": json.dumps({"failed_artifact": failed_output, "validation_error": detail, "original_request": compact_request}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)},
        ]
    journal_entry: dict[str, Any] = {
        "pass_id": pass_id,
        "slot": slot,
        "attempts": calls,
    }
    call_journal.setdefault("calls", []).append(journal_entry)
    try:
        # LLM slot: per-sub-pass injected creative/technical closure.
        result = structured_call(
            prompt=messages, schema=result_type, slot_fn=capture,
            base_temperature=base_temperature,
            structural_retry_temperature=structural_retry_temperature,
            max_new_tokens=max_new_tokens, max_attempts=3,
            repair_prompt_factory=typed_repair_factory,
            post_validator=post_validator,
            clamp_overlong_strings=clamp_overlong_strings,
            helper_name=f"scifi_codex:{pass_id}",
            on_attempt_complete=mark_attempt_complete,
        )
    except Exception as exc:
        journal_entry["terminal_error"] = (
            f"{type(exc).__name__}: {' '.join(str(exc).split())[:500]}"
        )
        raise CodexPassError(f"{pass_id} failed: {exc}") from exc
    journal_entry["accepted"] = result.model_dump(mode="json")
    return result


def _call_radio_score_draft(
    *,
    pass_id: Literal["P3", "P3_rewrite"],
    slot_fn: GenerateFn,
    pack: Any,
    seam_refs: tuple[str, ...],
    artifact_inputs: Mapping[str, Any],
    advisory: AdvisoryWordPlanV4,
    cast: CastPlanV4,
    fact_index: FactIndexV4,
    base_temperature: float,
    structural_retry_temperature: float,
    max_new_tokens: int,
    call_journal: MutableMapping[str, Any],
    expected_signature: tuple[Any, ...] | None = None,
) -> RadioScoreV4:
    """Accept only a compiled final score from the compact P3 transport."""
    compiled_by_draft_identity: dict[int, RadioScoreV4] = {}

    def validate_draft(candidate: BaseModel) -> str | None:
        if not isinstance(candidate, RadioScoreDraftV4):
            return "draft.score_schema at root: selected result is not a draft"
        try:
            compiled = compile_radio_score_draft(
                candidate, advisory, cast, fact_index,
            )
            if (
                expected_signature is not None
                and _radio_score_draft_structure_signature(candidate)
                != expected_signature
            ):
                raise RadioScoreDraftCompileError(
                    code="graph", path="rewrite.structure",
                    detail="rewrite changed a locked draft structural decision",
                )
        except RadioScoreDraftCompileError as exc:
            return str(exc)
        compiled_by_draft_identity[id(candidate)] = compiled
        return None

    result = invoke_codex_structured(
        pass_id=pass_id,
        slot="creative",
        slot_fn=slot_fn,
        pack=pack,
        seam_refs=seam_refs,
        artifact_inputs=artifact_inputs,
        result_type=RadioScoreDraftV4,
        post_validator=validate_draft,
        base_temperature=base_temperature,
        structural_retry_temperature=structural_retry_temperature,
        max_new_tokens=max_new_tokens,
        call_journal=call_journal,
        prompt_must_fit=True,
        clamp_overlong_strings=False,
        include_result_json_schema=False,
    )
    if not isinstance(result, RadioScoreDraftV4):
        raise CodexPassError(f"{pass_id} returned a non-draft structured result")
    compiled = compiled_by_draft_identity.get(id(result))
    if compiled is None:
        # The post-validator normally compiles this exact immutable instance.
        # Recompile defensively if a future structured-call implementation
        # materializes a value-equal replacement before returning it.
        compiled = compile_radio_score_draft(result, advisory, cast, fact_index)
        if (
            expected_signature is not None
            and _radio_score_draft_structure_signature(result) != expected_signature
        ):
            raise CodexPassError(
                f"{pass_id} accepted a draft that changed locked rewrite structure"
            )

    calls = call_journal.get("calls")
    if isinstance(calls, list) and calls and isinstance(calls[-1], dict):
        journal_entry = calls[-1]
        if journal_entry.get("pass_id") == pass_id:
            draft_wire = journal_entry.get("accepted")
            draft_serialized = json.dumps(
                draft_wire, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            )
            journal_entry["accepted_transport"] = {
                "schema": "RadioScoreDraftV4",
                "chars": len(draft_serialized),
                "sha256": hashlib.sha256(
                    draft_serialized.encode("utf-8")
                ).hexdigest(),
            }
            journal_entry["accepted"] = compiled.model_dump(mode="json")
    return compiled


def _script_digest(script: ScriptArtifactV4) -> str:
    return hashlib.sha256(json.dumps(script.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


class _CodexTailFinalizer:
    def __init__(self, expected: Mapping[str, str]):
        self.expected = dict(expected)

    def _proof(self, data: Mapping[str, Any]) -> None:
        lane = data.get("meta", {}).get("scifi_codex", {})
        wanted = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in self.expected.items()}
        if lane.get("line_text_sha256") != wanted:
            raise CodexPreTailAuditError("accepted text receipt does not match the in-memory ledger")
        for row in data.get("lines", []):
            if row.get("line_id") in self.expected and row.get("text") != self.expected[row["line_id"]]:
                raise CodexPreTailAuditError(f"line receipt mismatch for {row.get('line_id')}")

    def before_save(self, *, ctx: Any) -> None:
        self._proof(ctx.led.data)
        pre = _otr_ledger_freeze.phase_0_gap_audit_pre(ctx.led)
        post = _otr_ledger_freeze.phase_10_gap_audit_post_and_freeze(ctx.led)
        # A WARNING IS NOT AN ERROR. Errors block; warnings are recorded and the record
        # ships. This gate only ever passed because no warning happened to fire.
        notes = list(pre.warnings) + list(post.warnings)
        if notes:
            log.warning(
                "[scifi_codex] the freeze cascade raised %d warning(s); none is an "
                "error, so the record stands:\n  %s",
                len(notes), "\n  ".join(str(n) for n in notes),
            )
            ctx.led.data.setdefault("meta", {}).setdefault("scifi_codex", {})[
                "freeze_notes"
            ] = [str(n) for n in notes]
        if pre.errors or post.errors:
            raise CodexPreTailAuditError(
                "Codex ledger freeze has hard errors: "
                + "; ".join(str(e) for e in list(pre.errors) + list(post.errors))
            )
        # Same law as after_save, which this gate contradicted: a warning is not an
        # error. `frozen_with_warns` IS a clean freeze -- the reviewer read the ledger
        # and made no edits; the warns are soft gaps, i.e. notes. This line used to
        # demand `frozen_clean` outright, so the identical ledger was illegal here and
        # legal ten lines below, and a note could kill a finished episode (2026-07-11:
        # "frozen_with_warns -- 2 soft gap(s)" killed a codex roll after P0-P8 passed).
        # The structural verdicts -- frozen_with_doctor_edits, too_many_edits,
        # needs_full_rerun -- still block, because those are defects, not notes.
        verdict = ctx.led.data.get("meta", {}).get("freeze_verdict")
        if verdict not in ("frozen_clean", "frozen_with_warns"):
            raise CodexPreTailAuditError(
                f"Codex ledger freeze verdict is {verdict!r} -- not a clean freeze"
            )

    def after_save(self, *, saved_path: str, ledger_data: Mapping[str, Any]) -> None:
        try:
            with open(saved_path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
        except Exception as exc:
            raise CodexSavedLedgerAuditError(f"cannot reopen saved ledger: {exc}") from exc
        report = _otr_ledger_freeze.run_gap_audit(saved, label="saved")
        verdict = saved.get("meta", {}).get("freeze_verdict")
        # Same law on the saved ledger: errors and structural verdicts block; warnings do
        # not. frozen_with_warns is a CLEAN freeze -- the reviewer made no edits.
        if report.errors:
            raise CodexSavedLedgerAuditError(
                "saved ledger has hard errors: "
                + "; ".join(str(e) for e in report.errors)
            )
        if verdict not in ("frozen_clean", "frozen_with_warns"):
            raise CodexSavedLedgerAuditError(
                f"saved ledger freeze verdict is {verdict!r} -- not a clean freeze"
            )
        self._proof(saved)


@dataclass
class CodexTailParts:
    outline_view: Any
    canon: Any
    final_title_override: str
    run_story_spine: bool
    tail_finalizer: Any


def _build_codex_episode_canon(
    score: RadioScoreV4, script: ScriptArtifactV4, *, premise: str,
) -> EpisodeCanon:
    """Return the complete episode-canon protocol the shared tail writes."""
    return EpisodeCanon(
        title=script.title,
        premise=premise,
        setting=score.setting,
        # RadioScoreV4 deliberately does not author a time-of-day field. An
        # empty value records that absence honestly while still satisfying the
        # shared EpisodeCanon protocol; do not invent a setting detail here.
        time_of_day="",
        sound_palette=[],
    )


def _validate_script_post(script: ScriptArtifactV4, cast: CastPlanV4, score: RadioScoreV4) -> str | None:
    try:
        _validate_script_graph(script, score)
        validate_spoken_text_and_roster(script, cast, score)
    except ScifiCodexError as exc:
        return str(exc)
    return None


def _assemble_ledger(led: Any, score: RadioScoreV4, cast: CastPlanV4, script: ScriptArtifactV4, meta: MutableMapping[str, Any]) -> dict[str, str]:
    cast_rows = []
    for row in cast.cast:
        if row.char_id == "announcer":
            tts_model, preset = "kokoro", "bm_george"
        else:
            tts_model, preset = "bark", {"c01": "v2/en_speaker_6", "c02": "v2/en_speaker_3", "c03": "v2/en_speaker_0"}[row.char_id]
        cast_rows.append({"char_id": row.char_id, "name": row.name, "character_description": row.character_description, "gender": row.gender, "tts_model": tts_model, "voice_preset": preset})
    scenes = [{"scene_id": s.scene_id, "description": s.description, "env": s.env} for s in score.scenes]
    shots = [{"shot_id": sh.shot_id, "scene_id": sh.scene_id, "description": sh.description, "visual_prompt": sh.visual_prompt} for s in score.scenes for sh in s.shots]
    beats = [{"beat_id": b.beat_id, "shot_id": b.shot_id, "scene_id": b.scene_id, "speaker": b.speaker, "char_id": b.char_id, "line_ids": list(b.line_ids)} for s in score.scenes for b in s.beats]
    script_by_line = {x.line_id: x for x in script.lines}
    lines = []
    music_ids = []
    expected: dict[str, str] = {}
    for s in score.scenes:
        for b in s.beats:
            for lid in b.line_ids:
                src = script_by_line.get(lid)
                if src is None:
                    raise CodexGraphError(f"missing script line {lid}")
                row = {"line_id": lid, "beat_id": b.beat_id, "shot_id": b.shot_id, "char_id": src.char_id, "speaker_role": src.speaker_role, "text": src.text, "traits": src.traits, "boundary": src.boundary, "arc_phase": src.arc_phase, "compose_flags": list(src.compose_flags), "beat_intent": src.beat_intent, "dialogue_slot_id": src.dialogue_slot_id}
                lines.append(row)
                if src.char_id.startswith("music_"):
                    music_ids.append(lid)
                else:
                    expected[lid] = src.text
    led.set_cast(cast_rows)
    led.set_scenes(scenes)
    led.set_shots(shots)
    led.set_beats(beats)
    led.set_lines(lines)
    for row in led.data.get("lines", []):
        if row.get("line_id") in music_ids:
            row["skip"] = True
            row["text"] = ""
            row["tts_skip_reason"] = "music_cue"
    music = [{"cue_id": cue.cue_id, "description": cue.description, "generation_prompt": cue.generation_prompt, "placement": cue.placement, "anchor_line_id": cue.anchor_line_id} for cue in script.music_cues]
    led.set_music(music)
    led.data["clips"] = []
    stamp_word_counts(led)
    meta["scifi_codex"]["line_text_sha256"] = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in expected.items()}
    meta["scifi_codex"]["accepted_lines"] = dict(expected)
    return expected


def run_scifi_codex_episode(
    *, payload: dict[str, str], pack: Any, resolved: Mapping[str, Any], led: Any,
    meta: dict[str, Any], creative_fn: GenerateFn, technical_fn: GenerateFn,
    slot_scheduler: Any, source_bank_row: Any, story_rules: Mapping[str, Any],
    episode_root: Path, episode_id: str,
) -> CodexTailParts:
    del slot_scheduler, source_bank_row, story_rules, episode_root, episode_id
    env, steer = validate_payload_envelope(payload, resolved)
    p0_inputs = _p0_artifact_inputs(env)
    p0_allowed_fields = frozenset(p0_inputs["allowed_source_fields"])
    lane_meta: dict[str, Any] = {"source_digest": env.source_digest, "source_mode": env.source_mode, "call_journal": {}}
    meta["scifi_codex"] = lane_meta
    journal = lane_meta["call_journal"]
    p0_token_budget = p0_output_token_budget()
    journal["fact_index_token_budget"] = {
        **p0_contract_receipt(),
        "source_evidence_field_count": len(p0_inputs["allowed_source_fields"]),
        "source_evidence_characters": sum(
            len(value) for value in p0_inputs["payload"]["payload"].values()
        ),
    }
    p0 = invoke_codex_structured(pass_id="P0", slot="technical", slot_fn=technical_fn, pack=pack, seam_refs=("codex_fact_index_system",), artifact_inputs=p0_inputs, result_type=FactIndexV4, post_validator=lambda x: _validate_fact_index(x, payload, allowed_source_fields=p0_allowed_fields, expected_payload_sha256=env.source_digest), base_temperature=.20, structural_retry_temperature=.10, max_new_tokens=p0_token_budget, call_journal=journal, prompt_must_fit=True, clamp_overlong_strings=False)
    p1 = invoke_codex_structured(
        pass_id="P1", slot="creative", slot_fn=creative_fn, pack=pack,
        seam_refs=("codex_question_system",),
        artifact_inputs={"fact_index": p0.model_dump(mode="json")},
        result_type=DramaticQuestionV4, post_validator=lambda x: None,
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=1800, call_journal=journal,
        clamp_overlong_strings=False,
    )
    p2 = invoke_codex_structured(
        pass_id="P2", slot="creative", slot_fn=creative_fn, pack=pack,
        seam_refs=("codex_pressure_cast_system",),
        artifact_inputs={"question": p1.model_dump(mode="json")},
        result_type=CastPlanV4, post_validator=_validate_cast_plan,
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=1600, call_journal=journal,
        clamp_overlong_strings=False,
    )
    beat_ids = [f"b{i:03d}" for i in range(max(3, min(12, len(p2.cast) * 3)))]
    advisory = make_advisory_word_blueprint(steer.requested_words, beat_ids)
    score_token_budget = _radio_score_draft_output_token_budget(
        steer.requested_words, len(beat_ids),
    )
    journal["radio_score_draft_token_budget"] = {
        **_radio_score_draft_surface_receipt(),
        "requested_words": steer.requested_words,
        "beat_count": len(beat_ids),
    }
    p3_draft_inputs = {
        "question": p1.model_dump(mode="json"),
        "cast": p2.model_dump(mode="json"),
        "fact_index": _compact_p0_fact_context(p0),
        "advisory_word_plan": advisory.model_dump(mode="json"),
    }
    p3 = _call_radio_score_draft(
        pass_id="P3", slot_fn=creative_fn, pack=pack,
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs=p3_draft_inputs,
        advisory=advisory, cast=p2, fact_index=p0,
        base_temperature=.72, structural_retry_temperature=.32,
        max_new_tokens=score_token_budget, call_journal=journal,
    )
    review = invoke_codex_structured(
        pass_id="P4", slot="technical", slot_fn=technical_fn, pack=pack,
        seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
        artifact_inputs={"score": p3.model_dump(mode="json")},
        result_type=StructureReviewV4, post_validator=lambda x: None,
        base_temperature=.20, structural_retry_temperature=.10,
        max_new_tokens=1800, call_journal=journal,
        clamp_overlong_strings=False,
    )
    score = p3
    if review.verdict == "rewrite":
        projected_p3 = project_radio_score_to_draft(p3)
        rewrite_signature = _radio_score_draft_structure_signature(projected_p3)
        roundtrip_score = compile_radio_score_draft(projected_p3, advisory, p2, p0)
        if (
            _radio_score_draft_structure_signature(
                project_radio_score_to_draft(roundtrip_score)
            )
            != rewrite_signature
        ):
            raise CodexPassError(
                "P3_rewrite draft projection failed its structural round-trip"
            )
        score = _call_radio_score_draft(
            pass_id="P3_rewrite", slot_fn=creative_fn, pack=pack,
            seam_refs=("codex_radio_score_system", "codex_coda_contract_system"),
            artifact_inputs={
                **p3_draft_inputs,
                "previous_draft": projected_p3.model_dump(mode="json"),
                "review": review.model_dump(mode="json"),
            },
            advisory=advisory, cast=p2, fact_index=p0,
            base_temperature=.55, structural_retry_temperature=.20,
            max_new_tokens=score_token_budget, call_journal=journal,
            expected_signature=rewrite_signature,
        )
    # The whole-script reservation is only knowable once the score's accepted
    # line graph is final (P3, or P3_rewrite): the artifact serializes strict
    # metadata for every accepted line, so the line count -- not the word steer
    # alone -- drives its size.
    accepted_line_count = len(_accepted_script_line_metadata(score) or ())
    if not accepted_line_count:
        raise CodexGraphError("accepted score has no line graph to budget for")
    script_token_budget = _script_output_token_budget(steer.requested_words, accepted_line_count)
    journal["script_token_budget"] = {"requested_words": steer.requested_words, "accepted_line_count": accepted_line_count, "max_new_tokens": script_token_budget}
    script = invoke_codex_structured(pass_id="P5", slot="creative", slot_fn=creative_fn, pack=pack, seam_refs=("codex_play_system", "codex_coda_contract_system"), artifact_inputs=_script_artifact_inputs(score, p0, steer), result_type=ScriptArtifactV4, post_validator=lambda x: _validate_script_post(x, p2, score), base_temperature=.78, structural_retry_temperature=.35, max_new_tokens=script_token_budget, call_journal=journal, repair_score=score)
    listener = invoke_codex_structured(pass_id="P6", slot="technical", slot_fn=technical_fn, pack=pack, seam_refs=("codex_listening_room_system",), artifact_inputs={"script": script.model_dump(mode="json"), "score": score.model_dump(mode="json")}, result_type=ListenerReviewV4, post_validator=lambda x: None, base_temperature=.20, structural_retry_temperature=.10, max_new_tokens=2200, call_journal=journal)
    script = invoke_codex_structured(pass_id="P7", slot="creative", slot_fn=creative_fn, pack=pack, seam_refs=("codex_retake_system", "codex_coda_contract_system"), artifact_inputs={**_script_artifact_context(score), "previous": script.model_dump(mode="json"), "review": listener.model_dump(mode="json")}, result_type=ScriptArtifactV4, post_validator=lambda x: _validate_script_post(x, p2, score), base_temperature=.68, structural_retry_temperature=.30, max_new_tokens=script_token_budget, call_journal=journal, repair_score=score)
    audit = invoke_codex_structured(pass_id="P8", slot="technical", slot_fn=technical_fn, pack=pack, seam_refs=("codex_final_audit_system", "codex_coda_contract_system"), artifact_inputs={"script": script.model_dump(mode="json"), "fact_index": p0.model_dump(mode="json")}, result_type=FinalAuditV4, post_validator=lambda x: None, base_temperature=.20, structural_retry_temperature=.10, max_new_tokens=2400, call_journal=journal)
    if audit.verdict == "rewrite":
        script = invoke_codex_structured(pass_id="P9", slot="creative", slot_fn=creative_fn, pack=pack, seam_refs=("codex_retake_system", "codex_play_system", "codex_coda_contract_system"), artifact_inputs={**_script_artifact_context(score), "previous": script.model_dump(mode="json"), "audit": audit.model_dump(mode="json")}, result_type=ScriptArtifactV4, post_validator=lambda x: _validate_script_post(x, p2, score), base_temperature=.68, structural_retry_temperature=.30, max_new_tokens=script_token_budget, call_journal=journal, repair_score=score)
    validate_spoken_text_and_roster(script, p2, score)
    expected = _assemble_ledger(led, score, p2, script, meta)
    from ._otr_content_authorship import stamp_receipt
    stamp_receipt(
        led.data, owner_bank="scifi_codex",
        accepted_artifacts={"final_script": script},
    )
    actual = sum(_words(v) for v in expected.values())
    meta["scifi_codex"]["word_receipt"] = {"requested_words": steer.requested_words, "actual_split_words": actual, "actual_ledger_word_count": int(led.data.get("total_word_count") or 0)}
    meta["scifi_codex"]["fact_index"] = p0.model_dump(mode="json")
    meta["scifi_codex"]["script_digest"] = _script_digest(script)
    canon = _build_codex_episode_canon(score, script, premise=p1.question)
    return CodexTailParts(
        outline_view=SimpleNamespace(
            title=script.title,
            premise=p1.question,
            setting=score.setting,
            time_of_day=canon.time_of_day,
        ),
        canon=canon,
        final_title_override=script.title,
        run_story_spine=False,
        tail_finalizer=_CodexTailFinalizer(expected),
    )
