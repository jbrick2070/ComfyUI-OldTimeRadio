"""Sci-Fi Codex v4 additive source-bank runner.

The lane owns its schemas, prompt seams, provenance graph, and ledger assembly.
It never fetches a source, loads a model, or edits LLM-authored dialogue.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal, Mapping, MutableMapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


log = logging.getLogger("OTR")

try:
    from ._otr_canon import EpisodeCanon
    from ._otr_json import parse_first_json_object
    from ._otr_source_payload import validate_source_payload
    from ._otr_scifi_source_repair import repair_literal_source_metadata
    from ._otr_structured_call import schema_shape_instruction, structured_call
    from . import _otr_ledger_freeze
    from .production_ledger import stamp_word_counts
except ImportError:  # pragma: no cover
    from _otr_canon import EpisodeCanon  # type: ignore
    from _otr_json import parse_first_json_object  # type: ignore
    from _otr_source_payload import validate_source_payload  # type: ignore
    from _otr_scifi_source_repair import repair_literal_source_metadata  # type: ignore
    from _otr_structured_call import schema_shape_instruction, structured_call  # type: ignore
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
    quote: str

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
    fact_id: str = Field(pattern=r"F(?:0[1-9]|1[0-2])")
    claim: str
    source_spans: list[SourceSpanV4]
    numeric_tokens: list[str] = []


class EntityV4(_Strict):
    entity_id: str = Field(pattern=r"E(?:0[1-9]|1[0-2])")
    name: str
    source_spans: list[SourceSpanV4]


class NumberV4(_Strict):
    number_id: str = Field(pattern=r"N(?:0[1-9]|1[0-2])")
    verbatim: str
    fact_id: str
    source_span: SourceSpanV4


class FactIndexV4(_Strict):
    facts: list[FactV4] = Field(min_length=1, max_length=12)
    entities: list[EntityV4] = Field(max_length=12)
    numbers: list[NumberV4] = Field(max_length=12)
    tone: str
    payload_sha256: str


class DramaticQuestionV4(_Strict):
    question: str
    consequence: str
    ending_direction: str


class CastPlanRowV4(_Strict):
    char_id: Literal["announcer", "c01", "c02", "c03"]
    name: str
    character_description: str
    gender: str
    role_in_conflict: str
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


class AdvisoryWordPlanV4(_Strict):
    advisory_total_center: int = Field(ge=1)
    per_beat: list[dict[str, Any]]


class BeatPlanV4(_Strict):
    beat_id: str
    scene_id: str
    shot_id: str
    speaker: str
    char_id: Literal[
        "announcer", "c01", "c02", "c03",
        "music_open", "music_inter", "music_close",
    ]
    speaker_role: Literal[
        "character", "announcer", "music_open", "music_close", "music_inter",
    ]
    line_ids: list[str] = Field(min_length=1)
    order: int
    intent: str
    arc_phase: str
    fact_ids: list[str] = []
    advisory_voiced_word_center: int = Field(ge=0)


class ShotPlanV4(_Strict):
    shot_id: str
    scene_id: str
    description: str
    visual_prompt: str


class ScenePlanV4(_Strict):
    scene_id: str
    env: str
    description: str
    shots: list[ShotPlanV4] = Field(min_length=1)
    beats: list[BeatPlanV4] = Field(min_length=1)


class MusicCueV4(_Strict):
    cue_id: Literal["music_open", "music_inter", "music_close"]
    placement: Literal["open", "inter", "close"]
    description: str
    generation_prompt: str
    anchor_line_id: str
    anchor_beat_id: str


class RadioScoreV4(_Strict):
    title: str
    premise: str
    setting: str
    advisory_word_plan: AdvisoryWordPlanV4
    scenes: list[ScenePlanV4] = Field(min_length=1)
    music_cues: list[MusicCueV4] = Field(min_length=1, max_length=3)


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
        beat_id = row.get("beat_id") if isinstance(row, dict) else None
        if not isinstance(beat_id, str) or not beat_id:
            return "locked advisory plan has an invalid beat ID"
        expected_beat_ids.append(beat_id)
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


def repair_radio_score_metadata(
    failed_output: str, advisory: AdvisoryWordPlanV4,
) -> RadioScoreV4 | None:
    """Derive cue anchor beats from an otherwise closed accepted score graph.

    A cue's ``anchor_beat_id`` is executable metadata: once its anchor line is
    known, the owning beat is not a creative choice.  Do not repair an unknown
    anchor, missing beat, altered advisory plan, or noncanonical line manifest;
    those require a complete score repair and must fail closed here.
    """
    try:
        score = RadioScoreV4.model_validate(parse_first_json_object(failed_output))
    except Exception:
        return None
    if score.advisory_word_plan.model_dump(mode="json") != advisory.model_dump(mode="json"):
        return None
    expected_beats = [str(row["beat_id"]) for row in advisory.per_beat]
    observed_beats = [beat for scene in score.scenes for beat in scene.beats]
    if [beat.beat_id for beat in observed_beats] != expected_beats:
        return None
    if [beat.order for beat in observed_beats] != list(range(1, len(observed_beats) + 1)):
        return None
    line_metadata = _accepted_script_line_metadata(score)
    if line_metadata is None:
        return None
    line_ids = list(line_metadata)
    if line_ids != [f"l{i:03d}" for i in range(1, len(line_ids) + 1)]:
        return None

    seen_cue_ids: set[str] = set()
    cues: list[MusicCueV4] = []
    changed = False
    for cue in score.music_cues:
        if cue.cue_id in seen_cue_ids:
            return None
        seen_cue_ids.add(cue.cue_id)
        anchor = line_metadata.get(cue.anchor_line_id)
        if anchor is None:
            return None
        if cue.anchor_beat_id != anchor[0]:
            cue = cue.model_copy(update={"anchor_beat_id": anchor[0]})
            changed = True
        cues.append(cue)
    if not changed:
        return None
    repaired = score.model_copy(update={"music_cues": cues})
    return repaired if _validate_radio_score_graph(repaired, advisory) is None else None


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
    issues: list[str] = []
    rationale: str = ""


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
        if len(clean["full_text"].split()) < 80 or len(set(re.findall(r"[A-Za-z0-9]+", clean["full_text"].lower()))) < 12:
            raise CodexPayloadThinError("RSS payload is below the 80/12 thinness floor")
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
        per_beat=[{"beat_id": beat_id, "advisory_word_center": n} for beat_id, n in zip(ids, centers)],
    )


def _score_graph_contract(advisory: AdvisoryWordPlanV4) -> dict[str, Any]:
    """Render the non-creative P3 graph constraints as model-visible data."""
    beat_ids = [str(row["beat_id"]) for row in advisory.per_beat]
    return {
        "required_beat_ids": beat_ids,
        "required_beat_orders": [
            {"beat_id": beat_id, "order": index}
            for index, beat_id in enumerate(beat_ids, start=1)
        ],
        "line_id_policy": (
            "Flattened scene/beat line_ids must be unique contiguous canonical "
            "IDs l001, l002, and so on, with no gaps."
        ),
        "music_anchor_policy": (
            "Every music cue anchor_line_id must be one of those line_ids and "
            "anchor_beat_id must be that line's owning beat_id."
        ),
    }


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
    repair_advisory: AdvisoryWordPlanV4 | None = None,
    prompt_must_fit: bool = False,
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
    body = {"pass_id": pass_id, "artifact_inputs": artifact_inputs, "result_json_schema": result_type.model_json_schema()}
    schema_instruction = _schema_instruction(result_type)
    if script_artifact_pass:
        schema_instruction += _SCRIPT_ARTIFACT_ROOT_INSTRUCTION
    messages = [{"role": "system", "content": "\n".join(seams) + schema_instruction}, {"role": "user", "content": json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)}]
    calls: list[dict[str, Any]] = []
    def capture(messages_in, **kwargs):
        call_messages = (
            _PromptMustFitMessages(messages_in)
            if prompt_must_fit and isinstance(messages_in, list)
            else messages_in
        )
        raw = slot_fn(call_messages, **kwargs)
        calls.append({
            "temperature": kwargs.get("temperature"),
            "max_new_tokens": kwargs.get("max_new_tokens"),
            "raw_chars": len(str(raw)),
            "raw_sha256": hashlib.sha256(str(raw).encode("utf-8")).hexdigest(),
        })
        return raw
    def typed_repair_factory(*, original_prompt, failed_output, error):
        detail = str(error)
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
        elif pass_id in {"P3", "P3_rewrite"}:
            deterministic = (
                repair_radio_score_metadata(failed_output, repair_advisory)
                if repair_advisory is not None
                else None
            )
            if deterministic is not None and post_validator(deterministic) is None:
                log.info(
                    "[scifi_codex:%s] deterministic RadioScoreV4 cue-anchor repair accepted",
                    pass_id,
                )
                return deterministic
            repair_rules = (
                "This is a typed repair of the same RadioScoreV4, not a new "
                "unconstrained outline. The original request's score_graph_contract "
                "and advisory_word_plan are locked executable metadata. Copy the "
                "advisory_word_plan exactly. The flattened scenes[].beats array MUST "
                "contain every and only required_beat_ids, in that exact order, with "
                "the listed consecutive order values. Preserve existing title, premise, "
                "scene/shot descriptions, and beat intent wherever present; if a locked "
                "beat was omitted, add its complete score record using the existing "
                "story's cause-and-response progression rather than inventing another "
                "ID. Flatten all line_ids in assembly order into l001, l002, and so on "
                "with no gaps, and make every music cue anchor an existing line and its "
                "owning beat. Return one complete JSON object only."
            )
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
                "This is a typed repair of the same artifact, not a new creative response. "
                "Return one JSON object only. IDs are fixed lexical tokens: facts MUST use "
                "F01 through F12, entities MUST use E01 through E12, and numbers MUST use "
                "N01 through N12. Never emit bare F0, F1, E0, or N0. If the failed artifact "
                "used F0/F1/F2, change those references consistently to F01/F02/F03. "
                "For every source span, calculate quote from the original request exactly as "
                "payload[field][start:end]; do not paraphrase, infer, or retain a mismatched "
                "span. Preserve valid claims and remove only unsupported facts."
            )
            deterministic = repair_literal_source_metadata(
                failed_output,
                FactIndexV4,
                body["artifact_inputs"]["payload"]["payload"],
                zero_padded_ids=True,
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
        compact_request = {
            key: value for key, value in body.items()
            if key != "result_json_schema"
        }
        return [
            {"role": "system", "content": "\n".join(seams) + schema_instruction + "\n" + repair_rules},
            {"role": "user", "content": json.dumps({"failed_artifact": failed_output, "validation_error": detail, "original_request": compact_request}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)},
        ]
    try:
        # LLM slot: per-sub-pass injected creative/technical closure.
        result = structured_call(
            prompt=messages, schema=result_type, slot_fn=capture,
            base_temperature=base_temperature,
            structural_retry_temperature=structural_retry_temperature,
            max_new_tokens=max_new_tokens, max_attempts=3,
            repair_prompt_factory=typed_repair_factory,
            post_validator=post_validator, helper_name=f"scifi_codex:{pass_id}",
        )
    except Exception as exc:
        raise CodexPassError(f"{pass_id} failed: {exc}") from exc
    call_journal.setdefault("calls", []).append({"pass_id": pass_id, "slot": slot, "attempts": calls, "accepted": result.model_dump(mode="json")})
    return result


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
    p0 = invoke_codex_structured(pass_id="P0", slot="technical", slot_fn=technical_fn, pack=pack, seam_refs=("codex_fact_index_system",), artifact_inputs=p0_inputs, result_type=FactIndexV4, post_validator=lambda x: _validate_fact_index(x, payload, allowed_source_fields=p0_allowed_fields, expected_payload_sha256=env.source_digest), base_temperature=.20, structural_retry_temperature=.10, max_new_tokens=2000, call_journal=journal, prompt_must_fit=True)
    p1 = invoke_codex_structured(pass_id="P1", slot="creative", slot_fn=creative_fn, pack=pack, seam_refs=("codex_question_system",), artifact_inputs={"fact_index": p0.model_dump(mode="json")}, result_type=DramaticQuestionV4, post_validator=lambda x: None, base_temperature=.72, structural_retry_temperature=.32, max_new_tokens=1800, call_journal=journal)
    p2 = invoke_codex_structured(pass_id="P2", slot="creative", slot_fn=creative_fn, pack=pack, seam_refs=("codex_pressure_cast_system",), artifact_inputs={"question": p1.model_dump(mode="json")}, result_type=CastPlanV4, post_validator=_validate_cast_plan, base_temperature=.72, structural_retry_temperature=.32, max_new_tokens=1600, call_journal=journal)
    beat_ids = [f"b{i:03d}" for i in range(max(3, min(12, len(p2.cast) * 3)))]
    advisory = make_advisory_word_blueprint(steer.requested_words, beat_ids)
    score_graph_contract = _score_graph_contract(advisory)
    p3 = invoke_codex_structured(pass_id="P3", slot="creative", slot_fn=creative_fn, pack=pack, seam_refs=("codex_radio_score_system", "codex_coda_contract_system"), artifact_inputs={"question": p1.model_dump(mode="json"), "cast": p2.model_dump(mode="json"), "advisory_word_plan": advisory.model_dump(mode="json"), "score_graph_contract": score_graph_contract}, result_type=RadioScoreV4, post_validator=lambda x: _validate_radio_score_graph(x, advisory), base_temperature=.72, structural_retry_temperature=.32, max_new_tokens=3600, call_journal=journal, repair_advisory=advisory)
    review = invoke_codex_structured(pass_id="P4", slot="technical", slot_fn=technical_fn, pack=pack, seam_refs=("codex_radio_score_system", "codex_coda_contract_system"), artifact_inputs={"score": p3.model_dump(mode="json")}, result_type=StructureReviewV4, post_validator=lambda x: None, base_temperature=.20, structural_retry_temperature=.10, max_new_tokens=1800, call_journal=journal)
    score = p3
    if review.verdict == "rewrite":
        score = invoke_codex_structured(pass_id="P3_rewrite", slot="creative", slot_fn=creative_fn, pack=pack, seam_refs=("codex_radio_score_system", "codex_coda_contract_system"), artifact_inputs={"score": p3.model_dump(mode="json"), "review": review.model_dump(mode="json"), "advisory_word_plan": advisory.model_dump(mode="json"), "score_graph_contract": score_graph_contract}, result_type=RadioScoreV4, post_validator=lambda x: _validate_radio_score_graph(x, advisory), base_temperature=.55, structural_retry_temperature=.20, max_new_tokens=3600, call_journal=journal, repair_advisory=advisory)
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
