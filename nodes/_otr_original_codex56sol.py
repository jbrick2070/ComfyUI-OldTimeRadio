"""Strict original Lost and Found Frequency source-bank runner."""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Callable, Literal, Mapping

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from ._otr_canon import EpisodeCanon
from ._otr_content_authorship import stamp_receipt
from ._otr_json import parse_first_json_object
from ._otr_structured_call import schema_shape_instruction, structured_call
from . import _otr_ledger_freeze
from .production_ledger import stamp_word_counts


class OriginalCodex56SolError(RuntimeError): pass
class OriginalCodex56SolPassError(OriginalCodex56SolError): pass
class OriginalCodex56SolContractError(OriginalCodex56SolError): pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


Identifier = Annotated[
    str, BeforeValidator(lambda value: str(value) if isinstance(value, int)
                         and not isinstance(value, bool) else value),
]


def _role_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    lowered = value.strip().lower()
    if lowered == "announcer":
        return "announcer"
    if lowered in {"desk_operator", "desk operator"}:
        return "desk_operator"
    return "caller"


CastRole = Annotated[
    Literal["announcer", "desk_operator", "caller"],
    BeforeValidator(_role_value),
]


ArcPhase = Literal["opening", "rising", "reveal", "closing"]


class ConstraintDraw(StrictModel):
    deck_id: str
    deck_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    constraint_id: str
    lost_objects: list[str] = Field(min_length=3, max_length=6)
    acoustic_device: str
    helpful_ending: str


class ConstraintIngress(StrictModel):
    draw: ConstraintDraw


class FictionalName(StrictModel):
    name: str


class PossibilityCard(StrictModel):
    possibility_id: Identifier
    title_seed: str
    premise: str
    desk_operator: FictionalName
    callers: list[FictionalName] = Field(max_length=4)
    lost_objects: list[str] = Field(min_length=3, max_length=6)
    acoustic_device: str
    shared_cause: str
    clue_plan: list[str] = Field(min_length=3)
    helpful_resolution: str


class PossibilitySlate(StrictModel):
    possibilities: list[PossibilityCard] = Field(min_length=4, max_length=6)


class CandidateFinding(StrictModel):
    possibility_id: Identifier
    category: str
    detail: str
    blocking: bool


class SlateTriage(StrictModel):
    selected_possibility_id: Identifier
    findings: list[CandidateFinding]


class CallerThread(StrictModel):
    thread_id: Identifier
    caller_name: str
    lost_object: str
    practical_need: str


class CausalStep(StrictModel):
    step_id: Identifier
    cause: str
    effect: str


class AudibleClue(StrictModel):
    clue_id: Identifier
    thread_id: Identifier
    sound_or_phrase: str
    implication: str


class Interpretation(StrictModel):
    interpretation_id: Identifier
    clue_ids: list[Identifier] = Field(min_length=1)
    explanation: str
    is_true: bool


class ResolutionLink(StrictModel):
    thread_id: Identifier
    action: str
    result: str


class AudibleTruthMap(StrictModel):
    title: str
    premise: str
    setting: str
    desk_operator_name: str
    caller_threads: list[CallerThread] = Field(min_length=2)
    causal_steps: list[CausalStep] = Field(min_length=2)
    audible_clues: list[AudibleClue] = Field(min_length=3)
    interpretations: list[Interpretation] = Field(min_length=2)
    reveal: str
    resolution_links: list[ResolutionLink] = Field(min_length=2)


class FairPlayFinding(StrictModel):
    field_path: str = ""
    item_id: str = ""
    category: str
    detail: str
    blocking: bool


class FairPlayReport(StrictModel):
    accepted: bool
    findings: list[FairPlayFinding]


class CastConcept(StrictModel):
    char_id: Identifier
    name: str
    role: CastRole
    character_description: str


class SceneConcept(StrictModel):
    scene_id: Identifier
    description: str
    env: str


class ShotConcept(StrictModel):
    shot_id: Identifier
    scene_id: Identifier
    description: str
    visual_prompt: str
    env: str = ""


class LineIntent(StrictModel):
    intent: str
    arc_phase: ArcPhase
    clue_ids: list[Identifier] = Field(default_factory=list)


class BeatConcept(StrictModel):
    beat_id: Identifier
    shot_id: Identifier
    scene_id: Identifier
    char_id: Identifier
    speaker: str
    line_intent: LineIntent


class MusicBookend(StrictModel):
    description: str
    generation_prompt: str


class BroadcastScore(StrictModel):
    title: str
    premise: str
    setting: str
    cast: list[CastConcept] = Field(min_length=3, max_length=6)
    scenes: list[SceneConcept] = Field(min_length=2, max_length=4)
    shots: list[ShotConcept] = Field(min_length=2)
    beats: list[BeatConcept] = Field(min_length=5)
    orientation_beat_id: Identifier
    reveal_beat_id: Identifier
    closure_beat_id: Identifier
    opening_music: MusicBookend
    closing_music: MusicBookend


class ManifestLine(StrictModel):
    line_id: str
    beat_id: str
    shot_id: str
    scene_id: str
    char_id: str
    speaker: str
    speaker_role: Literal["announcer", "character"]
    boundary: Literal["shot_start", "beat_start", "continue"]
    arc_phase: ArcPhase
    intent: str
    clue_ids: list[Identifier] = Field(default_factory=list)
    orientation: bool | None = None
    clue: bool | None = None
    reveal: bool | None = None
    closure: bool | None = None


class ClosedLineManifest(StrictModel):
    lines: list[ManifestLine] = Field(min_length=5)
    orientation_line_id: str
    reveal_line_id: str
    closure_line_id: str


class SpokenLine(StrictModel):
    line_id: str
    char_id: str
    speaker: str
    text: str


class PerformanceScript(StrictModel):
    title: str
    lines: list[SpokenLine] = Field(min_length=5)


class ListenerFinding(StrictModel):
    line_id: str
    category: str
    detail: str
    blocking: bool


class BlindListenerReport(StrictModel):
    understood_cause: str
    understood_resolution: str
    findings: list[ListenerFinding]
    optional_notes: list[str]


class ContractFinding(StrictModel):
    field_path: str
    item_id: str
    exact_span: str
    category: str
    allowed_correction: str
    blocking: bool


class FinalContractAudit(StrictModel):
    accepted: bool
    findings: list[ContractFinding]
    warnings: list[str]


GenerateFn = Callable[..., str]


def _repair_rules(pass_id: str, error: Any) -> str:
    rules = ""
    if "forbidden" in str(error).lower():
        rules += (
            " Replace EVERY cited unsafe authored field/span. Unsafe prose "
            "is not immutable. Preserve the artifact structure, immutable "
            "constraint-draw fields, all IDs, and collection membership and "
            "cardinality, but rewrite every cited unsafe prose value."
        )
    if pass_id == "P3":
        rules += (
            " Preserve the complete artifact and every existing collection "
            "item. causal_steps MUST contain at least 2 items; audible_clues "
            "MUST contain at least 3 items; caller_threads and resolution_links "
            "MUST each contain at least 2 items. Never delete an item to repair "
            "an identifier or type error. Emit ONLY fields declared by the "
            "schema: never invent numbered, secondary, tertiary, or suffixed "
            "fields. Each caller_threads row has exactly one lost_object. If "
            "the failed artifact packed multiple lost objects into extra fields "
            "on one row, move each extra object into its own caller_threads row "
            "and give every caller thread exactly one resolution_links row; do "
            "not rename an unknown field and leave it in place. "
            "causal_steps, audible_clues, interpretations, and "
            "resolution_links MUST appear only as top-level arrays; never "
            "place them inside caller_threads rows. "
            "interpretations MUST retain at least "
            "2 items, including at least one true and one plausible false "
            "interpretation. To repair truth balance, change is_true on one "
            "existing interpretation; do not add or remove interpretations. "
            "Every interpretation.clue_ids MUST retain at least one existing "
            "audible_clues clue_id; an empty clue_ids list is invalid."
        )
    if pass_id == "P5":
        rules += (
            " Preserve the complete artifact and every existing beat while "
            "repairing it. Return exactly 4 scenes, at least 2 shots, and at "
            "least 5 beats; never delete a beat to repair another field. If "
            "the failed artifact has more than 4 scenes, consolidate it into "
            "exactly 4 and reassign every shot and beat to retained scene IDs; "
            "do not echo the oversized artifact. Put env inside every scene "
            "row and visual_prompt inside every shot row. "
            "Group every shot's beats into one adjacent contiguous block: "
            "once beats move to a different shot_id, never return to an "
            "earlier shot_id. Preserve orientation-before-reveal-before-closure. "
            "Never emit schema-path pseudo-fields such as `scenes[*].env` or "
            "`shots[*].visual_prompt` at the top level. "
            "scenes, shots, and beats MUST be separate top-level arrays. "
            "Never place shots inside scenes or beats inside shots. "
            "Every line_intent MUST "
            "have exactly the keys intent, arc_phase, and clue_ids. clue_ids "
            "MUST always be an array of existing clue IDs, or [] when there "
            "is no clue; singular clue_id is forbidden. Cast MUST include one "
            "separate row with char_id exactly "
            "`announcer` and role exactly `announcer`, exactly one different "
            "row with role `desk_operator`, and every remaining row with "
            "role `caller`. Never combine announcer and desk operator. Every "
            "scene MUST retain a non-empty env and every shot MUST retain a "
            "non-empty visual_prompt. line_intent.arc_phase MUST be exactly "
            "one of `opening`, `rising`, `reveal`, or `closing`: the "
            "orientation_beat_id beat is `opening`, the reveal_beat_id beat "
            "is `reveal`, the closure_beat_id beat is `closing`, and every "
            "other beat is `rising`. Never use orientation, resolution, "
            "closure, climax, or other synonyms as arc_phase values."
        )
    if pass_id == "P7":
        rules += (
            " understood_cause and understood_resolution MUST be strings; "
            "findings MUST be a list; optional_notes MUST be a list of "
            "strings, or [] when there are no optional notes."
        )
    if pass_id in {"P9", "P9_rerun"}:
        rules += (
            " Return only the compact audit envelope. accepted MUST be one "
            "boolean, findings MUST be a list, and warnings MUST be a list of "
            "strings. Never copy the manifest or script into accepted or any "
            "other output field. Every finding MUST include field_path, "
            "item_id, exact_span, category, allowed_correction, and blocking; "
            "blocking MUST be a boolean. If there are no concrete defects, "
            "return accepted=true, findings=[], and warnings=[]."
        )
    return rules


def _repair_duplicate_score_clues(
    score: BroadcastScore,
    truth: AudibleTruthMap,
) -> BroadcastScore | None:
    """Remove duplicate clue references without inventing placement.

    This repair is safe only when the score already covers the exact truth-map
    clue set. The first authored placement wins; later duplicate references are
    removed in beat order. Missing or unknown clues remain semantic failures
    for the LLM repair path.
    """
    expected = {clue.clue_id for clue in truth.audible_clues}
    assigned = [clue_id for beat in score.beats
                for clue_id in beat.line_intent.clue_ids]
    if set(assigned) != expected or len(assigned) == len(set(assigned)):
        return None

    repaired = score.model_copy(deep=True)
    seen: set[str] = set()
    for beat in repaired.beats:
        unique_ids = []
        for clue_id in beat.line_intent.clue_ids:
            if clue_id not in seen:
                seen.add(clue_id)
                unique_ids.append(clue_id)
        beat.line_intent.clue_ids = unique_ids
    return repaired


def _repair_truth_map_collection_placement(
    failed_output: str,
    selected: PossibilityCard,
) -> AudibleTruthMap | None:
    """Normalize declared collection placement without rewriting values.

    Existing top-level collections are authoritative; nested copies are
    forbidden extras. Lift nested rows only when the top-level collection is
    missing or empty, and return only a fully valid truth graph.
    """
    try:
        data = parse_first_json_object(failed_output)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    threads = data.get("caller_threads")
    if not isinstance(threads, list) or not all(
        isinstance(row, dict) for row in threads
    ):
        return None

    changed = False
    for collection in (
        "causal_steps", "audible_clues", "interpretations", "resolution_links",
    ):
        lifted: list[Any] = []
        for row in threads:
            if collection not in row:
                continue
            nested = row[collection]
            if not isinstance(nested, list):
                return None
            del row[collection]
            lifted.extend(nested)
            changed = True
        if not lifted:
            continue
        existing = data.get(collection)
        if existing is None:
            existing = []
        if not isinstance(existing, list):
            return None
        data[collection] = existing if existing else lifted

    if not changed:
        return None
    try:
        repaired = AudibleTruthMap.model_validate(data)
    except ValidationError:
        return None
    if _validate_truth_map(repaired, selected) is not None:
        return None
    return repaired


def _repair_score_collection_placement(
    failed_output: str,
    truth: AudibleTruthMap,
) -> BroadcastScore | None:
    """Normalize scene/shot/beat placement without rewriting authored values."""
    try:
        data = parse_first_json_object(failed_output)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    scenes = data.get("scenes")
    if not isinstance(scenes, list) or not all(
        isinstance(row, dict) for row in scenes
    ):
        return None

    changed = False
    nested_shots: list[Any] = []
    for scene in scenes:
        if "shots" not in scene:
            continue
        value = scene["shots"]
        if not isinstance(value, list):
            return None
        del scene["shots"]
        nested_shots.extend(value)
        changed = True
    top_shots = data.get("shots")
    if top_shots is None:
        top_shots = []
    if not isinstance(top_shots, list):
        return None
    data["shots"] = top_shots if top_shots else nested_shots
    if not all(isinstance(row, dict) for row in data["shots"]):
        return None

    nested_beats: list[Any] = []
    for shot in data["shots"]:
        if "beats" not in shot:
            continue
        value = shot["beats"]
        if not isinstance(value, list):
            return None
        del shot["beats"]
        nested_beats.extend(value)
        changed = True
    top_beats = data.get("beats")
    if top_beats is None:
        top_beats = []
    if not isinstance(top_beats, list):
        return None
    data["beats"] = top_beats if top_beats else nested_beats

    if not changed:
        return None
    try:
        repaired = BroadcastScore.model_validate(data)
    except ValidationError:
        return None
    if _validate_score(repaired, truth) is not None:
        return None
    return repaired


def _call(*, pass_id: str, slot: str, fn: GenerateFn, pack: Any,
          seam: str, inputs: Mapping[str, Any], schema: type[BaseModel],
          scheduler: Any, journal: list[dict], tokens: int = 2400,
          post_validator: Callable[[BaseModel], Any] = lambda value: None,
          ) -> BaseModel:
    system = str(pack.prompt_stages.get(seam) or "").strip()
    if not system:
        raise OriginalCodex56SolContractError(f"missing prompt seam {seam}")
    prompt = [
        {"role": "system", "content": system + "\n" + schema_shape_instruction(schema)},
        {"role": "user", "content": json.dumps(inputs, ensure_ascii=False, sort_keys=True)},
    ]
    attempts = []
    def capture(messages, **kwargs):
        raw = fn(messages, **kwargs)
        attempts.append(hashlib.sha256(str(raw).encode()).hexdigest())
        return raw
    def repair(*, original_prompt, failed_output, error):
        effective_error: Any = error
        if pass_id == "P3" and schema is AudibleTruthMap:
            try:
                selected = PossibilityCard.model_validate(inputs["selected"])
            except (KeyError, TypeError, ValidationError):
                pass
            else:
                repaired_truth = _repair_truth_map_collection_placement(
                    failed_output, selected,
                )
                if repaired_truth is not None:
                    content_error = post_validator(repaired_truth)
                    if content_error is None:
                        return repaired_truth
                    effective_error = RuntimeError(
                        f"{error}; after structural normalization: "
                        f"{content_error}"
                    )
        if pass_id == "P5" and schema is BroadcastScore:
            try:
                truth = AudibleTruthMap.model_validate(inputs["truth_map"])
            except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                pass
            else:
                placement_repair = _repair_score_collection_placement(
                    failed_output, truth,
                )
                if placement_repair is not None:
                    content_error = post_validator(placement_repair)
                    if content_error is None:
                        return placement_repair
                    effective_error = RuntimeError(
                        f"{error}; after structural normalization: "
                        f"{content_error}"
                    )
                if "each truth-map clue must be assigned to exactly one" in str(error):
                    try:
                        failed_score = BroadcastScore.model_validate(
                            parse_first_json_object(failed_output)
                        )
                    except (json.JSONDecodeError, ValueError, TypeError):
                        pass
                    else:
                        repaired_score = _repair_duplicate_score_clues(
                            failed_score, truth,
                        )
                        if repaired_score is not None:
                            content_error = post_validator(repaired_score)
                            if content_error is None:
                                return repaired_score
                            effective_error = RuntimeError(
                                f"{error}; after structural normalization: "
                                f"{content_error}"
                            )
        pass_rules = _repair_rules(pass_id, effective_error)
        return [
            {"role": "system", "content": system + "\nReturn the same complete artifact, repairing only the typed contract error." + pass_rules + " JSON only.\n" + schema_shape_instruction(schema)},
            {"role": "user", "content": json.dumps({"failed_artifact": failed_output, "error": str(effective_error), "inputs": inputs}, ensure_ascii=False, sort_keys=True)},
        ]
    try:
        with scheduler.helper_context(f"original_codex56sol:{pass_id}"):
            # LLM slot: per-sub-pass workflow creative/technical closure.
            result = structured_call(
                prompt=prompt, schema=schema, slot_fn=capture,
                base_temperature=.72 if slot == "creative" else .18,
                structural_retry_temperature=.25 if slot == "creative" else .08,
                max_new_tokens=tokens, max_attempts=3,
                post_validator=post_validator,
                repair_prompt_factory=repair,
                helper_name=f"original_codex56sol:{pass_id}",
            )
    except Exception as exc:
        raise OriginalCodex56SolPassError(f"{pass_id} failed: {exc}") from exc
    journal.append({"pass_id": pass_id, "slot": slot, "attempts": attempts})
    return result


def _validate_slate(slate: PossibilitySlate, draw: ConstraintDraw) -> str | None:
    ids = [card.possibility_id for card in slate.possibilities]
    if len(set(ids)) != len(ids):
        return "possibility ids must be unique"
    for card in slate.possibilities:
        if (card.lost_objects != draw.lost_objects
                or card.acoustic_device != draw.acoustic_device):
            return (
                f"{card.possibility_id}: lost_objects and acoustic_device "
                "must be copied verbatim from the immutable constraint draw"
            )
    return None


def _validate_triage(triage: SlateTriage, slate: PossibilitySlate) -> str | None:
    ids = {card.possibility_id for card in slate.possibilities}
    if triage.selected_possibility_id not in ids:
        return "selected_possibility_id must exactly match one slate id"
    if any(f.blocking and f.possibility_id == triage.selected_possibility_id
           for f in triage.findings):
        return "triage selected a possibility it marked blocking"
    return None


def _validate_truth_map(
    truth: AudibleTruthMap,
    selected: PossibilityCard | None = None,
) -> str | None:
    collections = {
        "caller_threads": [row.thread_id for row in truth.caller_threads],
        "causal_steps": [row.step_id for row in truth.causal_steps],
        "audible_clues": [row.clue_id for row in truth.audible_clues],
        "interpretations": [row.interpretation_id for row in truth.interpretations],
    }
    if any(len(values) != len(set(values)) for values in collections.values()):
        return "truth-map structural ids must be unique within each collection"
    thread_ids = set(collections["caller_threads"])
    clue_ids = set(collections["audible_clues"])
    if selected is not None and Counter(
        row.lost_object for row in truth.caller_threads
    ) != Counter(selected.lost_objects):
        return (
            "caller_threads must contain exactly one row per selected lost "
            "object, with one lost_object field per row"
        )
    if any(clue.thread_id not in thread_ids for clue in truth.audible_clues):
        return "every audible clue thread_id must resolve"
    clue_thread_ids = {clue.thread_id for clue in truth.audible_clues}
    if clue_thread_ids != thread_ids:
        return "every caller thread must have at least one audible clue"
    resolution_thread_ids = [link.thread_id for link in truth.resolution_links]
    if Counter(resolution_thread_ids) != Counter(thread_ids):
        return "every caller thread must have exactly one resolution link"
    if not any(row.is_true for row in truth.interpretations):
        return "interpretations need at least one true interpretation"
    if not any(not row.is_true for row in truth.interpretations):
        return "interpretations need at least one plausible false interpretation"
    if any(set(row.clue_ids) - clue_ids for row in truth.interpretations):
        return "every interpretation clue_id must resolve"
    return None


def _validate_score(score: BroadcastScore,
                    truth: AudibleTruthMap) -> str | None:
    if sum(c.char_id == "announcer" and c.role == "announcer"
           for c in score.cast) != 1:
        return "cast needs one separate char_id='announcer', role='announcer' row"
    if sum(c.role == "desk_operator" for c in score.cast) != 1:
        return "cast needs exactly one separate desk_operator row"
    if len({c.char_id for c in score.cast}) != len(score.cast):
        return "cast char_id values must be unique"
    cast_ids = {c.char_id for c in score.cast}
    scene_ids = {s.scene_id for s in score.scenes}
    shot_by_id = {s.shot_id: s for s in score.shots}
    beat_ids = [b.beat_id for b in score.beats]
    if len(scene_ids) != len(score.scenes) or len(shot_by_id) != len(score.shots):
        return "scene_id and shot_id values must be unique"
    if len(beat_ids) != len(set(beat_ids)):
        return "beat_id values must be unique"
    if any(shot.scene_id not in scene_ids for shot in score.shots):
        return "every shot scene_id must resolve"
    for beat in score.beats:
        shot = shot_by_id.get(beat.shot_id)
        if (shot is None or beat.scene_id not in scene_ids
                or shot.scene_id != beat.scene_id or beat.char_id not in cast_ids):
            return "every beat graph reference must resolve"
    closed_shots: set[str] = set()
    previous_shot = None
    for beat in score.beats:
        if beat.shot_id != previous_shot:
            if beat.shot_id in closed_shots:
                return "beats for each shot must form one contiguous block"
            if previous_shot is not None:
                closed_shots.add(previous_shot)
            previous_shot = beat.shot_id
    landmark_ids = [score.orientation_beat_id, score.reveal_beat_id,
                    score.closure_beat_id]
    if len(set(landmark_ids)) != 3 or any(value not in beat_ids for value in landmark_ids):
        return "orientation, reveal, and closure beat ids must be distinct and resolve"
    positions = {beat_id: index for index, beat_id in enumerate(beat_ids)}
    if not (positions[score.orientation_beat_id] < positions[score.reveal_beat_id]
            < positions[score.closure_beat_id]):
        return "landmark order must be orientation before reveal before closure"
    by_beat = {beat.beat_id: beat for beat in score.beats}
    expected_phases = {
        score.orientation_beat_id: "opening",
        score.reveal_beat_id: "reveal",
        score.closure_beat_id: "closing",
    }
    if any(by_beat[beat_id].line_intent.arc_phase != phase
           for beat_id, phase in expected_phases.items()):
        return "landmark beat arc phases must be opening, reveal, and closing"
    expected_clues = {clue.clue_id for clue in truth.audible_clues}
    assigned_clues = [clue_id for beat in score.beats
                      for clue_id in beat.line_intent.clue_ids]
    if set(assigned_clues) != expected_clues:
        return "line intents must cover every truth-map clue and no unknown clue"
    if len(assigned_clues) != len(set(assigned_clues)):
        return "each truth-map clue must be assigned to exactly one line intent"
    return None


def _iter_authored_strings(value: Any, path: str = ""):
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="python")
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            yield from _iter_authored_strings(item, child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{path}.{index}" if path else str(index)
            yield from _iter_authored_strings(item, child)
    elif isinstance(value, str):
        yield path, value


def _validate_authored_surface(value: Any, rules: Any) -> str | None:
    violations = []
    for path, text in _iter_authored_strings(value):
        for term in getattr(rules, "banned_phrases", ()):
            if re.search(rf"\b{re.escape(term)}\b", text, re.I):
                violations.append(
                    f"authored field {path!r} contains forbidden term {term!r}"
                )
        for pattern in getattr(rules, "stage_business", ()):
            if pattern.search(text):
                violations.append(
                    f"authored field {path!r} contains a forbidden authored surface"
                )
    if violations:
        return "; ".join(violations) + "; replace every cited authored detail"
    return None


def _truth_item_ids(truth: AudibleTruthMap) -> dict[str, set[str]]:
    return {
        "caller_threads": {row.thread_id for row in truth.caller_threads},
        "causal_steps": {row.step_id for row in truth.causal_steps},
        "audible_clues": {row.clue_id for row in truth.audible_clues},
        "interpretations": {row.interpretation_id for row in truth.interpretations},
        "resolution_links": {row.thread_id for row in truth.resolution_links},
    }


def _corroborated_fair_blocks(report: FairPlayReport,
                              truth: AudibleTruthMap) -> list[FairPlayFinding]:
    ids = _truth_item_ids(truth)
    blocks = []
    for finding in report.findings:
        root = finding.field_path.strip().split(".", 1)[0]
        if (finding.blocking and finding.category.strip()
                and finding.detail.strip() and finding.item_id in ids.get(root, set())):
            blocks.append(finding)
    return blocks


def _compile_manifest(score: BroadcastScore) -> ClosedLineManifest:
    cast = {row.char_id: row for row in score.cast}
    beat_to_line: dict[str, str] = {}
    lines = []
    previous_shot = None
    for index, beat in enumerate(score.beats, 1):
        line_id = f"line_{index:03d}"
        beat_to_line[beat.beat_id] = line_id
        intent = beat.line_intent
        lines.append(ManifestLine(
            line_id=line_id, beat_id=beat.beat_id, shot_id=beat.shot_id,
            scene_id=beat.scene_id, char_id=beat.char_id, speaker=beat.speaker,
            speaker_role="announcer" if cast[beat.char_id].role == "announcer"
            else "character",
            boundary="shot_start" if beat.shot_id != previous_shot else "beat_start",
            arc_phase=intent.arc_phase, intent=intent.intent,
            clue_ids=list(intent.clue_ids),
            orientation=beat.beat_id == score.orientation_beat_id,
            clue=bool(intent.clue_ids), reveal=beat.beat_id == score.reveal_beat_id,
            closure=beat.beat_id == score.closure_beat_id,
        ))
        previous_shot = beat.shot_id
    manifest = ClosedLineManifest(
        lines=lines,
        orientation_line_id=beat_to_line[score.orientation_beat_id],
        reveal_line_id=beat_to_line[score.reveal_beat_id],
        closure_line_id=beat_to_line[score.closure_beat_id],
    )
    error = _validate_manifest(score, manifest)
    if error:
        raise OriginalCodex56SolContractError(error)
    return manifest


def _validate_manifest(score: BroadcastScore,
                       manifest: ClosedLineManifest) -> str | None:
    beat_ids = {beat.beat_id for beat in score.beats}
    manifest_beat_ids = [line.beat_id for line in manifest.lines]
    if (set(manifest_beat_ids) != beat_ids
            or len(manifest_beat_ids) != len(beat_ids)):
        return "manifest must cover every score beat exactly once"
    landmarks = {
        "orientation": manifest.orientation_line_id,
        "reveal": manifest.reveal_line_id,
        "closure": manifest.closure_line_id,
    }
    line_ids = {line.line_id for line in manifest.lines}
    if len(line_ids) != len(manifest.lines):
        return "manifest line_id values must be unique"
    if any(line_id not in line_ids for line_id in landmarks.values()):
        return "every top-level landmark line id must resolve in manifest lines"
    if len(set(landmarks.values())) != 3:
        return "orientation, reveal, and closure line ids must be distinct"
    positions = {line.line_id: index for index, line in enumerate(manifest.lines)}
    if not (positions[manifest.orientation_line_id]
            < positions[manifest.reveal_line_id]
            < positions[manifest.closure_line_id]):
        return "landmark line order must be orientation before reveal before closure"
    expected_clues = {clue_id for beat in score.beats
                      for clue_id in beat.line_intent.clue_ids}
    manifest_clues = [clue_id for line in manifest.lines for clue_id in line.clue_ids]
    if set(manifest_clues) != expected_clues or len(manifest_clues) != len(set(manifest_clues)):
        return "manifest clue ids must exactly cover score line-intent clues"
    score_by_beat = {beat.beat_id: beat for beat in score.beats}
    cast_by_id = {row.char_id: row for row in score.cast}
    for line in manifest.lines:
        beat = score_by_beat[line.beat_id]
        expected_role = ("announcer" if cast_by_id[beat.char_id].role == "announcer"
                         else "character")
        if (line.shot_id != beat.shot_id or line.scene_id != beat.scene_id
                or line.char_id != beat.char_id or line.speaker != beat.speaker
                or line.speaker_role != expected_role
                or line.intent != beat.line_intent.intent
                or line.arc_phase != beat.line_intent.arc_phase
                or line.clue_ids != beat.line_intent.clue_ids):
            return "manifest row differs from its accepted score beat"
    for line in manifest.lines:
        for marker, expected in landmarks.items():
            if getattr(line, marker) is True and line.line_id != expected:
                return f"line {line.line_id!r} asserts {marker} but landmark is {expected!r}"
    return None


def _validate_graph(score: BroadcastScore, manifest: ClosedLineManifest,
                    script: PerformanceScript) -> str | None:
    cast = {c.char_id: c for c in score.cast}
    scenes = {s.scene_id for s in score.scenes}
    shots = {s.shot_id: s for s in score.shots}
    beats = {b.beat_id: b for b in score.beats}
    if len(cast) != len(score.cast) or "announcer" not in cast:
        return "cast ids are invalid"
    if sum(c.role == "desk_operator" for c in score.cast) != 1:
        return "exactly one desk operator is required"
    for shot in score.shots:
        if shot.scene_id not in scenes:
            return "shot scene reference is invalid"
    for beat in score.beats:
        if (beat.shot_id not in shots or beat.scene_id not in scenes
                or beat.char_id not in cast
                or shots[beat.shot_id].scene_id != beat.scene_id):
            return "beat graph reference is invalid"
    mids = [m.line_id for m in manifest.lines]
    manifest_beats = [m.beat_id for m in manifest.lines]
    if (len(mids) != len(set(mids)) or set(manifest_beats) != set(beats)
            or len(manifest_beats) != len(beats)):
        return "manifest must cover every beat exactly"
    sids = [line.line_id for line in script.lines]
    if sids != mids:
        return "script line order/coverage differs from manifest"
    by_manifest = {m.line_id: m for m in manifest.lines}
    for line in script.lines:
        m = by_manifest[line.line_id]
        if line.char_id != m.char_id or line.speaker != m.speaker:
            return "script roster differs from manifest"
    for required in (manifest.orientation_line_id, manifest.reveal_line_id,
                     manifest.closure_line_id):
        if required not in set(mids):
            return "manifest landmark is missing"
    return None


def _validate_text(script: PerformanceScript, rules: Any) -> str | None:
    for line in script.lines:
        text = line.text.strip()
        if not text:
            return "spoken text is empty"
        if re.search(r"^[A-Z][A-Z0-9 _-]{1,24}:\s*", text):
            return "speaker label in spoken text"
    return _validate_authored_surface(script, rules)


def _preceding_lines(manifest: ClosedLineManifest,
                     script: PerformanceScript) -> list[dict[str, str]]:
    script_by_id = {line.line_id: line for line in script.lines}
    packet = []
    reveal_found = False
    for row in manifest.lines:
        if row.line_id == manifest.reveal_line_id:
            reveal_found = True
            break
        spoken = script_by_id[row.line_id]
        packet.append({"line_id": spoken.line_id, "char_id": spoken.char_id,
                       "speaker": spoken.speaker, "text": spoken.text})
    if not reveal_found:
        raise OriginalCodex56SolContractError("blind-listener reveal line is missing")
    if not packet:
        raise OriginalCodex56SolContractError("blind-listener packet is empty")
    return packet


def _listener_blocks(report: BlindListenerReport,
                     allowed_line_ids: set[str]) -> list[ListenerFinding]:
    return [finding for finding in report.findings
            if finding.blocking and finding.line_id in allowed_line_ids
            and finding.category.strip() and finding.detail.strip()]


def _audit_blocks(report: FinalContractAudit,
                  script: PerformanceScript) -> list[ContractFinding]:
    text_by_id = {line.line_id: line.text for line in script.lines}
    return [finding for finding in report.findings
            if finding.blocking and finding.item_id in text_by_id
            and finding.field_path.strip() and finding.category.strip()
            and finding.allowed_correction.strip() and finding.exact_span
            and finding.exact_span in text_by_id[finding.item_id]]


def _validate_script(score: BroadcastScore, manifest: ClosedLineManifest,
                     script: PerformanceScript, rules: Any) -> str | None:
    return (_validate_graph(score, manifest, script)
            or _validate_text(script, rules))


def _assert_script_valid(score: BroadcastScore, manifest: ClosedLineManifest,
                         script: PerformanceScript, rules: Any) -> None:
    error = _validate_script(score, manifest, script, rules)
    if error:
        raise OriginalCodex56SolContractError(error)


def _cast_rows(score: BroadcastScore, episode_id: str) -> list[dict]:
    from config.cast_pools import VOICE_PROFILES, pick_announcer
    rng = random.Random(hashlib.sha256(episode_id.encode()).hexdigest())
    profiles = list(VOICE_PROFILES)
    rng.shuffle(profiles)
    rows = []
    used = set()
    for concept in score.cast:
        if concept.char_id == "announcer":
            row = pick_announcer(rng)
            row["char_id"] = "announcer"
            row["name"] = concept.name
            row["character_description"] = concept.character_description
        else:
            available = [p for p in profiles if p[0] not in used]
            if not available:
                raise OriginalCodex56SolContractError("voice inventory exhausted")
            preset, gender, _lang, _tags = available[0]
            used.add(preset)
            row = {"char_id": concept.char_id, "name": concept.name,
                   "character_description": concept.character_description,
                   "gender": gender, "tts_model": "bark",
                   "voice_preset": preset}
        rows.append(row)
    return rows


class OriginalCodex56SolFinalizer:
    def before_save(self, *, ctx: Any) -> None:
        from ._otr_content_authorship import validate_receipt
        validate_receipt(ctx.led.data)
        pre = _otr_ledger_freeze.phase_0_gap_audit_pre(ctx.led)
        post = _otr_ledger_freeze.phase_10_gap_audit_post_and_freeze(ctx.led)
        if pre.errors or post.errors:
            raise OriginalCodex56SolContractError("; ".join(map(str, [*pre.errors, *post.errors])))
        if ctx.led.data.get("meta", {}).get("freeze_verdict") not in ("frozen_clean", "frozen_with_warns"):
            raise OriginalCodex56SolContractError("ledger did not reach a clean freeze")

    def after_save(self, *, saved_path: str, ledger_data: Mapping[str, Any]) -> None:
        from ._otr_content_authorship import validate_receipt
        saved = json.loads(Path(saved_path).read_text(encoding="utf-8"))
        validate_receipt(saved)
        if saved.get("meta", {}).get("freeze_verdict") not in ("frozen_clean", "frozen_with_warns"):
            raise OriginalCodex56SolContractError("saved ledger freeze is invalid")


@dataclass
class OriginalCodex56SolTailParts:
    outline_view: Any
    canon: Any
    final_title_override: str
    run_story_spine: bool
    tail_finalizer: Any


def run_original_codex56sol_episode(
    *, payload: dict[str, str], pack: Any, resolved: Mapping[str, Any], led: Any,
    meta: dict[str, Any], creative_fn: GenerateFn, technical_fn: GenerateFn,
    slot_scheduler: Any, source_bank_row: Any, story_rules: Mapping[str, Any],
    episode_root: Path, episode_id: str,
) -> OriginalCodex56SolTailParts:
    del source_bank_row, episode_root
    try:
        draw = ConstraintDraw.model_validate(json.loads(payload["seed_text"]))
    except Exception as exc:
        raise OriginalCodex56SolContractError(f"invalid immutable constraint draw: {exc}") from exc
    journal: list[dict] = []
    ingress = ConstraintIngress(draw=draw)
    slate = _call(pass_id="P1", slot="creative", fn=creative_fn, pack=pack, seam="codex56_possibility_slate", inputs={"ingress": ingress.model_dump(mode="json"), "operator_hint": (meta.get("source_meta") or {}).get("operator_hint", "")}, schema=PossibilitySlate, scheduler=slot_scheduler, journal=journal, post_validator=lambda value: _validate_slate(value, draw) or _validate_authored_surface(value, story_rules))
    triage = _call(pass_id="P2", slot="technical", fn=technical_fn, pack=pack, seam="codex56_slate_triage", inputs={"slate": slate.model_dump(mode="json")}, schema=SlateTriage, scheduler=slot_scheduler, journal=journal, tokens=1600, post_validator=lambda value: _validate_triage(value, slate))
    selected = next((p for p in slate.possibilities if p.possibility_id == triage.selected_possibility_id), None)
    if selected is None or any(f.blocking and f.possibility_id == selected.possibility_id for f in triage.findings):
        raise OriginalCodex56SolContractError("triage did not select a valid possibility")
    truth = _call(pass_id="P3", slot="creative", fn=creative_fn, pack=pack, seam="codex56_audible_truth_map", inputs={"selected": selected.model_dump(mode="json"), "draw": draw.model_dump(mode="json")}, schema=AudibleTruthMap, scheduler=slot_scheduler, journal=journal, tokens=2800, post_validator=lambda value: _validate_truth_map(value, selected) or _validate_authored_surface(value, story_rules))
    fair = _call(pass_id="P4", slot="technical", fn=technical_fn, pack=pack, seam="codex56_fair_play_audit", inputs={"truth_map": truth.model_dump(mode="json")}, schema=FairPlayReport, scheduler=slot_scheduler, journal=journal, tokens=1600)
    corroborated_fair_blocks = _corroborated_fair_blocks(fair, truth)
    if corroborated_fair_blocks:
        raise OriginalCodex56SolContractError("fair-play audit rejected the truth map")
    score = _call(pass_id="P5", slot="creative", fn=creative_fn, pack=pack, seam="codex56_broadcast_score", inputs={"truth_map": truth.model_dump(mode="json"), "target_words_advisory": int(resolved["target_words"]), "num_characters_advisory": int(resolved["num_characters"])}, schema=BroadcastScore, scheduler=slot_scheduler, journal=journal, tokens=3600, post_validator=lambda value: _validate_score(value, truth) or _validate_authored_surface(value, story_rules))
    manifest = _compile_manifest(score)
    script = _call(pass_id="P6", slot="creative", fn=creative_fn, pack=pack, seam="codex56_performance_script", inputs={"score": score.model_dump(mode="json"), "manifest": manifest.model_dump(mode="json"), "target_words_advisory": int(resolved["target_words"])}, schema=PerformanceScript, scheduler=slot_scheduler, journal=journal, tokens=max(2600, int(resolved["target_words"]) * 6), post_validator=lambda value: _validate_script(score, manifest, value, story_rules))
    _assert_script_valid(score, manifest, script, story_rules)
    preceding_lines = _preceding_lines(manifest, script)
    listener = _call(pass_id="P7", slot="technical", fn=technical_fn, pack=pack, seam="codex56_blind_listener", inputs={"preceding_lines": preceding_lines}, schema=BlindListenerReport, scheduler=slot_scheduler, journal=journal, tokens=1800)
    listener_blocks = _listener_blocks(
        listener, {line["line_id"] for line in preceding_lines})
    if listener_blocks:
        script = _call(pass_id="P8", slot="creative", fn=creative_fn, pack=pack, seam="codex56_broadcast_retake", inputs={"manifest": manifest.model_dump(mode="json"), "previous_script": script.model_dump(mode="json"), "findings": [finding.model_dump(mode="json") for finding in listener_blocks]}, schema=PerformanceScript, scheduler=slot_scheduler, journal=journal, tokens=max(2600, int(resolved["target_words"]) * 6), post_validator=lambda value: _validate_script(score, manifest, value, story_rules))
        _assert_script_valid(score, manifest, script, story_rules)
    elif listener.optional_notes or any(not finding.blocking for finding in listener.findings):
        try:
            optional_script = _call(pass_id="P8_optional", slot="creative", fn=creative_fn, pack=pack, seam="codex56_broadcast_retake", inputs={"manifest": manifest.model_dump(mode="json"), "previous_script": script.model_dump(mode="json"), "findings": listener.model_dump(mode="json")}, schema=PerformanceScript, scheduler=slot_scheduler, journal=journal, tokens=max(2600, int(resolved["target_words"]) * 6), post_validator=lambda value: _validate_script(score, manifest, value, story_rules))
            _assert_script_valid(score, manifest, optional_script, story_rules)
            script = optional_script
        except OriginalCodex56SolError:
            pass
    audit = _call(pass_id="P9", slot="technical", fn=technical_fn, pack=pack, seam="codex56_final_contract_audit", inputs={"manifest": manifest.model_dump(mode="json"), "script": script.model_dump(mode="json")}, schema=FinalContractAudit, scheduler=slot_scheduler, journal=journal, tokens=1800)
    audit_blocks = _audit_blocks(audit, script)
    if audit_blocks:
        script = _call(pass_id="P9_retake", slot="creative", fn=creative_fn, pack=pack, seam="codex56_broadcast_retake", inputs={"manifest": manifest.model_dump(mode="json"), "previous_script": script.model_dump(mode="json"), "findings": [finding.model_dump(mode="json") for finding in audit_blocks]}, schema=PerformanceScript, scheduler=slot_scheduler, journal=journal, tokens=max(2600, int(resolved["target_words"]) * 6), post_validator=lambda value: _validate_script(score, manifest, value, story_rules))
        _assert_script_valid(score, manifest, script, story_rules)
        audit = _call(pass_id="P9_rerun", slot="technical", fn=technical_fn, pack=pack, seam="codex56_final_contract_audit", inputs={"manifest": manifest.model_dump(mode="json"), "script": script.model_dump(mode="json")}, schema=FinalContractAudit, scheduler=slot_scheduler, journal=journal, tokens=1800)
        if _audit_blocks(audit, script):
            raise OriginalCodex56SolContractError("final contract audit rejected the repaired script")

    led.set_cast(_cast_rows(score, episode_id))
    led.set_scenes([s.model_dump(mode="json") for s in score.scenes])
    led.set_shots([s.model_dump(mode="json") for s in score.shots])
    mby = {m.beat_id: m for m in manifest.lines}
    led.set_beats([{"beat_id": b.beat_id, "shot_id": b.shot_id, "scene_id": b.scene_id, "speaker": b.speaker, "char_id": b.char_id, "line_ids": [mby[b.beat_id].line_id]} for b in score.beats])
    sby = {s.line_id: s for s in script.lines}
    led.set_lines([{"line_id": m.line_id, "beat_id": m.beat_id, "shot_id": m.shot_id, "char_id": m.char_id, "speaker_role": m.speaker_role, "text": sby[m.line_id].text, "boundary": m.boundary, "traits": "", "arc_phase": m.arc_phase, "beat_intent": m.intent, "dialogue_slot_id": m.line_id} for m in manifest.lines])
    led.set_music([
        {"cue_id":"opening","placement":"opening","description":score.opening_music.description,"generation_prompt":score.opening_music.generation_prompt,"anchor_line_id":manifest.orientation_line_id},
        {"cue_id":"closing","placement":"closing","description":score.closing_music.description,"generation_prompt":score.closing_music.generation_prompt,"anchor_line_id":manifest.closure_line_id},
    ])
    led.data["clips"] = []
    stamp_word_counts(led)
    meta["original_codex56sol"] = {"call_journal": journal, "graph_proof": {"cast": len(score.cast), "scenes": len(score.scenes), "shots": len(score.shots), "beats": len(score.beats), "lines": len(manifest.lines)}, "phase_10_verdict": "accepted", "listener_report": listener.model_dump(mode="json"), "final_audit": audit.model_dump(mode="json")}
    stamp_receipt(led.data, owner_bank="original_codex56sol", accepted_artifacts={"performance_script": script, "closed_manifest": manifest})
    canon = EpisodeCanon(title=script.title, premise=score.premise, setting=score.setting, time_of_day="", sound_palette=[])
    return OriginalCodex56SolTailParts(SimpleNamespace(title=script.title, premise=score.premise, setting=score.setting, time_of_day=""), canon, script.title, False, OriginalCodex56SolFinalizer())
