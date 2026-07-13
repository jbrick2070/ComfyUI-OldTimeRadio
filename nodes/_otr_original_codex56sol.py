"""Strict original Lost and Found Frequency source-bank runner."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import unicodedata
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


log = logging.getLogger(__name__)


_SCORE_TOPOLOGY_ERROR = "beats for each shot must form one contiguous block"
_SCORE_DUPLICATE_CLUE_ERROR = (
    "each truth-map clue must be assigned to exactly one line intent"
)


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
    device_spoken_anchor: str
    resolution_spoken_anchor: str


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


class GroundingClue(StrictModel):
    clue_id: Identifier
    thread_id: Identifier
    lost_object: str
    sound_or_phrase: str
    implication: str


class GroundingContract(StrictModel):
    schema_version: Literal["original_codex56sol.grounding.v1"]
    constraint_id: str
    lost_object_anchors: list[str] = Field(min_length=3, max_length=6)
    device_anchor: str
    resolution_anchor: str
    expected_cause: str
    expected_resolution: str
    clues: list[GroundingClue] = Field(min_length=3)


class FairPlayFinding(StrictModel):
    field_path: str = ""
    item_id: str = ""
    category: str
    detail: str
    blocking: bool


class FairPlayReport(StrictModel):
    accepted: bool
    findings: list[FairPlayFinding]
    # The seam has always promised "taste notes are warnings", but StrictModel
    # forbids extra keys -- so a model that obeyed the prompt literally was
    # rejected for it.  Defaulted, never required: every existing report that
    # omits warnings stays valid.
    warnings: list[str] = Field(default_factory=list)


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


class ScoreIntentReplacement(StrictModel):
    beat_id: Identifier
    intent: str


class ScoreIntentPatch(StrictModel):
    replacements: list[ScoreIntentReplacement] = Field(min_length=1, max_length=6)


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


class ScriptLineReplacement(StrictModel):
    line_id: str
    text: str


class ScriptLinePatch(StrictModel):
    replacements: list[ScriptLineReplacement] = Field(min_length=1, max_length=8)


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
    if pass_id == "P1":
        rules += (
            " Return 4 to 6 possibilities and never delete one to repair "
            "another. Every possibility MUST copy the ingress lost_objects "
            "array and acoustic_device string verbatim. clue_plan MUST carry "
            "one distinct audible clue for EVERY lost object in that array, in "
            "the same order, so three lost objects require at least three "
            "clue_plan entries. Never merge two objects into one clue and "
            "never drop a lost object's clue: if a clue is missing, author the "
            "missing clue for the uncovered object and keep every existing "
            "clue as written."
        )
    if pass_id == "P2":
        rules += (
            " selected_possibility_id MUST be one possibility_id copied "
            "verbatim from the supplied slate: never a title, an index, or an "
            "invented id. NEVER mark the possibility you select as blocking. If "
            "every candidate carries a concern, select the least compromised one "
            "and record its concern with blocking=false. blocking MUST be a "
            "boolean, and every finding must name a concrete field."
        )
    if pass_id in {"P4", "P4_rerun"}:
        rules += (
            " Return only the compact audit envelope. accepted MUST be one "
            "boolean, findings MUST be a list, and warnings MUST be a list of "
            "strings. A finding may set blocking=true ONLY when it names a "
            "concrete truth-map item: field_path MUST start with the collection "
            "that owns the item (caller_threads, causal_steps, audible_clues, "
            "interpretations, or resolution_links) and item_id MUST be that "
            "item's exact id from the supplied truth_map. Taste, tone, warmth, "
            "and style remarks are NOT blocking: put them in warnings or set "
            "blocking=false. accepted MUST be false when a blocking finding "
            "exists and true otherwise."
        )
    if pass_id in {"P3", "P3_rerun"}:
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
            "earlier shot_id. The beats array is chronological and MUST "
            "never be reordered. If the camera returns to an earlier setup, "
            "create a new shot row with a new unique shot_id and assign that "
            "later contiguous beat run to it. Preserve "
            "orientation-before-reveal-before-closure. "
            "Never emit schema-path pseudo-fields such as `scenes[*].env` or "
            "`shots[*].visual_prompt` at the top level. "
            "scenes, shots, and beats MUST be separate top-level arrays. "
            "Never place shots inside scenes or beats inside shots. "
             "Every line_intent MUST "
             "have exactly the keys intent, arc_phase, and clue_ids. clue_ids "
             "MUST always be an array of existing clue IDs, or [] when there "
             "is no clue; singular clue_id is forbidden. Assign every truth-map "
             "clue ID exactly once across all line_intents; if a clue appears "
             "twice, retain its first authored placement and remove only its "
             "later duplicate reference. Cast MUST include one "
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
            "closure, climax, or other synonyms as arc_phase values. "
            "Use grounding_contract: each lost-object anchor MUST appear "
            "verbatim in at least one non-announcer clue-carrying intent for "
            "its thread, device_anchor MUST appear in the reveal intent, and "
            "resolution_anchor MUST appear in the closure intent."
        )
    if pass_id in {"P7", "P7_rerun"}:
        rules += (
            " understood_cause and understood_resolution MUST be strings; "
            "findings MUST be a list; optional_notes MUST be a list of "
            "strings, or [] when there are no optional notes."
        )
    if pass_id in {"P6", "P8", "P8_optional", "P9_retake"}:
        rules += (
            " Preserve every immutable spoken anchor in grounding_contract. "
            "Each lost-object anchor must be spoken on a line carrying a clue "
            "for that object's thread; the exact device_anchor must be spoken "
            "on the manifest reveal line; the exact resolution_anchor must be "
            "spoken on the manifest closure line. Never replace the mundane "
            "lost-and-found mechanism with speculative technology, an ancient "
            "artifact, a crime, or a supernatural cause."
        )
    if pass_id in {"P9", "P9_rerun"}:
        rules += (
            " Return only the compact audit envelope. accepted MUST be one "
            "boolean, findings MUST be a list, and warnings MUST be a list of "
            "strings. Never copy the manifest or script into accepted or any "
            "other output field. Every finding MUST include field_path, "
            "item_id, exact_span, category, allowed_correction, and blocking; "
            "blocking MUST be a boolean. "
            "You audit the spoken script only. The manifest, truth_map, and "
            "grounding_contract are already-accepted read-only evidence: the "
            "only correction this pass can order is a retake of a spoken line, "
            "so a defect you cannot locate in spoken text cannot block. Set "
            "blocking=true ONLY on a finding whose item_id is exactly one "
            "line_id from script.lines and whose exact_span is one string "
            "copied verbatim from that same line's text. exact_span MUST be a "
            "single quoted substring, never an array, an offset pair, a "
            "character range, or a paraphrase. Report any concern about the "
            "manifest, truth_map, grounding_contract, or clue bookkeeping in "
            "warnings, or as a finding with blocking=false. accepted MUST be "
            "false when at least one blocking finding exists and true "
            "otherwise. If no spoken line is defective, return accepted=true "
            "with findings=[] and put any remark in warnings."
        )
    return rules


def _project_duplicate_score_clues(
    score: BroadcastScore,
    truth: AudibleTruthMap,
) -> BroadcastScore | None:
    """Project duplicate clue references without judging the full graph.

    This repair is safe only when the score already covers the exact truth-map
    clue set. The first authored placement wins; later duplicate references are
    removed in beat order. A caller must run the complete score validator after
    applying this narrow projection; missing or unknown clues remain LLM
    failures.
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


def _repair_duplicate_score_clues(
    score: BroadcastScore,
    truth: AudibleTruthMap,
    grounding_contract: GroundingContract | None = None,
) -> BroadcastScore | None:
    """Return only a fully valid duplicate-clue projection."""
    repaired = _project_duplicate_score_clues(score, truth)
    if repaired is None:
        return None
    if _validate_score(repaired, truth, grounding_contract) is not None:
        return None
    return repaired


def _project_score_beat_topology(score: BroadcastScore) -> BroadcastScore | None:
    """Project reopened shot runs without judging the full graph.

    ``beats`` is chronological and compiles directly into spoken-line order,
    so a structural repair must never sort it.  For a model topology such as
    A, B, A, clone A's mechanical shot row under a collision-safe ID and retag
    only the later A run.  Beat order, IDs, prose, speakers, clues, phases, and
    landmark order remain unchanged. A caller must run full graph validation
    after this narrow projection.
    """
    shot_by_id = {shot.shot_id: shot for shot in score.shots}
    if len(shot_by_id) != len(score.shots):
        return None
    if any(beat.shot_id not in shot_by_id for beat in score.beats):
        return None

    repaired = score.model_copy(deep=True)
    repaired_shot_by_id = {
        shot.shot_id: shot for shot in repaired.shots
    }
    used_ids = set(repaired_shot_by_id)
    run_counts: Counter[str] = Counter()
    cloned_shots: list[ShotConcept] = []
    previous_source_id: str | None = None
    active_target_id: str | None = None
    changed = False

    for beat in repaired.beats:
        source_id = beat.shot_id
        if source_id != previous_source_id:
            run_counts[source_id] += 1
            if run_counts[source_id] == 1:
                active_target_id = source_id
            else:
                base_id = f"{source_id}_return_{run_counts[source_id]}"
                target_id = base_id
                collision = 2
                while target_id in used_ids:
                    target_id = f"{base_id}_{collision}"
                    collision += 1
                used_ids.add(target_id)
                clone = repaired_shot_by_id[source_id].model_copy(deep=True)
                clone.shot_id = target_id
                cloned_shots.append(clone)
                active_target_id = target_id
                changed = True
            previous_source_id = source_id
        if active_target_id is None:
            return None
        beat.shot_id = active_target_id

    if not changed:
        return None
    repaired.shots.extend(cloned_shots)
    return repaired


def _repair_score_beat_topology(
    score: BroadcastScore,
    truth: AudibleTruthMap,
    grounding_contract: GroundingContract | None = None,
) -> BroadcastScore | None:
    """Return only a fully valid reopened-shot projection."""
    repaired = _project_score_beat_topology(score)
    if repaired is None:
        return None
    if _validate_score(repaired, truth, grounding_contract) is not None:
        return None
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
    """Normalize non-authoritative score structure without rewriting values.

    The score owns music descriptions/prompts, never filenames or paths.  A
    local model may still echo a plausible ``music_file`` bookkeeping field;
    deleting any extra bookend key is the same structural projection as
    lifting nested collections below. Required authored fields are untouched,
    and the complete repaired score must still pass the strict model and graph
    validator before it can be accepted without another LLM call.
    """
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
    bookend_keys = set(MusicBookend.model_fields)
    for field_name in ("opening_music", "closing_music"):
        bookend = data.get(field_name)
        if not isinstance(bookend, dict):
            return None
        for extra_key in sorted(set(bookend) - bookend_keys):
            del bookend[extra_key]
            changed = True
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

    try:
        repaired = BroadcastScore.model_validate(data)
    except ValidationError:
        return None
    topology_repair = _repair_score_beat_topology(repaired, truth)
    if topology_repair is not None:
        repaired = topology_repair
        changed = True
    if not changed:
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

    def normalize_attempt_output(raw: str) -> str:
        """Apply narrow deterministic repairs to every ladder response.

        A repair-prompt factory sees only the response that *triggered* the
        typed-repair rung.  The typed-repair response itself normally goes
        straight to schema/content validation, so a local model can repeat a
        safe mechanical defect there and bypass the projection.  Normalize at
        the slot boundary instead, while retaining the hash of the actual raw
        model output.  Full post-validation remains the acceptance authority.
        """
        repaired: BaseModel | None = None
        if pass_id in {"P3", "P3_rerun"} and schema is AudibleTruthMap:
            try:
                selected = PossibilityCard.model_validate(inputs["selected"])
            except (KeyError, TypeError, ValidationError):
                selected = None
            if selected is not None:
                repaired = _repair_truth_map_collection_placement(
                    raw, selected,
                )
        elif pass_id == "P5" and schema is BroadcastScore:
            try:
                truth = AudibleTruthMap.model_validate(inputs["truth_map"])
            except (KeyError, TypeError, ValidationError):
                truth = None
            if truth is not None:
                repaired = _repair_score_collection_placement(raw, truth)
        if repaired is None or post_validator(repaired) is not None:
            return raw
        log.info(
            "[original_codex56sol] %s normalized one safe structural "
            "defect at the per-attempt boundary",
            pass_id,
        )
        return repaired.model_dump_json()

    def capture(messages, **kwargs):
        raw = fn(messages, **kwargs)
        attempts.append(hashlib.sha256(str(raw).encode()).hexdigest())
        return normalize_attempt_output(raw)
    def repair(*, original_prompt, failed_output, error):
        effective_error: Any = error
        if pass_id in {"P3", "P3_rerun"} and schema is AudibleTruthMap:
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
        pass_rules = _repair_rules(pass_id, effective_error)
        artifact_instruction = (
            "Return the same complete artifact, repairing only the typed "
            "contract error."
        )
        if pass_id.endswith("_grounding_patch"):
            artifact_instruction = (
                f"Return only the complete {schema.__name__} for the supplied "
                "targets; do not emit a full artifact or any other object."
            )
        return [
            {"role": "system", "content": system + "\n" + artifact_instruction + pass_rules + " JSON only.\n" + schema_shape_instruction(schema)},
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
        # Every lost object must reach the listener as its own audible clue.
        # The count is derived from the accepted draw, so Python can prove the
        # shortfall -- but a clue is authored story, so a missing one returns to
        # the model rather than being invented here.
        if len(card.clue_plan) < len(draw.lost_objects):
            return (
                f"{card.possibility_id}: clue_plan must carry one distinct "
                f"audible clue for each of the {len(draw.lost_objects)} lost "
                f"objects, but it has {len(card.clue_plan)}; author the missing "
                "clue instead of merging two objects into one"
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


def _build_grounding_contract(
    draw: ConstraintDraw,
    truth: AudibleTruthMap,
) -> GroundingContract:
    """Close the immutable draw over the truth map before dialogue exists."""
    truth_error = _validate_truth_map(truth)
    if truth_error is not None:
        raise OriginalCodex56SolContractError(
            f"cannot build grounding contract: {truth_error}"
        )
    thread_objects = {
        row.thread_id: row.lost_object for row in truth.caller_threads
    }
    if Counter(thread_objects.values()) != Counter(draw.lost_objects):
        raise OriginalCodex56SolContractError(
            "cannot build grounding contract: truth-map lost objects differ "
            "from the immutable draw"
        )
    clues = []
    for clue in truth.audible_clues:
        lost_object = thread_objects.get(clue.thread_id)
        if lost_object is None:
            raise OriginalCodex56SolContractError(
                f"cannot build grounding contract: clue {clue.clue_id!r} "
                f"references unknown thread {clue.thread_id!r}"
            )
        clues.append(GroundingClue(
            clue_id=clue.clue_id,
            thread_id=clue.thread_id,
            lost_object=lost_object,
            sound_or_phrase=clue.sound_or_phrase,
            implication=clue.implication,
        ))
    return GroundingContract(
        schema_version="original_codex56sol.grounding.v1",
        constraint_id=draw.constraint_id,
        lost_object_anchors=list(draw.lost_objects),
        device_anchor=draw.device_spoken_anchor,
        resolution_anchor=draw.resolution_spoken_anchor,
        expected_cause=draw.acoustic_device,
        expected_resolution=draw.helpful_ending,
        clues=clues,
    )


def _validate_score(
    score: BroadcastScore,
    truth: AudibleTruthMap,
    grounding_contract: GroundingContract | None = None,
) -> str | None:
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
                return _SCORE_TOPOLOGY_ERROR
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
        return _SCORE_DUPLICATE_CLUE_ERROR
    if grounding_contract is not None:
        clue_ids_by_object: dict[str, set[str]] = {}
        for clue in grounding_contract.clues:
            clue_ids_by_object.setdefault(
                clue.lost_object, set()
            ).add(clue.clue_id)
        for anchor in grounding_contract.lost_object_anchors:
            clue_ids = clue_ids_by_object.get(anchor, set())
            if not any(
                clue_ids.intersection(beat.line_intent.clue_ids)
                and _contains_grounding_anchor(
                    beat.line_intent.intent, anchor,
                )
                for beat in score.beats
                if beat.char_id != "announcer"
            ):
                return (
                    f"score needs a non-announcer clue intent naming exact "
                    f"lost-object anchor {anchor!r}"
                )
        if not _contains_grounding_anchor(
            by_beat[score.reveal_beat_id].line_intent.intent,
            grounding_contract.device_anchor,
        ):
            return (
                "reveal intent must name exact device anchor "
                f"{grounding_contract.device_anchor!r}"
            )
        if not _contains_grounding_anchor(
            by_beat[score.closure_beat_id].line_intent.intent,
            grounding_contract.resolution_anchor,
        ):
            return (
                "closure intent must name exact resolution anchor "
                f"{grounding_contract.resolution_anchor!r}"
            )
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


def _validate_score_attempt(
    score: BroadcastScore,
    truth: AudibleTruthMap,
    grounding_contract: GroundingContract | None,
    story_rules: Any,
) -> str | None:
    """Validate every P5 response at the exact structured-call boundary.

    P5 base and typed-repair outputs first become a strict ``BroadcastScore``
    inside ``structured_call``.  The contiguous-shot projection belongs here,
    not only in the raw-string normalization path: this is the one boundary
    every schema-valid ladder response must cross.  The projection changes
    only mechanical shot ownership and is accepted only when the complete
    requested score contract validates again.  P5 first accepts the structural
    score here; localized immutable-anchor omissions are handled immediately
    afterward by the bounded ``ScoreIntentPatch`` tool.
    """
    repaired_kinds: list[str] = []
    for _ in range(2):
        structural_error = _validate_score(score, truth, grounding_contract)
        repaired: BroadcastScore | None = None
        repaired_kind = ""
        if structural_error == _SCORE_TOPOLOGY_ERROR:
            repaired = _project_score_beat_topology(score)
            repaired_kind = "shot topology"
        elif structural_error == _SCORE_DUPLICATE_CLUE_ERROR:
            repaired = _project_duplicate_score_clues(score, truth)
            repaired_kind = "duplicate clue ownership"
        if repaired is None:
            break
        score.shots = repaired.shots
        score.beats = repaired.beats
        repaired_kinds.append(repaired_kind)
    structural_error = _validate_score(score, truth, grounding_contract)
    if repaired_kinds and structural_error is None:
        log.info(
            "[original_codex56sol] P5 normalized safe %s at the "
            "schema-validated attempt boundary",
            " + ".join(repaired_kinds),
        )
    if structural_error is not None:
        return structural_error
    return _validate_authored_surface(score, story_rules)


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


def _fair_play_advisories(report: FairPlayReport,
                          truth: AudibleTruthMap) -> list[FairPlayFinding]:
    """Blocking findings that name no truth-map item, kept as advice.

    A fair-play opinion with no coordinates ("the ending could be warmer") is a
    taste note the model mislabelled as blocking.  It cannot be acted on and it
    must not be fatal -- Python demotes it and records it verbatim.  The
    classification is mechanical: the item_id either resolves in a truth-map
    collection or it does not.
    """
    ids = _truth_item_ids(truth)
    resolvable = {item for collection in ids.values() for item in collection}
    return [finding for finding in report.findings
            if finding.blocking and finding.item_id.strip() not in resolvable]


def _validate_fair_play_envelope(report: FairPlayReport,
                                 truth: AudibleTruthMap) -> str | None:
    """Hold the fair-play verdict to coordinates Python can act on.

    Unlike the P9 audit -- whose target, the manifest, is Python-compiled and
    unfixable by any retake -- the truth map is authored by P3.  A fair-play
    rejection is therefore legitimate and gets a repair route, not a demotion.
    Python only insists that a blocking verdict either names a real truth-map
    item or is honest about naming none.

    Ambiguity fails closed: an item_id that resolves under a DIFFERENT
    field_path root leaves Python unable to tell which item is meant, so the
    defect returns to the owning model rather than being guessed at.
    """
    ids = _truth_item_ids(truth)
    resolvable = {item for collection in ids.values() for item in collection}
    blocking = [finding for finding in report.findings if finding.blocking]
    if report.accepted and blocking:
        return (
            "an accepted fair-play report must not carry a blocking finding; "
            "set accepted=false or make the finding non-blocking"
        )
    if not report.accepted and not blocking:
        return (
            "a rejected fair-play report must carry at least one blocking "
            "finding; return accepted=true and use warnings when nothing blocks"
        )
    for finding in blocking:
        item_id = finding.item_id.strip()
        if item_id not in resolvable:
            continue  # uncoordinated taste note -> demoted, never fatal
        root = finding.field_path.strip().split(".", 1)[0]
        if item_id not in ids.get(root, set()):
            return (
                f"blocking finding for item {item_id!r} sets field_path root "
                f"{root!r}, which does not own that item; name the collection "
                "that actually contains it"
            )
        if not (finding.category.strip() and finding.detail.strip()):
            return (
                f"blocking finding for item {item_id!r} must include a "
                "category and a detail"
            )
    return None


def _validate_score_clue_ownership(
    score: BroadcastScore,
    grounding: GroundingContract,
) -> str | None:
    """Every lost object must reach a listener through a non-announcer voice.

    OWNERSHIP ONLY -- never anchor text.  The bounded intent patch can rewrite a
    beat's intent prose, so a missing anchor WORD stays outside the ladder and
    is repaired there (see `_repair_score_grounding_intents`).  But the patch is
    forbidden from touching `clue_ids`, so if the announcer is the only voice
    carrying an object's clue, no patch can fix it and the score itself must be
    re-authored.  That is what this check returns to the P5 ladder.
    """
    clue_ids_by_object: dict[str, set[str]] = {}
    for clue in grounding.clues:
        clue_ids_by_object.setdefault(clue.lost_object, set()).add(clue.clue_id)
    for anchor in grounding.lost_object_anchors:
        clue_ids = clue_ids_by_object.get(anchor, set())
        if not any(
            clue_ids.intersection(beat.line_intent.clue_ids)
            for beat in score.beats
            if beat.char_id != "announcer"
        ):
            return (
                f"a non-announcer beat must carry an audible clue for lost "
                f"object {anchor!r}; the announcer may not be the only voice "
                "that holds it"
            )
    return None


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


def _normalize_grounding_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", normalized).split())


def _contains_grounding_anchor(text: str, anchor: str) -> bool:
    haystack = _normalize_grounding_text(text)
    needle = _normalize_grounding_text(anchor)
    return bool(needle) and f" {needle} " in f" {haystack} "


def _score_grounding_repair_plan(
    score: BroadcastScore,
    grounding: GroundingContract,
) -> list[dict[str, Any]] | None:
    """Select only LLM-owned intents that lack immutable grounding anchors."""
    targets: dict[str, set[str]] = {}
    clue_ids_by_object: dict[str, set[str]] = {}
    for clue in grounding.clues:
        clue_ids_by_object.setdefault(clue.lost_object, set()).add(clue.clue_id)
    for anchor in grounding.lost_object_anchors:
        relevant_clues = clue_ids_by_object.get(anchor, set())
        eligible = [
            beat for beat in score.beats
            if beat.char_id != "announcer"
            and relevant_clues.intersection(beat.line_intent.clue_ids)
        ]
        if not eligible:
            return None
        if not any(_contains_grounding_anchor(
                beat.line_intent.intent, anchor) for beat in eligible):
            targets.setdefault(eligible[0].beat_id, set()).add(anchor)

    by_id = {beat.beat_id: beat for beat in score.beats}
    reveal = by_id.get(score.reveal_beat_id)
    closure = by_id.get(score.closure_beat_id)
    if reveal is None or closure is None:
        return None
    if not _contains_grounding_anchor(
            reveal.line_intent.intent, grounding.device_anchor):
        targets.setdefault(reveal.beat_id, set()).add(grounding.device_anchor)
    if not _contains_grounding_anchor(
            closure.line_intent.intent, grounding.resolution_anchor):
        targets.setdefault(closure.beat_id, set()).add(
            grounding.resolution_anchor,
        )

    # A target can be a reveal or closure beat (or otherwise already carry a
    # different valid immutable anchor).  Replacing its intent must never
    # erase an anchor that the accepted score already relies on.
    protected_anchors = (
        *grounding.lost_object_anchors,
        grounding.device_anchor,
        grounding.resolution_anchor,
    )
    for beat in score.beats:
        if beat.beat_id not in targets:
            continue
        for anchor in protected_anchors:
            if _contains_grounding_anchor(beat.line_intent.intent, anchor):
                targets[beat.beat_id].add(anchor)

    return [
        {
            "beat_id": beat.beat_id,
            "current_intent": beat.line_intent.intent,
            "required_anchors": sorted(targets[beat.beat_id]),
        }
        for beat in score.beats if beat.beat_id in targets
    ]


def _validate_score_intent_patch(
    patch: ScoreIntentPatch,
    plan: list[dict[str, Any]],
) -> str | None:
    expected = {str(row["beat_id"]): row for row in plan}
    seen: set[str] = set()
    for replacement in patch.replacements:
        beat_id = replacement.beat_id
        if beat_id not in expected or beat_id in seen:
            return "score intent patch must replace every and only planned beat ids once"
        seen.add(beat_id)
        for anchor in expected[beat_id]["required_anchors"]:
            if not _contains_grounding_anchor(replacement.intent, anchor):
                return (
                    "score intent patch must name every required immutable "
                    f"anchor for beat {beat_id}"
                )
    if seen != set(expected):
        return "score intent patch must replace every planned beat id"
    return None


def _merge_score_intent_patch(
    score: BroadcastScore,
    patch: ScoreIntentPatch,
) -> BroadcastScore:
    intents = {row.beat_id: row.intent for row in patch.replacements}
    repaired = score.model_copy(deep=True)
    for beat in repaired.beats:
        if beat.beat_id in intents:
            beat.line_intent.intent = intents[beat.beat_id]
    return repaired


def _validate_score_intent_patch_application(
    patch: ScoreIntentPatch,
    score: BroadcastScore,
    plan: list[dict[str, Any]],
    truth: AudibleTruthMap,
    grounding: GroundingContract,
    story_rules: Any,
) -> str | None:
    patch_error = _validate_score_intent_patch(patch, plan)
    if patch_error is not None:
        return patch_error
    repaired = _merge_score_intent_patch(score, patch)
    score_error = _validate_score(repaired, truth, grounding)
    if score_error is not None:
        return f"score intent patch leaves the full score invalid: {score_error}"
    authored_error = _validate_authored_surface(repaired, story_rules)
    if authored_error is not None:
        return (
            "score intent patch leaves a forbidden authored surface: "
            f"{authored_error}"
        )
    return None


def _apply_score_intent_patch(
    score: BroadcastScore,
    patch: ScoreIntentPatch,
    plan: list[dict[str, Any]],
    truth: AudibleTruthMap,
    grounding: GroundingContract,
    story_rules: Any,
) -> BroadcastScore | None:
    if _validate_score_intent_patch_application(
            patch, score, plan, truth, grounding, story_rules) is not None:
        return None
    return _merge_score_intent_patch(score, patch)


def _repair_score_grounding_intents(
    *, score: BroadcastScore, truth: AudibleTruthMap,
    grounding: GroundingContract, pack: Any, creative_fn: GenerateFn,
    scheduler: Any, journal: list[dict], story_rules: Any,
) -> BroadcastScore:
    """Use a small LLM tool call for missing immutable P5 intent anchors."""
    plan = _score_grounding_repair_plan(score, grounding)
    if not plan:
        raise OriginalCodex56SolContractError(
            "P5 grounding repair has no eligible immutable-anchor plan"
        )
    patch = _call(
        pass_id="P5_grounding_patch", slot="creative", fn=creative_fn,
        pack=pack, seam="codex56_score_anchor_patch",
        inputs={"targets": plan}, schema=ScoreIntentPatch,
        scheduler=scheduler, journal=journal, tokens=900,
        post_validator=lambda value: _validate_score_intent_patch_application(
            value, score, plan, truth, grounding, story_rules,
        ),
    )
    repaired = _apply_score_intent_patch(
        score, patch, plan, truth, grounding, story_rules,
    )
    if repaired is None:
        raise OriginalCodex56SolContractError(
            "P5 grounding patch did not clear the full score contract"
        )
    return repaired


def _raw_anchor_span(text: str, anchor: str) -> str | None:
    tokens = _normalize_grounding_text(anchor).split()
    if not tokens:
        return None
    pattern = re.compile(
        r"(?<!\w)" + r"[\W_]+".join(re.escape(token) for token in tokens)
        + r"(?!\w)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    return match.group(0) if match else None


def _validate_script_grounding(
    contract: GroundingContract,
    manifest: ClosedLineManifest,
    script: PerformanceScript,
) -> str | None:
    """Prove opaque clue IDs still have audible story content."""
    script_by_id = {line.line_id: line for line in script.lines}
    full_script = "\n".join(line.text for line in script.lines)
    for anchor in contract.lost_object_anchors:
        if not _contains_grounding_anchor(full_script, anchor):
            return f"spoken script is missing lost-object anchor {anchor!r}"

    clue_ids_by_object: dict[str, set[str]] = {
        anchor: set() for anchor in contract.lost_object_anchors
    }
    for clue in contract.clues:
        clue_ids_by_object.setdefault(clue.lost_object, set()).add(clue.clue_id)
    for anchor in contract.lost_object_anchors:
        clue_ids = clue_ids_by_object.get(anchor, set())
        eligible_line_ids = [
            row.line_id for row in manifest.lines
            if clue_ids.intersection(row.clue_ids)
        ]
        if not eligible_line_ids:
            return (
                f"manifest has no clue-carrying line for lost-object anchor "
                f"{anchor!r}"
            )
        if not any(
            line_id in script_by_id
            and _contains_grounding_anchor(script_by_id[line_id].text, anchor)
            for line_id in eligible_line_ids
        ):
            return (
                f"lost-object anchor {anchor!r} is not spoken on any line "
                "carrying a clue for its thread"
            )

    reveal = script_by_id.get(manifest.reveal_line_id)
    if reveal is None or not _contains_grounding_anchor(
        reveal.text, contract.device_anchor,
    ):
        return (
            f"reveal line must speak exact device anchor "
            f"{contract.device_anchor!r}"
        )
    closure = script_by_id.get(manifest.closure_line_id)
    if closure is None or not _contains_grounding_anchor(
        closure.text, contract.resolution_anchor,
    ):
        return (
            f"closure line must speak exact resolution anchor "
            f"{contract.resolution_anchor!r}"
        )
    return None


def _script_grounding_repair_plan(
    script: PerformanceScript,
    manifest: ClosedLineManifest,
    grounding: GroundingContract,
) -> list[dict[str, Any]] | None:
    """Select only spoken lines that need immutable grounding literals."""
    script_by_id = {line.line_id: line for line in script.lines}
    targets: dict[str, set[str]] = {}
    clue_ids_by_object: dict[str, set[str]] = {}
    for clue in grounding.clues:
        clue_ids_by_object.setdefault(clue.lost_object, set()).add(clue.clue_id)
    for anchor in grounding.lost_object_anchors:
        clue_ids = clue_ids_by_object.get(anchor, set())
        eligible = [
            row for row in manifest.lines
            if clue_ids.intersection(row.clue_ids)
        ]
        if not eligible:
            return None
        if not any(
            row.line_id in script_by_id
            and _contains_grounding_anchor(
                script_by_id[row.line_id].text, anchor,
            )
            for row in eligible
        ):
            targets.setdefault(eligible[0].line_id, set()).add(anchor)

    reveal = script_by_id.get(manifest.reveal_line_id)
    closure = script_by_id.get(manifest.closure_line_id)
    if reveal is None or closure is None:
        return None
    if not _contains_grounding_anchor(reveal.text, grounding.device_anchor):
        targets.setdefault(reveal.line_id, set()).add(grounding.device_anchor)
    if not _contains_grounding_anchor(
            closure.text, grounding.resolution_anchor):
        targets.setdefault(closure.line_id, set()).add(
            grounding.resolution_anchor,
        )

    # A line can carry several valid story facts.  A replacement must retain
    # every immutable fact it already speaks, not merely the missing one.
    protected_anchors = (
        *grounding.lost_object_anchors,
        grounding.device_anchor,
        grounding.resolution_anchor,
    )
    for line_id in targets:
        line = script_by_id.get(line_id)
        if line is None:
            return None
        for anchor in protected_anchors:
            if _contains_grounding_anchor(line.text, anchor):
                targets[line_id].add(anchor)

    return [
        {
            "line_id": row.line_id,
            "current_text": script_by_id[row.line_id].text,
            "required_anchors": sorted(targets[row.line_id]),
        }
        for row in manifest.lines if row.line_id in targets
    ]


def _validate_script_line_patch(
    patch: ScriptLinePatch,
    plan: list[dict[str, Any]],
) -> str | None:
    expected = {str(row["line_id"]): row for row in plan}
    seen: set[str] = set()
    for replacement in patch.replacements:
        line_id = replacement.line_id
        if line_id not in expected or line_id in seen:
            return "script line patch must replace every and only planned line ids once"
        seen.add(line_id)
        for anchor in expected[line_id]["required_anchors"]:
            if not _contains_grounding_anchor(replacement.text, anchor):
                return (
                    "script line patch must speak every required immutable "
                    f"anchor for line {line_id}"
                )
    if seen != set(expected):
        return "script line patch must replace every planned line id"
    return None


def _merge_script_line_patch(
    script: PerformanceScript,
    patch: ScriptLinePatch,
) -> PerformanceScript:
    texts = {row.line_id: row.text for row in patch.replacements}
    repaired = script.model_copy(deep=True)
    for line in repaired.lines:
        if line.line_id in texts:
            line.text = texts[line.line_id]
    return repaired


def _validate_script_line_patch_application(
    patch: ScriptLinePatch,
    script: PerformanceScript,
    plan: list[dict[str, Any]],
    score: BroadcastScore,
    manifest: ClosedLineManifest,
    grounding: GroundingContract,
    story_rules: Any,
) -> str | None:
    patch_error = _validate_script_line_patch(patch, plan)
    if patch_error is not None:
        return patch_error
    repaired = _merge_script_line_patch(script, patch)
    script_error = _validate_script(
        score, manifest, repaired, story_rules, grounding,
    )
    if script_error is not None:
        return f"script line patch leaves the full script invalid: {script_error}"
    return None


def _repair_script_grounding_lines(
    *, script: PerformanceScript, score: BroadcastScore,
    manifest: ClosedLineManifest, grounding: GroundingContract, pack: Any,
    creative_fn: GenerateFn, scheduler: Any, journal: list[dict],
    story_rules: Any, origin_pass_id: str,
) -> PerformanceScript:
    """Use a small LLM tool call for a localized script grounding omission."""
    plan = _script_grounding_repair_plan(script, manifest, grounding)
    if not plan:
        raise OriginalCodex56SolContractError(
            f"{origin_pass_id} grounding repair has no eligible immutable-anchor plan"
        )
    patch = _call(
        pass_id=f"{origin_pass_id}_grounding_patch", slot="creative",
        fn=creative_fn,
        pack=pack, seam="codex56_script_anchor_patch",
        inputs={"targets": plan}, schema=ScriptLinePatch,
        scheduler=scheduler, journal=journal, tokens=900,
        post_validator=lambda value: _validate_script_line_patch_application(
            value, script, plan, score, manifest, grounding, story_rules,
        ),
    )
    return _merge_script_line_patch(script, patch)


def _call_grounded_script(
    *, pass_id: str, fn: GenerateFn, pack: Any, seam: str,
    inputs: Mapping[str, Any], score: BroadcastScore,
    manifest: ClosedLineManifest, grounding: GroundingContract,
    scheduler: Any, journal: list[dict], story_rules: Any, tokens: int,
) -> PerformanceScript:
    """Accept structural scripts first, then patch localized grounding gaps."""
    script = _call(
        pass_id=pass_id, slot="creative", fn=fn, pack=pack, seam=seam,
        inputs=inputs, schema=PerformanceScript, scheduler=scheduler,
        journal=journal, tokens=tokens,
        post_validator=lambda value: _validate_script(
            score, manifest, value, story_rules,
        ),
    )
    if _validate_script(score, manifest, script, story_rules, grounding) is not None:
        script = _repair_script_grounding_lines(
            script=script, score=score, manifest=manifest, grounding=grounding,
            pack=pack, creative_fn=fn, scheduler=scheduler, journal=journal,
            story_rules=story_rules, origin_pass_id=pass_id,
        )
    _assert_script_valid(score, manifest, script, story_rules, grounding)
    return script


def _grounding_receipt(
    contract: GroundingContract,
    manifest: ClosedLineManifest,
    script: PerformanceScript,
) -> dict[str, Any]:
    error = _validate_script_grounding(contract, manifest, script)
    if error is not None:
        raise OriginalCodex56SolContractError(error)
    script_by_id = {line.line_id: line for line in script.lines}
    clue_ids_by_object: dict[str, set[str]] = {}
    for clue in contract.clues:
        clue_ids_by_object.setdefault(clue.lost_object, set()).add(clue.clue_id)

    evidence = []
    for anchor in contract.lost_object_anchors:
        clue_ids = clue_ids_by_object[anchor]
        line_id = next(
            row.line_id for row in manifest.lines
            if clue_ids.intersection(row.clue_ids)
            and _contains_grounding_anchor(script_by_id[row.line_id].text, anchor)
        )
        text = script_by_id[line_id].text
        evidence.append({
            "kind": "lost_object", "anchor": anchor, "line_id": line_id,
            "exact_span": _raw_anchor_span(text, anchor) or text,
        })
    for kind, anchor, line_id in (
        ("device", contract.device_anchor, manifest.reveal_line_id),
        ("resolution", contract.resolution_anchor, manifest.closure_line_id),
    ):
        text = script_by_id[line_id].text
        evidence.append({
            "kind": kind, "anchor": anchor, "line_id": line_id,
            "exact_span": _raw_anchor_span(text, anchor) or text,
        })
    return {
        "schema_version": "original_codex56sol.grounding_receipt.v1",
        "constraint_id": contract.constraint_id,
        "complete": True,
        "evidence": evidence,
    }


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


def _listener_blocks(
    report: BlindListenerReport,
    allowed_line_ids: set[str],
    contract: GroundingContract | None = None,
    fallback_line_id: str = "",
) -> list[ListenerFinding]:
    blocks = [finding for finding in report.findings
              if finding.blocking and finding.line_id in allowed_line_ids
              and finding.category.strip() and finding.detail.strip()]
    if contract is not None:
        cause = _normalize_grounding_text(report.understood_cause)
        device_tokens = [
            token for token in _normalize_grounding_text(
                contract.device_anchor
            ).split()
            if len(token) >= 4
        ]
        if device_tokens and not any(
            f" {token} " in f" {cause} " for token in device_tokens
        ):
            line_id = fallback_line_id if fallback_line_id in allowed_line_ids else ""
            if not line_id and allowed_line_ids:
                line_id = sorted(allowed_line_ids)[-1]
            if line_id and not any(row.line_id == line_id
                                   and row.category == "Cause grounding"
                                   for row in blocks):
                blocks.append(ListenerFinding(
                    line_id=line_id,
                    category="Cause grounding",
                    detail=(
                        "Blind listener could not infer the declared mundane "
                        f"device {contract.device_anchor!r} from pre-reveal clues"
                    ),
                    blocking=True,
                ))
    return blocks


def _audit_blocks(report: FinalContractAudit,
                  script: PerformanceScript) -> list[ContractFinding]:
    """Blocking findings the P9 retake can actually act on.

    The retake re-authors spoken lines, so an actionable block names one real
    script line and quotes the offending span verbatim from that line's text.
    """
    text_by_id = {line.line_id: line.text for line in script.lines}
    return [finding for finding in report.findings
            if finding.blocking and finding.item_id in text_by_id
            and finding.field_path.strip() and finding.category.strip()
            and finding.allowed_correction.strip()
            and finding.exact_span.strip()
            and finding.exact_span.strip() in text_by_id[finding.item_id]]


def _audit_advisories(report: FinalContractAudit,
                      script: PerformanceScript) -> list[ContractFinding]:
    """Findings outside the audit's blocking authority, preserved verbatim.

    P9 owns the spoken script. Its other inputs are not model-owned at this
    point: the manifest is compiled by Python from the accepted score and
    proven row-for-row by `_validate_manifest` (exact clue coverage, no
    duplicates, landmark order), while the truth map and grounding contract are
    already-accepted artifacts. A finding that names one of those instead of a
    spoken line therefore cannot be repaired by a script retake and cannot
    overrule a deterministic validator that has already proven the invariant.

    Classifying such a finding is a mechanical fact -- its item_id either is a
    script line_id or it is not -- so Python may demote it without judging any
    authored meaning. It is recorded verbatim in the run receipt as advice and
    never blocks the episode.
    """
    line_ids = {line.line_id for line in script.lines}
    return [finding for finding in report.findings
            if finding.item_id not in line_ids]


def _validate_audit_envelope(report: FinalContractAudit,
                             script: PerformanceScript) -> str | None:
    """Hold the audit to the blocking authority its repair route can serve.

    Fail closed and return the defect to the owning model when a blocking
    finding names a real script line but does not quote it -- Python cannot
    infer which spoken text the model meant, so the coordinates are ambiguous
    and must not be guessed or normalized. A rejection with no blocking finding
    at all is the same class of defect: it orders a retake it refuses to
    locate.
    """
    text_by_id = {line.line_id: line.text for line in script.lines}
    blocking = [finding for finding in report.findings if finding.blocking]
    for finding in blocking:
        if finding.item_id not in text_by_id:
            continue
        missing = [name for name in
                   ("field_path", "category", "allowed_correction")
                   if not getattr(finding, name).strip()]
        if missing:
            return (
                f"blocking finding for script line {finding.item_id!r} is "
                f"missing {', '.join(missing)}"
            )
        span = finding.exact_span.strip()
        if not span or span not in text_by_id[finding.item_id]:
            return (
                f"blocking finding for script line {finding.item_id!r} must "
                "set exact_span to one exact substring copied verbatim from "
                "that line's spoken text"
            )
    if not report.accepted and not blocking:
        return (
            "a rejected audit must carry at least one blocking finding; "
            "return accepted=true and use warnings when no spoken line is "
            "defective"
        )
    if report.accepted and blocking:
        return "an accepted audit must not carry a blocking finding"
    return None


def _validate_script(score: BroadcastScore, manifest: ClosedLineManifest,
                     script: PerformanceScript, rules: Any,
                     grounding_contract: GroundingContract | None = None,
                     ) -> str | None:
    return (_validate_graph(score, manifest, script)
            or _validate_text(script, rules)
            or (_validate_script_grounding(
                    grounding_contract, manifest, script)
                if grounding_contract is not None else None))


def _assert_script_valid(score: BroadcastScore, manifest: ClosedLineManifest,
                         script: PerformanceScript, rules: Any,
                         grounding_contract: GroundingContract | None = None,
                         ) -> None:
    error = _validate_script(
        score, manifest, script, rules, grounding_contract,
    )
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
    grounding = _build_grounding_contract(draw, truth)
    fair = _call(pass_id="P4", slot="technical", fn=technical_fn, pack=pack, seam="codex56_fair_play_audit", inputs={"truth_map": truth.model_dump(mode="json"), "grounding_contract": grounding.model_dump(mode="json")}, schema=FairPlayReport, scheduler=slot_scheduler, journal=journal, tokens=1600, post_validator=lambda value, t=truth: _validate_fair_play_envelope(value, t))
    fair_advisories = _fair_play_advisories(fair, truth)
    corroborated_fair_blocks = _corroborated_fair_blocks(fair, truth)
    truth_map_retake_ran = False
    if corroborated_fair_blocks:
        # The truth map is authored by P3, so a corroborated fair-play block is
        # actionable: return it to its owner instead of ending the episode.
        # `selected`, `draw`, and `story_rules` are held fixed -- the retake may
        # re-author the truth map, never re-select the possibility -- and the P3
        # post_validator is reused verbatim so the retaken map still proves the
        # invariants `_build_grounding_contract` depends on.
        truth_map_retake_ran = True
        truth = _call(
            pass_id="P3_rerun", slot="creative", fn=creative_fn, pack=pack,
            seam="codex56_truth_map_retake",
            inputs={
                "selected": selected.model_dump(mode="json"),
                "draw": draw.model_dump(mode="json"),
                "previous_truth_map": truth.model_dump(mode="json"),
                "fair_play_findings": [
                    finding.model_dump(mode="json")
                    for finding in corroborated_fair_blocks
                ],
            },
            schema=AudibleTruthMap, scheduler=slot_scheduler, journal=journal,
            tokens=2800,
            post_validator=lambda value, s=selected: (
                _validate_truth_map(value, s)
                or _validate_authored_surface(value, story_rules)
            ),
        )
        # The old contract was derived from the rejected truth map: rebuild it
        # before anything audits or consumes it.
        grounding = _build_grounding_contract(draw, truth)
        fair = _call(pass_id="P4_rerun", slot="technical", fn=technical_fn, pack=pack, seam="codex56_fair_play_audit", inputs={"truth_map": truth.model_dump(mode="json"), "grounding_contract": grounding.model_dump(mode="json")}, schema=FairPlayReport, scheduler=slot_scheduler, journal=journal, tokens=1600, post_validator=lambda value, t=truth: _validate_fair_play_envelope(value, t))
        if _corroborated_fair_blocks(fair, truth):
            raise OriginalCodex56SolContractError(
                "fair-play audit rejected the retaken truth map"
            )
        fair_advisories = fair_advisories + _fair_play_advisories(fair, truth)

    fair_play_disposition = {
        "schema_version": "original_codex56sol.fair_play_disposition.v1",
        "model_verdict": fair.accepted,
        "effective_verdict": ("accepted_after_truth_map_retake"
                              if truth_map_retake_ran else "accepted"),
        "truth_map_retake_ran": truth_map_retake_ran,
        "corroborated_blocking_findings": 0,
        "advisory_findings": [row.model_dump(mode="json")
                              for row in fair_advisories],
        "warnings": list(fair.warnings),
    }
    score = _call(
        pass_id="P5", slot="creative", fn=creative_fn, pack=pack,
        seam="codex56_broadcast_score",
        inputs={
            "truth_map": truth.model_dump(mode="json"),
            "grounding_contract": grounding.model_dump(mode="json"),
            "target_words_advisory": int(resolved["target_words"]),
            "num_characters_advisory": int(resolved["num_characters"]),
        },
        schema=BroadcastScore, scheduler=slot_scheduler, journal=journal,
        tokens=3600,
        # The grounding contract stays out of `_validate_score_attempt` on
        # purpose: a missing anchor WORD is a localized leaf defect owned by the
        # bounded intent patch, not a reason to regenerate a whole score.  Clue
        # OWNERSHIP is different -- the patch may not touch clue_ids -- so it is
        # the one half of the contract the ladder must see while it can still
        # re-author the score.
        post_validator=lambda value, t=truth, g=grounding: (
            _validate_score_attempt(value, t, None, story_rules)
            or _validate_score_clue_ownership(value, g)
        ),
    )
    if _validate_score(score, truth, grounding) is not None:
        score = _repair_score_grounding_intents(
            score=score, truth=truth, grounding=grounding, pack=pack,
            creative_fn=creative_fn, scheduler=slot_scheduler, journal=journal,
            story_rules=story_rules,
        )
    manifest = _compile_manifest(score)
    script = _call_grounded_script(
        pass_id="P6", fn=creative_fn, pack=pack,
        seam="codex56_performance_script",
        inputs={
            "score": score.model_dump(mode="json"),
            "manifest": manifest.model_dump(mode="json"),
            "truth_map": truth.model_dump(mode="json"),
            "grounding_contract": grounding.model_dump(mode="json"),
            "target_words_advisory": int(resolved["target_words"]),
        },
        score=score, manifest=manifest, grounding=grounding,
        scheduler=slot_scheduler, journal=journal, story_rules=story_rules,
        tokens=max(2600, int(resolved["target_words"]) * 6),
    )
    preceding_lines = _preceding_lines(manifest, script)
    listener = _call(pass_id="P7", slot="technical", fn=technical_fn, pack=pack, seam="codex56_blind_listener", inputs={"preceding_lines": preceding_lines}, schema=BlindListenerReport, scheduler=slot_scheduler, journal=journal, tokens=1800)
    listener_blocks = _listener_blocks(
        listener, {line["line_id"] for line in preceding_lines}, grounding,
        preceding_lines[-1]["line_id"],
    )
    if listener_blocks:
        script = _call_grounded_script(
            pass_id="P8", fn=creative_fn, pack=pack,
            seam="codex56_broadcast_retake",
            inputs={
                "manifest": manifest.model_dump(mode="json"),
                "truth_map": truth.model_dump(mode="json"),
                "grounding_contract": grounding.model_dump(mode="json"),
                "previous_script": script.model_dump(mode="json"),
                "findings": [
                    finding.model_dump(mode="json")
                    for finding in listener_blocks
                ],
            },
            score=score, manifest=manifest, grounding=grounding,
            scheduler=slot_scheduler, journal=journal, story_rules=story_rules,
            tokens=max(2600, int(resolved["target_words"]) * 6),
        )
        preceding_lines = _preceding_lines(manifest, script)
        listener = _call(pass_id="P7_rerun", slot="technical", fn=technical_fn, pack=pack, seam="codex56_blind_listener", inputs={"preceding_lines": preceding_lines}, schema=BlindListenerReport, scheduler=slot_scheduler, journal=journal, tokens=1800)
        remaining_listener_blocks = _listener_blocks(
            listener, {line["line_id"] for line in preceding_lines},
            grounding, preceding_lines[-1]["line_id"],
        )
        if remaining_listener_blocks:
            raise OriginalCodex56SolContractError(
                "blind-listener rerun still could not infer the declared "
                "mundane cause"
            )
    elif listener.optional_notes or any(not finding.blocking for finding in listener.findings):
        try:
            optional_script = _call_grounded_script(
                pass_id="P8_optional", fn=creative_fn, pack=pack,
                seam="codex56_broadcast_retake",
                inputs={
                    "manifest": manifest.model_dump(mode="json"),
                    "truth_map": truth.model_dump(mode="json"),
                    "grounding_contract": grounding.model_dump(mode="json"),
                    "previous_script": script.model_dump(mode="json"),
                    "findings": listener.model_dump(mode="json"),
                },
                score=score, manifest=manifest, grounding=grounding,
                scheduler=slot_scheduler, journal=journal,
                story_rules=story_rules,
                tokens=max(2600, int(resolved["target_words"]) * 6),
            )
            script = optional_script
        except OriginalCodex56SolError:
            pass
    audited_script = script
    audit = _call(pass_id="P9", slot="technical", fn=technical_fn, pack=pack, seam="codex56_final_contract_audit", inputs={"manifest": manifest.model_dump(mode="json"), "truth_map": truth.model_dump(mode="json"), "grounding_contract": grounding.model_dump(mode="json"), "script": script.model_dump(mode="json")}, schema=FinalContractAudit, scheduler=slot_scheduler, journal=journal, tokens=1800, post_validator=lambda value: _validate_audit_envelope(value, audited_script))
    audit_blocks = _audit_blocks(audit, script)
    advisories = _audit_advisories(audit, script)
    retake_ran = False
    if audit_blocks:
        retake_ran = True
        script = _call_grounded_script(
            pass_id="P9_retake", fn=creative_fn, pack=pack,
            seam="codex56_broadcast_retake",
            inputs={
                "manifest": manifest.model_dump(mode="json"),
                "truth_map": truth.model_dump(mode="json"),
                "grounding_contract": grounding.model_dump(mode="json"),
                "previous_script": script.model_dump(mode="json"),
                "findings": [
                    finding.model_dump(mode="json")
                    for finding in audit_blocks
                ],
            },
            score=score, manifest=manifest, grounding=grounding,
            scheduler=slot_scheduler, journal=journal, story_rules=story_rules,
            tokens=max(2600, int(resolved["target_words"]) * 6),
        )
        audited_script = script
        audit = _call(pass_id="P9_rerun", slot="technical", fn=technical_fn, pack=pack, seam="codex56_final_contract_audit", inputs={"manifest": manifest.model_dump(mode="json"), "truth_map": truth.model_dump(mode="json"), "grounding_contract": grounding.model_dump(mode="json"), "script": script.model_dump(mode="json")}, schema=FinalContractAudit, scheduler=slot_scheduler, journal=journal, tokens=1800, post_validator=lambda value: _validate_audit_envelope(value, audited_script))
        if _audit_blocks(audit, script):
            raise OriginalCodex56SolContractError("final contract audit rejected the repaired script")
        advisories = advisories + _audit_advisories(audit, script)

    # Two verdicts, never conflated: `model_verdict` is what the audit model
    # said, verbatim; `effective_verdict` is what the pipeline did about it.
    # A single ambiguous `accepted` key would leave two competing authorities.
    audit_disposition = {
        "schema_version": "original_codex56sol.final_audit_disposition.v1",
        "model_verdict": audit.accepted,
        "effective_verdict": ("accepted_after_script_retake" if retake_ran
                              else "accepted"),
        "script_retake_ran": retake_ran,
        "blocking_script_findings": len(_audit_blocks(audit, script)),
        "advisory_findings": [row.model_dump(mode="json") for row in advisories],
        "warnings": list(audit.warnings),
    }

    _assert_script_valid(score, manifest, script, story_rules, grounding)

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
    meta["original_codex56sol"] = {
        "call_journal": journal,
        "graph_proof": {
            "cast": len(score.cast), "scenes": len(score.scenes),
            "shots": len(score.shots), "beats": len(score.beats),
            "lines": len(manifest.lines),
        },
        "phase_10_verdict": "accepted",
        "listener_report": listener.model_dump(mode="json"),
        "final_audit": audit.model_dump(mode="json"),
        "final_audit_disposition": audit_disposition,
        "fair_play_disposition": fair_play_disposition,
        "grounding_receipt": _grounding_receipt(
            grounding, manifest, script,
        ),
        "accepted_artifacts": {
            "selected_possibility": selected.model_dump(mode="json"),
            "triage": triage.model_dump(mode="json"),
            "truth_map": truth.model_dump(mode="json"),
            "fair_play_report": fair.model_dump(mode="json"),
            "grounding_contract": grounding.model_dump(mode="json"),
            "broadcast_score": score.model_dump(mode="json"),
            "performance_script": script.model_dump(mode="json"),
            "blind_listener_report": listener.model_dump(mode="json"),
            "final_contract_audit": audit.model_dump(mode="json"),
        },
    }
    stamp_receipt(led.data, owner_bank="original_codex56sol", accepted_artifacts={"performance_script": script, "closed_manifest": manifest})
    canon = EpisodeCanon(title=script.title, premise=score.premise, setting=score.setting, time_of_day="", sound_palette=[])
    return OriginalCodex56SolTailParts(SimpleNamespace(title=script.title, premise=score.premise, setting=score.setting, time_of_day=""), canon, script.title, False, OriginalCodex56SolFinalizer())
