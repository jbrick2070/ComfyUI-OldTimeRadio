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
from typing import Annotated, Any, Callable, Iterable, Literal, Mapping  # noqa: F401

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from ._otr_canon import EpisodeCanon
from ._otr_content_authorship import stamp_receipt
from ._otr_json import parse_first_json_object
from ._otr_structured_call import (
    schema_shape_instruction,
    structured_call,
)
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
    # One caller per lost object (P3 enforces exactly that), and a draw may
    # carry up to 6 lost objects -- so a cap of 4 made a 5- or 6-object draw
    # impossible to satisfy. The cap tracks the draw's own ceiling.
    callers: list[FictionalName] = Field(max_length=6)
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


class ScriptLineReplacement(StrictModel):
    line_id: str
    text: str


class ScriptLinePatch(StrictModel):
    replacements: list[ScriptLineReplacement] = Field(min_length=1, max_length=8)


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
            "invented id. Select the most solvable card; if every candidate "
            "carries a concern, select the least compromised one and record the "
            "concern in findings. findings are ADVISORY notes recorded with the "
            "episode: they never reject a card. blocking MUST be a boolean, and "
            "every finding must name a concrete field."
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
            "required_caller_threads is AUTHORITATIVE: caller_threads MUST have "
            "exactly one row for EVERY row of it, with that row's thread_id and "
            "lost_object copied EXACTLY -- never shortened, never reworded, "
            "never dropped. Write `folded timetable`, not `timetable`. If a "
            "lost object has no caller thread, ADD the missing thread; never "
            "delete an existing one to compensate. "
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
            "max_beats is a HARD CEILING: if the score has more beats than "
            "max_beats, MERGE beats until it fits -- never delete a clue to fit. "
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
            "resolution_anchor MUST appear in the closure intent. "
            "FAIR PLAY: device_anchor MUST ALSO appear verbatim in at least one "
            "non-announcer clue-carrying intent that comes BEFORE the reveal "
            "beat -- the device is planted in the clues and only explained at "
            "the reveal. At least one non-announcer beat carrying a clue MUST "
            "therefore be positioned before the reveal beat; if every clue beat "
            "sits at or after the reveal, move a clue beat earlier."
        )
    if pass_id == "P6":
        rules += (
            " Preserve every immutable spoken anchor in grounding_contract. "
            "Each lost-object anchor must be spoken on a line carrying a clue "
            "for that object's thread; the exact device_anchor must be spoken "
            "on the manifest reveal line; the exact resolution_anchor must be "
            "spoken on the manifest closure line. FAIR PLAY: the exact "
            "device_anchor must ALSO be spoken on at least one clue-carrying "
            "line BEFORE the reveal line, so the listener hears the device "
            "before it is explained. Never replace the mundane lost-and-found "
            "mechanism with speculative technology, an ancient artifact, a "
            "crime, or a supernatural cause."
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
    if _validate_score(repaired, truth) is not None:
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
    if _validate_score(repaired, truth) is not None:
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


def _project_arc_phases(data: dict[str, Any]) -> bool:
    """`arc_phase` is not a decision -- the landmark ids already made it.

    The score names its own orientation, reveal and closure beats, and the
    contract says those three are `opening`, `reveal`, `closing`, and every other
    beat is `rising`. So the value is DERIVABLE, and asking a 12B model to also
    remember the enum's exact spelling just loses episodes: it wrote `closure`
    for `closing` and the schema rejected the whole score (live: prompt
    55756bac). Derive it and move on. Returns True when anything changed.
    """
    beats = data.get("beats")
    if not isinstance(beats, list):
        return False
    landmarks = {
        str(data.get("orientation_beat_id")): "opening",
        str(data.get("reveal_beat_id")): "reveal",
        str(data.get("closure_beat_id")): "closing",
    }
    changed = False
    for beat in beats:
        if not isinstance(beat, dict):
            return False
        intent = beat.get("line_intent")
        if not isinstance(intent, dict):
            return False
        expected = landmarks.get(str(beat.get("beat_id")), "rising")
        if intent.get("arc_phase") != expected:
            intent["arc_phase"] = expected
            changed = True
    return changed


def _project_score_arc_phases_only(raw: str) -> BroadcastScore | None:
    """Derive the arc phases and nothing else, so a typo cannot kill the score.

    Schema-valid or nothing: this makes the artifact PARSEABLE, it does not
    pretend the score is correct. Whatever else is wrong then reaches the
    post_validator, which can return it to the ladder.
    """
    try:
        data = parse_first_json_object(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if not isinstance(data, dict) or not _project_arc_phases(data):
        return None
    try:
        return BroadcastScore.model_validate(data)
    except ValidationError:
        return None


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

    if _project_arc_phases(data):
        changed = True

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
            if repaired is None:
                # The full projection is gated on the complete score contract, so
                # ONE unrelated defect used to throw the whole normalization away
                # -- including the arc_phase derivation -- and the raw enum typo
                # then died at the schema, where no ladder can see it (live:
                # prompt f1bf4ec7). The arc phases are derivable on their own.
                # Derive them, and let the post_validator report the real defect.
                arc_only = _project_score_arc_phases_only(raw)
                if arc_only is not None:
                    return arc_only.model_dump_json()
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
            {"role": "user", "content": json.dumps({"failed_artifact": failed_output, "error": str(effective_error), "inputs": _repair_inputs(pass_id, inputs)}, ensure_ascii=False, sort_keys=True)},
        ]
    # A bounded anchor patch is a MECHANICAL edit -- put this exact string into
    # this exact intent/line and keep everything else. Running it at the creative
    # brainstorming temperature made the model keep writing past its own closing
    # brace, and the truncated JSON came back undecodable on all three attempts
    # (live: prompt 41faff33). Cool it down and give the output room to close.
    is_patch = pass_id.endswith("_grounding_patch")
    base_temperature = .25 if is_patch else (.72 if slot == "creative" else .18)
    retry_temperature = .10 if is_patch else (
        .25 if slot == "creative" else .08
    )
    try:
        with scheduler.helper_context(f"original_codex56sol:{pass_id}"):
            # LLM slot: per-sub-pass workflow creative/technical closure.
            result = structured_call(
                prompt=prompt, schema=schema, slot_fn=capture,
                base_temperature=base_temperature,
                structural_retry_temperature=retry_temperature,
                max_new_tokens=tokens, max_attempts=3,
                post_validator=post_validator,
                repair_prompt_factory=repair,
                helper_name=f"original_codex56sol:{pass_id}",
            )
    except Exception as exc:
        raise OriginalCodex56SolPassError(f"{pass_id} failed: {exc}") from exc
    journal.append({"pass_id": pass_id, "slot": slot, "attempts": attempts})
    return result


def _repair_inputs(pass_id: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """The bounded input surface a typed repair actually needs.

    A repair prompt carries the ORIGINAL seam, the schema contract, the repair
    rules, the failed artifact -- and then re-sent every input as well. For P5
    that reached 5,772 tokens against a 4,592-token usable window, so
    PROMPT_GUARD truncated it, and what fell off the end was the part telling the
    model to return a complete artifact. It answered with one cast row, and the
    ladder exhausted on a defect the model had already been told how to fix
    (live: prompt a89a46a4).

    The failed artifact already carries the graph the model must repair. What it
    cannot reconstruct from that alone are the IMMUTABLE anchors and the clue
    inventory -- so send those, and nothing else.
    """
    if pass_id not in {"P5", "P6"}:
        return dict(inputs)
    bounded: dict[str, Any] = {}
    grounding = inputs.get("grounding_contract")
    if isinstance(grounding, Mapping):
        bounded["grounding_contract"] = {
            "lost_object_anchors": grounding.get("lost_object_anchors"),
            "device_anchor": grounding.get("device_anchor"),
            "resolution_anchor": grounding.get("resolution_anchor"),
            "clues": [
                {
                    "clue_id": clue.get("clue_id"),
                    "lost_object": clue.get("lost_object"),
                }
                for clue in grounding.get("clues", [])
                if isinstance(clue, Mapping)
            ],
        }
    truth = inputs.get("truth_map")
    if isinstance(truth, Mapping):
        bounded["truth_map"] = {
            "audible_clues": [
                {
                    "clue_id": clue.get("clue_id"),
                    "thread_id": clue.get("thread_id"),
                }
                for clue in truth.get("audible_clues", [])
                if isinstance(clue, Mapping)
            ],
        }
    for key in ("manifest", "target_words_advisory", "num_characters_advisory"):
        if key in inputs:
            bounded[key] = inputs[key]
    return bounded


def _restore_slate_immutables(
    slate: PossibilitySlate,
    draw: ConstraintDraw,
) -> bool:
    """Put the draw's own words back into the fields that only echo it.

    `lost_objects` and `acoustic_device` are not authored: they are the immutable
    constraint draw, handed to the model and asked for back verbatim. The model
    paraphrases them ("a clockwork display chimes in the same rhythm" -> "a
    chiming clockwork display") and the run dies on a string compare -- while the
    STORY it wrote is about exactly the right device (live: prompt efafc6fa).

    Restoring an input is not authoring. Everything downstream is anchored to the
    draw anyway: the grounding contract is built from it, and the script must
    speak its exact anchors on the air. So overwrite the echo and keep the story.
    """
    changed = False
    for card in slate.possibilities:
        if card.lost_objects != draw.lost_objects:
            card.lost_objects = list(draw.lost_objects)
            changed = True
        if card.acoustic_device != draw.acoustic_device:
            card.acoustic_device = draw.acoustic_device
            changed = True
    return changed


def _validate_slate(slate: PossibilitySlate, draw: ConstraintDraw) -> str | None:
    ids = [card.possibility_id for card in slate.possibilities]
    if len(set(ids)) != len(ids):
        return "possibility ids must be unique"
    if _restore_slate_immutables(slate, draw):
        log.info(
            "[original_codex56sol] P1 restored the immutable draw fields the "
            "slate only echoes at the schema-validated attempt boundary",
        )
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
    """P2 owns one thing Python cannot: CHOOSING a card.

    The selection is its only mechanically checkable claim -- the id either names
    a real card or it does not. Its findings are advice and bind nothing.
    """
    ids = {card.possibility_id for card in slate.possibilities}
    if triage.selected_possibility_id not in ids:
        return "selected_possibility_id must exactly match one slate id"
    return None


def _restore_thread_lost_objects(
    truth: AudibleTruthMap,
    selected: PossibilityCard,
) -> bool:
    """Restore the immutable object name a caller thread only echoes.

    `lost_object` is not authored: it is one of the draw's strings, handed to the
    model and asked for back. The model writes the thread's real content -- the
    caller, the need -- and drops the modifier on the echo: "timetable" for
    "folded timetable", "whistle" for "tin whistle" (live: prompt 37a3cedf).

    Restore it ONLY when the correction is forced: same number of threads as
    objects, and each mismatched thread names exactly one remaining draw object
    by containment. Anything ambiguous -- a dropped thread, two candidates --
    is the model's to fix, and goes back to the ladder untouched.
    """
    objects = list(selected.lost_objects)
    threads = list(truth.caller_threads)
    if len(threads) != len(objects):
        return False
    if Counter(row.lost_object for row in threads) == Counter(objects):
        return False

    remaining = list(objects)
    unmatched: list[Any] = []
    for row in threads:
        if row.lost_object in remaining:
            remaining.remove(row.lost_object)
        else:
            unmatched.append(row)
    if not unmatched or len(unmatched) != len(remaining):
        return False

    for row in unmatched:
        spoken = _normalize_grounding_text(row.lost_object)
        candidates = [
            candidate for candidate in remaining
            if spoken
            and (spoken in _normalize_grounding_text(candidate)
                 or _normalize_grounding_text(candidate) in spoken)
        ]
        if len(candidates) != 1:
            # No name to go on -- but if this is the last thread and the last
            # unclaimed object, the assignment is FORCED: there is no other value
            # it could take. Python is not choosing, it is counting.
            if len(remaining) != 1 or len(unmatched) != 1:
                return False
            candidates = list(remaining)
        row.lost_object = candidates[0]
        remaining.remove(candidates[0])
    log.info(
        "[original_codex56sol] P3 restored %d immutable lost-object name(s) "
        "the caller threads only echo",
        len(unmatched),
    )
    return True


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
    if selected is not None:
        _restore_thread_lost_objects(truth, selected)
        written = Counter(row.lost_object for row in truth.caller_threads)
        required = Counter(selected.lost_objects)
        if written != required:
            missing = sorted((required - written).elements())
            invented = sorted((written - required).elements())
            detail = []
            if missing:
                detail.append("missing a caller thread for " + ", ".join(
                    repr(obj) for obj in missing))
            if invented:
                detail.append("wrote " + ", ".join(
                    repr(obj) for obj in invented) + " instead")
            return (
                "caller_threads must contain exactly one row per selected lost "
                "object, and each lost_object MUST be copied VERBATIM from the "
                "selected card's lost_objects array "
                f"{selected.lost_objects!r} -- never shortened, never reworded"
                + ("; " + "; ".join(detail) if detail else "")
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
) -> str | None:
    """The score's own contract: graph, roster, landmarks, clue coverage.

    The immutable ANCHORS are deliberately not checked here. A line_intent is a
    private note to the script writer, not something the listener hears -- so
    requiring the exact anchor strings inside it was a proxy for the thing that
    actually matters, and the bounded patch that enforced the proxy kept dying on
    it (live: prompts 41faff33, 6fe52216, 522e1581, 6a325375 -- four rolls). The
    anchors are proven where they are real: in the SPOKEN SCRIPT, by
    `_validate_script_grounding`, whose patch rewrites dialogue the model is
    actually good at writing. The seam still asks for anchors in the intents,
    because that steers P6 -- it is guidance, not a gate.
    """
    if sum(c.char_id == "announcer" and c.role == "announcer"
           for c in score.cast) != 1:
        return "cast needs one separate char_id='announcer', role='announcer' row"
    # ROLE is the announcer authority everywhere downstream: `_compile_manifest`
    # stamps `speaker_role` from it, and the fair-play rule is defined over
    # non-announcer voices. A second row carrying role='announcer' under another
    # char_id would make the cast and the manifest disagree about who is the
    # announcer, so there is exactly one, and every other row is a real voice.
    if sum(c.role == "announcer" for c in score.cast) != 1:
        return (
            "exactly one cast row may have role='announcer'; give every other "
            "row role='desk_operator' or role='caller'"
        )
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
        unassigned = sorted(expected_clues - set(assigned_clues))
        unknown = sorted(set(assigned_clues) - expected_clues)
        detail = []
        if unassigned:
            objects = {
                clue.clue_id: clue.thread_id for clue in truth.audible_clues
            }
            detail.append(
                "these truth-map clues are on NO beat: "
                + ", ".join(
                    f"{clue_id} (thread {objects.get(clue_id)})"
                    for clue_id in unassigned
                )
                + " -- add each one to the clue_ids of the beat where it is heard"
            )
        if unknown:
            detail.append(
                "these clue ids do not exist in the truth map: "
                + ", ".join(unknown)
            )
        return (
            "every truth-map clue must be assigned to exactly one line intent; "
            + "; ".join(detail)
        )
    if len(assigned_clues) != len(set(assigned_clues)):
        return _SCORE_DUPLICATE_CLUE_ERROR
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


def _drop_unknown_clue_ids(
    score: BroadcastScore,
    truth: AudibleTruthMap,
) -> bool:
    """Remove clue ids that name no clue. Noise, not story.

    The model wrote the SCHEMA PATH into the data -- clue_ids carrying
    "grounding_contract.device_anchor" (live: prompts ac762dd9, faffc134). A
    string that names no truth-map clue cannot be a clue, so deleting it invents
    nothing and loses nothing.

    Only when every REAL clue is still assigned afterwards. If dropping the noise
    would leave a clue uncovered, it was not noise, and the model must answer for
    it at the ladder.
    """
    known = {clue.clue_id for clue in truth.audible_clues}
    trimmed: list[tuple[Any, list[str]]] = []
    for beat in score.beats:
        kept = [
            clue_id for clue_id in beat.line_intent.clue_ids
            if clue_id in known
        ]
        if len(kept) != len(beat.line_intent.clue_ids):
            trimmed.append((beat, kept))
    if not trimmed:
        return False
    projected = {
        clue_id for beat in score.beats
        for clue_id in beat.line_intent.clue_ids
        if clue_id in known
    }
    if projected != known:
        return False
    for beat, kept in trimmed:
        beat.line_intent.clue_ids = kept
    return True


def _project_announcer_char_id(score: BroadcastScore) -> BroadcastScore | None:
    """Canonicalize the announcer's char_id -- an ID, never authored content.

    The model writes a perfectly good announcer row (`role: "announcer"`, name,
    description) and files it under `char_id: "a"`. That is a COORDINATE, not a
    story decision, and the contract fixes it: the announcer is `announcer`.
    Rejecting the whole score over it sends a 5,700-token repair prompt through
    a 4,592-token window (live: prompt a89a46a4), and the truncated model comes
    back with a fragment. Rename the id and the beats that point at it.
    """
    announcers = [row for row in score.cast if row.role == "announcer"]
    if len(announcers) != 1:
        return None
    row = announcers[0]
    if row.char_id == "announcer":
        return None
    if any(other.char_id == "announcer" for other in score.cast if other is not row):
        return None
    old_id = row.char_id
    repaired = score.model_copy(deep=True)
    for cast_row in repaired.cast:
        if cast_row.char_id == old_id:
            cast_row.char_id = "announcer"
    for beat in repaired.beats:
        if beat.char_id == old_id:
            beat.char_id = "announcer"
    return repaired


def _validate_score_attempt(
    score: BroadcastScore,
    truth: AudibleTruthMap,
    story_rules: Any,
) -> str | None:
    """Validate every P5 response at the exact structured-call boundary.

    P5 base and typed-repair outputs first become a strict ``BroadcastScore``
    inside ``structured_call``.  The contiguous-shot projection belongs here,
    not only in the raw-string normalization path: this is the one boundary
    every schema-valid ladder response must cross.  The projection changes
    only mechanical shot ownership and is accepted only when the complete
    requested score contract validates again.
    """
    repaired_kinds: list[str] = []
    announcer_id = _project_announcer_char_id(score)
    if announcer_id is not None:
        score.cast = announcer_id.cast
        score.beats = announcer_id.beats
        repaired_kinds.append("announcer char_id")
    if _drop_unknown_clue_ids(score, truth):
        repaired_kinds.append("clue ids naming no clue")
    for _ in range(3):
        structural_error = _validate_score(score, truth)
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
    structural_error = _validate_score(score, truth)
    if repaired_kinds and structural_error is None:
        log.info(
            "[original_codex56sol] P5 normalized safe %s at the "
            "schema-validated attempt boundary",
            " + ".join(repaired_kinds),
        )
    if structural_error is not None:
        return structural_error
    return _validate_authored_surface(score, story_rules)


_PRE_REVEAL_CLUE_BEAT_ERROR = (
    "at least one non-announcer clue-carrying beat must come BEFORE the reveal "
    "beat: the device is planted in the clues and explained at the reveal, so a "
    "mystery whose every clue lands at or after the reveal cannot be solved by "
    "the listener"
)


def _pre_reveal_clue_beats(score: BroadcastScore) -> list[BeatConcept]:
    """Non-announcer clue beats positioned before the reveal beat.

    This is the population the fair-play contract is defined over. It is a pure
    read of accepted beat order and clue ownership -- no judgement.

    "Non-announcer" means the beat's CAST ROLE is not announcer, which is the
    same authority `_compile_manifest` uses to stamp `speaker_role`. Keying off
    `char_id` instead would let a cast row with role='announcer' under some other
    id count here but not in the manifest, and the two halves of this rule would
    then disagree about which lines exist.
    """
    role_by_char = {row.char_id: row.role for row in score.cast}
    positions = {beat.beat_id: index for index, beat in enumerate(score.beats)}
    reveal_position = positions.get(score.reveal_beat_id)
    if reveal_position is None:
        return []
    return [
        beat for index, beat in enumerate(score.beats)
        if index < reveal_position
        and role_by_char.get(beat.char_id) != "announcer"
        and beat.line_intent.clue_ids
    ]


def _pre_reveal_clue_lines(manifest: ClosedLineManifest) -> list[ManifestLine]:
    """The spoken-line image of `_pre_reveal_clue_beats`, in manifest order."""
    positions = {line.line_id: index for index, line in enumerate(manifest.lines)}
    reveal_position = positions.get(manifest.reveal_line_id)
    if reveal_position is None:
        return []
    return [
        line for index, line in enumerate(manifest.lines)
        if index < reveal_position
        and line.speaker_role != "announcer"
        and line.clue_ids
    ]


def _beat_ceiling(target_words: int) -> int:
    """How many spoken beats a broadcast of this length can actually hold.

    The floor is 8, not 5: the show's own shape needs an announcer, one beat per
    caller, a reveal and a closure, and squeezing that into five beats is not a
    length rule, it is a different show. The ceiling exists to stop the model
    building a nineteen-beat graph for a thirty-word broadcast -- every beat is a
    line the script must then write.
    """
    return max(8, min(40, int(target_words) // 4))


def _validate_score_scale(score: BroadcastScore, target_words: int) -> str | None:
    """A 30-word episode is not a 19-beat episode.

    The seam gave the score a FLOOR (at least 5 beats) and no ceiling, so the
    model built a nineteen-beat graph for a thirty-word broadcast. Every line
    then had to be written, and the script blew its token budget and came back
    truncated (live: prompt 717f3a4f). Length is not a matter of taste here: the
    beats are what the script must fill.
    """
    ceiling = _beat_ceiling(target_words)
    if len(score.beats) > ceiling:
        return (
            f"a {target_words}-word broadcast holds at most {ceiling} beats, "
            f"but you wrote {len(score.beats)}. Tell the same story in "
            f"{ceiling} or fewer beats: merge beats, do not delete the clues."
        )
    return None


def _validate_score_clue_ownership(
    score: BroadcastScore,
    grounding: GroundingContract,
) -> str | None:
    """Every lost object must reach a listener through a non-announcer voice.

    OWNERSHIP AND ORDER ONLY -- never anchor text.  No patch may touch
    `clue_ids` or move a beat, so a defect in either is unrepairable below this
    rung and the score itself must be re-authored.  That is what this check
    returns to the P5 ladder.  The anchors are proven in the spoken script.
    """
    role_by_char = {row.char_id: row.role for row in score.cast}
    clue_ids_by_object: dict[str, set[str]] = {}
    for clue in grounding.clues:
        clue_ids_by_object.setdefault(clue.lost_object, set()).add(clue.clue_id)
    for anchor in grounding.lost_object_anchors:
        clue_ids = clue_ids_by_object.get(anchor, set())
        if not any(
            clue_ids.intersection(beat.line_intent.clue_ids)
            for beat in score.beats
            if role_by_char.get(beat.char_id) != "announcer"
        ):
            # Name the clue AND the beats that could carry it. "Move it" with no
            # candidates is a puzzle; "move it to b2 or b3" is an instruction.
            candidates = [
                f"{beat.beat_id} ({beat.speaker})" for beat in score.beats
                if role_by_char.get(beat.char_id) != "announcer"
            ]
            return (
                f"the clue(s) {', '.join(sorted(clue_ids))} for lost object "
                f"{anchor!r} are held by the announcer, who may not be the only "
                "voice that carries a clue. Move the clue id into the clue_ids "
                "of one of these beats: " + ", ".join(candidates)
            )
    # The bounded intent patch cannot create a pre-reveal clue beat -- only the
    # score author can -- so the existence of one belongs here, at the rung that
    # can still re-author the score.
    if not _pre_reveal_clue_beats(score):
        return _PRE_REVEAL_CLUE_BEAT_ERROR
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


def _restore_anchor_in_text(text: str, anchor: str) -> str | None:
    """Put the modifier back on an anchor the model shortened.

    The patch's whole job is to make a line name an immutable string. The model
    puts the thing in the right place and drops half its name -- "the grille" for
    "ventilation grille", "the stamp" for "library stamp" (live: prompts 6fe52216
    and 522e1581). It decided WHERE the anchor belongs; the exact wording was
    never its decision to make, so restore it.

    Only when the correction is FORCED: the longest sub-phrase of the anchor that
    appears exactly once in the text, as whole words. Otherwise return None and
    let the ladder deal with it.
    """
    words = _normalize_grounding_text(anchor).split()
    if len(words) < 2:
        return None
    for size in range(len(words) - 1, 0, -1):
        for start in range(0, len(words) - size + 1):
            phrase = " ".join(words[start:start + size])
            pattern = re.compile(
                r"(?<!\w)" + r"[\W_]+".join(
                    re.escape(word) for word in phrase.split()
                ) + r"(?!\w)",
                re.IGNORECASE,
            )
            matches = pattern.findall(text)
            if len(matches) != 1:
                continue
            restored = pattern.sub(anchor, text, count=1)
            if _contains_grounding_anchor(restored, anchor):
                return restored
    return None


def _restore_patch_anchors(
    replacements: Iterable[Any],
    required_by_id: Mapping[str, Any],
    id_field: str,
    text_field: str,
) -> None:
    """Restore every shortened immutable anchor in a bounded patch, in place."""
    for replacement in replacements:
        required = required_by_id.get(getattr(replacement, id_field))
        if not required:
            continue
        for anchor in required:
            text = getattr(replacement, text_field)
            if _contains_grounding_anchor(text, anchor):
                continue
            restored = _restore_anchor_in_text(text, anchor)
            if restored is not None:
                setattr(replacement, text_field, restored)
                log.info(
                    "[original_codex56sol] restored the immutable anchor %r "
                    "the patch shortened", anchor,
                )


def _plant_target(
    candidates: list[str],
    targets: dict[str, set[str]],
) -> str:
    """Where the device plant goes: EARLY, but never onto a loaded line.

    Fair play wants the plant early -- a clue the listener meets in the last
    breath before the reveal is fair only on paper. But the earliest clue line is
    usually the one already carrying its own lost-object anchor, and piling a
    third verbatim string onto one line is how the bounded patch fails: the model
    writes two of the three and the ladder exhausts (live: prompt 6fe52216, "must
    name every required immutable anchor for beat b9").

    So: the earliest candidate with the LIGHTEST anchor load. Deterministic, and
    it keeps each line speaking one immutable thing.
    """
    return min(
        candidates,
        key=lambda candidate: (len(targets.get(candidate, ())),
                               candidates.index(candidate)),
    )


def _patch_tokens(plan: list[dict[str, Any]]) -> int:
    """Room for one rewritten intent/line per target, plus the JSON envelope.

    900 was a flat guess. A four-target patch ran out of room mid-object and the
    truncated JSON was undecodable on every attempt -- so budget from the actual
    work: each replacement restates the current text plus the anchors it must
    carry, and the model needs headroom to close its own braces.
    """
    per_target = 0
    for row in plan:
        text = str(row.get("current_intent") or row.get("current_text") or "")
        anchors = " ".join(str(a) for a in row.get("required_anchors", ()))
        # ~4 chars/token, doubled: the replacement is a rewrite, not a copy.
        per_target += 2 * (len(text) + len(anchors)) // 4 + 60
    return max(900, 240 + per_target)


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
    # FAIR PLAY: the listener can only solve the mystery if the device is audible
    # before it is explained.  The device anchor is spoken on at least one
    # clue-carrying line BEFORE the reveal line, as well as on the reveal line.
    pre_reveal_clue_lines = _pre_reveal_clue_lines(manifest)
    if not pre_reveal_clue_lines:
        return _PRE_REVEAL_CLUE_BEAT_ERROR
    if not any(
        row.line_id in script_by_id
        and _contains_grounding_anchor(
            script_by_id[row.line_id].text, contract.device_anchor,
        )
        for row in pre_reveal_clue_lines
    ):
        return (
            f"device anchor {contract.device_anchor!r} must be spoken on at "
            "least one clue-carrying line before the reveal line, not only on "
            "the reveal line"
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
    # The pre-reveal plant of the device anchor (see `_validate_script_grounding`).
    pre_reveal_clue_lines = _pre_reveal_clue_lines(manifest)
    if not pre_reveal_clue_lines:
        return None
    if not any(
        row.line_id in script_by_id
        and _contains_grounding_anchor(
            script_by_id[row.line_id].text, grounding.device_anchor,
        )
        for row in pre_reveal_clue_lines
    ):
        targets.setdefault(
            _plant_target(
                [row.line_id for row in pre_reveal_clue_lines], targets,
            ),
            set(),
        ).add(grounding.device_anchor)
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
    _restore_patch_anchors(
        patch.replacements,
        {key: row["required_anchors"] for key, row in expected.items()},
        "line_id", "text",
    )
    seen: set[str] = set()
    for replacement in patch.replacements:
        line_id = replacement.line_id
        if line_id not in expected or line_id in seen:
            return "script line patch must replace every and only planned line ids once"
        seen.add(line_id)
        for anchor in expected[line_id]["required_anchors"]:
            if not _contains_grounding_anchor(replacement.text, anchor):
                return (
                    f"line {line_id}: the spoken text MUST contain the exact "
                    f"string {anchor!r}, spoken word for word. You wrote: "
                    f"{replacement.text!r}"
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


def _validate_script_line_patch_step(
    patch: ScriptLinePatch,
    script: PerformanceScript,
    target: list[dict[str, Any]],
    score: BroadcastScore,
    manifest: ClosedLineManifest,
    story_rules: Any,
) -> str | None:
    """One line's patch, checked at the rung that can still repair it.

    The GROUNDING contract is deliberately not checked here: this call owns one
    line, and the other lines' anchors are the next call's job. Graph, roster and
    SAFETY are checked -- a replacement that smuggles a banned term goes back to
    the model, not into the episode. The complete grounding proof runs once, over
    the finished script, in `_assert_script_valid`.
    """
    patch_error = _validate_script_line_patch(patch, target)
    if patch_error is not None:
        return patch_error
    repaired = _merge_script_line_patch(script, patch)
    script_error = _validate_script(score, manifest, repaired, story_rules)
    if script_error is not None:
        return f"script line patch leaves the script invalid: {script_error}"
    return None


def _repair_script_grounding_lines(
    *, script: PerformanceScript, score: BroadcastScore,
    manifest: ClosedLineManifest, grounding: GroundingContract, pack: Any,
    creative_fn: GenerateFn, scheduler: Any, journal: list[dict],
    story_rules: Any, origin_pass_id: str,
) -> PerformanceScript:
    """Rewrite ONE line per call -- the unit the model can actually deliver.

    Asking for every planned line in a single patch meant a small model that
    rewrote two of three lines failed the whole batch ("script line patch must
    replace every planned line id" -- live: prompt ec428576), and the two good
    lines went in the bin with the missing one. One line, one call, one set of
    anchors: each call is tiny, cannot truncate, and its success is kept.
    """
    plan = _script_grounding_repair_plan(script, manifest, grounding)
    if not plan:
        raise OriginalCodex56SolContractError(
            f"{origin_pass_id} grounding repair has no eligible immutable-anchor plan"
        )
    for row in plan:
        target = [row]
        patch = _call(
            pass_id=f"{origin_pass_id}_grounding_patch", slot="creative",
            fn=creative_fn,
            pack=pack, seam="codex56_script_anchor_patch",
            inputs={"targets": target}, schema=ScriptLinePatch,
            scheduler=scheduler, journal=journal,
            tokens=_patch_tokens(target),
            post_validator=lambda value, t=target, current=script: (
                _validate_script_line_patch_step(
                    value, current, t, score, manifest, story_rules,
                )
            ),
        )
        script = _merge_script_line_patch(script, patch)
    return script


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
    # The pre-reveal plant: the line on which the listener could actually hear the
    # device before it was explained.
    plant_line_id = next(
        row.line_id for row in _pre_reveal_clue_lines(manifest)
        if _contains_grounding_anchor(
            script_by_id[row.line_id].text, contract.device_anchor,
        )
    )
    for kind, anchor, line_id in (
        ("device_plant", contract.device_anchor, plant_line_id),
        ("device", contract.device_anchor, manifest.reveal_line_id),
        ("resolution", contract.resolution_anchor, manifest.closure_line_id),
    ):
        text = script_by_id[line_id].text
        evidence.append({
            "kind": kind, "anchor": anchor, "line_id": line_id,
            "exact_span": _raw_anchor_span(text, anchor) or text,
        })
    return {
        "schema_version": "original_codex56sol.grounding_receipt.v2",
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
    if selected is None:
        raise OriginalCodex56SolContractError("triage did not select a valid possibility")
    # Hand P3 the caller-thread ROWS instead of asking it to remember how many
    # to write. The schema's `min_length=2` says one thing and the real rule --
    # one thread per lost object -- says another, and a 12B model believes the
    # schema: it shipped two threads for a three-object draw twice in a row and
    # dropped 'tin whistle' entirely (live: prompts 5bd46a5e, d199a783). A
    # skeleton is not authorship -- Python states the contract as DATA and the
    # model fills in the caller and the need.
    thread_skeleton = [
        {
            "thread_id": f"t{index}",
            "lost_object": lost_object,
            "caller_name": "<write the caller's name>",
            "practical_need": "<write why this caller needs it back>",
        }
        for index, lost_object in enumerate(selected.lost_objects, 1)
    ]
    truth = _call(pass_id="P3", slot="creative", fn=creative_fn, pack=pack, seam="codex56_audible_truth_map", inputs={"selected": selected.model_dump(mode="json"), "draw": draw.model_dump(mode="json"), "required_caller_threads": thread_skeleton}, schema=AudibleTruthMap, scheduler=slot_scheduler, journal=journal, tokens=2800, post_validator=lambda value: _validate_truth_map(value, selected) or _validate_authored_surface(value, story_rules))
    grounding = _build_grounding_contract(draw, truth)
    score = _call(
        pass_id="P5", slot="creative", fn=creative_fn, pack=pack,
        seam="codex56_broadcast_score",
        inputs={
            "truth_map": truth.model_dump(mode="json"),
            "grounding_contract": grounding.model_dump(mode="json"),
            "target_words_advisory": int(resolved["target_words"]),
            "num_characters_advisory": int(resolved["num_characters"]),
            "max_beats": _beat_ceiling(int(resolved["target_words"])),
        },
        schema=BroadcastScore, scheduler=slot_scheduler, journal=journal,
        tokens=3600,
        # Clue OWNERSHIP and beat ORDER are the half of the contract the
        # ladder must see while it can still re-author the score: no patch may
        # touch clue_ids or move a beat. The immutable ANCHORS are not checked
        # here -- they are proven in the spoken script, where they are real.
        post_validator=lambda value, t=truth, g=grounding: (
            _validate_score_attempt(value, t, story_rules)
            or _validate_score_clue_ownership(value, g)
            or _validate_score_scale(value, int(resolved["target_words"]))
        ),
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
        # The script must fill every manifest line, so budget by LINES, not by
        # the word target alone: a long score with a small target still has to
        # be written, and a script cut off mid-object is undecodable JSON.
        tokens=max(
            2600,
            240 + 160 * len(manifest.lines) + int(resolved["target_words"]) * 4,
        ),
    )
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
    meta.setdefault("original_codex56sol", {}).update({
        "call_journal": journal,
        "graph_proof": {
            "cast": len(score.cast), "scenes": len(score.scenes),
            "shots": len(score.shots), "beats": len(score.beats),
            "lines": len(manifest.lines),
        },
        "grounding_receipt": _grounding_receipt(
            grounding, manifest, script,
        ),
        "accepted_artifacts": {
            "selected_possibility": selected.model_dump(mode="json"),
            "triage": triage.model_dump(mode="json"),
            "truth_map": truth.model_dump(mode="json"),
            "grounding_contract": grounding.model_dump(mode="json"),
            "broadcast_score": score.model_dump(mode="json"),
            "performance_script": script.model_dump(mode="json"),
        },
    })
    stamp_receipt(led.data, owner_bank="original_codex56sol", accepted_artifacts={"performance_script": script, "closed_manifest": manifest})
    canon = EpisodeCanon(title=script.title, premise=score.premise, setting=score.setting, time_of_day="", sound_palette=[])
    return OriginalCodex56SolTailParts(SimpleNamespace(title=script.title, premise=score.premise, setting=score.setting, time_of_day=""), canon, script.title, False, OriginalCodex56SolFinalizer())
