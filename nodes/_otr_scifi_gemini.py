"""Sci-Fi Gemini v4 source-bank runner."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import typing
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Annotated,
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    MutableMapping,
    Sequence,
)

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    from ._otr_structured_call import schema_shape_instruction, structured_call
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
    from _otr_structured_call import schema_shape_instruction, structured_call  # type: ignore
    import _otr_ledger_freeze  # type: ignore
    from production_ledger import stamp_word_counts  # type: ignore


class ScifiGeminiError(RuntimeError): pass
class SciFiGeminiPayloadContractError(ScifiGeminiError): pass
class SciFiGeminiPayloadRouteError(ScifiGeminiError): pass
class SciFiGeminiPayloadThinError(ScifiGeminiError): pass
class SciFiGeminiTargetRangeError(ScifiGeminiError): pass
class SciFiGeminiPackContractError(ScifiGeminiError): pass
class SciFiGeminiPassError(ScifiGeminiError): pass
class SciFiGeminiGraphError(ScifiGeminiError): pass
class SciFiGeminiProvenanceError(ScifiGeminiError): pass
class SciFiGeminiPreTailAuditError(ScifiGeminiError): pass
class SciFiGeminiSavedLedgerAuditError(ScifiGeminiError): pass


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


class GeminiPayloadV4(_Strict):
    payload: dict[str, str]
    source_mode: Literal["rss", "operator_pinned"]
    payload_sha256: str


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


class PitchV4(_Strict):
    premise: str
    setting: str
    tonal_palette: str


class PitchSlateV4(_Strict):
    # Exactly three pitches -- expressed as a length-pinned LIST, not a tuple.
    # These artifacts are parsed from model-emitted JSON, and JSON has no tuple:
    # the model can only ever send an array. Under _Strict (strict=True) Pydantic
    # will NOT coerce a list into a tuple, so a tuple annotation here is
    # unsatisfiable by construction -- the pass can never validate, no matter
    # what the model writes. This killed the Gemini lane at P1 on its first live
    # roll ("pitches: Input should be a valid tuple ... input_type=list").
    # min_length == max_length == 3 keeps the exact same three-pitch contract.
    pitches: list[PitchV4] = Field(min_length=3, max_length=3)


class PitchSelectionV4(_Strict):
    selected_index: Literal[0, 1, 2]
    rationale: str


class CastV4(_Strict):
    char_id: Literal["announcer", "c01", "c02", "c03"]
    name: str
    character_description: str
    gender: str


class AdvisoryBeatBandV4(_Strict):
    beat_id: str
    advisory_word_center: int = Field(ge=0)


class ShotV4(_Strict):
    shot_id: str
    scene_id: str
    description: str
    visual_prompt: str


class BeatV4(_Strict):
    beat_id: str
    line_id: str
    scene_id: str
    shot_id: str
    speaker: str
    char_id: Literal["announcer", "c01", "c02", "c03"]
    speaker_role: Literal["character", "announcer"]
    intent: str
    mood: str
    fact_ids: list[str] = []
    order: int


class MusicCueV4(_Strict):
    cue_id: Literal["music_open", "music_inter", "music_close"]
    placement: Literal["open", "inter", "close"]
    description: str
    generation_prompt: str
    anchor_beat_id: str


class SceneV4(_Strict):
    scene_id: str
    env: str
    description: str
    shots: list[ShotV4] = Field(min_length=1)
    beats: list[BeatV4] = Field(min_length=1)


class OutlineV4(_Strict):
    title: str
    premise: str
    setting: str
    time_of_day: str
    cast: list[CastV4] = Field(min_length=2, max_length=4)
    scenes: list[SceneV4] = Field(min_length=1)
    music_cues: list[MusicCueV4] = Field(min_length=1, max_length=3)
    advisory_word_bands: list[AdvisoryBeatBandV4]


class LineFactUseV4(_Strict):
    fact_id: str
    spoken_claim: str


class DraftLineV4(_Strict):
    beat_id: str
    text: str
    fact_uses: list[LineFactUseV4] = []
    non_fact: bool = False


class SceneDraftV4(_Strict):
    lines: list[DraftLineV4] = Field(min_length=1)


class SceneCritiqueV4(_Strict):
    passed: bool
    feedback: str
    # A clean critique has nothing to map, and the model omits the key entirely
    # rather than writing `{}`. Requiring it turns "the scene is fine" into a
    # validation failure. The empty map IS the honest value for a clean pass.
    line_fact_ids: dict[str, list[str]] = Field(default_factory=dict)


GenerateFn = Callable[..., str]
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’-]*")
_DECORATION_RE = re.compile(r"[\r\n\t\[\]()\x60*]|^\s*\x60\x60\x60|^\s*[-*]\s+")
_ALL_CAPS_RE = re.compile(r"\b[A-Z]{2,}\b")


def _digest(payload: Mapping[str, str]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _span_ok(span: SourceSpanV4, payload: Mapping[str, str]) -> bool:
    return span.quote == payload.get(span.field, "")[span.start:span.end]


def _span_mismatch(span: SourceSpanV4, payload: Mapping[str, str]) -> str:
    expected = payload.get(span.field, "")[span.start:span.end]
    return f"{span.field}[{span.start}:{span.end}] expected {expected[:300]!r}; returned {span.quote[:300]!r}"


def validate_gemini_payload(payload: Mapping[str, Any], resolved: Mapping[str, Any]) -> GeminiPayloadV4:
    try:
        clean = validate_source_payload(dict(payload), "scifi_gemini")
    except Exception as exc:
        raise SciFiGeminiPayloadContractError(str(exc)) from exc
    source = str(resolved.get("seed_source") or "")
    if source == "custom_premise":
        mode = "operator_pinned"
        text = clean["seed_text"]
        if len(text.split()) < 8 or len(set(re.findall(r"[A-Za-z0-9]+", text.lower()))) < 4:
            raise SciFiGeminiPayloadThinError("pinned source is below the 8/4 thinness floor")
    elif source == "rss_fetch":
        mode = "rss"
        text = clean["full_text"]
        if not meets_v4_rss_source_floor(text):
            raise SciFiGeminiPayloadThinError(
                "RSS source is below the 400/80/12 v4 source floor"
            )
    else:
        raise SciFiGeminiPayloadRouteError("scifi_gemini accepts only rss_fetch or custom_premise")
    try:
        target = resolved.get("target_words")
        if isinstance(target, bool) or not isinstance(target, int) or not 30 <= target <= 900:
            raise ValueError
    except Exception as exc:
        raise SciFiGeminiTargetRangeError("target_words must be an integer from 30 through 900") from exc
    return GeminiPayloadV4(payload=clean, source_mode=mode, payload_sha256=_digest(clean))


def _fact_validator(index: FactIndexV4, payload: Mapping[str, str]) -> str | None:
    fact_ids = {x.fact_id for x in index.facts}
    for fact in index.facts:
        if not fact.source_spans:
            return f"fact {fact.fact_id} must contain a source span"
        for span in fact.source_spans:
            if not _span_ok(span, payload):
                return f"fact {fact.fact_id} has an invalid source span: {_span_mismatch(span, payload)}"
    for number in index.numbers:
        if number.fact_id not in fact_ids:
            return f"number {number.number_id} has an invalid fact/span reference"
        if not _span_ok(number.source_span, payload):
            return f"number {number.number_id} has an invalid source span: {_span_mismatch(number.source_span, payload)}"
    for entity in index.entities:
        if not entity.source_spans:
            return f"entity {entity.entity_id} must contain a source span"
        for span in entity.source_spans:
            if not _span_ok(span, payload):
                return f"entity {entity.entity_id} has an invalid source span: {_span_mismatch(span, payload)}"
    return None


def make_advisory_word_blueprint(requested_words: int, locked_beats: Sequence[str]) -> list[AdvisoryBeatBandV4]:
    if isinstance(requested_words, bool) or not isinstance(requested_words, int) or not 30 <= requested_words <= 900:
        raise SciFiGeminiTargetRangeError("requested_words must be 30..900")
    ids = list(locked_beats)
    if not ids:
        raise SciFiGeminiGraphError("locked beat list is empty")
    weights = [1.0 + ((i % 4) * 0.25) for i in range(len(ids))]
    total = sum(weights)
    raw = [requested_words * w / total for w in weights]
    floors = [int(x) for x in raw]
    remainder = requested_words - sum(floors)
    for i in sorted(range(len(ids)), key=lambda n: raw[n] - floors[n], reverse=True)[:remainder]:
        floors[i] += 1
    return [AdvisoryBeatBandV4(beat_id=i, advisory_word_center=n) for i, n in zip(ids, floors)]


def _source_grounded_all_caps(index: "FactIndexV4 | None") -> frozenset[str]:
    """All-caps tokens that already appear in the accepted source evidence."""
    if index is None:
        return frozenset()
    surfaces: list[str] = []
    for fact in index.facts:
        surfaces.extend(span.quote for span in fact.source_spans)
    for entity in index.entities:
        surfaces.extend(span.quote for span in entity.source_spans)
    for number in index.numbers:
        surfaces.append(number.source_span.quote)
    return frozenset(
        match.group(0)
        for surface in surfaces
        for match in _ALL_CAPS_RE.finditer(surface or "")
    )


def allowed_spoken_all_caps(
    index: "FactIndexV4 | None",
    cast: "Iterable[CastV4]",
) -> frozenset[str]:
    """All-caps tokens a character may legitimately say out loud.

    The outline seam ORDERS cast names in ALL CAPS, so a character addressing
    another by name speaks an all-caps token by construction. Add the acronyms
    the source itself uses (RNA, NASA) -- those are the words the episode is
    about. Everything else all-caps is shouting or a stray label.
    """
    allowed = set(_source_grounded_all_caps(index))
    for row in cast:
        allowed.update(_ALL_CAPS_RE.findall(row.name or ""))
    return frozenset(allowed)


def _spoken_error(
    text: str,
    speaker: str = "",
    allowed_all_caps: frozenset[str] = frozenset(),
) -> str | None:
    if not (text or "").strip():
        return "spoken text is empty"
    if _DECORATION_RE.search(text):
        return "spoken text contains forbidden decoration"
    shouted = [
        token for token in _ALL_CAPS_RE.findall(text)
        if token not in allowed_all_caps
    ]
    if shouted:
        return (
            "spoken text contains all-caps lexical text "
            f"({', '.join(sorted(set(shouted)))}); write it in normal case -- "
            "only a locked cast name or a source acronym may be all-caps"
        )
    if re.match(r"""^\s*["'].*["']\s*$""", text):
        return "spoken text is wholly quoted"
    if re.match(r"^\s*(?:ANNOUNCER|[A-Z][A-Za-z]+)\s*:", text):
        return "spoken text contains a role label"
    if speaker and re.match(r"^\s*" + re.escape(speaker.split()[0]) + r"\s*[,!:]", text, re.I):
        return "spoken text is self-vocative"
    if any(not x.strip(".,!?;:'’-") for x in text.split()):
        return "spoken text contains a non-lexical token"
    return None


def validate_scene_draft(
    draft: SceneDraftV4,
    scene: "SceneV4",
    cast_lock: Mapping[str, CastV4],
    allowed_all_caps: frozenset[str] = frozenset(),
) -> str | None:
    """The complete deterministic contract for one drafted scene.

    Coverage, graph, and spoken text in ONE post_validator, so the P4/P6 repair
    ladder sees every defect while it can still fix it. Running the spoken check
    after the ladder had already accepted the draft meant the only response left
    to a bad line was to kill the episode.
    """
    coverage = validate_scene_draft_covers_beats(draft, scene)
    if coverage is not None:
        return coverage
    beat_map = {beat.beat_id: beat for beat in scene.beats}
    for line in draft.lines:
        beat = beat_map.get(line.beat_id)
        if beat is None:
            return f"draft line {line.beat_id} is not in the locked outline"
        if beat.char_id not in cast_lock:
            return f"beat {beat.beat_id} has no locked cast row"
        err = _spoken_error(line.text, beat.speaker, allowed_all_caps)
        if err:
            return f"{line.beat_id}: {err}"
    return None


def stamp_music_skip_contract_after_set_lines(led: Any, music_line_ids: Sequence[str]) -> None:
    for row in led.data.get("lines", []):
        if row.get("line_id") in set(music_line_ids):
            row["skip"] = True
            row["text"] = ""
            row["tts_skip_reason"] = "music_cue"


def _prompt(pack: Any, seam: str, pass_id: str, inputs: Mapping[str, Any], result_type: type[BaseModel]) -> list[dict[str, str]]:
    seam_text = str((getattr(pack, "prompt_stages", {}) or {}).get(seam) or "")
    if not seam_text:
        raise SciFiGeminiPackContractError(f"missing Gemini seam {seam!r}")
    body = {"pass_id": pass_id, "typed_inputs": inputs, "result_json_schema": result_type.model_json_schema()}
    return [{"role": "system", "content": seam_text + _schema_instruction(result_type)}, {"role": "user", "content": json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)}]


class _PromptMustFitMessages(list[dict[str, str]]):
    """Tell the local slot wrapper to fail before it slices this prompt.

    Parity with the Codex lane: a provenance-bearing prompt (the source payload
    plus its schema contract) must never be silently left-truncated. Losing the
    system/schema prefix produces a confidently wrong artifact instead of an
    honest failure.
    """

    _otr_prompt_must_fit = True


def _prune_to_schema(data: Any, model: type[BaseModel]) -> None:
    """Drop keys a strict model forbids, recursively, in place.

    An unrequested key is NOT authored content: the contract never had a slot for
    it, so nothing the writer meant to say lives there. Removing it discards no
    story and invents nothing -- it just stops `extra="forbid"` from rejecting an
    otherwise complete artifact because the model garnished it.
    """
    if not isinstance(data, dict):
        return
    fields = model.model_fields
    for key in [k for k in data if k not in fields]:
        del data[key]
    for name, field in fields.items():
        if name not in data:
            continue
        nested = _nested_model(field.annotation)
        if nested is None:
            continue
        value = data[name]
        if isinstance(value, list):
            for item in value:
                _prune_to_schema(item, nested)
        else:
            _prune_to_schema(value, nested)


def _nested_model(annotation: Any) -> type[BaseModel] | None:
    """The BaseModel inside `X`, `list[X]`, `X | None`, ... or None."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in typing.get_args(annotation):
        found = _nested_model(arg)
        if found is not None:
            return found
    return None


def validate_scene_draft_covers_beats(
    draft: "SceneDraftV4", scene: "SceneV4",
) -> str | None:
    """Every beat in the scene must get exactly one drafted line.

    Nothing used to require this. The model quietly skipped a beat, the scene passed
    its critique, and the run died much later in `_assemble` with a bare
    "missing draft for b004" -- after all the generation was already paid for.

    A missing line is AUTHORED work: Python cannot invent the dialogue, so it must
    not try. Catch it at the pass that owns it, where the typed repair can ask the
    model to write the line it forgot, and name the exact beat.
    """
    expected = [beat.beat_id for beat in scene.beats]
    drafted = [line.beat_id for line in draft.lines]
    duplicates = sorted({b for b in drafted if drafted.count(b) > 1})
    if duplicates:
        return f"beats drafted more than once: {', '.join(duplicates)}"
    missing = [b for b in expected if b not in set(drafted)]
    if missing:
        return f"no drafted line for beat(s): {', '.join(missing)}"
    foreign = [b for b in drafted if b not in set(expected)]
    if foreign:
        return (
            f"drafted a line for beat(s) outside this scene: {', '.join(sorted(set(foreign)))}"
        )
    return None


def repair_forbidden_extra_keys(
    failed_output: str, result_type: type[BaseModel],
) -> BaseModel | None:
    """Drop keys the strict artifact forbids -- for ANY pass, not just the outline.

    The local model garnishes every artifact it writes. It hands P4 a `fact_ids`
    list on each drafted line because the outline's beats carry one, even though
    DraftLineV4 has no such field -- and `extra="forbid"` rejects the whole scene
    over it, dialogue and all.

    An unrequested key is not authored content: the contract never had a slot for
    it, so no part of the writer's work is lost by removing it. Everything the
    schema DOES ask for -- the dialogue, the fact uses, the beat ids -- is left
    byte-identical. Fails closed if the artifact is still invalid afterwards; a
    genuinely missing field is the model's job, not ours.
    """
    try:
        data = parse_first_json_object(failed_output)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    _prune_to_schema(data, result_type)
    try:
        return result_type.model_validate(data)
    except Exception:
        return None


def outline_output_token_budget(requested_words: int, beat_count: int) -> int:
    """Reserve P3 output WITHOUT eating the input budget its own repair needs.

    P3 reserved a flat 3600 tokens. The local context cap is 8192, so that leaves
    only 4592 for input -- and the typed-repair prompt (failed artifact + the
    validation error + the original request) is bigger than that. The guard then
    LEFT-TRUNCATED it (live: "PROMPT_GUARD: Truncated 5408 -> 4592"), silently
    cutting off the system/schema prefix. The model was not ignoring the repair
    rules; it never received them. Every P3 repair was doomed before it was sent.

    A 30-word outline does not need 3600 output tokens. Scale the reservation from
    what the artifact actually costs -- the dialogue steer plus the per-beat graph
    metadata -- so the repair prompt still fits inside the cap.
    """
    if (
        not isinstance(requested_words, int)
        or isinstance(requested_words, bool)
        or not 30 <= requested_words <= 900
    ):
        raise SciFiGeminiTargetRangeError("requested_words must be 30..900")
    beats = max(1, int(beat_count))
    return min(3600, max(2000, int(requested_words * 3.0) + 110 * beats + 400))


def validate_outline_cast_labels(outline: "OutlineV4") -> str | None:
    """The announcer is a fixed ROLE LABEL, not a character the writer invents.

    CastLock's Gate 1 invariants (_assert_unique_bark_voices /
    _assert_voice_preset_invariant) skip the announcer row by the EXACT string
    "ANNOUNCER". Gemini leaves CastV4.name free-form, so the model returns
    "Narrator" or "Announcer" -- and then those invariants judge the announcer's
    kokoro preset (bm_george) as an invalid Bark voice and kill the run with
    CastingFailedError, deep in the media tail, ~15 minutes in.

    Reject it at P3 instead, where the deterministic repair can normalize the
    label (and the beats that refer to it) without touching any story content.
    Sonnet pins this with a Literal; Codex validates and repairs it. Parity.
    """
    for row in outline.cast:
        if row.char_id == "announcer" and row.name != "ANNOUNCER":
            return (
                f"the announcer cast row must use the exact fixed name ANNOUNCER, "
                f"not {row.name!r}"
            )
    for scene in outline.scenes:
        for beat in scene.beats:
            if beat.char_id == "announcer" and beat.speaker != "ANNOUNCER":
                return (
                    f"beat {beat.beat_id} is spoken by the announcer and must name "
                    f"the speaker ANNOUNCER, not {beat.speaker!r}"
                )
    return None


def normalize_outline_graph_metadata(failed_output: str) -> dict[str, Any] | None:
    """Fill only the OutlineV4 fields the nesting already determines.

    A shot or beat written INSIDE scene ``s001`` belongs to ``s001`` -- its
    ``scene_id`` is not a creative decision, it is a fact of where the model put
    it. A beat's ``order`` is likewise just its position in the emitted graph.
    The local model reliably omits both (live: Gemini P3, 18 then 22 validation
    errors, 2026-07-11), so derive them from the parent rather than spending a
    call asking the model to restate what it has already said.

    Returns the normalized object even when it is still INVALID -- a shot may
    still be missing its ``visual_prompt``, which is authored work this function
    must never invent. Handing that nearly-complete object back to the creative
    repair (instead of the raw failure plus a 22-error wall, most of it mechanical
    noise) shrinks the model's remaining job to the part only it can do.

    Fails closed (None) when there is nothing authoritative to copy FROM.
    """
    try:
        data = parse_first_json_object(failed_output)
    except Exception:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("scenes"), list):
        return None

    # Strict models forbid extra keys, and the local model garnishes its output
    # with fields nobody asked for (live: 5 x extra_forbidden after the truncation
    # fix let it finally read the contract). An unrequested key is not authored
    # content -- the contract never had a place for it -- so dropping it invents
    # nothing and discards no story. Codex's script repair prunes exactly this way.
    _prune_to_schema(data, OutlineV4)

    # The announcer's NAME is a fixed role label, not a character the writer
    # invented -- the show's cast contract says it is "ANNOUNCER" (Sonnet pins it
    # with a Literal; Codex validates and repairs it). Gemini leaves CastV4.name
    # free-form, so the model happily returns "Narrator" or "Announcer". That is
    # not cosmetic: CastLock's Gate 1 invariants skip the announcer row by the
    # EXACT string "ANNOUNCER", so a differently-cased name makes them judge the
    # announcer's kokoro preset (bm_george) as an invalid Bark voice and kill the
    # run with CastingFailedError. Normalize the label and every beat that refers
    # to it, so the cast and the beats stay consistent.
    for row in data.get("cast") or []:
        if isinstance(row, dict) and row.get("char_id") == "announcer":
            row["name"] = "ANNOUNCER"

    beat_order = 0
    for scene in data["scenes"]:
        if not isinstance(scene, dict):
            return None
        scene_id = scene.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            # Without an owning scene id there is nothing authoritative to copy.
            return None
        for key in ("shots", "beats"):
            children = scene.get(key)
            if not isinstance(children, list):
                return None
            for child in children:
                if not isinstance(child, dict):
                    return None
                child["scene_id"] = scene_id
                if key == "beats":
                    beat_order += 1
                    child["order"] = beat_order
                    if child.get("char_id") == "announcer":
                        child["speaker"] = "ANNOUNCER"
    return data


def repair_outline_metadata(failed_output: str) -> OutlineV4 | None:
    """The metadata-only repair: accept ONLY if the graph normalization is enough.

    Never invents content. ``visual_prompt``, ``description``, ``intent``,
    ``mood``, dialogue -- anything a writer had to think of -- is left exactly as
    the model wrote it, and a missing one fails closed so the creative typed
    repair still has to author it.
    """
    data = normalize_outline_graph_metadata(failed_output)
    if data is None:
        return None
    try:
        return OutlineV4.model_validate(data)
    except Exception:
        return None


def invoke_gemini_structured(
    *, pass_id: str, slot: Literal["creative", "technical"], slot_fn: GenerateFn,
    seam_ref: str, pack: Any, typed_inputs: Mapping[str, Any],
    result_type: type[BaseModel], post_validator: Callable[[BaseModel], str | None],
    base_temperature: float, structural_retry_temperature: float,
    max_new_tokens: int, journal: MutableMapping[str, Any],
    prompt_must_fit: bool = False,
) -> BaseModel:
    prompt = _prompt(pack, seam_ref, pass_id, typed_inputs, result_type)
    if pass_id == "P0":
        prompt[0]["content"] += p0_contract_instruction(has_numeric_tokens=True)
    attempts: list[dict[str, Any]] = []
    def capture(messages, **kwargs):
        call_messages = (
            _PromptMustFitMessages(messages)
            if prompt_must_fit and isinstance(messages, list)
            else messages
        )
        raw = slot_fn(call_messages, **kwargs)
        original_raw = str(raw)
        attempts.append({
            "temperature": kwargs.get("temperature"),
            "max_new_tokens": kwargs.get("max_new_tokens"),
            "raw_chars": len(original_raw),
            "raw_sha256": hashlib.sha256(original_raw.encode("utf-8")).hexdigest(),
        })
        return raw
    def typed_repair_factory(*, original_prompt, failed_output, error):
        if pass_id == "P0":
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
            p0_envelope = json.loads(prompt[1]["content"])["typed_inputs"]["payload"]
            deterministic = repair_literal_source_metadata(
                failed_output,
                FactIndexV4,
                p0_envelope["payload"],
                zero_padded_ids=True,
                max_quote_chars=MAX_QUOTE_CHARS,
            )
            if deterministic is not None:
                return deterministic
        else:
            # An OutlineV4 failure splits in two. The MECHANICAL half (a nested
            # shot/beat's parent scene_id, a beat's order) is already implied by
            # where the model put the node, so repair it deterministically rather
            # than spending a call asking the model to restate itself. The
            # CREATIVE half (a shot's visual_prompt) is authored work Python must
            # never invent -- name it explicitly so the model writes it.
            # Compare by NAME, not identity: this package is importable as both
            # `nodes._otr_scifi_gemini` and `_otr_scifi_gemini` (see the try/except
            # ImportError at the top), so the same class can exist as two distinct
            # objects and `is` silently misses.
            if getattr(result_type, "__name__", "") == "OutlineV4":
                deterministic = repair_outline_metadata(failed_output)
                if deterministic is not None and post_validator(deterministic) is None:
                    log.info(
                        "[scifi_gemini:%s] deterministic OutlineV4 graph-metadata "
                        "repair accepted; no LLM repair call made", pass_id,
                    )
                    return deterministic
                # Not fully repairable -- authored content is still missing. Hand
                # the model the graph-NORMALIZED artifact and the single job only
                # it can do, instead of the raw failure plus a wall of validation
                # errors that is mostly mechanical noise we have already fixed.
                normalized = normalize_outline_graph_metadata(failed_output)
                if normalized is not None:
                    missing_shots = [
                        shot.get("shot_id")
                        for scene in normalized.get("scenes", [])
                        for shot in scene.get("shots", [])
                        if not str(shot.get("visual_prompt") or "").strip()
                    ]
                    if missing_shots:
                        log.info(
                            "[scifi_gemini:%s] graph metadata normalized; asking the "
                            "model only for %d missing visual_prompt(s)",
                            pass_id, len(missing_shots),
                        )
                        return [
                            {"role": "system", "content": prompt[0]["content"] + "\n" + (
                                "This artifact is already structurally complete: every "
                                "scene_id and beat order is correct. Change NOTHING else -- "
                                "keep every title, premise, scene, shot, beat, id, and order "
                                "exactly as given. Your ONLY job: write the missing "
                                "visual_prompt for each shot listed in missing_visual_prompts. "
                                "A visual_prompt is one concrete, filmable image of that shot "
                                "-- subject, setting, lighting -- never empty, never a copy of "
                                "the shot description. Return the COMPLETE object."
                            )},
                            {"role": "user", "content": json.dumps({
                                "artifact": normalized,
                                "missing_visual_prompts": missing_shots,
                            }, sort_keys=True, separators=(",", ":"), ensure_ascii=False)},
                        ]
            # Any strict artifact can be rejected purely for keys the model added
            # that the contract never asked for (P4 hands every drafted line a
            # `fact_ids` list DraftLineV4 has no slot for, and the whole scene --
            # dialogue included -- is thrown out over it). Pruning those loses no
            # authored work, so try it for EVERY pass before spending an LLM call.
            generic = repair_forbidden_extra_keys(failed_output, result_type)
            if generic is not None and post_validator(generic) is None:
                log.info(
                    "[scifi_gemini:%s] deterministic repair dropped forbidden extra "
                    "keys; no LLM repair call made", pass_id,
                )
                return generic
            repair_rules = (
                "This is a typed repair of the same artifact. Preserve the selected premise, "
                "scene outline, beats, and dialogue content; repair only fields named by the "
                "validation error. Every required nested graph field must be present; copy each "
                "beat speaker from the cast row matching its char_id. "
                "Return ONLY the keys the schema declares -- no extra fields. "
                "Copy the parent scene's scene_id into every shot and every beat nested inside "
                "it, and number beats with a strictly increasing order starting at 1. "
                "EVERY shot MUST carry a visual_prompt: one concrete, filmable image of that "
                "shot -- subject, setting, and lighting -- never an empty string and never a "
                "restatement of the shot description."
            )
        if pass_id == "P0":
            return [
                {"role": "system", "content": prompt[0]["content"] + "\n" + repair_rules},
                {
                    "role": "user",
                    "content": compact_p0_repair_context(
                        failed_artifact=failed_output,
                        rejection=str(error),
                        source_evidence=p0_envelope["payload"],
                        source_digest=p0_envelope["payload_sha256"],
                        allowed_source_fields=sorted(p0_envelope["payload"]),
                    ),
                },
            ]
        return [
            {"role": "system", "content": prompt[0]["content"] + "\n" + repair_rules},
            {"role": "user", "content": json.dumps({"failed_artifact": failed_output, "validation_error": str(error), "original_request": json.loads(prompt[1]["content"])}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)},
        ]
    try:
        # LLM slot: per-sub-pass injected creative/technical closure.
        result = structured_call(prompt=prompt, schema=result_type, slot_fn=capture, base_temperature=base_temperature, structural_retry_temperature=structural_retry_temperature, max_new_tokens=max_new_tokens, max_attempts=3, post_validator=post_validator, repair_prompt_factory=typed_repair_factory, clamp_overlong_strings=pass_id != "P0", helper_name=f"scifi_gemini:{pass_id}")
    except Exception as exc:
        raise SciFiGeminiPassError(f"{pass_id} failed: {exc}") from exc
    journal.setdefault("calls", []).append({"pass_id": pass_id, "slot": slot, "attempts": attempts, "accepted": result.model_dump(mode="json")})
    return result


class _GeminiTailFinalizer:
    def __init__(self, expected: Mapping[str, str]):
        self.expected = dict(expected)

    def _proof(self, data: Mapping[str, Any]) -> None:
        lane = data.get("meta", {}).get("scifi_gemini", {})
        hashes = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in self.expected.items()}
        if lane.get("line_text_sha256") != hashes:
            raise SciFiGeminiPreTailAuditError("Gemini text receipt mismatch")
        for row in data.get("lines", []):
            if row.get("line_id") in self.expected and row.get("text") != self.expected[row["line_id"]]:
                raise SciFiGeminiPreTailAuditError(f"Gemini line changed: {row.get('line_id')}")

    def before_save(self, *, ctx: Any) -> None:
        self._proof(ctx.led.data)
        pre = _otr_ledger_freeze.phase_0_gap_audit_pre(ctx.led)
        post = _otr_ledger_freeze.phase_10_gap_audit_post_and_freeze(ctx.led)
        # A WARNING IS NOT AN ERROR. This gate killed the episode on any warning at all --
        # it only ever passed because no warning happened to fire. Errors block; the
        # cascade's structural verdicts block; warnings are RECORDED and the record ships.
        # (frozen_with_warns is a CLEAN freeze: reviewer made no edits.)
        notes = list(pre.warnings) + list(post.warnings)
        if notes:
            log.warning(
                "[scifi_gemini] the freeze cascade raised %d warning(s); none is an "
                "error, so the record stands:\n  %s",
                len(notes), "\n  ".join(str(n) for n in notes),
            )
            ctx.led.data.setdefault("meta", {}).setdefault("scifi_gemini", {})[
                "freeze_notes"
            ] = [str(n) for n in notes]
        if pre.errors or post.errors:
            raise SciFiGeminiPreTailAuditError(
                "Gemini freeze proof has hard errors: "
                + "; ".join(str(e) for e in list(pre.errors) + list(post.errors))
            )
        verdict = ctx.led.data.get("meta", {}).get("freeze_verdict")
        if verdict not in ("frozen_clean", "frozen_with_warns"):
            raise SciFiGeminiPreTailAuditError(
                f"Gemini freeze verdict is {verdict!r} -- not a clean freeze"
            )

    def after_save(self, *, saved_path: str, ledger_data: Mapping[str, Any]) -> None:
        try:
            with open(saved_path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
        except Exception as exc:
            raise SciFiGeminiSavedLedgerAuditError(str(exc)) from exc
        report = _otr_ledger_freeze.run_gap_audit(saved, label="saved")
        # Errors and structural verdicts block; warnings are recorded. frozen_with_warns
        # IS a clean freeze (the reviewer made no edits). This gate only ever passed
        # because no warning happened to fire on the rolls we ran.
        if report.warnings:
            log.warning(
                "[scifi_gemini] the saved ledger carries %d warning(s); none is an error:"
                "\n  %s",
                len(report.warnings), "\n  ".join(str(w) for w in report.warnings),
            )
        if report.errors:
            raise SciFiGeminiSavedLedgerAuditError(
                "saved Gemini ledger has hard errors: "
                + "; ".join(str(e) for e in report.errors)
            )
        verdict = saved.get("meta", {}).get("freeze_verdict")
        if verdict not in ("frozen_clean", "frozen_with_warns"):
            raise SciFiGeminiSavedLedgerAuditError(
                f"saved Gemini ledger freeze verdict is {verdict!r} -- not a clean freeze"
            )
        self._proof(saved)


@dataclass
class GeminiTailParts:
    outline_view: Any
    canon: Any
    final_title_override: str
    run_story_spine: bool
    tail_finalizer: Any


def _build_gemini_episode_canon(outline: OutlineV4) -> EpisodeCanon:
    """Build the complete shared-tail canon from Gemini's accepted outline."""
    return EpisodeCanon(
        title=outline.title,
        premise=outline.premise,
        setting=outline.setting,
        time_of_day=outline.time_of_day,
        sound_palette=[],
    )


def _assemble(led: Any, outline: OutlineV4, drafts: Mapping[str, SceneDraftV4], meta: MutableMapping[str, Any]) -> dict[str, str]:
    voice_map = {"announcer": ("kokoro", "bm_george"), "c01": ("bark", "v2/en_speaker_6"), "c02": ("bark", "v2/en_speaker_3"), "c03": ("bark", "v2/en_speaker_0")}
    led.set_cast([{"char_id": c.char_id, "name": c.name, "character_description": c.character_description, "gender": c.gender, "tts_model": voice_map[c.char_id][0], "voice_preset": voice_map[c.char_id][1]} for c in outline.cast])
    led.set_scenes([{"scene_id": s.scene_id, "description": s.description, "env": s.env} for s in outline.scenes])
    led.set_shots([{"shot_id": x.shot_id, "scene_id": x.scene_id, "description": x.description, "visual_prompt": x.visual_prompt} for s in outline.scenes for x in s.shots])
    beat_rows = []
    line_rows = []
    expected: dict[str, str] = {}
    all_music = {x.cue_id: x for x in outline.music_cues}
    for scene in outline.scenes:
        draft_by_beat = {x.beat_id: x for x in drafts[scene.scene_id].lines}
        for beat in scene.beats:
            line = draft_by_beat.get(beat.beat_id)
            if line is None:
                raise SciFiGeminiGraphError(f"missing draft for {beat.beat_id}")
            beat_rows.append({"beat_id": beat.beat_id, "shot_id": beat.shot_id, "scene_id": beat.scene_id, "speaker": beat.speaker, "char_id": beat.char_id, "line_ids": [beat.line_id]})
            line_rows.append({"line_id": beat.line_id, "beat_id": beat.beat_id, "shot_id": beat.shot_id, "char_id": beat.char_id, "speaker_role": beat.speaker_role, "text": line.text, "boundary": "beat_start", "traits": beat.mood, "arc_phase": "rising", "beat_intent": beat.intent, "dialogue_slot_id": beat.line_id})
            expected[beat.line_id] = line.text
    music_ids = []
    first_scene = outline.scenes[0]
    first_shot = first_scene.shots[0]
    for cue in outline.music_cues:
        bid = f"{cue.cue_id}_beat"
        lid = f"{cue.cue_id}_line"
        beat_rows.append({"beat_id": bid, "shot_id": first_shot.shot_id, "scene_id": first_scene.scene_id, "speaker": cue.cue_id, "char_id": cue.cue_id, "line_ids": [lid]})
        line_rows.append({"line_id": lid, "beat_id": bid, "shot_id": first_shot.shot_id, "char_id": cue.cue_id, "speaker_role": cue.cue_id, "text": "", "boundary": "continue", "arc_phase": "closing", "beat_intent": cue.description, "dialogue_slot_id": lid})
        music_ids.append(lid)
    led.set_beats(beat_rows)
    led.set_lines(line_rows)
    stamp_music_skip_contract_after_set_lines(led, music_ids)
    led.set_music([{"cue_id": c.cue_id, "placement": c.placement, "description": c.description, "generation_prompt": c.generation_prompt, "anchor_line_id": next((b.line_id for s in outline.scenes for b in s.beats if b.beat_id == c.anchor_beat_id), None)} for c in outline.music_cues])
    led.data["clips"] = []
    stamp_word_counts(led)
    meta["scifi_gemini"]["line_text_sha256"] = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in expected.items()}
    meta["scifi_gemini"]["accepted_lines"] = dict(expected)
    return expected


def run_scifi_gemini_episode(
    *, payload: dict[str, str], pack: Any, resolved: Mapping[str, Any], led: Any,
    meta: dict[str, Any], creative_fn: GenerateFn, technical_fn: GenerateFn,
    slot_scheduler: Any, source_bank_row: Any, story_rules: Mapping[str, Any],
    episode_root: Path, episode_id: str,
) -> GeminiTailParts:
    del slot_scheduler, source_bank_row, story_rules, episode_root, episode_id
    envelope = validate_gemini_payload(payload, resolved)
    lane_meta = {"source_digest": envelope.payload_sha256, "source_mode": envelope.source_mode, "call_journal": {}}
    meta["scifi_gemini"] = lane_meta
    journal = lane_meta["call_journal"]
    p0_token_budget = p0_output_token_budget()
    journal["fact_index_token_budget"] = {
        **p0_contract_receipt(),
        "source_evidence_field_count": len(envelope.payload),
        "source_evidence_characters": sum(
            len(value) for value in envelope.payload.values()
        ),
    }
    p0 = invoke_gemini_structured(pass_id="P0", slot="technical", slot_fn=technical_fn, seam_ref="gemini_fact_extraction", pack=pack, typed_inputs={"payload": envelope.model_dump(mode="json")}, result_type=FactIndexV4, post_validator=lambda x: _fact_validator(x, payload), base_temperature=.22, structural_retry_temperature=.12, max_new_tokens=p0_token_budget, journal=journal, prompt_must_fit=True)
    p1 = invoke_gemini_structured(pass_id="P1", slot="creative", slot_fn=creative_fn, seam_ref="gemini_pitch_generation", pack=pack, typed_inputs={"facts": p0.model_dump(mode="json")}, result_type=PitchSlateV4, post_validator=lambda x: None, base_temperature=.72, structural_retry_temperature=.36, max_new_tokens=1400, journal=journal)
    p2 = invoke_gemini_structured(pass_id="P2", slot="technical", slot_fn=technical_fn, seam_ref="gemini_pitch_critique", pack=pack, typed_inputs={"pitches": p1.model_dump(mode="json")}, result_type=PitchSelectionV4, post_validator=lambda x: None, base_temperature=.22, structural_retry_temperature=.12, max_new_tokens=700, journal=journal)
    ids = [f"b{i:03d}" for i in range(1, 7)]
    bands = make_advisory_word_blueprint(int(resolved["target_words"]), ids)
    p3 = invoke_gemini_structured(pass_id="P3", slot="creative", slot_fn=creative_fn, seam_ref="gemini_scene_outline", pack=pack, typed_inputs={"chosen_premise": p1.pitches[p2.selected_index].model_dump(mode="json"), "initial_outline_word_steer": {"requested_words": int(resolved["target_words"])}, "advisory_word_bands": [x.model_dump(mode="json") for x in bands]}, result_type=OutlineV4, post_validator=validate_outline_cast_labels, base_temperature=.68, structural_retry_temperature=.30, max_new_tokens=outline_output_token_budget(int(resolved["target_words"]), len(bands)), journal=journal, prompt_must_fit=True)
    casts = {c.char_id: c for c in p3.cast}
    # The outline seam orders cast names in ALL CAPS, so a line that addresses a
    # character by name is all-caps by construction. Give the spoken validator
    # the vocabulary it needs to tell a name from a shout.
    allowed_caps = allowed_spoken_all_caps(p0, p3.cast)
    drafts: dict[str, SceneDraftV4] = {}
    for scene in p3.scenes:
        draft = invoke_gemini_structured(pass_id=f"P4:{scene.scene_id}", slot="creative", slot_fn=creative_fn, seam_ref="gemini_scene_draft", pack=pack, typed_inputs={"scene_outline": scene.model_dump(mode="json"), "facts": p0.model_dump(mode="json")}, result_type=SceneDraftV4, post_validator=lambda x, s=scene: validate_scene_draft(x, s, casts, allowed_caps), base_temperature=.74, structural_retry_temperature=.34, max_new_tokens=3000, journal=journal)
        critique = invoke_gemini_structured(pass_id=f"P5:{scene.scene_id}", slot="technical", slot_fn=technical_fn, seam_ref="gemini_scene_critique", pack=pack, typed_inputs={"drafted_lines": draft.model_dump(mode="json"), "scene_outline": scene.model_dump(mode="json"), "facts": p0.model_dump(mode="json")}, result_type=SceneCritiqueV4, post_validator=lambda x: (("clean critique must have empty feedback" if x.passed and x.feedback else None) or ("failed critique needs feedback" if not x.passed and not x.feedback else None)), base_temperature=.20, structural_retry_temperature=.10, max_new_tokens=1400, journal=journal)
        if not critique.passed:
            # The critique buys ONE bounded rewrite. It does not get a veto: the
            # rewritten draft has already cleared the same deterministic contract
            # as the first, and a critic that is still unhappy is an opinion.
            draft = invoke_gemini_structured(pass_id=f"P6:{scene.scene_id}", slot="creative", slot_fn=creative_fn, seam_ref="gemini_scene_rewrite", pack=pack, typed_inputs={"feedback": critique.feedback, "previous_draft": draft.model_dump(mode="json"), "facts": p0.model_dump(mode="json"), "scene_outline": scene.model_dump(mode="json")}, result_type=SceneDraftV4, post_validator=lambda x, s=scene: validate_scene_draft(x, s, casts, allowed_caps), base_temperature=.62, structural_retry_temperature=.28, max_new_tokens=3000, journal=journal)
            journal[f"rewrite:{scene.scene_id}"] = {"critique": critique.feedback}
        drafts[scene.scene_id] = draft
    expected = _assemble(led, p3, drafts, meta)
    from ._otr_content_authorship import stamp_receipt
    stamp_receipt(
        led.data, owner_bank="scifi_gemini",
        accepted_artifacts={
            f"final_scene:{scene_id}": draft
            for scene_id, draft in drafts.items()
        },
    )
    actual = sum(len(_WORD_RE.findall(x)) for x in expected.values())
    meta["scifi_gemini"]["word_receipt"] = {"requested_words": int(resolved["target_words"]), "actual_split_words": actual, "actual_ledger_word_count": int(led.data.get("total_word_count") or 0)}
    meta["scifi_gemini"]["fact_index"] = p0.model_dump(mode="json")
    canon = _build_gemini_episode_canon(p3)
    return GeminiTailParts(
        outline_view=SimpleNamespace(
            title=p3.title,
            premise=p3.premise,
            setting=p3.setting,
            time_of_day=canon.time_of_day,
        ),
        canon=canon,
        final_title_override=p3.title,
        run_story_spine=False,
        tail_finalizer=_GeminiTailFinalizer(expected),
    )
