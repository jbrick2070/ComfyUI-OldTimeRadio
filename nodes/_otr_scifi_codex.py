"""Sci-Fi Codex v4 additive source-bank runner.

The lane owns its schemas, prompt seams, provenance graph, and ledger assembly.
It never fetches a source, loads a model, or edits LLM-authored dialogue.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal, Mapping, MutableMapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    from ._otr_source_payload import validate_source_payload
    from ._otr_scifi_source_repair import repair_literal_source_metadata
    from ._otr_structured_call import schema_shape_instruction, structured_call
    from . import _otr_ledger_freeze
    from .production_ledger import stamp_word_counts
except ImportError:  # pragma: no cover
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
    dialogue_slot_id: str
    fact_ids: list[str] = []


class ScriptArtifactV4(_Strict):
    schema_version: Literal["scifi_codex.script_artifact.v4"]
    title: str
    scenes: list[dict[str, Any]] = Field(min_length=1)
    lines: list[ScriptLineV4] = Field(min_length=1)
    music_cues: list[MusicCueV4] = Field(min_length=1)


class StructureReviewV4(_Strict):
    verdict: Literal["pass", "rewrite"]
    issues: list[str] = []
    rationale: str = ""


class ListenerReviewV4(_Strict):
    strengths: list[str] = []
    issues: list[dict[str, str]] = []
    require_full_retake: bool = True


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
    observed_word_counts: dict[str, int]


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


def _span_ok(span: SourceSpanV4, payload: Mapping[str, str]) -> bool:
    return span.quote == payload.get(span.field, "")[span.start:span.end]


def _span_mismatch(span: SourceSpanV4, payload: Mapping[str, str]) -> str:
    expected = payload.get(span.field, "")[span.start:span.end]
    return (
        f"{span.field}[{span.start}:{span.end}] expected exact slice "
        f"{expected[:300]!r}; returned quote {span.quote[:300]!r}"
    )


def _validate_fact_index(index: FactIndexV4, payload: Mapping[str, str]) -> str | None:
    fact_ids = {f.fact_id for f in index.facts}
    for fact in index.facts:
        if not fact.source_spans:
            return f"fact {fact.fact_id} must contain at least one source span"
        for span in fact.source_spans:
            if not _span_ok(span, payload):
                return f"fact {fact.fact_id} has a non-literal source span: {_span_mismatch(span, payload)}"
    for number in index.numbers:
        if number.fact_id not in fact_ids:
            return f"number {number.number_id} does not resolve to a literal fact/span"
        if not _span_ok(number.source_span, payload):
            return f"number {number.number_id} has a non-literal source span: {_span_mismatch(number.source_span, payload)}"
    for entity in index.entities:
        if not entity.source_spans:
            return f"entity {entity.entity_id} must contain at least one source span"
        for span in entity.source_spans:
            if not _span_ok(span, payload):
                return f"entity {entity.entity_id} has a non-literal source span: {_span_mismatch(span, payload)}"
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
        if row.char_id != "announcer" and (not re.fullmatch(r"[A-Z][a-z]+", row.name) or " " in row.name):
            raise CodexSpokenTextError(f"cast name {row.name!r} is not one canonical Title-Case token")
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


def invoke_codex_structured(
    *, pass_id: str, slot: Literal["creative", "technical"], slot_fn: GenerateFn,
    pack: Any, seam_refs: tuple[str, ...], artifact_inputs: Mapping[str, Any],
    result_type: type[BaseModel], post_validator: Callable[[BaseModel], str | None],
    base_temperature: float, structural_retry_temperature: float,
    max_new_tokens: int, call_journal: MutableMapping[str, Any],
) -> BaseModel:
    if not seam_refs:
        raise CodexPackContractError(f"{pass_id} has no prompt seam")
    seams = []
    for seam in seam_refs:
        text = str((getattr(pack, "prompt_stages", {}) or {}).get(seam) or "")
        if not text:
            raise CodexPackContractError(f"missing Codex seam {seam!r}")
        seams.append(text)
    body = {"pass_id": pass_id, "artifact_inputs": artifact_inputs, "result_json_schema": result_type.model_json_schema()}
    schema_instruction = _schema_instruction(result_type)
    messages = [{"role": "system", "content": "\n".join(seams) + schema_instruction}, {"role": "user", "content": json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)}]
    calls: list[dict[str, Any]] = []
    def capture(messages_in, **kwargs):
        raw = slot_fn(messages_in, **kwargs)
        calls.append({"temperature": kwargs.get("temperature"), "raw_sha256": hashlib.sha256(str(raw).encode("utf-8")).hexdigest()})
        return raw
    def typed_repair_factory(*, original_prompt, failed_output, error):
        detail = str(error)
        if pass_id == "P0":
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
                return deterministic
        else:
            repair_rules = (
                "This is a typed repair of the same artifact. Preserve the existing premise, "
                "scene descriptions, beats, and content; repair only the fields named by the "
                "validation error. Every required nested graph field must be present. Copy "
                "parent scene_id into each shot and beat, copy a valid shot_id into each beat, "
                "copy each beat's speaker from the cast row matching its char_id, and provide "
                "every required visual_prompt without dropping existing content."
            )
        return [
            {"role": "system", "content": "\n".join(seams) + schema_instruction + "\n" + repair_rules},
            {"role": "user", "content": json.dumps({"failed_artifact": failed_output, "validation_error": detail, "original_request": body}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)},
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
        if pre.errors or pre.warnings or post.errors or post.warnings:
            raise CodexPreTailAuditError("Codex ledger freeze is not warning-free")
        if ctx.led.data.get("meta", {}).get("freeze_verdict") != "frozen_clean":
            raise CodexPreTailAuditError("Codex ledger did not reach frozen_clean")

    def after_save(self, *, saved_path: str, ledger_data: Mapping[str, Any]) -> None:
        try:
            with open(saved_path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
        except Exception as exc:
            raise CodexSavedLedgerAuditError(f"cannot reopen saved ledger: {exc}") from exc
        report = _otr_ledger_freeze.run_gap_audit(saved, label="saved")
        if report.errors or report.warnings or saved.get("meta", {}).get("freeze_verdict") != "frozen_clean":
            raise CodexSavedLedgerAuditError("saved ledger is not warning-free and frozen_clean")
        self._proof(saved)


@dataclass
class CodexTailParts:
    outline_view: Any
    canon: Any
    final_title_override: str
    run_story_spine: bool
    tail_finalizer: Any


def _validate_script_post(script: ScriptArtifactV4, cast: CastPlanV4, score: RadioScoreV4) -> str | None:
    try:
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
    lane_meta: dict[str, Any] = {"source_digest": env.source_digest, "source_mode": env.source_mode, "call_journal": {}}
    meta["scifi_codex"] = lane_meta
    journal = lane_meta["call_journal"]
    p0 = invoke_codex_structured(pass_id="P0", slot="technical", slot_fn=technical_fn, pack=pack, seam_refs=("codex_fact_index_system",), artifact_inputs={"payload": env.model_dump(mode="json")}, result_type=FactIndexV4, post_validator=lambda x: _validate_fact_index(x, payload), base_temperature=.20, structural_retry_temperature=.10, max_new_tokens=2000, call_journal=journal)
    p1 = invoke_codex_structured(pass_id="P1", slot="creative", slot_fn=creative_fn, pack=pack, seam_refs=("codex_question_system",), artifact_inputs={"fact_index": p0.model_dump(mode="json")}, result_type=DramaticQuestionV4, post_validator=lambda x: None, base_temperature=.72, structural_retry_temperature=.32, max_new_tokens=1800, call_journal=journal)
    p2 = invoke_codex_structured(pass_id="P2", slot="creative", slot_fn=creative_fn, pack=pack, seam_refs=("codex_pressure_cast_system",), artifact_inputs={"question": p1.model_dump(mode="json")}, result_type=CastPlanV4, post_validator=lambda x: None, base_temperature=.72, structural_retry_temperature=.32, max_new_tokens=1600, call_journal=journal)
    beat_ids = [f"b{i:03d}" for i in range(max(3, min(12, len(p2.cast) * 3)))]
    advisory = make_advisory_word_blueprint(steer.requested_words, beat_ids)
    p3 = invoke_codex_structured(pass_id="P3", slot="creative", slot_fn=creative_fn, pack=pack, seam_refs=("codex_radio_score_system", "codex_coda_contract_system"), artifact_inputs={"question": p1.model_dump(mode="json"), "cast": p2.model_dump(mode="json"), "advisory_word_plan": advisory.model_dump(mode="json")}, result_type=RadioScoreV4, post_validator=lambda x: None, base_temperature=.72, structural_retry_temperature=.32, max_new_tokens=3600, call_journal=journal)
    review = invoke_codex_structured(pass_id="P4", slot="technical", slot_fn=technical_fn, pack=pack, seam_refs=("codex_radio_score_system", "codex_coda_contract_system"), artifact_inputs={"score": p3.model_dump(mode="json")}, result_type=StructureReviewV4, post_validator=lambda x: None, base_temperature=.20, structural_retry_temperature=.10, max_new_tokens=1800, call_journal=journal)
    score = p3
    if review.verdict == "rewrite":
        score = invoke_codex_structured(pass_id="P3_rewrite", slot="creative", slot_fn=creative_fn, pack=pack, seam_refs=("codex_radio_score_system", "codex_coda_contract_system"), artifact_inputs={"score": p3.model_dump(mode="json"), "review": review.model_dump(mode="json")}, result_type=RadioScoreV4, post_validator=lambda x: None, base_temperature=.55, structural_retry_temperature=.20, max_new_tokens=3600, call_journal=journal)
    script = invoke_codex_structured(pass_id="P5", slot="creative", slot_fn=creative_fn, pack=pack, seam_refs=("codex_play_system", "codex_coda_contract_system"), artifact_inputs={"score": score.model_dump(mode="json"), "fact_index": p0.model_dump(mode="json"), "initial_draft_word_steer": steer.model_dump(mode="json")}, result_type=ScriptArtifactV4, post_validator=lambda x: _validate_script_post(x, p2, score), base_temperature=.78, structural_retry_temperature=.35, max_new_tokens=6500, call_journal=journal)
    listener = invoke_codex_structured(pass_id="P6", slot="technical", slot_fn=technical_fn, pack=pack, seam_refs=("codex_listening_room_system",), artifact_inputs={"script": script.model_dump(mode="json"), "score": score.model_dump(mode="json")}, result_type=ListenerReviewV4, post_validator=lambda x: None, base_temperature=.20, structural_retry_temperature=.10, max_new_tokens=2200, call_journal=journal)
    script = invoke_codex_structured(pass_id="P7", slot="creative", slot_fn=creative_fn, pack=pack, seam_refs=("codex_retake_system", "codex_coda_contract_system"), artifact_inputs={"previous": script.model_dump(mode="json"), "review": listener.model_dump(mode="json"), "score": score.model_dump(mode="json")}, result_type=ScriptArtifactV4, post_validator=lambda x: _validate_script_post(x, p2, score), base_temperature=.68, structural_retry_temperature=.30, max_new_tokens=6500, call_journal=journal)
    audit = invoke_codex_structured(pass_id="P8", slot="technical", slot_fn=technical_fn, pack=pack, seam_refs=("codex_final_audit_system", "codex_coda_contract_system"), artifact_inputs={"script": script.model_dump(mode="json"), "fact_index": p0.model_dump(mode="json")}, result_type=FinalAuditV4, post_validator=lambda x: None, base_temperature=.20, structural_retry_temperature=.10, max_new_tokens=2400, call_journal=journal)
    if audit.verdict == "rewrite":
        script = invoke_codex_structured(pass_id="P9", slot="creative", slot_fn=creative_fn, pack=pack, seam_refs=("codex_retake_system", "codex_play_system", "codex_coda_contract_system"), artifact_inputs={"previous": script.model_dump(mode="json"), "audit": audit.model_dump(mode="json")}, result_type=ScriptArtifactV4, post_validator=lambda x: _validate_script_post(x, p2, score), base_temperature=.68, structural_retry_temperature=.30, max_new_tokens=6500, call_journal=journal)
    validate_spoken_text_and_roster(script, p2, score)
    expected = _assemble_ledger(led, score, p2, script, meta)
    actual = sum(_words(v) for v in expected.values())
    meta["scifi_codex"]["word_receipt"] = {"requested_words": steer.requested_words, "actual_split_words": actual, "actual_ledger_word_count": int(led.data.get("total_word_count") or 0)}
    meta["scifi_codex"]["fact_index"] = p0.model_dump(mode="json")
    meta["scifi_codex"]["script_digest"] = _script_digest(script)
    return CodexTailParts(
        outline_view=SimpleNamespace(title=script.title, premise=p1.question, setting=score.setting),
        canon=SimpleNamespace(title=script.title, premise=p1.question),
        final_title_override=script.title,
        run_story_spine=False,
        tail_finalizer=_CodexTailFinalizer(expected),
    )
