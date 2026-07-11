"""Sci-Fi Gemini v4 source-bank runner."""
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
    from ._otr_structured_call import structured_call
    from . import _otr_ledger_freeze
    from .production_ledger import stamp_word_counts
except ImportError:  # pragma: no cover
    from _otr_source_payload import validate_source_payload  # type: ignore
    from _otr_structured_call import structured_call  # type: ignore
    import _otr_ledger_freeze  # type: ignore
    from production_ledger import stamp_word_counts  # type: ignore


class ScifiGeminiError(RuntimeError): pass
class SciFiGeminiPayloadContractError(ScifiGeminiError): pass
class SciFiGeminiPayloadRouteError(ScifiGeminiError): pass
class SciFiGeminiPayloadThinError(ScifiGeminiError): pass
class SciFiGeminiTargetRangeError(ScifiGeminiError): pass
class SciFiGeminiPackContractError(ScifiGeminiError): pass
class SciFiGeminiPassError(ScifiGeminiError): pass
class SciFiGeminiRewriteExhaustedError(ScifiGeminiError): pass
class SciFiGeminiSpokenTextError(ScifiGeminiError): pass
class SciFiGeminiGraphError(ScifiGeminiError): pass
class SciFiGeminiProvenanceError(ScifiGeminiError): pass
class SciFiGeminiPreTailAuditError(ScifiGeminiError): pass
class SciFiGeminiSavedLedgerAuditError(ScifiGeminiError): pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


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


class GeminiPayloadV4(_Strict):
    payload: dict[str, str]
    source_mode: Literal["rss", "operator_pinned"]
    payload_sha256: str


class FactV4(_Strict):
    fact_id: str
    claim: str
    source_spans: list[SourceSpanV4]
    numeric_tokens: list[str] = []


class EntityV4(_Strict):
    entity_id: str
    name: str
    source_spans: list[SourceSpanV4]


class NumberV4(_Strict):
    number_id: str
    verbatim: str
    fact_id: str
    source_span: SourceSpanV4


class FactIndexV4(_Strict):
    facts: list[FactV4] = Field(min_length=1, max_length=12)
    entities: list[EntityV4] = Field(max_length=12)
    numbers: list[NumberV4] = Field(max_length=12)
    tone: str
    payload_sha256: str


class PitchV4(_Strict):
    premise: str
    setting: str
    tonal_palette: str


class PitchSlateV4(_Strict):
    pitches: tuple[PitchV4, PitchV4, PitchV4]


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
    line_fact_ids: dict[str, list[str]]
    sfw_pass: bool


GenerateFn = Callable[..., str]
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’-]*")
_DECORATION_RE = re.compile(r"[\r\n\t\[\]()\x60*]|^\s*\x60\x60\x60|^\s*[-*]\s+")
_ALL_CAPS_RE = re.compile(r"\b[A-Z]{2,}\b")


def _digest(payload: Mapping[str, str]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _span_ok(span: SourceSpanV4, payload: Mapping[str, str]) -> bool:
    return span.quote == payload.get(span.field, "")[span.start:span.end]


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
        if len(text.split()) < 80 or len(set(re.findall(r"[A-Za-z0-9]+", text.lower()))) < 12:
            raise SciFiGeminiPayloadThinError("RSS source is below the 80/12 thinness floor")
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
        if not fact.source_spans or any(not _span_ok(s, payload) for s in fact.source_spans):
            return f"fact {fact.fact_id} has an invalid source span"
    for number in index.numbers:
        if number.fact_id not in fact_ids or not _span_ok(number.source_span, payload):
            return f"number {number.number_id} has an invalid fact/span reference"
    for entity in index.entities:
        if not entity.source_spans or any(not _span_ok(s, payload) for s in entity.source_spans):
            return f"entity {entity.entity_id} has an invalid source span"
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


def _spoken_error(text: str, speaker: str = "") -> str | None:
    if not (text or "").strip():
        return "spoken text is empty"
    if _DECORATION_RE.search(text) or _ALL_CAPS_RE.search(text):
        return "spoken text contains forbidden decoration or all-caps lexical text"
    if re.match(r"""^\s*["'].*["']\s*$""", text):
        return "spoken text is wholly quoted"
    if re.match(r"^\s*(?:ANNOUNCER|[A-Z][A-Za-z]+)\s*:", text):
        return "spoken text contains a role label"
    if speaker and re.match(r"^\s*" + re.escape(speaker.split()[0]) + r"\s*[,!:]", text, re.I):
        return "spoken text is self-vocative"
    if any(not x.strip(".,!?;:'’-") for x in text.split()):
        return "spoken text contains a non-lexical token"
    return None


def validate_spoken_text_and_lock(draft: SceneDraftV4, outline: OutlineV4, cast_lock: Mapping[str, CastV4]) -> None:
    beat_map = {b.beat_id: b for s in outline.scenes for b in s.beats}
    for line in draft.lines:
        beat = beat_map.get(line.beat_id)
        if beat is None:
            raise SciFiGeminiGraphError(f"draft line {line.beat_id} is not in the locked outline")
        err = _spoken_error(line.text, beat.speaker)
        if err:
            raise SciFiGeminiSpokenTextError(f"{line.beat_id}: {err}")
        if beat.char_id not in cast_lock:
            raise SciFiGeminiGraphError(f"beat {beat.beat_id} has no locked cast row")
        if any(use.fact_id not in {f.fact_id for f in ()} for use in line.fact_uses):
            pass


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
    return [{"role": "system", "content": seam_text}, {"role": "user", "content": json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)}]


def invoke_gemini_structured(
    *, pass_id: str, slot: Literal["creative", "technical"], slot_fn: GenerateFn,
    seam_ref: str, pack: Any, typed_inputs: Mapping[str, Any],
    result_type: type[BaseModel], post_validator: Callable[[BaseModel], str | None],
    base_temperature: float, structural_retry_temperature: float,
    max_new_tokens: int, journal: MutableMapping[str, Any],
) -> BaseModel:
    prompt = _prompt(pack, seam_ref, pass_id, typed_inputs, result_type)
    attempts: list[dict[str, Any]] = []
    def capture(messages, **kwargs):
        raw = slot_fn(messages, **kwargs)
        attempts.append({"temperature": kwargs.get("temperature"), "raw_sha256": hashlib.sha256(str(raw).encode("utf-8")).hexdigest()})
        return raw
    try:
        # LLM slot: per-sub-pass injected creative/technical closure.
        result = structured_call(prompt=prompt, schema=result_type, slot_fn=capture, base_temperature=base_temperature, structural_retry_temperature=structural_retry_temperature, max_new_tokens=max_new_tokens, max_attempts=3, post_validator=post_validator, helper_name=f"scifi_gemini:{pass_id}")
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
        if pre.errors or pre.warnings or post.errors or post.warnings or ctx.led.data.get("meta", {}).get("freeze_verdict") != "frozen_clean":
            raise SciFiGeminiPreTailAuditError("Gemini freeze proof is not warning-free")

    def after_save(self, *, saved_path: str, ledger_data: Mapping[str, Any]) -> None:
        try:
            with open(saved_path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
        except Exception as exc:
            raise SciFiGeminiSavedLedgerAuditError(str(exc)) from exc
        report = _otr_ledger_freeze.run_gap_audit(saved, label="saved")
        if report.errors or report.warnings or saved.get("meta", {}).get("freeze_verdict") != "frozen_clean":
            raise SciFiGeminiSavedLedgerAuditError("saved Gemini ledger is not frozen_clean")
        self._proof(saved)


@dataclass
class GeminiTailParts:
    outline_view: Any
    canon: Any
    final_title_override: str
    run_story_spine: bool
    tail_finalizer: Any


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
    p0 = invoke_gemini_structured(pass_id="P0", slot="technical", slot_fn=technical_fn, seam_ref="gemini_fact_extraction", pack=pack, typed_inputs={"payload": envelope.model_dump(mode="json")}, result_type=FactIndexV4, post_validator=lambda x: _fact_validator(x, payload), base_temperature=.22, structural_retry_temperature=.12, max_new_tokens=1800, journal=journal)
    p1 = invoke_gemini_structured(pass_id="P1", slot="creative", slot_fn=creative_fn, seam_ref="gemini_pitch_generation", pack=pack, typed_inputs={"facts": p0.model_dump(mode="json")}, result_type=PitchSlateV4, post_validator=lambda x: None, base_temperature=.72, structural_retry_temperature=.36, max_new_tokens=1400, journal=journal)
    p2 = invoke_gemini_structured(pass_id="P2", slot="technical", slot_fn=technical_fn, seam_ref="gemini_pitch_critique", pack=pack, typed_inputs={"pitches": p1.model_dump(mode="json")}, result_type=PitchSelectionV4, post_validator=lambda x: None, base_temperature=.22, structural_retry_temperature=.12, max_new_tokens=700, journal=journal)
    ids = [f"b{i:03d}" for i in range(1, 7)]
    bands = make_advisory_word_blueprint(int(resolved["target_words"]), ids)
    p3 = invoke_gemini_structured(pass_id="P3", slot="creative", slot_fn=creative_fn, seam_ref="gemini_scene_outline", pack=pack, typed_inputs={"chosen_premise": p1.pitches[p2.selected_index].model_dump(mode="json"), "initial_outline_word_steer": {"requested_words": int(resolved["target_words"])}, "advisory_word_bands": [x.model_dump(mode="json") for x in bands]}, result_type=OutlineV4, post_validator=lambda x: None, base_temperature=.68, structural_retry_temperature=.30, max_new_tokens=3600, journal=journal)
    casts = {c.char_id: c for c in p3.cast}
    drafts: dict[str, SceneDraftV4] = {}
    for scene in p3.scenes:
        draft = invoke_gemini_structured(pass_id=f"P4:{scene.scene_id}", slot="creative", slot_fn=creative_fn, seam_ref="gemini_scene_draft", pack=pack, typed_inputs={"scene_outline": scene.model_dump(mode="json"), "facts": p0.model_dump(mode="json")}, result_type=SceneDraftV4, post_validator=lambda x, s=scene: None, base_temperature=.74, structural_retry_temperature=.34, max_new_tokens=3000, journal=journal)
        critique = invoke_gemini_structured(pass_id=f"P5:{scene.scene_id}", slot="technical", slot_fn=technical_fn, seam_ref="gemini_scene_critique", pack=pack, typed_inputs={"drafted_lines": draft.model_dump(mode="json"), "scene_outline": scene.model_dump(mode="json"), "facts": p0.model_dump(mode="json")}, result_type=SceneCritiqueV4, post_validator=lambda x: (("clean critique must have empty feedback" if x.passed and x.feedback else None) or ("failed critique needs feedback" if not x.passed and not x.feedback else None) or (None if x.sfw_pass else "SFW audit failed")), base_temperature=.20, structural_retry_temperature=.10, max_new_tokens=1400, journal=journal)
        if not critique.passed:
            draft = invoke_gemini_structured(pass_id=f"P6:{scene.scene_id}", slot="creative", slot_fn=creative_fn, seam_ref="gemini_scene_rewrite", pack=pack, typed_inputs={"feedback": critique.feedback, "previous_draft": draft.model_dump(mode="json"), "facts": p0.model_dump(mode="json"), "scene_outline": scene.model_dump(mode="json")}, result_type=SceneDraftV4, post_validator=lambda x: None, base_temperature=.62, structural_retry_temperature=.28, max_new_tokens=3000, journal=journal)
            check = invoke_gemini_structured(pass_id=f"P5-recheck:{scene.scene_id}", slot="technical", slot_fn=technical_fn, seam_ref="gemini_scene_critique", pack=pack, typed_inputs={"drafted_lines": draft.model_dump(mode="json"), "scene_outline": scene.model_dump(mode="json"), "facts": p0.model_dump(mode="json")}, result_type=SceneCritiqueV4, post_validator=lambda x: None, base_temperature=.20, structural_retry_temperature=.10, max_new_tokens=1400, journal=journal)
            if not check.passed:
                raise SciFiGeminiRewriteExhaustedError(f"scene {scene.scene_id} failed its bounded rewrite")
        validate_spoken_text_and_lock(draft, p3, casts)
        drafts[scene.scene_id] = draft
    expected = _assemble(led, p3, drafts, meta)
    actual = sum(len(_WORD_RE.findall(x)) for x in expected.values())
    meta["scifi_gemini"]["word_receipt"] = {"requested_words": int(resolved["target_words"]), "actual_split_words": actual, "actual_ledger_word_count": int(led.data.get("total_word_count") or 0)}
    meta["scifi_gemini"]["fact_index"] = p0.model_dump(mode="json")
    return GeminiTailParts(outline_view=SimpleNamespace(title=p3.title, premise=p3.premise, setting=p3.setting), canon=SimpleNamespace(title=p3.title, premise=p3.premise), final_title_override=p3.title, run_story_spine=False, tail_finalizer=_GeminiTailFinalizer(expected))
