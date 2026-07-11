"""Sci-Fi Sonnet v4 Continuity Archive lane."""
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
    from ._otr_structured_call import structured_call
    from . import _otr_ledger_freeze
    from .production_ledger import stamp_word_counts
except ImportError:  # pragma: no cover
    from _otr_source_payload import validate_source_payload  # type: ignore
    from _otr_scifi_source_repair import repair_literal_source_metadata  # type: ignore
    from _otr_structured_call import structured_call  # type: ignore
    import _otr_ledger_freeze  # type: ignore
    from production_ledger import stamp_word_counts  # type: ignore


class ScifiSonnetError(RuntimeError): pass
class SonnetPayloadContractError(ScifiSonnetError): pass
class SonnetPayloadRouteError(ScifiSonnetError): pass
class SonnetThinPayloadError(ScifiSonnetError): pass
class SonnetTargetRangeError(ScifiSonnetError): pass
class SonnetPackContractError(ScifiSonnetError): pass
class SonnetPassError(ScifiSonnetError): pass
class SonnetAuditExhaustedError(ScifiSonnetError): pass
class SonnetVoiceInventoryError(ScifiSonnetError): pass
class SonnetSpokenTextError(ScifiSonnetError): pass
class SonnetCompletenessError(ScifiSonnetError): pass
class SonnetPreTailAuditError(ScifiSonnetError): pass
class SonnetSavedLedgerAuditError(ScifiSonnetError): pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def _schema_instruction(schema: type[BaseModel]) -> str:
    keys = ", ".join(schema.model_fields)
    return (
        "\nReturn exactly one JSON object, with no Markdown, headings, or prose. "
        f"Its exact top-level keys are: {keys}. Do not copy input keys into the "
        "output in place of these keys."
    )


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


class PayloadV4(_Strict):
    payload: dict[str, str]
    source_mode: Literal["rss", "operator_pinned"]
    payload_sha256: str


class EvidenceFactV4(_Strict):
    fact_id: str = Field(pattern=r"fact_[0-9]+")
    claim: str
    source_spans: list[SourceSpanV4]


class EvidenceNumberV4(_Strict):
    number_id: str = Field(pattern=r"num_[0-9]+")
    verbatim: str
    fact_id: str
    source_span: SourceSpanV4


class EvidenceEntityV4(_Strict):
    entity_id: str = Field(pattern=r"entity_[0-9]+")
    name: str
    source_spans: list[SourceSpanV4]


class FragmentDossierV4(_Strict):
    verified_facts: list[EvidenceFactV4] = Field(min_length=1, max_length=12)
    key_numbers: list[EvidenceNumberV4] = Field(max_length=12)
    named_entities: list[EvidenceEntityV4] = Field(max_length=12)
    tone: str
    headline_clean: str
    provenance_note: str
    payload_sha256: str


class SessionFrameV4(_Strict):
    session_title: str
    session_premise: str
    registrar_cold_open: str
    orum_register: str
    thessaly_register: str
    vesh_register: str
    scene_description: str
    scene_env: str
    shot_description: str
    visual_prompt: str
    music_description: str
    music_generation_prompt: str


class CitedLineV4(_Strict):
    text: str
    cites: list[str] = Field(min_length=1, max_length=3)


class AuditVerdictV4(_Strict):
    status: Literal["clear", "defect"]
    defects: list[str] = Field(max_length=5)
    flagged_line_refs: list[int] = Field(max_length=5)
    invented_fact_flags: list[int] = Field(max_length=5)
    severity: Literal["critical", "advisory"]
    sfw_pass: bool

    @model_validator(mode="after")
    def coherent(self):
        if self.status == "clear" and (self.defects or self.flagged_line_refs):
            raise ValueError("clear audit must have empty defect and flag lists")
        if self.status == "defect" and (not self.defects or not self.flagged_line_refs):
            raise ValueError("defect audit must identify a real issue")
        if not set(self.invented_fact_flags).issubset(set(self.flagged_line_refs)):
            raise ValueError("invented fact flags must be a subset of flagged lines")
        return self


class WardenChallengeV4(_Strict):
    vesh_objection: str
    registrar_reopening: str


class WardenSatisfiedV4(_Strict):
    vesh_satisfied: str


class RewriteLineV4(_Strict):
    line_ref: int
    speaker: Literal["ORUM", "THESSALY"]
    text: str
    cites: list[str] = Field(min_length=1, max_length=3)


class RewriteResultV4(_Strict):
    corrected_lines: list[RewriteLineV4] = Field(min_length=1, max_length=6)
    vesh_resolution: str


class AttestationV4(_Strict):
    attestation: str
    attestation_cites: list[str] = Field(min_length=1, max_length=4)
    vesh_final_seal: str
    sign_off: str


class CalibrationEditV4(_Strict):
    line_ref: int
    text: str
    cites: list[str] = Field(min_length=1, max_length=3)


class CastLockV4(_Strict):
    char_id: Literal["announcer", "c02", "c03", "c04"]
    name: Literal["ANNOUNCER", "ORUM", "THESSALY", "VESH"]
    character_description: str
    tts_model: Literal["kokoro", "bark"]
    voice_preset: str


class DraftLineV4(_Strict):
    text: str
    cites: list[str] = Field(min_length=1, max_length=3)
    speaker: Literal["ANNOUNCER", "ORUM", "THESSALY", "VESH"]
    char_id: Literal["announcer", "c02", "c03", "c04"]
    source_pass: str


GenerateFn = Callable[..., str]
_DECORATION_RE = re.compile(r"[\r\n\t\[\]()\x60*]|^\s*\x60\x60\x60|^\s*[-*]\s+")
_ALL_CAPS_RE = re.compile(r"\b[A-Z]{2,}\b")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'’-]*")


def _digest(payload: Mapping[str, str]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _span_ok(span: SourceSpanV4, payload: Mapping[str, str]) -> bool:
    return span.quote == payload.get(span.field, "")[span.start:span.end]


def _span_mismatch(span: SourceSpanV4, payload: Mapping[str, str]) -> str:
    expected = payload.get(span.field, "")[span.start:span.end]
    return f"{span.field}[{span.start}:{span.end}] expected {expected[:300]!r}; returned {span.quote[:300]!r}"


def validate_sonnet_payload(payload: Mapping[str, Any], resolved: Mapping[str, Any]) -> tuple[PayloadV4, dict[str, int]]:
    try:
        clean = validate_source_payload(dict(payload), "scifi_sonnet")
    except Exception as exc:
        raise SonnetPayloadContractError(str(exc)) from exc
    source = str(resolved.get("seed_source") or "")
    if source == "custom_premise":
        mode, text = "operator_pinned", clean["seed_text"]
        if len(text.split()) < 8 or len(set(re.findall(r"[A-Za-z0-9]+", text.lower()))) < 4:
            raise SonnetThinPayloadError("pinned source is below the 8/4 thinness floor")
    elif source == "rss_fetch":
        mode, text = "rss", clean["full_text"]
        if len(text.split()) < 80 or len(set(re.findall(r"[A-Za-z0-9]+", text.lower()))) < 12:
            raise SonnetThinPayloadError("RSS source is below the 80/12 thinness floor")
    else:
        raise SonnetPayloadRouteError("scifi_sonnet accepts only rss_fetch or custom_premise")
    target = resolved.get("target_words")
    if isinstance(target, bool) or not isinstance(target, int) or not 30 <= target <= 900:
        raise SonnetTargetRangeError("target_words must be an integer from 30 through 900")
    return PayloadV4(payload=clean, source_mode=mode, payload_sha256=_digest(clean)), {"requested_words": target}


def _dossier_validator(dossier: FragmentDossierV4, payload: Mapping[str, str]) -> str | None:
    facts = {x.fact_id for x in dossier.verified_facts}
    for fact in dossier.verified_facts:
        if not fact.source_spans:
            return f"fact {fact.fact_id} must contain a source span"
        for span in fact.source_spans:
            if not _span_ok(span, payload):
                return f"fact {fact.fact_id} has a non-literal span: {_span_mismatch(span, payload)}"
    for number in dossier.key_numbers:
        if number.fact_id not in facts:
            return f"number {number.number_id} has an invalid fact/span reference"
        if not _span_ok(number.source_span, payload):
            return f"number {number.number_id} has an invalid source span: {_span_mismatch(number.source_span, payload)}"
    for entity in dossier.named_entities:
        if not entity.source_spans:
            return f"entity {entity.entity_id} must contain a source span"
        for span in entity.source_spans:
            if not _span_ok(span, payload):
                return f"entity {entity.entity_id} has a non-literal span: {_span_mismatch(span, payload)}"
    return None


def select_warden_mode_block(merged_seam: str, status: Literal["clear", "defect"]) -> str:
    """Select exactly one preserved Warden block from the merged seam."""
    if not isinstance(merged_seam, str) or not merged_seam.strip():
        raise SonnetPackContractError("merged Warden seam is empty")
    defect_marker = "[DEFECT MODE"
    clear_marker = "[CLEAR MODE"
    starts = [merged_seam.find(defect_marker), merged_seam.find(clear_marker)]
    if any(x < 0 for x in starts) or starts[0] == starts[1]:
        raise SonnetPackContractError("merged Warden seam must contain both mode markers")
    start = starts[1] if status == "clear" else starts[0]
    end = starts[0] if status == "clear" else starts[1]
    block = merged_seam[start:end] if end > start else merged_seam[start:]
    if not block.strip():
        raise SonnetPackContractError("selected Warden mode block is empty")
    return block


def lock_archive_cast(frame: SessionFrameV4) -> Mapping[str, CastLockV4]:
    return {
        "announcer": CastLockV4(char_id="announcer", name="ANNOUNCER", character_description=frame.session_premise, tts_model="kokoro", voice_preset="bm_george"),
        "c02": CastLockV4(char_id="c02", name="ORUM", character_description=frame.orum_register, tts_model="bark", voice_preset="v2/en_speaker_6"),
        "c03": CastLockV4(char_id="c03", name="THESSALY", character_description=frame.thessaly_register, tts_model="bark", voice_preset="v2/en_speaker_3"),
        "c04": CastLockV4(char_id="c04", name="VESH", character_description=frame.vesh_register, tts_model="bark", voice_preset="v2/en_speaker_0"),
    }


def _spoken_error(text: str, name: str = "") -> str | None:
    if not (text or "").strip():
        return "spoken text is empty"
    if _DECORATION_RE.search(text) or _ALL_CAPS_RE.search(text):
        return "spoken text contains decoration or all-caps lexical text"
    if re.match(r"""^\s*["'].*["']\s*$""", text):
        return "spoken text is wholly quoted"
    if re.match(r"^\s*(?:ANNOUNCER|ORUM|THESSALY|VESH)\s*:", text):
        return "spoken text contains a role label"
    if name and re.match(r"^\s*" + re.escape(name.split()[0]) + r"\s*[,!:]", text, re.I):
        return "spoken text is self-vocative"
    if any(not x.strip(".,!?;:'’-") for x in text.split()):
        return "spoken text contains a non-lexical token"
    return None


def validate_spoken_text_and_lock(events: Sequence[DraftLineV4], cast_lock: Mapping[str, CastLockV4]) -> None:
    for event in events:
        lock = cast_lock.get(event.char_id)
        if lock is None or event.speaker != lock.name:
            raise SonnetSpokenTextError(f"{event.char_id} is not locked to speaker {event.speaker}")
        err = _spoken_error(event.text, lock.name)
        if err:
            raise SonnetSpokenTextError(f"{event.speaker}: {err}")


def _prompt(pack: Any, seam: str, pass_id: str, inputs: Mapping[str, Any], schema: type[BaseModel]) -> list[dict[str, str]]:
    seam_text = str((getattr(pack, "prompt_stages", {}) or {}).get(seam) or "")
    if not seam_text:
        raise SonnetPackContractError(f"missing Sonnet seam {seam!r}")
    body = {"pass_id": pass_id, "typed_inputs": inputs, "result_json_schema": schema.model_json_schema()}
    return [{"role": "system", "content": seam_text + _schema_instruction(schema)}, {"role": "user", "content": json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)}]


def invoke_sonnet_structured(
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
    def source_span_repair_factory(*, original_prompt, failed_output, error):
        repair_rules = (
            "This is a typed repair of the same artifact, not a new creative response. "
            "Return one JSON object only. Use fact_1, fact_2, ... for facts; entity_1, "
            "entity_2, ... for entities; and num_1, num_2, ... for numbers. Keep every "
            "fact_id reference consistent. For every source span, calculate quote from "
            "the original request exactly as payload[field][start:end]; do not paraphrase, "
            "infer, or retain a mismatched span. Preserve valid claims and remove only "
            "unsupported facts."
        )
        deterministic = repair_literal_source_metadata(
            failed_output,
            FragmentDossierV4,
            json.loads(prompt[1]["content"])["typed_inputs"]["payload"]["payload"],
            zero_padded_ids=False,
        )
        if deterministic is not None:
            return deterministic
        return [
            {"role": "system", "content": prompt[0]["content"] + "\n" + repair_rules},
            {"role": "user", "content": json.dumps({"failed_artifact": failed_output, "validation_error": str(error), "original_request": json.loads(prompt[1]["content"])}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)},
        ]
    try:
        # LLM slot: per-sub-pass injected creative/technical closure.
        result = structured_call(prompt=prompt, schema=result_type, slot_fn=capture, base_temperature=base_temperature, structural_retry_temperature=structural_retry_temperature, max_new_tokens=max_new_tokens, max_attempts=3, post_validator=post_validator, repair_prompt_factory=(source_span_repair_factory if pass_id == "P0" else None), helper_name=f"scifi_sonnet:{pass_id}")
    except Exception as exc:
        raise SonnetPassError(f"{pass_id} failed: {exc}") from exc
    journal.setdefault("calls", []).append({"pass_id": pass_id, "slot": slot, "attempts": attempts, "accepted": result.model_dump(mode="json")})
    return result


class _SonnetTailFinalizer:
    def __init__(self, expected: Mapping[str, str]):
        self.expected = dict(expected)

    def _proof(self, data: Mapping[str, Any]) -> None:
        lane = data.get("meta", {}).get("scifi_sonnet", {})
        hashes = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in self.expected.items()}
        if lane.get("line_text_sha256") != hashes:
            raise SonnetPreTailAuditError("Sonnet text receipt mismatch")
        for row in data.get("lines", []):
            if row.get("line_id") in self.expected and row.get("text") != self.expected[row["line_id"]]:
                raise SonnetPreTailAuditError(f"Sonnet line changed: {row.get('line_id')}")

    def before_save(self, *, ctx: Any) -> None:
        self._proof(ctx.led.data)
        pre = _otr_ledger_freeze.phase_0_gap_audit_pre(ctx.led)
        post = _otr_ledger_freeze.phase_10_gap_audit_post_and_freeze(ctx.led)
        if pre.errors or pre.warnings or post.errors or post.warnings or ctx.led.data.get("meta", {}).get("freeze_verdict") != "frozen_clean":
            raise SonnetPreTailAuditError("Sonnet freeze proof is not warning-free")

    def after_save(self, *, saved_path: str, ledger_data: Mapping[str, Any]) -> None:
        try:
            with open(saved_path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
        except Exception as exc:
            raise SonnetSavedLedgerAuditError(str(exc)) from exc
        report = _otr_ledger_freeze.run_gap_audit(saved, label="saved")
        if report.errors or report.warnings or saved.get("meta", {}).get("freeze_verdict") != "frozen_clean":
            raise SonnetSavedLedgerAuditError("saved Sonnet ledger is not frozen_clean")
        self._proof(saved)


@dataclass
class SonnetTailParts:
    outline_view: Any
    canon: Any
    final_title_override: str
    run_story_spine: bool
    tail_finalizer: Any


def _assemble(led: Any, frame: SessionFrameV4, cast: Mapping[str, CastLockV4], events: Sequence[DraftLineV4], attestation: AttestationV4, meta: MutableMapping[str, Any]) -> dict[str, str]:
    led.set_cast([{"char_id": c.char_id, "name": c.name, "character_description": c.character_description, "gender": "unspecified", "tts_model": c.tts_model, "voice_preset": c.voice_preset} for c in cast.values()])
    led.set_scenes([{"scene_id": "scene_01", "description": frame.scene_description, "env": frame.scene_env}])
    led.set_shots([{"shot_id": "shot_001", "scene_id": "scene_01", "description": frame.shot_description, "visual_prompt": frame.visual_prompt}])
    lines = []
    beats = []
    expected: dict[str, str] = {}
    for i, event in enumerate(events):
        bid, lid = f"beat_{i:03d}", f"line_{i:03d}"
        role = "announcer" if event.char_id == "announcer" else "character"
        beats.append({"beat_id": bid, "shot_id": "shot_001", "scene_id": "scene_01", "speaker": event.speaker, "char_id": event.char_id, "line_ids": [lid]})
        lines.append({"line_id": lid, "beat_id": bid, "shot_id": "shot_001", "char_id": event.char_id, "speaker_role": role, "text": event.text, "traits": "", "boundary": "beat_start", "arc_phase": "rising", "beat_intent": event.source_pass, "dialogue_slot_id": lid})
        expected[lid] = event.text
    music_ids = []
    for cue_id, placement in (("music_open", "open"), ("music_close", "close")):
        bid, lid = f"{cue_id}_beat", f"{cue_id}_line"
        beats.append({"beat_id": bid, "shot_id": "shot_001", "scene_id": "scene_01", "speaker": cue_id, "char_id": cue_id, "line_ids": [lid]})
        lines.append({"line_id": lid, "beat_id": bid, "shot_id": "shot_001", "char_id": cue_id, "speaker_role": cue_id, "text": "", "boundary": "continue", "arc_phase": "closing", "beat_intent": frame.music_description, "dialogue_slot_id": lid})
        music_ids.append(lid)
    led.set_beats(beats)
    led.set_lines(lines)
    for row in led.data.get("lines", []):
        if row.get("line_id") in music_ids:
            row["skip"] = True
            row["text"] = ""
            row["tts_skip_reason"] = "music_cue"
    led.set_music([
        {"cue_id": "music_open", "placement": "open", "description": frame.music_description, "generation_prompt": frame.music_generation_prompt, "anchor_line_id": next(iter(expected), None)},
        {"cue_id": "music_close", "placement": "close", "description": frame.music_description, "generation_prompt": frame.music_generation_prompt, "anchor_line_id": next(reversed(expected), None)},
    ])
    led.data["clips"] = []
    stamp_word_counts(led)
    meta["scifi_sonnet"]["line_text_sha256"] = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in expected.items()}
    meta["scifi_sonnet"]["accepted_lines"] = dict(expected)
    return expected


def run_scifi_sonnet_episode(
    *, payload: dict[str, str], pack: Any, resolved: Mapping[str, Any], led: Any,
    meta: dict[str, Any], creative_fn: GenerateFn, technical_fn: GenerateFn,
    slot_scheduler: Any, source_bank_row: Any, story_rules: Mapping[str, Any],
    episode_root: Path, episode_id: str,
) -> SonnetTailParts:
    del slot_scheduler, source_bank_row, story_rules, episode_root, episode_id
    envelope, steer = validate_sonnet_payload(payload, resolved)
    meta["scifi_sonnet"] = {"source_digest": envelope.payload_sha256, "source_mode": envelope.source_mode, "call_journal": {}}
    journal = meta["scifi_sonnet"]["call_journal"]
    p0 = invoke_sonnet_structured(pass_id="P0", slot="technical", slot_fn=technical_fn, seam_ref="sonnet_intake_system", pack=pack, typed_inputs={"payload": envelope.model_dump(mode="json")}, result_type=FragmentDossierV4, post_validator=lambda x: _dossier_validator(x, payload), base_temperature=.20, structural_retry_temperature=.10, max_new_tokens=2000, journal=journal)
    p1 = invoke_sonnet_structured(pass_id="P1", slot="creative", slot_fn=creative_fn, seam_ref="sonnet_frame_system", pack=pack, typed_inputs={"dossier": p0.model_dump(mode="json"), "initial_session_word_steer": steer}, result_type=SessionFrameV4, post_validator=lambda x: None, base_temperature=.85, structural_retry_temperature=.40, max_new_tokens=2300, journal=journal)
    cast = lock_archive_cast(p1)
    orum = []
    thessaly = []
    for n in range(2):
        orum.append(invoke_sonnet_structured(pass_id=f"P2a:{n}", slot="creative", slot_fn=creative_fn, seam_ref="sonnet_literalist_system", pack=pack, typed_inputs={"dossier": p0.model_dump(mode="json"), "frame": p1.model_dump(mode="json"), "line_index": n}, result_type=CitedLineV4, post_validator=lambda x: None, base_temperature=.55, structural_retry_temperature=.25, max_new_tokens=2200, journal=journal))
        thessaly.append(invoke_sonnet_structured(pass_id=f"P2b:{n}", slot="creative", slot_fn=creative_fn, seam_ref="sonnet_speculator_system", pack=pack, typed_inputs={"dossier": p0.model_dump(mode="json"), "frame": p1.model_dump(mode="json"), "line_index": n}, result_type=CitedLineV4, post_validator=lambda x: None, base_temperature=.78, structural_retry_temperature=.35, max_new_tokens=2600, journal=journal))
    events: list[DraftLineV4] = [DraftLineV4(text=p1.registrar_cold_open, cites=["fact_0"], speaker="ANNOUNCER", char_id="announcer", source_pass="P1")]
    for i, (a, b) in enumerate(zip(orum, thessaly)):
        events.append(DraftLineV4(text=a.text, cites=a.cites, speaker="ORUM", char_id="c02", source_pass="P2a"))
        events.append(DraftLineV4(text=b.text, cites=b.cites, speaker="THESSALY", char_id="c03", source_pass="P2b"))
    audit = invoke_sonnet_structured(pass_id="P3", slot="technical", slot_fn=technical_fn, seam_ref="sonnet_audit_system", pack=pack, typed_inputs={"dossier": p0.model_dump(mode="json"), "draft_lines": [e.model_dump(mode="json") for e in events[1:]], "coverage": {"count": len(events) - 1}}, result_type=AuditVerdictV4, post_validator=lambda x: None, base_temperature=.25, structural_retry_temperature=.12, max_new_tokens=2000, journal=journal)
    warden_seam = str(pack.prompt_stages.get("sonnet_warden_system") or "")
    if audit.status == "clear":
        block = select_warden_mode_block(warden_seam, "clear")
        warden = invoke_sonnet_structured(pass_id="P4:clear", slot="creative", slot_fn=creative_fn, seam_ref="sonnet_warden_system", pack=SimpleNamespace(prompt_stages={"sonnet_warden_system": block}), typed_inputs={"audit": audit.model_dump(mode="json")}, result_type=WardenSatisfiedV4, post_validator=lambda x: None, base_temperature=.60, structural_retry_temperature=.25, max_new_tokens=1400, journal=journal)
        events.append(DraftLineV4(text=warden.vesh_satisfied, cites=["fact_0"], speaker="VESH", char_id="c04", source_pass="P4"))
    else:
        for round_no in range(2):
            block = select_warden_mode_block(warden_seam, "defect")
            challenge = invoke_sonnet_structured(pass_id=f"P4:defect:{round_no}", slot="creative", slot_fn=creative_fn, seam_ref="sonnet_warden_system", pack=SimpleNamespace(prompt_stages={"sonnet_warden_system": block}), typed_inputs={"audit": audit.model_dump(mode="json")}, result_type=WardenChallengeV4, post_validator=lambda x: None, base_temperature=.70, structural_retry_temperature=.30, max_new_tokens=1400, journal=journal)
            events.append(DraftLineV4(text=challenge.vesh_objection, cites=["fact_0"], speaker="VESH", char_id="c04", source_pass="P4"))
            events.append(DraftLineV4(text=challenge.registrar_reopening, cites=["fact_0"], speaker="ANNOUNCER", char_id="announcer", source_pass="P4"))
            rewrite = invoke_sonnet_structured(pass_id=f"P5:{round_no}", slot="creative", slot_fn=creative_fn, seam_ref="sonnet_rewrite_system", pack=pack, typed_inputs={"audit": audit.model_dump(mode="json"), "draft_lines": [e.model_dump(mode="json") for e in events[1:]]}, result_type=RewriteResultV4, post_validator=lambda x: None, base_temperature=.45, structural_retry_temperature=.20, max_new_tokens=3000, journal=journal)
            for item in rewrite.corrected_lines:
                if item.line_ref >= len(events) - 1:
                    raise SonnetCompletenessError("rewrite line reference is outside the locked draft")
            audit = invoke_sonnet_structured(pass_id=f"P3:recheck:{round_no}", slot="technical", slot_fn=technical_fn, seam_ref="sonnet_audit_system", pack=pack, typed_inputs={"dossier": p0.model_dump(mode="json"), "draft_lines": [e.model_dump(mode="json") for e in events[1:]], "coverage": {}}, result_type=AuditVerdictV4, post_validator=lambda x: None, base_temperature=.25, structural_retry_temperature=.12, max_new_tokens=2000, journal=journal)
            if audit.status == "clear":
                break
        if audit.status != "clear":
            raise SonnetAuditExhaustedError("Sonnet Warden audit remained defective after two rewrites")
        events.append(DraftLineV4(text="The record holds now.", cites=["fact_0"], speaker="VESH", char_id="c04", source_pass="P5"))
    att = invoke_sonnet_structured(pass_id="P6", slot="creative", slot_fn=creative_fn, seam_ref="sonnet_attestation_system", pack=pack, typed_inputs={"dossier": p0.model_dump(mode="json"), "events": [e.model_dump(mode="json") for e in events]}, result_type=AttestationV4, post_validator=lambda x: None, base_temperature=.50, structural_retry_temperature=.22, max_new_tokens=2600, journal=journal)
    events.extend([
        DraftLineV4(text=att.attestation, cites=att.attestation_cites, speaker="ANNOUNCER", char_id="announcer", source_pass="P6"),
        DraftLineV4(text=att.vesh_final_seal, cites=att.attestation_cites[:1], speaker="VESH", char_id="c04", source_pass="P6"),
        DraftLineV4(text=att.sign_off, cites=att.attestation_cites[:1], speaker="ANNOUNCER", char_id="announcer", source_pass="P6"),
    ])
    validate_spoken_text_and_lock(events, cast)
    expected = _assemble(led, p1, cast, events, att, meta)
    actual = sum(len(_WORD_RE.findall(x)) for x in expected.values())
    meta["scifi_sonnet"]["word_receipt"] = {"requested_words": steer["requested_words"], "actual_split_words": actual, "actual_ledger_word_count": int(led.data.get("total_word_count") or 0)}
    meta["scifi_sonnet"]["dossier"] = p0.model_dump(mode="json")
    return SonnetTailParts(outline_view=SimpleNamespace(title=p1.session_title, premise=p1.session_premise, setting=p1.scene_env), canon=SimpleNamespace(title=p1.session_title, premise=p1.session_premise), final_title_override=p1.session_title, run_story_spine=False, tail_finalizer=_SonnetTailFinalizer(expected))
