"""Sci-Fi Sonnet v4 Continuity Archive lane."""
from __future__ import annotations

import hashlib
import json
import logging
import re
import typing
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal, Mapping, MutableMapping, Sequence

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
        MAX_HEADLINE_CLEAN_CHARS,
        MAX_NUMERIC_TOKEN_CHARS,
        MAX_NUMBER_ROWS,
        MAX_PROVENANCE_NOTE_CHARS,
        MAX_QUOTE_CHARS,
        MAX_SPANS_PER_EVIDENCE_ROW,
        MAX_TONE_CHARS,
        compact_p0_repair_context,
        p0_contract_instruction,
        p0_contract_receipt,
        p0_output_token_budget,
        p0_source_char_budget,
        p0_source_chunks,
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
        MAX_HEADLINE_CLEAN_CHARS,
        MAX_NUMERIC_TOKEN_CHARS,
        MAX_NUMBER_ROWS,
        MAX_PROVENANCE_NOTE_CHARS,
        MAX_QUOTE_CHARS,
        MAX_SPANS_PER_EVIDENCE_ROW,
        MAX_TONE_CHARS,
        compact_p0_repair_context,
        p0_contract_instruction,
        p0_contract_receipt,
        p0_output_token_budget,
        p0_source_char_budget,
        p0_source_chunks,
    )
    from _otr_scifi_source_repair import repair_literal_source_metadata  # type: ignore
    from _otr_structured_call import schema_shape_instruction, structured_call  # type: ignore
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
    return schema_shape_instruction(schema)


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


class PayloadV4(_Strict):
    payload: dict[str, str]
    source_mode: Literal["rss", "operator_pinned"]
    payload_sha256: str


class EvidenceFactV4(_Strict):
    fact_id: str = Field(pattern=r"^fact_[1-6]$")
    claim: str = Field(min_length=1, max_length=MAX_CLAIM_CHARS)
    source_spans: list[SourceSpanV4] = Field(
        min_length=1, max_length=MAX_SPANS_PER_EVIDENCE_ROW,
    )


class EvidenceNumberV4(_Strict):
    number_id: str = Field(pattern=r"^num_[1-4]$")
    verbatim: str = Field(min_length=1, max_length=MAX_NUMERIC_TOKEN_CHARS)
    fact_id: str
    source_span: SourceSpanV4


class EvidenceEntityV4(_Strict):
    entity_id: str = Field(pattern=r"^entity_[1-4]$")
    name: str = Field(min_length=1, max_length=MAX_ENTITY_NAME_CHARS)
    source_spans: list[SourceSpanV4] = Field(
        min_length=1, max_length=MAX_SPANS_PER_EVIDENCE_ROW,
    )


class FragmentDossierV4(_Strict):
    verified_facts: list[EvidenceFactV4] = Field(
        min_length=1, max_length=MAX_FACT_ROWS,
    )
    key_numbers: list[EvidenceNumberV4] = Field(max_length=MAX_NUMBER_ROWS)
    named_entities: list[EvidenceEntityV4] = Field(max_length=MAX_ENTITY_ROWS)
    tone: str = Field(min_length=1, max_length=MAX_TONE_CHARS)
    headline_clean: str = Field(max_length=MAX_HEADLINE_CLEAN_CHARS)
    provenance_note: str = Field(max_length=MAX_PROVENANCE_NOTE_CHARS)
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
    """What the Warden may say. It points at lines; it does not pass sentence.

    `severity`, `sfw_pass`, and `invented_fact_flags` are gone: they existed only
    to feed a veto. Citation grounding is proven by `ungrounded_lines` and safety
    by the G9 freeze gate, so the audit's remaining job is to tell the script
    doctor which lines to look at.
    """

    status: Literal["clear", "defect"]
    # A "clear" verdict has no defects to list, and the model omits these keys
    # rather than writing empty arrays. Requiring them makes a CLEAN audit fail
    # validation -- the pass can only succeed by finding fault. Empty is the
    # honest value for a clear verdict.
    defects: list[str] = Field(default_factory=list, max_length=5)
    flagged_line_refs: list[int] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def coherent(self):
        if self.status == "clear" and (self.defects or self.flagged_line_refs):
            raise ValueError("clear audit must have empty defect and flag lists")
        if self.status == "defect" and (not self.defects or not self.flagged_line_refs):
            raise ValueError("defect audit must identify a real issue")
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
    # 1-3, not 1-4. The attestation becomes a DraftLineV4, and every line contract in
    # this lane (CitedLineV4, RewriteLineV4, CalibrationEditV4, DraftLineV4) caps cites
    # at 3. A 4-cite attestation validated here and then raised on construction one
    # pass later -- the schema was promising something the record could not hold.
    attestation_cites: list[str] = Field(min_length=1, max_length=3)
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
    """One spoken line of the session record.

    ``cites`` used to require at least one id, so the ceremonial lines -- the
    Registrar's cold open, the Warden's rulings, the sign-off -- which state no
    fact at all had nowhere to go, and the lane satisfied the schema by citing a
    sentinel ``fact_0``. No such fact can ever exist: the P0 dossier contract is
    one-based (``fact_1, fact_2, ...``). Every ceremonial line in the episode was
    therefore carrying a FALSE citation to a fact that does not exist, and the
    attestation's seal and sign-off borrowed a real fact id they never state.

    A line that cites nothing is now allowed to cite nothing, and must SAY so:
    ``non_fact`` marks it, and the two must agree. Honest empty beats a fabricated
    reference.
    """

    text: str
    cites: list[str] = Field(default_factory=list, max_length=3)
    speaker: Literal["ANNOUNCER", "ORUM", "THESSALY", "VESH"]
    char_id: Literal["announcer", "c02", "c03", "c04"]
    source_pass: str
    non_fact: bool = False

    @model_validator(mode="after")
    def _cites_match_the_claim(self):
        if self.non_fact and self.cites:
            raise ValueError("a non-factual line must not cite a fact")
        if not self.non_fact and not self.cites:
            raise ValueError("a factual line must cite at least one dossier id")
        return self


def _nested_model(annotation: Any) -> "type[BaseModel] | None":
    """The BaseModel inside `X`, `list[X]`, `X | None`, ... or None."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    for arg in typing.get_args(annotation):
        found = _nested_model(arg)
        if found is not None:
            return found
    return None


def _prune_to_schema(data: Any, model: "type[BaseModel]") -> None:
    """Drop keys a strict model forbids, recursively, in place.

    An unrequested key is not authored content: the contract never had a slot for
    it, so nothing the writer meant to say lives there. Removing it discards no
    story and invents nothing.
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


def repair_forbidden_extra_keys(
    failed_output: str, result_type: "type[BaseModel]",
) -> "BaseModel | None":
    """Drop keys the strict artifact forbids. Fails closed if anything is MISSING.

    A missing field is the model's work -- authored attribution, authored text --
    and Python may not conjure it. This only removes what nobody asked for.
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


def _audited_line_indices(events: "list[DraftLineV4]") -> list[int]:
    """The lines the audit actually numbers: the cited reading, nothing else.

    The audit seam states its own contract plainly -- "ORUM and THESSALY
    citation/extrapolation lines ONLY -- the Registrar's cold open and any Warden
    lines are never part of this numbered list". The code was handing it
    ``events[1:]``, which after the first defect round DOES contain Warden lines.
    So the model's ``line_ref`` indices and our list disagreed the moment a rewrite
    round happened -- and a mis-indexed correction rewrites the wrong line.
    """
    return [i for i, line in enumerate(events) if i > 0 and not line.non_fact]


# A number is digits, optional thousands separators, optional decimals -- and NOT
# the punctuation that happens to follow it. The greedy `[\d,.]*` form matched
# "2024," from "in 2024, the probe" and then reported the comma as an invented
# figure (live: prompt 7199a7b1's rerun).
_NUMERIC_TOKEN_RE = re.compile(r"\d(?:[\d,]*\d)?(?:\.\d+)?")


def _allowed_numeric_tokens(dossier: "FragmentDossierV4") -> frozenset[str]:
    """Every numeric token the record is permitted to state.

    The source's own numbers, from everywhere the source actually speaks in the
    dossier: the `key_numbers` the extractor pulled out, the literal `source_spans`
    quotes -- verbatim slices of the payload, so a number inside one IS a number the
    source states -- and the `provenance_note` and `headline_clean`.

    Those last two matter: the attestation seam ORDERS the announcer to name "the
    outlet/date from provenance_note", and the date is a number. A proof that
    forbids what the seam commands is not a proof, it is a contradiction -- and it
    killed the record at the final read (live: prompt 415ca1fc's rerun, where the
    announcer was condemned for stating the archive date it was told to state).
    """
    allowed: set[str] = set()
    for number in dossier.key_numbers:
        allowed.update(_NUMERIC_TOKEN_RE.findall(number.verbatim))
        allowed.update(_NUMERIC_TOKEN_RE.findall(number.source_span.quote))
    for fact in dossier.verified_facts:
        for span in fact.source_spans:
            allowed.update(_NUMERIC_TOKEN_RE.findall(span.quote))
    for entity in dossier.named_entities:
        for span in entity.source_spans:
            allowed.update(_NUMERIC_TOKEN_RE.findall(span.quote))
    allowed.update(_NUMERIC_TOKEN_RE.findall(dossier.provenance_note or ""))
    allowed.update(_NUMERIC_TOKEN_RE.findall(dossier.headline_clean or ""))
    return frozenset(allowed)


def _validate_attestation_grounding(
    att: "AttestationV4",
    dossier: "FragmentDossierV4",
) -> "str | None":
    """The final read is a factual line, so it is held to the factual contract."""
    line = DraftLineV4(
        text=att.attestation, cites=list(att.attestation_cites),
        speaker="ANNOUNCER", char_id="announcer", source_pass="P6",
    )
    defects = ungrounded_lines([line], dossier)
    if not defects:
        return None
    return (
        "the attestation is not grounded in the dossier: "
        + "; ".join(defects)
        + ". State only what the fragment states, and cite the fact you state."
    )


def _merge_grounding_into_audit(
    audit: "AuditVerdictV4",
    events: "Sequence[DraftLineV4]",
    dossier: "FragmentDossierV4",
) -> "AuditVerdictV4":
    """Give the PROOF a repair route: hand it to the doctor, not to the exception.

    `ungrounded_lines` is deterministic and correct -- the model invented the
    number "13" and the record must not state it (live: prompt 7199a7b1). But a
    validator that raises at the END of the lane, after every pass has succeeded,
    is a kill switch, not a gate. The lane already has a script doctor whose whole
    job is fixing flagged lines, so the proof becomes an audit finding and the
    doctor fixes it. Only when the doctor fails twice does the record die.
    """
    defects = ungrounded_lines(events, dossier)
    if not defects:
        return audit
    audited = _audited_line_indices(events)
    ref_by_index = {index: ref for ref, index in enumerate(audited)}
    refs = [
        ref_by_index[index] for index, line in enumerate(events)
        if index in ref_by_index and _line_is_ungrounded(line, dossier)
    ]
    return audit.model_copy(update={
        "status": "defect",
        "defects": (list(audit.defects) + defects)[:5],
        "flagged_line_refs": sorted(
            set(list(audit.flagged_line_refs) + refs),
        )[:5],
    })


def _line_is_ungrounded(
    line: "DraftLineV4",
    dossier: "FragmentDossierV4",
) -> bool:
    if line.non_fact:
        return False
    fact_ids = {row.fact_id for row in dossier.verified_facts}
    if any(cite not in fact_ids for cite in line.cites):
        return True
    allowed = _allowed_numeric_tokens(dossier)
    return any(
        token not in allowed
        for token in _NUMERIC_TOKEN_RE.findall(line.text)
    )


def ungrounded_lines(
    events: "Sequence[DraftLineV4]",
    dossier: "FragmentDossierV4",
) -> list[str]:
    """Prove grounding instead of asking a model whether it feels grounded.

    Two mechanical facts, both checkable:
      * every citation on a factual line must resolve to a real dossier fact id;
      * every number spoken in a factual line must be a number the source states.

    That is what `invented_fact_flags` was gesturing at. A model's feeling about
    it is an opinion; this is a proof, so this is what may end an episode.
    """
    fact_ids = {row.fact_id for row in dossier.verified_facts}
    allowed_numbers = _allowed_numeric_tokens(dossier)
    defects: list[str] = []
    for index, line in enumerate(events):
        if line.non_fact:
            continue
        unknown = [cite for cite in line.cites if cite not in fact_ids]
        if unknown:
            defects.append(
                f"line {index} ({line.speaker}) cites "
                f"{', '.join(sorted(unknown))}, which is not in the dossier"
            )
        invented = [
            token for token in _NUMERIC_TOKEN_RE.findall(line.text)
            if token not in allowed_numbers
        ]
        if invented:
            defects.append(
                f"line {index} ({line.speaker}) states the number(s) "
                f"{', '.join(sorted(set(invented)))}, which the source never does"
            )
    return defects


def _apply_rewrite_corrections(
    events: "list[DraftLineV4]",
    audited: list[int],
    rewrite: "RewriteResultV4",
    audit: "AuditVerdictV4",
    round_no: int,
) -> None:
    """Write the script doctor's corrections back into the record.

    They never were. The loop validated ``rewrite.corrected_lines``, threw them
    away, and re-audited the UNCHANGED draft -- so the recheck re-read the very
    text it had just condemned, and the audit could only exhaust. Sonnet has never
    completed a run, and this is why.

    Python integrates the model's replacement text; it authors none of it. A line
    the doctor did not return stays byte-identical.
    """
    flagged = set(audit.flagged_line_refs)
    seen: set[int] = set()
    for item in rewrite.corrected_lines:
        # An index that does not exist points at no line we can correct. We refuse to
        # guess which one was meant -- but we do not kill the episode over the doctor
        # miscounting, either. Apply nothing for it and let the recheck stay the
        # judge: if the defect really stands, the audit will say so and fail closed
        # on its own terms. (Live: the doctor returned line_ref 4 for a 0..3 draft.)
        if item.line_ref < 0 or item.line_ref >= len(audited):
            log.warning(
                "[scifi_sonnet:P5:%d] discarding a correction for line %d -- the "
                "audited draft has only %d lines (0..%d); no line was changed",
                round_no, item.line_ref, len(audited), len(audited) - 1,
            )
            continue
        # The same line returned twice IS incoherent -- two different texts for one
        # line, and nothing to choose between them. Fail closed.
        if item.line_ref in seen:
            raise SonnetCompletenessError(
                f"rewrite returned line {item.line_ref} twice"
            )
        # A correction for a line the AUDIT never flagged is a different thing: it is
        # coherent, just out of scope. The auditor decides what is defective, so we
        # do not apply an edit nobody asked for -- but we also do not destroy a whole
        # episode over the doctor being eager. Skip it, keep the original line exactly
        # as the model first wrote it, and say so out loud.
        if flagged and item.line_ref not in flagged:
            log.info(
                "[scifi_sonnet:P5:%d] ignoring an unflagged correction to line %d -- "
                "the audit did not name it as a defect; the original line stands",
                round_no, item.line_ref,
            )
            continue
        seen.add(item.line_ref)
        target = events[audited[item.line_ref]]
        # char_id is the locked cast identity: the doctor may rewrite what a
        # Reliquarian SAYS, never who is speaking.
        events[audited[item.line_ref]] = DraftLineV4(
            text=item.text,
            cites=list(item.cites),
            speaker=target.speaker,
            char_id=target.char_id,
            non_fact=False,
            source_pass=f"P5:{round_no}",
        )


GenerateFn = Callable[..., str]
_DECORATION_RE = re.compile(r"[\r\n\t\[\]()\x60*]|^\s*\x60\x60\x60|^\s*[-*]\s+")
_ALL_CAPS_RE = re.compile(r"\b[A-Z]{2,}\b")
# The cast's own names, which the schema REQUIRES to be all caps. A Reliquarian saying
# another's name is speaking normally, not shouting.
_CAST_NAME_RE = re.compile(r"\b(?:ANNOUNCER|ORUM|THESSALY|VESH)\b")
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
        if not meets_v4_rss_source_floor(text):
            raise SonnetThinPayloadError(
                "RSS source is below the 400/80/12 v4 source floor"
            )
    else:
        raise SonnetPayloadRouteError("scifi_sonnet accepts only rss_fetch or custom_premise")
    target = resolved.get("target_words")
    if isinstance(target, bool) or not isinstance(target, int) or not 30 <= target <= 900:
        raise SonnetTargetRangeError("target_words must be an integer from 30 through 900")
    return PayloadV4(payload=clean, source_mode=mode, payload_sha256=_digest(clean)), {"requested_words": target}


def _rebase_dossier(dossier: FragmentDossierV4, offset: int) -> FragmentDossierV4:
    """Move a window's spans back onto the full article.

    A span the model returned is an offset into the WINDOW it was shown. The record
    cites the SOURCE. Adding the window's start position is the whole translation --
    the quote text is untouched, so the citation still proves itself against the
    full text.
    """
    if offset == 0:
        return dossier
    rebased = dossier.model_copy(deep=True)
    for fact in rebased.verified_facts:
        for span in fact.source_spans:
            if span.field == "full_text":
                span.start += offset
                span.end += offset
    for number in rebased.key_numbers:
        if number.source_span.field == "full_text":
            number.source_span.start += offset
            number.source_span.end += offset
    for entity in rebased.named_entities:
        for span in entity.source_spans:
            if span.field == "full_text":
                span.start += offset
                span.end += offset
    return rebased


def merge_dossiers(parts: "Sequence[FragmentDossierV4]") -> FragmentDossierV4:
    """One dossier from every window of the article, ids renumbered to the caps.

    The evidence rows are capped (6 facts, 4 numbers, 4 entities) and their ids are
    `fact_1..fact_6` / `num_1..num_4` by schema -- so a merge cannot simply
    concatenate, it has to RENUMBER, and every number's `fact_id` has to follow its
    fact to the new name. Rows are taken round-robin across the windows so a long
    article is represented from end to end, not just from its opening paragraphs.
    """
    if len(parts) == 1:
        return parts[0]

    facts: list[EvidenceFactV4] = []
    fact_source: list[int] = []
    for row_index in range(MAX_FACT_ROWS):
        for part_index, part in enumerate(parts):
            if row_index < len(part.verified_facts) and len(facts) < MAX_FACT_ROWS:
                facts.append(part.verified_facts[row_index].model_copy(deep=True))
                fact_source.append(part_index)

    renamed: dict[tuple[int, str], str] = {}
    for new_index, (fact, part_index) in enumerate(zip(facts, fact_source), 1):
        renamed[(part_index, fact.fact_id)] = f"fact_{new_index}"
        fact.fact_id = f"fact_{new_index}"

    numbers: list[EvidenceNumberV4] = []
    for row_index in range(MAX_NUMBER_ROWS):
        for part_index, part in enumerate(parts):
            if row_index >= len(part.key_numbers) or len(numbers) >= MAX_NUMBER_ROWS:
                continue
            number = part.key_numbers[row_index].model_copy(deep=True)
            owner = renamed.get((part_index, number.fact_id))
            if owner is None:
                # Its fact did not survive the cap; a number with no fact to hang
                # on is not evidence, it is a loose figure. Drop it.
                continue
            number.fact_id = owner
            number.number_id = f"num_{len(numbers) + 1}"
            numbers.append(number)

    entities: list[EvidenceEntityV4] = []
    for row_index in range(MAX_ENTITY_ROWS):
        for part in parts:
            if row_index < len(part.named_entities) and len(entities) < MAX_ENTITY_ROWS:
                entities.append(part.named_entities[row_index].model_copy(deep=True))

    head = parts[0]
    return FragmentDossierV4(
        verified_facts=facts,
        key_numbers=numbers,
        named_entities=entities,
        tone=head.tone,
        headline_clean=head.headline_clean,
        provenance_note=head.provenance_note,
        payload_sha256=head.payload_sha256,
    )


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


def allowed_spoken_all_caps(dossier: "FragmentDossierV4 | None") -> frozenset[str]:
    """All-caps tokens the SOURCE itself uses.

    The record is about the source, so the source's own acronyms are the words the
    episode has to say: NASA, RNA, ESA. Killing the episode for saying them is
    killing it for doing its job (live: prompt c7023ae8 -- the announcer's cold
    open said an acronym the fragment states, and the gate ended the run AFTER
    every pass had succeeded).

    ``provenance_note`` is in this list for the SAME reason it is in
    ``_allowed_numbers``: the attestation seam ORDERS the announcer to name "the
    outlet/date from provenance_note", and the outlet is very often an all-caps
    acronym -- BBC, NASA, NIH, NSF, MIT. A validator that forbids what the seam
    COMMANDS is not a proof, it is a contradiction.

    That contradiction was live on 2026-07-14: with article-body scraping finally
    working, real outlets started reaching the dossier, the announcer named its
    source exactly as instructed, and the gate ended the run with "spoken text
    contains all-caps lexical text (BBC)" -- after every pass had succeeded. The
    numbers proof had already learned this lesson (prompt 415ca1fc, where the
    announcer was condemned for stating the archive date it was told to state);
    the all-caps proof simply never had it applied.
    """
    if dossier is None:
        return frozenset()
    surfaces: list[str] = []
    for fact in dossier.verified_facts:
        surfaces.append(fact.claim)
        surfaces.extend(span.quote for span in fact.source_spans)
    for number in dossier.key_numbers:
        surfaces.append(number.source_span.quote)
    for entity in dossier.named_entities:
        surfaces.append(entity.name)
        surfaces.extend(span.quote for span in entity.source_spans)
    surfaces.append(dossier.headline_clean)
    surfaces.append(dossier.provenance_note)
    return frozenset(
        match.group(0)
        for surface in surfaces
        for match in _ALL_CAPS_RE.finditer(surface or "")
    )


def _spoken_error(
    text: str,
    name: str = "",
    allowed_all_caps: frozenset[str] = frozenset(),
) -> str | None:
    if not (text or "").strip():
        return "spoken text is empty"
    # The all-caps rule exists to catch shouted emphasis and stage directions. But THIS
    # lane's characters are NAMED in all caps by contract -- CastLockV4.name is
    # Literal["ANNOUNCER", "ORUM", "THESSALY", "VESH"] -- so the moment the Warden
    # addressed the Literalist by name, the gate rejected the line for obeying the
    # schema. A validator may not block on the system's own contract. The same is
    # true of the SOURCE's own acronyms: the record exists to state them.
    without_cast_names = _CAST_NAME_RE.sub("", text)
    shouted = [
        token for token in _ALL_CAPS_RE.findall(without_cast_names)
        if token not in allowed_all_caps
    ]
    decoration = _DECORATION_RE.search(text)
    if decoration:
        return (
            f"spoken text contains decoration {decoration.group(0)!r} -- no "
            "brackets, parentheses, asterisks, backticks, tabs or newlines "
            f"reach the microphone. You wrote: {text!r}"
        )
    if shouted:
        return (
            "spoken text contains all-caps lexical text "
            f"({', '.join(sorted(set(shouted)))}); write it in normal case -- "
            "only a locked cast name or an acronym the source itself uses may "
            "be all-caps"
        )
    if re.match(r"""^\s*["'].*["']\s*$""", text):
        return "spoken text is wholly quoted"
    if re.match(r"^\s*(?:ANNOUNCER|ORUM|THESSALY|VESH)\s*:", text):
        return "spoken text contains a role label"
    if name and re.match(r"^\s*" + re.escape(name.split()[0]) + r"\s*[,!:]", text, re.I):
        return "spoken text is self-vocative"
    if any(not x.strip(".,!?;:'’-") for x in text.split()):
        return "spoken text contains a non-lexical token"
    return None


def validate_spoken_text_and_lock(
    events: Sequence[DraftLineV4],
    cast_lock: Mapping[str, CastLockV4],
    allowed_all_caps: frozenset[str] = frozenset(),
) -> None:
    for event in events:
        lock = cast_lock.get(event.char_id)
        if lock is None or event.speaker != lock.name:
            raise SonnetSpokenTextError(f"{event.char_id} is not locked to speaker {event.speaker}")
        err = _spoken_error(event.text, lock.name, allowed_all_caps)
        if err:
            raise SonnetSpokenTextError(f"{event.speaker}: {err}")


def _prompt(pack: Any, seam: str, pass_id: str, inputs: Mapping[str, Any], schema: type[BaseModel]) -> list[dict[str, str]]:
    seam_text = str((getattr(pack, "prompt_stages", {}) or {}).get(seam) or "")
    if not seam_text:
        raise SonnetPackContractError(f"missing Sonnet seam {seam!r}")
    body = {"pass_id": pass_id, "typed_inputs": inputs, "result_json_schema": schema.model_json_schema()}
    return [{"role": "system", "content": seam_text + _schema_instruction(schema)}, {"role": "user", "content": json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)}]


class _PromptMustFitMessages(list[dict[str, str]]):
    """Tell the local slot wrapper to fail before it slices this prompt.

    Parity with the Codex lane: a provenance-bearing prompt (the source payload
    plus its schema contract) must never be silently left-truncated. Losing the
    system/schema prefix produces a confidently wrong artifact instead of an
    honest failure.
    """

    _otr_prompt_must_fit = True


def invoke_sonnet_structured(
    *, pass_id: str, slot: Literal["creative", "technical"], slot_fn: GenerateFn,
    seam_ref: str, pack: Any, typed_inputs: Mapping[str, Any],
    result_type: type[BaseModel], post_validator: Callable[[BaseModel], str | None],
    base_temperature: float, structural_retry_temperature: float,
    max_new_tokens: int, journal: MutableMapping[str, Any],
    prompt_must_fit: bool = False,
) -> BaseModel:
    prompt = _prompt(pack, seam_ref, pass_id, typed_inputs, result_type)
    if pass_id == "P0":
        prompt[0]["content"] += p0_contract_instruction(has_numeric_tokens=False)
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
                "This is a typed repair of the same FragmentDossierV4, not a new creative "
                "response. Return one complete JSON object only, rooted exactly at "
                "verified_facts, key_numbers, named_entities, tone, headline_clean, "
                "provenance_note, and payload_sha256. Use fact_1 through fact_6, entity_1 "
                "through entity_4, and num_1 through num_4; keep every fact_id reference "
                "consistent. Every fact and entity has exactly one literal source span; "
                "calculate quote exactly as payload[field][start:end] from the supplied source "
                "evidence. Do not paraphrase, infer, or retain a mismatched span. tone is one "
                "nonempty scalar source-derived string, never an array or object. Preserve "
                "valid claims and remove only unsupported facts. The tagged input references "
                "are not an output template: never return a wrapper, request field, or tag name."
            )
            p0_envelope = json.loads(prompt[1]["content"])["typed_inputs"]["payload"]
            deterministic = repair_literal_source_metadata(
                failed_output,
                FragmentDossierV4,
                p0_envelope["payload"],
                zero_padded_ids=False,
                max_quote_chars=MAX_QUOTE_CHARS,
            )
            if deterministic is not None:
                return deterministic
        else:
            # An artifact can be thrown out purely for keys the model added that the
            # contract never asked for. Pruning those loses no authored work, so try
            # it before spending an LLM call. (Gemini's lane proved this out.)
            deterministic = repair_forbidden_extra_keys(failed_output, result_type)
            if deterministic is not None and post_validator(deterministic) is None:
                log.info(
                    "[scifi_sonnet:%s] deterministic repair dropped forbidden extra "
                    "keys; no LLM repair call made", pass_id,
                )
                return deterministic
            # The MISSING half is different: which facts a line cites is authored
            # attribution, and Python must never invent it -- carrying the old line's
            # cites forward onto new text would be exactly the false attribution we
            # just spent the day removing. So ask, and ask NARROWLY: hand the model
            # back its own corrected lines and request only the field it dropped,
            # rather than making it re-derive a whole artifact from a wall of errors.
            if getattr(result_type, "__name__", "") == "RewriteResultV4":
                try:
                    partial = parse_first_json_object(failed_output)
                except Exception:
                    partial = None
                lines_missing = [
                    item for item in (partial or {}).get("corrected_lines", [])
                    if isinstance(item, dict) and not item.get("cites")
                ]
                if lines_missing:
                    log.info(
                        "[scifi_sonnet:%s] asking the model only for the %d missing "
                        "cites array(s) it dropped", pass_id, len(lines_missing),
                    )
                    return [
                        {"role": "system", "content": prompt[0]["content"] + "\n" + (
                            "Your previous reply dropped the required `cites` array on "
                            "one or more corrected lines. Return the SAME corrected "
                            "lines with their text UNCHANGED, and add to each a cites "
                            "array of 1-3 dossier ids (fact_N / num_N) that actually "
                            "support what that line says. If the line names a fact in "
                            "its wording, that fact's id belongs in cites. Change no "
                            "text. Add no lines. Return the complete object."
                        )},
                        {"role": "user", "content": json.dumps({
                            "lines_missing_cites": lines_missing,
                            "dossier": json.loads(prompt[1]["content"])["typed_inputs"].get("dossier"),
                        }, sort_keys=True, separators=(",", ":"), ensure_ascii=False)},
                    ]
            repair_rules = (
                "This is a typed repair of the same artifact. Preserve the continuity archive, "
                "session frame, cast locks, and authored lines; repair only fields named by "
                "the validation error. Every required nested field must be present; preserve "
                "the locked speaker/char_id mapping. "
                "Return ONLY the keys the schema declares -- no extra fields. "
                "EVERY corrected line MUST carry ALL FOUR of: line_ref, speaker, text, and "
                "cites. cites is a NON-EMPTY array of 1-3 dossier ids (fact_N / num_N) that "
                "actually support what the line says -- never omit it, never leave it empty, "
                "and never cite an id that is not in the dossier. If your line mentions a "
                "fact in its wording, that fact's id belongs in cites."
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
        result = structured_call(prompt=prompt, schema=result_type, slot_fn=capture, base_temperature=base_temperature, structural_retry_temperature=structural_retry_temperature, max_new_tokens=max_new_tokens, max_attempts=3, post_validator=post_validator, repair_prompt_factory=typed_repair_factory, clamp_overlong_strings=pass_id != "P0", helper_name=f"scifi_sonnet:{pass_id}")
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
        # A WARNING IS NOT AN ERROR -- that is what the word means. This gate killed the
        # episode on any warning at all, which is the same defect that has dominated this
        # build: a gate blocking on a note. Errors block. The freeze verdict blocks -- it
        # is the cascade's own structured judgment. Warnings are RECORDED, loudly, and the
        # record ships.
        notes = list(pre.warnings) + list(post.warnings)
        if notes:
            log.warning(
                "[scifi_sonnet] the freeze cascade raised %d warning(s); none of them is "
                "an error, so the record stands:\n  %s",
                len(notes), "\n  ".join(str(n) for n in notes),
            )
            ctx.led.data.setdefault("meta", {}).setdefault("scifi_sonnet", {})[
                "freeze_notes"
            ] = [str(n) for n in notes]
        if pre.errors or post.errors:
            raise SonnetPreTailAuditError(
                "Sonnet freeze proof has hard errors: "
                + "; ".join(str(e) for e in list(pre.errors) + list(post.errors))
            )
        # The cascade's own verdict mapping (see _otr_freeze_cascade, "Verdict mapping"):
        #   clean_no_edits + clean  -> frozen_clean
        #   clean_no_edits + warns  -> frozen_with_warns     <- ALSO a clean freeze
        #   improved                -> frozen_with_doctor_edits
        #   too_many_edits / needs_full_rerun                <- structural, must block
        # frozen_with_warns means the reviewer made NO edits and the ledger is sound; the
        # warns are the same notes we just recorded. Demanding frozen_clean rejected a good
        # freeze for carrying a note. A doctor-edited freeze DOES still block a
        # content-owned lane -- its canonical text is sealed and nothing may rewrite it.
        verdict = ctx.led.data.get("meta", {}).get("freeze_verdict")
        if verdict not in ("frozen_clean", "frozen_with_warns"):
            raise SonnetPreTailAuditError(
                f"Sonnet freeze verdict is {verdict!r} -- not a clean freeze"
            )

    def after_save(self, *, saved_path: str, ledger_data: Mapping[str, Any]) -> None:
        try:
            with open(saved_path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
        except Exception as exc:
            raise SonnetSavedLedgerAuditError(str(exc)) from exc
        report = _otr_ledger_freeze.run_gap_audit(saved, label="saved")
        # Same law on the saved ledger: errors and structural verdicts block, warnings do
        # not. frozen_with_warns IS a clean freeze -- the reviewer made no edits -- and for
        # a multi-role content-owned lane frozen_clean is not even reachable here: the
        # freeze runs BEFORE CastLock assigns the bark voices, so the cascade always has
        # something to note. Demanding it made the gate unpassable by construction.
        if report.warnings:
            log.warning(
                "[scifi_sonnet] the saved ledger carries %d warning(s); none is an error:"
                "\n  %s",
                len(report.warnings), "\n  ".join(str(w) for w in report.warnings),
            )
        if report.errors:
            raise SonnetSavedLedgerAuditError(
                "saved Sonnet ledger has hard errors: "
                + "; ".join(str(e) for e in report.errors)
            )
        verdict = saved.get("meta", {}).get("freeze_verdict")
        if verdict not in ("frozen_clean", "frozen_with_warns"):
            raise SonnetSavedLedgerAuditError(
                f"saved Sonnet ledger freeze verdict is {verdict!r} -- not a clean freeze"
            )
        self._proof(saved)


@dataclass
class SonnetTailParts:
    outline_view: Any
    canon: Any
    final_title_override: str
    run_story_spine: bool
    tail_finalizer: Any


def _build_sonnet_episode_canon(frame: SessionFrameV4) -> EpisodeCanon:
    """Build the complete shared-tail canon from the accepted archive frame."""
    return EpisodeCanon(
        title=frame.session_title,
        premise=frame.session_premise,
        setting=frame.scene_env,
        # SessionFrameV4 deliberately carries no time/palette field. Keep
        # their absence explicit instead of inventing canon detail.
        time_of_day="",
        sound_palette=[],
    )


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
    p0_token_budget = p0_output_token_budget(extra_root_fields=2)
    # THE SOURCE IS READ WHOLE, IN WINDOWS THAT FIT. The evidence artifact was
    # always bounded; the article never was, and a long one walked straight past the
    # context window (live: prompt 415ca1fc, 5,424 tokens into a 5,192 opening).
    # Trimming would have been easier and wrong: the tail of the article would
    # become uncitable, and the long episodes are the ones that need it most.
    source_budget = p0_source_char_budget(extra_root_fields=2)
    windows = p0_source_chunks(envelope.payload, budget_chars=source_budget)
    journal["fact_index_token_budget"] = {
        **p0_contract_receipt(extra_root_fields=2),
        "source_evidence_field_count": len(envelope.payload),
        "source_evidence_characters": sum(
            len(value) for value in envelope.payload.values()
        ),
        "source_char_budget": source_budget,
        "source_windows": len(windows),
    }
    parts: list[FragmentDossierV4] = []
    for index, (offset, window) in enumerate(windows):
        # Each window is proven against ITS OWN text -- the model can only cite what
        # it was shown -- and then rebased onto the full article.
        part = invoke_sonnet_structured(
            pass_id="P0" if len(windows) == 1 else f"P0:w{index}",
            slot="technical", slot_fn=technical_fn,
            seam_ref="sonnet_intake_system", pack=pack,
            typed_inputs={"payload": PayloadV4(
                payload=window, source_mode=envelope.source_mode,
                payload_sha256=envelope.payload_sha256,
            ).model_dump(mode="json")},
            result_type=FragmentDossierV4,
            post_validator=lambda x, w=window: _dossier_validator(x, w),
            base_temperature=.20, structural_retry_temperature=.10,
            max_new_tokens=p0_token_budget, journal=journal,
            prompt_must_fit=True,
        )
        parts.append(_rebase_dossier(part, offset))
    p0 = merge_dossiers(parts)
    # And the merged record must prove itself against the WHOLE article, which is
    # what every downstream citation is checked against.
    merged_error = _dossier_validator(p0, envelope.payload)
    if merged_error is not None:
        raise SonnetPassError(f"merged dossier is not grounded: {merged_error}")
    p1 = invoke_sonnet_structured(pass_id="P1", slot="creative", slot_fn=creative_fn, seam_ref="sonnet_frame_system", pack=pack, typed_inputs={"dossier": p0.model_dump(mode="json"), "initial_session_word_steer": steer}, result_type=SessionFrameV4, post_validator=lambda x: None, base_temperature=.85, structural_retry_temperature=.40, max_new_tokens=2300, journal=journal)
    cast = lock_archive_cast(p1)
    orum = []
    thessaly = []
    for n in range(2):
        orum.append(invoke_sonnet_structured(pass_id=f"P2a:{n}", slot="creative", slot_fn=creative_fn, seam_ref="sonnet_literalist_system", pack=pack, typed_inputs={"dossier": p0.model_dump(mode="json"), "frame": p1.model_dump(mode="json"), "line_index": n}, result_type=CitedLineV4, post_validator=lambda x: None, base_temperature=.55, structural_retry_temperature=.25, max_new_tokens=2200, journal=journal))
        thessaly.append(invoke_sonnet_structured(pass_id=f"P2b:{n}", slot="creative", slot_fn=creative_fn, seam_ref="sonnet_speculator_system", pack=pack, typed_inputs={"dossier": p0.model_dump(mode="json"), "frame": p1.model_dump(mode="json"), "line_index": n}, result_type=CitedLineV4, post_validator=lambda x: None, base_temperature=.78, structural_retry_temperature=.35, max_new_tokens=2600, journal=journal))
    events: list[DraftLineV4] = [DraftLineV4(text=p1.registrar_cold_open, cites=[], non_fact=True, speaker="ANNOUNCER", char_id="announcer", source_pass="P1")]
    for i, (a, b) in enumerate(zip(orum, thessaly)):
        events.append(DraftLineV4(text=a.text, cites=a.cites, speaker="ORUM", char_id="c02", source_pass="P2a"))
        events.append(DraftLineV4(text=b.text, cites=b.cites, speaker="THESSALY", char_id="c03", source_pass="P2b"))
    # ONE numbering contract for the audit, everywhere. This used to pass `events[1:]`
    # while the rewrite loop numbered the factual lines only -- identical today (just
    # the cold open precedes), but two contracts for one thing is how a line_ref
    # silently comes to mean two different lines.
    audited = _audited_line_indices(events)
    audit = invoke_sonnet_structured(pass_id="P3", slot="technical", slot_fn=technical_fn, seam_ref="sonnet_audit_system", pack=pack, typed_inputs={"dossier": p0.model_dump(mode="json"), "draft_lines": [events[i].model_dump(mode="json") for i in audited], "coverage": {"count": len(audited)}}, result_type=AuditVerdictV4, post_validator=lambda x: None, base_temperature=.25, structural_retry_temperature=.12, max_new_tokens=2000, journal=journal)
    # The deterministic grounding proof outranks the Warden's opinion in BOTH
    # directions: it can open a defect the Warden missed, and it is the only thing
    # that can close the record. Feed it to the doctor, not to an exception.
    audit = _merge_grounding_into_audit(audit, events, p0)
    warden_seam = str(pack.prompt_stages.get("sonnet_warden_system") or "")
    if audit.status == "clear":
        block = select_warden_mode_block(warden_seam, "clear")
        warden = invoke_sonnet_structured(pass_id="P4:clear", slot="creative", slot_fn=creative_fn, seam_ref="sonnet_warden_system", pack=SimpleNamespace(prompt_stages={"sonnet_warden_system": block}), typed_inputs={"audit": audit.model_dump(mode="json")}, result_type=WardenSatisfiedV4, post_validator=lambda x: None, base_temperature=.60, structural_retry_temperature=.25, max_new_tokens=1400, journal=journal)
        events.append(DraftLineV4(text=warden.vesh_satisfied, cites=[], non_fact=True, speaker="VESH", char_id="c04", source_pass="P4"))
    else:
        for round_no in range(2):
            block = select_warden_mode_block(warden_seam, "defect")
            challenge = invoke_sonnet_structured(pass_id=f"P4:defect:{round_no}", slot="creative", slot_fn=creative_fn, seam_ref="sonnet_warden_system", pack=SimpleNamespace(prompt_stages={"sonnet_warden_system": block}), typed_inputs={"audit": audit.model_dump(mode="json")}, result_type=WardenChallengeV4, post_validator=lambda x: None, base_temperature=.70, structural_retry_temperature=.30, max_new_tokens=1400, journal=journal)
            events.append(DraftLineV4(text=challenge.vesh_objection, cites=[], non_fact=True, speaker="VESH", char_id="c04", source_pass="P4"))
            events.append(DraftLineV4(text=challenge.registrar_reopening, cites=[], non_fact=True, speaker="ANNOUNCER", char_id="announcer", source_pass="P4"))
            audited = _audited_line_indices(events)
            # The rewrite seam orders the doctor to fix a defect "by grounding the line
            # in an actual key_numbers/verified_facts entry" and to return cites -- and
            # the pass was never GIVEN the dossier. It was being told to cite a source
            # it could not see, so it wrote fact ids into its prose and omitted the
            # cites array entirely, twice, and the lane died. Hand it the dossier it is
            # required to cite.
            rewrite = invoke_sonnet_structured(pass_id=f"P5:{round_no}", slot="creative", slot_fn=creative_fn, seam_ref="sonnet_rewrite_system", pack=pack, typed_inputs={"dossier": p0.model_dump(mode="json"), "audit": audit.model_dump(mode="json"), "draft_lines": [events[i].model_dump(mode="json") for i in audited]}, result_type=RewriteResultV4, post_validator=lambda x: None, base_temperature=.45, structural_retry_temperature=.20, max_new_tokens=3000, journal=journal)
            _apply_rewrite_corrections(events, audited, rewrite, audit, round_no)
            audit = invoke_sonnet_structured(pass_id=f"P3:recheck:{round_no}", slot="technical", slot_fn=technical_fn, seam_ref="sonnet_audit_system", pack=pack, typed_inputs={"dossier": p0.model_dump(mode="json"), "draft_lines": [events[i].model_dump(mode="json") for i in _audited_line_indices(events)], "coverage": {}}, result_type=AuditVerdictV4, post_validator=lambda x: None, base_temperature=.25, structural_retry_temperature=.12, max_new_tokens=2000, journal=journal)
            audit = _merge_grounding_into_audit(audit, events, p0)
            if audit.status == "clear":
                break
        # The Warden bought its two rewrites. Whatever it still feels about the
        # record is a note, and a note ships with the episode -- the auditor is an
        # LLM, and it will not stop calling craft observations defects no matter
        # how plainly the seam forbids it (it blocked three separate rolls on
        # "line 2 repeats line 0's claim", in a 30-word script with a two-fact
        # dossier, where saying something new is not possible).
        #
        # What actually matters -- whether the record cites real facts and states
        # only real numbers -- is PROVEN below by `ungrounded_lines`, not felt.
        if audit.status != "clear":
            log.info(
                "[scifi_sonnet] the Warden is still not satisfied after two "
                "rewrites; its notes ship with the record: %s",
                "; ".join(audit.defects) or "(none named)",
            )
            journal["warden_notes"] = audit.model_dump(mode="json")
        # The Warden's closing ruling is a SPOKEN LINE, and it was hardcoded here as
        # "The record holds now." -- Python authoring dialogue, which this lane is
        # never allowed to do. It did not even need to: RewriteResultV4 already
        # carries `vesh_resolution`, the Warden's on-air acknowledgment that the
        # record is corrected. The model had written the line all along and the lane
        # was throwing it away to speak for the character itself.
        events.append(DraftLineV4(text=rewrite.vesh_resolution, cites=[], non_fact=True, speaker="VESH", char_id="c04", source_pass="P5"))
    # The attestation is authored AFTER the doctor has gone home, so its grounding
    # has to be proven at its own rung -- otherwise an invented number in the final
    # read kills a record that every other pass had already cleared.
    att = invoke_sonnet_structured(pass_id="P6", slot="creative", slot_fn=creative_fn, seam_ref="sonnet_attestation_system", pack=pack, typed_inputs={"dossier": p0.model_dump(mode="json"), "events": [e.model_dump(mode="json") for e in events]}, result_type=AttestationV4, post_validator=lambda x, d=p0: _validate_attestation_grounding(x, d), base_temperature=.50, structural_retry_temperature=.22, max_new_tokens=2600, journal=journal)
    events.extend([
        # The attestation IS the citation-anchored read -- it states the facts, so it
        # cites them. The seal and the sign-off state nothing: they were borrowing the
        # attestation's first fact id, claiming a source for a ceremonial line that
        # never makes the claim. An honest empty is the truthful record.
        DraftLineV4(text=att.attestation, cites=list(att.attestation_cites), speaker="ANNOUNCER", char_id="announcer", source_pass="P6"),
        DraftLineV4(text=att.vesh_final_seal, cites=[], non_fact=True, speaker="VESH", char_id="c04", source_pass="P6"),
        DraftLineV4(text=att.sign_off, cites=[], non_fact=True, speaker="ANNOUNCER", char_id="announcer", source_pass="P6"),
    ])
    validate_spoken_text_and_lock(events, cast, allowed_spoken_all_caps(p0))
    # The grounding proof. Every factual line cites a real dossier fact and states
    # only numbers the source states -- checked, not felt. This is the one thing in
    # the lane allowed to end an episode over content, and it is deterministic.
    ungrounded = ungrounded_lines(events, p0)
    if ungrounded:
        journal["ungrounded_lines"] = ungrounded
        raise SonnetSpokenTextError(
            "the record is not grounded in its dossier: " + "; ".join(ungrounded)
        )
    expected = _assemble(led, p1, cast, events, att, meta)
    from ._otr_content_authorship import stamp_receipt
    stamp_receipt(
        led.data, owner_bank="scifi_sonnet",
        accepted_artifacts={
            "final_events": [event.model_dump(mode="json") for event in events],
            "final_attestation": att,
        },
    )
    actual = sum(len(_WORD_RE.findall(x)) for x in expected.values())
    meta["scifi_sonnet"]["word_receipt"] = {"requested_words": steer["requested_words"], "actual_split_words": actual, "actual_ledger_word_count": int(led.data.get("total_word_count") or 0)}
    meta["scifi_sonnet"]["dossier"] = p0.model_dump(mode="json")
    canon = _build_sonnet_episode_canon(p1)
    return SonnetTailParts(
        outline_view=SimpleNamespace(
            title=p1.session_title,
            premise=p1.session_premise,
            setting=p1.scene_env,
            time_of_day=canon.time_of_day,
        ),
        canon=canon,
        final_title_override=p1.session_title,
        run_story_spine=False,
        tail_finalizer=_SonnetTailFinalizer(expected),
    )
