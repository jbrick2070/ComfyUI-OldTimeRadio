"""Strict original Lost and Found Frequency source-bank runner."""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from ._otr_canon import EpisodeCanon
from ._otr_content_authorship import stamp_receipt
from ._otr_structured_call import schema_shape_instruction, structured_call
from . import _otr_ledger_freeze
from .production_ledger import stamp_word_counts


class OriginalCodex56SolError(RuntimeError): pass
class OriginalCodex56SolPassError(OriginalCodex56SolError): pass
class OriginalCodex56SolContractError(OriginalCodex56SolError): pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


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
    possibility_id: str
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
    possibilities: list[PossibilityCard] = Field(min_length=3, max_length=3)


class CandidateFinding(StrictModel):
    possibility_id: str
    category: str
    detail: str
    blocking: bool


class SlateTriage(StrictModel):
    selected_possibility_id: str
    findings: list[CandidateFinding]


class CallerThread(StrictModel):
    thread_id: str
    caller_name: str
    lost_object: str
    practical_need: str


class CausalStep(StrictModel):
    step_id: str
    cause: str
    effect: str


class AudibleClue(StrictModel):
    clue_id: str
    thread_id: str
    sound_or_phrase: str
    implication: str


class ResolutionLink(StrictModel):
    thread_id: str
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
    reveal: str
    resolution_links: list[ResolutionLink] = Field(min_length=2)


class FairPlayFinding(StrictModel):
    field_path: str
    item_id: str
    category: str
    detail: str
    blocking: bool


class FairPlayReport(StrictModel):
    accepted: bool
    findings: list[FairPlayFinding]


class CastConcept(StrictModel):
    char_id: str
    name: str
    role: Literal["announcer", "desk_operator", "caller"]
    character_description: str


class SceneConcept(StrictModel):
    scene_id: str
    description: str
    env: str


class ShotConcept(StrictModel):
    shot_id: str
    scene_id: str
    description: str
    visual_prompt: str


class BeatConcept(StrictModel):
    beat_id: str
    shot_id: str
    scene_id: str
    char_id: str
    speaker: str
    intent: str


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
    arc_phase: Literal["opening", "rising", "reveal", "closing"]
    intent: str


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
        return [
            {"role": "system", "content": system + "\nReturn the same complete artifact, repairing only the typed contract error. JSON only.\n" + schema_shape_instruction(schema)},
            {"role": "user", "content": json.dumps({"failed_artifact": failed_output, "error": str(error), "inputs": inputs}, ensure_ascii=False, sort_keys=True)},
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


def _validate_graph(score: BroadcastScore, manifest: ClosedLineManifest,
                    script: PerformanceScript) -> None:
    cast = {c.char_id: c for c in score.cast}
    scenes = {s.scene_id for s in score.scenes}
    shots = {s.shot_id: s for s in score.shots}
    beats = {b.beat_id: b for b in score.beats}
    if len(cast) != len(score.cast) or "announcer" not in cast:
        raise OriginalCodex56SolContractError("cast ids are invalid")
    if sum(c.role == "desk_operator" for c in score.cast) != 1:
        raise OriginalCodex56SolContractError("exactly one desk operator is required")
    for shot in score.shots:
        if shot.scene_id not in scenes:
            raise OriginalCodex56SolContractError("shot scene reference is invalid")
    for beat in score.beats:
        if (beat.shot_id not in shots or beat.scene_id not in scenes
                or beat.char_id not in cast
                or shots[beat.shot_id].scene_id != beat.scene_id):
            raise OriginalCodex56SolContractError("beat graph reference is invalid")
    mids = [m.line_id for m in manifest.lines]
    if len(mids) != len(set(mids)) or set(m.beat_id for m in manifest.lines) != set(beats):
        raise OriginalCodex56SolContractError("manifest must cover every beat exactly")
    sids = [line.line_id for line in script.lines]
    if sids != mids:
        raise OriginalCodex56SolContractError("script line order/coverage differs from manifest")
    by_manifest = {m.line_id: m for m in manifest.lines}
    for line in script.lines:
        m = by_manifest[line.line_id]
        if line.char_id != m.char_id or line.speaker != m.speaker:
            raise OriginalCodex56SolContractError("script roster differs from manifest")
    for required in (manifest.orientation_line_id, manifest.reveal_line_id,
                     manifest.closure_line_id):
        if required not in set(mids):
            raise OriginalCodex56SolContractError("manifest landmark is missing")


def _validate_text(script: PerformanceScript, rules: Any) -> None:
    for line in script.lines:
        text = line.text.strip()
        if not text:
            raise OriginalCodex56SolContractError("spoken text is empty")
        for term in getattr(rules, "banned_phrases", ()):
            if re.search(rf"\b{re.escape(term)}\b", text, re.I):
                raise OriginalCodex56SolContractError(f"forbidden term {term!r}")
        for pattern in getattr(rules, "stage_business", ()):
            if pattern.search(text):
                raise OriginalCodex56SolContractError("forbidden authored surface")
        if re.search(r"^[A-Z][A-Z0-9 _-]{1,24}:\s*", text):
            raise OriginalCodex56SolContractError("speaker label in spoken text")


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
    ingress = _call(pass_id="P0", slot="technical", fn=technical_fn, pack=pack, seam="codex56_constraint_ingress", inputs={"draw": draw.model_dump(mode="json")}, schema=ConstraintIngress, scheduler=slot_scheduler, journal=journal, tokens=1200)
    if ingress.draw != draw:
        raise OriginalCodex56SolContractError("technical ingress changed the constraint draw")
    slate = _call(pass_id="P1", slot="creative", fn=creative_fn, pack=pack, seam="codex56_possibility_slate", inputs={"ingress": ingress.model_dump(mode="json"), "operator_hint": (meta.get("source_meta") or {}).get("operator_hint", "")}, schema=PossibilitySlate, scheduler=slot_scheduler, journal=journal, post_validator=lambda value: _validate_slate(value, draw))
    triage = _call(pass_id="P2", slot="technical", fn=technical_fn, pack=pack, seam="codex56_slate_triage", inputs={"slate": slate.model_dump(mode="json")}, schema=SlateTriage, scheduler=slot_scheduler, journal=journal, tokens=1600, post_validator=lambda value: _validate_triage(value, slate))
    selected = next((p for p in slate.possibilities if p.possibility_id == triage.selected_possibility_id), None)
    if selected is None or any(f.blocking and f.possibility_id == selected.possibility_id for f in triage.findings):
        raise OriginalCodex56SolContractError("triage did not select a valid possibility")
    truth = _call(pass_id="P3", slot="creative", fn=creative_fn, pack=pack, seam="codex56_audible_truth_map", inputs={"selected": selected.model_dump(mode="json"), "draw": draw.model_dump(mode="json")}, schema=AudibleTruthMap, scheduler=slot_scheduler, journal=journal, tokens=2800)
    fair = _call(pass_id="P4", slot="technical", fn=technical_fn, pack=pack, seam="codex56_fair_play_audit", inputs={"truth_map": truth.model_dump(mode="json")}, schema=FairPlayReport, scheduler=slot_scheduler, journal=journal, tokens=1600)
    if not fair.accepted or any(f.blocking for f in fair.findings):
        raise OriginalCodex56SolContractError("fair-play audit rejected the truth map")
    score = _call(pass_id="P5", slot="creative", fn=creative_fn, pack=pack, seam="codex56_broadcast_score", inputs={"truth_map": truth.model_dump(mode="json"), "target_words_advisory": int(resolved["target_words"]), "num_characters_advisory": int(resolved["num_characters"])}, schema=BroadcastScore, scheduler=slot_scheduler, journal=journal, tokens=3600)
    manifest = _call(pass_id="P5_manifest", slot="technical", fn=technical_fn, pack=pack, seam="codex56_closed_line_manifest", inputs={"score": score.model_dump(mode="json")}, schema=ClosedLineManifest, scheduler=slot_scheduler, journal=journal, tokens=3000)
    script = _call(pass_id="P6", slot="creative", fn=creative_fn, pack=pack, seam="codex56_performance_script", inputs={"score": score.model_dump(mode="json"), "manifest": manifest.model_dump(mode="json"), "target_words_advisory": int(resolved["target_words"])}, schema=PerformanceScript, scheduler=slot_scheduler, journal=journal, tokens=max(2600, int(resolved["target_words"]) * 6))
    _validate_graph(score, manifest, script); _validate_text(script, story_rules)
    listener = _call(pass_id="P7", slot="technical", fn=technical_fn, pack=pack, seam="codex56_blind_listener", inputs={"manifest": manifest.model_dump(mode="json"), "script": script.model_dump(mode="json")}, schema=BlindListenerReport, scheduler=slot_scheduler, journal=journal, tokens=1800)
    if any(f.blocking for f in listener.findings):
        script = _call(pass_id="P8", slot="creative", fn=creative_fn, pack=pack, seam="codex56_broadcast_retake", inputs={"manifest": manifest.model_dump(mode="json"), "previous_script": script.model_dump(mode="json"), "findings": listener.model_dump(mode="json")}, schema=PerformanceScript, scheduler=slot_scheduler, journal=journal, tokens=max(2600, int(resolved["target_words"]) * 6))
        _validate_graph(score, manifest, script); _validate_text(script, story_rules)
    audit = _call(pass_id="P9", slot="technical", fn=technical_fn, pack=pack, seam="codex56_final_contract_audit", inputs={"manifest": manifest.model_dump(mode="json"), "script": script.model_dump(mode="json")}, schema=FinalContractAudit, scheduler=slot_scheduler, journal=journal, tokens=1800)
    if not audit.accepted or any(f.blocking for f in audit.findings):
        raise OriginalCodex56SolContractError("final contract audit rejected the script")

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
