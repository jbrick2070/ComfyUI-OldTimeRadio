"""F2: applied scene corrections, bounded calls, and current dispatch receipts."""
import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from nodes import _otr_story_source as source
from nodes import otr_meta_brief_image_prompt as mb
from nodes import otr_image_gen_dispatcher as dispatcher
from nodes._otr_story_brief_helpers import NO_TEXT_CLAUSE


class Slot:
    def __init__(self, reply=None, fit=None):
        self.reply = reply if reply is not None else {
            "prompt": "Ada, her recognizable round face visible, shares dinner with her living mother."}
        self.fit = fit
        self.calls = []
        self.inspections = []

    def _otr_inspect_fit(self, messages, *, max_new_tokens):
        assert max_new_tokens is None
        self.inspections.append(copy.deepcopy(messages))
        return self.fit or {"supported": True, "capacity_known": True, "fits": True}

    def __call__(self, messages, *, max_new_tokens, **kwargs):
        assert max_new_tokens is None
        assert messages._otr_reserve_remaining_output_capacity
        self.calls.append(copy.deepcopy(messages))
        answer = self.reply(messages) if callable(self.reply) else self.reply
        return answer if isinstance(answer, str) else json.dumps(answer)


def _ledger():
    return {
        "episode_id": "source_scene_test",
        "meta": {
            "source_bank": "my_story", "visual_style": "sci_fi_radio",
            "story_brief": "A shared dinner in Los Angeles.", "story_brief_status": "ok",
            "story_brief_terms": {"setting": ["Los Angeles kitchen"]},
            "my_story": {"treatment": {"acts": [{"n": 1, "scene_setting": "kitchen"},
                                                 {"n": 2, "scene_setting": "garden"}]}},
            "source_meta": {"story_input": {"fields": {
                "idea": "  Ordinary background. " * 60 + "\nMother is alive 🌠 and joins Ada for dinner.\t",
                "characters": "Ada and Mother; Tom is their friend.",
                "plot": "Ada and Mother eat together. Tom speaks later in the garden.",
                "setting": "Los Angeles", "author": "Not a story fact"}}},
        },
        "cast": [
            {"char_id": "c1", "name": "Ada", "character_description": "round face, dark curls"},
            {"char_id": "c2", "name": "Mother", "character_description": "silver hair, kind eyes"},
            {"char_id": "c3", "name": "Tom", "character_description": "tall gardener in overalls"}],
        "lines": [
            {"line_id": "l1", "shot_id": "shot1", "speaker_role": "character", "char_id": "c1",
             "speaker": "Ada", "text": "Mother, sit beside me. " + "We talk. " * 40 + "A shared meal.",
             "start_s": 0.0, "dur_s": 5.0},
            {"line_id": "l2", "shot_id": "shot1", "speaker_role": "character", "char_id": "c2",
             "speaker": "Mother", "text": "I am right here.", "start_s": 5.0, "dur_s": 5.0},
            {"line_id": "l3", "shot_id": "shot2", "speaker_role": "character", "char_id": "c3",
             "speaker": "Tom", "text": "The garden is quiet.", "start_s": 10.0, "dur_s": 5.0}],
        "beats": [
            {"beat_id": "beat1", "shot_id": "shot1", "scene_id": "scene1", "line_ids": ["l1"]},
            {"beat_id": "beat2", "shot_id": "shot1", "scene_id": "scene1", "line_ids": ["l2"]},
            {"beat_id": "beat3", "shot_id": "shot2", "scene_id": "scene2", "line_ids": ["l3"]}],
        "shots": [{"shot_id": "shot1", "scene_id": "scene1", "description": "dinner"},
                  {"shot_id": "shot2", "scene_id": "scene2", "description": "garden"}],
        "scenes": [{"scene_id": "scene1", "description": "Los Angeles kitchen"},
                   {"scene_id": "scene2", "description": "garden"}],
    }


def _context(ledger):
    return mb._scene_source_context(
        ledger["meta"], ledger["cast"], ledger["lines"],
        {"beat_id": "l1", "char_id": "c1"}, ledger["lines"][0], ledger)


def _compose(slot, *, ledger=None, max_reseed=2, receipts=None):
    ledger = ledger or _ledger()
    journal = receipts if receipts is not None else []
    result = mb._compose_char_scene_prompt(
        ledger["meta"], ledger["cast"][0], "Los Angeles kitchen", ledger["lines"][0],
        None, [], "c1", max_reseed=max_reseed, source_context=_context(ledger),
        source_slot_fn=slot, source_model_id="configured-technical",
        source_binding_model_id="loaded-technical", source_receipts=journal)
    return result, journal[-1]


def _video_models():
    return {"announcer_video_model": {"engine_id": "viz_mxc_cpu"},
            "music_video_model": {"engine_id": "viz_mxc_mandala"},
            "character_video_model": {"engine_id": "still_motion"}}


def _derive(ledger, slot, **kwargs):
    return mb.derive_image_prompts(
        ledger["cast"], ledger["meta"], lines=ledger["lines"], ledger_context=ledger,
        source_slot_fn=slot, video_models=_video_models(), **kwargs)


def test_complete_source_scene_dialogue_and_target_reach_applied_correction():
    ledger = _ledger()
    original = copy.deepcopy(ledger)
    slot = Slot()
    (prompt, provenance), receipt = _compose(slot, ledger=ledger)
    request = json.loads(slot.calls[0][1]["content"])
    raw = ledger["meta"]["source_meta"]["story_input"]["fields"]
    assert request["source"] == {key: raw[key] for key in source.CREATIVE_FIELDS}
    context = request["authoring_context"]
    assert context["beat"]["beat_id"] == "beat1"  # line-id targets join real beat ids
    assert context["current_line"]["text"] == ledger["lines"][0]["text"]
    assert [row["line_id"] for row in context["ordered_scene_dialogue"]] == ["l1", "l2"]
    assert context["target_character"]["appearance"] == "round face, dark curls"
    companions = {row["name"]: row for row in context["candidate_companions"]}
    assert companions["Mother"]["speaks_in_scene"] is True
    assert companions["Tom"]["speaks_in_scene"] is False
    assert companions["Mother"]["appearance"] == "silver hair, kind eyes"
    assert "Do not force every act speaker into every frame" in slot.calls[0][0]["content"]
    assert prompt.startswith(slot.reply["prompt"]) and prompt.endswith(NO_TEXT_CLAUSE)
    assert provenance == "char_scene_source_rewrite"
    assert receipt["applied"] and receipt["status"] == "rewritten"
    assert receipt["retained_prompt"] == prompt
    assert receipt["base_prompt_hash"] == mb._content_hash(prompt)
    assert receipt["executed_model_id"] is None
    assert receipt["binding_model_id"] == "loaded-technical"
    assert receipt["configured_model_id"] == "configured-technical"
    assert not receipt["qualified"] and len(slot.calls) == 1
    assert ledger == original


def test_actual_shared_source_call_does_not_change_neutral_portrait_scope():
    ledger = _ledger()
    slot = Slot()
    _compose(slot, ledger=ledger)
    request = json.loads(slot.calls[0][1]["content"])
    assert request["source"]["idea"] == _context(ledger)["raw_fields"]["idea"]
    assert "candidate_companions" in request["authoring_context"]
    portrait = mb._build_char_prompt_request(ledger["cast"][0], ledger["meta"], "kitchen")
    assert "CURRENT SCENE CONTEXT" not in portrait and "candidate_companions" not in portrait


@pytest.mark.parametrize("row_age", [None, "", "  ", "n/a", " N/A "])
def test_scene_owner_receives_current_ages_and_shared_action_without_literalizing_memory(row_age):
    ledger = _ledger()
    for row in ledger["cast"]:
        row["age_band"] = row_age
    ledger["meta"]["my_story"]["treatment"]["cast"] = [
        {"name": "Ada", "age_band": "30s", "gender": "female"},
        {"name": "Mother", "age_band": "50s", "gender": "female"},
        {"name": "Tom", "age_band": "40s", "gender": "male"}]
    ledger["lines"][0]["text"] = "Mother, remember when I was small? I love sharing dinner with you now."
    before_hash = source.candidate_sha256(_context(ledger))
    slot = Slot()
    _compose(slot, ledger=ledger)
    request = json.loads(slot.calls[0][1]["content"])
    context = request["authoring_context"]
    assert context["target_character"]["age_band"] == "30s"
    assert context["candidate_companions"][0]["age_band"] == "50s"
    visual_request = context["visual_request"]
    assert "active speaker is the focus within that scene" in visual_request
    assert "Spoken memories" in visual_request and "objects those actions require" in visual_request
    assert "Do not force every act speaker into every frame" in slot.calls[0][0]["content"]
    assert context["candidate_companions"][1]["speaks_in_scene"] is False
    ledger["meta"]["my_story"]["treatment"]["cast"][0]["age_band"] = "40s"
    assert source.candidate_sha256(_context(ledger)) != before_hash
    portrait = mb._build_char_prompt_request(ledger["cast"][0], ledger["meta"], "kitchen")
    assert "shared actions" not in portrait and "Spoken memories" not in portrait


def test_scene_age_join_does_not_borrow_from_a_different_or_ambiguous_cast_name():
    ledger = _ledger()
    ledger["meta"]["my_story"]["treatment"]["cast"] = [
        {"name": "Ada", "age_band": "30s"}, {"name": "ADA", "age_band": "60s"},
        {"name": "Mother's friend", "age_band": "50s"}]
    context = _context(ledger)["scene"]
    assert context["target_character"]["age_band"] == ""
    assert context["candidate_companions"][0]["age_band"] == ""


def test_scene_unknown_treatment_age_is_absent_and_known_row_age_wins():
    ledger = _ledger()
    ledger["cast"][0]["age_band"] = "40s"
    ledger["meta"]["my_story"]["treatment"]["cast"] = [
        {"name": "Ada", "age_band": "30s"}, {"name": "Mother", "age_band": "n/a"}]
    context = _context(ledger)["scene"]
    assert context["target_character"]["age_band"] == "40s"
    assert context["candidate_companions"][0]["age_band"] == ""


def test_corrected_narrative_appearance_is_not_prepended_back_into_prompt():
    ledger = _ledger()
    ledger["cast"][0]["character_description"] = "round face, mourning her dead mother, waiting alone"
    (prompt, _), receipt = _compose(Slot(), ledger=ledger)
    assert "dead mother" not in prompt and "waiting alone" not in prompt
    assert "living mother" in prompt
    assert "dead mother" in receipt["scene_context"]["target_character"]["appearance"]


@pytest.mark.parametrize("reply", ["broken", "{}", '{"prompt": "   "}'])
@pytest.mark.parametrize("max_reseed,expected", [(0, 1), (2, 2), (50, 2)])
def test_scene_correction_has_one_budget_including_malformed_and_schema_repairs(reply, max_reseed, expected):
    slot = Slot(reply)
    (prompt, provenance), receipt = _compose(slot, max_reseed=max_reseed)
    assert len(slot.calls) == expected and receipt["attempt_limit"] == expected
    assert len(receipt["attempts"]) == expected
    assert provenance == "char_scene_source_unresolved" and prompt
    assert receipt["source_context_hash"] == source.candidate_sha256(_context(_ledger()))
    assert not receipt["qualified"] and not receipt["applied"]


def test_malformed_reply_is_repaired_once_and_actual_second_prompt_is_used():
    failed = {"prompt": ["a wrong shape " * 80 + "UNICODE TAIL 🌠"]}
    slot = Slot(lambda messages: failed if len(slot.calls) == 1 else {"prompt": "A shared dinner."})
    (prompt, _), receipt = _compose(slot)
    assert prompt.startswith("A shared dinner.") and len(slot.calls) == 2
    assert any(m["role"] == "assistant" and json.loads(m["content"]) == failed for m in slot.calls[1])
    assert receipt["status"] == "rewritten"


@pytest.mark.parametrize("error", [RuntimeError("provider failed"), MemoryError("OOM"),
                                  KeyboardInterrupt("cancelled")])
def test_provider_oom_and_cancellation_escape_without_becoming_empty_scene(error):
    def fail(messages):
        raise error
    slot = Slot(fail)
    journal = []
    with pytest.raises(type(error), match=str(error)):
        _compose(slot, receipts=journal)
    assert len(slot.calls) == 1
    assert journal[0]["status"] == "provider_error"
    assert journal[0]["scene_context"]["beat_id"] == "l1"


def test_failed_visual_operation_is_logged_with_context_before_provider_error_escapes(caplog, monkeypatch):
    error = RuntimeError("provider unavailable")
    def fail(messages):
        raise error
    with caplog.at_level(logging.ERROR, logger=mb.log.name):
        with pytest.raises(RuntimeError) as raised:
            _compose(Slot(fail))
    assert raised.value is error
    records = [record.getMessage() for record in caplog.records if "SOURCE_REWRITE_FAILED " in record.getMessage()]
    assert len(records) == 1
    receipt = json.loads(records[0].split("SOURCE_REWRITE_FAILED ", 1)[1])
    assert receipt["scene_context"]["beat_id"] == "l1"
    assert receipt["source_context_hash"] == source.candidate_sha256(_context(_ledger()))
    assert receipt["attempts"][0]["error"] == "provider unavailable"
    assert receipt["status"] == "provider_error" and not receipt["qualified"]
    assert receipt["executed_model_id"] is None

    def broken_log(*args, **kwargs):
        raise KeyboardInterrupt("logging interrupted")
    monkeypatch.setattr(mb.log, "error", broken_log)
    with pytest.raises(RuntimeError) as raised:
        _compose(Slot(fail))
    assert raised.value is error


def test_unavailable_or_measured_no_room_keeps_fallback_and_no_semantic_pass():
    for slot in (None, Slot(fit={"supported": True, "capacity_known": True, "fits": False})):
        (prompt, provenance), receipt = _compose(slot)
        assert prompt and provenance == "char_scene_source_unresolved"
        assert not receipt["qualified"] and not receipt["applied"]
        assert receipt["executed_model_id"] is None
        assert slot is None or not slot.calls


def test_derived_payload_applies_each_correction_and_forwards_zero_reseed_budget():
    ledger = _ledger()
    slot = Slot("{}")
    payload, _ = _derive(ledger, slot, max_reseed=0)
    scenes = [row for row in payload["objects"] if row["kind"] == "scene_character"]
    assert len(scenes) == 3 and len(slot.calls) == 3
    assert all(row["source_rewrite"]["attempt_limit"] == 1 for row in scenes)
    assert all(row["source_context_hash"] for row in scenes)
    assert all("source_rewrite" not in row for row in payload["objects"] if row["kind"] == "portrait")


def test_legacy_scene_reseed_is_forwarded_too():
    ledger = _ledger()
    ledger["meta"].pop("my_story")
    scene_calls = []
    def call(request):
        if "cinematic STILL-image" in request:
            scene_calls.append(request)
            return ""
        return "a neutral portrait"
    _derive(ledger, None, llm_fn=call, max_reseed=0)
    assert len(scene_calls) == 3


def test_wired_node_resolves_raw_owner_once_and_keeps_portraits_separate(monkeypatch):
    from nodes import otr_shot_lock as shot_lock
    ledger = _ledger()
    slot = Slot()
    resolutions = []
    monkeypatch.setattr(shot_lock, "_resolve_writer_llm_binding",
                        lambda meta, warnings: (resolutions.append(meta) or slot, "loaded-technical"))
    monkeypatch.setattr(shot_lock, "writer_model_id_from_meta", lambda meta: "configured-technical")
    monkeypatch.setattr(shot_lock, "_writer_call_at", lambda gen, budget: lambda prompt: "a neutral portrait")
    result, _ = mb.OTRMetaBriefImagePromptGen().generate(
        json.dumps(ledger), json.dumps({"policy_version": 2, "video_models": _video_models()}))
    scenes = [row for row in json.loads(result)["objects"] if row["kind"] == "scene_character"]
    assert len(resolutions) == 1 and len(slot.calls) == 3
    assert all(row["source_rewrite"]["executed_model_id"] is None for row in scenes)
    assert all(row["source_rewrite"]["binding_model_id"] == "loaded-technical" for row in scenes)
    assert all(row["prompt"].startswith(slot.reply["prompt"]) for row in scenes)


def test_normalizer_preserves_old_fields_and_records_actual_final_text():
    from nodes._otr_visual_styles import get_visual_style
    legacy = dispatcher._NormalizedPrompt("x", False, "", None, {}, "hash")
    assert legacy.prompt_hash == "hash" and legacy.base_prompt_hash == ""
    raw = "  Ada carries a revolver.  "
    normalized = dispatcher.normalize_prompt_for_render(
        raw, vstyle=get_visual_style({"visual_style": "archival_documentary"}),
        banana_on=True, banana_key="test-episode")
    assert normalized.base_prompt_hash == dispatcher._prompt_content_hash(raw)
    assert normalized.prompt_hash == dispatcher._prompt_content_hash(normalized.text)
    receipt = normalized.normalization_receipt
    assert receipt["safety_changed"] and receipt["style_changed"] and receipt["banana_changed"]
    assert receipt["final_prompt"] == normalized.text and "revolver" not in normalized.text
    assert not receipt["source_qualified"]


@pytest.fixture
def image_dispatch(monkeypatch, tmp_path):
    import numpy as np
    from nodes._otr_image_engines import registry
    from nodes._otr_shared import role_compat
    from nodes import production_ledger
    saved = dict(registry._IMAGE_REGISTRY._registry)
    registry._IMAGE_REGISTRY._registry.clear()
    registry.register(SimpleNamespace(
        name="source_image_stub", roles=role_compat.ROLES, default_roles=role_compat.ROLES,
        commercial_clean=True, requires_flag=None, required_inputs=("text_prompt",), engine_version="1"))
    monkeypatch.setattr(dispatcher._levers, "free_otr_pipeline_residue", lambda **kw: {})
    stamps = []
    monkeypatch.setattr(production_ledger, "stamp_durable", lambda **kw: stamps.append(copy.deepcopy(kw)))
    calls = []
    def generate(request):
        calls.append(copy.deepcopy(request))
        return np.full((8, 8, 3), 70 + len(calls), dtype=np.uint8)
    def dispatch(ledger, objects):
        policy = {"policy_version": 2, "video_models": _video_models(),
                  "image_models": {"character_image_model": {"engine_id": "source_image_stub"}},
                  "seed": {"request_seed": 1}, "granularity": {}}
        return dispatcher.dispatch_images(
            ledger, policy, {"version": 1, "objects": objects}, gen_fn=generate,
            output_dir=str(tmp_path), lockdir=tmp_path / "source-lease")
    try:
        yield dispatch, calls, stamps
    finally:
        registry._IMAGE_REGISTRY._registry.clear()
        registry._IMAGE_REGISTRY._registry.update(saved)


def _scene_object(ledger):
    payload, _ = _derive(ledger, Slot())
    obj = next(row for row in payload["objects"] if row["object_id"] == "still_l1")
    obj["identity"] = "none"
    obj.pop("identity_prompt", None)
    return obj


def test_fresh_cache_and_durable_manifest_use_current_receipt(image_dispatch, monkeypatch):
    monkeypatch.setenv("OTR_BANANA_STILLS", "0")
    dispatch, calls, stamps = image_dispatch
    ledger = _ledger()
    obj = _scene_object(ledger)
    ledger, *_ = dispatch(ledger, [copy.deepcopy(obj)])
    fresh = ledger["images"]["images"][-1]
    assert fresh["source_rewrite"]["base_matches_rewrite"]
    assert fresh["source_rewrite"]["final_prompt"] == calls[0]["prompt"]
    assert fresh["prompt_hash"] == dispatcher._prompt_content_hash(calls[0]["prompt"])
    current = copy.deepcopy(obj)
    current["source_rewrite"]["operation_id"] = "current-operation"
    ledger, *_ = dispatch(ledger, [current])
    hit = ledger["images"]["images"][-1]
    assert len(calls) == 1 and hit["provenance"]["source"] == "cache_hit"
    assert hit["source_rewrite"]["operation_id"] == "current-operation"
    assert not hit["source_rewrite"]["qualified"]
    assert stamps[-1]["sections"]["images"]["images"][-1]["source_rewrite"] == hit["source_rewrite"]
    manifest = json.loads((Path(hit["path"]).parent / "stills_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stills"][-1]["source_rewrite"] == hit["source_rewrite"]
    assert manifest["stills"][-1]["source_context_hash"] == obj["source_context_hash"]


def test_changed_source_or_dialogue_invalidates_cache_even_with_stale_payload(image_dispatch):
    dispatch, calls, _ = image_dispatch
    ledger = _ledger()
    obj = _scene_object(ledger)
    ledger, *_ = dispatch(ledger, [copy.deepcopy(obj)])
    old_hash = ledger["images"]["images"][-1]["source_context_hash"]
    ledger["meta"]["source_meta"]["story_input"]["fields"]["plot"] += " The door stays open."
    ledger, *_ = dispatch(ledger, [copy.deepcopy(obj)])
    revised = ledger["images"]["images"][-1]
    assert len(calls) == 2 and revised["source_context_hash"] != old_hash
    assert revised["source_rewrite"]["dispatch_disposition"] == "stale_source_receipt"
    ledger["lines"][1]["text"] = "I join you by the open door."
    ledger, *_ = dispatch(ledger, [copy.deepcopy(obj)])
    assert len(calls) == 3
    ledger["meta"]["story_brief_terms"]["setting"] = ["Los Angeles garden"]
    ledger, *_ = dispatch(ledger, [copy.deepcopy(obj)])
    revised = ledger["images"]["images"][-1]
    assert len(calls) == 4
    assert revised["source_rewrite"]["dispatch_disposition"] == "stale_source_receipt"


def test_real_jump_merge_keeps_scene_source_scope_and_invalidates_cache(image_dispatch):
    from tests.test_multiclip_jump_stills import _request_ledger
    dispatch, calls, _ = image_dispatch
    ledger = _ledger()
    obj = _scene_object(ledger)
    ledger["video"] = _request_ledger(beat="l1")["video"]
    merged, _, _ = dispatcher.merge_jump_still_requests(ledger, [obj], [])
    assert merged[1]["source_scene_scope"] == "scene_character"
    assert dispatcher._current_source_context_hash(ledger, merged[1]) == obj["source_context_hash"]
    ledger, *_ = dispatch(ledger, [copy.deepcopy(obj)])
    assert len(calls) == 2
    assert all(row["source_rewrite"]["base_matches_rewrite"] for row in ledger["images"]["images"])
    ledger, *_ = dispatch(ledger, [copy.deepcopy(obj)])
    assert len(calls) == 2
    ledger["meta"]["source_meta"]["story_input"]["fields"]["plot"] += " Mother opens the door."
    ledger, *_ = dispatch(ledger, [copy.deepcopy(obj)])
    assert len(calls) == 4
    assert all(row["source_rewrite"]["dispatch_disposition"] == "stale_source_receipt"
               for row in ledger["images"]["images"][-2:])


def test_changed_prompt_discloses_stale_evidence_and_banana_never_claims_pass(image_dispatch, monkeypatch):
    monkeypatch.setenv("OTR_BANANA_STILLS", "1")
    dispatch, calls, _ = image_dispatch
    ledger = _ledger()
    obj = _scene_object(ledger)
    obj["prompt"] += ", a revolver on the table"
    ledger, *_ = dispatch(ledger, [obj])
    row = ledger["images"]["images"][-1]
    receipt = row["source_rewrite"]
    assert receipt["dispatch_disposition"] == "stale_source_receipt"
    assert receipt["normalization"]["banana_changed"] and "revolver" not in calls[0]["prompt"]
    assert receipt["final_prompt_hash"] == row["prompt_hash"]
    assert not receipt["qualified"]


def test_cache_hit_clears_removed_source_receipt(image_dispatch):
    dispatch, calls, _ = image_dispatch
    ledger = _ledger()
    obj = _scene_object(ledger)
    ledger, *_ = dispatch(ledger, [copy.deepcopy(obj)])
    obj.pop("source_rewrite")
    ledger, *_ = dispatch(ledger, [obj])
    assert len(calls) == 1
    assert ledger["images"]["images"][-1]["source_rewrite"] is None
