"""Source correction tests use canned slots and the Windows CPU test runtime."""
import copy
import hashlib
import json

import pytest
from pydantic import BaseModel

from nodes import _otr_story_source as source
from nodes._otr_generation_budget import PromptContextOverflowError
from nodes._otr_story_input import RawStoryFields


class Draft(BaseModel):
    ending: str
    people: list[str]


class Slot:
    def __init__(self, reply=None, fit=None):
        self.reply = reply if reply is not None else {"ending": "Mother lives.", "people": ["Mother"]}
        self.fit = fit
        self.calls = []
        self.inspections = []

    def _otr_inspect_fit(self, messages, *, max_new_tokens):
        assert max_new_tokens is None
        assert messages._otr_reserve_remaining_output_capacity
        self.inspections.append(copy.deepcopy(messages))
        return self.fit or {"supported": True, "capacity_known": True, "fits": True}

    def __call__(self, messages, *, max_new_tokens, **kwargs):
        assert max_new_tokens is None
        assert messages._otr_reserve_remaining_output_capacity
        self.calls.append(copy.deepcopy(messages))
        reply = self.reply(messages) if callable(self.reply) else self.reply
        return reply if isinstance(reply, str) else json.dumps(reply)


def _run(slot, *, raw=None, candidate=None, receipts=None, **kwargs):
    return source.rewrite_story_source(
        raw if raw is not None else RawStoryFields(idea="Mother is alive and joins dinner."),
        candidate if candidate is not None else {"ending": "Mother died.", "people": []},
        slot, schema=Draft, receipts=receipts if receipts is not None else [],
        pass_id="treatment", configured_model_id="test-creative", **kwargs)


def test_raw_documents_keep_exact_unicode_whitespace_and_exclude_author():
    raw = RawStoryFields(idea="  Mother 🌠\n lives.\t", plot=" \n", author="A. Listener")
    documents = source.build_raw_documents(raw)
    assert set(documents) == {"idea"}
    assert documents["idea"].canonical_body == raw.idea
    assert documents["idea"].body_sha256 == hashlib.sha256(raw.idea.encode("utf-8")).hexdigest()
    assert documents["idea"].normalization_version == source.RAW_COORDINATE_VERSION
    assert raw.idea in source.raw_source_block(raw)
    assert raw.author not in source.raw_source_block(raw)
    with pytest.raises(TypeError):
        json.dumps(documents)


def test_combined_operation_returns_corrected_artifact_including_missing_list_items():
    raw = RawStoryFields(idea="Ordinary life. " * 70 + "Mother is alive and joins dinner.")
    slot = Slot()
    result, receipt = _run(slot, raw=raw)
    assert result.ending == "Mother lives." and result.people == ["Mother"]
    assert len(slot.calls) == 1
    payload = json.loads(slot.calls[0][1]["content"])
    assert payload["source"]["idea"] == raw.idea
    assert payload["draft"]["ending"] == "Mother died."
    assert receipt["source_intervals"] == [{"field": "idea", "start_char": 0, "end_char": len(raw.idea)}]
    assert not receipt["qualified"] and not receipt["applied"]
    assert receipt["returned_artifact"] == result.model_dump()
    assert receipt["executed_model_id"] is None


@pytest.mark.parametrize("response", ["broken", "{}", '{"ending": "x"}'])
def test_stubborn_failure_stops_at_two_actual_calls_without_a_fourth_or_fifth_round(response):
    slot = Slot(response)
    journal = []
    for revision in range(5):
        result, receipt = _run(slot, candidate={"ending": "draft %s" % revision, "people": []},
                               receipts=journal)
        assert result is None
    assert len(slot.calls) == 2
    assert journal[0]["status"] == "unresolved"
    assert all(row["status"] == "budget_already_spent" for row in journal[1:])
    assert all(row["attempt_limit"] == 0 for row in journal[1:])


def test_second_attempt_repairs_schema_and_keeps_the_entire_failed_artifact():
    failed = {"ending": "long text " * 80 + "THE VERY LAST WORD", "people": 7}
    slot = Slot(lambda messages: failed if len(slot.calls) == 1 else
                {"ending": "Mother lives.", "people": ["Mother"]})
    result, receipt = _run(slot)
    assert result.people == ["Mother"]
    assert len(slot.calls) == 2
    repair = slot.calls[1]
    assert any(m["role"] == "assistant" and json.loads(m["content"]) == failed for m in repair)
    assert "THE VERY LAST WORD" in str(repair)
    assert len(receipt["attempts"]) == 2


def test_existing_structural_validator_runs_on_the_exact_returned_model():
    def validate(model):
        if "Outsider" in model.people:
            return "Keep the locked cast"
        model.people[:] = [name.upper() for name in model.people]
    slot = Slot(lambda messages: {"ending": "Mother lives.",
                                  "people": ["Outsider" if len(slot.calls) == 1 else "Mother"]})
    result, receipt = _run(slot, post_validator=validate)
    assert len(slot.calls) == 2
    assert result.people == ["MOTHER"]
    assert receipt["returned_artifact"]["people"] == ["MOTHER"]


def _rewrite_interpretation(candidate, reply, **kwargs):
    from nodes._otr_my_story import StoryInterpretation
    slot = Slot(reply)
    result, receipt = source.rewrite_story_source(
        RawStoryFields(idea="Jeffrey and his mother share dinner."), candidate, slot,
        schema=StoryInterpretation, receipts=[], pass_id="interpret",
        preserve_omitted={("requirements",): "id", ("named_cast",): "name"}, **kwargs)
    return result, receipt, slot


def test_sparse_source_correction_conserves_metadata_by_identity_not_position():
    from nodes._otr_my_story import StoryInterpretation
    candidate = StoryInterpretation.model_validate({
        "requirements": [{"id": "dinner", "text": "Share dinner", "kind": "event",
                          "source_field": "plot", "strength": "preferred"},
                         {"id": "place", "text": "In LA", "kind": "setting",
                          "source_field": "setting"}],
        "named_cast": [{"name": "Jeffrey", "stated_gender": "male"},
                       {"name": "Mother", "notes": "Present throughout"}],
        "assumptions": ["A warm evening"],
    }).model_dump(mode="json")
    before = copy.deepcopy(candidate)
    reply = {"requirements": [{"id": "place", "text": "In LA"},
                              {"id": "dinner", "text": "Share dinner"}],
             "named_cast": [{"name": "Mother"}, {"name": "Jeffrey"}]}
    result, receipt, slot = _rewrite_interpretation(candidate, reply)
    assert result.requirements[1].source_field == "plot"
    assert result.requirements[1].strength == "preferred"
    assert result.requirements[0].kind == "setting"
    assert result.named_cast[1].stated_gender == "male"
    assert result.named_cast[0].notes == "Present throughout"
    assert result.assumptions == candidate["assumptions"]
    assert receipt["returned_artifact"] == result.model_dump(mode="json")
    assert candidate == before and len(slot.calls) == 1


def test_explicit_corrections_clears_and_list_membership_are_authoritative():
    candidate = {"named_cast": [{"name": "Jeffrey", "stated_gender": "male", "notes": "old"},
                                {"name": "Removed", "notes": "Do not resurrect"}],
                 "requirements": [{"id": "old", "kind": "event"}],
                 "assumptions": ["old"]}
    reply = {"named_cast": [{"name": "Jeffrey", "stated_gender": "", "notes": "",
                             "speaking": False, "required": False},
                            {"name": "Added"}], "requirements": [], "assumptions": []}
    result, _, _ = _rewrite_interpretation(candidate, reply)
    assert [r.name for r in result.named_cast] == ["Jeffrey", "Added"]
    assert result.named_cast[0].stated_gender == result.named_cast[0].notes == ""
    assert not result.named_cast[0].speaking and not result.named_cast[0].required
    assert not result.requirements and not result.assumptions


@pytest.mark.parametrize("rows", [[{"name": "A"}, {"name": "A"}], [{}], [{"name": "Renamed"}]])
def test_ambiguous_or_changed_identity_never_inherits_another_cast_member(rows):
    result, _, _ = _rewrite_interpretation(
        {"named_cast": [{"name": "A", "stated_gender": "female"}]}, {"named_cast": rows})
    assert all(not row.stated_gender for row in result.named_cast)


def test_rejected_proposal_is_not_the_next_baseline_and_validator_result_is_journaled():
    from nodes._otr_my_story import StoryInterpretation
    seen = []

    def validate(model):
        seen.append(model)
        if model.setting_brief == "wrong":
            return "wrong setting"
        model.setting_brief = "normalized"

    slot = Slot(lambda messages: {"setting_brief": "wrong", "assumptions": ["bad"]}
                if len(slot.calls) == 1 else {"setting_brief": "right"})
    result, receipt = source.rewrite_story_source(
        RawStoryFields(idea="Dinner"), {"assumptions": ["original"]}, slot,
        schema=StoryInterpretation, receipts=[], pass_id="interpret",
        preserve_omitted={}, post_validator=validate)
    assert len(slot.calls) == 2 and result is seen[-1]
    assert result.assumptions == ["original"] and result.setting_brief == "normalized"
    assert receipt["returned_artifact"] == result.model_dump(mode="json")


@pytest.mark.parametrize("register,expected", [(None, "warm"), ("", ""), ("formal", "formal")])
def test_treatment_alias_and_explicit_act_identity_preserve_the_correct_metadata(register, expected):
    from nodes._otr_my_story import StoryTreatment
    candidate = StoryTreatment.model_validate({
        "cast": [{"name": "Jeffrey", "register": "warm"}],
        "acts": [{"n": 1, "purpose": "dinner"}, {"n": 2, "purpose": "departure"}],
    }).model_dump(mode="json")
    member = {"name": "Jeffrey"}
    if register is not None:
        member["register"] = register
    slot = Slot({"cast": [member], "acts": [{"n": 2}, {}]})
    result, _ = source.rewrite_story_source(
        RawStoryFields(idea="Dinner"), candidate, slot, schema=StoryTreatment,
        receipts=[], pass_id="treatment", preserve_omitted={("cast",): "name", ("acts",): "n"})
    assert result.cast[0].speech_register == expected
    assert result.acts[0].purpose == "departure"
    assert result.acts[1].purpose == ""  # default n=1 is not an explicit identity


@pytest.mark.parametrize("error", [RuntimeError("provider failed"), MemoryError("real OOM"),
                                  KeyboardInterrupt("cancelled")])
def test_real_failures_propagate_once_and_keep_attempt_evidence(error):
    def fail(messages):
        raise error
    slot = Slot(fail)
    journal = []
    with pytest.raises(type(error), match=str(error)):
        _run(slot, receipts=journal)
    assert len(slot.calls) == 1
    assert journal[0]["status"] == "provider_error"
    assert journal[0]["attempts"][0]["error_type"] == type(error).__name__
    assert json.loads(json.dumps(journal)) == journal


def test_measured_no_fit_does_not_waste_generation_or_reject_accepted_text():
    slot = Slot(fit={"supported": True, "capacity_known": True, "fits": False})
    result, receipt = _run(slot)
    assert result is None and not slot.calls
    assert receipt["status"] == "unresolved_capacity"
    assert receipt["output_sha256"] == receipt["input_sha256"]


@pytest.mark.parametrize("fits", [True, False])
def test_unknown_native_capacity_is_not_a_measured_refusal_or_pass(fits):
    slot = Slot(fit={"supported": True, "capacity_known": False, "fits": fits})
    result, receipt = _run(slot)
    assert result is not None and len(slot.calls) == 1
    assert not receipt["qualified"]
    assert not receipt["attempts"][0]["fit"]["capacity_known"]


def test_output_limit_completion_stays_separate_and_total_budget_is_two():
    def fail(messages):
        error = PromptContextOverflowError("ran out of output", phase="output_limit")
        error.raw_completion = '{"ending": "unfinished'
        raise error
    slot = Slot(fail)
    result, receipt = _run(slot)
    assert result is None and len(slot.calls) == 2
    assert all(row["raw_output"] == "" for row in receipt["attempts"])
    assert all(row["raw_completion"].endswith("unfinished") for row in receipt["attempts"])


def _ledger():
    return {"meta": {"my_story": {"source_rewrites": []},
                     "source_meta": {"story_input": {"fields": {"idea": "Mother is alive."}}}},
            "lines": [{"line_id": "l1", "speaker": "Ada", "speaker_role": "character",
                       "text": "  Keep this. Mother died.  Keep that!"},
                      {"line_id": "l2", "speaker": "Tom", "speaker_role": "character",
                       "text": "A different line."}]}


def _edit(**overrides):
    edit = {"line_id": "l1", "source_field": "idea", "source_quote": "Mother is alive",
            "original_quote": "Mother died.", "replacement": "Mother lives."}
    edit.update(overrides)
    return edit


def test_spoken_correction_is_applied_without_changing_surrounding_bytes_ids_or_order():
    data = _ledger()
    before = copy.deepcopy(data)
    slot = Slot({"edits": [_edit()]})
    receipt = source.rewrite_spoken_from_source(data, slot_fn=slot)
    assert len(slot.calls) == 1 and receipt["applied"]
    system = slot.calls[0][0]["content"]
    assert "A partial artifact need not repeat source facts outside its scope." in system
    assert "An act need not repeat every fact" not in system
    assert "Preserve compatible elaboration and unaffected wording." in system
    assert "unrelated byte unchanged. Never change speakers, order or ids." in system
    assert data["lines"][0]["text"] == "  Keep this. Mother lives.  Keep that!"
    assert data["lines"][1] == before["lines"][1]
    assert data["lines"][0]["speaker"] == "Ada"
    assert data["lines"][0]["line_id"] == "l1"
    assert data["lines"][0]["char_count"] == len(data["lines"][0]["text"])


def test_spoken_source_alias_repairs_to_an_applied_missing_action_within_two_calls():
    data = _ledger()
    data['meta']['source_meta']['story_input']['fields'] = {
        'plot': 'Jeffrey briefly mentions his girlfriend as a separate person.'}
    data['lines'][0]['text'] = '  Mom, I loved the carousel.  '
    before = copy.deepcopy(data['lines'])
    correction = _edit(
        source_field='plot', source_quote='Jeffrey briefly mentions his girlfriend',
        original_quote='Mom, I loved the carousel.',
        replacement='Mom, I loved the carousel. My girlfriend would enjoy it too.')
    responses = iter([{'edits': [dict(correction, source_field='text')]},
                      {'edits': [correction]}])
    slot = Slot(lambda _messages: next(responses))
    receipt = source.rewrite_spoken_from_source(data, slot_fn=slot)
    assert len(slot.calls) == 2 and receipt['applied']
    assert data['lines'][0]['text'] == '  Mom, I loved the carousel. My girlfriend would enjoy it too.  '
    assert data['lines'][1] == before[1]
    assert data['lines'][0]['speaker'] == before[0]['speaker']
    assert data['lines'][0]['line_id'] == before[0]['line_id']
    assert receipt['input_sha256'] != receipt['output_sha256']
    assert receipt['attempts'][0]['status'] == 'failed'
    assert receipt['attempts'][1]['status'] == 'usable'
    assert receipt['qualified'] is False  # application is not a semantic certificate


@pytest.mark.parametrize("override", [
    {"source_field": "author"}, {"source_quote": "Invented source"},
    {"line_id": "unknown"}, {"original_quote": "invented original"},
    {"start_char": True, "end_char": 22}, {"start_char": -1, "end_char": 22},
    {"start_char": "13", "end_char": 22}, {"start_char": 0, "end_char": 1},
])
def test_invalid_edits_exhaust_two_calls_and_preserve_original(override):
    data = _ledger()
    before = copy.deepcopy(data["lines"])
    slot = Slot({"edits": [_edit(**override)]})
    receipt = source.rewrite_spoken_from_source(data, slot_fn=slot)
    assert receipt["status"] == "unresolved" and len(slot.calls) == 2
    assert data["lines"] == before and not receipt["applied"]


def test_overlapping_edits_are_not_applied():
    data = _ledger()
    before = copy.deepcopy(data["lines"])
    slot = Slot({"edits": [_edit(), _edit()]})
    receipt = source.rewrite_spoken_from_source(data, slot_fn=slot)
    assert receipt["status"] == "unresolved" and len(slot.calls) == 2
    assert data["lines"] == before


def test_repeated_quote_requires_exact_coordinates():
    candidate = {"lines": [{"line_id": "l1", "text": "Mother died. Mother died."}]}
    raw = {"idea": "Mother is alive."}
    with pytest.raises(ValueError, match="unambiguous"):
        source._apply_spoken_edits(source.SpokenSourceEdits(edits=[_edit()]), candidate, raw)
    result = source._apply_spoken_edits(source.SpokenSourceEdits(
        edits=[_edit(start_char=13, end_char=25)]), candidate, raw)
    assert result["lines"][0]["text"] == "Mother died. Mother lives."


def test_protected_fact_rows_are_not_editable():
    from nodes._otr_ledger_clean import PROTECTED_FACT_COMPONENT_FLAG
    data = _ledger()
    data["lines"][0]["compose_flags"] = [PROTECTED_FACT_COMPONENT_FLAG]
    before = copy.deepcopy(data["lines"])
    slot = Slot({"edits": [_edit()]})
    receipt = source.rewrite_spoken_from_source(data, slot_fn=slot)
    assert receipt["status"] == "unresolved" and data["lines"] == before


def test_no_source_correction_is_one_call_not_a_self_recheck():
    data = _ledger()
    before = copy.deepcopy(data["lines"])
    slot = Slot({"edits": []})
    receipt = source.rewrite_spoken_from_source(data, slot_fn=slot)
    assert receipt["status"] == "unchanged" and len(slot.calls) == 1
    assert data["lines"] == before


def test_spoken_identity_includes_order_and_speaker():
    data = _ledger()
    original = source.candidate_sha256(source.spoken_projection(data))
    data["lines"].reverse()
    assert source.candidate_sha256(source.spoken_projection(data)) != original
    data["lines"].reverse()
    data["lines"][0]["speaker"] = "Tom"
    assert source.candidate_sha256(source.spoken_projection(data)) != original


def _source_tail(tmp_path, monkeypatch):
    from tests.test_scifi_news_pro_tail_context import _make_ctx
    from nodes import _otr_ledger_clean
    from nodes._otr_content_authorship import stamp_receipt
    ctx = _make_ctx(tmp_path, monkeypatch, final_title_override="The Source Repair")
    meta = ctx.led.data["meta"]
    meta.update(source_bank="my_story", my_story={"source_rewrites": []},
                source_meta={"story_input": {"fields": {"idea": "Mother is alive."}}})
    for row in ctx.led.data["lines"]:
        if row["line_id"] == "b003":
            row["text"] = "Keep this. Mother died. Keep that."
    stamp_receipt(ctx.led.data, owner_bank="my_story", accepted_artifacts={"script": "accepted"})
    slot = Slot({"edits": [_edit(line_id="b003")]})

    def clean(data, **kwargs):
        receipt = source.rewrite_spoken_from_source(data, slot_fn=slot)
        data["meta"]["ledger_clean"] = {"source_rewrite": receipt}
        return data["meta"]["ledger_clean"]

    monkeypatch.setattr(_otr_ledger_clean, "run_ledger_clean", clean)
    return ctx, slot


@pytest.mark.parametrize("error", [RuntimeError("cleanup failed"), KeyboardInterrupt("cleanup cancelled")])
def test_tail_persists_source_repair_before_propagating_later_cleanup_failure(tmp_path, monkeypatch, error):
    from pathlib import Path
    from nodes import _otr_ledger_cleanup
    from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter
    ctx, slot = _source_tail(tmp_path, monkeypatch)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(_otr_ledger_cleanup, "run_ledger_cleanup", fail)
    with pytest.raises(type(error), match=str(error)) as caught:
        OTR_LedgerScriptWriter()._run_writer_tail(ctx)
    assert caught.value is error
    saved = json.loads(Path(ctx.led.path).read_text(encoding="utf-8"))
    journal = saved["meta"]["my_story"]["source_rewrites"]
    assert len(slot.calls) == 1 and len(journal) == 1
    assert journal[0]["status"] == "rewritten" and journal[0]["applied"]
    assert any(row.get("text") == "Keep this. Mother lives. Keep that." for row in saved["lines"])


def test_tail_rollback_keeps_attempt_history_and_actual_retained_hash_without_rechecking(tmp_path, monkeypatch):
    from pathlib import Path
    from tests.test_clean_transaction import _LaneProof
    from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter
    from nodes._otr_content_authorship import validate_receipt
    ctx, slot = _source_tail(tmp_path, monkeypatch)

    class FailedReseal(_LaneProof):
        def before_save(self, **kwargs):
            pass

        def after_save(self, **kwargs):
            pass

    proof = FailedReseal({"b003": "Keep this. Mother died. Keep that."}, fail_reseal=True)
    OTR_LedgerScriptWriter()._run_writer_tail(ctx, tail_finalizer=proof)
    saved = json.loads(Path(ctx.led.path).read_text(encoding="utf-8"))
    story = saved["meta"]["my_story"]
    assert len(slot.calls) == 1 and len(story["source_rewrites"]) == 1
    assert story["source_rewrites"][0]["status"] == "rewritten"  # attempted correction
    assert story["source_rewrites"][0]["retained"] is False
    assert story["source_rewrites"][0]["clean_outcome"] == "restored_pre_clean"
    assert story["retained_spoken"]["clean_outcome"] == "restored_pre_clean"
    assert story["retained_spoken"]["sha256"] == source.candidate_sha256(source.spoken_projection(saved))
    assert story["final_spoken_sha256"] == source.candidate_sha256(source.spoken_projection(saved, delivery=True))
    assert any(row.get("text") == "Keep this. Mother died. Keep that." for row in saved["lines"])
    validate_receipt(saved)
