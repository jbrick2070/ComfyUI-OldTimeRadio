"""Successful per-call identities, truthful credits, and final ledger parity."""
import gc
import json
import weakref
from pathlib import Path

import pytest

from nodes import OTR_LedgerScriptWriter as writer
from nodes import _otr_model_loader as loader
from nodes import _otr_openrouter_backend as orb
from nodes import _otr_writer_tail as tail
from tests.test_openrouter_model_gone import enabled_env, _reset, _row, _ok, _gone


def scheduler():
    return writer._SlotScheduler(creative_id="configured/creative", technical_id="configured/technical",
                                 top_p=.9, min_p=0, repetition_penalty=1)


def invoke(sched, helper="compose_line", slot="creative"):
    with sched.helper_context(helper):
        return sched.for_slot(slot)([{"role": "user", "content": "A scene."}],
                                    temperature=.5, max_new_tokens=16)


@pytest.mark.parametrize("provider,key", [("local", "model_id"), ("gguf", "model_id"),
                                         ("comfy_credits", "slug"), ("google_api", "google_model")])
def test_request_identity_is_primitive_and_episode_local(monkeypatch, provider, key):
    class Model:
        pass
    handles = []
    def request(*args, **kwargs):
        model = Model()
        handles.append(weakref.ref(model))
        return {"provider": provider, key: "actual/request", "model": model}
    monkeypatch.setattr(loader, "request_slot", request)
    monkeypatch.setattr(writer, "_build_truncating_generate_fn",
                        lambda entry, **kw: lambda *a, **k: "Completed scene.")
    first, second = scheduler(), scheduler()
    assert invoke(first) == "Completed scene."
    assert not second.successful_model_calls
    invoke(second, "ledger_cleanup", "technical")
    gc.collect()
    assert all(ref() is None for ref in handles)
    row = first.successful_model_calls[0]
    assert row == {"helper": "compose_line", "slot": "creative", "provider": provider,
                   "configured_model_id": "configured/creative", "requested_model_id": "actual/request",
                   "executed_model_id": "actual/request", "reported_model_id": None,
                   "identity_basis": "request_identity"}
    assert second.successful_model_calls[0]["helper"] == "ledger_cleanup"
    json.dumps(first.successful_model_calls)


def test_failures_and_fit_inspection_are_not_successes(monkeypatch):
    monkeypatch.setattr(loader, "request_slot", lambda *a, **k: {"model_id": "local/model"})
    monkeypatch.setattr(loader, "inspect_native_prompt_fit", lambda *a, **k: {"supported": True})
    def fail(*a, **k):
        raise RuntimeError("provider failed")
    monkeypatch.setattr(writer, "_build_truncating_generate_fn", lambda *a, **k: fail)
    sched = scheduler()
    sched.for_slot("creative")._otr_inspect_fit([], max_new_tokens=16)
    assert sched.calls_by_slot["creative"] == 0
    with pytest.raises(RuntimeError, match="provider failed"):
        invoke(sched)
    assert sched.calls_by_slot["creative"] == 1
    assert sched.successful_model_calls == []


def _reported(model, text="A complete scene."):
    result = _ok(text)
    if model is not None:
        result["json"]["model"] = model
    return result


def test_openrouter_fallback_and_unreported_do_not_leak_across_schedulers(enabled_env, monkeypatch):
    orb.set_slot_bindings(slot_a="vendor/dead-model", slot_b=None)
    entry = orb.OpenRouterBackend().load(orb.SLOT_A_ID, _row())
    monkeypatch.setattr(loader, "request_slot", lambda *a, **k: entry)
    replies = iter([_gone("vendor/dead-model"), _reported("vendor/resolved-a"),
                    _reported("vendor/resolved-b"), _reported(None)])
    monkeypatch.setattr(orb, "_post_chat_completion", lambda **kw: next(replies))
    first, second = scheduler(), scheduler()
    invoke(first)
    invoke(second)
    invoke(first)
    a, missing = first.successful_model_calls
    assert a["requested_model_id"] == "anthropic/claude-opus-4.8"
    assert a["executed_model_id"] == "vendor/resolved-a"
    assert second.successful_model_calls[0]["executed_model_id"] == "vendor/resolved-b"
    assert missing["requested_model_id"] == "vendor/dead-model"
    assert missing["executed_model_id"] is missing["reported_model_id"] is None
    assert missing["identity_basis"] == "unreported"


def test_reused_remote_closure_clears_receipt_on_malformed_and_runaway(enabled_env, monkeypatch):
    from nodes import _otr_decode_guard as guard
    entry = orb.OpenRouterBackend().load(orb.SLOT_A_ID, _row())
    replies = iter([_reported("vendor/first"),
                    {"status_code": 200, "json": {"model": "vendor/malformed", "choices": []}},
                    _reported("vendor/runaway")])
    monkeypatch.setattr(orb, "_post_chat_completion", lambda **kw: next(replies))
    fn = orb.make_openrouter_generate_fn(entry)
    kwargs = {"temperature": .5, "max_new_tokens": 16}
    fn([], **kwargs)
    assert fn._otr_response_model_receipt["executed_model_id"] == "vendor/first"
    with pytest.raises(orb.OpenRouterCallFailedError):
        fn([], **kwargs)
    assert fn._otr_response_model_receipt is None
    def reject(*a, **k):
        raise RuntimeError("runaway")
    monkeypatch.setattr(guard, "assert_no_verbatim_cycle", reject)
    with pytest.raises(RuntimeError, match="runaway"):
        fn([], **kwargs)
    assert fn._otr_response_model_receipt is None


def _call(helper, model, slot="creative"):
    return {"helper": helper, "slot": slot, "provider": "local", "executed_model_id": model}


def test_credit_classification_order_and_preservation():
    sched = scheduler()
    sched.successful_model_calls = [
        _call("compose_line", "model/a"), _call("generate_title", "model/b"),
        _call("lock_cast", "model/a"), _call("compose_announcer_outro", None),
        _call("ledger_cleanup", "model/clean", "technical"),
        _call("dramatic_state", "model/support"), _call("unknown_helper", "model/other"),
        _call("compose_line", "wrong-slot", "technical"),
    ]
    meta = {"source_bank": "original", "credits_source_line": "Story by Jeffrey"}
    tail._stamp_model_call_provenance(meta, sched)
    receipt = meta["model_call_provenance"]
    assert meta["credits_source_line"] == "Story generation models used: model/a, model/b, model identity unreported"
    assert receipt["finishing_models"] == ["model/clean"]
    assert receipt["support_models"] == ["model/support"]
    assert receipt["unclassified_helpers"] == ["unknown_helper", "compose_line"]
    for bank in ("my_story", "public_domain", "shakespeare"):
        kept = {"source_bank": bank, "credits_source_line": "Original author credit 🌠"}
        tail._stamp_model_call_provenance(kept, sched)
        assert kept["credits_source_line"] == "Original author credit 🌠"
    tail._stamp_model_call_provenance(meta, scheduler())
    assert meta["credits_source_line"] == "Story generation models used: none recorded"


def test_dispatched_tail_stamps_before_return(monkeypatch):
    sched = scheduler()
    sched.successful_model_calls = [_call("compose_line", "model/story")]
    monkeypatch.setattr(tail._LANES, "is_dispatched", lambda pipeline: True)
    meta = {"source_bank": "original"}
    tail._stamp_final_slot_telemetry(meta=meta, resolved={}, slot_scheduler=sched,
                                    pipeline_id="custom", title_source="fixed")
    assert meta["model_call_provenance"]["generation_models"] == ["model/story"]


def test_final_credit_matches_wire_disk_and_credits_renderer(tmp_path, monkeypatch):
    from tests.test_scifi_news_pro_tail_context import _make_ctx
    from tests.test_credits_roll_spec import _led
    from nodes import otr_credits_roll as credits
    from nodes import production_ledger
    saved = production_ledger._CURRENT
    try:
        ctx = _make_ctx(tmp_path, monkeypatch)
        ctx.meta["source_bank"] = "original"
        ctx.slot_scheduler.successful_model_calls = [_call("compose_line", "model/story")]
        output = writer.OTR_LedgerScriptWriter()._run_writer_tail(ctx)
        wire = json.loads(output[1])["meta"]
        disk = json.loads(Path(ctx.led.path).read_text(encoding="utf-8"))["meta"]
        for key in ("credits_source_line", "model_call_provenance"):
            assert wire[key] == disk[key] == ctx.meta[key]
        led = _led()
        led["meta"].update({key: wire[key] for key in ("credits_source_line", "model_call_provenance")})
        layout = credits.build_credits_layout(led, w=1920, h=1080, manifest={"clips": []})
        assert ">> SOURCE: " + wire["credits_source_line"] in json.dumps(layout)
    finally:
        production_ledger._CURRENT = saved
