from types import SimpleNamespace

from nodes import OTR_LedgerScriptWriter as writer
from nodes import _otr_story_routing as routing


def test_custom_slot_receipt_comes_only_from_executed_helpers():
    meta = {}
    scheduler = SimpleNamespace(
        transitions=3,
        calls_by_slot={"creative": 2, "technical": 1},
        slot_calls_by_helper={
            "lane:P1": {"creative": 1, "technical": 0},
            "lane:P2": {"creative": 0, "technical": 1},
            "story_brief_reflection": {"creative": 0, "technical": 1},
        },
        slot_transitions_by_phase=[{"phase": "lane:P2"}],
    )
    resolved = {
        "creative_writing_model": "creative/model",
        "technical_model": "technical/model",
    }
    writer._stamp_final_slot_telemetry(
        meta=meta, resolved=resolved, slot_scheduler=scheduler,
        pipeline_id="scifi_news_circuit", title_source="lane",
    )
    assert set(meta["gen_params_by_phase"]) == {
        "lane:P1", "lane:P2", "story_brief_reflection",
    }
    assert not ({"cast_lock", "outline", "dialogue_composer"}
                & set(meta["gen_params_by_phase"]))
    assert meta["slot_calls_by_slot"] == {"creative": 2, "technical": 1}


def test_runnable_custom_pipelines_and_writer_map_are_bijective():
    routing._REGISTRY = None
    registry = routing._ensure_loaded()
    expected = {
        bank.default_story_pipeline
        for bank in registry.banks.values()
        if bank.runnable
    } - writer._LEGACY_INLINE_PIPELINES
    assert set(writer._RUNNER_BY_PIPELINE) == expected
    for pipeline_id in writer._RUNNER_BY_PIPELINE:
        assert registry.pipelines[pipeline_id].executable is True
