from types import SimpleNamespace

from nodes import OTR_LedgerScriptWriter as writer
from nodes import _otr_lane_specs as lanes
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
        pipeline_id="scifi_news_pro_multipass", title_source="lane",
    )
    assert set(meta["gen_params_by_phase"]) == {
        "lane:P1", "lane:P2", "story_brief_reflection",
    }
    assert not ({"cast_lock", "outline", "dialogue_composer"}
                & set(meta["gen_params_by_phase"]))
    assert meta["slot_calls_by_slot"] == {"creative": 2, "technical": 1}


def test_runnable_custom_pipelines_and_lane_table_are_bijective():
    """Every runnable bank's pipeline is EITHER a dispatched lane or inline.

    The authority moved out of the writer into `_otr_lane_specs`; the
    bijection it enforces did not change. A new runnable bank whose
    pipeline lands in neither set fails here rather than at run().
    """
    routing._REGISTRY = None
    registry = routing._ensure_loaded()
    expected = {
        bank.default_story_pipeline
        for bank in registry.banks.values()
        if bank.runnable
    } - lanes.INLINE_PIPELINES
    assert set(lanes.LANE_SPECS) == expected
    for pipeline_id in lanes.LANE_SPECS:
        assert registry.pipelines[pipeline_id].executable is True


def test_writer_no_longer_owns_a_second_lane_table():
    """The writer keeps NO copy, alias or view of the lane authority.

    Two tables keyed by the same pipeline ids is the drift hazard this
    move exists to remove -- a shim left behind would recreate it.
    """
    for dead in (
        "_RUNNER_BY_PIPELINE", "_LEGACY_INLINE_PIPELINES",
        "_resolve_lane_runner", "_run_fable2_lane", "_run_scifi_codex_lane",
    ):
        assert not hasattr(writer, dead), (
            f"{dead} is back in the writer; _otr_lane_specs is the ONE "
            f"lane authority"
        )
