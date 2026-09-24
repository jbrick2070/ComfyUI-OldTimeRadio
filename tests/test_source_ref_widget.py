"""Source Banks v2 `source_ref` surface guardrails.

This chunk only adds the inert, append-only source-reference surface. Bank
fetchers consume it in later chunks; until then blank is byte-stable and
nonblank is just preserved for downstream fail-loud consumers.

Every position in this file is resolved BY NAME through
``tests/fixtures/writer_slots``. Nothing here pins an absolute widget index,
because the writer's layout has shifted under this file three times and each
shift was silent: the saved values around `source_ref` are mostly ``""``, so
an assertion that slid onto its neighbour kept passing and stopped checking.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from nodes.OTR_LedgerScriptWriter import (  # noqa: E402
    OTR_LedgerScriptWriter,
    _resolve_inputs,
)
from tests.fixtures.writer_slots import (  # noqa: E402
    assert_relative_order,
    value,
)

_CANONICAL_WORKFLOW = _REPO / "workflows" / "otr_canonical.json"


def test_source_ref_slot_pinned_after_source_bank():
    """`source_ref` keeps its declared neighbourhood, and the tail is `gate_in`.

    This test used to pin `source_ref` between `google_api_slot_b_model` and
    the six LLM runtime-policy widgets, because that was its neighbourhood
    from the S5 platform-portability pass (2026-07-10) onward. That
    neighbourhood is gone: the 2026-09-14 writer reorder (three independent
    readers converged on one order -- see the docstring on
    `_EXPECTED_INPUT_ORDER` in test_openrouter_slot_widgets_s2.py, the single
    place that order is pinned in full) moved `source_ref` from the tail of
    the model-picker run to the very front, directly after `source_bank` --
    load-bearing now, not cosmetic: it opens the "what are we making" group
    that used to be scattered across the node.

    Before this reorder the position had already moved three times, every
    time because a widget AHEAD of `source_ref` was deleted: target_words
    (2026-08-14), refine_target_grade (2026-08-28) and
    perfect_run_spacesaver (2026-09-13). That is exactly the class of change
    an absolute index cannot survive and a relative-order claim does not care
    about, so this test makes the relative claim -- now anchored on
    `source_ref`'s real neighbours instead of its old ones.
    """
    spec = OTR_LedgerScriptWriter.INPUT_TYPES()
    order = list(spec["required"].keys()) + list(spec["optional"].keys())

    assert_relative_order(order, [
        "source_bank",
        "source_ref",
        "visual_style",
        "episode_title",
        "custom_premise",
        "story_characters",
        "story_plot",
        "story_setting",
        "story_author",
        "act_count",
        "include_act_breaks",
        "num_characters",
        "lemmy_cameo",
        "story_scaffold",
        "creativity",
        "min_p",
        "repetition_penalty",
    ])
    # `gate_in` is the last thing appended (the forceInput validator socket),
    # so it closes the declared vector now -- not `story_author`, which the
    # reorder moved up next to `source_ref` (see the chain above).
    assert order[-1] == "gate_in"
    # A COUNT, not a position -- this one is a literal on purpose. 36 declared
    # inputs carry a 35-wide saved vector because gate_in is a forceInput
    # socket and consumes no widgets_values slot (asserted just below).
    # 36 since 2026-09-24, when the two retired quant widgets came out.
    assert len(order) == 36

    source_ref_type, meta = spec["optional"]["source_ref"]
    assert source_ref_type == "STRING"
    assert meta["default"] == ""

    # gate_in is declared in the order above but is a forceInput socket, so it
    # takes no saved value. That is the whole reason the declared order runs
    # one longer than widgets_values -- verified against the live canonical
    # node in test_patch_widget_by_name_lands_on_source_ref below, and in
    # tests/test_otr_api_companions.py::
    # test_round_trip_canonical_node1_inputs_correct.
    gate_type, gate_meta = spec["optional"]["gate_in"]
    assert gate_type == "STRING"
    assert gate_meta["forceInput"] is True


def test_source_ref_is_real_run_parameter_and_resolved_value():
    params = inspect.signature(OTR_LedgerScriptWriter.run).parameters
    assert "source_ref" in params

    resolved = _resolve_inputs(custom_premise="seed")
    assert resolved["source_ref"] == ""

    resolved2 = _resolve_inputs(
        custom_premise="seed",
        source_ref="https://example.invalid/source.txt",
    )
    assert resolved2["source_ref"] == "https://example.invalid/source.txt"


def test_source_ref_on_both_headless_whitelists():
    from nodes._otr_workflow_apply import CREATIVE_WHITELIST as pkg_wl
    import otr_api

    assert "source_ref" in pkg_wl
    assert "source_ref" in otr_api.CREATIVE_WHITELIST


def test_patch_widget_by_name_lands_on_source_ref():
    import otr_api

    spec = OTR_LedgerScriptWriter.INPUT_TYPES()
    schemas = {
        "OTR_LedgerScriptWriter": {
            "input": {
                "required": spec["required"],
                "optional": spec["optional"],
            },
        },
    }
    workflow = otr_api.load_workflow(str(_CANONICAL_WORKFLOW))
    otr_api.patch_widget_by_name(
        workflow,
        1,
        "source_ref",
        "https://example.invalid/source.txt",
        schemas,
    )
    node1 = next(n for n in workflow["nodes"] if n["id"] == 1)

    # A COUNT, not a position: the writer's saved vector has been 35 wide
    # since 2026-09-24, when the two retired quant widgets came out with the
    # backend they configured. Patching must not change the width -- a patch
    # that grew or shrank the vector would be corrupting every later widget in
    # the graph.
    assert len(node1["widgets_values"]) == 35

    # Neighbour checks. They exist to prove the patch landed on source_ref
    # ALONE and left the widgets on either side of it holding canonical's own
    # saved values, so they track canonical rather than stating a preference.
    # 2026-08-15 (operator): canonical ships the roll sentinels on bank and
    # style so an unattended run varies both.
    assert value(node1, "source_bank") == "roll (any eligible bank)"
    assert value(node1, "visual_style") == "roll (any style)"
    assert (value(node1, "google_api_slot_a_model")
            == "(select Google API model)")
    assert (value(node1, "google_api_slot_b_model")
            == "(select Google API model)")
    assert value(node1, "source_ref") == "https://example.invalid/source.txt"


def test_patch_creative_allows_source_ref():
    import otr_api

    spec = OTR_LedgerScriptWriter.INPUT_TYPES()
    schemas = {
        "OTR_LedgerScriptWriter": {
            "input": {
                "required": spec["required"],
                "optional": spec["optional"],
            },
        },
    }
    workflow = otr_api.load_workflow(str(_CANONICAL_WORKFLOW))
    otr_api.patch_creative(
        workflow,
        1,
        "source_ref",
        "pd://sherlock/case-001",
        schemas,
    )
    node1 = next(n for n in workflow["nodes"] if n["id"] == 1)
    assert value(node1, "source_ref") == "pd://sherlock/case-001"
