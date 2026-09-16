"""tests/test_visual_style_widget_3c.py

Multi-modal story schema STAGE 3 CHUNK 3C -- the `visual_style` selector
widget on OTR_LedgerScriptWriter (the 2C playbook applied per
STAGE3_SUBPLAN v5 section 4 + the r4 verify-at-build checklist).

Positions are resolved BY NAME through tests/fixtures/writer_slots.py. This
file used to pin them as bare integers, and every widget added to or removed
from the writer silently slid those integers onto a neighbour -- the saved
values around here are mostly "" and a repeated placeholder string, so a
drifted assertion kept passing while checking nothing. See that module's
docstring for the full account.

Pins:
  1. Widget surface: visual_style sits with its neighbours in the declared
     order -- immediately after source_ref and immediately before
     episode_title (the 2026-09-14 writer reorder, per
     tests/test_openrouter_slot_widgets_s2.py::_EXPECTED_INPUT_ORDER, moved
     the Google API slots away to sit after the openrouter/comfy slot
     pickers instead); choices == the roll sentinel followed by
     eligible_style_ids() exactly (registry order, all styles live);
     canvas default is the roll command, same as every shipping JSON.
  2. Registration fail-loud: a broken style registry RAISES out of
     INPUT_TYPES (deliberate convention exception -- no baked-in list).
  3. Gate order: an unknown visual_style raises UnknownVisualStyleError
     with ZERO side effects, and the bank gate fires BEFORE the style gate
     (non-runnable custom bank wins even when both are bad).
  4. _resolve_inputs carries visual_style as the authoritative value.
  5. Headless: on both CREATIVE_WHITELISTs; patch_widget_by_name writes the
     visual_style widget in the canonical graph and leaves its neighbours --
     source_bank before it, the Google API slots and source_ref after it --
     carrying their own saved values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from nodes import _otr_story_routing as routing  # noqa: E402
from nodes import _otr_visual_styles as vs  # noqa: E402
from nodes import OTR_LedgerScriptWriter as W_mod  # noqa: E402
from nodes.OTR_LedgerScriptWriter import (  # noqa: E402
    OTR_LedgerScriptWriter,
    _resolve_inputs,
)
from tests.fixtures.writer_slots import (  # noqa: E402
    assert_relative_order,
    value,
)

_CANONICAL_WORKFLOW = _REPO / "workflows" / "otr_canonical.json"


@pytest.fixture(autouse=True)
def _fresh_registry():
    vs._clear_caches()
    yield
    vs._clear_caches()


class TestWidgetSurface:
    def test_visual_style_neighbour_pin(self):
        """source_ref leads visual_style, which leads episode_title.

        What the four absolute indexes here used to say was a claim about
        this GROUP, not about where the group starts: the declared order used
        to run visual_style, both Google API slots, then source_ref,
        contiguously. The 2026-09-14 writer reorder (see
        tests/test_openrouter_slot_widgets_s2.py::_EXPECTED_INPUT_ORDER)
        moved the Google API slots down to sit after the openrouter/comfy
        slot pickers, and put visual_style between source_ref and
        episode_title instead. The relative-order claim survives an unrelated
        widget being added or removed elsewhere in the node, which an
        absolute index does not.
        """
        spec = OTR_LedgerScriptWriter.INPUT_TYPES()
        order = list(spec["required"].keys()) + list(spec["optional"].keys())
        assert_relative_order(order, [
            "source_ref",
            "visual_style",
            "episode_title",
        ])

    def test_choices_are_the_roll_sentinel_then_the_registry(self):
        """2026-07-31: this dropdown owns the SECOND randomizer's command,
        prepended as choice 0 -- independent of the source_bank roll.

        It is a UI command, not a style row; everything after it is still
        exactly the registry.
        """
        from nodes import _otr_rolls as rolls

        spec = OTR_LedgerScriptWriter.INPUT_TYPES()
        choices, meta = spec["optional"]["visual_style"]
        assert choices[0] == rolls.STYLE_SENTINEL
        assert choices[1:] == list(rolls.eligible_style_ids())
        assert rolls.STYLE_SENTINEL not in rolls.eligible_style_ids()
        assert meta["default"] == rolls.STYLE_SENTINEL
        # ALL styles are live (no execution lane); known non-defaults listed.
        for sid in ("anime", "cartoon", "paper_origami",
                    "archival_documentary", "recur_frac", "shakespeare_stage_realism",
                    "storybook_engraving", "video_art"):
            assert sid in choices


class TestRegistrationFailLoud:
    def test_broken_style_registry_raises_out_of_input_types(
            self, monkeypatch):
        def _boom():
            raise vs.VisualStyleValidationError("test: styles dir unreadable")
        monkeypatch.setattr(W_mod._otr_visual_styles, "list_style_ids", _boom)
        with pytest.raises(vs.VisualStyleError):
            OTR_LedgerScriptWriter.INPUT_TYPES()


class TestGateOrder:
    def test_unknown_style_raises_with_zero_side_effects(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            W_mod, "_apply_story_scaffold_env",
            lambda *_a, **_k: calls.append("scaffold_env") or "auto")
        monkeypatch.setattr(
            W_mod, "_resolve_inputs",
            lambda *_a, **_k: calls.append("resolve_inputs") or {})
        node = OTR_LedgerScriptWriter()
        with pytest.raises(vs.UnknownVisualStyleError) as ei:
            node.run(visual_style="no_such_style")
        assert "no_such_style" in str(ei.value)
        assert calls == [], (
            f"run() performed side effects before the style gate: {calls}")

    def test_bank_gate_fires_before_style_gate(self):
        node = OTR_LedgerScriptWriter()
        with pytest.raises(routing.StoryBankNotRunnableError):
            node.run(source_bank="custom_source_bank",
                     visual_style="no_such_style")

    def test_every_registered_style_passes_the_gate(self, monkeypatch):
        # The gate must accept ALL registered styles (they are live by
        # design); prove by stubbing everything after the gates.
        seen = []
        monkeypatch.setattr(
            W_mod, "_apply_story_scaffold_env",
            lambda *_a, **_k: seen.append("past-the-gates") or (_ for _ in ()).throw(
                RuntimeError("stop-after-gate")))
        node = OTR_LedgerScriptWriter()
        for sid in vs.list_style_ids():
            seen.clear()
            with pytest.raises(RuntimeError, match="stop-after-gate"):
                node.run(visual_style=sid)
            assert seen == ["past-the-gates"]



class TestResolvedSurface:
    def test_resolve_inputs_carries_visual_style(self):
        resolved = _resolve_inputs(custom_premise="test premise")
        assert resolved["visual_style"] == "sci_fi_radio"
        resolved2 = _resolve_inputs(custom_premise="test premise",
                                    visual_style="anime")
        assert resolved2["visual_style"] == "anime"


class TestHeadlessSurface:
    def test_visual_style_on_both_whitelists(self):
        from nodes._otr_workflow_apply import CREATIVE_WHITELIST as pkg_wl
        import otr_api
        assert "visual_style" in pkg_wl
        assert "visual_style" in otr_api.CREATIVE_WHITELIST

    def test_patch_widget_by_name_lands_on_visual_style(self):
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
            workflow, 1, "visual_style", "anime", schemas)
        node1 = next(n for n in workflow["nodes"] if n["id"] == 1)
        # The count is the one number that belongs here: it is a measured
        # total, not a position, so it is allowed to be a literal. schemas
        # comes from the LIVE INPUT_TYPES() above, so the patch is resolved
        # against the real widget vector rather than a remembered one.
        assert len(node1["widgets_values"]) == 36
        assert value(node1, "visual_style") == "anime"
        # The NEIGHBOUR checks: the patch wrote visual_style and nothing else,
        # so each of these still carries canonical's own saved value. The bank
        # roll sentinel is what canonical ships (2026-08-15, operator).
        assert value(node1, "source_bank") == "roll (any eligible bank)"
        assert value(node1, "google_api_slot_a_model") == "(select Google API model)"
        assert value(node1, "google_api_slot_b_model") == "(select Google API model)"
        assert value(node1, "source_ref") == ""
