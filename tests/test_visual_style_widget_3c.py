"""tests/test_visual_style_widget_3c.py

Multi-modal story schema STAGE 3 CHUNK 3C -- the `visual_style` selector
widget on OTR_LedgerScriptWriter (workflow slot 26; the 2C playbook applied
per STAGE3_SUBPLAN v5 section 4 + the r4 verify-at-build checklist).

Pins:
  1. Widget surface: visual_style stays pinned at slot 24; choices ==
     list_style_ids() exactly (registry order, all styles live); default
     sci_fi_radio.
  2. Registration fail-loud: a broken style registry RAISES out of
     INPUT_TYPES (deliberate convention exception -- no baked-in list).
  3. Gate order: an unknown visual_style raises UnknownVisualStyleError
     with ZERO side effects, and the bank gate fires BEFORE the style gate
     (non-runnable custom bank wins even when both are bad).
  4. Refine capture carries visual_style (the 2C signature-filtered fix).
  5. _resolve_inputs carries visual_style as the authoritative value.
  6. Headless: on both CREATIVE_WHITELISTs; patch_widget_by_name lands
     slot 24 (shifted -2 by the 2026-07-05 style-engine consolidation,
     which deleted the style / style_custom widgets). Later widgets append
     after it.
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

_CANONICAL_WORKFLOW = _REPO / "workflows" / "otr_canonical.json"


@pytest.fixture(autouse=True)
def _fresh_registry():
    vs._clear_caches()
    yield
    vs._clear_caches()


class TestWidgetSurface:
    def test_visual_style_positional_pin(self):
        spec = OTR_LedgerScriptWriter.INPUT_TYPES()
        order = list(spec["required"].keys()) + list(spec["optional"].keys())
        assert order[24] == "visual_style"
        assert order[25] == "google_api_slot_a_model"
        assert order[26] == "google_api_slot_b_model"
        assert order[27] == "source_ref"

    def test_choices_are_exactly_the_registry(self):
        spec = OTR_LedgerScriptWriter.INPUT_TYPES()
        choices, meta = spec["optional"]["visual_style"]
        assert choices == list(vs.list_style_ids())
        assert meta["default"] == "sci_fi_radio"
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


class TestRefineCarry:
    def test_refine_core_carries_visual_style(self, monkeypatch):
        from nodes import _otr_story_select as sel

        class _Rcfg:
            target_grade = "B"
            bar = 0
            effective_passes = 2

        monkeypatch.setattr(
            sel, "resolve_refine_passes", lambda *_a, **_k: _Rcfg())
        captured = {}

        def _fake_loop(self, _rcfg, _core):
            captured.update(_core)
            return ("", "", "", 0, "")

        monkeypatch.setattr(OTR_LedgerScriptWriter, "_refine_loop", _fake_loop)
        node = OTR_LedgerScriptWriter()
        out = node.run(visual_style="anime", refine_target_grade="B")
        assert out == ("", "", "", 0, "")
        assert captured["visual_style"] == "anime"
        assert "os" not in captured and "_scaffold" not in captured


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

    def test_patch_widget_by_name_lands_slot_24(self):
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
        # S5 platform-portability (2026-07-10): OLD pin 28 -> NEW pin 34
        # (llm_device..gguf_quant appended at 28-33; visual_style stays 24).
        # schemas comes from the LIVE INPUT_TYPES() above, so this already
        # reflects the new vector -- only the pin needed updating.
        assert len(node1["widgets_values"]) == 34
        assert node1["widgets_values"][24] == "anime"
        assert node1["widgets_values"][23] == "science_news"
        assert node1["widgets_values"][25] == "(select Google API model)"
        assert node1["widgets_values"][26] == "(select Google API model)"
        assert node1["widgets_values"][27] == ""
