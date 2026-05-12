"""tests/test_lfc_c1_standalone_phase_nodes.py

LFC commit 12.11 -- C1 standalone per-phase ComfyUI nodes for
Phase 4 / 5 / 6. Each is an ADDITIONAL entry point next to the
main OTR_LedgerFreezeCascade; defaults OFF so dropping the node
on the canvas is a no-op until opted in.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Class surface
# ---------------------------------------------------------------------------


class TestStandaloneNodeSurface:
    def test_phase_4_class_surface(self):
        from nodes.OTR_LFCPhase4Scene import OTR_LFCPhase4Scene
        assert OTR_LFCPhase4Scene.CATEGORY == "OldTimeRadio/v2"
        assert OTR_LFCPhase4Scene.RETURN_NAMES == (
            "script_json", "phase_4_verdict",
        )
        assert OTR_LFCPhase4Scene.RETURN_TYPES == ("STRING", "STRING")

    def test_phase_5_class_surface(self):
        from nodes.OTR_LFCPhase5Voice import OTR_LFCPhase5Voice
        assert OTR_LFCPhase5Voice.CATEGORY == "OldTimeRadio/v2"
        assert OTR_LFCPhase5Voice.RETURN_NAMES == (
            "script_json", "phase_5_verdict",
        )

    def test_phase_6_class_surface(self):
        from nodes.OTR_LFCPhase6Arc import OTR_LFCPhase6Arc
        assert OTR_LFCPhase6Arc.CATEGORY == "OldTimeRadio/v2"
        assert OTR_LFCPhase6Arc.RETURN_NAMES == (
            "script_json", "phase_6_verdict",
        )

    @pytest.mark.parametrize("module_name,class_name", [
        ("nodes.OTR_LFCPhase4Scene", "OTR_LFCPhase4Scene"),
        ("nodes.OTR_LFCPhase5Voice", "OTR_LFCPhase5Voice"),
        ("nodes.OTR_LFCPhase6Arc",   "OTR_LFCPhase6Arc"),
    ])
    def test_input_types_has_enable_default_off(self, module_name, class_name):
        import importlib
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        schema = cls.INPUT_TYPES()
        assert "enable" in schema["optional"], (
            f"{class_name}: expected `enable` BOOLEAN widget"
        )
        assert schema["optional"]["enable"][0] == "BOOLEAN"
        assert schema["optional"]["enable"][1]["default"] is False, (
            f"{class_name}: enable widget must default OFF"
        )

    @pytest.mark.parametrize("module_name,class_name", [
        ("nodes.OTR_LFCPhase4Scene", "OTR_LFCPhase4Scene"),
        ("nodes.OTR_LFCPhase5Voice", "OTR_LFCPhase5Voice"),
        ("nodes.OTR_LFCPhase6Arc",   "OTR_LFCPhase6Arc"),
    ])
    def test_input_types_has_model_id_default_mistral_nemo(
        self, module_name, class_name,
    ):
        import importlib
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        schema = cls.INPUT_TYPES()
        assert "model_id" in schema["optional"]
        assert schema["optional"]["model_id"][1]["default"] == (
            "mistralai/Mistral-Nemo-Instruct-2407"
        )


# ---------------------------------------------------------------------------
# Default-off no-op contract
# ---------------------------------------------------------------------------


class TestStandaloneNodeDefaultOff:
    def _instantiate(self, module_name, class_name):
        import importlib
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        return cls()

    @pytest.mark.parametrize("module_name,class_name", [
        ("nodes.OTR_LFCPhase4Scene", "OTR_LFCPhase4Scene"),
        ("nodes.OTR_LFCPhase5Voice", "OTR_LFCPhase5Voice"),
        ("nodes.OTR_LFCPhase6Arc",   "OTR_LFCPhase6Arc"),
    ])
    def test_disabled_is_no_op_passthrough(self, module_name, class_name):
        # When enable=False the node MUST return without touching
        # the ledger / loading the LLM. Pass an obviously-shaped
        # script_json; assert it's returned verbatim (modulo the
        # `or "{}"` falsy normalization) and the verdict is "skipped".
        inst = self._instantiate(module_name, class_name)
        result = inst.run(
            script_json='{"schema_version": "l3-2026-05-14"}',
            enable=False,
            model_id="some-model",
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        out_json, verdict = result
        assert out_json == '{"schema_version": "l3-2026-05-14"}'
        assert verdict == "skipped"

    @pytest.mark.parametrize("module_name,class_name", [
        ("nodes.OTR_LFCPhase4Scene", "OTR_LFCPhase4Scene"),
        ("nodes.OTR_LFCPhase5Voice", "OTR_LFCPhase5Voice"),
        ("nodes.OTR_LFCPhase6Arc",   "OTR_LFCPhase6Arc"),
    ])
    def test_empty_input_normalizes_to_braces(self, module_name, class_name):
        inst = self._instantiate(module_name, class_name)
        result = inst.run(script_json="", enable=False)
        assert result == ("{}", "skipped")
