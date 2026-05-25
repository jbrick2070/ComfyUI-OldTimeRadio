"""Unit tests for OTR_HuMoTierLoader (BUG-LOCAL-265, round-robin Option C).

Covers the parts that need no ComfyUI / torch / model files:
  - the INPUT_TYPES / RETURN contract (so the workflow contract
    validator's introspection of node 72 stays in lockstep with this
    class);
  - the pure per-tier config table (_tier_config);
  - the hard auto-downgrade rule (_resolve_tier);
  - the registration entry in the root __init__.py.

The node module is loaded standalone via importlib -- it imports only
its sibling _otr_vram_levers (which does lazy torch/comfy imports), so
this test has no heavy dependency.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NODES_DIR = REPO_ROOT / "nodes"
TIER_LOADER_PY = NODES_DIR / "_otr_humo_tier_loader.py"
INIT_PY = REPO_ROOT / "__init__.py"


def _load_tier_loader():
    """Load nodes/_otr_humo_tier_loader.py standalone, return the class."""
    if str(NODES_DIR) not in sys.path:
        sys.path.insert(0, str(NODES_DIR))
    spec = importlib.util.spec_from_file_location(
        "_otr_humo_tier_loader", TIER_LOADER_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HuMoTierLoader


@pytest.fixture(scope="module")
def H():
    return _load_tier_loader()


# ---------------------------------------------------------------------------
# INPUT_TYPES / RETURN contract
# ---------------------------------------------------------------------------

class TestNodeContract:
    def test_input_types_required_widget_set(self, H):
        it = H.INPUT_TYPES()
        req = list(it["required"].keys())
        assert req == [
            "tier",
            "humo_1p7b_model",
            "humo_highq_model",
            "humo_gguf_model",
            "clip_model",
            "vae_model",
            "audio_encoder_model",
            "distill_lora",
            "auto_downgrade",
            "vram_safety_threshold_gb",
        ], req

    def test_no_model_id_widget(self, H):
        """Prime Directive 6: no widget literally named 'model_id'."""
        it = H.INPUT_TYPES()
        all_widgets = list(it["required"]) + list(it.get("optional", {}))
        assert "model_id" not in all_widgets

    def test_unload_gate_is_optional_force_input(self, H):
        it = H.INPUT_TYPES()
        assert list(it.get("optional", {}).keys()) == ["unload_gate"]
        spec = it["optional"]["unload_gate"]
        assert spec[0] == "STRING"
        assert spec[1].get("forceInput") is True

    def test_tier_choices(self, H):
        it = H.INPUT_TYPES()
        choices = it["required"]["tier"][0]
        assert choices == [
            "low_vram_default", "high_quality", "experimental_gguf"]
        # The shipped default is the low-VRAM tier.
        assert it["required"]["tier"][1]["default"] == "low_vram_default"

    def test_return_contract(self, H):
        assert H.RETURN_TYPES == (
            "MODEL", "CLIP", "VAE", "AUDIO_ENCODER", "INT", "FLOAT", "STRING")
        assert H.RETURN_NAMES == (
            "model", "clip", "vae", "audio_encoder", "steps", "cfg", "report")
        assert H.FUNCTION == "load"
        assert H.OUTPUT_NODE is False
        # Model-file pickers opt-in marker (not LLM model_id widgets).
        assert getattr(H, "NON_LLM_MODEL_WIDGET_OK", False) is True

    def test_widget_count_matches_return_for_validator(self, H):
        """The workflow JSON node 72 ships a 10-value widgets_values
        vector; the contract validator checks count/order against this
        INPUT_TYPES. Pin the count so a future widget add/remove fails
        loud here before it reaches the JSON."""
        it = H.INPUT_TYPES()
        assert len(it["required"]) == 10


# ---------------------------------------------------------------------------
# Per-tier config table
# ---------------------------------------------------------------------------

class TestTierConfig:
    def test_low_vram_default_is_1p7b_20_steps_no_lora(self, H):
        spec = H._tier_config("low_vram_default")
        assert spec["use_gguf"] is False
        assert spec["use_lora"] is False  # distill LoRA is 14B-only
        assert spec["steps"] == 20
        assert spec["cfg"] == 5.0

    def test_high_quality_is_fp8_6_steps_with_lora(self, H):
        spec = H._tier_config("high_quality")
        assert spec["use_gguf"] is False
        assert spec["use_lora"] is True
        assert spec["steps"] == 6
        assert spec["cfg"] == 1.0

    def test_experimental_gguf_uses_gguf_with_lora(self, H):
        """GGUF is demoted to an experimental tier -- it is a tier, not
        the default, and it carries the GGUF loader path."""
        spec = H._tier_config("experimental_gguf")
        assert spec["use_gguf"] is True
        assert spec["use_lora"] is True
        assert spec["steps"] == 6

    def test_unknown_tier_raises(self, H):
        with pytest.raises(ValueError):
            H._tier_config("turbo")


# ---------------------------------------------------------------------------
# Auto-downgrade rule (_resolve_tier)
# ---------------------------------------------------------------------------

class TestAutoDowngrade:
    def test_low_tier_never_checks_vram(self, H):
        # Even with near-zero free VRAM the low tier loads unchanged.
        resolved, _ = H._resolve_tier("low_vram_default", True, 10.0, 0.1)
        assert resolved == "low_vram_default"

    def test_high_tier_clears_with_enough_vram(self, H):
        resolved, _ = H._resolve_tier("high_quality", True, 10.0, 14.0)
        assert resolved == "high_quality"

    def test_high_tier_clears_exactly_at_threshold(self, H):
        resolved, _ = H._resolve_tier("high_quality", True, 10.0, 10.0)
        assert resolved == "high_quality"

    def test_high_tier_downgrades_when_vram_low(self, H):
        resolved, note = H._resolve_tier("high_quality", True, 10.0, 3.0)
        assert resolved == "low_vram_default"
        assert "AUTO-DOWNGRADE" in note

    def test_experimental_gguf_downgrades_too(self, H):
        resolved, _ = H._resolve_tier("experimental_gguf", True, 10.0, 3.0)
        assert resolved == "low_vram_default"

    def test_low_vram_without_autodowngrade_raises(self, H):
        """auto_downgrade OFF + insufficient VRAM = hard stop, never a
        silent thrash."""
        with pytest.raises(RuntimeError):
            H._resolve_tier("high_quality", False, 10.0, 3.0)

    def test_unknown_vram_loads_as_requested(self, H):
        """No CUDA / headless: probe returns NaN -> load the requested
        tier rather than guess."""
        resolved, note = H._resolve_tier(
            "high_quality", True, 10.0, float("nan"))
        assert resolved == "high_quality"
        assert "probe unavailable" in note.lower()

    def test_unknown_vram_none_loads_as_requested(self, H):
        resolved, _ = H._resolve_tier("high_quality", True, 10.0, None)
        assert resolved == "high_quality"


# ---------------------------------------------------------------------------
# Registration in the root __init__.py
# ---------------------------------------------------------------------------

class TestRegistration:
    def _node_modules(self) -> dict:
        """AST-parse the _NODE_MODULES dict literal out of __init__.py
        without importing it (importing pulls in all ComfyUI deps)."""
        tree = ast.parse(INIT_PY.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "_NODE_MODULES":
                        return ast.literal_eval(node.value)
        raise AssertionError("_NODE_MODULES not found in __init__.py")

    def test_humo_tier_loader_registered(self):
        mods = self._node_modules()
        assert "OTR_HuMoTierLoader" in mods, (
            "OTR_HuMoTierLoader must be registered in __init__.py "
            "_NODE_MODULES or the workflow contract validator rejects "
            "node 72 as an unknown OTR type."
        )
        module_path, class_name, _display = mods["OTR_HuMoTierLoader"]
        assert module_path == ".nodes._otr_humo_tier_loader"
        assert class_name == "HuMoTierLoader"

    def test_batch_humo_render_still_registered(self):
        # The tier loader feeds OTR_BatchHumoRender; the renderer must
        # stay registered for the rewired workflow to load.
        mods = self._node_modules()
        assert "OTR_BatchHumoRender" in mods


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
