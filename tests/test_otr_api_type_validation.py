"""tests/test_otr_api_type_validation.py -- BUG-LOCAL-002 follow-up.

Locks the type/choice validation contract for `otr_api.patch_widget_by_name`:

  * COMBO values must be in the declared choice list.
  * INT values must be int (NOT bool, even though bool is a subclass of int).
  * FLOAT values may be int or float, NOT bool.
  * BOOLEAN values must be bool.
  * STRING values must be str.
  * `None` is accepted as "use the node's default".
  * Unknown spec types are skipped (best-effort, never block).

Round-robin follow-up 2026-05-02 (gpt-5.5 + nemotron-49b both recommended
adding type validation; reviewed independently and merged).
"""

from __future__ import annotations

import os
import sys

import pytest

_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_PACK_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from otr_api import (  # noqa: E402
    _validate_widget_value,
    _spec_for,
    patch_widget_by_name,
)


# ---------------------------------------------------------------------------
# Synthetic schemas used across tests. Mirrors the shape ComfyUI's
# /object_info returns for OTR_LLMScriptWriter (subset).
# ---------------------------------------------------------------------------
SYNTH_SCHEMAS = {
    "OTR_LLMScriptWriter": {
        "input": {
            "required": {
                "episode_title": ("STRING", {"default": "", "multiline": False}),
                "target_words": ("INT", {"default": 350, "min": 30, "max": 3500}),
                "num_characters": ("INT", {"default": 4, "min": 2, "max": 9}),
                "model_id": (["mistralai/Mistral-Nemo-Instruct-2407",
                              "google/gemma-4-E2B-it"], {}),
                "include_act_breaks": ("BOOLEAN", {"default": True}),
                "self_critique": ("BOOLEAN", {"default": True}),
                "target_length": (["30 words (smoke, 1 act)",
                                   "tiny (smoke, 1 act)",
                                   "short (3 acts)",
                                   "medium (5 acts)"], {}),
                "creativity": (["safe & tight", "balanced",
                                "wild & rough", "maximum chaos"], {}),
                "optimization_profile": (["Pro (Ultra Quality)",
                                          "Standard",
                                          "Obsidian (UNSTABLE/4GB)"], {}),
                "weight_multiplier": ("FLOAT", {"default": 1.0,
                                                 "min": 0.0, "max": 2.0}),
            },
            "optional": {
                "custom_premise": ("STRING", {"default": "", "multiline": True}),
            },
        }
    }
}


# ---------------------------------------------------------------------------
# _spec_for
# ---------------------------------------------------------------------------
class TestSpecFor:
    def test_required_widget_returns_spec(self):
        spec = _spec_for("OTR_LLMScriptWriter", "target_words", SYNTH_SCHEMAS)
        assert spec[0] == "INT"

    def test_optional_widget_returns_spec(self):
        spec = _spec_for("OTR_LLMScriptWriter", "custom_premise", SYNTH_SCHEMAS)
        assert spec[0] == "STRING"

    def test_unknown_widget_raises(self):
        with pytest.raises(KeyError):
            _spec_for("OTR_LLMScriptWriter", "no_such_widget", SYNTH_SCHEMAS)

    def test_unknown_node_type_raises(self):
        with pytest.raises(KeyError):
            _spec_for("UnknownNode", "anything", SYNTH_SCHEMAS)


# ---------------------------------------------------------------------------
# _validate_widget_value -- happy paths
# ---------------------------------------------------------------------------
class TestValidationHappyPaths:
    def test_int_accepts_int(self):
        spec = _spec_for("OTR_LLMScriptWriter", "target_words", SYNTH_SCHEMAS)
        _validate_widget_value("OTR_LLMScriptWriter", "target_words", spec, 30)
        _validate_widget_value("OTR_LLMScriptWriter", "target_words", spec, 350)
        _validate_widget_value("OTR_LLMScriptWriter", "target_words", spec, 0)

    def test_string_accepts_str(self):
        spec = _spec_for("OTR_LLMScriptWriter", "episode_title", SYNTH_SCHEMAS)
        _validate_widget_value("OTR_LLMScriptWriter", "episode_title", spec, "")
        _validate_widget_value("OTR_LLMScriptWriter", "episode_title", spec, "Hello")

    def test_bool_accepts_bool(self):
        spec = _spec_for("OTR_LLMScriptWriter", "self_critique", SYNTH_SCHEMAS)
        _validate_widget_value("OTR_LLMScriptWriter", "self_critique", spec, True)
        _validate_widget_value("OTR_LLMScriptWriter", "self_critique", spec, False)

    def test_float_accepts_int_or_float(self):
        spec = _spec_for(
            "OTR_LLMScriptWriter", "weight_multiplier", SYNTH_SCHEMAS
        )
        _validate_widget_value(
            "OTR_LLMScriptWriter", "weight_multiplier", spec, 1.0
        )
        _validate_widget_value(
            "OTR_LLMScriptWriter", "weight_multiplier", spec, 1
        )

    def test_combo_accepts_declared_choice(self):
        spec = _spec_for("OTR_LLMScriptWriter", "creativity", SYNTH_SCHEMAS)
        _validate_widget_value(
            "OTR_LLMScriptWriter", "creativity", spec, "balanced"
        )
        _validate_widget_value(
            "OTR_LLMScriptWriter", "creativity", spec, "maximum chaos"
        )

    def test_none_is_permissive(self):
        for widget in ("target_words", "creativity", "self_critique",
                       "episode_title", "weight_multiplier"):
            spec = _spec_for("OTR_LLMScriptWriter", widget, SYNTH_SCHEMAS)
            _validate_widget_value("OTR_LLMScriptWriter", widget, spec, None)


# ---------------------------------------------------------------------------
# _validate_widget_value -- rejection paths (the actual bug class this catches)
# ---------------------------------------------------------------------------
class TestValidationRejections:
    def test_int_rejects_bool(self):
        spec = _spec_for("OTR_LLMScriptWriter", "target_words", SYNTH_SCHEMAS)
        with pytest.raises(ValueError, match="INT"):
            _validate_widget_value(
                "OTR_LLMScriptWriter", "target_words", spec, True
            )

    def test_int_rejects_str(self):
        spec = _spec_for("OTR_LLMScriptWriter", "target_words", SYNTH_SCHEMAS)
        with pytest.raises(ValueError, match="INT"):
            _validate_widget_value(
                "OTR_LLMScriptWriter", "target_words", spec, "30"
            )

    def test_int_rejects_float(self):
        spec = _spec_for("OTR_LLMScriptWriter", "target_words", SYNTH_SCHEMAS)
        with pytest.raises(ValueError, match="INT"):
            _validate_widget_value(
                "OTR_LLMScriptWriter", "target_words", spec, 30.0
            )

    def test_string_rejects_bool(self):
        spec = _spec_for("OTR_LLMScriptWriter", "episode_title", SYNTH_SCHEMAS)
        with pytest.raises(ValueError, match="STRING"):
            _validate_widget_value(
                "OTR_LLMScriptWriter", "episode_title", spec, True
            )

    def test_string_rejects_int(self):
        spec = _spec_for("OTR_LLMScriptWriter", "episode_title", SYNTH_SCHEMAS)
        with pytest.raises(ValueError, match="STRING"):
            _validate_widget_value(
                "OTR_LLMScriptWriter", "episode_title", spec, 5
            )

    def test_bool_rejects_int(self):
        spec = _spec_for("OTR_LLMScriptWriter", "self_critique", SYNTH_SCHEMAS)
        with pytest.raises(ValueError, match="BOOL"):
            _validate_widget_value(
                "OTR_LLMScriptWriter", "self_critique", spec, 1
            )

    def test_bool_rejects_str(self):
        spec = _spec_for("OTR_LLMScriptWriter", "self_critique", SYNTH_SCHEMAS)
        with pytest.raises(ValueError, match="BOOL"):
            _validate_widget_value(
                "OTR_LLMScriptWriter", "self_critique", spec, "True"
            )

    def test_float_rejects_bool(self):
        spec = _spec_for(
            "OTR_LLMScriptWriter", "weight_multiplier", SYNTH_SCHEMAS
        )
        with pytest.raises(ValueError, match="FLOAT"):
            _validate_widget_value(
                "OTR_LLMScriptWriter", "weight_multiplier", spec, True
            )

    def test_float_rejects_str(self):
        spec = _spec_for(
            "OTR_LLMScriptWriter", "weight_multiplier", SYNTH_SCHEMAS
        )
        with pytest.raises(ValueError, match="FLOAT"):
            _validate_widget_value(
                "OTR_LLMScriptWriter", "weight_multiplier", spec, "1.0"
            )

    def test_combo_rejects_undeclared_choice(self):
        spec = _spec_for("OTR_LLMScriptWriter", "creativity", SYNTH_SCHEMAS)
        with pytest.raises(ValueError, match="COMBO"):
            _validate_widget_value(
                "OTR_LLMScriptWriter", "creativity", spec, "absolute chaos"
            )

    def test_combo_rejects_typo(self):
        spec = _spec_for("OTR_LLMScriptWriter", "creativity", SYNTH_SCHEMAS)
        with pytest.raises(ValueError):
            _validate_widget_value(
                "OTR_LLMScriptWriter", "creativity", spec, "Balanced"  # capital B
            )

    def test_combo_rejects_int_index(self):
        spec = _spec_for("OTR_LLMScriptWriter", "creativity", SYNTH_SCHEMAS)
        with pytest.raises(ValueError):
            _validate_widget_value(
                "OTR_LLMScriptWriter", "creativity", spec, 1
            )


# ---------------------------------------------------------------------------
# patch_widget_by_name -- end-to-end happy + rejection paths
# ---------------------------------------------------------------------------
def _minimal_workflow():
    """Tiny one-node workflow so patch_widget_by_name has somewhere to write."""
    return {
        "nodes": [
            {
                "id": 1,
                "type": "OTR_LLMScriptWriter",
                "widgets_values": [
                    "",       # episode_title
                    350,      # target_words
                    4,        # num_characters
                    "mistralai/Mistral-Nemo-Instruct-2407",  # model_id
                    True,     # include_act_breaks
                    True,     # self_critique
                    "medium (5 acts)",  # target_length
                    "balanced",          # creativity
                    "Standard",          # optimization_profile
                    1.0,                  # weight_multiplier
                    "",                   # custom_premise (optional)
                ],
                "inputs": [],
            }
        ],
        "links": [],
    }


class TestPatchWidgetByName:
    def test_happy_path_combo(self):
        wf = _minimal_workflow()
        patch_widget_by_name(
            wf, 1, "creativity", "maximum chaos", SYNTH_SCHEMAS
        )
        assert wf["nodes"][0]["widgets_values"][7] == "maximum chaos"

    def test_happy_path_int(self):
        wf = _minimal_workflow()
        patch_widget_by_name(wf, 1, "target_words", 30, SYNTH_SCHEMAS)
        assert wf["nodes"][0]["widgets_values"][1] == 30

    def test_happy_path_string(self):
        wf = _minimal_workflow()
        patch_widget_by_name(
            wf, 1, "episode_title", "Test Episode", SYNTH_SCHEMAS
        )
        assert wf["nodes"][0]["widgets_values"][0] == "Test Episode"

    def test_rejects_wrong_type_combo(self):
        wf = _minimal_workflow()
        with pytest.raises(ValueError, match="COMBO"):
            patch_widget_by_name(
                wf, 1, "creativity", "absolute chaos", SYNTH_SCHEMAS
            )

    def test_rejects_bool_into_int_widget(self):
        wf = _minimal_workflow()
        with pytest.raises(ValueError, match="INT"):
            patch_widget_by_name(wf, 1, "target_words", True, SYNTH_SCHEMAS)

    def test_rejects_int_into_string_widget(self):
        wf = _minimal_workflow()
        with pytest.raises(ValueError, match="STRING"):
            patch_widget_by_name(wf, 1, "episode_title", 42, SYNTH_SCHEMAS)

    def test_none_skips_patch_does_not_write_null(self):
        """Round-robin round 2 blocker (Gemini + NVIDIA): patch_widget_by_name
        must NOT write `None` into the widgets_values slot when value is None.
        Many ComfyUI core nodes parse `null` as an error rather than a
        default-fallback, which would crash the workflow at queue-time.
        `None` is documented as "use node default" -- leave the slot alone.
        """
        wf = _minimal_workflow()
        original = list(wf["nodes"][0]["widgets_values"])  # snapshot
        patch_widget_by_name(wf, 1, "episode_title", None, SYNTH_SCHEMAS)
        # Episode title slot must still hold the original value, NOT None.
        assert wf["nodes"][0]["widgets_values"] == original
        assert wf["nodes"][0]["widgets_values"][0] != None  # noqa: E711

    def test_none_skips_combo(self):
        """Same None-skip semantics on a COMBO widget (the original
        on-disk value `"balanced"` for `creativity` must survive).
        """
        wf = _minimal_workflow()
        original_creativity = wf["nodes"][0]["widgets_values"][7]
        patch_widget_by_name(wf, 1, "creativity", None, SYNTH_SCHEMAS)
        assert wf["nodes"][0]["widgets_values"][7] == original_creativity
        assert wf["nodes"][0]["widgets_values"][7] != None  # noqa: E711


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
