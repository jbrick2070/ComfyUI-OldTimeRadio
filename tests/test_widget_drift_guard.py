"""
Regression tests for BUG-LOCAL-027 (widget-drift in _workflow_to_api_prompt)
and BUG-LOCAL-026 (scene regex matches non-numeric tokens).

BUG-LOCAL-027: The soak operator's _workflow_to_api_prompt mapper walked
widgets_values positionally against INPUT_TYPES order, without filtering
socket-only params (types like PROJECT_STATE that have no widget). Any
socket-only param interleaved in the optional block shifted every
subsequent widget up by one slot. The canonical symptom was a literal
string ("Standard", "Pro (Ultra Quality)") landing in the project_state
key, while optimization_profile disappeared from the payload.

These tests lock down:
  - project_state is NEVER emitted as a string; it is either absent or a
    link [src_id, slot].
  - optimization_profile ALWAYS survives in the emitted payload with its
    correct string value.
  - Socket-only inputs never consume widget slots.
  - The scene regex only matches numeric scene tokens and never captures
    the word "FINAL" as a scene number.

Run:  python -m pytest tests/test_widget_drift_guard.py -v
"""
import os
import re
import sys

import pytest

PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACK_ROOT not in sys.path:
    sys.path.insert(0, PACK_ROOT)


# ---------------------------------------------------------------------------
# Shared schema fixture: mirrors the real Gemma4ScriptWriter input layout
# with a socket-only PROJECT_STATE interleaved between widget-backed params.
# ---------------------------------------------------------------------------
def _script_writer_schema():
    return {
        "OTR_Gemma4ScriptWriter": {
            "input": {
                "required": {
                    "episode_title": ("STRING", {"default": ""}),
                    "genre_flavor": (
                        ["hard_sci_fi", "space_opera", "dystopian"],
                        {"default": "hard_sci_fi"},
                    ),
                    "target_words": ("INT", {"default": 700}),
                    "num_characters": ("INT", {"default": 4}),
                },
                "optional": {
                    "model_id": (
                        ["google/gemma-2-2b-it", "mistralai/Mistral-Nemo-Instruct-2407"],
                        {"default": "mistralai/Mistral-Nemo-Instruct-2407"},
                    ),
                    "custom_premise": ("STRING", {"multiline": True, "default": ""}),
                    "include_act_breaks": ("BOOLEAN", {"default": True}),
                    "self_critique": ("BOOLEAN", {"default": True}),
                    "open_close": ("BOOLEAN", {"default": True}),
                    "target_length": (
                        ["short (3 acts)", "medium (5 acts)", "long (7-8 acts)"],
                        {"default": "medium (5 acts)"},
                    ),
                    "style_variant": (
                        ["tense claustrophobic", "space opera epic"],
                        {"default": "tense claustrophobic"},
                    ),
                    "creativity": (
                        ["safe & tight", "balanced", "wild & rough"],
                        {"default": "balanced"},
                    ),
                    "arc_enhancer": ("BOOLEAN", {"default": True}),
                    "optimization_profile": (
                        ["Pro (Ultra Quality)", "Standard", "Obsidian (UNSTABLE/4GB)"],
                        {"default": "Standard"},
                    ),
                    # Socket-only, intentionally at the tail per BUG-LOCAL-027.
                    "project_state": ("PROJECT_STATE", {}),
                },
            }
        }
    }


def _legacy_drift_schema():
    """Schema with project_state INTERLEAVED before optimization_profile.

    Exercises the worst-case drift scenario the reviewer observed on the
    wire. Even with this declaration order, the mapper must emit clean
    inputs because socket-only params cannot consume widget slots.
    """
    schema = _script_writer_schema()
    opt = schema["OTR_Gemma4ScriptWriter"]["input"]["optional"]
    reordered = {}
    for k, v in opt.items():
        if k == "optimization_profile":
            # Insert project_state immediately before optimization_profile
            reordered["project_state"] = ("PROJECT_STATE", {})
            reordered[k] = v
        elif k == "project_state":
            continue  # will be placed above
        else:
            reordered[k] = v
    schema["OTR_Gemma4ScriptWriter"]["input"]["optional"] = reordered
    return schema


def _widgets_values():
    """Widget values in correct widget-display order (no slot for project_state)."""
    return [
        "The Last Frequency",     # 0  episode_title
        "hard_sci_fi",            # 1  genre_flavor
        700,                      # 2  target_words
        4,                        # 3  num_characters
        "mistralai/Mistral-Nemo-Instruct-2407",  # 4  model_id
        "",                       # 5  custom_premise
        True,                     # 6  include_act_breaks
        True,                     # 7  self_critique
        True,                     # 8  open_close
        "medium (5 acts)",        # 9  target_length
        "tense claustrophobic",   # 10 style_variant
        "balanced",               # 11 creativity
        True,                     # 12 arc_enhancer
        "Standard",               # 13 optimization_profile
    ]


def _workflow(schema):
    return {
        "links": [],
        "nodes": [
            {
                "id": 1,
                "type": "OTR_Gemma4ScriptWriter",
                "widgets_values": _widgets_values(),
                "inputs": [
                    # project_state declared as a socket input, not linked here.
                    {"name": "project_state", "type": "PROJECT_STATE", "link": None},
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Import the mapper from soak_operator without executing the module's
# runtime init (no network, no ntfy, no mutex). The module uses guard
# clauses and module-level constants that are safe to import.
# ---------------------------------------------------------------------------
def _import_mapper():
    import importlib
    scripts_path = os.path.join(PACK_ROOT, "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    mod = importlib.import_module("soak_operator")
    return mod


class TestMapperSocketOnlyFilter:
    def test_project_state_never_appears_as_string(self):
        mod = _import_mapper()
        schema = _script_writer_schema()
        wf = _workflow(schema)
        prompt = mod._workflow_to_api_prompt(wf, schema)

        inputs = prompt["1"]["inputs"]
        assert "project_state" not in inputs or isinstance(
            inputs["project_state"], list
        ), (
            f"project_state must be absent or a link, got "
            f"{inputs.get('project_state')!r} (type {type(inputs.get('project_state')).__name__})"
        )

    def test_optimization_profile_survives(self):
        mod = _import_mapper()
        schema = _script_writer_schema()
        wf = _workflow(schema)
        prompt = mod._workflow_to_api_prompt(wf, schema)

        inputs = prompt["1"]["inputs"]
        assert inputs.get("optimization_profile") == "Standard", (
            f"optimization_profile must be present and correct, got "
            f"{inputs.get('optimization_profile')!r}"
        )

    def test_drift_resistant_with_interleaved_schema(self):
        """Even if a future refactor puts project_state BEFORE
        optimization_profile in INPUT_TYPES, the mapper must still emit
        clean inputs. Socket-only params cannot consume widget slots."""
        mod = _import_mapper()
        schema = _legacy_drift_schema()
        wf = _workflow(schema)
        prompt = mod._workflow_to_api_prompt(wf, schema)

        inputs = prompt["1"]["inputs"]
        assert not isinstance(inputs.get("project_state"), str), (
            "Socket-only project_state must NEVER be populated as a string"
        )
        assert inputs.get("optimization_profile") == "Standard"
        assert inputs.get("creativity") == "balanced"
        assert inputs.get("style_variant") == "tense claustrophobic"
        assert inputs.get("target_length") == "medium (5 acts)"

    def test_all_widget_values_land_in_correct_slots(self):
        mod = _import_mapper()
        schema = _script_writer_schema()
        wf = _workflow(schema)
        prompt = mod._workflow_to_api_prompt(wf, schema)

        inputs = prompt["1"]["inputs"]
        assert inputs["episode_title"] == "The Last Frequency"
        assert inputs["genre_flavor"] == "hard_sci_fi"
        assert inputs["target_words"] == 700
        assert inputs["num_characters"] == 4
        assert inputs["creativity"] == "balanced"
        assert inputs["arc_enhancer"] is True


class TestLinkedConvertedWidgetSlots:
    """BUG-LOCAL-028: linked converted widgets must NOT consume widgets_values
    slots. ComfyUI's workflow JSON stores widgets_values only for inputs still
    shown as widgets in the UI. When a widget is converted to a socket and then
    linked, its slot is stripped from widgets_values entirely. The mapper must
    not "consume and skip" a slot for these, or every downstream param shifts.

    Canonical failure: Node 15 (OTR_BatchAudioGenGenerator) crashes with
    HTTP 400 value_not_in_list because model_id receives 3.0 instead of
    'facebook/audiogen-medium'.
    """

    def _node15_schema(self):
        return {
            "OTR_BatchAudioGenGenerator": {
                "input": {
                    "required": {
                        "script_json": ("STRING", {"multiline": True, "default": "[]"}),
                        "production_plan_json": ("STRING", {"multiline": True, "default": "{}"}),
                    },
                    "optional": {
                        "episode_seed": ("STRING", {"default": ""}),
                        "model_id": (
                            ["facebook/audiogen-medium", "facebook/audiogen-small"],
                            {"default": "facebook/audiogen-medium"},
                        ),
                        "guidance_scale": ("FLOAT", {"default": 3.0}),
                        "default_duration": ("FLOAT", {"default": 3.0}),
                    },
                }
            }
        }

    def _node15_workflow(self):
        """Mirrors the on-disk shape in otr_scifi_16gb_full.json where
        script_json and production_plan_json are linked AND carry a
        'widget' metadata block from a prior widget-to-socket conversion."""
        return {
            "links": [
                [24, 10, 0, 15, 0, "STRING"],
                [26, 11, 0, 15, 1, "STRING"],
            ],
            "nodes": [
                {
                    "id": 15,
                    "type": "OTR_BatchAudioGenGenerator",
                    # Only 4 slots: one per widget-backed param still shown
                    # as a widget (episode_seed, model_id, guidance_scale,
                    # default_duration). NOT 6. The linked script_json /
                    # production_plan_json have no slot here.
                    "widgets_values": ["", "facebook/audiogen-medium", 3.0, 3.0],
                    "inputs": [
                        {
                            "name": "script_json",
                            "type": "STRING",
                            "link": 24,
                            "widget": {"name": "script_json"},
                        },
                        {
                            "name": "production_plan_json",
                            "type": "STRING",
                            "link": 26,
                            "widget": {"name": "production_plan_json"},
                        },
                    ],
                }
            ],
        }

    def test_model_id_is_not_float_after_mapping(self):
        mod = _import_mapper()
        wf = self._node15_workflow()
        schema = self._node15_schema()
        prompt = mod._workflow_to_api_prompt(wf, schema)
        inputs = prompt["15"]["inputs"]

        assert inputs.get("model_id") == "facebook/audiogen-medium", (
            f"model_id must land in its own slot, got {inputs.get('model_id')!r}"
        )

    def test_episode_seed_is_empty_string(self):
        mod = _import_mapper()
        wf = self._node15_workflow()
        schema = self._node15_schema()
        prompt = mod._workflow_to_api_prompt(wf, schema)
        inputs = prompt["15"]["inputs"]

        assert inputs.get("episode_seed") == "", (
            f"episode_seed must be empty string, got {inputs.get('episode_seed')!r}"
        )

    def test_guidance_and_duration_land_correctly(self):
        mod = _import_mapper()
        wf = self._node15_workflow()
        schema = self._node15_schema()
        prompt = mod._workflow_to_api_prompt(wf, schema)
        inputs = prompt["15"]["inputs"]

        assert inputs.get("guidance_scale") == 3.0
        assert inputs.get("default_duration") == 3.0

    def test_links_preserved(self):
        mod = _import_mapper()
        wf = self._node15_workflow()
        schema = self._node15_schema()
        prompt = mod._workflow_to_api_prompt(wf, schema)
        inputs = prompt["15"]["inputs"]

        assert inputs.get("script_json") == ["10", 0]
        assert inputs.get("production_plan_json") == ["11", 0]


class TestIsWidgetBacked:
    def test_string_is_widget_backed(self):
        mod = _import_mapper()
        assert mod._is_widget_backed(("STRING", {"default": ""}))

    def test_int_is_widget_backed(self):
        mod = _import_mapper()
        assert mod._is_widget_backed(("INT", {"default": 0}))

    def test_float_is_widget_backed(self):
        mod = _import_mapper()
        assert mod._is_widget_backed(("FLOAT", {"default": 0.0}))

    def test_boolean_is_widget_backed(self):
        mod = _import_mapper()
        assert mod._is_widget_backed(("BOOLEAN", {"default": True}))

    def test_dropdown_is_widget_backed(self):
        mod = _import_mapper()
        assert mod._is_widget_backed((["a", "b", "c"], {"default": "a"}))

    def test_project_state_is_socket_only(self):
        mod = _import_mapper()
        assert not mod._is_widget_backed(("PROJECT_STATE", {}))

    def test_model_is_socket_only(self):
        mod = _import_mapper()
        assert not mod._is_widget_backed(("MODEL", {}))

    def test_audio_is_socket_only(self):
        mod = _import_mapper()
        assert not mod._is_widget_backed(("AUDIO", {}))

    def test_image_is_socket_only(self):
        mod = _import_mapper()
        assert not mod._is_widget_backed(("IMAGE", {}))


# ---------------------------------------------------------------------------
# BUG-LOCAL-026: scene regex tightening.
# The old pattern \S+ matched literals like "FINAL". The new pattern
# captures digits only, and a separate terminator regex picks up
# '=== SCENE FINAL ===' and appends 'END'.
# ---------------------------------------------------------------------------
def _import_orchestrator_regexes():
    """Read the regex source literals out of story_orchestrator.py so we
    don't have to import the module (which pulls torch/transformers).

    We locate the line 'r\\'===...==='' literal directly. This is fragile
    by design: any regression on the pattern shape will be caught by the
    source-level assertion before the compiled regex is exercised."""
    src = os.path.join(PACK_ROOT, "nodes", "story_orchestrator.py")
    text = open(src, encoding="utf-8").read()
    marker_line = None
    terminator_line = None
    for line in text.splitlines():
        stripped = line.strip()
        if marker_line is None and stripped.startswith("r'===\\s*SCENE\\s+(\\d+)"):
            marker_line = stripped.rstrip(",")
        if terminator_line is None and stripped.startswith("r'===\\s*SCENE\\s+FINAL"):
            terminator_line = stripped.rstrip(",")
    assert marker_line, "Could not locate numeric SCENE marker pattern"
    assert terminator_line, "Could not locate SCENE FINAL terminator pattern"
    # Strip surrounding r'...' quotes.
    def _extract_literal(line):
        # Finds the r'...' literal
        m = re.match(r"r'(.+)'\s*$", line)
        assert m, f"Could not parse pattern literal from line: {line!r}"
        return m.group(1)
    marker = re.compile(_extract_literal(marker_line), re.IGNORECASE)
    terminator = re.compile(_extract_literal(terminator_line), re.IGNORECASE)
    return marker, terminator


class TestSceneRegex:
    def test_matches_numeric_scene(self):
        marker, _ = _import_orchestrator_regexes()
        assert marker.search("=== SCENE 1 ===")

    def test_matches_numeric_scene_with_title(self):
        marker, _ = _import_orchestrator_regexes()
        m = marker.search("=== SCENE 2: The Confrontation ===")
        assert m
        assert m.group(1) == "2"

    def test_does_not_capture_final_as_number(self):
        marker, _ = _import_orchestrator_regexes()
        # The old pattern matched this and captured "FINAL". The new one
        # must NOT match it as a scene-number marker.
        assert not marker.search("=== SCENE FINAL ===")
        assert not marker.search("=== SCENE FINAL: Payoff ===")

    def test_terminator_catches_final(self):
        _, terminator = _import_orchestrator_regexes()
        assert terminator.search("=== SCENE FINAL ===")
        assert terminator.search("=== SCENE FINAL: The Reveal ===")

    def test_mixed_script_tokens(self):
        marker, terminator = _import_orchestrator_regexes()
        script = (
            "=== SCENE 1 ===\nOPENING\n"
            "=== SCENE 2: Rising Action ===\nMIDDLE\n"
            "=== SCENE FINAL ===\nCLOSING\n"
        )
        numeric_tokens = [m.group(1) for m in marker.finditer(script)]
        assert numeric_tokens == ["1", "2"]
        assert terminator.search(script)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
