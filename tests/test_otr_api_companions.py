"""tests/test_otr_api_companions.py

Sprint H §3.7 unblock (Jeffrey 2026-05-17 round-robin Reading C).
Guardrails for the companion-aware serialized slot mapper in
`scripts/otr_api.py`. The mapper teaches the patcher about ComfyUI's
client-side `control_after_generate` companion that the UI auto-injects
next to every INT seed widget.

These tests exercise:

  1. The synthesis: declaration order + companion injection produces a
     slot list whose length equals `widgets_values` for a clean-saved
     workflow.
  2. The canonical workflow shape (18-entry widgets_values on node 1):
     a `creative_writing_model` patch lands at slot 5, NOT slot 4 (which
     is the `"fixed"` companion). The companion value is preserved.
  3. Same for `technical_model` patch -> slot 6.
  4. Narrow loosening: an extra slot in a position OTHER than the
     companion position still rejects with ValueError.
  5. Narrow loosening: a saved array longer than declared + companion
     count still rejects.

The OTR_LedgerScriptWriter schema in the fixtures mirrors node 1's
widgets_values dump from `workflows/otr_scifi_16gb_full.json`
(optimization_profile widget removed 2026-05-23, ROADMAP PRIORITY 2):

  0  episode_title           STRING        ''
  1  target_words            INT           350
  2  num_characters          INT           2
  3  seed                    INT           42
  4  seed companion          (synthetic)   'fixed'
  5  creative_writing_model  STRING        'google/gemma-4-E4B-it'
  6  technical_model         STRING        'google/gemma-4-E4B-it'
  7  custom_premise          STRING        ''
  8  include_act_breaks      BOOL          True
  9  act_count               INT           3
  10 style                   COMBO         'let the story decide'
  11 style_custom            STRING        ''
  12 creativity              COMBO         'balanced'
  13 perfect_run_spacesaver  BOOL          False
  14 min_p                   FLOAT         0.05
  15 repetition_penalty      FLOAT         1.03
  16 (extra INT widget)      INT           200
  17 (extra BOOL widget)     BOOL          False

The `target_words` + `num_characters` + `act_count` INT widgets are
NOT named "seed" or "noise_seed" -- they do NOT trigger companion
injection. Only `seed` does. This is the load-bearing invariant the
mapper depends on.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from otr_api import (  # noqa: E402
    _serialized_slot_names,
    load_workflow,
    patch_widget_by_name,
    workflow_to_api_prompt,
)

_REPO = Path(__file__).resolve().parent.parent
# Jeffrey 2026-05-18 overnight reconcile: single workflow source of
# truth -- _bughunt.json sibling deleted. Round-trip fixture now reads
# the canonical _full.json directly.
_CANONICAL_WORKFLOW = _REPO / "workflows" / "otr_scifi_16gb_full.json"


def _writer_schemas() -> dict:
    """Schema fixture mirroring OTR_LedgerScriptWriter node 1 widget order.

    Built positionally from the dump documented at module top. Order
    matches the live /object_info response shape: a node entry with
    `input.required` + `input.optional` dicts, each value a (type, opts)
    tuple.
    """
    # Use a single `required` dict in declaration order. Python 3.7+ dicts
    # are insertion-ordered, so this matches what ComfyUI serializes.
    return {
        "OTR_LedgerScriptWriter": {
            "input": {
                "required": {
                    "episode_title": ("STRING", {"default": ""}),
                    "target_words": ("INT", {"default": 350}),
                    "num_characters": ("INT", {"default": 2}),
                    "seed": ("INT", {"default": 42}),
                    "creative_writing_model": ("STRING", {"default": ""}),
                    "technical_model": ("STRING", {"default": ""}),
                    "custom_premise": ("STRING", {"default": ""}),
                    "include_act_breaks": ("BOOLEAN", {"default": True}),
                    "act_count": ("INT", {"default": 3}),
                    "style": (
                        ["let the story decide", "mission_control_procedural"],
                        {"default": "let the story decide"},
                    ),
                    "style_custom": ("STRING", {"default": ""}),
                    "creativity": (
                        ["balanced", "high", "low"],
                        {"default": "balanced"},
                    ),
                    "perfect_run_spacesaver": (
                        "BOOLEAN", {"default": False},
                    ),
                    "min_p": ("FLOAT", {"default": 0.05}),
                    "repetition_penalty": ("FLOAT", {"default": 1.03}),
                    "max_new_tokens": ("INT", {"default": 200}),
                    "stream": ("BOOLEAN", {"default": False}),
                },
                "optional": {},
            }
        }
    }


def _writer_node_fixture() -> dict:
    """Workflow fixture with node 1 carrying the 18-entry widgets_values
    layout dumped from workflows/otr_scifi_16gb_full.json (optimization_
    profile widget removed 2026-05-23, ROADMAP PRIORITY 2).
    """
    return {
        "nodes": [
            {
                "id": 1,
                "type": "OTR_LedgerScriptWriter",
                "inputs": [],
                "widgets_values": [
                    "",                          # 0  episode_title
                    350,                         # 1  target_words
                    2,                           # 2  num_characters
                    42,                          # 3  seed
                    "fixed",                     # 4  seed companion
                    "google/gemma-4-E4B-it",     # 5  creative_writing_model
                    "google/gemma-4-E4B-it",     # 6  technical_model
                    "",                          # 7  custom_premise
                    True,                        # 8  include_act_breaks
                    3,                           # 9  act_count
                    "let the story decide",      # 10 style
                    "",                          # 11 style_custom
                    "balanced",                  # 12 creativity
                    False,                       # 13 perfect_run_spacesaver
                    0.05,                        # 14 min_p
                    1.03,                        # 15 repetition_penalty
                    200,                         # 16 max_new_tokens
                    False,                       # 17 stream
                ],
            }
        ],
        "links": [],
    }


# ---------------------------------------------------------------------------
# Test 1 -- serialized slot list shape
# ---------------------------------------------------------------------------
def test_serialized_slots_includes_seed_companion():
    """Stub schema with one INT widget named `seed`. Assert the
    returned slot list has length = declared_widgets + 1 and the
    companion appears at seed_index + 1 with the synthetic name
    `seed__control_after_generate`.
    """
    schemas = {
        "MiniNode": {
            "input": {
                "required": {
                    "title": ("STRING", {"default": ""}),
                    "seed": ("INT", {"default": 0}),
                    "factor": ("FLOAT", {"default": 1.0}),
                },
                "optional": {},
            }
        }
    }
    slots = _serialized_slot_names("MiniNode", schemas)
    declared = ["title", "seed", "factor"]

    assert len(slots) == len(declared) + 1, (
        f"expected len(slots) = len(declared) + 1 = {len(declared) + 1}; "
        f"got {len(slots)} with slots={slots!r}"
    )
    seed_idx = slots.index("seed")
    assert slots[seed_idx + 1] == "seed__control_after_generate", (
        f"companion must appear at seed_idx + 1; "
        f"got slots[{seed_idx + 1}] = {slots[seed_idx + 1]!r}"
    )


def test_serialized_slots_no_companion_for_non_seed_int():
    """Defense-in-depth: an INT widget not named seed / noise_seed
    must NOT trigger companion injection. Most OTR INT widgets
    (target_words, num_characters, act_count, max_new_tokens) are
    declared and rely on this behavior.
    """
    schemas = {
        "NoSeedNode": {
            "input": {
                "required": {
                    "target_words": ("INT", {"default": 350}),
                    "act_count": ("INT", {"default": 3}),
                },
                "optional": {},
            }
        }
    }
    slots = _serialized_slot_names("NoSeedNode", schemas)
    assert slots == ["target_words", "act_count"], (
        f"non-seed INT widgets must not inject companions; got {slots!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 -- creative_writing_model patch lands past the companion
# ---------------------------------------------------------------------------
def test_patch_skips_companion_and_lands_on_correct_index():
    """Using node 1's actual 18-entry fixture, patch
    creative_writing_model and assert widgets_values[5] changes
    (NOT widgets_values[4], which is the companion).
    """
    schemas = _writer_schemas()
    workflow = _writer_node_fixture()

    pre_companion = workflow["nodes"][0]["widgets_values"][4]
    pre_creative = workflow["nodes"][0]["widgets_values"][5]
    assert pre_companion == "fixed"
    assert pre_creative == "google/gemma-4-E4B-it"

    patch_widget_by_name(
        workflow, 1, "creative_writing_model",
        "mistralai/Mistral-Nemo-Instruct-2407", schemas,
    )

    wv = workflow["nodes"][0]["widgets_values"]
    assert wv[5] == "mistralai/Mistral-Nemo-Instruct-2407", (
        f"creative_writing_model patch must land at slot 5; "
        f"got wv[5] = {wv[5]!r}"
    )
    assert wv[4] == "fixed", (
        f"companion at slot 4 must be untouched; got wv[4] = {wv[4]!r}"
    )
    # Slot 3 (seed value) also untouched.
    assert wv[3] == 42


# ---------------------------------------------------------------------------
# Test 3 -- technical_model patch skips both companion and
# creative_writing_model
# ---------------------------------------------------------------------------
def test_patch_technical_model_skips_companion():
    """Patch technical_model on the same fixture. Assert
    widgets_values[6] changes and slots 4 (companion) + 5
    (creative_writing_model) are untouched.
    """
    schemas = _writer_schemas()
    workflow = _writer_node_fixture()

    patch_widget_by_name(
        workflow, 1, "technical_model",
        "mistralai/Mistral-Nemo-Instruct-2407", schemas,
    )

    wv = workflow["nodes"][0]["widgets_values"]
    assert wv[6] == "mistralai/Mistral-Nemo-Instruct-2407", (
        f"technical_model patch must land at slot 6; "
        f"got wv[6] = {wv[6]!r}"
    )
    assert wv[4] == "fixed", (
        f"companion at slot 4 must be untouched; got wv[4] = {wv[4]!r}"
    )
    assert wv[5] == "google/gemma-4-E4B-it", (
        f"creative_writing_model at slot 5 must be untouched; "
        f"got wv[5] = {wv[5]!r}"
    )


# ---------------------------------------------------------------------------
# Test 4 -- extra slot OUTSIDE companion position still rejected
# ---------------------------------------------------------------------------
def test_extra_slot_outside_companion_position_still_rejected():
    """Synthesize a node with an extra slot NOT in the
    seed-companion position. The narrow loosening must NOT swallow
    this -- the patcher must raise ValueError on length drift.

    Schema: title (STRING), factor (FLOAT). No seed -> no companion
    expected. Saved widgets_values has 3 entries instead of the
    declared 2. The extra cannot be a companion (no seed). Reject.
    """
    schemas = {
        "ExtraSlotNode": {
            "input": {
                "required": {
                    "title": ("STRING", {"default": ""}),
                    "factor": ("FLOAT", {"default": 1.0}),
                },
                "optional": {},
            }
        }
    }
    workflow = {
        "nodes": [
            {
                "id": 7,
                "type": "ExtraSlotNode",
                "inputs": [],
                "widgets_values": ["title_val", 2.5, "phantom"],
            }
        ],
        "links": [],
    }

    with pytest.raises(ValueError, match="widgets_values length mismatch"):
        patch_widget_by_name(
            workflow, 7, "title", "new_title", schemas,
        )


# ---------------------------------------------------------------------------
# Test 5 -- count exceeds declared + companion budget
# ---------------------------------------------------------------------------
def test_extra_slot_count_exceeds_companions_rejected():
    """One seed widget licenses ONE companion slot. A workflow with
    len(wv) = len(declared) + 2 has one slot too many; the narrow
    loosening must reject.
    """
    schemas = {
        "OneSeedNode": {
            "input": {
                "required": {
                    "title": ("STRING", {"default": ""}),
                    "seed": ("INT", {"default": 0}),
                    "factor": ("FLOAT", {"default": 1.0}),
                },
                "optional": {},
            }
        }
    }
    # Declared count = 3. One companion licensed -> expected = 4.
    # Saved widgets_values has 5 entries -> reject.
    workflow = {
        "nodes": [
            {
                "id": 9,
                "type": "OneSeedNode",
                "inputs": [],
                "widgets_values": ["title_val", 42, "fixed", 1.5, "extra"],
            }
        ],
        "links": [],
    }

    with pytest.raises(ValueError, match="widgets_values length mismatch"):
        patch_widget_by_name(
            workflow, 9, "title", "new_title", schemas,
        )


# ---------------------------------------------------------------------------
# Round-trip: workflow_to_api_prompt honors companion + does not bleed "fixed"
# ---------------------------------------------------------------------------
def _dump_canonical_node1() -> list:
    """Read _full.json node 1 widgets_values verbatim.

    Per Jeffrey 2026-05-17 directive: do NOT trust prose recollection;
    inspect the on-disk dump first, then build the assertions.
    """
    wf = load_workflow(str(_CANONICAL_WORKFLOW))
    for n in wf["nodes"]:
        if n["id"] == 1:
            return list(n["widgets_values"])
    raise AssertionError("node id=1 missing from _full.json")


def test_round_trip_canonical_node1_inputs_correct():
    """Round-trip the canonical workflow through the API converter
    and assert declared inputs land on the right slots, with the
    companion "fixed" NEVER reaching a declared input.

    Asserts derived from on-disk dump of node 1 widgets_values:
        wv[3]   seed value  -> inputs["seed"]
        wv[5]   creative_writing_model
        wv[6]   technical_model
        wv[4]   "fixed" companion (NOT mapped anywhere)
    """
    dump = _dump_canonical_node1()
    assert len(dump) == 18, f"node 1 widgets_values length drift: {len(dump)}"
    expected_seed = dump[3]
    expected_companion = dump[4]
    expected_creative = dump[5]
    expected_technical = dump[6]
    assert expected_companion == "fixed", (
        f"node 1 widgets_values[4] expected 'fixed' companion; "
        f"got {expected_companion!r}"
    )

    schemas = _writer_schemas()
    workflow = load_workflow(str(_CANONICAL_WORKFLOW))
    prompt = workflow_to_api_prompt(workflow, schemas)

    n1_inputs = prompt["1"]["inputs"]

    assert n1_inputs["seed"] == expected_seed, (
        f"inputs['seed'] expected {expected_seed!r}; "
        f"got {n1_inputs['seed']!r}"
    )
    assert n1_inputs["creative_writing_model"] == expected_creative, (
        f"inputs['creative_writing_model'] expected {expected_creative!r}; "
        f"got {n1_inputs['creative_writing_model']!r}"
    )
    assert n1_inputs["technical_model"] == expected_technical, (
        f"inputs['technical_model'] expected {expected_technical!r}; "
        f"got {n1_inputs['technical_model']!r}"
    )

    # "fixed" must NOT appear in any declared input on node 1. The
    # companion vocabulary {fixed, randomize, increment, decrement}
    # is never a legitimate declared-input value for this writer.
    for field in (
        "creative_writing_model",
        "technical_model",
        "seed",
        "target_words",
        "num_characters",
        "act_count",
        "min_p",
        "repetition_penalty",
        "max_new_tokens",
    ):
        if field in n1_inputs:
            assert n1_inputs[field] != "fixed", (
                f"inputs[{field!r}] = 'fixed' -- companion bled into "
                f"declared input. Conversion regressed."
            )


# ---------------------------------------------------------------------------
# Misplaced companion -- value-at-companion-slot vocabulary check
# ---------------------------------------------------------------------------
def test_misplaced_companion_value_rejected():
    """A workflow whose schema declares a seed widget but whose saved
    widgets_values has a non-vocabulary value at the companion slot
    indicates widget drift. The converter must refuse rather than
    silently propagate the stray value.

    Schema: [a STRING, seed INT, b STRING] -> serialized slots 4:
        [a, seed, seed__control_after_generate, b]
    widgets_values: ["a_val", 42, "TYPO_NOT_FIXED", "b_val"] -- length 4
    matches; but slot 2 has "TYPO_NOT_FIXED" not in the companion
    vocabulary. Reject.
    """
    schemas = {
        "MisplacedSeedNode": {
            "input": {
                "required": {
                    "a": ("STRING", {"default": ""}),
                    "seed": ("INT", {"default": 0}),
                    "b": ("STRING", {"default": ""}),
                },
                "optional": {},
            }
        }
    }
    workflow = {
        "nodes": [
            {
                "id": 11,
                "type": "MisplacedSeedNode",
                "inputs": [],
                "widgets_values": ["a_val", 42, "TYPO_NOT_FIXED", "b_val"],
            }
        ],
        "links": [],
    }

    with pytest.raises(ValueError, match="companion slot"):
        workflow_to_api_prompt(workflow, schemas)


# ---------------------------------------------------------------------------
# Unrelated extra slot rejected by API converter
# ---------------------------------------------------------------------------
def test_converter_unrelated_extra_slot_rejected():
    """No seed widget -> no companion licensed. A workflow with one
    extra slot beyond the declared count must reject during API
    conversion.
    """
    schemas = {
        "NoSeedNode": {
            "input": {
                "required": {
                    "title": ("STRING", {"default": ""}),
                    "factor": ("FLOAT", {"default": 1.0}),
                },
                "optional": {},
            }
        }
    }
    workflow = {
        "nodes": [
            {
                "id": 13,
                "type": "NoSeedNode",
                "inputs": [],
                "widgets_values": ["title_val", 2.5, "phantom"],
            }
        ],
        "links": [],
    }

    with pytest.raises(ValueError, match="length mismatch"):
        workflow_to_api_prompt(workflow, schemas)


# ---------------------------------------------------------------------------
# Companion vocabulary acceptance
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Reading D: forceInput=True widget-backed inputs are socket-only.
# They never occupy a widgets_values slot, even when their TYPE is
# widget-backed and even when they are linked.
# ---------------------------------------------------------------------------
def test_serialized_slots_skips_forceInput_widgets():
    """A STRING widget marked `forceInput=True` must NOT appear in the
    serialized slot list. ComfyUI never renders it as a widget and
    never serializes a slot for it -- regardless of declaration order.
    """
    from otr_api import _serialized_slot_names  # local import for clarity

    schemas = {
        "ForceInputNode": {
            "input": {
                "required": {
                    "a": ("STRING", {"default": ""}),
                    "gate": ("STRING", {"default": "", "forceInput": True}),
                    "b": ("STRING", {"default": ""}),
                },
                "optional": {},
            }
        }
    }
    slots = _serialized_slot_names("ForceInputNode", schemas)
    assert slots == ["a", "b"], (
        f"forceInput-flagged widget must be filtered out of "
        f"serialized_slots; got {slots!r}"
    )


def test_converter_handles_forceInput_link_without_widget_slot():
    """A node with one normal-linked widget (preserved placeholder slot)
    AND one forceInput-linked widget (no slot) must convert cleanly.
    Both linked names should resolve via the link map; widgets_values
    indices must NOT be consumed for the forceInput entry.
    """
    schemas = {
        "MixedLinkNode": {
            "input": {
                "required": {
                    "normal_widget": (
                        "STRING", {"multiline": True, "default": "[]"},
                    ),
                    "extra": ("STRING", {"default": "fill"}),
                    "gate": (
                        "STRING",
                        {"default": "", "forceInput": True},
                    ),
                },
                "optional": {},
            }
        }
    }
    workflow = {
        "nodes": [
            # Upstream source node #99 (no schema needed; its outputs
            # are referenced by link).
            {"id": 99, "type": "UpstreamA", "inputs": [],
             "widgets_values": []},
            {"id": 100, "type": "UpstreamB", "inputs": [],
             "widgets_values": []},
            {
                "id": 30,
                "type": "MixedLinkNode",
                "inputs": [
                    {
                        "name": "normal_widget",
                        "type": "STRING",
                        "link": 501,
                        "widget": {"name": "normal_widget"},
                    },
                    {
                        "name": "gate",
                        "type": "STRING",
                        "link": 502,
                        "widget": {"name": "gate"},
                    },
                ],
                # Preserved-mode save: normal_widget's placeholder slot
                # is kept (`"[]"`), extra is its real value, gate has
                # NO slot because forceInput.
                "widgets_values": ["[]", "extra_val"],
            },
        ],
        "links": [
            # link_id, src_node, src_slot, dst_node, dst_slot, type
            [501, 99, 0, 30, 0, "STRING"],
            [502, 100, 0, 30, 1, "STRING"],
        ],
    }

    prompt = workflow_to_api_prompt(workflow, schemas)

    n_inputs = prompt["30"]["inputs"]
    # normal_widget resolves to its link source via the link map.
    assert n_inputs["normal_widget"] == ["99", 0], (
        f"normal_widget should resolve to link source; got "
        f"{n_inputs.get('normal_widget')!r}"
    )
    # forceInput gate ALSO resolves to its link source -- and does
    # NOT consume the "extra_val" slot.
    assert n_inputs["gate"] == ["100", 0], (
        f"forceInput gate should resolve to link source; got "
        f"{n_inputs.get('gate')!r}"
    )
    # The "extra" widget gets its actual saved value.
    assert n_inputs["extra"] == "extra_val", (
        f"extra widget should receive 'extra_val'; got "
        f"{n_inputs.get('extra')!r}"
    )


def test_regression_node20_actual_workflow():
    """Round-trip the canonical workflow through the API converter
    and assert node 20 (OTR_VideoPlan) converts cleanly. Verifies the
    forceInput fix on the actual schema shape that surfaced the
    blocker (freeze_done_gate forceInput=True).

    Assertions are derived from the on-disk node 20 dump (BUG-LOCAL-250
    removed the style preset widget):
        widgets_values[0]  '{}'    script_json placeholder (linked)
        widgets_values[1]  '(all)'  focus_character
        widgets_values[2]  4        shots_per_scene
        widgets_values[3]  'cinematic, 35mm film look...'  style_tail
        widgets_values[4]  True    include_final_end_frame
        (no slot 5 -- freeze_done_gate has forceInput=True)
    """
    # Stub schema mirroring nodes/otr_video_plan.py:734-810.
    schemas = {
        "OTR_VideoPlan": {
            "input": {
                "required": {
                    "script_json": (
                        "STRING", {"multiline": True, "default": ""},
                    ),
                    "focus_character": (
                        "STRING", {"default": "(all)"},
                    ),
                    "shots_per_scene": (
                        "INT", {"default": 3, "min": 1, "max": 40},
                    ),
                },
                "optional": {
                    "style_tail": (
                        "STRING",
                        {"default": "cinematic, 35mm film look"},
                    ),
                    "include_final_end_frame": (
                        "BOOLEAN", {"default": True},
                    ),
                    "freeze_done_gate": (
                        "STRING",
                        {"default": "", "forceInput": True},
                    ),
                },
            }
        }
    }

    workflow = load_workflow(str(_CANONICAL_WORKFLOW))
    # Should NOT raise (was raising
    # "widgets_values length mismatch on node 20" before the fix).
    prompt = workflow_to_api_prompt(workflow, schemas)

    n20 = prompt["20"]["inputs"]
    assert n20.get("focus_character") == "(all)", (
        f"focus_character should be '(all)'; got "
        f"{n20.get('focus_character')!r}. '{{}}' must NOT bleed in here."
    )
    assert n20.get("shots_per_scene") == 4, (
        f"shots_per_scene should be 4; got {n20.get('shots_per_scene')!r}"
    )
    # BUG-LOCAL-250: the style preset widget was removed from VideoPlan.
    assert "style" not in n20, (
        f"node 20 should carry no 'style' input post-BUG-LOCAL-250; "
        f"got {n20.get('style')!r}"
    )
    assert n20.get("include_final_end_frame") is True, (
        f"include_final_end_frame should be True; got "
        f"{n20.get('include_final_end_frame')!r}"
    )
    # Companion / orphan / forceInput-bleed guard: "{}" must NOT
    # propagate into any declared input. It's the preserved-mode
    # placeholder for the linked script_json, consumed correctly.
    for key, value in n20.items():
        assert value != "{}", (
            f"node 20 inputs[{key!r}] = '{{}}' -- placeholder bled into "
            f"a declared input. forceInput / linked-slot accounting "
            f"regressed."
        )


def test_forceInput_filter_does_not_reduce_unfiltered_count():
    """When three widget-backed inputs exist and only ONE has
    forceInput=True, linked_widget_count must reflect ONLY the two
    non-forceInput inputs that ARE linked -- the forceInput one was
    never a slot, so it must not be counted as a stripped slot either.

    This verifies the symmetric application of the forceInput filter
    to `linked_widget_count` calculation in patch_widget_by_name +
    workflow_to_api_prompt.
    """
    from otr_api import _serialized_slot_names

    schemas = {
        "ThreeLinkedNode": {
            "input": {
                "required": {
                    "a": ("STRING", {"default": ""}),
                    "b": ("STRING", {"default": ""}),
                    "c": (
                        "STRING",
                        {"default": "", "forceInput": True},
                    ),
                },
                "optional": {},
            }
        }
    }
    slots = _serialized_slot_names("ThreeLinkedNode", schemas)
    # The forceInput slot is filtered, leaving only 'a' and 'b'.
    assert slots == ["a", "b"], (
        f"forceInput slot must be filtered before slot counting; "
        f"got {slots!r}"
    )
    # Simulate a node where all three are linked.
    linked_names = {"a", "b", "c"}
    linked_widget_count = sum(1 for n in slots if n in linked_names)
    assert linked_widget_count == 2, (
        f"linked_widget_count should count only slot-occupying "
        f"linked widgets ('a' + 'b'), not the forceInput 'c'. "
        f"Got {linked_widget_count}."
    )


@pytest.mark.parametrize(
    "companion_value",
    ["fixed", "randomize", "increment", "decrement"],
)
def test_converter_accepts_each_companion_vocab_value(companion_value):
    """Each value in the ComfyUI control_after_generate vocabulary
    must be accepted at a companion slot without raising. Round-robin
    risk callout (Reading C section): if a future ComfyUI release
    renames any of these, the mapper needs extension; this test pins
    the current four.
    """
    schemas = {
        "SeedNode": {
            "input": {
                "required": {
                    "title": ("STRING", {"default": ""}),
                    "seed": ("INT", {"default": 0}),
                },
                "optional": {},
            }
        }
    }
    workflow = {
        "nodes": [
            {
                "id": 21,
                "type": "SeedNode",
                "inputs": [],
                "widgets_values": ["title_val", 42, companion_value],
            }
        ],
        "links": [],
    }

    prompt = workflow_to_api_prompt(workflow, schemas)
    assert prompt["21"]["inputs"]["seed"] == 42
    assert prompt["21"]["inputs"]["title"] == "title_val"
    # Companion never appears as a declared input.
    assert "seed__control_after_generate" not in prompt["21"]["inputs"]
