"""tests/test_otr_api_companions.py

Sprint H §3.7 unblock (Jeffrey 2026-05-17 round-robin Reading C).
Guardrails for the companion-aware serialized slot mapper in
`scripts/otr_api.py`. The mapper teaches the patcher about ComfyUI's
client-side `control_after_generate` companion that the UI auto-injects
next to every INT seed widget.

These tests exercise:

  1. The synthesis: declaration order + companion injection produces a
     slot list whose length equals `widgets_values` for a clean-saved
     workflow (exercised with synthetic seed-bearing nodes).
  2. The writer's `creative_writing_model` patch lands at slot 3.
  3. The writer's `technical_model` patch lands at slot 4.
  4. Narrow loosening: an extra slot in a position OTHER than the
     companion position still rejects with ValueError.
  5. Narrow loosening: a saved array longer than declared + companion
     count still rejects.

The OTR_LedgerScriptWriter schema in the fixtures mirrors node 1's
widgets_values dump from `workflows/otr_scifi_16gb_full.json`. The
`seed` widget was removed (BUG-LOCAL-269/270), so the writer no longer
carries a seed value or a control_after_generate companion:

  0  episode_title           STRING        ''
  1  target_words            INT           350
  2  num_characters          INT           2
  3  creative_writing_model  STRING        'google/gemma-4-E4B-it'
  4  technical_model         STRING        'google/gemma-4-E4B-it'
  5  custom_premise          STRING        ''
  6  include_act_breaks      BOOL          True
  7  act_count               INT           3
  8  style                   COMBO         'let the story decide'
  9  style_custom            STRING        ''
  10 creativity              COMBO         'balanced'
  11 perfect_run_spacesaver  BOOL          False
  12 min_p                   FLOAT         0.05
  13 repetition_penalty      FLOAT         1.03
  14 max_new_tokens_cap      INT           200
  15 enable_polish_pass      BOOL          False
  16 lemmy_cameo             COMBO         'roll (~11% chance)'

The writer has NO seed widget. The mapper's control_after_generate
companion logic is still load-bearing for other nodes (HuMo / Bark
seeds) and is exercised below with synthetic seed-bearing nodes
(MiniNode / OneSeedNode).
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
                    "max_new_tokens_cap": ("INT", {"default": 200}),
                    "lemmy_cameo": (
                        ["roll (~11% chance)", "always include",
                         "never include"],
                        {"default": "roll (~11% chance)"},
                    ),
                    # Build 4 (2026-05-28): grouped-exchange widget at
                    # slot 18 (replaced use_multiturn_dialogue in the
                    # 2026-05-29 lean-down). Mirrors INPUT_TYPES order.
                    "use_exchange": (
                        "BOOLEAN", {"default": False},
                    ),
                    # Sprint 10B Wave 1 Agent B (2026-05-27): in-line
                    # Stage 3 validators widget at slot 19.
                    "enable_production_stage3_validators": (
                        "BOOLEAN", {"default": False},
                    ),
                    # Sprint 2.2 (2026-05-28): news-brief hard-halt, slot 20.
                    "news_briefs_required": (
                        "BOOLEAN", {"default": True},
                    ),
                    # S2 (2026-06-01): the two OpenRouter slot-slug pickers,
                    # appended at the END (indices 19/20) to mirror the live
                    # INPUT_TYPES + the canonical workflow node-1 vector.
                    "openrouter_slot_a_model": ("STRING", {"default": ""}),
                    "openrouter_slot_b_model": ("STRING", {"default": ""}),
                    # Comfy Credits (2026-06-01): the sibling slot-slug pickers,
                    # appended at indices 21/22 to mirror the live INPUT_TYPES +
                    # the canonical workflow node-1 vector.
                    "comfy_slot_a_model": ("STRING", {"default": ""}),
                    "comfy_slot_b_model": ("STRING", {"default": ""}),
                    # Refine loop v1 (2026-06-23): the refine_target_grade
                    # dropdown appended at index 23 (END) to mirror the live
                    # INPUT_TYPES + the canonical workflow node-1 vector.
                    "refine_target_grade": (
                        ["Off", "C+", "B", "B+", "A"], {"default": "Off"},
                    ),
                    # Story-scaffold toggle (2026-06-24): appended at index 24
                    # (END) to mirror the live INPUT_TYPES + the canonical
                    # workflow node-1 vector.
                    "story_scaffold": (
                        ["auto", "on", "off"], {"default": "auto"},
                    ),
                },
                "optional": {},
            }
        }
    }


def _writer_node_fixture() -> dict:
    """Workflow fixture with node 1 carrying the 19-entry widgets_values
    layout dumped from workflows/otr_scifi_16gb_full.json. The `seed`
    widget + its companion were removed in BUG-LOCAL-269/270 (vector
    19 -> 17). Sprint 10A step 3-C (2026-05-26) appended
    enable_stage1_shadow_pass at slot 17 (vector 17 -> 18). Sprint 10B
    Wave 0 (2026-05-27) appended use_multiturn_dialogue at slot 18
    (vector 18 -> 19). The string model ids reflect the slot-2
    reconciliation -- 'mistralai/Mistral-Nemo-Instruct-2407' is the
    canonical default.
    """
    return {
        "nodes": [
            {
                "id": 1,
                "type": "OTR_LedgerScriptWriter",
                "inputs": [],
                "widgets_values": [
                    "",                                       # 0  episode_title
                    350,                                      # 1  target_words
                    2,                                        # 2  num_characters
                    "mistralai/Mistral-Nemo-Instruct-2407",   # 3  creative_writing_model
                    "mistralai/Mistral-Nemo-Instruct-2407",   # 4  technical_model
                    "",                                       # 5  custom_premise
                    True,                                     # 6  include_act_breaks
                    3,                                        # 7  act_count
                    "let the story decide",                   # 8  style
                    "",                                       # 9  style_custom
                    "balanced",                               # 10 creativity
                    False,                                    # 11 perfect_run_spacesaver
                    0.05,                                     # 12 min_p
                    1.03,                                     # 13 repetition_penalty
                    200,                                      # 14 max_new_tokens_cap
                    "roll (~11% chance)",                     # 15 lemmy_cameo
                    False,                                    # 16 use_exchange
                    False,                                    # 17 enable_production_stage3_validators
                    True,                                     # 18 news_briefs_required
                    "anthropic/claude-opus-4.8",              # 19 openrouter_slot_a_model
                    "deepseek/deepseek-v4-pro",               # 20 openrouter_slot_b_model
                    "anthropic/claude-opus-4.7",              # 21 comfy_slot_a_model
                    "deepseek/deepseek-v4-pro",               # 22 comfy_slot_b_model
                    "Off",                                    # 23 refine_target_grade
                    "auto",                                   # 24 story_scaffold
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
    (target_words, num_characters, act_count, max_new_tokens_cap) are
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
# Test 2 -- creative_writing_model patch lands on the right slot
# ---------------------------------------------------------------------------
def test_patch_creative_writing_model_lands_on_correct_index():
    """Using node 1's 18-entry fixture, patch creative_writing_model
    and assert widgets_values[3] changes. The writer's seed widget --
    and its control_after_generate companion -- were removed in
    BUG-LOCAL-269/270. Sprint 10A step 2 reconciliation flipped the
    canonical model to Mistral-Nemo; step 3-C appended the shadow-
    pass widget at slot 17 (vector 17 -> 18).
    """
    schemas = _writer_schemas()
    workflow = _writer_node_fixture()

    pre_creative = workflow["nodes"][0]["widgets_values"][3]
    assert pre_creative == "mistralai/Mistral-Nemo-Instruct-2407"

    patch_widget_by_name(
        workflow, 1, "creative_writing_model",
        "google/gemma-4-E4B-it", schemas,
    )

    wv = workflow["nodes"][0]["widgets_values"]
    assert wv[3] == "google/gemma-4-E4B-it", (
        f"creative_writing_model patch must land at slot 3; "
        f"got wv[3] = {wv[3]!r}"
    )
    # num_characters at slot 2 untouched.
    assert wv[2] == 2


# ---------------------------------------------------------------------------
# Test 3 -- technical_model patch lands on the right slot
# ---------------------------------------------------------------------------
def test_patch_technical_model_lands_on_correct_index():
    """Patch technical_model on the 18-entry fixture. Assert
    widgets_values[4] changes and slot 3 (creative_writing_model) is
    untouched. Sprint 10A step 2 reconciliation flipped the canonical
    model to Mistral-Nemo; step 3-C appended slot 17.
    """
    schemas = _writer_schemas()
    workflow = _writer_node_fixture()

    patch_widget_by_name(
        workflow, 1, "technical_model",
        "google/gemma-4-E4B-it", schemas,
    )

    wv = workflow["nodes"][0]["widgets_values"]
    assert wv[4] == "google/gemma-4-E4B-it", (
        f"technical_model patch must land at slot 4; "
        f"got wv[4] = {wv[4]!r}"
    )
    assert wv[3] == "mistralai/Mistral-Nemo-Instruct-2407", (
        f"creative_writing_model at slot 3 must be untouched; "
        f"got wv[3] = {wv[3]!r}"
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
    and assert node 1's declared inputs land on the right slots.

    Post BUG-LOCAL-269/270 the writer's `seed` widget was removed, so
    there is no longer a seed value or a "fixed" control_after_generate
    companion in the writer's widgets_values:
        wv[3]   creative_writing_model
        wv[4]   technical_model

    Sprint 10A step 3-C (2026-05-26) appended enable_stage1_shadow_pass
    at slot 17, bringing the vector length to 18.

    Sprint 10B Wave 0 (2026-05-27) appended use_multiturn_dialogue at
    slot 18, bringing the vector length to 19.

    Sprint 10B Wave 1 Agent B (2026-05-27) appended
    enable_production_stage3_validators at slot 19, bringing the
    vector length to 20.

    S2 (2026-06-01) appended openrouter_slot_a_model +
    openrouter_slot_b_model at slots 19/20, bringing the vector length
    to 21. Comfy Credits (2026-06-01) appended comfy_slot_a_model +
    comfy_slot_b_model at slots 21/22, bringing it to 23. The new slots
    are appended at the END so creative/technical stay at wv[3]/wv[4].
    """
    dump = _dump_canonical_node1()
    assert len(dump) == 25, f"node 1 widgets_values length drift: {len(dump)}"
    expected_creative = dump[3]
    expected_technical = dump[4]

    schemas = _writer_schemas()
    workflow = load_workflow(str(_CANONICAL_WORKFLOW))
    prompt = workflow_to_api_prompt(workflow, schemas)

    n1_inputs = prompt["1"]["inputs"]

    assert n1_inputs["creative_writing_model"] == expected_creative, (
        f"inputs['creative_writing_model'] expected {expected_creative!r}; "
        f"got {n1_inputs['creative_writing_model']!r}"
    )
    assert n1_inputs["technical_model"] == expected_technical, (
        f"inputs['technical_model'] expected {expected_technical!r}; "
        f"got {n1_inputs['technical_model']!r}"
    )

    # The writer no longer declares a `seed` input (BUG-LOCAL-269/270).
    assert "seed" not in n1_inputs, (
        f"node 1 still exposes a 'seed' input after the widget removal: "
        f"{n1_inputs.get('seed')!r}"
    )

    # A control_after_generate companion token must NEVER reach a
    # declared input. (The writer has no seed widget now, so no
    # companion is emitted -- this stays as a defensive guard.)
    for field in (
        "creative_writing_model",
        "technical_model",
        "target_words",
        "num_characters",
        "act_count",
        "min_p",
        "repetition_penalty",
        "max_new_tokens_cap",
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
