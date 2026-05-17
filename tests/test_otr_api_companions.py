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
  2. The bughunt workflow shape (19-entry widgets_values on node 1):
     a `creative_writing_model` patch lands at slot 5, NOT slot 4 (which
     is the `"fixed"` companion). The companion value is preserved.
  3. Same for `technical_model` patch -> slot 6.
  4. Narrow loosening: an extra slot in a position OTHER than the
     companion position still rejects with ValueError.
  5. Narrow loosening: a saved array longer than declared + companion
     count still rejects.

The OTR_LedgerScriptWriter schema in the fixtures mirrors node 1's
widgets_values dump from `workflows/otr_scifi_16gb_bughunt.json`:

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
  13 optimization_profile    COMBO         'Standard'
  14 perfect_run_spacesaver  BOOL          False
  15 min_p                   FLOAT         0.05
  16 repetition_penalty      FLOAT         1.03
  17 (extra INT widget)      INT           200
  18 (extra BOOL widget)     BOOL          False

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
    patch_widget_by_name,
)


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
                    "optimization_profile": (
                        ["Standard", "Obsidian", "8-bit"],
                        {"default": "Standard"},
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
    """Workflow fixture with node 1 carrying the 19-entry widgets_values
    layout dumped from workflows/otr_scifi_16gb_bughunt.json on 2026-05-17.
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
                    "Standard",                  # 13 optimization_profile
                    False,                       # 14 perfect_run_spacesaver
                    0.05,                        # 15 min_p
                    1.03,                        # 16 repetition_penalty
                    200,                         # 17 max_new_tokens
                    False,                       # 18 stream
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
    """Using node 1's actual 19-entry fixture, patch
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
