"""S21.1 + S21.2 drift guards for VRAM-budget tuning in story_orchestrator.

S21.1: flagship GPU-pinning threshold lowered 15.0 -> 14.5 to catch
RTX 5080 Laptop class hardware reporting ~14.7 GiB after OS reservations.

S21.2: Gemma 4 E2B/E4B context cap aligned with the rest of the table
at 8192 (was 16384). The 16K cap added 2-3 GiB dynamic VRAM during
long generation right when audio models want the envelope.
"""
from __future__ import annotations

import ast
import pathlib

import pytest


# S31.5 B1 refactor: S31 B2 ported the `_MODEL_CONTEXT_CAPS` dict and
# `>= 14.5 GiB` flagship-sovereignty threshold from
# `nodes/story_orchestrator.py` to `nodes/_otr_model_loader.py`
# (canonical LLM loader home post-S31 B2). Update the source target
# so the S21.1/S21.2 drift guards still pin the same patterns.
# Bucket B in the BUG-LOCAL-227 triage classification.
_ORCH_SRC = (
    pathlib.Path(__file__).resolve().parent.parent
    / "nodes" / "_otr_model_loader.py"
).read_text(encoding="utf-8")


# ----- S21.1 ----------------------------------------------------------------


def test_flagship_vram_threshold_is_14_5():
    """AST-walk the file and find the ``total_vram >= 14.5`` comparison.

    Pinned so a future revert to 15.0 (or any other value) is caught
    in CI before it ships.
    """
    tree = ast.parse(_ORCH_SRC)
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        # We're looking for: <Name 'total_vram'> >= <Constant 14.5>
        if not (
            isinstance(node.left, ast.Name)
            and node.left.id == "total_vram"
        ):
            continue
        if not node.comparators:
            continue
        comp = node.comparators[0]
        if not isinstance(comp, ast.Constant):
            continue
        if isinstance(node.ops[0], ast.GtE) and comp.value == 14.5:
            found = True
            break
    assert found, (
        "S21.1: total_vram >= 14.5 comparison not found in "
        "story_orchestrator.py. The flagship GPU-pinning threshold "
        "was either reverted or moved."
    )


# ----- S21.2 ----------------------------------------------------------------


def test_all_model_context_caps_at_8192():
    """Every entry in _MODEL_CONTEXT_CAPS must be 8192. S21.2 aligned
    Gemma 4 E2B/E4B (formerly 16384) with the rest of the table.

    Source-level AST walk so we don't have to import story_orchestrator
    (which pulls torch + transformers + the OTR class graph).
    """
    tree = ast.parse(_ORCH_SRC)
    found_dict = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if (
                    isinstance(tgt, ast.Name)
                    and tgt.id == "_MODEL_CONTEXT_CAPS"
                ):
                    found_dict = node.value
                    break
        if found_dict is not None:
            break
    assert isinstance(found_dict, ast.Dict), (
        "S21.2: _MODEL_CONTEXT_CAPS dict not found at expected name."
    )
    for k_node, v_node in zip(found_dict.keys, found_dict.values):
        if not isinstance(k_node, ast.Constant):
            continue
        if not isinstance(v_node, ast.Constant):
            continue
        assert v_node.value == 8192, (
            f"S21.2: model {k_node.value!r} has cap {v_node.value} "
            f"!= 8192. Aligning Gemma 4 E-series to 8192 was the "
            f"S21.2 change; new entries must follow the same cap "
            f"unless the audit-and-decide path is documented inline."
        )


def test_default_cap_is_8192():
    """The fallback ``_cap = _MODEL_CONTEXT_CAPS.get(_resolved_id, 8192)``
    line still uses 8192 as the default."""
    assert "_MODEL_CONTEXT_CAPS.get(_resolved_id, 8192)" in _ORCH_SRC, (
        "S21.2: default-cap fallback no longer uses 8192. Audit any "
        "change to this default in lockstep with the per-model caps "
        "above."
    )
