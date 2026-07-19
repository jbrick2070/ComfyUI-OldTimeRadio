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


def test_no_hardcoded_model_context_caps_table():
    """The function-local ``_MODEL_CONTEXT_CAPS`` fallback table is GONE
    (2026-07-19). S30 B1b migrated context-cap resolution to
    ``_otr_model_catalog.resolve_context_cap``; this removed the last hardcoded
    vestige so a no-cap ``load_llm`` caller (the ``_LegacyTransformersBackendBase``
    delegate) can no longer silently load a model at a stale value that
    disagrees with the catalog. Guard against anyone re-introducing a parallel
    table (that is how Mistral-Nemo would load at a stale 8192 on the no-cap
    path while ``request_slot`` loaded 16384).

    Source-level AST walk so we don't import torch + transformers.
    """
    tree = ast.parse(_ORCH_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "_MODEL_CONTEXT_CAPS":
                    raise AssertionError(
                        "A hardcoded _MODEL_CONTEXT_CAPS table was re-added to "
                        "nodes/_otr_model_loader.py. Context caps are the "
                        "catalog's single source of truth "
                        "(_otr_model_catalog.resolve_context_cap); a parallel "
                        "table drifts."
                    )


def test_load_llm_delegates_context_cap_to_catalog():
    """When called without an explicit ``context_cap``, ``load_llm`` resolves
    through ``_otr_model_catalog.resolve_context_cap`` -- the SAME path
    ``request_slot`` uses -- so every load path agrees on the effective
    window."""
    assert "resolve_context_cap(_resolved_id)" in _ORCH_SRC, (
        "load_llm no longer delegates its context-cap fallback to "
        "resolve_context_cap. The catalog is the single source of truth; a "
        "hardcoded default here is exactly the drift this guard prevents."
    )
