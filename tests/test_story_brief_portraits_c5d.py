"""Sprint C C5d: FLUX portraits wiring tests.

Per refinement section 6.2: portraits narrow to
`get_story_brief_lighting(meta)` rather than `get_story_brief_full`
because portraits do NOT want setting / prop terms pulling
composition toward the env shot. The helper returns lighting +
atmosphere terms only.

Behavioral tests use the E-06 canary token
`zebra_lantern_atmosphere_731` so a regression that bypasses the
helper would surface even when the helper itself misbehaves.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


# `visual/batch_flux_portrait_render.py` does `import folder_paths` at
# module level. `folder_paths` is provided by the ComfyUI runtime and is
# not available in the pure-pytest sandbox; stub it before importing
# so the test module collects without raising.
if "folder_paths" not in sys.modules:
    _stub = types.ModuleType("folder_paths")
    _stub.get_output_directory = lambda: str(Path.cwd())  # noqa: E731
    sys.modules["folder_paths"] = _stub

from visual import batch_flux_portrait_render as bfpr


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_PORTRAITS_PATH = _REPO_ROOT / "visual" / "batch_flux_portrait_render.py"


_CANARY = "zebra_lantern_atmosphere_731"


# ---------------------------------------------------------------------------
# Source-level: helper is used (lighting), full helper is NOT used
# ---------------------------------------------------------------------------


def test_portraits_use_lighting_helper():
    src = _PORTRAITS_PATH.read_text(encoding="utf-8")
    assert "get_story_brief_lighting" in src, (
        "visual/batch_flux_portrait_render.py does not reference "
        "get_story_brief_lighting; C5d wiring missing"
    )


def test_portraits_do_not_use_full_helper():
    """Refinement section 6.2: portraits use the lighting helper, NOT
    the full helper. Setting / prop terms would pull composition
    toward env shots."""
    src = _PORTRAITS_PATH.read_text(encoding="utf-8")
    assert "get_story_brief_full" not in src, (
        "visual/batch_flux_portrait_render.py imports / calls "
        "get_story_brief_full; portraits should use "
        "get_story_brief_lighting instead per refinement section 6.2"
    )


def test_portraits_log_story_brief_status():
    """E-07: report surfaces story_brief_status."""
    src = _PORTRAITS_PATH.read_text(encoding="utf-8")
    assert "story_brief_status=" in src, (
        "portrait report does not surface story_brief_status; "
        "E-07 observability missing"
    )


# ---------------------------------------------------------------------------
# Behavioral -- _build_portrait_prompt
# ---------------------------------------------------------------------------


def test_portrait_prompt_includes_lighting_when_supplied():
    """BUG-LOCAL-250 follow-up: when the helper output is non-empty
    the brief-derived lighting LEADS the prompt -- ahead of the
    generic style anchor, the speaker subject, the appearance, and
    the fixed composition guidance."""
    prompt = bfpr._build_portrait_prompt(
        speaker="Jones",
        appearance="tall detective, weathered face",
        style_anchor="head-and-shoulders studio portrait, neutral lighting",
        lighting=f"swinging bulb, harsh shadow, {_CANARY}, tense",
    )
    assert _CANARY in prompt, (
        "E-06 canary token missing from portrait prompt; "
        "lighting parameter not threaded through _build_portrait_prompt"
    )
    # Brief-leads order: the lighting block is FIRST -- before the
    # style anchor, the appearance, and the story-linked expression
    # composition-guidance block (Sprint 9 Narrative Face).
    canary_idx = prompt.find(_CANARY)
    anchor_idx = prompt.find("head-and-shoulders studio portrait")
    appearance_idx = prompt.find("weathered face")
    composition_idx = prompt.find(
        "subtle expression matching the character's role in the story"
    )
    assert canary_idx > -1
    assert canary_idx < anchor_idx < appearance_idx < composition_idx, (
        f"brief-leads order wrong: canary@{canary_idx} "
        f"anchor@{anchor_idx} appearance@{appearance_idx} "
        f"composition@{composition_idx}"
    )


def test_portrait_prompt_falls_through_when_lighting_empty():
    """Empty lighting: legacy prompt builds without crashing."""
    prompt = bfpr._build_portrait_prompt(
        speaker="Jones",
        appearance="tall detective",
        style_anchor="head-and-shoulders studio portrait, neutral lighting",
        lighting="",
    )
    assert "tall detective" in prompt
    assert (
        "subtle expression matching the character's role in the story"
        in prompt
    )
    assert _CANARY not in prompt


def test_portrait_prompt_no_setting_terms_when_lighting_used():
    """Refinement section 6.2: the lighting helper deliberately
    EXCLUDES setting terms. The prompt should not contain anything
    like 'steel table' (a typical setting term) even when supplied
    in adjacent ledger meta.

    This test exercises the helper itself (via the canary structure)
    rather than asserting empty: helper output may include lighting
    or atmosphere tokens that happen to share names with setting
    concepts; the structural lock is "we called the lighting helper,
    not the full helper" (tested above via source-level assertions).
    """
    # Confirm via helper directly that setting terms are filtered out.
    from nodes._otr_story_brief_helpers import get_story_brief_lighting
    meta = {
        "story_brief":        "the brief",
        "story_brief_status": "ok",
        "story_brief_terms": {
            "setting":    ["steel table", "interrogation room"],
            "lighting":   ["swinging bulb", "harsh shadow"],
            "atmosphere": ["tense", "smoky"],
        },
    }
    lighting = get_story_brief_lighting(meta)
    assert "steel table" not in lighting
    assert "interrogation room" not in lighting
    assert "swinging bulb" in lighting
    assert "tense" in lighting
