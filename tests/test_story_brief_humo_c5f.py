"""Sprint C C5f: HuMo lip-sync wiring tests.

Per refinement section 6 / C5f: HuMo per-clip prompt builder appends
`get_story_brief_lighting(meta)` between the character description
and `_DEFAULT_POS_SUFFIX`. Same helper as C5d (portraits) -- both
consumers want lighting + atmosphere, no setting noise.

E-06 meta-threading canary token surfaces regressions.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest


# `nodes/batch_humo_render.py` may import ComfyUI runtime modules
# at module level. Stub the ones safe to no-op for prompt-construction
# tests.
for _stub_name in ("folder_paths",):
    if _stub_name not in sys.modules:
        _stub = types.ModuleType(_stub_name)
        _stub.get_output_directory = lambda: str(Path.cwd())  # noqa: E731
        sys.modules[_stub_name] = _stub

from nodes import batch_humo_render as bhumo


_CANARY = "zebra_lantern_atmosphere_731"

_HUMO_PATH = Path(__file__).resolve().parent.parent / "nodes" / "batch_humo_render.py"


# ---------------------------------------------------------------------------
# Source-level: helper is used
# ---------------------------------------------------------------------------


def test_humo_uses_lighting_helper():
    src = _HUMO_PATH.read_text(encoding="utf-8")
    assert "get_story_brief_lighting" in src, (
        "nodes/batch_humo_render.py does not reference "
        "get_story_brief_lighting; C5f wiring missing"
    )


def test_humo_logs_story_brief_status():
    """E-07: ledger-load log surfaces story_brief_status."""
    src = _HUMO_PATH.read_text(encoding="utf-8")
    assert "story_brief_status=" in src


# ---------------------------------------------------------------------------
# Behavioral -- _build_pos_prompt
# ---------------------------------------------------------------------------


_CAST = [
    {"name": "Jones", "character_description": "tall detective, weathered face"},
    {"name": "Smith", "character_description": "quiet suspect, sweating"},
]


def _meta_ok(*, atmosphere=None, lighting=None) -> dict:
    return {
        "story_brief":        "the brief prose",
        "story_brief_status": "ok",
        "story_brief_terms": {
            "setting":    ["steel table"],     # excluded by helper
            "lighting":   list(lighting or ["swinging bulb"]),
            "atmosphere": list(atmosphere or ["tense", "smoky"]),
        },
    }


def test_humo_prompt_preserves_default_pos_suffix():
    """The legacy _DEFAULT_POS_SUFFIX content must still appear in the
    composed prompt; the brief insertion happens BEFORE it."""
    meta = _meta_ok()
    prompt = bhumo._build_pos_prompt("Jones", {}, _CAST, meta=meta)
    # Spot-check a couple of words from _DEFAULT_POS_SUFFIX.
    suffix = bhumo._DEFAULT_POS_SUFFIX
    # Whichever first short token appears in suffix should also appear in prompt.
    for token in suffix.split(",")[:2]:
        token = token.strip()
        if token and len(token) >= 4:
            assert token in prompt, (
                f"suffix token {token!r} missing from composed HuMo prompt"
            )


def test_humo_prompt_includes_lighting_atmosphere_when_ok():
    meta = _meta_ok(lighting=["swinging bulb"], atmosphere=[_CANARY, "tense"])
    prompt = bhumo._build_pos_prompt("Jones", {}, _CAST, meta=meta)
    assert _CANARY in prompt, (
        "E-06 canary token missing from HuMo prompt; "
        "get_story_brief_lighting not wired through"
    )
    assert "swinging bulb" in prompt
    # Setting term EXCLUDED.
    assert "steel table" not in prompt


def test_humo_prompt_falls_through_when_meta_missing():
    """No meta supplied -> legacy prompt; no canary, no crash."""
    prompt = bhumo._build_pos_prompt("Jones", {}, _CAST, meta=None)
    assert "tall detective" in prompt
    assert _CANARY not in prompt
    # _DEFAULT_POS_SUFFIX still present.
    for token in bhumo._DEFAULT_POS_SUFFIX.split(",")[:2]:
        token = token.strip()
        if token and len(token) >= 4:
            assert token in prompt


def test_humo_prompt_falls_through_when_status_failed():
    """Failed brief -> legacy prompt; the helper returns ''."""
    meta = {
        "story_brief":        "",
        "story_brief_status": "failed",
        "story_brief_terms":  {"setting": [], "lighting": [], "atmosphere": []},
    }
    prompt = bhumo._build_pos_prompt("Jones", {}, _CAST, meta=meta)
    assert _CANARY not in prompt
    assert "tall detective" in prompt
