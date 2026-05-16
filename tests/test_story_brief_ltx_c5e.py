"""Sprint C C5e: LTX motion wiring tests.

Per refinement section 6.1: 220-240 char total budget; motion-first;
drop brief if it pushes motion past char 140; 80-100 char brief
fragment.

Structural proxy ONLY per R-05: these tests prove the prompt body
is well-formed and threads meta.story_brief through correctly. They
do NOT prove LTX actually renders better motion -- empirical motion-
fidelity verification is Sprint A scope per ROADMAP.

E-06 meta-threading canary: `zebra_lantern_atmosphere_731` token
makes regressions visible even when the helper itself misbehaves.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest


# `nodes/batch_ltx_render.py` may import ComfyUI runtime modules
# (folder_paths, comfy.*) at module level. Stub the ones that are
# safe to no-op for prompt-construction tests.
for _stub_name in ("folder_paths",):
    if _stub_name not in sys.modules:
        _stub = types.ModuleType(_stub_name)
        _stub.get_output_directory = lambda: str(Path.cwd())  # noqa: E731
        sys.modules[_stub_name] = _stub

from nodes import batch_ltx_render as bltx


_CANARY = "zebra_lantern_atmosphere_731"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ledger_with_brief(*, brief: str | None, atmosphere=None) -> dict:
    meta: dict = {}
    if brief is not None:
        meta["story_brief"] = brief
        meta["story_brief_status"] = "ok" if brief else "absent"
        meta["story_brief_terms"] = {
            "setting":    [],
            "lighting":   [],
            "atmosphere": list(atmosphere or []),
        }
    return {"meta": meta, "lines": [], "cast": []}


def _empty_line() -> dict:
    return {"speaker_role": "announcer", "text": ""}


# ---------------------------------------------------------------------------
# Source-level: helper is used
# ---------------------------------------------------------------------------


_LTX_PATH = Path(__file__).resolve().parent.parent / "nodes" / "batch_ltx_render.py"


def test_ltx_uses_get_story_brief_ltx_helper():
    src = _LTX_PATH.read_text(encoding="utf-8")
    assert "get_story_brief_ltx" in src, (
        "nodes/batch_ltx_render.py does not reference get_story_brief_ltx; "
        "C5e wiring missing"
    )


def test_ltx_logs_story_brief_status():
    """E-07: render log surfaces story_brief_status."""
    src = _LTX_PATH.read_text(encoding="utf-8")
    assert "story_brief_status=" in src


# ---------------------------------------------------------------------------
# Behavioral -- prompt structure
# ---------------------------------------------------------------------------


def test_ltx_prompt_total_under_240_when_brief_present():
    """Refinement section 6.1: total LTX prompt <= 240 chars."""
    led = _ledger_with_brief(
        brief="a tight detective room with rain on tin and a swinging bulb",
    )
    for role in ("announcer", "music_open", "music_close", "music_inter", "sfx"):
        prompt = bltx._build_ltx_role_prompt(role, _empty_line(), led)
        assert len(prompt) <= 240, (
            f"role={role} prompt is {len(prompt)} chars; cap is 240"
        )


def test_ltx_motion_verb_before_char_140():
    """Refinement section 6.1 motion-first: the first motion verb
    must appear at char index <= 140 in every role prompt."""
    led = _ledger_with_brief(
        brief="a tight detective room with rain on tin and a swinging bulb",
    )
    for role in ("announcer", "music_open", "music_close", "music_inter", "sfx"):
        prompt = bltx._build_ltx_role_prompt(role, _empty_line(), led)
        first_motion = bltx._first_motion_verb_index(prompt)
        assert first_motion <= 140, (
            f"role={role}: first motion verb at char {first_motion}, "
            f"ceiling is 140. Prompt: {prompt[:200]}"
        )


def test_ltx_brief_dropped_when_combined_exceeds_240():
    """Per spec: brief is dropped if total prompt exceeds 240 chars."""
    long_brief = (
        "an extraordinarily long brief that combined with the role "
        "template would exceed the 240 char budget, forcing the "
        "drop-fallback to fire and the role template alone to ship"
    )
    led = _ledger_with_brief(brief=long_brief)
    role = "announcer"
    template_only = bltx._PROMPT_BY_ROLE[role]
    prompt = bltx._build_ltx_role_prompt(role, _empty_line(), led)
    # The brief should NOT appear in the output (dropped).
    # The prompt should equal the legacy template.
    assert prompt == template_only, (
        f"long-brief drop did not fire; got: {prompt[:200]}"
    )


def test_ltx_brief_fragment_when_in_range():
    """Behavioral: a short canary brief should reach the prompt
    (canary token visible). Locks the helper-to-callsite wiring."""
    led = _ledger_with_brief(brief=f"a tight room with {_CANARY}")
    prompt = bltx._build_ltx_role_prompt("announcer", _empty_line(), led)
    assert _CANARY in prompt, (
        "E-06 canary token missing from LTX prompt; helper not wired"
    )


def test_ltx_falls_through_when_brief_absent():
    """No brief -> legacy template returned verbatim."""
    led = _ledger_with_brief(brief=None)
    role = "music_open"
    prompt = bltx._build_ltx_role_prompt(role, _empty_line(), led)
    assert prompt == bltx._PROMPT_BY_ROLE[role]
    assert _CANARY not in prompt


def test_ltx_logs_status_failed(caplog):
    """E-07: failed-status surfaces in render log."""
    led = {
        "meta": {
            "story_brief":        "",
            "story_brief_status": "failed",
            "story_brief_terms":  {"setting": [], "lighting": [], "atmosphere": []},
        },
        "lines": [], "cast": [],
    }
    with caplog.at_level(logging.INFO, logger="OTR"):
        bltx._build_ltx_role_prompt("announcer", _empty_line(), led)
    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "story_brief_status=failed" in log_text
