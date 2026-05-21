"""Sprint C C5c: FLUX env + radio bookend wiring tests.

File-path retargeted to `visual/batch_flux_render.py` per C1 audit
(E-24 / RR-B8 contingency activated -- batch_flux_env_render.py does
not exist at HEAD).

Both wirings use `get_story_brief_full(meta)`:

  * `_parse_env_prompts`: the brief LEADS the composed env prompt
    (BUG-LOCAL-250 follow-up); the env description and the generic
    style_suffix tail follow.
  * `_build_dynamic_radio_prompt`: brief is the PRIMARY radio
    descriptor (BUG-LOCAL-249). The upstream style preset is not
    read; an absent/failed brief falls through to the episode_id
    slug tier.

Behavioral tests use a unique sentinel token `zebra_lantern_atmosphere_731`
(per E-06 meta-threading canary) so a regression that bypasses the
helper would be caught even if the helper itself misbehaves silently.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from visual import batch_flux_render as bfr


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_BFR_PATH = _REPO_ROOT / "visual" / "batch_flux_render.py"


_CANARY = "zebra_lantern_atmosphere_731"


# ---------------------------------------------------------------------------
# Source-level: helper is imported
# ---------------------------------------------------------------------------


def test_flux_env_uses_get_story_brief_full():
    src = _BFR_PATH.read_text(encoding="utf-8")
    assert "get_story_brief_full" in src, (
        "visual/batch_flux_render.py does not reference "
        "get_story_brief_full; C5c wiring missing"
    )


def test_flux_env_uses_get_story_brief_status():
    """E-07 observability: render log records story_brief_status."""
    src = _BFR_PATH.read_text(encoding="utf-8")
    assert "get_story_brief_status" in src
    assert "story_brief_status=" in src, (
        "render log does not surface story_brief_status; E-07 violation"
    )


# ---------------------------------------------------------------------------
# Behavioral -- _parse_env_prompts
# ---------------------------------------------------------------------------


def _payload_with_env_tokens(brief: str | None) -> str:
    meta = {}
    if brief is not None:
        meta = {
            "story_brief":        brief,
            "story_brief_status": "ok" if brief else "absent",
            "story_brief_terms":  {"setting": [], "lighting": [], "atmosphere": []},
        }
    payload = {
        "tokens": [
            {"type": "environment", "description": "a dim hallway"},
            {"type": "dialogue",    "content": "[VOICE: JONES] line"},
        ],
        "meta": meta,
    }
    return json.dumps(payload)


def test_flux_env_prompt_includes_brief_when_status_ok():
    script_json = _payload_with_env_tokens(
        brief=f"a tight room with {_CANARY} and rain on tin"
    )
    prompts = bfr._parse_env_prompts(
        script_json, batch_limit=4,
        fallback="fallback prompt", style_suffix="35mm grain",
    )
    assert prompts, "no env prompts emitted"
    assert _CANARY in prompts[0], (
        "E-06 meta-threading canary missing from env prompt; "
        "get_story_brief_full not wired through"
    )


def test_flux_env_prompt_falls_through_when_brief_absent():
    """Empty brief: prompt builds without crashing; legacy desc only."""
    script_json = _payload_with_env_tokens(brief="")
    prompts = bfr._parse_env_prompts(
        script_json, batch_limit=4,
        fallback="fallback prompt", style_suffix="35mm grain",
    )
    assert prompts
    # First env token's description survives; no canary expected since
    # brief is empty.
    assert "a dim hallway" in prompts[0]
    assert _CANARY not in prompts[0]


def test_flux_env_prompt_logs_status_failed(caplog):
    """E-07: failed-status surfaces in render log."""
    payload = {
        "tokens": [{"type": "environment", "description": "a dim hallway"}],
        "meta": {
            "story_brief":        "",
            "story_brief_status": "failed",
            "story_brief_terms":  {"setting": [], "lighting": [], "atmosphere": []},
        },
    }
    with caplog.at_level(logging.INFO, logger="OTR"):
        bfr._parse_env_prompts(
            json.dumps(payload), batch_limit=4,
            fallback="fallback prompt", style_suffix="35mm grain",
        )
    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "story_brief_status=failed" in log_text, (
        "render log does not contain 'story_brief_status=failed' on "
        "the failed path; E-07 observability missing"
    )


# ---------------------------------------------------------------------------
# Behavioral -- _build_dynamic_radio_prompt
# ---------------------------------------------------------------------------


def _ledger_with_brief(brief: str | None, *, style: str = "") -> dict:
    meta: dict = {
        "gen_params_initial": {"style": style},
    }
    if brief is not None:
        meta["story_brief"] = brief
        meta["story_brief_status"] = "ok" if brief else "absent"
        meta["story_brief_terms"] = {
            "setting": [], "lighting": [], "atmosphere": [],
        }
    return {"meta": meta, "scenes": [], "episode_id": "ep_test"}


def test_flux_radio_bookend_uses_brief():
    """BUG-LOCAL-249: meta.story_brief is the primary radio-prompt
    descriptor."""
    led = _ledger_with_brief(
        brief=f"a tight detective room with {_CANARY} and rain",
        style="",
    )
    prompt = bfr._build_dynamic_radio_prompt(led)
    assert _CANARY in prompt, (
        f"radio bookend missing canary token; got: {prompt[:200]}"
    )


def test_flux_radio_bookend_brief_wins_over_style():
    """BUG-LOCAL-249: the radio prompt is downstream of story
    creation, so meta.story_brief drives it. The upstream style
    preset is NOT read -- even when the user set one, the brief
    wins and the style string never reaches the radio prompt."""
    led = _ledger_with_brief(
        brief=f"a tight detective room with {_CANARY}",
        style="noir mystery",
    )
    prompt = bfr._build_dynamic_radio_prompt(led)
    assert _CANARY in prompt, (
        f"brief should drive the radio prompt; got: {prompt[:200]}"
    )
    assert "noir mystery" not in prompt, (
        "upstream style preset must NOT appear in the radio prompt"
    )


def test_flux_radio_bookend_falls_through_on_absent_brief():
    """When the brief is absent (legacy ledger), the chain falls
    through to Tier 5+ (episode_id slug) without crashing."""
    led = _ledger_with_brief(brief=None, style="")
    prompt = bfr._build_dynamic_radio_prompt(led)
    # episode_id_slug branch -> "ep test radio broadcast unit, ..."
    assert "radio broadcast unit" in prompt


def test_flux_radio_bookend_logs_story_brief_status(caplog):
    """E-07: render log surfaces story_brief_status."""
    led = _ledger_with_brief(brief="", style="")
    # brief="" sets status="absent" via _ledger_with_brief.
    with caplog.at_level(logging.INFO, logger="OTR"):
        bfr._build_dynamic_radio_prompt(led)
    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "story_brief_status=" in log_text, (
        "render log does not contain 'story_brief_status='; "
        "E-07 observability missing"
    )
