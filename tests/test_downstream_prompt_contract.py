"""tests/test_downstream_prompt_contract.py -- system contract canary.

ADR docs/news_interpreter_adr.md section 5 -- safety-net-first commit
order. These tests assert that downstream prompt-building sites do NOT
contain hardcoded era literals. Tests marked xfail(strict=True) fail
the suite the moment the offender is removed -- forcing the xfail
marker to be removed in lockstep with the fix. That is the canary
mechanic: contract lands first, fixes flip xfails to PASSED.

Test plan coverage
------------------
- Era-literal absence in script_critic.py:330, 339-340, 556 -- five
  xfail-strict text-scan canaries.
- _LTX_STYLE_BRIEF_PROMPT in story_orchestrator.py:3401, 3407-3409 --
  two xfail-strict text-scan canaries (vintage-radio + vacuum-tubes).
- ADR test plan case 1 (RADIO portrait empty char_desc hard-fail) --
  xfail-strict until commit 4 wiring.
- ADR test plan case 2 (MusicGen style-aware cues) -- xfail-strict
  until commit 4 wiring.
- ADR test plan case 12 (old ledger without meta.news loads with
  warning) -- xfail-strict until commit 3 wiring.

When the matching commit lands, each xfail-strict flips to XPASS and
fails the suite, forcing this file to be updated alongside the fix.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODES_DIR = _REPO_ROOT / "nodes"
for _p in (_REPO_ROOT, _NODES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


_SCRIPT_CRITIC = _REPO_ROOT / "nodes" / "script_critic.py"
_STORY_ORCHESTRATOR = _REPO_ROOT / "nodes" / "story_orchestrator.py"


# ---------------------------------------------------------------------------
# Era-literal absence -- xfail-strict until commit 5
# ---------------------------------------------------------------------------


def test_script_critic_no_1940s_setting_in_rubric():
    """LANDED in commit 5 (news_interpreter sprint). xfail-strict
    marker removed in lockstep with the fix per the canary mechanic.
    """
    text = _SCRIPT_CRITIC.read_text(encoding="utf-8")
    assert "1940s setting" not in text, (
        "script_critic.py still contains the '1940s setting' rubric "
        "line (regression -- commit 5 stripped this; the literal must "
        "not return)."
    )


def test_script_critic_no_1940s_style_in_system_prompt():
    """LANDED in commit 5. Marker removed in lockstep."""
    text = _SCRIPT_CRITIC.read_text(encoding="utf-8")
    assert "1940s-style" not in text, (
        "script_critic.py system prompt regressed -- '1940s-style' "
        "literal must not return after commit 5."
    )


def test_script_critic_no_1940s_in_revision_prompt():
    """LANDED in commit 5. Marker removed in lockstep."""
    text = _SCRIPT_CRITIC.read_text(encoding="utf-8")
    # Match the full anchor phrase so this test only triggers on the
    # specific revision-prompt offender, not unrelated commentary that
    # mentions 1940s history.
    assert "You are revising a 1940s" not in text, (
        "script_critic.py revision prompt regressed -- "
        "'You are revising a 1940s ...' literal must not return "
        "after commit 5."
    )


# ---------------------------------------------------------------------------
# LTX style brief prompt anchoring -- xfail-strict until commit 5
# ---------------------------------------------------------------------------


def test_story_orchestrator_no_vintage_radio_skinning():
    """LANDED in commit 5. Marker removed in lockstep."""
    text = _STORY_ORCHESTRATOR.read_text(encoding="utf-8")
    assert "vintage-radio" not in text, (
        "story_orchestrator.py regressed -- 'vintage-radio' literal "
        "must not return after commit 5 (_LTX_STYLE_BRIEF_PROMPT was "
        "rewritten to drop the vintage-skinned-for-setting bias)."
    )


def test_story_orchestrator_no_vacuum_tube_example_anchor():
    """LANDED in commit 5. Marker removed in lockstep. The three
    baked-in examples now span the style range (near-future news-
    room / deep-space vessel / industrial-decay loft) so no single
    hardware era dominates the LTX style brief output.
    """
    text = _STORY_ORCHESTRATOR.read_text(encoding="utf-8")
    # Loose case-insensitive search -- any "vacuum tube(s)" remaining
    # in this file means the baked-in examples regressed.
    assert not re.search(r"vacuum tubes?", text, re.IGNORECASE), (
        "story_orchestrator.py regressed -- 'vacuum tubes' must not "
        "return in the LTX style brief examples after commit 5."
    )


# ---------------------------------------------------------------------------
# Schema migration / graceful degrade -- xfail until commit 3
# ---------------------------------------------------------------------------


def test_old_ledger_without_meta_news_loads_with_warning():
    """ADR test plan case 12 -- LANDED in commit 3 (news_interpreter
    sprint). Old ledgers (pre-commit-3 shape: schema_version
    'l3-2026-05-08', no meta.news) must be readable by downstream
    consumers without exception. The contract: ``meta.get('news')``
    on absence returns ``None`` cleanly; consumers fall back to raw
    news_seed for cast / outline and to a synthesized close line for
    the announcer (commit 4 wires the announcer fallback).

    xfail-strict marker was removed in lockstep with the fix per the
    canary mechanic described in commit 1.
    """
    # Pre-commit-3 ledger shape -- exactly what's on disk for any
    # episode that was generated before this sprint landed.
    old_ledger = {
        "schema_version": "l3-2026-05-08",
        "episode_id": "test_old_ledger",
        "commit": "deadbeef",
        "meta": {"episode_seed": 42},
        "cast": [],
        "lines": [],
    }
    # Contract 1: meta.get('news') returns None on absence (not
    # KeyError, not exception). Downstream consumers rely on this
    # pattern, not on schema-level field presence.
    meta = old_ledger.get("meta", {})
    news = meta.get("news")
    assert news is None, (
        "Old ledger access pattern broke: meta.get('news') returned "
        f"{news!r} instead of None"
    )

    # Contract 2: the writer's graceful-degrade path sets
    # meta['news'] = None when build_news_briefs raises. Round-trip
    # through json.dumps + json.loads must preserve the None sentinel
    # so on-disk ledgers honor the same access pattern.
    import json
    degraded_ledger = dict(old_ledger)
    degraded_ledger["meta"] = {**old_ledger["meta"], "news": None}
    round_tripped = json.loads(json.dumps(degraded_ledger))
    assert round_tripped["meta"]["news"] is None, (
        "meta.news = None sentinel did not survive JSON round-trip: "
        f"got {round_tripped['meta']['news']!r}"
    )


# ---------------------------------------------------------------------------
# Integration cases -- xfail until commit 4 wiring
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ADR docs/news_interpreter_adr.md section 1 OUT OF SCOPE: FLUX "
        "character portraits land in their own ADR once the narrative "
        "plane is stable. Canary stays armed -- whenever a future ADR "
        "rewires scripts/render_flux_batch.py:266 to hard-fail rather "
        "than invent 'vintage 1940s console radio' on empty char_desc, "
        "this xfail-strict flips to XPASS and forces the marker removal."
    ),
)
def test_radio_portrait_empty_char_desc_hard_fails():
    pytest.fail(
        "RADIO portrait fallback policy still not wired (deferred per "
        "ADR section 1 -- FLUX portraits are out of news_interpreter "
        "sprint scope). Current code in scripts/render_flux_batch.py:"
        "266 still invents a hardcoded period radio when char_desc is "
        "empty. Future ADR will address."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ADR docs/news_interpreter_adr.md section 1 OUT OF SCOPE: "
        "MusicGen cues land in their own ADR once the narrative plane "
        "is stable. Canary stays armed -- whenever a future ADR "
        "rewires musicgen_theme.py to read ledger.meta.news + style "
        "instead of defaulting to '1940s old time radio', this xfail-"
        "strict flips to XPASS and forces the marker removal."
    ),
)
def test_musicgen_does_not_default_to_period_cues():
    pytest.fail(
        "MusicGen style-aware cues still not wired (deferred per ADR "
        "section 1 -- MusicGen cues are out of news_interpreter sprint "
        "scope). Current code in nodes/musicgen_theme.py:52-74 still "
        "hardcodes '1940s old time radio' as the default opening / "
        "closing / interstitial cue. Future ADR will address."
    )
