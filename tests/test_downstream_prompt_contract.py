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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Commit 5 strips '1940s setting' from script_critic.py:330 "
        "(rubric fallback line). xfail-strict marker fails the suite "
        "once the literal is gone, forcing this marker to be removed."
    ),
)
def test_script_critic_no_1940s_setting_in_rubric():
    text = _SCRIPT_CRITIC.read_text(encoding="utf-8")
    assert "1940s setting" not in text, (
        "script_critic.py still contains the '1940s setting' rubric "
        "line (line 330 of the audited snapshot)."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Commit 5 strips '1940s-style' from script_critic.py:339-340 "
        "(critic system prompt). xfail-strict forces marker removal "
        "once the fix lands."
    ),
)
def test_script_critic_no_1940s_style_in_system_prompt():
    text = _SCRIPT_CRITIC.read_text(encoding="utf-8")
    assert "1940s-style" not in text, (
        "script_critic.py system prompt still anchors to '1940s-style' "
        "(line 339 of the audited snapshot)."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Commit 5 strips '1940s' from script_critic.py:556 (revision "
        "prompt). xfail-strict forces marker removal once the fix "
        "lands."
    ),
)
def test_script_critic_no_1940s_in_revision_prompt():
    text = _SCRIPT_CRITIC.read_text(encoding="utf-8")
    # Match the full anchor phrase so this test only triggers on the
    # specific revision-prompt offender, not unrelated commentary that
    # mentions 1940s history.
    assert "You are revising a 1940s" not in text, (
        "script_critic.py revision prompt still contains "
        "'You are revising a 1940s ...' (line 556 of the audited "
        "snapshot)."
    )


# ---------------------------------------------------------------------------
# LTX style brief prompt anchoring -- xfail-strict until commit 5
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Commit 5 rewrites story_orchestrator._LTX_STYLE_BRIEF_PROMPT "
        "to drop the 'vintage-radio elements ... skinned for the "
        "setting' clause (line 3401). xfail-strict canary."
    ),
)
def test_story_orchestrator_no_vintage_radio_skinning():
    text = _STORY_ORCHESTRATOR.read_text(encoding="utf-8")
    assert "vintage-radio" not in text, (
        "story_orchestrator.py still contains 'vintage-radio' in the "
        "LTX style brief prompt (line 3401 of the audited snapshot)."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Commit 5 rewrites _LTX_STYLE_BRIEF_PROMPT examples that all "
        "hardcode vacuum-tube + brass-speaker + radio-bay anchors "
        "(lines 3407-3409). xfail-strict canary."
    ),
)
def test_story_orchestrator_no_vacuum_tube_example_anchor():
    text = _STORY_ORCHESTRATOR.read_text(encoding="utf-8")
    # Loose case-insensitive search -- any "vacuum tube(s)" remaining
    # in this file means the baked-in examples haven't been replaced
    # with a style-spanning rotation.
    assert not re.search(r"vacuum tubes?", text, re.IGNORECASE), (
        "story_orchestrator.py still has 'vacuum tubes' baked into "
        "LTX style brief examples (lines 3407-3409 of the audited "
        "snapshot)."
    )


# ---------------------------------------------------------------------------
# Schema migration / graceful degrade -- xfail until commit 3
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Commit 3 adds meta.news graceful-degrade for old ledgers "
        "(ADR section 9.2). Test asserts a ledger without meta.news "
        "loads with a warning rather than failing -- not yet wired."
    ),
)
def test_old_ledger_without_meta_news_loads_with_warning():
    """ADR test plan case 12. Concrete fixture build lands with the
    graceful-degrade code in commit 3.
    """
    pytest.fail(
        "Graceful-degrade path for old ledgers is not yet implemented "
        "(commit 3 deliverable)."
    )


# ---------------------------------------------------------------------------
# Integration cases -- xfail until commit 4 wiring
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ADR test plan case 1: RADIO portrait fallback must hard-fail "
        "rather than invent 'vintage 1940s console radio' when the "
        "cast row has empty character_description. Commit 4 wiring."
    ),
)
def test_radio_portrait_empty_char_desc_hard_fails():
    pytest.fail(
        "Commit 4 RADIO portrait fallback policy not yet wired; the "
        "current code in scripts/render_flux_batch.py:266 still "
        "invents a hardcoded period radio when char_desc is empty."
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ADR test plan case 2: MusicGen with style 'rust-belt cyber-"
        "noir' must NOT default to '1940s old time radio'. Requires "
        "musicgen_theme to read ledger.meta.news + style. Commit 4."
    ),
)
def test_musicgen_does_not_default_to_period_cues():
    pytest.fail(
        "Commit 4 MusicGen style-aware cues not yet wired; the "
        "current code in nodes/musicgen_theme.py:52-74 still hardcodes "
        "'1940s old time radio' as the default opening/closing/"
        "interstitial cue."
    )
