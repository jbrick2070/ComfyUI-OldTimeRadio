"""Story-quality C4 (S3) -- body-gate text-scored accept locks.

Pins the v2 body gate's total-order defect score (``_otr_body_score``) and the
locked-cast roster-caps signals:

  * the per-feature weights (10*grounding + 3*hard_leak + 2*trunc + 2*run_on +
    1*roster_caps) and lower-wins / ORIGINAL-on-tie ordering;
  * ``run_on`` shares C2's ``derive_one_breath_cap`` + relaxed
    ``max_clause_markers = max(3, cap // 8)`` (first-pass == reroll == scan ==
    body-gate), so a budget-length multi-clause line C2 allowed is not counted
    against a reroll here;
  * roster-caps matches the episode's LOCKED cast FULL names only (NASA/UCLA-
    safe), case-sensitive on the ALL-CAPS shout;
  * the MID-CLAUSE roster-caps reroll TRIGGER -- a leading/trailing vocative is
    scrubbed by ``scrub_roster_vocative`` and does NOT trip it; only a mid-clause
    hit rerolls.

The v2-OFF byte-identity of the gate itself is covered by the full-suite golden +
``test_audio_byte_identical``, not here. Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

from types import SimpleNamespace

from nodes.OTR_LedgerScriptWriter import (
    _otr_allcaps_cast_hits,
    _otr_body_score,
    _otr_cast_fullnames,
    _otr_roster_caps_midclause,
)

_ROSTER = frozenset({"CLARISSE GORDON", "YUKI MARTIN", "NASA"})


def _req(words_per_beat_range=(0, 0), roster=frozenset()):
    """Minimal stand-in for the LineRequest fields the helpers read."""
    return SimpleNamespace(
        words_per_beat_range=words_per_beat_range, allowed_roster=roster)


# --- roster helpers -----------------------------------------------------------

def test_cast_fullnames_full_names_only():
    names = _otr_cast_fullnames(_req(roster=_ROSTER))
    assert set(names) == {"CLARISSE GORDON", "YUKI MARTIN"}
    assert "NASA" not in names  # a single ALL-CAPS token is NASA/UCLA-safe


def test_allcaps_cast_hits_case_sensitive():
    fns = ("CLARISSE GORDON",)
    assert _otr_allcaps_cast_hits("I trust CLARISSE GORDON.", fns) == 1
    assert _otr_allcaps_cast_hits("I trust Clarisse Gordon.", fns) == 0
    assert _otr_allcaps_cast_hits("Call NASA right now.", ("NASA",)) == 0


def test_roster_caps_midclause_only_mid():
    fns = ("CLARISSE GORDON",)
    # leading + trailing vocatives are scrub-safe -> NOT a reroll trigger
    assert _otr_roster_caps_midclause("CLARISSE GORDON, stop!", fns) is False
    assert _otr_roster_caps_midclause("Stop it, CLARISSE GORDON!", fns) is False
    # mid-clause subject/object -> would mangle on strip -> reroll
    assert _otr_roster_caps_midclause("I think CLARISSE GORDON lied.", fns) is True


# --- the total-order score ----------------------------------------------------

def test_clean_line_scores_zero():
    assert _otr_body_score(
        "The cat sat on the warm rug.", {}, frozenset(), None, _req((50, 60))
    ) == 0


def test_roster_caps_adds_one():
    req = _req((50, 60), roster=frozenset({"CLARISSE GORDON"}))
    shout = _otr_body_score("I trust CLARISSE GORDON.", {}, frozenset(), None, req)
    plain = _otr_body_score("I trust Clarisse Gordon.", {}, frozenset(), None, req)
    assert shout - plain == 1


def test_truncation_adds_two():
    req = _req((50, 60))
    cut = _otr_body_score("We need the reactor and", {}, frozenset(), None, req)
    whole = _otr_body_score("We need the reactor.", {}, frozenset(), None, req)
    assert cut - whole == 2


def test_hard_leak_adds_three():
    req = _req((50, 60))
    leak = _otr_body_score('He said "wait.', {}, frozenset(), None, req)
    clean = _otr_body_score("He said wait.", {}, frozenset(), None, req)
    assert leak - clean == 3


def test_run_on_uses_c2_cap():
    # ~39-word, multi-clause budget-length line: flagged at the legacy (0,0)->28
    # cap (word count alone) but NOT at a (50,60)->60 cap with the relaxed clause
    # threshold -- the exact C2 one-breath contract. Only run_on differs, so the
    # score delta is exactly its weight (2).
    line = (
        "We moved the cargo, checked the seals, logged the readings, "
        "and waited by the bay while the slow tide of the long gray "
        "morning rolled on past the harbor and the quiet boats at rest "
        "near the pier today.")
    legacy = _otr_body_score(line, {}, frozenset(), None, _req((0, 0)))
    budget = _otr_body_score(line, {}, frozenset(), None, _req((50, 60)))
    assert legacy - budget == 2


def test_original_wins_on_tie():
    # Two equally-clean lines score the same; the gate's strict "<" then keeps
    # the ORIGINAL (a reroll must be STRICTLY cleaner to win).
    a = _otr_body_score(
        "The hull held against the cold.", {}, frozenset(), None, _req((50, 60)))
    b = _otr_body_score(
        "The crew slept through the night.", {}, frozenset(), None, _req((50, 60)))
    assert a == b == 0
    assert not (b < a)
