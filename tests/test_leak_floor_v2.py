"""tests/test_leak_floor_v2.py -- leak-floor-v2 acceptance (2026-06-25).

Build spec: docs/2026-06-25-leaking-words/roundtable/pass02_plan.md.

ONE mandatory deterministic offline verifier over four named leak classes, plus
the news-bleed fix at build_allowed_roster. Acceptance contract:

  POSITIVE fixtures = the four real shipped leaks -> each MUST be caught:
    1. `Gasping, "..."`              -> rule 1 capitalised-participle extract
    2. `President Trump's orders...` -> rule 4 news-bleed (roster gate + echo)
    3. `YUKI MARTIN, no!`            -> rule 2 ALL-CAPS full-name vocative drop
    4. unclosed internal quote       -> rule 3 malformed-quote -> recompose
  NEGATIVE fixtures = 0 false-positive:
    a. first-name vocative `YUKI!`         -> untouched (full names only)
    b. legit in-world org `NASA confirmed`  -> stays allowed, not banned
    c. non-stage `-ing` open `Running to...` -> untouched (no leading quote)

Require 0 leak survive + 0 false-positive. Pure-Python; no GPU; no LLM.

GROUNDED root cause: the `Gasping,` leak is _leading_stage_strip's lowercase-
start guard (hygiene line 271), NOT the verb whitelist -- so rule 1 is a NEW
sibling and _NARRATION_VERBS is NOT widened (asserted below).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when pytest is invoked from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_hygiene import (  # noqa: E402
    EntityPolicy,
    has_malformed_internal_quote,
    scrub_roster_vocative,
    strip_participle_before_quote,
    verify_and_repair_line,
)
from nodes._otr_line_composer import (  # noqa: E402
    build_allowed_roster,
    build_banned_source_proper_nouns,
    detect_phantom_names,
)


# ---------------------------------------------------------------------------
# POSITIVE 1 -- capitalised participle before a quote (rule 1)
# ---------------------------------------------------------------------------
def test_pos1_participle_before_quote_extracts_dialogue():
    line = 'Gasping, "I can\'t believe it."'
    vr = verify_and_repair_line(line, policy=EntityPolicy())
    assert vr.changed is True
    assert vr.text == "I can't believe it."
    assert [d.reason_code for d in vr.defects] == [
        "capitalized_participle_before_quote"
    ]
    assert vr.needs_recompose is False
    assert "leak_floor:participle_before_quote" in vr.compose_flags


def test_pos1_pure_rule_is_idempotent():
    once, changed = strip_participle_before_quote('Frowning, "Stay back."')
    assert changed is True and once == "Stay back."
    twice, changed2 = strip_participle_before_quote(once)
    assert changed2 is False and twice == once


# ---------------------------------------------------------------------------
# POSITIVE 2 -- news-bleed real-person / political figure (rule 4)
# ---------------------------------------------------------------------------
def test_pos2_news_bleed_banned_at_roster_and_echoed_by_verifier():
    line = "President Trump's orders came down at dawn."
    banned = build_banned_source_proper_nouns(
        terms=["President Trump", "NASA"], raw_text=line,
    )
    # "President Trump" is classified real-person; NASA is NOT.
    assert "TRUMP" in banned
    assert "NASA" not in banned

    roster = build_allowed_roster(
        cast_rows=[{"name": "DESMOND CHEN"}],
        key_terms=["NASA"],
        banned_terms=banned,
    )
    # EntityPolicy invariant: banned & allowed is empty.
    banned_u = frozenset(b.upper() for b in banned)
    assert banned_u & roster == frozenset()
    assert "NASA" in roster and "TRUMP" not in roster

    # The EXISTING phantom/roster gate now REJECTS the name (it is no longer
    # allowlisted) -> flagged for downstream repair.
    phantoms = detect_phantom_names(line, "DESMOND CHEN", roster)
    assert any("Trump" in p for p in phantoms)

    # The verifier ECHOES it: a surviving banned entity -> recompose.
    policy = EntityPolicy(allowed=roster, banned=banned_u)
    vr = verify_and_repair_line(line, policy=policy)
    assert vr.needs_recompose is True
    assert "banned_source_entity" in [d.reason_code for d in vr.defects]


def test_pos2_honorific_variants_classified():
    banned = build_banned_source_proper_nouns(
        terms=["Senator Warren", "Governor Newsom", "Dr. Fauci"],
    )
    # the honorific phrase + the surname are both banned
    assert "TRUMP" not in banned  # control: not in input
    assert any("WARREN" in b for b in banned)
    assert any("NEWSOM" in b for b in banned)
    assert any("FAUCI" in b for b in banned)


# ---------------------------------------------------------------------------
# POSITIVE 3 -- ALL-CAPS full-name roster vocative (rule 2)
# ---------------------------------------------------------------------------
def test_pos3_roster_vocative_leading_drop():
    policy = EntityPolicy(allowed=frozenset({"YUKI MARTIN", "DESMOND CHEN"}))
    vr = verify_and_repair_line("YUKI MARTIN, no!", policy=policy)
    assert vr.changed is True
    assert vr.text == "no!"
    assert "roster_vocative" in [d.reason_code for d in vr.defects]


def test_pos3_roster_vocative_trailing_keeps_terminal():
    out = scrub_roster_vocative("Get out, YUKI MARTIN!", ["YUKI MARTIN"])
    assert out == "Get out!"


# ---------------------------------------------------------------------------
# POSITIVE 4 -- malformed internal (unclosed) double quote (rule 3)
# ---------------------------------------------------------------------------
def test_pos4_malformed_internal_quote_requests_recompose():
    line = 'He shouted, "We have to go now'  # one internal, unclosed "
    assert has_malformed_internal_quote(line) is True
    vr = verify_and_repair_line(line, policy=EntityPolicy())
    assert vr.needs_recompose is True
    assert "malformed_quote" in [d.reason_code for d in vr.defects]


def test_pos4_edge_wrapper_is_not_malformed():
    # a single edge wrapper quote is NOT an internal malformation
    assert has_malformed_internal_quote('"We have to go now') is False
    assert has_malformed_internal_quote('We have to go now"') is False
    # a balanced pair is fine
    assert has_malformed_internal_quote('She said "go" and left.') is False


# ---------------------------------------------------------------------------
# NEGATIVE a -- first-name vocative (NOT a full name) -> untouched
# ---------------------------------------------------------------------------
def test_neg_a_first_name_vocative_untouched():
    policy = EntityPolicy(allowed=frozenset({"YUKI MARTIN"}))
    vr = verify_and_repair_line("YUKI!", policy=policy)
    assert vr.changed is False
    assert vr.text == "YUKI!"
    assert vr.defects == ()
    assert vr.needs_recompose is False


# ---------------------------------------------------------------------------
# NEGATIVE b -- legit in-world org -> stays allowed, never banned
# ---------------------------------------------------------------------------
def test_neg_b_legit_org_not_banned_not_flagged():
    line = "NASA confirmed the launch window."
    banned = build_banned_source_proper_nouns(
        terms=["NASA", "CERN", "Voyager"], raw_text=line,
    )
    assert banned == frozenset()
    roster = build_allowed_roster(
        cast_rows=[{"name": "DESMOND CHEN"}],
        key_terms=["NASA", "CERN"],
        banned_terms=banned,
    )
    assert "NASA" in roster
    assert detect_phantom_names(line, "DESMOND CHEN", roster) == []
    policy = EntityPolicy(allowed=roster, banned=frozenset())
    vr = verify_and_repair_line(line, policy=policy)
    assert vr.changed is False
    assert vr.defects == ()
    assert vr.needs_recompose is False


# ---------------------------------------------------------------------------
# NEGATIVE c -- a non-stage `-ing` opening with NO leading quote -> untouched
# ---------------------------------------------------------------------------
def test_neg_c_non_stage_ing_opening_untouched():
    line = "Running to the door, I shouted for help."
    out, changed = strip_participle_before_quote(line)
    assert changed is False and out == line
    vr = verify_and_repair_line(line, policy=EntityPolicy())
    assert vr.changed is False
    assert vr.defects == ()


def test_neg_c_quote_led_line_untouched():
    # `"Running," she said` starts WITH the quote -> outside-segment-0 empty
    line = '"Running," she said, "we have to move."'
    out, changed = strip_participle_before_quote(line)
    assert changed is False and out == line


# ---------------------------------------------------------------------------
# Contract: 0 leak survive + 0 false-positive (summary)
# ---------------------------------------------------------------------------
def _leak_detected(vr) -> bool:
    return bool(vr.defects) or vr.needs_recompose or vr.changed


def test_zero_leak_zero_false_positive_summary():
    pol_cast = EntityPolicy(
        allowed=frozenset({"YUKI MARTIN", "DESMOND CHEN"}),
        banned=frozenset({"TRUMP", "PRESIDENT TRUMP"}),
    )
    positives = [
        'Gasping, "I can\'t believe it."',
        "President Trump's orders came down at dawn.",
        "YUKI MARTIN, no!",
        'He shouted, "We have to go now',
    ]
    for line in positives:
        vr = verify_and_repair_line(line, policy=pol_cast)
        assert _leak_detected(vr), f"leak survived: {line!r}"

    pol_clean = EntityPolicy(
        allowed=frozenset({"YUKI MARTIN", "NASA", "CERN"}),
        banned=frozenset(),
    )
    negatives = [
        "YUKI!",
        "NASA confirmed the launch window.",
        "Running to the door, I shouted for help.",
        "I can't believe it.",  # already-clean dialogue with an apostrophe
    ]
    for line in negatives:
        vr = verify_and_repair_line(line, policy=pol_clean)
        assert not _leak_detected(vr), f"false positive: {line!r}"


# ---------------------------------------------------------------------------
# Off-path byte-identity + grounded-root-cause guards
# ---------------------------------------------------------------------------
def test_build_allowed_roster_banned_default_is_noop():
    a = build_allowed_roster(cast_rows=[{"name": "A B"}], key_terms=["NASA"])
    b = build_allowed_roster(
        cast_rows=[{"name": "A B"}], key_terms=["NASA"], banned_terms=(),
    )
    assert a == b


def test_narration_verbs_not_widened():
    # the GROUNDED fix does NOT widen the verb whitelist to catch the leak --
    # "gasping" (the real `Gasping,` leak token) and "weeping" stay absent;
    # rule 1 is structural (participle + required quote), not verb-based.
    from nodes import _otr_line_hygiene as hy

    assert "gasping" not in hy._NARRATION_VERBS
    assert "weeping" not in hy._NARRATION_VERBS


def test_leading_stage_strip_lowercase_guard_intact():
    # line 271 guard: a capitalised lead is returned UNCHANGED by the existing
    # function (this is WHY `Gasping,` leaked, and why rule 1 is a new sibling).
    from nodes._otr_line_hygiene import _leading_stage_strip

    assert _leading_stage_strip("Gasping, the lever.", 10) == "Gasping, the lever."
