"""Story-quality G1 GOLDEN fixtures (2026-06-28, kibitz option C).

Locks the G1 gate-decompression target. Each fixture row pairs a REAL shipped
FAILING line (``raw_text`` -- the over-correction collapsing rich beats into short
noun-salad) with a hand-authored ``corrected_text`` at the per-beat budget.

The test asserts, per row:
  RAW  -- documents the defect we are fixing (collapse below budget_lo, the worn
          cliche span, or an ALL-CAPS roster full name mid-line).
  CORRECTED -- the regression lock for G1: not truncated; NOT one-breath-flagged at
          the BUDGET cap (derive_one_breath_cap); WOULD be flagged at the legacy 28
          cap (so the budget cap is demonstrably what permits it); within budget.

Plus direct unit tests for the three new leaf helpers. Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nodes import _otr_line_hygiene as HY

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_story_quality.json"
_ROWS = json.loads(_FIXTURE.read_text(encoding="utf-8"))["rows"]
_WORD_RE = re.compile(r"[A-Za-z']+")
#: an ALL-CAPS multi-token run = a leaked roster FULL NAME (NASA/UCLA single tokens
#: are not full names, so require >= 2 caps tokens).
_ALLCAPS_FULLNAME_RE = re.compile(r"\b[A-Z][A-Z]+(?:\s+[A-Z][A-Z]+)+\b")


def _wc(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _ids():
    return [f"{r['episode_id'].replace('signal_lost_', '')}-{r['beat_id']}" for r in _ROWS]


def test_fixture_loads_six_rows():
    assert len(_ROWS) == 6
    for r in _ROWS:
        for k in ("raw_text", "corrected_text", "words_per_beat_range",
                  "expected_cliche_span", "raw_collapsed", "raw_roster_caps"):
            assert k in r, f"{r.get('beat_id')} missing {k}"


@pytest.mark.parametrize("row", _ROWS, ids=_ids())
def test_corrected_not_truncated(row):
    assert HY.is_truncated(row["corrected_text"]) is False


@pytest.mark.parametrize("row", _ROWS, ids=_ids())
def test_corrected_one_breath_false_at_budget_cap(row):
    """The whole point of G1: a budget-length corrected line is clean at the
    BUDGET-derived cap (no fragmentary reroll)."""
    cap = HY.derive_one_breath_cap(row["words_per_beat_range"])
    flagged, _ = HY.flag_one_breath(row["corrected_text"], max_words=cap)
    assert flagged is False, (
        f"corrected line one-breath-flagged at budget cap {cap} "
        f"(wc={_wc(row['corrected_text'])})")


@pytest.mark.parametrize("row", _ROWS, ids=_ids())
def test_corrected_would_flag_at_legacy_cap(row):
    """Regression proof: at the legacy hard 28-word cap the SAME corrected line IS
    flagged -- so a silent revert of the cap to 28 fails this test, not just the
    real render."""
    flagged, _ = HY.flag_one_breath(row["corrected_text"], max_words=28)
    assert flagged is True, (
        "corrected line not flagged at legacy 28 cap -- the budget-cap proof is "
        f"vacuous (wc={_wc(row['corrected_text'])})")


@pytest.mark.parametrize("row", _ROWS, ids=_ids())
def test_corrected_within_budget(row):
    lo, _hi = row["words_per_beat_range"]
    budget_hi = HY.derive_one_breath_cap(row["words_per_beat_range"])
    cwc = _wc(row["corrected_text"])
    assert lo <= cwc <= budget_hi, f"corrected wc {cwc} not in [{lo},{budget_hi}]"


@pytest.mark.parametrize("row", _ROWS, ids=_ids())
def test_corrected_has_no_cliche(row):
    assert HY.find_cliche_phrase(row["corrected_text"]) == ""


@pytest.mark.parametrize("row", _ROWS, ids=_ids())
def test_raw_defect_present(row):
    """Each raw line carries at least one documented defect: a collapse below
    budget_lo, a worn cliche, or an ALL-CAPS roster full name."""
    raw = row["raw_text"]
    lo = row["words_per_beat_range"][0]
    defects = []
    if row["raw_collapsed"]:
        assert _wc(raw) < lo, f"raw wc {_wc(raw)} not < budget_lo {lo}"
        defects.append("collapsed")
    if row["expected_cliche_span"]:
        assert HY.find_cliche_phrase(raw) == row["expected_cliche_span"]
        defects.append("cliche")
    if row["raw_roster_caps"]:
        assert _ALLCAPS_FULLNAME_RE.search(raw), "expected ALL-CAPS roster full name"
        defects.append("roster_caps")
    assert defects, f"{row['beat_id']}: no defect flagged on a golden FAILING line"


# --- direct leaf-helper unit tests -----------------------------------------

@pytest.mark.parametrize("rng,exp", [
    ((0, 0), 28),
    ([0, 0], 28),
    ([10, 50], 50),
    ([28, 50], 50),
    ([28, 80], 60),    # clamp to 60
    ([22, 40], 40),
    ([18, 18], 28),    # hi below 28 -> legacy 28
    (None, 28),
    ([5], 28),         # too few elements
    ("x", 28),         # not a list/tuple
    (["a", "b"], 28),  # non-numeric
    ((20, 35), 35),
])
def test_derive_one_breath_cap(rng, exp):
    assert HY.derive_one_breath_cap(rng) == exp


@pytest.mark.parametrize("text,exp", [
    ("", 0),
    ("plain spoken line here", 0),
    ("a, b; c: d", 3),               # , ; :
    ("run and hide but wait", 2),    # and, but
    ("so, then we go", 2),           # 'so' is FANBOYS (+1), 'then' is NOT, plus 1 comma -> 2
    ("nor yet for or", 4),           # all FANBOYS
])
def test_hard_clauses(text, exp):
    assert HY._hard_clauses(text) == exp


@pytest.mark.parametrize("text,exp", [
    ("Over my dead body, Lemmy.", "Over my dead body"),
    ("Not on my watch, Pim.", "Not on my watch"),
    ("You're playing with fire here.", "You're playing with fire"),
    ("A perfectly clean spoken line.", ""),
    ("", ""),
])
def test_find_cliche_phrase(text, exp):
    assert HY.find_cliche_phrase(text) == exp
