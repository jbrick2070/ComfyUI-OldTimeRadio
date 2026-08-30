"""The duration receipt must not print OK on an episode that blew its budget.

PBUG-20260830-01, found by the 4060 on live 8 GB hardware. Two lines from the
same node, in the same run, seconds apart:

    [OTR_MasterAudioMux] DURATION MISMATCH (publishing anyway): ... over the
      credits-tail budget (25.2110s [declared] + 0.1200s tol) by 8.8608s
    [OTR_MasterAudioMux] duration_check v=155.120s a=120.928s
      tail_budget=25.2s (declared) OK

The gate failed by 8.86 s and the receipt said OK.

**The mechanism was a mis-bound `else`.** The budget check only ever *logged*;
it recorded no state. The receipt's `else` therefore bound to the PROBE check
immediately above it, so a successful ffprobe printed OK unconditionally and the
only branch that could print anything else was a *failed probe*. The over-budget
case had no branch at all.

**These tests CALL the function.** The first version asserted on the AST -- branch
shape and string presence -- because reaching the real branch needs ffprobe, two
rendered media files and a full mux. It was mutation-verified and it did detect
the defect, but the 4060 named the ceiling correctly: a source-shape assertion
"proves the code is written correctly, not that it runs correctly", cannot catch
a behavioural regression that preserves the shape, and goes red on a refactor
that preserves the behaviour. So the decision was extracted into
``duration_receipt_line`` -- pure, numbers in, one line out -- and is now
exercised with real values from both machines instead of read as source.
"""
from __future__ import annotations

import pytest

from nodes.otr_master_audio_mux import duration_receipt_line

TOL = 0.12  # 3 frames @ 25fps, the shipping duration_tol_frames


# --- the exact leg that exposed the defect (4060, leg 2204f5f8) -----------

def test_the_leg_that_printed_ok_now_prints_over_budget():
    line = duration_receipt_line(155.120, 120.928, 25.211, "declared", TOL)
    assert "OVER_BUDGET" in line, (
        "the receipt for the exact episode that failed its gate by 8.86s still "
        "does not say so: %r" % line)
    assert " OK" not in line, "the over-budget receipt still claims OK: %r" % line
    assert "8.86" in line, (
        "the overage is missing from the line, so the receipt is not "
        "self-contained: %r" % line)


def test_the_leg_that_passed_still_prints_ok():
    """The other real leg, same bank, same box -- must NOT be false-positived."""
    line = duration_receipt_line(155.280, 130.709, 24.600, "declared", TOL)
    assert line.endswith(" OK"), (
        "an episode comfortably inside its tail budget no longer reads OK: %r"
        % line)


def test_the_5080_five_act_residual_reads_over_budget():
    """16.83s over on 4 bridges -- the accepted 'let it fly' residual."""
    line = duration_receipt_line(281.4565, 228.7930, 35.711, "declared", TOL)
    assert "OVER_BUDGET" in line and "published anyway" in line, (
        "the 5-act residual must be visible in the receipt AND state that the "
        "episode still shipped: %r" % line)


# --- the boundary, which is where an off-by-one would hide ----------------

@pytest.mark.parametrize("excess_over_tail,expect_over", [
    (-1.0, False),    # comfortably inside
    (-0.001, False),  # just inside
    (0.0, False),     # exactly at budget+tol is NOT over (gate uses >)
    (0.001, True),    # just past
    (5.0, True),      # well past
])
def test_the_budget_boundary(excess_over_tail, expect_over):
    a_dur, tail = 100.0, 20.0
    v_dur = a_dur + tail + TOL + excess_over_tail
    line = duration_receipt_line(v_dur, a_dur, tail, "declared", TOL)
    if expect_over:
        assert "OVER_BUDGET" in line, "v=%.4f should be over: %r" % (v_dur, line)
    else:
        assert line.endswith(" OK"), "v=%.4f should be OK: %r" % (v_dur, line)


def test_the_receipt_matches_the_warnings_arithmetic():
    """Receipt and log must never disagree about the overage."""
    v_dur, a_dur, tail = 155.120, 120.928, 25.211
    expected = (v_dur - a_dur) - tail - TOL
    line = duration_receipt_line(v_dur, a_dur, tail, "declared", TOL)
    assert ("by %.4fs" % expected) in line, (
        "the receipt reports a different overage than the warning computes "
        "(expected %.4fs): %r" % (expected, line))


# --- a failed probe must never read as a passed gate ---------------------

@pytest.mark.parametrize("v_dur,a_dur", [
    (-1.0, -1.0),   # both unprobeable
    (-1.0, 130.0),  # video only
    (155.0, -1.0),  # audio only
])
def test_a_failed_probe_is_unproven_not_ok(v_dur, a_dur):
    line = duration_receipt_line(v_dur, a_dur, 24.6, "declared", TOL)
    assert "UNPROVEN" in line, (
        "a failed probe (v=%r a=%r) must not report a verdict it did not "
        "compute: %r" % (v_dur, a_dur, line))
    assert "SKIPPED" in line, (
        "UNPROVEN must say the gate was skipped rather than passed: %r" % line)
    assert "OVER_BUDGET" not in line, (
        "a failed probe cannot claim an overage it could not measure: %r" % line)


def test_the_three_verdicts_are_mutually_exclusive():
    """Whatever the inputs, exactly one verdict appears."""
    cases = [(155.120, 120.928, 25.211), (155.280, 130.709, 24.600),
             (-1.0, -1.0, 24.600), (100.0, 100.0, 0.0)]
    for v_dur, a_dur, tail in cases:
        line = duration_receipt_line(v_dur, a_dur, tail, "declared", TOL)
        hits = sum(v in line for v in ("UNPROVEN", "OVER_BUDGET"))
        is_ok = line.endswith(" OK")
        assert hits + int(is_ok) == 1, (
            "v=%r a=%r produced %d verdicts, not exactly one: %r"
            % (v_dur, a_dur, hits + int(is_ok), line))


def test_the_tail_source_is_named_so_a_reader_knows_which_budget():
    for src_name in ("declared", "env_ceiling"):
        line = duration_receipt_line(150.0, 120.0, 24.6, src_name, TOL)
        assert "(%s)" % src_name in line, (
            "the receipt does not say which budget it compared against, so a "
            "reader cannot tell a declared tail from the env ceiling: %r" % line)
