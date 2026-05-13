"""S13.2 -- G8 line_id uniqueness invariant tests.

ProcSFX's filename scheme (``proc_<sfx_type>_<line_id>_<perm>.wav``)
and every ledger write-back path (``apply_line_timings``,
``patch_line_fields``, etc.) key by ``line_id``. Duplicates produce
silent overwrites in BOTH places. G8 catches the duplication at
freeze time -- before any render fires.

Phase 0 collects (warn-mode), Phase 10 raises FreezeAssertionError.
The check is wired into ``run_gap_audit`` alongside G1-G7.
"""
from __future__ import annotations

import pytest

from nodes import _otr_ledger_freeze as _LFC


def _mk_minimal_ledger(lines: list[dict]) -> dict:
    return {
        "schema_version": "l3-2026-05-14",
        "cast": [],
        "lines": lines,
        "meta": {},
    }


def test_g8_unique_line_ids_pass():
    """Distinct line_ids across lines[] -> G8 reports no errors."""
    led = _mk_minimal_ledger([
        {"line_id": "ln_001", "char_id": "c01",
         "speaker_role": "character", "text": "alpha", "traits": "calm"},
        {"line_id": "ln_002", "char_id": "c01",
         "speaker_role": "character", "text": "beta", "traits": "calm"},
        {"line_id": "sfx_001", "char_id": "sfx",
         "speaker_role": "sfx", "text": "alarm", "traits": "ambient"},
    ])
    report = _LFC.run_gap_audit(led, label="test")
    g8_errors = [e for e in report.errors if "G8" in e]
    assert not g8_errors, (
        f"G8 fired on a clean ledger: {g8_errors}"
    )


def test_g8_duplicate_line_ids_raise_at_phase_10():
    """Phase 10 hard-fails on G8 violation. Inspects the structured
    ``.errors`` attribute on the exception (not the formatted message,
    which only includes the FIRST error) so the assertion is robust
    against null-rejection / per-line invariants firing on the same
    minimal fixture."""
    led = _mk_minimal_ledger([
        {"line_id": "ln_001", "char_id": "c01",
         "speaker_role": "character", "text": "alpha", "traits": "calm"},
        {"line_id": "ln_001", "char_id": "c02",
         "speaker_role": "character", "text": "beta", "traits": "calm"},
    ])
    with pytest.raises(_LFC.FreezeAssertionError) as excinfo:
        _LFC.phase_10_gap_audit_post_and_freeze(led)
    g8_errors = [e for e in excinfo.value.errors if "G8" in e]
    assert g8_errors, (
        f"FreezeAssertionError.errors should include a G8 entry. "
        f"Got: {excinfo.value.errors}"
    )


def test_g8_phase_0_collects_does_not_raise():
    """Phase 0 collects but does NOT raise -- the soft-warn behavior
    matches G1-G7. Phase 10 is the hard gate."""
    led = _mk_minimal_ledger([
        {"line_id": "dup_id", "char_id": "c01",
         "speaker_role": "character", "text": "first", "traits": "calm"},
        {"line_id": "dup_id", "char_id": "c02",
         "speaker_role": "character", "text": "second", "traits": "calm"},
    ])
    # Should NOT raise.
    report = _LFC.phase_0_gap_audit_pre(led)
    g8_errors = [e for e in report.errors if "G8" in e]
    assert g8_errors, (
        "Phase 0 must collect G8 violations even though it doesn't raise. "
        "Got no G8 errors."
    )


def test_g8_three_or_more_duplicates_reported_with_more_suffix():
    """When >5 duplicates exist, the diagnostic shows the first 5 + a
    ``(+N more)`` suffix so the operator sees severity but the message
    stays short."""
    lines = []
    for i in range(8):
        lines.append({
            "line_id": "ln_dup",  # all share the same id
            "char_id": f"c{i:02d}",
            "speaker_role": "character",
            "text": f"line {i}",
            "traits": "calm",
        })
    led = _mk_minimal_ledger(lines)
    report = _LFC.run_gap_audit(led, label="test")
    g8_errors = [e for e in report.errors if "G8" in e]
    assert len(g8_errors) == 1, f"Expected 1 G8 error, got {len(g8_errors)}"
    msg = g8_errors[0]
    # 8 lines with same line_id -> 7 duplicates after the first one is
    # added to `seen`. 7 > 5 so the +more suffix should fire.
    assert "+2 more" in msg, (
        f"G8 diagnostic should cap displayed dups at 5 + suffix; got: {msg}"
    )


def test_g8_skips_lines_without_line_id():
    """Lines without line_id (or with empty/non-string line_id) are
    skipped -- they're already caught by per-line type invariants;
    no need for G8 to double-report."""
    led = _mk_minimal_ledger([
        {"char_id": "c01", "speaker_role": "character",
         "text": "no id", "traits": "calm"},  # no line_id
        {"line_id": "", "char_id": "c02", "speaker_role": "character",
         "text": "empty", "traits": "calm"},  # empty line_id
        {"line_id": None, "char_id": "c03", "speaker_role": "character",
         "text": "none", "traits": "calm"},  # None line_id
    ])
    report = _LFC.run_gap_audit(led, label="test")
    g8_errors = [e for e in report.errors if "G8" in e]
    assert not g8_errors, (
        f"G8 should skip lines without line_id (already caught by other "
        f"checks). Got: {g8_errors}"
    )


def test_g8_handles_non_list_lines_defensively():
    """Non-list ``lines`` value -> G8 returns silently. The top-level
    null-rejection / per-line type invariants flag the structural
    issue; G8 doesn't double-report."""
    led = {
        "schema_version": "l3-2026-05-14",
        "cast": [],
        "lines": "not a list",
        "meta": {},
    }
    report = _LFC.run_gap_audit(led, label="test")
    g8_errors = [e for e in report.errors if "G8" in e]
    assert not g8_errors


def test_g8_runs_alongside_g7_independently():
    """G7 and G8 are independent checks. A ledger with both a duplicate
    line_id AND an out-of-bounds SFX dur_s should produce both errors
    in a single run."""
    led = _mk_minimal_ledger([
        {"line_id": "sfx_001", "char_id": "sfx",
         "speaker_role": "sfx", "text": "alarm", "dur_s": 25.0,
         "traits": "ambient"},
        {"line_id": "sfx_001", "char_id": "sfx",
         "speaker_role": "sfx", "text": "alarm again", "dur_s": 1.0,
         "traits": "ambient"},
    ])
    report = _LFC.run_gap_audit(led, label="test")
    g7_errors = [e for e in report.errors if "G7" in e]
    g8_errors = [e for e in report.errors if "G8" in e]
    assert g7_errors and g8_errors, (
        f"Expected both G7 and G8 errors. G7: {g7_errors}, G8: {g8_errors}"
    )
