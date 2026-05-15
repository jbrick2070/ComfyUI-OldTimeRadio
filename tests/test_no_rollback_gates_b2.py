"""tests/test_no_rollback_gates_b2.py — S33 B2 deletion-proof tests.

S33 B2 (2026-05-15) retired the two cascade rollback gates per the
refined no-auditors rule:

  * `speaker_unknowns` rollback gate -> drove `cast_unrecoverable`
    verdict
  * `post_audit_pass` rollback gate -> drove `post_audit_failed`
    verdict

Both gates "cut the pipeline" rather than developing the story, so
they fail the refined rule ("Audit calls are OK if they USE the
audit to develop / edit the story. NOT OK if they just cut the
pipeline").

This file proves the deletion stuck:

  1. `ReviewerVerdict` Literal no longer contains the retired
     verdict strings.
  2. `REVIEWER_TO_FREEZE_VERDICT` mapping has no entries for them.
  3. `FREEZE_TERMINAL_FAILURE_VERDICTS` frozenset has no entries.
  4. Tree-wide string-ref sweep across `nodes/` proves 0 hits
     for the retired verdict strings (excluding forensic docs +
     this test file).
  5. Tree-wide string-ref sweep across `tests/` proves 0 hits
     (excluding this test file).
  6. Tree-wide string-ref sweep across `workflows/` proves 0 hits.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import pytest

from nodes import _otr_freeze_cascade as _LFC_ORCH
from nodes import _otr_ledger_reviewer as _OTRLR


_RETIRED_VERDICTS = ("cast_unrecoverable", "post_audit_failed")

# Repo root = parent of `tests/` parent of this file.
_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Verdict literal + mapping deletions
# ---------------------------------------------------------------------------


class TestReviewerVerdictLiteralTrim:
    @pytest.mark.parametrize("retired", _RETIRED_VERDICTS)
    def test_retired_verdict_not_in_reviewer_literal(self, retired):
        # B2 deletion proof: the retired verdict strings must not
        # appear in the ReviewerVerdict Literal arg list.
        lits = set(get_args(_OTRLR.ReviewerVerdict))
        assert retired not in lits, (
            f"S33 B2 deletion regressed: {retired!r} still in "
            f"ReviewerVerdict literal set {lits!r}"
        )

    def test_reviewer_literal_size_is_4(self):
        # Surviving verdicts: clean_no_edits, improved,
        # too_many_edits, needs_full_rerun.
        lits = set(get_args(_OTRLR.ReviewerVerdict))
        assert lits == {
            "clean_no_edits",
            "improved",
            "too_many_edits",
            "needs_full_rerun",
        }


class TestReviewerToFreezeVerdictTrim:
    @pytest.mark.parametrize("retired", _RETIRED_VERDICTS)
    def test_retired_verdict_not_in_mapping(self, retired):
        assert retired not in _LFC_ORCH.REVIEWER_TO_FREEZE_VERDICT, (
            f"S33 B2 deletion regressed: REVIEWER_TO_FREEZE_VERDICT "
            f"still maps {retired!r}"
        )

    def test_mapping_size_is_4(self):
        # One entry per surviving ReviewerVerdict literal.
        assert set(_LFC_ORCH.REVIEWER_TO_FREEZE_VERDICT.keys()) == {
            "clean_no_edits",
            "improved",
            "too_many_edits",
            "needs_full_rerun",
        }


class TestFreezeTerminalFailureVerdictsTrim:
    @pytest.mark.parametrize("retired", _RETIRED_VERDICTS)
    def test_retired_verdict_not_in_terminal_set(self, retired):
        assert retired not in _LFC_ORCH.FREEZE_TERMINAL_FAILURE_VERDICTS, (
            f"S33 B2 deletion regressed: terminal-failure verdict "
            f"set still contains {retired!r}"
        )

    def test_terminal_set_size_is_2(self):
        assert _LFC_ORCH.FREEZE_TERMINAL_FAILURE_VERDICTS == frozenset({
            "too_many_edits",
            "needs_full_rerun",
        })


# ---------------------------------------------------------------------------
# Tree-wide string-ref sweep
# ---------------------------------------------------------------------------


def _iter_text_files(*dirs: str, suffix: tuple[str, ...]):
    for d in dirs:
        base = _REPO_ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix not in suffix:
                continue
            if "__pycache__" in p.parts:
                continue
            yield p


_THIS_FILE = Path(__file__).resolve()


def _grep_string(text: str, needle: str) -> bool:
    """Return True if needle appears in text as a bareword or
    quoted literal. Word-boundary regex to dodge near-misses."""
    pat = re.compile(r"\b" + re.escape(needle) + r"\b")
    return pat.search(text) is not None


def _within_s33_b2_callout(lines: list[str], idx: int, window: int = 12) -> bool:
    """Return True if any of the surrounding `window` lines (either
    side of `idx`) carry the "S33 B2" marker or the "retired" marker.

    S33 B2 deletion-callout comments / docstrings often span multiple
    lines; the marker appears at the head of the block but the actual
    verdict-string mention may be many lines down. A bare per-line
    filter misses those. The window catches them.

    Window size 12 is generous enough to cover the longest
    explanatory docstring while staying tight enough to catch
    accidentally-stranded references.
    """
    lo = max(0, idx - window)
    hi = min(len(lines), idx + window + 1)
    region = "\n".join(lines[lo:hi])
    return "S33 B" in region or "retired" in region.lower()


class TestStringRefSweep:
    """0 hits anywhere outside this test file + forensic docs."""

    @pytest.mark.parametrize("retired", _RETIRED_VERDICTS)
    def test_no_refs_in_nodes(self, retired):
        # S33 B2 docstring callouts intentionally name the retired
        # verdicts to explain WHY they're gone. The callout markers
        # ("S33 B2" / "retired") may sit a few lines above the mention,
        # so check a +/-6 line window around each hit.
        hits: list[str] = []
        for p in _iter_text_files("nodes", suffix=(".py",)):
            text = p.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
            for idx, line in enumerate(lines):
                if _grep_string(line, retired):
                    if _within_s33_b2_callout(lines, idx):
                        continue
                    hits.append(
                        f"{p.relative_to(_REPO_ROOT)}:{idx + 1}:"
                        f"{line.strip()}"
                    )
        assert hits == [], (
            f"S33 B2 deletion regressed: {retired!r} still referenced "
            f"in nodes/: {hits}"
        )

    @pytest.mark.parametrize("retired", _RETIRED_VERDICTS)
    def test_no_refs_in_tests(self, retired):
        hits: list[str] = []
        for p in _iter_text_files("tests", suffix=(".py",)):
            if p.resolve() == _THIS_FILE:
                continue  # this file legitimately names the retired verdicts
            text = p.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
            for idx, line in enumerate(lines):
                if _grep_string(line, retired):
                    if _within_s33_b2_callout(lines, idx):
                        continue
                    hits.append(
                        f"{p.relative_to(_REPO_ROOT)}:{idx + 1}:"
                        f"{line.strip()}"
                    )
        assert hits == [], (
            f"S33 B2 deletion regressed: {retired!r} still referenced "
            f"in tests/: {hits}"
        )

    @pytest.mark.parametrize("retired", _RETIRED_VERDICTS)
    def test_no_refs_in_workflows(self, retired):
        hits: list[str] = []
        for p in _iter_text_files("workflows", suffix=(".json",)):
            text = p.read_text(encoding="utf-8", errors="ignore")
            for ln_no, line in enumerate(text.splitlines(), 1):
                if _grep_string(line, retired):
                    hits.append(f"{p.relative_to(_REPO_ROOT)}:{ln_no}:{line.strip()}")
        assert hits == [], (
            f"S33 B2 deletion regressed: {retired!r} still referenced "
            f"in workflows/: {hits}"
        )


# ---------------------------------------------------------------------------
# Rollback-gate machinery deletion
# ---------------------------------------------------------------------------


def _scan_for_leaky_token(src: str, token: str) -> list[str]:
    """Return source lines that mention `token` outside of an S33
    callout. A line is considered "inside a callout" when any of the
    surrounding +/-6 lines carry an "S33 B" marker or the "retired"
    marker. Catches multi-line explanatory blocks where the marker
    sits at the head of the block."""
    lines = src.splitlines()
    leaky: list[str] = []
    for idx, line in enumerate(lines):
        if token not in line:
            continue
        if _within_s33_b2_callout(lines, idx):
            continue
        leaky.append(line)
    return leaky


class TestRollbackGateMachineryDeleted:
    def test_speaker_unknowns_local_not_in_review_ledger_source(self):
        # The local variable `speaker_unknowns` lived inside
        # review_ledger to host the rollback gate filter. After B2
        # the variable is gone. Explanatory callouts about its
        # retirement are allowed (window-based exclusion).
        import inspect
        src = inspect.getsource(_OTRLR.review_ledger)
        leaky_lines = _scan_for_leaky_token(src, "speaker_unknowns")
        assert leaky_lines == [], (
            "S33 B2 deletion regressed: speaker_unknowns rollback "
            f"gate still present in review_ledger: {leaky_lines}"
        )

    def test_post_audit_pass_local_not_in_review_ledger_source(self):
        import inspect
        src = inspect.getsource(_OTRLR.review_ledger)
        leaky_lines = _scan_for_leaky_token(src, "post_audit_pass")
        assert leaky_lines == [], (
            "S33 B2 deletion regressed: post_audit_pass rollback "
            f"gate still present in review_ledger: {leaky_lines}"
        )

    def test_final_phantoms_call_not_in_review_ledger_source(self):
        # The `final_phantoms = _final_phantom_check(...)` call only
        # served the post_audit_pass gate. After B2 it's gone.
        # (B4 deletes the function itself.)
        import inspect
        src = inspect.getsource(_OTRLR.review_ledger)
        leaky_lines = _scan_for_leaky_token(src, "final_phantoms")
        assert leaky_lines == [], (
            "S33 B2 deletion regressed: final_phantoms call still "
            f"present in review_ledger: {leaky_lines}"
        )
