"""tests/test_no_phantom_handlers_b4.py — S33 B4 deletion-proof tests.

S33 B4 (2026-05-15) retired the two phantom handlers classified
DELETE in S33 B1.5 per the refined no-auditors rule:

  * `apply_phantom_skip_fallback` -- mutated `line["skip"] = True`
    and cleared text. The mutation SILENCED the line rather than
    fixing it; a mute is a pipeline cut, not a story edit.
  * `_final_phantom_check` -- pure report-only function whose only
    consumer was the `post_audit_pass` rollback gate retired in
    S33 B2.

`auto_remap_phantom` survives (KEEP per B1.5) -- it powers
`apply_deterministic_cast_repairs` (an editor that rewrites phantom
names into real cast names).

This file proves the deletions stuck:

  1. The retired functions do not exist as attributes of
     `_otr_ledger_reviewer`.
  2. `__all__` does not export them.
  3. `ReviewerDisposition` no longer carries the `phantom_skip_count`
     field (it was the dataclass surface of the retired function).
  4. Tree-wide string-ref sweep across `nodes/` proves 0
     production-code hits for the retired function names (S33
     callouts allowed via context window).
  5. Tree-wide string-ref sweep across `tests/` proves 0 hits
     (excluding this test file).
  6. `auto_remap_phantom` SURVIVES -- explicit positive check.
"""
from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest

from nodes import _otr_ledger_reviewer as _OTRLR


_REPO_ROOT = Path(__file__).resolve().parent.parent
_THIS_FILE = Path(__file__).resolve()

_RETIRED_FUNCS = ("apply_phantom_skip_fallback", "_final_phantom_check")


# ---------------------------------------------------------------------------
# Symbol-level deletion proof
# ---------------------------------------------------------------------------


class TestPhantomHandlerSymbolsDeleted:
    @pytest.mark.parametrize("func_name", _RETIRED_FUNCS)
    def test_function_not_in_module(self, func_name):
        assert not hasattr(_OTRLR, func_name), (
            f"S33 B4 deletion regressed: `_otr_ledger_reviewer.{func_name}` "
            f"still exists as a module attribute"
        )

    @pytest.mark.parametrize("func_name", _RETIRED_FUNCS)
    def test_function_not_in_all_exports(self, func_name):
        all_exports = getattr(_OTRLR, "__all__", [])
        assert func_name not in all_exports, (
            f"S33 B4 deletion regressed: `{func_name}` still listed in "
            f"_otr_ledger_reviewer.__all__"
        )

    def test_auto_remap_phantom_SURVIVES(self):
        # auto_remap_phantom was classified KEEP in B1.5 because it
        # powers apply_deterministic_cast_repairs (an editor). It
        # MUST still exist.
        assert hasattr(_OTRLR, "auto_remap_phantom"), (
            "S33 B4 over-deletion: `auto_remap_phantom` was classified "
            "KEEP in B1.5 but is now missing from the module"
        )
        assert "auto_remap_phantom" in getattr(_OTRLR, "__all__", []), (
            "S33 B4 over-deletion: `auto_remap_phantom` no longer in "
            "__all__ but it should survive"
        )


# ---------------------------------------------------------------------------
# ReviewerDisposition field trim
# ---------------------------------------------------------------------------


class TestReviewerDispositionFieldTrim:
    def test_phantom_skip_count_field_deleted(self):
        field_names = {f.name for f in fields(_OTRLR.ReviewerDisposition)}
        assert "phantom_skip_count" not in field_names, (
            "S33 B4 deletion regressed: ReviewerDisposition still "
            f"carries `phantom_skip_count` field: {sorted(field_names)}"
        )

    def test_remaining_fields_intact(self):
        # Surviving fields after B4.
        field_names = {f.name for f in fields(_OTRLR.ReviewerDisposition)}
        expected = {
            "verdict",
            "pre_audit_violations",
            "pre_audit_repairs_applied",
            "doctor_edits_proposed",
            "doctor_edits_applied",
            "post_audit_violations",
        }
        assert field_names == expected, (
            f"ReviewerDisposition field surface drifted from expected "
            f"post-B4 set; got {sorted(field_names)}, expected "
            f"{sorted(expected)}"
        )


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


def _grep_string(text: str, needle: str) -> bool:
    pat = re.compile(r"\b" + re.escape(needle) + r"\b")
    return pat.search(text) is not None


def _within_s33_callout(lines: list[str], idx: int, window: int = 12) -> bool:
    lo = max(0, idx - window)
    hi = min(len(lines), idx + window + 1)
    region = "\n".join(lines[lo:hi])
    return (
        "S33 B" in region
        or "retired" in region.lower()
        or "retire" in region.lower()
    )


class TestStringRefSweep:
    @pytest.mark.parametrize("func_name", _RETIRED_FUNCS)
    def test_no_refs_in_nodes(self, func_name):
        hits: list[str] = []
        for p in _iter_text_files("nodes", suffix=(".py",)):
            text = p.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
            for idx, line in enumerate(lines):
                if _grep_string(line, func_name):
                    if _within_s33_callout(lines, idx):
                        continue
                    hits.append(
                        f"{p.relative_to(_REPO_ROOT)}:{idx + 1}:"
                        f"{line.strip()}"
                    )
        assert hits == [], (
            f"S33 B4 deletion regressed: {func_name!r} still referenced "
            f"in nodes/: {hits}"
        )

    @pytest.mark.parametrize("func_name", _RETIRED_FUNCS)
    def test_no_refs_in_tests(self, func_name):
        hits: list[str] = []
        for p in _iter_text_files("tests", suffix=(".py",)):
            if p.resolve() == _THIS_FILE:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
            for idx, line in enumerate(lines):
                if _grep_string(line, func_name):
                    if _within_s33_callout(lines, idx):
                        continue
                    hits.append(
                        f"{p.relative_to(_REPO_ROOT)}:{idx + 1}:"
                        f"{line.strip()}"
                    )
        assert hits == [], (
            f"S33 B4 deletion regressed: {func_name!r} still referenced "
            f"in tests/: {hits}"
        )


# ---------------------------------------------------------------------------
# review_ledger source: no Step 2.5 dispatch
# ---------------------------------------------------------------------------


def _scan_source_for_leaky_token(src: str, token: str) -> list[str]:
    """Return source lines mentioning `token` outside an S33 callout
    window (+/-12 lines). Catches multi-line explanatory comment
    blocks where the marker is at the head and the token reference
    is several lines deeper."""
    lines = src.splitlines()
    offenders: list[str] = []
    for idx, line in enumerate(lines):
        if token not in line:
            continue
        if _within_s33_callout(lines, idx):
            continue
        offenders.append(line.strip())
    return offenders


class TestStep25DispatchDeleted:
    def test_no_apply_phantom_skip_fallback_call_in_review_ledger(self):
        import inspect
        src = inspect.getsource(_OTRLR.review_ledger)
        offenders = _scan_source_for_leaky_token(
            src, "apply_phantom_skip_fallback"
        )
        assert offenders == [], (
            "S33 B4 deletion regressed: `apply_phantom_skip_fallback` "
            f"still invoked in review_ledger: {offenders}"
        )

    def test_no_final_phantom_check_call_in_review_ledger(self):
        import inspect
        src = inspect.getsource(_OTRLR.review_ledger)
        offenders = _scan_source_for_leaky_token(
            src, "_final_phantom_check"
        )
        assert offenders == [], (
            "S33 B4 deletion regressed: `_final_phantom_check` "
            f"still invoked in review_ledger: {offenders}"
        )
