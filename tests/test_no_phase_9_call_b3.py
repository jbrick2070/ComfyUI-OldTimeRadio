"""tests/test_no_phase_9_call_b3.py — S33 B3 deletion-proof tests.

S33 B3 (2026-05-15) retired the Phase 9 LLM call
`audit_cast_contract(..., label="post")` per the refined
no-auditors rule. Its only consumer was the `post_audit_pass`
rollback gate, which B2 already removed.

The shared `audit_cast_contract` function survives because its
remaining call site (Phase 1, `label="pre"`) feeds
`apply_deterministic_cast_repairs` -- an editor that uses the
audit output to develop the story.

This file proves the deletion stuck:

  1. `review_ledger` source has no `label="post"` call site.
  2. `review_ledger` source has exactly ONE `audit_cast_contract`
     call (Phase 1 only).
  3. End-to-end `review_ledger` happy-path runs with only 2 LLM
     calls (down from 3 pre-B3).
  4. Tree-wide string-ref sweep across `nodes/` proves 0
     production-code hits for `label="post"` (excluding S33 B3
     callout comments + retired pre-S33 docs).
"""
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from nodes import _otr_ledger_reviewer as _OTRLR


_REPO_ROOT = Path(__file__).resolve().parent.parent
_THIS_FILE = Path(__file__).resolve()


# ---------------------------------------------------------------------------
# review_ledger source inspection
# ---------------------------------------------------------------------------


class TestReviewLedgerSourceShape:
    def test_no_label_post_call_in_review_ledger(self):
        # The Phase 9 invocation `audit_cast_contract(..., label="post")`
        # has been deleted. Comments/docstrings citing the historical
        # call are allowed only if they include the "S33 B3" marker.
        src = inspect.getsource(_OTRLR.review_ledger)
        offenders: list[str] = []
        for line in src.splitlines():
            if "label=\"post\"" in line or "label='post'" in line:
                if "S33 B3" in line or "retired" in line.lower():
                    continue
                offenders.append(line.strip())
        assert offenders == [], (
            "S33 B3 deletion regressed: `label=\"post\"` still appears in "
            f"review_ledger: {offenders}"
        )

    def test_audit_cast_contract_called_once_in_review_ledger(self):
        # After B3, only the Phase 1 (label="pre") call remains.
        # AST-walk the function body so comments / docstrings citing
        # the historical `audit_cast_contract(label="post")` shape
        # don't get miscounted as live calls.
        import ast as _ast
        src = inspect.getsource(_OTRLR.review_ledger)
        # `inspect.getsource` preserves the original function's
        # indentation; ast.parse needs dedented input.
        import textwrap
        tree = _ast.parse(textwrap.dedent(src))
        call_count = 0
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Call):
                func = node.func
                # Match bare-name calls (`audit_cast_contract(...)`)
                # and attribute calls (`mod.audit_cast_contract(...)`).
                if isinstance(func, _ast.Name) and func.id == "audit_cast_contract":
                    call_count += 1
                elif isinstance(func, _ast.Attribute) and func.attr == "audit_cast_contract":
                    call_count += 1
        assert call_count == 1, (
            "S33 B3 deletion regressed: review_ledger should invoke "
            f"audit_cast_contract exactly once after Phase 9 retirement; "
            f"found {call_count} live invocation(s) (AST walk)."
        )

    def test_no_post_audit_local_in_review_ledger(self):
        # The `post_audit = ...` local variable assignment is gone.
        src = inspect.getsource(_OTRLR.review_ledger)
        offenders: list[str] = []
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("post_audit ="):
                offenders.append(stripped)
                continue
            # Skip explanatory callouts.
            if "post_audit" in line:
                if "S33 B" in line or "retired" in line.lower():
                    continue
                # Bare references to `post_audit` outside the local
                # variable should be flagged.
                if "post_audit_violations" in line:
                    # The ReviewerDisposition counter `post_audit_violations`
                    # is a stable dataclass field name -- not an indication
                    # of the deleted local.
                    continue
                offenders.append(stripped)
        assert offenders == [], (
            "S33 B3 deletion regressed: `post_audit` local variable / "
            f"references still present in review_ledger: {offenders}"
        )


# ---------------------------------------------------------------------------
# audit_cast_contract signature
# ---------------------------------------------------------------------------


class TestAuditCastContractSignature:
    def test_label_default_is_pre(self):
        # After B3 the only call site is Phase 1 (label="pre"); the
        # default value should reflect that.
        sig = inspect.signature(_OTRLR.audit_cast_contract)
        param = sig.parameters.get("label")
        assert param is not None, (
            "audit_cast_contract is missing the `label` parameter"
        )
        assert param.default == "pre", (
            "S33 B3: `label` should default to 'pre' (only surviving "
            f"call site); got default={param.default!r}"
        )


# ---------------------------------------------------------------------------
# review_ledger happy-path -- two LLM calls
# ---------------------------------------------------------------------------


def _line(line_id: str, char_id: str, text: str, role: str = "character") -> dict:
    return {
        "line_id": line_id,
        "char_id": char_id,
        "speaker_role": role,
        "speaker": char_id,
        "text": text,
        "char_count": len(text),
        "word_count": len(text.split()),
        "scene_id": "s01",
        "arc_phase": "act_1",
    }


class _LedgerStub:
    """Minimal stand-in for production_ledger.Ledger used by the
    review_ledger contract. Same shape as the production test stub."""

    def __init__(self, data: dict, tmp_path: Path) -> None:
        self.data = data
        self.episode_id = "test_b3"
        self._path = tmp_path / "led.json"

    def save(self) -> None:
        self._path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")


def _audit_clean() -> str:
    return json.dumps({"violations": [], "pass_clean": True})


def _doctor_diagnosis_clean() -> str:
    # Sprint 3C: the Script Doctor diagnosis pass output.
    return json.dumps({"diagnoses": [
        {"line_id": "b001", "failure": "none", "note": "sound"},
    ]})


def _doctor_edits_clean() -> str:
    # Sprint 3C: the Script Doctor edits pass output.
    return json.dumps({"edits": [], "overall_verdict": "clean"})


class TestReviewLedgerLLMCallCount:
    def test_happy_path_uses_three_llm_calls(self, tmp_path):
        """Pass 1 (audit pre) + Pass 2 Script Doctor (Sprint 3C split:
        diagnosis + edits) = 3 LLM calls.

        S33 B3 retired the Phase 9 `audit_cast_contract(label="post")`
        call -- and it stays retired (the call-count is 3, not the
        pre-B3 4). Sprint 3C (2026-05-25) then split the single Script
        Doctor call into a diagnosis pass + an edits pass, taking the
        happy path from 2 LLM calls to 3. The +1 is the new diagnosis
        call, NOT a Phase 9 regression: this test's `label="post"`
        guard (in TestReviewLedgerSourceShape / TestStringRefSweep)
        independently proves Phase 9 stayed deleted.
        """
        data = {
            "lines": [_line("b001", "c01", "Hello there.", role="character")],
            "cast": [
                {"char_id": "c01", "name": "ALICE",
                 "speaker_role": "character"},
            ],
            "meta": {
                "allowed_roster": ["ALICE", "ANNOUNCER"],
                "model_id": "stub",
            },
        }
        led = _LedgerStub(data, tmp_path)

        call_count = {"n": 0}
        responses = [
            _audit_clean(),
            _doctor_diagnosis_clean(),
            _doctor_edits_clean(),
        ]

        def fake_generate(messages, *, temperature, max_new_tokens):
            i = call_count["n"]
            call_count["n"] += 1
            if i < len(responses):
                return responses[i]
            pytest.fail(
                f"review_ledger made {call_count['n']} LLM calls; expected 3 "
                f"(Phase 1 audit pre + Phase 2 doctor diagnosis + edits). "
                f"Phase 9 was retired in S33 B3 -- a fourth call would "
                f"indicate a Phase 9 regression."
            )

        disp = _OTRLR.review_ledger(fake_generate, led)
        assert call_count["n"] == 3, (
            f"Expected 3 LLM calls (Phase 1 audit + Sprint 3C Script "
            f"Doctor diagnosis + edits), got {call_count['n']}"
        )
        assert disp.verdict == "clean_no_edits"


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
    """Production code in `nodes/` should have 0 live references to
    the retired Phase 9 call shape (`label="post"`) outside of S33
    explanatory comments."""

    def test_no_label_post_in_nodes(self):
        # Note: `label="post"` is still a legitimate string inside
        # `nodes/_otr_ledger_freeze.py` for Phase 10 deterministic
        # gap-audit (label="pre" vs label="post" on the GapAudit
        # function), which is unrelated to Phase 9. So this test
        # specifically targets the reviewer / cascade trail, not
        # ledger_freeze.
        hits: list[str] = []
        for p in _iter_text_files("nodes", suffix=(".py",)):
            # Skip ledger_freeze (uses label="post" for the unrelated
            # Phase 10 deterministic gap audit, not the retired Phase 9).
            if p.name == "_otr_ledger_freeze.py":
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
            for idx, line in enumerate(lines):
                if 'label="post"' in line or "label='post'" in line:
                    if _within_s33_callout(lines, idx):
                        continue
                    hits.append(
                        f"{p.relative_to(_REPO_ROOT)}:{idx + 1}:"
                        f"{line.strip()}"
                    )
        assert hits == [], (
            "S33 B3 deletion regressed: `label=\"post\"` still referenced "
            f"in nodes/ outside ledger_freeze: {hits}"
        )

    def test_no_phase_9_method_in_tests(self):
        # Live references to a Phase 9 call site in tests/ must be
        # either S33 callouts or the historical `test_phase3_*` file
        # comments (which themselves carry S33 markers post-B3).
        hits: list[str] = []
        for p in _iter_text_files("tests", suffix=(".py",)):
            if p.resolve() == _THIS_FILE:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            lines = text.splitlines()
            for idx, line in enumerate(lines):
                if 'label="post"' in line or "label='post'" in line:
                    if _within_s33_callout(lines, idx):
                        continue
                    hits.append(
                        f"{p.relative_to(_REPO_ROOT)}:{idx + 1}:"
                        f"{line.strip()}"
                    )
        assert hits == [], (
            "S33 B3 deletion regressed: `label=\"post\"` still referenced "
            f"in tests/ outside this test file: {hits}"
        )
