"""tests/test_phase3_ledger_reviewer.py — two-pass cast-gated reviewer.

Covers (per script-writing-architecture synthesis §3 Phase 3 + §6.A + M2 + G1-G9):

  * Levenshtein auto_remap_phantom               §6.A + G8 test table
  * compute_edit_cap scales with voiced beats   G1
  * audit_cast_contract single-fn + label       G4 (Pass 1 + Pass 3 same fn)
  * apply_deterministic_cast_repairs            each repair kind branch
  * (S33 B4 retired apply_phantom_skip_fallback M2 safety net)
  * review_ledger disposition for each verdict:
        clean_no_edits / improved /
        too_many_edits / needs_full_rerun
    (S33 B2 retired cast_unrecoverable + post_audit_failed with
    their rollback gates per refined no-auditors rule.)
  * skip_reviewer programmatic bypass            G9

Pure-Python. Mock generate_fn for every LLM call. No GPU. No HF.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import production_ledger as PL  # noqa: E402
from nodes._otr_ledger_reviewer import (  # noqa: E402
    CastViolation,
    PreAuditReport,
    ReviewerEdit,
    ScriptDoctorReport,
    apply_deterministic_cast_repairs,
    audit_cast_contract,
    auto_remap_phantom,
    compute_edit_cap,
    review_ledger,
)
# S33 B4 (2026-05-15): `apply_phantom_skip_fallback` retired.


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _build_ledger(tmp_path, lines):
    """Build a fresh ledger with the given lines + ALICE/BOB cast."""
    d = tmp_path / "audio"
    d.mkdir(parents=True, exist_ok=True)
    led = PL.Ledger(episode_id="pending_test", out_dir=str(d))
    led.set_cast([
        {"char_id": "c01", "name": "ALICE"},
        {"char_id": "c02", "name": "BOB"},
        {"char_id": "c03", "name": "ANNOUNCER"},
    ])
    # Skip set_lines normalization for compose_flags/skip fields by
    # writing directly.
    led.data["lines"] = lines
    return led


def _line(line_id, char_id, text, role="character", arc_phase=None):
    return {
        "line_id": line_id,
        "beat_id": line_id,
        "char_id": char_id,
        "speaker_role": role,
        "text": text,
        "traits": None,
        "boundary": None,
        "compose_flags": [],
        "char_count": len(text),
        "word_count": len(text.split()),
        "bark_wav_path": None,
        "start_s": None,
        "dur_s": None,
        "arc_phase": arc_phase,
    }


def _mk_generate_fn(*replies):
    """Mock generate_fn that returns the given JSON strings in order."""
    state = {"i": 0}

    def fn(messages, *, temperature, max_new_tokens):
        i = state["i"]
        state["i"] += 1
        if i < len(replies):
            return replies[i]
        return replies[-1] if replies else "{}"
    return fn


def _audit_clean_json():
    return json.dumps({"violations": [], "pass_clean": True})


def _audit_phantom_json(line_id, phantom):
    return json.dumps({
        "violations": [{
            "line_id": line_id,
            "kind": "invented_name",
            "found": phantom,
            "expected": "",
            "confidence": 0.9,
        }],
        "pass_clean": False,
    })


def _doctor_json(edits=None, verdict="clean"):
    return json.dumps({
        "edits": edits or [],
        "overall_verdict": verdict,
    })


# ---------------------------------------------------------------------------
# auto_remap_phantom -- §6.A G8 test table
# ---------------------------------------------------------------------------


class TestAutoRemapPhantom:

    @pytest.mark.parametrize("phantom,cast,expected", [
        ("alice",     ["ALICE", "BOB"], "ALICE"),
        ("Allice",    ["ALICE", "BOB"], "ALICE"),
        ("Alyce",     ["ALICE", "BOB"], "ALICE"),
        ("BOBB",      ["ALICE", "BOB"], "BOB"),
    ])
    def test_close_matches_remap(self, phantom, cast, expected):
        assert auto_remap_phantom(phantom, cast) == expected

    @pytest.mark.parametrize("phantom,cast", [
        ("Robert",    ["ALICE", "BOB"]),       # dist 5 to BOB
        ("Patel",     ["ALICE", "BOB"]),       # dist 4 to ALICE
        ("Dr. Patel", ["ALICE", "BOB"]),       # dist 8+
        ("the council", ["ALICE", "BOB"]),     # dist 9+
    ])
    def test_far_matches_return_none(self, phantom, cast):
        assert auto_remap_phantom(phantom, cast) is None

    def test_substring_containment_fast_path(self):
        # "ALICE" substring of "Doctor Alice" (case-folded) -> match.
        assert auto_remap_phantom("Doctor Alice", ["ALICE", "BOB"]) == "ALICE"

    def test_empty_roster_returns_none(self):
        assert auto_remap_phantom("alice", []) is None

    def test_empty_phantom_returns_none(self):
        assert auto_remap_phantom("", ["ALICE"]) is None


# ---------------------------------------------------------------------------
# compute_edit_cap -- G1
# ---------------------------------------------------------------------------


class TestComputeEditCap:

    @pytest.mark.parametrize("voiced,expected", [
        (0,   3),    # floor 3
        (6,   3),
        (9,   3),
        (12,  4),
        (15,  5),
        (18,  6),
        (19,  6),
        (24,  8),    # ceiling 8
        (50,  8),    # ceiling holds
    ])
    def test_scale_with_voiced_beats(self, voiced, expected):
        assert compute_edit_cap(voiced) == expected


# ---------------------------------------------------------------------------
# audit_cast_contract -- single function used once (G4 superseded by S33 B3)
# ---------------------------------------------------------------------------
# S33 B3 (2026-05-15): the second call site (label="post", Phase 9)
# was retired per the refined no-auditors rule. Only Phase 1
# (label="pre") still invokes this function; its output drives
# `apply_deterministic_cast_repairs` (editor).


class TestAuditCastContract:

    def test_clean_audit_returns_clean(self, tmp_path):
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "Hello there.", role="character"),
        ])
        fn = _mk_generate_fn(_audit_clean_json())
        report = audit_cast_contract(fn, led.data, label="pre")
        assert isinstance(report, PreAuditReport)
        assert report.pass_clean is True
        assert report.violations == []

    def test_violations_parsed(self, tmp_path):
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "Dr. Patel is here.", role="character"),
        ])
        fn = _mk_generate_fn(_audit_phantom_json("b001", "Dr. Patel"))
        report = audit_cast_contract(fn, led.data, label="pre")
        assert len(report.violations) == 1
        assert report.violations[0].kind == "invented_name"
        assert report.violations[0].found == "Dr. Patel"

    def test_llm_failure_returns_audit_failed_sentinel(self, tmp_path):
        """Wiring-review #8 (2026-05-11): audit fail-loud. LLM call
        raising must NOT default to pass_clean=True; map to the
        sentinel so the caller's branch fires."""
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "Hello.", role="character"),
        ])
        def boom(messages, *, temperature, max_new_tokens):
            raise RuntimeError("simulated OOM")
        report = audit_cast_contract(boom, led.data, label="pre")
        assert report.pass_clean is False
        assert getattr(report, "audit_failed", False) is True
        assert "simulated OOM" in getattr(report, "audit_failure_reason", "")

    def test_malformed_json_returns_audit_failed_sentinel(self, tmp_path):
        """Wiring-review #8 (2026-05-11): JSON parse failure also
        sentinel + pass_clean=False."""
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "Hello.", role="character"),
        ])
        fn = _mk_generate_fn("not json at all")
        report = audit_cast_contract(fn, led.data, label="pre")
        assert report.pass_clean is False
        assert getattr(report, "audit_failed", False) is True


# ---------------------------------------------------------------------------
# apply_deterministic_cast_repairs
# ---------------------------------------------------------------------------


class TestDeterministicCastRepairs:

    def test_bad_casing_replaced(self, tmp_path):
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "alice waits.", role="character"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="bad_casing",
                found="alice", expected="ALICE", confidence=0.95,
            )
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 1
        assert led.data["lines"][0]["text"] == "ALICE waits."

    # Wiring-review #11 supersedes the original `test_wrong_char_id
    # _overwritten` test: deterministic repair no longer writes a raw
    # LLM-suggested char_id literal. The replacement coverage lives at
    # test_wrong_char_id_validates_against_cast_contract and
    # test_wrong_char_id_no_match_leaves_row_unrepaired (further
    # down in this module) which exercise the new lookup-then-write
    # contract on both the happy and the unmapped paths.

    def test_role_mismatch_overwritten(self, tmp_path):
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "Hello.", role="announcer"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="role_mismatch",
                found="announcer", expected="character", confidence=0.95,
            )
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 1
        assert led.data["lines"][0]["speaker_role"] == "character"

    def test_alias_used_substituted(self, tmp_path):
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "AL is in trouble.", role="character"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="alias_used",
                found="AL", expected="ALICE", confidence=0.7,
            )
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 1
        assert "ALICE" in led.data["lines"][0]["text"]

    def test_invented_name_auto_remap_close_match(self, tmp_path):
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "Allice was here.", role="character"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="invented_name",
                found="Allice", expected="", confidence=0.6,
            )
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 1
        # "Allice" -> "ALICE" (Levenshtein 1).
        assert "ALICE" in led.data["lines"][0]["text"]
        assert "Allice" not in led.data["lines"][0]["text"]

    def test_invented_name_no_match_left_in_place(self, tmp_path):
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "Dr. Patel arrives.", role="character"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="invented_name",
                found="Dr. Patel", expected="", confidence=0.9,
            )
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        # No remap (too far from any cast name) -- not counted as
        # repaired; Pass 2 Script Doctor owns the next step.
        # (S33 B4 retired the Step 2.5 phantom-skip safety net.)
        assert n == 0
        assert "Dr. Patel" in led.data["lines"][0]["text"]


# S33 B4 (2026-05-15): TestPhantomSkipFallback class deleted.
# `apply_phantom_skip_fallback` was retired per B1.5 classification
# (it muted lines via skip=True, a pipeline cut not a story edit).


# ---------------------------------------------------------------------------
# review_ledger -- end-to-end disposition tests for each verdict branch
# ---------------------------------------------------------------------------


class TestReviewLedgerDispositions:

    def test_clean_no_edits(self, tmp_path):
        """All passes clean, no edits proposed -> clean_no_edits."""
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "Hello there.", role="character"),
            _line("b002", "c02", "Likewise.", role="character"),
        ])
        # Pass 1 clean, Pass 2 clean no edits.
        # S33 B3 (2026-05-15): Pass 3 (post-audit) call retired; only
        # 2 LLM responses needed now.
        fn = _mk_generate_fn(
            _audit_clean_json(),
            _doctor_json(edits=[], verdict="clean"),
        )
        disp = review_ledger(fn, led)
        assert disp.verdict == "clean_no_edits"
        assert led.data["meta"]["reviewer_verdict"] == "clean_no_edits"

    def test_improved(self, tmp_path):
        """All passes clean, doctor proposes one rewrite -> improved."""
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "yeah I dunno", role="character"),
            _line("b002", "c02", "OK then.", role="character"),
        ])
        # S33 B3 (2026-05-15): Pass 3 (post-audit) call retired; only
        # 2 LLM responses needed now.
        fn = _mk_generate_fn(
            _audit_clean_json(),
            _doctor_json(edits=[{
                "line_id": "b001",
                "action": "rewrite",
                "payload": "I cannot say for certain.",
                "rationale": "voice fit",
            }], verdict="improved"),
        )
        disp = review_ledger(fn, led)
        assert disp.verdict == "improved"
        assert disp.doctor_edits_applied == 1
        # The rewrite landed on disk.
        b001 = next(r for r in led.data["lines"] if r["line_id"] == "b001")
        assert b001["text"] == "I cannot say for certain."

    # S33 B2 (2026-05-15): test_cast_unrecoverable removed --
    # speaker_unknowns rollback gate retired per refined no-auditors
    # rule. High-confidence speaker_unknown rows now flow into Phase 2
    # Script Doctor as ordinary violations.

    def test_too_many_edits(self, tmp_path):
        """Doctor proposes > edit_cap edits -> too_many_edits."""
        # 6 voiced beats -> edit_cap = max(3, 6//3) = 3
        lines = [
            _line(f"b{i:03d}", "c01", f"line {i}", role="character")
            for i in range(1, 7)
        ]
        led = _build_ledger(tmp_path, lines)
        # Doctor proposes 5 edits > cap 3.
        doctor_edits = [
            {"line_id": f"b{i:03d}", "action": "rewrite",
             "payload": f"new {i}", "rationale": "test"}
            for i in range(1, 6)
        ]
        fn = _mk_generate_fn(
            _audit_clean_json(),
            _doctor_json(edits=doctor_edits, verdict="improved"),
        )
        disp = review_ledger(fn, led)
        assert disp.verdict == "too_many_edits"
        # Original text preserved.
        for i in range(1, 7):
            row = next(r for r in led.data["lines"]
                       if r["line_id"] == f"b{i:03d}")
            assert row["text"] == f"line {i}"

    def test_needs_full_rerun(self, tmp_path):
        """Doctor verdict needs_full_rerun -> verdict propagates."""
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "broken structure", role="character"),
        ])
        fn = _mk_generate_fn(
            _audit_clean_json(),
            _doctor_json(edits=[], verdict="needs_full_rerun"),
        )
        disp = review_ledger(fn, led)
        assert disp.verdict == "needs_full_rerun"
        # Original preserved.
        assert led.data["lines"][0]["text"] == "broken structure"

    # S33 B2 (2026-05-15): test_post_audit_failed removed --
    # post_audit_pass rollback gate retired per refined no-auditors
    # rule. Per Jeffrey's phantom-ship policy, occasional phantoms
    # reaching the audience is the accepted trade-off vs preserving
    # the rollback.

    def test_skip_reviewer_bypass(self, tmp_path):
        """meta.skip_reviewer=True short-circuits the reviewer (G9)."""
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "untouched", role="character"),
        ])
        led.data.setdefault("meta", {})["skip_reviewer"] = True
        # No generate_fn calls should happen -- pass a fn that raises
        # if invoked.
        def boom(messages, *, temperature, max_new_tokens):
            raise AssertionError("reviewer should be bypassed")
        disp = review_ledger(boom, led)
        assert disp.verdict == "clean_no_edits"
        assert led.data["lines"][0]["text"] == "untouched"

    def test_pre_audit_llm_failure_maps_to_needs_full_rerun(self, tmp_path):
        """Wiring-review #8 (2026-05-11): pre-audit LLM failure maps
        to needs_full_rerun verdict (NOT clean_no_edits). Doctor
        + Pass 3 never run on garbage data."""
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "Hello there.", role="character"),
        ])
        call_count = {"n": 0}

        def fail_first_then_unreachable(messages, *, temperature, max_new_tokens):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated audit LLM OOM")
            raise AssertionError(
                "doctor / pass 3 must NOT run after pre-audit failure"
            )

        disp = review_ledger(fail_first_then_unreachable, led)
        assert disp.verdict == "needs_full_rerun"
        assert call_count["n"] == 1
        assert led.data["meta"]["reviewer_audit_failure_reason"]

    # S33 B4 (2026-05-15): test_phantom_skip_clears_text_belt_and_suspenders
    # deleted -- `apply_phantom_skip_fallback` retired per B1.5.

    def test_doctor_skip_action_clears_text(self, tmp_path):
        """Wiring-review #14: Script Doctor `skip` action also clears
        text in lockstep with skip=True."""
        from nodes._otr_ledger_reviewer import (  # noqa: E402
            apply_doctor_edits,
        )
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "Will be muted.", role="character"),
        ])
        report = ScriptDoctorReport(edits=[
            ReviewerEdit(
                line_id="b001", action="skip",
                payload="mute reason", rationale="x",
            ),
        ], overall_verdict="improved")
        applied = apply_doctor_edits(led.data, report, edit_cap=8)
        assert applied == 1
        row = led.data["lines"][0]
        assert row["skip"] is True
        assert row["text"] == ""

    def test_doctor_scope_guard_rejects_non_character_edits(self, tmp_path):
        """Wiring-review (Pass 2 scope guard): doctor edits targeting
        announcer / music / sfx beats are REJECTED, stamped on
        meta.reviewer_doctor_rejected_edits[], and do not count."""
        from nodes._otr_ledger_reviewer import (  # noqa: E402
            apply_doctor_edits,
        )
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "Character line.", role="character"),
            _line("b002", "announcer", "Announcer text.", role="announcer"),
        ])
        report = ScriptDoctorReport(edits=[
            ReviewerEdit(
                line_id="b001", action="rewrite",
                payload="OK to edit.", rationale="x",
            ),
            ReviewerEdit(
                line_id="b002", action="rewrite",
                payload="DOCTOR SHOULD NOT TOUCH",
                rationale="x",
            ),
        ], overall_verdict="improved")
        applied = apply_doctor_edits(led.data, report, edit_cap=8)
        # Only the character-targeted edit counts.
        assert applied == 1
        rows = {ln["line_id"]: ln for ln in led.data["lines"]}
        assert rows["b001"]["text"] == "OK to edit."
        assert rows["b002"]["text"] == "Announcer text."  # untouched
        rejected = led.data["meta"]["reviewer_doctor_rejected_edits"]
        assert any(r["line_id"] == "b002" for r in rejected)

    def test_wrong_char_id_validates_against_cast_contract(self, tmp_path):
        """Wiring-review #11 (2026-05-11): deterministic repair NEVER
        writes a raw LLM-suggested name as char_id. It must look up
        the canonical char_id in cast_contract. On miss, leave the
        row unrepaired."""
        led = _build_ledger(tmp_path, [
            _line("b001", "c99", "Hello.", role="character"),  # wrong cid
        ])
        # LLM suggests "ALICE" → should look up to "c01" and write c01.
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="wrong_char_id",
                found="c99", expected="ALICE", confidence=0.99,
            ),
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 1
        assert led.data["lines"][0]["char_id"] == "c01"

    def test_wrong_char_id_no_match_leaves_row_unrepaired(self, tmp_path):
        """Wiring-review #11: LLM suggests a name not in the cast →
        repair does NOT write the raw name; row stays as-is."""
        led = _build_ledger(tmp_path, [
            _line("b001", "c99", "Hello.", role="character"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="wrong_char_id",
                found="c99", expected="ZARGON",  # not in cast
                confidence=0.99,
            ),
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 0
        assert led.data["lines"][0]["char_id"] == "c99"  # unrepaired

    def test_role_mismatch_validates_against_enum(self, tmp_path):
        """Wiring-review #11: role_mismatch with valid enum value →
        written; invalid value → leave row alone."""
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "Hello.", role="announcer"),
        ])
        # Valid enum value: switch to character.
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="role_mismatch",
                found="announcer", expected="character",
                confidence=0.95,
            ),
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 1
        assert led.data["lines"][0]["speaker_role"] == "character"

    def test_role_mismatch_invalid_enum_leaves_unrepaired(self, tmp_path):
        """Wiring-review #11: invalid LLM-suggested role → leave alone."""
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "Hello.", role="announcer"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="role_mismatch",
                found="announcer", expected="lead_singer",  # not in enum
                confidence=0.95,
            ),
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 0
        assert led.data["lines"][0]["speaker_role"] == "announcer"
