"""tests/test_phase3_ledger_reviewer.py — two-pass cast-gated reviewer.

Covers (per script-writing-architecture synthesis §3 Phase 3 + §6.A + M2 + G1-G9):

  * Levenshtein auto_remap_phantom               §6.A + G8 test table
  * compute_edit_cap scales with voiced beats   G1
  * audit_cast_contract single-fn + label       G4 (Pass 1 + Pass 3 same fn)
  * apply_deterministic_cast_repairs            each repair kind branch
  * Sprint 3F deterministic resolution           confidence field removed;
        case-fold / Levenshtein resolution + ambiguous-tie escalation
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
    LineDiagnosis,
    PreAuditReport,
    ReviewerEdit,
    ScriptDoctorDiagnosis,
    ScriptDoctorReport,
    _render_cast_contract_table,
    _render_lines_for_doctor,
    _resolve_cast_member,
    apply_deterministic_cast_repairs,
    audit_cast_contract,
    auto_remap_phantom,
    compute_edit_cap,
    run_script_doctor,
    run_script_doctor_diagnosis,
    run_script_doctor_edits,
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


def test_cast_contract_table_never_renders_engine_name_as_role():
    """STEP 1+2 (2026-06-22 story+cast fix): the cast-contract table the
    auditor sees must NOT read the TTS engine name as the cast member's
    role (STEP 1), and must give the auditor a REAL contract role instead
    of '' (STEP 2 / roundtable Option A). Cast rows carry no speaker_role,
    so the role is derived: ANNOUNCER by name, else character. Cue roles
    (music_*/sfx) never appear here -- they are not cast members.
    """
    table = _render_cast_contract_table([
        {"char_id": "c01", "name": "ANNOUNCER",
         "speaker_role": "", "tts_model": "kokoro"},
        {"char_id": "c02", "name": "ALICE",
         "tts_model": "bark"},  # no speaker_role at all (real cast row shape)
    ])
    assert "kokoro" not in table
    assert "bark" not in table
    assert "role='announcer'" in table   # derived from ANNOUNCER name
    assert "role='character'" in table   # derived default for a cast member
    assert "role=''" not in table        # never the empty/engine-name role


def test_cast_contract_table_explicit_speaker_role_wins():
    """An explicit speaker_role on a cast row (future) wins over the
    name-derived default."""
    table = _render_cast_contract_table([
        {"char_id": "c01", "name": "MONTY", "speaker_role": "announcer"},
    ])
    assert "role='announcer'" in table


def test_lines_for_audit_lists_only_spoken_rows():
    """STEP 2 (roundtable Option A): the cast auditor sees ONLY spoken
    rows. music_*/sfx cue rows are line-level render contracts with no
    cast entry; auditing them invites spurious speaker_unknown /
    role_mismatch. The existing is_spoken_role predicate is the filter."""
    from nodes._otr_ledger_reviewer import _render_lines_for_audit
    rendered = _render_lines_for_audit([
        {"line_id": "l001", "speaker_role": "music_open", "char_id": "music_open",
         "text": ""},
        {"line_id": "l002", "speaker_role": "character", "char_id": "c01",
         "text": "We have to move."},
        {"line_id": "l003", "speaker_role": "announcer", "char_id": "announcer",
         "text": "Tonight on..."},
        {"line_id": "l004", "speaker_role": "sfx", "char_id": "sfx",
         "text": "door slams"},
    ])
    assert "l002" in rendered
    assert "l003" in rendered          # announcer is spoken
    assert "l001" not in rendered      # music cue excluded
    assert "l004" not in rendered      # sfx cue excluded


def test_lines_for_audit_all_cue_rows_yields_placeholder():
    """A ledger of only cue rows renders the no-spoken-lines placeholder,
    never an empty audit body."""
    from nodes._otr_ledger_reviewer import _render_lines_for_audit
    rendered = _render_lines_for_audit([
        {"line_id": "l001", "speaker_role": "music_open", "text": ""},
        {"line_id": "l002", "speaker_role": "sfx", "text": "thud"},
    ])
    assert "no spoken lines" in rendered
    assert "l001" not in rendered


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
    # Sprint 3F (2026-05-25): the auditor no longer emits a confidence
    # score -- pure anomaly extraction (line_id / kind / found /
    # expected) only.
    return json.dumps({
        "violations": [{
            "line_id": line_id,
            "kind": "invented_name",
            "found": phantom,
            "expected": "",
        }],
        "pass_clean": False,
    })


def _doctor_json(edits=None, verdict="clean"):
    return json.dumps({
        "edits": edits or [],
        "overall_verdict": verdict,
    })


def _diagnosis_json(diagnoses=None):
    """Sprint 3C: a ScriptDoctorDiagnosis JSON payload.

    `diagnoses` is a list of {line_id, failure, note} dicts. With no
    argument it yields an empty (all-clean) diagnosis.
    """
    return json.dumps({"diagnoses": diagnoses or []})


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
                found="alice", expected="ALICE",
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
                found="announcer", expected="character",
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
                found="AL", expected="ALICE",
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
                found="Allice", expected="",
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
                found="Dr. Patel", expected="",
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
        # Pass 1 audit clean; Sprint 3C Pass 2 = diagnosis (all `none`)
        # then edits (no edits). 3 LLM responses now.
        fn = _mk_generate_fn(
            _audit_clean_json(),
            _diagnosis_json([
                {"line_id": "b001", "failure": "none", "note": "sound"},
                {"line_id": "b002", "failure": "none", "note": "sound"},
            ]),
            _doctor_json(edits=[], verdict="clean"),
        )
        disp = review_ledger(fn, led)
        assert disp.verdict == "clean_no_edits"
        assert led.data["meta"]["reviewer_verdict"] == "clean_no_edits"

    def test_transport_failure_ships_unreviewed_not_rerun(self, tmp_path):
        """2026-06-20: a reviewer-LLM TRANSPORT failure (e.g. OpenRouter
        ~latest alias momentarily 404 'no endpoints') must FAIL SOFT --
        ship the writer's ledger unreviewed (clean_no_edits), NOT a terminal
        needs_full_rerun that CastLock would refuse. A cloud hiccup is not a
        story defect."""
        from nodes._otr_openrouter_backend import OpenRouterModelGoneError

        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "Hello there.", role="character"),
            _line("b002", "c02", "Likewise.", role="character"),
        ])

        def _gone(*a, **k):
            raise OpenRouterModelGoneError(
                "OpenRouter model ~openai/gpt-latest is gone (HTTP 404: "
                "No endpoints found)"
            )

        disp = review_ledger(_gone, led)
        assert disp.verdict == "clean_no_edits"
        assert led.data["meta"]["reviewer_verdict"] == "clean_no_edits"
        assert "reviewer_transport_skip" in led.data["meta"]

    def test_nontransport_llm_failure_still_rerun(self, tmp_path):
        """A NON-transport reviewer failure (e.g. a raising slot fn that is not
        an availability error) still maps to needs_full_rerun -- the fail-soft
        path is transport-only."""
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "Hello there.", role="character"),
        ])

        def _boom(*a, **k):
            raise ValueError("malformed model output")

        disp = review_ledger(_boom, led)
        assert disp.verdict == "needs_full_rerun"

    def test_improved(self, tmp_path):
        """All passes clean, doctor proposes one rewrite -> improved."""
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "yeah I dunno", role="character"),
            _line("b002", "c02", "OK then.", role="character"),
        ])
        # Sprint 3C: diagnosis flags b001 (flat_exposition); the edits
        # pass rewrites it. 3 LLM responses now.
        fn = _mk_generate_fn(
            _audit_clean_json(),
            _diagnosis_json([
                {"line_id": "b001", "failure": "flat_exposition",
                 "note": "no dramatic life"},
                {"line_id": "b002", "failure": "none", "note": "sound"},
            ]),
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
        # Doctor proposes 5 edits > cap 3. Sprint 3C: the diagnosis
        # must flag all 5 lines so the edits survive the undiagnosed-
        # edit guard and reach the cap check.
        doctor_edits = [
            {"line_id": f"b{i:03d}", "action": "rewrite",
             "payload": f"new {i}", "rationale": "test"}
            for i in range(1, 6)
        ]
        diagnoses = [
            {"line_id": f"b{i:03d}", "failure": "pacing", "note": "drags"}
            for i in range(1, 6)
        ]
        fn = _mk_generate_fn(
            _audit_clean_json(),
            _diagnosis_json(diagnoses),
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
        """Doctor edits-pass verdict needs_full_rerun -> propagates."""
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "broken structure", role="character"),
        ])
        # Sprint 3C: audit clean, diagnosis runs, edits pass returns the
        # needs_full_rerun verdict.
        fn = _mk_generate_fn(
            _audit_clean_json(),
            _diagnosis_json([
                {"line_id": "b001", "failure": "arc", "note": "inert"},
            ]),
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
                found="c99", expected="ALICE",
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
            ),
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 0
        assert led.data["lines"][0]["char_id"] == "c99"  # unrepaired

    def test_wrong_char_id_charid_shaped_expected_repairs(self, tmp_path):
        """BUG-LOCAL-271: the cast auditor emits a CHAR_ID in `expected` (not a
        name). The repair must validate it directly against the locked cast's
        char_id set and write it. The prior code only resolved `expected` as a
        NAME, so every char_id-shaped suggestion missed and the repair was dead
        (all lines escalated to the Script Doctor, which flagged nothing)."""
        led = _build_ledger(tmp_path, [
            _line("b001", "c99", "Hello.", role="character"),  # wrong cid
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="wrong_char_id",
                found="c99", expected="c01",  # auditor emits a CHAR_ID, not a name
            ),
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 1
        assert led.data["lines"][0]["char_id"] == "c01"

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
            ),
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 0
        assert led.data["lines"][0]["speaker_role"] == "announcer"


# ---------------------------------------------------------------------------
# Sprint 3F (2026-05-25) -- confidence field removed; Python decides
# replacement deterministically (exact case-fold / Levenshtein <= 3),
# escalating ambiguous ties exactly as it escalates an unresolved
# violation (row left for the Script Doctor, never silently picked).
# ---------------------------------------------------------------------------


class TestCastViolationNoConfidenceField:
    """The pydantic model must no longer carry a confidence score."""

    def test_confidence_field_removed_from_model(self):
        assert "confidence" not in CastViolation.model_fields

    def test_violation_constructs_without_confidence(self):
        v = CastViolation(
            line_id="b001", kind="bad_casing",
            found="alice", expected="ALICE",
        )
        assert v.found == "alice"
        assert v.expected == "ALICE"
        assert not hasattr(v, "confidence")

    def test_auditor_prompt_drops_confidence(self):
        from nodes._otr_ledger_reviewer import _AUDITOR_SYSTEM_PROMPT
        assert "confidence" not in _AUDITOR_SYSTEM_PROMPT.lower()


class TestResolveCastMember:
    """`_resolve_cast_member` -- exact case-fold first, then Levenshtein
    via the existing `auto_remap_phantom` matcher; ambiguous ties to
    None."""

    def test_exact_case_fold_match_returns_canonical_spelling(self):
        # Auditor sends a lowercased name -> resolver returns the
        # canonical roster spelling, not the auditor's casing.
        assert _resolve_cast_member("alice", ["ALICE", "BOB"]) == "ALICE"
        assert _resolve_cast_member("ALICE", ["ALICE", "BOB"]) == "ALICE"

    def test_levenshtein_fallback_resolves_close_typo(self):
        # No exact match; Levenshtein 1 -> resolves.
        assert _resolve_cast_member("Allice", ["ALICE", "BOB"]) == "ALICE"

    def test_far_name_returns_none(self):
        assert _resolve_cast_member("Zargon", ["ALICE", "BOB"]) is None

    def test_empty_candidate_returns_none(self):
        assert _resolve_cast_member("", ["ALICE", "BOB"]) is None

    def test_empty_roster_returns_none(self):
        assert _resolve_cast_member("ALICE", []) is None

    def test_ambiguous_levenshtein_tie_returns_none(self):
        # "CARO" is Levenshtein 1 from both "CARL" and "CARA" --
        # an ambiguous tie escalates to None rather than guessing.
        assert _resolve_cast_member("CARO", ["CARL", "CARA"]) is None


class TestDeterministicRepairResolution:
    """Sprint 3F: `apply_deterministic_cast_repairs` resolves the
    auditor's `expected` deterministically -- no confidence gate."""

    def test_bad_casing_resolves_lowercase_expected(self, tmp_path):
        # Auditor emits a lowercased `expected`; the resolver maps it
        # to the canonical roster spelling and the repair lands.
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "alice waits.", role="character"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="bad_casing",
                found="alice", expected="alice",
            )
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 1
        assert led.data["lines"][0]["text"] == "ALICE waits."

    def test_bad_casing_typo_expected_resolves_via_levenshtein(self, tmp_path):
        # Auditor's `expected` is itself slightly misspelled
        # ("ALICEE"); Levenshtein <= 3 still resolves it to ALICE.
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "alice waits.", role="character"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="bad_casing",
                found="alice", expected="ALICEE",
            )
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 1
        assert led.data["lines"][0]["text"] == "ALICE waits."

    def test_bad_casing_unresolvable_expected_escalates(self, tmp_path):
        # Auditor's `expected` is not a cast member and not close to
        # one -- the repair is NOT applied; the row is left untouched
        # for the Script Doctor.
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "alice waits.", role="character"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="bad_casing",
                found="alice", expected="Zargon",
            )
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 0
        assert led.data["lines"][0]["text"] == "alice waits."

    def test_alias_used_resolves_expected_to_cast_member(self, tmp_path):
        # Lowercased `expected` for an alias still resolves.
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "AL is in trouble.", role="character"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="alias_used",
                found="AL", expected="alice",
            )
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 1
        assert "ALICE" in led.data["lines"][0]["text"]

    def test_alias_used_unresolvable_expected_escalates(self, tmp_path):
        # Alias the auditor maps to a non-cast name -> escalate, do
        # not write an unverified literal.
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "AL is in trouble.", role="character"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="alias_used",
                found="AL", expected="Mxyzptlk",
            )
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 0
        assert led.data["lines"][0]["text"] == "AL is in trouble."

    def test_wrong_char_id_resolves_typo_expected(self, tmp_path):
        # Auditor suggests a slightly misspelled name; Levenshtein
        # resolves it to the canonical cast member, then the canonical
        # char_id is written.
        led = _build_ledger(tmp_path, [
            _line("b001", "c99", "Hello.", role="character"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="wrong_char_id",
                found="c99", expected="ALICEE",  # typo of ALICE
            ),
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 1
        assert led.data["lines"][0]["char_id"] == "c01"

    def test_wrong_char_id_no_confidence_gate(self, tmp_path):
        # Old behavior gated wrong_char_id on confidence >= 0.9.
        # Sprint 3F removed the gate -- a clean exact-name match
        # repairs regardless of any (now absent) score.
        led = _build_ledger(tmp_path, [
            _line("b001", "c99", "Hello.", role="character"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="wrong_char_id",
                found="c99", expected="ALICE",
            ),
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 1
        assert led.data["lines"][0]["char_id"] == "c01"

    def test_bad_casing_no_confidence_gate(self, tmp_path):
        # Old behavior gated bad_casing on confidence >= 0.8. Sprint 3F
        # removed the gate -- the repair lands on a resolvable
        # `expected` regardless of any (now absent) score.
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "bob speaks.", role="character"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="bad_casing",
                found="bob", expected="BOB",
            )
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 1
        assert led.data["lines"][0]["text"] == "BOB speaks."


class TestAmbiguousTieEscalation:
    """An ambiguous tie (two cast members equally close) must escalate
    -- the row is left for the Script Doctor, never silently picked."""

    def _build_tie_ledger(self, tmp_path, lines):
        """Ledger whose cast has two near-identical names so a typo
        can sit equidistant between them."""
        d = tmp_path / "audio"
        d.mkdir(parents=True, exist_ok=True)
        led = PL.Ledger(episode_id="pending_tie", out_dir=str(d))
        led.set_cast([
            {"char_id": "c01", "name": "CARL"},
            {"char_id": "c02", "name": "CARA"},
            {"char_id": "c03", "name": "ANNOUNCER"},
        ])
        led.data["lines"] = lines
        return led

    def test_bad_casing_ambiguous_tie_left_for_doctor(self, tmp_path):
        # `expected="CARO"` is Levenshtein 1 from both CARL and CARA.
        # No exact match, ambiguous tie -> repair NOT applied.
        led = self._build_tie_ledger(tmp_path, [
            _line("b001", "c01", "caro speaks.", role="character"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="bad_casing",
                found="caro", expected="CARO",
            )
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 0
        assert led.data["lines"][0]["text"] == "caro speaks."

    def test_wrong_char_id_ambiguous_tie_left_for_doctor(self, tmp_path):
        # Ambiguous `expected` -> no char_id written; row escalates.
        led = self._build_tie_ledger(tmp_path, [
            _line("b001", "c99", "Hello.", role="character"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="wrong_char_id",
                found="c99", expected="CARO",  # tie between CARL/CARA
            ),
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 0
        assert led.data["lines"][0]["char_id"] == "c99"

    def test_invented_name_ambiguous_tie_left_in_place(self, tmp_path):
        # invented_name routes through auto_remap_phantom, which
        # already returns None on a tie -- the phantom stays put.
        led = self._build_tie_ledger(tmp_path, [
            _line("b001", "c01", "Caro arrives.", role="character"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="invented_name",
                found="Caro", expected="",
            )
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(
            led.data, report, led.data["cast"],
        )
        assert n == 0
        assert "Caro" in led.data["lines"][0]["text"]


# ---------------------------------------------------------------------------
# Sprint 3C (2026-05-25) -- Script Doctor split: diagnosis then edits.
#
#   * Enriched Doctor input rows carry beat_id / arc_phase / mood /
#     actual_words / text (pulled from ledger per-line state).
#   * run_script_doctor_diagnosis NAMES the per-line failure, no edits.
#   * run_script_doctor_edits emits the edits array and cannot rewrite
#     a line the diagnosis did not flag -- enforced deterministically.
#   * run_script_doctor orchestrates diagnosis -> edits and preserves
#     the never-raises / needs_full_rerun loud-failure contract.
# ---------------------------------------------------------------------------


def _doctor_line(line_id, char_id, text, *, beat_id=None, arc_phase=None,
                 mood=None, word_count=None):
    """A character line with the per-line state the Doctor enriches on."""
    row = _line(line_id, char_id, text, role="character", arc_phase=arc_phase)
    row["beat_id"] = beat_id or line_id
    # The writer stamps beat.mood into the line's `traits` field.
    row["traits"] = mood
    if word_count is not None:
        row["word_count"] = word_count
    return row


class TestEnrichedDoctorRows:
    """The Sprint 3C Doctor rows must carry the per-line narrative
    context the cast-auditor rows never did."""

    def test_rows_carry_beat_id_arc_phase_mood_actual_words(self):
        lines = [
            _doctor_line(
                "b001", "c01", "We have to move now.",
                beat_id="beat_07", arc_phase="rising_tension",
                mood="urgent", word_count=5,
            ),
        ]
        rendered = _render_lines_for_doctor(lines)
        assert "beat_id=beat_07" in rendered
        assert "arc_phase=rising_tension" in rendered
        assert "mood=urgent" in rendered
        assert "actual_words=5" in rendered
        assert "We have to move now." in rendered

    def test_mood_pulled_from_traits_field(self):
        # The writer stamps mood into `traits`; the renderer reads it.
        lines = [_doctor_line("b001", "c01", "Hello.", mood="wistful")]
        assert "mood=wistful" in _render_lines_for_doctor(lines)

    def test_actual_words_falls_back_to_word_split(self):
        # No word_count stamped -> renderer counts the text itself.
        row = _line("b001", "c01", "one two three four", role="character")
        row["word_count"] = None
        assert "actual_words=4" in _render_lines_for_doctor([row])

    def test_missing_fields_render_as_placeholder_not_fabricated(self):
        # arc_phase / mood absent -> a neutral '(unset)' marker, never
        # an invented value.
        row = _line("b001", "c01", "Hello.", role="character")
        row["arc_phase"] = None
        row["traits"] = None
        rendered = _render_lines_for_doctor([row])
        assert "arc_phase=(unset)" in rendered
        assert "mood=(unset)" in rendered

    def test_beat_intent_rendered_only_when_present(self):
        # beat_intent is not on the ledger today; the renderer omits it
        # rather than fabricating. When a future stamping change adds
        # it to the line dict, the renderer surfaces it.
        without = _render_lines_for_doctor([
            _doctor_line("b001", "c01", "Hello."),
        ])
        assert "beat_intent" not in without
        row = _doctor_line("b002", "c01", "Hello.")
        row["beat_intent"] = "introduce the threat"
        with_intent = _render_lines_for_doctor([row])
        assert "beat_intent='introduce the threat'" in with_intent

    def test_doctor_renderer_distinct_from_auditor_renderer(self):
        # The cast-auditor renderer is unchanged: it must NOT carry the
        # enriched fields (Sprint 3C only touched the Doctor path).
        from nodes._otr_ledger_reviewer import _render_lines_for_audit
        lines = [_doctor_line("b001", "c01", "Hello.", arc_phase="climax")]
        audit_rendered = _render_lines_for_audit(lines)
        assert "arc_phase" not in audit_rendered
        assert "speaker_role" in audit_rendered

    def test_doctor_renderer_target_tension_optional(self):
        """STEP 6 (2026-06-22): _render_lines_for_doctor renders
        target_tension=N/5 ONLY when a tension map is supplied (the critic
        path); with no map (the Script Doctor path) it is byte-identical."""
        from nodes._otr_ledger_reviewer import _render_lines_for_doctor
        lines = [_doctor_line("b001", "c01", "Hello."),
                 _doctor_line("b002", "c02", "Hi.")]
        # no map -> no tension column (Doctor path unchanged)
        assert "target_tension" not in _render_lines_for_doctor(lines)
        # with a map -> the column appears for the mapped line only
        rendered = _render_lines_for_doctor(lines, {"b001": 4})
        assert "target_tension=4/5" in rendered
        assert rendered.count("target_tension") == 1   # b002 had no target
        # out-of-range / non-int targets are ignored
        assert "target_tension" not in _render_lines_for_doctor(
            lines, {"b001": 9, "b002": "x"})


class TestScriptDoctorDiagnosisPass:
    """run_script_doctor_diagnosis NAMES the per-line failure."""

    def test_diagnosis_names_failures(self, tmp_path):
        led = _build_ledger(tmp_path, [
            _doctor_line("b001", "c01", "yeah I dunno", mood="flat"),
            _doctor_line("b002", "c02", "OK then.", mood="flat"),
        ])
        fn = _mk_generate_fn(_diagnosis_json([
            {"line_id": "b001", "failure": "flat_exposition",
             "note": "states a fact, no life"},
            {"line_id": "b002", "failure": "none", "note": "sound"},
        ]))
        diagnosis = run_script_doctor_diagnosis(
            fn, led.data, led.data["cast"],
        )
        assert isinstance(diagnosis, ScriptDoctorDiagnosis)
        assert len(diagnosis.diagnoses) == 2
        by_id = {d.line_id: d for d in diagnosis.diagnoses}
        assert by_id["b001"].failure == "flat_exposition"
        assert by_id["b002"].failure == "none"

    def test_diagnosis_produces_no_edits(self, tmp_path):
        # The diagnosis model has no `edits` field at all -- the pass
        # cannot emit edits by construction.
        assert "edits" not in ScriptDoctorDiagnosis.model_fields
        assert "diagnoses" in ScriptDoctorDiagnosis.model_fields

    def test_diagnosis_llm_failure_returns_none(self, tmp_path):
        led = _build_ledger(tmp_path, [
            _doctor_line("b001", "c01", "Hello."),
        ])

        def boom(messages, *, temperature, max_new_tokens):
            raise RuntimeError("simulated diagnosis OOM")

        diagnosis = run_script_doctor_diagnosis(
            boom, led.data, led.data["cast"],
        )
        assert diagnosis is None

    def test_diagnosis_malformed_json_returns_none(self, tmp_path):
        led = _build_ledger(tmp_path, [
            _doctor_line("b001", "c01", "Hello."),
        ])
        fn = _mk_generate_fn("not json at all")
        diagnosis = run_script_doctor_diagnosis(
            fn, led.data, led.data["cast"],
        )
        assert diagnosis is None


class TestScriptDoctorEditsPass:
    """run_script_doctor_edits emits the edits array and may NOT touch
    a line the diagnosis did not flag."""

    def test_edits_pass_can_edit_a_diagnosed_line(self, tmp_path):
        led = _build_ledger(tmp_path, [
            _doctor_line("b001", "c01", "yeah I dunno"),
        ])
        diagnosis = ScriptDoctorDiagnosis(diagnoses=[
            LineDiagnosis(line_id="b001", failure="voice_drift",
                          note="off register"),
        ])
        fn = _mk_generate_fn(_doctor_json(edits=[{
            "line_id": "b001", "action": "rewrite",
            "payload": "I cannot say.", "rationale": "voice fit",
        }], verdict="improved"))
        report = run_script_doctor_edits(
            fn, led.data, led.data["cast"], diagnosis, edit_cap=8,
        )
        assert len(report.edits) == 1
        assert report.edits[0].line_id == "b001"

    def test_edits_pass_cannot_touch_an_undiagnosed_line(self, tmp_path):
        # The diagnosis flagged ONLY b001. The edits pass (a misbehaving
        # model) also proposes an edit on b002 -- which the diagnosis
        # marked `none`. The deterministic guard must drop the b002 edit.
        led = _build_ledger(tmp_path, [
            _doctor_line("b001", "c01", "yeah I dunno"),
            _doctor_line("b002", "c02", "A perfectly fine line."),
        ])
        diagnosis = ScriptDoctorDiagnosis(diagnoses=[
            LineDiagnosis(line_id="b001", failure="flat_exposition",
                          note="flat"),
            LineDiagnosis(line_id="b002", failure="none", note="sound"),
        ])
        fn = _mk_generate_fn(_doctor_json(edits=[
            {"line_id": "b001", "action": "rewrite",
             "payload": "A real line.", "rationale": "fix flat"},
            {"line_id": "b002", "action": "rewrite",
             "payload": "MEDDLING WITH A SOUND LINE",
             "rationale": "should not happen"},
        ], verdict="improved"))
        report = run_script_doctor_edits(
            fn, led.data, led.data["cast"], diagnosis, edit_cap=8,
        )
        # Only the diagnosed line's edit survives the guard.
        kept_ids = {e.line_id for e in report.edits}
        assert kept_ids == {"b001"}

    def test_edits_pass_drops_edit_on_line_with_no_diagnosis_row(self, tmp_path):
        # b002 has NO diagnosis row at all (not even `none`). An edit on
        # it must be dropped exactly as a `none`-diagnosed edit is.
        led = _build_ledger(tmp_path, [
            _doctor_line("b001", "c01", "yeah I dunno"),
            _doctor_line("b002", "c02", "Untouched."),
        ])
        diagnosis = ScriptDoctorDiagnosis(diagnoses=[
            LineDiagnosis(line_id="b001", failure="pacing", note="drags"),
        ])
        fn = _mk_generate_fn(_doctor_json(edits=[
            {"line_id": "b002", "action": "rewrite",
             "payload": "SHOULD BE DROPPED", "rationale": "x"},
        ], verdict="improved"))
        report = run_script_doctor_edits(
            fn, led.data, led.data["cast"], diagnosis, edit_cap=8,
        )
        assert report.edits == []

    def test_edits_pass_llm_failure_returns_needs_full_rerun(self, tmp_path):
        led = _build_ledger(tmp_path, [
            _doctor_line("b001", "c01", "Hello."),
        ])
        diagnosis = ScriptDoctorDiagnosis(diagnoses=[
            LineDiagnosis(line_id="b001", failure="arc", note="inert"),
        ])

        def boom(messages, *, temperature, max_new_tokens):
            raise RuntimeError("simulated edits OOM")

        report = run_script_doctor_edits(
            boom, led.data, led.data["cast"], diagnosis, edit_cap=8,
        )
        assert report.overall_verdict == "needs_full_rerun"


class TestScriptDoctorOrchestrator:
    """run_script_doctor chains diagnosis -> edits and preserves the
    never-raises / needs_full_rerun contract."""

    def test_orchestrator_diagnosis_then_edits(self, tmp_path):
        led = _build_ledger(tmp_path, [
            _doctor_line("b001", "c01", "yeah I dunno"),
        ])
        # Call 1 = diagnosis, call 2 = edits.
        fn = _mk_generate_fn(
            _diagnosis_json([
                {"line_id": "b001", "failure": "flat_exposition",
                 "note": "flat"},
            ]),
            _doctor_json(edits=[{
                "line_id": "b001", "action": "rewrite",
                "payload": "A real line.", "rationale": "fix flat",
            }], verdict="improved"),
        )
        report = run_script_doctor(fn, led.data, led.data["cast"], 8)
        assert isinstance(report, ScriptDoctorReport)
        assert len(report.edits) == 1
        assert report.edits[0].line_id == "b001"

    def test_orchestrator_drops_undiagnosed_edit_end_to_end(self, tmp_path):
        # End-to-end: the edits pass over-reaches; the orchestrator's
        # edits pass drops the undiagnosed edit.
        led = _build_ledger(tmp_path, [
            _doctor_line("b001", "c01", "yeah I dunno"),
            _doctor_line("b002", "c02", "Fine line."),
        ])
        fn = _mk_generate_fn(
            _diagnosis_json([
                {"line_id": "b001", "failure": "pacing", "note": "drags"},
                {"line_id": "b002", "failure": "none", "note": "sound"},
            ]),
            _doctor_json(edits=[
                {"line_id": "b001", "action": "rewrite",
                 "payload": "Tighter.", "rationale": "pace"},
                {"line_id": "b002", "action": "rewrite",
                 "payload": "MEDDLE", "rationale": "x"},
            ], verdict="improved"),
        )
        report = run_script_doctor(fn, led.data, led.data["cast"], 8)
        assert {e.line_id for e in report.edits} == {"b001"}

    def test_orchestrator_never_raises_on_diagnosis_failure(self, tmp_path):
        # Diagnosis pass fails -> orchestrator returns needs_full_rerun,
        # never raises.
        led = _build_ledger(tmp_path, [
            _doctor_line("b001", "c01", "Hello."),
        ])

        def boom(messages, *, temperature, max_new_tokens):
            raise RuntimeError("simulated diagnosis crash")

        report = run_script_doctor(boom, led.data, led.data["cast"], 8)
        assert isinstance(report, ScriptDoctorReport)
        assert report.overall_verdict == "needs_full_rerun"

    def test_orchestrator_never_raises_on_edits_failure(self, tmp_path):
        # Diagnosis succeeds, edits pass fails -> needs_full_rerun.
        led = _build_ledger(tmp_path, [
            _doctor_line("b001", "c01", "Hello."),
        ])
        call_count = {"n": 0}

        def fail_on_edits(messages, *, temperature, max_new_tokens):
            i = call_count["n"]
            call_count["n"] += 1
            if i == 0:
                return _diagnosis_json([
                    {"line_id": "b001", "failure": "arc", "note": "inert"},
                ])
            raise RuntimeError("simulated edits crash")

        report = run_script_doctor(fail_on_edits, led.data, led.data["cast"], 8)
        assert report.overall_verdict == "needs_full_rerun"

    def test_orchestrator_malformed_diagnosis_json_needs_full_rerun(self, tmp_path):
        led = _build_ledger(tmp_path, [
            _doctor_line("b001", "c01", "Hello."),
        ])
        fn = _mk_generate_fn("not json at all")
        report = run_script_doctor(fn, led.data, led.data["cast"], 8)
        assert report.overall_verdict == "needs_full_rerun"


class TestBugLocal276RefuseAnnouncerOnCharacterLine:
    """BUG-LOCAL-276 family (2026-05-29): the deterministic wrong_char_id
    repair must NEVER remap a speaker_role='character' line onto the
    announcer's cast row (a Kokoro / non-v2 voice). Honoring such an
    auditor suggestion stamps an unrenderable voice on a Bark line and
    hard-crashes BatchBarkGenerator at render. The row is left for the
    Script Doctor instead, so the locked character speaker stays constant
    and announcer lines are never managed onto a character.

    _build_ledger cast: c01=ALICE, c02=BOB, c03=ANNOUNCER (no voice_preset).
    """

    def test_character_line_not_remapped_onto_announcer(self, tmp_path):
        led = _build_ledger(tmp_path, [
            _line("b001", "c01", "Hello there.", role="character"),
        ])
        # Auditor wrongly suggests the ANNOUNCER's char_id (c03) for a
        # character-role line.
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="wrong_char_id",
                found="c01", expected="c03",
            ),
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(led.data, report, led.data["cast"])
        # Refused: char_id unchanged (NOT the announcer's c03), not counted.
        assert led.data["lines"][0]["char_id"] == "c01"
        assert n == 0

    def test_character_to_character_remap_still_works(self, tmp_path):
        # Positive control: a legitimate character->character remap is
        # unaffected by the announcer guard.
        led = _build_ledger(tmp_path, [
            _line("b001", "c99", "Hello there.", role="character"),
        ])
        report = PreAuditReport(violations=[
            CastViolation(
                line_id="b001", kind="wrong_char_id",
                found="c99", expected="c02",
            ),
        ], pass_clean=False)
        n = apply_deterministic_cast_repairs(led.data, report, led.data["cast"])
        assert led.data["lines"][0]["char_id"] == "c02"
        assert n == 1
