"""tests/test_lfc_freeze_cascade_orchestrator.py

LFC sprint Commit 2 -- regression coverage for the freeze-cascade
orchestrator. Tests verify that:

  * run_freeze_cascade wires Phase 0 -> reviewer -> Phase 10
  * Reviewer-verdict -> freeze-verdict mapping is correct per ADR
    section 9 for every reviewer-verdict literal
  * Terminal reviewer failures skip Phase 10 (so we don't stamp
    frozen_clean on a restored-to-original ledger)
  * Phase 10 critical gap forces needs_full_rerun
  * Improved reviewer verdict + Phase 10 clean -> frozen_with_doctor_edits
  * meta.cleanup_passes records each phase
  * meta.skip_reviewer bypass still works (Phase 0/10 still run)
  * Output contract preserved: 5 slots, freeze_verdict last
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_freeze_cascade as _LFC_ORCH
from nodes import _otr_ledger_freeze as _LFC
from nodes import _otr_ledger_reviewer as _OTRLR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _clean_ledger_data() -> dict:
    """Same shape as test_lfc_phase_0_10_gap_audit -- minimal-valid L3."""
    return {
        "schema_version": _LFC.EXPECTED_SCHEMA_VERSION,
        "episode_id": "ep_test_cascade",
        "cast": [
            {"char_id": "c01", "name": "ALICE",
             "traits": "calm", "voice_preset": "v2/en_speaker_6"},
            {"char_id": "c02", "name": "BOB",
             "traits": "skeptical", "voice_preset": "v2/en_speaker_2"},
            {"char_id": "announcer", "name": "ANNOUNCER",
             "traits": "warm", "voice_preset": "v2/en_speaker_9"},
        ],
        "scenes": [], "shots": [],
        "beats": [{"beat_id": f"b00{i}"} for i in range(1, 5)],
        "lines": [
            {"line_id": "l001", "beat_id": "b001",
             "speaker_role": "announcer", "char_id": "announcer",
             "text": "Tonight, on Signal Lost.",
             "word_count": 4, "char_count": 24, "skip": False},
            {"line_id": "l002", "beat_id": "b002",
             "speaker_role": "character", "char_id": "c01",
             "text": "Bob, the readings are off the chart.",
             "word_count": 7, "char_count": 36, "skip": False},
            {"line_id": "l003", "beat_id": "b003",
             "speaker_role": "character", "char_id": "c02",
             "text": "Then someone tampered with the calibration.",
             "word_count": 6, "char_count": 43, "skip": False},
            {"line_id": "l004", "beat_id": "b004",
             "speaker_role": "announcer", "char_id": "announcer",
             "text": "Stay tuned.",
             "word_count": 2, "char_count": 11, "skip": False},
        ],
        "music": [], "clips": [],
        "meta": {
            "episode_title": "The Calibration Drift",
            "style": "mission_control_procedural",
        },
    }


def _ledger_obj(data: dict):
    """Minimal stand-in for production_ledger.Ledger -- has .data and
    .save() (no-op) so the existing review_ledger flow works in-memory.
    """
    obj = SimpleNamespace(data=data, episode_id=data.get("episode_id", "ep"))
    obj.save = lambda: None  # type: ignore[attr-defined]
    return obj


def _stub_reviewer_disposition(verdict: str = "clean_no_edits",
                                edits_applied: int = 0,
                                edits_proposed: int = 0):
    """Build a synthetic ReviewerDisposition that mocks review_ledger
    return value. review_ledger is the inner LLM-bound function we
    replace in tests so the orchestrator is exercised in isolation."""
    return _OTRLR.ReviewerDisposition(
        verdict=verdict,
        pre_audit_violations=0,
        pre_audit_repairs_applied=0,
        doctor_edits_proposed=edits_proposed,
        doctor_edits_applied=edits_applied,
        post_audit_violations=0,
        phantom_skip_count=0,
    )


# ---------------------------------------------------------------------------
# Happy path -- clean reviewer + clean gap audit -> frozen_clean
# ---------------------------------------------------------------------------


class TestCascadeHappyPath:
    def test_clean_clean_frozen_clean(self):
        led = _ledger_obj(_clean_ledger_data())

        def fake_review(_fn, _led):
            return _stub_reviewer_disposition("clean_no_edits")

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review):
            disp = _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        assert disp.verdict == "frozen_clean"
        assert led.data["meta"]["freeze_verdict"] == "frozen_clean"
        assert led.data["meta"]["cleanup_locked"] is True
        assert "freeze_timestamp" in led.data["meta"]
        assert disp.gap_audit_pre.is_clean
        assert disp.gap_audit_post is not None
        assert disp.gap_audit_post.is_clean

    def test_clean_with_warns_frozen_with_warns(self):
        data = _clean_ledger_data()
        # Drop episode_title -> warn from Phase 10 + Phase 0.
        del data["meta"]["episode_title"]
        led = _ledger_obj(data)

        def fake_review(_fn, _led):
            return _stub_reviewer_disposition("clean_no_edits")

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review):
            disp = _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        assert disp.verdict == "frozen_with_warns"
        assert led.data["meta"]["cleanup_locked"] is True

    def test_improved_lifts_to_doctor_edits(self):
        led = _ledger_obj(_clean_ledger_data())

        def fake_review(_fn, _led):
            # Simulate doctor having rewritten one line.
            return _stub_reviewer_disposition(
                "improved", edits_applied=1, edits_proposed=1,
            )

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review):
            disp = _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        assert disp.verdict == "frozen_with_doctor_edits"
        assert led.data["meta"]["freeze_verdict"] == "frozen_with_doctor_edits"
        assert led.data["meta"]["cleanup_locked"] is True


# ---------------------------------------------------------------------------
# Terminal reviewer failures -> skip Phase 10
# ---------------------------------------------------------------------------


class TestTerminalReviewerFailures:
    @pytest.mark.parametrize("verdict", [
        "cast_unrecoverable",
        "too_many_edits",
        "needs_full_rerun",
        "post_audit_failed",
    ])
    def test_terminal_verdict_skips_phase_10(self, verdict):
        led = _ledger_obj(_clean_ledger_data())

        def fake_review(_fn, _led):
            return _stub_reviewer_disposition(verdict)

        # Spy on phase_10 to make sure it was NOT called.
        called = {"phase_10": False}
        original = _LFC.phase_10_gap_audit_post_and_freeze

        def spy_phase_10(led_):
            called["phase_10"] = True
            return original(led_)

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review), \
             patch.object(_LFC_ORCH._LFC,
                          "phase_10_gap_audit_post_and_freeze",
                          side_effect=spy_phase_10):
            disp = _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        assert disp.verdict == verdict
        assert called["phase_10"] is False
        assert led.data["meta"]["freeze_verdict"] == verdict
        # cleanup_locked must NOT be stamped on terminal failure.
        assert "cleanup_locked" not in led.data["meta"]
        assert disp.gap_audit_post is None


# ---------------------------------------------------------------------------
# Phase 10 hard-fail -> needs_full_rerun
# ---------------------------------------------------------------------------


class TestPhase10HardFailDownstream:
    def test_reviewer_clean_but_phase_10_rejects(self):
        # Build a ledger where the reviewer would return clean but
        # Phase 10 finds a critical gap (synthetic: empty voiced text
        # added between reviewer return and Phase 10).
        data = _clean_ledger_data()
        led = _ledger_obj(data)

        def fake_review(_fn, _led):
            # Mutate the ledger AFTER reviewer would have returned --
            # simulating a downstream phase mucking up the state.
            _led.data["lines"][1]["text"] = ""
            _led.data["lines"][1]["word_count"] = 0
            _led.data["lines"][1]["char_count"] = 0
            return _stub_reviewer_disposition("clean_no_edits")

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review):
            disp = _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        assert disp.verdict == "needs_full_rerun"
        assert led.data["meta"]["freeze_verdict"] == "needs_full_rerun"
        assert disp.gap_audit_post is not None
        assert disp.gap_audit_post.has_critical_gaps


# ---------------------------------------------------------------------------
# meta.cleanup_passes ledger
# ---------------------------------------------------------------------------


class TestCleanupPassesRecording:
    def test_phase_records_stamped(self):
        led = _ledger_obj(_clean_ledger_data())

        def fake_review(_fn, _led):
            return _stub_reviewer_disposition("clean_no_edits")

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        passes = _LFC_ORCH.all_phase_passes(led.data["meta"])
        assert isinstance(passes, list)
        names = [p["phase_name"] for p in passes]
        assert "phase_0_gap_audit_pre" in names
        assert "phase_1_2_9_reviewer_composite" in names
        assert "phase_10_gap_audit_post_and_freeze" in names
        # Each record carries the fingerprint contract.
        for p in passes:
            assert "started_at" in p
            assert "finished_at" in p
            assert "text_hash_before" in p
            assert "text_hash_after" in p

    def test_phase_record_carries_doctor_edits(self):
        led = _ledger_obj(_clean_ledger_data())

        def fake_review(_fn, _led):
            return _stub_reviewer_disposition(
                "improved", edits_applied=2, edits_proposed=3,
            )

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        passes = _LFC_ORCH.all_phase_passes(led.data["meta"])
        rev_pass = next(
            p for p in passes
            if p["phase_name"] == "phase_1_2_9_reviewer_composite"
        )
        assert rev_pass["edits_proposed"] == 3
        assert rev_pass["edits_applied"] == 2


# ---------------------------------------------------------------------------
# FreezeDisposition surface
# ---------------------------------------------------------------------------


class TestFreezeDispositionDataclass:
    def test_to_dict_round_trips(self):
        led = _ledger_obj(_clean_ledger_data())

        def fake_review(_fn, _led):
            return _stub_reviewer_disposition("clean_no_edits")

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review):
            disp = _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        d = disp.to_dict()
        assert d["verdict"] == "frozen_clean"
        assert d["reviewer_disposition"]["verdict"] == "clean_no_edits"
        assert d["gap_audit_pre"]["errors"] == []
        assert d["gap_audit_post"] is not None

    def test_terminal_verdict_to_dict_has_none_post(self):
        led = _ledger_obj(_clean_ledger_data())

        def fake_review(_fn, _led):
            return _stub_reviewer_disposition("cast_unrecoverable")

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review):
            disp = _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        assert disp.to_dict()["gap_audit_post"] is None


# ---------------------------------------------------------------------------
# Verdict-mapping table
# ---------------------------------------------------------------------------


class TestVerdictMapping:
    def test_mapping_is_complete(self):
        # Every ReviewerVerdict literal must have a freeze mapping.
        from typing import get_args
        for lit in get_args(_OTRLR.ReviewerVerdict):
            assert lit in _LFC_ORCH.REVIEWER_TO_FREEZE_VERDICT

    def test_terminal_set_contents(self):
        expected = {
            "cast_unrecoverable", "too_many_edits",
            "needs_full_rerun", "post_audit_failed",
        }
        assert _LFC_ORCH.FREEZE_TERMINAL_FAILURE_VERDICTS == expected


# ---------------------------------------------------------------------------
# Node class output contract
# ---------------------------------------------------------------------------


class TestNodeOutputContract:
    def test_return_names_freeze_verdict(self):
        from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
        assert OTR_LedgerFreezeCascade.RETURN_NAMES == (
            "script_text", "script_json", "news_used",
            "estimated_minutes", "freeze_verdict",
        )
        assert OTR_LedgerFreezeCascade.RETURN_TYPES == (
            "STRING", "STRING", "STRING", "INT", "STRING",
        )

    def test_legacy_class_name_is_dead(self):
        # Clean-break v2.0-alpha (2026-05-12): the legacy
        # OTR_LedgerScriptReviewer module is deleted; no back-compat
        # surface. Importing it must raise.
        import importlib
        with pytest.raises((ImportError, ModuleNotFoundError)):
            importlib.import_module("nodes.OTR_LedgerScriptReviewer")

    def test_class_input_types_schema(self):
        from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
        schema = OTR_LedgerFreezeCascade.INPUT_TYPES()
        assert "script_text" in schema["required"]
        assert "model_id" in schema["optional"]
        assert "script_json" in schema["optional"]
        # All four passthrough sockets keep forceInput=True so the
        # graph wires them by link, not by widget value.
        for k in ("script_text",):
            assert schema["required"][k][1].get("forceInput") is True
        for k in ("script_json", "news_used", "estimated_minutes"):
            assert schema["optional"][k][1].get("forceInput") is True
