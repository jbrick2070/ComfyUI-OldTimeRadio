"""tests/test_lfc_g3_phase_verdicts.py

LFC commit 12.13 -- G3 fix regression. Shared PhaseVerdict
literal + Phase 5 surfaces explicit `failed` verdict on
two_step_generate None (previously masked as
completed_no_edits_after_flag).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_lfc_phase_verdicts as _PV  # noqa: E402
from nodes import _otr_lfc_phase_5_voice_drift as _P5  # noqa: E402


# ---------------------------------------------------------------------------
# Shared verdict module surface
# ---------------------------------------------------------------------------


class TestSharedVerdictModule:
    def test_universal_set_pinned(self):
        assert _PV.UNIVERSAL_VERDICTS == frozenset({
            "skipped",
            "no_ledger",
            "completed_no_edits",
            "completed_with_edits",
            "failed",
        })

    def test_interlock_set_pinned(self):
        assert _PV.INTERLOCK_VERDICTS == frozenset({
            "already_ran_in_cascade",
            "missing_prerequisite",
        })

    def test_phase_specific_set_pinned(self):
        assert _PV.PHASE_SPECIFIC_VERDICTS == frozenset({
            "completed_with_failures",
            "completed_no_edits_after_flag",
            "completed_no_flagged_lines",
        })

    def test_all_phase_verdicts_is_union(self):
        assert _PV.ALL_PHASE_VERDICTS == (
            _PV.UNIVERSAL_VERDICTS
            | _PV.INTERLOCK_VERDICTS
            | _PV.PHASE_SPECIFIC_VERDICTS
        )

    def test_no_duplicate_verdict_strings_across_sets(self):
        # A string should only appear in one set -- duplicates
        # would mean the categorization is wrong.
        all_strings: list = []
        for bucket in (
            _PV.UNIVERSAL_VERDICTS,
            _PV.INTERLOCK_VERDICTS,
            _PV.PHASE_SPECIFIC_VERDICTS,
        ):
            all_strings.extend(bucket)
        assert len(all_strings) == len(set(all_strings)), (
            f"Duplicate verdict string across sets: "
            f"{[v for v in set(all_strings) if all_strings.count(v) > 1]}"
        )


# ---------------------------------------------------------------------------
# Phase 5 report carries the new `failed` field
# ---------------------------------------------------------------------------


class TestPhase5ReportFailedField:
    def test_failed_defaults_false(self):
        rep = _P5.Phase5VoiceDriftReport()
        assert rep.failed is False
        assert rep.failure_reason == ""

    def test_to_dict_includes_failed(self):
        rep = _P5.Phase5VoiceDriftReport(
            failed=True, failure_reason="x",
        )
        d = rep.to_dict()
        assert d["failed"] is True
        assert d["failure_reason"] == "x"


# ---------------------------------------------------------------------------
# Phase 5 sets failed=True on two_step_generate None
# ---------------------------------------------------------------------------


def _line(line_id, char_id, text):
    return {
        "line_id": line_id, "beat_id": line_id,
        "speaker_role": "character", "char_id": char_id,
        "text": text, "skip": False,
        "char_count": len(text), "word_count": len(text.split()),
    }


def _ledger_with_outlier():
    """Build a ledger where exactly one line trips the drift
    heuristic. Includes a phase_3_per_line_polish record so the
    G1 interlock added in commit 12.14 doesn't short-circuit
    this G3 test.
    """
    return SimpleNamespace(
        data={
            "schema_version": "l3-2026-05-14",
            "episode_id": "ep_g3_test",
            "cast": [
                {"char_id": "c01", "name": "ALICE",
                 "traits": "calm"},
            ],
            "scenes": [], "shots": [], "beats": [],
            "lines": [
                _line("l001", "c01",
                      "alpha beta gamma delta epsilon zeta eta theta iota."),
                _line("l002", "c01",
                      "lambda mu nu xi omicron pi rho sigma tau."),
                _line("l003", "c01",
                      "phi chi psi omega aleph beth gimel daleth he."),
                _line("l004", "c01", "x " * 30),  # outlier
            ],
            "music": [], "clips": [],
            "meta": {
                "episode_title": "g3 test",
                "style": "mission_control_procedural",
                # G1 prereq: phase_3 must have a real record so the
                # standalone Phase 5 node's interlock allows the
                # actual run we're testing here.
                "cleanup_passes": [{
                    "phase_name": "phase_3_per_line_polish",
                    "failures": [],
                }],
            },
        },
        episode_id="ep_g3_test",
    )


class TestPhase5FailedOnTwoStepNone:
    def test_failed_set_when_formatter_exhausts(self):
        led = _ledger_with_outlier()
        # Feed garbage that exhausts parse_with_repair's 3 retries.
        replies = [
            "REVISE l004",
            "not json", "still not json",
            "still definitely not json", "absolutely not json",
        ]
        state = {"i": 0}

        def fake_llm(messages, *, temperature, max_new_tokens):
            i = state["i"]
            state["i"] += 1
            return replies[i] if i < len(replies) else replies[-1]

        rep = _P5.phase_5_voice_drift(fake_llm, led, enable=True)
        assert rep.failed is True
        assert "two_step_generate" in rep.failure_reason
        assert rep.edits_applied == 0
        # flagged_lines was populated before the LLM call; the
        # failed flag distinguishes parse-exhaustion from
        # "editor declined all".
        assert len(rep.flagged_lines) >= 1


# ---------------------------------------------------------------------------
# Standalone Phase 5 node returns `failed` verdict on report.failed
# ---------------------------------------------------------------------------


class TestPhase5StandaloneFailedVerdict:
    def test_standalone_phase_5_returns_failed_on_two_step_none(self):
        """G3 fix: pre-fix the standalone Phase 5 node returned
        `completed_no_edits_after_flag` for an LLM parse failure
        (silent failure mode). Post-fix it returns `failed`."""
        from unittest.mock import patch as _patch
        from nodes import _otr_model_loader as _real_loader
        from nodes import production_ledger as _real_pl

        led = _ledger_with_outlier()
        gen_replies = [
            "REVISE l004",
            "not json", "still not json",
            "still definitely not json", "absolutely not json",
        ]
        state = {"i": 0}

        def fake_gen(messages, *, temperature, max_new_tokens):
            i = state["i"]
            state["i"] += 1
            return gen_replies[i] if i < len(gen_replies) else gen_replies[-1]

        from nodes.OTR_LFCPhase5Voice import OTR_LFCPhase5Voice
        with _patch.object(_real_loader, "load_llm",
                           return_value={"model": object(),
                                         "tokenizer": object()}), \
             _patch.object(_real_loader, "make_generate_fn",
                           return_value=fake_gen), \
             _patch.object(_real_loader, "unload_llm",
                           return_value=None), \
             _patch.object(_real_pl, "peek_ledger", return_value=led), \
             _patch.object(_real_pl, "get_ledger", return_value=led):
            inst = OTR_LFCPhase5Voice()
            result = inst.run(
                script_json="{}", enable=True,
            )
        assert result[1] == "failed", (
            f"G3 fix: standalone Phase 5 must return 'failed' on "
            f"two_step_generate None. got verdict={result[1]!r}"
        )
