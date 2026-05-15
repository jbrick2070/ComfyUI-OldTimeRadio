"""tests/test_script_doctor_hardfail.py — S34 B1 P0 deletion proof.

S34 B1 (2026-05-15) changes `run_script_doctor` in
`nodes/_otr_ledger_reviewer.py` so that all three failure paths
(LLM exception, JSON parse failure, schema validation failure)
return `ScriptDoctorReport(overall_verdict="needs_full_rerun")`
instead of the silent fail-soft `ScriptDoctorReport()` default
(which yielded `overall_verdict="clean"` so the caller committed
the candidate as if the doctor had succeeded).

The Phase 1 auditor (audit_cast_contract) already fails loud via
`_audit_failed_sentinel(pass_clean=False)`; Phase 2 now matches.

Tests prove:
  1. Each of the three failure paths returns
     `overall_verdict="needs_full_rerun"`.
  2. Empty / no-content output also routes to needs_full_rerun.
  3. The docstring no longer carries the stale fail-soft phrases.
  4. `review_ledger` propagates doctor failure into a
     `ReviewerDisposition(verdict="needs_full_rerun")` WITHOUT
     invoking `apply_doctor_edits` (no candidate commit).
  5. The freeze cascade maps reviewer `needs_full_rerun` through
     `REVIEWER_TO_FREEZE_VERDICT` and into the cascade's output
     `freeze_verdict` slot (index 4).
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_ledger_freeze as _LFC
from nodes import _otr_ledger_reviewer as _OTRLR
from nodes._otr_ledger_reviewer import (
    ScriptDoctorReport,
    run_script_doctor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _line(line_id: str, char_id: str, text: str,
          role: str = "character", beat_id: str = "b001") -> dict:
    return {
        "line_id": line_id,
        "beat_id": beat_id,
        "char_id": char_id,
        "speaker_role": role,
        "speaker": char_id,
        "text": text,
        "char_count": len(text),
        "word_count": len(text.split()),
        "skip": False,
    }


def _minimal_ledger_data() -> dict:
    return {
        "schema_version": _LFC.EXPECTED_SCHEMA_VERSION,
        "episode_id": "ep_test_b1",
        "cast": [
            {"char_id": "c01", "name": "ALICE",
             "speaker_role": "character",
             "voice_preset": "v2/en_speaker_6"},
            {"char_id": "announcer", "name": "ANNOUNCER",
             "speaker_role": "announcer",
             "voice_preset": "v2/en_speaker_9"},
        ],
        "scenes": [], "shots": [],
        "beats": [{"beat_id": "b001"}, {"beat_id": "b002"}],
        "lines": [
            _line("l001", "announcer", "Tonight, on Signal Lost.",
                  role="announcer", beat_id="b001"),
            _line("l002", "c01", "The readings are off the chart.",
                  role="character", beat_id="b002"),
        ],
        "music": [], "clips": [],
        "meta": {
            "episode_title": "Test Episode",
            "allowed_roster": ["ALICE", "ANNOUNCER"],
        },
    }


class _LedgerStub:
    """Minimal Ledger stand-in: .data, .episode_id, .save() no-op."""

    def __init__(self, data: dict, tmp_path: Path) -> None:
        self.data = data
        self.episode_id = data.get("episode_id", "ep")
        self._path = tmp_path / "led.json"

    def save(self) -> None:
        self._path.write_text(
            json.dumps(self.data, indent=2), encoding="utf-8"
        )


def _audit_clean_json() -> str:
    return json.dumps({"violations": [], "pass_clean": True})


def _doctor_args(ledger_data: dict) -> tuple:
    """Build the positional + kw args run_script_doctor expects."""
    cast_rows = ledger_data.get("cast", [])
    edit_cap = 3  # arbitrary; failure-path tests don't gate on this.
    return (ledger_data, cast_rows, edit_cap)


# ---------------------------------------------------------------------------
# 1-3. Direct run_script_doctor failure paths
# ---------------------------------------------------------------------------


class TestScriptDoctorFailurePaths:
    def test_script_doctor_llm_exception_returns_needs_full_rerun(self):
        ledger_data = _minimal_ledger_data()

        def boom(messages, *, temperature, max_new_tokens):
            raise RuntimeError("simulated LLM OOM")

        report = run_script_doctor(boom, *_doctor_args(ledger_data))
        assert isinstance(report, ScriptDoctorReport)
        assert report.overall_verdict == "needs_full_rerun", (
            f"S34 B1 P0 regressed: LLM exception path returned "
            f"{report.overall_verdict!r} instead of 'needs_full_rerun'"
        )

    def test_script_doctor_invalid_json_returns_needs_full_rerun(self):
        ledger_data = _minimal_ledger_data()

        def fn(messages, *, temperature, max_new_tokens):
            return "not json at all -- prose response"

        report = run_script_doctor(fn, *_doctor_args(ledger_data))
        assert report.overall_verdict == "needs_full_rerun", (
            "S34 B1 P0 regressed: JSON parse failure path returned "
            f"{report.overall_verdict!r}"
        )

    def test_script_doctor_schema_validation_returns_needs_full_rerun(self):
        # Valid JSON, but missing the required `edits` field /
        # invalid `overall_verdict` value -> pydantic ValidationError.
        ledger_data = _minimal_ledger_data()

        def fn(messages, *, temperature, max_new_tokens):
            return json.dumps({
                "edits": "not a list",     # wrong type
                "overall_verdict": "made_up_verdict",  # not in Literal
            })

        report = run_script_doctor(fn, *_doctor_args(ledger_data))
        assert report.overall_verdict == "needs_full_rerun", (
            "S34 B1 P0 regressed: schema validation failure path "
            f"returned {report.overall_verdict!r}"
        )

    def test_script_doctor_empty_output_returns_needs_full_rerun(self):
        # Empty string -> JSON parse failure.
        ledger_data = _minimal_ledger_data()

        def fn(messages, *, temperature, max_new_tokens):
            return ""

        report = run_script_doctor(fn, *_doctor_args(ledger_data))
        assert report.overall_verdict == "needs_full_rerun", (
            "S34 B1 P0 regressed: empty output path returned "
            f"{report.overall_verdict!r}"
        )


# ---------------------------------------------------------------------------
# 4. Docstring stale-phrase scrub
# ---------------------------------------------------------------------------


class TestScriptDoctorDocstring:
    def test_script_doctor_docstring_replaces_stale_phrases(self):
        doc = inspect.getdoc(run_script_doctor) or ""
        # Stale fail-soft language MUST be gone (the original phrases
        # from pre-S34).
        assert "fail-soft on the doctor" not in doc, (
            "S34 B1 docstring scrub regressed: 'fail-soft on the "
            "doctor' phrase still present"
        )
        assert 'overall_verdict="clean"' not in doc, (
            "S34 B1 docstring scrub regressed: 'overall_verdict=\"clean\"' "
            "phrase still present (the documented soft-fail return value)"
        )
        # The new fail-loud rationale must be present.
        assert "needs_full_rerun" in doc, (
            "S34 B1 docstring scrub regressed: 'needs_full_rerun' "
            "not present in run_script_doctor docstring"
        )


# ---------------------------------------------------------------------------
# 5. review_ledger propagation
# ---------------------------------------------------------------------------


class TestReviewLedgerDoctorFailurePropagation:
    def test_review_ledger_doctor_failure_returns_needs_full_rerun_without_commit(
        self, tmp_path
    ):
        # Phase 1 clean; Phase 2 doctor raises -> needs_full_rerun.
        led = _LedgerStub(_minimal_ledger_data(), tmp_path)

        call_count = {"n": 0}

        def fake_generate(messages, *, temperature, max_new_tokens):
            i = call_count["n"]
            call_count["n"] += 1
            if i == 0:
                # Phase 1 audit_cast_contract
                return _audit_clean_json()
            # Phase 2 run_script_doctor -> raise
            raise RuntimeError("doctor crashed")

        # Spy on apply_doctor_edits to confirm it never runs.
        with patch.object(
            _OTRLR, "apply_doctor_edits", wraps=_OTRLR.apply_doctor_edits
        ) as spy:
            disp = _OTRLR.review_ledger(fake_generate, led)

        assert disp.verdict == "needs_full_rerun", (
            f"S34 B1 P0 propagation regressed: review_ledger returned "
            f"{disp.verdict!r} instead of 'needs_full_rerun' on doctor "
            f"failure"
        )
        assert spy.call_count == 0, (
            "S34 B1 P0 propagation regressed: review_ledger invoked "
            "apply_doctor_edits despite the doctor failing -- candidate "
            "would have been committed corrupted"
        )


# ---------------------------------------------------------------------------
# 6. Cascade verdict mapping
# ---------------------------------------------------------------------------


class TestCascadeVerdictMapping:
    def test_freeze_cascade_maps_reviewer_needs_full_rerun_to_freeze_verdict(self):
        # Validate the verdict-mapping table contains the route from
        # reviewer needs_full_rerun -> freeze_verdict needs_full_rerun.
        # Direct table check is the load-bearing assertion; the
        # orchestrator consumes this map.
        from nodes import _otr_freeze_cascade as _LFC_ORCH
        assert _LFC_ORCH.REVIEWER_TO_FREEZE_VERDICT["needs_full_rerun"] == (
            "needs_full_rerun"
        ), (
            "S34 B1 P0: reviewer needs_full_rerun must map to freeze "
            "needs_full_rerun (the cascade's freeze_verdict slot value)"
        )
        # And it's in the terminal-failure set (cascade skips Phase 10
        # and returns the verdict directly).
        assert "needs_full_rerun" in _LFC_ORCH.FREEZE_TERMINAL_FAILURE_VERDICTS, (
            "S34 B1 P0: needs_full_rerun must be terminal so the cascade "
            "skips Phase 10 (which would otherwise stamp frozen_clean on "
            "an un-doctored ledger)"
        )
