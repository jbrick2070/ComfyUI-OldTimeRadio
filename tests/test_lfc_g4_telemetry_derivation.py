"""tests/test_lfc_g4_telemetry_derivation.py

LFC commit 12.16 -- G4 regression. The cascade stamps
`meta.freeze_phase_telemetry` at end-of-cascade by walking the
three bucket lists (audit_passes / cleanup_passes /
readiness_passes). Reviewer raised the silent-drift risk: if any
code path edits a bucket record without re-deriving the
telemetry, the two views diverge and both become untrustworthy.

This test pins the contract: after the cascade runs,
`meta.freeze_phase_telemetry` MUST equal
`build_phase_telemetry(meta)` when called fresh on the same
meta. Any future regression that writes telemetry independently
of the bucket lists fails this test.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_freeze_cascade as _LFC_ORCH  # noqa: E402
from nodes import _otr_ledger_reviewer as _OTRLR  # noqa: E402


def _line(line_id, char_id, role, text, *, skip=False):
    return {
        "line_id": line_id, "beat_id": line_id,
        "speaker_role": role, "char_id": char_id,
        "text": text, "skip": skip,
        "char_count": len(text), "word_count": len(text.split()),
    }


def _ledger(lines, cast=None):
    return SimpleNamespace(
        data={
            "schema_version": "l3-2026-05-14",
            "episode_id": "ep_g4_test",
            "cast": cast or [
                {"char_id": "c01", "name": "ALICE",
                 "portrait_path": "ALICE.png"},
                {"char_id": "announcer", "name": "ANNOUNCER",
                 "portrait_path": "ANNOUNCER.png"},
            ],
            "scenes": [], "shots": [], "beats": [],
            "lines": lines, "music": [], "clips": [],
            "meta": {
                "episode_title": "g4 test",
                "style": "mission_control_procedural",
            },
        },
        episode_id="ep_g4_test",
    )


def _stub_reviewer(verdict="clean_no_edits"):
    def fake_review(_fn, _led):
        return _OTRLR.ReviewerDisposition(
            verdict=verdict,
            pre_audit_violations=0, pre_audit_repairs_applied=0,
            doctor_edits_proposed=0, doctor_edits_applied=0,
            post_audit_violations=0, phantom_skip_count=0,
        )
    return fake_review


class TestTelemetryDerivedFromBuckets:
    """G4 contract: meta.freeze_phase_telemetry IS A DERIVED VIEW.
    The source of truth is the three bucket lists. Any future
    code path that writes telemetry independently breaks this
    test."""

    def test_clean_run_telemetry_equals_fresh_derivation(self):
        led = _ledger([
            _line("l001", "c01", "character", "Hello there general."),
        ])
        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=_stub_reviewer("clean_no_edits")):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        meta = led.data["meta"]
        stored = meta.get("freeze_phase_telemetry")
        assert stored is not None, (
            "G4: cascade must stamp meta.freeze_phase_telemetry on "
            "every exit path"
        )

        # Re-derive on the same meta and compare.
        rederived = _LFC_ORCH.build_phase_telemetry(meta)
        assert stored == rederived, (
            f"G4: meta.freeze_phase_telemetry diverged from "
            f"build_phase_telemetry(meta).\n"
            f"  stored:    {stored}\n"
            f"  rederived: {rederived}"
        )

    def test_telemetry_phase_count_matches_bucket_record_count(self):
        led = _ledger([
            _line("l001", "c01", "character", "Hello."),
        ])
        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=_stub_reviewer("clean_no_edits")):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        meta = led.data["meta"]
        bucket_records = (
            (meta.get("audit_passes") or [])
            + (meta.get("cleanup_passes") or [])
            + (meta.get("readiness_passes") or [])
        )
        bucket_count = sum(
            1 for r in bucket_records if isinstance(r, dict)
        )
        tel_count = len(meta.get("freeze_phase_telemetry") or [])
        assert tel_count == bucket_count, (
            f"G4: telemetry entry count {tel_count} != "
            f"bucket record count {bucket_count}. Drift detected."
        )

    def test_telemetry_bucket_field_matches_PHASE_BUCKETS(self):
        """Each telemetry entry's `bucket` field must match the
        canonical _PHASE_BUCKETS mapping. Catches the bug where
        a bucket-routing change touches one path but not the
        other."""
        led = _ledger([
            _line("l001", "c01", "character", "Hello."),
        ])
        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=_stub_reviewer("clean_no_edits")):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        tel = led.data["meta"]["freeze_phase_telemetry"]
        for entry in tel:
            phase = entry["phase"]
            expected_bucket = _LFC_ORCH._PHASE_BUCKETS.get(
                phase, "cleanup_passes",
            )
            assert entry["bucket"] == expected_bucket, (
                f"G4: telemetry entry for {phase!r} carries bucket "
                f"{entry['bucket']!r}; _PHASE_BUCKETS says "
                f"{expected_bucket!r}"
            )

    def test_terminal_run_telemetry_derives_correctly(self):
        # On terminal-reviewer-verdict the cascade stamps
        # terminal_skipped records for the downstream phases.
        # Verify the telemetry derivation still matches.
        # S33 B2 (2026-05-15): `cast_unrecoverable` retired; use
        # `too_many_edits` -- still a terminal verdict, same
        # telemetry-derivation path.
        led = _ledger([
            _line("l001", "c01", "character", "Hello."),
        ])
        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=_stub_reviewer("too_many_edits")):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        meta = led.data["meta"]
        stored = meta["freeze_phase_telemetry"]
        rederived = _LFC_ORCH.build_phase_telemetry(meta)
        assert stored == rederived

    def test_idempotent_re_derivation(self):
        """build_phase_telemetry called twice on the same meta
        returns equal lists. Pure function over the bucket
        state."""
        led = _ledger([
            _line("l001", "c01", "character", "Hello."),
        ])
        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=_stub_reviewer("clean_no_edits")):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        meta = led.data["meta"]
        first = _LFC_ORCH.build_phase_telemetry(meta)
        second = _LFC_ORCH.build_phase_telemetry(meta)
        assert first == second
