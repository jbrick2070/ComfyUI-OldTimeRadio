"""tests/test_lfc_c3_phase_telemetry.py

LFC commit 12.6 -- C3 per-phase telemetry payload on
meta.freeze_phase_telemetry. Compact shape for soak diagnostics:
  {phase, bucket, skipped, changed, warnings, edits_proposed,
   edits_applied}

The freeze_verdict OUTPUT STRING stays the verdict literal --
this commit only adds the meta-side detail.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_freeze_cascade as _LFC_ORCH
from nodes import _otr_ledger_reviewer as _OTRLR


def _line(line_id, char_id, role, text):
    return {
        "line_id": line_id, "beat_id": line_id,
        "speaker_role": role, "char_id": char_id,
        "text": text, "skip": False,
        "char_count": len(text),
        "word_count": sum(1 for _ in text.split()),
    }


def _ledger(lines):
    return SimpleNamespace(
        data={
            "schema_version": "l3-2026-05-14",
            "episode_id": "ep_c3_test",
            "cast": [
                # voice_preset populated per G6 invariant (voice-path-cleanbreak):
                # non-ANNOUNCER rows must carry a v2/* Bark preset; ANNOUNCER
                # carries a Kokoro namespace id.
                {"char_id": "c01", "name": "ALICE",
                 "voice_preset": "v2/en_speaker_4",
                 "portrait_path": "ALICE.png"},
                {"char_id": "c02", "name": "BOB",
                 "voice_preset": "v2/en_speaker_6",
                 "portrait_path": "BOB.png"},
                {"char_id": "announcer", "name": "ANNOUNCER",
                 "voice_preset": "bm_george",
                 "portrait_path": "ANNOUNCER.png"},
            ],
            "scenes": [], "shots": [], "beats": [],
            "lines": lines, "music": [], "clips": [],
            "meta": {
                "episode_title": "c3 test",
                "style": "mission_control_procedural",
            },
        },
        episode_id="ep_c3_test",
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


# ---------------------------------------------------------------------------
# build_phase_telemetry shape
# ---------------------------------------------------------------------------


class TestBuildPhaseTelemetryShape:
    def test_empty_meta_returns_empty_list(self):
        assert _LFC_ORCH.build_phase_telemetry({}) == []

    def test_unrelated_keys_ignored(self):
        meta = {"episode_title": "x", "style": "y"}
        assert _LFC_ORCH.build_phase_telemetry(meta) == []

    def test_entry_shape(self):
        meta = {
            "cleanup_passes": [{
                "phase_name": "phase_3_per_line_polish",
                "started_at": "2026-05-12T00:00:00+00:00",
                "finished_at": "2026-05-12T00:00:01+00:00",
                "text_hash_before": 100,
                "text_hash_after": 200,
                "edits_proposed": 2,
                "edits_applied": 1,
                "failures": [],
            }],
        }
        tel = _LFC_ORCH.build_phase_telemetry(meta)
        assert len(tel) == 1
        entry = tel[0]
        assert entry["phase"] == "phase_3_per_line_polish"
        assert entry["bucket"] == "cleanup_passes"
        assert entry["skipped"] is False
        assert entry["changed"] is True   # hash_before != hash_after
        assert entry["warnings"] == 0
        assert entry["edits_proposed"] == 2
        assert entry["edits_applied"] == 1
        assert "started_at" not in entry  # compact shape

    def test_skipped_phase_marked_via_stub_bypassed_reason(self):
        meta = {
            "cleanup_passes": [{
                "phase_name": "phase_6_episode_arc",
                "started_at": "2026-05-12T00:00:00+00:00",
                "finished_at": "2026-05-12T00:00:00+00:00",
                "text_hash_before": 100,
                "text_hash_after": 100,
                "failures": [{
                    "line_id": "__phase_6_episode_arc__",
                    "reason": "stub_bypassed",
                }],
            }],
        }
        tel = _LFC_ORCH.build_phase_telemetry(meta)
        assert tel[0]["skipped"] is True

    def test_skipped_phase_marked_via_terminal_skipped_reason(self):
        meta = {
            "readiness_passes": [{
                "phase_name": "phase_7_audio_readiness",
                "started_at": "2026-05-12T00:00:00+00:00",
                "finished_at": "2026-05-12T00:00:00+00:00",
                "text_hash_before": 50,
                "text_hash_after": 50,
                "failures": [{
                    "line_id": "__phase_7__",
                    "reason": "terminal_skipped",
                }],
            }],
        }
        tel = _LFC_ORCH.build_phase_telemetry(meta)
        assert tel[0]["skipped"] is True
        # terminal_skipped counts as skipped, NOT as a real warning.
        assert tel[0]["warnings"] == 0

    def test_real_failure_counts_as_warning(self):
        meta = {
            "cleanup_passes": [{
                "phase_name": "phase_3_per_line_polish",
                "started_at": "2026-05-12T00:00:00+00:00",
                "finished_at": "2026-05-12T00:00:01+00:00",
                "text_hash_before": 100,
                "text_hash_after": 100,
                "failures": [
                    {"line_id": "l001", "reason": "polish refused"},
                    {"line_id": "l002", "reason": "still leaky"},
                ],
            }],
        }
        tel = _LFC_ORCH.build_phase_telemetry(meta)
        assert tel[0]["skipped"] is False
        assert tel[0]["warnings"] == 2

    def test_chronological_order_across_buckets(self):
        meta = {
            "audit_passes": [{
                "phase_name": "phase_0_gap_audit_pre",
                "started_at": "2026-05-12T00:00:01+00:00",
                "finished_at": "2026-05-12T00:00:02+00:00",
                "text_hash_before": 1, "text_hash_after": 1,
                "failures": [],
            }, {
                "phase_name": "phase_10_gap_audit_post_and_freeze",
                "started_at": "2026-05-12T00:00:10+00:00",
                "finished_at": "2026-05-12T00:00:11+00:00",
                "text_hash_before": 9, "text_hash_after": 9,
                "failures": [],
            }],
            "cleanup_passes": [{
                "phase_name": "phase_3_per_line_polish",
                "started_at": "2026-05-12T00:00:03+00:00",
                "finished_at": "2026-05-12T00:00:04+00:00",
                "text_hash_before": 2, "text_hash_after": 3,
                "failures": [],
            }],
            "readiness_passes": [{
                "phase_name": "phase_7_audio_readiness",
                "started_at": "2026-05-12T00:00:08+00:00",
                "finished_at": "2026-05-12T00:00:09+00:00",
                "text_hash_before": 8, "text_hash_after": 9,
                "failures": [],
            }],
        }
        tel = _LFC_ORCH.build_phase_telemetry(meta)
        phases_in_order = [e["phase"] for e in tel]
        assert phases_in_order == [
            "phase_0_gap_audit_pre",
            "phase_3_per_line_polish",
            "phase_7_audio_readiness",
            "phase_10_gap_audit_post_and_freeze",
        ]


# ---------------------------------------------------------------------------
# Cascade wires it
# ---------------------------------------------------------------------------


class TestCascadeStampsTelemetry:
    def test_clean_run_stamps_telemetry(self):
        led = _ledger([
            _line("l001", "c01", "character", "Hello there general."),
        ])
        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=_stub_reviewer("clean_no_edits")):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        tel = led.data["meta"]["freeze_phase_telemetry"]
        assert isinstance(tel, list)
        # Every cascade phase contributes an entry.
        phases = [e["phase"] for e in tel]
        for required in (
            "phase_0_gap_audit_pre",
            "phase_3_per_line_polish",
            "phase_1_2_9_reviewer_composite",
            "phase_4_per_scene_coherence",
            "phase_4_5_smart_suggestion",
            "phase_5_voice_drift",
            "phase_6_episode_arc",
            "phase_7_audio_readiness",
            "phase_8_video_readiness",
            "phase_10_gap_audit_post_and_freeze",
        ):
            assert required in phases, (
                f"missing {required} in freeze_phase_telemetry"
            )

    def test_terminal_run_stamps_telemetry(self):
        led = _ledger([
            _line("l001", "c01", "character", "Hello."),
        ])
        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=_stub_reviewer("needs_full_rerun")):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        tel = led.data["meta"]["freeze_phase_telemetry"]
        # Phase 4-10 should all be marked skipped on a terminal verdict.
        for entry in tel:
            if entry["phase"] in (
                "phase_4_per_scene_coherence",
                "phase_4_5_smart_suggestion",
                "phase_5_voice_drift",
                "phase_6_episode_arc",
                "phase_7_audio_readiness",
                "phase_8_video_readiness",
                "phase_10_gap_audit_post_and_freeze",
            ):
                assert entry["skipped"] is True, (
                    f"{entry['phase']} should be skipped on terminal"
                )

    def test_freeze_verdict_output_string_unchanged(self):
        # C3 explicitly says the OUTPUT STRING stays the verdict
        # literal -- only meta carries the telemetry detail.
        led = _ledger([
            _line("l001", "c01", "character", "Hello there general."),
        ])
        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=_stub_reviewer("clean_no_edits")):
            disp = _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)
        # disp.verdict is the literal; should be a plain string.
        assert isinstance(disp.verdict, str)
        assert disp.verdict in (
            "frozen_clean",
            "frozen_with_warns",
            "frozen_with_doctor_edits",
        )
