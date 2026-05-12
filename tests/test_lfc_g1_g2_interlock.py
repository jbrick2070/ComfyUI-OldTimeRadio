"""tests/test_lfc_g1_g2_interlock.py

LFC commit 12.14 -- G1 + G2 interlock regression.

G1: standalone Phase 5 refuses to run on un-polished ledger
    (missing Phase 3 prereq) unless force=True.
G2: standalone Phase 4 / 5 / 6 refuse to re-run when the main
    cascade already ran the corresponding phase, unless
    force=True.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_lfc_phase_verdicts as _PV  # noqa: E402


# ---------------------------------------------------------------------------
# Verdict-helper unit tests
# ---------------------------------------------------------------------------


class TestPhaseAlreadyRanInCascade:
    def test_real_run_record_is_detected(self):
        meta = {
            "cleanup_passes": [{
                "phase_name": "phase_4_per_scene_coherence",
                "failures": [],
            }]
        }
        assert _PV.phase_already_ran_in_cascade(
            meta, "phase_4_per_scene_coherence",
        ) is True

    def test_stub_bypassed_record_is_not_a_real_run(self):
        meta = {
            "cleanup_passes": [{
                "phase_name": "phase_4_per_scene_coherence",
                "failures": [{
                    "line_id": "__phase_4_per_scene_coherence__",
                    "reason": "stub_bypassed",
                }],
            }]
        }
        assert _PV.phase_already_ran_in_cascade(
            meta, "phase_4_per_scene_coherence",
        ) is False

    def test_terminal_skipped_record_is_not_a_real_run(self):
        meta = {
            "cleanup_passes": [{
                "phase_name": "phase_5_voice_drift",
                "failures": [{
                    "reason": "terminal_skipped",
                }],
            }]
        }
        assert _PV.phase_already_ran_in_cascade(
            meta, "phase_5_voice_drift",
        ) is False

    def test_enable_false_record_is_not_a_real_run(self):
        meta = {
            "cleanup_passes": [{
                "phase_name": "phase_6_episode_arc",
                "failures": [{
                    "reason": "enable_false",
                }],
            }]
        }
        assert _PV.phase_already_ran_in_cascade(
            meta, "phase_6_episode_arc",
        ) is False

    def test_other_phase_record_does_not_match(self):
        meta = {
            "cleanup_passes": [{
                "phase_name": "phase_3_per_line_polish",
                "failures": [],
            }]
        }
        # We're asking about phase 5 -- phase 3's record doesn't
        # count.
        assert _PV.phase_already_ran_in_cascade(
            meta, "phase_5_voice_drift",
        ) is False

    def test_empty_meta_returns_false(self):
        assert _PV.phase_already_ran_in_cascade({}, "anything") is False

    def test_walks_all_three_buckets(self):
        # phase_0 lives in audit_passes; the helper must find it
        # there.
        meta = {
            "audit_passes": [{
                "phase_name": "phase_0_gap_audit_pre",
                "failures": [],
            }],
            "cleanup_passes": [],
            "readiness_passes": [],
        }
        assert _PV.phase_already_ran_in_cascade(
            meta, "phase_0_gap_audit_pre",
        ) is True


# ---------------------------------------------------------------------------
# Standalone node interlock behaviour
# ---------------------------------------------------------------------------


def _line(line_id, char_id, role, text):
    return {
        "line_id": line_id, "beat_id": line_id,
        "speaker_role": role, "char_id": char_id,
        "text": text, "skip": False,
        "char_count": len(text), "word_count": len(text.split()),
    }


def _ledger_with_meta(meta_overrides=None):
    data = {
        "schema_version": "l3-2026-05-14",
        "episode_id": "ep_g1g2_test",
        "cast": [{"char_id": "c01", "name": "ALICE"}],
        "scenes": [], "shots": [], "beats": [],
        "lines": [_line("l001", "c01", "character", "Hello.")],
        "sfx": [], "music": [], "clips": [],
        "meta": {"episode_title": "g1/g2", "style": "x"},
    }
    if meta_overrides:
        data["meta"].update(meta_overrides)
    return SimpleNamespace(data=data, episode_id=data["episode_id"])


def _patched(led):
    """Patch loader + production_ledger so standalone nodes can run
    without a live LLM or singleton ledger."""
    from unittest.mock import patch as _patch
    from nodes import _otr_model_loader as _real_loader
    from nodes import production_ledger as _real_pl
    return [
        _patch.object(_real_loader, "load_llm",
                      return_value={"model": object(),
                                    "tokenizer": object()}),
        _patch.object(_real_loader, "make_generate_fn",
                      return_value=lambda *a, **k: ""),
        _patch.object(_real_loader, "make_polish_generate_fn",
                      return_value=lambda *a, **k: ""),
        _patch.object(_real_loader, "unload_llm",
                      return_value=None),
        _patch.object(_real_pl, "peek_ledger", return_value=led),
        _patch.object(_real_pl, "get_ledger", return_value=led),
    ]


class TestPhase4Interlock:
    def test_already_ran_returns_interlock_verdict(self):
        from nodes.OTR_LFCPhase4Scene import OTR_LFCPhase4Scene
        led = _ledger_with_meta({
            "cleanup_passes": [{
                "phase_name": "phase_4_per_scene_coherence",
                "failures": [],
            }],
        })
        with _stack(_patched(led)):
            inst = OTR_LFCPhase4Scene()
            _, verdict = inst.run(
                script_json="{}", enable=True, force=False,
            )
        assert verdict == "already_ran_in_cascade"

    def test_force_overrides_interlock(self):
        from unittest.mock import patch as _patch
        from nodes.OTR_LFCPhase4Scene import OTR_LFCPhase4Scene
        from nodes import _otr_lfc_phase_4_scene_coherence as _P4

        led = _ledger_with_meta({
            "cleanup_passes": [{
                "phase_name": "phase_4_per_scene_coherence",
                "failures": [],
            }],
        })
        with _stack(_patched(led)):
            with _patch.object(
                _P4, "phase_4_scene_coherence",
                return_value=_P4.Phase4SceneCoherenceReport(
                    scenes_scanned=0, scenes_examined=0,
                    scenes_failed=0, total_edits_applied=0,
                ),
            ):
                inst = OTR_LFCPhase4Scene()
                _, verdict = inst.run(
                    script_json="{}", enable=True, force=True,
                )
        # With force=True the phase actually ran; verdict reflects
        # the real outcome ("completed_no_edits"), not the
        # interlock.
        assert verdict == "completed_no_edits"


class TestPhase5Interlock:
    def test_missing_phase_3_prereq_returns_interlock(self):
        from nodes.OTR_LFCPhase5Voice import OTR_LFCPhase5Voice
        # No phase_3 record on meta.
        led = _ledger_with_meta({})
        with _stack(_patched(led)):
            inst = OTR_LFCPhase5Voice()
            _, verdict = inst.run(
                script_json="{}", enable=True, force=False,
            )
        assert verdict == "missing_prerequisite"

    def test_phase_3_present_then_runs(self):
        from unittest.mock import patch as _patch
        from nodes.OTR_LFCPhase5Voice import OTR_LFCPhase5Voice
        from nodes import _otr_lfc_phase_5_voice_drift as _P5

        led = _ledger_with_meta({
            "cleanup_passes": [{
                "phase_name": "phase_3_per_line_polish",
                "failures": [],
            }],
        })
        with _stack(_patched(led)):
            with _patch.object(
                _P5, "phase_5_voice_drift",
                return_value=_P5.Phase5VoiceDriftReport(),
            ):
                inst = OTR_LFCPhase5Voice()
                _, verdict = inst.run(
                    script_json="{}", enable=True, force=False,
                )
        # No flagged lines (default report) -> completed_no_flagged_lines.
        assert verdict == "completed_no_flagged_lines"

    def test_double_run_overrides_prereq_check(self):
        # If Phase 5 already ran AND Phase 3 also ran, the
        # double-run check fires first.
        from nodes.OTR_LFCPhase5Voice import OTR_LFCPhase5Voice
        led = _ledger_with_meta({
            "cleanup_passes": [
                {"phase_name": "phase_3_per_line_polish",
                 "failures": []},
                {"phase_name": "phase_5_voice_drift",
                 "failures": []},
            ],
        })
        with _stack(_patched(led)):
            inst = OTR_LFCPhase5Voice()
            _, verdict = inst.run(
                script_json="{}", enable=True, force=False,
            )
        assert verdict == "already_ran_in_cascade"


class TestPhase6Interlock:
    def test_already_ran_returns_interlock(self):
        from nodes.OTR_LFCPhase6Arc import OTR_LFCPhase6Arc
        led = _ledger_with_meta({
            "cleanup_passes": [{
                "phase_name": "phase_6_episode_arc",
                "failures": [],
            }],
        })
        with _stack(_patched(led)):
            inst = OTR_LFCPhase6Arc()
            _, verdict = inst.run(
                script_json="{}", enable=True, force=False,
            )
        assert verdict == "already_ran_in_cascade"

    def test_no_prereq_check_phase_6_has_synopses_fallback(self):
        # Phase 6 deliberately does NOT check for Phase 4 prereq
        # -- the underlying phase function has a documented
        # fallback (outline-intents). Verify the standalone node
        # runs cleanly even without a phase 4 record.
        from unittest.mock import patch as _patch
        from nodes.OTR_LFCPhase6Arc import OTR_LFCPhase6Arc
        from nodes import _otr_lfc_phase_6_episode_arc as _P6

        led = _ledger_with_meta({})  # no phase records
        with _stack(_patched(led)):
            with _patch.object(
                _P6, "phase_6_episode_arc",
                return_value=_P6.Phase6EpisodeArcReport(),
            ):
                inst = OTR_LFCPhase6Arc()
                _, verdict = inst.run(
                    script_json="{}", enable=True, force=False,
                )
        # Default report: voiced_beats=0 (no voiced beats), but
        # the verdict bucket logic returns "completed_no_edits"
        # for that path (failed=False, edits_applied=0).
        assert verdict == "completed_no_edits"


# ---------------------------------------------------------------------------
# Tiny stack-of-context-managers helper
# ---------------------------------------------------------------------------


class _stack:
    def __init__(self, cms):
        self._cms = list(cms)
        self._entered: list = []

    def __enter__(self):
        for cm in self._cms:
            self._entered.append(cm.__enter__())
        return self

    def __exit__(self, *exc):
        for cm in reversed(self._cms):
            try:
                cm.__exit__(*exc)
            except Exception:
                pass
        return False
