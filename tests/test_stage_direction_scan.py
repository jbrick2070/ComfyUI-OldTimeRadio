"""Unit test for scripts/stage_direction_scan.py (the precision-gate tool)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

_spec = importlib.util.spec_from_file_location(
    "stage_direction_scan", str(_REPO / "scripts" / "stage_direction_scan.py"))
_scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scan)


class TestScanLine:
    def test_flags_and_proposes_strip(self):
        rec = _scan.scan_line("ep1", {
            "line_id": "b004", "speaker_role": "character",
            "text": "clenches jaw You mean it?",
        })
        assert rec is not None
        assert rec["detect_hit"] is True
        assert rec["would_mutate"] is True
        assert rec["proposed_text"] == "You mean it?"
        assert rec["guard_reasons"] == "floor_strip"

    def test_clean_line_not_flagged(self):
        rec = _scan.scan_line("ep1", {
            "line_id": "b002", "speaker_role": "character",
            "text": "We need a plan, Pinky.",
        })
        assert rec is None

    def test_non_character_role_skipped(self):
        rec = _scan.scan_line("ep1", {
            "line_id": "b001", "speaker_role": "announcer",
            "text": "twirls his pen You mean it?",
        })
        assert rec is None

    def test_detect_only_long_lead_not_mutated(self):
        rec = _scan.scan_line("ep1", {
            "line_id": "b005", "speaker_role": "character",
            "text": "twirls his pen and taps the desk twice You mean it?",
        })
        assert rec is not None
        assert rec["detect_hit"] is True
        assert rec["would_mutate"] is False
        assert rec["proposed_text"] is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
