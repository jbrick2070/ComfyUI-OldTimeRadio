"""tests/test_lfc_phase_7_8_readiness.py

LFC sprint Commit 5 -- regression coverage for Phase 7 (audio
readiness) + Phase 8 (video readiness). Both phases are
deterministic, no LLM calls, default ON.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_readiness as _LFC_RDY  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ledger(lines, cast=None):
    return SimpleNamespace(
        data={
            "schema_version": "l3-2026-05-14",
            "episode_id": "ep_test_readiness",
            "cast": cast or [
                {"char_id": "c01", "name": "ALICE",
                 "portrait_path": "ALICE_portrait.png"},
                {"char_id": "c02", "name": "BOB",
                 "portrait_path": "BOB_portrait.png"},
                {"char_id": "announcer", "name": "ANNOUNCER",
                 "portrait_path": "ANNOUNCER_portrait.png"},
            ],
            "scenes": [], "shots": [], "beats": [],
            "lines": lines, "sfx": [], "music": [], "clips": [],
            "meta": {},
        },
        episode_id="ep_test_readiness",
    )


def _line(line_id, char_id, role, text, *, skip=False):
    return {
        "line_id": line_id, "char_id": char_id,
        "speaker_role": role, "text": text, "skip": skip,
        "char_count": len(text),
        "word_count": sum(1 for _ in text.split()),
    }


# ---------------------------------------------------------------------------
# Phase 7 -- audio readiness
# ---------------------------------------------------------------------------


class TestPhase7Abbreviations:
    def test_doctor_expanded(self):
        led = _ledger([
            _line("l001", "c01", "character", "Dr. Hawking confirmed it."),
        ])
        _LFC_RDY.phase_7_audio_readiness(led)
        assert "Doctor Hawking" in led.data["lines"][0]["text"]
        assert "Dr." not in led.data["lines"][0]["text"]

    def test_mister_expanded(self):
        led = _ledger([
            _line("l001", "c01", "character", "Mr. Smith and Mrs. Smith arrived."),
        ])
        _LFC_RDY.phase_7_audio_readiness(led)
        text = led.data["lines"][0]["text"]
        assert "Mister Smith" in text
        assert "Missus Smith" in text

    def test_versus_expanded(self):
        led = _ledger([
            _line("l001", "c01", "character", "It was Earth vs. Mars."),
        ])
        _LFC_RDY.phase_7_audio_readiness(led)
        assert "versus" in led.data["lines"][0]["text"]

    def test_no_op_when_no_abbreviations(self):
        original = "Then we should go now."
        led = _ledger([_line("l001", "c01", "character", original)])
        rep = _LFC_RDY.phase_7_audio_readiness(led)
        assert led.data["lines"][0]["text"] == original
        assert rep.abbreviations_expanded == 0


class TestPhase7Symbols:
    def test_ampersand(self):
        led = _ledger([
            _line("l001", "c01", "character", "Black & White."),
        ])
        _LFC_RDY.phase_7_audio_readiness(led)
        assert "and" in led.data["lines"][0]["text"]
        assert "&" not in led.data["lines"][0]["text"]

    def test_percent(self):
        led = _ledger([
            _line("l001", "c01", "character", "Off by 50%."),
        ])
        _LFC_RDY.phase_7_audio_readiness(led)
        text = led.data["lines"][0]["text"]
        assert "%" not in text
        assert "percent" in text

    def test_at_sign(self):
        led = _ledger([
            _line("l001", "c01", "character", "Meet @ noon."),
        ])
        _LFC_RDY.phase_7_audio_readiness(led)
        assert "@" not in led.data["lines"][0]["text"]
        assert " at " in led.data["lines"][0]["text"]


class TestPhase7Numbers:
    def test_integer_expanded(self):
        pytest.importorskip("num2words")
        led = _ledger([
            _line("l001", "c01", "character", "We need 42 samples."),
        ])
        rep = _LFC_RDY.phase_7_audio_readiness(led)
        text = led.data["lines"][0]["text"]
        assert "42" not in text
        assert "forty-two" in text or "forty two" in text
        assert rep.numbers_expanded >= 1

    def test_large_integer_expanded(self):
        pytest.importorskip("num2words")
        led = _ledger([
            _line("l001", "c01", "character", "Total: 1,234 units."),
        ])
        _LFC_RDY.phase_7_audio_readiness(led)
        text = led.data["lines"][0]["text"]
        assert "1234" not in text  # comma stripped, then converted
        assert "1,234" not in text

    def test_num2words_unavailable_skips_with_warning(self, monkeypatch):
        monkeypatch.setattr(
            _LFC_RDY, "_try_import_num2words",
            lambda: None,
        )
        led = _ledger([
            _line("l001", "c01", "character", "We need 42 samples."),
        ])
        rep = _LFC_RDY.phase_7_audio_readiness(led)
        # Number stays intact, no crash, warning recorded.
        assert "42" in led.data["lines"][0]["text"]
        assert any("num2words" in w for w in rep.warnings)


class TestPhase7Boundaries:
    def test_skipped_lines_not_touched(self):
        led = _ledger([
            _line("l001", "c01", "character", "Dr. Hawking.", skip=True),
        ])
        _LFC_RDY.phase_7_audio_readiness(led)
        # Skipped lines bypassed.
        assert led.data["lines"][0]["text"] == "Dr. Hawking."

    def test_sfx_lines_not_touched(self):
        led = _ledger([
            _line("l001", "sfx", "sfx", "[door slam]"),
        ])
        _LFC_RDY.phase_7_audio_readiness(led)
        assert led.data["lines"][0]["text"] == "[door slam]"

    def test_word_count_updated_after_expansion(self):
        led = _ledger([
            _line("l001", "c01", "character", "Mr. Smith spoke."),
        ])
        _LFC_RDY.phase_7_audio_readiness(led)
        # "Mister Smith spoke." = 3 words.
        assert led.data["lines"][0]["word_count"] == 3
        assert led.data["lines"][0]["char_count"] == len(
            "Mister Smith spoke."
        )

    def test_meta_audio_readiness_stamped(self):
        led = _ledger([
            _line("l001", "c01", "character", "Mr. Smith."),
        ])
        _LFC_RDY.phase_7_audio_readiness(led)
        meta = led.data["meta"]["audio_readiness"]
        assert meta["lines_scanned"] == 1
        assert meta["lines_normalized"] == 1
        assert meta["abbreviations_expanded"] >= 1


# ---------------------------------------------------------------------------
# Phase 8 -- video readiness
# ---------------------------------------------------------------------------


class TestPhase8VideoReadiness:
    def test_all_portraits_present_passes(self):
        led = _ledger([
            _line("l001", "c01", "character", "Hello."),
            _line("l002", "c02", "character", "Hi."),
        ])
        rep = _LFC_RDY.phase_8_video_readiness(led)
        assert rep.cast_with_portrait == 3
        assert rep.cast_missing_portrait == []
        assert rep.voiced_lines == 2
        assert rep.voiced_lines_with_visual_mapping == 2

    def test_missing_portrait_warned_not_failed(self):
        led = _ledger([
            _line("l001", "c01", "character", "Hello."),
        ], cast=[
            {"char_id": "c01", "name": "ALICE"},  # no portrait field
        ])
        rep = _LFC_RDY.phase_8_video_readiness(led)
        assert "c01" in rep.cast_missing_portrait
        assert rep.voiced_lines_with_visual_mapping == 0
        assert "l001" in rep.voiced_lines_missing_visual
        assert rep.warnings

    def test_phase_8_does_not_mutate_lines(self):
        led = _ledger([
            _line("l001", "c01", "character", "Hello."),
        ])
        original_text = led.data["lines"][0]["text"]
        _LFC_RDY.phase_8_video_readiness(led)
        assert led.data["lines"][0]["text"] == original_text

    def test_meta_video_readiness_stamped(self):
        led = _ledger([
            _line("l001", "c01", "character", "Hello."),
        ])
        _LFC_RDY.phase_8_video_readiness(led)
        meta = led.data["meta"]["video_readiness"]
        assert meta["cast_total"] == 3
        assert meta["voiced_lines"] == 1

    def test_portrait_path_recognised_alongside_alt_keys(self):
        led = _ledger([], cast=[
            {"char_id": "c01", "name": "ALICE",
             "portrait_path": "ALICE.png"},
            {"char_id": "c02", "name": "BOB",
             "portrait_image": "BOB.png"},
            {"char_id": "c03", "name": "CARL", "portrait": "C.png"},
            {"char_id": "c04", "name": "DOE", "image_path": "D.png"},
        ])
        rep = _LFC_RDY.phase_8_video_readiness(led)
        # All four cast members are recognised as having a portrait.
        assert rep.cast_with_portrait == 4
        assert rep.cast_missing_portrait == []


# ---------------------------------------------------------------------------
# Cascade integration
# ---------------------------------------------------------------------------


class TestCascadeWiring:
    def test_phase_7_record_appended_to_cleanup_passes(self):
        from unittest.mock import patch
        from nodes import _otr_freeze_cascade as _LFC_ORCH
        from nodes import _otr_ledger_reviewer as _OTRLR

        led = _ledger([
            _line("l001", "c01", "character", "Mr. Smith arrived."),
            _line("l002", "c02", "character", "He said 100 percent."),
        ])

        def fake_review(_fn, _led):
            return _OTRLR.ReviewerDisposition(
                verdict="clean_no_edits",
                pre_audit_violations=0, pre_audit_repairs_applied=0,
                doctor_edits_proposed=0, doctor_edits_applied=0,
                post_audit_violations=0, phantom_skip_count=0,
            )

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        passes = led.data["meta"]["cleanup_passes"]
        names = [p["phase_name"] for p in passes]
        assert "phase_7_audio_readiness" in names
        assert "phase_8_video_readiness" in names
        # Phase 7 normalized "Mr." -> "Mister" so meta.audio_readiness
        # has lines_normalized >= 1.
        ar = led.data["meta"]["audio_readiness"]
        assert ar["lines_normalized"] >= 1
        # Phase 8 stamped meta.video_readiness.
        assert "video_readiness" in led.data["meta"]

    def test_phase_7_can_be_disabled(self):
        from unittest.mock import patch
        from nodes import _otr_freeze_cascade as _LFC_ORCH
        from nodes import _otr_ledger_reviewer as _OTRLR

        led = _ledger([
            _line("l001", "c01", "character", "Mr. Smith arrived."),
        ])

        def fake_review(_fn, _led):
            return _OTRLR.ReviewerDisposition(
                verdict="clean_no_edits",
                pre_audit_violations=0, pre_audit_repairs_applied=0,
                doctor_edits_proposed=0, doctor_edits_applied=0,
                post_audit_violations=0, phantom_skip_count=0,
            )

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review):
            _LFC_ORCH.run_freeze_cascade(
                lambda *a, **k: "", led,
                enable_phase_7_audio_readiness=False,
            )

        # Text NOT normalized.
        assert "Mr. Smith" in led.data["lines"][0]["text"]
        # Skip marker recorded.
        ar = led.data["meta"]["audio_readiness"]
        assert ar.get("skipped") is True
