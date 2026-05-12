"""tests/test_lfc_phase_4_5_smart_suggestion.py

LFC sprint Commit 11 -- regression coverage for the optional
deterministic SFX / music synthesis phase. Default OFF per ADR
section 6.17.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_lfc_smart_suggestion as _SS  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ledger(lines):
    return SimpleNamespace(
        data={
            "schema_version": "l3-2026-05-14",
            "episode_id": "ep_ss_test",
            "cast": [
                {"char_id": "c01", "name": "ALICE"},
                {"char_id": "c02", "name": "BOB"},
            ],
            "scenes": [], "shots": [],
            "beats": [], "lines": lines,
            "sfx": [], "music": [], "clips": [],
            "meta": {},
        },
        episode_id="ep_ss_test",
    )


def _line(line_id, role, char_id, text):
    return {
        "line_id": line_id, "beat_id": line_id,
        "speaker_role": role, "char_id": char_id,
        "text": text, "skip": False,
        "char_count": len(text),
        "word_count": sum(1 for _ in text.split()),
    }


# ---------------------------------------------------------------------------
# Disabled default
# ---------------------------------------------------------------------------


class TestDisabledByDefault:
    def test_disabled_no_op(self):
        led = _ledger([
            _line("l001", "character", "c01",
                  "Start the car, Bob."),
        ])
        rep = _SS.phase_4_5_smart_suggestion(led, enable=False)
        assert rep.sfx_suggested == 0
        assert len(led.data["lines"]) == 1
        meta = led.data["meta"]["phase_4_5_smart_suggestion"]
        assert meta["skipped"] is True


# ---------------------------------------------------------------------------
# SFX synthesis
# ---------------------------------------------------------------------------


class TestSFXSynthesis:
    def test_car_engine_start_inferred(self):
        led = _ledger([
            _line("l001", "character", "c01", "Start the car, Bob."),
        ])
        rep = _SS.phase_4_5_smart_suggestion(led, enable=True)
        assert rep.sfx_suggested == 1
        # Synthesized line appended.
        assert len(led.data["lines"]) == 2
        sfx_line = led.data["lines"][1]
        assert sfx_line["speaker_role"] == "sfx"
        assert sfx_line.get("auto_generated") is True
        assert "car_engine_start" in sfx_line["text"]

    def test_door_slam_inferred(self):
        led = _ledger([
            _line("l001", "character", "c01",
                  "She slammed the door behind her."),
        ])
        rep = _SS.phase_4_5_smart_suggestion(led, enable=True)
        assert rep.sfx_suggested == 1
        assert "door_slam" in led.data["lines"][1]["text"]

    def test_phone_ring_inferred(self):
        led = _ledger([
            _line("l001", "character", "c01",
                  "The phone rang twice before he answered."),
        ])
        rep = _SS.phase_4_5_smart_suggestion(led, enable=True)
        assert rep.sfx_suggested == 1
        assert "phone_ring" in led.data["lines"][1]["text"]

    def test_no_false_positive_on_clean_dialogue(self):
        led = _ledger([
            _line("l001", "character", "c01",
                  "I told you it was rigged."),
        ])
        rep = _SS.phase_4_5_smart_suggestion(led, enable=True)
        assert rep.sfx_suggested == 0
        assert len(led.data["lines"]) == 1

    def test_de_duplication_against_existing_sfx(self):
        # An sfx beat for car_engine_start is already present;
        # the inferred suggestion should NOT be added again.
        led = _ledger([
            _line("l001", "character", "c01", "Start the car, Bob."),
            _line("l002", "sfx", "", "car_engine_start"),
        ])
        rep = _SS.phase_4_5_smart_suggestion(led, enable=True)
        assert rep.sfx_suggested == 0


# ---------------------------------------------------------------------------
# Music synthesis
# ---------------------------------------------------------------------------


class TestMusicSynthesis:
    def test_music_swell_inferred(self):
        led = _ledger([
            _line("l001", "character", "c01",
                  "And then the music swelled to a crescendo."),
        ])
        rep = _SS.phase_4_5_smart_suggestion(led, enable=True)
        assert rep.music_suggested == 1
        music_line = led.data["lines"][1]
        assert music_line["speaker_role"] == "music_inter"
        assert music_line.get("auto_generated") is True
        assert "Swelling" in music_line["text"]

    def test_dedupe_against_existing_music_inter(self):
        led = _ledger([
            _line("l001", "character", "c01",
                  "The music swelled at that moment."),
            _line("l002", "music_inter", "", "Existing music"),
        ])
        rep = _SS.phase_4_5_smart_suggestion(led, enable=True)
        assert rep.music_suggested == 0


# ---------------------------------------------------------------------------
# Mutation contract
# ---------------------------------------------------------------------------


class TestMutationContract:
    def test_original_lines_text_untouched(self):
        original_text = "Start the car, Bob."
        led = _ledger([
            _line("l001", "character", "c01", original_text),
        ])
        _SS.phase_4_5_smart_suggestion(led, enable=True)
        # First (original) line still says exactly what it did.
        assert led.data["lines"][0]["text"] == original_text

    def test_original_line_id_stable(self):
        led = _ledger([
            _line("l001", "character", "c01", "Start the car, Bob."),
        ])
        _SS.phase_4_5_smart_suggestion(led, enable=True)
        assert led.data["lines"][0]["line_id"] == "l001"

    def test_synthesized_line_id_does_not_collide(self):
        led = _ledger([
            _line("l001", "character", "c01", "Start the car."),
            _line("auto_sfx_000", "sfx", "", "pre-existing"),
        ])
        _SS.phase_4_5_smart_suggestion(led, enable=True)
        # New synthesized line picks the next unused auto_sfx_NNN id.
        ids = [ln["line_id"] for ln in led.data["lines"]]
        assert len(set(ids)) == len(ids)  # no duplicate ids

    def test_meta_record_full_counts(self):
        led = _ledger([
            _line("l001", "character", "c01", "Start the car."),
            _line("l002", "character", "c01", "Door slammed shut."),
            _line("l003", "character", "c01", "I told you so."),
        ])
        _SS.phase_4_5_smart_suggestion(led, enable=True)
        rec = led.data["meta"]["phase_4_5_smart_suggestion"]
        assert rec["lines_scanned"] == 3
        assert rec["sfx_suggested"] == 2
        assert len(rec["sfx_added"]) == 2


# ---------------------------------------------------------------------------
# Cascade wiring
# ---------------------------------------------------------------------------


class TestCascadeWiring:
    def test_cascade_default_off_no_lines_added(self):
        from unittest.mock import patch
        from nodes import _otr_freeze_cascade as _LFC_ORCH
        from nodes import _otr_ledger_reviewer as _OTRLR

        led = _ledger([
            _line("l001", "character", "c01", "Start the car."),
            _line("l002", "announcer", "announcer",
                  "Tonight on Signal Lost."),
        ])

        def fake_review(_fn, _led):
            return _OTRLR.ReviewerDisposition(
                verdict="clean_no_edits",
                pre_audit_violations=0, pre_audit_repairs_applied=0,
                doctor_edits_proposed=0, doctor_edits_applied=0,
                post_audit_violations=0, phantom_skip_count=0,
            )

        # Default cascade call -- phase 4.5 disabled.
        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        # No synthesized lines added.
        roles = [ln["speaker_role"] for ln in led.data["lines"]]
        assert roles.count("sfx") == 0
        # Phase record still stamped (skipped reason).
        assert "phase_4_5_smart_suggestion" in led.data["meta"]

    def test_cascade_enabled_synthesizes_sfx(self):
        from unittest.mock import patch
        from nodes import _otr_freeze_cascade as _LFC_ORCH
        from nodes import _otr_ledger_reviewer as _OTRLR

        led = _ledger([
            _line("l001", "character", "c01", "Start the car."),
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
                enable_phase_4_5_smart_suggestion=True,
            )

        # Synthesized SFX line present.
        roles = [ln["speaker_role"] for ln in led.data["lines"]]
        assert roles.count("sfx") == 1
