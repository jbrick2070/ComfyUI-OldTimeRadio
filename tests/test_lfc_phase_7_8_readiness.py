"""Regression coverage for non-destructive audio/video readiness."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_readiness as readiness  # noqa: E402
from nodes._otr_text_metrics import canonical_word_count  # noqa: E402


def _ledger(lines, cast=None):
    data = {
        "schema_version": "l3-2026-05-14",
        "episode_id": "ep_test_readiness",
        "cast": cast or [
            {
                "char_id": "c01",
                "name": "ALICE",
                "portrait_path": "ALICE_portrait.png",
            },
            {
                "char_id": "c02",
                "name": "BOB",
                "portrait_path": "BOB_portrait.png",
            },
            {
                "char_id": "announcer",
                "name": "ANNOUNCER",
                "portrait_path": "ANNOUNCER_portrait.png",
            },
        ],
        "scenes": [],
        "shots": [],
        "beats": [],
        "lines": lines,
        "music": [],
        "clips": [],
        "meta": {
            "episode_title": "Readiness Test",
            "style": "radio_drama",
        },
    }
    return SimpleNamespace(
        data=data,
        episode_id=data["episode_id"],
        save=lambda: None,
    )


def _line(line_id, char_id, role, text, *, skip=False):
    return {
        "line_id": line_id,
        "beat_id": line_id,
        "char_id": char_id,
        "speaker_role": role,
        "text": text,
        "skip": skip,
        "char_count": len(text),
        "word_count": canonical_word_count(text),
    }


@pytest.mark.parametrize(
    ("canonical", "expected"),
    [
        ("Dr. Hawking confirmed it.", "Doctor Hawking confirmed it."),
        ("Mr. Smith and Mrs. Smith arrived.", "Mister Smith and Missus Smith arrived."),
        ("It was Earth vs. Mars.", "It was Earth versus Mars."),
        ("Black & White.", "Black and White."),
        ("Off by half%.", "Off by half percent."),
        ("Meet @ noon.", "Meet at noon."),
    ],
)
def test_phase_7_stamps_pronounceable_projection_without_mutating_canonical(
    canonical,
    expected,
):
    line = _line("l001", "c01", "character", canonical)
    original_metrics = (line["word_count"], line["char_count"])
    led = _ledger([line])

    report = readiness.phase_7_audio_readiness(led)

    row = led.data["lines"][0]
    assert row["text"] == canonical
    assert (row["word_count"], row["char_count"]) == original_metrics
    assert row["text_for_tts"] == expected
    assert row["text_for_tts_source_sha256"] == (
        readiness.text_for_tts_source_sha256(canonical)
    )
    assert row["text_for_tts_receipt"]["changed"] is True
    assert report.lines_normalized == 1


def test_phase_7_stamps_identity_projection_when_no_normalization_is_needed():
    canonical = "Then we should go now."
    led = _ledger([_line("l001", "c01", "character", canonical)])

    report = readiness.phase_7_audio_readiness(led)

    row = led.data["lines"][0]
    assert row["text"] == canonical
    assert row["text_for_tts"] == canonical
    assert row["text_for_tts_receipt"]["changed"] is False
    assert report.lines_normalized == 0


def test_phase_7_expands_numbers_only_in_tts_projection():
    pytest.importorskip("num2words")
    canonical = "We need 42 samples and 1,234 controls."
    led = _ledger([_line("l001", "c01", "character", canonical)])

    report = readiness.phase_7_audio_readiness(led)

    row = led.data["lines"][0]
    assert row["text"] == canonical
    assert "42" not in row["text_for_tts"]
    assert "1,234" not in row["text_for_tts"]
    assert report.numbers_expanded == 2


def test_phase_7_without_num2words_keeps_numeric_projection_and_warns(monkeypatch):
    monkeypatch.setattr(readiness, "_try_import_num2words", lambda: None)
    canonical = "We need 42 samples."
    led = _ledger([_line("l001", "c01", "character", canonical)])

    report = readiness.phase_7_audio_readiness(led)

    row = led.data["lines"][0]
    assert row["text"] == canonical
    assert row["text_for_tts"] == canonical
    assert row["text_for_tts_receipt"]["num2words_available"] is False
    assert any("num2words" in warning for warning in report.warnings)


@pytest.mark.parametrize(
    "line",
    [
        _line("l001", "c01", "character", "", skip=True),
        _line("l002", "sfx", "sfx", "[door slam]"),
    ],
)
def test_phase_7_does_not_project_skipped_or_non_voiced_rows(line):
    led = _ledger([line])

    readiness.phase_7_audio_readiness(led)

    assert "text_for_tts" not in led.data["lines"][0]
    assert "text_for_tts_source_sha256" not in led.data["lines"][0]


def test_phase_7_stamps_audio_readiness_summary():
    led = _ledger([_line("l001", "c01", "character", "Mr. Smith.")])

    readiness.phase_7_audio_readiness(led)

    summary = led.data["meta"]["audio_readiness"]
    assert summary["lines_scanned"] == 1
    assert summary["lines_normalized"] == 1
    assert summary["abbreviations_expanded"] >= 1


class TestPhase8VideoReadiness:
    def test_all_portraits_present(self):
        led = _ledger([
            _line("l001", "c01", "character", "Hello."),
            _line("l002", "c02", "character", "Hi."),
        ])

        report = readiness.phase_8_video_readiness(led)

        assert report.cast_with_portrait == 3
        assert report.cast_missing_portrait == []
        assert report.voiced_lines == 2
        assert report.voiced_lines_with_visual_mapping == 2

    def test_missing_portrait_warns_without_failing(self):
        led = _ledger(
            [_line("l001", "c01", "character", "Hello.")],
            cast=[{"char_id": "c01", "name": "ALICE"}],
        )

        report = readiness.phase_8_video_readiness(led)

        assert report.cast_missing_portrait == ["c01"]
        assert report.voiced_lines_missing_visual == ["l001"]
        assert report.warnings

    def test_audit_preserves_lines_and_stamps_meta(self):
        led = _ledger([_line("l001", "c01", "character", "Hello.")])
        before = dict(led.data["lines"][0])

        readiness.phase_8_video_readiness(led)

        assert led.data["lines"][0] == before
        assert led.data["meta"]["video_readiness"]["voiced_lines"] == 1

    def test_all_supported_portrait_keys_are_recognized(self):
        led = _ledger([], cast=[
            {"char_id": "c01", "name": "ALICE", "portrait_path": "A.png"},
            {"char_id": "c02", "name": "BOB", "portrait_image": "B.png"},
            {"char_id": "c03", "name": "CARL", "portrait": "C.png"},
            {"char_id": "c04", "name": "DOE", "image_path": "D.png"},
        ])

        report = readiness.phase_8_video_readiness(led)

        assert report.cast_with_portrait == 4
        assert report.cast_missing_portrait == []


class TestCascadeReadiness:
    def test_metric_refresh_preserves_pre_diagnosis_and_cleans_post(self):
        from nodes import _otr_freeze_cascade as cascade

        text = "The dial says off\u2014it's actually elsewhere."
        led = _ledger([_line("l001", "c01", "character", text)])
        led.data["lines"][0].update({"word_count": 999, "char_count": 999})

        disposition = cascade.run_freeze_cascade(
            lambda *args, **kwargs: "",
            led,
            enable_phase_7_audio_readiness=False,
        )

        assert any(
            "word_count=999" in warning
            for warning in disposition.gap_audit_pre.warnings
        )
        assert not any(
            "word_count=" in warning
            for warning in disposition.gap_audit_post.warnings
        )
        row = led.data["lines"][0]
        expected = canonical_word_count(text)
        assert row["text"] == text
        assert row["word_count"] == expected
        assert row["char_count"] == len(text)
        assert led.data["total_word_count"] == expected
        assert led.data["meta"]["character_word_count"] == expected

    def test_phase_7_and_8_records_land_in_readiness_bucket(self):
        from nodes import _otr_freeze_cascade as cascade

        canonical = "Mr. Smith arrived."
        led = _ledger([_line("l001", "c01", "character", canonical)])

        cascade.run_freeze_cascade(lambda *args, **kwargs: "", led)

        names = [
            record["phase_name"]
            for record in led.data["meta"]["readiness_passes"]
        ]
        assert "phase_7_audio_readiness" in names
        assert "phase_8_video_readiness" in names
        cleanup_names = [
            record["phase_name"]
            for record in led.data["meta"].get("cleanup_passes", [])
        ]
        assert "phase_7_audio_readiness" not in cleanup_names
        assert "phase_8_video_readiness" not in cleanup_names
        row = led.data["lines"][0]
        assert row["text"] == canonical
        assert row["text_for_tts"] == "Mister Smith arrived."
        assert "video_readiness" in led.data["meta"]

    def test_phase_7_can_be_disabled(self):
        from nodes import _otr_freeze_cascade as cascade

        led = _ledger([
            _line("l001", "c01", "character", "Mr. Smith arrived."),
        ])

        cascade.run_freeze_cascade(
            lambda *args, **kwargs: "",
            led,
            enable_phase_7_audio_readiness=False,
        )

        assert "text_for_tts" not in led.data["lines"][0]
        assert led.data["meta"]["audio_readiness"]["skipped"] is True
