"""Story-Quality LIFT L3 (composer ACTION-marker strip) + L4 (minimal
transcript sanitizer). Both are AUDIO-AFFECTING and default-OFF: the flag-off
paths must be byte-identical (asserted), and the flag-on paths strip exactly
what they advertise."""
from __future__ import annotations

from nodes import _otr_line_hygiene as H
from nodes import _otr_line_composer as LC
from nodes._otr_ledger_scrub import scrub_ledger


# ---------------------------------------------------------------------------
# L3 -- strip_action_marker (pure)
# ---------------------------------------------------------------------------
class TestStripActionMarker:
    def test_strips_trailing_action_line(self):
        clean, action, n = H.strip_action_marker(
            "We have to decide now. ACTION: she slams the folder shut"
        )
        assert n == 1
        assert "slams" not in clean
        assert clean == "We have to decide now."
        assert "slams the folder shut" in action

    def test_no_marker_no_op(self):
        s = "Just a normal spoken line."
        clean, action, n = H.strip_action_marker(s)
        assert (clean, action, n) == (s, "", 0)

    def test_idempotent(self):
        s = "Hold on. ACTION: he steps back"
        once, _, _ = H.strip_action_marker(s)
        twice, _, n2 = H.strip_action_marker(once)
        assert twice == once
        assert n2 == 0

    def test_entire_line_is_action_collapses(self):
        clean, action, n = H.strip_action_marker("ACTION: lights dim")
        assert n == 1
        assert clean == ""


# ---------------------------------------------------------------------------
# L4 -- sanitize_transcript_text / detect_mojibake (pure)
# ---------------------------------------------------------------------------
class TestSanitizeTranscript:
    def test_strips_director_note(self):
        clean, reasons = H.sanitize_transcript_text(
            "Note: speak softly\nWe begin the broadcast."
        )
        assert "Note:" not in clean
        assert "We begin the broadcast." in clean
        assert any(r.startswith("prompt_leak") for r in reasons)

    def test_strips_voice_should(self):
        clean, reasons = H.sanitize_transcript_text(
            "The voice should be calm. The signal is gone."
        )
        assert "voice should" not in clean.casefold()
        assert "The signal is gone." in clean

    def test_balances_stray_wrapper_quote(self):
        clean, reasons = H.sanitize_transcript_text('"We are live')
        assert clean == "We are live"
        assert "wrapper_quote_balanced" in reasons

    def test_clean_line_unchanged(self):
        s = "We are on the air in ten seconds."
        clean, reasons = H.sanitize_transcript_text(s)
        assert clean == s
        assert reasons == []

    def test_apostrophes_preserved(self):
        s = "It's Dana's call, and we're out of time."
        clean, reasons = H.sanitize_transcript_text(s)
        assert clean == s

    def test_mojibake_detected_not_repaired(self):
        bad = "the cafÃ© is closed"   # "café" mojibake
        clean, reasons = H.sanitize_transcript_text(bad)
        assert "mojibake_detected" in reasons
        assert clean == bad   # verify-only: NEVER mutated in v0


# ---------------------------------------------------------------------------
# L3 -- composer prompt instruction gated on the flag
# ---------------------------------------------------------------------------
class TestComposerActionPrompt:
    def _req(self):
        return LC.LineRequest(
            speaker="ALEX", intent="say something", mood="tense",
            target_words=20, speaker_role="character",
            canon_header="", last_lines=[],
        )

    def test_flag_off_no_action_instruction(self, monkeypatch):
        monkeypatch.delenv("OTR_COMPOSER_ACTION_STRIP", raising=False)
        prompt = LC._build_user_prompt(self._req())
        assert "ACTION:" not in prompt

    def test_flag_on_adds_action_instruction(self, monkeypatch):
        monkeypatch.setenv("OTR_COMPOSER_ACTION_STRIP", "1")
        prompt = LC._build_user_prompt(self._req())
        assert "ACTION:" in prompt

    def test_byte_identical_when_off(self, monkeypatch):
        monkeypatch.delenv("OTR_COMPOSER_ACTION_STRIP", raising=False)
        a = LC._build_user_prompt(self._req())
        monkeypatch.setenv("OTR_COMPOSER_ACTION_STRIP", "0")
        b = LC._build_user_prompt(self._req())
        assert a == b


# ---------------------------------------------------------------------------
# L4 -- scrub integration gated on the flag
# ---------------------------------------------------------------------------
class TestScrubTranscriptSanitizer:
    def _ledger(self) -> dict:
        return {
            "schema_version": "l3", "episode_id": "san", "meta": {},
            "cast": [
                {"char_id": "c01", "name": "DANA", "gender": "female",
                 "tts_model": "bark", "voice_preset": "v2/en_speaker_4"},
            ],
            "lines": [
                {"line_id": "b001", "beat_id": "b001", "char_id": "c01",
                 "speaker_role": "character",
                 "text": "Note: speak softly. The signal is gone.",
                 "word_count": 7, "char_count": 39},
            ],
        }

    def test_flag_on_sanitizes(self, monkeypatch):
        monkeypatch.setenv("OTR_TRANSCRIPT_SANITIZER", "1")
        led = self._ledger()
        scrub_ledger(led, repair_available=True)
        row = led["lines"][0]
        assert "Note:" not in row["text"]
        assert any(
            f.startswith("transcript_sanitized:")
            for f in row.get("compose_flags", [])
        )

    def test_flag_off_byte_identical(self, monkeypatch):
        monkeypatch.delenv("OTR_TRANSCRIPT_SANITIZER", raising=False)
        led = self._ledger()
        before = led["lines"][0]["text"]
        scrub_ledger(led, repair_available=True)
        row = led["lines"][0]
        assert row["text"] == before
        assert not any(
            f.startswith("transcript_sanitized:")
            for f in row.get("compose_flags", [])
        )
