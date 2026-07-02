"""Story-Quality R2 -- S1: music / non-dialogue beats never render as
spoken or caption text.

Root cause (pre-S1): production_ledger.init_lines_from_outline stamped a
non-voiced row's `text` from `sfx_cue or intent`, so a music_inter beat
leaked its generic intent -- e.g. "Musical interlude bridging <phase> into
the next phase." -- into the line text, which then bled into the writer's
script transcript and any transcript consumer.

S1 fix (+ rip-sfx-broll 2026-07-01):
  * production_ledger: non-voiced rows carry NO text at all (the sfx_cue
    carrier and the [SFX:] transcript token died with the sfx subsystem);
  * _otr_outline: the inserted music_inter beat carries a neutral intent;
  * _otr_ledger_scrub.is_spoken_role: the single canonical spoken-role test.

These tests prove the suppression is robust even against a WORST-CASE
leaked intent, and that the music_inter row + voiced slot ids are unchanged.

Pure Python; no GPU, no LLM, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_ledger_scrub import is_spoken_role  # noqa: E402
from nodes._otr_outline import (  # noqa: E402
    Beat,
    Outline,
    stamp_dialogue_slot_ids,
)
from nodes.production_ledger import (  # noqa: E402
    Ledger,
    assemble_script_text_from_ledger,
)

_FILLER = "Musical interlude bridging the setup into the next phase."


# ---------------------------------------------------------------------------
# is_spoken_role -- the canonical helper
# ---------------------------------------------------------------------------


class TestIsSpokenRole:
    def test_voiced_roles_true(self):
        assert is_spoken_role("character") is True
        assert is_spoken_role("announcer") is True

    def test_non_voiced_roles_false(self):
        for role in ("music_open", "music_inter", "music_close"):
            assert is_spoken_role(role) is False

    def test_defensive_inputs_false(self):
        assert is_spoken_role(None) is False
        assert is_spoken_role("") is False
        assert is_spoken_role(123) is False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _outline_with_music_filler() -> Outline:
    """An outline whose music_inter beat carries a WORST-CASE leaked
    filler intent."""
    beats = [
        Beat(beat_id="b001", speaker="ANNOUNCER", speaker_role="announcer",
             intent="open the episode", target_words=15, mood="welcoming",
             arc_phase="setup"),
        Beat(beat_id="b002", speaker="ALICE", speaker_role="character",
             intent="raise the question", target_words=20, mood="wry",
             arc_phase="setup"),
        Beat(beat_id="b003", speaker="NARRATOR", speaker_role="music_inter",
             intent=_FILLER, target_words=5, mood="transitional",
             arc_phase="setup"),
        Beat(beat_id="b004", speaker="BOB", speaker_role="character",
             intent="reveal the obstacle", target_words=25, mood="grave",
             arc_phase="complication"),
        Beat(beat_id="b006", speaker="ANNOUNCER", speaker_role="announcer",
             intent="close the episode", target_words=15, mood="reflective",
             arc_phase="resolution"),
    ]
    outline = Outline(
        title="Test Episode",
        premise="A short test premise for S1 music suppression verification.",
        setting="Test setting",
        time_of_day="midnight",
        beats=beats,
    )
    stamp_dialogue_slot_ids(outline)
    return outline


def _ledger(tmp_path) -> Ledger:
    d = tmp_path / "audio"
    d.mkdir(parents=True, exist_ok=True)
    led = Ledger(episode_id="s1_test", out_dir=str(d))
    led.set_cast([
        {"char_id": "c01", "name": "ALICE"},
        {"char_id": "c02", "name": "BOB"},
        {"char_id": "c03", "name": "ANNOUNCER"},
    ])
    return led


# ---------------------------------------------------------------------------
# Materialization: non-voiced text suppression
# ---------------------------------------------------------------------------


class TestMaterialization:
    def test_music_inter_text_suppressed_intent_preserved(self, tmp_path):
        led = _ledger(tmp_path)
        led.init_lines_from_outline(
            _outline_with_music_filler(),
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        rows = {r["beat_id"]: r for r in led.data["lines"]}
        # The music_inter row exists but carries NO spoken/caption text.
        assert rows["b003"]["speaker_role"] == "music_inter"
        assert rows["b003"]["text"] == ""
        # The intent is preserved for visual / music-prompt consumers.
        assert rows["b003"]["beat_intent"] == _FILLER
        # Non-voiced rows never get a dialogue slot id.
        assert rows["b003"]["dialogue_slot_id"] is None

    def test_sfx_beats_cannot_exist_in_outline(self):
        # rip-sfx-broll: the SpeakerRole Literal has no "sfx" member --
        # pydantic rejects it at Beat construction (the outline CONTRACT
        # is the first gate).
        with pytest.raises(Exception):
            Beat(beat_id="b005", speaker="NARRATOR", speaker_role="sfx",
                 intent="footsteps approach off-stage", target_words=5,
                 mood="tense", arc_phase="complication")

    def test_beat_has_no_sfx_cue_field(self):
        assert "sfx_cue" not in Beat.model_fields

    def test_voiced_slot_ids_unchanged(self, tmp_path):
        """S1 must not perturb the voiced dialogue_slot_id stamping."""
        outline = _outline_with_music_filler()
        # Slot ids as stamped on the outline beats themselves.
        want = {
            b.beat_id: getattr(b, "dialogue_slot_id", None)
            for b in outline.beats
        }
        led = _ledger(tmp_path)
        led.init_lines_from_outline(
            outline, char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        got = {
            r["beat_id"]: r.get("dialogue_slot_id") for r in led.data["lines"]
        }
        assert got == want
        # Voiced beats carry a real slot id; non-voiced beats do not.
        assert got["b002"] is not None  # character
        assert got["b003"] is None      # music_inter


# ---------------------------------------------------------------------------
# Transcript: the assembled script text is filler-free
# ---------------------------------------------------------------------------


class TestTranscript:
    def test_no_filler_in_script_text(self, tmp_path):
        led = _ledger(tmp_path)
        led.init_lines_from_outline(
            _outline_with_music_filler(),
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        # Compose the voiced lines so the transcript has real content.
        led.update_line_text("b001", "Good evening, listeners.")
        led.update_line_text("b002", "Did you hear that signal?")
        led.update_line_text("b004", "We are not alone out here.")
        led.update_line_text("b006", "Until next time, stay tuned.")

        transcript = assemble_script_text_from_ledger(led.data)

        # The whole point of S1: no music filler reaches the transcript.
        assert "Musical interlude" not in transcript
        assert _FILLER not in transcript
        # rip-sfx-broll: the [SFX:] token no longer exists at all.
        assert "[SFX:" not in transcript
        # Voiced dialogue is intact.
        assert "Did you hear that signal?" in transcript

    def test_music_inter_row_count_unchanged(self, tmp_path):
        led = _ledger(tmp_path)
        led.init_lines_from_outline(
            _outline_with_music_filler(),
            char_id_by_name={"ALICE": "c01", "BOB": "c02"},
        )
        music_rows = [
            r for r in led.data["lines"]
            if r["speaker_role"] == "music_inter"
        ]
        assert len(music_rows) == 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
