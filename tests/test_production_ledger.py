"""Unit tests for the production ledger (L1 scope: write-only, incremental
saves, fault-tolerant writes)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nodes.production_ledger import (  # noqa: E402
    Ledger,
    get_ledger,
    new_ledger,
    _slugify,
    _word_count,
    _char_count,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_out(tmp_path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# Utility coverage
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_slugify_basic(self):
        assert _slugify("Black Sphere") == "black_sphere"

    def test_slugify_strips_junk(self):
        assert _slugify("  I am Artemis!  Peter Rossoni ") == "i_am_artemis_peter_rossoni"

    def test_slugify_empty_returns_default(self):
        assert _slugify("") == "episode"
        assert _slugify(None) == "episode"

    def test_slugify_limits_length(self):
        assert len(_slugify("a" * 200, limit=40)) == 40

    def test_word_count_ignores_punctuation(self):
        assert _word_count("Hello, world! How's it going?") == 5

    def test_word_count_handles_empty(self):
        assert _word_count("") == 0
        assert _word_count(None) == 0

    def test_char_count_is_len(self):
        assert _char_count("abcdef") == 6
        assert _char_count("") == 0
        assert _char_count(None) == 0


# ---------------------------------------------------------------------------
# Ledger construction + save
# ---------------------------------------------------------------------------

class TestLedgerBasics:
    def test_new_ledger_creates_structure(self, tmp_out):
        led = Ledger("signal_lost_test_20260424_000000", str(tmp_out))
        assert led.episode_id == "signal_lost_test_20260424_000000"
        # Schema bumped 2026-04-25 PM (l1 -> l2) for the beats[] hierarchy.
        assert led.data["schema_version"].startswith("l2-")
        for key in ("cast", "scenes", "shots", "beats", "lines",
                    "sfx", "music", "clips"):
            assert led.data[key] == []
        assert led.data["final_audio_path"] is None
        assert led.data["final_video_path"] is None
        assert led.data["total_episode_dur_s"] is None
        assert led.data["total_beats"] == 0


class TestLedgerBeats:
    def test_set_beats_round_trips(self, tmp_out):
        led = Ledger("test_beats", str(tmp_out))
        led.set_beats([
            {"beat_id": "shot_001_b1", "shot_id": "shot_001",
             "scene_id": "scene_lab", "speaker": "ALICE", "char_id": "c01",
             "line_ids": ["l001", "l002"], "start_s": 0.0, "dur_s": 5.5},
            {"beat_id": "shot_001_b2", "shot_id": "shot_001",
             "scene_id": "scene_lab", "speaker": "BOB", "char_id": "c02",
             "line_ids": ["l003"], "start_s": 5.5, "dur_s": 3.2},
        ])
        assert len(led.data["beats"]) == 2
        assert led.data["total_beats"] == 2
        b1 = led.data["beats"][0]
        assert b1["beat_id"] == "shot_001_b1"
        assert b1["speaker"] == "ALICE"
        assert b1["line_ids"] == ["l001", "l002"]
        assert b1["start_s"] == 0.0
        assert b1["dur_s"] == 5.5

    def test_set_lines_carries_beat_id_and_boundary(self, tmp_out):
        led = Ledger("test_beat_lines", str(tmp_out))
        led.set_lines([
            {"line_id": "l001", "shot_id": "shot_001",
             "beat_id": "shot_001_b1", "boundary": "shot_start",
             "char_id": "c01", "text": "Hello", "start_s": 0.0, "dur_s": 1.0},
            {"line_id": "l002", "shot_id": "shot_001",
             "beat_id": "shot_001_b2", "boundary": "beat_start",
             "char_id": "c02", "text": "Hi", "start_s": 1.0, "dur_s": 1.0},
            {"line_id": "l003", "shot_id": "shot_001",
             "beat_id": "shot_001_b2", "boundary": "continue",
             "char_id": "c02", "text": "There", "start_s": 2.0, "dur_s": 1.0},
        ])
        rows = led.data["lines"]
        assert rows[0]["beat_id"] == "shot_001_b1"
        assert rows[0]["boundary"] == "shot_start"
        assert rows[1]["boundary"] == "beat_start"
        assert rows[2]["boundary"] == "continue"

    def test_set_lines_omitting_beat_fields_back_compat(self, tmp_out):
        """Older callers pre-dating beats can omit beat_id + boundary;
        ledger stores None and downstream treats as shot_start."""
        led = Ledger("test_back_compat", str(tmp_out))
        led.set_lines([
            {"line_id": "l001", "shot_id": "shot_001", "char_id": "c01",
             "text": "Hello", "start_s": 0.0, "dur_s": 1.0},
        ])
        row = led.data["lines"][0]
        assert row["beat_id"] is None
        assert row["boundary"] is None

    def test_path_follows_episode_id(self, tmp_out):
        led = Ledger("signal_lost_BLACK SPHERE_20260424", str(tmp_out))
        assert led.path.endswith("signal_lost_black_sphere_20260424_ledger.json")

    def test_save_writes_valid_json(self, tmp_out):
        led = Ledger("test_ep", str(tmp_out))
        path = led.save()
        assert path is not None
        assert Path(path).exists()
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        assert loaded["episode_id"] == "test_ep"
        assert loaded["total_dialogue_lines"] == 0

    def test_save_is_atomic(self, tmp_out):
        led = Ledger("atomic_test", str(tmp_out))
        led.save()
        # No stray .tmp file should remain after a successful save
        assert not Path(tmp_out / "atomic_test_ledger.json.tmp").exists()

    def test_rename_updates_path_and_data(self, tmp_out):
        led = Ledger("pending_20260424_000000", str(tmp_out))
        led.rename_episode("signal_lost_black_sphere_20260424_142006")
        assert led.data["episode_id"] == "signal_lost_black_sphere_20260424_142006"
        assert "black_sphere" in led.path


# ---------------------------------------------------------------------------
# Setters
# ---------------------------------------------------------------------------

class TestSetters:
    def _sample_cast(self):
        return [
            {"char_id": "c01", "name": "EDNA",
             "gender": "female", "voice_preset": "v2/en_speaker_2"},
            {"char_id": "c02", "name": "BOB",
             "gender": "male", "voice_preset": "v2/en_speaker_1"},
        ]

    def _sample_lines(self):
        return [
            {"line_id": "l001", "shot_id": "sh01", "char_id": "c01",
             "text": "We're flying blind without GPS.",
             "traits": "Female, 40s, urgent"},
            {"line_id": "l002", "shot_id": "sh01", "char_id": "c02",
             "text": "Edna, I've taken control of some military satellites.",
             "traits": "Male, 30s, clipped"},
            {"line_id": "l003", "shot_id": "sh02", "char_id": "c01",
             "text": "Bob? What are you doing here?",
             "traits": "startled"},
        ]

    def test_set_cast_normalizes_rows(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_cast(self._sample_cast())
        assert len(led.data["cast"]) == 2
        assert led.data["cast"][0]["char_id"] == "c01"
        assert led.data["cast"][0]["gender"] == "female"

    def test_set_lines_computes_counts(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_lines(self._sample_lines())
        rows = led.data["lines"]
        assert len(rows) == 3
        assert rows[0]["char_count"] == len("We're flying blind without GPS.")
        assert rows[0]["word_count"] == 5
        assert led.data["total_dialogue_lines"] == 3
        total_words = sum(r["word_count"] for r in rows)
        assert led.data["total_word_count"] == total_words

    def test_totals_roll_up_per_character(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_cast(self._sample_cast())
        led.set_lines(self._sample_lines())
        edna = next(c for c in led.data["cast"] if c["char_id"] == "c01")
        bob = next(c for c in led.data["cast"] if c["char_id"] == "c02")
        assert edna["line_count"] == 2
        assert bob["line_count"] == 1
        assert edna["word_count"] > 0
        assert bob["word_count"] > 0

    def test_totals_roll_up_per_scene_via_shot(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_scenes([{"scene_id": "s01", "env": "NASA control room"}])
        led.set_shots([
            {"shot_id": "sh01", "scene_id": "s01", "visual_prompt": "X"},
            {"shot_id": "sh02", "scene_id": "s01", "visual_prompt": "Y"},
        ])
        led.set_lines(self._sample_lines())
        s01 = led.data["scenes"][0]
        assert s01["line_count"] == 3  # all three lines land in s01

    def test_set_final_paths(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_final_paths(
            audio_path="/tmp/final.wav",
            video_path="/tmp/final.mp4",
            total_episode_dur_s=421.5,
        )
        assert led.data["final_audio_path"] == "/tmp/final.wav"
        assert led.data["final_video_path"] == "/tmp/final.mp4"
        assert led.data["total_episode_dur_s"] == 421.5


# ---------------------------------------------------------------------------
# Timing back-fill
# ---------------------------------------------------------------------------

class TestTimingBackfill:
    def test_apply_line_timings(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_lines([
            {"line_id": "l001", "char_id": "c01", "text": "A"},
            {"line_id": "l002", "char_id": "c02", "text": "B"},
        ])
        led.apply_line_timings({
            "l001": {"start_s": 12.4, "dur_s": 2.1, "bark_wav_path": "/tmp/a.wav"},
            "l002": {"start_s": 14.5, "dur_s": 1.8, "bark_wav_path": "/tmp/b.wav"},
        })
        rows = led.data["lines"]
        assert rows[0]["start_s"] == 12.4
        assert rows[0]["bark_wav_path"] == "/tmp/a.wav"
        assert rows[1]["start_s"] == 14.5

    def test_apply_line_timings_ignores_unknown_ids(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_lines([{"line_id": "l001", "text": "Hi"}])
        led.apply_line_timings({"l999": {"start_s": 1.0, "dur_s": 1.0}})
        assert led.data["lines"][0]["start_s"] is None

    def test_apply_sfx_and_music_timings(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_sfx([{"cue_id": "sfx_001", "description": "alarm"}])
        led.set_music([{"cue_id": "opening"}])
        led.apply_sfx_timings({"sfx_001": {"start_s": 5.0, "dur_s": 1.2, "wav_path": "/tmp/a.wav"}})
        led.apply_music_timings({"opening": {"start_s": 0.0, "dur_s": 12.0, "wav_path": "/tmp/m.wav"}})
        assert led.data["sfx"][0]["start_s"] == 5.0
        assert led.data["music"][0]["dur_s"] == 12.0


# ---------------------------------------------------------------------------
# Fault tolerance
# ---------------------------------------------------------------------------

class TestFaultTolerance:
    def test_bad_input_does_not_raise(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_cast([{"char_id": None, "line_count": "bogus"}])
        # Should have coerced line_count to 0 rather than raising.
        assert led.data["cast"][0]["line_count"] == 0

    def test_save_never_raises_even_with_bad_out_dir(self, tmp_out, monkeypatch):
        led = Ledger("t", str(tmp_out))
        # Simulate disk failure by monkeypatching os.replace.
        import nodes.production_ledger as pl
        def _boom(*a, **kw):
            raise OSError("disk full")
        monkeypatch.setattr(pl.os, "replace", _boom)
        assert led.save() is None  # returned None, did not raise

    def test_save_is_idempotent(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_lines([{"line_id": "l001", "text": "Hi"}])
        p1 = led.save()
        p2 = led.save()
        assert p1 == p2
        assert Path(p1).exists()


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------

class TestSingletonAccessors:
    def test_new_ledger_resets_current(self, tmp_out):
        a = new_ledger("ep_a", str(tmp_out))
        assert get_ledger() is a
        b = new_ledger("ep_b", str(tmp_out))
        assert get_ledger() is b
        assert b.episode_id == "ep_b"

    def test_get_ledger_creates_placeholder_if_none(self, tmp_out):
        # Reset singleton via new_ledger then null it through a fresh import would
        # be overkill; just confirm get_ledger never returns None.
        led = get_ledger()
        assert led is not None
