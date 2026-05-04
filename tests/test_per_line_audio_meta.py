"""
tests/test_per_line_audio_meta.py -- BUG-LOCAL-030 audit-completion
============================================================================

Regression suite for the new per-line audio render metadata helpers
(``compute_audio_sample_hash``, ``stamp_per_line_audio_meta``) and the
end-to-end stamping flow used by BatchBark / KokoroAnnouncer /
MusicGenTheme / BatchAudioGen.

Acceptance gates:
  * ``compute_audio_sample_hash`` returns a stable 8-char hex for the
    same byte input, different hex for different input
  * ``stamp_per_line_audio_meta`` skips empty/zero values so a partial
    bundle does not overwrite previously-stamped fields
  * ``stamp_per_line_audio_meta`` returns False when the line_id does
    not exist (no silent ledger pollution)
  * Stamping is additive — does NOT clobber pre-existing per-line
    fields like text_for_tts or bark_render_ms
"""
from __future__ import annotations

import numpy as np
import pytest

from nodes import _otr_ledger as OTRL


class TestComputeAudioSampleHash:
    def test_deterministic_for_same_bytes(self):
        buf = b"\x00\x01\x02\x03" * 256
        h1 = OTRL.compute_audio_sample_hash(buf)
        h2 = OTRL.compute_audio_sample_hash(buf)
        assert h1 == h2
        assert len(h1) == 8

    def test_differs_for_different_bytes(self):
        h1 = OTRL.compute_audio_sample_hash(b"\x00" * 1024)
        h2 = OTRL.compute_audio_sample_hash(b"\x01" * 1024)
        assert h1 != h2

    def test_handles_numpy_array(self):
        arr = np.zeros(2048, dtype=np.float32)
        h = OTRL.compute_audio_sample_hash(arr)
        assert isinstance(h, str)
        assert len(h) == 8

    def test_returns_empty_on_unhashable(self):
        # Plain int has no .tobytes() and is not bytes -> ""
        h = OTRL.compute_audio_sample_hash(42)
        assert h == ""

    def test_only_hashes_leading_n_bytes(self):
        # First 1024 bytes identical, rest differs -> same hash
        prefix = bytes(range(256)) * 4  # 1024 bytes
        a = prefix + b"\x00" * 100
        b = prefix + b"\xff" * 100
        assert OTRL.compute_audio_sample_hash(a) == OTRL.compute_audio_sample_hash(b)


class TestStampPerLineAudioMeta:
    @pytest.fixture
    def ledger(self):
        return {
            "lines": [
                {"line_id": "l001", "text": "Hello", "bark_render_ms": 1234},
                {"line_id": "l002", "text": "World"},
            ]
        }

    def test_stamps_full_bundle(self, ledger):
        ok = OTRL.stamp_per_line_audio_meta(
            ledger, "l001",
            tts_engine="bark",
            voice_preset="v2/en_speaker_6",
            render_ms=987,
            generated_dur_s=2.5,
            audio_sample_hash="abcd1234",
        )
        assert ok is True
        row = ledger["lines"][0]
        assert row["tts_engine"] == "bark"
        assert row["voice_preset"] == "v2/en_speaker_6"
        assert row["render_ms"] == 987
        assert row["generated_dur_s"] == 2.5
        assert row["audio_sample_hash"] == "abcd1234"
        # Pre-existing field preserved
        assert row["bark_render_ms"] == 1234

    def test_skips_empty_values(self, ledger):
        ok = OTRL.stamp_per_line_audio_meta(
            ledger, "l002",
            tts_engine="kokoro",
            voice_preset="",
            render_ms=0,
            generated_dur_s=0.0,
            audio_sample_hash="",
        )
        assert ok is True
        row = ledger["lines"][1]
        assert row["tts_engine"] == "kokoro"
        # Optional fields omitted entirely (no zero / empty entries)
        assert "voice_preset" not in row
        assert "render_ms" not in row
        assert "generated_dur_s" not in row
        assert "audio_sample_hash" not in row

    def test_returns_false_for_unknown_line(self, ledger):
        ok = OTRL.stamp_per_line_audio_meta(
            ledger, "l999",
            tts_engine="bark",
        )
        assert ok is False
        # Existing rows untouched
        assert "tts_engine" not in ledger["lines"][0]
        assert "tts_engine" not in ledger["lines"][1]

    def test_engine_only_partial_bundle(self, ledger):
        # MusicGen / AudioGen path: stamp engine + render_ms + hash only
        ok = OTRL.stamp_per_line_audio_meta(
            ledger, "l001",
            tts_engine="musicgen",
            render_ms=4500,
            audio_sample_hash="deadbeef",
        )
        assert ok is True
        row = ledger["lines"][0]
        assert row["tts_engine"] == "musicgen"
        assert row["render_ms"] == 4500
        assert row["audio_sample_hash"] == "deadbeef"
        assert "voice_preset" not in row
        assert "generated_dur_s" not in row

    def test_never_raises_on_bad_ledger(self):
        # Passing something without a lines key should return False
        ok = OTRL.stamp_per_line_audio_meta(
            {}, "l001", tts_engine="bark",
        )
        assert ok is False
