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

import json

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
        # render_ms defaults to None (skip); passing 0 explicitly PERSISTS 0
        # under the cloud-audio-cache chunk 2 signature (2026-08-08). See
        # test_render_ms_zero_persists_when_explicit below.
        ok = OTRL.stamp_per_line_audio_meta(
            ledger, "l002",
            tts_engine="kokoro",
            voice_preset="",
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

    def test_render_ms_zero_persists_when_explicit(self, ledger):
        # r4 MF#4: cache-hit path stamps render_ms=0 to mean "no generation
        # time consumed". None still means "skip"; an explicit 0 must land.
        ok = OTRL.stamp_per_line_audio_meta(
            ledger, "l001",
            tts_engine="google_tts",
            render_ms=0,
        )
        assert ok is True
        row = ledger["lines"][0]
        assert row["render_ms"] == 0
        assert row["tts_engine"] == "google_tts"

    def test_stamps_cache_provenance_fields(self, ledger):
        # cloud-audio-cache chunk 2 (2026-08-08): new optional string kwargs.
        ok = OTRL.stamp_per_line_audio_meta(
            ledger, "l001",
            tts_engine="google_tts",
            audio_cache_key="cachekey123",
            audio_sha256="sha256abc",
            provider_model_id="gemini-2.5-flash-preview-tts",
        )
        assert ok is True
        row = ledger["lines"][0]
        assert row["audio_cache_key"] == "cachekey123"
        assert row["audio_sha256"] == "sha256abc"
        assert row["provider_model_id"] == "gemini-2.5-flash-preview-tts"

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


# ---------------------------------------------------------------------------
# PARTIAL-STAMP AND FLUSH-REPORTING COVERAGE, moved here 2026-09-24.
#
# These lived in tests/test_voice_route_reference_contract.py, which is slated
# for deletion with the voice-route subsystem. They are not about routes:
# `stamp_per_line_audio_meta` and `_persist_ledger_stamps` are shared ledger
# infrastructure every voice line goes through regardless of engine.
#
# Two of them were MIXED and have been split rather than moved whole -- the
# route-id assertions stayed behind to die with their field, the sample-rate and
# engine assertions came here. Moving a mixed test whole would have carried a
# dying field into a surviving file.
# ---------------------------------------------------------------------------

@pytest.fixture()
def no_in_flight_ledger(monkeypatch):
    """Pin the in-flight ledger singleton to absent.

    `_persist_ledger_stamps` PREFERS the in-flight singleton over the
    caller-supplied `meta.paths.ledger_path`, deliberately -- the wire value is
    attacker-controlled and can go stale, so the singleton wins when it exists.
    That makes these three tests depend on process-wide state: any earlier test
    in the session that constructs a Ledger leaves the singleton set, and then
    the tmp_path ledger below is never the one written.

    Found the hard way. These tests used to live in a file whose name sorted
    later in the run, where the singleton happened to be clear; moving them here
    put them after a test that leaves it set, and the full suite failed while the
    file passed alone. The position was never the contract -- this is. Every
    other test touching this helper pins it the same way.
    """
    from nodes import _otr_ledger as _OTRL
    monkeypatch.setattr(_OTRL, "in_flight_ledger_path", lambda: None)


def test_one_unstampable_line_does_not_degrade_the_whole_batch(
        tmp_path, no_in_flight_ledger):
    """One line missing from the ledger counts ONE degraded stamp, not two.

    This used to also assert WHICH line failed, through an out-parameter the
    flush filled for the voice route's receipt gate. The gate and the parameter
    went on 2026-09-24; the count is the surviving contract, and it is the half
    that says the good line was not blamed for the bad one.
    """
    from nodes._otr_voice_node_common import _persist_ledger_stamps

    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(
        {"lines": [{"line_id": "L1", "text": "hi"}]}), encoding="utf-8")
    meta = {"paths": {"ledger_path": str(ledger_path)}}

    degraded = _persist_ledger_stamps(
        meta,
        [("L1", {"tts_engine": "indextts2"}),
         ("L_NOT_IN_LEDGER", {"tts_engine": "indextts2"})],
        __import__("logging").getLogger("test"),
    )
    assert degraded == 1, (
        "the good line must not be blamed for the bad one")
    # ...and the good line really was written, which the count alone cannot say.
    rows = {r["line_id"]: r for r in
            json.loads(ledger_path.read_text(encoding="utf-8"))["lines"]}
    assert rows["L1"].get("tts_engine") == "indextts2"

def test_a_missing_ledger_path_blames_every_stamp(tmp_path, no_in_flight_ledger):
    from nodes._otr_voice_node_common import _persist_ledger_stamps

    degraded = _persist_ledger_stamps(
        {"paths": {}}, [("L1", {"tts_engine": "bark"})],
        __import__("logging").getLogger("test"))
    assert degraded == 1


# ---------------------------------------------------------------------------
# Per-line receipts.
# ---------------------------------------------------------------------------
def test_the_stamp_helper_records_the_sample_rate_and_engine():
    """The route half of this test went with the route subsystem; the sample
    rate and engine are ordinary per-line receipt fields every lane writes."""
    from nodes._otr_ledger import stamp_per_line_audio_meta

    led = {"lines": [{"line_id": "L1", "text": "hi"}]}
    assert stamp_per_line_audio_meta(
        led, "L1", tts_engine="indextts2", sample_rate=24000)
    line = led["lines"][0]
    assert line["sample_rate"] == 24000
    assert line["tts_engine"] == "indextts2"

def test_the_new_receipt_fields_are_skipped_when_empty():
    """Skip-when-empty, like every field beside them: a later role must not be
    able to blank a receipt an earlier one wrote."""
    from nodes._otr_ledger import stamp_per_line_audio_meta

    led = {"lines": [{"line_id": "L1", "text": "hi", "sample_rate": 24000}]}
    stamp_per_line_audio_meta(led, "L1", tts_engine="bark")
    assert led["lines"][0]["sample_rate"] == 24000
    assert led["lines"][0]["tts_engine"] == "bark"
