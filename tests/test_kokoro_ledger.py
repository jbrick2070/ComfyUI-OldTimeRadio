"""tests/test_kokoro_ledger.py -- self-test for the ledger-pure
kokoro_announcer rewrite.

Mocks the kokoro pipeline + the HF voice file fetch + the on-disk
ledger save so the parsing + stamping logic runs CPU-only in well
under a second.

Covers the four canonical Pattern 7 cases:
  - test_kokoro_iter_announcer_only -- character / sfx / music lines
    are NOT stamped; only announcer lines reach per_line_meta
  - test_kokoro_stamps_by_line_id -- multiple announcer lines stamped
    correctly via line_id (works even with duplicate text)
  - test_kokoro_voice_fallback -- empty production_plan + voice_override
    'random' still produces a valid grab-bag pick (no None / empty)
  - test_kokoro_legacy_list_raises -- load_ledger ValueError surfaces
    (Kokoro is in the loud-fail group)
"""
from __future__ import annotations

import json
import sys
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.fixtures.ledger_stub import make_legacy_list, make_stub_ledger


# ---------------------------------------------------------------------------
# Fake kokoro module (the real package may not be installed in CI; even
# if it is, we don't want to download voice .pt files in a unit test).
# ---------------------------------------------------------------------------

class _FakeKPipeline:
    """Stand-in for kokoro.KPipeline. Yields a deterministic 1.0s mono
    buffer per call so each line gets a slightly different dur_s
    proportional to text length."""

    def __init__(self, lang_code, device, repo_id):
        self.model = MagicMock()

    def __call__(self, text, voice, speed=0.95, split_pattern=None):
        # Length encodes input length so the test can assert distinct
        # dur_s per line_id. Tiny sleep so the consumer's render_ms
        # measurement clears the integer-millisecond floor (Kokoro
        # stamps render_ms only when > 0; 0 = "didn't measure").
        time.sleep(0.005)
        samples = 24000 + min(len(text) * 10, 5000)
        audio = np.zeros(samples, dtype=np.float32)
        yield (None, None, audio)


@pytest.fixture
def patched_kokoro_env(tmp_path, monkeypatch):
    """Patch the GPU + ledger I/O surfaces so render() is hermetic.

    Pre-seed state['led_disk'] before calling render(); read it back
    after to inspect the merged ledger.
    """
    state = {"led_disk": None}

    # Inject a fake `kokoro` module BEFORE the consumer does its lazy
    # `from kokoro import KPipeline` inside render().
    fake_kokoro = MagicMock()
    fake_kokoro.KPipeline = _FakeKPipeline
    monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro)

    def _fake_in_flight_path():
        return tmp_path / "stub_ledger.json"

    def _fake_load_ledger_safe(_path):
        return state["led_disk"]

    def _fake_save_ledger_safe(_path, led):
        state["led_disk"] = led
        return True

    def _fake_ensure_voice_file(voice_id):
        return f"/fake/voices/{voice_id}.pt"

    with patch(
        "nodes.kokoro_announcer._ensure_voice_file",
        side_effect=_fake_ensure_voice_file,
    ), patch(
        "nodes._otr_ledger.in_flight_ledger_path",
        side_effect=_fake_in_flight_path,
    ), patch(
        "nodes._otr_ledger.load_ledger_safe",
        side_effect=_fake_load_ledger_safe,
    ), patch(
        "nodes._otr_ledger.save_ledger_safe",
        side_effect=_fake_save_ledger_safe,
    ):
        yield state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_kokoro_iter_announcer_only(patched_kokoro_env):
    """character / sfx / music lines must NOT be touched. Only the
    announcer line gets stamped."""
    from nodes.kokoro_announcer import KokoroAnnouncer

    led_in = make_stub_ledger()  # 2 character + 1 announcer + 1 sfx + 2 music
    patched_kokoro_env["led_disk"] = json.loads(json.dumps(led_in))
    script_json = json.dumps(led_in)

    audio_out, _render_log, voice_id = KokoroAnnouncer().render(
        script_json=script_json,
        episode_seed="alpha-test-seed",
        voice_override="random",
        speed=0.95,
    )

    waveform = audio_out["waveform"]
    assert waveform.shape[0] == 1, (
        f"expected 1 announcer clip; got {waveform.shape[0]}"
    )
    assert audio_out["sample_rate"] == 24000

    saved_led = patched_kokoro_env["led_disk"]
    by_id = {ln["line_id"]: ln for ln in saved_led["lines"]}

    # Announcer row b002 stamped with kokoro fields.
    assert by_id["b002"].get("tts_engine") == "kokoro"
    assert by_id["b002"].get("voice_preset") == voice_id
    assert by_id["b002"].get("dur_s", 0) > 0
    assert by_id["b002"].get("start_s") == 0.0  # first announcer
    assert "render_ms" in by_id["b002"]
    assert "generated_dur_s" in by_id["b002"]

    # Character / sfx / music rows must NOT have a tts_engine field
    # stamped by Kokoro (they go through Bark / AudioGen / LTX).
    assert "tts_engine" not in by_id["b001"], "music_open not stamped"
    assert "tts_engine" not in by_id["b003"], "character c01 not stamped"
    assert "tts_engine" not in by_id["b004"], "character c02 not stamped"
    assert "tts_engine" not in by_id["b005"], "sfx not stamped"
    assert "tts_engine" not in by_id["b006"], "music_close not stamped"


def test_kokoro_stamps_by_line_id_with_duplicate_text(patched_kokoro_env):
    """Two announcer lines with IDENTICAL text -- the old text-match
    block would have collided. line_id-keyed patching stamps both
    correctly."""
    from nodes.kokoro_announcer import KokoroAnnouncer

    # Hand-craft a ledger with two announcer lines saying the same thing.
    led_in = {
        "meta":  {"title": "DUP-ANN", "news_seed": {"headline": "x"}},
        "cast":  [],
        "lines": [
            {"line_id": "ann_first", "char_id": "announcer",
             "speaker_role": "announcer", "text": "Stand by.",
             "traits": "warm", "start_s": None, "dur_s": None,
             "boundary": "ep_open", "bark_wav_path": None},
            {"line_id": "ann_second", "char_id": "announcer",
             "speaker_role": "announcer", "text": "Stand by.",
             "traits": "warm", "start_s": None, "dur_s": None,
             "boundary": "ep_close", "bark_wav_path": None},
        ],
    }
    patched_kokoro_env["led_disk"] = json.loads(json.dumps(led_in))
    script_json = json.dumps(led_in)

    KokoroAnnouncer().render(
        script_json=script_json,
        episode_seed="dup-test",
        voice_override="random",
        speed=0.95,
    )

    saved_led = patched_kokoro_env["led_disk"]
    by_id = {ln["line_id"]: ln for ln in saved_led["lines"]}

    # Both rows stamped, despite identical text.
    for lid in ("ann_first", "ann_second"):
        row = by_id[lid]
        assert row["tts_engine"] == "kokoro", f"{lid} missing tts_engine"
        assert row.get("dur_s", 0) > 0, f"{lid} missing dur_s"
        assert row.get("voice_preset"), f"{lid} missing voice_preset"
        assert "render_ms" in row, f"{lid} missing render_ms"
        assert "audio_sample_hash" in row, f"{lid} missing audio_sample_hash"

    # start_s monotonic across the announcer cumulative timeline.
    assert by_id["ann_first"]["start_s"] == 0.0
    assert by_id["ann_second"]["start_s"] == pytest.approx(
        by_id["ann_first"]["dur_s"], rel=1e-6,
    )


def test_kokoro_voice_fallback(patched_kokoro_env):
    """Empty episode_seed + voice_override='random' -- the seeded
    grab-bag picker must still produce a valid voice (one of the four
    pool entries, never empty/None)."""
    from nodes.kokoro_announcer import ANNOUNCER_VOICE_POOL, KokoroAnnouncer

    led_in = make_stub_ledger()
    patched_kokoro_env["led_disk"] = json.loads(json.dumps(led_in))
    script_json = json.dumps(led_in)

    _audio_out, _render_log, voice_id = KokoroAnnouncer().render(
        script_json=script_json,
        episode_seed="",  # explicitly empty -- must still pick
        voice_override="random",
        speed=0.95,
    )

    assert voice_id in ANNOUNCER_VOICE_POOL, (
        f"voice fallback returned {voice_id!r}; expected one of "
        f"{ANNOUNCER_VOICE_POOL}"
    )

    # Same empty seed must produce the same voice on a second call
    # (deterministic seeded grab-bag).
    _audio2, _log2, voice_id2 = KokoroAnnouncer().render(
        script_json=script_json,
        episode_seed="",
        voice_override="random",
        speed=0.95,
    )
    assert voice_id == voice_id2, (
        "seeded grab-bag should be deterministic for identical seed"
    )


def test_kokoro_legacy_list_raises():
    """Legacy parser-list shape on the wire must raise ValueError --
    Kokoro is in the loud-fail group (Pattern 1)."""
    from nodes.kokoro_announcer import KokoroAnnouncer

    legacy_json = json.dumps(make_legacy_list())

    with pytest.raises(ValueError) as excinfo:
        KokoroAnnouncer().render(
            script_json=legacy_json,
            episode_seed="seed",
            voice_override="random",
            speed=0.95,
        )

    msg = str(excinfo.value)
    assert "legacy parser-list" in msg or "OTR_LedgerScriptWriter" in msg, (
        f"ValueError message should name the legacy shape; got: {msg!r}"
    )
