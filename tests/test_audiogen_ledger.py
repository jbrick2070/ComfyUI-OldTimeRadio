"""tests/test_audiogen_ledger.py -- self-test for the ledger-pure
batch_audiogen_generator rewrite.

Pre-seeds the AudioGen cache with valid WAVs so every cue is a
cache-hit (no real generation). Mocks the on-disk ledger I/O via
state-backed in_flight_ledger_path / load_ledger_safe / save_ledger_safe.

Covers four canonical Pattern 7 cases:
  - test_audiogen_iter_sfx_only -- character/announcer/music lines
    are NOT stamped; only sfx lines reach render_queue
  - test_audiogen_stamps_by_line_id_with_duplicate_text -- two sfx
    lines with identical cue text both stamped via line_id
  - test_audiogen_cache_hit_path -- cache-hit path still stamps
    line_id-keyed write-back (BUG-LOCAL-095 surface)
  - test_audiogen_legacy_list_raises -- load_ledger ValueError surfaces
"""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from tests.fixtures.ledger_stub import make_legacy_list, make_stub_ledger

AUDIOGEN_SAMPLE_RATE = 32000

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _no_op(*_args, **_kwargs):
    return None

@pytest.fixture
def patched_audiogen_env(tmp_path):
    """Patch the GPU + ledger I/O surfaces. Cache dir is the tmp_path
    (state['cache_dir']) and writable. Pre-seed state['led_disk']
    before calling generate(); inspect after."""
    cache_dir = tmp_path / "sfx_cache"
    cache_dir.mkdir()

    state = {"led_disk": None, "cache_dir": str(cache_dir)}

    def _fake_cache_dir():
        return state["cache_dir"]

    def _fake_in_flight_path():
        return tmp_path / "stub_ledger.json"

    def _fake_load_ledger_safe(_path):
        return state["led_disk"]

    def _fake_save_ledger_safe(_path, led):
        state["led_disk"] = led
        return True

    with patch(
        "nodes.batch_audiogen_generator.force_vram_offload",
        side_effect=_no_op,
    ), patch(
        "nodes.batch_audiogen_generator._cache_dir",
        side_effect=_fake_cache_dir,
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

def _seed_cache_for_cue(cache_dir: str, prompt: str, duration: float,
                       episode_seed: str) -> str:
    """Pre-write a valid WAV file at the canonical cache filename so
    the consumer's _find_cached returns a hit and skips real generation."""
    from nodes.batch_audiogen_generator import (
        _cache_filename_for_write,
    )
    fname = _cache_filename_for_write(prompt, duration, episode_seed)
    fpath = os.path.join(cache_dir, fname)
    # 1.0s mono buffer at AudioGen sample rate
    samples = AUDIOGEN_SAMPLE_RATE
    data = np.zeros(samples, dtype=np.float32)
    sf.write(fpath, data, AUDIOGEN_SAMPLE_RATE, subtype="FLOAT", format="WAV")
    return fpath

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_audiogen_iter_sfx_only(patched_audiogen_env):
    """character / announcer / music lines must NOT be stamped. Only
    the sfx line gets the audiogen write-back."""
    from nodes.batch_audiogen_generator import BatchAudioGenGenerator

    led_in = make_stub_ledger()  # 2 character + 1 announcer + 1 sfx + 2 music
    patched_audiogen_env["led_disk"] = json.loads(json.dumps(led_in))
    script_json = json.dumps(led_in)

    # Pre-seed cache for the single sfx cue ("metal door slam")
    # at the default duration so generation is skipped entirely.
    _seed_cache_for_cue(
        patched_audiogen_env["cache_dir"],
        prompt="metal door slam",
        duration=3.0,
        episode_seed="test-iter",
    )

    BatchAudioGenGenerator().generate(
        script_json=script_json,

        episode_seed="test-iter",
        default_duration=3.0,
    )

    saved_led = patched_audiogen_env["led_disk"]
    by_id = {ln["line_id"]: ln for ln in saved_led["lines"]}

    # SFX line b005 stamped with audiogen fields.
    assert by_id["b005"].get("sfx_engine") == "audiogen"
    assert by_id["b005"].get("sfx_wav_path", "").endswith(".wav")
    assert by_id["b005"].get("dur_s", 0) > 0
    assert by_id["b005"].get("generated_dur_s", 0) > 0

    # Non-sfx rows must NOT have any audiogen fields stamped.
    for non_sfx in ("b001", "b002", "b003", "b004", "b006"):
        row = by_id[non_sfx]
        assert "sfx_engine" not in row, (
            f"{non_sfx} should not have sfx_engine stamped"
        )
        assert "sfx_wav_path" not in row, (
            f"{non_sfx} should not have sfx_wav_path stamped"
        )

def test_audiogen_stamps_by_line_id_with_duplicate_text(patched_audiogen_env):
    """Two sfx lines with IDENTICAL cue text but distinct line_ids.
    Both stamped correctly via line_id. The legacy parallel-index walk
    on ledger.sfx[] would also handle this, but the test specifically
    asserts ledger.lines[] line_id stamping."""
    from nodes.batch_audiogen_generator import BatchAudioGenGenerator

    led_in = {
        "meta":  {"title": "DUP-SFX", "news_seed": {"headline": "x"}},
        "cast":  [],
        "lines": [
            {"line_id": "sfx_first", "char_id": "sfx",
             "speaker_role": "sfx", "text": "metal door slam",
             "traits": "neutral", "start_s": None, "dur_s": None,
             "boundary": None, "bark_wav_path": None},
            {"line_id": "sfx_second", "char_id": "sfx",
             "speaker_role": "sfx", "text": "metal door slam",
             "traits": "neutral", "start_s": None, "dur_s": None,
             "boundary": None, "bark_wav_path": None},
        ],
    }
    patched_audiogen_env["led_disk"] = json.loads(json.dumps(led_in))
    script_json = json.dumps(led_in)

    # Same cue text -> same cache prefix -> both lines hit the same
    # cache file. Pre-seed it once.
    _seed_cache_for_cue(
        patched_audiogen_env["cache_dir"],
        prompt="metal door slam",
        duration=3.0,
        episode_seed="dup-test",
    )

    BatchAudioGenGenerator().generate(
        script_json=script_json,

        episode_seed="dup-test",
        default_duration=3.0,
    )

    saved_led = patched_audiogen_env["led_disk"]
    by_id = {ln["line_id"]: ln for ln in saved_led["lines"]}

    # Both rows stamped, despite identical cue text.
    for lid in ("sfx_first", "sfx_second"):
        row = by_id[lid]
        assert row.get("sfx_engine") == "audiogen", (
            f"{lid} missing sfx_engine"
        )
        assert row.get("sfx_wav_path", "").endswith(".wav"), (
            f"{lid} missing sfx_wav_path"
        )
        assert row.get("dur_s", 0) > 0, f"{lid} missing dur_s"

def test_audiogen_cache_hit_path(patched_audiogen_env):
    """Cache-hit path (no real model load) still stamps line_id-keyed
    write-back. This was the BUG-LOCAL-095 surface -- legacy code
    sometimes failed to stamp on cache hits because the dur_s computation
    relied on a freshly-loaded WAV. Verify the new contract holds."""
    from nodes.batch_audiogen_generator import BatchAudioGenGenerator

    led_in = make_stub_ledger()  # contains 1 sfx line at b005
    patched_audiogen_env["led_disk"] = json.loads(json.dumps(led_in))
    script_json = json.dumps(led_in)

    # Pre-seed cache so the sfx cue resolves to a hit.
    cache_path = _seed_cache_for_cue(
        patched_audiogen_env["cache_dir"],
        prompt="metal door slam",
        duration=3.0,
        episode_seed="cache-hit-test",
    )

    _audio, batch_log = BatchAudioGenGenerator().generate(
        script_json=script_json,

        episode_seed="cache-hit-test",
        default_duration=3.0,
    )

    # Confirm cache-hit was the path taken (pre-seeded cache file
    # guarantees it; the assertion documents the test's intent).
    assert "CACHE HIT" in batch_log, (
        f"expected CACHE HIT in batch_log; got:\n{batch_log}"
    )
    # Cache miss path would log "MISS"; confirm we did NOT take it.
    assert "MISS" not in batch_log, (
        f"unexpected cache MISS on a pre-seeded cache run; got:\n{batch_log}"
    )

    # Confirm the line_id stamping fired despite cache-hit.
    saved_led = patched_audiogen_env["led_disk"]
    by_id = {ln["line_id"]: ln for ln in saved_led["lines"]}
    row = by_id["b005"]
    assert row.get("sfx_engine") == "audiogen"
    assert row.get("sfx_wav_path") == str(cache_path)
    assert row.get("dur_s", 0) > 0
    assert row.get("generated_dur_s", 0) > 0

def test_audiogen_legacy_list_raises():
    """Legacy parser-list shape -> load_ledger ValueError surfaces.
    AudioGen is in the loud-fail group."""
    from nodes.batch_audiogen_generator import BatchAudioGenGenerator

    legacy_json = json.dumps(make_legacy_list())

    with patch(
        "nodes.batch_audiogen_generator.force_vram_offload",
        side_effect=_no_op,
    ):
        with pytest.raises(ValueError) as excinfo:
            BatchAudioGenGenerator().generate(
                script_json=legacy_json,

            )

    msg = str(excinfo.value)
    assert "legacy parser-list" in msg or "OTR_LedgerScriptWriter" in msg, (
        f"ValueError message should name the legacy shape; got: {msg!r}"
    )
