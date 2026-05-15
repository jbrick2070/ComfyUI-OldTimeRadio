"""tests/test_bark_ledger.py -- self-test for the ledger-pure
batch_bark_generator rewrite.

Mocks the heavy GPU paths (_load_bark, _generate_single_line,
force_vram_offload, _unload_llm) so the parsing + stamping logic is
exercised CPU-only in well under a second.

Covers four paths:
  - test_bark_iter_voiced_lines_only -- sfx/music/announcer skipped,
    only character lines reach dialogue_items
  - test_bark_stamps_by_line_id -- write-back patches by line_id even
    when two characters share identical text (the failure mode of the
    old text-match block)
  - test_bark_voice_preset_fallback -- character with cast.voice_preset
    missing falls back to the deterministic char-based default
  - test_bark_legacy_list_input_raises -- load_ledger ValueError
    propagates (Bark is allowed to fail loud on bad wiring)
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from tests.fixtures.ledger_stub import make_legacy_list, make_stub_ledger

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

def _fake_load_bark(_model_id):
    """Return placeholder model + processor; never touched because we also
    mock _generate_single_line."""
    return MagicMock(name="model"), MagicMock(name="processor")

def _fake_gen_single_line(text, _voice_preset, _model, _processor,
                          _temperature=0.7, is_first_line=False):
    """Return a deterministic 1.0s mono clip @ 24 kHz so the assembly
    block doesn't divide by zero or crash on shape mismatches.

    Encodes the call's text length into the audio length so each line
    has a slightly different duration (~1.0 - 1.2s); lets the test
    assert distinct dur_s values per line_id.
    """
    samples = 24000 + min(len(text) * 10, 5000)
    return np.zeros(samples, dtype=np.float32), 24000

@pytest.fixture
def patched_bark_env(tmp_path):
    """Patch the GPU + ledger I/O surfaces so generate_batch is hermetic.

    in_flight_ledger_path returns a temp file (so the write-back block
    actually runs and we can inspect the merged ledger). load_ledger_safe
    returns whatever was last written. save_ledger_safe writes through.
    """
    state = {"led_disk": None, "saved_path": None}

    def _fake_in_flight_path():
        # A fake path object whose .name attribute reads cleanly and
        # whose presence is truthy. The actual file isn't read; we
        # short-circuit load_ledger_safe and save_ledger_safe via the
        # state dict.
        return tmp_path / "stub_ledger.json"

    def _fake_load_ledger_safe(_path):
        return state["led_disk"]

    def _fake_save_ledger_safe(path, led):
        state["saved_path"] = path
        state["led_disk"] = led
        return True

    def _no_op(*_args, **_kwargs):
        return None

    # Voice-path-cleanbreak 2026-05-12 (P3): nodes.bark_tts was kept
    # as a library-only module (the OTR_BarkTTS node class was deleted
    # but _load_bark survives because batch_bark_generator imports it).
    # _load_bark stays patched so the test never tries to download
    # the actual Bark model.
    #
    # Sprint 7.2 (2026-05-12): module renamed nodes._bark_lib ->
    # nodes._otr_bark_lib per docs/conventions.md. Patch path updated
    # below in lockstep.
    with patch(
        "nodes.batch_bark_generator.force_vram_offload", side_effect=_no_op,
    ), patch(
        "nodes.batch_bark_generator._runtime_log", side_effect=_no_op,
    ), patch(
        "nodes.batch_bark_generator._generate_single_line",
        side_effect=_fake_gen_single_line,
    ), patch(
        "nodes._otr_bark_lib._load_bark", side_effect=_fake_load_bark,
    ), patch(
        # S31.5 B1: post-S31 B4 `story_orchestrator._unload_llm` was
        # DELETED. The canonical teardown surface is now
        # `_otr_model_loader.unload_llm`. Mock the new location.
        # Bucket B in the BUG-LOCAL-227 triage classification.
        "nodes._otr_model_loader.unload_llm", side_effect=_no_op,
    ), patch(
        "nodes._otr_ledger.in_flight_ledger_path",
        side_effect=_fake_in_flight_path,
    ), patch(
        "nodes._otr_ledger.load_ledger_safe", side_effect=_fake_load_ledger_safe,
    ), patch(
        "nodes._otr_ledger.save_ledger_safe", side_effect=_fake_save_ledger_safe,
    ):
        yield state

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_bark_iter_voiced_lines_only(patched_bark_env):
    """sfx, music, and announcer lines must NOT reach dialogue_items.
    Only character lines pass through. announcer is tallied separately
    for the diagnostic log line."""
    from nodes.batch_bark_generator import BatchBarkGenerator

    led_in = make_stub_ledger()  # 2 character + 1 announcer + 1 sfx + 2 music
    patched_bark_env["led_disk"] = json.loads(json.dumps(led_in))  # deep copy for save_ledger_safe
    script_json = json.dumps(led_in)

    audio_out, _batch_log = BatchBarkGenerator().generate_batch(
        script_json=script_json,

        temperature=0.7,
    )

    # Assembly tensor reflects exactly 2 character clips -- announcer,
    # sfx, and music_open / music_close lines were filtered out by the
    # iter_lines roles={"character"} call.
    waveform = audio_out["waveform"]
    assert waveform.shape[0] == 2, (
        f"expected 2 dialogue clips (character only), got {waveform.shape[0]}"
    )
    assert audio_out["sample_rate"] == 24000

    # The patched led_disk should now have updates on the two character
    # line_ids only (b003 LEMMY, b004 ASTRA per the stub fixture).
    saved_led = patched_bark_env["led_disk"]
    by_id = {ln["line_id"]: ln for ln in saved_led["lines"]}
    assert "tts_engine" in by_id["b003"], "character line b003 should be stamped"
    assert "tts_engine" in by_id["b004"], "character line b004 should be stamped"
    assert by_id["b003"]["tts_engine"] == "bark"
    # Non-character lines must NOT have a tts_engine field stamped by Bark
    # (announcer goes to Kokoro on a separate bus; sfx/music aren't TTS).
    assert "tts_engine" not in by_id["b002"], "announcer line not stamped by Bark"
    assert "tts_engine" not in by_id["b005"], "sfx line not stamped by Bark"
    assert "tts_engine" not in by_id["b001"], "music_open line not stamped by Bark"
    assert "tts_engine" not in by_id["b006"], "music_close line not stamped by Bark"

def test_bark_stamps_by_line_id_with_duplicate_text(patched_bark_env):
    """Two character lines with IDENTICAL text -- the old text-match
    block would have collided. line_id-keyed patching must stamp both
    correctly without ambiguity."""
    from nodes.batch_bark_generator import BatchBarkGenerator

    # Hand-craft a ledger with two character lines saying exactly "Okay."
    led_in = {
        "meta":  {"title": "DUP", "news_seed": {"headline": "x"}},
        "cast": [
            {"char_id": "c01", "name": "ALPHA",
             "voice_preset": "v2/en_speaker_3", "line_count": 1, "word_count": 1},
            {"char_id": "c02", "name": "BETA",
             "voice_preset": "v2/en_speaker_9", "line_count": 1, "word_count": 1},
        ],
        "lines": [
            {"line_id": "L_alpha", "char_id": "c01", "speaker_role": "character",
             "text": "Okay.", "traits": "neutral", "start_s": None, "dur_s": None,
             "boundary": None, "bark_wav_path": None},
            {"line_id": "L_beta", "char_id": "c02", "speaker_role": "character",
             "text": "Okay.", "traits": "neutral", "start_s": None, "dur_s": None,
             "boundary": None, "bark_wav_path": None},
        ],
    }
    patched_bark_env["led_disk"] = json.loads(json.dumps(led_in))
    script_json = json.dumps(led_in)

    BatchBarkGenerator().generate_batch(
        script_json=script_json,

        temperature=0.7,
    )

    saved_led = patched_bark_env["led_disk"]
    by_id = {ln["line_id"]: ln for ln in saved_led["lines"]}

    # Both rows stamped with bark + per-line forensic fields.
    for lid in ("L_alpha", "L_beta"):
        row = by_id[lid]
        assert row["tts_engine"] == "bark", f"{lid} missing tts_engine"
        assert row.get("dur_s", 0) > 0, f"{lid} missing dur_s"
        assert row.get("voice_preset", "").startswith("v2/"), (
            f"{lid} voice_preset not v2/* form"
        )

    # start_s must be monotonic -- L_alpha at 0.0, L_beta at L_alpha's dur.
    assert by_id["L_alpha"]["start_s"] == 0.0
    assert by_id["L_beta"]["start_s"] == pytest.approx(
        by_id["L_alpha"]["dur_s"], rel=1e-6,
    )

def test_bark_voice_preset_fallback_is_gone(patched_bark_env):
    """Voice-path-cleanbreak (Gate 3): Bark no longer has a fallback for
    cast.voice_preset=None. The legacy gender-aware grab-bag was deleted.
    Empty / non-v2 cast.voice_preset hard-raises ValueError. The new
    contract is fully covered by tests/test_bark_cast_contract.py; this
    test pins the regression: a cast row with voice_preset=None must
    raise, not silently grab a default.
    """
    from nodes.batch_bark_generator import BatchBarkGenerator

    led_in = {
        "meta":  {"title": "FB", "news_seed": {"headline": "x"}},
        "cast": [
            {"char_id": "c01", "name": "GHOST",
             "voice_preset": None, "line_count": 1, "word_count": 4,
             "gender": "male"},
        ],
        "lines": [
            {"line_id": "g001", "char_id": "c01", "speaker_role": "character",
             "text": "Static on the line.", "traits": "male, gravel",
             "start_s": None, "dur_s": None, "boundary": None,
             "bark_wav_path": None},
        ],
    }
    patched_bark_env["led_disk"] = json.loads(json.dumps(led_in))
    script_json = json.dumps(led_in)

    with pytest.raises(ValueError, match=r"voice_preset|v2/"):
        BatchBarkGenerator().generate_batch(
            script_json=script_json,
            temperature=0.7,
        )

def test_bark_legacy_list_input_raises():
    """Legacy parser-list shape on the wire must raise ValueError --
    Bark fails loud at the consumer boundary."""
    from nodes.batch_bark_generator import BatchBarkGenerator

    legacy_json = json.dumps(make_legacy_list())

    # Patch the heavy entrypoints so we know any failure has to come
    # from load_ledger, not from the GPU path.
    def _no_op(*_args, **_kwargs):
        return None

    with patch(
        "nodes.batch_bark_generator.force_vram_offload", side_effect=_no_op,
    ), patch(
        "nodes.batch_bark_generator._runtime_log", side_effect=_no_op,
    ):
        with pytest.raises(ValueError) as excinfo:
            BatchBarkGenerator().generate_batch(
                script_json=legacy_json,

                temperature=0.7,
            )

    msg = str(excinfo.value)
    assert "legacy parser-list" in msg or "OTR_LedgerScriptWriter" in msg, (
        f"ValueError message should name the legacy shape; got: {msg!r}"
    )
