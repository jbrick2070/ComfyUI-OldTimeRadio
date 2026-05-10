"""tests/test_procsfx_ledger.py -- self-test for the ledger-pure
batch_procedural_sfx rewrite (Option 2: write wavs + stamp ledger).

Mocks the on-disk ledger I/O via state-backed in_flight_ledger_path /
load_ledger_safe / save_ledger_safe. Procedural synthesis is CPU-only
math, no model mocks needed.

Covers four canonical Pattern 7 cases:
  - test_procsfx_iter_sfx_only -- character / announcer / music lines
    are NOT stamped; only sfx lines reach render + write-back
  - test_procsfx_stamps_by_line_id -- two sfx lines, both stamped with
    sfx_wav_path + sfx_engine + sfx_type + dur_s; wav files exist on disk
  - test_procsfx_disk_write_failure_graceful -- mock the wav writer to
    raise; pipeline doesn't crash, sfx_wav_path stamped as None,
    AUDIO batch still returns
  - test_procsfx_legacy_list_raises -- load_ledger ValueError surfaces
"""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from tests.fixtures.ledger_stub import make_legacy_list, make_stub_ledger


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def patched_procsfx_env(tmp_path):
    """Patch on-disk ledger I/O. The fake in_flight_ledger_path returns
    a path under tmp_path; <tmp_path>/sfx/ is where ProcSFX will write
    its wavs (the consumer's _resolve_proc_sfx_dir resolves
    ledger_path.parent / "sfx")."""
    state = {"led_disk": None, "ledger_path": tmp_path / "stub_ledger.json"}

    def _fake_in_flight_path():
        return state["ledger_path"]

    def _fake_load_ledger_safe(_path):
        return state["led_disk"]

    def _fake_save_ledger_safe(_path, led):
        state["led_disk"] = led
        return True

    with patch(
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

def test_procsfx_iter_sfx_only(patched_procsfx_env):
    """character / announcer / music lines must NOT be stamped. Only
    the sfx line gets the procedural write-back."""
    from nodes.batch_procedural_sfx import BatchProceduralSFX

    led_in = make_stub_ledger()  # 2 character + 1 announcer + 1 sfx + 2 music
    patched_procsfx_env["led_disk"] = json.loads(json.dumps(led_in))
    script_json = json.dumps(led_in)

    BatchProceduralSFX().generate(
        script_json=script_json,
        production_plan_json="{}",
        default_duration=2.0,
        volume_db=0.0,
    )

    saved_led = patched_procsfx_env["led_disk"]
    by_id = {ln["line_id"]: ln for ln in saved_led["lines"]}

    # SFX line b005 stamped: sfx_engine + sfx_type + dur_s + sfx_wav_path.
    # Stub cue is "metal door slam" -> "door" alias -> sfx_type "door_knock".
    assert by_id["b005"].get("sfx_engine") == "procedural"
    assert by_id["b005"].get("sfx_type") == "door_knock", (
        f"expected door_knock from 'metal door slam' fuzzy match; "
        f"got {by_id['b005'].get('sfx_type')!r}"
    )
    assert by_id["b005"].get("dur_s", 0) > 0
    # sfx_wav_path stamped (disk write succeeded under tmp_path).
    wav_path = by_id["b005"].get("sfx_wav_path")
    assert wav_path is not None and wav_path.endswith(".wav"), (
        f"expected wav path ending in .wav; got {wav_path!r}"
    )
    assert os.path.exists(wav_path), f"wav file not written: {wav_path}"

    # Non-sfx rows must NOT have any procedural fields stamped.
    for non_sfx in ("b001", "b002", "b003", "b004", "b006"):
        row = by_id[non_sfx]
        assert "sfx_engine" not in row, (
            f"{non_sfx} should not have sfx_engine stamped"
        )
        assert "sfx_type" not in row, (
            f"{non_sfx} should not have sfx_type stamped"
        )
        assert "sfx_wav_path" not in row, (
            f"{non_sfx} should not have sfx_wav_path stamped"
        )


def test_procsfx_stamps_by_line_id(patched_procsfx_env):
    """Two sfx lines (distinct line_ids, sample text variety). Both
    stamped via line_id, both wav files exist on disk."""
    from nodes.batch_procedural_sfx import BatchProceduralSFX

    led_in = {
        "meta":  {"title": "DUP-PROC", "news_seed": {"headline": "x"}},
        "cast":  [],
        "lines": [
            {"line_id": "ps_first", "char_id": "sfx",
             "speaker_role": "sfx", "text": "static crackle",
             "traits": "neutral", "start_s": None, "dur_s": None,
             "boundary": None, "bark_wav_path": None},
            {"line_id": "ps_second", "char_id": "sfx",
             "speaker_role": "sfx", "text": "loud knock",
             "traits": "neutral", "start_s": None, "dur_s": None,
             "boundary": None, "bark_wav_path": None},
        ],
    }
    patched_procsfx_env["led_disk"] = json.loads(json.dumps(led_in))
    script_json = json.dumps(led_in)

    BatchProceduralSFX().generate(
        script_json=script_json,
        production_plan_json="{}",
        default_duration=1.5,
        volume_db=0.0,
    )

    saved_led = patched_procsfx_env["led_disk"]
    by_id = {ln["line_id"]: ln for ln in saved_led["lines"]}

    # Both rows stamped with full field set.
    expected = {
        "ps_first":  ("white_noise", "static crackle"),
        "ps_second": ("door_knock",  "loud knock"),
    }
    for lid, (expected_type, _cue) in expected.items():
        row = by_id[lid]
        assert row.get("sfx_engine") == "procedural", f"{lid} sfx_engine"
        assert row.get("sfx_type") == expected_type, (
            f"{lid} expected sfx_type={expected_type!r}; "
            f"got {row.get('sfx_type')!r}"
        )
        assert row.get("dur_s", 0) > 0, f"{lid} dur_s"
        wav_path = row.get("sfx_wav_path")
        assert wav_path is not None, f"{lid} missing sfx_wav_path"
        assert wav_path.endswith(".wav"), f"{lid} wav_path should end .wav"
        assert os.path.exists(wav_path), (
            f"{lid} wav file not written: {wav_path}"
        )
        # Filename format: proc_<sfx_type>_<line_id>.wav (per spec).
        fname = os.path.basename(wav_path)
        assert fname.startswith(f"proc_{expected_type}_"), (
            f"{lid} filename {fname!r} should start with proc_{expected_type}_"
        )
        assert lid in fname, f"{lid} should appear in filename {fname!r}"


def test_procsfx_disk_write_failure_graceful(patched_procsfx_env):
    """Mock the wav writer to raise. Pipeline must NOT crash. The
    ledger row gets sfx_wav_path=None (best-effort fallback). The
    AUDIO batch return still ships."""
    from nodes.batch_procedural_sfx import BatchProceduralSFX

    led_in = make_stub_ledger()
    patched_procsfx_env["led_disk"] = json.loads(json.dumps(led_in))
    script_json = json.dumps(led_in)

    # Patch the wav-writer helper to return False (simulates any I/O
    # error: permission denied, disk full, soundfile import failure,
    # etc.). The consumer's best-effort try/except in _save_proc_sfx_wav
    # already catches exceptions and returns False; here we just
    # short-circuit to that exit.
    with patch(
        "nodes.batch_procedural_sfx._save_proc_sfx_wav",
        return_value=False,
    ):
        audio_out, batch_log = BatchProceduralSFX().generate(
            script_json=script_json,
            production_plan_json="{}",
            default_duration=2.0,
            volume_db=0.0,
        )

    # AUDIO batch still returns -- pipeline didn't crash.
    waveform = audio_out["waveform"]
    assert waveform.shape[0] == 1, (
        f"AUDIO batch must still return one clip; got {waveform.shape[0]}"
    )
    assert audio_out["sample_rate"] == 48000

    # Ledger stamped, but sfx_wav_path is None.
    saved_led = patched_procsfx_env["led_disk"]
    by_id = {ln["line_id"]: ln for ln in saved_led["lines"]}
    row = by_id["b005"]
    assert row.get("sfx_engine") == "procedural"
    assert row.get("sfx_type") == "door_knock"
    assert row.get("dur_s", 0) > 0
    assert row.get("sfx_wav_path") is None, (
        f"on disk-write failure sfx_wav_path must be None; "
        f"got {row.get('sfx_wav_path')!r}"
    )

    # Log line confirms the failure was surfaced (not silently swallowed).
    assert "disk write failed" in batch_log


def test_procsfx_legacy_list_raises():
    """Legacy parser-list shape -> load_ledger ValueError surfaces.
    ProcSFX is in the loud-fail group."""
    from nodes.batch_procedural_sfx import BatchProceduralSFX

    legacy_json = json.dumps(make_legacy_list())

    with pytest.raises(ValueError) as excinfo:
        BatchProceduralSFX().generate(
            script_json=legacy_json,
            production_plan_json="{}",
        )

    msg = str(excinfo.value)
    assert "legacy parser-list" in msg or "OTR_LedgerScriptWriter" in msg, (
        f"ValueError message should name the legacy shape; got: {msg!r}"
    )
