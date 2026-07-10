"""Ownership-aware disk merge + reviewer skip contract + lane capability
gate (external QA root-cause fold, 2026-07-10, fable2 S1b).

The corrupt-row chain the QA analysis proved: Script Doctor 'skip'
intentionally clears a row's text -> Ledger.save()'s stale-disk merge
treated the empty text as ABSENT and resurrected the OLD text into the
cleared row -> skip=True + non-empty text (+ no tts_skip_reason) ->
Phase 10 critical gap -> needs_full_rerun -> dead run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import production_ledger as _PL  # noqa: E402


@pytest.fixture(autouse=True)
def _preserve_current_ledger():
    saved = _PL._CURRENT
    yield
    _PL._CURRENT = saved


def _row(**over):
    r = {
        "line_id": "b001", "beat_id": "b001", "shot_id": None,
        "char_id": "c02", "speaker_role": "character",
        "boundary": None, "text": "The tape is the tape.",
    }
    r.update(over)
    return r


def test_intentionally_cleared_text_survives_save_roundtrip(tmp_path):
    # QA regression 1: save a non-empty row, clear it with a valid skip
    # state, save again -- the cleared text/counts survive while an
    # unrelated disk-only durable field ALSO survives (BUG-LOCAL-108).
    led = _PL.new_ledger(episode_id="merge_own",
                         out_dir=str(tmp_path / "ep"))
    led.set_cast([{"char_id": "c02", "name": "SELA", "tts_model": "bark",
                   "voice_preset": "v2/en_speaker_4",
                   "character_description": "x" * 40, "gender": "female",
                   "voice_params": None}])
    led.set_lines([_row()])
    led.save()
    # out-of-band durable stamp lands on DISK only (the BUG-108 shape)
    import json
    p = Path(led.path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["lines"][0]["bark_wav_path"] = "durable.wav"
    p.write_text(json.dumps(data), encoding="utf-8")
    # the doctor-skip mutation: clear text, stamp the full skip contract
    row = led.data["lines"][0]
    row["skip"] = True
    row["reviewer_skip_reason"] = "flat"
    row["tts_skip_reason"] = "reviewer_skip: flat"
    row["text"] = ""
    row["char_count"] = 0
    row["word_count"] = 0
    led.save()
    merged = led.data["lines"][0]
    assert merged["text"] == ""            # NOT resurrected from disk
    assert merged["skip"] is True
    assert merged["tts_skip_reason"] == "reviewer_skip: flat"
    assert merged["bark_wav_path"] == "durable.wav"  # durable field kept


def test_doctor_skip_stamps_the_phase10_contract():
    # QA regression 2: the skip edit must stamp tts_skip_reason (Phase
    # 10 requires it on every skip row).
    from nodes._otr_ledger_reviewer import (
        ReviewerEdit, ScriptDoctorReport, apply_doctor_edits)
    lines = [_row()]
    report = ScriptDoctorReport(edits=[
        ReviewerEdit(line_id="b001", action="skip", payload="flat")])
    n = apply_doctor_edits({"lines": lines}, report, edit_cap=5)
    assert n == 1
    assert lines[0]["skip"] is True
    assert lines[0]["text"] == ""
    assert lines[0]["tts_skip_reason"] == "reviewer_skip: flat"


def test_freeze_policy_replaces_capability_gate():
    # QA r2 P0.1/P1.2 supersedes the round-1 fail-open boolean: the
    # policy is typed, resolved once, and a TAGGED bank that fails to
    # resolve is TERMINAL -- never fail-open into legacy mutation.
    # (Deeper cascade-level coverage: tests/test_freeze_policy_readonly.py)
    from nodes._otr_freeze_cascade import resolve_freeze_policy
    from nodes import _otr_story_routing as ROUTING
    ROUTING._REGISTRY = None
    try:
        assert resolve_freeze_policy(
            {"source_bank": "scifi_fable2"}
        ).run_legacy_content_passes is False
        assert resolve_freeze_policy(
            {"source_bank": "science_news"}
        ).run_legacy_content_passes is True
        assert resolve_freeze_policy({}).run_legacy_content_passes is True
        bad = resolve_freeze_policy({"source_bank": "no_such_bank"})
        assert bad.run_legacy_content_passes is False
        assert bad.terminal_error  # terminal, NOT fail-open
    finally:
        ROUTING._REGISTRY = None


def test_set_lines_rebuild_does_not_resurrect_stale_skip(tmp_path):
    # QA r2 P0.2: persist a SKIPPED b001, rebuild it through the public
    # set_lines() API with non-empty text (set_lines omits the skip
    # family by schema), save, and assert the stale disk skip flag +
    # reasons do NOT come back. Pre-fix the key-presence test read the
    # omission as "restore old disk state" -> skip=True on a non-empty
    # line -> Phase 10 freeze error.
    led = _PL.new_ledger(episode_id="rebuild_own",
                         out_dir=str(tmp_path / "ep"))
    led.set_lines([_row()])
    row = led.data["lines"][0]
    row["skip"] = True
    row["reviewer_skip_reason"] = "old"
    row["tts_skip_reason"] = "reviewer_skip: old"
    row["text"] = ""
    row["char_count"] = 0
    row["word_count"] = 0
    led.save()  # skipped row now on disk
    led.set_lines([_row(text="Fresh authoritative rebuild.")])
    led.save()
    merged = led.data["lines"][0]
    assert merged["text"] == "Fresh authoritative rebuild."
    assert "skip" not in merged
    assert "tts_skip_reason" not in merged
    assert "reviewer_skip_reason" not in merged
    # reopen from disk: the persisted row is equally clean
    import json
    on_disk = json.loads(
        Path(led.path).read_text(encoding="utf-8"))["lines"][0]
    assert on_disk["text"] == "Fresh authoritative rebuild."
    assert "skip" not in on_disk
    assert "tts_skip_reason" not in on_disk
    assert "reviewer_skip_reason" not in on_disk


def test_set_lines_rebuild_with_empty_text_reflects_new_row(tmp_path):
    # QA r2 P0.2 (variant): an explicitly EMPTY replacement text is the
    # new authoritative state -- not an invitation to restore the old
    # disk text (round-1 regression held at the rebuild boundary too).
    led = _PL.new_ledger(episode_id="rebuild_empty",
                         out_dir=str(tmp_path / "ep"))
    led.set_lines([_row()])
    led.save()
    led.set_lines([_row(text="")])
    led.save()
    merged = led.data["lines"][0]
    assert merged["text"] == ""
    assert merged["word_count"] == 0
    # durable out-of-band fields still survive a rebuild (BUG-108)
    import json
    p = Path(led.path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data["lines"][0]["bark_wav_path"] = "durable.wav"
    p.write_text(json.dumps(data), encoding="utf-8")
    led.set_lines([_row(text="Third rebuild.")])
    led.save()
    assert led.data["lines"][0]["bark_wav_path"] == "durable.wav"
