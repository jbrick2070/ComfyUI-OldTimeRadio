"""CastLock's announcer-role repair must survive the next durable save.

Voice review V3 (Muse Wonder Full, Codex source QA; reproduced 2026-10-14 by
a Sonnet 5.5 round trip on the real modules). CastLock re-stamps a
mis-stamped announcer line to ``speaker_role="announcer"`` on the WIRE only.
SceneSequencer then patched the role on disk, but ``speaker_role`` is
memory-owned in the singleton's save merge, so the next unrelated
``stamp_durable`` (image dispatch, render engines, ShotLock) re-saved the
stale "character" role -- and ``stamp_word_counts`` fed the credits' "Words"
row from it (char 13 / ann 0 instead of 6 / 7).

The fix is a narrow, validated per-line patch channel on ``stamp_durable``
that CastLock uses for exactly the lines it rerouted.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from nodes import production_ledger as pl


@pytest.fixture
def fresh_singleton(tmp_path):
    saved = pl._CURRENT
    try:
        yield pl.new_ledger("ep_role_repair", str(tmp_path))
    finally:
        with pl._LEDGER_LOCK:
            pl._CURRENT = saved


def _lines():
    return [
        {"line_id": "l001", "char_id": "c1", "speaker_role": "character",
         "text": "Hello there."},
        {"line_id": "l002", "char_id": "a1", "speaker_role": "character",
         "text": "Good evening."},
    ]


# --------------------------------------------------------------------------- #
# The patch channel's contract
# --------------------------------------------------------------------------- #
def test_a_line_patch_corrects_an_existing_row(fresh_singleton):
    fresh_singleton.data["lines"] = _lines()
    pl.stamp_durable(line_patches=[{"line_id": "l002", "char_id": "a1",
                                    "fields": {"speaker_role": "announcer"}}],
                     source="spec")
    roles = {r["line_id"]: r["speaker_role"] for r in pl.get_ledger().data["lines"]}
    assert roles == {"l001": "character", "l002": "announcer"}


def test_a_line_missing_from_a_populated_singleton_is_warned_not_failed(
        fresh_singleton, caplog):
    """The save keeps memory's rows when memory holds lines, so a line memory
    lacks cannot be persisted from here. That is a singleton/wire divergence
    (a cached writer on a re-queue, say), not something to fail a render on:
    it is warned, naming the line, and nothing changes."""
    import logging

    fresh_singleton.data["lines"] = _lines()
    with caplog.at_level(logging.WARNING, logger="OTR"):
        pl.stamp_durable(line_patches=[{"line_id": "l999", "char_id": "a1",
                                        "fields": {"speaker_role": "announcer"}}],
                         source="spec")
    assert [r["speaker_role"] for r in pl.get_ledger().data["lines"]] == [
        "character", "character"]
    assert any("l999" in r.getMessage() and "NOT persisted" in r.getMessage()
               for r in caplog.records), caplog.text


def test_a_singleton_with_no_lines_leaves_the_disk_lines_standing(tmp_path, monkeypatch):
    """When memory holds NO lines, a save keeps the disk's lines wholesale, so
    skipping the patch quietly is safe -- proven on a real save."""
    monkeypatch.setenv("OTR_TEST_MODE", "0")
    saved = pl._CURRENT
    try:
        pl._CURRENT = None
        led = pl.new_ledger("ep_no_mem_lines", out_dir=str(tmp_path))
        led.data["lines"] = [dict(r, speaker_role="announcer" if r["line_id"] == "l002"
                                  else r["speaker_role"]) for r in _lines()]
        assert led.save()
        led.data["lines"] = []
        pl.stamp_durable(line_patches=[{"line_id": "l002", "char_id": "a1",
                                        "fields": {"speaker_role": "announcer"}}],
                         meta_updates={"note": "x"}, source="spec")
        with open(pl.peek_ledger().path, encoding="utf-8") as fh:
            disk = json.load(fh)
        assert {r["line_id"]: r["speaker_role"] for r in disk["lines"]} == {
            "l001": "character", "l002": "announcer"}
    finally:
        with pl._LEDGER_LOCK:
            pl._CURRENT = saved


@pytest.mark.parametrize("patch,sections,why", [
    ({"line_id": "l002", "char_id": ""}, None, "needs both a line_id and a char_id"),
    ({"line_id": "", "char_id": "a1"}, None, "needs both a line_id and a char_id"),
    ({"line_id": "l002", "char_id": "a1"}, {"lines": []}, "cannot be combined"),
], ids=["no_char_id", "no_line_id", "with_lines_section"])
def test_a_malformed_patch_is_refused_before_any_change(fresh_singleton, patch, sections, why):
    fresh_singleton.data["lines"] = _lines()
    before = json.dumps(pl.get_ledger().data, sort_keys=True)
    with pytest.raises(ValueError, match=why):
        pl.stamp_durable(sections=sections,
                         line_patches=[{**patch, "fields": {"speaker_role": "announcer"}}],
                         source="spec")
    assert json.dumps(pl.get_ledger().data, sort_keys=True) == before


@pytest.mark.parametrize("lines,patch,why", [
    (_lines() + [{"line_id": "l002", "char_id": "a1",
                  "speaker_role": "character", "text": "dup"}],
     {"line_id": "l002", "char_id": "a1"}, "appears 2 times"),
    (_lines(), {"line_id": "l002", "char_id": "c9"}, "refusing to patch a different line"),
], ids=["duplicate_line_id", "char_id_mismatch"])
def test_an_ambiguous_patch_fails_loud_and_changes_nothing(fresh_singleton, lines, patch, why):
    fresh_singleton.data["lines"] = lines
    before = json.dumps(pl.get_ledger().data, sort_keys=True)
    with pytest.raises(pl.LedgerStampError, match=why):
        pl.stamp_durable(
            sections={"cast": [{"char_id": "x"}]},
            line_patches=[{**patch, "fields": {"speaker_role": "announcer"}}],
            source="spec")
    assert json.dumps(pl.get_ledger().data, sort_keys=True) == before, (
        "validation must run before ANY state changes, sections included")


def test_only_allowlisted_fields_can_be_patched(fresh_singleton):
    fresh_singleton.data["lines"] = _lines()
    with pytest.raises(ValueError, match="not patchable"):
        pl.stamp_durable(line_patches=[{"line_id": "l001", "char_id": "c1",
                                        "fields": {"text": "rewritten"}}],
                         source="spec")


# --------------------------------------------------------------------------- #
# The real round trip: CastLock -> SceneSequencer -> unrelated stamp -> disk
# --------------------------------------------------------------------------- #
_CAST = [
    {"char_id": "c1", "name": "MONTY", "gender": "male", "tts_model": "bark",
     "voice_preset": "v2/en_speaker_1", "character_description": "x" * 30},
    {"char_id": "c2", "name": "VERA", "gender": "female", "tts_model": "bark",
     "voice_preset": "v2/en_speaker_9", "character_description": "y" * 30},
    {"char_id": "a1", "name": "ANNOUNCER", "gender": "male", "tts_model": "bark",
     "voice_preset": "v2/en_speaker_6", "character_description": "z" * 30},
]
_ROUNDTRIP_LINES = [
    # The mis-stamped row the repair targets: the announcer marker, role "character".
    {"line_id": "l001", "char_id": "announcer", "speaker_role": "character",
     "text": "Good evening, and welcome to the program."},
    {"line_id": "l002", "char_id": "c1", "speaker_role": "character",
     "text": "Hello there, Vera."},
    {"line_id": "l003", "char_id": "c2", "speaker_role": "character",
     "text": "Monty. At last."},
]


def _tone(seconds, f, sr=24000):
    t = np.arange(int(seconds * sr)) / sr
    return torch.from_numpy((0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)).view(1, 1, -1)


@pytest.mark.parametrize("later", ["image_dispatcher", "render_engines", "shot_lock"])
def test_the_repaired_role_survives_a_later_unrelated_stamp(tmp_path, monkeypatch, later):
    from nodes.cast_lock import CastLock
    from nodes.scene_sequencer import SceneSequencer

    monkeypatch.setenv("OTR_TEST_MODE", "0")      # stamp_durable writes disk only then
    saved = pl._CURRENT
    try:
        pl._CURRENT = None
        led = pl.new_ledger("ep_v3_" + later, out_dir=str(tmp_path / "audio"))
        led.data["meta"] = {"episode_seed": 42}
        led.set_cast(_CAST)
        led.set_lines(_ROUNDTRIP_LINES)
        assert led.save()

        ledger_json, _rev, report, _done = CastLock().lock(
            script_json=json.dumps(led.data),
            cast_voice_policy="preserve_ledger", voice_device="cpu")
        assert "reroute" in report

        two_char = torch.cat([_tone(0.6, 330.0), _tone(0.6, 440.0)], dim=0)
        SceneSequencer().sequence(
            script_json=ledger_json,
            tts_audio_clips={"waveform": two_char, "sample_rate": 24000},
            announcer_audio_clips={"waveform": _tone(0.8, 180.0), "sample_rate": 24000})

        {   # the exact call shapes those stages make
            "image_dispatcher": lambda: pl.stamp_durable(
                sections={"images": {"rows": []}, "visual": {"style_id": "s"}},
                meta_updates={"image_engines": {"by_role": {}}},
                source="image_dispatcher"),
            "render_engines": lambda: pl.stamp_durable(
                meta_updates={"render_engines": {"x": 1}},
                source="video_render_batch"),
            "shot_lock": lambda: pl.stamp_durable(
                sections={"video": {"shots": []}}, source="shot_lock"),
        }[later]()

        led = pl.peek_ledger()
        with open(led.path, encoding="utf-8") as fh:
            disk = json.load(fh)
        roles = {r["line_id"]: r["speaker_role"] for r in disk["lines"]}
        assert roles["l001"] == "announcer", roles
        assert roles["l002"] == roles["l003"] == "character"
        assert (disk.get("meta") or {}).get("announcer_word_count", 0) > 0, (
            "the credits' word counts must follow the repaired role")
        assert (disk.get("meta") or {}).get("voice_device") == "cpu"
    finally:
        with pl._LEDGER_LOCK:
            pl._CURRENT = saved


def test_only_the_voiced_roles_can_be_written(fresh_singleton):
    """The channel exists to move a line between the two voiced roles; a
    patch cannot write any other role string (sonnet QA r2)."""
    fresh_singleton.data["lines"] = _lines()
    with pytest.raises(ValueError, match="not a patchable value"):
        pl.stamp_durable(line_patches=[{"line_id": "l002", "char_id": "a1",
                                        "fields": {"speaker_role": "music_open"}}],
                         source="spec")
    assert [r["speaker_role"] for r in pl.get_ledger().data["lines"]] == [
        "character", "character"]


def test_a_padded_line_id_in_the_singleton_still_matches(fresh_singleton):
    lines = _lines()
    lines[1]["line_id"] = " l002 "
    fresh_singleton.data["lines"] = lines
    pl.stamp_durable(line_patches=[{"line_id": "l002", "char_id": "a1",
                                    "fields": {"speaker_role": "announcer"}}],
                     source="spec")
    assert pl.get_ledger().data["lines"][1]["speaker_role"] == "announcer"


# --------------------------------------------------------------------------- #
# The treatment sidecar's word counts follow the repaired roles too
# --------------------------------------------------------------------------- #
def test_the_treatment_takes_its_word_counts_from_the_durable_ledger(fresh_singleton):
    """Node 12's `led` is FreezeCascade's script, upstream of CastLock, so its
    counts still read a rerouted announcer line as a character's (sonnet QA:
    "char 13 / announcer 0"). The singleton's recomputed counts win."""
    from nodes.video_engine import _treatment_ledger

    fresh_singleton.data["meta"] = {"total_word_count": 13,
                                    "character_word_count": 6,
                                    "announcer_word_count": 7}
    wire = {"meta": {"total_word_count": 13, "character_word_count": 13,
                     "announcer_word_count": 0, "episode_title": "T"},
            "lines": [], "cast": []}
    out = _treatment_ledger(wire)
    assert out["meta"]["character_word_count"] == 6
    assert out["meta"]["announcer_word_count"] == 7
    assert out["meta"]["episode_title"] == "T", "only the counts are overlaid"
    assert wire["meta"]["character_word_count"] == 13, "the wire ledger is not mutated"


def test_without_durable_counts_the_treatment_keeps_the_wire_ledger():
    from nodes.video_engine import _treatment_ledger

    saved = pl._CURRENT
    try:
        pl._CURRENT = None
        wire = {"meta": {"character_word_count": 13}, "lines": []}
        assert _treatment_ledger(wire) is wire
    finally:
        with pl._LEDGER_LOCK:
            pl._CURRENT = saved
