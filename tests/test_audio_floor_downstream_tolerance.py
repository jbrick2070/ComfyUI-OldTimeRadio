"""A FLOORED LINE OR CUE MUST NOT BREAK ANYTHING THAT READS IT.

Operator, 2026-09-16: *"remember the downstream consumers -- when they see a
gap they can't fail either."*

That is the harder half of the audio floor and the reason this file exists
separately from the floor's own tests. Substituting silence keeps the run
alive at the point of failure; it is worth nothing if the sequencer, the
packer, the loudness pass or the ledger then falls over on the thing we
substituted. The floor would simply have moved the crash later and lost the
paid work anyway -- which is precisely what dropping the line would have done.

WHY SILENCE AND NOT A DROP. ``scene_sequencer._verify_bus_clip_counts`` raises
on any consumed/provided mismatch ("no silent tolerance"), so a dropped line
does not survive to the mux. A clip that HOLDS THE SLOT is the only shape that
keeps every positional contract intact.

These tests deliberately use the REAL downstream functions, not stubs. A stub
would prove the floor agrees with itself.
"""
from __future__ import annotations

import pytest

from nodes._otr_shared import cloud_media_backend as cmb
from nodes import _otr_voice_node_common as voice
from nodes._otr_audio_engines.base import (
    assert_audio_batch_contract, pack_audio_batch,
)

SR = 44100


# --------------------------------------------------------------------------- #
# what the floor produces
# --------------------------------------------------------------------------- #
def test_a_floored_line_is_a_real_clip_of_a_sane_length():
    quiet = voice.floored_line_silence("this line never came back", SR)
    assert quiet["sample_rate"] == SR
    assert quiet["waveform"].shape[0] == 1 and quiet["waveform"].shape[1] == 1
    seconds = quiet["waveform"].shape[-1] / float(SR)
    # 5 words at the documented rate, floored at the minimum.
    assert voice.FLOORED_LINE_MIN_S <= seconds < 10.0, seconds
    assert float(quiet["waveform"].abs().max()) == 0.0, "must be pure silence"


def test_even_an_empty_line_gets_air_rather_than_a_zero_length_clip():
    """A zero-length clip is the shape that divides by zero downstream."""
    for text in ("", None, "   "):
        quiet = voice.floored_line_silence(text, SR)
        assert quiet["waveform"].shape[-1] >= 1, text
        seconds = quiet["waveform"].shape[-1] / float(SR)
        assert seconds >= voice.FLOORED_LINE_MIN_S


def test_a_longer_line_gets_proportionally_more_air():
    short = voice.floored_line_silence("one two three", SR)
    long = voice.floored_line_silence(" ".join(["word"] * 120), SR)
    assert long["waveform"].shape[-1] > short["waveform"].shape[-1]


def test_floor_text_uses_spoken_words_not_the_engine_wrapper():
    """Google TTS prefixes `Say <style>: ` onto `prepared`. Counting that
    inflates the gap with words nobody would have spoken.

    THE LIVE SHAPE is a job with ``spoken_text`` plus a
    ``ResolvedVoiceRequest`` that has ``prepared_text``, not ``text``. A
    fake ``_Req.text`` would pass while production still counted the wrapper.
    """
    from nodes._otr_resolved_request import ResolvedVoiceRequest

    req = ResolvedVoiceRequest(prepared_text="Say warmly: hello there")
    spoken = voice._floor_text({
        "spoken_text": "hello there",
        "request": req,
        "prepared": "Say warmly: hello there",
    })
    assert spoken == "hello there"
    # Without spoken_text the live request cannot supply words -- do not
    # silently count the wrapper as speech.
    assert voice._floor_text({
        "request": req,
        "prepared": "Say warmly: hello there",
    }) == "Say warmly: hello there"
    # A non-string prepared (the defect Opus named) must not explode.
    assert voice._floor_text({"prepared": {"prompt": "nope"}}) == ""
    assert voice._floor_text({"prepared": None}) == ""


def test_voice_floor_is_a_recognized_optional_ledger_string():
    from nodes._otr_ledger_consumers import _OPTIONAL_STRING_FIELDS
    assert "voice_floor" in _OPTIONAL_STRING_FIELDS


# --------------------------------------------------------------------------- #
# the consumers
# --------------------------------------------------------------------------- #
def test_a_floored_line_packs_into_the_audio_batch_contract():
    """pack_audio_batch is the first thing every voice node hands downstream."""
    import torch
    real = {"waveform": torch.ones(1, 1, SR), "sample_rate": SR}
    quiet = voice.floored_line_silence("a floored line", SR)
    packed = pack_audio_batch([real, quiet, real], sample_rate=SR, mono=True)
    assert_audio_batch_contract(packed)
    # THE COUNT IS THE POINT: three lines in, three lines out. The sequencer
    # verifies consumed-vs-provided and raises on any mismatch.
    assert packed["waveform"].shape[0] == 3


def test_an_all_floored_role_still_packs():
    """The worst case must not be the one that crashes."""
    quiet = [voice.floored_line_silence("x %d" % i, SR) for i in range(4)]
    packed = pack_audio_batch(quiet, sample_rate=SR, mono=True)
    assert_audio_batch_contract(packed)
    assert packed["waveform"].shape[0] == 4


def test_the_loudness_pass_does_not_divide_by_silence():
    """RMS normalization on a pure-silence clip must return, not explode."""
    import numpy as np
    from nodes import scene_sequencer as seq
    silent = np.zeros((2, SR), dtype=np.float32)
    got = seq._loudness_normalize_clip(
        silent, -18.0, 12.0, -24.0, -60.0, 0.98)
    assert np.asarray(got).shape == silent.shape
    assert not np.any(np.isnan(np.asarray(got))), "silence became NaN"
    # It returns the clip UNCHANGED rather than applying a gain to nothing.
    assert float(np.abs(np.asarray(got)).max()) == 0.0
    # A near-silent-but-not-zero clip must also survive (the gate path).
    nearly = np.full((2, SR), 1e-9, dtype=np.float32)
    got2 = seq._loudness_normalize_clip(
        nearly, -18.0, 12.0, -24.0, -60.0, 0.98)
    assert not np.any(np.isnan(np.asarray(got2)))


def test_a_floored_cue_is_exactly_the_duration_it_was_asked_for():
    """Music placement offsets are computed from requested_duration_s.

    If the substitute were any other length every later cue, caption span and
    assembly offset would shift -- a floor that silently re-times the episode
    is not a floor.

    THIS TEST WAS A TAUTOLOGY ONCE. The first draft rebuilt the same
    `int(round(seconds * SR))` arithmetic inline and asserted it against
    itself, because the production helper was a closure nobody could import --
    it would have passed with the music floor deleted. The helper is module
    scope now and this calls it.
    """
    from nodes.stable_audio_theme import floored_cue_silence
    for seconds in (1.0, 7.5, 30.0):
        quiet = floored_cue_silence(seconds, SR)
        assert quiet["sample_rate"] == SR
        assert abs(quiet["waveform"].shape[-1] / float(SR) - seconds) < 1e-4
        assert float(quiet["waveform"].abs().max()) == 0.0
    # A zero/absurd duration must still yield a usable clip, never length 0.
    for bad in (0.0, -3.0):
        assert floored_cue_silence(bad, SR)["waveform"].shape[-1] >= 1


def test_only_a_stamped_job_scoped_verdict_floors_a_cue():
    from nodes.stable_audio_theme import cloud_cue_floor_reason
    for code in cmb.JOB_SCOPED_CODES:
        assert cloud_cue_floor_reason(
            cmb.CloudMediaError(code, "no bed")) == code.value
    for code in cmb.RUN_SCOPED_CODES | {cmb.CloudErrorCode.INTERRUPTED}:
        assert cloud_cue_floor_reason(cmb.CloudMediaError(code, "stop")) == ""
    assert cloud_cue_floor_reason(RuntimeError("adapter blew up")) == ""


# --------------------------------------------------------------------------- #
# the LEDGER, which a floored line must not damage
# --------------------------------------------------------------------------- #
def test_a_floored_line_stamp_lands_and_spares_its_healthy_neighbours(tmp_path):
    """THE DEFECT THIS FILE MISSED THE FIRST TIME.

    `stamp_per_line_audio_meta` is keyword-only with no **kwargs. An earlier
    draft stamped `voice_floor` anyway; the TypeError was swallowed by
    `_persist_ledger_stamps`' broad except, which then marked EVERY line in
    the role as failed -- so one floored line destroyed the render evidence of
    every healthy line beside it. The floor is supposed to cost one line.
    """
    import json
    import logging
    from nodes._otr_voice_node_common import _persist_ledger_stamps

    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps({"lines": [
        {"line_id": "L1", "text": "a real take"},
        {"line_id": "L2", "text": "the one that failed"},
    ]}), encoding="utf-8")

    failed = set()
    degraded = _persist_ledger_stamps(
        {"paths": {"ledger_path": str(ledger_path)}},
        [("L1", {"tts_engine": "elevenlabs", "audio_sample_hash": "abc123",
                 "generated_dur_s": 1.5}),
         ("L2", {"tts_engine": "elevenlabs", "render_ms": 0,
                 "generated_dur_s": 2.0, "sample_rate": SR,
                 "voice_floor": "timeout", "voice_floor_words": 4})],
        logging.getLogger("test"), failed_line_ids=failed)

    assert degraded == 0, "no stamp should have failed"
    assert failed == set(), failed
    rows = {r["line_id"]: r for r in
            json.loads(ledger_path.read_text(encoding="utf-8"))["lines"]}
    # The healthy line keeps its evidence.
    assert rows["L1"]["audio_sample_hash"] == "abc123"
    # The floored line is MARKED, and carries no hash -- nothing was produced.
    assert rows["L2"]["voice_floor"] == "timeout"
    assert rows["L2"]["voice_floor_words"] == 4
    assert not rows["L2"].get("audio_sample_hash")
    assert not rows["L2"].get("audio_sha256")


def test_a_healthy_re_render_cannot_blank_or_invent_a_voice_floor(tmp_path):
    """Skip-when-empty, like every neighbouring field."""
    import json
    from nodes._otr_ledger import stamp_per_line_audio_meta
    led = {"lines": [{"line_id": "L1", "text": "hi", "voice_floor": "timeout"}]}
    stamp_per_line_audio_meta(led, "L1", tts_engine="cloud_elevenlabs")
    assert led["lines"][0]["voice_floor"] == "timeout", "must not be blanked"
    led2 = {"lines": [{"line_id": "L1", "text": "hi"}]}
    stamp_per_line_audio_meta(led2, "L1", tts_engine="cloud_elevenlabs")
    assert "voice_floor" not in led2["lines"][0], "must not be invented"


# --------------------------------------------------------------------------- #
# the gate -- a floor is only for a stamped, job-scoped provider verdict
# --------------------------------------------------------------------------- #
def test_only_a_stamped_job_scoped_verdict_floors_a_line():
    for code in cmb.JOB_SCOPED_CODES:
        err = cmb.CloudMediaError(code, "no audio")
        assert voice.cloud_line_floor_reason(err) == code.value
    for code in cmb.RUN_SCOPED_CODES | {cmb.CloudErrorCode.INTERRUPTED}:
        assert voice.cloud_line_floor_reason(
            cmb.CloudMediaError(code, "stop")) == ""
    # An ordinary crash is NOT a provider verdict and must still fail loud.
    assert voice.cloud_line_floor_reason(RuntimeError("adapter blew up")) == ""
    assert voice.cloud_line_floor_reason(MemoryError("oom")) == ""
    # No prose fallback on this funnel either.
    assert voice.cloud_line_floor_reason(
        RuntimeError("Content filtered due to policy restrictions")) == ""


def test_the_floor_reason_survives_a_wrapped_exception():
    refused = cmb.CloudMediaError(
        cmb.CloudErrorCode.CONTENT_REFUSED, "Content filtered")
    wrap = RuntimeError("voice line failed")
    wrap.__cause__ = refused
    assert voice.cloud_line_floor_reason(wrap) == "content_refused"
