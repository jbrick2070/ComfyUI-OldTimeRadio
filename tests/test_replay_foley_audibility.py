"""The foley REPLAY harness's audibility verdict -- the number a human reads.

WHAT THIS FILE IS FOR. ``scripts/otr_replay_foley_mix.py`` became a proof tool
on 2026-08-27: PBUG-20260827-03 shipped a foley bed that was mixed, levelled
and green on every receipt while being 35-56 dB under the programme, so
``foley_bed=mixed`` was retired as evidence and the per-beat
``RMS(bed in window) - RMS(programme in window)`` delta replaced it.

THE POINT OF THESE TESTS IS THAT A DIAGNOSTIC MUST NOT LIE QUIETLY. The bug
being reported was invisible precisely because a green receipt answered a
question nobody had asked. A measurement tool that prints a wrong number in
place of a right one repeats that failure one level up, so the cases below are
mostly about the summary line disagreeing with the rows above it.

DIGITAL SILENCE IS THE WHOLE HAZARD. ``rms_db`` returns a true ``-inf`` for a
silent block, and the two ways that reaches the delta are different findings:
``-inf - -inf`` is NaN ("no evidence either way"), while a live bed over a
silent programme is ``+inf`` ("the bed is the only thing here"). NaN is the
dangerous one -- every comparison against it is False, so one NaN arriving
first makes ``min``/``max`` return NaN for a list of otherwise ordinary beats.

CPU-safe and pure: no renders, no CUDA, no model loads, no episode on disk
except the few bytes of JSON ``detect_fps`` is handed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
_SRC = _REPO / "scripts" / "otr_replay_foley_mix.py"


def _load():
    spec = importlib.util.spec_from_file_location("otr_replay_foley_mix", _SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPLAY = _load()

NAN = float("-inf") - float("-inf")
INF = float("inf")


def _row(beat_id, delta):
    """A placed row carrying ``delta``; the other columns are never judged."""
    return {"beat_id": beat_id, "lane": "ltx25_foley_plus", "state": "placed",
            "raw_stem_db": -40.0, "bed_db": -54.0, "programme_db": -14.0,
            "delta_db": delta}


def test_nan_first_does_not_poison_the_range(capsys):
    """The regression that QA caught before this tool was ever trusted.

    RUN THIS RED AGAINST A ``min(deltas)`` OVER THE RAW LIST and it prints
    ``nan`` for both ends of a range whose real beats are -40 and -35. That is
    a wrong summary sitting directly under a correct table, which is worse
    than a crash because it reads as an answer.
    """
    REPLAY.print_audibility([_row("b001_dead", NAN), _row("b002", -40.0),
                             _row("b003", -35.0)])
    out = capsys.readouterr().out
    assert "delta range    : -40.00 dB (quietest) .. -35.00 dB (loudest)" in out
    assert "nan" not in out.split("delta range")[1]
    assert "b001_dead" in out and "no evidence" in out
    assert "2 judgeable beat(s)" in out


def test_infinite_delta_is_held_out_and_named(capsys):
    """A live bed over a digitally silent programme is not "infinitely good".

    It is a degenerate window and it says nothing about whether the bed sits
    correctly under DIALOGUE, so it is named and excluded rather than allowed
    to become the "loudest" end of the range.
    """
    REPLAY.print_audibility([_row("b001", -40.0), _row("b002_quiet", INF),
                             _row("b003", -20.0)])
    out = capsys.readouterr().out
    assert "delta range    : -40.00 dB (quietest) .. -20.00 dB (loudest)" in out
    assert "b002_quiet" in out
    assert "programme digitally silent under a live bed" in out


def test_every_beat_non_finite_refuses_to_judge(capsys):
    """No judgeable beat is a LOUD outcome, never a silent pass."""
    REPLAY.print_audibility([_row("b001", NAN), _row("b002", NAN)])
    out = capsys.readouterr().out
    assert "NO JUDGEABLE BEAT" in out
    assert "delta range" not in out


def test_ordinary_episode_is_unchanged_and_says_nothing_about_holdouts(capsys):
    """The happy path must not grow a hold-out line it has no reason to print."""
    REPLAY.print_audibility([_row("b001", -40.0), _row("b002", -20.0)])
    out = capsys.readouterr().out
    assert "held out" not in out
    assert "silent window" not in out
    assert "audible band   : -25.0 .. -15.0 dB -- 1 of 2 judgeable beat(s)" in out


def test_a_bed_inside_the_band_reads_as_audible(capsys):
    """The verdict this whole exercise is trying to eventually produce."""
    REPLAY.print_audibility([_row("b001", -20.0), _row("b002", -18.0)])
    out = capsys.readouterr().out
    assert "VERDICT        : AUDIBLE on every judgeable beat" in out


def test_rms_db_of_silence_is_negative_infinity_not_a_crash():
    """Silence has no dB value; -inf is the honest answer and it must not raise."""
    import numpy as np

    assert REPLAY.rms_db(np.zeros((2, 128), dtype=np.float32)) == float("-inf")
    assert REPLAY.rms_db(np.zeros((0,), dtype=np.float32)) == float("-inf")
    assert REPLAY.rms_db(np.full((2, 8), 1.0, dtype=np.float32)) == pytest.approx(0.0)


def _write_episode(root, frames_and_durs, name="ep_20260827_000000"):
    """The two artifacts ``detect_fps`` reads, and nothing else."""
    episode = root / name
    (episode / "audio").mkdir(parents=True)
    beats = [{"beat_id": bid, "shot_id": bid, "frame_count": frames}
             for bid, frames, _dur in frames_and_durs]
    lines = [{"line_id": bid, "start_s": 0.0, "dur_s": dur}
             for bid, _frames, dur in frames_and_durs]
    (episode / (name + "_silent.mp4.qa.json")).write_text(
        json.dumps({"beats": beats}), encoding="utf-8")
    (episode / "audio" / (name + "_ledger.json")).write_text(
        json.dumps({"lines": lines}), encoding="utf-8")
    return episode


def test_detect_fps_derives_the_rate_rather_than_assuming_it(tmp_path):
    """fps is in NEITHER artifact, so it is recovered from frames over seconds.

    The r1 panel asked for "fps from the QA/ledger" and that turned out not to
    be satisfiable -- neither file records it. Real episodes carry per-beat
    counts that agree to within 0.05 fps, which is where it actually survives.
    """
    episode = _write_episode(tmp_path, [
        ("b001", 417, 16.668041666666667),
        ("b002", 220, 8.804958333333333),
        ("b003", 226, 9.036375),
    ])
    fps, note = REPLAY.detect_fps(str(episode))
    assert fps == 25
    assert note == ""


def test_detect_fps_says_so_when_the_beats_disagree(tmp_path):
    """A disagreement falls back to the default and NAMES the disagreement."""
    episode = _write_episode(tmp_path, [
        ("b001", 250, 10.0),   # 25 fps
        ("b002", 300, 10.0),   # 30 fps
    ])
    fps, note = REPLAY.detect_fps(str(episode))
    assert fps == REPLAY.DEFAULT_FPS
    assert "disagree" in note and "25" in note and "30" in note


def test_detect_fps_survives_missing_artifacts_with_a_note(tmp_path):
    """An unreadable episode degrades to a labelled default, never a crash."""
    fps, note = REPLAY.detect_fps(str(tmp_path / "does_not_exist"))
    assert fps == REPLAY.DEFAULT_FPS
    assert "assumed" in note


def test_the_audible_band_is_the_one_the_bug_recorded():
    """15-25 dB under the programme, from PBUG-20260827-03. A guard on drift."""
    assert REPLAY.AUDIBLE_BAND_DB == (-25.0, -15.0)
