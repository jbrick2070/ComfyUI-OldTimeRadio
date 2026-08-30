"""A music cue's duration must reach the beat that the cue anchors to.

PBUG-20260829-16. A bank naming its music anchors `<shot>_music` produced
ledger LINE rows with dur_s=None, while the music CUE anchored to them carried
the real duration and a rendered wav. Nothing crossed `anchor_line_id`, so the
beat budgeted to ZERO frames and the failure surfaced 72 minutes later at the
video stage as "Ghost Signal cadence needs a delivered target of at least 1" --
naming neither music nor duration.

Measured on ledger signal_lost_the_millisecond_map_20260829_163343: of 13 line
rows, the ONLY two with dur_s=None were the two music anchors.
"""
from __future__ import annotations

from nodes.otr_shot_lock import compute_clip_budget, extract_beats


def _ledger_with_anchored_music():
    return {
        "lines": [
            {"line_id": "shot_000_b1", "speaker_role": "announcer",
             "dur_s": 3.783, "text": "hello", "char_id": "narrator"},
            # the shape that crashed: real cue, line carries nothing
            {"line_id": "shot_000_music", "speaker_role": "music",
             "dur_s": None, "text": "", "char_id": ""},
        ],
        "music": [
            {"cue_id": "opening", "anchor_line_id": "shot_000_music",
             "start_s": 0.0, "dur_s": 10.0, "start_s_space": "master_mix",
             "wav_path": "music_cue_opening.wav"},
        ],
        "cast": [],
    }


def test_the_cue_duration_reaches_the_anchored_beat():
    beats = extract_beats(_ledger_with_anchored_music())
    music = [b for b in beats if b["beat_id"] == "shot_000_music"]
    assert music, "the music line did not produce a beat at all"
    assert music[0]["dur_s"] == 10.0, (
        "the anchored beat still has dur_s=%r -- the cue's duration did not "
        "cross anchor_line_id" % music[0]["dur_s"])


def test_the_beat_no_longer_budgets_to_zero_frames():
    """The exact precondition of the GhostCadenceError."""
    beats = extract_beats(_ledger_with_anchored_music())
    budget = compute_clip_budget(beats, fps=25)
    frames = budget["per_beat"]["shot_000_music"]
    assert frames >= 1, (
        "the music beat budgets to %d frames; ghost_unique_source_count "
        "raises below 1 and every delivered-target engine will refuse it"
        % frames)
    assert frames == 250, "10.0 s at 25 fps should be 250 frames, got %d" % frames


def test_a_real_line_duration_still_wins_over_a_cue():
    """Propagation must fill a GAP, never override stamped timing."""
    led = _ledger_with_anchored_music()
    led["lines"][1]["dur_s"] = 4.0
    beats = extract_beats(led)
    music = [b for b in beats if b["beat_id"] == "shot_000_music"][0]
    assert music["dur_s"] == 4.0, (
        "the cue overrode a line that already had its own duration")


def test_start_s_is_NOT_copied_from_the_cue():
    """The cue's start_s is in `master_mix` space, not the line's.

    Copying it would trade a loud crash for silently wrong placement, which is
    the worse failure. Only the space-independent duration may cross.
    """
    beats = extract_beats(_ledger_with_anchored_music())
    music = [b for b in beats if b["beat_id"] == "shot_000_music"][0]
    assert "start_s" not in music or music.get("start_s") in (None, 0.0), (
        "start_s was taken from a cue stamped start_s_space=master_mix")


def test_an_unanchored_music_cue_changes_nothing():
    led = _ledger_with_anchored_music()
    led["music"][0]["anchor_line_id"] = "some_other_line"
    beats = extract_beats(led)
    music = [b for b in beats if b["beat_id"] == "shot_000_music"][0]
    assert music["dur_s"] is None, "a cue anchored elsewhere leaked its duration"
