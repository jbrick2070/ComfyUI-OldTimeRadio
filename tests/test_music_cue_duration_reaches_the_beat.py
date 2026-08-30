"""Music rows must own exactly the timeline they actually occupy.

PBUG-20260829-16, and this file previously asserted the WRONG half of it.

The original reading was "a music cue's duration must reach the beat the cue
anchors to", and the fix resolved a duration through ``anchor_line_id``. The
4060 then measured the result on real 8 GB hardware and the master mux rejected
the episode for a silent video 18.93 s longer than its audio. The reason is that
the anchored row is not the row that renders:

* the **sentinel** (``shot_000_music``) is authored PRE-audio by the bank,
  carries the bank's own id, is untimed BY DESIGN, and is load-bearing -- it is
  what tells the assembler to mint the mirror and reserve that beat's still.
  It is also what the cue's ``anchor_line_id`` points at.
* the **mirror** (``music_opening_001``) is minted POST-audio under the
  assembler's deterministic id and carries the real ``start_s``/``dur_s``.

They are one beat in two lifecycle stages, not two copies. Only the mirror is a
timeline segment. Handing the sentinel the cue's duration emitted those seconds
a SECOND time on top of the mirror that already carried them.

The separate, real gap is ``music_inter``: commit 59286499 ("rip interstitial
audio insertion", 2026-07-22) removed the only code that stamped ``start_s`` and
``dur_s`` on act-break bridges, while the writer kept planning them. Those rows
have no mirror to defer to -- they are genuinely missing a duration. Measured
across every ledger on this box: 742 carry ``music_inter`` rows and none has
ever published.

Fixtures below are the shapes actually measured on the two machines, not
simplifications.
"""
from __future__ import annotations

from nodes.otr_shot_lock import (
    MUSIC_BRIDGE_FALLBACK_DUR_S,
    compute_clip_budget,
    extract_beats,
)


def _ledger_sentinel_and_mirror():
    """scifi_news_pro's real shape, measured on the 4060.

    Four music rows for two cues: the timed mirrors interleaved in position,
    the untimed sentinels appended after every real line, and the cues anchored
    to the sentinels.
    """
    return {
        "lines": [
            {"line_id": "music_opening_001", "speaker_role": "music_open",
             "start_s": 0.0, "dur_s": 10.0, "text": "", "char_id": ""},
            {"line_id": "shot_000_b1", "speaker_role": "announcer",
             "dur_s": 3.783, "text": "hello", "char_id": "narrator"},
            {"line_id": "music_closing_001", "speaker_role": "music_close",
             "start_s": 89.62775, "dur_s": 8.0, "text": "", "char_id": ""},
            # appended, untimed, and what anchor_line_id points at
            {"line_id": "shot_000_music", "speaker_role": "music_open",
             "dur_s": None, "text": "", "char_id": ""},
            {"line_id": "shot_002_music", "speaker_role": "music_close",
             "dur_s": None, "text": "", "char_id": ""},
        ],
        "music": [
            {"cue_id": "opening", "anchor_line_id": "shot_000_music",
             "start_s": 0.0, "dur_s": 10.0, "start_s_space": "master_mix"},
            {"cue_id": "closing", "anchor_line_id": "shot_002_music",
             "start_s": 89.62775, "dur_s": 8.0, "start_s_space": "master_mix"},
        ],
        "cast": [],
    }


def _ledger_with_act_break_bridges():
    """The 5080's real shape: timed bookends plus untimed act-break bridges."""
    return {
        "lines": [
            {"line_id": "music_opening_001", "speaker_role": "music_open",
             "start_s": 0.0, "dur_s": 10.0, "text": "", "char_id": ""},
            {"line_id": "shot_000_b1", "speaker_role": "announcer",
             "dur_s": 3.783, "text": "hello", "char_id": "narrator"},
            {"line_id": "music_closing_001", "speaker_role": "music_close",
             "start_s": 222.694, "dur_s": 8.0, "text": "", "char_id": ""},
            {"line_id": "b006", "speaker_role": "music_inter",
             "dur_s": None, "text": "", "char_id": ""},
            {"line_id": "b011", "speaker_role": "music_inter",
             "dur_s": None, "text": "", "char_id": ""},
        ],
        "music": [
            {"cue_id": "opening", "anchor_line_id": None, "start_s": 0.0,
             "dur_s": 10.0, "start_s_space": "master_mix"},
            {"cue_id": "closing", "anchor_line_id": None, "start_s": 222.694,
             "dur_s": 8.0, "start_s_space": "master_mix"},
        ],
        "cast": [],
    }


# --------------------------------------------------------------- sentinels

def test_the_sentinel_never_becomes_a_timeline_segment():
    """The 18.93 s overshoot, asserted at its source."""
    beats = extract_beats(_ledger_sentinel_and_mirror())
    ids = [b["beat_id"] for b in beats]
    for sentinel in ("shot_000_music", "shot_002_music"):
        assert sentinel not in ids, (
            "the pre-audio sentinel %r became a video beat. Its mirror already "
            "owns that timeline, so rendering it emits those seconds twice -- "
            "measured as an 18.93 s silent-video overshoot at the master mux."
            % sentinel)


def test_the_mirror_is_kept_and_keeps_its_own_timing():
    """Dropping the sentinel must not drop the row that actually renders."""
    beats = extract_beats(_ledger_sentinel_and_mirror())
    by_id = {b["beat_id"]: b for b in beats}
    assert "music_opening_001" in by_id, "the timed mirror was dropped"
    assert by_id["music_opening_001"]["dur_s"] == 10.0
    assert by_id["music_closing_001"]["dur_s"] == 8.0


def test_the_cue_duration_does_not_cross_the_anchor():
    """The specific mechanism of the earlier wrong fix, kept as a guard.

    A cue anchors to the SENTINEL, so reading ``anchor_line_id`` to supply a
    duration always selects the row that must not be timed.
    """
    beats = extract_beats(_ledger_sentinel_and_mirror())
    total = sum(b["dur_s"] or 0.0 for b in beats)
    assert total == 21.783, (
        "beats total %.3f s; the mirrors (10.0 + 8.0) plus the one spoken line "
        "(3.783) is 21.783. Anything near 39.783 means the sentinels were "
        "timed from their cues and the bookends are on the timeline twice."
        % total)


# ------------------------------------------------------------ act breaks

def test_an_act_break_bridge_gets_a_duration():
    """The 07-22 regression: 742 ledgers, none published."""
    beats = extract_beats(_ledger_with_act_break_bridges())
    bridges = [b for b in beats if b["beat_id"] in ("b006", "b011")]
    assert len(bridges) == 2, "an act-break bridge stopped producing a beat"
    for b in bridges:
        assert b["dur_s"] == MUSIC_BRIDGE_FALLBACK_DUR_S, (
            "bridge %s has dur_s=%r; with no mirror to defer to it must take "
            "the measured fallback" % (b["beat_id"], b["dur_s"]))


def test_the_bridge_no_longer_budgets_to_zero_frames():
    """The exact precondition of the GhostCadenceError that killed every leg."""
    beats = extract_beats(_ledger_with_act_break_bridges())
    budget = compute_clip_budget(beats, fps=25)
    for bid in ("b006", "b011"):
        frames = budget["per_beat"][bid]
        assert frames >= 1, (
            "bridge %s budgets to %d frames; ghost_unique_source_count raises "
            "below 1 and every delivered-target engine refuses it" % (bid, frames))
        assert frames == 100, (
            "%.1f s at 25 fps should be 100 frames, got %d"
            % (MUSIC_BRIDGE_FALLBACK_DUR_S, frames))


def test_a_real_duration_still_wins_over_the_fallback():
    """The fallback fills a GAP; it never overrides stamped timing."""
    led = _ledger_with_act_break_bridges()
    led["lines"][3]["dur_s"] = 6.5
    beats = extract_beats(led)
    b006 = [b for b in beats if b["beat_id"] == "b006"][0]
    assert b006["dur_s"] == 6.5, (
        "a bridge that WAS timed got overwritten with the fallback")


def test_the_fallback_is_scoped_to_act_breaks_only():
    """A missing duration anywhere else must stay loud, not get a number."""
    led = _ledger_with_act_break_bridges()
    led["lines"].append({"line_id": "shot_009_b2", "speaker_role": "character",
                         "dur_s": None, "text": "quiet", "char_id": "c01"})
    beats = extract_beats(led)
    spoken = [b for b in beats if b["beat_id"] == "shot_009_b2"][0]
    assert spoken["dur_s"] is None, (
        "an untimed CHARACTER line was handed the music-bridge fallback; the "
        "zero-frame warning is the correct outcome there")
