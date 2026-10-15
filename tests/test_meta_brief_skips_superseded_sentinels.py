"""MetaBrief plans exactly ShotLock's beats -- no still for a superseded sentinel.

Plan item 10, 2026-10-14. ShotLock drops a pre-audio opening/closing music
SENTINEL once a timed mirror of the same role exists: the mirror owns the
timeline, and the sentinel must never render. MetaBrief's beat iterator kept
the sentinel, so the image side minted a scene still (paid per image on the
cloud graphs) for a beat no shot ever showed. The iterator now applies
ShotLock's own rule inside the same enumeration.
"""
from __future__ import annotations

from nodes.otr_meta_brief_image_prompt import (
    _iter_beat_lines, derive_scene_still_targets,
)
from nodes.otr_shot_lock import extract_beats


def _row(line_id, role, *, timed=True, start=None, dur=3.0, char="", **extra):
    row = {"line_id": line_id, "speaker_role": role, "text": "words",
           "char_id": char, "dur_s": dur if timed else None}
    if start is not None:
        row["start_s"] = start
    row.update(extra)
    return row


def _sentinel_and_mirror():
    """scifi_news_pro's measured shape: timed mirrors in place, untimed
    sentinels appended after every real line."""
    return [
        _row("music_opening_001", "music_open", start=0.0, dur=10.0),
        _row("shot_000_b1", "announcer", dur=3.783, char="narrator"),
        _row("music_closing_001", "music_close", start=89.6, dur=8.0),
        _row("shot_000_music", "music_open", timed=False),
        _row("shot_002_music", "music_close", timed=False),
    ]


def _agree(lines):
    mine = [bid for bid, _ln in _iter_beat_lines(lines)]
    shotlock = [b["beat_id"] for b in extract_beats({"lines": lines, "cast": []})]
    assert mine == shotlock, (mine, shotlock)
    return mine


def test_a_superseded_sentinel_gets_no_beat_and_no_still():
    lines = _sentinel_and_mirror()
    assert _agree(lines) == ["music_opening_001", "shot_000_b1", "music_closing_001"]
    targets, _warn = derive_scene_still_targets(lines)
    ids = {t["beat_id"] for t in targets}
    assert not ids & {"shot_000_music", "shot_002_music"}
    assert {"music_opening_001", "shot_000_b1", "music_closing_001"} <= ids


def test_an_untimed_sentinel_with_no_mirror_stays_a_beat():
    """Pre-audio, before any mirror exists, the sentinel is the only music row
    and still reserves its still."""
    lines = [_row("shot_000_music", "music_open", timed=False),
             _row("shot_000_b1", "announcer", char="narrator"),
             _row("shot_002_music", "music_close", timed=False)]
    assert _agree(lines) == ["shot_000_music", "shot_000_b1", "shot_002_music"]
    ids = {t["beat_id"] for t in derive_scene_still_targets(lines)[0]}
    assert {"shot_000_music", "shot_002_music"} <= ids


def test_interstitial_music_is_never_suppressed():
    lines = _sentinel_and_mirror() + [_row("b006", "music_inter", timed=False),
                                      _row("b011", "music_inter", timed=False)]
    assert _agree(lines)[-2:] == ["b006", "b011"]


def test_fallback_ids_stay_stable_across_the_suppression():
    """beat_NNNN counts every non-skipped row, sentinels included, exactly as
    ShotLock counts -- suppression happens inside the enumeration."""
    lines = [
        _row("music_opening_001", "music_open", start=0.0, dur=10.0),
        _row("", "announcer", char="narrator"),
        _row("shot_000_music", "music_open", timed=False),
        _row("", "character", char="c01"),
        _row("x", "character", char="c01", skip=True),
    ]
    assert _agree(lines) == ["music_opening_001", "beat_0001", "beat_0003"]


def test_a_skipped_mirror_does_not_supersede_its_sentinel():
    lines = [_row("music_opening_001", "music_open", start=0.0, dur=10.0, skip=True),
             _row("shot_000_b1", "announcer", char="narrator"),
             _row("shot_000_music", "music_open", timed=False)]
    assert "shot_000_music" in _agree(lines)


def test_the_ledger_rows_themselves_are_untouched():
    lines = _sentinel_and_mirror()
    before = [dict(r) for r in lines]
    list(_iter_beat_lines(lines))
    derive_scene_still_targets(lines)
    assert lines == before
