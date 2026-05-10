"""tests/fixtures/ledger_stub.py -- shared stub ledger factory for v2 consumer self-tests.

Mirrors the on-wire shape OTR_LedgerScriptWriter emits: a dict with
``meta``, ``cast`` (LIST of dicts, each carrying its own ``char_id``),
and ``lines`` (LIST of dicts with ``char_id``, ``speaker_role``, ``text``,
``traits``, ``start_s``, ``dur_s``, ``boundary``, ``line_id``).

Used by the seven consumer self-tests built during the ledger-consumer
rewrite sprint. Kept tiny (two characters, six lines spanning every
``speaker_role``) so each test reads at a glance.
"""
from __future__ import annotations

from typing import Optional


def make_stub_ledger(
    *,
    with_sfx: bool = True,
    with_music: bool = True,
    title: Optional[str] = None,
    headline: Optional[str] = None,
) -> dict:
    """Build a stub ledger dict covering character / announcer / sfx / music
    speaker_roles.

    Default shape:

      cast  -> 2 entries (LEMMY c01, ASTRA c02), each with voice_preset
      lines -> 6 entries when with_sfx=with_music=True:
                 music_open, announcer, character (c01), character (c02),
                 sfx, music_close.
               4 entries when both flags are False.

    The cast is a LIST of dicts to match production_ledger.py's
    ``set_cast`` output (NOT a dict keyed by char_id).
    """
    cast = [
        {
            "char_id":      "c01",
            "name":         "LEMMY",
            "description":  None,
            "gender":       "male",
            "voice_preset": "v2/en_speaker_3",
            "line_count":   1,
            "word_count":   6,
        },
        {
            "char_id":      "c02",
            "name":         "ASTRA",
            "description":  None,
            "gender":       "female",
            "voice_preset": "v2/en_speaker_9",
            "line_count":   1,
            "word_count":   2,
        },
    ]

    lines: list[dict] = []
    if with_music:
        lines.append({
            "line_id":      "b001",
            "shot_id":      None,
            "beat_id":      "b001",
            "char_id":      "music_open",
            "speaker_role": "music_open",
            "text":         "Theme up.",
            "traits":       "neutral",
            "start_s":      0.0,
            "dur_s":        4.0,
            "boundary":     "ep_open",
            "bark_wav_path": None,
        })
    lines.append({
        "line_id":      "b002",
        "shot_id":      None,
        "beat_id":      "b002",
        "char_id":      "announcer",
        "speaker_role": "announcer",
        "text":         "From the studio, tonight's broadcast.",
        "traits":       "warm baritone",
        "start_s":      4.0,
        "dur_s":        3.5,
        "boundary":     "ep_open",
        "bark_wav_path": None,
    })
    lines.append({
        "line_id":      "b003",
        "shot_id":      None,
        "beat_id":      "b003",
        "char_id":      "c01",
        "speaker_role": "character",
        "text":         "Get the kit out, fast.",
        "traits":       "urgent, gruff",
        "start_s":      7.5,
        "dur_s":        2.1,
        "boundary":     "scene_1_open",
        "bark_wav_path": None,
    })
    lines.append({
        "line_id":      "b004",
        "shot_id":      None,
        "beat_id":      "b004",
        "char_id":      "c02",
        "speaker_role": "character",
        "text":         "On it.",
        "traits":       "calm",
        "start_s":      9.6,
        "dur_s":        1.0,
        "boundary":     "scene_1_open",
        "bark_wav_path": None,
    })
    if with_sfx:
        lines.append({
            "line_id":      "b005",
            "shot_id":      None,
            "beat_id":      "b005",
            "char_id":      "sfx",
            "speaker_role": "sfx",
            "text":         "metal door slam",
            "traits":       "neutral",
            "start_s":      10.6,
            "dur_s":        0.8,
            "boundary":     "scene_1_open",
            "bark_wav_path": None,
        })
    if with_music:
        lines.append({
            "line_id":      "b006",
            "shot_id":      None,
            "beat_id":      "b006",
            "char_id":      "music_close",
            "speaker_role": "music_close",
            "text":         "Theme out.",
            "traits":       "neutral",
            "start_s":      11.4,
            "dur_s":        4.0,
            "boundary":     "ep_close",
            "bark_wav_path": None,
        })

    return {
        "meta": {
            "title": title or "STUB EPISODE",
            "news_seed": {"headline": headline or "Test headline"},
        },
        "cast":  cast,
        "lines": lines,
    }


def make_legacy_list() -> list:
    """Legacy parser-list shape used to verify ``load_ledger`` raises
    ``ValueError`` on stale wirings."""
    return [
        {"type": "dialogue", "content": "[VOICE: LEMMY, gruff] Hello there."},
        {"type": "sfx",      "description": "metal door slam"},
    ]


__all__ = ["make_stub_ledger", "make_legacy_list"]
