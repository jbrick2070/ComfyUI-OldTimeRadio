"""The shipped per-bank music defaults are documented where a user reads them.

On 2026-09-12 four banks gained a fixed genre and `OTR_StableAudioTheme` a
`music_style` widget honoured only on My Story -- "these are the new
defaults" -- and no user-facing file said so. README and every generated
launch recipe now carry the table, and this test reads all three off the one
source, `nodes/_otr_music_palette.bank_music_table`, so none can drift.
"""
from __future__ import annotations

import pathlib

from nodes import _otr_music_palette as P

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_the_table_is_the_declared_palettes_sorted():
    table = P.bank_music_table()
    assert [row[0] for row in table] == sorted(P._BANK_PALETTES)
    for bank, idiom, rhythmic in table:
        assert idiom == P._BANK_PALETTES[bank].idiom
        assert rhythmic is P._BANK_PALETTES[bank].rhythmic
    assert "my_story" not in dict((b, i) for b, i, _ in table)


