"""The shipped per-bank music defaults are documented where a user reads them.

On 2026-09-12 four banks gained a fixed genre and `OTR_StableAudioTheme` a
`music_style` widget honoured only on My Story -- "these are the new
defaults" -- and no user-facing file said so. README and every generated
launch recipe now carry the table, and this test reads all three off the one
source, `nodes/_otr_music_palette.bank_music_table`, so none can drift.
"""
from __future__ import annotations

import importlib.util
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


def test_readme_names_every_bank_idiom_verbatim():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for bank, idiom, _rhythmic in P.bank_music_table():
        assert f"`{bank}`" in readme, bank
        assert idiom in readme, (bank, idiom)
    assert "`music_style`" in readme and "`my_story`" in readme


def test_every_generated_launch_recipe_carries_the_table():
    spec = importlib.util.spec_from_file_location(
        "_bv_for_music_docs", ROOT / "scripts" / "build_variants.py")
    bv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bv)
    _variant, _rel, recipe = bv.build_variant("otr_16gb_foley")
    assert "## Music" in recipe
    committed = (ROOT / "workflows" / "variants" / "otr_16gb_foley.launch.md"
                 ).read_text(encoding="utf-8")
    for bank, idiom, rhythmic in P.bank_music_table():
        # A rhythmic palette is the bank's fixed identity; a non-rhythmic one
        # is only its default when the source has no year (`story_palette`),
        # and the recipe must not overclaim the second as the first.
        expected = (f"- `{bank}`: {idiom} (fixed)" if rhythmic
                    else f"- `{bank}`: chosen by the source's year; "
                         f"its own default is {idiom}")
        assert expected in recipe, (bank, expected)
        # And the committed recipes were regenerated from this generator.
        assert expected in committed, (bank, expected)
    assert "`music_style`" in recipe and "`my_story`" in recipe
