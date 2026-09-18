"""An indicator the operator can read: iso in the filename, label on the card.

Operator 2026-09-18: "it would be nice for me to easily know what language
each one is in as I don't know Japanese or Portuguese or Chinese". Non-English
episodes publish as ``signal_lost_<title>_<iso>_<ts>.mp4`` and bake
``<TITLE> · <LANGUAGE>`` on the hero card. English, Off and legacy ledgers are
byte-identical. CPU only.
"""
from __future__ import annotations

import inspect

import pytest

from nodes import video_engine as VE


@pytest.mark.parametrize("iso, label", [
    ("es", "Spanish"), ("pt", "Portuguese"), ("it", "Italian"), ("fr", "French"),
    ("hi", "Hindi"), ("ja", "Japanese"), ("zh", "Mandarin"),
])
def test_a_non_english_ledger_yields_its_iso_and_english_label(iso, label):
    assert VE._language_marks({"meta": {"episode_language": iso}}) == ("_" + iso, label)


@pytest.mark.parametrize("led", [
    {"meta": {"episode_language": "en"}},
    {"meta": {}},
    {},
    None,
    "not a ledger",
    {"meta": {"episode_language": "tlh"}},
])
def test_english_off_legacy_and_unreadable_add_nothing(led):
    assert VE._language_marks(led) == ("", "")


def test_the_filename_carries_the_iso_before_the_timestamp():
    src = inspect.getsource(VE)
    assert 'f"signal_lost_{safe_title}{lang_suffix}_{ts}.mp4"' in src
    assert 'f"signal_lost_{safe_title}_{ts}.mp4"' not in src


def _renderer(title, card_label=""):
    return VE._CRTRenderer(64, 64, title, [], [], [], 24, card_label=card_label)


def test_the_hero_card_carries_the_english_label_only_off_english():
    assert _renderer("El mapa prohibido")._hero_text() == "EL MAPA PROHIBIDO"
    assert _renderer("El mapa prohibido", "Spanish")._hero_text() == \
        "EL MAPA PROHIBIDO · SPANISH"
    assert _renderer("", "Japanese")._hero_text() == "SIGNAL · JAPANESE"


def test_the_label_never_reaches_the_ident_hud_or_the_scramble_seed():
    r = _renderer("El mapa prohibido", "Spanish")
    assert r.title == "El mapa prohibido"
    src = inspect.getsource(VE._CRTRenderer)
    ident = src[src.index("def _draw_ident"):]
    ident = ident[:ident.index("\n    def ")]
    assert "card_label" not in ident and "_hero_text" not in ident
    assert "rng_title=self.title" in src


def test_the_renderer_receives_the_plain_title_and_the_label_separately():
    src = inspect.getsource(VE)
    assert "_CRTRenderer(W, H, episode_title, volume, freqs, waves, fps," in src
    assert "card_label=lang_label)" in src
    assert "display_title" not in src
