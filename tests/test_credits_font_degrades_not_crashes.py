"""A machine with no monospace font must still PUBLISH the episode.

Operator ruling 2026-09-11: *"don't assume people have fonts installed. I'm open to
some bad formatting as long as it doesn't crash."*

This REVERSES the earlier no-fallback policy ("a point-size-less bitmap hero is
unacceptable", Fable risk #3). That ruling optimised for how the card looks. The
standing rule is that a leg which does not reach ``otr/obs/`` did not pass, and the
bar is "as long as it doesn't crash when it's not supposed to".

The old behaviour raised ``CreditsDataError`` at the credits stage -- i.e. it killed a
FULLY RENDERED episode, after the script, the cast, every voice, the audio master and
every clip had already been paid for, because a font file was missing. An ugly card
ships; a CreditsDataError ships nothing.

A fresh install on a box without DejaVu/consola/Menlo is exactly where this bites, so
it is a portability defect as much as a crash one.
"""
from __future__ import annotations

import pytest
from PIL import ImageFont

from nodes import otr_credits_roll as cr


@pytest.fixture(autouse=True)
def _clear_font_cache():
    cr._FONT_CACHE.clear()
    yield
    cr._FONT_CACHE.clear()


def _no_fonts_anywhere(monkeypatch):
    """Every candidate FILE fails to load, as on a bare container.

    Deliberately surgical: only PATH loads fail. PIL's `load_default()` resolves
    an EMBEDDED base64 font through this same `truetype` entry point, so a blunt
    patch would break the fallback itself and test nothing real -- a box with no
    font files still has PIL's built-in one. Verified by reading
    PIL.ImageFont.load_default's source.
    """
    real = ImageFont.truetype

    def _paths_fail(font=None, size=10, *a, **kw):
        if isinstance(font, (str, bytes)):
            raise OSError("cannot open resource: %s" % (font,))
        return real(font, size, *a, **kw)

    monkeypatch.setattr(ImageFont, "truetype", _paths_fail)


def test_a_machine_with_no_monospace_font_still_gets_a_font(monkeypatch):
    """THE DEFECT: this used to raise CreditsDataError and lose the episode."""
    _no_fonts_anywhere(monkeypatch)

    font = cr._load_font(48)

    assert font is not None, "credits must degrade, not refuse"


def test_the_degraded_font_is_usable_for_measurement(monkeypatch):
    """An unusable fallback would just move the crash into the layout pass."""
    _no_fonts_anywhere(monkeypatch)

    font = cr._load_font(32)
    # the layout ladder measures text; the fallback must survive that
    from PIL import Image, ImageDraw
    draw = ImageDraw.Draw(Image.new("RGB", (64, 32)))
    width = cr._fw(draw, "THE BAY AREA TABLE", font)
    assert isinstance(width, int) and width > 0


def test_the_explicit_override_is_still_tried_first(monkeypatch):
    """OTR_CREDITS_FONT remains the documented remedy and must still win."""
    seen = []

    real = ImageFont.truetype

    def _record(font=None, size=10, *a, **kw):
        if isinstance(font, (str, bytes)):
            seen.append(font)
            raise OSError("nope")
        return real(font, size, *a, **kw)

    monkeypatch.setenv("OTR_CREDITS_FONT", r"C:\fonts\MyMono.ttf")
    monkeypatch.setattr(ImageFont, "truetype", _record)

    cr._load_font(20)

    assert seen and seen[0] == r"C:\fonts\MyMono.ttf", seen[:3]


def test_a_resolvable_font_is_unchanged(monkeypatch):
    """Exception-only in spirit: a box WITH fonts behaves exactly as before."""
    sentinel = object()
    calls = []

    def _first_wins(font=None, size=10, *a, **kw):
        calls.append(font)
        return sentinel

    monkeypatch.setattr(ImageFont, "truetype", _first_wins)

    assert cr._load_font(24) is sentinel
    assert len(calls) == 1, "it must stop at the first font that loads"


def test_the_degraded_font_is_cached_like_any_other(monkeypatch):
    """The fallback must not re-walk every candidate on every call -- the
    credits ladder asks for several point sizes per card."""
    _no_fonts_anywhere(monkeypatch)

    first = cr._load_font(18)
    second = cr._load_font(18)
    assert first is second
