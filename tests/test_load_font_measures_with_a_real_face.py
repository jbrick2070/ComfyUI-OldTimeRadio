"""`_load_font` must return a REAL scalable face, because centring is arithmetic.

WHAT SHIPPED BROKEN, and why it was invisible for so long.

Nothing in the title path passes a "centre" flag to anything. `_otr_title_card`
centres the hero itself -- `x = cx_centre - tw // 2` -- using a width measured
by PIL through `video_engine._load_font`. `_otr_captions` then emits that x as
an ASS `\\pos()` under Alignment 7, which is top-LEFT, so libass plants the
glyphs' left edge exactly where the measurement said and draws them in the ASS
style's own font at the real size. Measurement and drawing are therefore two
DIFFERENT font stacks that must agree, and only the arithmetic connects them.

`_load_font`'s non-win32 branch listed two Linux distro paths and nothing else.
On macOS neither exists, so every requested size fell through to
`ImageFont.load_default()` -- a ~10px bitmap face that IGNORES the size argument.
At title size 96 it measured a 21-character string at tw=131 instead of 1218, so
the subtraction produced x=895 and Menlo drew from 895 to 2113 on a 1920 frame:
off the right edge and clipped. The operator's report was "flush right, not
centre", and Windows was flawless the whole time because consola.ttf is really
in C:\\Windows\\Fonts. A platform gap in a fallback chain, not a layout bug.

These tests are deliberately split: the first runs everywhere and catches the
generic defect (a bitmap fallback silently standing in for a scalable face), the
second is macOS-only and pins the specific agreement that was broken.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _font(size):
    from nodes.video_engine import _FONT_CACHE, _load_font
    _FONT_CACHE.pop(size, None)          # the cache is global and per-size
    return _load_font(size)


def test_the_face_actually_scales_with_the_requested_size():
    """The property that matters, stated without naming any font file.

    A bitmap fallback returns the same glyphs for size 24 and size 96, so its
    measured widths are equal. A scalable face's are not. This is the assertion
    that would have failed on the Mac from the day the title card was written,
    on any platform whose candidate list goes stale.
    """
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (1920, 1080)))
    text = "LIGHTNING_MAC_PROOF_3"
    small = draw.textbbox((0, 0), text, font=_font(24))[2]
    large = draw.textbbox((0, 0), text, font=_font(96))[2]

    assert large > small * 2, (
        "_load_font is not returning a scalable face on %s: size 24 measured "
        "%dpx and size 96 measured %dpx. A face that ignores `size` makes every "
        "centred element mis-place, because `_otr_title_card` centres with "
        "`x = centre - tw // 2` and libass then draws at the REAL size."
        % (sys.platform, small, large))


def test_the_advance_per_character_tracks_the_requested_size():
    """NOT "does the centred string fit" -- that assertion is a tautology.

    The obvious test is `x = 960 - tw // 2; assert x + tw <= 1920`, and it is
    WORTHLESS: `(960 - tw // 2) + tw` is `960 + ceil(tw / 2)`, which is <= 1920
    for ANY tw <= 1920. It passes on the bitmap fallback's tw=131 exactly as
    happily as on Menlo's tw=1218. This file shipped that version for one
    revision and a reviewer proved it green against the broken code; it is
    written down here so nobody re-derives it.

    What actually separates a real face from the fallback is whether the glyph
    advance responds to `size` at all. A 21-character monospace string at size
    96 measures ~58px per character (0.60 of the requested size); PIL's bitmap
    default measures ~6px per character (0.065) no matter what is asked for.
    The 0.30 floor sits an order of magnitude clear of the fallback and well
    under any real monospace, so it is font-agnostic rather than Menlo-specific.
    """
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (1920, 1080)))
    text = "LIGHTNING_MAC_PROOF_3"
    size = 96
    tw = draw.textbbox((0, 0), text, font=_font(size))[2]
    advance = tw / float(len(text))

    assert advance >= size * 0.30, (
        "_load_font(%d) on %s measures %.1fpx per character (%.3f of the "
        "requested size); a real monospace face gives ~0.60 and PIL's bitmap "
        "default gives ~0.065. The face is ignoring `size`, so every centred "
        "element is placed from a width an order of magnitude too small -- "
        "which is what put a title card off the right edge of the frame."
        % (size, sys.platform, advance, advance / size))


def test_this_platform_measures_in_the_family_libass_draws():
    """The two halves must name the SAME face, or the arithmetic is meaningless.

    `_otr_captions` already carried a per-platform family map; the measurement
    side never did, and nothing detected the disagreement because both halves
    individually "worked". Deliberately resolved through `mono_font()` rather
    than the raw map, so that setting `OTR_CAPTION_MONO_FONT` without the
    matching `OTR_VIDEO_FONT` path fails HERE instead of on a published frame.

    Runs on every platform, not just macOS: the same class of gap can open on
    any of them the moment a candidate list goes stale.
    """
    pytest.importorskip("PIL")
    from nodes._otr_captions import mono_font

    path = getattr(_font(96), "path", "")
    if not isinstance(path, str) or not path:
        pytest.fail(
            "_load_font(96) on %s returned PIL's bitmap default rather than a "
            "file-backed face; there is no family to compare." % sys.platform)

    def norm(v):
        return "".join(ch for ch in v.lower() if ch.isalnum())

    family = norm(mono_font())
    stem = norm(os.path.splitext(os.path.basename(path))[0])

    assert family.startswith(stem) or stem.startswith(family), (
        "the ASS style declares %r for %s but _load_font measured with %r. "
        "Measurement and drawing must agree on the family, or centred text is "
        "positioned with one font's metrics and drawn with another's. If this "
        "is a deliberate OTR_CAPTION_MONO_FONT override, set OTR_VIDEO_FONT to "
        "the matching TTF path." % (mono_font(), sys.platform, path))
