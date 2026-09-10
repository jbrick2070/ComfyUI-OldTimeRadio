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


def _resolve_on(monkeypatch, platform, present):
    """Run the REAL resolver against a simulated host carrying `present`.

    Deliberately drives `_find_mono_font_path` itself rather than re-deriving
    what it ought to pick. An earlier version of this file compared hardcoded
    strings through a helper of its own and passed green while the shipped
    candidate list had Arch's Liberation path ahead of Arch's DejaVu path --
    a reviewer found the defect the test was nominally guarding. A test that
    re-implements the logic tests the re-implementation.
    """
    from PIL import ImageFont

    import nodes.video_engine as ve

    have = set(present)
    monkeypatch.setattr(ve.sys, "platform", platform, raising=False)
    monkeypatch.setattr(ve.os.path, "isfile", lambda q: q in have)

    class Stub:
        def __init__(self, path):
            self.path = path

    def fake_truetype(name, size=10, *a, **k):
        if name in have:
            return Stub(name)
        raise OSError("absent on this simulated host: %r" % (name,))

    monkeypatch.setattr(ImageFont, "truetype", fake_truetype)
    monkeypatch.setattr(ve, "_FONT_PATH", None, raising=False)
    monkeypatch.setattr(ve, "_FONT_PATH_RESOLVED", False, raising=False)
    return ve._find_mono_font_path(96)


#: Real package layouts, each taken from the distribution's own file list.
_DEJAVU = {
    "debian": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "opensuse": "/usr/share/fonts/truetype/DejaVuSansMono.ttf",
    "fedora": "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
    "arch": "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
}
_LIBERATION = {
    "debian": "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "fedora": "/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf",
    "arch": "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
}


@pytest.mark.parametrize("distro", sorted(_LIBERATION))
def test_dejavu_beats_liberation_when_a_linux_host_has_both(monkeypatch, distro):
    """THE regression, and it shipped for one commit.

    `_otr_captions._MONO_FALLBACK` names "DejaVu Sans Mono" as the family
    libass DRAWS with on every non-Mac, non-Windows host. So whenever DejaVu is
    on the box, the measuring side must resolve DejaVu -- otherwise metrics
    come from one family and glyphs from another, which is the whole defect
    class this file exists for.

    Ordering the candidates by DISTRO put Arch's Liberation path ahead of
    Arch's DejaVu path, so an ordinary Arch box carrying both packages -- not
    the documented Liberation-only edge case -- measured the wrong family. The
    list is now grouped by FAMILY, which makes the class unreachable instead of
    fixing the one pair that happened to be wrong.
    """
    picked = _resolve_on(monkeypatch, "linux",
                         [_DEJAVU[distro], _LIBERATION[distro]])
    assert picked == _DEJAVU[distro], (
        "on a %s host carrying BOTH families the resolver chose %r; it must "
        "choose DejaVu (%r), because that is the family the ASS style names "
        "and libass will actually draw."
        % (distro, picked, _DEJAVU[distro]))


@pytest.mark.parametrize("distro", sorted(_LIBERATION))
def test_liberation_is_still_reached_when_it_is_all_the_host_has(monkeypatch,
                                                                distro):
    """Grouping by family must not strand a Liberation-only box.

    It is the wrong family and it is still enormously better than the bitmap
    fallback: Liberation Mono is metric-compatible with DejaVu Sans Mono, while
    the fallback is off by roughly a factor of ten. Preferring DejaVu must not
    turn a working host into a broken one.
    """
    picked = _resolve_on(monkeypatch, "linux", [_LIBERATION[distro]])
    assert picked == _LIBERATION[distro], (
        "a %s host with only Liberation resolved %r instead of falling back to "
        "it; regrouping the candidates must not strand these hosts on the "
        "bitmap default." % (distro, picked))


def test_debian_ubuntu_resolution_is_unchanged_by_the_regrouping(monkeypatch):
    """The boxes that already worked must keep working, byte for byte.

    Debian/Ubuntu is the layout the original two-entry list was written for, so
    it is the one that must be shown unaffected rather than assumed to be.
    """
    both = _resolve_on(monkeypatch, "linux",
                       [_DEJAVU["debian"], _LIBERATION["debian"]])
    assert both == _DEJAVU["debian"], "DejaVu must still win when both present"

    only_lib = _resolve_on(monkeypatch, "linux", [_LIBERATION["debian"]])
    assert only_lib == _LIBERATION["debian"], (
        "a Debian box with only Liberation must still resolve it")


def test_windows_and_macos_are_untouched_by_the_linux_regrouping(monkeypatch):
    """Neither branch shares the Linux list, and that is worth pinning.

    The Windows path is BUILT with `os.path.join`, exactly as the resolver
    builds it, rather than written out with backslashes. The first draft of
    this test hardcoded the separator and failed on this Mac, where join emits
    `C:\\Windows/Fonts/consola.ttf` -- the same "re-implement instead of
    invoke" mistake this file was just rewritten to stop making.
    """
    windir = os.environ.get("WINDIR", r"C:\Windows")
    win = os.path.join(os.path.join(windir, "Fonts"), "consola.ttf")
    assert _resolve_on(monkeypatch, "win32", [win]) == win, (
        "the win32 branch must still resolve consola.ttf; the Linux "
        "regrouping shares no list with it")

    mac = "/System/Library/Fonts/Menlo.ttc"
    assert _resolve_on(monkeypatch, "darwin", [mac]) == mac, (
        "the darwin branch must still resolve Menlo, the family the ASS style "
        "names for macOS")
