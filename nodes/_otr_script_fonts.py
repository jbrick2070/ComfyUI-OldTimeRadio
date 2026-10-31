"""nodes/_otr_script_fonts.py -- the script fonts the pack ships.

A Linux image carries no Chinese, Japanese or Hindi font, and nothing failed:
a Mandarin episode rendered on a RunPod pod (2026-10-21) burned every caption,
title card and credit as empty boxes. So the pack ships one face per non-Latin
script, under ``nodes/fonts`` (``assets/`` does not ship to the registry), and
every text surface reaches it
only AFTER the operating system's own faces miss -- Windows (Microsoft YaHei,
Nirmala UI) and macOS (PingFang, Devanagari Sangam) never use these, so their
output does not change.

* Noto Sans SC (``cjk``): Simplified Chinese, and Japanese kana; on a fontless
  box a Japanese line draws some kanji in their Chinese forms -- read, not boxed.
* OTR Noto Devanagari Latin (``devanagari``): Hindi. Noto Sans Devanagari has
  no Latin letters, so a credits line mixing a Hindi heading with names and
  model ids boxed its Latin half (RunPod, 2026-10-21); this is that font merged
  with Noto Sans and renamed. For the same reason it is tried BEFORE a Linux
  system Devanagari face (``otr_credits_roll._credits_script_font_paths``).

All are the Noto Project's fonts under the SIL Open Font License 1.1
(``nodes/fonts/OFL-*.txt``). Pure lookup: nothing is fetched.

LATIN, FOR A BOX WITH NO FONTS AT ALL (2026-10-31). Comfy Cloud's serverless
image carries no font files, not even DejaVu, and a five-episode run published
every episode with no captions, no hero title and the credits drawn in Pillow's
built-in face -- squashed letters, the third column off the right edge. libass
with nothing to select draws nothing and exits 0, so no step failed. The pack
ships DejaVu Sans Mono (every monospace surface) and DejaVu Sans (the SDH
caption's Arial) under ``LATIN``, reached the same way: only when the machine
has no Latin face of its own, so Windows, macOS and any Linux with a font
package render exactly as before. DejaVu's licence is
``nodes/fonts/LICENSE-DejaVu.txt`` (Bitstream Vera terms; DejaVu's changes are
public domain).
UTF-8 no BOM, ASCII-only source.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat (sys.path) load
    from _otr_shared import env as otr_env  # type: ignore

log = logging.getLogger("OTR")

FONTS_DIR = Path(__file__).resolve().parent / "fonts"

#: captions.font_policy -> (file in nodes/fonts, the font's family name).
BUNDLED = {
    "cjk": ("NotoSansSC-Regular.otf", "Noto Sans SC"),
    "devanagari": ("OTRNotoDevanagariLatin-Regular.ttf", "OTR Noto Devanagari Latin"),
}

#: Latin faces for a font-less machine: key -> (file in nodes/fonts, family).
#: Kept apart from ``BUNDLED`` because those keys are language-row font
#: policies and these are not.
LATIN = {
    "latin_mono": ("DejaVuSansMono.ttf", "DejaVu Sans Mono"),
    "latin_sans": ("DejaVuSans.ttf", "DejaVu Sans"),
}

#: The install line that gives a Linux box its own face for a script.
LINUX_PACKAGE = {"cjk": "fonts-noto-cjk", "devanagari": "fonts-noto-core",
                 "latin_mono": "fonts-dejavu-core", "latin_sans": "fonts-dejavu-core"}

#: Where the operating systems keep their own Latin faces, absolute paths
#: first, then names Pillow resolves through the platform font folders (on
#: Linux the XDG set), which finds a face under a layout not listed here.
_OS_LATIN_PATHS = {
    "latin_mono": [
        "{fd}/consola.ttf", "{fd}/cour.ttf", "{fd}/lucon.ttf",
        "/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf",
        "/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    ],
    "latin_sans": [
        "{fd}/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ],
}
_OS_LATIN_NAMES = {
    "latin_mono": ["DejaVuSansMono", "LiberationMono-Regular", "NotoSansMono-Regular"],
    "latin_sans": ["DejaVuSans", "LiberationSans-Regular", "NotoSans-Regular", "Arial"],
}

_WARNED: set = set()


def _entry(policy: str):
    key = str(policy or "").strip()
    return BUNDLED.get(key) or LATIN.get(key)


def bundled_path(policy: str) -> "str | None":
    """The shipped face for ``policy``, or None when the policy has none or the
    file is missing from this install (warned once, with the fix)."""
    entry = _entry(policy)
    if entry is None:
        return None
    path = FONTS_DIR / entry[0]
    if path.is_file():
        return str(path)
    if policy not in _WARNED:
        _WARNED.add(policy)
        log.warning(
            "OTR: the pack's %s font %s is missing from this install, so %s text "
            "draws as empty boxes unless the machine has its own face (Linux: "
            "apt install %s). Reinstall the pack to restore it.",
            policy, path, policy, LINUX_PACKAGE.get(policy, ""))
    return None


def bundled_family(policy: str) -> "str | None":
    entry = _entry(policy)
    return entry[1] if entry else None


def os_latin_face(policy: str) -> "str | None":
    """The machine's own Latin face for ``policy`` (``latin_mono`` or
    ``latin_sans``), or None on a font-less box such as Comfy Cloud's image."""
    from PIL import ImageFont

    fd = os.path.join(otr_env.get("WINDIR", r"C:\Windows"), "Fonts")
    found = os_face(policy, [p.format(fd=fd) for p in _OS_LATIN_PATHS.get(policy, ())])
    if found:
        return found
    shipped = bundled_path(policy)
    for name in _OS_LATIN_NAMES.get(policy, ()):
        try:
            path = ImageFont.truetype(name, 12).path
        except OSError:
            continue
        if path != shipped:
            return path
    return None


def os_face(policy: str, candidates) -> "str | None":
    """The first of ``candidates`` that is a file and is not the shipped face:
    the machine's own font for the script, or None."""
    shipped = bundled_path(policy)
    for candidate in candidates or ():
        if candidate and candidate != shipped and os.path.isfile(candidate):
            return candidate
    return None


def families_in(text: str) -> list:
    """The shipped families named in an ASS script, so the burn can hand libass
    exactly those files. A Latin face is handed over only when the machine has
    none of its own, so a burn on a box with fonts is unchanged."""
    named = [policy for policy, (_file, family) in {**BUNDLED, **LATIN}.items()
             if ("," + family + ",") in str(text or "")]
    return [policy for policy in named
            if policy not in LATIN or os_latin_face(policy) is None]
