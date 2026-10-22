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
UTF-8 no BOM, ASCII-only source.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("OTR")

FONTS_DIR = Path(__file__).resolve().parent / "fonts"

#: captions.font_policy -> (file in nodes/fonts, the font's family name).
BUNDLED = {
    "cjk": ("NotoSansSC-Regular.otf", "Noto Sans SC"),
    "devanagari": ("OTRNotoDevanagariLatin-Regular.ttf", "OTR Noto Devanagari Latin"),
}

#: The install line that gives a Linux box its own face for a script.
LINUX_PACKAGE = {"cjk": "fonts-noto-cjk", "devanagari": "fonts-noto-core"}

_WARNED: set = set()


def bundled_path(policy: str) -> "str | None":
    """The shipped face for ``policy``, or None when the policy has none or the
    file is missing from this install (warned once, with the fix)."""
    entry = BUNDLED.get(str(policy or "").strip())
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
    entry = BUNDLED.get(str(policy or "").strip())
    return entry[1] if entry else None


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
    exactly those files."""
    return [policy for policy, (_file, family) in BUNDLED.items()
            if ("," + family + ",") in str(text or "")]
