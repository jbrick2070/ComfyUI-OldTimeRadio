"""Cast names in the episode's own script, for the surfaces a viewer reads.

Operator 2026-11-27: on a Japanese, Mandarin or Hindi episode the burned
captions opened every turn with the speaker in Latin letters ("ROBIN:",
"MALVOLIO:", "REGINALD SIRIKIT:"), the one thing on screen a native viewer
could not read. Only the reserved ANNOUNCER was localized.

WHAT THIS OWNS: one optional cast-row field, ``native_display_name``, filled
by one small structured call per non-Latin-script episode, after the cast is
final. Captions and credits paint it; nothing else reads it.

WHAT IT NEVER TOUCHES: ``cast[].name``. That string is an identity key in
roughly forty places (the verbatim executor's exact speaker match, cast
partition, voice coverage, scene sequencing), so the display name is a
separate field painted at the surface, the same way the announcer is.

FAIL-SOFT BY CONSTRUCTION. Every name is judged on its own; a missing,
echoed, merged, wrong-script, colliding or overlong name keeps its Latin
spelling, and a failed call keeps them all. Nothing here raises into the
episode. English and the Latin-script rows make no call and change no byte.

Design read by Codex and agy (2026-11-27), both picking this shape; agy
supplied the rule for role names (translate role words, transliterate
personal names) and the reference spellings the tests use. UTF-8, no BOM.
"""
from __future__ import annotations

import logging
import unicodedata
from typing import Any, Callable

from pydantic import BaseModel, Field

try:
    from . import _otr_episode_languages as _EPLANG
    from ._otr_repair_prompts import make_dispatching_repair_factory
    from ._otr_structured_call import structured_call
except ImportError:  # pragma: no cover -- flat test/standalone load
    import _otr_episode_languages as _EPLANG  # type: ignore
    from _otr_repair_prompts import make_dispatching_repair_factory  # type: ignore
    from _otr_structured_call import structured_call  # type: ignore

log = logging.getLogger("OTR")

FIELD = "native_display_name"
RECEIPT_KEY = "cast_display_names"
RECEIPT_VERSION = "otr_cast_display_names_v1"

#: Longest display name kept. A caption prefix longer than this is a sentence,
#: and CAST & VOICES draws "<native> (<NAME>)" on one line of a 424 px column.
MAX_CHARS = 16

#: Characters a name never carries: the caption separator, line breaks, JSON
#: punctuation (a merged or broken reply), list separators, a backslash (an
#: ASS override: "\N" is a line break on the burned caption), and brackets (a
#: reply that glosses the Latin name, "Viola (ヴァイオラ)").
_FORBIDDEN = set(":：\n\r\t[]{}\"/|;；\\()（）।॥")

#: The scripts a name may be written in, by episode language, read from each
#: letter's Unicode name. Sonnet QA on 003c3b84f: "any non-Latin letter" let
#: Cyrillic, Greek or Hangul through on a Japanese episode.
_HAN = ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH",
        "IDEOGRAPHIC ITERATION MARK", "VERTICAL IDEOGRAPHIC ITERATION MARK",
        "IDEOGRAPHIC NUMBER ZERO", "IDEOGRAPHIC CLOSING MARK")
_SCRIPTS = {
    "ja": _HAN + ("HIRAGANA", "KATAKANA"),
    "zh": _HAN,
    "hi": ("DEVANAGARI",),
}

#: Non-letters a name may carry besides spaces and the digits 0-9: the
#: katakana middle dot between given and family name (and its half-width
#: form before NFKC), and the Devanagari joiners that shape a conjunct.
_ALLOWED_MARKS = {
    "ja": set("・"),
    "zh": set("·・"),
    "hi": set("‌‍"),
}


class _NameRow(BaseModel):
    """One numbered answer. The row is keyed by the NUMBER the name was asked
    with, never by the name echoed back: Qwen 3.5 4B translated the "name"
    key too (Mac, 2026-11-27), and pairing by position instead would put a
    name on the wrong character when the reply is reordered or slips in an
    extra row (Sonnet QA on 5b6dcedb0)."""
    n: int
    display_name: str


class CastDisplayNames(BaseModel):
    names: list[_NameRow] = Field(min_length=1)


_SYSTEM = (
    "You write the character names of a radio play the way a viewer of this "
    "episode reads them in captions. Return JSON only: {\"names\": "
    "[{\"n\": <the number the name is listed with, starting at 1>, "
    "\"display_name\": \"...\"}]} with one entry per numbered name. Transliterate personal "
    "and place names into the episode's script, using the established "
    "rendering where a famous one exists (Shakespeare's characters have "
    "standard published names). Translate generic role words, ordinals and "
    "prepositions instead of spelling them out (FIRST WITCH is a first witch; "
    "ANTIPHOLUS OF EPHESUS keeps the place as a place). Keep each name short. "
    "Write the name only: no colon, no notes, no Latin letters unless the "
    "name itself is an acronym."
)


def _is_latin_or_ascii(ch: str) -> bool:
    return ord(ch) <= 127 or "LATIN" in unicodedata.name(ch, "")


def _output_budget(name_count: int) -> int:
    """Tokens for the reply: a fixed allowance for the JSON frame (and the
    code fence a small model adds) plus room per name for the echoed Latin
    name, the native name and the indentation around them."""
    return 128 + 64 * max(1, int(name_count))


def needs_native_names(meta: Any) -> bool:
    """True for a non-English episode whose own language name is not written
    in Latin letters (Hindi, Japanese, Mandarin today). English, Off, legacy,
    an unreadable row and the Romance rows make no call."""
    try:
        if not isinstance(meta, dict) or not str(meta.get("episode_language") or "").strip():
            return False
        row = _EPLANG.row_from_meta(meta)
        if row.iso == _EPLANG.ENGLISH_ISO:
            return False
        native = str(row.native_header or "")
        letters = [c for c in native if c.isalpha()]
        return bool(letters) and not all(_is_latin_or_ascii(c) for c in letters)
    except Exception:  # noqa: BLE001 -- a display nicety never breaks a run
        return False


def judge_display_name(name: str, display: str, iso: str = "") -> str | None:
    """Why ``display`` cannot stand for ``name`` on screen, or None if it can.

    With ``iso``, every letter must be in that language's script (kana and
    kanji for Japanese, Han for Mandarin, Devanagari for Hindi); without it,
    every letter must at least be non-Latin."""
    value = " ".join(unicodedata.normalize("NFKC", str(display or "")).split())
    if not value:
        return "empty"
    if len(value) > MAX_CHARS:
        return "too long"
    marks = _ALLOWED_MARKS.get(str(iso or ""), set())
    if any(c in _FORBIDDEN for c in value) or any(
            unicodedata.category(c).startswith("C") and c not in marks
            for c in value):
        return "punctuation a name does not carry"
    if value.casefold() == " ".join(str(name or "").split()).casefold():
        return "the Latin name echoed"
    letters = [c for c in value if c.isalpha()]
    if not letters or any(_is_latin_or_ascii(c) for c in letters):
        return "not in the episode's script"
    allowed = _SCRIPTS.get(str(iso or ""))
    if allowed:
        # EVERY character, not only letters: an emoji, a circled symbol, a
        # danda or a stray combining accent is not a name either (Sonnet QA).
        for c in value:
            if c == " " or c in "0123456789" or c in marks:
                continue
            if not unicodedata.name(c, "").startswith(allowed):
                return "not in the episode's script"
    return None


def display_name(cast_row: Any) -> str:
    """What a viewer-facing surface paints for a cast row: the native display
    name when one was accepted, else the identity name, else the legacy
    ``display_name`` fallback the captions always had."""
    if not isinstance(cast_row, dict):
        return ""
    return str(cast_row.get(FIELD) or cast_row.get("name")
               or cast_row.get("display_name") or "").strip()


def populate(led_data: dict, *, technical_fn: Callable[..., str]) -> dict:
    """Ask once, judge every name, write the accepted ones. Returns a receipt
    of counts (no names copied: the cast row is the one owner) and stores it
    at ``meta[RECEIPT_KEY]``. Never raises."""
    meta = led_data.setdefault("meta", {})
    receipt: dict = {"receipt_version": RECEIPT_VERSION, "asked": 0,
                     "accepted": 0, "rejected": {}}
    try:
        if not needs_native_names(meta):
            return {}
        rows = [r for r in (led_data.get("cast") or [])
                if isinstance(r, dict) and str(r.get("name") or "").strip()
                and str(r.get("name")).strip().upper() != "ANNOUNCER"]
        if not rows or any(r.get(FIELD) for r in rows):
            # Nothing to name, or a replay of a ledger that already has them.
            return {}
        wanted = [str(r["name"]).strip() for r in rows]
        receipt["asked"] = len(wanted)
        iso = _EPLANG.row_from_meta(meta).iso
        lead = _EPLANG.native_translation_instruction(meta)
        user = ("Character names:\n"
                + "\n".join(f"{i}. {n}" for i, n in enumerate(wanted, 1)))
        result = structured_call(
            prompt=[{"role": "system", "content": _EPLANG.lead_system(_SYSTEM, lead)},
                    {"role": "user", "content": user}],
            schema=CastDisplayNames,
            slot_fn=technical_fn,
            base_temperature=0.2,
            structural_retry_temperature=0.1,
            repair_prompt_factory=make_dispatching_repair_factory(),
            # Room for a fenced, pretty-printed reply: Gemma 4 E2B on the 4060
            # (2026-11-27) wrapped three names in ```json and indented them,
            # ran out at 96 tokens mid-name ("尤里·肯德尔…") on both attempts,
            # and every name fell back to Latin.
            max_new_tokens=_output_budget(len(wanted)),
            max_attempts=2,
            helper_name="cast_display_names",
        )
        proposed: dict[str, str] = {}
        # Each row names the number its name was asked with; a number outside
        # the list (an invented row) matches nothing, and the first answer for
        # a number wins.
        if any(item.n == 0 for item in result.names):
            # Counted from zero, every answer would land one character early,
            # and nothing would show it (Sonnet QA on 7502d17bd). The whole
            # reply is void and every name keeps its Latin spelling.
            raise ValueError("the reply numbered the names from 0")
        seen_numbers: set[int] = set()
        for item in result.names:
            if not 1 <= item.n <= len(wanted) or item.n in seen_numbers:
                continue
            seen_numbers.add(item.n)
            match = wanted[item.n - 1]
            reason = judge_display_name(match, item.display_name, iso)
            if reason:
                receipt["rejected"][match] = reason
                continue
            receipt["rejected"].pop(match, None)
            proposed[match] = " ".join(
                unicodedata.normalize("NFKC", str(item.display_name)).split())
        # Two speakers who would read the same on screen keep their Latin names.
        seen: dict[str, list[str]] = {}
        for latin, native in proposed.items():
            seen.setdefault(native.casefold(), []).append(latin)
        for group in seen.values():
            if len(group) > 1:
                for latin in group:
                    proposed.pop(latin, None)
                    receipt["rejected"][latin] = "shared with another speaker"
        for latin in wanted:
            if latin not in proposed and latin not in receipt["rejected"]:
                receipt["rejected"][latin] = "missing from the reply"
        for r in rows:
            native = proposed.get(str(r["name"]).strip())
            if native:
                r[FIELD] = native
        receipt["accepted"] = len(proposed)
    except Exception as exc:  # noqa: BLE001 -- every name keeps its Latin spelling
        receipt["error"] = type(exc).__name__
        log.warning("[cast_display_names] kept every Latin name: %s",
                    type(exc).__name__)
    meta[RECEIPT_KEY] = receipt
    log.info("[cast_display_names] %d of %d name(s) in the episode's script",
             receipt.get("accepted", 0), receipt.get("asked", 0))
    return receipt


__all__ = [
    "FIELD", "RECEIPT_KEY", "RECEIPT_VERSION", "CastDisplayNames",
    "display_name", "judge_display_name", "needs_native_names", "populate",
]
