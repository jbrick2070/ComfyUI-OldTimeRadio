"""The published (obs) filename rule, spelled ONCE.

TWO MODULES HAVE TO AGREE ABOUT THIS NAME AND THEY CANNOT IMPORT EACH OTHER.
`otr_master_audio_mux` WRITES the published name; `_otr_ledger` VALIDATES it
and records `meta.paths.obs_final`. The mux already imports the ledger, so the
ledger can never import the mux, and any rule spelled in the mux is invisible
to the validator that has to accept its output.

That seam has come apart twice:

* **PBUG-20260904-06** -- the mux stripped `SHOW_PREFIX` from the published
  name and the validator demanded it, so every `meta.paths.obs_final` after the
  rename pointed at a file that did not exist.
* **PBUG-20260918-09** -- the title went through an ASCII-only sanitiser, so a
  Japanese or Hindi title erased and the stem no longer began with the episode
  id. Same failure MODE, different cause: the file was on disk and the ledger
  could not say so, on every non-Latin episode.

Those were different bugs, and this module is not a retelling of either. It is
here because the NEXT split is already designed: the two-word gloss is
sanitised by the mux and bound by the ledger. Spelling that in one place is
what stops a third recurrence.

Nothing here opens a socket, a model or a file.
"""
from __future__ import annotations

import re
import unicodedata

#: Cap on the whole published basename, counted in UTF-8 BYTES. A filesystem
#: counts UTF-16 units (NTFS) or bytes (ext4, and any share landing on one),
#: never codepoints -- a CJK title is about three bytes per codepoint, so a cap
#: applied to `len()` is roughly a third of what it claims.
OBS_NAME_MAX = 150

#: What a filesystem actually refuses, and nothing more. The Windows set is a
#: superset of POSIX's, so obeying it everywhere keeps one name per episode
#: across the 5080, the 4060 and the share between them.
FILENAME_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')

#: Windows refuses these as a basename whatever the extension.
WINDOWS_RESERVED = frozenset(
    ["con", "prn", "aux", "nul"]
    + ["com%d" % n for n in range(1, 10)]
    + ["lpt%d" % n for n in range(1, 10)])


def trim_to_bytes(text, budget):
    """Trim to ``budget`` UTF-8 BYTES without splitting a character.

    Slicing at an arbitrary codepoint splits a Devanagari cluster: U+0930 plus
    a vowel sign is one grapheme, and cutting between them leaves a combining
    mark that renders as a dotted circle. The walk is append-only so a base is
    always taken before its mark is considered, and the tail strip catches the
    virama (U+094D), which IS a combining mark by category.
    """
    text = str(text or "")
    if len(text.encode("utf-8")) <= budget:
        return text
    out = ""
    used = 0
    for character in text:
        size = len(character.encode("utf-8"))
        if used + size > budget:
            break
        out += character
        used += size
    while out and unicodedata.combining(out[-1]):
        out = out[:-1]
    return out


def sanitise_name_part(text, fallback="episode"):
    """One filename-safe part, in ITS OWN SCRIPT.

    THE TITLE IS THE OPERATOR'S CONTENT AND IS NOT TRANSLITERATED. Romanising
    was measured and rejected (PBUG-20260918-09): `anyascii` renders the
    Japanese 嘘の夜明け as `XunoYeMingke`, which is the CHINESE reading of the
    kanji, and drops the vowels out of Devanagari; `misaki.cutlet` returns IPA
    phonemes because misaki vendored it as a G2P. A wrong romanisation is worse
    than none, because nothing about it announces that it is wrong.

    Strip what a filesystem genuinely refuses, normalise, keep the rest.
    """
    # ORDER MATTERS: collapsing whitespace to `_` before trimming turns a
    # trailing space into a trailing underscore, and a whitespace-only input
    # into a bare `_` rather than the fallback.
    part = unicodedata.normalize("NFC", str(text or "")).strip()
    part = FILENAME_FORBIDDEN.sub("-", part)
    part = re.sub(r"\s+", "_", part)
    # Windows silently drops a trailing dot or space, which would make the name
    # on disk differ from the name we recorded.
    part = part.strip("-._ \t　")
    if not part:
        return fallback
    if part.split(".")[0].lower() in WINDOWS_RESERVED:
        part = "_" + part
    return part.lower()


#: `_<iso>_<YYYYMMDD>_<HHMMSS>`, plus a replay stamp when one is present. The
#: iso is ABSENT on English, matching `video_engine._language_marks`.
_TIMESTAMP_TAIL = re.compile(
    r"_(\d{8}_\d{6}(?:_replay_\d{8}_\d{6}_\d{6})?)$")


def identity_tail(episode_id, iso=""):
    """The part of an episode id that identifies it: `[_<iso>]_<timestamp>`.

    THE ISO COMES FROM THE LEDGER, NOT FROM THE STRING. Reading a two-letter
    code off the id looks like it works and does not: a title whose last word
    is two English letters is indistinguishable from a language tag.
    `..._of_it_20260918_190917` parses `it` and the file reads as an ITALIAN
    episode; `..._to_go_...` eats `go`. Narrowing the pattern to the eight
    admitted rows does not help, because `it` IS one of them. Both the writer
    and the validator would agree on the same wrong answer, so the BINDING
    holds -- the damage is a wrong language in a name the operator reads.

    So the caller passes the iso it already has (`meta.episode_language` through
    the same row lookup that minted the id) and this only confirms the id
    really ends that way. Returns "" when it does not, and the caller keeps
    today's name rather than guessing.
    """
    text = str(episode_id or "")
    match = _TIMESTAMP_TAIL.search(text)
    if not match:
        return ""
    stamp = match.group(1)
    iso = str(iso or "").strip().lower()
    if iso:
        expected = "_%s_%s" % (iso, stamp)
        return expected if text.endswith(expected) else ""
    return "_%s" % stamp


#: A model handed "summarise this title in two words" sometimes hands back the
#: instruction instead. These are the words of the frame, never of a title.
_FRAME_ECHO = frozenset(
    ["title", "episode", "summary", "filename", "label", "words", "untitled",
     "gloss", "index", "name"])

#: Two words is the ask. One or three are ACCEPTED: rejecting a good
#: three-word answer costs a retry and then an unreadable native fallback,
#: which is a worse outcome than a slightly long gloss.
_GLOSS_MAX_WORDS = 3
_GLOSS_MAX_WORD_CHARS = 16
_GLOSS_MAX_CHARS = 24


def validate_gloss(raw):
    """``(gloss, reason)`` -- the accepted two-word gloss, or "" and why not.

    NO PROFANITY CHECK, deliberately: operator directive 2026-08-03 forbids a
    content filter on the generation path.

    A Latin-script NON-ENGLISH reply (`reloj inquieto`) is accepted. It is
    readable, which is the whole purpose, and this repo already refused to add
    a language detector for exactly this case.
    """
    text = unicodedata.normalize("NFKD", str(raw or "")).strip()
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    # A letter that survives NFKD folding is a script that does not fold --
    # CJK, Devanagari, Cyrillic. `cafe` folds from `café` and passes.
    if any(c.isalpha() and ord(c) > 127 for c in text):
        return "", "not English script"
    words = [w for w in re.split(r"[^a-z0-9]+", text) if w]
    if not words:
        return "", "empty"
    if len(words) > _GLOSS_MAX_WORDS:
        return "", "%d words; wanted two" % len(words)
    if any(len(w) > _GLOSS_MAX_WORD_CHARS for w in words):
        return "", "a word over %d characters" % _GLOSS_MAX_WORD_CHARS
    if set(words) <= _FRAME_ECHO:
        return "", "echoed the instruction"
    gloss = "_".join(words)
    if len(gloss) > _GLOSS_MAX_CHARS:
        return "", "over %d characters" % _GLOSS_MAX_CHARS
    return gloss, ""


def episode_iso(meta):
    """The row's iso for naming, or "" for English / legacy / unreadable.

    `video_engine._language_marks` adds `_<iso>` to the episode id for a
    non-English row and NOTHING for English, so every reader of the identity
    tail has to derive the iso the same way or they disagree about where the
    tail starts. Both the mux (which WRITES the name) and the ledger (which
    VALIDATES it) call this, for the same reason the sanitiser lives here.

    Takes it from the ROW, never from a `meta` key: two plausible keys were
    tried while writing this -- `episode_language_label` and
    `episode_language_iso` -- and NEITHER exists. `meta["episode_language"]`
    holds the iso already on a real ledger, and `iso_from_meta` is the reader
    `cast_lock` uses. Never raises: a name may not cost an episode.
    """
    try:
        try:
            from .. import _otr_episode_languages as _EPLANG
        except ImportError:  # pragma: no cover -- flat import harnesses
            import _otr_episode_languages as _EPLANG  # type: ignore
        iso = _EPLANG.iso_from_meta(meta if isinstance(meta, dict) else {})
        return "" if iso == _EPLANG.ENGLISH_ISO else str(iso or "")
    except Exception:  # noqa: BLE001 -- a name never costs an episode
        return ""


def obs_stem(gloss, episode_id, iso=""):
    """`<gloss><identity tail>` -- the published stem, or "" to keep today's.

    Returns "" whenever the gloss is missing or the id does not end in the tail
    the ledger expects, so the caller falls back to the name it already writes.
    Never guesses.
    """
    gloss = str(gloss or "").strip()
    if not gloss:
        return ""
    tail = identity_tail(episode_id, iso)
    if not tail:
        return ""
    return sanitise_name_part(gloss + tail, "episode")


__all__ = [
    "FILENAME_FORBIDDEN", "OBS_NAME_MAX", "WINDOWS_RESERVED",
    "episode_iso", "identity_tail", "obs_stem", "sanitise_name_part", "trim_to_bytes",
    "validate_gloss",
]
