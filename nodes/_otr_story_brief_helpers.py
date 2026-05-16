"""nodes/_otr_story_brief_helpers.py -- central consumer helpers (C5b).

Visual + audio consumers do NOT parse the brief prose themselves. They
call one of these five helpers, each shaped to the consumer's prompt
budget and narrative concern:

  get_story_brief_full     -> full brief prose, "" if absent or failed
  get_story_brief_ltx      -> sentence-boundary-trimmed fragment for LTX
  get_story_brief_lighting -> lighting + atmosphere terms, comma joined
  get_story_brief_music_mood -> list[str] of in-vocab mood terms
  get_story_brief_status   -> 'ok' | 'failed' | 'absent'

One helper per consumer shape. The alternative was N slightly-different
bad implementations across N consumer files (refinement section 5).

Module is PURE: no I/O, no GPU, no ComfyUI imports, no MusicGen import.
The dependency direction is consumer -> helper (e.g. C5g wires
`nodes/musicgen_theme.py` to import `get_story_brief_music_mood`),
NEVER the reverse. The `test_get_music_mood_no_musicgen_import` test
in `tests/test_story_brief_helpers_c5b.py` locks this property via
AST inspection so a future refactor cannot accidentally introduce a
circular import.

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
"""
from __future__ import annotations

from typing import Any


# MusicGen mood vocabulary -- 16 terms that the audio model handles
# well as a prepended mood prefix. Atmosphere terms outside this set
# are filtered out so the MusicGen prompt stays in known-vocabulary
# space (refinement section 6.3).
_MUSIC_MOOD_VOCAB: frozenset[str] = frozenset({
    "tense", "ominous", "melancholic", "hopeful", "urgent", "calm",
    "eerie", "sombre", "playful", "menacing", "wistful", "frantic",
    "reverent", "uneasy", "stoic", "yearning",
})


def _meta(obj: Any) -> dict:
    """Accept either a meta dict OR a parent dict carrying a meta key."""
    if isinstance(obj, dict):
        if "story_brief_status" in obj or "story_brief" in obj:
            return obj
        sub = obj.get("meta")
        if isinstance(sub, dict):
            return sub
    return {}


def get_story_brief_status(meta: Any) -> str:
    """Return 'ok' / 'failed' / 'absent'.

    'absent' is the legacy-ledger value (no story_brief_status key);
    consumers fall through to legacy prompt construction on this path
    per refinement section 8.2. 'failed' surfaces the L-6 sentinel so
    the consumer can log story_brief_status in its render output
    (E-07 pattern).
    """
    m = _meta(meta)
    status = m.get("story_brief_status")
    if status in ("ok", "failed"):
        return status
    return "absent"


def get_story_brief_full(meta: Any) -> str:
    """Full brief prose, empty string if absent or failed.

    Returning "" on non-ok status lets consumers fall through to
    legacy prompt construction without branching on status -- a
    one-liner truthiness check (`if brief: ...`) does the right
    thing in both ok and non-ok cases.
    """
    m = _meta(meta)
    if get_story_brief_status(m) != "ok":
        return ""
    return (m.get("story_brief") or "").strip()


def get_story_brief_ltx(meta: Any, max_chars: int = 90) -> str:
    """Brief fragment safe for LTX motion prompts.

    Per refinement section 6.1: LTX motion budget is 220-240 chars
    total with 80-100 chars for the brief fragment. Default max_chars
    is 90 (centerpoint of the 80-100 window). Trimmed at the nearest
    sentence-end or clause-boundary before max_chars; NEVER mid-word.
    """
    full = get_story_brief_full(meta)
    if not full or len(full) <= max_chars:
        return full

    candidate = full[:max_chars]
    # Prefer sentence-end (. ! ?) then clause boundary (, ; :) then
    # word boundary. All inside the [0:max_chars] window.
    for sep in (". ", "! ", "? ", "; ", ", ", ": ", " "):
        idx = candidate.rfind(sep)
        if idx >= 20:  # avoid trimming so aggressively the brief is empty
            return candidate[: idx + (0 if sep == " " else 1)].rstrip()
    # Fall back: hard-trim and append ellipsis. Never returns mid-word
    # because the " " sep above catches that case.
    return candidate.rstrip()


def get_story_brief_lighting(meta: Any) -> str:
    """lighting + atmosphere terms, comma-joined.

    Per refinement section 6.2: portrait builders want lighting and
    atmosphere terms WITHOUT setting-terms noise (which would push
    the portrait prompt toward env composition). Returns empty
    string when the brief is absent or failed.
    """
    m = _meta(meta)
    if get_story_brief_status(m) != "ok":
        return ""
    terms = m.get("story_brief_terms") or {}
    if not isinstance(terms, dict):
        return ""
    lighting = [str(t).strip() for t in (terms.get("lighting") or []) if str(t).strip()]
    atmosphere = [str(t).strip() for t in (terms.get("atmosphere") or []) if str(t).strip()]
    return ", ".join(lighting + atmosphere)


def get_story_brief_music_mood(meta: Any) -> list[str]:
    """Mood keywords from atmosphere_terms, intersected with the
    MusicGen mood vocabulary.

    Per refinement section 6.3: MusicGen never sees prose. The helper
    extracts in-vocab terms from atmosphere_terms so the MusicGen
    prompt stays in known-vocabulary space. Returns an empty list
    when the brief is absent, failed, or carries no in-vocab atmosphere
    terms -- the caller (wired at C5g per E-12) treats an empty list
    as "fall through to legacy prompt construction" per refinement
    section 8.2.

    Dependency direction: consumer -> helper. This module does NOT
    import musicgen_theme; the consumer (`nodes/musicgen_theme.py`)
    imports THIS helper at C5g. The `test_get_music_mood_no_musicgen_
    import` test locks the direction.
    """
    m = _meta(meta)
    if get_story_brief_status(m) != "ok":
        return []
    terms = m.get("story_brief_terms") or {}
    if not isinstance(terms, dict):
        return []
    atmosphere = terms.get("atmosphere") or []
    if not isinstance(atmosphere, list):
        return []
    # Preserve order, intersect against vocab.
    out: list[str] = []
    for raw in atmosphere:
        t = str(raw).strip().lower()
        if t in _MUSIC_MOOD_VOCAB and t not in out:
            out.append(t)
    return out
