"""
_otr_speaker_role.py
====================

Speaker-role taxonomy for the v2.0-alpha "100% HuMo coverage" lock
(ROADMAP P0, 2026-04-30).  Every line in ``ledger.lines[]`` carries
a ``speaker_role`` that tells BatchHumoRender which I2V reference
image to use:

    character   -> existing PASS3 cast portrait resolver
                   (BUG-088 fallback chain stays intact)
    announcer   -> ledger.radio_bookend_path  (the radio still)
    music_open  -> ledger.radio_bookend_path
    music_close -> ledger.radio_bookend_path
    music_inter -> ledger.radio_bookend_path
    sfx         -> ledger.radio_bookend_path

The radio is the visual performer for everything that isn't a
dialogue line: announcer voice, opening/closing/interstitial music
windows, and standalone SFX cues.  Per Jeffrey 2026-04-30: people
speaking get people lip-syncing, everything else gets the radio
lip-syncing.

This module is pure stdlib -- no torch, no comfy imports -- so it's
safe to load from tests, scripts, and any node without adding
import-time cost.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


# Canonical role string constants.  Use these instead of literals
# wherever you need to test for / set a role; the linter will catch
# typos in constants but not in string literals.
SPEAKER_ROLE_CHARACTER = "character"
SPEAKER_ROLE_ANNOUNCER = "announcer"
SPEAKER_ROLE_MUSIC_OPEN = "music_open"
SPEAKER_ROLE_MUSIC_CLOSE = "music_close"
SPEAKER_ROLE_MUSIC_INTER = "music_inter"
SPEAKER_ROLE_SFX = "sfx"


# All valid roles, in canonical order.  Used by validators and tests.
VALID_SPEAKER_ROLES = (
    SPEAKER_ROLE_CHARACTER,
    SPEAKER_ROLE_ANNOUNCER,
    SPEAKER_ROLE_MUSIC_OPEN,
    SPEAKER_ROLE_MUSIC_CLOSE,
    SPEAKER_ROLE_MUSIC_INTER,
    SPEAKER_ROLE_SFX,
)


# All roles whose I2V reference is the radio still.  Lookup table
# kept separate from the constant tuple so future role additions
# (e.g. a new music tier) can be classified without changing the
# canonical role list.
_RADIO_ROLES = frozenset({
    SPEAKER_ROLE_ANNOUNCER,
    SPEAKER_ROLE_MUSIC_OPEN,
    SPEAKER_ROLE_MUSIC_CLOSE,
    SPEAKER_ROLE_MUSIC_INTER,
    SPEAKER_ROLE_SFX,
})


# Music-tier roles, in case a downstream consumer wants the music
# vs. announcer vs. sfx split (e.g. for separate VRAM pipelines or
# different ref-image families).  Currently all three route to the
# same radio still, but exposing the split keeps the door open.
_MUSIC_ROLES = frozenset({
    SPEAKER_ROLE_MUSIC_OPEN,
    SPEAKER_ROLE_MUSIC_CLOSE,
    SPEAKER_ROLE_MUSIC_INTER,
})


def resolve_speaker_role(line: Any) -> str:
    """Return the canonical ``speaker_role`` for a ledger line.

    Behavior:
      - If ``line`` is a ``Mapping`` with key ``speaker_role`` set to
        a known role string, return it (case-insensitive, stripped).
      - If ``line`` is missing the field, has a non-string value, or
        the value isn't in :data:`VALID_SPEAKER_ROLES`, default to
        :data:`SPEAKER_ROLE_CHARACTER` -- the safest fallback because
        unknown lines render with the existing portrait resolver,
        same as legacy behavior pre-2026-04-30.
      - Hostile inputs (``None``, lists, scalars, garbled types)
        also default to ``character`` rather than raising.
    """
    if not isinstance(line, Mapping):
        return SPEAKER_ROLE_CHARACTER
    raw = line.get("speaker_role")
    if not isinstance(raw, str):
        return SPEAKER_ROLE_CHARACTER
    norm = raw.strip().lower()
    if norm in VALID_SPEAKER_ROLES:
        return norm
    return SPEAKER_ROLE_CHARACTER


def is_dialogue_role(role: str) -> bool:
    """True iff the role drives a character portrait HuMo render.

    Currently only ``character``; future expansion (e.g. multiple
    dialogue subroles) would extend this set.
    """
    return role == SPEAKER_ROLE_CHARACTER


def is_radio_role(role: str) -> bool:
    """True iff the role drives a radio still HuMo render.

    Covers announcer + all music_* + sfx.  The complement of
    :func:`is_dialogue_role` for valid roles, but kept as an
    explicit predicate so future additions (e.g. a hypothetical
    ``narrator`` role driving a separate visual) can be classified
    independently of the dialogue/radio split.
    """
    return role in _RADIO_ROLES


def is_music_role(role: str) -> bool:
    """True iff the role is one of the music tiers (open/close/inter).

    Used by consumers that want music-vs-other-radio behavior splits
    (e.g. different ledger fields, different render budgets).
    """
    return role in _MUSIC_ROLES


def stamp_default_role(line: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate ``line`` in place to set ``speaker_role`` to
    ``character`` if missing.

    Used by ScriptParser-style call sites that want to backfill
    legacy lines.  Returns the same dict for chaining.

    Raises ``TypeError`` if ``line`` is not a dict (this helper
    expects to mutate, unlike :func:`resolve_speaker_role` which
    is read-only and hostile-input safe).
    """
    if not isinstance(line, dict):
        raise TypeError(
            f"stamp_default_role expects a dict, got {type(line).__name__}"
        )
    if not isinstance(line.get("speaker_role"), str) or \
            line["speaker_role"].strip().lower() not in VALID_SPEAKER_ROLES:
        line["speaker_role"] = SPEAKER_ROLE_CHARACTER
    return line


__all__ = [
    "SPEAKER_ROLE_CHARACTER",
    "SPEAKER_ROLE_ANNOUNCER",
    "SPEAKER_ROLE_MUSIC_OPEN",
    "SPEAKER_ROLE_MUSIC_CLOSE",
    "SPEAKER_ROLE_MUSIC_INTER",
    "SPEAKER_ROLE_SFX",
    "VALID_SPEAKER_ROLES",
    "resolve_speaker_role",
    "is_dialogue_role",
    "is_radio_role",
    "is_music_role",
    "stamp_default_role",
]
