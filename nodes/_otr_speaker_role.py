"""
_otr_speaker_role.py
====================

Speaker-role taxonomy for the v2.0-alpha architecture.

**Routing contract (locked 2026-05-01 after BUG-LOCAL-129):**

Every line in ``ledger.lines[]`` carries a ``speaker_role``. Routing
in BatchHumoRender + VideoComposite is now:

    character   -> HuMo, with PASS3 cast portrait resolver
                   (BUG-088 fallback chain) as I2V reference
    announcer   -> HuMo, with the ANNOUNCER cast portrait if the
                   LLM emitted ANNOUNCER as a cast member; otherwise
                   falls through to the VideoComposite static-radio
                   fill path (BUG-129a).
    music_open  -> non-HuMo. VideoComposite generates a deterministic
                   static-radio segment for visual coverage.
    music_close -> same as music_open.
    music_inter -> same as music_open.
    sfx         -> same as music_open (standalone SFX). SFX concurrent
                   with dialogue should NOT have its own ledger.lines[]
                   entry -- it's part of the surrounding character's
                   audio and stays on that character's HuMo clip.

**Why the old "radio is the visual performer" premise was retired:**

BUG-LOCAL-129 (2026-05-01) discovered that HuMo's finetuned weights
will not animate non-face references. Passing the radio still as
HuMo's ``ref_image`` for announcer/music/sfx produced two unrelated
generic faces (l001 + l021 of the 2026-05-01_110019 run) instead of
the radio itself. The architectural premise that "the radio is the
visual performer for everything that isn't dialogue" is incompatible
with HuMo as the renderer. Round-robin consult (gpt-5.4, gemini-3-
pro-preview, mistral-nemotron) + external code review converged on:
HuMo for speaking faces only; everything else through a deterministic
static-video editorial path. See ``docs/2026-05-01-humo-radio-
architecture__*.md`` for full transcripts.

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


# BUG-LOCAL-129 fix (2026-05-01): no role routes to the radio still
# as a HuMo I2V reference any more. HuMo's weights only animate faces;
# passing the radio still produces unconstrained generic-face output
# (BUG-129's two-blonde-women symptom). Roles that previously routed
# here now fall through the portrait chain in BatchHumoRender; if no
# portrait is found the line gets a deterministic static-radio fill
# in VideoComposite (BUG-129a). The empty set is preserved as a
# defense-in-depth signal: if a future commit re-populates this set,
# is_radio_role() flips True and the regression resurfaces visibly.
_RADIO_ROLES: frozenset[str] = frozenset()


# Roles that must NEVER trigger a HuMo render even if a portrait is
# somehow resolvable. music_*/sfx-standalone visual coverage is the
# job of the VideoComposite static-fill path (BUG-129a). Defense in
# depth so a future regression that adds these to the cast portrait
# pool can't reintroduce the problem.
_NEVER_HUMO_ROLES = frozenset({
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
    """Always returns ``False`` post-BUG-LOCAL-129 (2026-05-01).

    Historical contract: True for announcer + music_* + sfx, which
    used the radio still PNG as HuMo's I2V reference. Retired because
    HuMo's weights only animate faces -- passing a non-face produced
    unconstrained generic-face output (BUG-129).

    The predicate is preserved (rather than deleted) as a defense-in-
    depth flag: callers that historically routed on this branch
    (BatchHumoRender) keep the dead code as documentation of the
    failed experiment, and any test that asserts ``is_radio_role(r)``
    is True will fail loudly if a future commit re-populates
    :data:`_RADIO_ROLES`.
    """
    return role in _RADIO_ROLES


def is_never_humo_role(role: str) -> bool:
    """True iff the role must NEVER dispatch a HuMo render.

    Currently covers ``music_open``, ``music_close``, ``music_inter``,
    ``sfx``. These roles get visual coverage via VideoComposite's
    deterministic static-radio fill path (BUG-129a). Even if a
    portrait somehow resolves for one of these speakers (e.g., an
    SFX line gets the same speaker name as a real character), the
    dispatch must short-circuit before HuMo is invoked.

    ``announcer`` is intentionally NOT in this set: announcer lines
    SHOULD render via HuMo if the LLM emits ANNOUNCER as a cast
    member with a portrait; if no portrait resolves they fall through
    to the VideoComposite static-fill path naturally.
    """
    return role in _NEVER_HUMO_ROLES


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
    "is_never_humo_role",
    "stamp_default_role",
]
