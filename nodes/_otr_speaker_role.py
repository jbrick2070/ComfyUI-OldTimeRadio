"""
_otr_speaker_role.py
====================

Speaker-role taxonomy for the v2.0-alpha architecture.

**Routing contract (locked 2026-05-01 after BUG-LOCAL-129; sfx role RIPPED
2026-07-01 -- see docs/2026-07-01-rip-sfx-broll/BUILD_PLAN.md):**

Every line in ``ledger.lines[]`` carries a ``speaker_role``. Routing:

    character   -> HuMo, with PASS3 cast portrait resolver
                   (BUG-088 fallback chain) as I2V reference
    announcer   -> never HuMo ("the radio IS the host"): visual coverage
                   from the animated radio console path.
    music_open  -> non-HuMo. Deterministic radio-console visual coverage.
    music_close -> same as music_open.
    music_inter -> same as music_open.

The historical ``sfx`` role was removed 2026-07-01 (kibitz r1+r2 grounded:
it produced ZERO script/audio/video content -- the writer nudge never fired,
TTS never saw it, and its SceneSequencer overlay inputs were unwired). A
ledger that still carries ``speaker_role: "sfx"`` is an OLD ledger and is
rejected LOUD by :func:`resolve_speaker_role`, the ledger-freeze per-line
invariant, and the SceneSequencer dispatch. NO FALLBACKS.

**Why the old "radio is the visual performer" premise was retired:**

BUG-LOCAL-129 (2026-05-01) discovered that HuMo's finetuned weights
will not animate non-face references. Passing the radio still as
HuMo's ``ref_image`` for announcer/music produced two unrelated
generic faces (l001 + l021 of the 2026-05-01_110019 run) instead of
the radio itself. HuMo is for speaking faces only; everything else
goes through the deterministic radio-console editorial path. See
``docs/2026-05-01-humo-radio-architecture__*.md`` for transcripts.

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


# All valid roles, in canonical order.  Used by validators and tests.
# 2026-07-01: "sfx" REMOVED (rip-sfx-broll). Old sfx ledgers fail loud.
VALID_SPEAKER_ROLES = (
    SPEAKER_ROLE_CHARACTER,
    SPEAKER_ROLE_ANNOUNCER,
    SPEAKER_ROLE_MUSIC_OPEN,
    SPEAKER_ROLE_MUSIC_CLOSE,
    SPEAKER_ROLE_MUSIC_INTER,
)


# BUG-LOCAL-129 fix (2026-05-01): no role routes to the radio still
# as a HuMo I2V reference any more. HuMo's weights only animate faces;
# passing the radio still produces unconstrained generic-face output
# (BUG-129's two-blonde-women symptom). Roles that previously routed
# here now fall through the portrait chain; if no portrait is found
# the line gets a deterministic static-radio fill (BUG-129a). The
# empty set is preserved as a defense-in-depth signal: if a future
# commit re-populates this set, is_radio_role() flips True and the
# regression resurfaces visibly.
_RADIO_ROLES: frozenset[str] = frozenset()


# Roles that must NEVER trigger a HuMo render. These get visual
# coverage from the animated radio console (post-2026-05-01
# architecture, BUG-LOCAL-134). Aesthetic intent: classic 1940s
# old-time radio -- the listener doesn't see the announcer's face,
# sees the radio set. Only character DIALOGUE gets a face (HuMo
# lip-sync).
#
# Jeffrey 2026-05-01: announcer added to this set alongside music.
# "the radio IS the host" -- announcer voice plays under animated radio,
# no human face needed for the host. Closes the BUG-131 announcer-
# cast-portrait dependency entirely.
_NEVER_HUMO_ROLES = frozenset({
    SPEAKER_ROLE_ANNOUNCER,
    SPEAKER_ROLE_MUSIC_OPEN,
    SPEAKER_ROLE_MUSIC_CLOSE,
    SPEAKER_ROLE_MUSIC_INTER,
})


# Music-tier roles, in case a downstream consumer wants the music
# vs. announcer split (e.g. for separate VRAM pipelines or
# different ref-image families).
_MUSIC_ROLES = frozenset({
    SPEAKER_ROLE_MUSIC_OPEN,
    SPEAKER_ROLE_MUSIC_CLOSE,
    SPEAKER_ROLE_MUSIC_INTER,
})


def resolve_speaker_role(line: Any) -> str:
    """Return the canonical ``speaker_role`` for a ledger line, or RAISE.

    NO FALLBACKS (rip-sfx-broll, 2026-07-01 -- the silent
    default-to-character path was removed; repo grep confirmed zero
    production callers at conversion time, so no production path relied
    on the old fallback):

      - ``line`` must be a ``Mapping`` carrying a string ``speaker_role``
        whose normalized value is in :data:`VALID_SPEAKER_ROLES`;
        the normalized value is returned.
      - Anything else -- a non-mapping, a missing field, a non-string,
        or an unknown value (including the retired ``"sfx"``) -- raises
        ``ValueError``. An old ledger with ``speaker_role: "sfx"``
        fails LOUD here.
    """
    if not isinstance(line, Mapping):
        raise ValueError(
            f"resolve_speaker_role: line must be a mapping with a "
            f"'speaker_role' field, got {type(line).__name__}"
        )
    raw = line.get("speaker_role")
    if not isinstance(raw, str):
        raise ValueError(
            f"resolve_speaker_role: line "
            f"{str(line.get('line_id') or line.get('beat_id') or '?')!r} "
            f"has missing/non-string speaker_role ({raw!r}); valid roles: "
            f"{VALID_SPEAKER_ROLES}"
        )
    norm = raw.strip().lower()
    if norm in VALID_SPEAKER_ROLES:
        return norm
    raise ValueError(
        f"resolve_speaker_role: line "
        f"{str(line.get('line_id') or line.get('beat_id') or '?')!r} "
        f"carries unknown speaker_role {raw!r} (valid: "
        f"{VALID_SPEAKER_ROLES}). The 'sfx' role was removed 2026-07-01 "
        f"(rip-sfx-broll) -- an old sfx ledger must be regenerated."
    )


def is_dialogue_role(role: str) -> bool:
    """True iff the role drives a character portrait HuMo render.

    Currently only ``character``; future expansion (e.g. multiple
    dialogue subroles) would extend this set.
    """
    return role == SPEAKER_ROLE_CHARACTER


def is_radio_role(role: str) -> bool:
    """Always returns ``False`` post-BUG-LOCAL-129 (2026-05-01).

    Historical contract: True for announcer + music_*, which used the
    radio still PNG as HuMo's I2V reference. Retired because HuMo's
    weights only animate faces -- passing a non-face produced
    unconstrained generic-face output (BUG-129).

    The predicate is preserved (rather than deleted) as a defense-in-
    depth flag: any test that asserts ``is_radio_role(r)`` is True will
    fail loudly if a future commit re-populates :data:`_RADIO_ROLES`.
    """
    return role in _RADIO_ROLES


def is_never_humo_role(role: str) -> bool:
    """True iff the role must NEVER dispatch a HuMo render.

    Covers ``announcer``, ``music_open``, ``music_close``,
    ``music_inter``. These roles get visual coverage from the animated
    radio console (BUG-LOCAL-134 architecture, locked 2026-05-01). Only
    ``character`` dispatches HuMo for dialogue lip-sync; everything
    else is "the radio is the performer."

    Even if a portrait somehow resolves for one of these speakers,
    the dispatch must short-circuit before HuMo is invoked.

    Pre-2026-05-01: ``announcer`` was NOT in this set -- it was
    expected to render via HuMo with a host portrait. That changed
    with the LTX-radio routing decision: the announcer voice plays
    under the animated radio set, no human face for the host. Closes
    the BUG-131 dependency on having an ANNOUNCER cast member with
    a portrait.
    """
    return role in _NEVER_HUMO_ROLES


def is_music_role(role: str) -> bool:
    """True iff the role is one of the music tiers (open/close/inter).

    Used by consumers that want music-vs-other-radio behavior splits
    (e.g. different ledger fields, different render budgets).
    """
    return role in _MUSIC_ROLES


def stamp_default_role(line: Dict[str, Any]) -> Dict[str, Any]:
    """Validate ``line``'s ``speaker_role`` in place; RAISE if bad.

    NO FALLBACKS (rip-sfx-broll, 2026-07-01): the historical backfill
    behavior (silently stamping ``character`` over a missing or invalid
    role -- including on lines that had NO speaker_role at all) was
    removed. Every producer stamps a valid role at init; a line that
    reaches this helper without one is a bug upstream, not a legacy
    shape to repair. Repo grep at conversion time confirmed zero
    production callers.

    Raises ``TypeError`` if ``line`` is not a dict (this helper
    expects a mutable row) and ``ValueError`` on a missing/invalid
    ``speaker_role``. Returns the same dict for chaining.
    """
    if not isinstance(line, dict):
        raise TypeError(
            f"stamp_default_role expects a dict, got {type(line).__name__}"
        )
    raw = line.get("speaker_role")
    if not isinstance(raw, str) or \
            raw.strip().lower() not in VALID_SPEAKER_ROLES:
        raise ValueError(
            f"stamp_default_role: line "
            f"{str(line.get('line_id') or line.get('beat_id') or '?')!r} "
            f"has missing/invalid speaker_role ({raw!r}); valid: "
            f"{VALID_SPEAKER_ROLES}. NO FALLBACKS -- fix the producer."
        )
    line["speaker_role"] = raw.strip().lower()
    return line


__all__ = [
    "SPEAKER_ROLE_CHARACTER",
    "SPEAKER_ROLE_ANNOUNCER",
    "SPEAKER_ROLE_MUSIC_OPEN",
    "SPEAKER_ROLE_MUSIC_CLOSE",
    "SPEAKER_ROLE_MUSIC_INTER",
    "VALID_SPEAKER_ROLES",
    "resolve_speaker_role",
    "is_dialogue_role",
    "is_radio_role",
    "is_music_role",
    "is_never_humo_role",
    "stamp_default_role",
]
