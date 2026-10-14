"""
_otr_speaker_role.py
====================

Speaker-role taxonomy for the v2.0-alpha architecture.

**Routing contract (locked 2026-05-01 after BUG-LOCAL-129):**

Every line in ``ledger.lines[]`` carries a ``speaker_role``. Routing:

    character   -> HuMo, with PASS3 cast portrait resolver
                   (BUG-088 fallback chain) as I2V reference
    announcer   -> never HuMo ("the radio IS the host"): visual coverage
                   from the animated radio console path.
    music_open  -> non-HuMo. Deterministic radio-console visual coverage.
    music_close -> same as music_open.
    music_inter -> same as music_open.

A ledger that carries ``speaker_role: "sfx"`` is an OLD ledger and is
rejected LOUD by the ledger-freeze per-line invariant and the SceneSequencer
dispatch. NO FALLBACKS.

**Why the radio is never HuMo's reference:**

HuMo's finetuned weights will not animate non-face references
(BUG-LOCAL-129). Passing the radio still as HuMo's ``ref_image`` for
announcer/music produces unrelated generic faces instead of the radio
itself. HuMo is for speaking faces only; everything else goes through the
deterministic radio-console editorial path.

This module is pure stdlib -- no torch, no comfy imports -- so it's
safe to load from tests, scripts, and any node without adding
import-time cost.
"""

from __future__ import annotations


# Canonical role string constants.  Use these instead of literals
# wherever you need to test for / set a role; the linter will catch
# typos in constants but not in string literals.
SPEAKER_ROLE_CHARACTER = "character"
SPEAKER_ROLE_ANNOUNCER = "announcer"
SPEAKER_ROLE_MUSIC_OPEN = "music_open"
SPEAKER_ROLE_MUSIC_CLOSE = "music_close"
SPEAKER_ROLE_MUSIC_INTER = "music_inter"


# All valid roles, in canonical order.  Used by validators and tests.
# A ledger carrying "sfx" is rejected loud.
VALID_SPEAKER_ROLES = (
    SPEAKER_ROLE_CHARACTER,
    SPEAKER_ROLE_ANNOUNCER,
    SPEAKER_ROLE_MUSIC_OPEN,
    SPEAKER_ROLE_MUSIC_CLOSE,
    SPEAKER_ROLE_MUSIC_INTER,
)


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


__all__ = [
    "SPEAKER_ROLE_CHARACTER",
    "SPEAKER_ROLE_ANNOUNCER",
    "SPEAKER_ROLE_MUSIC_OPEN",
    "SPEAKER_ROLE_MUSIC_CLOSE",
    "SPEAKER_ROLE_MUSIC_INTER",
    "VALID_SPEAKER_ROLES",
    "is_never_humo_role",
]
