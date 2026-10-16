"""
_otr_speaker_role.py
====================

Speaker-role taxonomy for the v2.0-alpha architecture.

**Routing contract (locked 2026-05-01 after BUG-LOCAL-129; revised
2026-10-15 -- the picked lane is the lane):**

Every line in ``ledger.lines[]`` carries a ``speaker_role``. Each role
renders on the video engine picked for it -- there is no role-based engine
swap. What a role changes is WHO appears:

    character   -> a cast member; on HuMo, the PASS3 cast portrait
                   (BUG-088 fallback chain) is the I2V reference.
    announcer   -> "the radio IS the host": never a human face. On HuMo
                   it is the brief-driven radio-with-a-face host.
    music_open  -> same as announcer.
    music_close -> same as announcer.
    music_inter -> same as announcer.

A ledger that carries ``speaker_role: "sfx"`` is an OLD ledger and is
rejected LOUD by the ledger-freeze per-line invariant and the SceneSequencer
dispatch. NO FALLBACKS.

**Why the HuMo host is a radio WITH a face:**

HuMo's finetuned weights will not animate non-face references
(BUG-LOCAL-129): a plain radio still as HuMo's ``ref_image`` produced an
unrelated generic person. So a HuMo bookend is fed the anthropomorphic
``console_face`` radio (dial-eyes, needle-fan mouth) -- a face HuMo can
animate that is still the radio.

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


# The radio-host roles: they never show a person. "The radio IS the host"
# (Jeffrey 2026-05-01: announcer added alongside music; closes the BUG-131
# announcer-cast-portrait dependency). On HuMo these render the
# radio-with-a-face host; on any other lane, that lane's radio picture.
_RADIO_HOST_ROLES = frozenset({
    SPEAKER_ROLE_ANNOUNCER,
    SPEAKER_ROLE_MUSIC_OPEN,
    SPEAKER_ROLE_MUSIC_CLOSE,
    SPEAKER_ROLE_MUSIC_INTER,
})


def is_radio_host_role(role: str) -> bool:
    """True iff the role is a radio-host role (``announcer``,
    ``music_open``, ``music_close``, ``music_inter``): it never shows a
    person, and on HuMo it animates the radio-with-a-face host instead of a
    cast portrait. Only ``character`` shows a cast member. The role never
    changes which engine renders the beat."""
    return role in _RADIO_HOST_ROLES


__all__ = [
    "SPEAKER_ROLE_CHARACTER",
    "SPEAKER_ROLE_ANNOUNCER",
    "SPEAKER_ROLE_MUSIC_OPEN",
    "SPEAKER_ROLE_MUSIC_CLOSE",
    "SPEAKER_ROLE_MUSIC_INTER",
    "VALID_SPEAKER_ROLES",
    "is_radio_host_role",
]
