"""
test_speaker_role.py
====================

Coverage for the speaker_role taxonomy introduced 2026-04-30 as
part of the ROADMAP P0 architecture lock, REWRITTEN 2026-07-01 for the
rip-sfx-broll cleanbreak: the "sfx" role is GONE.
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_NODES_DIR = os.path.join(_REPO_ROOT, "nodes")
for p in (_REPO_ROOT, _NODES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import _otr_speaker_role as SR  # noqa: E402  -- after sys.path tweak


# ---------------------------------------------------------------------------
# Constants + canonical set
# ---------------------------------------------------------------------------

class TestRoleConstants:
    """Constants are the single source of truth -- linter catches
    typos in constants, not in string literals."""

    def test_canonical_role_strings(self):
        assert SR.SPEAKER_ROLE_CHARACTER == "character"
        assert SR.SPEAKER_ROLE_ANNOUNCER == "announcer"
        assert SR.SPEAKER_ROLE_MUSIC_OPEN == "music_open"
        assert SR.SPEAKER_ROLE_MUSIC_CLOSE == "music_close"
        assert SR.SPEAKER_ROLE_MUSIC_INTER == "music_inter"

    def test_valid_roles_set_membership(self):
        # All five constants must appear in VALID_SPEAKER_ROLES.
        for const_value in (
            SR.SPEAKER_ROLE_CHARACTER,
            SR.SPEAKER_ROLE_ANNOUNCER,
            SR.SPEAKER_ROLE_MUSIC_OPEN,
            SR.SPEAKER_ROLE_MUSIC_CLOSE,
            SR.SPEAKER_ROLE_MUSIC_INTER,
        ):
            assert const_value in SR.VALID_SPEAKER_ROLES

    def test_valid_roles_count_matches_constants(self):
        # Guards against silently dropping OR re-adding a constant
        # to the canonical tuple (sfx must never come back).
        assert len(SR.VALID_SPEAKER_ROLES) == 5
        assert "sfx" not in SR.VALID_SPEAKER_ROLES

    def test_canonical_order_first_is_character(self):
        # character is the most common case; keep it first so
        # iteration / display starts there.
        assert SR.VALID_SPEAKER_ROLES[0] == SR.SPEAKER_ROLE_CHARACTER


# ---------------------------------------------------------------------------
# is_radio_host_role
# ---------------------------------------------------------------------------

class TestRadioHostRoles:
    """Announcer and the music tiers are the RADIO-HOST roles: the radio is the
    host (BUG-LOCAL-134, 2026-05-01). Character dialogue is not. This table is
    only a role classifier -- it never rewrites a picked engine (operator
    2026-10-15: the picked lane is the lane)."""

    @pytest.mark.parametrize("role", [
        SR.SPEAKER_ROLE_ANNOUNCER,
        SR.SPEAKER_ROLE_MUSIC_OPEN,
        SR.SPEAKER_ROLE_MUSIC_CLOSE,
        SR.SPEAKER_ROLE_MUSIC_INTER,
    ])
    def test_radio_host_roles_include_announcer_and_music(self, role):
        assert SR.is_radio_host_role(role) is True

    def test_only_character_is_not_a_radio_host_role(self):
        assert SR.is_radio_host_role(SR.SPEAKER_ROLE_CHARACTER) is False
        non_host = [r for r in SR.VALID_SPEAKER_ROLES
                    if not SR.is_radio_host_role(r)]
        assert non_host == [SR.SPEAKER_ROLE_CHARACTER]
