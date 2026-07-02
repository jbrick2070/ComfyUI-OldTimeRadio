"""
test_speaker_role.py
====================

Coverage for the speaker_role taxonomy introduced 2026-04-30 as
part of the ROADMAP P0 architecture lock, REWRITTEN 2026-07-01 for the
rip-sfx-broll cleanbreak: the "sfx" role is GONE and the silent
default-to-character fallbacks in resolve_speaker_role /
stamp_default_role are now LOUD ValueErrors (NO FALLBACKS).
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

    def test_sfx_constant_is_gone(self):
        # rip-sfx-broll (2026-07-01): the constant itself was deleted.
        assert not hasattr(SR, "SPEAKER_ROLE_SFX")

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
# resolve_speaker_role -- LOUD contract (no fallbacks)
# ---------------------------------------------------------------------------

class TestResolveSpeakerRole:
    """Reads a line dict and returns the canonical role string;
    anything else RAISES (rip-sfx-broll NO-FALLBACKS conversion --
    the historical silent default-to-character path is gone)."""

    @pytest.mark.parametrize("role", SR.VALID_SPEAKER_ROLES)
    def test_known_role_round_trips(self, role):
        line = {"text": "...", "speaker_role": role}
        assert SR.resolve_speaker_role(line) == role

    def test_case_insensitive(self):
        for variant in ("ANNOUNCER", "Announcer", "  announcer  ", "AnNoUnCeR"):
            line = {"speaker_role": variant}
            assert SR.resolve_speaker_role(line) == SR.SPEAKER_ROLE_ANNOUNCER

    def test_missing_field_raises(self):
        line = {"text": "Hello", "speaker": "ALICE"}
        with pytest.raises(ValueError):
            SR.resolve_speaker_role(line)

    def test_unknown_role_raises(self):
        with pytest.raises(ValueError):
            SR.resolve_speaker_role({"speaker_role": "narrator-v3-experimental"})

    def test_old_sfx_ledger_rejected_loud(self):
        # The rip's old-ledger gate: speaker_role="sfx" is a hard error.
        with pytest.raises(ValueError) as exc:
            SR.resolve_speaker_role({"line_id": "x00", "speaker_role": "sfx"})
        assert "sfx" in str(exc.value)

    @pytest.mark.parametrize("hostile", [
        None, [], "not a dict", 42, 3.14, True, False, b"bytes",
        {"speaker_role": None},
        {"speaker_role": 42},
        {"speaker_role": []},
        {"speaker_role": ""},
    ])
    def test_hostile_input_raises(self, hostile):
        # NO FALLBACKS: garbage input is a producer bug and raises
        # instead of being silently rendered as a character line.
        with pytest.raises(ValueError):
            SR.resolve_speaker_role(hostile)


# ---------------------------------------------------------------------------
# is_dialogue_role / is_radio_role / is_music_role
# ---------------------------------------------------------------------------

class TestRolePredicates:
    """Routing predicates for speaker_role.

    BUG-LOCAL-129 (2026-05-01) retired the "radio is the visual
    performer" premise. is_radio_role() now always returns False
    (defense-in-depth dead predicate). Routing is split into:
      - is_dialogue_role()   -> True for character only (HuMo with
                                portrait)
      - is_never_humo_role() -> True for announcer + music_* (skip
                                HuMo; radio-console coverage)
    """

    def test_character_is_dialogue_only(self):
        assert SR.is_dialogue_role(SR.SPEAKER_ROLE_CHARACTER) is True
        assert SR.is_radio_role(SR.SPEAKER_ROLE_CHARACTER) is False
        assert SR.is_never_humo_role(SR.SPEAKER_ROLE_CHARACTER) is False

    @pytest.mark.parametrize("role", SR.VALID_SPEAKER_ROLES)
    def test_is_radio_role_always_false_post_bug129(self, role):
        # BUG-LOCAL-129 fix: _RADIO_ROLES is empty. Predicate kept
        # as defense-in-depth -- if this assertion ever fails it
        # means the regression has been reintroduced.
        assert SR.is_radio_role(role) is False, (
            f"BUG-LOCAL-129 regression: role {role!r} routes back to "
            f"radio still as HuMo ref_image. _RADIO_ROLES has been "
            f"re-populated in _otr_speaker_role.py. Revert."
        )

    @pytest.mark.parametrize("role", [
        SR.SPEAKER_ROLE_ANNOUNCER,
        SR.SPEAKER_ROLE_MUSIC_OPEN,
        SR.SPEAKER_ROLE_MUSIC_CLOSE,
        SR.SPEAKER_ROLE_MUSIC_INTER,
    ])
    def test_never_humo_roles_include_announcer_and_music(self, role):
        # BUG-LOCAL-134 architecture (2026-05-01): announcer joins
        # music_* in never-humo. The radio IS the host -- announcer
        # voice plays under the animated radio, no face render.
        assert SR.is_never_humo_role(role) is True

    def test_only_character_can_use_humo(self):
        assert SR.is_never_humo_role(SR.SPEAKER_ROLE_CHARACTER) is False

    @pytest.mark.parametrize("role", [
        SR.SPEAKER_ROLE_MUSIC_OPEN,
        SR.SPEAKER_ROLE_MUSIC_CLOSE,
        SR.SPEAKER_ROLE_MUSIC_INTER,
    ])
    def test_music_predicate_covers_only_music_tiers(self, role):
        assert SR.is_music_role(role) is True

    @pytest.mark.parametrize("role", [
        SR.SPEAKER_ROLE_CHARACTER,
        SR.SPEAKER_ROLE_ANNOUNCER,
    ])
    def test_non_music_roles_arent_music(self, role):
        assert SR.is_music_role(role) is False

    def test_predicates_partition_valid_set_post_bug129(self):
        # Routing partition:
        #   character  -> dialogue (HuMo)
        #   announcer + music_* -> never-humo (radio-console coverage)
        for role in SR.VALID_SPEAKER_ROLES:
            d = SR.is_dialogue_role(role)
            n = SR.is_never_humo_role(role)
            # No role can be both dialogue and never-humo.
            assert not (d and n), (
                f"Role {role!r}: dialogue={d}, never_humo={n} -- "
                f"contradictory."
            )
            # Every valid role is exactly one of the two post-rip.
            assert d or n, f"Role {role!r} unrouted"
            # is_radio_role() is always False post-BUG-129.
            assert SR.is_radio_role(role) is False


# ---------------------------------------------------------------------------
# stamp_default_role -- validate-or-raise (no backfill)
# ---------------------------------------------------------------------------

class TestStampDefaultRole:
    """rip-sfx-broll: the legacy backfill (silently stamping
    'character' over missing/invalid roles) is GONE. The helper now
    validates + normalizes, raising on anything unmapped."""

    def test_missing_field_raises(self):
        line = {"text": "Hello", "speaker": "ALICE"}
        with pytest.raises(ValueError):
            SR.stamp_default_role(line)

    def test_existing_valid_role_preserved(self):
        line = {"speaker_role": "announcer"}
        SR.stamp_default_role(line)
        assert line["speaker_role"] == "announcer"

    def test_valid_role_normalized_in_place(self):
        line = {"speaker_role": "  MUSIC_OPEN  "}
        SR.stamp_default_role(line)
        assert line["speaker_role"] == SR.SPEAKER_ROLE_MUSIC_OPEN

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError):
            SR.stamp_default_role({"speaker_role": "narrator-experimental"})

    def test_sfx_role_raises(self):
        with pytest.raises(ValueError):
            SR.stamp_default_role({"speaker_role": "sfx"})

    def test_returns_same_dict_for_chaining(self):
        line = {"speaker_role": "character"}
        out = SR.stamp_default_role(line)
        assert out is line

    def test_non_dict_raises_type_error(self):
        for hostile in (None, "string", 42, [], (1, 2)):
            with pytest.raises(TypeError):
                SR.stamp_default_role(hostile)

    def test_iterating_full_ledger_of_valid_rows(self):
        # Realistic call shape: walk ledger.lines[] and validate.
        lines = [
            {"speaker_role": "announcer"},
            {"speaker_role": "character"},
            {"speaker_role": "music_open"},
        ]
        for ln in lines:
            SR.stamp_default_role(ln)
        roles = [ln["speaker_role"] for ln in lines]
        assert roles == [
            SR.SPEAKER_ROLE_ANNOUNCER,
            SR.SPEAKER_ROLE_CHARACTER,
            SR.SPEAKER_ROLE_MUSIC_OPEN,
        ]
