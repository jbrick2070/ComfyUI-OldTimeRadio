"""
test_speaker_role.py
====================

Coverage for the speaker_role taxonomy introduced 2026-04-30 as
part of the ROADMAP P0 architecture lock (100% HuMo coverage).
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
        assert SR.SPEAKER_ROLE_SFX == "sfx"

    def test_valid_roles_set_membership(self):
        # All six constants must appear in VALID_SPEAKER_ROLES.
        for const_value in (
            SR.SPEAKER_ROLE_CHARACTER,
            SR.SPEAKER_ROLE_ANNOUNCER,
            SR.SPEAKER_ROLE_MUSIC_OPEN,
            SR.SPEAKER_ROLE_MUSIC_CLOSE,
            SR.SPEAKER_ROLE_MUSIC_INTER,
            SR.SPEAKER_ROLE_SFX,
        ):
            assert const_value in SR.VALID_SPEAKER_ROLES

    def test_valid_roles_count_matches_constants(self):
        # Guards against silently dropping a constant from the
        # canonical tuple.
        assert len(SR.VALID_SPEAKER_ROLES) == 6

    def test_canonical_order_first_is_character(self):
        # character is the most common case and the safe default;
        # keep it first so iteration / display starts there.
        assert SR.VALID_SPEAKER_ROLES[0] == SR.SPEAKER_ROLE_CHARACTER


# ---------------------------------------------------------------------------
# resolve_speaker_role
# ---------------------------------------------------------------------------

class TestResolveSpeakerRole:
    """Reads a line dict and returns the canonical role string,
    defaulting to character on any ambiguity."""

    @pytest.mark.parametrize("role", SR.VALID_SPEAKER_ROLES)
    def test_known_role_round_trips(self, role):
        line = {"text": "...", "speaker_role": role}
        assert SR.resolve_speaker_role(line) == role

    def test_case_insensitive(self):
        for variant in ("ANNOUNCER", "Announcer", "  announcer  ", "AnNoUnCeR"):
            line = {"speaker_role": variant}
            assert SR.resolve_speaker_role(line) == SR.SPEAKER_ROLE_ANNOUNCER

    def test_missing_field_defaults_to_character(self):
        # Legacy lines (pre-2026-04-30) won't have speaker_role.
        # They must default to "character" so existing behavior is
        # preserved.
        line = {"text": "Hello", "speaker": "ALICE"}
        assert SR.resolve_speaker_role(line) == SR.SPEAKER_ROLE_CHARACTER

    def test_unknown_role_defaults_to_character(self):
        # Future / typo'd values fall back to "character" so HuMo's
        # portrait resolver kicks in (safer than rendering with the
        # radio still for an ambiguous line).
        line = {"speaker_role": "narrator-v3-experimental"}
        assert SR.resolve_speaker_role(line) == SR.SPEAKER_ROLE_CHARACTER

    @pytest.mark.parametrize("hostile", [
        None, [], "not a dict", 42, 3.14, True, False, b"bytes",
        {"speaker_role": None},
        {"speaker_role": 42},
        {"speaker_role": []},
        {"speaker_role": ""},
    ])
    def test_hostile_input_defaults_to_character(self, hostile):
        # Builder must NEVER raise on garbage input -- a downstream
        # ledger-renderer crash from a single weird line is a worse
        # failure mode than rendering that line with a portrait.
        assert SR.resolve_speaker_role(hostile) == SR.SPEAKER_ROLE_CHARACTER


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
      - is_never_humo_role() -> True for music_*/sfx (skip HuMo;
                                VideoComposite static-fill covers them)
      - announcer            -> not dialogue, not never-humo: routes
                                through portrait chain (HuMo if a
                                portrait resolves, otherwise falls
                                through to VideoComposite static-fill).
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
        SR.SPEAKER_ROLE_MUSIC_OPEN,
        SR.SPEAKER_ROLE_MUSIC_CLOSE,
        SR.SPEAKER_ROLE_MUSIC_INTER,
        SR.SPEAKER_ROLE_SFX,
    ])
    def test_never_humo_roles_are_music_and_sfx(self, role):
        assert SR.is_never_humo_role(role) is True

    @pytest.mark.parametrize("role", [
        SR.SPEAKER_ROLE_CHARACTER,
        SR.SPEAKER_ROLE_ANNOUNCER,
    ])
    def test_character_and_announcer_can_use_humo(self, role):
        # BUG-LOCAL-129b: announcer is intentionally NOT in the
        # never-humo set -- if the LLM emits ANNOUNCER as a cast
        # member with a portrait, HuMo renders it like any character.
        assert SR.is_never_humo_role(role) is False

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
        SR.SPEAKER_ROLE_SFX,
    ])
    def test_non_music_roles_arent_music(self, role):
        assert SR.is_music_role(role) is False

    def test_predicates_partition_valid_set_post_bug129(self):
        # New routing partition (BUG-LOCAL-129b, 2026-05-01):
        #   character  -> dialogue (HuMo)
        #   announcer  -> dialogue-eligible (HuMo if portrait, else
        #                 falls through to VideoComposite)
        #   music_*/sfx-> never-humo (VideoComposite static-fill)
        # Every valid role lands in exactly one of: dialogue (only
        # character), never-humo (music/sfx), or "humo-eligible"
        # (announcer - the residual).
        for role in SR.VALID_SPEAKER_ROLES:
            d = SR.is_dialogue_role(role)
            n = SR.is_never_humo_role(role)
            # No role can be both dialogue and never-humo.
            assert not (d and n), (
                f"Role {role!r}: dialogue={d}, never_humo={n} -- "
                f"contradictory."
            )
            # is_radio_role() is always False post-BUG-129.
            assert SR.is_radio_role(role) is False


# ---------------------------------------------------------------------------
# stamp_default_role
# ---------------------------------------------------------------------------

class TestStampDefaultRole:
    """In-place backfill helper for legacy lines."""

    def test_missing_field_gets_character(self):
        line = {"text": "Hello", "speaker": "ALICE"}
        SR.stamp_default_role(line)
        assert line["speaker_role"] == SR.SPEAKER_ROLE_CHARACTER

    def test_existing_valid_role_preserved(self):
        line = {"speaker_role": "announcer"}
        SR.stamp_default_role(line)
        assert line["speaker_role"] == "announcer"

    def test_existing_invalid_role_overwritten(self):
        # Garbage role gets normalized to "character" (the safe
        # default).
        line = {"speaker_role": "narrator-experimental"}
        SR.stamp_default_role(line)
        assert line["speaker_role"] == SR.SPEAKER_ROLE_CHARACTER

    def test_returns_same_dict_for_chaining(self):
        line = {"text": "Hello"}
        out = SR.stamp_default_role(line)
        assert out is line

    def test_non_dict_raises_type_error(self):
        for hostile in (None, "string", 42, [], (1, 2)):
            with pytest.raises(TypeError):
                SR.stamp_default_role(hostile)

    def test_iterating_full_ledger(self):
        # Realistic call shape: walk ledger.lines[] and backfill.
        lines = [
            {"text": "Greetings, listeners.", "speaker": "ANNOUNCER"},
            {"text": "Hello there.", "speaker": "ALICE"},
            {"speaker_role": "music_open"},
            {"speaker_role": "garbled"},   # gets reset to character
            {},                            # empty -- gets character
        ]
        for ln in lines:
            SR.stamp_default_role(ln)
        roles = [ln["speaker_role"] for ln in lines]
        assert roles == [
            SR.SPEAKER_ROLE_CHARACTER,
            SR.SPEAKER_ROLE_CHARACTER,
            SR.SPEAKER_ROLE_MUSIC_OPEN,
            SR.SPEAKER_ROLE_CHARACTER,
            SR.SPEAKER_ROLE_CHARACTER,
        ]
