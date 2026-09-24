# -*- coding: utf-8 -*-
"""The recurring-character voice table, and the promise that it scales.

A recurring character is one a story lane writes on purpose and who should sound
like himself across episodes. There is exactly one today. The design claim this
file exists to hold is that a SECOND one is a new key in a dict and nothing
else -- no new branch, no new selector, no per-name special case anywhere
downstream. That claim is cheap to make and easy to quietly break, so the
scalability test below injects a character who does not ship and proves the same
code path stamps him.

WHAT THE TABLE DOES NOT OWN, stated because a voice table is a tempting place to
put all of it: story inclusion, the character's name, his description and his
gender all live elsewhere and are unchanged by anything here.

Headless. No engine, no model, no GPU.
"""
from __future__ import annotations

import json

import pytest

from config import cast_pools as POOLS
from config.cast_pools import (
    RECURRING_CHARACTER_VOICES,
    recurring_character_key,
    recurring_character_voice,
)


# ---------------------------------------------------------------------------
# The shipped table
# ---------------------------------------------------------------------------

def test_the_shipped_table_has_exactly_one_character():
    """Not a style preference -- the plan ships ONE and says a second must be
    added by a story lane that actually writes that row. A character appearing
    here without one would be a voice for somebody nobody casts."""
    assert set(RECURRING_CHARACTER_VOICES) == {"LEMMY"}


def test_the_assigned_ids_are_shared_catalogue_rows_not_reserved_ones():
    """THE DISTINCTION THE WHOLE DESIGN RESTS ON. An assignment is not a
    reservation. These ids stay castable for everyone else -- `bm_george` is a
    kokoro row tagged preferred_announcer, and pulling it out of the pool to
    protect a recurring character would starve the announcer draw. What IS
    reserved is the clone recording of his own voice, which lives on the bank
    row and is not in this table.
    """
    from nodes._otr_voice_bank import reserved_voice_ref_ids

    reserved = reserved_voice_ref_ids()
    for character, by_engine in RECURRING_CHARACTER_VOICES.items():
        for engine, voice_ref_id in by_engine.items():
            assert voice_ref_id not in reserved, (
                "%s's %s voice %r is RESERVED; an assignment must name a shared "
                "catalogue row" % (character, engine, voice_ref_id))


def test_every_assigned_id_exists_in_the_bank_on_the_engine_it_names():
    """An id that is not in the bank is an assignment that silently misses and
    falls through to an ordinary draw -- the character sounds like a stranger
    and nothing reports it."""
    from nodes._otr_voice_bank import load_voice_bank

    bank = load_voice_bank()[0]
    by_id = {}
    for entry in bank:
        by_id.setdefault(entry.voice_ref_id, set()).add(entry.engine)

    for character, by_engine in RECURRING_CHARACTER_VOICES.items():
        for engine, voice_ref_id in by_engine.items():
            assert voice_ref_id in by_id, (
                "%s/%s names %r, which is not in the voice bank"
                % (character, engine, voice_ref_id))
            assert engine in by_id[voice_ref_id], (
                "%s/%s names %r, which exists but only on %s"
                % (character, engine, voice_ref_id, sorted(by_id[voice_ref_id])))


def test_no_two_keys_collide_case_insensitively():
    """The lookup folds case, so "Lemmy" and "LEMMY" as separate keys would make
    which one wins depend on dict order."""
    folded = [str(k).strip().casefold() for k in RECURRING_CHARACTER_VOICES]
    assert len(folded) == len(set(folded)), folded


# ---------------------------------------------------------------------------
# The lookup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row", [
    {"name": "LEMMY"},
    {"name": "lemmy"},
    {"name": "  Lemmy  "},
    {"char_id": "LEMMY"},
    {"char_id": "lemmy"},
    {"name": "", "char_id": "lemmy"},
])
def test_a_recurring_row_is_recognised_by_name_or_char_id(row):
    assert recurring_character_key(row) == "LEMMY"


@pytest.mark.parametrize("row", [
    {"name": "MONTY"},
    {"name": "ANNOUNCER"},
    {"name": "", "char_id": ""},
    {},
])
def test_an_ordinary_row_matches_nothing(row):
    assert recurring_character_key(row) == ""


@pytest.mark.parametrize("bad", ["LEMMY", None, 42, ["LEMMY"], ("LEMMY",)])
def test_a_row_that_is_not_a_dict_is_ordinary_not_an_exception(bad):
    """Casting must not raise on a malformed row: it takes the ordinary draw."""
    assert recurring_character_key(bad) == ""


def test_an_explicitly_empty_registry_stays_empty():
    """`{}` means "no recurring characters", NOT "fall back to the default".
    Without this a test could not prove the ordinary path, because the shipped
    table would keep answering."""
    assert recurring_character_key({"name": "LEMMY"}, registry={}) == ""
    assert recurring_character_voice("LEMMY", "kokoro", registry={}) == ""


def test_the_canonical_key_is_returned_not_the_rows_spelling():
    """Callers index the table with the answer, so it has to be the table's own
    spelling -- otherwise a lowercase row produces a KeyError one layer down."""
    key = recurring_character_key({"name": "lemmy"})
    assert key in RECURRING_CHARACTER_VOICES
    assert recurring_character_voice(key, "kokoro") == "bm_george"


def test_an_unmapped_engine_returns_empty_rather_than_raising():
    """Bark is deliberately absent (it keeps its preset path), and a clone
    engine is absent too. Both must mean "take the ordinary draw"."""
    assert recurring_character_voice("LEMMY", "bark") == ""
    assert recurring_character_voice("LEMMY", "indextts2") == ""
    assert recurring_character_voice("LEMMY", "") == ""
    assert recurring_character_voice("NOBODY", "kokoro") == ""


# ---------------------------------------------------------------------------
# THE SCALABILITY CLAIM
# ---------------------------------------------------------------------------

def test_a_second_character_works_through_the_same_code_path():
    """THE POINT OF THE TABLE, and the operator's own question about it.

    A character who does not ship is injected into the registry and resolved by
    the SAME two helpers, with no argument naming him and no branch added. If
    adding a character ever requires touching code rather than data, this test
    is where that shows up -- it will still pass while the real integration has
    grown a Lemmy-only branch, which is why the CastLock-level version of this
    check lives beside the casting integration as well.
    """
    injected = {
        "LEMMY": {"kokoro": "bm_george"},
        "MRS_PEEL": {"kokoro": "bf_emma", "google_tts": "gt_someone"},
    }
    assert recurring_character_key({"name": "MRS_PEEL"}, injected) == "MRS_PEEL"
    assert recurring_character_key({"char_id": "mrs_peel"}, injected) == "MRS_PEEL"
    assert recurring_character_voice("MRS_PEEL", "kokoro", injected) == "bf_emma"
    assert recurring_character_voice("MRS_PEEL", "google_tts", injected) == "gt_someone"
    # ...and the shipped character is unaffected by the injection.
    assert recurring_character_voice("LEMMY", "kokoro", injected) == "bm_george"
    # ...and an engine the new character has no entry for still falls through.
    assert recurring_character_voice("MRS_PEEL", "cloud_elevenlabs", injected) == ""


def test_the_helpers_never_name_a_character_in_their_own_source():
    """A per-name branch inside the lookup would make the table a decoration.

    Source inspection is the right tool for exactly one job -- "is this written
    generically" -- and this is that job. The shipped DATA names LEMMY; the CODE
    must not.
    """
    import inspect

    for fn in (POOLS.recurring_character_key, POOLS.recurring_character_voice):
        src = inspect.getsource(fn)
        assert "LEMMY" not in src.upper().replace("RECURRING_CHARACTER_VOICES", ""), (
            "%s names a character in its own body; the table is data, not a "
            "list of things the code already knows" % fn.__name__)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# THE ANNOUNCER RULE, pinned across the two modules that each spell it.
#
# `_otr_voice_node_common._is_announcer_row` duplicates
# `cast_lock._is_announcer_entry` rather than importing it, so the cache-key
# probe stays cold-import clean. That is a deliberate duplication and it is
# also a standing invitation to drift, which is why its docstring promises a
# pin. Until 2026-09-24 that promise was false and no such test existed.
# ---------------------------------------------------------------------------

_ANNOUNCER_ROW_CASES = [
    {"char_id": "a1", "name": "ANNOUNCER"},
    {"char_id": "ANNOUNCER", "name": "MONTY"},
    {"char_id": "c01", "name": "announcer"},
    {"char_id": "c01", "name": "  Announcer  "},
    {"char_id": "c01", "name": "MONTY", "speaker_role": "announcer"},
    {"char_id": "c01", "name": "MONTY", "role": "ANNOUNCER"},
    {"char_id": "c02", "name": "LEMMY"},
    {"char_id": "c01", "name": "MONTY"},
    {"char_id": "", "name": ""},
    {"char_id": "c01", "name": "ANNOUNCERS"},
    {"char_id": "c01", "name": "THE ANNOUNCER"},
    {},
]


@pytest.mark.parametrize("row", _ANNOUNCER_ROW_CASES)
def test_the_two_announcer_rules_agree_row_for_row(row):
    """Whatever the rule is, both modules must answer it the same way."""
    from nodes._otr_voice_node_common import _is_announcer_row
    from nodes.cast_lock import _is_announcer_entry

    assert _is_announcer_row(row) == _is_announcer_entry(dict(row)), (
        "the cache-key probe and the caster disagree about whether %r is the "
        "announcer; the fingerprint would then cover a different set of rows "
        "than casting actually stamps" % (row,))


def test_only_the_probe_survives_a_non_dict():
    """The one DELIBERATE difference, stated so it is not read as drift.

    The probe adds an isinstance guard because it reads a ledger that may have
    come off disk in any shape; the caster is called with rows it has already
    validated. Pinning the difference keeps the test above honest about what
    "agree" means.
    """
    from nodes._otr_voice_node_common import _is_announcer_row

    for junk in (None, "ANNOUNCER", 7, ["ANNOUNCER"]):
        assert _is_announcer_row(junk) is False


def test_an_announcer_sharing_the_recurring_voice_is_still_excluded():
    """ANNOUNCER holding the SAME catalogue voice is not fingerprinted.

    `bm_george` is a shared kokoro row, not anybody's private recording, so the
    announcer and the recurring character can legitimately hold it at once.

    MEASURED HONESTLY: this test passes even with the announcer guard removed
    from the selection, because `recurring_character_key` already returns ""
    for a name the table does not list. It pins the OUTCOME, which is the thing
    that matters to the graph. The test below is the one that pins the guard.
    """
    from nodes.batch_character_voices import BatchCharacterVoices

    shared = "bm_george"
    announcer_only = json.dumps({
        "meta": {"episode_seed": 42},
        "cast": [
            {"char_id": "c01", "name": "MONTY", "gender": "male"},
            {"char_id": "a1", "name": "ANNOUNCER", "gender": "male",
             "voice_ref_id": shared, "voice_engine": "kokoro"},
        ],
        "lines": [],
    })
    assert BatchCharacterVoices.IS_CHANGED(
        script_json=announcer_only, engine="kokoro") == "static", (
        "an ANNOUNCER holding a shared catalogue voice was fingerprinted; it "
        "is a role, not a recurring character")


def test_the_announcer_is_excluded_even_if_the_table_names_him(monkeypatch):
    """THE TEST THAT ACTUALLY PINS THE GUARD, and the reason it is kept.

    The announcer check in the IS_CHANGED selection is redundant against
    today's table: `recurring_character_key` already returns "" for ANNOUNCER
    because ANNOUNCER is not a key in it. Remove the guard with the table as
    shipped and every test still passes -- which is exactly the kind of guard
    that gets deleted as dead and is then missed.

    It is not dead. `cast_lock` skips announcer rows at its CALL SITES, before
    it consults the table -- note the skip is in `_auto_registry` and
    `_apply_recurring_character_voices`, NOT inside `_recurring_character_key`,
    which happily answers "ANNOUNCER" for a table that names him. So if the
    table ever named ANNOUNCER, casting would still refuse to stamp him while
    an unguarded IS_CHANGED would start fingerprinting him. The two selections
    would then disagree about what a character is, which is what the
    cross-module pin above exists to prevent.

    Casting's own half of that is already covered by
    `test_only_the_recurring_row_takes_the_assignment`; this pins the probe's
    half, and it FAILS if the guard is removed from the selection.
    """
    from nodes.batch_character_voices import BatchCharacterVoices

    monkeypatch.setattr(
        "config.cast_pools.RECURRING_CHARACTER_VOICES",
        {"ANNOUNCER": {"kokoro": "bm_george"}})

    row = {"char_id": "a1", "name": "ANNOUNCER", "gender": "male",
           "voice_ref_id": "bm_george", "voice_engine": "kokoro"}
    led = json.dumps({"meta": {"episode_seed": 42}, "cast": [row], "lines": []})
    assert BatchCharacterVoices.IS_CHANGED(
        script_json=led, engine="kokoro") == "static", (
        "IS_CHANGED fingerprinted the announcer while casting refuses to "
        "stamp him -- the cache key now covers a different set of rows than "
        "casting actually writes")
