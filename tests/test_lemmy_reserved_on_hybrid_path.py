"""The RESERVED voices are unreachable from EVERY pool that can draw a voice.

Three separate pools have had to learn the reservation, one per discovery, and
the file is organised in that order: the deterministic selector (fixed
2026-08-17, covered by `tests/test_lemmy_voice_stays_reserved.py`), the HYBRID
LLM voice-fit (fixed 2026-08-18, first half of this file), and the
gender-agnostic fallback (fixed 2026-08-18 after a QA pass, second half).
`test_reserved_ids_are_unreachable_from_every_pool_at_once` at the end is the
assertion to EXTEND if a fourth pool ever appears -- do not write a fourth file.

PBUG-20260817-08, re-opened 2026-08-18. The 2026-08-17 fix (`8f3c7615`) filtered
`reserved_voice_ref_ids()` out of `assign_voice_for_slot` and
`tests/test_lemmy_voice_stays_reserved.py` proved that selector obeys it across
480 seeded draws. **Both were correct and both were measuring the wrong path.**

`hybrid_voice_fit_enabled()` was **default-ON when this leak happened** (it went
default-OFF on 2026-08-18, after these tests were written). While it was on, the
LLM was shown `build_voice_cards()` and proposed one id per character,
`validate_voice_proposal()` checked it, and `cast_lock.py:884-906` stamped the
accepted id and `continue`d -- never reaching the deterministic selector at all.
**These tests still matter with the pass off**: it remains reachable by explicit
opt-in until the code is ripped, and a reserved voice must not leak on a lane
merely because that lane is not the default today.
Measured over 1711 ledgers in the episodes tree: **1871 character rows came from
an accepted proposal against 82 fallbacks**, so the path those 480 draws
exercised carries roughly 4% of production casting.

Neither `build_voice_cards` nor `validate_voice_proposal` knew reserved ids
existed. Because ids are offered in alphabetical order capped at 12,
`idx_lemmy_algenib_cockney_v1` sorted FIRST among indextts2 male entries and was
handed to the model as CARD #1 on every male slot. The corpus records the LLM
proposing a reserved id **21 times, accepted 21 times**, putting his qualified
Cockney on DON PEDRO, MARCELLUS, BANQUO, FLETCHER CORBEN, STARBUCK, FERDINAND,
MOE GORDON and Dr. Alexei Petrov -- 20 leaked rows against 5 legitimate LEMMY
ones.

**This also corrects the record on `signal_lost_rivers_embrace_20260817_233013`.**
It was read as the resident-server trap: a leg 16h after the fix still running
the module loaded that morning. Process age did not produce that row -- the
hybrid path was unguarded at that commit and stayed unguarded, which a live call
against a freshly imported module confirms. A stale process and an uncovered
code path read identically from the outside, and only a test on the path that
actually runs tells them apart.

A 2026-08-04 kibitz round had already named this exact layer -- *"the hybrid LLM
voice-fit ... whose 12-card truncation is a harder variety cap"* -- and asked for
"one assertion ... that runs with the hybrid path ENABLED so a future working LLM
cannot silently undo the win". This file is that assertion, fourteen days late.
"""
from __future__ import annotations

import pytest

from nodes._otr_voice_bank import (_reserved_ids_from_policy,
                                   build_voice_cards, default_char_engine,
                                   gender_agnostic_fallback_ref,
                                   load_voice_bank, reserved_voice_ref_ids,
                                   validate_voice_proposal)

GENDERS = ("male", "female")
#: Every engine that draws a cloned character voice. The fallback below is
#: engine-scoped, so a per-engine sweep is the only honest coverage.
CLONER_ENGINES = ("indextts2", "chatterbox", "dia")


def _bank():
    return load_voice_bank()[0]


def test_the_reservation_is_not_empty():
    """Every assertion below is vacuous if nothing is reserved, so this fails
    loudly rather than letting the file pass while the guard does nothing."""
    assert reserved_voice_ref_ids(), (
        "nothing is reserved -- every assertion in this file would pass while "
        "the LLM was being offered Lemmy's own voice"
    )


@pytest.mark.parametrize("gender", GENDERS)
def test_no_reserved_id_is_ever_offered_to_the_llm(gender):
    """The card list is what the model chooses FROM. A reserved id on it is the
    defect: it was card #1 for male slots, the position a model favours most."""
    bank = _bank()
    engine = default_char_engine(bank)
    ids = [c["voice_ref_id"] for c in build_voice_cards(engine, gender, bank=bank)]
    leaked = sorted(set(ids) & reserved_voice_ref_ids())
    assert not leaked, (
        f"{gender} card list offers reserved voice(s) {leaked} to the LLM; "
        f"an accepted proposal is stamped straight onto the character"
    )


@pytest.mark.parametrize("gender", GENDERS)
def test_the_card_list_is_not_empty_and_did_not_shrink_to_nothing(gender):
    """TEETH. Filtering the pool must not be satisfiable by returning no cards --
    an empty list makes the test above pass and silently disables the whole
    hybrid pass (every row would fall back with reason ``no_cards``)."""
    bank = _bank()
    engine = default_char_engine(bank)
    cards = build_voice_cards(engine, gender, bank=bank)
    assert len(cards) >= 2, (
        f"{gender} card list collapsed to {len(cards)} -- the reservation must "
        f"remove reserved ids, not the pool"
    )


@pytest.mark.parametrize("gender", GENDERS)
def test_a_reserved_proposal_is_refused_even_when_not_on_the_card_list(gender):
    """A proposal is free text from a model. It does not have to name a card it
    was shown, so the card filter alone is not a gate -- this is the last check
    before CastLock stamps the row."""
    bank = _bank()
    engine = default_char_engine(bank)
    for reserved_id in sorted(reserved_voice_ref_ids()):
        assert validate_voice_proposal(
            reserved_id, engine, gender, bank=bank, used_ids=set()
        ) == "", (
            f"validate_voice_proposal accepted reserved id {reserved_id!r} for a "
            f"{gender} slot; CastLock stamps whatever this returns"
        )


@pytest.mark.parametrize("gender", GENDERS)
def test_an_ordinary_proposal_still_validates(gender):
    """TEETH. If validation started refusing everything the test above would pass
    for entirely the wrong reason and the hybrid pass would be dead."""
    bank = _bank()
    engine = default_char_engine(bank)
    cards = build_voice_cards(engine, gender, bank=bank)
    probe = cards[0]["voice_ref_id"]
    assert validate_voice_proposal(
        probe, engine, gender, bank=bank, used_ids=set()
    ) == probe, (
        f"ordinary proposal {probe!r} was refused -- the reservation is now "
        f"rejecting legitimate voices"
    )


def test_reserved_ids_are_still_present_in_the_unfiltered_bank():
    """TEETH, and the one that matters most. If Lemmy's rows were simply deleted
    from the bank every test here would pass while he lost his own voice. The
    reservation must hide him from the CHOOSERS, never from the catalogue -- his
    qualified route stamps these ids directly in CastLock."""
    bank_ids = {e.voice_ref_id for e in _bank()}
    missing = sorted(reserved_voice_ref_ids() - bank_ids)
    assert not missing, (
        f"reserved ids {missing} are not in the bank at all -- the guard would "
        f"pass for the wrong reason and Lemmy's own route cannot resolve"
    )


# --------------------------------------------------------------------------- #
# THE THIRD POOL. Found by a Sonnet QA pass on the hybrid fix above, 2026-08-18.
#
# `gender_agnostic_fallback_ref` is not a "caster" by name, which is exactly why
# it was missed twice. It draws a real reference that BOTH the ledger stamp and
# the render path use, so a reserved id reached from here is heard identically
# to one reached from the selector. And it is not a rare branch: canonical
# gender `other` is 20% of every roll, the bank carries zero rows for it, so
# `assign_voice_for_slot` raises and every one of those rows lands here.
# Measured before the filter: Lemmy's clone came back in 7-9 of 200 draws per
# engine -- roughly the odds of any other single voice, because nothing excluded
# it. The corpus agrees: 2 of the 20 leaked rows did NOT come from an accepted
# LLM proposal, so at least one non-hybrid path was leaking all along.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("engine", CLONER_ENGINES)
def test_the_gender_agnostic_fallback_never_draws_a_reserved_ref(engine):
    """The `gender="other"` fallback draws uniformly over the engine's refs.
    Before 2026-08-18 that pool included reserved ids."""
    bank = _bank()
    reserved = reserved_voice_ref_ids()
    drawn = set()
    for i in range(200):
        entry = gender_agnostic_fallback_ref(
            bank, engine=engine, char_id=f"char_{i}", episode_seed=i,
            role="char_voice",
        )
        if entry is not None:
            drawn.add(entry.voice_ref_id)
    leaked = sorted(drawn & reserved)
    assert not leaked, (
        f"{engine} gender-agnostic fallback drew reserved voice(s) {leaked} "
        f"across 200 seeded draws"
    )


@pytest.mark.parametrize("engine", CLONER_ENGINES)
def test_the_gender_agnostic_fallback_still_returns_a_voice(engine):
    """TEETH. Excluding reserved ids must not empty this pool -- returning None
    here does not fail safely, it drops a cloning engine back to bark and the
    ledger names a voice the render never opened."""
    bank = _bank()
    entry = gender_agnostic_fallback_ref(
        bank, engine=engine, char_id="somebody", episode_seed=7,
        role="char_voice",
    )
    assert entry is not None, (
        f"{engine} fallback returned None -- an uncastable row now has no voice"
    )
    assert entry.engine == engine


def test_a_broken_policy_reserves_nothing_instead_of_raising():
    """The helper has four callers and two of them promise never to raise. A
    malformed policy must degrade to reserving nothing -- the exact
    pre-reservation behaviour -- rather than killing a render."""
    for broken in ("not-a-dict", 12345, ["also", "wrong"]):
        assert _reserved_ids_from_policy({}) == frozenset()
        with pytest.raises(AttributeError):
            # proves the raw walk really would raise on this input, so the
            # wrapper's except clause is load-bearing and not decorative
            _reserved_ids_from_policy(broken)


def test_reserved_ids_are_unreachable_from_every_pool_at_once():
    """The whole point, stated once. Three pools have now had to learn this
    separately; if a fourth appears, this is the assertion that should be
    extended rather than a fourth file being written."""
    bank = _bank()
    reserved = reserved_voice_ref_ids()
    engine = default_char_engine(bank)
    reachable = set()
    for gender in GENDERS:
        reachable |= {c["voice_ref_id"] for c in build_voice_cards(engine, gender, bank=bank)}
    for eng in CLONER_ENGINES:
        for i in range(60):
            e = gender_agnostic_fallback_ref(
                bank, engine=eng, char_id=f"c{i}", episode_seed=i, role="char_voice")
            if e is not None:
                reachable.add(e.voice_ref_id)
    assert not (reachable & reserved), (
        f"reserved ids reachable from a production pool: "
        f"{sorted(reachable & reserved)}"
    )
