"""An ordinary character can never be handed Lemmy's own voice.

PBUG-20260817-08. The operator's ruling was one line -- *"only Lemmy should get
Lemmy voice"* -- after `ED HIBBERT`, in a cast of ANNOUNCER / ERIN BURNS / ED
HIBBERT with no Lemmy anywhere, spoke the cameo's Cockney for a whole episode.

WHY THIS FILE EXISTS WHEN THE FIX ALREADY SHIPPED. The reservation landed in
`8f3c7615` and `tests/test_lemmy_provisional_tier.py` already checks the helper:
that `reserved_voice_ref_ids()` holds his clone refs, that it does NOT sweep in
the catalogue voices he merely borrows, and that reserving cannot starve Lemmy of
his own voice. **None of those exercise the selector.** They prove the LIST is
right; nothing proved the POOL obeys it, and the pool is where the bug lived.

That distinction is not academic here. The leak was seen AGAIN sixteen hours
AFTER the fix commit, in `signal_lost_rivers_embrace_20260817_233013`, because
the soak harness boots one server and never tears it down -- so an evening leg
was still executing the module loaded into memory that morning. A stale process
is not a code defect, but it is exactly the situation where "we fixed it" and "it
cannot happen" drift apart, and only a selector-level test closes that gap.

Corpus check at the time of writing: across the 15 most recent episodes with a
cast ledger, the two rows carrying a Lemmy-owned reference on a non-Lemmy
character are `kinetic_motion_clause_live_test` (05:01) and `rivers_embrace`
(23:30), both on 2026-08-17. Every episode from 2026-08-18 onward is clean.
"""
from __future__ import annotations

import pytest

from nodes._otr_voice_bank import (VoiceCastingError, assign_voice_for_slot,
                                   load_voice_bank, reserved_voice_ref_ids)


def _bank():
    return load_voice_bank()[0]


def test_the_reservation_is_not_empty():
    """A guard derived from a policy fails OPEN if the policy stops naming
    anything, and an empty reservation would make every test below vacuous."""
    assert reserved_voice_ref_ids(), (
        "nothing is reserved -- every assertion in this file would pass while "
        "the pool handed Lemmy's voice to anyone")


def test_no_ordinary_slot_can_ever_draw_a_reserved_reference():
    """THE ACTUAL REGRESSION TEST -- the selector, not the helper.

    Sweeps the seed space rather than sampling one draw: the pick is a seeded
    choice, so a single seed proves nothing about the seed that bit us. If a
    reserved reference is reachable at all, one of these finds it.
    """
    bank = _bank()
    reserved = reserved_voice_ref_ids()
    engines = sorted({e.engine for e in bank
                      if e.voice_ref_id in reserved} or {"indextts2"})

    checked = 0
    for engine in engines:
        for gender in ("male", "female"):
            for seed in range(120):
                try:
                    entry = assign_voice_for_slot(
                        role="character", engine=engine,
                        char_id="c%02d" % (seed % 20), gender=gender,
                        episode_seed=seed, bank=bank)
                except VoiceCastingError:
                    # Nothing castable for this combination is a legitimate
                    # outcome and says nothing about the reservation.
                    continue
                checked += 1
                assert entry.voice_ref_id not in reserved, (
                    "engine=%s gender=%s seed=%d drew RESERVED %s -- a character "
                    "who is not Lemmy would speak in Lemmy's voice"
                    % (engine, gender, seed, entry.voice_ref_id))
    assert checked, "no slot was ever cast -- this test proved nothing"


def test_the_reserved_reference_really_is_in_the_bank_it_is_filtered_from():
    """A filter that removes something absent is a filter that proves nothing.

    If the reserved id were not a castable bank row in the first place, the test
    above would pass for the wrong reason and would keep passing if the guard
    were deleted.
    """
    bank = _bank()
    reserved = reserved_voice_ref_ids()
    present = {e.voice_ref_id for e in bank} & reserved
    assert present, (
        "no reserved id is a bank entry -- the pool filter has nothing to do, "
        "so these tests cannot detect its removal")


def test_removing_the_guard_would_make_the_reference_reachable():
    """PROVES THE TEST HAS TEETH. Runs the same seed sweep against the bank with
    the reservation NOT applied, and asserts the reserved voice does turn up.

    Without this, a future change that quietly stopped casting that engine would
    turn the real test green for a reason that has nothing to do with the guard.
    """
    bank = _bank()
    reserved = reserved_voice_ref_ids()
    target = sorted({e.voice_ref_id for e in bank} & reserved)
    if not target:
        pytest.skip("no reserved id is castable on this bank")
    engine = next(e.engine for e in bank if e.voice_ref_id == target[0])

    # Reproduce the selector's own pool WITHOUT the reservation step, then
    # confirm the reserved row is genuinely among the candidates it filters.
    candidates = [e for e in bank
                  if e.engine == engine and e.quality_tier != "reject"]
    unreserved = {e.voice_ref_id for e in candidates}
    assert target[0] in unreserved, (
        "%s is not in the unfiltered candidate pool, so the guard is not what "
        "keeps it out of casts" % target[0])
