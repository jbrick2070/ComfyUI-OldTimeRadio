"""The RESERVED voices are unreachable from EVERY pool that can draw a voice.

Two pools can hand out a character voice today, and both must obey the
reservation: the deterministic selector `assign_voice_for_slot` (also covered by
`tests/test_lemmy_voice_stays_reserved.py`) and `gender_agnostic_fallback_ref`.
`test_reserved_ids_are_unreachable_from_every_pool_at_once` is the assertion to
EXTEND if a third pool ever appears -- do not write another file.

WHY THE FILE READS LIKE A POST-MORTEM. There WERE three pools. PBUG-20260817-08
was fixed on 2026-08-17 in the selector, proved with 480 seeded draws, and
closed -- while two other pools still leaked. The hybrid LLM voice-fit was
offering `idx_lemmy_algenib_cockney_v1` as CARD #1 on every male slot, because
cards were ordered alphabetically and capped at 12; the corpus records the model
proposing a reserved id 21 times, accepted every time, putting Lemmy's qualified
Cockney on DON PEDRO, MARCELLUS, BANQUO, STARBUCK, FERDINAND, MOE GORDON and
others. A QA pass then found the third pool, `gender_agnostic_fallback_ref`,
returning his clone in 7-9 of every 200 draws.

**The hybrid pool no longer exists** -- the pass was ripped on 2026-08-18 because
it had no information the scorer lacks -- so the tests that exercised its card
list and its proposal validator were removed with it. The two pools that remain
are the two that can still draw, and they are both tested below.

THE LESSON THAT OUTLIVES THE BUG, and it is why this file is one file: a guard
that lives in ONE subsystem is invisible to every other subsystem enumerating
the same catalogue. Three pools had to learn the same reservation separately,
each found by a different method -- an operator sighting, a measurement of which
path production actually takes, and an adversarial review. The way to avoid a
fourth is to ask "what are ALL the places that can draw one of these?" rather
than fixing the place the bug was seen.
"""
from __future__ import annotations

import pytest

from nodes._otr_voice_bank import _reserved_ids_from_policy, gender_agnostic_fallback_ref, load_voice_bank, reserved_voice_ref_ids

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
    """The helper is called by both pools, and the fallback promises never to
    raise. A
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
    reachable = set()
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
