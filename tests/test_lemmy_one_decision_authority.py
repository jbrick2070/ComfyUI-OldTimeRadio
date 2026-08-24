"""ONE cameo decision, and the legacy lanes finally get their receipt.

THE GAP (found 2026-08-24 by tracing character selection end to end, after the
operator asked to "trace the path of char selection and story gen and find the
gap"). The cameo decision existed THREE times over:

  1. `resolve_lemmy_cameo`            -- the real one, returning a full record
  2. `assemble_pre_locked_rows`       -- the same three branches, inline,
                                         returning a bare bool
  3. `lock_cast`'s meta stamp         -- the policy re-derived a THIRD time
                                         from `_source_bank_excludes_lemmy`

Only the dispatched `scifi_news_pro` lane called (1). Every legacy/invention
lane ran (2) and (3), which between them cannot express WHY the cameo did or
did not land -- so `media_archive`, `original` and `science_news` shipped 413
episodes with ZERO cameo receipts, while `scifi_news_pro` stamped one every
time. Measured on 1853 real ledgers, not inferred.

That is Bug Bible 12.132's class ("one matcher, never two"): the copies agreed
by luck, no test compared them, and a change to the exclusion set or to the
precedence had to be remembered in three places.

WHAT THIS FILE PINS:
* the three branches agree with the one authority, on every bank and knob;
* the roll is spent EXACTLY ONCE per cast lock (a second roll could disagree
  with the rows it is supposed to describe);
* the receipt reaches `lock_cast`'s meta, carrying the fields a bare bool
  cannot -- knob_state and roll_executed;
* the fidelity exclusion still outranks the operator knob.

CPU-only: no LLM, no GPU. `lock_cast` itself needs a model, so the seam is
tested through `assemble_pre_locked_rows` plus a direct read of the decision
authority -- the two things the legacy path actually consumes.
"""

from __future__ import annotations

import inspect

import pytest

from nodes import _otr_casting as CASTING
from nodes._otr_casting import (
    LEMMY_POLICY_OPERATOR_CAMEO,
    LEMMY_POLICY_SOURCE_FIDELITY_EXCLUSION,
    assemble_pre_locked_rows,
    resolve_lemmy_cameo,
)

#: A bank that MAY carry the cameo, and one the fidelity rule excludes.
_CAMEO_BANK = "media_archive"
_EXCLUDED_BANK = "shakespeare"


def _names(rows) -> "set[str]":
    return {str(r.get("name") or "") for r in rows}


# --------------------------------------------------------------------------- #
# one authority -- the assembler must not re-decide
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bank", [_CAMEO_BANK, "original", _EXCLUDED_BANK])
@pytest.mark.parametrize("knob", [True, False])
def test_the_assembler_agrees_with_the_decision_authority(bank, knob):
    """A forced knob is deterministic on both sides, so the two can be
    compared directly. Before the fix these were separate implementations
    that merely happened to agree."""
    decision = resolve_lemmy_cameo(bank, knob)
    rows, _slots, hit = assemble_pre_locked_rows(
        num_characters=3, force_lemmy=knob, source_bank_id=bank,
    )

    assert hit is decision.lemmy_hit
    assert ("LEMMY" in _names(rows)) is decision.lemmy_hit


def test_a_supplied_decision_is_obeyed_and_no_second_roll_happens(monkeypatch):
    """THE NEGATIVE CONTROL FOR THE DOUBLE-ROLL HAZARD. `lock_cast` resolves
    the decision and hands it down; if the assembler rolled again anyway, the
    cast rows could contradict the receipt describing them. Detonate the roll
    so a second call is impossible to miss."""
    def _detonate():
        raise AssertionError(
            "roll_lemmy() was called even though a decision was supplied -- "
            "the cameo would be decided twice and could disagree with itself")

    monkeypatch.setattr(CASTING._POOLS, "roll_lemmy", _detonate)

    decision = resolve_lemmy_cameo(_CAMEO_BANK, True)
    rows, _slots, hit = assemble_pre_locked_rows(
        num_characters=3,
        force_lemmy=None,          # natural roll -- would fire without the fix
        source_bank_id=_CAMEO_BANK,
        decision=decision,
    )

    assert hit is True
    assert "LEMMY" in _names(rows)


def test_the_natural_roll_is_spent_exactly_once(monkeypatch):
    """Not zero (the cameo must still be possible) and not twice."""
    calls = []

    def _counted():
        calls.append(1)
        return True

    monkeypatch.setattr(CASTING._POOLS, "roll_lemmy", _counted)

    assemble_pre_locked_rows(
        num_characters=3, force_lemmy=None, source_bank_id=_CAMEO_BANK,
    )

    assert len(calls) == 1, f"roll spent {len(calls)} times, expected exactly 1"


def test_the_assembler_no_longer_carries_its_own_copy_of_the_rule():
    """The duplication itself is the defect, so pin its absence. If a future
    edit re-inlines the branches, this fails even while behaviour still
    happens to match -- which is the whole point: they agreed by luck before."""
    source = inspect.getsource(assemble_pre_locked_rows)

    assert "resolve_lemmy_cameo(" in source, (
        "the assembler must defer to the one decision authority")
    assert "_POOLS.roll_lemmy()" not in source, (
        "the assembler must not roll the cameo itself -- that is the "
        "duplicate implementation this fix deleted")
    assert "_source_bank_excludes_lemmy(" not in source, (
        "the assembler must not re-derive the fidelity exclusion -- "
        "resolve_lemmy_cameo owns it")


# --------------------------------------------------------------------------- #
# the receipt the legacy lanes never had
# --------------------------------------------------------------------------- #
def test_lock_cast_stamps_the_full_roll_receipt():
    """The producing line, read off the real source. `lock_cast` needs a model
    to run, so its STAMP is what is pinned here; the writer-side copy is
    pinned below."""
    source = inspect.getsource(CASTING.lock_cast)

    assert '"lemmy_roll_receipt":     lemmy_decision.to_meta()' in source, (
        "lock_cast must stamp the full decision receipt, not just a bool")
    assert '"lemmy_policy":           lemmy_decision.lemmy_policy' in source, (
        "the policy must come from the carried decision, never be re-derived")


def test_the_writer_copies_the_receipt_into_the_ledger():
    """lock_cast's meta is copied KEY BY KEY into the ledger, so a key stamped
    upstream and not named in the writer never reaches a shipped episode --
    the file says so itself. Without this line the fix would be inert, and the
    suite would still be green."""
    from pathlib import Path

    writer = (Path(__file__).resolve().parents[1]
              / "nodes" / "OTR_LedgerScriptWriter.py").read_text(
                  encoding="utf-8")

    assert 'meta["lemmy_roll_receipt"] = cast_meta.get(' in writer, (
        "the writer must copy lemmy_roll_receipt, or the receipt dies in "
        "lock_cast's local meta and no episode ever carries it")


def test_the_receipt_says_whether_the_roll_was_actually_SPENT():
    """The distinction the bare bool could never make, and the reason the
    shipped rate was unreadable: a forced-off run and a roll that came up
    short both stamp lemmy_hit=False."""
    forced_off = resolve_lemmy_cameo(_CAMEO_BANK, False).to_meta()
    natural = resolve_lemmy_cameo(_CAMEO_BANK, None).to_meta()

    assert forced_off["lemmy_hit"] is False
    assert forced_off["roll_executed"] is False, (
        "a forced decision spends no roll")
    assert natural["roll_executed"] is True, (
        "a natural decision must record that the roll was spent, whichever "
        "way it landed")
    # Both are 'operator_cameo' -- which is exactly why roll_executed has to
    # exist for the two to be distinguishable at all.
    assert forced_off["lemmy_policy"] == LEMMY_POLICY_OPERATOR_CAMEO
    assert natural["lemmy_policy"] == LEMMY_POLICY_OPERATOR_CAMEO


def test_the_receipt_is_primitive_only_so_the_ledger_cannot_lose_it():
    """`Ledger.save()` never raises -- a non-serializable value logs a warning,
    returns None, and the episode loses its receipt in silence."""
    import json

    receipt = resolve_lemmy_cameo(_CAMEO_BANK, None).to_meta()

    assert json.loads(json.dumps(receipt)) == receipt
    for value in receipt.values():
        assert isinstance(value, (bool, int, str)), value


# --------------------------------------------------------------------------- #
# the fidelity rule still outranks everything
# --------------------------------------------------------------------------- #
def test_a_fidelity_bank_refuses_the_cameo_even_when_forced(monkeypatch):
    """Unchanged behaviour, pinned because the consolidation moved the code
    that enforces it. shakespeare/public_domain casts must contain only
    source-appropriate people -- a 0% there is CORRECT, not a defect."""
    monkeypatch.setattr(
        CASTING._POOLS, "roll_lemmy",
        lambda: (_ for _ in ()).throw(AssertionError(
            "an excluded bank must never reach the roll")))

    decision = resolve_lemmy_cameo(_EXCLUDED_BANK, True)
    rows, _slots, hit = assemble_pre_locked_rows(
        num_characters=3, force_lemmy=True, source_bank_id=_EXCLUDED_BANK,
    )

    assert hit is False
    assert "LEMMY" not in _names(rows)
    assert decision.lemmy_policy == LEMMY_POLICY_SOURCE_FIDELITY_EXCLUSION
    assert decision.roll_executed is False
