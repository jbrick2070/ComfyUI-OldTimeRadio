"""Voice-reference genders the OPERATOR settled by ear stay settled.

A bank row's `gender` is an ear call, not a code call. Nothing in the repo can
hear a recording, so once the operator has ruled on a row that ruling is the
authority and a later import, bulk edit or heuristic must not quietly undo it.

WHY THIS FILE EXISTS. `vz_donor_glenn` was imported FEMALE on 2026-06-05 and
cast on 115 female slots -- it became the #2 female voice in the corpus -- until
the operator heard it and it was flipped to MALE in `e1c84cf6` (2026-08-17,
"the operator heard four bugs the code could not"), together with its
`cb_`/`dia_` siblings. That is two and a half months of female characters
speaking with a male voice, found by ear because no test could find it.

His instruction on 2026-08-18, in his words: *"We [spent] so much time fixing
glenn please [don't] break it again but if you can look for other fixes fine but
i may need to confirm."* This file is the "don't break it again" half.

THIS IS A PIN, NOT AN AUDIT. It asserts only what the operator has actually
ruled on. Rows that merely LOOK mis-stamped are deliberately NOT asserted here --
as of 2026-08-18 nine of them are open and unconfirmed (the `hillbilly_jim`,
`rup` and `james` trios, all stamped female on indextts2/chatterbox/dia).
Adding an unconfirmed row to this file would turn a guess into a contract.
When he confirms one, add it here in the same change.
"""
from __future__ import annotations

import pytest

from nodes._otr_voice_bank import load_voice_bank

#: voice_ref_id -> gender the operator settled by ear, with the ruling's commit.
#: Only add a row here AFTER he has confirmed it.
OPERATOR_SETTLED_GENDERS = {
    # e1c84cf6, 2026-08-17 -- heard as male after months cast as female.
    "vz_donor_glenn": "male",
    "cb_donor_glenn": "male",
    "dia_donor_glenn": "male",
}


def _bank_by_id():
    return {e.voice_ref_id: e for e in load_voice_bank()[0]}


@pytest.mark.parametrize(
    "voice_ref_id,expected_gender", sorted(OPERATOR_SETTLED_GENDERS.items())
)
def test_operator_settled_gender_is_unchanged(voice_ref_id, expected_gender):
    """The ruling stands until he changes it, not until an import disagrees."""
    entry = _bank_by_id().get(voice_ref_id)
    assert entry is not None, (
        f"{voice_ref_id} is gone from the bank -- the operator ruled on this row "
        f"by ear; removing it silently discards that ruling"
    )
    assert entry.gender == expected_gender, (
        f"{voice_ref_id} is stamped {entry.gender!r} but the operator settled it "
        f"as {expected_gender!r} by ear (e1c84cf6). This is a REGRESSION of a fix "
        f"that cost two and a half months of wrong-voiced episodes to find. Do "
        f"not 'correct' this test -- restore the bank value, or get a fresh "
        f"ruling from him first"
    )


def test_the_pin_list_is_not_empty():
    """TEETH. An empty pin list would make every assertion above vacuous while
    the file still reported green."""
    assert OPERATOR_SETTLED_GENDERS, "nothing pinned -- this file proves nothing"
