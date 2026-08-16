"""tests/test_content_owned_cast_contract.py

THE DEFECT (PBUG-20260811-03, corrected 2026-08-16): both content-owned lanes
built their cast from the finished script, never reached ``lock_cast()``, and
never stamped ``meta.cast_contract`` AT ALL -- the key was absent (not ``{}``;
that was a probe's ``or {}`` fallback rendering). A reader could not tell a
declined cameo from one that was never considered, and nothing failed and
nothing logged.

Chunk A of the Lemmy sprint: ``content_owned_cast_contract`` in
``nodes/_otr_casting.py`` builds the decision receipt, and the runner stamps
it -- ``_otr_scifi_fable2._stamp_cast_contract`` after ``_assemble`` (which
saves and rebinds ``led.data``, so the stamp reacquires the live meta).

The codex runner carried the twin stamp until the 2026-08-16 ``scifi_news``
rip; its cases went with it. The helper is lane-agnostic and still covers
both announcer row shapes, because a future content-owned lane may ship
either one.

Load-bearing absences, pinned here: NO ``cast_seed`` and NO
``cast_seed_source``. ``cast_seed`` is a replay claim -- OTR_CastLock replays
the writer's picker whenever it sees the key, and a lane-owned cast has no
``num_characters_request`` to replay with ("num_characters must be 1-6, got
0"). Credits read ``cast_contract.get("cast_seed", meta.episode_seed)``, so
the key must be ABSENT rather than present-and-None.
"""

import os
import re
from types import SimpleNamespace

from nodes import _otr_casting as casting
from nodes._otr_casting import (
    CONTENT_OWNED_CONTRACT_FORBIDDEN_KEYS,
    CONTENT_OWNED_NO_CAMEO_ROLL,
    content_owned_cast_contract,
    count_locked_characters,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The exact key set the legacy writer stamps (OTR_LedgerScriptWriter's
#: ``meta["cast_contract"] = {...}`` block). The content-owned contract is
#: this minus the two replay keys -- held equal from BOTH sides below.
LEGACY_CONTRACT_KEYS = {
    "lemmy_hit", "lemmy_policy", "casting_attempts",
    "num_characters_request", "num_characters_locked",
    "cast_seed", "cast_seed_source",
}


def _read(rel_path: str) -> str:
    with open(os.path.join(REPO_ROOT, rel_path), encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------- shape


def test_shape_is_legacy_minus_replay_keys():
    contract = content_owned_cast_contract(
        source_bank_id="scifi_news_pro",
        num_characters_request=2,
        num_characters_locked=3,
    )
    assert set(contract) == (
        LEGACY_CONTRACT_KEYS - CONTENT_OWNED_CONTRACT_FORBIDDEN_KEYS
    )


def test_legacy_writer_block_still_stamps_exactly_the_pinned_keys():
    """Parity from the OTHER side: if the writer's contract block gains or
    loses a key, this fails loudly instead of letting the two shapes drift."""
    src = _read(os.path.join("nodes", "OTR_LedgerScriptWriter.py"))
    start = src.index('meta["cast_contract"] = {')
    end = src.index("}", start)
    keys = set(re.findall(r'"([a-z_]+)"\s*:', src[start:end]))
    assert keys == LEGACY_CONTRACT_KEYS


def test_replay_keys_are_absent_not_none():
    """Credits read .get("cast_seed", episode_seed): a present-and-None key
    would satisfy .get and hand credits a null seed. Absence is the contract."""
    contract = content_owned_cast_contract(
        source_bank_id="scifi_news_pro",
        num_characters_request=2,
        num_characters_locked=2,
    )
    for key in CONTENT_OWNED_CONTRACT_FORBIDDEN_KEYS:
        assert key not in contract


# ---------------------------------------------------------------- policy


def test_policy_is_fidelity_exclusion_on_adaptation_banks():
    for bank in ("public_domain", "shakespeare",
                 "public_domain_v3", "shakespeare_v2"):
        contract = content_owned_cast_contract(
            source_bank_id=bank,
            num_characters_request=2,
            num_characters_locked=2,
        )
        assert contract["lemmy_policy"] == "source_fidelity_exclusion", bank


def test_policy_is_content_owned_no_roll_on_the_scifi_family():
    for bank in ("scifi_news_pro", "", None):
        contract = content_owned_cast_contract(
            source_bank_id=bank,
            num_characters_request=2,
            num_characters_locked=2,
        )
        assert contract["lemmy_policy"] == CONTENT_OWNED_NO_CAMEO_ROLL, bank


def test_policy_is_not_operator_cameo():
    """operator_cameo asserts the 11% roll RAN and declined. These lanes never
    roll, and stamping a decline nobody made is the same silence in a better
    disguise."""
    contract = content_owned_cast_contract(
        source_bank_id="scifi_news_pro",
        num_characters_request=2,
        num_characters_locked=2,
    )
    assert contract["lemmy_policy"] != "operator_cameo"


def test_no_cameo_and_no_attempts():
    contract = content_owned_cast_contract(
        source_bank_id="scifi_news_pro",
        num_characters_request=2,
        num_characters_locked=3,
    )
    assert contract["lemmy_hit"] is False
    assert contract["casting_attempts"] == []
    assert contract["num_characters_request"] == 2
    assert contract["num_characters_locked"] == 3


# ------------------------------------------------- count_locked_characters


def test_count_excludes_announcer_by_char_id_shape():
    # A lane whose announcer row uses char_id "announcer" (the shape the
    # retired codex lane shipped; kept because it is a real ledger shape
    # a future content-owned lane may reproduce).
    cast = [
        {"char_id": "announcer", "name": "ANNOUNCER"},
        {"char_id": "c01", "name": "Dr. Aris Thorne"},
        {"char_id": "c02", "name": "Elias Vance"},
        {"char_id": "c03", "name": "Unit 7"},
    ]
    assert count_locked_characters(cast) == 3


def test_count_excludes_announcer_by_name_fable2_shape():
    # scifi_news_pro ships ANNOUNCER at char_id "c01" (blood_red_water shape),
    # so len(cast) - 1 arithmetic is exactly what this helper must NOT be.
    cast = [
        {"char_id": "c01", "name": "ANNOUNCER"},
        {"char_id": "c02", "name": "Elias"},
        {"char_id": "c03", "name": "Sarah"},
    ]
    assert count_locked_characters(cast) == 2


def test_count_tolerates_junk_and_empty():
    assert count_locked_characters([]) == 0
    assert count_locked_characters(None) == 0
    assert count_locked_characters(["not-a-dict", 7]) == 0


# ---------------------------------------------------------- runner stamps


# `test_codex_runner_stamps_after_assemble` lived here and pinned the twin
# stamp on the codex runner, including that it closed `cast_status` (which
# froze "building" on published episodes). It went with the 2026-08-16
# `scifi_news` rip. The fable2 case below is now the only runner stamp.


def test_fable2_runner_stamps_live_meta():
    led = SimpleNamespace(data={
        "meta": {},
        "cast": [
            {"char_id": "c01", "name": "ANNOUNCER"},
            {"char_id": "c02", "name": "Elias"},
            {"char_id": "c03", "name": "Sarah"},
        ],
    })
    from nodes import _otr_scifi_fable2 as fable2

    contract = fable2._stamp_cast_contract(
        led,
        resolved={"num_characters": 2},
        source_bank_row=SimpleNamespace(source_bank_id="scifi_news_pro"),
    )
    meta = led.data["meta"]
    assert meta["cast_contract"] is contract
    assert contract["lemmy_policy"] == CONTENT_OWNED_NO_CAMEO_ROLL
    assert contract["num_characters_locked"] == 2
    for key in CONTENT_OWNED_CONTRACT_FORBIDDEN_KEYS:
        assert key not in contract


def test_fable2_call_site_sits_between_assemble_and_receipt_save():
    src = _read(os.path.join("nodes", "_otr_scifi_fable2.py"))
    call = src.rindex("_stamp_cast_contract(")
    assert call > src.index("def run_scifi_fable2_episode(")
    assert call > src.index("    _assemble(")
    assert call < src.index('require_ledger_save(led, "the fable2 pass receipts")')
