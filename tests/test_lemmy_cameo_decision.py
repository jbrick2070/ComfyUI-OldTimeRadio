"""The cameo decision is made ONCE, and every consumer reads that one answer.

WHY A DEDICATED FILE. The content-owned lanes derive their cast from the
finished script and then gate on it, so the cameo has to be decided BEFORE any
authoring -- and once it is decided, the prompt contract, the cast contract, the
roll receipt and the voice deal must all agree. The failure this pins is a lane
stamping `lemmy_hit: False` beside a Lemmy who is standing in the script.
"""
import json
import os

import pytest

from nodes import _otr_casting as CASTING
from nodes._otr_casting import (
    CONTENT_OWNED_NO_CAMEO_ROLL,
    LEMMY_CAMEO_DECISION_SCHEMA,
    LEMMY_KNOB_FORCED_EXCLUDE,
    LEMMY_KNOB_FORCED_INCLUDE,
    LEMMY_KNOB_NATURAL_ROLL,
    LEMMY_POLICY_OPERATOR_CAMEO,
    LEMMY_POLICY_SOURCE_FIDELITY_EXCLUSION,
    LemmyCameoDecision,
    content_owned_cast_contract,
    resolve_lemmy_cameo,
)

_EXCLUDED_BANK = "public_domain"
_CAMEO_BANK = "scifi_news_pro"


@pytest.fixture
def counted_roll(monkeypatch):
    """Count roll_lemmy calls so 'exactly once, and only here' is provable."""
    calls = []

    def _fake_roll():
        calls.append(True)
        return True

    monkeypatch.setattr(CASTING._POOLS, "roll_lemmy", _fake_roll)
    return calls


# --- the normative truth table -------------------------------------------


def test_an_excluded_bank_refuses_the_cameo_and_never_rolls(counted_roll):
    """Fidelity outranks everything: the source's cast is the point of the lane."""
    for knob in (None, True, False):
        decision = resolve_lemmy_cameo(_EXCLUDED_BANK, knob)

        assert decision.lemmy_hit is False
        assert decision.lemmy_policy == LEMMY_POLICY_SOURCE_FIDELITY_EXCLUSION
        assert decision.roll_executed is False

    assert counted_roll == [], "an excluded bank must not spend the roll"


def test_exclusion_outranks_an_operator_who_forced_the_cameo(counted_roll):
    """The knob still records what was ASKED, even though the bank refused --
    that is how a reader tells a refusal from a decline."""
    decision = resolve_lemmy_cameo(_EXCLUDED_BANK, True)

    assert decision.knob_state == LEMMY_KNOB_FORCED_INCLUDE
    assert decision.lemmy_hit is False


def test_forced_include_does_not_roll(counted_roll):
    decision = resolve_lemmy_cameo(_CAMEO_BANK, True)

    assert decision.lemmy_hit is True
    assert decision.lemmy_policy == LEMMY_POLICY_OPERATOR_CAMEO
    assert decision.knob_state == LEMMY_KNOB_FORCED_INCLUDE
    assert decision.roll_executed is False
    assert counted_roll == [], "a forced decision must not consult chance"


def test_forced_exclude_does_not_roll(counted_roll):
    decision = resolve_lemmy_cameo(_CAMEO_BANK, False)

    assert decision.lemmy_hit is False
    assert decision.lemmy_policy == LEMMY_POLICY_OPERATOR_CAMEO
    assert decision.knob_state == LEMMY_KNOB_FORCED_EXCLUDE
    assert decision.roll_executed is False
    assert counted_roll == []


def test_the_natural_roll_spends_exactly_one_roll(counted_roll):
    decision = resolve_lemmy_cameo(_CAMEO_BANK, None)

    assert decision.lemmy_hit is True          # the fake roll returns True
    assert decision.knob_state == LEMMY_KNOB_NATURAL_ROLL
    assert decision.roll_executed is True
    assert counted_roll == [True], "the roll runs once, not zero or twice"


def test_a_bank_variant_inherits_the_family_exclusion(counted_roll):
    """Fidelity is a family behaviour, so a bake-off variant cannot opt in."""
    decision = resolve_lemmy_cameo("public_domain_v3", None)

    assert decision.lemmy_policy == LEMMY_POLICY_SOURCE_FIDELITY_EXCLUSION
    assert counted_roll == []


# --- the receipt ----------------------------------------------------------


def test_to_meta_is_json_serializable_with_a_pinned_key_set():
    """`Ledger.save()` NEVER RAISES -- a non-serializable object in meta logs a
    warning, returns None, and a dozen call sites never check. So the episode
    would lose its receipt in silence. Everything here must be a primitive."""
    meta = resolve_lemmy_cameo(_CAMEO_BANK, True).to_meta()

    assert set(meta) == {
        "schema_version", "lemmy_hit", "lemmy_policy",
        "knob_state", "source_bank_id", "roll_executed",
    }
    assert meta["schema_version"] == LEMMY_CAMEO_DECISION_SCHEMA
    assert json.loads(json.dumps(meta)) == meta
    for value in meta.values():
        assert isinstance(value, (bool, int, str)), value


def test_the_decision_is_immutable():
    """Every consumer reads the same answer, so nobody may edit it in flight."""
    decision = resolve_lemmy_cameo(_CAMEO_BANK, True)

    with pytest.raises(Exception):
        decision.lemmy_hit = False


def test_the_receipt_records_the_bank_it_was_handed():
    assert resolve_lemmy_cameo("  media_archive  ", False).source_bank_id == (
        "media_archive")


# --- the cast contract ----------------------------------------------------


def test_no_decision_preserves_the_pre_chunk_b_contract_exactly():
    """The API lands green BEFORE its callers migrate, so an unmigrated lane
    must stamp byte-for-byte what it stamped before."""
    contract = content_owned_cast_contract(
        source_bank_id=_CAMEO_BANK,
        num_characters_request=3,
        num_characters_locked=4,
    )

    assert contract == {
        "lemmy_hit": False,
        "lemmy_policy": CASTING.CONTENT_OWNED_NO_CAMEO_ROLL,
        "casting_attempts": [],
        "num_characters_request": 3,
        "num_characters_locked": 4,
        # 2026-09-11: both contract shapes carry the widget value AS ASKED
        # (the verbatim lane executes at the passage's own speaker count and
        # records the ask beside it); on a content-owned lane it equals the
        # request. Held equal to the legacy block by the parity test.
        "num_characters_operator_request": 3,
    }


def test_no_decision_still_reports_the_fidelity_exclusion():
    contract = content_owned_cast_contract(
        source_bank_id=_EXCLUDED_BANK,
        num_characters_request=2,
        num_characters_locked=2,
    )

    assert contract["lemmy_policy"] == LEMMY_POLICY_SOURCE_FIDELITY_EXCLUSION


def test_a_decision_drives_the_contract_so_the_two_cannot_disagree():
    decision = resolve_lemmy_cameo(_CAMEO_BANK, True)
    contract = content_owned_cast_contract(
        source_bank_id=_CAMEO_BANK,
        num_characters_request=3,
        num_characters_locked=4,
        decision=decision,
    )

    assert contract["lemmy_hit"] is True
    assert contract["lemmy_policy"] == LEMMY_POLICY_OPERATOR_CAMEO
    assert contract["lemmy_hit"] == decision.to_meta()["lemmy_hit"]


def test_the_contract_never_grows_a_cast_seed():
    """`cast_seed` is a CLAIM that the writer's seeded picker produced this
    cast and can be replayed from it. A lane-owned cast has no such seed, and
    CastLock detonates when it tries to replay one."""
    contract = content_owned_cast_contract(
        source_bank_id=_CAMEO_BANK,
        num_characters_request=1,
        num_characters_locked=1,
        decision=resolve_lemmy_cameo(_CAMEO_BANK, False),
    )

    for forbidden in CASTING.CONTENT_OWNED_CONTRACT_FORBIDDEN_KEYS:
        assert forbidden not in contract


def test_the_decision_type_is_exported_for_the_runners():
    assert isinstance(resolve_lemmy_cameo(_CAMEO_BANK, False), LemmyCameoDecision)


# ---------------------------------------------------------------------------
# WHICH SHIPPED BANKS MAY CARRY THE CAMEO, moved here 2026-09-24.
#
# These lived in tests/test_cast_lock_policy_repin.py, which is slated for
# deletion with the voice-route subsystem. They have nothing to do with routes:
# they pin the shipped bank policy map against the real registry, and that
# `always include` never overrides source fidelity. Moving them BEFORE that file
# is deleted is the point -- coverage that dies because it was sharing a file
# with a dying subsystem is the defect this move exists to avoid.
#
# The map's own comment records the sweep that corrected it, and that is kept
# verbatim: it is the reason the map is asserted against the registry rather
# than hand-maintained.
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Every SHIPPED source bank, and whether the Lemmy cameo may appear in it.
#: An EXHAUSTIVE map whose keys are asserted equal to the shipped registry --
#: the same shape `ENGINE_COVERAGE` uses in tests/test_slug_provenance.py, and
#: for the same reason.
#:
#: CORRECTED 2026-08-11 FROM A LIVE SWEEP, and the correction is the lesson.
#: The first version of this map was written from the bank list without
#: measuring anything, and it asserted a cameo rule for the since-retired
#: sci-fi news lane. A six-bank
#: render sweep with the cameo FORCED on every leg proved otherwise: that lane
#: ran a content-owned pipeline that never calls `lock_cast()`, so
#: it writes an EMPTY `cast_contract` -- no cast_seed, no lemmy_hit, no
#: lemmy_policy -- and ignores both `lemmy_cameo` and `num_characters` (asked
#: for 2 characters, produced 3).
#:
#: A map that claims a rule nobody measured is the exact defect this file's
#: neighbours exist to catch, so the values below now say what was OBSERVED.
BANK_CAMEO_POLICY = {
    # PROVEN by live render, 2026-08-10/11: forced cameo produced a LEMMY row
    # on the qualified IndexTTS2 route.
    "original": "cameo_allowed",
    "media_archive": "cameo_allowed",
    # PROVEN by live render: forced cameo REFUSED, with
    # lemmy_policy="source_fidelity_exclusion" recorded on the ledger.
    "public_domain": "source_fidelity_excluded",
    "shakespeare": "source_fidelity_excluded",
    # NOT MEASURED. Its leg failed in the WRITER before casting ran
    # ("[scifi_news_pro] pass 'script' failed after 4 attempt(s): markup ladder
    # exhausted"), so nothing about its cameo behaviour was observed. Marked
    # unmeasured rather than assumed from its sibling -- that assumption is the
    # one that made this map wrong the first time.
    "scifi_news_pro": "unmeasured",
    # The creator bank NEVER ROLLS FOR A CAMEO AT ALL, which is a third thing
    # and not a shade of the other two. The cast belongs to the person who
    # described it, so the runner passes decision=None and the ledger records
    # lemmy_policy="content_owned_cast_no_cameo_roll" -- a decision that was
    # made, not a roll that came up short.
    #
    # `_source_bank_excludes_lemmy("my_story")` is False, exactly as it is for
    # the cameo_allowed banks: nothing in the fidelity exclusion applies here.
    # What stops the cameo is the lane never asking, which is why this needs
    # its own label rather than borrowing "cameo_allowed" (it is not) or
    # "unmeasured" (it is decided).
    "my_story": "no_cameo_roll",
    # Operator-supplied; the shipped map cannot see a bank this repo has never
    # heard of. See the docstring on the completeness test below.
    "custom_source_bank": "cameo_allowed",
}

#: The policies whose claim is "this lane does not run the roll". Held apart
#: from `_CAMEO_ALLOWED_POLICIES` because they answer different questions: that
#: tuple means Lemmy MAY appear, this one means nobody ever asked.
_NO_CAMEO_ROLL_POLICIES = ("no_cameo_roll",)

#: The policies whose claim is "Lemmy may appear here" -- the only ones the
#: `_source_bank_excludes_lemmy` cross-check can speak about.
_CAMEO_ALLOWED_POLICIES = ("cameo_allowed", "unmeasured")

def _shipped_bank_ids():
    import json
    import os

    path = os.path.join(REPO_ROOT, "nodes", "story_packs", "banks.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    banks = data if isinstance(data, list) else data.get("banks", data)
    if isinstance(banks, dict):
        banks = list(banks.values())
    return {str(b.get("source_bank_id") or b.get("id"))
            for b in banks if isinstance(b, dict)}

def test_the_exclusion_the_contract_rests_on_is_real_and_covers_both_banks():
    """If this ever narrows, the test above is guarding a rule that no longer
    exists. `always include` must never override source fidelity."""
    from nodes._otr_casting import _source_bank_excludes_lemmy

    for excluded in ("shakespeare", "public_domain",
                     "shakespeare_v2", "public_domain_v3"):
        assert _source_bank_excludes_lemmy(excluded), excluded
    for allowed in ("original", "archive", None, ""):
        assert not _source_bank_excludes_lemmy(allowed), allowed

def test_every_shipped_bank_has_a_DECIDED_cameo_policy():
    """THE MAINTENANCE HAZARD, converted into a build failure.

    The exclusion is a hardcoded frozenset of two names
    (`_LEMMY_EXCLUDED_SOURCE_BANK_IDS`), so a NEW source-faithful bank would
    escape it silently -- nothing would fail, Lemmy would simply start appearing
    in an adaptation. That is not hypothetical bookkeeping: the operator's rule
    is that `always include` must NEVER override source fidelity, and today that
    rule rests entirely on someone remembering to edit a set.

    Asserting the map is EQUAL to the shipped registry means a new bank cannot be
    added without someone deciding, in writing, which side it is on.

    WHAT THIS CANNOT COVER, said plainly: `custom_source_bank` and any user bank
    dropped under `user_packs/source_banks/<id>/` carry ids this repo has never
    seen. No static map can know whether an operator's own bank is a faithful
    adaptation. If a user ships a fidelity bank, its exclusion is their call.
    """
    shipped = _shipped_bank_ids()
    undecided = sorted(shipped - set(BANK_CAMEO_POLICY))
    stale = sorted(set(BANK_CAMEO_POLICY) - shipped)
    assert not undecided, (
        "shipped bank(s) with no cameo decision: %s -- add each to "
        "BANK_CAMEO_POLICY as cameo_allowed or source_fidelity_excluded, and if "
        "it is a faithful adaptation add it to _LEMMY_EXCLUDED_SOURCE_BANK_IDS "
        "too" % undecided)
    assert not stale, "BANK_CAMEO_POLICY names bank(s) no longer shipped: %s" % stale

def test_the_decided_policy_matches_what_the_code_actually_does():
    """The map above is a claim; this is the check that it is true.

    Scoped to what `_source_bank_excludes_lemmy` can actually answer: it decides
    SOURCE FIDELITY, and nothing else. `lane_owns_its_cast` and `unmeasured`
    banks are not excluded BY IT -- they simply never reach the cameo path -- so
    asserting exclusion for them would be testing the wrong function.
    """
    from nodes._otr_casting import _source_bank_excludes_lemmy

    for bank, policy in sorted(BANK_CAMEO_POLICY.items()):
        expected = (policy == "source_fidelity_excluded")
        assert _source_bank_excludes_lemmy(bank) is expected, (bank, policy)
        if policy in _CAMEO_ALLOWED_POLICIES:
            assert not _source_bank_excludes_lemmy(bank), bank

def test_every_policy_value_is_one_we_defined():
    known = (set(_CAMEO_ALLOWED_POLICIES)
             | set(_NO_CAMEO_ROLL_POLICIES)
             | {"source_fidelity_excluded"})
    for bank, policy in BANK_CAMEO_POLICY.items():
        assert policy in known, (bank, policy)

def test_a_no_roll_lane_is_not_a_fidelity_exclusion():
    """The two reasons a cameo is absent are not the same reason.

    A fidelity lane REFUSES the cameo -- `_source_bank_excludes_lemmy` says so,
    and it overrides the operator's knob. A creator lane never rolls at all,
    because the cast belongs to the person who described it. Both end with no
    cameo and they must not be recorded as the same decision, or a reader
    cannot tell a refusal from a question nobody asked.
    """
    from nodes._otr_casting import (
        CONTENT_OWNED_NO_CAMEO_ROLL, _source_bank_excludes_lemmy,
        content_owned_cast_contract,
    )

    for bank, policy in BANK_CAMEO_POLICY.items():
        if policy not in _NO_CAMEO_ROLL_POLICIES:
            continue
        assert not _source_bank_excludes_lemmy(bank), bank
        # What the runner actually stamps for such a lane: decided, not rolled.
        contract = content_owned_cast_contract(
            source_bank_id=bank, num_characters_request=2,
            num_characters_locked=2, decision=None,
        )
        assert contract["lemmy_hit"] is False, bank
        assert contract["lemmy_policy"] == CONTENT_OWNED_NO_CAMEO_ROLL, bank

def test_public_domain_story_is_a_CORPUS_DIRECTORY_not_a_bank_id():
    """Pinned because a review mistook one for the other and filed it as a
    fidelity breach.

    `config/source_banks/public_domain_story/` holds fetched source TEXTS
    (`scripts/otr_fetch_public_domain.py --dest-dir ...`), and every occurrence
    of the string as a `source_bank_id` VALUE lives in
    `docs/multimodal-story-schema/schema-examples/`. The shipped runtime id is
    `public_domain`, which IS excluded. If that ever stops being true -- if a
    bank really is named `public_domain_story` -- this test fails and the
    exclusion set genuinely does need it.
    """
    assert "public_domain_story" not in _shipped_bank_ids()
    assert "public_domain" in _shipped_bank_ids()
