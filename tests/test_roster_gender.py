"""The source-roster gender join (_otr_roster_gender).

Run against the REAL shipped provenance sidecars, not fixtures: the whole point
of the module is that it reads a confirmed record, so a synthetic roster would
prove nothing about the 44 mis-gendered rows this exists to stop.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from nodes import _otr_roster_gender as RG

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BANK_DIR = REPO_ROOT / "config" / "source_banks" / "shakespeare"
SOURCES = BANK_DIR / "sources"
MANIFEST = BANK_DIR / "curated_scenes.sample.json"

# ARIEL and PUCK are deliberately left to the roll -- see the supplement's own
# comment. 40 of 42 is the ceiling, and asserting 42 would be unsatisfiable.
UNRESOLVABLE_BY_DESIGN = {"ARIEL", "PUCK"}


def _chars(stem: str):
    return RG.load_roster_characters(SOURCES / (stem + ".txt"))


def _manifest_scenes():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))["scenes"]


# ---------------------------------------------------------------- the ladder
def test_exact_tier_resolves_a_named_character():
    v = RG.resolve_roster_gender("MALVOLIO", _chars("twelfth_night__act2_scene5"))
    assert v.tier == "exact"
    assert v.gender == "male"
    assert v.evidence  # a quotable reason, not a bare flag


def test_qualified_tier_abstains_for_a_single_unknown_candidate():
    """ANTIPHOLUS matches ANTIPHOLUS OF EPHESUS; when the roster records him as
    gender='unknown' the tier matched but the source declined to gender him.
    (Synthetic rows: since the 2026-09-02 stamp the live sidecar carries the
    supplement's answer for this row, see the test below.)
    """
    rows = [{"name": "ANTIPHOLUS OF EPHESUS", "roster_name": "ANTIPHOLUS OF EPHESUS",
             "description": "a citizen of Ephesus", "gender": "unknown",
             "gender_source": "unknown"}]
    v = RG.resolve_roster_gender("ANTIPHOLUS", rows)
    assert v.tier == "qualified"
    assert v.gender == "unknown"


def test_both_dromios_abstain_rather_than_agreeing():
    """Pins the corrected fact. An earlier analysis claimed 'you do not need to
    know which Dromio to know he is a man' -- but BOTH Dromios are recorded
    gender='unknown', so the join has nothing to agree about and must abstain.
    """
    rows = [{"name": n, "roster_name": n, "description": "", "gender": "unknown",
             "gender_source": "unknown"} for n in ("DROMIO OF EPHESUS", "DROMIO OF SYRACUSE")]
    v = RG.resolve_roster_gender("DROMIO", rows)
    assert v.gender == "unknown"
    assert len(v.matched) == 2, v.matched
    # Since the 2026-09-02 stamp the LIVE rows carry the supplement's answer, and
    # two rows that AGREE are a pin, not an abstention.
    live = RG.resolve_roster_gender("DROMIO", _chars("comedy_errors__act3_scene1"))
    assert (live.gender, live.gender_source, len(live.matched)) == ("male", "supplement", 2)


def test_disagreeing_candidates_are_reported_as_ambiguous():
    synthetic = [
        {"name": "TWIN OF EPHESUS", "gender": "male"},
        {"name": "TWIN OF SYRACUSE", "gender": "female"},
    ]
    v = RG.resolve_roster_gender("TWIN", synthetic)
    assert v.gender == "unknown"
    assert v.evidence == "ambiguous_join"


def test_a_name_absent_from_the_roster_does_not_match():
    v = RG.resolve_roster_gender("PUCK", _chars("midsummer__act3_scene1"))
    assert v.gender == "unknown"
    assert v.tier == "none"


def test_missing_sidecar_returns_empty_rather_than_raising():
    assert RG.load_roster_characters(SOURCES / "no_such_source.txt") == ()


def test_roster_rows_are_plain_json_serializable_dicts():
    """_copy_sidecar is a shallow dict(), so these alias into durable ledger meta
    and get json-dumped. A MappingProxyType or frozen dataclass breaks a render.
    """
    rows = _chars("twelfth_night__act2_scene5")
    assert rows and all(type(r) is dict for r in rows)
    json.dumps(rows)


# ------------------------------------------------------------ the supplement
def test_supplement_loads_and_every_entry_carries_evidence():
    supp = RG.load_gender_supplement(BANK_DIR)
    assert supp
    total = 0
    for play_code, names in supp.items():
        for name, spec in names.items():
            total += 1
            assert spec["gender"] in ("male", "female"), (play_code, name)
            assert spec["evidence"].strip(), (play_code, name)
    assert total == 12, ("12 curated entries (LUCE and BALTHASAR of Comedy of Errors "
                         "added 2026-09-02); ARIEL and PUCK are excluded by design")


def test_supplement_may_not_overrule_a_confirmed_sidecar_gender():
    chars = _chars("twelfth_night__act2_scene5")
    hostile = {"TN": {"MALVOLIO": {"gender": "female", "evidence": "wrong on purpose"}}}
    with pytest.raises(RG.RosterGenderError):
        RG.resolve_with_supplement(
            "MALVOLIO", chars, play_code="TN", supplement=hostile)


def test_supplement_closes_the_names_the_roster_cannot_reach():
    """Two layers, both live. The stamper (2026-09-02) writes the supplement's
    answer INTO the sidecar row (gender_source 'supplement'), so the roster join
    alone now pins ANTIPHOLUS and DROMIO; and on a roster that still says unknown,
    the render-time supplement closes the same names."""
    chars = _chars("comedy_errors__act3_scene1")
    supp = RG.load_gender_supplement(BANK_DIR)
    for name in ("ANTIPHOLUS", "DROMIO"):
        v = RG.resolve_roster_gender(name, chars)
        assert (v.gender, v.gender_source, v.gender_confidence) == ("male", "supplement", "known"), name
    unknown_rows = [dict(r, gender="unknown", gender_source="unknown") for r in chars]
    for name in ("ANTIPHOLUS", "DROMIO"):
        assert RG.resolve_roster_gender(name, unknown_rows).gender == "unknown"
        v = RG.resolve_with_supplement(
            name, unknown_rows, play_code="Err", supplement=supp)
        assert v.gender == "male", name
        assert v.tier == "supplement"
        assert (v.gender_source, v.gender_confidence) == ("supplement", "known")


def test_supplement_does_not_adjudicate_an_ambiguous_join():
    synthetic = [
        {"name": "TWIN OF EPHESUS", "gender": "male"},
        {"name": "TWIN OF SYRACUSE", "gender": "female"},
    ]
    supp = {"Err": {"TWIN": {"gender": "male", "evidence": "should not win"}}}
    v = RG.resolve_with_supplement("TWIN", synthetic, play_code="Err", supplement=supp)
    assert v.evidence == "ambiguous_join"


# ------------------------------------------------------------- whole corpus
def test_every_shipped_cast_hint_resolves_except_the_two_left_to_the_operator():
    """The corpus gate. 42 shipped hints; 40 must resolve to male|female.

    Deleting the supplement drops this to 30 and fails on ten named hints.
    """
    supp = RG.load_gender_supplement(BANK_DIR)
    resolved, unresolved = [], []
    for scene in _manifest_scenes():
        chars = RG.load_roster_characters(BANK_DIR / scene["text_path"])
        for hint in scene["cast_hints"]:
            v = RG.resolve_with_supplement(
                hint, chars, play_code=scene["play_code"], supplement=supp)
            (resolved if v.is_pinned else unresolved).append(hint.upper())

    assert len(resolved) + len(unresolved) == 42
    assert set(unresolved) == UNRESOLVABLE_BY_DESIGN, unresolved
    assert len(resolved) == 40


def test_the_named_production_defects_now_resolve_correctly():
    """The eight characters measured as mis-gendered across published ledgers."""
    supp = RG.load_gender_supplement(BANK_DIR)
    expected = {
        ("twelfth_night__act2_scene5", "TN", "MALVOLIO"): "male",
        ("twelfth_night__act2_scene5", "TN", "MARIA"): "female",
        ("romeo_juliet__act2_scene2", "Rom", "ROMEO"): "male",
        ("romeo_juliet__act2_scene2", "Rom", "JULIET"): "female",
        ("tempest__act1_scene2", "Tmp", "MIRANDA"): "female",
        ("as_you_like_it__act3_scene2", "AYL", "CELIA"): "female",
        ("as_you_like_it__act3_scene2", "AYL", "ROSALIND"): "female",
        ("king_lear__act1_scene1", "Lr", "LEAR"): "male",
    }
    for (stem, play_code, name), want in expected.items():
        v = RG.resolve_with_supplement(
            name, _chars(stem), play_code=play_code, supplement=supp)
        assert v.gender == want, (name, v)


def test_gender_map_omits_names_it_could_not_resolve():
    supp = RG.load_gender_supplement(BANK_DIR)
    chars = _chars("midsummer__act3_scene1")
    got = RG.gender_map_for_names(
        ["Bottom", "Titania", "Puck"], chars, play_code="MND", supplement=supp)
    assert "PUCK" not in got, "an unresolved name must be omitted, never guessed"
    assert got["BOTTOM"]["gender"] == "male"
    assert got["TITANIA"]["gender"] == "female"
    for spec in got.values():
        assert spec["evidence"] and spec["tier"] and spec["roster_name"]


def test_supplement_dir_is_none_for_a_bank_with_no_manifest():
    class _Bank:
        defaults = {}

    assert RG.supplement_dir_for_bank(_Bank()) is None
    assert RG.load_gender_supplement(None) == {}
