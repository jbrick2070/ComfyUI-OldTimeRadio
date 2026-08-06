"""Item 8 chunk 1 -- the single gender normalization boundary.

The published corpus carries eighteen distinct gender strings across 5,123 cast
rows. Every consumer used to compare the RAW value against `male`/`female`, so a
row recorded as `woman` matched nothing: it got no portrait anchor and its voice
fell through to the gender-agnostic draw. These tests pin the boundary and the
two read sites that were provably wrong.
"""
import logging

import pytest

from nodes._otr_roster_gender import (
    VOICE_PORTRAIT_CONSISTENCY_POLICY_KEY,
    VOICE_PORTRAIT_CONSISTENCY_POLICY_REVISION,
    canonical_bank_gender,
    normalize_gender,
)
from nodes.otr_meta_brief_image_prompt import _ensure_gender_anchor


# --------------------------------------------------------------------------- #
# The normalizer


@pytest.mark.parametrize("raw", ["male", "Male", "MALE", " male ", "m", "M", "man"])
def test_male_synonyms_normalize(raw):
    assert normalize_gender(raw) == "male"


@pytest.mark.parametrize(
    "raw", ["female", "Female", "FEMALE", " female ", "f", "F", "woman"])
def test_female_synonyms_normalize(raw):
    assert normalize_gender(raw) == "female"


@pytest.mark.parametrize("raw", [
    "other", "non-binary", "nonbinary", "neutral", "unspecified", "any",
    "unknown", "n/a", "various", "artificial", "ai", "synthetic",
    "genderfluid", "child-like",
])
def test_every_corpus_nonbinary_token_maps_to_other(raw):
    """All 18 live strings are accounted for -- an unlisted one is a defect."""
    assert normalize_gender(raw) == "other"


def test_blank_is_other_in_legacy_mode():
    assert normalize_gender("") == "other"
    assert normalize_gender(None) == "other"


def test_strict_rejects_blank_and_unlisted():
    """A policy-revision producer writing an unresolvable gender is a defect."""
    with pytest.raises(ValueError):
        normalize_gender("", strict=True)
    with pytest.raises(ValueError):
        normalize_gender("wombat", strict=True)


def test_legacy_maps_unlisted_to_other_and_warns(caplog):
    import nodes._otr_roster_gender as rg
    rg._UNMAPPED_SEEN.discard("wombat")
    with caplog.at_level(logging.WARNING, logger="OTR"):
        assert normalize_gender("wombat") == "other"
    assert any("wombat" in r.getMessage() for r in caplog.records)


def test_policy_revision_is_an_integer_and_has_a_durable_key():
    """`cast_lock_revision` cannot substitute -- it increments per RUN."""
    assert isinstance(VOICE_PORTRAIT_CONSISTENCY_POLICY_REVISION, int)
    assert VOICE_PORTRAIT_CONSISTENCY_POLICY_REVISION >= 1
    assert VOICE_PORTRAIT_CONSISTENCY_POLICY_KEY == (
        "voice_portrait_consistency_policy_revision")


def test_normalizer_module_stays_import_light():
    """It is the LEAF every consumer calls; a heavy import here risks a cycle."""
    import nodes._otr_roster_gender as rg
    src = open(rg.__file__, encoding="utf-8").read()
    for forbidden in ("import torch", "from .cast_lock", "from ._otr_casting",
                      "from .otr_meta_brief_image_prompt"):
        assert forbidden not in src, f"{forbidden} would create an import cycle"


# --------------------------------------------------------------------------- #
# The portrait anchor -- the read site that was provably dropping rows


def test_woman_row_now_gets_an_anchor():
    """THE BUG: `woman` fell through the ("female","male") test unanchored."""
    out = _ensure_gender_anchor("a tall figure in a long coat",
                                {"gender": "woman"})
    assert out.startswith("adult woman,")


def test_man_row_now_gets_an_anchor():
    out = _ensure_gender_anchor("a tall figure in a long coat", {"gender": "man"})
    assert out.startswith("adult man,")


def test_title_case_still_works():
    """Regression guard: :81 already lower-cased, so this was never broken."""
    out = _ensure_gender_anchor("a tall figure", {"gender": "Male"})
    assert out.startswith("adult man,")


def test_other_row_gets_a_neutral_anchor_not_a_binary_one():
    # No subject noun of any kind in this prompt -- "figure"/"person"/"adult"
    # would already establish one and correctly suppress the anchor.
    out = _ensure_gender_anchor("seated at a cluttered desk, lamplight",
                                {"gender": "other"})
    assert out.startswith("person,")
    assert "woman" not in out.lower() and "adult man" not in out.lower()


def test_other_row_with_an_existing_subject_noun_is_left_alone():
    """`figure` already names a subject; stacking "person," would read badly."""
    already = "a tall figure in a long coat"
    assert _ensure_gender_anchor(already, {"gender": "other"}) == already


def test_anchor_is_not_stacked_when_already_present():
    already = "adult woman, seated at a desk"
    assert _ensure_gender_anchor(already, {"gender": "female"}) == already
    neutral = "a person at a desk"
    assert _ensure_gender_anchor(neutral, {"gender": "other"}) == neutral


def test_anchor_never_rejects_or_rewrites_a_prompt():
    """The portrait payload builder forbids any Python classifier from
    rejecting, rewriting or blocking a prompt. A contradicting cue is REPORTED
    by the audit, never gated here -- so the original text must survive whole."""
    contradictory = "a stern gentleman adjusting her collar"
    out = _ensure_gender_anchor(contradictory, {"gender": "female"})
    assert contradictory in out, "the authored text must survive unmodified"


def test_empty_prompt_is_left_alone():
    assert _ensure_gender_anchor("", {"gender": "female"}) == ""


# --------------------------------------------------------------------------- #
# canonical_bank_gender -- the VOICE side, which must not use the tri-state


@pytest.mark.parametrize("raw,expected", [
    ("woman", "female"), ("man", "male"), ("f", "female"), ("m", "male"),
    ("Woman", "female"), (" MAN ", "male"),
])
def test_bank_canonicalizer_fixes_the_synonyms(raw, expected):
    assert canonical_bank_gender(raw) == expected


def test_bank_canonicalizer_does_NOT_collapse_neutral():
    """The bank carries exactly one `neutral` reference (el_river) and ~27
    corpus rows are recorded `neutral`. The tri-state normalizer folds those
    into `other`, which would skip the one voice that actually fits them."""
    assert canonical_bank_gender("neutral") == "neutral"
    assert normalize_gender("neutral") == "other"
    assert canonical_bank_gender("neutral") != normalize_gender("neutral")


def test_bank_canonicalizer_keeps_blank_blank():
    """Callers short-circuit on a falsy gender. Mapping blank to `other` emptied
    the same-gender candidate tiers and dropped the not-already-in-use
    preference, which can hand two characters the same voice."""
    assert canonical_bank_gender("") == ""
    assert canonical_bank_gender(None) == ""
    assert not canonical_bank_gender(None)


def test_bank_canonicalizer_passes_unknown_values_through():
    """It is a synonym fixer, not a bucketer -- the bank owns its vocabulary."""
    assert canonical_bank_gender("androgynous") == "androgynous"
    assert canonical_bank_gender("other") == "other"


def test_the_two_helpers_are_not_interchangeable():
    """Regression guard for the actual mistake: using the tri-state normalizer
    on the voice path erased a bank-servable gender."""
    for raw in ("woman", "man", "male", "female"):
        assert canonical_bank_gender(raw) == normalize_gender(raw)
    assert canonical_bank_gender("neutral") != normalize_gender("neutral")


def test_castlock_char_casting_uses_the_bank_canonicalizer():
    """THE path that runs. CastLock stamps voice_ref_id before any render, so
    the render-time resolvers never reach their own gender fallback -- fixing
    only those left the defect live."""
    import inspect

    from nodes import cast_lock

    src = inspect.getsource(cast_lock.CastLock._auto_registry)
    assert "canonical_bank_gender" in src, (
        "_auto_registry must canonicalize gender before assign_voice_for_slot; "
        "a row recorded 'woman' otherwise raises and takes the "
        "gender-agnostic draw"
    )
    assert 'str(entry.get("gender") or "").strip().lower()' not in src, (
        "the raw-gender comparison is the defect; it must not survive"
    )
