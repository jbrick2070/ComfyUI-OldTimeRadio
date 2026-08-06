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
    get_presentation_gender,
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


# --------------------------------------------------------------------------- #
# chunk 3 -- the scifi_codex lane allocates bark presets by gender


def test_bark_pools_are_gendered_and_non_empty():
    from nodes._otr_scifi_codex import _bark_preset_pools

    male, female = _bark_preset_pools()
    assert male and female, "both bark gender columns must be populated"
    assert not set(male) & set(female), "a preset cannot be in both columns"
    assert all(p.startswith("v2/") for p in male + female)


def test_bark_pools_come_from_cast_pools_not_a_hand_kept_copy():
    """A second copy of the gender column is how the two drift apart."""
    from config import cast_pools

    from nodes._otr_scifi_codex import _bark_preset_pools

    male, female = _bark_preset_pools()
    expected_male = [p for p, g, _l, _t in cast_pools.VOICE_PROFILES
                     if g.strip().lower() == "male"]
    expected_female = [p for p, g, _l, _t in cast_pools.VOICE_PROFILES
                       if g.strip().lower() == "female"]
    assert male == expected_male
    assert female == expected_female


def test_the_hardcoded_all_male_triple_is_gone():
    """Every sampled episode on this lane shipped v2/en_speaker_6 / _3 / _0 --
    all three male -- keyed on char_id with the row's gender never read."""
    import inspect

    from nodes import _otr_scifi_codex

    src = inspect.getsource(_otr_scifi_codex._assemble_ledger)
    assert '{"c01": "v2/en_speaker_6"' not in src, (
        "the hardcoded per-char_id preset map must not survive")
    assert "canonical_bank_gender" in src, (
        "preset choice must consult the row's gender")


def test_other_gender_presentation_pick_is_seed_stable():
    """The pick is keyed on char_id via sha1, never hash() -- Python's builtin
    hash is PYTHONHASHSEED-salted, so it would hand the same character
    different voices from run to run."""
    import hashlib
    import inspect

    from nodes import _otr_scifi_codex

    src = inspect.getsource(_otr_scifi_codex._assemble_ledger)
    # Strip comments first -- the code's own comment explains why hash() is
    # forbidden, and a naive substring search matches that explanation.
    code_only = "\n".join(
        line.split("#", 1)[0] for line in src.splitlines())
    assert "sha1" in code_only
    assert "hash(" not in code_only, "builtin hash() is PYTHONHASHSEED-salted"

    def pick(char_id):
        d = hashlib.sha1(str(char_id).encode("utf-8")).hexdigest()
        return "male" if int(d, 16) % 2 == 0 else "female"

    assert pick("c01") == pick("c01")
    assert pick("c02") == pick("c02")


# --------------------------------------------------------------------------- #
# presentation_gender -- chunk 4


def test_presentation_gender_prefers_the_stamped_value():
    """The stamped value is what the audience actually HEARD."""
    row = {"gender": "male", "presentation_gender": "female"}
    assert get_presentation_gender(row) == "female"


def test_presentation_gender_falls_back_to_the_normalized_label():
    """Legacy: 1,595 ledgers were frozen before the field existed. They are read
    through, never treated as violations and never rewritten."""
    assert get_presentation_gender({"gender": "woman"}) == "female"
    assert get_presentation_gender({"gender": "male"}) == "male"
    assert get_presentation_gender({"gender": "non-binary"}) == "other"


def test_presentation_gender_treats_empty_stamp_as_absent():
    """A row the caster never stamped must not read as 'presents as nothing'."""
    assert get_presentation_gender({"gender": "female",
                                    "presentation_gender": ""}) == "female"


def test_presentation_gender_survives_set_cast(tmp_path):
    """THE persistence test the earlier plan lacked. set_cast rebuilds a FIXED
    row and drops every key it does not name -- a field written upstream and
    omitted there evaporates on the next save."""
    from nodes.production_ledger import Ledger

    led = Ledger(episode_id="EP-GENDER-TEST", out_dir=str(tmp_path))
    led.set_cast([{
        "char_id": "c01", "name": "SCROOGE",
        "character_description": "a miser", "gender": "female",
        "presentation_gender": "female", "tts_model": "bark",
        "voice_preset": "v2/en_speaker_9",
    }])
    row = led.data["cast"][0]
    assert "presentation_gender" in row, (
        "set_cast dropped presentation_gender -- the field would evaporate on save"
    )
    assert row["presentation_gender"] == "female"


def test_set_cast_omits_presentation_gender_when_unstamped(tmp_path):
    """Carried conditionally, like provider_voice_id: a writer-stage row nobody
    has cast yet stays byte-identical to the legacy cast-row contract, so the
    drift guard in test_fable2_assembly keeps its teeth. Absence is read through
    by get_presentation_gender."""
    from nodes.production_ledger import Ledger

    led = Ledger(episode_id="EP-GENDER-TEST-2", out_dir=str(tmp_path))
    led.set_cast([{"char_id": "c01", "name": "X",
                   "character_description": "y", "gender": "male",
                   "tts_model": "bark", "voice_preset": "v2/en_speaker_0"}])
    row = led.data["cast"][0]
    assert "presentation_gender" not in row
    assert get_presentation_gender(row) == "male"


def test_castlock_stamp_records_the_delivered_reference_gender():
    """Stamped from the reference ACTUALLY chosen, not from the row's label --
    the announcer's reference is drawn from the episode seed and never reads its
    row's gender, so the label alone cannot answer for it."""
    import inspect

    from nodes import cast_lock

    src = inspect.getsource(cast_lock.CastLock._stamp)
    assert "presentation_gender" in src, (
        "_stamp is the one place every stamped row passes through; the field "
        "must be written there or the announcer and fallback rows miss it"
    )


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
