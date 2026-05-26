"""Sprint 10A step 4 regression -- Stage 1 cast audit.

Covers:
  * Name-gender oracle:
      - Curated NAME_GENDER table covers every entry of
        cast_pools.FIRST_NAMES (no name slips through to
        'unknown' for a name we KNOW the project uses).
      - lookup_first_name_gender handles full names ('FIRST LAST'),
        empty strings, None-equivalents, punctuated names.
      - is_inversion respects unisex + unknown carveouts.
  * Cast audit:
      - Happy path: a clean plan produces ok=True, 0 errors.
      - Each error code fires under the right input.
      - Named regressions from real soak runs:
          * ROBINSON VOSS  -> female  (operator-flagged 2026-05-26)
          * COLE           -> female  (run 2 of prior session)
          * MIRA           -> male    (run 2 of prior session)
          * REGINALD       -> female  (run 1 of prior session)
          * OSCAR          -> female  (run 3 of prior session)
        Each is a named test method so a soak diff can grep the
        specific case.
      - Soft warn for unknown names does NOT count toward errors.
      - Audit raises NO exception on edge cases (empty cast, all
        unisex, etc.).
  * Voice gender lookup:
      - The hard-coded fallback _FALLBACK_VOICE_GENDER stays in
        sync with cast_pools.VOICE_PROFILES.
"""

from __future__ import annotations

import pytest

from nodes._otr_name_gender import (
    NAME_GENDER,
    is_inversion,
    lookup_first_name_gender,
)
from nodes._otr_stage1_cast_audit import (
    CODE_GENDER_PRONOUNS_MISMATCH,
    CODE_GENDER_VOICE_MISMATCH,
    CODE_NAME_GENDER_MISMATCH,
    CODE_NAME_UNKNOWN_SOFT,
    CODE_VOICE_UNKNOWN,
    CastAuditFinding,
    CastAuditResult,
    _FALLBACK_VOICE_GENDER,
    _build_voice_gender_lookup,
    audit_cast,
)
from nodes._otr_stage1_plan import (
    Stage1Arc,
    Stage1Beat,
    Stage1CastMember,
    Stage1Plan,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _good_plan() -> Stage1Plan:
    """A canonical clean plan: 2 cast members, no inversions, no warns."""
    return Stage1Plan(
        premise="Two technicians at a Mars relay catch a signal that should not exist.",
        arc=Stage1Arc(
            setup="Routine night shift, calm comms traffic.",
            complication="An off-band carrier locks in and refuses to drop.",
            resolution="They identify the source and decide whether to acknowledge it.",
        ),
        cast=[
            Stage1CastMember(
                name="DALE PORTER",
                gender="male",
                pronouns="he/him",
                voice_id="v2/en_speaker_5",
                persona="Experienced relay technician with a measured cadence.",
                arc_role="anchor of competence",
            ),
            Stage1CastMember(
                name="MEG SAARI",
                gender="female",
                pronouns="she/her",
                voice_id="v2/en_speaker_2",
                persona="Junior tech, six months in, quick to question.",
                arc_role="pressure source",
            ),
        ],
        beats=[
            Stage1Beat(
                beat_id="b001", speaker="ANNOUNCER",
                intent="Open the episode.", length_target_words=18,
                emotional_register="neutral", callback_to=None,
            ),
            Stage1Beat(
                beat_id="b002", speaker="DALE PORTER",
                intent="Set the scene.", length_target_words=24,
                emotional_register="calm", callback_to=None,
            ),
            Stage1Beat(
                beat_id="b003", speaker="MEG SAARI",
                intent="Notice the signal.", length_target_words=22,
                emotional_register="alert", callback_to="b002",
            ),
        ],
        running_facts=["The relay is on Mars."],
    )


def _swap_cast(plan: Stage1Plan, **member0_overrides) -> Stage1Plan:
    """Return a clone of `plan` with cast[0] field overrides applied."""
    base = plan.cast[0].model_dump()
    base.update(member0_overrides)
    new_cast = [Stage1CastMember(**base)] + list(plan.cast[1:])
    data = plan.model_dump()
    data["cast"] = [c.model_dump() if hasattr(c, "model_dump") else c
                    for c in new_cast]
    return Stage1Plan(**data)


# ---------------------------------------------------------------------------
# Name-gender oracle
# ---------------------------------------------------------------------------


class TestNameGenderOracle:
    def test_robinson_is_male(self):
        """The operator-flagged ROBINSON case must resolve to male."""
        assert lookup_first_name_gender("ROBINSON VOSS") == "male"
        assert lookup_first_name_gender("Robinson") == "male"
        assert lookup_first_name_gender("robinson") == "male"

    def test_classic_female_names(self):
        assert lookup_first_name_gender("Alice Carter") == "female"
        assert lookup_first_name_gender("MARGOT") == "female"
        assert lookup_first_name_gender("Anya") == "female"

    def test_unisex_names(self):
        assert lookup_first_name_gender("Quinn Stone") == "unisex"
        assert lookup_first_name_gender("Ren") == "unisex"
        assert lookup_first_name_gender("Sora Vance") == "unisex"

    def test_unknown_name_returns_unknown(self):
        assert lookup_first_name_gender("Zog Frellnax") == "unknown"
        assert lookup_first_name_gender("Xylvanik") == "unknown"

    def test_empty_and_garbage_returns_unknown(self):
        assert lookup_first_name_gender("") == "unknown"
        assert lookup_first_name_gender("   ") == "unknown"
        assert lookup_first_name_gender(None) == "unknown"  # type: ignore[arg-type]
        assert lookup_first_name_gender(42) == "unknown"  # type: ignore[arg-type]

    def test_punctuated_first_name_stripped(self):
        assert lookup_first_name_gender("'Dale' Porter") == "male"
        assert lookup_first_name_gender("Alice, Carter") == "female"

    def test_is_inversion_unisex_carveout(self):
        # Unisex name + either gender -> not an inversion.
        assert is_inversion("unisex", "male") is False
        assert is_inversion("unisex", "female") is False

    def test_is_inversion_unknown_carveout(self):
        # Unknown name -> not an inversion regardless of LLM choice.
        assert is_inversion("unknown", "male") is False
        assert is_inversion("unknown", "female") is False

    def test_is_inversion_male_vs_female(self):
        assert is_inversion("male", "female") is True
        assert is_inversion("female", "male") is True

    def test_is_inversion_match_is_clean(self):
        assert is_inversion("male", "male") is False
        assert is_inversion("female", "female") is False


# ---------------------------------------------------------------------------
# Coverage: every name in cast_pools.FIRST_NAMES has a NAME_GENDER row
# ---------------------------------------------------------------------------


class TestNameGenderCoverage:
    def test_every_first_name_is_covered(self):
        """If a name lives in the project's pool, the audit must
        be able to look it up. Drift between the pool and the
        oracle is a maintenance bug that lets gender inversions
        slip past the audit silently."""
        try:
            from config import cast_pools
        except ImportError:
            pytest.skip("config.cast_pools not importable in this env")
        missing = []
        for name in cast_pools.FIRST_NAMES:
            key = name.strip().lower()
            if key not in NAME_GENDER:
                missing.append(name)
        assert missing == [], (
            f"{len(missing)} cast_pools.FIRST_NAMES entries are missing "
            f"from nodes/_otr_name_gender.NAME_GENDER. Update the oracle "
            f"so the audit can catch inversions on these names. Missing: "
            f"{missing}"
        )


# ---------------------------------------------------------------------------
# Voice-gender fallback consistency
# ---------------------------------------------------------------------------


class TestVoiceGenderFallback:
    def test_fallback_matches_live_cast_pools(self):
        """_FALLBACK_VOICE_GENDER is hard-coded for sandbox / test
        contexts where the config package can't be imported. Pin
        it to match the live cast_pools.VOICE_PROFILES so the audit
        never drifts on the wrong oracle."""
        try:
            from config import cast_pools
        except ImportError:
            pytest.skip("config.cast_pools not importable in this env")
        live = {
            preset: gender
            for preset, gender, _lang, _tags in cast_pools.VOICE_PROFILES
        }
        assert live == _FALLBACK_VOICE_GENDER, (
            f"_FALLBACK_VOICE_GENDER in nodes/_otr_stage1_cast_audit.py "
            f"has drifted from cast_pools.VOICE_PROFILES. Update the "
            f"fallback to match. live={live!r} fallback="
            f"{_FALLBACK_VOICE_GENDER!r}"
        )

    def test_builder_returns_voice_gender_dict(self):
        d = _build_voice_gender_lookup()
        assert isinstance(d, dict)
        assert "v2/en_speaker_5" in d
        assert d["v2/en_speaker_5"] in ("male", "female")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_clean_plan_ok_zero_errors(self):
        result = audit_cast(_good_plan())
        assert result.ok is True
        assert len(result.errors) == 0

    def test_to_dict_serializes(self):
        d = audit_cast(_good_plan()).to_dict()
        assert d["ok"] is True
        assert d["error_count"] == 0
        assert "findings" in d


# ---------------------------------------------------------------------------
# Named regression fixtures from real soak runs
# ---------------------------------------------------------------------------


class TestNamedRegressions:
    """Each test pins a real-world gender inversion the operator
    flagged in a soak run. Step 4's job is to catch every one of
    these as a name_gender_mismatch error."""

    def test_robinson_voss_female_caught(self):
        """Operator-flagged 2026-05-26 (pending_20260526_114111).
        LLM cast 'ROBINSON VOSS' as gender=female with
        voice_id=v2/en_speaker_7. Robinson is canonically male.
        Audit must fire name_gender_mismatch.
        """
        plan = _swap_cast(
            _good_plan(),
            name="ROBINSON VOSS",
            gender="female",
            pronouns="she/her",
            voice_id="v2/en_speaker_7",
        )
        result = audit_cast(plan)
        codes = {f.code for f in result.errors}
        assert CODE_NAME_GENDER_MISMATCH in codes, (
            f"ROBINSON VOSS=female must trigger name_gender_mismatch; "
            f"got error codes {codes!r}"
        )
        finding = next(
            f for f in result.errors
            if f.code == CODE_NAME_GENDER_MISMATCH
            and f.char_name == "ROBINSON VOSS"
        )
        assert finding.expected == "male"
        assert finding.got == "female"
        assert result.ok is False

    def test_cole_female_caught(self):
        """Prior session run 2: COLE cast as female. Cole is
        canonically male."""
        plan = _swap_cast(
            _good_plan(),
            name="COLE",
            gender="female",
            pronouns="she/her",
            voice_id="v2/en_speaker_2",
        )
        result = audit_cast(plan)
        assert any(
            f.code == CODE_NAME_GENDER_MISMATCH and f.char_name == "COLE"
            for f in result.errors
        )

    def test_mira_male_caught(self):
        """Prior session run 2: MIRA cast as male. Mira is
        canonically female."""
        plan = _swap_cast(
            _good_plan(),
            name="MIRA",
            gender="male",
            pronouns="he/him",
            voice_id="v2/en_speaker_5",
        )
        result = audit_cast(plan)
        assert any(
            f.code == CODE_NAME_GENDER_MISMATCH and f.char_name == "MIRA"
            for f in result.errors
        )

    def test_reginald_female_caught(self):
        """Prior session run 1: REGINALD cast as female. Reginald
        is canonically male."""
        plan = _swap_cast(
            _good_plan(),
            name="REGINALD CARTER",
            gender="female",
            pronouns="she/her",
            voice_id="v2/en_speaker_2",
        )
        result = audit_cast(plan)
        assert any(
            f.code == CODE_NAME_GENDER_MISMATCH
            and f.char_name == "REGINALD CARTER"
            for f in result.errors
        )

    def test_oscar_female_caught(self):
        """Prior session run 3: OSCAR cast as female. Oscar is
        canonically male."""
        plan = _swap_cast(
            _good_plan(),
            name="OSCAR SIRIKIT",
            gender="female",
            pronouns="she/her",
            voice_id="v2/en_speaker_2",
        )
        result = audit_cast(plan)
        assert any(
            f.code == CODE_NAME_GENDER_MISMATCH
            and f.char_name == "OSCAR SIRIKIT"
            for f in result.errors
        )


# ---------------------------------------------------------------------------
# Each error code under a clean trigger
# ---------------------------------------------------------------------------


class TestErrorCodes:
    def test_gender_pronouns_mismatch(self):
        # 'male' canonical but pronouns 'she/her'
        plan = _swap_cast(
            _good_plan(),
            name="DALE PORTER",
            gender="male",
            pronouns="she/her",   # mismatch
            voice_id="v2/en_speaker_5",
        )
        result = audit_cast(plan)
        assert any(
            f.code == CODE_GENDER_PRONOUNS_MISMATCH for f in result.errors
        )

    def test_gender_voice_mismatch(self):
        # gender=male but voice is the v2/en_speaker_2 (female)
        plan = _swap_cast(
            _good_plan(),
            name="DALE PORTER",
            gender="male",
            pronouns="he/him",
            voice_id="v2/en_speaker_2",   # female-coded
        )
        result = audit_cast(plan)
        codes = {f.code for f in result.errors}
        assert CODE_GENDER_VOICE_MISMATCH in codes

    def test_voice_unknown_caught_when_schema_lets_through(self):
        """A voice_id outside the v2/en_speaker_[0-9] pool is rejected
        at Stage1Plan schema parse time, so the audit's voice_unknown
        branch is a backstop -- exercise it via the dict-mutation
        path that bypasses pydantic re-validation."""
        plan = _good_plan()
        # Bypass the pydantic re-construction: mutate the in-memory
        # object directly. This simulates a future code path that
        # might construct a Stage1Plan from a partial dict.
        plan.cast[0].voice_id = "v2/en_speaker_42"  # not in the pool
        result = audit_cast(plan)
        codes = {f.code for f in result.errors}
        assert CODE_VOICE_UNKNOWN in codes

    def test_name_unknown_is_warn_not_error(self):
        """A name the LLM invented from outside the pool should
        produce a SOFT warn, NOT an error -- the soak gate keys off
        errors only, and we don't want to fail every novel cast on
        a curation gap."""
        plan = _swap_cast(
            _good_plan(),
            name="ZOG FRELLNAX",        # not in NAME_GENDER
            gender="male",
            pronouns="he/him",
            voice_id="v2/en_speaker_5",
        )
        result = audit_cast(plan)
        # No error finding for the unknown name itself.
        unknown_errors = [
            f for f in result.errors
            if f.char_name == "ZOG FRELLNAX"
        ]
        assert unknown_errors == [], (
            f"unknown name should produce no error findings; got "
            f"{[f.code for f in unknown_errors]}"
        )
        # But a soft warn IS expected.
        unknown_warns = [
            f for f in result.warns
            if f.char_name == "ZOG FRELLNAX"
            and f.code == CODE_NAME_UNKNOWN_SOFT
        ]
        assert len(unknown_warns) == 1
        # ok is True (errors only count for the gate)
        assert result.ok is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unisex_name_does_not_fire_inversion(self):
        """Unisex names accept either gender from the LLM."""
        plan = _swap_cast(
            _good_plan(),
            name="QUINN STONE",
            gender="female",
            pronouns="she/her",
            voice_id="v2/en_speaker_2",
        )
        result = audit_cast(plan)
        # No name_gender_mismatch for QUINN
        gender_findings = [
            f for f in result.errors
            if f.code == CODE_NAME_GENDER_MISMATCH and f.char_name == "QUINN STONE"
        ]
        assert gender_findings == []

    def test_nonbinary_does_not_fire_voice_mismatch(self):
        """Nonbinary characters get whichever Bark voice the LLM
        picked -- the binary v2 pool can't represent nonbinary, so
        the audit must not flag this as gender_voice_mismatch."""
        plan = _swap_cast(
            _good_plan(),
            name="QUINN STONE",
            gender="nonbinary",
            pronouns="they/them",
            voice_id="v2/en_speaker_5",     # male-coded in Bark pool
        )
        result = audit_cast(plan)
        voice_mismatches = [
            f for f in result.errors
            if f.code == CODE_GENDER_VOICE_MISMATCH
        ]
        assert voice_mismatches == [], (
            f"nonbinary character should not fire gender_voice_mismatch; "
            f"got {[f.message for f in voice_mismatches]}"
        )

    def test_audit_swallows_no_exceptions(self):
        """The audit is on a non-fatal path -- it must never raise
        out, even on degenerate input."""
        # Trigger every check; should produce findings, not raise.
        plan = _swap_cast(
            _good_plan(),
            name="ROBINSON VOSS",       # name_gender_mismatch
            gender="female",
            pronouns="he/him",          # gender_pronouns_mismatch
            voice_id="v2/en_speaker_5", # male-coded -> gender_voice_mismatch
        )
        result = audit_cast(plan)
        codes = {f.code for f in result.errors}
        assert CODE_NAME_GENDER_MISMATCH in codes
        assert CODE_GENDER_PRONOUNS_MISMATCH in codes
        assert CODE_GENDER_VOICE_MISMATCH in codes
        # And the result is well-formed.
        assert isinstance(result, CastAuditResult)
        assert result.ok is False
