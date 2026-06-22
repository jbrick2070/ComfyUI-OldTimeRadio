"""Voice-casting architecture (B): each approved voice model must have a SOLID
voice library. compute_bank_coverage is the gate + the operator-facing report."""

from nodes._otr_voice_bank import (
    APPROVED_VOICE_ENGINES,
    compute_bank_coverage,
    load_voice_bank,
)


def test_every_approved_engine_meets_the_casting_floor():
    """HARD bar: each approved engine has >= 5 ADULT voices for BOTH male and
    female, so a worst-case same-gender 5-character cast casts with no reuse."""
    cov = compute_bank_coverage(cast_size=5)
    for eng in APPROVED_VOICE_ENGINES:
        assert eng in cov, f"{eng} missing from the bank"
        c = cov[eng]
        assert c["meets_floor"], (
            f"{eng} below the casting floor: adult_male={c['adult_male']} "
            f"adult_female={c['adult_female']} (need >= 5 each)")


def test_coverage_report_shape_and_gaps_surfaced():
    """The report carries the per-engine matrix + an informational `gaps` list so
    the operator can see thin spots (male-light etc.) without failing the suite."""
    cov = compute_bank_coverage()
    for eng, c in cov.items():
        assert set(c) >= {"total", "by_gender", "by_gender_age", "adult_male",
                          "adult_female", "meets_floor", "gaps"}
        assert c["total"] >= 1
        assert isinstance(c["gaps"], list)
    # The current bank is known male-light; the report must SURFACE that (it is a
    # gap to remediate, not a hard failure).
    assert any("male-light" in g
               for c in cov.values() for g in c["gaps"]), \
        "expected the male-light imbalance to be surfaced as a gap"


def test_coverage_is_pure_and_deterministic():
    bank, _ = load_voice_bank()
    assert compute_bank_coverage(bank) == compute_bank_coverage(bank)
