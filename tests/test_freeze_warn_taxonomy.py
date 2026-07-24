"""Deterministic freeze-warning telemetry never controls publication."""

from nodes._otr_freeze_cascade import (
    FREEZE_WARN_TIERS,
    build_freeze_warn_taxonomy,
    classify_freeze_warning,
)


def test_tiers_are_stable():
    assert FREEZE_WARN_TIERS == (
        "structural_error",
        "story_accuracy_warning",
        "cosmetic_warning",
    )


def test_warning_classification():
    assert classify_freeze_warning("missing speaker for beat b004") == "structural_error"
    assert classify_freeze_warning("canon divergence on setting") == "story_accuracy_warning"
    assert classify_freeze_warning("title casing looks off") == "cosmetic_warning"


def test_structural_wins_when_terms_overlap():
    assert (
        classify_freeze_warning("continuity: beat b004 has no voiced text")
        == "structural_error"
    )


def test_taxonomy_buckets_gap_and_consistency_telemetry():
    tax = build_freeze_warn_taxonomy(
        gap_warnings=[
            "minor spacing nit",
            "a continuity hiccup",
            "no voiced text on l002",
        ],
        consistency_status={
            "clean": False,
            "defects": [{"field": "sound_palette"}],
        },
    )
    assert "no voiced text on l002" in " ".join(tax["structural_error"])
    assert "continuity hiccup" in " ".join(tax["story_accuracy_warning"])
    assert "sound_palette" in " ".join(tax["story_accuracy_warning"])
    assert "minor spacing nit" in " ".join(tax["cosmetic_warning"])


def test_taxonomy_is_empty_for_clean_input():
    tax = build_freeze_warn_taxonomy(
        gap_warnings=[],
        consistency_status={"clean": True},
    )
    assert all(tax[tier] == [] for tier in FREEZE_WARN_TIERS)


def test_taxonomy_tolerates_bad_diagnostic_shapes():
    tax = build_freeze_warn_taxonomy(
        gap_warnings=None,
        consistency_status=123,
    )
    assert set(tax) == set(FREEZE_WARN_TIERS)
