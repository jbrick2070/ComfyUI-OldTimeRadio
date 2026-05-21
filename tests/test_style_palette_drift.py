"""S25 / MG-6 (BUG-LOCAL-216): pin the three style-slug surfaces to a
single source of truth.

Background: writer pool (OTR_LedgerScriptWriter._STYLE_PICKER_SEED_POOL)
and music palette (musicgen_theme._STYLE_PALETTE) used to be maintained
as two parallel lists in two files. Drift caused MusicGen to halt
mid-pipeline -- after the writer + freeze cascade had already spent
time and tokens.

Hoisting to nodes/_otr_style_palette.py fixed the root cause. This
test pins the palette and the writer seed pool to set-equality with
KNOWN_STYLE_SLUGS so any future drift fires at unit-test time, not
soak time. The freeze validator is no longer set-pinned: BUG-LOCAL-240
relaxed it to validate slug SHAPE (snake_case) rather than membership,
because the style picker invents new slugs by design.
"""
from __future__ import annotations

from nodes._otr_style_palette import KNOWN_STYLE_SLUGS, STYLE_PALETTE


def test_palette_matches_known_slugs():
    """STYLE_PALETTE keys are the canonical slug set."""
    assert set(STYLE_PALETTE.keys()) == set(KNOWN_STYLE_SLUGS)


def test_writer_pool_matches_known_slugs():
    """OTR_LedgerScriptWriter._STYLE_PICKER_SEED_POOL must match the
    canonical slug set. Drift here is the original BUG-LOCAL-216
    failure mode -- writer emits a slug the palette doesn't cover."""
    from nodes.OTR_LedgerScriptWriter import _STYLE_PICKER_SEED_POOL
    assert set(_STYLE_PICKER_SEED_POOL) == set(KNOWN_STYLE_SLUGS), (
        "writer pool drift; one source of truth violated"
    )


def test_every_palette_entry_has_all_three_cues():
    """Each style entry must carry opening / closing / interstitial.
    Missing cue -> musicgen falls through to an exception path during
    cue-prompt resolution."""
    for slug, cues in STYLE_PALETTE.items():
        assert set(cues.keys()) == {"opening", "closing", "interstitial"}, (
            f"{slug} missing cue keys"
        )


def test_freeze_validator_rejects_malformed_slug():
    """BUG-LOCAL-240 relaxed the freeze slug check from KNOWN_STYLE_SLUGS
    membership to snake_case SHAPE. A genuinely malformed slug
    (uppercase, spaces) is still real writer drift and must still be
    rejected."""
    from nodes._otr_ledger_freeze import _check_meta_invariants

    malformed = "Mission Control Procedural"  # spaces + capitals
    ledger_data = {
        "schema_version": "l3-2026-05-14",
        "meta": {
            "gen_params_initial": {"style": malformed},
        },
    }
    errors: list[str] = []
    warnings: list[str] = []
    _check_meta_invariants(ledger_data, errors, warnings)
    assert any("well-formed snake_case" in e for e in errors), (
        f"freeze validator did not reject malformed slug; errors={errors!r}"
    )


def test_freeze_validator_accepts_invented_snake_case_slug():
    """BUG-LOCAL-240: the style picker's "let the story decide" sentinel
    invents NEW snake_case slugs from the story content. A well-formed
    invented slug that is NOT in the seed palette must NOT be flagged as
    writer drift -- inventing slugs is the intended feature."""
    from nodes._otr_ledger_freeze import _check_meta_invariants

    invented = "geosynchronous_orbital_rivalry"
    assert invented not in KNOWN_STYLE_SLUGS  # genuinely outside the palette
    ledger_data = {
        "schema_version": "l3-2026-05-14",
        "meta": {
            "gen_params_initial": {"style": invented},
        },
    }
    errors: list[str] = []
    warnings: list[str] = []
    _check_meta_invariants(ledger_data, errors, warnings)
    assert not any("snake_case" in e for e in errors), (
        f"freeze validator falsely rejected invented snake_case slug "
        f"{invented!r}; errors={errors!r}"
    )


def test_freeze_validator_accepts_known_slug():
    """Sanity: a slug that IS in the seed palette must NOT trigger the
    style-slug error path."""
    from nodes._otr_ledger_freeze import _check_meta_invariants

    good_slug = next(iter(KNOWN_STYLE_SLUGS))
    ledger_data = {
        "schema_version": "l3-2026-05-14",
        "meta": {
            "gen_params_initial": {"style": good_slug},
        },
    }
    errors: list[str] = []
    warnings: list[str] = []
    _check_meta_invariants(ledger_data, errors, warnings)
    assert not any("snake_case" in e for e in errors), (
        f"freeze validator falsely rejected known slug {good_slug!r}; "
        f"errors={errors!r}"
    )
