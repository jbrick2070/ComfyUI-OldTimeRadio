"""S25 / MG-6 (BUG-LOCAL-216): pin the three style-slug surfaces to a
single source of truth.

Background: writer pool (OTR_LedgerScriptWriter._STYLE_PICKER_SEED_POOL)
and music palette (musicgen_theme._STYLE_PALETTE) used to be maintained
as two parallel lists in two files. Drift caused MusicGen to halt
mid-pipeline -- after the writer + freeze cascade had already spent
time and tokens.

Hoisting to nodes/_otr_style_palette.py fixed the root cause. This
test pins all three downstream surfaces (palette, writer pool, freeze
validator) to set-equality with KNOWN_STYLE_SLUGS so any future drift
fires at unit-test time, not soak time.
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


def test_freeze_validator_rejects_unknown_slug():
    """Freeze-time slug validation pins drift to a single error path
    instead of halting in musicgen. S25/MG-6."""
    from nodes._otr_ledger_freeze import _check_meta_invariants

    # Build a minimal ledger-data shape that hits the meta-invariants
    # path with a known-bad slug at meta.gen_params_initial.style.
    bad_slug = "not_a_real_style_anywhere"
    assert bad_slug not in KNOWN_STYLE_SLUGS  # guard against future name collision

    ledger_data = {
        "schema_version": "l3-2026-05-14",
        "meta": {
            "gen_params_initial": {"style": bad_slug},
        },
    }
    errors: list[str] = []
    warnings: list[str] = []
    _check_meta_invariants(ledger_data, errors, warnings)
    assert any("KNOWN_STYLE_SLUGS" in e for e in errors), (
        f"freeze validator did not reject unknown slug; errors={errors!r}"
    )


def test_freeze_validator_accepts_known_slug():
    """Sanity: a slug that IS in the palette must NOT trigger the
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
    assert not any("KNOWN_STYLE_SLUGS" in e for e in errors), (
        f"freeze validator falsely rejected known slug {good_slug!r}; "
        f"errors={errors!r}"
    )
