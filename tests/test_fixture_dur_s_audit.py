"""S13.3 -- Python-based fixture dur_s audit (IMP-9).

Replaces the S8.1 regex sweep (one-shot script in the session
outputs dir) with a structural test that runs in CI on every commit.

Walks the actual object graph of every L3 ledger fixture under
``tests/fixtures/`` (both standalone JSON files AND ``make_*_ledger``
factory output) and asserts every SFX line's ``dur_s`` falls inside
the G7 window ``[SFX_DUR_MIN_S, SFX_DUR_MAX_S]`` -- the exact same
constants the FreezeCascade uses, imported live so the audit moves
with the contract.

Catches what regex misses: variables, indirection, helper-built
dicts, kwargs spread, conditional branches.
"""
from __future__ import annotations

import pytest

from nodes._otr_ledger_freeze import SFX_DUR_MIN_S, SFX_DUR_MAX_S
from tests._helpers import load_all_ledger_fixtures


def test_no_fixture_seeds_out_of_bounds_sfx_dur_s():
    """Every SFX line in every L3 ledger fixture must have a dur_s
    within the live G7 window."""
    violations: list[tuple[str, str, float]] = []
    fixture_count = 0
    sfx_count = 0
    for path, ledger in load_all_ledger_fixtures():
        fixture_count += 1
        lines = ledger.get("lines") or []
        for line in lines:
            if not isinstance(line, dict):
                continue
            if line.get("speaker_role") != "sfx":
                continue
            sfx_count += 1
            d = line.get("dur_s")
            if d is None:
                continue  # absent dur_s falls back to default; not in scope
            try:
                d_f = float(d)
            except (TypeError, ValueError):
                continue  # type-invariant catches this elsewhere
            if not (SFX_DUR_MIN_S <= d_f <= SFX_DUR_MAX_S):
                violations.append(
                    (str(path), line.get("line_id") or "<no line_id>", d_f)
                )
    assert not violations, (
        f"Out-of-bounds fixture SFX dur_s: {violations}. "
        f"G7 bounds [{SFX_DUR_MIN_S}, {SFX_DUR_MAX_S}]. Tighten the "
        f"fixture or, if the value is intentionally exercising "
        f"rejection, mark the test name with one of the intentional-"
        f"OOB tags from docs/fixtures-audit-S8.md."
    )


def test_audit_visited_at_least_one_fixture():
    """Sanity check: the helper actually walks something. If
    load_all_ledger_fixtures yields zero pairs (e.g., fixtures dir
    was renamed, all factories require args), this test catches
    the silent-empty audit and fails loudly."""
    fixtures = list(load_all_ledger_fixtures())
    assert fixtures, (
        "load_all_ledger_fixtures yielded zero fixtures. The audit "
        "is silently passing because there's nothing to walk. Verify "
        "tests/fixtures/ contents and the factory naming convention "
        "(make_*_ledger with all-defaulted kwargs)."
    )
