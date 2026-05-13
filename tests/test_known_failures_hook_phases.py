"""S19.1 (IMP-24): the known-failures hook must track all three
pytest phases (setup, call, teardown).

Source-level pins -- exercising the hook in a subprocess is expensive
and brittle in CI; the hook's correctness is structural so the
relevant phrasing in conftest.py is the contract.
"""
from __future__ import annotations

import pathlib


CONFTEST = (
    pathlib.Path(__file__).resolve().parent / "conftest.py"
).read_text(encoding="utf-8")


def test_hook_iterates_all_three_phases():
    """The pytest_sessionfinish hook must iterate setup/call/teardown.

    The S15.1 original implementation read only rep_call; tests that
    failed in setup or teardown leaked past the diff and silently
    passed. S19.1 extends to all three phases via a getattr loop
    over the phase tuple.
    """
    # The 3-phase tuple is the implementation signal.
    assert '("setup", "call", "teardown")' in CONFTEST, (
        "S19.1: known-failures hook must iterate the 3-tuple of "
        "pytest phases. Either the phrasing changed or the hook "
        "was reverted to single-phase. Restore the 3-tuple iteration."
    )


def test_hook_uses_phase_aware_attribute_lookup():
    """The hook reads rep_setup / rep_call / rep_teardown via
    f-string attribute name -- mirrors the
    ``setattr(item, f"rep_{rep.when}", rep)`` line in
    pytest_runtest_makereport."""
    # Either form satisfies the contract; we just want SOMETHING
    # that produces a phase-aware attribute name.
    assert 'f"rep_{phase}"' in CONFTEST, (
        "S19.1: known-failures hook must compute the phase attribute "
        "name dynamically (f-string over the phase) to mirror "
        "pytest_runtest_makereport's setattr line."
    )


def test_makereport_still_stashes_all_phases():
    """pytest_runtest_makereport must still stash all three reports
    on the item (this was the S15.1 contract; S19.1's hook depends
    on it)."""
    assert 'setattr(item, f"rep_{rep.when}", rep)' in CONFTEST
