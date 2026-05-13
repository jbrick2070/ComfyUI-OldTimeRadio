"""tests/conftest.py -- pytest fixtures for OTR-OldTimeRadio.

The autouse `_otr_no_cuda_during_collection` fixture sets
``CUDA_VISIBLE_DEVICES=""`` before any OTR test module imports torch.
Without this, pytest collection hangs deterministically when the same
GPU is held by a running ComfyUI Desktop process: torch's lazy CUDA
init blocks waiting for the primary context the other process owns.

BUG-LOCAL-006 fix (2026-05-02). The unit tests under ``tests/`` exercise
parser logic, schema validation, dropdown guardrails, and ledger math.
None of them need a real GPU. If any future test does need CUDA, mark
it with ``@pytest.mark.requires_cuda`` and override the env in that
test (left as an exercise — no current test claims that mark).
"""

from __future__ import annotations

import os
import sys

import pytest


# Set the env var BEFORE pytest collects any test module. conftest.py at
# tests/ root is imported very early -- before tests/test_*.py modules
# are collected -- so this is the right spot.
#
# We also strip any inherited NVIDIA env from the parent shell so the
# Triton / bitsandbytes / transformers device-probe paths short-circuit
# cleanly to CPU. This is the test-only path; ComfyUI Desktop is unaffected.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")

# Some OTR modules tolerate `torch.cuda.is_available()` returning False
# at import time but blow up on `torch.cuda.empty_cache()` calls inside
# decorators. The decorators are guarded already, but pin the device
# string for any module that reads it eagerly.
os.environ.setdefault("OTR_TEST_MODE", "1")


@pytest.fixture(autouse=True, scope="session")
def _otr_no_cuda_during_collection():
    """Autouse session-scope fixture; documents the env-var approach.

    The actual env-var set happens at module import (above) because pytest
    collection imports test modules before any fixture runs. This fixture
    exists so the policy shows up in `pytest --fixtures` output and so
    any test that needs to opt out can do so explicitly.
    """
    yield


def pytest_configure(config):
    """Register the optional `requires_cuda` marker for tests that need GPU.

    No current test uses this marker. Reserved for future use.
    """
    config.addinivalue_line(
        "markers",
        "requires_cuda: test needs a real CUDA device (override CUDA_VISIBLE_DEVICES)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip `requires_cuda`-marked tests when CUDA is masked.

    Cheap, idempotent. Lets a future GPU test sit in the same suite
    without unmasking globally.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        return
    skip_marker = pytest.mark.skip(
        reason="CUDA masked in tests/conftest.py (BUG-LOCAL-006)"
    )
    for item in items:
        if "requires_cuda" in item.keywords:
            item.add_marker(skip_marker)


# Diagnostic: print this once at session start so a future debugger
# can see the mask is in place.
def pytest_sessionstart(session):
    sys.stderr.write(
        f"[OTR conftest] CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}' "
        f"OTR_TEST_MODE={os.environ.get('OTR_TEST_MODE','0')}\n"
    )


# ---------------------------------------------------------------------------
# S15.1 known-failures with nodeid tracking (per Q-D11)
#
# Tracks the failure SET, not just the count. A known failure that
# starts passing while a new failure appears would slip past a
# count-only guard (6 in, 6 out, but the IDs differ). The hook below
# diffs the actual failed-nodeid set against EXPECTED_FAILED_NODEIDS:
#   - new failures (actual - expected): hard-fail with exit code 2
#     so CI surfaces a regression.
#   - promotable (expected - actual): print a PROMOTABLE message --
#     the known-fail entry is now passing and should be removed
#     from this set + docs/known-failures.md.
#
# Matches docs/known-failures.md schema rewrite (S15.2).
# ---------------------------------------------------------------------------


EXPECTED_FAILED_NODEIDS = frozenset({
    # KNOWN-FAIL-001
    "tests/test_production_ledger.py::TestDualLedgerFix::test_save_merges_schema_l3_fields_from_disk",
    # KNOWN-FAIL-002
    "tests/test_save_to_episode_workspace.py::test_save_to_per_episode_dir_when_singleton_active",
    # KNOWN-FAIL-003
    "tests/test_save_to_episode_workspace.py::test_portraits_role_routes_to_portraits_dir",
    # KNOWN-FAIL-004
    "tests/test_save_to_episode_workspace.py::test_falls_back_to_legacy_dir_when_no_singleton",
    # KNOWN-FAIL-005
    "tests/test_save_to_episode_workspace.py::test_per_episode_counter_starts_at_1",
    # KNOWN-FAIL-006
    "tests/test_video_composite.py::test_default_canvas_is_native_832x480_at_25fps",
})


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Stash each test's per-phase report (setup / call / teardown) on
    the item so pytest_sessionfinish can read .rep_call.failed below.
    Standard pytest pattern -- there's no built-in "did this test fail"
    accessor, so we plumb it ourselves."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


def pytest_sessionfinish(session, exitstatus):
    """Diff the actual-failed-nodeid set against EXPECTED_FAILED_NODEIDS.

    NEW failures (actual - expected): regression -- hard-fail with
    exit code 2 so CI surfaces it distinctly from a normal pytest
    failed-test exit.

    PROMOTABLE (expected - actual): a known-fail is now passing.
    Print a PROMOTABLE banner naming the nodeid; the contributor
    updates EXPECTED_FAILED_NODEIDS + docs/known-failures.md in
    lockstep. Don't fail the session for promotables -- they're
    good news, just news.

    Subset-run guard: only enforce the diff when at least 80% of
    the expected nodeids were collected. So a focused
    ``pytest tests/test_xyz.py`` doesn't trip "PROMOTABLE" on every
    known-fail it didn't run.
    """
    items = getattr(session, "items", None)
    if not items:
        return
    collected_ids = {it.nodeid for it in items}
    expected_seen = collected_ids & EXPECTED_FAILED_NODEIDS
    if len(expected_seen) < 0.8 * len(EXPECTED_FAILED_NODEIDS):
        return  # focused subset run; skip the diff

    actual_failed = set()
    for it in items:
        rep = getattr(it, "rep_call", None)
        if rep is not None and getattr(rep, "failed", False):
            actual_failed.add(it.nodeid)

    new_failures = actual_failed - EXPECTED_FAILED_NODEIDS
    promotable = (EXPECTED_FAILED_NODEIDS & collected_ids) - actual_failed

    if promotable:
        sys.stderr.write(
            "\n[KNOWN-FAIL-GUARD] PROMOTABLE -- the following expected "
            "failures now PASS. Remove them from "
            "tests/conftest.py::EXPECTED_FAILED_NODEIDS AND from "
            "docs/known-failures.md in lockstep:\n"
        )
        for nid in sorted(promotable):
            sys.stderr.write(f"  - {nid}\n")

    if new_failures:
        sys.stderr.write(
            "\n[KNOWN-FAIL-GUARD] NEW failures (REGRESSION) -- the "
            "following nodeids failed but are not in "
            "EXPECTED_FAILED_NODEIDS:\n"
        )
        for nid in sorted(new_failures):
            sys.stderr.write(f"  - {nid}\n")
        sys.stderr.write(
            "If these are intentional new known-fails, add them to "
            "EXPECTED_FAILED_NODEIDS + docs/known-failures.md in the "
            "same commit. Otherwise fix the regression.\n"
        )
        # Hard exit so CI distinguishes this from a normal pytest
        # failed-test exit. Exit code 2 per S15.1 spec.
        raise SystemExit(2)
