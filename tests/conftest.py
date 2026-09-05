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
# Portable installs are expected to persist this override. The test suite still
# needs the shipped bank as its deterministic baseline; individual portability
# tests opt back in with monkeypatch so production inheritance cannot silently
# change unrelated casting assertions.
os.environ.pop("OTR_VOICE_REFERENCE_BANK", None)
# A pinned publication root belongs to a launcher, never to a test run: with
# it inherited, any test that resolves otr_obs_dir() would write into the
# operator's live watched folder (kibitz runpod-found-fixes r3, 2026-09-04).
os.environ.pop("OTR_OBS_DIR", None)
for _google_key_env in ("OTR_GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
    os.environ.pop(_google_key_env, None)


# ---------------------------------------------------------------------------
# Shared `folder_paths` stub (2026-05-21).
#
# `folder_paths` is a ComfyUI-runtime module, absent in the bare pytest
# sandbox. Node modules import it at module level (the original case was
# the legacy `nodes/_otr_deferred_loaders.py`, retired in the Chunk E
# cleanbreak), and a few node INPUT_TYPES() bodies
# call `folder_paths.get_filename_list(...)`.
#
# Before this block, three test modules
# (test_story_brief_{humo_c5f,ltx_c5e,portraits_c5d}.py) each installed
# their own PARTIAL stub (`get_output_directory` only) at import time,
# each guarded by `if "folder_paths" not in sys.modules`. Whichever
# pytest collected first won -- and none carried `get_filename_list`.
# In the full `tests/` walk that made
# `OTR_DeferredCheckpointLoader.INPUT_TYPES()` raise
# `AttributeError: module 'folder_paths' has no attribute
# 'get_filename_list'`, which the workflow contract validator wrapped
# into a `WorkflowValidationError`. Result: 9 order-dependent failures
# that PASSED in isolation (no stub installed -> deferred-loader import
# skipped) but FAILED in the walk (partial stub present -> import
# succeeds -> INPUT_TYPES blows up).
#
# Installing ONE complete stub here -- before any test module is
# collected -- makes every downstream `if "folder_paths" not in
# sys.modules` guard a no-op, so the whole suite sees a single
# consistent stub and the order-dependence is gone.
if "folder_paths" not in sys.modules:
    import types as _types
    from pathlib import Path as _Path

    _fp_stub = _types.ModuleType("folder_paths")
    # Matches the attribute the prior partial stubs provided.
    _fp_stub.get_output_directory = lambda: str(_Path.cwd())  # noqa: E731
    # The missing piece: node INPUT_TYPES() bodies call this to build
    # checkpoint / text-encoder pickers. An empty list is a valid combo
    # spec -- the contract validator introspects key NAMES, not members.
    _fp_stub.get_filename_list = lambda *args, **kwargs: []  # noqa: E731
    sys.modules["folder_paths"] = _fp_stub


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
    # THE SUITE WRITES INTO pytest's tmp TREE, AND THAT TREE IS DECLARED HERE
    # (2026-09-05). The output-tree containment added for the registry punch
    # list refuses a destination outside `comfy_output_dir()`, the operator's
    # `$OTR_OBS_DIR`, and `$OTR_EXTRA_OUTPUT_ROOTS`. Roughly forty tests pass a
    # `tmp_path` destination on purpose -- that IS the shape of the operator
    # naming a location, so the honest way to make them run is the same knob an
    # operator would use, not a test-only bypass inside the guard. Declaring the
    # base temp dir keeps the production default (env unset) untouched, and a
    # test that escapes even this root still fails, which is what we want.
    try:
        base = str(session.config._tmp_path_factory.getbasetemp())
    except Exception:  # noqa: BLE001 -- never let a fixture detail stop the run
        import tempfile
        base = tempfile.gettempdir()
    existing = (os.environ.get("OTR_EXTRA_OUTPUT_ROOTS") or "").strip()
    os.environ["OTR_EXTRA_OUTPUT_ROOTS"] = (
        "%s;%s" % (existing, base) if existing else base)
    sys.stderr.write(
        f"[OTR conftest] CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}' "
        f"OTR_TEST_MODE={os.environ.get('OTR_TEST_MODE','0')} "
        f"OTR_EXTRA_OUTPUT_ROOTS='{os.environ['OTR_EXTRA_OUTPUT_ROOTS']}'\n"
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


# CW-1 (2026-06-06) unwired the legacy FLUX/HuMo/LTX render path; CW-4
# (2026-06-07) rebuilt it as SignalLostVideo -> OTR_SilentComposite ->
# OTR_MasterAudioMux and DELETED the 14 quarantined render-path tests that
# asserted the old wiring (their coverage is replaced by the new-path tests
# in tests/test_video_render_path_cw4.py).
# CW-5 (2026-06-09) promoted the chatterbox entry: the test now skips when
# the sidecar venv is present on disk (same pattern as indextts2), so it is
# no longer an expected failure -- the guard would fire every run.
EXPECTED_FAILED_NODEIDS: frozenset[str] = frozenset()
# KNOWN-FAIL-001 through KNOWN-FAIL-006 all promoted 2026-05-13
# (s26-downstream missed-regression sweep). The 5 BUG-108 + save
# workspace entries went green when the test fixtures + the
# production_ledger meta-merge bug got fixed; KNOWN-FAIL-006 was
# renamed in lockstep with the BUG-LOCAL-030 canvas geometry bump
# and now passes under the new name
# (test_default_canvas_is_layered_1472x832_at_25fps). See commits
# ba8a02e (meta merge), a70aeb8 (fixture size), 8181950 (canvas
# rename) for the migration trail.
#
# KNOWN-FAIL-007 + KNOWN-FAIL-008 (added 2026-05-21, full-walk
# 27-failure triage; promoted 2026-05-21, same day). Both asserted the
# Sprint D / D3 creative-binding gate, which rejected the shipped
# workflow's 'google/gemma-4-E4B-it' binding because that catalog row
# carried license_audit_status='pending'. Resolved by a license
# re-audit: the Gemma 4 family ships under Apache 2.0 -- confirmed on
# the official Google HuggingFace model card -- not the older
# restricted Gemma Terms of Use. The catalog rows and the
# docs/model-license-google--gemma-4-e{2,4}b-it.md audit files were
# corrected in lockstep to apache_2_0 / mit_equivalent, so the gate
# now passes. See docs/known-failures.md Resolution log
# KNOWN-FAIL-007/008.
#
# The quarantine set is now empty -- there are no known failures. The
# pytest_sessionfinish guard below still runs (an empty set does not
# trigger the subset-run early return), so any new failure is caught
# as a regression.


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):  # kept: pytest hook signature contract
    """Stash each test's per-phase report (setup / call / teardown) on
    the item so pytest_sessionfinish can read .rep_call.failed below.
    Standard pytest pattern -- there's no built-in "did this test fail"
    accessor, so we plumb it ourselves."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):  # kept: pytest hook signature contract
    """Diff the actual-failed-nodeid set against EXPECTED_FAILED_NODEIDS.

    `trylast=True` IS LOAD-BEARING (2026-08-16). The `raise SystemExit(2)`
    below aborts every remaining `pytest_sessionfinish` implementation --
    including the terminal reporter's, which is what prints the `FAILURES`
    section and the short summary. Without this decorator a regression is
    reported as a bare nodeid with NO TRACEBACK, so the guard built to
    surface regressions was hiding the one thing needed to fix them. Running
    last lets the report print first, then the hard exit still happens.

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

    # S19.1 (IMP-24): track all three pytest phases. Reading only
    # rep_call meant tests that errored in setup or teardown leaked
    # past the diff as silent-passes. pytest_runtest_makereport
    # above stashes rep_setup / rep_call / rep_teardown on the item.
    actual_failed = set()
    for it in items:
        failed_in_any_phase = any(
            getattr(getattr(it, f"rep_{phase}", None), "failed", False)
            for phase in ("setup", "call", "teardown")
        )
        if failed_in_any_phase:
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
