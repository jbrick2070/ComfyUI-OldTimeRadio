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
import socket
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
# Likewise a phone-ping topic and a preview probe belong to an operator's shell,
# never to a test run: inherited, a test that drives the headless runner would
# ping a real phone, or submit a prompt that asks the server for previews
# (planning/creatoros-concepts/PLAN.md rows 1.1 and 1.3).
os.environ.pop("OTR_NOTIFY_URL", None)
os.environ.pop("OTR_PREVIEW_METHOD", None)
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
import tempfile as _tempfile

#: Filled by pytest_sessionstart; the fallback only serves an import before it.
_TEST_COMFY_OUTPUT = [os.path.join(_tempfile.gettempdir(), "otr_pytest_comfy_output")]

if "folder_paths" not in sys.modules:
    import types as _types
    from pathlib import Path as _Path

    _fp_stub = _types.ModuleType("folder_paths")
    # ComfyUI's output dir, AS THE SUITE SEES IT: a folder inside pytest's own
    # session temp tree, set in pytest_sessionstart below. Until 2026-09-25 this
    # returned the cwd -- the repo root when pytest runs from it -- so every test
    # that resolved an output path (comfy_output_dir() tier 2) wrote into
    # <repo>/otr/, inside the installed pack, and the OpenRouter "cache leaves
    # the pack" guard failed on the harness rather than on the code.
    _fp_stub.get_output_directory = lambda: _TEST_COMFY_OUTPUT[0]  # noqa: E731
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
    """Mark this machine's known failures xfail(strict), and skip
    `requires_cuda`-marked tests when CUDA is masked.

    A known failure that fails as recorded is an xfail, so the run ends
    green; one that PASSES fails the suite (strict), which is how the
    ledger's own 2026-08-12 entry describes the pattern worth keeping: the
    marker cannot outlive its cause, because fixing the cause fails the
    suite until the entry leaves MACHINE_FACTS and apple/known-failures.md.
    pytest_sessionfinish then holds each xfail to its recorded signature.

    Cheap, idempotent. Lets a future GPU test sit in the same suite
    without unmasking globally.
    """
    if EXPECTED_FAILURES:
        known = pytest.mark.xfail(
            strict=True,
            reason="a known failure on %s: apple/known-failures.md" % MACHINE)
        for item in items:
            if item.nodeid in EXPECTED_FAILURES:
                item.add_marker(known)
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
    _TEST_COMFY_OUTPUT[0] = os.path.join(base, "comfy_output")
    os.makedirs(_TEST_COMFY_OUTPUT[0], exist_ok=True)
    sys.stderr.write(
        f"[OTR conftest] CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}' "
        f"OTR_TEST_MODE={os.environ.get('OTR_TEST_MODE','0')} "
        f"OTR_EXTRA_OUTPUT_ROOTS='{os.environ['OTR_EXTRA_OUTPUT_ROOTS']}' "
        f"machine={MACHINE} known_failures={len(EXPECTED_FAILURES)} "
        f"OTR_COMFYUI_MODELS_ROOT='{os.environ.get('OTR_COMFYUI_MODELS_ROOT','<unset>')}'\n"
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
#     from this set + apple/known-failures.md.
#
# Matches apple/known-failures.md schema rewrite (S15.2).
#
# KEYED BY MACHINE, AND EACH ENTRY CARRIES ITS REASON (2026-10-30). The set
# is empty on the box that holds every weight and reference (the 5080,
# IDREAM) and names 19 nodeids on the RTX 4060 (MRKT), where no voice
# reference WAV, no kokoro voice tensor, no upscale weight at the headless
# roots and no `s29-clean-slate-gate` git ref exist. A flat set could only
# have hidden those behind a count, which is what apple/known-failures.md
# forbids. Two things keep this honest: an entry is a nodeid AND a
# substring of the failure text measured for it, so the same test failing
# for ANOTHER reason is reported as a regression, not excused; and every
# nodeid must appear in apple/known-failures.md with its cause and exit
# condition (tests/test_known_fail_guard.py holds both files together).
# ---------------------------------------------------------------------------


def machine_key(hostname) -> str:
    """The first DNS label of a hostname, upper-cased: "mrkt.lan" -> "MRKT".

    The key MACHINE_FACTS is read with. The banner pytest_sessionstart prints
    names it, so a renamed box reads as "no facts" there, not as 19 silent
    regressions."""
    return str(hostname or "").split(".")[0].strip().upper()


#: What the suite knows about one machine, by machine_key(gethostname()).
#:   models_root     where this box keeps the weights when the pack's default
#:                   root (nodes/_otr_models_root.py, C:\ComfyUI-Models) is
#:                   not it. Set below at import, like the other env defaults
#:                   above, unless the environment already names a root or
#:                   the directory is gone.
#:   expected_failed nodeid -> a substring of that test's failure MESSAGE,
#:                   as measured on the box: the exception's own words (the
#:                   "E   ..." lines), never a traceback source line, which
#:                   `--tb=no` leaves out of the report. See
#:                   apple/known-failures.md for the cause and the exit
#:                   condition of each.
#: A machine with no row here has an empty set and no env change.
MACHINE_FACTS: dict = {
    "MRKT": {
        "models_root": r"C:\Users\jeffr\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models",
        "expected_failed": {
            # The git ref the sweep diffs against exists only on the 5080's clone.
            "tests/test_b7_forbidden_sweep.py::test_forbidden_sweep_runs_clean":
                "s29-clean-slate-gate",
            # models/TTS/refs/indextts2/*.wav: the clone references are not on this box.
            "tests/test_cast_lock_voice_ref_completeness.py::"
            "test_castlock_stamps_the_same_reference_the_render_path_would_open":
                "the stamped reference must resolve to a real file on disk",
            "tests/test_cast_lock_voice_ref_completeness.py::"
            "test_unspecified_gender_names_the_real_render_reference_without_changing_identity[None]":
                "the render reference must resolve to a real file on disk",
            "tests/test_cast_lock_voice_ref_completeness.py::"
            "test_unspecified_gender_names_the_real_render_reference_without_changing_identity[]":
                "the render reference must resolve to a real file on disk",
            "tests/test_cast_lock_voice_ref_completeness.py::"
            "test_unspecified_gender_names_the_real_render_reference_without_changing_identity[   ]":
                "the render reference must resolve to a real file on disk",
            "tests/test_cast_lock_voice_ref_completeness.py::"
            "test_the_render_fallback_still_works_for_a_row_castlock_never_stamped":
                "a cloning engine must never be left without a reference",
            "tests/test_voice_casting_identity.py::"
            "test_a_clone_engine_takes_the_ordinary_draw_and_reaches_a_real_file":
                "which is not a file on disk",
            "tests/test_voice_casting_identity.py::"
            "test_an_assigned_row_is_fingerprinted_rather_than_static":
                "AssertionError: nan",
            "tests/test_voice_casting_identity.py::"
            "test_the_reserved_recording_is_delivered_to_the_character_it_names[indextts2-auto_registry]":
                "went to nobody",
            "tests/test_voice_casting_identity.py::"
            "test_the_reserved_recording_is_delivered_to_the_character_it_names[indextts2-preserve_ledger]":
                "went to nobody",
            "tests/test_voice_casting_identity.py::"
            "test_the_reserved_recording_is_delivered_to_the_character_it_names[chatterbox-auto_registry]":
                "went to nobody",
            "tests/test_voice_casting_identity.py::"
            "test_the_reserved_recording_is_delivered_to_the_character_it_names[chatterbox-preserve_ledger]":
                "went to nobody",
            "tests/test_voice_casting_identity.py::"
            "test_the_reservation_outranks_a_catalogue_assignment[auto_registry]":
                "beat his own recording",
            "tests/test_voice_casting_identity.py::"
            "test_the_reservation_outranks_a_catalogue_assignment[preserve_ledger]":
                "beat his own recording",
            # kokoro runs on the ONNX backend here: no per-voice .pt tensor to fingerprint.
            "tests/test_voice_casting_identity.py::test_a_pt_voice_tensor_is_fingerprinted_too":
                "AssertionError: nan",
            "tests/test_voice_identity_fix.py::"
            "test_a_frozen_ledger_route_does_not_override_the_stamped_voice":
                "assert nan == nan",
            # The upscale gate reads the 5080's headless roots and the checkout's
            # grandparent models dir; neither holds a weight here.
            "tests/test_upscale_weight_resolution_gate.py::"
            "test_every_registered_upscale_engine_can_reach_its_checkpoint":
                "cannot reach its checkpoint",
            "tests/test_upscale_weight_resolution_gate.py::"
            "test_the_headless_boot_maps_upscale_models_somewhere_that_has_weights":
                "NONE of the upscale_models roots",
            "tests/test_upscale_weight_resolution_gate.py::"
            "test_the_pinned_esrgan_filename_matches_what_is_on_disk":
                "resolved None but the engine pins",
        },
    },
}


def expected_failures(hostname) -> dict:
    """nodeid -> failure-text signature for the machine `hostname` names;
    empty for a machine MACHINE_FACTS does not know."""
    return dict(MACHINE_FACTS.get(machine_key(hostname), {}).get("expected_failed", {}))


def report_outcome(reports, known=False) -> "tuple[bool, str | None]":
    """`(executed, failure_text)` for one item's phase reports (setup, call,
    teardown; None for a phase that never ran).

    `executed` is False when no phase ran -- after `-x`, or a crash -- which
    is no evidence either way. `failure_text` is the report text when the
    item failed; for a `known` item (one this guard marked xfail) also when
    it failed the way that mark expects, which pytest files as skipped with
    `wasxfail`, so the signature can be checked. An xfail a TEST AUTHOR
    placed is an accepted outcome, as it always was, never a failure here.
    None when the item passed, and None too when a known item passed
    AGAINST its strict xfail, which pytest already reports as a failure of
    its own ("[XPASS(strict)] ...") and the guard reports as promotable."""
    reports = [rep for rep in reports if rep is not None]
    if not reports:
        return False, None
    texts = []
    for rep in reports:
        text = str(getattr(rep, "longreprtext", "") or "")
        if getattr(rep, "failed", False) and not text.startswith("[XPASS(strict)]"):
            texts.append(text)
        elif known and getattr(rep, "skipped", False) and hasattr(rep, "wasxfail"):
            texts.append(text)
    return True, ("\n".join(texts) if texts else None)


def guard_verdict(executed, failed, expected) -> "tuple[dict, set]":
    """The diff the session-finish guard reports, as a pure function.

    `executed`: nodeids that STARTED (setup ran); a collected item that never
    started -- after `-x`, or a crash -- is no evidence either way, so it is
    neither a failure nor promotable. `failed`: nodeid -> its failure text.
    `expected`: nodeid -> signature. Returns `(new_failures, promotable)`:
    new_failures maps a nodeid to why it is new -- unknown to the set, or known
    but failing WITHOUT its signature, which is a different break wearing a
    known test's name; promotable is every expected nodeid that ran and passed.
    """
    new_failures = {}
    for nodeid, text in failed.items():
        signature = expected.get(nodeid)
        if signature is None:
            new_failures[nodeid] = "not a known failure"
        elif signature not in (text or ""):
            new_failures[nodeid] = (
                "a known failure, but it did not fail the known way: %r is not "
                "in its report" % (signature,))
    promotable = (set(expected) & set(executed)) - set(failed)
    return new_failures, promotable


MACHINE: str = machine_key(socket.gethostname())
EXPECTED_FAILURES: dict = expected_failures(MACHINE)
EXPECTED_FAILED_NODEIDS: frozenset[str] = frozenset(EXPECTED_FAILURES)

# This box's weights, for the resolvers that read the configured root
# (nodes/_otr_models_root.py): set here at import for the same reason the env
# defaults above are, and only when nothing already names a root. An explicit
# root is exclusive to those resolvers, so a directory that is gone is left
# unset and the banner says so.
_MACHINE_MODELS_ROOT = str(MACHINE_FACTS.get(MACHINE, {}).get("models_root") or "")
if (_MACHINE_MODELS_ROOT and os.path.isdir(_MACHINE_MODELS_ROOT)
        and not any(os.environ.get(k, "").strip()
                    for k in ("OTR_COMFYUI_MODELS_ROOT", "COMFYUI_MODELS_ROOT"))):
    os.environ["OTR_COMFYUI_MODELS_ROOT"] = _MACHINE_MODELS_ROOT


# CW-1 (2026-06-06) unwired the legacy FLUX/HuMo/LTX render path; CW-4
# (2026-06-07) rebuilt it as SignalLostVideo -> OTR_SilentComposite ->
# OTR_MasterAudioMux and DELETED the 14 quarantined render-path tests that
# asserted the old wiring (their coverage is replaced by the new-path tests
# in tests/test_video_render_path_cw4.py).
# CW-5 (2026-06-09) promoted the chatterbox entry: the test now skips when
# the sidecar venv is present on disk (same pattern as indextts2), so it is
# no longer an expected failure -- the guard would fire every run.
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
# now passes. See apple/known-failures.md Resolution log
# KNOWN-FAIL-007/008.
#
# From 2026-09-26 to 2026-10-30 the set was empty everywhere; it is empty
# on the 5080 still. The MRKT entries above are the first machine-keyed
# ones (see the header of this section).


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

    NEW failures: regression -- hard-fail with exit code 2 so CI
    surfaces it distinctly from a normal pytest failed-test exit. A
    failure is new when its nodeid is not in this machine's set, OR
    when it is but its report lacks the signature the set recorded
    for it: a known test failing a new way is a new break.

    PROMOTABLE: a known-fail ran and passed. Its strict xfail has
    already failed the session (pytest's own "[XPASS(strict)]"); this
    banner names the nodeid and says what to do: remove it from
    MACHINE_FACTS + apple/known-failures.md in lockstep.

    Only items that STARTED count, in either direction. A focused
    ``pytest tests/test_xyz.py`` therefore reports exactly what it
    ran, and a run cut short by ``-x`` or a crash does not read its
    unexecuted known-fails as passing. (Until 2026-10-30 the diff was
    skipped whenever under 80% of the expected set was collected,
    which with a non-empty set would have blinded every focused run.)
    """
    items = getattr(session, "items", None)
    if not items:
        return

    # S19.1 (IMP-24): track all three pytest phases. Reading only
    # rep_call meant tests that errored in setup or teardown leaked
    # past the diff as silent-passes. pytest_runtest_makereport
    # above stashes rep_setup / rep_call / rep_teardown on the item.
    executed = set()
    failed = {}
    for it in items:
        ran, text = report_outcome(
            (getattr(it, f"rep_{phase}", None) for phase in ("setup", "call", "teardown")),
            known=it.nodeid in EXPECTED_FAILURES)
        if not ran:
            continue  # collected, never started
        executed.add(it.nodeid)
        if text is not None:
            failed[it.nodeid] = text

    new_failures, promotable = guard_verdict(executed, failed, EXPECTED_FAILURES)

    if promotable:
        sys.stderr.write(
            "\n[KNOWN-FAIL-GUARD] PROMOTABLE -- the following expected "
            "failures now PASS on %s (their strict xfail is what failed "
            "this run). Remove them from tests/conftest.py::MACHINE_FACTS "
            "AND from apple/known-failures.md in lockstep:\n" % MACHINE
        )
        for nid in sorted(promotable):
            sys.stderr.write(f"  - {nid}\n")

    if new_failures:
        sys.stderr.write(
            "\n[KNOWN-FAIL-GUARD] NEW failures (REGRESSION) -- the "
            "following nodeids failed and are not in %s's known "
            "failures (tests/conftest.py::MACHINE_FACTS):\n" % MACHINE
        )
        for nid in sorted(new_failures):
            sys.stderr.write(f"  - {nid}: {new_failures[nid]}\n")
        sys.stderr.write(
            "If these are intentional new known-fails, add them to "
            "MACHINE_FACTS (nodeid and a signature of the failure text) "
            "+ apple/known-failures.md in the same commit. Otherwise fix "
            "the regression.\n"
        )
        # Hard exit so CI distinguishes this from a normal pytest
        # failed-test exit. Exit code 2 per S15.1 spec.
        raise SystemExit(2)


@pytest.fixture(autouse=True)
def _openrouter_catalog_fetch_is_never_real(monkeypatch):
    """NO UNIT TEST MAY FETCH THE OPENROUTER CATALOG (2026-09-12).

    `resolve_context_window` now refreshes a cold catalog cache inline when an
    API key is present, and OpenRouter's model list needs no auth -- so a test
    that set a fake key and a cold cache fetched the REAL catalog mid-suite and
    then failed on the window it had just learned. This box also carries a real
    key in its shell environment. Every test therefore starts with the fetch
    seam raised and the once-per-process flag cleared; a test that wants a
    fetch replaces the seam itself (see tests/test_openrouter_cold_cache_self_heals.py).
    """
    try:
        from nodes import _otr_openrouter_backend as _orb
    except Exception:  # pragma: no cover -- the backend is importable everywhere
        return

    def _no_network(**_kw):
        raise AssertionError("OpenRouter catalog fetch fired from a unit test")

    monkeypatch.setattr(_orb, "_fetch_models_json", _no_network, raising=False)
    monkeypatch.setattr(_orb, "_COLD_CACHE_REFRESH_TRIED", False, raising=False)

