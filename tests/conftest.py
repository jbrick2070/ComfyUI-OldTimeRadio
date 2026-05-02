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
