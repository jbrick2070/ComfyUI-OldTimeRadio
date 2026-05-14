"""Standalone DeprecationWarning capture for QA-5.

Monkeypatches `tests/conftest.py::pytest_sessionfinish` BEFORE pytest
collects so the SystemExit(2) hook becomes a no-op. The FAILURES
traceback then survives the run."""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OTR_TEST_MODE", "1")

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

# Import the conftest module and replace the offending hook BEFORE
# pytest discovers it.
import tests.conftest as _conftest  # noqa: E402

def _noop_session_finish(session, exitstatus):
    return None

_conftest.pytest_sessionfinish = _noop_session_finish

import pytest  # noqa: E402

if __name__ == "__main__":
    rc = pytest.main(
        [
            "tests/test_audiogen_ledger.py::test_audiogen_iter_sfx_only",
            "-p", "no:cacheprovider",
            "-W", "error::DeprecationWarning",
            "--tb=long",
            "-v",
            "--no-header",
        ],
    )
    print(f"\nPYTEST_RC={rc}")
