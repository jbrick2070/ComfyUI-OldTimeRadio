# tests/test_llm_slot_sweep.py -- regression suite for the LLM-slot-tag
# CI sweep (docs/_s28_llm_slot_sweep.py).
#
# Validates that:
#   1. Every LLM call site in nodes/ carries a ``# LLM slot:`` tag.
#   2. The sweep finds a floor count of call sites (guards against
#      vacuous passes).
#   3. The sweep catches a missing tag in a synthetic test file.
#   4. Exempt files are never scanned.
#   5. No node file fails to AST-parse (a broken file would silently
#      drop out of the sweep, hiding its call sites).
#   6. The sweep catches an unparseable file in a synthetic dir.
from __future__ import annotations

import pathlib
import sys
import textwrap

import pytest

# Resolve paths relative to this test file so ``pytest`` finds the
# sweep module regardless of CWD.
_TEST_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _TEST_DIR.parent
NODES_DIR = str(_REPO_ROOT / "nodes")

# Import the sweep module from docs/.
sys.path.insert(0, str(_REPO_ROOT / "docs"))
from _s28_llm_slot_sweep import (  # noqa: E402
    CALL_SITE_NAMES,
    EXEMPT_FILES,
    find_llm_call_sites,
    find_parse_failures,
    find_untagged_call_sites,
)


# ------------------------------------------------------------------
# 1. Every call site must be tagged
# ------------------------------------------------------------------

def test_every_llm_call_site_has_slot_tag():
    """No LLM call site in nodes/ should be missing a # LLM slot: tag."""
    untagged = find_untagged_call_sites(NODES_DIR)
    if untagged:
        import os
        lines = [
            f"  {os.path.relpath(fp, str(_REPO_ROOT))}:{ln} {name}()"
            for fp, ln, name in untagged
        ]
        details = "\n".join(lines)
        pytest.fail(
            f"{len(untagged)} LLM call site(s) missing "
            f"'# LLM slot:' tag:\n{details}"
        )


# ------------------------------------------------------------------
# 2. Floor count -- the sweep must find a non-trivial number of sites
# ------------------------------------------------------------------

# The codebase currently has ~20 LLM call sites. A floor of 12
# guards against accidental removal of call sites that would make
# the sweep vacuously pass while still leaving room for legitimate
# refactors that consolidate a few call sites.
_MIN_EXPECTED_CALL_SITES = 12


def test_sweep_finds_known_call_sites():
    """Sweep must find at least _MIN_EXPECTED_CALL_SITES call sites."""
    all_sites = find_llm_call_sites(NODES_DIR)
    assert len(all_sites) >= _MIN_EXPECTED_CALL_SITES, (
        f"Expected >= {_MIN_EXPECTED_CALL_SITES} LLM call sites, "
        f"found {len(all_sites)}. Did a refactor remove call sites "
        f"without updating this test?"
    )


# ------------------------------------------------------------------
# 3. The sweep catches a missing tag in a synthetic file
# ------------------------------------------------------------------

def test_sweep_catches_missing_tag(tmp_path):
    """A file with structured_call(...) but no tag must be flagged."""
    source = textwrap.dedent("""\
        import ast

        def my_fn(slot_fn):
            result = structured_call(
                prompt=[],
                schema=object,
                slot_fn=slot_fn,
            )
            return result
    """)
    test_file = tmp_path / "fake_node.py"
    test_file.write_text(source, encoding="utf-8")
    untagged = find_untagged_call_sites(str(tmp_path))
    assert len(untagged) == 1, (
        f"Expected 1 untagged site in synthetic file, found {len(untagged)}"
    )
    assert untagged[0][2] == "structured_call"


# ------------------------------------------------------------------
# 4. Exempt files are never scanned
# ------------------------------------------------------------------

def test_exempt_files_not_scanned():
    """No result should come from an exempt file."""
    assert "_otr_structured_call.py" in EXEMPT_FILES, (
        "_otr_structured_call.py must be in EXEMPT_FILES"
    )
    all_sites = find_llm_call_sites(NODES_DIR)
    for filepath, _, _ in all_sites:
        basename = pathlib.Path(filepath).name
        assert basename not in EXEMPT_FILES, (
            f"Call site found in exempt file {basename} -- "
            f"the sweep should skip it."
        )


# ------------------------------------------------------------------
# 5. No node file may fail to AST-parse
# ------------------------------------------------------------------

def test_no_node_file_fails_to_parse():
    """An unparseable node file is skipped by the call-site walk, so its
    LLM call sites silently vanish and the sweep passes vacuously for it.
    Guard against that: every file under nodes/ must AST-parse."""
    failures = find_parse_failures(NODES_DIR)
    if failures:
        import os
        lines = [
            f"  {os.path.relpath(fp, str(_REPO_ROOT))}: {err}"
            for fp, err in failures
        ]
        pytest.fail(
            f"{len(failures)} node file(s) failed to AST-parse -- their "
            f"LLM call sites are invisible to the sweep:\n"
            + "\n".join(lines)
        )


# ------------------------------------------------------------------
# 6. The sweep catches an unparseable file
# ------------------------------------------------------------------

def test_sweep_catches_parse_failure(tmp_path):
    """A file with a syntax error must be reported by find_parse_failures,
    and the call-site walk must not crash on it."""
    broken = tmp_path / "broken_node.py"
    broken.write_text("def oops(:\n    pass\n", encoding="utf-8")
    failures = find_parse_failures(str(tmp_path))
    assert len(failures) == 1, (
        f"Expected 1 parse failure in synthetic dir, found {len(failures)}"
    )
    assert pathlib.Path(failures[0][0]).name == "broken_node.py"
    # The call-site walk must swallow the broken file, not raise.
    sites = find_llm_call_sites(str(tmp_path))
    assert sites == [], (
        "find_llm_call_sites must skip an unparseable file without crashing"
    )
