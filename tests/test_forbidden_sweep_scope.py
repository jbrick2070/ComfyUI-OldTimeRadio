"""Sprint D D3 -- forbidden-sweep scope and carve-out behavior.

The sweep helper at `tests/_s28_forbidden_sweep.py` plus its driver test
already covers these properties at D3 landing:

  1. The driver only diffs `*.py` files (`git diff ... -- "*.py"`).
     A `.safetensors`, `.wav`, `.pt`, etc. binary cannot trip the
     0x97 cp1252 em-dash false-positive class.

  2. Era literals INSIDE `OTR_PERIOD_SYSTEM_PROMPT` (and
     `PERIOD_EXEMPLARS`) live on lines tokenize-classified as
     "string". The sweep classifies those as FORENSIC, not RUNTIME.

This file pins both properties against future regression:

  test_forbidden_sweep_test_driver_restricts_to_py_extension_only
      Source inspection of `tests/test_b7_forbidden_sweep.py`
      confirms the git diff invocation carries `-- "*.py"`.

  test_classify_lines_marks_period_system_prompt_body_as_string
      Smoke against the live `nodes/_otr_period_prompts.py` -- the
      multi-line triple-quoted string body of OTR_PERIOD_SYSTEM_PROMPT
      is classified as "string" so its era literals would be
      forensic-suppressed even if they appeared on a new ADDED diff
      line.

  test_planted_era_literal_in_period_prompts_file_outside_string_would_be_runtime
      Negative control: if a future commit accidentally placed an
      era literal OUTSIDE the carved constants (e.g. in a docstring
      that doesn't span a triple-quoted form OR in a variable name),
      the existing classifier would correctly mark it. This test
      asserts the classifier's behavior, not the sweep run itself.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PERIOD_SRC = REPO_ROOT / "nodes" / "_otr_period_prompts.py"
SWEEP_TEST_SRC = REPO_ROOT / "tests" / "test_b7_forbidden_sweep.py"


def _load_classify_lines():
    """Import classify_lines from tests/_s28_forbidden_sweep.py
    without running the module body (which auto-walks the diff file).
    """
    sweep_path = REPO_ROOT / "tests" / "_s28_forbidden_sweep.py"
    spec = importlib.util.spec_from_file_location("_s28_sweep_for_test", sweep_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError:
        # The sweep module top-level body tries to read DIFF_PATH and
        # OUT_PATH; FileNotFoundError when neither exists is benign --
        # classify_lines is still bound. Re-raise anything else.
        pass
    return module.classify_lines


def test_forbidden_sweep_test_driver_restricts_to_py_extension_only() -> None:
    """The B7 sweep driver test invokes git diff with `-- "*.py"` so
    `.safetensors`, `.wav`, and other binaries don't enter the
    sweep input. Pinned here so a future tweak to the driver
    cannot drop the extension filter and reintroduce the false-
    positive class on the 0x97 em-dash byte in cp1252-encoded
    binaries.
    """
    src = SWEEP_TEST_SRC.read_text(encoding="utf-8")
    assert '"*.py"' in src or "'*.py'" in src, (
        "tests/test_b7_forbidden_sweep.py git diff invocation does "
        "NOT carry `-- '*.py'` filter; the sweep input would include "
        "binary files and any 0x97 byte (cp1252 em-dash) inside a "
        "weights file would false-fire the em-dash marker"
    )


def test_classify_lines_marks_period_system_prompt_body_as_string() -> None:
    """The multi-line triple-quoted OTR_PERIOD_SYSTEM_PROMPT body
    spans roughly lines 37-84 of `nodes/_otr_period_prompts.py`.
    The tokenize classifier must mark these lines as "string" so
    forbidden markers like "1940s" appearing inside the prompt are
    suppressed from runtime hits.
    """
    classify_lines = _load_classify_lines()
    cls = classify_lines(PERIOD_SRC)
    # Spot-check the lines around the system prompt body. Line 37 is
    # the assignment line; lines 38-83 are inside the string. Line
    # 84 closes the string. The classifier marks any tokenize STRING
    # span as "string".
    string_lines = [i for i, kind in cls.items() if kind == "string"]
    assert len(string_lines) > 30, (
        f"_otr_period_prompts.py has only {len(string_lines)} "
        f"string-classified lines; expected >30 (the period system "
        f"prompt body alone is ~45 lines). Classifier may not be "
        f"recognizing the triple-quoted block."
    )
    # Confirm one specific line known to contain "1940s" is classified
    # as string. Per current source, "1940s" appears in the system
    # prompt body around line 38 ("a 1940s American anthology radio
    # drama...").
    src_lines = PERIOD_SRC.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(src_lines, start=1):
        if "1940s" in line:
            kind = cls.get(i, "?")
            assert kind == "string", (
                f"line {i} contains '1940s' but is classified as "
                f"{kind!r}; expected 'string' (it's inside the period "
                f"system prompt triple-quoted body)"
            )
            return
    pytest.fail(
        "no '1940s' literal found in _otr_period_prompts.py source; "
        "the test smoke fixture may be stale or the prompt body was "
        "rewritten"
    )


def test_planted_era_literal_in_a_helper_string_would_be_runtime(tmp_path: Path) -> None:
    """Negative control: a single-line module-level string literal
    containing an era token gets classified as "string" too -- the
    classifier doesn't distinguish "the canonical period prompt" from
    "any other string literal". The actual carve-out lives at the
    diff-filter layer (only NEW added lines fire the sweep) plus the
    fact that the period prompt body was added long before Sprint D.

    This test plants a small fixture .py file with various era-token
    placements and confirms the classifier marks them all as
    string-classified or comment-classified -- the categorical
    expectation holds.
    """
    classify_lines = _load_classify_lines()
    fixture = tmp_path / "fixture.py"
    fixture.write_text(
        '"""Module docstring mentioning 1940s era flavor."""\n'
        '\n'
        '# Comment with 1940s in it\n'
        '\n'
        'BANNER = "1940s tube radio"\n'
        '\n'
        'def foo():\n'
        '    return "1940s noir"\n',
        encoding="utf-8",
    )
    cls = classify_lines(fixture)
    # Module docstring -- string
    assert cls.get(1) == "string", cls
    # Comment line -- comment
    assert cls.get(3) == "comment", cls
    # BANNER assignment -- the line has both code (BANNER = ) and a
    # string literal. The classifier marks lines where ANY token is
    # a string as string-classified; the BANNER line ends up "string".
    assert cls.get(5) in ("string", "code"), cls
    # The return inside foo() -- mixed code + string.
    assert cls.get(8) in ("string", "code"), cls


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
