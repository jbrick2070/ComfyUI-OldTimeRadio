"""Sprint D D3 -- forbidden-sweep scope and carve-out behavior.

The sweep helper at `tests/_s28_forbidden_sweep.py` plus its driver test
already covers these properties at D3 landing:

  1. The driver only diffs `*.py` files (`git diff ... -- "*.py"`).
     A `.safetensors`, `.wav`, `.pt`, etc. binary cannot trip the
     0x97 cp1252 em-dash false-positive class.

  2. Era literals INSIDE `OTR_PERIOD_SYSTEM_PROMPT` (and
     `PERIOD_EXEMPLARS`) sit inside tokenize string spans. The sweep
     judges each match by its span, so those are FORENSIC, not RUNTIME.

This file pins both properties against future regression:

  test_forbidden_sweep_test_driver_restricts_to_py_extension_only
      Source inspection of `tests/test_b7_forbidden_sweep.py`
      confirms the git diff invocation carries `-- "*.py"`.

  test_the_period_system_prompt_era_literals_are_forensic
      Smoke against the live `nodes/_otr_period_prompts.py` -- every
      era literal in it sits inside a string span, so each would be
      forensic even if it appeared on a new ADDED diff line.

  test_era_literals_in_strings_and_comments_are_forensic
      A planted fixture: a docstring, a comment, a module string and a
      returned string each carry an era literal, and each is forensic.
      The executable-hit side is pinned in test_b7_forbidden_sweep.py.
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


def _load_sweep():
    """Import tests/_s28_forbidden_sweep.py. The sweep itself runs only
    under ``main()``, so importing it walks no diff."""
    sweep_path = REPO_ROOT / "tests" / "_s28_forbidden_sweep.py"
    spec = importlib.util.spec_from_file_location("_s28_sweep_for_test", sweep_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_the_period_system_prompt_era_literals_are_forensic() -> None:
    """The multi-line triple-quoted OTR_PERIOD_SYSTEM_PROMPT body is one
    string span, so every era literal in it is a forensic mention, not a
    runtime hit.
    """
    sweep = _load_sweep()
    spans = sweep.string_and_comment_spans(PERIOD_SRC)
    assert spans is not None, "tokenize could not read _otr_period_prompts.py"
    covered = [i for i, s in spans.items() if s]
    assert len(covered) > 30, (
        f"_otr_period_prompts.py has only {len(covered)} lines inside a "
        f"string or comment; the period system prompt body alone is ~45")
    src_lines = PERIOD_SRC.read_text(encoding="utf-8").splitlines()
    hits = [(i, line) for i, line in enumerate(src_lines, start=1)
            if "1940s" in line and sweep.forbidden.search(line)]
    assert hits, ("no '1940s' literal found in _otr_period_prompts.py; the "
                  "smoke fixture is stale or the prompt body was rewritten")
    for i, line in hits:
        assert sweep.classify_hit(spans, i, line) == "forensic", (
            f"line {i} carries '1940s' outside every string span")


def test_era_literals_in_strings_and_comments_are_forensic(tmp_path: Path) -> None:
    """A docstring, a comment, a module string and a returned string each
    carry an era literal; the sweep calls every one of them a mention."""
    sweep = _load_sweep()
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
    spans = sweep.string_and_comment_spans(fixture)
    lines = fixture.read_text(encoding="utf-8").splitlines()
    for lineno in (1, 3, 5, 8):
        text = lines[lineno - 1]
        assert sweep.forbidden.search(text), "fixture line %d must hit" % lineno
        assert sweep.classify_hit(spans, lineno, text) == "forensic", lineno


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
