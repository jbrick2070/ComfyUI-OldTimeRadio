"""S32 B1 -- writer wires paired generators end-to-end.

`OTR_LedgerScriptWriter._resolve_inputs` returns a dict containing
`creative_generate_fn` + `technical_generate_fn` (constructed by the
slot scheduler at write_script entry). The 4 helper call sites
post-S32 B1 must pass BOTH callables as paired kwargs, not as a
single positional `generate_fn`.

Test:
* `test_writer_passes_paired_generators` -- AST scan over
  `OTR_LedgerScriptWriter.py` asserts every call to `pick_style`,
  `lock_cast`, `compose_line`, `build_news_briefs` carries both
  `creative_fn=` and `technical_fn=` keyword arguments. Tripwire
  against partial-port states (rule R4 atomicity violation).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
WRITER_PATH = REPO / "nodes" / "OTR_LedgerScriptWriter.py"


_HELPER_NAMES = ("pick_style", "lock_cast", "compose_line", "build_news_briefs")


def test_writer_passes_paired_generators():
    """For every call site in `OTR_LedgerScriptWriter.py` whose
    callable's terminal attribute is in `_HELPER_NAMES`, assert the
    keyword arguments include both `creative_fn` and `technical_fn`.
    """
    tree = ast.parse(WRITER_PATH.read_text(encoding="utf-8"))

    failures: list[str] = []
    sites_seen: dict[str, int] = {n: 0 for n in _HELPER_NAMES}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match `_OTRSP.pick_style(...)`, `_OTRCAST.lock_cast(...)`, etc.
        if isinstance(func, ast.Attribute) and func.attr in _HELPER_NAMES:
            helper = func.attr
        # Also catch bare `pick_style(...)` if anyone imports directly.
        elif isinstance(func, ast.Name) and func.id in _HELPER_NAMES:
            helper = func.id
        else:
            continue

        sites_seen[helper] += 1
        kwarg_names = {kw.arg for kw in node.keywords if kw.arg is not None}
        missing = []
        if "creative_fn" not in kwarg_names:
            missing.append("creative_fn=")
        if "technical_fn" not in kwarg_names:
            missing.append("technical_fn=")
        if missing:
            failures.append(
                f"line {node.lineno}: `{helper}(...)` missing kwarg(s): "
                f"{', '.join(missing)}"
            )

    # Every helper must have at least one call site in the writer (we
    # know the writer drives all 4).
    not_called = [n for n, count in sites_seen.items() if count == 0]
    assert not not_called, (
        f"Writer is missing call sites for {not_called}; the paired-"
        f"contract wiring can't be verified without them."
    )

    assert not failures, (
        "S32 B1 rule R4 (atomic paired-contract): every helper call "
        "in the writer must pass BOTH `creative_fn=` and "
        "`technical_fn=` kwargs.\nOffenders:\n  "
        + "\n  ".join(failures)
    )
