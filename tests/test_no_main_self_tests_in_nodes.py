"""No module in `nodes/` carries a `__main__` self-test block.

OPERATOR RULING 2026-08-15: *"if they aren't doing anything delete em."*

Eight modules carried one, totalling 1,089 lines, and nothing executed any of
them -- no script, no doc, no CI step invoked a node module as `__main__`. So
they were not coverage; they were a second set of assertions that nobody ran
and that nothing kept honest. Three of them had already been found asserting
things that were false, and `OTR_LedgerScriptWriter.py`'s block still stopped
on a `creative_writing_model` default that had drifted out from under it.

The real suite is the suite. Anything worth keeping from a block like that
gets promoted into a test that actually runs, first.
"""
from __future__ import annotations

import ast
from pathlib import Path


NODES = Path(__file__).resolve().parents[1] / "nodes"


def _has_main_guard(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for sub in ast.walk(node.test):
            if isinstance(sub, ast.Name) and sub.id == "__name__":
                return True
    return False


def test_no_node_module_carries_a_main_self_test():
    offenders = []
    for path in sorted(NODES.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # a genuinely broken file is another test's job
            continue
        if _has_main_guard(tree):
            offenders.append(str(path.relative_to(NODES.parent)))

    assert not offenders, (
        "these modules grew a __main__ block back: "
        + ", ".join(offenders)
        + ". Nothing runs it, so it is not coverage -- put the assertion in "
        "the real suite instead."
    )
