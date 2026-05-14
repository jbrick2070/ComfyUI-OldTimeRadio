"""S30 B2c -- cleanup_model_id extinction guard.

The legacy `cleanup_model_id` widget + its associated normalization
helper were retired during S15.5+ writer slim-down; B2c arms the
forbidden-pattern sweep marker + this dedicated guard so a future
contributor cannot accidentally reintroduce the symbol as a Python
identifier (function, parameter, variable, attribute, kwarg).

Forensic string literals in legacy-widget guard lists (e.g. the
writer's INPUT_TYPES self-test's `for legacy in ("cleanup_model_id",
...)` tuple) are explicitly allowed -- the sweep classifies STRING
tokens as forensic, and this test scans Python identifiers, not
arbitrary strings.

Scope: every Python file under nodes/, visual/, scripts/, tests/.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


PACK_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIRS = ("nodes", "visual", "scripts", "tests")
FORBIDDEN_NAME = "cleanup_model_id"


def _python_files() -> list[Path]:
    out: list[Path] = []
    for d in TARGET_DIRS:
        root = PACK_ROOT / d
        if not root.is_dir():
            continue
        out.extend(sorted(root.rglob("*.py")))
    return out


def _identifier_hits_in_tree(tree: ast.AST) -> list[tuple[int, str]]:
    """Walk `tree` and return (lineno, what) for every identifier-form
    use of the forbidden name. Excludes ast.Constant (string literals)
    so forensic mentions in legacy-guard lists, docstrings, etc. don't
    register.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == FORBIDDEN_NAME:
            hits.append((node.lineno, "Name"))
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == FORBIDDEN_NAME
        ):
            hits.append((node.lineno, "Attribute"))
        elif isinstance(node, ast.arg) and node.arg == FORBIDDEN_NAME:
            hits.append((node.lineno, "arg"))
        elif isinstance(node, ast.keyword) and node.arg == FORBIDDEN_NAME:
            hits.append((node.lineno, "keyword"))
        elif isinstance(node, ast.FunctionDef) and node.name == FORBIDDEN_NAME:
            hits.append((node.lineno, "FunctionDef"))
        elif (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == FORBIDDEN_NAME
        ):
            hits.append((node.lineno, "AsyncFunctionDef"))
        elif isinstance(node, ast.ClassDef) and node.name == FORBIDDEN_NAME:
            hits.append((node.lineno, "ClassDef"))
    return hits


def test_no_cleanup_model_id_python_identifiers():
    """AST scan: the forbidden name must not appear as a Python
    identifier anywhere under nodes/, visual/, scripts/, tests/.
    """
    offenders: list[str] = []
    for path in _python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            offenders.append(f"{path}: parse failure -- {exc}")
            continue
        for lineno, kind in _identifier_hits_in_tree(tree):
            offenders.append(
                f"{path.relative_to(PACK_ROOT).as_posix()}:{lineno} [{kind}]"
            )
    assert not offenders, (
        "S30 B2c violation: 'cleanup_model_id' reintroduced as a "
        "Python identifier:\n  " + "\n  ".join(offenders)
    )


def test_no_cleanup_model_id_test_two_llm_split_module():
    """The legacy `tests/test_two_llm_split.py` pinned the
    `_resolve_cleanup_model_id` resolver semantics that the S30
    selector replaces. Its deletion is part of B2c.
    """
    legacy = PACK_ROOT / "tests" / "test_two_llm_split.py"
    assert not legacy.exists(), (
        f"{legacy} must be deleted; it pinned the legacy "
        f"_resolve_cleanup_model_id contract"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
