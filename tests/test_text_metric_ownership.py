"""Regression contract for canonical ledger text-metric ownership."""
from __future__ import annotations

import ast
from pathlib import Path

from nodes._otr_text_metrics import (
    canonical_char_count,
    canonical_word_count,
    set_line_text_metrics,
)


REPO = Path(__file__).resolve().parents[1]
NODES = REPO / "nodes"


class _DirectTextWriteVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.function_stack: list[str] = []
        self.direct_writes: list[int] = []
        self.non_atomic_updates: list[int] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _is_self_test(self) -> bool:
        return bool(
            self.function_stack
            and self.function_stack[-1] == "_run_self_test"
        )

    @staticmethod
    def _is_text_target(target: ast.expr) -> bool:
        return (
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "text"
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        if not self._is_self_test() and any(
            self._is_text_target(target) for target in node.targets
        ):
            self.direct_writes.append(node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not self._is_self_test() and self._is_text_target(node.target):
            self.direct_writes.append(node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            not self._is_self_test()
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and node.args
            and isinstance(node.args[0], ast.Dict)
        ):
            keys = {
                key.value
                for key in node.args[0].keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if "text" in keys and not {"char_count", "word_count"} <= keys:
                self.non_atomic_updates.append(node.lineno)
        self.generic_visit(node)


def test_canonical_metric_punctuation_contract():
    assert canonical_word_count("forty-two") == 1
    assert canonical_word_count("don't don\u2019t") == 2
    assert canonical_word_count("off\u2014it's") == 2
    assert canonical_word_count("off\u2013it\u2019s") == 2
    assert canonical_char_count("off\u2014it's") == len("off\u2014it's")


def test_atomic_text_metric_mutator_sets_all_owned_fields():
    row = {"text": "stale", "char_count": 999, "word_count": 999}
    assert set_line_text_metrics(row, "off\u2014it's")
    assert row == {
        "text": "off\u2014it's",
        "char_count": len("off\u2014it's"),
        "word_count": 2,
    }


def test_production_nodes_do_not_bypass_canonical_text_metric_owner():
    violations: list[str] = []
    for path in sorted(NODES.glob("*.py")):
        if path.name == "_otr_text_metrics.py":
            continue
        visitor = _DirectTextWriteVisitor()
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        violations.extend(
            f"{path.name}:{line}: direct text assignment"
            for line in visitor.direct_writes
        )
        violations.extend(
            f"{path.name}:{line}: non-atomic text update"
            for line in visitor.non_atomic_updates
        )
    assert violations == []
