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
        """True for a LEDGER LINE's `text`, which must move with its counts.

        `ui["text"]` is exempt, and it is a different `text` entirely rather
        than a loophole. It is ComfyUI's node-preview contract -- a list of
        strings the canvas draws under the node -- so it has no char_count or
        word_count to keep in step and nothing downstream reads it as spoken
        content. The rule this file enforces is about a ledger row whose
        metrics would silently disagree with its words.

        Kept as narrow as the evidence allows: exactly one `ui["text"]` write
        exists in nodes/ (otr_master_audio_mux, naming where the episode was
        published or why it was withheld), and no ledger write anywhere uses a
        variable named `ui`. Matching on the OBJECT rather than skipping the
        file means a real `row["text"]` added to that same module is still
        caught.
        """
        if not (
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "text"
        ):
            return False
        # NAME *AND* SHAPE. Matching only the object name would excuse a real
        # ledger write that happened to be bound to a local called `ui`.
        # ComfyUI's preview contract is a LIST of strings; a ledger row's text
        # is a plain string, so requiring a list literal on the right-hand side
        # separates them by shape as well as by name.
        if not (isinstance(target.value, ast.Name)
                and target.value.id == "ui"):
            return True
        assigned = getattr(target, "_otr_assigned_value", None)
        return not isinstance(assigned, (ast.List, ast.ListComp))

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            # The shape of the assigned VALUE decides the ui exemption, so the
            # target has to know what is being assigned to it.
            setattr(target, "_otr_assigned_value", node.value)
        if not self._is_self_test() and any(
            self._is_text_target(target) for target in node.targets
        ):
            self.direct_writes.append(node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # The attribute has to be set HERE too. Without it the shape check read
        # a missing value on this path, so an annotated `ui["text"]` could
        # never take the exemption -- the check was dead on one of its two
        # branches and nobody would have noticed until someone annotated one.
        setattr(node.target, "_otr_assigned_value", node.value)
        if not self._is_self_test() and self._is_text_target(node.target):
            self.direct_writes.append(node.lineno)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """`row["text"] += ...` was invisible to this visitor entirely.

        An augmented assignment mutates the text without touching char_count or
        word_count, which is precisely the drift this file exists to prevent,
        and it walked past every branch. It is caught on ANY object -- there is
        no ui exemption here, because ComfyUI's preview list is built, not
        appended to, so a `+=` on a "text" key is a ledger write or a mistake.
        """
        if self._is_self_test():
            self.generic_visit(node)
            return
        target = node.target
        is_text_key = (
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "text"
        )
        if is_text_key:
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
