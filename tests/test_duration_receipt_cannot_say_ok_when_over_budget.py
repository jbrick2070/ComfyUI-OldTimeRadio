"""The duration receipt must not print OK on an episode that blew its budget.

PBUG-20260830-01, found by the 4060 on live 8 GB hardware. Two lines from the
same node, in the same run, seconds apart:

    [OTR_MasterAudioMux] DURATION MISMATCH (publishing anyway): ... over the
      credits-tail budget (25.2110s [declared] + 0.1200s tol) by 8.8608s
    [OTR_MasterAudioMux] duration_check v=155.120s a=120.928s
      tail_budget=25.2s (declared) OK

The gate failed by 8.86 s and the receipt said OK.

**The mechanism was a mis-bound `else`.** The budget check only ever *logged*;
it recorded no state. The receipt's `else` therefore bound to the PROBE check
immediately above it, so a successful ffprobe printed OK unconditionally and the
only branch that could print anything else was a *failed probe*. The over-budget
case had no branch at all.

This is the same rule the file already states one branch over, for the UNPROVEN
case: *"What must not happen is a receipt claiming a proof it does not have."*
It was applied to the probe failure and missed for the budget failure.

**It got sharper, not milder, when the raise became a warning.** The operator's
2026-08-30 directive -- "don't kill a duration mismatch, just let it fly" -- is
right and stands: refusing to mux discards a finished episode over a length
disagreement. But it means the receipt is now the ONLY compact signal a reader
gets, so an always-OK receipt makes every overshoot invisible in exactly the
summary the node exists to emit.

Checked structurally rather than behaviourally: reaching the real branch needs
ffprobe, two rendered media files and a full mux, and the defect was never in
the arithmetic -- it was in which `if` the `else` attached to. The AST is where
that lives.
"""
from __future__ import annotations

import ast
import pathlib

from nodes import otr_master_audio_mux as mux


def _receipt_chain():
    """The if/elif/else chain that emits the ``duration_check`` verdicts."""
    src = pathlib.Path(mux.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        rendered = ast.unparse(node)
        if "duration_check" in rendered and ") OK" in rendered:
            return node
    raise AssertionError("the duration_check verdict chain was not found")


def _branch_emitting(chain: ast.If, verdict: str):
    """``(test_source, branch)`` for the branch printing ``verdict``."""
    node = chain
    while True:
        body_src = "".join(ast.unparse(s) for s in node.body)
        if verdict in body_src:
            return ast.unparse(node.test), node.body
        rest = node.orelse
        if len(rest) == 1 and isinstance(rest[0], ast.If):
            node = rest[0]
            continue
        if rest:
            else_src = "".join(ast.unparse(s) for s in rest)
            if verdict in else_src:
                return None, rest  # the terminal else
        return None, None


def test_all_three_verdicts_exist():
    chain = _receipt_chain()
    rendered = ast.unparse(chain)
    for verdict in ("UNPROVEN", "OVER_BUDGET", ") OK"):
        assert verdict in rendered, (
            "the duration receipt lost its %r verdict -- OK, UNPROVEN and "
            "OVER_BUDGET are three different claims and collapsing any two "
            "loses information a reader needs" % verdict)


def test_ok_is_not_reachable_while_over_budget():
    """The defect itself: OK must be guarded by the budget verdict."""
    chain = _receipt_chain()

    over_test, over_body = _branch_emitting(chain, "OVER_BUDGET")
    assert over_body is not None, "no branch emits OVER_BUDGET"
    assert over_test and "over_budget" in over_test, (
        "the OVER_BUDGET branch is not keyed on the budget verdict (its test is "
        "%r) -- if it keys on anything else the two failures are conflated again"
        % over_test)

    ok_test, ok_body = _branch_emitting(chain, ") OK")
    assert ok_body is not None, "no branch emits OK"
    assert ok_test is None, (
        "OK is emitted from a guarded branch (%r) rather than the terminal "
        "else. It must be the fall-through that only runs when neither the "
        "probe nor the budget failed." % ok_test)


def test_the_budget_check_records_state_and_does_not_only_log():
    """A check that only warns cannot inform a receipt."""
    src = pathlib.Path(mux.__file__).read_text(encoding="utf-8")
    assert "over_budget_by = None" in src, (
        "the budget verdict is not initialised, so the receipt cannot "
        "distinguish 'not over budget' from 'never checked'")
    assert "over_budget_by = _excess - max_tail_s - tol" in src, (
        "the budget check no longer records its overage. If it only logs, the "
        "receipt has nothing to read and the mis-bound else returns.")


def test_the_overage_rides_the_receipt_line():
    """A reader must not have to go find the warning to get the number.

    Read off the AST branch rather than a character window around the first
    textual match -- the first ``OVER_BUDGET`` in the file is the comment
    explaining the name, and anchoring there measures the prose, not the code.
    """
    chain = _receipt_chain()
    _test_src, over_body = _branch_emitting(chain, "OVER_BUDGET")
    assert over_body is not None, "no branch emits OVER_BUDGET"

    emitted = "".join(ast.unparse(stmt) for stmt in over_body)
    assert "over_budget_by" in emitted, (
        "the OVER_BUDGET line does not carry the overage, so the receipt is "
        "not self-contained -- the number would live only in a warning the "
        "reader may not have in front of them")
