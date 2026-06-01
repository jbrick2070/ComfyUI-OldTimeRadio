"""Sprint 3 -- REJECT gate regression (story-spine go-forward refactor).

The Targeted Story QA router can return verdict="REJECT" for a structural
defect a one-line edit cannot fix (a dead ending, a broken turn, an
unclear premise). The abort has two halves, and this guards both
STATICALLY (source / AST scans, no heavy import, no GPU):

  1. Spine side (NEVER raises, PD1): run_post_script_spine, on a REJECT
     verdict, sets meta["story_verdict"]="REJECT", and RETURNS out of the
     branch -- skipping the deterministic scrub.
  2. Writer side (fail-loud): OTR_LedgerScriptWriter.run() raises
     RuntimeError when meta["story_verdict"] == "REJECT", at the writer
     boundary AFTER the spine call, so an aborted run produces no node
     output and the graph stops before the render cascade.

Behavioral coverage of the spine verdict routing lives in the spine's
own __main__ self-test (PASS / fail-open / micro-repair / REJECT). These
pytest checks are the permanent suite-level anchor for the contract.
"""
from __future__ import annotations

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_WRITER_SRC = (_REPO / "nodes" / "OTR_LedgerScriptWriter.py").read_text(encoding="utf-8")
_SPINE_SRC = (_REPO / "nodes" / "_otr_story_spine.py").read_text(encoding="utf-8")


def _find_function(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def test_writer_raises_runtimeerror_on_reject_signal():
    """The writer must raise RuntimeError when meta['story_verdict'] is
    'REJECT' -- the fail-loud half of the abort (the spine never raises)."""
    tree = ast.parse(_WRITER_SRC)
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        left = node.test.left
        is_story_verdict_get = (
            isinstance(left, ast.Call)
            and isinstance(left.func, ast.Attribute)
            and left.func.attr == "get"
            and any(isinstance(a, ast.Constant) and a.value == "story_verdict"
                    for a in left.args)
        )
        compares_reject = any(
            isinstance(c, ast.Constant) and c.value == "REJECT"
            for c in node.test.comparators
        )
        if not (is_story_verdict_get and compares_reject):
            continue
        for stmt in node.body:
            if (isinstance(stmt, ast.Raise)
                    and isinstance(stmt.exc, ast.Call)
                    and isinstance(stmt.exc.func, ast.Name)
                    and stmt.exc.func.id == "RuntimeError"):
                found = True
    assert found, (
        "OTR_LedgerScriptWriter.run() must raise RuntimeError when "
        "meta['story_verdict'] == 'REJECT' (Sprint 3 writer-boundary abort)"
    )


def test_writer_reject_raise_is_after_the_spine_call():
    """The REJECT check must come AFTER run_post_script_spine in source
    order: the spine sets the signal, then the writer acts on it."""
    spine_call = _WRITER_SRC.find("run_post_script_spine")
    reject_check = _WRITER_SRC.find('meta.get("story_verdict") == "REJECT"')
    assert spine_call != -1, "writer must call run_post_script_spine"
    assert reject_check != -1, "writer must check the REJECT signal"
    assert reject_check > spine_call, (
        "the REJECT raise must be checked after the spine call"
    )


def test_spine_reject_branch_returns_and_skips_scrub():
    """run_post_script_spine, on _verdict_value == 'REJECT', must RETURN
    out of the branch (so the deterministic scrub is skipped) and never
    raise -- the PD1 half of the abort."""
    tree = ast.parse(_SPINE_SRC)
    fn = _find_function(tree, "run_post_script_spine")
    assert fn is not None, "run_post_script_spine must exist"
    reject_if = None
    for node in ast.walk(fn):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        left = node.test.left
        if (isinstance(left, ast.Name) and left.id == "_verdict_value"
                and any(isinstance(c, ast.Constant) and c.value == "REJECT"
                        for c in node.test.comparators)):
            reject_if = node
            break
    assert reject_if is not None, (
        "spine must branch on _verdict_value == 'REJECT'"
    )
    assert any(isinstance(s, ast.Return) for s in reject_if.body), (
        "the REJECT branch must return (skip the scrub) -- the spine never "
        "falls through to the scrub on a rejected story"
    )
    # And the branch must not contain a raise (PD1 -- the spine never raises).
    assert not any(isinstance(s, ast.Raise) for s in ast.walk(reject_if)), (
        "the spine REJECT branch must NEVER raise (PD1); the writer raises"
    )


def test_spine_calls_scrub_with_repair_available_false():
    """Sprint 4 wiring: micro-repair runs upstream now, so the scrub is
    always told repair_available=False (mechanical, fail-closed, last)."""
    assert "repair_available=False" in _SPINE_SRC, (
        "scrub_ledger must be called with repair_available=False"
    )
    assert "repair_available=not repair_used" not in _SPINE_SRC, (
        "the old repair_available=not repair_used wiring must be gone"
    )
