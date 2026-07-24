"""Sprint D D2b -- the resolver is wired into 4 phase sites.

Phase-name inventory tests check each phase identifier appears
exactly once as a literal string argument across production code.
Catches both under-wiring (missed phase) and silent coverage swap
where one phase is removed and a different one is added.

Per v3 plan §6 D2b acceptance row SD-08:
    test_router_has_exactly_4_production_callers_at_d2b_boundary
        (lives in test_creative_prompt_router.py; flipped from 0
        to 4 at D2b)
    test_outline_phase_wired_to_router_by_name_inventory
    test_line_composer_system_phase_wired_to_router_by_name_inventory
    test_polish_character_phase_wired_to_router_by_name_inventory
    test_polish_announcer_phase_wired_to_router_by_name_inventory

The phase-name inventory tests count the literal phase strings via
AST so they aren't fooled by f-strings or string concatenation.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


PROD_ROOTS = (
    REPO_ROOT / "nodes",
    REPO_ROOT / "visual",
)


def _count_phase_literal_in_router_calls(phase: str) -> tuple[int, list[str]]:
    """Count `resolve_creative_system_prompt(...)` calls where the
    `phase` argument is the literal string `phase`. Excludes the
    router module's own definition file.

    AST-based so docstrings, comments, and string-literal mentions
    in other contexts do not count -- only actual call arguments.
    """
    router_src = (REPO_ROOT / "nodes" / "_otr_creative_prompt_router.py").resolve()
    hits: list[str] = []
    for root in PROD_ROOTS:
        if not root.is_dir():
            continue
        for py in root.rglob("*.py"):
            if py.resolve() == router_src:
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                called_name: str | None = None
                if isinstance(func, ast.Name):
                    called_name = func.id
                elif isinstance(func, ast.Attribute):
                    called_name = func.attr
                if called_name != "resolve_creative_system_prompt":
                    continue
                # Look for `phase=<literal>` kwarg, or 2nd positional.
                phase_value: str | None = None
                for kw in node.keywords:
                    if kw.arg == "phase" and isinstance(kw.value, ast.Constant):
                        if isinstance(kw.value.value, str):
                            phase_value = kw.value.value
                if phase_value is None and len(node.args) >= 2:
                    arg2 = node.args[1]
                    if (
                        isinstance(arg2, ast.Constant)
                        and isinstance(arg2.value, str)
                    ):
                        phase_value = arg2.value
                if phase_value == phase:
                    lineno = getattr(node, "lineno", -1)
                    hits.append(f"{py.relative_to(REPO_ROOT)}:{lineno}")
    return len(hits), hits


@pytest.mark.parametrize("phase", [
    "outline",
    "line_composer_system",
    # Closing layers share a dynamic phase resolver, so their behavior is
    # pinned by tests/test_closing_seams_bank_routing.py rather than this
    # literal-only inventory.
])
def test_each_creative_phase_has_exactly_one_production_callsite(
    phase: str,
) -> None:
    """Phase-name inventory: each literal-wired creative phase has
    exactly one production call site.
    """
    count, hits = _count_phase_literal_in_router_calls(phase)
    assert count == 1, (
        f"phase {phase!r} has {count} production callsites; expected "
        f"exactly 1. Hits:\n  " + "\n  ".join(hits)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
