"""Sprint D D2a -- creative prompt router resolver.

Five assertions per v3 plan:

  test_router_returns_modern_for_default_mistral_nemo
      The default writer slot (Mistral-Nemo) resolves to the modern
      phase prompt for every phase.

  test_router_returns_period_for_talkie_otr_1940s_v1
      Talkie's row resolves to OTR_PERIOD_SYSTEM_PROMPT for every
      phase (one prompt covers all four creative phases under the
      otr_1940s_v1 profile).

  test_router_zero_production_callers_at_d2a_boundary
      `resolve_creative_system_prompt` has zero production callers
      at D2a. D2b will flip to exactly 4 (one per phase site).

  test_router_raises_on_unknown_phase
      Typo at a call site fails loud with ValueError.

  test_render_few_shot_block_has_zero_production_callers
      Encodes the D2c few-shot OMIT decision as a zero-callers test.
      Re-introducing render_few_shot_block fires the test and
      forces a deliberate scope decision.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_creative_prompt_router as router  # noqa: E402
from nodes import _otr_line_composer  # noqa: E402
from nodes import _otr_outline  # noqa: E402
from nodes import _otr_period_prompts  # noqa: E402

MISTRAL_NEMO = "mistralai/Mistral-Nemo-Instruct-2407"
TALKIE = "talkie-lm/talkie-1930-13b-it"

PHASES: tuple[str, ...] = (
    "outline",
    "line_composer_system",
    "polish_character",
    "polish_announcer",
)


# ---------------------------------------------------------------------------
# Resolver behavior
# ---------------------------------------------------------------------------


def test_router_returns_modern_for_default_mistral_nemo() -> None:
    """Every phase under Mistral-Nemo returns the corresponding
    modern phase prompt (object-identity, not just equality, so
    string-interning behavior cannot mask a drift).
    """
    expected_modern_by_phase = {
        "outline":              _otr_outline._SYSTEM_PROMPT,
        "line_composer_system": _otr_line_composer._SYSTEM_PROMPT,
        "polish_character":     _otr_line_composer._POLISH_SYSTEM_PROMPT_CHARACTER,
        "polish_announcer":     _otr_line_composer._POLISH_SYSTEM_PROMPT_ANNOUNCER,
    }
    for phase in PHASES:
        out = router.resolve_creative_system_prompt(MISTRAL_NEMO, phase)
        assert out is expected_modern_by_phase[phase], (
            f"router({MISTRAL_NEMO!r}, {phase!r}) did NOT return the "
            f"modern phase prompt by object identity; phase prompt "
            f"may have been reassigned or the router rebuilt it"
        )


def test_router_returns_period_for_talkie_otr_1940s_v1() -> None:
    """Every phase under talkie returns OTR_PERIOD_SYSTEM_PROMPT
    (one prompt covers all four creative phases under the
    otr_1940s_v1 profile -- the period system prompt is a full
    voice anchor, not phase-specific).
    """
    for phase in PHASES:
        out = router.resolve_creative_system_prompt(TALKIE, phase)
        assert out is _otr_period_prompts.OTR_PERIOD_SYSTEM_PROMPT, (
            f"router({TALKIE!r}, {phase!r}) did NOT return "
            f"OTR_PERIOD_SYSTEM_PROMPT by object identity"
        )


def test_router_raises_on_unknown_phase() -> None:
    """Typo at a call site fails loud with ValueError naming the
    bad phase identifier.
    """
    with pytest.raises(ValueError, match="unknown creative phase"):
        router.resolve_creative_system_prompt(MISTRAL_NEMO, "outlineX")


# ---------------------------------------------------------------------------
# Caller-count invariants (D2a boundary)
# ---------------------------------------------------------------------------


# Production code search roots (excludes tests/ + docs/ + scripts/ scratch).
PROD_ROOTS = (
    REPO_ROOT / "nodes",
    REPO_ROOT / "visual",
)


def _count_callers(symbol_name: str, exclude_path: Path) -> tuple[int, list[str]]:
    """Count `ast.Call` nodes where the called function is named
    `symbol_name`, across production .py files. Excludes
    `exclude_path` (the symbol's own definition file).

    Matches both bare-name calls `fn(...)` and attribute-access
    calls `mod.fn(...)`. AST-based so docstrings, comments,
    string-literal mentions, and `__all__` entries do not count.
    A future commit that re-introduces the symbol by adding any
    of those still passes this test -- only a real CALL site fires
    the counter.

    Returns (count, list of "path:line" hits).
    """
    hits: list[str] = []
    for root in PROD_ROOTS:
        if not root.is_dir():
            continue
        for py in root.rglob("*.py"):
            if py.resolve() == exclude_path.resolve():
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
                if called_name == symbol_name:
                    lineno = getattr(node, "lineno", -1)
                    hits.append(f"{py.relative_to(REPO_ROOT)}:{lineno}")
    return len(hits), hits


def test_router_has_exactly_4_production_callers_at_d2b_boundary() -> None:
    """`resolve_creative_system_prompt` has EXACTLY 4 production
    call-site references at the D2b boundary -- one per creative
    phase. Catches both under-wiring (missed phase) and over-wiring
    (accidental extra call site).

    D2a shipped with 0 callers. D2b wired:
      _otr_outline.generate_outline      (phase="outline")
      _otr_line_composer.compose_line    (phase="line_composer_system")
      _otr_line_composer.polish_line     (phase="polish_announcer")
      _otr_line_composer.polish_line     (phase="polish_character")

    The router module's own definition file is excluded -- this
    test counts CALL sites, not definitions or imports.
    """
    router_src = REPO_ROOT / "nodes" / "_otr_creative_prompt_router.py"
    count, hits = _count_callers("resolve_creative_system_prompt", router_src)
    assert count == 4, (
        f"resolve_creative_system_prompt has {count} production "
        f"caller(s) at the D2b boundary; expected exactly 4. Hits:\n  "
        + "\n  ".join(hits)
    )


def test_render_few_shot_block_has_zero_production_callers() -> None:
    """Encodes the D2c few-shot OMIT decision. The helper exists in
    `_otr_period_prompts` but is intentionally not wired into any
    production-side prompt assembly at D2a. Re-introducing it would
    cost ~600 tokens of context budget per call and should be a
    deliberate D-future scope decision, not a silent add.
    """
    period_src = REPO_ROOT / "nodes" / "_otr_period_prompts.py"
    count, hits = _count_callers("render_few_shot_block", period_src)
    assert count == 0, (
        f"render_few_shot_block has {count} production caller(s) at "
        f"the D2a boundary; expected 0 per the v3 plan D2c few-shot "
        f"OMIT decision. If a deliberate re-introduction is intended, "
        f"update this test to assert the new expected count and "
        f"document the few-shot context-budget cost in the commit "
        f"body. Hits:\n  " + "\n  ".join(hits)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
