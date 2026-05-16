"""Sprint D D3 -- news_interpreter does NOT route through the creative resolver.

The news_interpreter pass reads modern news articles and produces
the per-episode news brief used by downstream outline / dialogue
generation. It runs on the writer's TECHNICAL slot (modern
English, structured JSON output) -- NOT the creative slot. If a
future commit accidentally routed news_interpreter through
`resolve_creative_system_prompt`, a period model on the creative
slot would route news_interpreter prompts through
OTR_PERIOD_SYSTEM_PROMPT and the modern news grounding would
break.

Single AST assertion: `nodes/news_interpreter.py` (and any
sibling news wiring) contains zero `resolve_creative_system_prompt`
call sites and zero imports from `_otr_creative_prompt_router`.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


NEWS_FILES = (
    REPO_ROOT / "nodes" / "news_interpreter.py",
    REPO_ROOT / "nodes" / "_otr_news_wiring.py",
)


def test_news_interpreter_prompts_come_from_technical_slot_never_routed_through_creative_resolver() -> None:
    """Walk every news_interpreter source file. None of them should
    call resolve_creative_system_prompt or import from
    _otr_creative_prompt_router.
    """
    violations: list[str] = []
    for src_path in NEWS_FILES:
        if not src_path.is_file():
            continue
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # Import-from check
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if "creative_prompt_router" in mod or mod.endswith(
                    "_otr_creative_prompt_router",
                ):
                    violations.append(
                        f"{src_path.relative_to(REPO_ROOT)}:"
                        f"{node.lineno}: from {mod!r} import "
                        f"{[a.name for a in node.names]}"
                    )
            # Call-site check
            if isinstance(node, ast.Call):
                func = node.func
                called_name = None
                if isinstance(func, ast.Name):
                    called_name = func.id
                elif isinstance(func, ast.Attribute):
                    called_name = func.attr
                if called_name == "resolve_creative_system_prompt":
                    violations.append(
                        f"{src_path.relative_to(REPO_ROOT)}:"
                        f"{node.lineno}: calls "
                        f"resolve_creative_system_prompt"
                    )
    assert not violations, (
        "news_interpreter routes through the creative resolver -- "
        "modern news grounding could be polluted by a period model "
        "on the creative slot. Violations:\n  "
        + "\n  ".join(violations)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
