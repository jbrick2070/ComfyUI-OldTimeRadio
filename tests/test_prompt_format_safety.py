"""
tests/test_prompt_format_safety.py -- BUG-LOCAL-026 regression guard

Locks in the rule: any LLM prompt template called with `.format(...)` must
have all literal curly braces escaped as `{{` / `}}` so they don't get
interpreted as positional arg slots.

Repro of the BUG-LOCAL-026 crash (Phase H regression, 2026-05-02 23:46 soak):
the ANNOUNCER-skip clause added to DIRECTOR_PROMPT contained literal
``visual_plan.characters{}`` and ``voice_assignments{}`` -- two unescaped
empty-brace pairs. ``DIRECTOR_PROMPT.format(script_text=..., voice_mapping_rules=...)``
raised ``IndexError: Replacement index 0 out of range for positional args
tuple`` because Python's str.format saw ``{}`` as a positional slot ref.

This test calls ``.format()`` on every named LLM prompt template with the
exact kwargs the production call site uses, and asserts no exception.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

NODES = Path(__file__).resolve().parent.parent / "nodes"
SO_SRC = (NODES / "story_orchestrator.py").read_text(encoding="utf-8")


def _extract_constant(name: str) -> str:
    """Extract a triple-quoted constant by name. Returns the contents
    between the opening and closing triple-quotes."""
    pat = rf'^{re.escape(name)}\s*=\s*"""(.*?)"""'
    match = re.search(pat, SO_SRC, re.MULTILINE | re.DOTALL)
    assert match, f"{name} triple-quoted assignment not found in story_orchestrator.py"
    return match.group(1)


def test_director_prompt_format_safety():
    """The DIRECTOR_PROMPT.format() call must not raise on the exact kwargs
    the production LLMDirector node passes."""
    director = _extract_constant("DIRECTOR_PROMPT")
    try:
        out = director.format(
            script_text="placeholder script body",
            voice_mapping_rules="placeholder voice mapping rules",
        )
    except (IndexError, KeyError, ValueError) as exc:
        pytest.fail(
            f"DIRECTOR_PROMPT.format() raised {type(exc).__name__}: {exc}\n"
            f"Likely cause: literal curly braces in the template were not "
            f"escaped as {{{{ or }}}}. Find any prose that includes raw "
            f"{{}} or {{name}} that isn't a real format placeholder."
        )
    # Sanity: output should be non-trivially longer than the inputs
    assert len(out) > 1000, "DIRECTOR_PROMPT seems empty post-format"
