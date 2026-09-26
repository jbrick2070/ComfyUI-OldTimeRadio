"""Assert every registered node class in NODE_CLASS_MAPPINGS has a non-empty DESCRIPTION.

Plan 0f item 1: ComfyUI shows a node's DESCRIPTION class attribute and each input's
"tooltip" in its "?" help panel. Every registered node class must declare a non-empty
DESCRIPTION string explaining what the node does and what a person would change on it.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _mappings() -> dict:
    """The pack's registered nodes, imported the way ComfyUI imports them."""
    sys.path.insert(0, str(_REPO_ROOT.parent))
    try:
        pkg = importlib.import_module(_REPO_ROOT.name)
    finally:
        sys.path.pop(0)
    return dict(getattr(pkg, "NODE_CLASS_MAPPINGS", {}) or {})


NODE_CLASS_MAPPINGS = _mappings()


def test_node_mappings_has_expected_floor():
    """Vacuity guard: ensure NODE_CLASS_MAPPINGS loaded the full registered node suite."""
    assert len(NODE_CLASS_MAPPINGS) >= 20, (
        f"NODE_CLASS_MAPPINGS has only {len(NODE_CLASS_MAPPINGS)} entries; expected at least 20"
    )


@pytest.mark.parametrize("node_name,cls", sorted(NODE_CLASS_MAPPINGS.items()))
def test_every_node_class_has_non_empty_description(node_name, cls):
    """Every registered node class must carry a non-empty string DESCRIPTION."""
    desc = getattr(cls, "DESCRIPTION", None)
    assert desc is not None, f"Node {node_name} ({cls.__name__}) lacks a DESCRIPTION class attribute"
    assert isinstance(desc, str), (
        f"Node {node_name} ({cls.__name__}) DESCRIPTION must be a string, got {type(desc).__name__}"
    )
    stripped = desc.strip()
    assert stripped, f"Node {node_name} ({cls.__name__}) has an empty DESCRIPTION"
    assert "dummy" not in stripped.lower(), (
        f"Node {node_name} ({cls.__name__}) DESCRIPTION contains forbidden word 'dummy'"
    )


def _declared_inputs(cls):
    """(section, name, declaration) for every required and optional input.

    Hidden inputs are excluded: ComfyUI never draws them, so the "?" panel
    has nothing to show for them."""
    spec = cls.INPUT_TYPES()
    for section in ("required", "optional"):
        for name, decl in (spec.get(section) or {}).items():
            yield section, name, decl


@pytest.mark.parametrize("node_name,cls", sorted(NODE_CLASS_MAPPINGS.items()))
def test_every_input_carries_a_tooltip(node_name, cls):
    """The other half of plan 0f item 1: every widget AND every socket the
    "?" panel lists explains itself. 184 of 184 carried one on 2026-09-25."""
    missing = []
    for section, name, decl in _declared_inputs(cls):
        opts = decl[1] if len(decl) > 1 and isinstance(decl[1], dict) else {}
        tip = str(opts.get("tooltip") or "").strip()
        if not tip:
            missing.append(f"{section}.{name}")
        assert "dummy" not in tip.lower(), f"{node_name}.{name}"
    assert not missing, f"{node_name} inputs without a tooltip: {missing}"
