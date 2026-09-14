"""Every COMBO widget's default must be one of its OWN choices.

THE FAILURE THIS CATCHES, and it has happened at least twice. ComfyUI resolves
a COMBO whose declared default matches no entry in its choice list by falling
through to INDEX 0. So the node renders, runs, and produces a result -- using a
different option from the one it claims. Nothing errors and nothing logs.

Measured on 2026-09-14, before this guard existed: `OTR_LedgerScriptWriter`'s
`technical_model` declared `default="Qwen/Qwen3.5-4B"` while its choices read
`"Qwen/Qwen3.5-4B (8.7 GB, mac16-tight nv8 nv16 nv24)"` -- the label carries a
size and fit badge and THE BADGE IS PART OF THE VALUE. The default matched
nothing. It was correct only by the accident that that row sorts first in the
catalog, and would have become a silent wrong-model bug the day the catalog
order changed.

`nodes/_otr_model_catalog.py::default_llm_option()` exists precisely to hand out
the badged label, and its docstring records the previous occurrence (both writer
widgets rendering red, 2026-08-04). The helper existed and the call site did not
use it, which is why a guard is needed rather than a fix.

WHY THIS IS REPO-WIDE rather than pinned to the writer: the defect class is "a
COMBO default was hand-built or taken from a bare constant instead of from the
same function that builds the choices". Any node with a dynamic choice list can
grow it. There are 25 registered nodes.

A NOTE ON DYNAMIC LISTS. Several choice lists change with the environment -- the
remote-lane pickers show a sentinel until their lane is enabled, and the model
catalog grows rows when a key is set. That is fine and is exactly why this
compares the default against the choices AS DECLARED IN THE SAME CALL, rather
than against a frozen expectation. Whatever the environment, a node must not
offer a default it is not also offering as a choice.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _mappings() -> dict:
    """The pack's registered nodes, imported the way ComfyUI imports them."""
    sys.path.insert(0, str(_REPO.parent))
    try:
        pkg = importlib.import_module(_REPO.name)
    finally:
        sys.path.pop(0)
    return dict(getattr(pkg, "NODE_CLASS_MAPPINGS", {}) or {})


NODE_CLASS_MAPPINGS = _mappings()


def _combo_widgets(cls):
    """`(section, name, choices, default)` for every COMBO this class declares."""
    spec = cls.INPUT_TYPES()
    out = []
    for section in ("required", "optional"):
        block = spec.get(section) or {}
        if not isinstance(block, dict):
            continue
        for name, decl in block.items():
            if not isinstance(decl, (list, tuple)) or not decl:
                continue
            choices = decl[0]
            if not isinstance(choices, (list, tuple)):
                continue                      # not a COMBO
            opts = decl[1] if len(decl) > 1 and isinstance(decl[1], dict) else {}
            if "default" not in opts:
                continue                      # no default -> index 0 on purpose
            out.append((section, name, list(choices), opts["default"]))
    return out


def test_the_registry_actually_loaded():
    """A guard over an empty registry passes on everything."""
    assert len(NODE_CLASS_MAPPINGS) >= 20, (
        "only %d node classes resolved -- the comparisons below would silently "
        "cover almost nothing" % len(NODE_CLASS_MAPPINGS))


@pytest.mark.parametrize("node_name", sorted(NODE_CLASS_MAPPINGS))
def test_every_combo_default_is_one_of_its_own_choices(node_name):
    cls = NODE_CLASS_MAPPINGS[node_name]
    if not hasattr(cls, "INPUT_TYPES"):
        pytest.skip("%s declares no INPUT_TYPES" % node_name)

    problems = []
    for section, name, choices, default in _combo_widgets(cls):
        if default in choices:
            continue
        near = [c for c in choices if isinstance(c, str)
                and isinstance(default, str) and c.startswith(default)]
        hint = ""
        if near:
            hint = ("\n      the choices DO offer %r -- the default is the same "
                    "value with its suffix stripped, which is the badge-vs-bare "
                    "mistake this guard is named for" % near[0])
        problems.append(
            "  %s.%s (%s): default=%r is not among its %d choices%s\n"
            "      choices[0]=%r"
            % (node_name, name, section, default, len(choices), hint, choices[0]))

    assert not problems, (
        "a COMBO offers a default it does not offer as a choice. ComfyUI "
        "resolves an unmatched COMBO to INDEX 0, so the node runs a different "
        "option from the one it declares -- silently, with no error and no log "
        "line. Build the default with the SAME function that builds the "
        "choices.\n%s" % "\n".join(problems))


def test_the_guard_would_catch_a_stripped_suffix():
    """THE MUTATION TEST. A guard that cannot fire is indistinguishable from no
    guard, and this file exists because a real helper was ignored at a real call
    site for weeks."""
    choices = ["Qwen/Qwen3.5-4B (8.7 GB, mac16-tight nv8 nv16 nv24)",
               "google/gemma-4-12b-it (23.9 GB, nv16 nv24)"]
    bare = "Qwen/Qwen3.5-4B"
    assert bare not in choices, "precondition: the bare id must not be a choice"
    near = [c for c in choices if c.startswith(bare)]
    assert near, "the comparison cannot see that the badged twin exists"
    assert choices[0].startswith(bare), (
        "the fixture should mirror the real case, where the unmatched default "
        "silently resolves to the RIGHT row by accident of ordering")
