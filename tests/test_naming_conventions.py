"""S10.3 -- Test enforcement of docs/conventions.md.

Pins the two structural rules from the conventions doc:

  1. Library-only private modules MUST follow ``_otr_<name>_lib.py``
     (leading underscore + project prefix + library suffix).
  2. ``_otr_*_lib.py`` modules MUST be library-only (no
     ``NODE_CLASS_MAPPINGS`` export).

Without these tests, the next contributor adding a library module
would have no automated reminder of the rule.
"""
from __future__ import annotations

import ast
import pathlib

NODES_DIR = pathlib.Path(__file__).resolve().parent.parent / "nodes"


def test_lib_modules_have_otr_prefix():
    """Any module whose name ends in ``_lib.py`` MUST also start with
    ``_otr_`` (or the underscore-prefix base form ``_otr_``).

    Catches: a contributor adds ``nodes/_helpers_lib.py`` instead of
    ``nodes/_otr_helpers_lib.py``. The leading ``_`` makes it
    package-private; the ``otr_`` infix scopes the name to this
    project so it doesn't collide visually with sibling projects in
    the same Cowork session.
    """
    bad = []
    for f in NODES_DIR.glob("*_lib.py"):
        if not f.name.startswith("_otr_"):
            bad.append(f.name)
    assert not bad, (
        f"Library modules without _otr_ prefix: {bad}. "
        "Per docs/conventions.md, private library modules under "
        "nodes/ must follow _otr_<name>_lib.py."
    )


def test_lib_modules_have_no_node_class_mappings():
    """``_otr_*_lib.py`` modules MUST be library-only -- they expose
    helpers to other node modules but never register a ComfyUI node
    themselves. AST-walk catches a NODE_CLASS_MAPPINGS assignment at
    any scope inside the file.

    Catches: a contributor adds ``NODE_CLASS_MAPPINGS = {...}`` to
    ``_otr_bark_lib.py`` because they want a node class wrapping the
    helper. The right move is a separate non-prefixed module that
    imports from the lib.
    """
    bad = []
    for f in NODES_DIR.glob("_otr_*_lib.py"):
        src = f.read_text(encoding="utf-8")
        tree = ast.parse(src, str(f))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "NODE_CLASS_MAPPINGS":
                        bad.append((f.name, node.lineno))
                        break
            elif isinstance(node, ast.AnnAssign):
                tgt = node.target
                if isinstance(tgt, ast.Name) and tgt.id == "NODE_CLASS_MAPPINGS":
                    bad.append((f.name, node.lineno))
    assert not bad, (
        f"_otr_*_lib.py modules defining NODE_CLASS_MAPPINGS: {bad}. "
        "Per docs/conventions.md, library-only modules must not "
        "register node classes. Move the class to a non-prefixed "
        "module and import the helpers from the _lib module."
    )
