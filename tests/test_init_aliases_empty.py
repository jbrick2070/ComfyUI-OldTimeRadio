"""S29 Phase 6.1 -- pin the _RENAME_ALIASES empty-state.

Per the 2026-05-12 voice-path-cleanbreak directive, the
``_RENAME_ALIASES`` dict in ``__init__.py`` was deleted outright.
The package no longer mirrors legacy class names to the
NODE_CLASS_MAPPINGS keys; every workflow JSON references current
canonical names directly.

This test fires RED if any future contributor re-introduces the
dict (or any equivalent alias-mirror loop). Captures the rule that
adding aliases is an EXPLICIT contract decision -- not something
that can slip in without surfacing in CI.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_rename_aliases_dict_does_not_exist():
    """``__init__._RENAME_ALIASES`` must not be defined.

    Either as an empty dict (`= {}`) or with entries -- both are
    rejected. The package's alias-mirror loop was deleted in
    lockstep; resurrecting the dict means the loop is back too.
    """
    sys.path.insert(0, str(_REPO_ROOT.parent))
    try:
        pkg = importlib.import_module(_REPO_ROOT.name)
    finally:
        sys.path.pop(0)

    assert not hasattr(pkg, "_RENAME_ALIASES"), (
        "__init__._RENAME_ALIASES was deleted in the 2026-05-12 "
        "voice-path-cleanbreak. If you re-introduced it, either "
        "delete the dict and the alias-mirror loop, or amend this "
        "test with an explicit ADR pointer documenting the new "
        "contract."
    )


def test_node_class_mappings_no_bare_name_aliases():
    """NODE_CLASS_MAPPINGS keys must all carry the ``OTR_`` prefix.

    The legacy alias-mirror loop registered both the OTR_-prefixed
    name and a bare alias (e.g. ``OTR_SceneSequencer`` AND
    ``SceneSequencer``). The 2026-05-12 cleanbreak removed the mirror
    loop, but the test guards against re-introduction by walking the
    actual mapping at import time.
    """
    sys.path.insert(0, str(_REPO_ROOT.parent))
    try:
        pkg = importlib.import_module(_REPO_ROOT.name)
    finally:
        sys.path.pop(0)

    bare = [k for k in pkg.NODE_CLASS_MAPPINGS if not k.startswith("OTR_")]
    assert not bare, (
        f"NODE_CLASS_MAPPINGS contains bare-name keys without OTR_ "
        f"prefix: {bare}. Per S29 Phase 6.1, alias mirrors are "
        "extinct -- every registered key must carry the OTR_ prefix."
    )
