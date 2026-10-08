"""S29 Phase 6.1 -- every registered node key carries the ``OTR_`` prefix.

The package does not mirror legacy class names to the NODE_CLASS_MAPPINGS
keys; every workflow JSON references current canonical names directly. This
test fires RED if a bare-name alias key shows up in the live mapping.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


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
