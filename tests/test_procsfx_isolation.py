"""S12.2 -- AST-based isolation guard for ProcSFX vs AudioGen cache.

Per S6-S8 round-robin finding F-7: the prior guard
(``test_procsfx_module_does_not_define_cache_lookup_helpers`` in
``tests/test_audiogen_cache_keys.py``) hardcoded four AudioGen
helper symbol names and grepped for ``def <name>`` in the ProcSFX
source. That guard was rename-blind: if AudioGen renamed
``_cache_prefix`` to ``_lookup_prefix``, the grep silently passed
even if ProcSFX imported the new name.

This module replaces that guard with an AST-walk over ProcSFX's
import statements. The structural invariant is "ProcSFX does not
import from ``nodes.batch_audiogen_generator``" -- catches both
``from .batch_audiogen_generator import _cache_prefix`` and
``import nodes.batch_audiogen_generator as ag`` shapes.

Why this is the right invariant
  Cache helpers are AudioGen-specific. ProcSFX is procedural,
  deterministic, and uses content-addressed filenames (post-S12.1).
  The two SFX paths must not share a cache layer; if a future
  patch makes them share, the dur_s invariant on AudioGen's hash
  must explicitly extend to whatever ProcSFX imports. Forcing the
  shared import to surface in a test is the cleanest way to make
  that decision deliberate.
"""
from __future__ import annotations

import ast
import pathlib


def test_procsfx_does_not_import_from_audiogen_cache():
    """Structural invariant: ProcSFX must not import from AudioGen's
    cache module. AST-based so AudioGen helper renames don't bypass
    the guard."""
    src = (
        pathlib.Path(__file__).resolve().parent.parent
        / "nodes" / "batch_procedural_sfx.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)

    forbidden = {"batch_audiogen_generator"}
    violations: list[tuple[str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            # Match exact module name OR suffix match (handles relative
            # imports like ``from .batch_audiogen_generator import ...``
            # which AST parses with module="batch_audiogen_generator").
            for f in forbidden:
                if node.module == f or node.module.endswith("." + f):
                    violations.append((node.module, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                for f in forbidden:
                    if alias.name == f or alias.name.endswith("." + f):
                        violations.append((alias.name, node.lineno))

    assert not violations, (
        f"ProcSFX has forbidden imports from AudioGen cache module: "
        f"{violations}. Cache helpers must not cross module boundaries. "
        f"If you intentionally want to share a cache layer between SFX "
        f"paths, the shared key MUST include dur_s (S12.1 invariant) "
        f"and the new test guard must pin that explicitly."
    )
