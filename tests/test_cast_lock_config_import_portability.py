"""``config`` imports inside ``nodes/`` must be relative-first, never bare.

PBUG-20260825-02. ``CastLock._assign_bark_announcer`` imported the cast-pools
module with a bare ``from config import cast_pools`` -- no relative attempt
first, no fallback. That works only when the repo root happens to be on
``sys.path`` as a bare entry, which is true under pytest and under
``otr_canonical_api_run.py`` (both add the repo root themselves), but NOT under
a real ComfyUI installation, where the pack is loaded as a submodule of
``custom_nodes`` and its own root is never added to ``sys.path`` directly.

So every proof leg run against the API script tonight passed by ACCIDENT, and
the bug reproduced on the very first real ComfyUI Desktop render on the 4060 --
18 minutes into a run, at the last step (announcer voice casting), with
``ModuleNotFoundError: No module named 'config'``.

This is not a one-off: the same two-tier shape (relative attempt, absolute
fallback) already existed at 10+ other ``cast_pools`` call sites across
``nodes/_otr_casting.py``, ``nodes/_otr_voice_bank.py``,
``nodes/_otr_voice_route.py``, ``nodes/_otr_scifi_news_pro.py``, and ANOTHER
function in this exact same file three lines above
(``_resolve_lemmy_voice_policy``, :49-55) -- proof that the pattern was known
and simply not applied here. This test pins the shape so it cannot regress at
any of the guarded call sites, and would have failed on the bug as filed.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Every production file with a cast_pools import, and every function inside it
# that imports cast_pools. A new call site added here without the guard is
# exactly the PBUG-20260825-02 shape.
_FILES = (
    "nodes/cast_lock.py",
    "nodes/_otr_casting.py",
    "nodes/_otr_voice_bank.py",
    "nodes/_otr_voice_route.py",
    "nodes/_otr_scifi_news_pro.py",
)


def _functions_importing_cast_pools(tree: ast.AST):
    """Map enclosing-function-name -> list of ImportFrom nodes that import
    `cast_pools`, for every such import in the module (module-level imports
    are keyed under "<module level>")."""
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    def enclosing_function(node):
        cur = node
        while id(cur) in parents:
            cur = parents[id(cur)]
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return cur
        return None

    hits: dict[str, list[ast.ImportFrom]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in (
            "config", "..config", ".config",
        ):
            names = {a.name for a in node.names}
            if "cast_pools" not in names:
                continue
            fn = enclosing_function(node)
            key = fn.name if fn is not None else "<module level>"
            hits.setdefault(key, []).append(node)
    return hits


def test_every_cast_pools_import_site_is_guarded():
    """No bare `from config import cast_pools` may exist outside a `try` whose
    `except ImportError` (or `except (ImportError, ValueError)`) falls back to
    the other import form. Mirrors the shape at cast_lock.py:49-55."""
    violations = []

    for rel in _FILES:
        path = REPO_ROOT / rel
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))

        # Map each `ast.Try` node's line RANGE so we can ask "is this import
        # inside some try block whose handler falls back to the sibling
        # import form".
        try_blocks = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]

        hits = _functions_importing_cast_pools(tree)
        for fn_name, imports in hits.items():
            for imp in imports:
                guarded = False
                for tb in try_blocks:
                    body_lines = {
                        s.lineno for s in ast.walk(tb) if hasattr(s, "lineno")
                    }
                    if imp.lineno not in body_lines:
                        continue
                    # It sits inside a try. The try must have at least one
                    # ImportError-catching handler (directly, or nested via a
                    # second try inside the handler -- the fallback pattern).
                    for handler in tb.handlers:
                        exc = handler.type
                        names = set()
                        if isinstance(exc, ast.Name):
                            names.add(exc.id)
                        elif isinstance(exc, ast.Tuple):
                            names.update(
                                e.id for e in exc.elts if isinstance(e, ast.Name)
                            )
                        if "ImportError" in names:
                            guarded = True
                    if guarded:
                        break
                if not guarded:
                    violations.append(
                        f"{rel}:{imp.lineno} in {fn_name}() -- bare "
                        f"'from {imp.module} import cast_pools', not inside "
                        "a try/except ImportError fallback"
                    )

    assert not violations, (
        "unguarded cast_pools import(s) found -- these work by accident under "
        "pytest/otr_canonical_api_run.py (repo root on sys.path) and crash "
        "under a real ComfyUI install (PBUG-20260825-02):\n"
        + "\n".join(violations)
    )


def test_assign_bark_announcer_specifically_has_the_fallback():
    """Direct regression pin for the exact function that broke on the 4060."""
    src = (REPO_ROOT / "nodes/cast_lock.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    target = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_assign_bark_announcer"
    )
    fn_src = ast.get_source_segment(src, target)
    assert fn_src is not None

    assert "from ..config import cast_pools" in fn_src, (
        "the relative-first import is gone from _assign_bark_announcer"
    )
    assert "from config import cast_pools" in fn_src, (
        "the absolute fallback import is gone from _assign_bark_announcer"
    )
    # The bare, unguarded form (no leading whitespace before `from config`,
    # i.e. not nested inside the except block) must not be present.
    bare = [
        line for line in fn_src.splitlines()
        if line.strip() == "from config import cast_pools as _POOLS  # type: ignore"
        and not line.startswith((" " * 12, "\t\t\t"))
    ]
    assert not bare, f"found an unguarded absolute import: {bare}"
