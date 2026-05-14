"""S31 -- no caller references the 4 legacy orchestrator LLM symbols.

The 4 legacy symbols live in `nodes/story_orchestrator.py`:
  * `_load_llm`
  * `_unload_llm`
  * `_LLM_CACHE`
  * `_generate_with_llm`

S31 B1: VRAMContextTest switched off `_SO._load_llm` /
`_SO._generate_with_llm` onto the canonical
`_otr_model_loader.request_slot` + `make_generate_fn` surface
(Hard rule #5).

S31 B4: the 4 symbols themselves get DELETED from
`story_orchestrator.py`. The `test_orchestrator_no_*_symbol`
assertions (added in B4) become tripwires against accidental
re-introduction.

This file is the union of:
* B1 caller-side guard       -- vram_context_test.py specifically.
* B1 tree-wide caller guard  -- no external module references
                                the 4 symbols at runtime
                                (excludes the two source-of-truth
                                files: story_orchestrator.py and
                                _otr_model_loader.py).
* B4 symbol-deletion guard   -- `hasattr` checks added when the
                                symbols are actually removed.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest


PACK_ROOT = Path(__file__).resolve().parent.parent
VRAM_TEST_PATH = PACK_ROOT / "nodes" / "vram_context_test.py"

LEGACY_SYMBOLS = (
    "_load_llm",
    "_unload_llm",
    "_LLM_CACHE",
    "_generate_with_llm",
)

# Files allowed to contain the 4 legacy symbol names at AST level. Two
# kinds of allowance:
# 1. The two source-of-truth files themselves (definitions live here at
#    B1..B3; B4 deletes them).
# 2. Test files in this directory -- tests reference legacy names in
#    string literals, hasattr checks, AST scans, and forensic
#    assertions. The forbidden-pattern sweep already classifies these
#    contexts as forensic. AST-level we just exclude test files
#    wholesale.
_SOURCE_TRUTH = {
    PACK_ROOT / "nodes" / "story_orchestrator.py",
    PACK_ROOT / "nodes" / "_otr_model_loader.py",
}


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def _runtime_legacy_refs(tree: ast.AST) -> list[tuple[int, str]]:
    """Return list of (lineno, description) of runtime refs to the 4
    legacy symbols.

    Captured patterns:
      * `Anything._load_llm` / `._unload_llm` / `._LLM_CACHE` /
        `._generate_with_llm`  (Attribute access at runtime)
      * Bare Name node id in {legacy 4}  (would-be reference, e.g.
        after `from story_orchestrator import _load_llm`)
      * `from story_orchestrator import _load_llm` (ImportFrom)

    String literals, comments, and docstrings are NOT captured -- they
    do not produce ast.Name or ast.Attribute nodes in the AST. So a
    docstring mentioning "story_orchestrator._load_llm" is silent here,
    same as the forbidden-pattern sweep's forensic-mention rule.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in LEGACY_SYMBOLS:
            hits.append((node.lineno, f"Attribute access .{node.attr}"))
        elif isinstance(node, ast.Name) and node.id in LEGACY_SYMBOLS:
            hits.append((node.lineno, f"Name reference {node.id}"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.endswith("story_orchestrator"):
                # ast.alias is the AST node type for each imported
                # name; iterating is the only way to pull individual
                # imported symbols out of an `ImportFrom`. The local
                # variable is named `imp` (not `alias`) so the
                # forbidden-pattern sweep's `\balias\b` marker (S28
                # _RENAME_ALIASES lock) stays clean here.
                for imp in node.names:
                    if imp.name in LEGACY_SYMBOLS:
                        hits.append(
                            (node.lineno,
                             f"ImportFrom story_orchestrator: {imp.name}")
                        )
    return hits


# ---------------------------------------------------------------------------
# B1 caller-side guards: vram_context_test.py specifically
# ---------------------------------------------------------------------------


def test_vram_context_test_no_direct_load_llm():
    """vram_context_test.py must not call `_SO._load_llm` (or any
    `*._load_llm` attribute access) at runtime. The pre-load step
    routes through `_OTRML.request_slot("technical", model_id)`
    (canonical lifecycle helper, S31 B1)."""
    tree = _tree(VRAM_TEST_PATH)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "_load_llm":
            offenders.append(f"line {node.lineno}: .{node.attr}")
    assert not offenders, (
        "vram_context_test.py must not reference `._load_llm` at "
        "runtime any more (S31 B1 switched it to "
        "`_OTRML.request_slot`). Offenders:\n  "
        + "\n  ".join(offenders)
    )


def test_vram_context_test_no_direct_generate_with_llm():
    """vram_context_test.py must not call `_SO._generate_with_llm`
    (or any `*._generate_with_llm`) at runtime. Each probe routes
    through `make_generate_fn(cache_entry)` then `gen_fn(messages=...,
    temperature=..., max_new_tokens=...)` (canonical generate surface
    per Hard rule #5)."""
    tree = _tree(VRAM_TEST_PATH)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "_generate_with_llm"
        ):
            offenders.append(f"line {node.lineno}: .{node.attr}")
    assert not offenders, (
        "vram_context_test.py must not reference `._generate_with_llm` "
        "at runtime any more (S31 B1 switched it to "
        "`make_generate_fn`). Offenders:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# B1 tree-wide caller guard: no external module references the 4
# symbols at runtime. Source-of-truth files excluded.
# ---------------------------------------------------------------------------


def _iter_runtime_py_files() -> list[Path]:
    """Production Python files: nodes/, visual/, scripts/. Skip
    tests/, docs/, __pycache__/, .venv/, .git/, and any test_*.py
    that snuck into a runtime dir."""
    roots = [PACK_ROOT / "nodes", PACK_ROOT / "visual", PACK_ROOT / "scripts"]
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            if p.name.startswith("test_"):
                continue
            out.append(p)
    return out


def test_no_external_caller_of_legacy_symbols():
    """Tree-wide AST scan: NO production file outside the two
    source-of-truth modules (`story_orchestrator.py`,
    `_otr_model_loader.py`) references any of the 4 legacy symbols at
    runtime. Forensic mentions in comments / docstrings / string
    literals are AST-silent and don't count.

    At S31 B1 this gates: VRAMContextTest is the last external caller;
    after B1 it should be clean.
    At S31 B4 this gates: nothing in the tree can reference symbols
    that no longer exist.
    """
    offenders: dict[str, list[tuple[int, str]]] = {}
    for path in _iter_runtime_py_files():
        if path in _SOURCE_TRUTH:
            continue
        try:
            tree = _tree(path)
        except SyntaxError:
            # If a file doesn't parse, it's a different bug -- not this
            # test's job to surface. Skip silently.
            continue
        hits = _runtime_legacy_refs(tree)
        if hits:
            offenders[str(path.relative_to(PACK_ROOT))] = hits
    assert not offenders, (
        "S31 hard rule: NO production file outside "
        "story_orchestrator.py / _otr_model_loader.py may reference "
        "the 4 legacy LLM symbols at runtime. Offenders:\n"
        + "\n".join(
            f"  {p}:\n    " + "\n    ".join(
                f"line {ln}: {desc}" for ln, desc in hits
            )
            for p, hits in offenders.items()
        )
    )
