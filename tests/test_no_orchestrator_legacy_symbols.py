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


# ---------------------------------------------------------------------------
# Folded in from `tests/test_b4b_rss_rewire.py` @ S31.5 B2
# ---------------------------------------------------------------------------
# `test_b4b_rss_rewire.py` was originally the S30 B4b "RSS news LLM
# path rewire" regression test file. S31 B4 deleted its primary
# subjects (`_generate_with_llm`, `_unload_llm`, `_LLM_CACHE`,
# `_load_llm`) wholesale. After S31 B4, only the importer-pattern
# guards and the BUG_LOG status check remained; those are folded
# below as the canonical structural pin against re-introducing the
# `from story_orchestrator import _unload_llm` import path.


_IMPORTER_PATHS = (
    # nodes/batch_bark_generator.py retired in the audio clean-break (1a); the
    # bark inference lib (_otr_bark_lib.py) carries the unload_llm handoff now.
    "nodes/_otr_bark_lib.py",
    "nodes/scene_sequencer.py",
)


@pytest.mark.parametrize("rel_path", _IMPORTER_PATHS)
def test_importers_use_new_unload_path(rel_path):
    """Production importers must use a modern `_otr_model_loader` unload
    handoff, NOT `_unload_llm` from `story_orchestrator`. Cloud-safe handoff
    sites may import `unload_llm_if_local_resident`; local-only handoff sites
    may import `unload_llm`.
    """
    path = PACK_ROOT / rel_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    legacy_imports: list[str] = []
    new_imports: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        for imp in node.names:
            if (
                module.endswith("story_orchestrator")
                and imp.name == "_unload_llm"
            ):
                legacy_imports.append(f"line {node.lineno}")
            if (
                module.endswith("_otr_model_loader")
                and imp.name in ("unload_llm", "unload_llm_if_local_resident")
            ):
                new_imports.append(f"line {node.lineno}")
    assert not legacy_imports, (
        f"{rel_path} still imports _unload_llm from "
        f"story_orchestrator. Offenders:\n  "
        + "\n  ".join(legacy_imports)
    )
    assert new_imports, (
        f"{rel_path} must import a modern unload helper from "
        f"`_otr_model_loader` to free local LLM VRAM; no such import found"
    )


def test_no_orchestrator_unload_llm_import_in_packages():
    """Broader guarantee: NO module under nodes/, visual/, scripts/
    pulls `_unload_llm` from `story_orchestrator` after S31 B4. The
    modern entry point is `_otr_model_loader.unload_llm`. Folded
    from `test_b4b_rss_rewire.py` at S31.5 B2.
    """
    offenders: list[str] = []
    for d in ("nodes", "visual", "scripts"):
        root = PACK_ROOT / d
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = node.module or ""
                if not module.endswith("story_orchestrator"):
                    continue
                for imp in node.names:
                    if imp.name == "_unload_llm":
                        offenders.append(
                            f"{path.relative_to(PACK_ROOT).as_posix()}"
                            f":{node.lineno}"
                        )
    assert not offenders, (
        "Production code still imports _unload_llm from "
        "story_orchestrator:\n  " + "\n  ".join(offenders)
    )


def test_bug_local_226_marked_fixed_in_bug_log():
    """BUG-LOCAL-226 entry header must read `[FIXED ...]` in
    `BUG_LOG.md` after S31 B4. Folded from `test_b4b_rss_rewire.py`
    at S31.5 B2.
    """
    text = (PACK_ROOT / "BUG_LOG.md").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("### BUG-LOCAL-226"):
            assert "[FIXED" in line, (
                f"BUG-LOCAL-226 header must carry "
                f"`[FIXED <hash> <date>]` after S31 B4; got: {line!r}"
            )
            return
    raise AssertionError("BUG-LOCAL-226 heading not found in BUG_LOG.md")


# ---------------------------------------------------------------------------
# Folded in from `tests/test_unload_synchronize_guard.py` @ S31.5 B2
# ---------------------------------------------------------------------------
# `test_unload_synchronize_guard.py` was originally the BUG-LOCAL-073
# regression test pinning `synchronize` before `model.cpu()` inside
# the legacy `story_orchestrator._unload_llm` body. After S31 B4
# deletion of that function the file collapsed to a single deletion
# guard which is now an exact duplicate of
# `test_orchestrator_no_unload_llm_symbol` above. Nothing to fold
# beyond what already exists; the file was DELETED at S31.5 B2.


# ---------------------------------------------------------------------------
# B2 file-deletion sanity guards
# ---------------------------------------------------------------------------


def test_b4b_rss_rewire_file_does_not_exist():
    """`tests/test_b4b_rss_rewire.py` was DELETED at S31.5 B2.
    Its surviving assertions live in the section above this one.
    Tripwire against accidental restoration.
    """
    assert not (PACK_ROOT / "tests" / "test_b4b_rss_rewire.py").exists(), (
        "tests/test_b4b_rss_rewire.py was deleted at S31.5 B2 -- "
        "surviving assertions folded into this file. Restoration "
        "would create duplicate tests."
    )


def test_unload_synchronize_guard_file_does_not_exist():
    """`tests/test_unload_synchronize_guard.py` was DELETED at
    S31.5 B2. Its single deletion-guard test is already covered by
    `test_orchestrator_no_unload_llm_symbol` above. Tripwire against
    accidental restoration.
    """
    assert not (PACK_ROOT / "tests" / "test_unload_synchronize_guard.py").exists(), (
        "tests/test_unload_synchronize_guard.py was deleted at S31.5 "
        "B2 -- its single deletion guard duplicates "
        "`test_orchestrator_no_unload_llm_symbol`."
    )


# ---------------------------------------------------------------------------
# S31.5 B3: `_vram_cleanup_via_loader` wrapper elimination (Outcome A)
# ---------------------------------------------------------------------------


def test_no_vram_cleanup_via_loader_wrapper():
    """`story_orchestrator._vram_cleanup_via_loader` was a thin try/except
    wrapper added at S31 B4 to satisfy `register_vram_cleanup`'s callback
    contract. S31.5 B3 audited the contract:
    `_vram_log.force_vram_offload` already wraps each registered
    callback in `try/except: pass`, so a no-raise wrapper is NOT
    required -- the contract is "no-arg callable" only. The wrapper
    was eliminated; `_otr_model_loader.unload_llm` registers directly.
    """
    from nodes import story_orchestrator as _so

    assert not hasattr(_so, "_vram_cleanup_via_loader"), (
        "`_vram_cleanup_via_loader` was eliminated at S31.5 B3 "
        "(Outcome A). Reintroduction creates pure delegation "
        "overhead -- the cleanup-callback contract does not require "
        "a no-raise wrapper (see `_vram_log.force_vram_offload`). "
        "If a future signature change forces the wrapper back, "
        "document the requirement in the new function's docstring."
    )


# ---------------------------------------------------------------------------
# B4 symbol-deletion guards: the 4 legacy symbols MUST NOT EXIST on
# `story_orchestrator` at all post-S31 B4. Tripwires against re-introduction.
# ---------------------------------------------------------------------------


def test_orchestrator_no_load_llm_symbol():
    """S31 B4 deletion guard: `_load_llm` removed from
    `story_orchestrator`."""
    from nodes import story_orchestrator as _so

    assert not hasattr(_so, "_load_llm"), (
        "`_load_llm` deleted at S31 B4 (Hard rule #1A non-deferrable). "
        "Re-introduction is forbidden -- file a new sprint plan if a "
        "caller emerges. Canonical surface: "
        "`_otr_model_loader.load_llm`."
    )


def test_orchestrator_no_unload_llm_symbol():
    """S31 B4 deletion guard: `_unload_llm` removed."""
    from nodes import story_orchestrator as _so

    assert not hasattr(_so, "_unload_llm"), (
        "`_unload_llm` deleted at S31 B4. Canonical surface: "
        "`_otr_model_loader.unload_llm` (or "
        "`invalidate_cache_no_gpu_teardown` for timeout recovery "
        "paths where GPU teardown is unsafe)."
    )


def test_orchestrator_no_llm_cache_symbol():
    """S31 B4 deletion guard: module-level `_LLM_CACHE` dict
    removed."""
    from nodes import story_orchestrator as _so

    assert not hasattr(_so, "_LLM_CACHE"), (
        "`_LLM_CACHE` module-level dict deleted at S31 B4. Canonical "
        "cache surface: `_otr_model_loader.LLM_CACHE` (modern 3-key "
        "shape: model_id / slot / cache_entry)."
    )


def test_orchestrator_no_generate_with_llm_symbol():
    """S31 B4 deletion guard: `_generate_with_llm` legacy wrapper
    removed."""
    from nodes import story_orchestrator as _so

    assert not hasattr(_so, "_generate_with_llm"), (
        "`_generate_with_llm` deleted at S31 B4. Hard rule #5: ONE "
        "generate surface. Canonical: "
        "`request_slot(slot, model_id) -> make_generate_fn(cache_entry)"
        " -> fn(messages, *, temperature, max_new_tokens)`."
    )
