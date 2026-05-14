"""S30 B4b -- RSS news path rewired through request_slot; importers
switched to the modern unload_llm; BUG-LOCAL-226 audit-miss closed.

The full deletion of `story_orchestrator._load_llm` / `_unload_llm` /
`_LLM_CACHE` / `_generate_with_llm` symbols was deferred -- the modern
`_otr_model_loader.load_llm` still delegates to the legacy
implementation for its bitsandbytes / profile-specific body (~600 LOC).
That deeper port is its own follow-up sprint. The audit-miss fix
that B4b ships is structural: the RSS news path no longer holds a
parallel reference to the legacy cache, and the timeout-recovery
path goes through the canonical unload_llm helper.

Tests:
1. test_generate_with_llm_uses_request_slot -- AST scan of the
   refactored `_generate_with_llm` body asserts a `request_slot`
   call exists and the prior `_load_llm(...)` call site is gone.
2. test_run_with_timeout_uses_unload_llm     -- AST scan of
   `_run_with_timeout` asserts the canonical `unload_llm()` call
   replaces the manual `_LLM_CACHE` invalidation block.
3. test_importers_use_new_unload_path        -- AST scan over the
   three production importers (batch_bark_generator, _otr_bark_lib,
   scene_sequencer) asserts each imports `unload_llm` from
   `_otr_model_loader` and NOT from `story_orchestrator`.
4. test_no_orchestrator_unload_llm_import_in_packages -- the broader
   guarantee: NO module under nodes/ / visual/ / scripts/ imports
   `_unload_llm` from `story_orchestrator` after B4b.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


PACK_ROOT = Path(__file__).resolve().parent.parent
ORCH_PATH = PACK_ROOT / "nodes" / "story_orchestrator.py"


def _orch_tree() -> ast.AST:
    return ast.parse(ORCH_PATH.read_text(encoding="utf-8"))


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"function {name!r} not found in {ORCH_PATH}")


def test_generate_with_llm_uses_request_slot():
    """`_generate_with_llm` body must call `request_slot(...)` and
    must NOT call `_load_llm(...)` directly any more.
    """
    tree = _orch_tree()
    func = _find_function(tree, "_generate_with_llm")
    has_request_slot = False
    legacy_load_calls: list[str] = []
    for sub in ast.walk(func):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        if isinstance(fn, ast.Attribute) and fn.attr == "request_slot":
            has_request_slot = True
        elif isinstance(fn, ast.Name) and fn.id == "request_slot":
            has_request_slot = True
        elif isinstance(fn, ast.Name) and fn.id == "_load_llm":
            legacy_load_calls.append(f"line {sub.lineno}")
    assert has_request_slot, (
        "_generate_with_llm must call request_slot(...) to acquire its "
        "cache entry; no such call found"
    )
    assert not legacy_load_calls, (
        "_generate_with_llm must not call _load_llm(...) directly any "
        "more (B4b rewire). Offenders:\n  " + "\n  ".join(legacy_load_calls)
    )


def test_run_with_timeout_uses_unload_llm():
    """`_run_with_timeout` recovery path must call `unload_llm()` from
    `_otr_model_loader`; the manual `_LLM_CACHE` shuffle is gone.
    """
    tree = _orch_tree()
    func = _find_function(tree, "_run_with_timeout")
    has_unload = False
    manual_cache_assign: list[str] = []
    for sub in ast.walk(func):
        if isinstance(sub, ast.Call):
            fn = sub.func
            if (
                isinstance(fn, ast.Attribute)
                and fn.attr == "unload_llm"
            ):
                has_unload = True
            elif isinstance(fn, ast.Name) and fn.id == "unload_llm":
                has_unload = True
        # The deleted block had statements like:
        #   _LLM_CACHE["model"] = None
        # Subscript assignment to Name(_LLM_CACHE) signals the legacy
        # manual-shuffle pattern.
        if (
            isinstance(sub, ast.Assign)
            and len(sub.targets) == 1
            and isinstance(sub.targets[0], ast.Subscript)
            and isinstance(sub.targets[0].value, ast.Name)
            and sub.targets[0].value.id == "_LLM_CACHE"
        ):
            manual_cache_assign.append(f"line {sub.lineno}")
    assert has_unload, (
        "_run_with_timeout must call unload_llm() on timeout recovery "
        "(B4b clean-break)"
    )
    assert not manual_cache_assign, (
        "_run_with_timeout must not manually shuffle _LLM_CACHE keys "
        "(B4b clean-break). Offenders:\n  "
        + "\n  ".join(manual_cache_assign)
    )


@pytest.mark.parametrize(
    "rel_path",
    [
        "nodes/batch_bark_generator.py",
        "nodes/_otr_bark_lib.py",
        "nodes/scene_sequencer.py",
    ],
)
def test_importers_use_new_unload_path(rel_path):
    """Each of the three production importers (batch_bark_generator,
    _otr_bark_lib, scene_sequencer) must import `unload_llm` from
    `_otr_model_loader`, not `_unload_llm` from `story_orchestrator`.
    """
    path = PACK_ROOT / rel_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    legacy_imports: list[str] = []
    new_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for imp_alias in node.names:
                if (
                    module.endswith("story_orchestrator")
                    and imp_alias.name == "_unload_llm"
                ):
                    legacy_imports.append(f"line {node.lineno}")
                if (
                    module.endswith("_otr_model_loader")
                    and imp_alias.name == "unload_llm"
                ):
                    new_imports.append(f"line {node.lineno}")
    assert not legacy_imports, (
        f"{rel_path} still imports _unload_llm from story_orchestrator "
        f"(B4b clean-break). Offenders:\n  " + "\n  ".join(legacy_imports)
    )
    assert new_imports, (
        f"{rel_path} must import `unload_llm` from `_otr_model_loader` "
        f"to free LLM VRAM; no such import found"
    )


def test_no_orchestrator_unload_llm_import_in_packages():
    """Broader guarantee: NO module under nodes/, visual/, scripts/
    pulls `_unload_llm` from `story_orchestrator` after B4b. The
    modern entry point is `_otr_model_loader.unload_llm`.
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
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if not module.endswith("story_orchestrator"):
                        continue
                    for imp_alias in node.names:
                        if imp_alias.name == "_unload_llm":
                            offenders.append(
                                f"{path.relative_to(PACK_ROOT).as_posix()}"
                                f":{node.lineno}"
                            )
    assert not offenders, (
        "B4b violation: production code still imports "
        "_unload_llm from story_orchestrator:\n  "
        + "\n  ".join(offenders)
    )


def test_bug_local_226_marked_fixed_in_bug_log():
    """BUG-LOCAL-226 entry header must read [FIXED ...] in BUG_LOG.md
    (not [LOGGED ...]) after B4b lands.
    """
    text = (PACK_ROOT / "BUG_LOG.md").read_text(encoding="utf-8")
    # Find the BUG-LOCAL-226 heading line.
    for line in text.splitlines():
        if line.startswith("### BUG-LOCAL-226"):
            assert "[FIXED" in line, (
                f"BUG-LOCAL-226 header must carry [FIXED <hash> <date>] "
                f"after B4b; got: {line!r}"
            )
            return
    raise AssertionError("BUG-LOCAL-226 heading not found in BUG_LOG.md")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
