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


def test_generate_with_llm_deleted_at_s31_b4():
    """S30 B4b rewired `_generate_with_llm` to route through
    `request_slot`. S31 B4 took the next step: `_generate_with_llm`
    itself is DELETED (Hard rule #1A non-deferrable). The two RSS
    callers (`_llm_rank_news_candidates`, `_llm_rerank_with_bodies`)
    were refactored onto the canonical `request_slot +
    make_generate_fn` surface at S31 B3, then `_generate_with_llm`
    deleted at B4.

    Original B4b assertion: AST scan that `_generate_with_llm`'s body
    calls `request_slot(...)`. Post-S31 B4: the function doesn't
    exist; assert deletion instead.
    """
    from nodes import story_orchestrator as _so

    assert not hasattr(_so, "_generate_with_llm"), (
        "`_generate_with_llm` deleted at S31 B4. Canonical generate "
        "surface is `_otr_model_loader.request_slot(slot, model_id) "
        "+ make_generate_fn(cache_entry) + fn(messages=..., "
        "temperature=..., max_new_tokens=...)`."
    )


def test_run_with_timeout_uses_safe_invalidation_post_s31_b4():
    """S30 B4b assertion was: `_run_with_timeout` calls `unload_llm()`
    on timeout recovery (replacing a manual `_LLM_CACHE` shuffle).
    S31 B4 fix: that `unload_llm()` call CAUSED the cudaErrorIllegalAddress
    race it claimed to avoid (BUG-LOCAL-228); replaced with the new
    `invalidate_cache_no_gpu_teardown()` helper. This test renames
    accordingly.
    """
    tree = _orch_tree()
    func = _find_function(tree, "_run_with_timeout")
    has_safe_helper = False
    forbidden_unload_calls: list[str] = []
    for sub in ast.walk(func):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        attr = (
            fn.attr if isinstance(fn, ast.Attribute)
            else (fn.id if isinstance(fn, ast.Name) else None)
        )
        if attr == "invalidate_cache_no_gpu_teardown":
            has_safe_helper = True
        elif attr == "unload_llm":
            forbidden_unload_calls.append(f"line {sub.lineno}")
    assert has_safe_helper, (
        "_run_with_timeout must call "
        "invalidate_cache_no_gpu_teardown() on timeout recovery "
        "(S31 B4 fix for BUG-LOCAL-228 CUDA race)."
    )
    assert not forbidden_unload_calls, (
        "_run_with_timeout must NOT call unload_llm() in any branch "
        "post-S31 B4. The B4 fix replaced that call with the GPU-safe "
        "invalidate_cache_no_gpu_teardown helper. See BUG-LOCAL-228."
        " Offenders:\n  " + "\n  ".join(forbidden_unload_calls)
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
