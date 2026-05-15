"""S31 B4 -- `_run_with_timeout` uses GPU-safe invalidation.

S30 B4b introduced a regression: `_run_with_timeout`'s timeout-
recovery path called `_otr_model_loader.unload_llm()`. The comment
claimed the unload "avoids cudaErrorIllegalAddress from orphan
worker still on GPU" -- but the new behavior actively CAUSES that
error: `model.to("cpu")` + `torch.cuda.empty_cache()` race with the
orphan worker thread that is still executing CUDA kernels on the
cached model.

S31 B4 fix: replaced with `invalidate_cache_no_gpu_teardown()` -- a
new lifecycle helper that clears `LLM_CACHE` dict references WITHOUT
touching the GPU. The orphan thread holds the model in its stack
frame and exits naturally. The next `request_slot` loads a fresh
model cleanly.

See BUG-LOCAL-228 in BUG_LOG.md for the regression details.

This test pins the structural fix via AST scan of `_run_with_timeout`:
* `invalidate_cache_no_gpu_teardown(...)` call MUST be present in the
  timeout-recovery branch.
* `unload_llm(...)` MUST NOT be called in the timeout-recovery branch
  (it would re-introduce the race).
"""

from __future__ import annotations

import ast
from pathlib import Path


PACK_ROOT = Path(__file__).resolve().parent.parent
ORCH_PATH = PACK_ROOT / "nodes" / "story_orchestrator.py"


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"function {name!r} not found")


def test_run_with_timeout_uses_safe_invalidation():
    tree = ast.parse(ORCH_PATH.read_text(encoding="utf-8"))
    fn = _find_function(tree, "_run_with_timeout")

    helper_calls: list[int] = []
    unload_calls: list[int] = []
    for sub in ast.walk(fn):
        if not isinstance(sub, ast.Call):
            continue
        f = sub.func
        attr = (
            f.attr if isinstance(f, ast.Attribute)
            else (f.id if isinstance(f, ast.Name) else None)
        )
        if attr == "invalidate_cache_no_gpu_teardown":
            helper_calls.append(sub.lineno)
        elif attr == "unload_llm":
            unload_calls.append(sub.lineno)

    assert helper_calls, (
        "S31 B4 contract: `_run_with_timeout` MUST call "
        "`invalidate_cache_no_gpu_teardown()` in the timeout-recovery "
        "path. No such call found."
    )
    assert not unload_calls, (
        "S31 B4 contract: `_run_with_timeout` MUST NOT call "
        "`unload_llm()` in any branch (S30 B4b regression that caused "
        "BUG-LOCAL-228 CUDA-race). Offenders:\n  "
        + "\n  ".join(f"line {ln}: unload_llm(...)" for ln in unload_calls)
    )


def test_invalidate_helper_is_canonical_for_timeout_recovery():
    """The TIMEOUT_RECOVERY log message documents the new behavior:
    'LLM_CACHE invalidated (GPU untouched...)'. Audit that the
    structural fix landed with the log message updated to match
    (no stale message claiming `unload_llm()` was called)."""
    src = ORCH_PATH.read_text(encoding="utf-8")
    # The literal text expected in the post-fix log message.
    assert "TIMEOUT_RECOVERY: LLM_CACHE invalidated" in src, (
        "B4 log message for the timeout-recovery branch did not land. "
        "Expected the post-fix message containing "
        "'TIMEOUT_RECOVERY: LLM_CACHE invalidated' (signaling GPU was "
        "untouched). A stale message claiming `unload_llm()` was "
        "called would mask the BUG-LOCAL-228 regression in logs."
    )
    # The stale message must be gone.
    assert "TIMEOUT_RECOVERY: unload_llm()" not in src, (
        "Stale TIMEOUT_RECOVERY log message still present in source. "
        "Expected the B4 update to remove the prior message claiming "
        "`unload_llm()` was invoked. See BUG-LOCAL-228."
    )
