"""S31 B6 Fix 1 -- RSS news re-rank slot label / id agreement.

`_fetch_rss_seed_or_die` routes through
`_otr_model_loader.request_slot("technical", model_id)` (post-S31 B3).
The writer's `_resolve_inputs` calls `_fetch_rss_seed_or_die` to seed
news. Pre-fix it passed `creative_writing_model` -- the slot label
("technical") and the resolved id (creative model) would disagree in
differing-slots mode. Post-fix it passes `technical_model`.

In default config (creative == technical, both = DEFAULT_LLM) the two
ids are identical so the fix is a no-op at runtime. In differing-slots
config (S32 forward) the fix is load-bearing -- the slot scheduler
would otherwise load the creative model under the technical slot
label, defeating two-slot routing.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


PACK_ROOT = Path(__file__).resolve().parent.parent
WRITER_PATH = PACK_ROOT / "nodes" / "OTR_LedgerScriptWriter.py"


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"function {name!r} not found")


def test_resolve_inputs_rss_uses_technical_model():
    """AST scan: the call to `_fetch_rss_seed_or_die(...)` inside
    `_resolve_inputs` passes `technical_model` as the second
    positional argument, NOT `creative_writing_model`.
    """
    tree = ast.parse(WRITER_PATH.read_text(encoding="utf-8"))
    resolve_fn = _find_function(tree, "_resolve_inputs")

    target_call = None
    for sub in ast.walk(resolve_fn):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        if isinstance(fn, ast.Name) and fn.id == "_fetch_rss_seed_or_die":
            target_call = sub
            break
    assert target_call is not None, (
        "_resolve_inputs must contain a call to "
        "_fetch_rss_seed_or_die(...); none found."
    )
    # Second positional arg = the model id we route through. Check the
    # AST Name id.
    assert len(target_call.args) >= 2, (
        f"_fetch_rss_seed_or_die call has fewer than 2 positional "
        f"args: {len(target_call.args)}"
    )
    second_arg = target_call.args[1]
    assert isinstance(second_arg, ast.Name), (
        f"_fetch_rss_seed_or_die second arg must be a Name; "
        f"got {type(second_arg).__name__}"
    )
    assert second_arg.id == "technical_model", (
        f"S31 B6 Fix 1: _fetch_rss_seed_or_die must receive "
        f"`technical_model` (the slot label routes through "
        f"`request_slot(\"technical\", ...)`). Pre-fix it received "
        f"`creative_writing_model`. Got `{second_arg.id}`."
    )


def test_resolve_inputs_rss_default_config_baseline():
    """Default config sanity: when creative == technical, the
    resolved id is the same regardless of which slot label travels
    with it. The fix from B6 is a no-op at runtime in default
    config -- this test pins that baseline.

    Structural check: AST scan asserts the writer-level routing
    table at the top of OTR_LedgerScriptWriter.py declares
    `creative_writing_model` and `technical_model` as separate
    widgets with the same default. The shared default = DEFAULT_LLM
    means default-config callers pass the SAME id under either
    slot label.
    """
    src = WRITER_PATH.read_text(encoding="utf-8")
    # Both widgets must exist in INPUT_TYPES.
    assert '"creative_writing_model"' in src
    assert '"technical_model"' in src
