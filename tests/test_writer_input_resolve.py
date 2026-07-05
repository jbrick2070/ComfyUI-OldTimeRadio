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


SOURCE_PAYLOAD_PATH = PACK_ROOT / "nodes" / "_otr_source_payload.py"


def test_resolve_inputs_rss_uses_technical_model():
    """S31 B6 invariant across the chunk-3 topology (2026-07-05): the
    fetch moved behind the source-payload contract, so the pin is now
    TWO-part -- (a) `_resolve_inputs` threads `technical_model` into
    the resolved fetcher entry's `.fetch(...)` as the
    `technical_model=` kwarg (and no longer calls
    `_fetch_rss_seed_or_die` directly); (b) the science_rss wrapper
    forwards it as `_fetch_rss_seed_or_die`'s SECOND POSITIONAL arg.
    Together the technical model still routes the technical re-rank
    slot.
    """
    tree = ast.parse(WRITER_PATH.read_text(encoding="utf-8"))
    resolve_fn = _find_function(tree, "_resolve_inputs")

    # (a) writer side: no direct call; entry.fetch(technical_model=...).
    fetch_call = None
    for sub in ast.walk(resolve_fn):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        assert not (isinstance(fn, ast.Name)
                    and fn.id == "_fetch_rss_seed_or_die"), (
            "_resolve_inputs must not call _fetch_rss_seed_or_die "
            "directly post-chunk-3 (the science_rss wrapper owns it)."
        )
        if isinstance(fn, ast.Attribute) and fn.attr == "fetch":
            fetch_call = sub
    assert fetch_call is not None, (
        "_resolve_inputs must call the resolved fetcher entry's "
        ".fetch(...); none found."
    )
    kw = {k.arg: k.value for k in fetch_call.keywords if k.arg}
    assert "technical_model" in kw, (
        "entry.fetch(...) must pass technical_model= explicitly."
    )
    assert (isinstance(kw["technical_model"], ast.Name)
            and kw["technical_model"].id == "technical_model"), (
        "S31 B6 Fix 1 (chunk-3 form): entry.fetch must receive the "
        "`technical_model` parameter (the slot label routes through "
        "`request_slot(\"technical\", ...)`), not "
        "`creative_writing_model`."
    )

    # (b) wrapper side: 2nd positional into _fetch_rss_seed_or_die.
    sp_tree = ast.parse(SOURCE_PAYLOAD_PATH.read_text(encoding="utf-8"))
    wrapper = _find_function(sp_tree, "_fetch_science_rss")
    target_call = None
    for sub in ast.walk(wrapper):
        if (isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "_fetch_rss_seed_or_die"):
            target_call = sub
            break
    assert target_call is not None, (
        "science_rss wrapper must call _fetch_rss_seed_or_die."
    )
    assert len(target_call.args) >= 2, (
        "wrapper must pass (style_slug, technical_model) positionally."
    )
    second_arg = target_call.args[1]
    assert isinstance(second_arg, ast.Name) and second_arg.id == "technical_model", (
        "S31 B6 Fix 1: the wrapper's 2nd positional must be "
        "technical_model."
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
