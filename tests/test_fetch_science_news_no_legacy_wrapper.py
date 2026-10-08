"""S31 B3 -- RSS news path runs on the canonical generate surface.

The orchestrator's `_fetch_science_news` internal callers
(`_llm_rank_news_candidates`, `_llm_rerank_with_bodies`) call
`request_slot("technical", ...) + make_generate_fn` (Hard rule #5: one
generate surface, no wrapper-by-another-name), thread the resolved policy
through, and pin their sampling arguments.
"""

from __future__ import annotations

import ast
from pathlib import Path


PACK_ROOT = Path(__file__).resolve().parent.parent
ORCH_PATH = PACK_ROOT / "nodes" / "story_orchestrator.py"


def _orch_tree() -> ast.AST:
    return ast.parse(ORCH_PATH.read_text(encoding="utf-8"))


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"function {name!r} not found in {ORCH_PATH}")


def _calls_named(func: ast.FunctionDef, target: str) -> list[ast.Call]:
    """Return all `ast.Call` nodes inside `func` whose function is a bare
    `Name(target)` or `Attribute(attr=target)`."""
    out: list[ast.Call] = []
    for sub in ast.walk(func):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        if isinstance(fn, ast.Name) and fn.id == target:
            out.append(sub)
        elif isinstance(fn, ast.Attribute) and fn.attr == target:
            out.append(sub)
    return out


def _kwarg_value(call: ast.Call, name: str):
    """Return the literal value of a keyword argument `name` on `call`,
    or None if absent / non-literal."""
    for kw in call.keywords:
        if kw.arg == name:
            try:
                return ast.literal_eval(kw.value)
            except (ValueError, SyntaxError):
                return None
    return None


def _has_kwarg(call: ast.Call, name: str) -> bool:
    """True iff `call` passes a keyword argument named `name` (any value)."""
    return any(kw.arg == name for kw in call.keywords)


# ---------------------------------------------------------------------------
# Structural assertions on the refactored RSS news path
# ---------------------------------------------------------------------------


def test_fetch_science_news_uses_request_slot():
    """Both `_llm_rank_news_candidates` and `_llm_rerank_with_bodies`
    must call `request_slot(...)` AND `make_generate_fn(...)`. Canonical
    generate surface per Hard rule #5."""
    tree = _orch_tree()

    offenders: dict[str, list[str]] = {}
    for fname in ("_llm_rank_news_candidates", "_llm_rerank_with_bodies"):
        fn = _find_function(tree, fname)
        request_slot_calls = _calls_named(fn, "request_slot")
        make_gen_fn_calls = _calls_named(fn, "make_generate_fn")
        if not request_slot_calls:
            offenders.setdefault(fname, []).append(
                "missing required `request_slot(...)` call"
            )
        if not make_gen_fn_calls:
            offenders.setdefault(fname, []).append(
                "missing required `make_generate_fn(...)` call"
            )

    assert not offenders, (
        "S31 B3 contract: RSS news path uses canonical "
        "request_slot + make_generate_fn surface. Offenders:\n"
        + "\n".join(f"  {k}: {v}" for k, v in offenders.items())
    )


def test_fetch_science_news_news_rank_args():
    """_llm_rank_news_candidates calls gen_fn with `temperature=0.05`
    (the argmax-stable tiny-positive trick) and `max_new_tokens=64`
    (~64 tokens of comma-separated indices)."""
    tree = _orch_tree()
    fn = _find_function(tree, "_llm_rank_news_candidates")
    # Look for the gen_fn(...) call -- the one inside `_do_rank_call`.
    # Identify it by `messages=[...]` keyword shape.
    target_call = None
    for sub in ast.walk(fn):
        if not isinstance(sub, ast.Call):
            continue
        if any(kw.arg == "messages" for kw in sub.keywords):
            target_call = sub
            break
    assert target_call is not None, (
        "no gen_fn(messages=...) call found in _llm_rank_news_candidates"
    )
    assert _kwarg_value(target_call, "temperature") == 0.05, (
        f"_llm_rank_news_candidates must call with temperature=0.05; "
        f"got {_kwarg_value(target_call, 'temperature')!r}"
    )
    assert _kwarg_value(target_call, "max_new_tokens") == 64, (
        f"_llm_rank_news_candidates must call with max_new_tokens=64; "
        f"got {_kwarg_value(target_call, 'max_new_tokens')!r}"
    )


def test_fetch_science_news_body_rerank_args():
    """_llm_rerank_with_bodies calls gen_fn with `max_new_tokens=8`
    (single-index pick output)."""
    tree = _orch_tree()
    fn = _find_function(tree, "_llm_rerank_with_bodies")
    target_call = None
    for sub in ast.walk(fn):
        if not isinstance(sub, ast.Call):
            continue
        if any(kw.arg == "messages" for kw in sub.keywords):
            target_call = sub
            break
    assert target_call is not None, (
        "no gen_fn(messages=...) call found in _llm_rerank_with_bodies"
    )
    assert _kwarg_value(target_call, "max_new_tokens") == 8, (
        f"_llm_rerank_with_bodies must call with max_new_tokens=8; "
        f"got {_kwarg_value(target_call, 'max_new_tokens')!r}"
    )
    assert _kwarg_value(target_call, "temperature") == 0.05, (
        f"_llm_rerank_with_bodies must call with temperature=0.05; "
        f"got {_kwarg_value(target_call, 'temperature')!r}"
    )


# ---------------------------------------------------------------------------
# The preflight-resolved policy threads the whole RSS technical rerank chain,
# so the rerank runs under the SAME policy the writer resolved rather than
# whatever request_slot would derive for itself.
#
# This pair used to pin a per-slot `load_config=` alongside it, for a writer
# backend removed 2026-09-24. The policy half is the half that survived, and
# it is the half that was always load-bearing for every other lane.
# ---------------------------------------------------------------------------


def test_rss_rank_rerank_thread_the_policy_to_request_slot():
    """`_llm_rank_news_candidates` + `_llm_rerank_with_bodies` must pass
    `policy=` into every `request_slot(...)` call.

    An unthreaded `request_slot("technical", model_id)` makes request_slot
    resolve its own policy, so the rerank can run under a different device /
    attention / quantisation than the writer already committed to -- a silent
    divergence inside one episode."""
    tree = _orch_tree()
    offenders: dict[str, list[str]] = {}
    for fname in ("_llm_rank_news_candidates", "_llm_rerank_with_bodies"):
        fn = _find_function(tree, fname)
        rs_calls = _calls_named(fn, "request_slot")
        if not rs_calls:
            offenders.setdefault(fname, []).append("no request_slot(...) call")
            continue
        for call in rs_calls:
            if not _has_kwarg(call, "policy"):
                offenders.setdefault(fname, []).append(
                    f"request_slot at line {call.lineno} missing policy="
                )
    assert not offenders, (
        "RSS rank/rerank must thread the resolved policy into "
        "request_slot. Offenders:\n"
        + "\n".join(f"  {k}: {v}" for k, v in offenders.items())
    )


def test_fetch_science_news_forwards_the_policy():
    """`_fetch_science_news` is the middle link: it must forward `policy=`
    into BOTH `_llm_rank_news_candidates` and `_llm_rerank_with_bodies` so the
    preflight-resolved policy reaches the request_slot calls above."""
    tree = _orch_tree()
    fn = _find_function(tree, "_fetch_science_news")
    offenders: dict[str, list[str]] = {}
    for target in ("_llm_rank_news_candidates", "_llm_rerank_with_bodies"):
        calls = _calls_named(fn, target)
        if not calls:
            offenders.setdefault(target, []).append("not called")
            continue
        for call in calls:
            if not _has_kwarg(call, "policy"):
                offenders.setdefault(target, []).append(
                    f"call at line {call.lineno} missing policy="
                )
    assert not offenders, (
        "_fetch_science_news must forward the resolved policy to "
        "rank/rerank. Offenders:\n"
        + "\n".join(f"  {k}: {v}" for k, v in offenders.items())
    )
