"""S31 B3 -- RSS news path off legacy `_generate_with_llm` wrapper.

Per plan B3: the orchestrator's `_fetch_science_news` internal callers
(`_llm_rank_news_candidates`, `_llm_rerank_with_bodies`) refactored
onto the canonical `request_slot("technical", ...) + make_generate_fn`
surface (Hard rule #5: one generate surface, no wrapper-by-another-name).

Additionally per plan B3 pre-grep: `_generate_ltx_style_brief` had zero
callers tree-wide outside its own definition (checked across nodes/,
visual/, scripts/, tests/). Per plan: "0 hits outside its own
definition -> orphaned; add to B3 deletion list (do not refactor dead
code)". The function AND the `_LTX_STYLE_BRIEF_PROMPT` constant it
templated against were both DELETED in B3.

`_generate_with_llm` function definition STAYS alive but uncalled --
deletes at S31 B4 alongside `_load_llm`, `_unload_llm`, `_LLM_CACHE`.
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
    must call `request_slot(...)` AND `make_generate_fn(...)` AND NOT
    call the legacy `_generate_with_llm(...)`. Canonical generate
    surface per Hard rule #5."""
    tree = _orch_tree()

    offenders: dict[str, list[str]] = {}
    for fname in ("_llm_rank_news_candidates", "_llm_rerank_with_bodies"):
        fn = _find_function(tree, fname)
        legacy = _calls_named(fn, "_generate_with_llm")
        request_slot_calls = _calls_named(fn, "request_slot")
        make_gen_fn_calls = _calls_named(fn, "make_generate_fn")
        if legacy:
            offenders.setdefault(fname, []).append(
                f"calls legacy _generate_with_llm at lines "
                f"{[c.lineno for c in legacy]}"
            )
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


def test_scifi_v4_source_floor_requires_length_words_and_token_diversity():
    from nodes.story_orchestrator import _science_body_meets_v4_floor

    diverse = " ".join(
        ["alpha", "beta", "gamma", "delta", "epsilon", "zeta",
         "eta", "theta", "iota", "kappa", "lambda", "mu"] * 7
    )
    assert _science_body_meets_v4_floor(diverse)
    assert not _science_body_meets_v4_floor("alpha " * 90)
    assert not _science_body_meets_v4_floor(" ".join(diverse.split()[:79]))


# ---------------------------------------------------------------------------
# Dead-code deletion + remaining-callers guard
# ---------------------------------------------------------------------------


def test_generate_ltx_style_brief_deleted():
    """Plan B3 dead-code deletion: `_generate_ltx_style_brief` (and the
    `_LTX_STYLE_BRIEF_PROMPT` it templated against) had zero callers
    tree-wide outside their own definitions. Deleted in B3 rather than
    refactored (do not refactor dead code).
    """
    from nodes import story_orchestrator as _so

    assert not hasattr(_so, "_generate_ltx_style_brief"), (
        "_generate_ltx_style_brief should be deleted at S31 B3 "
        "(zero tree-wide callers found at pre-grep). Reintroducing "
        "this function without callers is forbidden -- file a new "
        "sprint plan if a caller emerges."
    )
    assert not hasattr(_so, "_LTX_STYLE_BRIEF_PROMPT"), (
        "_LTX_STYLE_BRIEF_PROMPT should be deleted at S31 B3 alongside "
        "the function that templated against it."
    )


def test_orchestrator_no_remaining_generate_with_llm_callers():
    """Tree-wide assertion: NO function in `story_orchestrator.py`
    (other than `_generate_with_llm`'s own definition) contains a call
    to `_generate_with_llm(...)`. After B3 the legacy wrapper is
    uncalled; B4 deletes it.
    """
    tree = _orch_tree()
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "_generate_with_llm":
            # The function's own body -- recursive calls are checked
            # elsewhere if needed; skip self-walks.
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            fn = sub.func
            if isinstance(fn, ast.Name) and fn.id == "_generate_with_llm":
                offenders.append(
                    f"{node.name}() line {sub.lineno}: _generate_with_llm(...)"
                )
            elif (
                isinstance(fn, ast.Attribute)
                and fn.attr == "_generate_with_llm"
            ):
                offenders.append(
                    f"{node.name}() line {sub.lineno}: .{fn.attr}(...)"
                )
    assert not offenders, (
        "S31 B3 contract: NO orchestrator function may call "
        "`_generate_with_llm(...)` -- the legacy wrapper is uncalled "
        "going into B4 deletion. Offenders:\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# GGUF row registry (2026-07-16): preflight load_config + policy thread the
# whole RSS technical rerank chain, so a gguf technical slot loads its real
# per-row artifact instead of the gemma env-fallback path.
# ---------------------------------------------------------------------------


def test_rss_rank_rerank_thread_load_config_and_policy_to_request_slot():
    """`_llm_rank_news_candidates` + `_llm_rerank_with_bodies` must pass
    BOTH `load_config=` and `policy=` into every `request_slot(...)` call.

    An unthreaded `request_slot("technical", model_id)` leaves load_config
    None, so the gguf backend's env-fallback resolves the GEMMA artifact
    for a Qwen id (silent wrong-model path) -- the exact bug this wiring
    closes for both-slots-Qwen runs."""
    tree = _orch_tree()
    offenders: dict[str, list[str]] = {}
    for fname in ("_llm_rank_news_candidates", "_llm_rerank_with_bodies"):
        fn = _find_function(tree, fname)
        rs_calls = _calls_named(fn, "request_slot")
        if not rs_calls:
            offenders.setdefault(fname, []).append("no request_slot(...) call")
            continue
        for call in rs_calls:
            if not _has_kwarg(call, "load_config"):
                offenders.setdefault(fname, []).append(
                    f"request_slot at line {call.lineno} missing load_config="
                )
            if not _has_kwarg(call, "policy"):
                offenders.setdefault(fname, []).append(
                    f"request_slot at line {call.lineno} missing policy="
                )
    assert not offenders, (
        "GGUF row registry: RSS rank/rerank must thread load_config + "
        "policy into request_slot. Offenders:\n"
        + "\n".join(f"  {k}: {v}" for k, v in offenders.items())
    )


def test_fetch_science_news_forwards_load_config_and_policy():
    """`_fetch_science_news` is the middle link: it must forward
    `load_config=` + `policy=` into BOTH `_llm_rank_news_candidates` and
    `_llm_rerank_with_bodies` so the preflight-resolved config reaches the
    request_slot calls above."""
    tree = _orch_tree()
    fn = _find_function(tree, "_fetch_science_news")
    offenders: dict[str, list[str]] = {}
    for target in ("_llm_rank_news_candidates", "_llm_rerank_with_bodies"):
        calls = _calls_named(fn, target)
        if not calls:
            offenders.setdefault(target, []).append("not called")
            continue
        for call in calls:
            if not _has_kwarg(call, "load_config"):
                offenders.setdefault(target, []).append(
                    f"call at line {call.lineno} missing load_config="
                )
            if not _has_kwarg(call, "policy"):
                offenders.setdefault(target, []).append(
                    f"call at line {call.lineno} missing policy="
                )
    assert not offenders, (
        "GGUF row registry: _fetch_science_news must forward load_config + "
        "policy to rank/rerank. Offenders:\n"
        + "\n".join(f"  {k}: {v}" for k, v in offenders.items())
    )
