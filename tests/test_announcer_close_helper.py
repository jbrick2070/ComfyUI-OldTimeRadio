"""B4: the announcer-close helper must be WIRED, not merely correct.

WHY THIS FILE EXISTS. On 2026-08-04 a fix for the spoken-citation defect was
applied to ``spoken_coda_line()`` -- a function with ZERO callers -- and thirty
episodes leaked after it "landed". The lesson promoted to the Bug Bible from
that incident is flat: **a fix applied to a function with no callers is not a
fix.** ``_compose_and_stamp_announcer_close`` was extracted so tests can reach
the production reader, and that is worth nothing if ``run()`` stops calling it.

These assertions are STATIC (AST over the real source) rather than behavioural,
and that is deliberate: booting ``OTR_LedgerScriptWriter.run()`` needs an
outline, a cast lock, a slot scheduler and a live model lane, so a behavioural
reachability test would be so heavily mocked that it would prove the mocks call
the helper rather than that the NODE does. Reading the shipped source cannot be
satisfied by a stub.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WRITER = REPO / "nodes" / "OTR_LedgerScriptWriter.py"
HELPER = "_compose_and_stamp_announcer_close"

#: The complete keyword contract. `script_brief` is in this list because
#: omitting it was a real, independently-found defect: the fictional-outro
#: branch passes it to `compose_announcer_outro` and it is bound far outside
#: the extracted range, so a helper without it raises NameError on that branch
#: alone -- invisible to any test that only exercises the coda route.
EXPECTED_KEYWORDS = frozenset({
    "first_announcer_id",
    "last_announcer_id",
    "provenance_owned",
    "style_grammar_on",
    "effective_spoken_fact",
    "nc_brief",
    "script_brief",
    "premise",
    "resolved",
    "slot_scheduler",
    "creative_generate_fn",
})


def _tree() -> ast.Module:
    return ast.parse(WRITER.read_text(encoding="utf-8"))


def _run_function() -> ast.FunctionDef:
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "run":
                    return sub
    raise AssertionError("OTR_LedgerScriptWriter.run() not found")


def _helper_calls(fn: ast.FunctionDef) -> list[ast.Call]:
    return [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == HELPER
    ]


def test_helper_exists_at_module_level():
    """Module level, so a test can import it without constructing the node."""
    names = [n.name for n in _tree().body if isinstance(n, ast.FunctionDef)]
    assert HELPER in names, (
        f"{HELPER} must be a module-level function; module-level defs are {names}")


def test_run_calls_the_helper_exactly_once():
    """THE ZERO-READER GUARD. If this drops to 0 the extraction is decorative."""
    calls = _helper_calls(_run_function())
    assert len(calls) == 1, (
        f"run() must call {HELPER} exactly once, found {len(calls)}")


def test_the_call_passes_the_complete_keyword_contract():
    call = _helper_calls(_run_function())[0]
    passed = {kw.arg for kw in call.keywords if kw.arg}
    missing = EXPECTED_KEYWORDS - passed
    extra = passed - EXPECTED_KEYWORDS
    assert not missing, f"call omits {sorted(missing)} -- NameError on some branch"
    assert not extra, f"call passes unexpected {sorted(extra)}"
    # led and meta ride positionally, matching the signature.
    assert len(call.args) == 2, "led and meta are positional"


def test_the_result_is_assigned_not_discarded():
    """The caller logs `outro_res.compose_flags`. Dropping the return value is a
    NameError at that log line -- which is exactly how the FIRST proposed
    boundary for this extraction was wrong."""
    run = _run_function()
    assigned = False
    for node in ast.walk(run):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == HELPER:
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                assert targets == ["outro_res"], (
                    f"expected `outro_res = {HELPER}(...)`, got {targets}")
                assigned = True
    assert assigned, f"{HELPER}'s return value is discarded"


def _helper_function() -> ast.FunctionDef:
    for node in _tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == HELPER:
            return node
    raise AssertionError(f"{HELPER} not found at module level")


def test_caller_still_saves_and_logs_after_the_call():
    """`led.save()` and the flag log stay in the CALLER -- that is what keeps the
    helper in-memory and therefore testable without touching disk.

    Located by AST line number, not by string search: searching the source text
    for the helper name finds its DEFINITION long before its call site.
    """
    run = _run_function()
    call_line = _helper_calls(run)[0].lineno

    saves_after = [
        n.lineno for n in ast.walk(run)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "save"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "led"
        and n.lineno > call_line
    ]
    assert saves_after, "led.save() must follow the helper call in run()"

    flags_after = [
        n.lineno for n in ast.walk(run)
        if isinstance(n, ast.Attribute)
        and n.attr == "compose_flags"
        and isinstance(n.value, ast.Name)
        and n.value.id == "outro_res"
        and n.lineno > call_line
    ]
    assert flags_after, "the caller must still log outro_res.compose_flags"


def test_news_meta_stays_in_the_caller():
    """The trap this extraction is famous for: `news_meta` is defined above the
    block and read below it by the key-term audit. Moving it into the helper is
    a NameError on every episode."""
    run = _run_function()
    bound = {
        t.id
        for node in ast.walk(run)
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    assert "news_meta" in bound, "news_meta must remain bound in run()"

    # AST over the helper's own node -- string slicing on module text picked up
    # the class below it and reported a leak that was not there.
    leaked = [
        n.id for n in ast.walk(_helper_function())
        if isinstance(n, ast.Name) and n.id == "news_meta"
    ]
    assert not leaked, (
        "news_meta leaked into the helper -- NameError in the caller's key-term "
        "audit on every episode")
