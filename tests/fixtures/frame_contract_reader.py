"""The ONE reader of "did this lane DECLARE its continuity mode?".

MOVED HERE FROM `nodes/_otr_video_engines/frame_contract.py` (2026-09-05). It
lived in the shipped engine module with a stated reason -- "the gate and every
lane's own test must ask the SAME question, and two readers of one invariant is
how they drift" -- and that reason is honoured here, not broken: this file is
still the single reader, imported by the preflight gate and by all nine lane
tests.

What changed is only WHERE it lives. Three independent reviewers (GPT-6 Astra,
Gemini 3.8 Flash, Sonnet) each confirmed no production code calls it; its only
consumers are tests. Shipping a test helper inside `nodes/` costs the registry
scanner a file to read and the reader a false impression that the render path
uses it. Neither is worth paying.

`can_chain` came along for the same reason: it is a one-line predicate over
`frame_contract_for(...)`, and `coverage_plan.py` inlines the identical
comparison rather than calling it.
"""
from __future__ import annotations

import ast
import inspect
import textwrap

try:
    from nodes._otr_video_engines import frame_contract as _fc
except ImportError:  # pragma: no cover -- flat (sys.path) load
    from _otr_video_engines import frame_contract as _fc  # type: ignore


def declares_continuity_kwarg(engine) -> bool:
    """True iff a ``FrameContract(...)`` call in this engine's class body passes
    ``continuity`` as a KEYWORD -- i.e. the join mode was DECIDED, not defaulted.

    The value alone cannot answer this. ``CONTINUITY_NONE`` is the dataclass
    default, so a lane that never thought about chaining and a lane that
    concluded NONE after reading its own render path are byte-identical at
    runtime. That is exactly why G3.3 asks the question at all (lesson L3).

    AST-BASED, AND THAT IS THE POINT (lane 12, 2026-08-11). The check used to be
    a substring search for ``"continuity="`` over the class's SOURCE TEXT. It
    worked while nobody discussed continuity in a comment, and it silently
    rotted the moment lanes 10-12 began WRITING DOWN why each lane's value is
    NONE -- every one of those comments contains that literal, so the check
    would have gone green for a lane whose real declaration had been deleted,
    satisfied by the paragraph explaining the declaration it no longer had.

    Comments are not AST nodes, so prose cannot satisfy this.

    Never raises: unreadable or unparseable source answers False, which is the
    safe direction (it reports "not declared", never a false pass).
    """
    if engine is None:
        return False
    for base in type(engine).__mro__:
        if base is object:
            continue
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(base)))
        except Exception:  # noqa: BLE001 -- unreadable source is not a pass
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
            if name == "FrameContract" and any(
                    kw.arg == "continuity" for kw in node.keywords):
                return True
    return False


def can_chain(engine) -> bool:
    """True iff segment N+1 may begin exactly on segment N's terminal frame.

    The one thing still EARNED per adapter. Multi-clip is universal; a seamless
    join is not, because only an engine that genuinely locks frame 0 to a
    supplied image can deliver one. Everything else jump cuts, honestly.
    """
    return (_fc.frame_contract_for(engine).continuity
            == _fc.CONTINUITY_STRICT_FIRST_FRAME)
