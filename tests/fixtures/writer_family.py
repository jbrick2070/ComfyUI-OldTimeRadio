"""Where the writer's source actually lives, for the tests that read it.

A large family of tests in this suite do not CALL the writer -- they read its
source and pin WHERE its wiring sits: that the story-brief reflection runs at
K.5.5 and not L.5, that the resolved-models stamp precedes the terminal save,
that the title regen passes a premise, that a slot tag accompanies every LLM
call. Those pins are valuable precisely because no behavioural test would
notice if the order quietly changed.

They all hard-coded one path: ``nodes/OTR_LedgerScriptWriter.py``. The lean-mean
order-9 split moved 1,927 lines out of that file into two siblings -- byte for
byte, with a sha256 per block -- and every one of those pins went red at once,
not because a contract broke but because the address did.

So the address lives HERE now, once. ``find_function`` searches the whole family
and returns the definition wherever it is; ``family_source`` concatenates it for
substring pins. A future slice moves a name again and this list is the only
thing that changes.

WHAT THIS DELIBERATELY DOES NOT DO: soften a pin. ``find_function`` raises when
a name is in none of the files, exactly as each test's own private lookup did,
so a function that genuinely disappears still fails loudly. And the ORDER pins
keep working unchanged, because a returned node carries the line numbers of the
one file it came from -- never a concatenation.
"""
from __future__ import annotations

import ast
from pathlib import Path

_NODES = Path(__file__).resolve().parents[2] / "nodes"

#: The writer, in the order a reader should think about it: the node itself,
#: the widget-values resolver (order 9 slice 1), the tail (order 9 slice 2).
WRITER_FAMILY = (
    _NODES / "OTR_LedgerScriptWriter.py",
    _NODES / "_otr_writer_inputs.py",
    _NODES / "_otr_writer_tail.py",
)


def family_source() -> str:
    """Every byte of the writer family, in family order.

    For substring pins ("this call site exists somewhere in the writer"). Do not
    use it for anything that compares LINE NUMBERS -- use :func:`find_function`
    or :func:`tree_of`, which stay inside a single file.
    """
    return "\n".join(path.read_text(encoding="utf-8") for path in WRITER_FAMILY)


def trees() -> "list[tuple[Path, ast.Module]]":
    """(path, parsed module) for each family member, in family order."""
    return [(path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
            for path in WRITER_FAMILY]


def tree_of(name: str) -> ast.Module:
    """The parsed module that DEFINES ``name`` (function, class or method)."""
    for path, tree in trees():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)) and node.name == name:
                return tree
    raise AssertionError(
        "%r is defined in none of the writer family: %s"
        % (name, ", ".join(p.name for p in WRITER_FAMILY)))


def find_function(name: str) -> "ast.FunctionDef | ast.AsyncFunctionDef":
    """The definition of ``name`` anywhere in the family, module level or method.

    Raises ``AssertionError`` when nothing defines it -- a missing pin target is
    still a failure, which is the whole reason these tests exist.
    """
    for _path, tree in trees():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == name:
                return node
    raise AssertionError(
        "function %r is defined in none of the writer family: %s"
        % (name, ", ".join(p.name for p in WRITER_FAMILY)))


def source_defining(name: str) -> str:
    """The full text of the family member that defines ``name``."""
    for path, tree in trees():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)) and node.name == name:
                return path.read_text(encoding="utf-8")
    raise AssertionError(
        "%r is defined in none of the writer family: %s"
        % (name, ", ".join(p.name for p in WRITER_FAMILY)))
