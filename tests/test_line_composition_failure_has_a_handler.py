"""An unfillable beat is not a dead render, and the ledger already knew that.

`compose_line` raises `LineCompositionFailedError` when the model returns
nothing sayable after every attempt. Until 2026-08-15 NO `except` for it
existed anywhere in `nodes/`, and both writer call sites were bare -- so one
empty beat surfaced as a traceback out of the middle of the episode loop.

The disposition is not invented here. `_otr_ledger_freeze` already states the
row contract as "(text non-empty) OR (skip=True with a reason)", and
`_otr_ledger_cleanup` already converts a voiced row with nothing sayable into
an explicit skip carrying `empty_spoken_text_at_ledger_cleanup`. So the writer
leaves the row EMPTY, writes no prose of its own, and the existing owner
completes the record. One beat shorter is an observation; length is not a
contract.

An AST scan rather than a text grep: a `try` that stopped covering the call,
or an `except` narrowed to something else, has to fail this.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

WRITER_PATH = REPO / "nodes" / "OTR_LedgerScriptWriter.py"
COMPOSER_PATH = REPO / "nodes" / "_otr_line_composer.py"
CLEANUP_PATH = REPO / "nodes" / "_otr_ledger_cleanup.py"


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _handler_names(handler: ast.ExceptHandler) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(handler.type) if handler.type is not None else ():
        if isinstance(sub, ast.Attribute):
            names.add(sub.attr)
        elif isinstance(sub, ast.Name):
            names.add(sub.id)
    return names


def test_every_writer_compose_line_call_handles_the_failure():
    tree = ast.parse(WRITER_PATH.read_text(encoding="utf-8"))

    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not any(
            "LineCompositionFailedError" in _handler_names(handler)
            for handler in node.handlers
        ):
            continue
        for body_node in node.body:
            for sub in ast.walk(body_node):
                if isinstance(sub, ast.Call) and _call_name(sub) == "compose_line":
                    guarded.add(sub.lineno)

    all_calls = {
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "compose_line"
    }

    assert all_calls, "no compose_line call sites found; the scan moved"
    unguarded = sorted(all_calls - guarded)
    assert not unguarded, (
        f"compose_line at OTR_LedgerScriptWriter.py lines {unguarded} can "
        f"raise LineCompositionFailedError with no handler -- one beat the "
        f"model cannot fill would kill the whole episode"
    )


def test_the_failure_is_the_only_thing_that_call_site_swallows():
    """A bare `except Exception` there would hide real corruption."""
    tree = ast.parse(WRITER_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        calls = {
            _call_name(sub)
            for body_node in node.body for sub in ast.walk(body_node)
            if isinstance(sub, ast.Call)
        }
        if "compose_line" not in calls:
            continue
        for handler in node.handlers:
            names = _handler_names(handler)
            assert "LineCompositionFailedError" in names, (
                f"the try around compose_line (line {node.lineno}) catches "
                f"{sorted(names) or ['everything']}, which is wider than the "
                f"one failure it is there to dispose of"
            )


def test_the_ledger_cleanup_really_does_own_the_empty_row():
    """The disposition depends on this; prove the owner still exists."""
    source = CLEANUP_PATH.read_text(encoding="utf-8")
    assert '_EMPTY_TEXT_SKIP_REASON = "empty_spoken_text_at_ledger_cleanup"' in source
    assert 'row["skip"] = True' in source
    assert 'row["tts_skip_reason"] = _EMPTY_TEXT_SKIP_REASON' in source


def test_the_writer_writes_no_sentence_of_its_own_on_that_path():
    """Python may not author broadcast prose, not even a placeholder."""
    tree = ast.parse(WRITER_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not any(
            "LineCompositionFailedError" in _handler_names(handler)
            for handler in node.handlers
        ):
            continue
        for handler in node.handlers:
            for sub in ast.walk(handler):
                if isinstance(sub, ast.Call) and _call_name(sub) == "LineResult":
                    text_kwargs = [
                        kw for kw in sub.keywords if kw.arg == "text"
                    ]
                    assert text_kwargs, "LineResult must state its text"
                    value = text_kwargs[0].value
                    assert isinstance(value, ast.Constant) and value.value == "", (
                        "the failure path must leave the row EMPTY for the "
                        "ledger cleanup to mark skipped, never fill it with a "
                        "Python-authored line"
                    )


def test_the_composer_still_raises_when_it_truly_cannot_compose():
    """The handler is a disposition, not a reason to stop failing."""
    source = COMPOSER_PATH.read_text(encoding="utf-8")
    assert "raise LineCompositionFailedError(attempts=attempts, request=req)" in source
