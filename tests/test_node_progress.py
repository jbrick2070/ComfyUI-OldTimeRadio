# -*- coding: utf-8 -*-
"""The long silent loops draw ComfyUI's progress bar (plan 0f item 2).

The helper is exercised directly against a stand-in ``comfy.utils``; the
WIRING is asserted at its three real call sites by source inspection,
because a test that only calls the helper proves the helper, never that a
node uses it.

Headless. No ComfyUI server, no model, no GPU.
"""
from __future__ import annotations

import ast
import pathlib
import sys
import types

import pytest

from nodes._otr_shared import node_progress as np_mod
from nodes._otr_shared.node_progress import NodeProgress

REPO = pathlib.Path(__file__).resolve().parents[1]


class _Interrupted(BaseException):
    """Stands in for comfy.model_management.InterruptProcessingException."""


@pytest.fixture()
def fake_comfy(monkeypatch):
    calls = []
    behaviour = {"raise": None}

    class ProgressBar:
        def __init__(self, total, node_id=None):
            calls.append(("new", total))

        def update_absolute(self, value, total=None, preview=None):
            if behaviour["raise"] is not None:
                raise behaviour["raise"]
            calls.append(("at", value, total))

    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")
    utils.ProgressBar = ProgressBar
    comfy.utils = utils
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)
    return calls, behaviour


def test_the_bar_follows_the_items(fake_comfy):
    calls, _ = fake_comfy
    bar = NodeProgress(3, "t")
    bar.at(0)
    bar.step()
    bar.step()
    bar.finish()
    assert calls == [("new", 3), ("at", 0, 3), ("at", 1, 3), ("at", 2, 3), ("at", 3, 3)]


def test_values_are_clamped_to_the_range(fake_comfy):
    calls, _ = fake_comfy
    bar = NodeProgress(2)
    bar.at(-4)
    bar.at(9)
    assert calls[1:] == [("at", 0, 2), ("at", 2, 2)]


def test_an_empty_loop_builds_no_bar(fake_comfy):
    calls, _ = fake_comfy
    bar = NodeProgress(0)
    bar.step()
    bar.finish()
    assert calls == []


def test_a_broken_bar_never_breaks_the_render(fake_comfy):
    calls, behaviour = fake_comfy
    bar = NodeProgress(2)
    behaviour["raise"] = RuntimeError("socket gone")
    bar.step()            # swallowed, bar hidden
    behaviour["raise"] = None
    bar.step()            # no further updates are attempted
    assert calls == [("new", 2)]
    assert bar.done == 2


def test_cancel_still_stops_the_run(fake_comfy):
    """ComfyUI's hook raises InterruptProcessingException, a BaseException."""
    _, behaviour = fake_comfy
    bar = NodeProgress(2)
    behaviour["raise"] = _Interrupted()
    with pytest.raises(_Interrupted):
        bar.step()


def test_without_comfy_it_is_a_counter(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.utils", None)
    bar = NodeProgress("4")
    bar.step(3)
    assert (bar.total, bar.done, bar._bar) == (4, 3, None)


# ---------------------------------------------------------------------------
# Wiring at the real call sites
# ---------------------------------------------------------------------------

def _tree(rel):
    return ast.parse((REPO / rel).read_text(encoding="utf-8"))


def _loops_over(tree, iter_src):
    return [n for n in ast.walk(tree) if isinstance(n, ast.For)
            and ast.unparse(n.iter) == iter_src]


def _first_call(loop):
    first = loop.body[0]
    assert isinstance(first, ast.Expr), ast.unparse(first)
    return ast.unparse(first.value)


def _assigned(tree, name):
    return [ast.unparse(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(getattr(t, "id", None) == name for t in n.targets)]


def test_the_writer_steps_once_per_beat():
    tree = _tree("nodes/OTR_LedgerScriptWriter.py")
    (loop,) = _loops_over(tree, "enumerate(outline.beats)")
    assert _first_call(loop) == "_beat_progress.at(_beat_index)"
    assert _assigned(tree, "_beat_progress") == [
        "NodeProgress(len(outline.beats), 'writer beats')"]
    assert "_beat_progress.finish()" in ast.unparse(tree)


def test_the_sequencer_steps_once_per_line():
    tree = _tree("nodes/scene_sequencer.py")
    (loop,) = _loops_over(tree, "enumerate(lines_to_render)")
    assert _first_call(loop) == "_line_progress.at(i)"
    assert _assigned(tree, "_line_progress") == [
        "NodeProgress(len(lines_to_render), 'sequencer lines')"]
    assert "_line_progress.finish()" in ast.unparse(tree)


def test_local_voices_step_per_line_and_cloud_voices_keep_the_heartbeat():
    tree = _tree("nodes/_otr_voice_node_common.py")
    assert _assigned(tree, "_line_progress") == [
        "NodeProgress(0 if adapter_is_cloud_side(adapter) else len(misses), "
        "'voice lines')"]
    serial = [loop for loop in _loops_over(tree, "misses")
              if any(isinstance(s, ast.Expr)
                     and ast.unparse(s.value) == "_line_progress.step()"
                     for s in loop.body)]
    assert len(serial) == 1


def test_the_helper_has_production_callers_only_at_those_three_sites():
    callers = sorted(
        p.relative_to(REPO).as_posix()
        for p in (REPO / "nodes").rglob("*.py")
        if "NodeProgress(" in p.read_text(encoding="utf-8")
        and p.name != "node_progress.py")
    assert callers == ["nodes/OTR_LedgerScriptWriter.py",
                       "nodes/_otr_voice_node_common.py",
                       "nodes/scene_sequencer.py"]
    assert np_mod.__all__ == ["NodeProgress"]
