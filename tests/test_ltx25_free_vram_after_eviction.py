"""The free-VRAM figure LTX 2.5 logs after its pre-decode eviction, on CPU.

ONE READING since 2026-09-26 (operator: "single"). It used to be a settle loop
of up to twelve 1 s reads, written when the 4060 was still releasing memory
for about three seconds after the eviction call (2026-09-23). By 2026-09-26
every settle line on both boxes read "never moved" -- four identical readings
from 0.0 s -- so the loop cost 3 s a clip for a figure that reaches a log line
and nothing else. The loop's own tests died with it.

WHAT THIS VALUE IS FOR, because it bounds what is worth asserting: it decides
only what gets LOGGED -- an info line, or which warning follows it.
`_make_room_for_decode` logs and returns; nothing shortens, skips or refuses the
decode on it. Do not add a test here that asserts a gate -- there isn't one, by
operator ruling ("never refuse on a number, only an OOM decides").
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_video_engines import eng_ltx25 as E  # noqa: E402


def _engine_cls():
    """Any engine class carrying the helper -- they share one base."""
    for obj in vars(E).values():
        if isinstance(obj, type) and hasattr(obj, "_free_vram_after_eviction_mb"):
            return obj
    raise AssertionError("no class exposes _free_vram_after_eviction_mb")


class _ScriptedMC:
    """Stands in for `motion_common`, handing out a fixed reading sequence."""

    def __init__(self, readings):
        self._readings = list(readings)
        self.calls = 0

    def free_vram_mb(self):
        self.calls += 1
        return self._readings[min(self.calls - 1, len(self._readings) - 1)]


@pytest.fixture
def read_once(monkeypatch):
    """Call the helper against a scripted card; any sleep is a failure."""
    def no_sleep(*_a, **_k):
        raise AssertionError("the post-eviction reading must not wait")

    monkeypatch.setattr(time, "sleep", no_sleep)

    def run(readings, floor=1000.0):
        mc = _ScriptedMC(readings)
        monkeypatch.setattr(E, "_MC", mc)
        engine = object.__new__(_engine_cls())
        engine.name = "ltx25_test"
        return engine._free_vram_after_eviction_mb(floor), mc

    return run


def test_it_takes_exactly_one_reading_and_never_waits(read_once):
    value, mc = read_once([7399.0, 9000.0, 9500.0])
    assert value == 7399.0
    assert mc.calls == 1


def test_an_unreadable_card_reports_the_floor_not_none(read_once):
    value, mc = read_once([None], floor=4940.0)
    assert value == 4940.0 and mc.calls == 1


def test_it_decides_only_what_is_logged_never_the_decode():
    """Guard the operator ruling structurally: no refusal may grow back here.

    `_make_room_for_decode` may log, may warn, and must return. If a future
    edit makes the settled figure shorten, skip or refuse a decode, this fails
    and the ruling gets re-read before the change lands."""
    import ast as _ast
    import inspect
    import textwrap

    cls = _engine_cls()
    if not hasattr(cls, "_make_room_for_decode"):
        pytest.skip("no _make_room_for_decode on this engine class")
    src = textwrap.dedent(inspect.getsource(cls._make_room_for_decode))
    tree = _ast.parse(src)

    # Where is the measurement? Find the line that assigns `after_mb`.
    assign_line = None
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Name) and node.id == "after_mb" and                 isinstance(getattr(node, "ctx", None), _ast.Store):
            assign_line = node.lineno if assign_line is None else min(
                assign_line, node.lineno)
    assert assign_line is not None, "after_mb is never assigned"

    # AN AST WALK, NOT A SUBSTRING SEARCH. The first version of this guard did
    # `"raise" not in src.split("after_mb")[-1]`, which a QA pass showed was
    # both evadable (move the raise into a helper and the token disappears)
    # and fragile (any comment using the word "raise" failed it spuriously).
    raises = [n.lineno for n in _ast.walk(tree)
              if isinstance(n, _ast.Raise) and n.lineno > assign_line]
    assert not raises, (
        "_make_room_for_decode raises at line(s) %s after measuring free "
        "VRAM; the operator ruling is that only a real OOM decides" % raises)

    # The evasion the substring version allowed: a refusal routed through a
    # helper. Catch the shape by name.
    calls = [n for n in _ast.walk(tree)
             if isinstance(n, _ast.Call) and n.lineno > assign_line]
    suspicious = []
    for call in calls:
        name = getattr(call.func, "attr", None) or getattr(
            call.func, "id", None) or ""
        if any(word in str(name).lower()
               for word in ("refuse", "reject", "abort", "shrink", "skip")):
            suspicious.append((call.lineno, name))
    assert not suspicious, (
        "a refusal-shaped call follows the measurement: %s" % suspicious)
