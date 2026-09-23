"""The settle loop that reports free VRAM after an eviction, on CPU.

WHY THIS FILE EXISTS. `_settled_free_vram_mb` shipped twice on 2026-09-23 with
no test at all, and a contrarian review said so plainly: a loop whose whole job
is to produce one number, carrying four tuned constants, with nothing pinning
its sequence behaviour. Both of the bugs it has had were sequence bugs that no
static read would catch:

  1. It polled every 0.5 s against a release that lands in ~1 s steps, so two
     reads inside one step were identical and it returned on the first
     plateau -- 4940 MB when the settled figure was ~7399.
  2. The first fix seeded the plateau test with `floor_mb`, the UNSETTLED
     reading, so the first real sample matched it and counted as flat
     immediately. Same bug, one level down.

Nothing here touches a GPU: `_MC.free_vram_mb` is replaced with a scripted
sequence and `time.sleep` is neutered, so a case runs in microseconds.

WHAT THIS VALUE IS FOR, because it bounds what is worth asserting: it reaches a
LOG LINE and nothing else. `_make_room_for_decode` logs it and returns; no
branch shortens, skips or refuses the decode on it. Do not add a test here that
asserts a gate -- there isn't one, by operator ruling ("never refuse on a
number, only an OOM decides").
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
        if isinstance(obj, type) and hasattr(obj, "_settled_free_vram_mb"):
            return obj
    raise AssertionError("no class exposes _settled_free_vram_mb")


class _ScriptedMC:
    """Stands in for `motion_common`, handing out a fixed reading sequence."""

    def __init__(self, readings):
        self._readings = list(readings)
        self.calls = 0

    def free_vram_mb(self):
        self.calls += 1
        idx = min(self.calls - 1, len(self._readings) - 1)
        return self._readings[idx]


@pytest.fixture
def settle(monkeypatch):
    """Call the helper against a scripted sequence, with no real waiting."""
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)

    def run(readings, floor=None):
        mc = _ScriptedMC(readings)
        monkeypatch.setattr(E, "_MC", mc)
        engine = object.__new__(_engine_cls())
        first = readings[0]
        seed = floor if floor is not None else (
            float(first) if isinstance(first, (int, float)) else 1000.0)
        return engine._settled_free_vram_mb(seed), mc

    return run


# ---------------------------------------------------------------------------
# the two bugs this loop has actually had
# ---------------------------------------------------------------------------

def test_a_plateau_at_the_start_is_not_settled(settle):
    """THE ORIGINAL BUG. The opening reading is taken while the release is
    still in flight, so repeated identical samples prove only that nothing has
    happened yet. Both shipped versions returned 4940 here."""
    got, _ = settle([4940, 4940, 4940, 7399, 7399, 7399])
    assert got == 7399, "returned a plateau that had never seen a rise"


def test_the_measured_staged_release(settle):
    """The 4060's own shape: free memory climbing in steps with flat pairs
    inside each step, which is what defeated 0.5 s sampling."""
    got, _ = settle([4940, 4940, 6100, 6100, 7399, 7399, 7399, 7399])
    assert got == 7399


def test_floor_is_never_republished_as_the_answer(settle):
    """`floor_mb` is the unsettled pre-eviction figure. It must not come back
    as the settled one just because a probe failed."""
    got, _ = settle([None, 4940, 7399, 7399, 7399], floor=4940.0)
    assert got == 7399, "a failed first probe republished the unsettled number"


# ---------------------------------------------------------------------------
# robustness: it may never hang, and it may never raise
# ---------------------------------------------------------------------------

def test_a_reading_that_never_settles_is_bounded(settle):
    """Rising forever must exit on the try cap, not spin."""
    got, mc = settle([1000.0 * i for i in range(1, 40)])
    assert got is not None
    assert mc.calls <= 13, "more reads than the cap allows: %d" % mc.calls


def test_an_unreadable_card_does_not_spin(settle):
    """All-None (no CUDA, no driver) must terminate on the cap."""
    got, mc = settle([None] * 40, floor=1234.0)
    assert mc.calls <= 13, "spun on an unreadable card: %d reads" % mc.calls
    assert got is None or got == pytest.approx(1234.0)


def test_a_transient_dip_does_not_read_as_settled(settle):
    """A drop below the high-water mark means something else took memory, or
    the release is still moving. It must not count toward the plateau."""
    got, _ = settle([4940, 6469, 4940, 6100, 7399, 7399, 7399, 7399])
    assert got == 7399


def test_a_big_card_that_settles_immediately_still_terminates(settle):
    got, mc = settle([15000.0] * 12)
    assert got == 15000.0
    assert mc.calls <= 13


# ---------------------------------------------------------------------------
# the contract on the value itself
# ---------------------------------------------------------------------------

def test_the_cap_reports_what_is_free_now_not_a_stale_peak(settle):
    """On try-cap exhaustion the CURRENT figure is reported, not the high-water
    mark.

    A QA pass caught the earlier version of this test being vacuous: its
    sequence let the loop exit normally at the peak, so `best` and `now` were
    the same value and the assertion could not tell the two behaviours apart.
    It also caught the behaviour itself being wrong -- reporting a peak seen
    for a single read while a sustained collapse to a fraction of it went
    unmentioned. This number is logged immediately before the decode runs, so
    what is free NOW is the honest answer to the question being asked.

    This sequence forces the divergence: the peak is never confirmed settled,
    and the collapse lasts to the cap."""
    got, _ = settle([1000, 5000, 5000, 100, 100, 100, 100, 100,
                     100, 100, 100, 100])
    assert got == 100, "reported a peak the card had left nine samples earlier"


def test_an_unreadable_card_reports_the_floor_not_none(settle):
    """`floor_mb` must come back when nothing can be read.

    It briefly became a dead parameter, so an all-None probe returned None and
    the caller formats None as "-1 MB" -- a worse figure in the receipt than
    the under-read this whole helper exists to prevent."""
    got, _ = settle([None] * 15, floor=1234.0)
    assert got == pytest.approx(1234.0), "an unreadable card published None"


def test_a_card_that_never_moves_does_not_burn_the_whole_cap(settle):
    """Eviction that frees nothing must not cost the full try cap.

    The rise-then-flats predicate made this case wait out every try, which is
    SLOWER than the buggy version it replaced. Three still reads at 1 s is
    already longer than the release takes to begin moving."""
    slept = []
    mc_calls = settle([7399.0] * 12)[1].calls
    assert mc_calls <= 5, (
        "a card with nothing to release took %d reads" % mc_calls)


def test_it_reaches_a_log_line_and_nothing_branches_on_it():
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
