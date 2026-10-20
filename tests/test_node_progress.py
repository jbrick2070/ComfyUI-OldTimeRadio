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


def test_voices_step_per_line_unless_a_partner_heartbeat_draws_the_bar():
    tree = _tree("nodes/_otr_voice_node_common.py")
    assert _assigned(tree, "_line_progress") == [
        "NodeProgress(0 if adapter_draws_partner_heartbeat(adapter) else "
        "len(misses), 'voice lines')"]
    src = ast.unparse(tree)
    assert "on_item_done=lambda _job: _line_progress.step()" in src
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
    assert np_mod.__all__ == ["NodeProgress", "QuietInnerBars", "WeightedProgress"]


# ---------------------------------------------------------------------------
# The local video walk: frames rendered of the episode, inner bars quiet
# ---------------------------------------------------------------------------

def test_weighted_beats_start_where_the_beats_before_them_end(fake_comfy):
    calls, _ = fake_comfy
    bar = np_mod.WeightedProgress([97, 0, 194], "video frames")
    bar.begin(0)
    bar.part(97)
    bar.begin(1)          # a floored beat moves the bar the same as a rendered one
    bar.begin(2)
    bar.part(97)
    bar.part(500)         # never past the beat's own end
    bar.finish()
    assert calls == [("new", 291), ("at", 0, 291), ("at", 97, 291), ("at", 97, 291),
                     ("at", 97, 291), ("at", 194, 291), ("at", 291, 291), ("at", 291, 291)]


def test_a_beat_that_fails_part_way_is_not_counted_twice(fake_comfy):
    calls, _ = fake_comfy
    bar = np_mod.WeightedProgress([100, 100], "video frames")
    bar.begin(0)
    bar.part(60)          # one segment rendered, then the beat was floored
    bar.begin(1)
    assert calls[-1] == ("at", 100, 200)


@pytest.fixture()
def fake_hook(monkeypatch):
    sent, interrupts = [], []
    utils = types.ModuleType("comfy.utils")
    mm = types.ModuleType("comfy.model_management")

    def real(value, total, preview, prompt_id=None, node_id=None):
        sent.append((value, total))

    utils.PROGRESS_BAR_HOOK = real

    def set_hook(fn):
        utils.PROGRESS_BAR_HOOK = fn

    utils.set_progress_bar_global_hook = set_hook
    mm.throw_exception_if_processing_interrupted = lambda: interrupts.append(1)
    comfy = types.ModuleType("comfy")
    comfy.utils, comfy.model_management = utils, mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    return utils, real, sent, interrupts


def test_inner_bars_on_the_render_thread_show_nothing_but_still_check_cancel(fake_hook):
    utils, real, sent, interrupts = fake_hook
    with np_mod.QuietInnerBars():
        utils.PROGRESS_BAR_HOOK(3, 8, None, node_id=None)
    assert sent == [] and interrupts == [1]
    assert utils.PROGRESS_BAR_HOOK is real


def test_other_threads_keep_the_real_bar_while_the_walk_is_quiet(fake_hook):
    import threading
    utils, real, sent, _ = fake_hook
    with np_mod.QuietInnerBars():
        hook = utils.PROGRESS_BAR_HOOK
        worker = threading.Thread(target=lambda: hook(1, 2, None, node_id="5"))
        worker.start()
        worker.join()
    assert sent == [(1, 2)]


def test_close_never_overwrites_a_hook_someone_else_installed(fake_hook):
    utils, real, _, _ = fake_hook
    quiet = np_mod.QuietInnerBars().open()
    other = lambda *a, **k: None  # noqa: E731
    utils.set_progress_bar_global_hook(other)
    quiet.close()
    quiet.close()
    assert utils.PROGRESS_BAR_HOOK is other


def test_without_comfy_the_quiet_is_a_no_op(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.utils", None)
    with np_mod.QuietInnerBars() as quiet:
        assert quiet._hook is None


def test_the_local_video_walk_draws_frames_and_quiets_its_inner_bars():
    tree = _tree("nodes/_otr_video_engines/render_driver.py")
    (loop,) = _loops_over(tree, "enumerate(section['shots'])")
    assert _first_call(loop) == "_video_progress.begin(_beat_no)"
    assert _assigned(tree, "_video_progress") == [
        "_WeightedProgress([beat_render_frames(s) for s in section['shots']], 'video frames')"]
    src = ast.unparse(tree)
    assert "on_segment_done=_video_progress.part" in src
    assert "_video_progress.finish()" in src
    (run,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_episode"]
    run_src = ast.unparse(run)
    assert "_quiet_bars = _QuietInnerBars()" in run_src and "_quiet_bars.open()" in run_src
    finals = [ast.unparse(st) for n in ast.walk(run) if isinstance(n, ast.Try) for st in n.finalbody]
    assert "_quiet_bars.close()" in finals
    # The quiet never opens inside render_beat_coverage, which cloud worker threads run.
    (beat,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "render_beat_coverage"]
    assert "QuietInnerBars" not in ast.unparse(beat) and "_quiet_bars" not in ast.unparse(beat)


def test_the_fanout_reports_each_finished_job_on_the_calling_thread():
    import threading
    from nodes._otr_shared.cloud_fanout import run_cloud_fanout

    caller = threading.get_ident()
    seen = []
    outcome = run_cloud_fanout(
        ["a", "b", "c"], item_id=lambda s: s,
        execute=lambda s: (_ for _ in ()).throw(RuntimeError(s)) if s == "b" else s,
        workers=2,
        on_item_done=lambda s: seen.append((s, threading.get_ident())))
    assert sorted(s for s, _ in seen) == ["a", "b", "c"]
    assert {tid for _, tid in seen} == {caller}
    assert set(outcome.results) == {"a", "c"} and set(outcome.errors) == {"b"}


@pytest.mark.parametrize("name,heartbeat", [
    ("cloud_elevenlabs", True), ("cloud_sonilo", True),
    ("google_tts", False), ("kokoro", False), ("chatterbox", False)])
def test_only_partner_engines_count_as_drawing_the_bar(name, heartbeat):
    from nodes._otr_shared.cloud_fanout import adapter_draws_partner_heartbeat
    assert adapter_draws_partner_heartbeat(types.SimpleNamespace(name=name)) is heartbeat
