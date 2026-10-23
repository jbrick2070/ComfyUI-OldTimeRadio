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


def test_the_dispatcher_draws_one_stage_bar_on_a_producer_run():
    # Scene Producer chunk 2 (plan, "Progress"): one node-level bar over the
    # objects, the inner samplers quieted, the quiet closed by the node.
    tree = _tree("nodes/otr_image_gen_dispatcher.py")
    (loop,) = _loops_over(tree, "enumerate(objects)")
    assert _first_call(loop) == "_stage_bar.at(_object_index)"
    assert _assigned(tree, "_stage_bar") == [
        "_NodeProgress(_stage_bar_items(objects, image_policy) if quiet_inner_bars is not "
        "None else 0, 'Scene Producer stills')"]
    (images,) = [n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "dispatch_images"]
    body = ast.unparse(images)
    assert "_stage_bar.finish()" in body
    # The node's own bar is built before the quiet opens, so it keeps the real hook.
    assert body.index("_stage_bar = _NodeProgress(") < body.index("quiet_inner_bars.open()")
    (node,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "dispatch"]
    assert "quiet = _QuietInnerBars() if direction is not None else None" in ast.unparse(node)
    finals = [ast.unparse(st) for n in ast.walk(node) if isinstance(n, ast.Try)
              for st in n.finalbody]
    assert any("quiet.close()" in text for text in finals)


def test_the_helper_has_production_callers_only_at_those_four_sites():
    callers = sorted(
        p.relative_to(REPO).as_posix()
        for p in (REPO / "nodes").rglob("*.py")
        if "NodeProgress(" in p.read_text(encoding="utf-8")
        and p.name != "node_progress.py")
    assert callers == ["nodes/OTR_LedgerScriptWriter.py",
                       "nodes/_otr_voice_node_common.py",
                       "nodes/otr_image_gen_dispatcher.py",
                       "nodes/scene_sequencer.py"]
    assert np_mod.__all__ == ["NodeProgress", "QuietInnerBars", "WeightedProgress"]


@pytest.fixture()
def fake_comfy_bars(monkeypatch):
    """A stand-in ComfyUI with the real ProgressBar contract: a bar reads the
    global hook when it is BUILT, and reports ``(value, total)`` through it."""
    sent, interrupts = [], []
    utils = types.ModuleType("comfy.utils")
    mm = types.ModuleType("comfy.model_management")

    def real(value, total, preview, prompt_id=None, node_id=None):
        sent.append((value, total))

    utils.PROGRESS_BAR_HOOK = real

    def set_hook(fn):
        utils.PROGRESS_BAR_HOOK = fn

    class ProgressBar:
        def __init__(self, total, node_id=None):
            self.total, self.hook = total, utils.PROGRESS_BAR_HOOK

        def update_absolute(self, value, total=None, preview=None):
            if total is not None:
                self.total = total
            if self.hook is not None:
                self.hook(min(value, self.total), self.total, preview)

    utils.ProgressBar = ProgressBar
    utils.set_progress_bar_global_hook = set_hook
    mm.throw_exception_if_processing_interrupted = lambda: interrupts.append(1)
    comfy = types.ModuleType("comfy")
    comfy.utils, comfy.model_management = utils, mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    return utils, real, sent, interrupts


def _dispatch_two_portraits(tmp_path, quiet):
    """Two portraits through the real dispatch loop on a local stand-in
    engine whose render builds a 20-step sampler bar on this thread."""
    import numpy as np

    from nodes import otr_image_gen_dispatcher as disp
    from nodes._otr_image_engines import registry as ireg
    from nodes._otr_shared import role_compat as rc

    saved = dict(ireg._IMAGE_REGISTRY._registry)
    ireg._IMAGE_REGISTRY._registry.clear()
    ireg.register(types.SimpleNamespace(
        name="flux_gen1", roles=rc.ROLES, default_roles=rc.ROLES, commercial_clean=True,
        requires_flag=None, required_inputs=("text_prompt",), engine_version="1"))
    policy = {"policy_version": 2,
              "image_models": {"character_image_model": {"engine_id": "flux_gen1"}},
              "video_models": {"announcer_video_model": {"engine_id": "viz_mxc_cpu"},
                               "music_video_model": {"engine_id": "viz_mxc_mandala"},
                               "character_video_model": {"engine_id": "still_motion"}},
              "seed": {"request_seed": 0}}
    payload = {"version": 1, "objects": [
        {"object_id": cid, "kind": "portrait", "role": "character_video", "char_id": cid,
         "w": 832, "h": 1216, "prompt": "a radio actor " + cid, "prompt_hash": "ph_" + cid,
         "source": "llm"} for cid in ("c1", "c2")]}

    def render(_request):
        sampler = sys.modules["comfy.utils"].ProgressBar(20)
        sampler.update_absolute(10)
        sampler.update_absolute(20)
        return np.full((8, 8, 3), 9, dtype=np.uint8)

    try:
        return disp.dispatch_images(
            {"episode_id": "ep_stage_bar", "cast": [{"char_id": "c1"}, {"char_id": "c2"}]},
            policy, payload, gen_fn=render, output_dir=str(tmp_path),
            lockdir=tmp_path / "lease.lockdir", quiet_inner_bars=quiet)
    finally:
        ireg._IMAGE_REGISTRY._registry.clear()
        ireg._IMAGE_REGISTRY._registry.update(saved)


def test_a_producer_run_shows_its_items_and_hides_its_samplers(fake_comfy_bars, tmp_path):
    utils, real, sent, interrupts = fake_comfy_bars
    quiet = np_mod.QuietInnerBars()
    try:
        _dispatch_two_portraits(tmp_path, quiet)
    finally:
        quiet.close()
    assert sent == [(0, 2), (1, 2), (2, 2)], "only the stage bar reached the node"
    assert len(interrupts) == 4, "every quieted sampler step still checked for Cancel"
    assert utils.PROGRESS_BAR_HOOK is real


def test_a_partner_engine_keeps_its_own_heartbeat_bar():
    # Two bars on one node fight: a cloud_* engine's wait loop redraws this
    # node's bar from fan-out threads the quiet does not reach.
    from nodes import otr_image_gen_dispatcher as disp
    objects = [{"role": "character_video"}, {"role": "character_video"}]
    local = {"image_models": {"character_image_model": {"engine_id": "flux_gen1"}}}
    partner = {"image_models": {"character_image_model": {"engine_id": "cloud_flux_pro"}}}
    assert disp._stage_bar_items(objects, local) == 2
    assert disp._stage_bar_items(objects, partner) == 0


def test_an_ordinary_run_draws_no_stage_bar_and_quiets_nothing(fake_comfy_bars, tmp_path):
    utils, real, sent, interrupts = fake_comfy_bars
    _dispatch_two_portraits(tmp_path, None)
    assert sent == [(10, 20), (20, 20), (10, 20), (20, 20)], "the samplers' own bars, as before"
    assert interrupts == [] and utils.PROGRESS_BAR_HOOK is real


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
        "_WeightedProgress([beat_render_frames(s, frame_count) for s in section['shots']], 'video frames')"]
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


def test_overlapping_quiets_always_end_on_the_real_hook(fake_hook):
    utils, real, _, _ = fake_hook
    a = np_mod.QuietInnerBars().open()
    b = np_mod.QuietInnerBars().open()
    a.close()
    b.close()
    assert utils.PROGRESS_BAR_HOOK is real


def test_a_bar_built_while_quiet_shows_again_after_close(fake_hook):
    utils, real, sent, _ = fake_hook
    quiet = np_mod.QuietInnerBars().open()
    kept = utils.PROGRESS_BAR_HOOK          # what a ProgressBar built now would hold
    quiet.close()
    kept(2, 4, None, node_id=None)
    assert sent == [(2, 4)]


def test_a_beat_weighs_the_frames_it_renders():
    from nodes._otr_video_engines.render_driver import beat_render_frames
    plan = {"target_visible_frames": 100, "join_mode": "chain",
            "segments": [{"index": 0, "render_frames": 97}, {"index": 1, "render_frames": 49, "drop_head": 1}]}
    assert beat_render_frames({"target_frame_count": 100, "coverage_plan": plan}) == 146
    assert beat_render_frames({"target_frame_count": 50}) == 50
    assert beat_render_frames({}, 25) == 25
    assert beat_render_frames({"target_frame_count": 50, "coverage_plan": {"target_visible_frames": 50, "segments": []}}) == 50
