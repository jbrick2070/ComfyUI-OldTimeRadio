"""Models are unloaded when an episode's prompt ends (operator, 2026-09-26:
"we need to unload models after they are used").

The mechanism is ComfyUI's own post-prompt flag, asked for at the first node,
so it holds on success and on failure. CPU-safe: a stand-in PromptServer.
"""
from __future__ import annotations

import inspect
import sys
import types

from nodes import _otr_vram_levers as levers
from nodes import _otr_workflow_validator as wv


class _Queue:
    def __init__(self):
        self.flags = {}

    def set_flag(self, name, data):
        self.flags[name] = data


def _install_server(monkeypatch):
    queue = _Queue()
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(prompt_queue=queue))
    monkeypatch.setitem(sys.modules, "server", server)
    return queue


def test_the_release_sets_comfyuis_own_free_memory_flag(monkeypatch):
    """free_memory is what makes ComfyUI's worker call unload_all_models and
    reset its node cache after the prompt (main.py prompt_worker)."""
    queue = _install_server(monkeypatch)
    assert levers.release_models_after_this_prompt() is True
    assert queue.flags == {"free_memory": True}


def test_without_a_running_comfyui_the_release_is_a_quiet_no(monkeypatch):
    monkeypatch.setitem(sys.modules, "server", types.ModuleType("server"))
    assert levers.release_models_after_this_prompt() is False


def test_the_validator_asks_before_anything_can_refuse():
    """Wiring at the real site, first thing in validate(): a refusal raised by
    any later check must still leave the release requested."""
    src = inspect.getsource(wv.WorkflowValidator.validate)
    ask = src.index("release_models_after_this_prompt()")
    assert ask < src.index("self._assert_stamp(")
    assert ask < src.index("self._admit_story_input(")


def test_a_refused_episode_still_releases(monkeypatch):
    """Drive the real validate(): a story admission refusal raises, and the
    release was already requested."""
    queue = _install_server(monkeypatch)

    def _refuse(self, prompt, unique_id):
        raise ValueError("no story")

    monkeypatch.setattr(wv.WorkflowValidator, "_admit_story_input", _refuse)
    node = wv.WorkflowValidator()
    try:
        node.validate("", True, True, prompt={}, unique_id="1")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("the stand-in refusal did not fire")
    assert queue.flags == {"free_memory": True}
