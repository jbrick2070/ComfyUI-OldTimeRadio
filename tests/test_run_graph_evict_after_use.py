"""run_graph(evict_after_use=...) -- the DynamicVRAM-safe encoder eviction (2026-09-02).

On ComfyUI 0.34 with comfy-aimdo, dropping the pack's last reference to a text
encoder leaves its VBAR resident; only ``comfy.model_management.unload_model_and_clones``
releases it mid-prompt (4060 clean room, docs/2026-09-02-encoder-eviction). The executor
performs that unload for NAMED node ids only, at the moment free_after_use drops them,
and only for dynamic patchers. These tests fake ``comfy.model_management`` so the contract
is pinned without a GPU.
"""
from __future__ import annotations

import sys
import types

import pytest

from nodes._otr_video_engines import wrapper_bridge as wb


class _Patcher:
    def __init__(self, name, dynamic=True):
        self.name = name
        self.model = object()
        self._dynamic = dynamic

    def detach(self, unpatch_all=True):
        return self.model

    def is_dynamic(self):
        return self._dynamic


class _Clip:
    """Shape of comfy.sd.CLIP: the model lives behind ``.patcher``."""

    def __init__(self, name, dynamic=True):
        self.patcher = _Patcher(name, dynamic)


def _node(fn):
    return type("N", (), {"FUNCTION": "run", "run": fn})


def _graph(clip_obj, trace=None):
    """clip -> pos -> sample -> decode, plus unet -> sample (the MODEL stays kept).
    ``trace`` (a list) records the moment the sampler runs, so a test can prove the
    encoder was unloaded BEFORE the sampler, not swept at the end of the graph."""
    def _sample(self, model, positive):
        if trace is not None:
            trace.append("sample-ran")
        return ("latent",)
    return {
        "clip": {"class": _node(lambda self: (clip_obj,)), "inputs": {}},
        "unet": {"class": _node(lambda self: (_Patcher("unet"),)), "inputs": {}},
        "pos": {"class": _node(lambda self, clip: ("cond",)),
                "inputs": {"clip": wb.Wire("clip", 0)}},
        "sample": {"class": _node(_sample),
                   "inputs": {"model": wb.Wire("unet", 0), "positive": wb.Wire("pos", 0)}},
        "decode": {"class": _node(lambda self, samples: ("image",)),
                   "inputs": {"samples": wb.Wire("sample", 0)}},
    }


@pytest.fixture
def fake_mm(monkeypatch):
    """Install a fake ``comfy.model_management`` that records unload calls."""
    calls = []
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.unload_model_and_clones = lambda patcher: calls.append(patcher)
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    return calls


def test_named_dynamic_encoder_is_unloaded_at_its_drop_and_nothing_else(fake_mm):
    clip = _Clip("te", dynamic=True)
    # fake_mm is one ordered record: the unload appends the patcher, the sampler
    # appends "sample-ran" -- so the order proves the unload happened at the drop
    # (after the text encode, BEFORE the sampler), not in an end-of-graph sweep.
    out = wb.run_graph(_graph(clip, trace=fake_mm), terminal="decode",
                       free_after_use=True, keep={"unet"}, evict_after_use={"clip"})
    assert out == ("image",)
    assert [getattr(x, "name", x) for x in fake_mm] == ["te", "sample-ran"]


def test_unnamed_nodes_are_never_unloaded_even_when_they_carry_a_patcher(fake_mm):
    clip = _Clip("te", dynamic=True)
    wb.run_graph(_graph(clip), terminal="decode", free_after_use=True, keep={"unet"})
    assert fake_mm == []                                 # opt-in: no names, no unloads


def test_classic_patcher_is_left_to_the_reference_drop(fake_mm):
    clip = _Clip("te", dynamic=False)
    wb.run_graph(_graph(clip), terminal="decode", free_after_use=True,
                 keep={"unet"}, evict_after_use={"clip"})
    assert fake_mm == []                                 # is_dynamic() False -> skipped


def test_evict_requires_free_after_use():
    with pytest.raises(wb.GraphExecutionError, match="requires free_after_use"):
        wb.run_graph(_graph(_Clip("te")), terminal="decode", evict_after_use={"clip"})


def test_evict_names_must_exist_in_the_graph():
    with pytest.raises(wb.GraphExecutionError, match="not in the graph"):
        wb.run_graph(_graph(_Clip("te")), terminal="decode", free_after_use=True,
                     evict_after_use={"text_encoder"})


def test_evict_and_keep_cannot_name_the_same_node():
    with pytest.raises(wb.GraphExecutionError, match="cannot be evicted"):
        wb.run_graph(_graph(_Clip("te")), terminal="decode", free_after_use=True,
                     keep={"unet", "clip"}, evict_after_use={"clip"})


def test_missing_comfy_api_degrades_to_the_reference_drop(monkeypatch):
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")          # no unload_model_and_clones
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    out = wb.run_graph(_graph(_Clip("te")), terminal="decode", free_after_use=True,
                       keep={"unet"}, evict_after_use={"clip"})
    assert out == ("image",)                             # rendered; no exception


def test_no_comfy_at_all_is_a_no_op(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy", None)      # import raises -> return 0
    out = wb.run_graph(_graph(_Clip("te")), terminal="decode", free_after_use=True,
                       keep={"unet"}, evict_after_use={"clip"})
    assert out == ("image",)
