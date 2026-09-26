# -*- coding: utf-8 -*-
"""The writer loads onto the GPU the policy names, not GPU 0.

WHY (2026-09-25). 0c item 1 let ``LLMRuntimePolicy`` admit ``"cuda:N"`` --
the device ``device_options.resolve_device`` produces for a second GPU. That
turned a loud crash into a silent wrong: the policy said ``cuda:1`` while
``load_llm`` sized VRAM from ``get_device_properties(0)``, forced
``device_map={"": 0}`` on 14.5 GiB+ cards, and skipped ``model.to(device)``
for quantized loads, so the writer landed on GPU 0. A contrarian review
caught it the same day.

Bare ``"cuda"`` is index 0, so every single-GPU box (the 5080, the 4060) is
byte-identical: the overflow map below is asserted equal to its old value.

``load_llm`` is a large function that needs real weights, so the WIRING is
asserted at its real call sites by source inspection -- the one job source
inspection is the right tool for -- and the helpers are exercised directly.

Headless. No engine, no model, no GPU.
"""
from __future__ import annotations

import inspect

import pytest

from nodes import _otr_model_loader as loader
from nodes._otr_shared import llm_policy as lp


@pytest.mark.parametrize("device,index", [
    ("cuda", 0), ("cuda:1", 1), ("cuda:3", 3), ("cpu", 0), ("mps", 0), ("", 0),
])
def test_cuda_index_reads_the_ordinal(device, index):
    assert loader._cuda_index(device) == index


def test_overflow_map_is_unchanged_on_gpu_zero():
    """The 5080/4060 value, byte for byte, before and after."""
    assert loader._cpu_overflow_max_memory(8.0) == {0: "8.00GiB", "cpu": "64GiB"}
    assert loader._cpu_overflow_max_memory(8.0, gpu_index=0) == {0: "8.00GiB", "cpu": "64GiB"}


def test_overflow_map_names_the_chosen_gpu():
    assert loader._cpu_overflow_max_memory(8.0, gpu_index=1) == {1: "8.00GiB", "cpu": "64GiB"}


def test_every_policy_admitted_cuda_device_parses():
    """The loader int()s the ordinal, so the policy must never admit one it
    cannot parse."""
    for device in ("cuda", "cuda:1", "cuda:12"):
        lp.LLMRuntimePolicy(device=device)
        loader._cuda_index(device)
    for device in ("cuda:", "cuda:foo", "cuda:1x", "cuda1", " cuda"):
        with pytest.raises(lp.LLMPolicyError):
            lp.LLMRuntimePolicy(device=device)


def test_load_llm_probes_and_places_on_the_policy_gpu():
    src = inspect.getsource(loader.load_llm)
    assert "_gpu_index = _cuda_index(device)" in src
    assert "get_device_properties(0)" not in src
    assert "memory_reserved(0)" not in src
    assert '{"": 0}' not in src
    assert "get_device_properties(_gpu_index)" in src
    assert '{"": _gpu_index}' in src
    assert "_cpu_overflow_max_memory(\n                        total_vram, gpu_index=_gpu_index)" in src


# ---------------------------------------------------------------------------
# THE CURRENT DEVICE (third pass, 2026-09-25). 1c0b1dd4 fixed the literal-0
# sites, but torch.cuda.synchronize / empty_cache / ipc_collect / memory stats
# and bitsandbytes' own placement act on the CURRENT device, still 0. The fix
# makes the policy's GPU current for the whole load and for teardown, so every
# such call -- including ones not yet written -- follows it.
# ---------------------------------------------------------------------------

import ast
import contextlib
import pathlib
import types

_SRC = pathlib.Path(loader.__file__).read_text(encoding="utf-8")
_TREE = ast.parse(_SRC)


def _fn(name):
    return next(n for n in _TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _is_device_with(node):
    return (isinstance(node, ast.With) and len(node.items) == 1
            and isinstance(node.items[0].context_expr, ast.Call)
            and getattr(node.items[0].context_expr.func, "id", None) == "_on_cuda_device")


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cpu", "mps", "", None])
def test_gpu_zero_and_non_cuda_get_a_nullcontext(device):
    """The single-GPU box enters no torch context at all."""
    assert isinstance(loader._on_cuda_device(device), contextlib.nullcontext)


def _fake_torch(monkeypatch, *, available, count):
    made = []

    class _Dev:
        def __init__(self, index):
            self.index = index
            made.append(self)

    fake_cuda = types.SimpleNamespace(is_available=lambda: available,
                                      device_count=lambda: count, device=_Dev)
    import torch
    monkeypatch.setattr(torch, "cuda", fake_cuda)
    return made


def test_a_cuda_less_host_is_left_alone(monkeypatch):
    _fake_torch(monkeypatch, available=False, count=0)
    assert isinstance(loader._on_cuda_device("cuda:1"), contextlib.nullcontext)


def test_a_second_gpu_gets_a_fresh_device_context_each_time(monkeypatch):
    made = _fake_torch(monkeypatch, available=True, count=2)
    first = loader._on_cuda_device("cuda:1")
    second = loader._on_cuda_device("cuda:1")
    assert first is not second, "a cached torch.cuda.device loses its saved index"
    assert [d.index for d in made] == [1, 1]


def test_a_gpu_the_host_does_not_have_is_refused_by_name(monkeypatch):
    _fake_torch(monkeypatch, available=True, count=1)
    with pytest.raises(loader.ModelLoaderError, match="names GPU 1.*shows 1"):
        loader._on_cuda_device("cuda:1")


def test_the_real_load_runs_inside_the_policy_device_context():
    """request_slot is the one production load site (the transformers backends'
    .load() has no production caller). The context must be the IMMEDIATE parent
    of the load and key on the same policy the load receives."""
    rs = _fn("request_slot")
    hits = []
    for node in ast.walk(rs):
        if _is_device_with(node):
            for stmt in node.body:
                if (isinstance(stmt, ast.Assign)
                        and isinstance(stmt.value, ast.Call)
                        and getattr(stmt.value.func, "id", None) == "load_llm"):
                    arg = ast.unparse(node.items[0].context_expr.args[0])
                    policy_kw = next(k for k in stmt.value.keywords if k.arg == "policy")
                    hits.append((arg, ast.unparse(policy_kw.value)))
    assert hits == [("getattr(_policy, 'device', '')", "_policy")], hits
    # and no load_llm call in request_slot escapes the context
    bare = [n for n in ast.walk(rs) if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) == "load_llm"]
    assert len(bare) == 1


def test_teardown_washes_on_the_gpu_the_model_was_placed_on():
    td = _fn("_teardown_gpu_for_entry")
    read_at = [n.lineno for n in ast.walk(td) if isinstance(n, ast.Assign)
               and any(getattr(t, "id", None) == "placed_on" for t in n.targets)]
    del_at = [n.lineno for n in ast.walk(td) if isinstance(n, ast.Delete)
              and any(getattr(t, "id", None) == "entry" for t in n.targets)]
    assert len(read_at) == 1 and len(del_at) == 1, (read_at, del_at)
    assert read_at[0] < del_at[0], "the device must be read before the entry is deleted"
    withs = [n for n in ast.walk(td) if _is_device_with(n)]
    assert len(withs) == 1
    assert ast.unparse(withs[0].items[0].context_expr.args[0]) == "placed_on"
    inside = {id(n) for n in ast.walk(withs[0])}
    for node in ast.walk(td):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and ast.unparse(node.func).startswith("torch.cuda.")
                and node.func.attr in ("empty_cache", "ipc_collect", "synchronize")):
            assert id(node) in inside, ast.unparse(node)
