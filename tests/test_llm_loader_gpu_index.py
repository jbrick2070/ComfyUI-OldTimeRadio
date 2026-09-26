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
