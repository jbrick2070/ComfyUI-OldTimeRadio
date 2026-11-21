"""Memory telemetry reads Metal on Apple Silicon instead of logging nan / 0.00.

A Mac render log (2026-11-20, M6, otr_mac16_animatediff) printed
``allocated nan -> nan MB ... free=nan MB`` from the residue freer and
``peak_accelerator_gb=0.00 physical_vram_gb=0.00`` from the writer load, while
the model sat on ``mps:0``. A fake torch with CUDA off and Metal on stands in
for the Mac.
"""

from __future__ import annotations

import sys
import types

import pytest

from nodes import _otr_model_loader, _otr_vram_levers

GIB = 1024 ** 3


def _fake_torch(*, cuda: bool):
    mps = types.SimpleNamespace(
        recommended_max_memory=lambda: 18 * GIB,
        driver_allocated_memory=lambda: 6 * GIB,
        current_allocated_memory=lambda: 4 * GIB,
    )
    cuda_ns = types.SimpleNamespace(
        is_available=lambda: cuda,
        memory_allocated=lambda: 1 * GIB,
        memory_reserved=lambda: 2 * GIB,
        mem_get_info=lambda: (10 * GIB, 16 * GIB),
        max_memory_allocated=lambda: 3 * GIB,
    )
    return types.SimpleNamespace(
        cuda=cuda_ns,
        mps=mps,
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )


def test_snapshot_reads_metal_when_cuda_is_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    snap = _otr_vram_levers._vram_snapshot()
    assert snap == {
        "alloc_mb": 4 * 1024.0,
        "reserved_mb": 6 * 1024.0,
        "free_mb": 12 * 1024.0,
        "total_mb": 18 * 1024.0,
    }


def test_snapshot_keeps_reading_cuda_first(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True))
    snap = _otr_vram_levers._vram_snapshot()
    assert snap == {
        "alloc_mb": 1 * 1024.0,
        "reserved_mb": 2 * 1024.0,
        "free_mb": 10 * 1024.0,
        "total_mb": 16 * 1024.0,
    }


def test_load_placement_reports_metal_figures(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    rec = _otr_model_loader._record_load_placement(
        object(), total_vram=0, quant_policy="none", actual_quant=False, cpu_overflow=False,
    )
    assert rec["peak_accelerator_gb"] == pytest.approx(6.0)
    assert rec["physical_vram_gb"] == pytest.approx(18.0)


def test_load_placement_on_cuda_is_unchanged(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True))
    rec = _otr_model_loader._record_load_placement(
        object(), total_vram=15.99, quant_policy="none", actual_quant=False, cpu_overflow=False,
    )
    assert rec["peak_accelerator_gb"] == pytest.approx(3.0)
    assert rec["physical_vram_gb"] == pytest.approx(15.99)
