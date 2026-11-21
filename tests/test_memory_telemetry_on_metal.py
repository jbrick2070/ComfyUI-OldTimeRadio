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


def test_lfc_watchdog_reads_metal_allocation(monkeypatch):
    from nodes import _otr_lfc_watchdog as wd

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    assert wd._torch_vram_allocated_gb() == pytest.approx(4.0)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True))
    assert wd._torch_vram_allocated_gb() == pytest.approx(1.0)


def test_system_specs_name_the_mac_chip_and_metal_gpu(monkeypatch):
    """A Mac episode's SYSTEM block read "CPU arm", "GPU (unknown)",
    "CUDA (unknown)"."""
    import platform

    from nodes import _otr_sys_specs as specs

    fake = _fake_torch(cuda=False)
    fake.version = types.SimpleNamespace(cuda=None)
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(specs.sys, "platform", "darwin")
    monkeypatch.setattr(platform, "processor", lambda: "arm")
    monkeypatch.setattr(
        specs.otr_proc, "run",
        lambda *a, **k: types.SimpleNamespace(stdout="Apple M6\n", returncode=0),
    )
    assert specs._cpu_model() == "Apple M6"
    assert specs._gpu_via_torch() == ("Apple M6 GPU (Metal)", "18.0 GiB", "n/a (Metal)")


def test_system_specs_off_a_mac_never_ask_sysctl(monkeypatch):
    from nodes import _otr_sys_specs as specs

    def boom(*_a, **_k):
        raise AssertionError("sysctl asked off a Mac")

    monkeypatch.setattr(specs.sys, "platform", "win32")
    monkeypatch.setattr(specs.otr_proc, "run", boom)
    assert specs._mac_chip_name() == ""


def test_video_peak_sampler_reads_metal_when_nvml_answers_nothing(monkeypatch):
    """The credits' "VRAM: x of y" peak comes from VramPeakProbe, which read
    NVML only, so every Mac episode printed "(unknown)"."""
    from nodes._otr_video_engines import motion_common as MC

    monkeypatch.setattr(MC._GR, "sample_used_mb", lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    assert MC.vram_used_mb() == pytest.approx(6 * 1024.0)


def test_video_peak_sampler_keeps_nvml_first(monkeypatch):
    from nodes._otr_video_engines import motion_common as MC

    monkeypatch.setattr(MC._GR, "sample_used_mb", lambda *a, **k: 1234.0)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    assert MC.vram_used_mb() == 1234.0
    monkeypatch.setattr(MC._GR, "sample_used_mb", lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True))
    assert MC.vram_used_mb() is None
