"""Memory estimates are recommendations. Runtime placement is the authority.

Pins the 2026-09-17 rule: google/gemma-4-12b-it + bnb_nf4 on an 8 GB card
must not be refused because OTR predicted it would not fit. Hugging Face
auto-download already worked; the remaining failure is Accelerate's own
device-0 allocation, which load_llm retries with CPU overflow.
"""
from __future__ import annotations

import types
from pathlib import Path

from nodes import _otr_model_catalog as cat
from nodes._otr_model_loader import (
    _assert_policy_admits_vram,
    _cpu_overflow_max_memory,
    _is_memory_placement_failure,
    _plan_max_memory,
    _summarize_hf_device_map,
)


def test_gemma_12b_on_8gb_ceiling_is_fail_estimate_not_a_raise():
    verdict = cat.check_vram_fit(
        "google/gemma-4-12b-it", 8192, ceiling_gb=6.8,
    )
    assert verdict.tier == "FAIL"
    _assert_policy_admits_vram(
        "google/gemma-4-12b-it",
        types.SimpleNamespace(value=8192, tier="WARN"),
        types.SimpleNamespace(vram_ceiling_gb=6.8),
    )


def test_70b_fail_estimate_still_does_not_raise():
    verdict = cat.check_vram_fit(cat.TEST_OVERSIZED_LLM, 8192)
    assert verdict.tier == "FAIL"
    _assert_policy_admits_vram(
        cat.TEST_OVERSIZED_LLM,
        types.SimpleNamespace(value=8192, tier="UNKNOWN"),
        types.SimpleNamespace(vram_ceiling_gb=14.5),
    )


def test_first_try_plan_stays_none_on_8gb_and_16gb():
    """16 GB Gemma NF4 must keep the flagship {\"\": 0} path; that path
    only fires when max_memory is None."""
    assert _plan_max_memory(
        "google/gemma-4-12b-it", 8.0,
        cuda_available=True, quant_policy="bnb_nf4",
    ) is None
    assert _plan_max_memory(
        "google/gemma-4-12b-it", 15.99,
        cuda_available=True, quant_policy="bnb_nf4",
    ) is None


def test_cpu_overflow_budget_is_physical_vram_plus_ram_not_a_name_tag():
    eight = _cpu_overflow_max_memory(8.0)
    assert eight == {0: "8.00GiB", "cpu": "64GiB"}
    assert _cpu_overflow_max_memory(15.99) == {0: "15.99GiB", "cpu": "64GiB"}
    assert "2b-it" not in str(eight)


def test_runtime_oom_is_a_placement_failure_unrelated_errors_are_not():
    oom = RuntimeError(
        "Allocation on device 0 would exceed allowed memory. (out of memory)\n"
        "Currently allocated     : 15.91 GiB\n"
        "Device limit            : 8.00 GiB\n"
        "PyTorch limit (set by user-supplied memory fraction)"
    )
    assert _is_memory_placement_failure(oom)
    assert _is_memory_placement_failure(
        ValueError("Some modules are dispatched on the CPU or the disk")
    )
    assert not _is_memory_placement_failure(ValueError("unrelated configuration failure"))
    assert not _is_memory_placement_failure(RuntimeError("gemma4_unified is not supported"))


def test_device_map_summary_reports_cpu_layers_truthfully():
    single = types.SimpleNamespace(hf_device_map=None, device="cuda:0")
    mixed = types.SimpleNamespace(hf_device_map={
        "model.layers.0": 0,
        "model.layers.1": "cpu",
    })
    assert _summarize_hf_device_map(single)["cpu_modules"] is False
    summary = _summarize_hf_device_map(mixed)
    assert summary["cpu_modules"] is True
    assert summary["disk_modules"] is False
    assert summary["cpu_module_count"] == 1


def test_no_test_still_expects_vram_fit_failed_error_raise():
    """Skipped retired rows used to keep a VRAMFitFailedError raise expectation
    alive after request_slot stopped raising. Scan so that cannot rot."""
    needle = "pytest.raises(" + "VRAMFitFailedError"
    root = Path(__file__).resolve().parent
    offenders = []
    for path in sorted(root.glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        if needle in text:
            offenders.append(path.name)
    assert offenders == []


def test_production_never_raises_vram_fit_failed_error():
    needle = "raise " + "VRAMFitFailedError"
    nodes = Path(__file__).resolve().parents[1] / "nodes"
    offenders = []
    for path in sorted(nodes.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if needle in text:
            offenders.append(path.name)
    assert offenders == []
