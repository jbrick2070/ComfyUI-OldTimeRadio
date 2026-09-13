"""Probe reporting tests with simulated libraries; these are not AMD receipts."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def probe(monkeypatch):
    # The script adjusts sys.path for standalone use; contain that in the test.
    monkeypatch.setattr(sys, "path", list(sys.path))
    path = Path(__file__).resolve().parents[1] / "scripts/otr_rocm_probe.py"
    spec = importlib.util.spec_from_file_location("_rocm_probe_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_COMFY", None)
    monkeypatch.delitem(sys.modules, "_otr_probe_device_options", raising=False)
    yield module
    sys.modules.pop("_otr_probe_device_options", None)


@pytest.fixture
def fake_torch(monkeypatch):
    torch = SimpleNamespace(
        __version__="test-rocm", version=SimpleNamespace(hip="test", cuda=None),
        cuda=SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1,
            get_device_name=lambda index: "simulated AMD",
            get_device_properties=lambda index: SimpleNamespace(total_memory=16 * 2**30),
        ),
    )
    # No tensor methods: compute attempts fail, and later checks must still run.
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_no_torch_is_an_explicit_setup_failure(probe, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert probe.main() == 2
    assert "NOT IMPORTABLE" in capsys.readouterr().out


@pytest.mark.parametrize("root", [None, "found-but-broken-comfy"])
def test_failed_comfy_import_never_reports_fallback_as_hardware(
        probe, fake_torch, monkeypatch, capsys, root):
    monkeypatch.setattr(probe, "_COMFY", root)
    monkeypatch.setitem(sys.modules, "comfy", None)
    monkeypatch.setitem(sys.modules, "comfy.model_management", None)
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    assert probe.main() == 0
    out = capsys.readouterr().out
    assert "SKIPPED: ComfyUI import failed" in out
    assert not any(line.startswith("vendor()") for line in out.splitlines())
    assert not any(line.startswith("resolve_device(") for line in out.splitlines())
    assert "bf16 matmul on the card" in out
    assert "fp16 matmul on the card" in out
    assert "what to do with this" in out


@pytest.mark.parametrize("devices", [{"cuda", "cpu"}, lambda: {"cuda", "cpu"}])
def test_real_device_wrapper_and_bnb_metadata_reporting(
        probe, fake_torch, monkeypatch, capsys, devices):
    mm = ModuleType("comfy.model_management")
    mm.is_amd = lambda: True
    mm.is_nvidia = lambda: False
    mm.get_torch_device = lambda: SimpleNamespace(type="cuda", index=0)
    mm.get_gpu_device_options = lambda: ["default", "cpu"]
    comfy = ModuleType("comfy")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setattr(probe, "_COMFY", "simulated-comfy")
    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace(
        __version__="test", supported_torch_devices=devices))
    assert probe.main() == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert next(line for line in lines if line.startswith("vendor()")).endswith("amd")
    assert next(line for line in lines if line.startswith("resolve_device('default')")).endswith("cuda")
    assert next(line for line in lines if line.startswith("resolve_device('tpu')")).endswith("tpu")
    assert next(line for line in lines if line.startswith("bitsandbytes declared devices")).endswith("['cpu', 'cuda']")
    assert "NOT TESTED: import/metadata" in out
