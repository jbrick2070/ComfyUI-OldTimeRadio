"""Wan-lane VRAM residue freer -- surgical (V-4/V-5) contract.

free_otr_pipeline_residue() is the canonical pre-heavy-engine residue freer. The
2026-06-15 live data proved the dominant pin is the OUT-OF-BAND writer LLM +
Bark (invisible to comfy.model_management), NOT ComfyUI patchers. The surgical
variant frees those + DETACHES any tracked patcher, and NEVER calls
unload_all_models (the V-4/V-5 single-resident-engine invariant).

These inject fake comfy.model_management + writer-LLM/Bark modules so the
contract is locked with no GPU and no real models.
"""

from __future__ import annotations

import sys
import types

import pytest

from nodes import _otr_bark_lib, _otr_model_loader
from nodes._otr_vram_levers import free_otr_pipeline_residue


class _Patcher:
    def __init__(self):
        self.detached = False

    def detach(self, unpatch_all=True):
        self.detached = True


class _LoadedModel:
    def __init__(self, patcher):
        self.model = patcher


def _install_fakes(monkeypatch, *, patchers=()):
    """Inject fakes for comfy.model_management + the OTR writer-LLM/Bark loaders.
    Returns a call-record dict."""
    rec = {"unload_llm": 0, "unload_bark": 0, "unload_all_models": 0,
           "soft_empty": 0}

    mm = types.ModuleType("comfy.model_management")
    mm.current_loaded_models = list(patchers)

    def unload_all_models():            # MUST NOT be called (V-4/V-5)
        rec["unload_all_models"] += 1

    def soft_empty_cache(force=False):
        rec["soft_empty"] += 1

    mm.unload_all_models = unload_all_models
    mm.soft_empty_cache = soft_empty_cache
    comfy_pkg = sys.modules.get("comfy") or types.ModuleType("comfy")
    monkeypatch.setitem(sys.modules, "comfy", comfy_pkg)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    # `from . import _otr_model_loader` binds the REAL module attribute when it
    # is already imported (in-suite), so setattr the real functions rather than
    # inject sys.modules fakes (which the package-attribute binding bypasses).
    monkeypatch.setattr(
        _otr_model_loader, "unload_llm",
        lambda: rec.__setitem__("unload_llm", rec["unload_llm"] + 1))
    monkeypatch.setattr(
        _otr_bark_lib, "_unload_bark",
        lambda: rec.__setitem__("unload_bark", rec["unload_bark"] + 1))
    return rec


def test_never_calls_unload_all_models(monkeypatch):
    rec = _install_fakes(monkeypatch)
    report = free_otr_pipeline_residue(reason="surgical-test")
    assert rec["unload_all_models"] == 0
    assert "unload_all_models" not in report["steps_run"]


def test_frees_the_out_of_band_writer_llm_and_bark(monkeypatch):
    rec = _install_fakes(monkeypatch)
    report = free_otr_pipeline_residue(reason="t")
    assert rec["unload_llm"] == 1
    assert rec["unload_bark"] == 1
    assert "unload_llm" in report["steps_run"]
    assert "_unload_bark" in report["steps_run"]


def test_detaches_tracked_flux_patchers(monkeypatch):
    # Defense-in-depth: any ComfyUI-tracked patcher is detached surgically
    # (weights -> offload device), no registry nuke.
    patcher = _Patcher()
    rec = _install_fakes(monkeypatch, patchers=[_LoadedModel(patcher)])
    report = free_otr_pipeline_residue(reason="t")
    assert patcher.detached is True
    assert any(s.startswith("detach_patchers") for s in report["steps_run"])
    assert rec["unload_all_models"] == 0


def test_returns_telemetry_and_soft_empties(monkeypatch):
    rec = _install_fakes(monkeypatch)
    report = free_otr_pipeline_residue(reason="t")
    assert rec["soft_empty"] >= 1
    assert "free_gb_after" in report and "steps_run" in report
    assert report["steps_failed"] == []          # all canonical steps ran clean


def test_best_effort_when_writer_llm_unload_raises(monkeypatch):
    rec = _install_fakes(monkeypatch)
    def boom():
        raise RuntimeError("llm cache busy")
    monkeypatch.setattr(_otr_model_loader, "unload_llm", boom)
    # Must not raise; the failure is recorded, the rest still runs, and
    # unload_all_models is STILL never called.
    report = free_otr_pipeline_residue(reason="t")
    assert any("unload_llm" in f for f in report["steps_failed"])
    assert rec["unload_all_models"] == 0
    assert rec["unload_bark"] == 1
