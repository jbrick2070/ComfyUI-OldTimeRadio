"""Wan-lane OOM fix: free_idle_models_before_phase (stills -> video boundary).

The detach-only pre-render reclaim detached 0 under ComfyUI's dynamic-VRAM
"Staged" model, so the ~16GB flux portrait stack stayed pinned and the 14B
wan_i2v OOM'd at the ksampler. The phase-boundary free uses the canonical
ComfyUI eviction APIs (cleanup_models_gc + free_memory + soft_empty_cache --
never unload_all_models) and is MEASURED.

These tests inject a fake comfy.model_management so the canonical-call sequence
+ the measured return are locked with no GPU, no real models.
"""

from __future__ import annotations

import sys
import types

from nodes._otr_video_engines import wrapper_bridge as wb


def _install_fake_mm(monkeypatch, *, free_before=1000, free_after=5000,
                     loaded=None, unloaded=None):
    """Inject a fake comfy.model_management; returns the call-record dict."""
    rec = {"cleanup_gc": 0, "free_memory": 0, "soft_empty": 0,
           "unload_all": 0, "free_args": None}
    calls = {"n": 0}

    mm = types.ModuleType("comfy.model_management")
    mm.current_loaded_models = list(loaded or [])

    def get_torch_device():
        return "cuda:0"

    def get_free_memory(dev):
        # before on the 1st call, after on subsequent calls (post-free).
        calls["n"] += 1
        return (free_before if calls["n"] == 1 else free_after) * 1024 * 1024

    def get_total_memory(dev):
        return 16 * 1024 * 1024 * 1024

    def cleanup_models_gc():
        rec["cleanup_gc"] += 1

    def free_memory(required, dev, keep_loaded=None, **kw):
        rec["free_memory"] += 1
        rec["free_args"] = (required, dev, keep_loaded)
        return list(unloaded or [])

    def soft_empty_cache(force=False):
        rec["soft_empty"] += 1

    def unload_all_models():  # MUST NOT be called (V-4/V-5)
        rec["unload_all"] += 1

    mm.get_torch_device = get_torch_device
    mm.get_free_memory = get_free_memory
    mm.get_total_memory = get_total_memory
    mm.cleanup_models_gc = cleanup_models_gc
    mm.free_memory = free_memory
    mm.soft_empty_cache = soft_empty_cache
    mm.unload_all_models = unload_all_models

    comfy_pkg = sys.modules.get("comfy") or types.ModuleType("comfy")
    monkeypatch.setitem(sys.modules, "comfy", comfy_pkg)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    return rec


def test_phase_free_calls_canonical_apis_and_measures(monkeypatch):
    rec = _install_fake_mm(monkeypatch, free_before=1000, free_after=5000,
                           unloaded=["m1", "m2"])
    freed = wb.free_idle_models_before_phase(" (pre-render)")
    # 1000MB -> 5000MB free == 4000MB reclaimed.
    assert freed == 4000
    assert rec["cleanup_gc"] == 1        # dead-LoadedModel purge ran
    assert rec["free_memory"] == 1       # canonical graduated free ran
    assert rec["soft_empty"] >= 1        # blocks returned to the pool
    assert rec["unload_all"] == 0        # V-4/V-5: never the nuke


def test_phase_free_requires_full_headroom_not_keep_loaded(monkeypatch):
    # free_memory must be asked to free DOWN TO empty (required == total,
    # keep_loaded == []), i.e. evict every idle model at the boundary.
    rec = _install_fake_mm(monkeypatch)
    wb.free_idle_models_before_phase("")
    required, _dev, keep_loaded = rec["free_args"]
    assert required == 16 * 1024 * 1024 * 1024
    assert keep_loaded == []


def test_phase_free_off_box_returns_minus_one(monkeypatch):
    # No comfy importable (off the GPU box) -> graceful -1, never raises.
    monkeypatch.setitem(sys.modules, "comfy.model_management", None)
    # Force the `import comfy.model_management` to fail.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("comfy"):
            raise ImportError("no comfy off-box")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert wb.free_idle_models_before_phase("x") == -1


def test_phase_free_survives_free_memory_raising(monkeypatch):
    # free_memory throwing must NOT propagate (the detach already ran); best-effort.
    rec = _install_fake_mm(monkeypatch, free_before=2000, free_after=2000)

    def boom(*a, **k):
        raise RuntimeError("allocator hiccup")

    sys.modules["comfy.model_management"].free_memory = boom
    # Does not raise; freed computed from the (equal) before/after measurement.
    freed = wb.free_idle_models_before_phase("")
    assert isinstance(freed, int)
    assert rec["unload_all"] == 0
