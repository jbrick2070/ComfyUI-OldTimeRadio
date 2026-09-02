"""4060 clean-room residency probe (2026-09-02): does dropping the CLIP object actually
release the Qwen3-4B encoder's VRAM under ComfyUI 0.34 DynamicVRAM, and what does?

Phase A (manual): CLIPLoader -> CLIPTextEncode, report VRAM; drop every reference + gc +
soft_empty_cache, report; comfy.model_management.free_memory(huge), report;
unload_all_models(), report.
Phase B (the engine): the pack's flux2_klein render with load_models_gpu instrumented so
we see free VRAM + current_loaded_models at "Requested to load Flux2". Steps capped to 2
by env for probe time only (not a recipe change).

HISTORICAL RECEIPT (pre-fix probe). Phase B here monkeypatches wrapper_bridge._soft_free
with a CRUDE all-registry free_memory(1e30) to test the hypothesis; it is NOT the shipped
shape (run_graph evict_after_use={"clip"}: named node, dynamic patchers only, unloaded
through unload_model_and_clones at its drop). The shipped shape is measured by Leg C4 in
docs/ship-audit-2026-09-01/4060_CLEANROOM.md. 4060_probe_residency_aimdo.log is the full
run of this file; 4060_probe_residency.log is from an earlier revision without the A1b
line or the Phase B patch.
"""
from __future__ import annotations

import gc
import importlib
import importlib.util
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
import subprocess
import sys
import time
import types

COMFY = r"C:\OTR-CleanRoom\ComfyUI_windows_portable\ComfyUI"
REPO = os.path.join(COMFY, "custom_nodes", "ComfyUI-OldTimeRadio")
os.chdir(COMFY)
sys.path.insert(0, COMFY)
sys.path.insert(1, os.path.join(REPO, "nodes"))
os.environ.setdefault("PYTHONUTF8", "1")
os.environ["OTR_FLUX2_KLEIN_STEPS"] = "2"

pkg = types.ModuleType("otrnodes")
pkg.__path__ = [os.path.join(REPO, "nodes")]
sys.modules["otrnodes"] = pkg

# main.py calls comfy_aimdo.control.init() BEFORE any torch / comfy import; mirror that.
if os.environ.get("OTR_PROBE_AIMDO") == "1":
    import comfy_aimdo.control
    try:
        comfy_aimdo.control.init(simple_vram_headroom=None, nvml_pressure=True)
    except TypeError:
        comfy_aimdo.control.init()
    print("aimdo control.init done; lib=%r" % (getattr(comfy_aimdo.control, "lib", None) is not None,), flush=True)

import folder_paths  # noqa: E402,F401
import asyncio  # noqa: E402
import inspect  # noqa: E402
import nodes as comfy_nodes  # noqa: E402
for fn_name in ("init_builtin_extra_nodes", "init_extra_nodes"):
    fn = getattr(comfy_nodes, fn_name, None)
    if fn is None:
        continue
    try:
        r = fn(init_custom_nodes=False) if fn_name == "init_extra_nodes" else fn()
        if inspect.iscoroutine(r):
            r = asyncio.run(r)
    except Exception as exc:  # noqa: BLE001
        print(f"{fn_name} raised {type(exc).__name__}: {exc}", flush=True)
    if "EmptyFlux2LatentImage" in comfy_nodes.NODE_CLASS_MAPPINGS:
        break
gguf_dir = os.path.join(COMFY, "custom_nodes", "ComfyUI-GGUF")
spec = importlib.util.spec_from_file_location(
    "ComfyUI_GGUF", os.path.join(gguf_dir, "__init__.py"), submodule_search_locations=[gguf_dir])
gg = importlib.util.module_from_spec(spec)
sys.modules["ComfyUI_GGUF"] = gg
spec.loader.exec_module(gg)
comfy_nodes.NODE_CLASS_MAPPINGS.update(getattr(gg, "NODE_CLASS_MAPPINGS", {}))

import torch  # noqa: E402
import comfy.model_management as mm  # noqa: E402
import comfy.memory_management as cmm  # noqa: E402
import comfy.model_patcher as cmp  # noqa: E402

DEV = mm.get_torch_device()

# OTR_PROBE_AIMDO=1 reproduces the server's DynamicVRAM path exactly as main.py enables it
# (comfy_aimdo.control.init -> init_devices -> CoreModelPatcher = ModelPatcherDynamic).
if os.environ.get("OTR_PROBE_AIMDO") == "1":
    import comfy_aimdo.control
    try:
        ok = comfy_aimdo.control.init_devices((d.index, 0) for d in mm.get_all_torch_devices())
    except TypeError:
        ok = comfy_aimdo.control.init_devices(d.index for d in mm.get_all_torch_devices())
    if ok:
        cmp.CoreModelPatcher = cmp.ModelPatcherDynamic
        cmm.aimdo_enabled = True
        print("DynamicVRAM enabled in-process (aimdo)", flush=True)
    else:
        print("aimdo init_devices FAILED; running classic", flush=True)


def smi():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True).strip() + " MiB(smi)"
    except Exception:  # noqa: BLE001
        return "smi n/a"


def report(tag):
    names = []
    for lm in list(mm.current_loaded_models):
        m = lm.model
        names.append("dead" if m is None else
                     f"{m.model.__class__.__name__}[loaded={m.loaded_size() / 2**20:.0f}MB dyn={m.is_dynamic()}]")
    print(f"[{tag}] free={mm.get_free_memory(DEV) / 2**20:.0f}MB alloc={torch.cuda.memory_allocated() / 2**20:.0f}MB "
          f"reserved={torch.cuda.memory_reserved() / 2**20:.0f}MB {smi()} loaded={names}", flush=True)


print(f"aimdo_enabled={getattr(cmm, 'aimdo_enabled', None)} vram_state={mm.vram_state}", flush=True)
report("start")

# ---- Phase A ----
CL = comfy_nodes.NODE_CLASS_MAPPINGS["CLIPLoader"]
TE = comfy_nodes.NODE_CLASS_MAPPINGS["CLIPTextEncode"]
t = time.time()
clip = CL().load_clip("qwen_3_4b.safetensors", type="flux2")[0]
cond = TE().encode(clip, "A 1940s radio set on a wooden desk, warm tungsten light")[0]
print(f"encode took {time.time() - t:.1f}s", flush=True)
report("A1 after encode (clip referenced)")
# Candidate eviction WHILE the encoder is still registered (the server's failure mode is
# an orphaned vbar after the reference is dropped): unload this model and its clones.
mm.unload_model_and_clones(clip.patcher)
report("A1b after unload_model_and_clones(clip.patcher) (clip still referenced)")
del clip
gc.collect()
mm.soft_empty_cache()
report("A2 after del clip + gc + soft_empty_cache")
mm.cleanup_models_gc()
mm.cleanup_models()
report("A2b after cleanup_models_gc + cleanup_models")
mm.free_memory(64 * 2**30, DEV)
report("A3 after free_memory(64GiB)")
mm.unload_all_models()
gc.collect()
mm.soft_empty_cache()
report("A4 after unload_all_models")
del cond
gc.collect()
mm.soft_empty_cache()
report("A5 after del cond")

# ---- Phase B ----
# Simulate the executor-side fix candidate: when free_after_use drops an intermediate,
# unload every registered model FIRST (only the encoder is registered at that point; the
# DiT has not been requested yet), then the usual gc + soft cache empty.
import otrnodes._otr_video_engines.wrapper_bridge as _wb  # noqa: E402
_orig_soft_free = _wb._soft_free


def _evicting_soft_free():
    names = [getattr(getattr(lm.model, "model", None), "__class__", type(None)).__name__ for lm in list(mm.current_loaded_models)]
    if names:
        mm.free_memory(1e30, DEV)
        print(f"[probe] _soft_free evicted registered models: {names}", flush=True)
    _orig_soft_free()


_wb._soft_free = _evicting_soft_free
_orig = mm.load_models_gpu


def _instrumented(models, *a, **k):
    names = [getattr(getattr(m, "model", None), "__class__", type(None)).__name__ for m in models]
    report(f"B before load_models_gpu({names})")
    out = _orig(models, *a, **k)
    report(f"B after load_models_gpu({names})")
    return out


mm.load_models_gpu = _instrumented
zi = importlib.import_module("otrnodes._otr_image_engines.flux2_klein")
eng = zi.Flux2KleinEngine()
t = time.time()
frame = eng.render_image({"object_id": "c01", "prompt": "A 1940s radio set on a wooden desk, warm tungsten light",
                          "seed": 4242, "width": 832, "height": 480})
print(f"B render {frame.shape} took {time.time() - t:.1f}s (2 steps)", flush=True)
report("B after render")
print("PROBE_DONE", flush=True)
